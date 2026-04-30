//! HLSSA -> LLSSA lowering pass
//!
//! Translates the high-level SSA (with abstract Field/U types and a separate
//! constant map) into low-level SSA (explicit integer widths, field-as-struct,
//! constants-as-instructions).
//!
//! Array types lower to heap-allocated RC'd structs behind `Ptr`, following
//! the layout in `docs/llssa.md`. MkSeq, ArrayGet, ArraySet, and MemOp
//! (Bump/Drop) are lowered to explicit memory operations.

use std::collections::{BTreeMap, HashMap};

use crate::compiler::analysis::types::{FunctionTypeInfo, TypeInfo};
use crate::compiler::block_builder::{LLBlockEmitter, LLEmitter};
use crate::compiler::flow_analysis::FlowAnalysis;
use crate::compiler::ir::r#type::{Type, TypeExpr};
use crate::compiler::llssa::{
    FieldArithOp, IntArithOp, IntCmpOp, LLFieldType, LLFunction, LLOp, LLSSA, LLStruct, LLType,
};
use crate::compiler::ssa::{
    BinaryArithOpKind, BlockId, CmpKind, DMatrix, FunctionId, HLFunction, HLSSA, Terminator,
    ValueId,
};
use mavros_artifacts::{ConstraintsLayout, WitnessLayout};

// =============================================================================
// Type helpers
// =============================================================================

/// Map an HLSSA type to an LLType.
fn lower_type(ty: &Type) -> LLType {
    match &ty.expr {
        TypeExpr::Field => LLType::Struct(LLStruct::field_elem()),
        TypeExpr::U(bits) | TypeExpr::I(bits) => LLType::Int(*bits as u32),
        TypeExpr::Array(..) => LLType::Ptr,
        // In the AD path, WitnessOf values are heap-allocated AD nodes
        TypeExpr::WitnessOf(_) => LLType::Ptr,
        TypeExpr::Tuple(_) => LLType::Ptr,
        _ => panic!("Unsupported type in HLSSA->LLSSA lowering: {}", ty),
    }
}

/// Get the LLStruct layout for a single element of the given HLSSA type,
/// for use in InlineArray fields. Scalar types become single-field structs.
fn elem_struct(ty: &Type) -> LLStruct {
    match &ty.expr {
        TypeExpr::Field => LLStruct::field_elem(),
        TypeExpr::U(bits) | TypeExpr::I(bits) => {
            LLStruct::new(vec![LLFieldType::Int(*bits as u32)])
        }
        TypeExpr::Array(..) => LLStruct::new(vec![LLFieldType::Ptr]),
        TypeExpr::Tuple(_) => LLStruct::new(vec![LLFieldType::Ptr]),
        TypeExpr::WitnessOf(_) => LLStruct::new(vec![LLFieldType::Ptr]),
        _ => panic!("Unsupported element type: {}", ty),
    }
}

/// Get the RC'd array struct for an Array<T, N> type.
fn rc_array_struct(elem_type: &Type, count: usize) -> LLStruct {
    LLStruct::rc_array(elem_struct(elem_type), count)
}

/// Convert an HLSSA element type to an LLFieldType for use in tuple struct layouts.
fn tuple_field_type(ty: &Type) -> LLFieldType {
    match &ty.expr {
        TypeExpr::Field => LLFieldType::Inline(LLStruct::field_elem()),
        TypeExpr::U(bits) | TypeExpr::I(bits) => LLFieldType::Int(*bits as u32),
        TypeExpr::Array(..) => LLFieldType::Ptr,
        TypeExpr::Tuple(_) => LLFieldType::Ptr,
        TypeExpr::WitnessOf(_) => LLFieldType::Ptr,
        _ => panic!("Unsupported tuple element type: {}", ty),
    }
}

/// Build the LLStruct layout for a heap-allocated RC'd tuple.
/// Layout: { Inline(RcHeader), field0, field1, ... }
fn rc_tuple_struct(element_types: &[Type]) -> LLStruct {
    let mut fields = vec![LLFieldType::Inline(LLStruct::rc_header())];
    for elem_ty in element_types {
        fields.push(tuple_field_type(elem_ty));
    }
    LLStruct::new(fields)
}

/// Extract (element_type, count) from an HLSSA array type.
fn array_info(ty: &Type) -> (&Type, usize) {
    match &ty.expr {
        TypeExpr::Array(inner, n) => (inner.as_ref(), *n),
        _ => panic!("Expected array type, got: {}", ty),
    }
}

// =============================================================================
// Drop function tracking
// =============================================================================

struct DropFnEntry {
    hlssa_type: Type,
    fn_id: FunctionId,
}

/// Get or create a drop function for a type that needs dropping (currently Array or WitnessOf).
/// For arrays, recursively creates drop functions for inner elements that need dropping.
fn get_or_create_drop_fn(
    ty: &Type,
    llssa: &mut LLSSA,
    drop_fns: &mut Vec<DropFnEntry>,
    ad_fns: &mut AdFunctions,
) -> FunctionId {
    // Check if already exists
    for entry in drop_fns.iter() {
        if entry.hlssa_type == *ty {
            return entry.fn_id;
        }
    }

    // Recursively create drop fns for inner heap-allocated elements first
    match &ty.expr {
        TypeExpr::Array(inner, _) => {
            if needs_drop(&inner.expr) {
                get_or_create_drop_fn(inner, llssa, drop_fns, ad_fns);
            }
        }
        TypeExpr::Tuple(elements) => {
            for elem in elements {
                if needs_drop(&elem.expr) {
                    get_or_create_drop_fn(elem, llssa, drop_fns, ad_fns);
                }
            }
        }
        _ => {}
    }

    // Resolve or create the drop function ID
    let fn_id = match &ty.expr {
        TypeExpr::WitnessOf(_) => ad_fns.get_drop_fn(llssa),
        TypeExpr::Array(_inner, _) => llssa.add_function(format!("drop_{}", ty)),
        TypeExpr::Tuple(_) => llssa.add_function(format!("drop_{}", ty)),
        _ => panic!("{} is not supported yet", ty),
    };
    drop_fns.push(DropFnEntry {
        hlssa_type: ty.clone(),
        fn_id,
    });
    fn_id
}

// =============================================================================
// AD generated function tracking
// =============================================================================

/// Holds function IDs for the generated AD dispatch functions.
/// These are created lazily on first use.
struct AdBumpIds {
    da: FunctionId,
    db: FunctionId,
    dc: FunctionId,
}

impl AdBumpIds {
    fn get(&self, matrix: DMatrix) -> FunctionId {
        match matrix {
            DMatrix::A => self.da,
            DMatrix::B => self.db,
            DMatrix::C => self.dc,
        }
    }
}

struct AdFunctions {
    bumps: Option<AdBumpIds>,
    ad_drop: Option<FunctionId>,
}

impl AdFunctions {
    fn new() -> Self {
        Self {
            bumps: None,
            ad_drop: None,
        }
    }

    fn ensure_bumps(&mut self, llssa: &mut LLSSA) -> &AdBumpIds {
        self.bumps.get_or_insert_with(|| AdBumpIds {
            da: llssa.add_function("__ad_bump_da".to_string()),
            db: llssa.add_function("__ad_bump_db".to_string()),
            dc: llssa.add_function("__ad_bump_dc".to_string()),
        })
    }

    fn get_bump_fn(&mut self, matrix: DMatrix, llssa: &mut LLSSA) -> FunctionId {
        self.ensure_bumps(llssa).get(matrix)
    }

    fn get_drop_fn(&mut self, llssa: &mut LLSSA) -> FunctionId {
        if let Some(id) = self.ad_drop {
            return id;
        }
        self.ensure_bumps(llssa);
        let id = llssa.add_function("__ad_drop".to_string());
        self.ad_drop = Some(id);
        id
    }
}

// =============================================================================
// Lookup generated function tracking
// =============================================================================

/// IDs of the lookup helper functions generated on demand.
struct LookupFunctions {
    /// Forward-pass rangecheck-8 helper id, set on first registration.
    rngchk_8: Option<FunctionId>,
    /// AD-path rangecheck-8 helper id. Branches internally on the runtime
    /// rangecheck-8 sentinel: first call allocates the table region and runs
    /// the init body; subsequent calls reuse the snapshot.
    drngchk_8_call: Option<FunctionId>,
    /// Forward-pass spread lookup helpers, keyed by spread input bit-width.
    spread: BTreeMap<u8, FunctionId>,
    /// AD-path spread lookup helpers, keyed by spread input bit-width.
    dspread_call: BTreeMap<u8, FunctionId>,
    /// Internal zero-initialized global storing `table_idx + 1` for the
    /// forward helper. Zero means unallocated.
    rngchk_8_table_idx_global: Option<usize>,
    /// Internal zero-initialized globals storing `table_idx + 1` for forward
    /// spread helpers. Zero means unallocated.
    spread_table_idx_globals: BTreeMap<u8, usize>,
    /// Internal zero-initialized globals storing `inv_cnst_off + 1` for AD
    /// spread helpers. Zero means unallocated.
    dspread_inv_cnst_off_globals: BTreeMap<u8, usize>,
    /// Internal zero-initialized global storing `inv_cnst_off + 1` for the
    /// AD helper. Zero means unallocated.
    drngchk_8_inv_cnst_off_global: Option<usize>,
}

impl LookupFunctions {
    fn new() -> Self {
        Self {
            rngchk_8: None,
            drngchk_8_call: None,
            spread: BTreeMap::new(),
            dspread_call: BTreeMap::new(),
            rngchk_8_table_idx_global: None,
            spread_table_idx_globals: BTreeMap::new(),
            dspread_inv_cnst_off_globals: BTreeMap::new(),
            drngchk_8_inv_cnst_off_global: None,
        }
    }

    fn get_rngchk_8_fn(&mut self, llssa: &mut LLSSA) -> FunctionId {
        if let Some(id) = self.rngchk_8 {
            return id;
        }
        let id = llssa.add_function("__rngchk_8".to_string());
        self.rngchk_8 = Some(id);
        id
    }

    fn get_spread_fn(&mut self, bits: u8, llssa: &mut LLSSA) -> FunctionId {
        if let Some(id) = self.spread.get(&bits) {
            return *id;
        }
        let id = llssa.add_function(format!("__spread_{}_lookup", bits));
        self.spread.insert(bits, id);
        id
    }

    fn get_dspread_call_fn(&mut self, bits: u8, llssa: &mut LLSSA) -> FunctionId {
        if let Some(id) = self.dspread_call.get(&bits) {
            return *id;
        }
        let id = llssa.add_function(format!("__dspread_{}_ad_call", bits));
        self.dspread_call.insert(bits, id);
        id
    }

    /// Lazily register the AD rangecheck-8 helper. Allocation of the table
    /// region happens lazily inside the helper on first call, so there's no
    /// separate init step or main-prologue hoist.
    fn get_drngchk_8_call_fn(&mut self, llssa: &mut LLSSA) -> FunctionId {
        if let Some(id) = self.drngchk_8_call {
            return id;
        }
        let id = llssa.add_function("__drngchk_8_ad_call".to_string());
        self.drngchk_8_call = Some(id);
        id
    }

    fn allocate_internal_globals(&mut self, llssa: &mut LLSSA) {
        if self.rngchk_8.is_some() && self.rngchk_8_table_idx_global.is_none() {
            self.rngchk_8_table_idx_global = Some(add_ll_global(llssa, LLType::i32()));
        }
        for bits in self.spread.keys() {
            if !self.spread_table_idx_globals.contains_key(bits) {
                let global = add_ll_global(llssa, LLType::i32());
                self.spread_table_idx_globals.insert(*bits, global);
            }
        }
        for bits in self.dspread_call.keys() {
            if !self.dspread_inv_cnst_off_globals.contains_key(bits) {
                let global = add_ll_global(llssa, LLType::i32());
                self.dspread_inv_cnst_off_globals.insert(*bits, global);
            }
        }
        if self.drngchk_8_call.is_some() && self.drngchk_8_inv_cnst_off_global.is_none() {
            self.drngchk_8_inv_cnst_off_global = Some(add_ll_global(llssa, LLType::i32()));
        }
    }
}

fn add_ll_global(llssa: &mut LLSSA, ty: LLType) -> usize {
    let mut global_types = llssa.get_global_types().to_vec();
    let id = global_types.len();
    global_types.push(ty);
    llssa.set_global_types(global_types);
    id
}

// =============================================================================
// Main entry point
// =============================================================================

/// Lower with R1CS layout info — required when the program contains
/// `OpCode::Lookup` or `OpCode::DLookup`, since the generated helpers bake
/// absolute witness/constraint offsets into their bodies.
pub fn lower_with_layout(
    hlssa: &HLSSA,
    flow_analysis: &FlowAnalysis,
    type_info: &TypeInfo,
    witness_layout: WitnessLayout,
    constraints_layout: ConstraintsLayout,
) -> LLSSA {
    lower_inner(
        hlssa,
        flow_analysis,
        type_info,
        Some((witness_layout, constraints_layout)),
    )
}

fn lower_inner(
    hlssa: &HLSSA,
    flow_analysis: &FlowAnalysis,
    type_info: &TypeInfo,
    layout: Option<(WitnessLayout, ConstraintsLayout)>,
) -> LLSSA {
    let main_id = hlssa.get_main_id();
    let main_name = hlssa.get_main().get_name().to_string();
    let mut llssa = LLSSA::with_main(main_name);
    let mut fn_map: HashMap<FunctionId, FunctionId> = HashMap::new();
    let mut drop_fns: Vec<DropFnEntry> = Vec::new();
    let mut ad_fns = AdFunctions::new();
    let mut lookup_fns = LookupFunctions::new();

    // Transfer global types from HLSSA to LLSSA
    let hlssa_global_types: Vec<Type> = hlssa.get_global_types().to_vec();
    let ll_global_types: Vec<LLType> = hlssa_global_types.iter().map(|ty| lower_type(ty)).collect();
    llssa.set_global_types(ll_global_types);
    let forward_a_base_global = if hlssa_has_forward_spread_lookup(hlssa) {
        Some(add_ll_global(&mut llssa, LLType::Ptr))
    } else {
        None
    };

    // First pass: create all functions (so we can map FunctionIds)
    fn_map.insert(main_id, llssa.get_main_id());

    for (fn_id, function) in hlssa.iter_functions() {
        if *fn_id == main_id {
            continue;
        }
        let ll_fn_id = llssa.add_function(function.get_name().to_string());
        fn_map.insert(*fn_id, ll_fn_id);
    }

    // Second pass: lower each function body
    for (fn_id, function) in hlssa.iter_functions() {
        let ll_fn_id = fn_map[fn_id];
        let fn_type_info = type_info.get_function(*fn_id);
        let cfg = flow_analysis.get_function_cfg(*fn_id);

        let ll_func = lower_function(
            function,
            fn_type_info,
            cfg,
            &fn_map,
            &mut llssa,
            &mut drop_fns,
            &mut ad_fns,
            &mut lookup_fns,
            &hlssa_global_types,
            if *fn_id == main_id {
                forward_a_base_global
            } else {
                None
            },
        );

        let _old = llssa.take_function(ll_fn_id);
        llssa.put_function(ll_fn_id, ll_func);
    }

    // Third pass: generate drop function bodies
    generate_all_drop_functions(&mut llssa, &drop_fns);

    // The AD rangecheck helper calls `ad_fns.get_bump_fn(...)` inside its body
    // generator. If we let that happen *after* `generate_all_ad_functions`
    // runs, the bumps would get FunctionIds but never bodies. Pre-allocate
    // now.
    if lookup_fns.drngchk_8_call.is_some() || !lookup_fns.dspread_call.is_empty() {
        ad_fns.ensure_bumps(&mut llssa);
    }
    lookup_fns.allocate_internal_globals(&mut llssa);

    // Fourth pass: generate AD dispatch function bodies
    generate_all_ad_functions(&mut llssa, &ad_fns);

    // Fifth pass: generate lookup helper function bodies (needs layout).
    generate_all_lookup_functions(
        &mut llssa,
        &lookup_fns,
        layout,
        &mut ad_fns,
        forward_a_base_global,
    );

    llssa
}

fn hlssa_has_forward_spread_lookup(hlssa: &HLSSA) -> bool {
    for (_, function) in hlssa.iter_functions() {
        for (_, block) in function.get_blocks() {
            for instruction in block.get_instructions() {
                if matches!(
                    instruction,
                    crate::compiler::ssa::OpCode::Lookup {
                        target: crate::compiler::ssa::LookupTarget::Spread(_),
                        ..
                    }
                ) {
                    return true;
                }
            }
        }
    }
    false
}

// =============================================================================
// Per-function lowering
// =============================================================================

#[allow(clippy::too_many_arguments)]
fn lower_function(
    function: &HLFunction,
    fn_type_info: &FunctionTypeInfo,
    cfg: &crate::compiler::flow_analysis::CFG,
    fn_map: &HashMap<FunctionId, FunctionId>,
    llssa: &mut LLSSA,
    drop_fns: &mut Vec<DropFnEntry>,
    ad_fns: &mut AdFunctions,
    lookup_fns: &mut LookupFunctions,
    hlssa_global_types: &[Type],
    forward_a_base_global: Option<usize>,
) -> LLFunction {
    let mut ll_func = LLFunction::empty(function.get_name().to_string());
    let mut val_map: HashMap<ValueId, ValueId> = HashMap::new();
    let mut block_map: HashMap<BlockId, BlockId> = HashMap::new();

    let hl_entry_id = function.get_entry_id();
    let ll_entry_id = ll_func.get_entry_id();
    block_map.insert(hl_entry_id, ll_entry_id);
    add_vm_parameter(&mut ll_func);

    // Create blocks for non-entry blocks
    for (block_id, _) in function.get_blocks() {
        if *block_id != hl_entry_id {
            let ll_block_id = ll_func.add_block();
            block_map.insert(*block_id, ll_block_id);
        }
    }

    // Add return types
    for ret_type in function.get_returns() {
        ll_func.add_return_type(lower_type(ret_type));
    }

    // Add block parameters
    for (block_id, _) in function.get_blocks() {
        let block = function.get_block(*block_id);
        let ll_block_id = block_map[block_id];
        for (param_id, param_type) in block.get_parameters() {
            let ll_type = lower_type(param_type);
            let ll_param_id = ll_func.add_parameter(ll_block_id, ll_type);
            val_map.insert(*param_id, ll_param_id);
        }
    }

    // Lower instructions and terminators in domination order
    for block_id in cfg.get_domination_pre_order() {
        let block = function.get_block(block_id);
        let ll_block_id = block_map[&block_id];

        // Create a BlockEmitter for this block
        let mut emitter = LLBlockEmitter::new(&mut ll_func, ll_block_id);
        if block_id == hl_entry_id {
            if let Some(global) = forward_a_base_global {
                capture_witgen_a_base(&mut emitter, global);
            }
        }

        // Lower instructions
        for instruction in block.get_instructions() {
            lower_instruction(
                instruction,
                &mut emitter,
                &mut val_map,
                fn_type_info,
                fn_map,
                llssa,
                drop_fns,
                ad_fns,
                lookup_fns,
                hlssa_global_types,
            );
        }

        // Lower terminator (into current block, which may have changed due to block splits)
        if let Some(terminator) = block.get_terminator() {
            lower_terminator(terminator, &mut emitter, &val_map, &block_map);
        }
    }

    ll_func
}

fn new_ll_function(name: impl Into<String>) -> LLFunction {
    let mut func = LLFunction::empty(name.into());
    add_vm_parameter(&mut func);
    func
}

fn add_vm_parameter(func: &mut LLFunction) -> ValueId {
    let entry = func.get_entry_id();
    func.add_parameter(entry, LLType::Ptr)
}

fn capture_witgen_a_base(e: &mut LLBlockEmitter<'_>, global: usize) {
    let a_slot = e.witgen_vm_field_ptr(LLStruct::WITGEN_VM_A);
    let a_base = e.ll_load(a_slot, LLType::Ptr);
    let global_slot = e.global_addr(global);
    e.ll_store(global_slot, a_base);
}

// =============================================================================
// Instruction lowering
// =============================================================================

/// Lower a single HLSSA instruction to LLSSA ops.
#[allow(clippy::too_many_arguments)]
fn lower_instruction(
    instruction: &crate::compiler::ssa::OpCode,
    e: &mut LLBlockEmitter<'_>,
    val_map: &mut HashMap<ValueId, ValueId>,
    fn_type_info: &FunctionTypeInfo,
    fn_map: &HashMap<FunctionId, FunctionId>,
    llssa: &mut LLSSA,
    drop_fns: &mut Vec<DropFnEntry>,
    ad_fns: &mut AdFunctions,
    lookup_fns: &mut LookupFunctions,
    hlssa_global_types: &[Type],
) {
    use crate::compiler::ssa::{CallTarget, CastTarget, ConstValue, MemOp, OpCode, Radix, SeqType};

    match instruction {
        OpCode::BinaryArithOp {
            kind,
            result,
            lhs,
            rhs,
        } => {
            let ll_lhs = val_map[lhs];
            let ll_rhs = val_map[rhs];
            let result_type = fn_type_info.get_value_type(*result);

            let ll_result = match &result_type.expr {
                TypeExpr::Field => {
                    let op = match kind {
                        BinaryArithOpKind::Mul => FieldArithOp::Mul,
                        BinaryArithOpKind::Add => FieldArithOp::Add,
                        BinaryArithOpKind::Sub => FieldArithOp::Sub,
                        BinaryArithOpKind::Div => FieldArithOp::Div,
                        _ => panic!("Unsupported field arith op: {:?}", kind),
                    };
                    e.field_arith(op, ll_lhs, ll_rhs)
                }
                TypeExpr::U(_) => {
                    let op = match kind {
                        BinaryArithOpKind::Add => IntArithOp::Add,
                        BinaryArithOpKind::Sub => IntArithOp::Sub,
                        BinaryArithOpKind::Mul => IntArithOp::Mul,
                        BinaryArithOpKind::And => IntArithOp::And,
                        BinaryArithOpKind::Or => IntArithOp::Or,
                        BinaryArithOpKind::Xor => IntArithOp::Xor,
                        BinaryArithOpKind::Shl => IntArithOp::Shl,
                        BinaryArithOpKind::Shr => IntArithOp::UShr,
                        BinaryArithOpKind::Div => IntArithOp::UDiv,
                        BinaryArithOpKind::Mod => IntArithOp::URem,
                        _ => panic!("Unsupported int arith op: {:?}", kind),
                    };
                    e.int_arith(op, ll_lhs, ll_rhs)
                }
                TypeExpr::I(_) => {
                    let op = match kind {
                        BinaryArithOpKind::Add => IntArithOp::Add,
                        BinaryArithOpKind::Sub => IntArithOp::Sub,
                        BinaryArithOpKind::Mul => IntArithOp::Mul,
                        BinaryArithOpKind::And => IntArithOp::And,
                        BinaryArithOpKind::Or => IntArithOp::Or,
                        BinaryArithOpKind::Xor => IntArithOp::Xor,
                        BinaryArithOpKind::Shl => IntArithOp::Shl,
                        BinaryArithOpKind::Shr => IntArithOp::UShr,
                        BinaryArithOpKind::Div => panic!("Signed div not yet implemented in LLSSA"),
                        BinaryArithOpKind::Mod => panic!("Signed mod not yet implemented in LLSSA"),
                        _ => panic!("Unsupported signed int arith op: {:?}", kind),
                    };
                    e.int_arith(op, ll_lhs, ll_rhs)
                }
                TypeExpr::WitnessOf(_) => {
                    // AD path: Add on WitnessOf → allocate ADSumNode
                    match kind {
                        BinaryArithOpKind::Add => lower_ad_sum(e, ll_lhs, ll_rhs),
                        _ => panic!(
                            "Unsupported WitnessOf arith op: {:?} (should be lowered by WitnessLowering)",
                            kind
                        ),
                    }
                }
                _ => panic!(
                    "Unsupported type for BinaryArithOp in lowering: {:?}",
                    result_type
                ),
            };
            val_map.insert(*result, ll_result);
        }

        OpCode::Cmp {
            kind,
            result,
            lhs,
            rhs,
        } => {
            let ll_lhs = val_map[lhs];
            let ll_rhs = val_map[rhs];
            let lhs_type = fn_type_info.get_value_type(*lhs);

            let ll_result = match &lhs_type.strip_witness().expr {
                TypeExpr::U(_) => {
                    let op = match kind {
                        CmpKind::Lt => IntCmpOp::ULt,
                        CmpKind::Eq => IntCmpOp::Eq,
                    };
                    e.int_cmp(op, ll_lhs, ll_rhs)
                }
                TypeExpr::I(_) => match kind {
                    CmpKind::Eq => e.int_cmp(IntCmpOp::Eq, ll_lhs, ll_rhs),
                    CmpKind::Lt => e.int_cmp(IntCmpOp::SLt, ll_lhs, ll_rhs),
                },
                TypeExpr::Field => match kind {
                    CmpKind::Eq => e.field_eq(ll_lhs, ll_rhs),
                    _ => panic!("Unsupported field comparison: {:?}", kind),
                },
                _ => panic!("Unsupported type for Cmp in lowering: {:?}", lhs_type),
            };
            val_map.insert(*result, ll_result);
        }

        OpCode::Constrain { a, b, c } => {
            let ll_a = val_map[a];
            let ll_b = val_map[b];
            let ll_c = val_map[c];
            e.constrain(ll_a, ll_b, ll_c);
        }

        OpCode::WriteWitness { result, value, .. } => {
            let ll_value = val_map[value];
            e.write_witness(ll_value);
            if let Some(result_id) = result {
                val_map.insert(*result_id, ll_value);
            }
        }

        OpCode::Call {
            results,
            function: CallTarget::Static(callee),
            args,
            unconstrained: _,
        } => {
            let ll_callee = fn_map[callee];
            let ll_args: Vec<ValueId> = args.iter().map(|a| val_map[a]).collect();
            let ll_results = e.call(ll_callee, ll_args, results.len());
            for (hl_result, ll_result) in results.iter().zip(ll_results.iter()) {
                val_map.insert(*hl_result, *ll_result);
            }
        }

        OpCode::Not { result, value } => {
            let ll_value = val_map[value];
            let ll_result = e.not(ll_value);
            val_map.insert(*result, ll_result);
        }

        OpCode::Select {
            result,
            cond,
            if_t,
            if_f,
        } => {
            let ll_cond = val_map[cond];
            let ll_if_t = val_map[if_t];
            let ll_if_f = val_map[if_f];
            let ll_result = e.select(ll_cond, ll_if_t, ll_if_f);
            val_map.insert(*result, ll_result);
        }

        OpCode::Const { result, value } => match value {
            ConstValue::U(bits, val) | ConstValue::I(bits, val) => {
                let ll_val = e.int_const(*bits as u32, *val as u64);
                val_map.insert(*result, ll_val);
            }
            ConstValue::Field(fr) => {
                let field_struct = LLStruct::field_elem();
                let limbs = fr.0.0;
                let l0 = e.int_const(64, limbs[0]);
                let l1 = e.int_const(64, limbs[1]);
                let l2 = e.int_const(64, limbs[2]);
                let l3 = e.int_const(64, limbs[3]);
                let mk = e.mk_struct(field_struct, vec![l0, l1, l2, l3]);
                val_map.insert(*result, mk);
            }
            ConstValue::FnPtr(_) => {
                panic!("FnPtr constants not supported in HLSSA->LLSSA lowering");
            }
        },

        // -- Array operations --
        OpCode::MkSeq {
            result,
            elems,
            seq_type: SeqType::Array(count),
            elem_type,
        } => {
            lower_mk_array(e, val_map, *result, elems, elem_type, *count);
        }

        OpCode::ArrayGet {
            result,
            array,
            index,
        } => {
            lower_array_get(e, val_map, fn_type_info, *result, *array, *index);
        }

        OpCode::ArraySet {
            result,
            array,
            index,
            value,
        } => {
            lower_array_set(
                e,
                val_map,
                fn_type_info,
                *result,
                *array,
                *index,
                *value,
                llssa,
                drop_fns,
                ad_fns,
            );
        }

        // -- RC operations --
        OpCode::MemOp {
            kind: MemOp::Bump(n),
            value,
        } => {
            let val_type = fn_type_info.get_value_type(*value);
            if val_type.is_witness_of() {
                lower_ad_rc_bump(e, val_map, *n, *value);
            } else {
                lower_rc_bump(e, val_map, fn_type_info, *n, *value);
            }
        }

        OpCode::MemOp {
            kind: MemOp::Drop,
            value,
        } => {
            let val_type = fn_type_info.get_value_type(*value);
            if val_type.is_witness_of() {
                lower_ad_rc_drop(e, val_map, *value, llssa, ad_fns);
            } else {
                lower_rc_drop(e, val_map, fn_type_info, *value, llssa, drop_fns, ad_fns);
            }
        }

        // -- AD operations --
        OpCode::NextDCoeff { result } => {
            let ll_result = e.next_d_coeff();
            val_map.insert(*result, ll_result);
        }

        OpCode::BumpD {
            matrix,
            variable,
            sensitivity,
        } => {
            let ll_var = val_map[variable];
            let ll_sens = val_map[sensitivity];
            let bump_fn = ad_fns.get_bump_fn(*matrix, llssa);
            e.call(bump_fn, vec![ll_var, ll_sens], 0);
        }

        OpCode::FreshWitness { result, .. } => {
            lower_ad_fresh_witness(e, val_map, *result);
        }

        OpCode::MulConst {
            result,
            const_val,
            var,
        } => {
            let coeff_type = fn_type_info.get_value_type(*const_val);
            lower_ad_mul_const(e, val_map, *result, *const_val, *var, &coeff_type);
        }

        OpCode::Cast {
            result,
            value,
            target,
        } => {
            let ll_value = val_map[value];
            let source_type = fn_type_info.get_value_type(*value);
            let result_type = fn_type_info.get_value_type(*result);
            if source_type.is_witness_of()
                && result_type.is_witness_of()
                && !matches!(target, CastTarget::WitnessOf)
            {
                // WitnessOf(A) → WitnessOf(B): no-op at runtime (both are AD refs)
                val_map.insert(*result, ll_value);
                return;
            }
            match target {
                CastTarget::WitnessOf => {
                    // Pure value → AD constant node
                    lower_ad_const_wrap(e, val_map, *result, *value, &source_type);
                }
                CastTarget::Field => {
                    if source_type.is_field() {
                        // Field → Field: identity cast (no-op)
                        val_map.insert(*result, ll_value);
                    } else {
                        // U(n)/I(n) → Field: zero-extend to i64, build {val, 0, 0, 0} limbs, FieldFromLimbs
                        let val64 = match &source_type.expr {
                            TypeExpr::U(bits) | TypeExpr::I(bits) if *bits < 64 => {
                                e.zext(ll_value, 64)
                            }
                            TypeExpr::U(64) | TypeExpr::I(64) => ll_value,
                            _ => panic!("Cast to Field from unsupported type: {}", source_type),
                        };
                        let zero = e.int_const(64, 0);
                        let limbs = e.mk_struct(LLStruct::limbs(), vec![val64, zero, zero, zero]);
                        let field_val = e.field_from_limbs(limbs);
                        val_map.insert(*result, field_val);
                    }
                }
                CastTarget::U(target_bits) | CastTarget::I(target_bits) => {
                    let ll_result = match &source_type.expr {
                        TypeExpr::Field => {
                            // Field → U(n)/I(n): FieldToLimbs, extract limb 0, truncate
                            let limbs = e.field_to_limbs(ll_value);
                            let limb0 = e.extract_field(limbs, LLStruct::limbs(), 0);
                            if *target_bits < 64 {
                                e.truncate(limb0, *target_bits as u32)
                            } else {
                                limb0
                            }
                        }
                        TypeExpr::U(source_bits) | TypeExpr::I(source_bits) => {
                            // Integer → Integer: zext or truncate
                            if *target_bits > *source_bits {
                                e.zext(ll_value, *target_bits as u32)
                            } else if *target_bits < *source_bits {
                                e.truncate(ll_value, *target_bits as u32)
                            } else {
                                ll_value
                            }
                        }
                        _ => panic!(
                            "Cast to U({})/I({}) from unsupported type: {}",
                            target_bits, target_bits, source_type
                        ),
                    };
                    val_map.insert(*result, ll_result);
                }
                _ => panic!(
                    "Unsupported cast target in HLSSA->LLSSA lowering: {:?}",
                    target
                ),
            }
        }

        OpCode::Truncate {
            result,
            value,
            to_bits,
            from_bits: _,
        } => {
            assert!(*to_bits <= 64, "Truncate to_bits > 64 not supported");
            let ll_value = val_map[value];
            let source_type = fn_type_info.get_value_type(*value);
            let to_bits = *to_bits as u32;
            let mask64: u64 = if to_bits == 64 {
                u64::MAX
            } else {
                (1u64 << to_bits) - 1
            };

            let ll_result = match &source_type.expr {
                TypeExpr::Field => {
                    let limbs = e.field_to_limbs(ll_value);
                    let limb0 = e.extract_field(limbs, LLStruct::limbs(), 0);
                    let masked_low = if to_bits == 64 {
                        limb0
                    } else {
                        let mask = e.int_const(64, mask64);
                        e.int_arith(IntArithOp::And, limb0, mask)
                    };
                    let zero = e.int_const(64, 0);
                    let new_limbs =
                        e.mk_struct(LLStruct::limbs(), vec![masked_low, zero, zero, zero]);
                    e.field_from_limbs(new_limbs)
                }
                TypeExpr::U(bits) | TypeExpr::I(bits) => {
                    let bits = *bits as u32;
                    if to_bits >= bits {
                        ll_value
                    } else {
                        let mask = e.int_const(bits, mask64);
                        e.int_arith(IntArithOp::And, ll_value, mask)
                    }
                }
                _ => panic!(
                    "Unsupported source type for Truncate in HLSSA->LLSSA lowering: {}",
                    source_type
                ),
            };
            val_map.insert(*result, ll_result);
        }

        OpCode::Spread {
            result,
            value,
            bits,
        } => {
            let value_type = fn_type_info.get_value_type(*value);
            let result_type = fn_type_info.get_value_type(*result);
            let ll_value = val_map[value];
            let ll_result = lower_spread(e, ll_value, &value_type, &result_type, *bits);
            val_map.insert(*result, ll_result);
        }

        OpCode::Unspread {
            result_odd,
            result_even,
            value,
            bits,
        } => {
            let value_type = fn_type_info.get_value_type(*value);
            let odd_type = fn_type_info.get_value_type(*result_odd);
            let even_type = fn_type_info.get_value_type(*result_even);
            let ll_value = val_map[value];
            let (ll_odd, ll_even) =
                lower_unspread(e, ll_value, &value_type, &odd_type, &even_type, *bits);
            val_map.insert(*result_odd, ll_odd);
            val_map.insert(*result_even, ll_even);
        }

        OpCode::MkTuple {
            result,
            elems,
            element_types,
        } => {
            lower_mk_tuple(e, val_map, *result, elems, element_types);
        }

        OpCode::TupleProj { result, tuple, idx } => {
            lower_tuple_proj(e, val_map, fn_type_info, *result, *tuple, *idx);
        }

        OpCode::AssertEq { lhs, rhs } => {
            let ll_lhs = val_map[lhs];
            let ll_rhs = val_map[rhs];
            let lhs_type = fn_type_info.get_value_type(*lhs);

            let eq = match &lhs_type.expr {
                TypeExpr::Field => e.field_eq(ll_lhs, ll_rhs),
                TypeExpr::U(_) | TypeExpr::I(_) => e.int_cmp(IntCmpOp::Eq, ll_lhs, ll_rhs),
                _ => panic!(
                    "Unsupported type for AssertEq in HLSSA->LLSSA lowering: {:?}",
                    lhs_type
                ),
            };

            assert(e, eq);
        }

        OpCode::AssertR1C { a, b, c } => {
            let ll_a = val_map[a];
            let ll_b = val_map[b];
            let ll_c = val_map[c];
            let a_type = fn_type_info.get_value_type(*a);
            let b_type = fn_type_info.get_value_type(*b);
            let c_type = fn_type_info.get_value_type(*c);
            if !a_type.is_field() || !b_type.is_field() || !c_type.is_field() {
                panic!(
                    "Unsupported type for AssertR1C in HLSSA->LLSSA lowering: {:?}, {:?}, {:?}",
                    a_type, b_type, c_type
                );
            }

            let product = e.field_arith(FieldArithOp::Mul, ll_a, ll_b);
            let eq = e.field_eq(product, ll_c);
            assert(e, eq);
        }

        OpCode::InitGlobal { global, value } => {
            let ll_value = val_map[value];
            let r = e.fresh_value();
            e.emit_ll(LLOp::GlobalAddr {
                result: r,
                global_id: *global,
            });
            e.ll_store(r, ll_value);
        }

        OpCode::ReadGlobal {
            result,
            offset,
            result_type,
        } => {
            let r = e.fresh_value();
            e.emit_ll(LLOp::GlobalAddr {
                result: r,
                global_id: *offset as usize,
            });
            let ll_type = lower_type(result_type);
            let loaded = e.ll_load(r, ll_type);
            val_map.insert(*result, loaded);
        }

        OpCode::DropGlobal { global } => {
            let global_type = &hlssa_global_types[*global];
            // Load the value from the global slot, then call its drop function.
            let r = e.fresh_value();
            e.emit_ll(LLOp::GlobalAddr {
                result: r,
                global_id: *global,
            });
            let ll_type = lower_type(global_type);
            let ll_value = e.ll_load(r, ll_type);
            let drop_fn_id = get_or_create_drop_fn(global_type, llssa, drop_fns, ad_fns);
            e.call(drop_fn_id, vec![ll_value], 0);
        }

        // Supported ToRadix: Radix::Bytes on a pure Field input.
        // All other ToRadix shapes are rejected by the explicit catch-all below.
        OpCode::ToRadix {
            result,
            value,
            radix: Radix::Bytes,
            endianness,
            count,
        } if matches!(fn_type_info.get_value_type(*value).expr, TypeExpr::Field) => {
            lower_to_bytes(e, val_map, *result, *value, *endianness, *count);
        }

        OpCode::ToRadix { value, radix, .. } => {
            let value_type = fn_type_info.get_value_type(*value);
            let reason = match (radix, &value_type.expr) {
                (Radix::Dyn(_), _) => "ToRadix with a dynamic radix is not supported".to_string(),
                (Radix::Bytes, TypeExpr::WitnessOf(_)) => {
                    "ToRadix on a witness value is not supported; witness byte decomposition \
                     must be lowered via constrained byte decomposition before this pass"
                        .to_string()
                }
                (Radix::Bytes, _) => format!(
                    "ToRadix(Bytes) only supports Field inputs, got {}",
                    value_type
                ),
            };
            panic!("HLSSA->LLSSA lowering: {}: {:?}", reason, instruction);
        }

        OpCode::Lookup {
            target: crate::compiler::ssa::LookupTarget::Rangecheck(8),
            keys,
            results,
            flag,
        } => {
            assert!(
                results.is_empty(),
                "Rangecheck(8) lookup must have no results"
            );
            assert_eq!(
                keys.len(),
                1,
                "Rangecheck(8) lookup must have exactly one key"
            );
            let key = val_map[&keys[0]];
            let flag_val = val_map[flag];
            let fn_id = lookup_fns.get_rngchk_8_fn(llssa);
            e.call(fn_id, vec![key, flag_val], 0);
        }
        OpCode::Lookup {
            target: crate::compiler::ssa::LookupTarget::Spread(bits),
            keys,
            results,
            flag,
        } => {
            assert_eq!(keys.len(), 1, "Spread lookup must have exactly one key");
            assert_eq!(
                results.len(),
                1,
                "Spread lookup must have exactly one result"
            );
            let key = val_map[&keys[0]];
            let result = val_map[&results[0]];
            let flag_val = val_map[flag];
            let fn_id = lookup_fns.get_spread_fn(*bits, llssa);
            e.call(fn_id, vec![key, result, flag_val], 0);
        }
        OpCode::Lookup { .. } => {
            panic!(
                "Unsupported Lookup variant in HLSSA->LLSSA lowering: {:?}",
                instruction
            );
        }

        OpCode::DLookup {
            target: crate::compiler::ssa::LookupTarget::Rangecheck(8),
            keys,
            results,
            flag,
        } => {
            assert!(
                results.is_empty(),
                "Rangecheck(8) dlookup must have no results"
            );
            assert_eq!(
                keys.len(),
                1,
                "Rangecheck(8) dlookup must have exactly one key"
            );
            let key = val_map[&keys[0]];
            let flag_val = val_map[flag];
            let fn_id = lookup_fns.get_drngchk_8_call_fn(llssa);
            e.call(fn_id, vec![key, flag_val], 0);
        }
        OpCode::DLookup {
            target: crate::compiler::ssa::LookupTarget::Spread(bits),
            keys,
            results,
            flag,
        } => {
            assert_eq!(keys.len(), 1, "Spread dlookup must have exactly one key");
            assert_eq!(
                results.len(),
                1,
                "Spread dlookup must have exactly one result"
            );
            let key = val_map[&keys[0]];
            let result = val_map[&results[0]];
            let flag_val = val_map[flag];
            let fn_id = lookup_fns.get_dspread_call_fn(*bits, llssa);
            e.call(fn_id, vec![key, result, flag_val], 0);
        }
        OpCode::DLookup { .. } => {
            panic!(
                "Unsupported DLookup variant in HLSSA->LLSSA lowering: {:?}",
                instruction
            );
        }

        _ => panic!(
            "Unsupported opcode in HLSSA->LLSSA lowering: {:?}",
            instruction
        ),
    }
}

fn integer_width(ty: &Type, op_name: &str) -> u32 {
    let scalar_ty = ty.strip_witness();
    match scalar_ty.expr {
        TypeExpr::U(bits) | TypeExpr::I(bits) => bits as u32,
        _ => panic!("{} expects an integer type, got {}", op_name, ty),
    }
}

fn zext_to_width(
    e: &mut LLBlockEmitter<'_>,
    value: ValueId,
    from_bits: u32,
    to_bits: u32,
) -> ValueId {
    assert!(
        from_bits <= to_bits,
        "Cannot zero-extend i{} to narrower i{}",
        from_bits,
        to_bits
    );
    if from_bits == to_bits {
        value
    } else {
        e.zext(value, to_bits)
    }
}

fn truncate_to_width(
    e: &mut LLBlockEmitter<'_>,
    value: ValueId,
    from_bits: u32,
    to_bits: u32,
) -> ValueId {
    assert!(
        to_bits <= from_bits,
        "Cannot truncate i{} to wider i{}",
        from_bits,
        to_bits
    );
    if from_bits == to_bits {
        value
    } else {
        e.truncate(value, to_bits)
    }
}

fn lower_spread(
    e: &mut LLBlockEmitter<'_>,
    value: ValueId,
    value_type: &Type,
    result_type: &Type,
    bits: u8,
) -> ValueId {
    let input_bits = integer_width(value_type, "Spread");
    let result_bits = integer_width(result_type, "Spread");
    assert!(
        input_bits <= 64,
        "Spread only supports integer widths up to 64 bits, got {}",
        value_type
    );
    assert_eq!(
        result_bits,
        input_bits * 2,
        "Spread result width must be twice the input width"
    );
    assert_eq!(
        bits as u32, input_bits,
        "Spread opcode bit-width does not match input type width"
    );

    let value = zext_to_width(e, value, input_bits, result_bits);
    let mut acc = e.int_const(result_bits, 0);
    let one = e.int_const(result_bits, 1);

    for i in 0..input_bits {
        let src_shift = e.int_const(result_bits, i as u64);
        let shifted_down = if i == 0 {
            value
        } else {
            e.int_arith(IntArithOp::UShr, value, src_shift)
        };
        let bit = e.int_arith(IntArithOp::And, shifted_down, one);
        let dst_shift = e.int_const(result_bits, (i * 2) as u64);
        let spread_bit = if i == 0 {
            bit
        } else {
            e.int_arith(IntArithOp::Shl, bit, dst_shift)
        };
        acc = e.int_arith(IntArithOp::Or, acc, spread_bit);
    }

    acc
}

fn lower_unspread(
    e: &mut LLBlockEmitter<'_>,
    value: ValueId,
    value_type: &Type,
    odd_type: &Type,
    even_type: &Type,
    bits: u8,
) -> (ValueId, ValueId) {
    let input_bits = integer_width(value_type, "Unspread");
    let odd_bits = integer_width(odd_type, "Unspread");
    let even_bits = integer_width(even_type, "Unspread");
    assert!(
        input_bits <= 128 && input_bits % 2 == 0,
        "Unspread expects an even integer width up to 128 bits, got {}",
        value_type
    );
    let half_bits = input_bits / 2;
    assert_eq!(
        odd_bits, half_bits,
        "Unspread odd result width must be half the input width"
    );
    assert_eq!(
        even_bits, half_bits,
        "Unspread even result width must be half the input width"
    );
    assert_eq!(
        bits as u32, half_bits,
        "Unspread opcode bit-width does not match result type width"
    );

    let mut odd_acc = e.int_const(input_bits, 0);
    let mut even_acc = e.int_const(input_bits, 0);
    let one = e.int_const(input_bits, 1);

    for i in 0..half_bits {
        let even_src_shift = e.int_const(input_bits, (i * 2) as u64);
        let even_shifted_down = if i == 0 {
            value
        } else {
            e.int_arith(IntArithOp::UShr, value, even_src_shift)
        };
        let even_bit = e.int_arith(IntArithOp::And, even_shifted_down, one);
        let dst_shift = e.int_const(input_bits, i as u64);
        let even_compact_bit = if i == 0 {
            even_bit
        } else {
            e.int_arith(IntArithOp::Shl, even_bit, dst_shift)
        };
        even_acc = e.int_arith(IntArithOp::Or, even_acc, even_compact_bit);

        let odd_src_shift = e.int_const(input_bits, (i * 2 + 1) as u64);
        let odd_shifted_down = e.int_arith(IntArithOp::UShr, value, odd_src_shift);
        let odd_bit = e.int_arith(IntArithOp::And, odd_shifted_down, one);
        let odd_compact_bit = if i == 0 {
            odd_bit
        } else {
            e.int_arith(IntArithOp::Shl, odd_bit, dst_shift)
        };
        odd_acc = e.int_arith(IntArithOp::Or, odd_acc, odd_compact_bit);
    }

    (
        truncate_to_width(e, odd_acc, input_bits, odd_bits),
        truncate_to_width(e, even_acc, input_bits, even_bits),
    )
}

// =============================================================================
// Array lowering helpers
// =============================================================================

/// Lower MkSeq(Array) to heap allocation + element stores.
fn lower_mk_array(
    e: &mut LLBlockEmitter<'_>,
    val_map: &mut HashMap<ValueId, ValueId>,
    result: ValueId,
    elems: &[ValueId],
    elem_type: &Type,
    count: usize,
) {
    let rc_struct = rc_array_struct(elem_type, count);
    let es = elem_struct(elem_type);

    // Allocate
    let arr = e.heap_alloc(rc_struct.clone(), None);

    // Init RC to 1
    let rc_hdr = e.struct_field_ptr(arr, rc_struct.clone(), 0);
    let rc_word = e.struct_field_ptr(rc_hdr, LLStruct::rc_header(), 0);
    let one = e.int_const(64, 1);
    e.ll_store(rc_word, one);

    // Store elements
    let data = e.struct_field_ptr(arr, rc_struct, 2);
    for (i, elem) in elems.iter().enumerate() {
        let idx = e.int_const(64, i as u64);
        let elem_ptr = e.array_elem_ptr(data, es.clone(), idx);
        let ll_elem = val_map[elem];
        e.ll_store(elem_ptr, ll_elem);
    }

    val_map.insert(result, arr);
}

fn lower_to_bytes(
    e: &mut LLBlockEmitter<'_>,
    val_map: &mut HashMap<ValueId, ValueId>,
    result: ValueId,
    value: ValueId,
    endianness: crate::compiler::ssa::Endianness,
    count: usize,
) {
    use crate::compiler::ssa::Endianness;

    let ll_value = val_map[&value];

    // Decompose field → 4 × u64 limbs (little-endian: limb 0 = least significant)
    let limbs_val = e.field_to_limbs(ll_value);
    let limb = [
        e.extract_field(limbs_val, LLStruct::limbs(), 0),
        e.extract_field(limbs_val, LLStruct::limbs(), 1),
        e.extract_field(limbs_val, LLStruct::limbs(), 2),
        e.extract_field(limbs_val, LLStruct::limbs(), 3),
    ];

    // Extract `count` bytes in little-endian order (byte 0 = LSB).
    // Byte i lives in limb[i/8] at bit offset (i%8)*8 for i < 32.
    // Byte positions i >= 32 are structurally zero: the field representation
    // is only 32 bytes, so any higher byte is definitionally 0. Matches the
    // old VM `to_bytes_be` behavior of zero-padding past the field width.
    let zero_byte = e.int_const(8, 0);
    let mut bytes_le = Vec::with_capacity(count);
    for i in 0..count {
        if i >= 32 {
            bytes_le.push(zero_byte);
            continue;
        }
        let limb_idx = i / 8;
        let byte_offset = (i % 8) * 8;
        let shifted = if byte_offset == 0 {
            limb[limb_idx]
        } else {
            let shift = e.int_const(64, byte_offset as u64);
            e.int_arith(IntArithOp::UShr, limb[limb_idx], shift)
        };
        let byte_val = e.truncate(shifted, 8);
        bytes_le.push(byte_val);
    }

    // Allocate RC'd array of u8
    let u8_type = Type::u(8);
    let rc_struct = rc_array_struct(&u8_type, count);
    let es = elem_struct(&u8_type);

    let arr = e.heap_alloc(rc_struct.clone(), None);

    // Init RC to 1
    let rc_hdr = e.struct_field_ptr(arr, rc_struct.clone(), 0);
    let rc_word = e.struct_field_ptr(rc_hdr, LLStruct::rc_header(), 0);
    let one = e.int_const(64, 1);
    e.ll_store(rc_word, one);

    // Store bytes into the array
    let data = e.struct_field_ptr(arr, rc_struct, 2);
    for i in 0..count {
        let src_idx = match endianness {
            Endianness::Big => count - 1 - i,
            Endianness::Little => i,
        };
        let idx = e.int_const(64, i as u64);
        let elem_ptr = e.array_elem_ptr(data, es.clone(), idx);
        e.ll_store(elem_ptr, bytes_le[src_idx]);
    }

    val_map.insert(result, arr);
}

/// Lower MkTuple to heap allocation + RC init + field stores.
fn lower_mk_tuple(
    e: &mut LLBlockEmitter<'_>,
    val_map: &mut HashMap<ValueId, ValueId>,
    result: ValueId,
    elems: &[ValueId],
    element_types: &[Type],
) {
    let rc_struct = rc_tuple_struct(element_types);

    // Allocate
    let tuple_ptr = e.heap_alloc(rc_struct.clone(), None);

    // Init RC to 1
    let rc_hdr = e.struct_field_ptr(tuple_ptr, rc_struct.clone(), 0);
    let rc_word = e.struct_field_ptr(rc_hdr, LLStruct::rc_header(), 0);
    let one = e.int_const(64, 1);
    e.ll_store(rc_word, one);

    // Store each element into its field (fields are at index 1, 2, 3, ...)
    for (i, elem) in elems.iter().enumerate() {
        let field_ptr = e.struct_field_ptr(tuple_ptr, rc_struct.clone(), i + 1);
        let ll_elem = val_map[elem];
        e.ll_store(field_ptr, ll_elem);
    }

    val_map.insert(result, tuple_ptr);
}

/// Lower TupleProj(Static) to StructFieldPtr + Load.
fn lower_tuple_proj(
    e: &mut LLBlockEmitter<'_>,
    val_map: &mut HashMap<ValueId, ValueId>,
    fn_type_info: &FunctionTypeInfo,
    result: ValueId,
    tuple: ValueId,
    field_idx: usize,
) {
    let tuple_type = fn_type_info.get_value_type(tuple);
    let element_types = tuple_type.get_tuple_elements();
    let rc_struct = rc_tuple_struct(&element_types);

    let ll_tuple = val_map[&tuple];

    // Field is at index field_idx + 1 (field 0 is RC header)
    let field_ptr = e.struct_field_ptr(ll_tuple, rc_struct, field_idx + 1);

    // Load the field value
    let field_ll_type = lower_type(&element_types[field_idx]);
    let val = e.ll_load(field_ptr, field_ll_type);

    val_map.insert(result, val);
}

/// Lower ArrayGet to struct_field_ptr + array_elem_ptr + load.
fn lower_array_get(
    e: &mut LLBlockEmitter<'_>,
    val_map: &mut HashMap<ValueId, ValueId>,
    fn_type_info: &FunctionTypeInfo,
    result: ValueId,
    array: ValueId,
    index: ValueId,
) {
    let arr_type = fn_type_info.get_value_type(array);
    let (et, count) = array_info(arr_type);
    let rc_struct = rc_array_struct(et, count);
    let es = elem_struct(et);
    let ll_elem_type = lower_type(et);

    let ll_arr = val_map[&array];
    let ll_idx = val_map[&index];

    // ZExt index from u32 to i64 for pointer arithmetic
    let idx64 = e.zext(ll_idx, 64);

    let data = e.struct_field_ptr(ll_arr, rc_struct, 2);
    let elem_ptr = e.array_elem_ptr(data, es, idx64);
    let val = e.ll_load(elem_ptr, ll_elem_type);

    val_map.insert(result, val);
}

/// Lower ArraySet with copy-on-write semantics.
/// Creates new blocks for the RC check + conditional copy.
///
/// When elements are themselves RC'd (arrays of arrays):
/// - Reuse branch: drop the old element's RC (it's being replaced)
/// - Copy branch: bump RC of every copied element except the one at `index`
///   (which is overwritten), then decrement old array's RC
fn lower_array_set(
    e: &mut LLBlockEmitter<'_>,
    val_map: &mut HashMap<ValueId, ValueId>,
    fn_type_info: &FunctionTypeInfo,
    result: ValueId,
    array: ValueId,
    index: ValueId,
    value: ValueId,
    llssa: &mut LLSSA,
    drop_fns: &mut Vec<DropFnEntry>,
    ad_fns: &mut AdFunctions,
) {
    let arr_type = fn_type_info.get_value_type(array);
    let (et, count) = array_info(arr_type);
    let rc_struct = rc_array_struct(et, count);
    let es = elem_struct(et);
    let elem_is_rc = needs_drop(&et.expr);

    let inner_drop_fn = if elem_is_rc {
        Some(get_or_create_drop_fn(et, llssa, drop_fns, ad_fns))
    } else {
        None
    };
    let inner_rc_struct = if elem_is_rc {
        Some(match &et.expr {
            TypeExpr::Array(inner, n) => rc_array_struct(inner, *n),
            TypeExpr::Tuple(elements) => rc_tuple_struct(elements),
            TypeExpr::WitnessOf(_) => LLStruct::ad_node_base(),
            _ => panic!("Unsupported RC element type: {}", et),
        })
    } else {
        None
    };

    let ll_arr = val_map[&array];
    let ll_idx = val_map[&index];
    let ll_val = val_map[&value];

    // ZExt index
    let idx64 = e.zext(ll_idx, 64);

    // Load RC
    let hdr = e.struct_field_ptr(ll_arr, rc_struct.clone(), 0);
    let rc_ptr = e.struct_field_ptr(hdr, LLStruct::rc_header(), 0);
    let rc = e.ll_load(rc_ptr, LLType::i64());
    let one = e.int_const(64, 1);
    let unique = e.int_eq(rc, one);

    let merge_results = e.build_if_else(
        unique,
        vec![LLType::Ptr],
        // -- Mutate in place --
        |me| {
            let data = me.struct_field_ptr(ll_arr, rc_struct.clone(), 2);
            let slot = me.array_elem_ptr(data, es.clone(), idx64);

            // Drop old element's RC before overwriting
            if let Some(drop_fn) = inner_drop_fn {
                let old_elem = me.ll_load(slot, LLType::Ptr);
                me.call(drop_fn, vec![old_elem], 0);
            }

            me.ll_store(slot, ll_val);
            vec![ll_arr]
        },
        // -- Copy then mutate --
        |ce| {
            // Decrement old array's RC
            let new_rc = ce.int_sub(rc, one);
            ce.ll_store(rc_ptr, new_rc);

            // Allocate new array
            let new_arr = ce.heap_alloc(rc_struct.clone(), None);

            // Init new RC to 1
            let new_hdr = ce.struct_field_ptr(new_arr, rc_struct.clone(), 0);
            let new_rc_ptr = ce.struct_field_ptr(new_hdr, LLStruct::rc_header(), 0);
            ce.ll_store(new_rc_ptr, one);

            // Copy all data
            let old_data = ce.struct_field_ptr(ll_arr, rc_struct.clone(), 2);
            let new_data = ce.struct_field_ptr(new_arr, rc_struct.clone(), 2);
            let count_val = ce.int_const(64, count as u64);
            ce.memcpy(new_data, old_data, es.clone(), Some(count_val));

            // Bump RC of all copied elements except the one we're overwriting
            if inner_drop_fn.is_some() {
                ce.build_counted_loop(count, vec![], |ce, i_val, _| {
                    let elem_ptr = ce.array_elem_ptr(new_data, es.clone(), i_val);
                    let elem_val = ce.ll_load(elem_ptr, LLType::Ptr);
                    let is_replaced = ce.int_eq(i_val, idx64);
                    // Bump RC only if this is NOT the replaced element
                    let not_replaced = ce.not(is_replaced);
                    ce.build_if_else(
                        not_replaced,
                        vec![],
                        |be| {
                            let inner_hdr =
                                be.struct_field_ptr(elem_val, inner_rc_struct.clone().unwrap(), 0);
                            let inner_rc_ptr =
                                be.struct_field_ptr(inner_hdr, LLStruct::rc_header(), 0);
                            let inner_rc = be.ll_load(inner_rc_ptr, LLType::i64());
                            let new_inner_rc = be.int_add(inner_rc, one);
                            be.ll_store(inner_rc_ptr, new_inner_rc);
                            vec![]
                        },
                        |_| vec![],
                    );
                    vec![]
                });
            }

            // Write new value at index
            let new_slot = ce.array_elem_ptr(new_data, es.clone(), idx64);
            ce.ll_store(new_slot, ll_val);

            vec![new_arr]
        },
    );

    val_map.insert(result, merge_results[0]);
}

/// Lower MemOp::Bump(n) -- increment refcount by n.
fn lower_rc_bump(
    e: &mut LLBlockEmitter<'_>,
    val_map: &HashMap<ValueId, ValueId>,
    fn_type_info: &FunctionTypeInfo,
    n: usize,
    value: ValueId,
) {
    let val_type = fn_type_info.get_value_type(value);

    let rc_struct = match &val_type.expr {
        TypeExpr::Array(inner, count) => rc_array_struct(inner, *count),
        TypeExpr::Tuple(elements) => rc_tuple_struct(elements),
        _ => panic!("lower_rc_bump: unexpected type {}", val_type),
    };

    let ll_arr = val_map[&value];

    let hdr = e.struct_field_ptr(ll_arr, rc_struct, 0);
    let rc_ptr = e.struct_field_ptr(hdr, LLStruct::rc_header(), 0);
    let rc = e.ll_load(rc_ptr, LLType::i64());
    let n_val = e.int_const(64, n as u64);
    let new_rc = e.int_add(rc, n_val);
    e.ll_store(rc_ptr, new_rc);
}

/// Lower MemOp::Drop -- call the generated drop function.
fn lower_rc_drop(
    e: &mut LLBlockEmitter<'_>,
    val_map: &HashMap<ValueId, ValueId>,
    fn_type_info: &FunctionTypeInfo,
    value: ValueId,
    llssa: &mut LLSSA,
    drop_fns: &mut Vec<DropFnEntry>,
    ad_fns: &mut AdFunctions,
) {
    let arr_type = fn_type_info.get_value_type(value);
    let drop_fn_id = get_or_create_drop_fn(arr_type, llssa, drop_fns, ad_fns);
    let ll_arr = val_map[&value];
    e.call(drop_fn_id, vec![ll_arr], 0);
}

// =============================================================================
// AD lowering helpers
// =============================================================================

/// Ensure a value is Field-sized ({i64, i64, i64, i64}).
/// Non-Field integer types are zero-extended to i64 and packed into limbs.
fn ensure_field_sized(e: &mut LLBlockEmitter<'_>, ll_val: ValueId, source_type: &Type) -> ValueId {
    if source_type.is_field() || source_type.is_witness_of() {
        return ll_val;
    }
    // U(n)/I(n) → build {val_as_i64, 0, 0, 0}, then FieldFromLimbs
    let val64 = match &source_type.expr {
        TypeExpr::U(bits) | TypeExpr::I(bits) if *bits < 64 => e.zext(ll_val, 64),
        TypeExpr::U(64) | TypeExpr::I(64) => ll_val,
        _ => panic!("ensure_field_sized: unsupported type: {}", source_type),
    };
    let zero = e.int_const(64, 0);
    let limbs = e.mk_struct(LLStruct::limbs(), vec![val64, zero, zero, zero]);
    e.field_from_limbs(limbs)
}

/// Allocate an ADConstNode wrapping a pure field value.
fn lower_ad_const_wrap(
    e: &mut LLBlockEmitter<'_>,
    val_map: &mut HashMap<ValueId, ValueId>,
    result: ValueId,
    value: ValueId,
    source_type: &Type,
) {
    let ll_val = ensure_field_sized(e, val_map[&value], source_type);
    let node_struct = LLStruct::ad_const_node();
    let node = e.heap_alloc(node_struct.clone(), None);

    // RC = 1
    let rc_hdr = e.struct_field_ptr(node, node_struct.clone(), 0);
    let rc_word = e.struct_field_ptr(rc_hdr, LLStruct::rc_header(), 0);
    let one = e.int_const(64, 1);
    e.ll_store(rc_word, one);

    // tag = AD_TAG_CONST
    let tag_ptr = e.struct_field_ptr(node, node_struct.clone(), 1);
    let tag = e.int_const(32, LLStruct::AD_TAG_CONST);
    e.ll_store(tag_ptr, tag);

    // value = field element
    let val_ptr = e.struct_field_ptr(node, node_struct, 2);
    e.ll_store(val_ptr, ll_val);

    val_map.insert(result, node);
}

/// Allocate an ADWitnessNode with the next witness index.
fn lower_ad_fresh_witness(
    e: &mut LLBlockEmitter<'_>,
    val_map: &mut HashMap<ValueId, ValueId>,
    result: ValueId,
) {
    let node_struct = LLStruct::ad_witness_node();
    let node = e.heap_alloc(node_struct.clone(), None);

    // RC = 1
    let rc_hdr = e.struct_field_ptr(node, node_struct.clone(), 0);
    let rc_word = e.struct_field_ptr(rc_hdr, LLStruct::rc_header(), 0);
    let one = e.int_const(64, 1);
    e.ll_store(rc_word, one);

    // tag = AD_TAG_WITNESS
    let tag_ptr = e.struct_field_ptr(node, node_struct.clone(), 1);
    let tag = e.int_const(32, LLStruct::AD_TAG_WITNESS);
    e.ll_store(tag_ptr, tag);

    // index = next witness index from VM
    let index = e.ad_fresh_witness();
    let index64 = e.zext(index, 64);
    let idx_ptr = e.struct_field_ptr(node, node_struct, 2);
    e.ll_store(idx_ptr, index64);

    val_map.insert(result, node);
}

/// Allocate an ADSumNode for two AD values.
fn lower_ad_sum(e: &mut LLBlockEmitter<'_>, ll_a: ValueId, ll_b: ValueId) -> ValueId {
    let node_struct = LLStruct::ad_sum_node();
    let field_elem = LLStruct::field_elem();
    let node = e.heap_alloc(node_struct.clone(), None);

    // RC = 1
    let rc_hdr = e.struct_field_ptr(node, node_struct.clone(), 0);
    let rc_word = e.struct_field_ptr(rc_hdr, LLStruct::rc_header(), 0);
    let one = e.int_const(64, 1);
    e.ll_store(rc_word, one);

    // tag = AD_TAG_SUM
    let tag_ptr = e.struct_field_ptr(node, node_struct.clone(), 1);
    let tag = e.int_const(32, LLStruct::AD_TAG_SUM);
    e.ll_store(tag_ptr, tag);

    // a, b = children
    let a_ptr = e.struct_field_ptr(node, node_struct.clone(), 2);
    e.ll_store(a_ptr, ll_a);
    let b_ptr = e.struct_field_ptr(node, node_struct.clone(), 3);
    e.ll_store(b_ptr, ll_b);

    // da, db, dc = zero
    let zero_field = make_field_zero(e);
    let da_ptr = e.struct_field_ptr(node, node_struct.clone(), 4);
    e.ll_store(da_ptr, zero_field);
    let db_ptr = e.struct_field_ptr(node, node_struct.clone(), 5);
    e.ll_store(db_ptr, zero_field);
    let dc_ptr = e.struct_field_ptr(node, node_struct, 6);
    e.ll_store(dc_ptr, zero_field);

    node
}

/// Allocate an ADMulConstNode: coeff * var.
fn lower_ad_mul_const(
    e: &mut LLBlockEmitter<'_>,
    val_map: &mut HashMap<ValueId, ValueId>,
    result: ValueId,
    const_val: ValueId,
    var: ValueId,
    coeff_type: &Type,
) {
    let ll_coeff = ensure_field_sized(e, val_map[&const_val], coeff_type);
    let ll_var = val_map[&var];
    let node_struct = LLStruct::ad_mul_const_node();
    let node = e.heap_alloc(node_struct.clone(), None);

    // RC = 1
    let rc_hdr = e.struct_field_ptr(node, node_struct.clone(), 0);
    let rc_word = e.struct_field_ptr(rc_hdr, LLStruct::rc_header(), 0);
    let one = e.int_const(64, 1);
    e.ll_store(rc_word, one);

    // tag = AD_TAG_MUL_CONST
    let tag_ptr = e.struct_field_ptr(node, node_struct.clone(), 1);
    let tag = e.int_const(32, LLStruct::AD_TAG_MUL_CONST);
    e.ll_store(tag_ptr, tag);

    // coeff
    let coeff_ptr = e.struct_field_ptr(node, node_struct.clone(), 2);
    e.ll_store(coeff_ptr, ll_coeff);

    // value (child)
    let val_ptr = e.struct_field_ptr(node, node_struct.clone(), 3);
    e.ll_store(val_ptr, ll_var);

    // da, db, dc = zero
    let zero_field = make_field_zero(e);
    let da_ptr = e.struct_field_ptr(node, node_struct.clone(), 4);
    e.ll_store(da_ptr, zero_field);
    let db_ptr = e.struct_field_ptr(node, node_struct.clone(), 5);
    e.ll_store(db_ptr, zero_field);
    let dc_ptr = e.struct_field_ptr(node, node_struct, 6);
    e.ll_store(dc_ptr, zero_field);

    val_map.insert(result, node);
}

/// RC bump for AD nodes: load RC, add n, store.
fn lower_ad_rc_bump(
    e: &mut LLBlockEmitter<'_>,
    val_map: &HashMap<ValueId, ValueId>,
    n: usize,
    value: ValueId,
) {
    let ll_node = val_map[&value];
    let base = LLStruct::ad_node_base();
    let hdr = e.struct_field_ptr(ll_node, base, 0);
    let rc_ptr = e.struct_field_ptr(hdr, LLStruct::rc_header(), 0);
    let rc = e.ll_load(rc_ptr, LLType::i64());
    let n_val = e.int_const(64, n as u64);
    let new_rc = e.int_add(rc, n_val);
    e.ll_store(rc_ptr, new_rc);
}

/// RC drop for AD nodes: call __ad_drop.
fn lower_ad_rc_drop(
    e: &mut LLBlockEmitter<'_>,
    val_map: &HashMap<ValueId, ValueId>,
    value: ValueId,
    llssa: &mut LLSSA,
    ad_fns: &mut AdFunctions,
) {
    let ll_node = val_map[&value];
    let drop_fn = ad_fns.get_drop_fn(llssa);
    e.call(drop_fn, vec![ll_node], 0);
}

/// Construct a zero field element as a struct value.
fn make_field_zero(e: &mut LLBlockEmitter<'_>) -> ValueId {
    let z = e.int_const(64, 0);
    e.mk_struct(LLStruct::field_elem(), vec![z, z, z, z])
}

// =============================================================================
// AD generated function bodies
// =============================================================================

/// Generate all AD dispatch function bodies.
fn generate_all_ad_functions(llssa: &mut LLSSA, ad_fns: &AdFunctions) {
    if let Some(bumps) = &ad_fns.bumps {
        for matrix in [DMatrix::A, DMatrix::B, DMatrix::C] {
            let id = bumps.get(matrix);
            let func = generate_ad_bump_function(matrix);
            let _old = llssa.take_function(id);
            llssa.put_function(id, func);
        }
    }

    if let Some(drop_id) = ad_fns.ad_drop {
        let bumps = ad_fns.bumps.as_ref().expect(
            "ICE: ad_drop allocated without bump functions. \
             This is a bug in AdFunctions — get_drop_fn must call ensure_bumps.",
        );
        let func = generate_ad_drop_function(bumps, drop_id);
        let _old = llssa.take_function(drop_id);
        llssa.put_function(drop_id, func);
    }
}

/// Generate __ad_bump_d{a,b,c}(node: Ptr, amount: FieldElem):
///
/// Reads the tag from the node, then:
///   CONST:     ad_write_const(matrix, node.value, amount)
///   WITNESS:   ad_write_witness(matrix, node.index, amount)
///   SUM:       node.d{a,b,c} += amount  (field add)
///   MUL_CONST: node.d{a,b,c} += amount  (field add)
fn generate_ad_bump_function(matrix: DMatrix) -> LLFunction {
    let name = match matrix {
        DMatrix::A => "__ad_bump_da",
        DMatrix::B => "__ad_bump_db",
        DMatrix::C => "__ad_bump_dc",
    };
    let d_field: usize = match matrix {
        DMatrix::A => 4,
        DMatrix::B => 5,
        DMatrix::C => 6,
    };

    let mut func = new_ll_function(name.to_string());
    let entry = func.get_entry_id();

    {
        let mut e = LLBlockEmitter::new(&mut func, entry);
        let node = e.add_parameter(LLType::Ptr);
        let amount = e.add_parameter(LLType::Struct(LLStruct::field_elem()));

        let tag_ptr = e.struct_field_ptr(node, LLStruct::ad_node_base(), 1);
        let tag = e.ll_load(tag_ptr, LLType::i32());

        // CONST?
        let const_tag = e.int_const(32, LLStruct::AD_TAG_CONST);
        let is_const = e.int_eq(tag, const_tag);
        e.build_if_else(
            is_const,
            vec![],
            |e| {
                // CONST: write to output matrix directly
                let val_ptr = e.struct_field_ptr(node, LLStruct::ad_const_node(), 2);
                let const_val = e.ll_load(val_ptr, LLType::Struct(LLStruct::field_elem()));
                e.ad_write_const(matrix, const_val, amount);
                vec![]
            },
            |e| {
                // Not CONST — check WITNESS
                let tag_ptr = e.struct_field_ptr(node, LLStruct::ad_node_base(), 1);
                let tag = e.ll_load(tag_ptr, LLType::i32());
                let wit_tag = e.int_const(32, LLStruct::AD_TAG_WITNESS);
                let is_wit = e.int_eq(tag, wit_tag);
                e.build_if_else(
                    is_wit,
                    vec![],
                    |e| {
                        // WITNESS: write to output matrix at witness index
                        let idx_ptr = e.struct_field_ptr(node, LLStruct::ad_witness_node(), 2);
                        let index64 = e.ll_load(idx_ptr, LLType::i64());
                        let index32 = e.truncate(index64, 32);
                        e.ad_write_witness(matrix, index32, amount);
                        vec![]
                    },
                    |e| {
                        // SUM or MUL_CONST: accumulate into node.d{a,b,c}
                        let tag_ptr = e.struct_field_ptr(node, LLStruct::ad_node_base(), 1);
                        let tag = e.ll_load(tag_ptr, LLType::i32());
                        let sum_tag = e.int_const(32, LLStruct::AD_TAG_SUM);
                        let is_sum = e.int_eq(tag, sum_tag);
                        e.build_if_else(
                            is_sum,
                            vec![],
                            |e| {
                                // SUM node
                                let d_ptr =
                                    e.struct_field_ptr(node, LLStruct::ad_sum_node(), d_field);
                                let old_d =
                                    e.ll_load(d_ptr, LLType::Struct(LLStruct::field_elem()));
                                let new_d = e.field_arith(FieldArithOp::Add, old_d, amount);
                                e.ll_store(d_ptr, new_d);
                                vec![]
                            },
                            |e| {
                                // MUL_CONST node
                                let d_ptr = e.struct_field_ptr(
                                    node,
                                    LLStruct::ad_mul_const_node(),
                                    d_field,
                                );
                                let old_d =
                                    e.ll_load(d_ptr, LLType::Struct(LLStruct::field_elem()));
                                let new_d = e.field_arith(FieldArithOp::Add, old_d, amount);
                                e.ll_store(d_ptr, new_d);
                                vec![]
                            },
                        );
                        vec![]
                    },
                );
                vec![]
            },
        );

        e.terminate_return(vec![]);
    }

    func
}

/// Generate __ad_drop(node: Ptr):
///
/// Decrements RC. If RC hits zero, dispatches on tag:
///   CONST/WITNESS: free
///   SUM: propagate da/db/dc to children, drop children, free
///   MUL_CONST: propagate da*coeff/db*coeff/dc*coeff to child, drop child, free
fn generate_ad_drop_function(bumps: &AdBumpIds, ad_drop_id: FunctionId) -> LLFunction {
    let mut func = new_ll_function("__ad_drop".to_string());
    let entry = func.get_entry_id();

    let field_type = LLType::Struct(LLStruct::field_elem());

    let bump_da = bumps.da;
    let bump_db = bumps.db;
    let bump_dc = bumps.dc;

    {
        let mut e = LLBlockEmitter::new(&mut func, entry);
        let node = e.add_parameter(LLType::Ptr);

        // Decrement RC
        let rc_hdr = e.struct_field_ptr(node, LLStruct::ad_node_base(), 0);
        let rc_ptr = e.struct_field_ptr(rc_hdr, LLStruct::rc_header(), 0);
        let rc = e.ll_load(rc_ptr, LLType::i64());
        let one = e.int_const(64, 1);
        let new_rc = e.int_sub(rc, one);
        e.ll_store(rc_ptr, new_rc);

        // If RC hit zero, dispatch on tag and tear down
        let zero = e.int_const(64, 0);
        let dead = e.int_eq(new_rc, zero);
        e.build_if_else(
            dead,
            vec![],
            |e| {
                // Read tag
                let tag_ptr = e.struct_field_ptr(node, LLStruct::ad_node_base(), 1);
                let tag = e.ll_load(tag_ptr, LLType::i32());

                // CONST or WITNESS (tag < 2) → just free
                let two = e.int_const(32, 2);
                let is_simple = e.int_ult(tag, two);
                e.build_if_else(
                    is_simple,
                    vec![],
                    |e| {
                        e.free(node);
                        vec![]
                    },
                    |e| {
                        // SUM or MUL_CONST
                        let tag_ptr = e.struct_field_ptr(node, LLStruct::ad_node_base(), 1);
                        let tag = e.ll_load(tag_ptr, LLType::i32());
                        let sum_tag = e.int_const(32, LLStruct::AD_TAG_SUM);
                        let is_sum = e.int_eq(tag, sum_tag);
                        e.build_if_else(
                            is_sum,
                            vec![],
                            |e| {
                                // SUM: propagate da/db/dc to both children
                                let sum = LLStruct::ad_sum_node();
                                let a_ptr = e.struct_field_ptr(node, sum.clone(), 2);
                                let a = e.ll_load(a_ptr, LLType::Ptr);
                                let b_ptr = e.struct_field_ptr(node, sum.clone(), 3);
                                let b = e.ll_load(b_ptr, LLType::Ptr);

                                let da_ptr = e.struct_field_ptr(node, sum.clone(), 4);
                                let da = e.ll_load(da_ptr, field_type.clone());
                                let db_ptr = e.struct_field_ptr(node, sum.clone(), 5);
                                let db = e.ll_load(db_ptr, field_type.clone());
                                let dc_ptr = e.struct_field_ptr(node, sum, 6);
                                let dc = e.ll_load(dc_ptr, field_type.clone());

                                e.call(bump_da, vec![a, da], 0);
                                e.call(bump_db, vec![a, db], 0);
                                e.call(bump_dc, vec![a, dc], 0);
                                e.call(bump_da, vec![b, da], 0);
                                e.call(bump_db, vec![b, db], 0);
                                e.call(bump_dc, vec![b, dc], 0);
                                e.call(ad_drop_id, vec![a], 0);
                                e.call(ad_drop_id, vec![b], 0);
                                e.free(node);
                                vec![]
                            },
                            |e| {
                                // MUL_CONST: scale sensitivities by coeff, propagate
                                let mul = LLStruct::ad_mul_const_node();
                                let coeff_ptr = e.struct_field_ptr(node, mul.clone(), 2);
                                let coeff = e.ll_load(coeff_ptr, field_type.clone());
                                let val_ptr = e.struct_field_ptr(node, mul.clone(), 3);
                                let child = e.ll_load(val_ptr, LLType::Ptr);

                                let da_ptr = e.struct_field_ptr(node, mul.clone(), 4);
                                let da = e.ll_load(da_ptr, field_type.clone());
                                let db_ptr = e.struct_field_ptr(node, mul.clone(), 5);
                                let db = e.ll_load(db_ptr, field_type.clone());
                                let dc_ptr = e.struct_field_ptr(node, mul, 6);
                                let dc = e.ll_load(dc_ptr, field_type.clone());

                                let scaled_da = e.field_arith(FieldArithOp::Mul, da, coeff);
                                let scaled_db = e.field_arith(FieldArithOp::Mul, db, coeff);
                                let scaled_dc = e.field_arith(FieldArithOp::Mul, dc, coeff);

                                e.call(bump_da, vec![child, scaled_da], 0);
                                e.call(bump_db, vec![child, scaled_db], 0);
                                e.call(bump_dc, vec![child, scaled_dc], 0);
                                e.call(ad_drop_id, vec![child], 0);
                                e.free(node);
                                vec![]
                            },
                        );
                        vec![]
                    },
                );
                vec![]
            },
            |_| vec![],
        );

        e.terminate_return(vec![]);
    }

    func
}

// =============================================================================
// Drop function generation
// =============================================================================

/// Generate all drop function bodies after user functions are lowered.
fn generate_all_drop_functions(llssa: &mut LLSSA, drop_fns: &[DropFnEntry]) {
    for entry in drop_fns {
        let func = match &entry.hlssa_type.expr {
            TypeExpr::Array(..) => generate_drop_function_for_array(&entry.hlssa_type, drop_fns),
            TypeExpr::Tuple(elements) => {
                generate_drop_function_for_tuple(elements, &entry.hlssa_type, drop_fns)
            }
            // WitnessOf points to ad_drop, whose body is generated by generate_all_ad_functions
            TypeExpr::WitnessOf(_) => continue,
            other => panic!("No drop function generator for type: {:?}", other),
        };
        let _old = llssa.take_function(entry.fn_id);
        llssa.put_function(entry.fn_id, func);
    }
}

fn needs_drop(expr: &TypeExpr) -> bool {
    matches!(
        expr,
        TypeExpr::Array(..) | TypeExpr::Tuple(..) | TypeExpr::WitnessOf(..)
    )
}

/// Generate a drop function for an array type.
///
/// Pseudocode:
///   fn drop(ptr):
///     rc = --ptr.header.rc
///     if rc == 0:
///       for i in 0..count:        // only if elements are RC'd
///         drop(ptr.data[i])
///       free(ptr)
///     return
fn generate_drop_function_for_array(ty: &Type, drop_fns: &[DropFnEntry]) -> LLFunction {
    let (et, count) = array_info(ty);
    let rc_struct = rc_array_struct(et, count);
    let es = elem_struct(et);
    let elem_is_rc = needs_drop(&et.expr);

    let mut func = new_ll_function(format!("drop_{}", ty));
    let entry = func.get_entry_id();

    {
        let mut e = LLBlockEmitter::new(&mut func, entry);

        let ptr = e.add_parameter(LLType::Ptr);

        // Decrement RC
        let hdr = e.struct_field_ptr(ptr, rc_struct.clone(), 0);
        let rc_ptr = e.struct_field_ptr(hdr, LLStruct::rc_header(), 0);
        let rc = e.ll_load(rc_ptr, LLType::i64());
        let one = e.int_const(64, 1);
        let new_rc = e.int_sub(rc, one);
        e.ll_store(rc_ptr, new_rc);

        // If RC hit zero, drop inner elements and free
        let zero = e.int_const(64, 0);
        let dead = e.int_eq(new_rc, zero);
        e.build_if_else(
            dead,
            vec![],
            |e| {
                if elem_is_rc {
                    let inner_drop_fn = drop_fns
                        .iter()
                        .find(|entry| entry.hlssa_type == *et)
                        .expect("inner drop fn should exist")
                        .fn_id;

                    let data = e.struct_field_ptr(ptr, rc_struct, 2);
                    e.build_counted_loop(count, vec![], |e, i_val, _| {
                        let elem_ptr = e.array_elem_ptr(data, es.clone(), i_val);
                        let elem_val = e.ll_load(elem_ptr, LLType::Ptr);
                        e.call(inner_drop_fn, vec![elem_val], 0);
                        vec![]
                    });
                }
                e.free(ptr);
                vec![]
            },
            |_| vec![],
        );

        e.terminate_return(vec![]);
    }

    func
}

/// Generate a drop function for an RC'd tuple.
///
/// Pseudocode:
///   fn drop(ptr):
///     rc = --ptr.header.rc
///     if rc == 0:
///       for each heap-allocated field i:
///         drop_FieldType(ptr.field[i+1])
///       free(ptr)
///     return
fn generate_drop_function_for_tuple(
    element_types: &[Type],
    ty: &Type,
    drop_fns: &[DropFnEntry],
) -> LLFunction {
    let rc_struct = rc_tuple_struct(element_types);

    let mut func = new_ll_function(format!("drop_{}", ty));
    let entry = func.get_entry_id();

    {
        let mut e = LLBlockEmitter::new(&mut func, entry);
        let ptr = e.add_parameter(LLType::Ptr);

        // Decrement RC
        let hdr = e.struct_field_ptr(ptr, rc_struct.clone(), 0);
        let rc_ptr = e.struct_field_ptr(hdr, LLStruct::rc_header(), 0);
        let rc = e.ll_load(rc_ptr, LLType::i64());
        let one = e.int_const(64, 1);
        let new_rc = e.int_sub(rc, one);
        e.ll_store(rc_ptr, new_rc);

        // If RC hit zero, drop inner heap-allocated elements and free
        let zero = e.int_const(64, 0);
        let dead = e.int_eq(new_rc, zero);
        e.build_if_else(
            dead,
            vec![],
            |e| {
                for (i, elem_ty) in element_types.iter().enumerate() {
                    if needs_drop(&elem_ty.expr) {
                        let inner_drop_fn = drop_fns
                            .iter()
                            .find(|entry| entry.hlssa_type == *elem_ty)
                            .expect("inner drop fn should exist for tuple element")
                            .fn_id;

                        // Field is at index i + 1 (field 0 is RC header)
                        let field_ptr = e.struct_field_ptr(ptr, rc_struct.clone(), i + 1);
                        let field_val = e.ll_load(field_ptr, LLType::Ptr);
                        e.call(inner_drop_fn, vec![field_val], 0);
                    }
                }
                e.free(ptr);
                vec![]
            },
            |_| vec![],
        );

        e.terminate_return(vec![]);
    }

    func
}

// =============================================================================
// Terminator lowering
// =============================================================================

fn lower_terminator(
    terminator: &Terminator,
    e: &mut LLBlockEmitter<'_>,
    val_map: &HashMap<ValueId, ValueId>,
    block_map: &HashMap<BlockId, BlockId>,
) {
    match terminator {
        Terminator::Jmp(target, args) => {
            let ll_target = block_map[target];
            let ll_args: Vec<ValueId> = args.iter().map(|a| val_map[a]).collect();
            e.terminate_jmp(ll_target, ll_args);
        }
        Terminator::JmpIf(cond, then_target, else_target) => {
            let ll_cond = val_map[cond];
            let ll_then = block_map[then_target];
            let ll_else = block_map[else_target];
            e.terminate_jmp_if(ll_cond, ll_then, ll_else);
        }
        Terminator::Return(values) => {
            let ll_values: Vec<ValueId> = values.iter().map(|v| val_map[v]).collect();
            e.terminate_return(ll_values);
        }
    }
}

// =============================================================================
// Lookup generated function bodies
// =============================================================================

/// Generate all lookup helper function bodies that were registered during
/// per-function lowering.
fn generate_all_lookup_functions(
    llssa: &mut LLSSA,
    lookup_fns: &LookupFunctions,
    layout: Option<(WitnessLayout, ConstraintsLayout)>,
    ad_fns: &mut AdFunctions,
    forward_a_base_global: Option<usize>,
) {
    if let Some(id) = lookup_fns.rngchk_8 {
        let table_idx_global = lookup_fns
            .rngchk_8_table_idx_global
            .expect("rangecheck-8 helper registered without internal table-id global");
        let func = generate_rngchk_8_function(table_idx_global);
        let _old = llssa.take_function(id);
        llssa.put_function(id, func);
    }

    for (bits, id) in &lookup_fns.spread {
        let table_idx_global = lookup_fns
            .spread_table_idx_globals
            .get(bits)
            .copied()
            .expect("spread helper registered without internal table-id global");
        let a_base_global =
            forward_a_base_global.expect("spread helper registered without forward A base global");
        let func = generate_spread_lookup_function(*bits, table_idx_global, a_base_global);
        let _old = llssa.take_function(*id);
        llssa.put_function(*id, func);
    }

    if let Some(call_id) = lookup_fns.drngchk_8_call {
        let inv_cnst_off_global = lookup_fns
            .drngchk_8_inv_cnst_off_global
            .expect("AD rangecheck-8 helper registered without internal offset global");
        let (witness_layout, constraints_layout) =
            layout.expect("R1CS layout required to generate AD rangecheck-8 helper");
        let bump_db_id = ad_fns.get_bump_fn(DMatrix::B, llssa);
        let bump_dc_id = ad_fns.get_bump_fn(DMatrix::C, llssa);
        let call_fn = generate_drngchk_8_ad_call(
            inv_cnst_off_global,
            witness_layout,
            constraints_layout,
            bump_db_id,
            bump_dc_id,
        );
        let _old = llssa.take_function(call_id);
        llssa.put_function(call_id, call_fn);
    }

    for (bits, call_id) in &lookup_fns.dspread_call {
        let inv_cnst_off_global = lookup_fns
            .dspread_inv_cnst_off_globals
            .get(bits)
            .copied()
            .expect("AD spread helper registered without internal offset global");
        let (witness_layout, constraints_layout) =
            layout.expect("R1CS layout required to generate AD spread helper");
        let bump_da_id = ad_fns.get_bump_fn(DMatrix::A, llssa);
        let bump_db_id = ad_fns.get_bump_fn(DMatrix::B, llssa);
        let bump_dc_id = ad_fns.get_bump_fn(DMatrix::C, llssa);
        let call_fn = generate_dspread_ad_call(
            *bits,
            inv_cnst_off_global,
            witness_layout,
            constraints_layout,
            bump_da_id,
            bump_db_id,
            bump_dc_id,
        );
        let _old = llssa.take_function(*call_id);
        llssa.put_function(*call_id, call_fn);
    }
}

/// Extract the four raw limbs of a Field value.
fn field_limbs(e: &mut LLBlockEmitter<'_>, val: ValueId) -> (ValueId, ValueId, ValueId, ValueId) {
    let limbs = e.field_to_limbs(val);
    let l0 = e.extract_field(limbs.clone(), LLStruct::limbs(), 0);
    let l1 = e.extract_field(limbs.clone(), LLStruct::limbs(), 1);
    let l2 = e.extract_field(limbs.clone(), LLStruct::limbs(), 2);
    let l3 = e.extract_field(limbs, LLStruct::limbs(), 3);
    (l0, l1, l2, l3)
}

/// Assert that `ok` is true.
fn assert(e: &mut LLBlockEmitter<'_>, ok: ValueId) {
    e.build_if_else(
        ok,
        vec![],
        |_ok_e| vec![],
        |bad_e| {
            bad_e.trap();
            vec![]
        },
    );
}

/// Build a Field whose low limb is `lo` and upper limbs are zero.
fn u64_as_field(e: &mut LLBlockEmitter<'_>, lo: ValueId) -> ValueId {
    let zero = e.int_const(64, 0);
    let limbs = e.mk_struct(LLStruct::limbs(), vec![lo, zero, zero, zero]);
    e.field_from_limbs(limbs)
}

fn field_neg_via_sub(e: &mut LLBlockEmitter<'_>, value: ValueId) -> ValueId {
    let zero_i64 = e.int_const(64, 0);
    let zero_field = u64_as_field(e, zero_i64);
    e.field_arith(FieldArithOp::Sub, zero_field, value)
}

/// Emit: `bump_u64_at(cursor, delta)` — advance the cursor stored at
/// `cursor_slot_ptr` by one Field, writing `(value, 0, 0, 0)` as raw limbs
/// into the slot the old cursor pointed at.
fn write_tape_entry_u64(e: &mut LLBlockEmitter<'_>, cursor_field: usize, value_u64: ValueId) {
    let cursor_slot = e.witgen_vm_field_ptr(cursor_field);
    let cursor = e.ll_load(cursor_slot, LLType::Ptr);
    // Store value into low limb via struct_field_ptr.
    let low_ptr = e.struct_field_ptr(cursor, LLStruct::field_elem(), 0);
    e.ll_store(low_ptr, value_u64);
    // Advance cursor by one Field (4 i64s).
    let one = e.int_const(32, 1);
    let next = e.array_elem_ptr(cursor, LLStruct::field_elem(), one);
    e.ll_store(cursor_slot, next);
}

fn write_tape_entry_field(e: &mut LLBlockEmitter<'_>, cursor_field: usize, value: ValueId) {
    let cursor_slot = e.witgen_vm_field_ptr(cursor_field);
    let cursor = e.ll_load(cursor_slot, LLType::Ptr);
    e.ll_store(cursor, value);
    let one = e.int_const(32, 1);
    let next = e.array_elem_ptr(cursor, LLStruct::field_elem(), one);
    e.ll_store(cursor_slot, next);
}

/// Pointer to the host-visible table metadata slot for runtime `table_idx`.
/// Subsequent field reads go through
/// `struct_field_ptr(slot_ptr, table_info_slot(), TABLE_INFO_*)`.
fn witgen_table_info_ptr(e: &mut LLBlockEmitter<'_>, table_idx: ValueId) -> ValueId {
    let arr_base_slot = e.witgen_vm_field_ptr(LLStruct::WITGEN_VM_TABLES_PTR);
    let arr_base = e.ll_load(arr_base_slot, LLType::Ptr);
    e.array_elem_ptr(arr_base, LLStruct::table_info_slot(), table_idx)
}

/// GEP a field within a `TableInfoSlot` (already-pointed-to). Wraps the
/// `struct_field_ptr` boilerplate so call sites read as
/// `table_info_field_ptr(e, slot_ptr, LLStruct::TABLE_INFO_MULTS_BASE)`.
fn table_info_field_ptr(e: &mut LLBlockEmitter<'_>, slot_ptr: ValueId, field: usize) -> ValueId {
    e.struct_field_ptr(slot_ptr, LLStruct::table_info_slot(), field)
}

fn generate_spread_lookup_function(
    bits: u8,
    table_idx_global: usize,
    a_base_global: usize,
) -> LLFunction {
    assert!(
        bits <= 16,
        "Spread lookup helper currently supports bit-widths up to 16, got {}",
        bits
    );
    let length = 1usize << bits;
    let mut func = new_ll_function(format!("__spread_{}_lookup", bits));
    let entry = func.get_entry_id();

    {
        let mut e = LLBlockEmitter::new(&mut func, entry);
        let key_field = e.add_parameter(LLType::Struct(LLStruct::field_elem()));
        let result_field = e.add_parameter(LLType::Struct(LLStruct::field_elem()));
        let flag_field = e.add_parameter(LLType::Struct(LLStruct::field_elem()));

        let (key_l0, key_l1, key_l2, key_l3) = field_limbs(&mut e, key_field);
        let (flag_l0, flag_l1, flag_l2, flag_l3) = field_limbs(&mut e, flag_field);
        let key = key_l0;
        let flag_u64 = flag_l0;

        let zero_i64 = e.int_const(64, 0);
        let table_len_i64 = e.int_const(64, length as u64);
        let in_range = e.int_ult(key, table_len_i64);
        assert(&mut e, in_range);
        for high in [key_l1, key_l2, key_l3, flag_l1, flag_l2, flag_l3] {
            let ok = e.int_eq(high, zero_i64);
            assert(&mut e, ok);
        }

        let snap_idx_slot = e.global_addr(table_idx_global);
        let snap_idx_plus_one = e.ll_load(snap_idx_slot, LLType::i32());
        let zero_i32 = e.int_const(32, 0);
        let is_unalloc = e.int_eq(snap_idx_plus_one, zero_i32);
        let merge = e.build_if_else(
            is_unalloc,
            vec![LLType::Ptr, LLType::Int(32)],
            |e| {
                let mults_cursor_slot = e.witgen_vm_field_ptr(LLStruct::WITGEN_VM_MULTS_CURSOR);
                let mults_base = e.ll_load(mults_cursor_slot, LLType::Ptr);
                let len_slot = e.witgen_vm_field_ptr(LLStruct::WITGEN_VM_TABLES_LEN);
                let table_idx = e.ll_load(len_slot, LLType::i32());
                let cap_slot = e.witgen_vm_field_ptr(LLStruct::WITGEN_VM_TABLES_CAP);
                let tables_cap = e.ll_load(cap_slot, LLType::i32());
                let has_capacity = e.int_ult(table_idx, tables_cap);
                assert(e, has_capacity);
                let cnst_cursor_slot =
                    e.witgen_vm_field_ptr(LLStruct::WITGEN_VM_CURRENT_CNST_TABLES_OFF);
                let inv_cnst_off = e.ll_load(cnst_cursor_slot, LLType::i32());
                let wit_cursor_slot =
                    e.witgen_vm_field_ptr(LLStruct::WITGEN_VM_CURRENT_WIT_TABLES_OFF);
                let inv_wit_off = e.ll_load(wit_cursor_slot, LLType::i32());

                let slot_ptr = witgen_table_info_ptr(e, table_idx);
                let one_i32 = e.int_const(32, 1);
                let table_len_i32 = e.int_const(32, length as u64);
                let table_wit_bump = e.int_const(32, (2 * length) as u64);
                let table_cnst_bump = e.int_const(32, (2 * length + 1) as u64);
                let table_info_writes = [
                    (LLStruct::TABLE_INFO_MULTS_BASE, mults_base),
                    (LLStruct::TABLE_INFO_INV_CNST_OFF, inv_cnst_off),
                    (LLStruct::TABLE_INFO_INV_WIT_OFF, inv_wit_off),
                    (LLStruct::TABLE_INFO_NUM_INDICES, one_i32),
                    (LLStruct::TABLE_INFO_NUM_VALUES, one_i32),
                    (LLStruct::TABLE_INFO_LENGTH, table_len_i32),
                ];
                for (field, value) in table_info_writes {
                    let p = table_info_field_ptr(e, slot_ptr, field);
                    e.ll_store(p, value);
                }

                let a_base_slot = e.global_addr(a_base_global);
                let a_base = e.ll_load(a_base_slot, LLType::Ptr);
                let input_ty = Type::u(bits as usize);
                let result_ty = Type::u(bits as usize * 2);
                e.build_counted_loop(length, vec![], |e, i_i64, _| {
                    let i_key = e.truncate(i_i64, bits as u32);
                    let spread = lower_spread(e, i_key, &input_ty, &result_ty, bits);
                    let spread_u64 = if bits as u32 * 2 == 64 {
                        spread
                    } else {
                        e.zext(spread, 64)
                    };
                    let spread_field = u64_as_field(e, spread_u64);
                    let i_i32 = e.truncate(i_i64, 32);
                    let two_i32 = e.int_const(32, 2);
                    let doubled_i = e.int_arith(IntArithOp::Mul, i_i32, two_i32);
                    let table_idx = e.int_arith(IntArithOp::Add, inv_cnst_off, doubled_i);
                    let table_slot = e.array_elem_ptr(a_base, LLStruct::field_elem(), table_idx);
                    e.ll_store(table_slot, spread_field);
                    vec![]
                });

                let table_idx_plus_one = e.int_arith(IntArithOp::Add, table_idx, one_i32);
                e.ll_store(snap_idx_slot, table_idx_plus_one);

                let next_mults =
                    e.array_elem_ptr(mults_base, LLStruct::field_elem(), table_len_i32);
                e.ll_store(mults_cursor_slot, next_mults);
                e.ll_store(len_slot, table_idx_plus_one);
                let next_cnst = e.int_arith(IntArithOp::Add, inv_cnst_off, table_cnst_bump);
                e.ll_store(cnst_cursor_slot, next_cnst);
                let next_wit = e.int_arith(IntArithOp::Add, inv_wit_off, table_wit_bump);
                e.ll_store(wit_cursor_slot, next_wit);

                vec![mults_base, table_idx]
            },
            |e| {
                let snap_idx_slot = e.global_addr(table_idx_global);
                let snap_idx_plus_one = e.ll_load(snap_idx_slot, LLType::i32());
                let one_i32 = e.int_const(32, 1);
                let table_idx = e.int_arith(IntArithOp::Sub, snap_idx_plus_one, one_i32);
                let slot_ptr = witgen_table_info_ptr(e, table_idx);
                let mults_p = table_info_field_ptr(e, slot_ptr, LLStruct::TABLE_INFO_MULTS_BASE);
                let mults_base = e.ll_load(mults_p, LLType::Ptr);
                vec![mults_base, table_idx]
            },
        );
        let mults_base = merge[0];
        let table_idx_i32 = merge[1];

        let slot_ptr = e.array_elem_ptr(mults_base, LLStruct::field_elem(), key);
        let low_ptr = e.struct_field_ptr(slot_ptr, LLStruct::field_elem(), 0);
        let old_low = e.ll_load(low_ptr, LLType::i64());
        let new_low = e.int_add(old_low, flag_u64);
        e.ll_store(low_ptr, new_low);

        let table_id = e.zext(table_idx_i32, 64);
        write_tape_entry_u64(&mut e, LLStruct::WITGEN_VM_LOOKUPS_A, table_id);
        write_tape_entry_field(&mut e, LLStruct::WITGEN_VM_LOOKUPS_B, result_field);
        write_tape_entry_u64(&mut e, LLStruct::WITGEN_VM_LOOKUPS_C, zero_i64);
        write_tape_entry_u64(&mut e, LLStruct::WITGEN_VM_LOOKUPS_A, table_id);
        write_tape_entry_u64(&mut e, LLStruct::WITGEN_VM_LOOKUPS_B, key);
        write_tape_entry_u64(&mut e, LLStruct::WITGEN_VM_LOOKUPS_C, flag_u64);

        e.terminate_return(vec![]);
    }

    func
}

/// Generate __rngchk_8(val: FieldElem, flag: FieldElem):
///
/// Emits one entry into the LogUp lookup argument for the 8-bit rangecheck
/// table. Extracts the raw low u64 limb of both `val` and `flag`, asserts
/// upper limbs are zero and `val < 256`, then lazily allocates a runtime
/// table id for rangecheck-8. On first use it appends one `TableInfoSlot`
/// at `tables[tables_len]`, stores that id in the rangecheck sentinel, and
/// bumps the table-region cursors. On every call it bumps the multiplicity
/// slot at `mults_base[key]` and writes `(table_id, key, flag)` into the
/// forward lookup tape.
///
/// Mirrors `rngchk_8_field` in `vm/src/bytecode.rs`: the table id and
/// multiplicities-base pointer come from runtime cursors, not compile-time
/// constants, so adding other lookup kinds (which will allocate their own
/// table regions before or after this one) doesn't break the rangecheck-8
/// path.
///
/// Phase 2 — which runs on the host after WASM returns — fixes the raw-u64
/// multiplicity slots into Montgomery form and materializes the per-slot
/// inverses + sum constraint.
fn generate_rngchk_8_function(table_idx_global: usize) -> LLFunction {
    let mut func = new_ll_function("__rngchk_8".to_string());
    let entry = func.get_entry_id();

    {
        let mut e = LLBlockEmitter::new(&mut func, entry);
        let val = e.add_parameter(LLType::Struct(LLStruct::field_elem()));
        let flag = e.add_parameter(LLType::Struct(LLStruct::field_elem()));

        let (val_l0, val_l1, val_l2, val_l3) = field_limbs(&mut e, val);
        let (flag_l0, flag_l1, flag_l2, flag_l3) = field_limbs(&mut e, flag);
        let flag_u64 = flag_l0;
        let key = val_l0;

        // Invariants: val < 256, all high limbs zero (for both val and flag).
        let zero_i64 = e.int_const(64, 0);
        let table_len_i64 = e.int_const(64, 256);
        let in_range = e.int_ult(val_l0, table_len_i64);
        assert(&mut e, in_range);
        for high in [val_l1, val_l2, val_l3, flag_l1, flag_l2, flag_l3] {
            let ok = e.int_eq(high, zero_i64);
            assert(&mut e, ok);
        }

        // First-use check: the helper's private zero-initialized global
        // stores `table_idx + 1`. Zero means rangecheck-8 has not claimed a
        // runtime table id yet.
        //
        // Static metadata for rangecheck-8: length=256, num_indices=1,
        // num_values=0 (width-1 table; Phase 2 dispatches on num_values).
        // Constraints footprint = 256 + 1 = 257 (sum constraint).
        let snap_idx_slot = e.global_addr(table_idx_global);
        let snap_idx_plus_one = e.ll_load(snap_idx_slot, LLType::i32());
        let zero_i32 = e.int_const(32, 0);
        let is_unalloc = e.int_eq(snap_idx_plus_one, zero_i32);
        let merge = e.build_if_else(
            is_unalloc,
            vec![LLType::Ptr, LLType::Int(32)],
            |e| {
                let mults_cursor_slot = e.witgen_vm_field_ptr(LLStruct::WITGEN_VM_MULTS_CURSOR);
                let mults_base = e.ll_load(mults_cursor_slot, LLType::Ptr);
                let len_slot = e.witgen_vm_field_ptr(LLStruct::WITGEN_VM_TABLES_LEN);
                let table_idx = e.ll_load(len_slot, LLType::i32());
                let cap_slot = e.witgen_vm_field_ptr(LLStruct::WITGEN_VM_TABLES_CAP);
                let tables_cap = e.ll_load(cap_slot, LLType::i32());
                let has_capacity = e.int_ult(table_idx, tables_cap);
                assert(e, has_capacity);
                let cnst_cursor_slot =
                    e.witgen_vm_field_ptr(LLStruct::WITGEN_VM_CURRENT_CNST_TABLES_OFF);
                let inv_cnst_off = e.ll_load(cnst_cursor_slot, LLType::i32());
                let wit_cursor_slot =
                    e.witgen_vm_field_ptr(LLStruct::WITGEN_VM_CURRENT_WIT_TABLES_OFF);
                let inv_wit_off = e.ll_load(wit_cursor_slot, LLType::i32());

                // Append the registry slot for this runtime table id.
                let slot_ptr = witgen_table_info_ptr(e, table_idx);
                let one_i32 = e.int_const(32, 1);
                let bump_256 = e.int_const(32, 256);
                let bump_257 = e.int_const(32, 257);
                let zero_i32 = e.int_const(32, 0);
                let table_info_writes = [
                    (LLStruct::TABLE_INFO_MULTS_BASE, mults_base),
                    (LLStruct::TABLE_INFO_INV_CNST_OFF, inv_cnst_off),
                    (LLStruct::TABLE_INFO_INV_WIT_OFF, inv_wit_off),
                    (LLStruct::TABLE_INFO_NUM_INDICES, one_i32),
                    (LLStruct::TABLE_INFO_NUM_VALUES, zero_i32),
                    (LLStruct::TABLE_INFO_LENGTH, bump_256),
                ];
                for (field, value) in table_info_writes {
                    let p = table_info_field_ptr(e, slot_ptr, field);
                    e.ll_store(p, value);
                }
                let table_idx_plus_one = e.int_arith(IntArithOp::Add, table_idx, one_i32);
                e.ll_store(snap_idx_slot, table_idx_plus_one);

                // Bump cursors by this kind's footprint.
                let next_mults = e.array_elem_ptr(mults_base, LLStruct::field_elem(), bump_256);
                e.ll_store(mults_cursor_slot, next_mults);
                e.ll_store(len_slot, table_idx_plus_one);
                let next_cnst = e.int_arith(IntArithOp::Add, inv_cnst_off, bump_257);
                e.ll_store(cnst_cursor_slot, next_cnst);
                let next_wit = e.int_arith(IntArithOp::Add, inv_wit_off, bump_256);
                e.ll_store(wit_cursor_slot, next_wit);

                vec![mults_base, table_idx]
            },
            |e| {
                // Already claimed — reload this table's metadata.
                let snap_idx_slot = e.global_addr(table_idx_global);
                let snap_idx_plus_one = e.ll_load(snap_idx_slot, LLType::i32());
                let one_i32 = e.int_const(32, 1);
                let table_idx = e.int_arith(IntArithOp::Sub, snap_idx_plus_one, one_i32);
                let slot_ptr = witgen_table_info_ptr(e, table_idx);
                let mults_p = table_info_field_ptr(e, slot_ptr, LLStruct::TABLE_INFO_MULTS_BASE);
                let mults_base = e.ll_load(mults_p, LLType::Ptr);
                vec![mults_base, table_idx]
            },
        );
        let mults_base = merge[0];
        let table_idx_i32 = merge[1];

        // Bump: multiplicities[key].low_u64 += flag_u64. `mults_base` is the
        // snapshotted base for *this* table's slab, so we offset by `key`
        // Field-elements from it.
        let slot_ptr = e.array_elem_ptr(mults_base, LLStruct::field_elem(), key);
        let low_ptr = e.struct_field_ptr(slot_ptr, LLStruct::field_elem(), 0);
        let old_low = e.ll_load(low_ptr, LLType::i64());
        let new_low = e.int_add(old_low, flag_u64);
        e.ll_store(low_ptr, new_low);

        // Append one tape entry: (table_id, key, flag_u64) into (a, b, c).
        // The table id is the snapshotted `vm.rgchk_8` analog — once a real
        // value, since the merge-block parameters always carry an allocated
        // snapshot. The tape stores u64s, so widen the i32 snapshot.
        let table_id = e.zext(table_idx_i32, 64);
        write_tape_entry_u64(&mut e, LLStruct::WITGEN_VM_LOOKUPS_A, table_id);
        write_tape_entry_u64(&mut e, LLStruct::WITGEN_VM_LOOKUPS_B, key);
        write_tape_entry_u64(&mut e, LLStruct::WITGEN_VM_LOOKUPS_C, flag_u64);

        e.terminate_return(vec![]);
    }

    func
}

/// Read `ad_coeffs_base[offset]` via random access (does not advance the
/// AdCoeffs cursor). Used by AD helpers that need to peek at a coefficient at
/// a layout-determined absolute index.
fn ad_read_coeff_at(e: &mut LLBlockEmitter<'_>, offset: usize) -> ValueId {
    let base_slot = e.ad_vm_field_ptr(LLStruct::AD_VM_COEFFS_BASE);
    let base = e.ll_load(base_slot, LLType::Ptr);
    let idx = e.int_const(32, offset as u64);
    let slot = e.array_elem_ptr(base, LLStruct::field_elem(), idx);
    e.ll_load(slot, LLType::Struct(LLStruct::field_elem()))
}

/// Post-increment `AdCurrentLookupWitOff` by one, returning the old i32 value.
fn ad_next_lookup_wit_off(e: &mut LLBlockEmitter<'_>) -> ValueId {
    let slot = e.ad_vm_field_ptr(LLStruct::AD_VM_CURRENT_LOOKUP_WIT_OFF);
    let idx = e.ll_load(slot, LLType::i32());
    let one = e.int_const(32, 1);
    let next = e.int_add(idx, one);
    e.ll_store(slot, next);
    idx
}

/// Emit SSA to read `ad_coeffs[offset_i32]` via random access (GEP on
/// AdCoeffsBase; no cursor advance). Used when we need a coefficient whose
/// absolute index is determined by R1CS layout, not by the main AD cursor.
fn ad_read_coeff_at_dyn(e: &mut LLBlockEmitter<'_>, offset_i32: ValueId) -> ValueId {
    let base_slot = e.ad_vm_field_ptr(LLStruct::AD_VM_COEFFS_BASE);
    let base = e.ll_load(base_slot, LLType::Ptr);
    let slot = e.array_elem_ptr(base, LLStruct::field_elem(), offset_i32);
    e.ll_load(slot, LLType::Struct(LLStruct::field_elem()))
}

/// Emit the per-element AD bumps that allocate the rangecheck-8 table region.
///
/// Mirrors the first-call branch of `drngchk_8_field` in `vm/src/bytecode.rs`:
/// snapshots the three table-region cursors out of the AD VM struct, advances
/// them by the rangecheck-8 footprint (256 / 256 / 257), and runs the loop
/// that bumps `out_da`, `out_db`, `out_dc` for each of the 256 entries plus
/// the post-loop `out_db[0] += inv_sum_coeff`. Stores `inv_cnst_off` in the
/// AD rangecheck sentinel so subsequent calls find the same region. Returns
/// `inv_cnst_off` so the caller can immediately read `inv_sum_coeff` from it
/// without reloading.
///
/// Reads `logup_challenge_off` from `witness_layout` because it's a structural
/// constant of the layout (always at `challenges_start`), not a
/// dynamically-allocated region.
fn emit_rngchk_8_ad_init_body(
    e: &mut LLBlockEmitter<'_>,
    inv_cnst_off_global: usize,
    witness_layout: WitnessLayout,
) -> ValueId {
    // Snapshot the three table-region cursors. These are the AD analogue of
    // VM `current_cnst_tables_off` / `current_wit_tables_off` /
    // `current_wit_multiplicities_off`. The host seeds them at
    // {constraints,witness}_layout starts; first-use lookups bump them.
    let cnst_tables_slot = e.ad_vm_field_ptr(LLStruct::AD_VM_CURRENT_CNST_TABLES_OFF);
    let inv_cnst_off = e.ll_load(cnst_tables_slot, LLType::i32());
    let wit_tables_slot = e.ad_vm_field_ptr(LLStruct::AD_VM_CURRENT_WIT_TABLES_OFF);
    let inv_wit_off = e.ll_load(wit_tables_slot, LLType::i32());
    let wit_mults_slot = e.ad_vm_field_ptr(LLStruct::AD_VM_CURRENT_WIT_MULTIPLICITIES_OFF);
    let mults_wit_off = e.ll_load(wit_mults_slot, LLType::i32());

    // Bump each cursor by the rangecheck-8 table's footprint:
    //   constraints: 256 (per-elem) + 1 (sum) = 257
    //   witness tables: 256 (per-elem inverses)
    //   witness multiplicities: 256
    let bump_257 = e.int_const(32, 257);
    let next_cnst = e.int_arith(IntArithOp::Add, inv_cnst_off, bump_257);
    e.ll_store(cnst_tables_slot, next_cnst);
    let bump_256 = e.int_const(32, 256);
    let next_wit = e.int_arith(IntArithOp::Add, inv_wit_off, bump_256);
    e.ll_store(wit_tables_slot, next_wit);
    let next_mults = e.int_arith(IntArithOp::Add, mults_wit_off, bump_256);
    e.ll_store(wit_mults_slot, next_mults);

    // Mark the rangecheck-8 table as allocated by stashing
    // `inv_cnst_off + 1` in a private zero-initialized global.
    let snap_slot = e.global_addr(inv_cnst_off_global);
    let one_i32 = e.int_const(32, 1);
    let snap = e.int_arith(IntArithOp::Add, inv_cnst_off, one_i32);
    e.ll_store(snap_slot, snap);

    // inv_sum_coeff sits at the sum-constraint AD coefficient (one past the
    // 256 per-element coefficients for this table).
    let two_fifty_six_i32 = e.int_const(32, 256);
    let sum_idx = e.int_arith(IntArithOp::Add, inv_cnst_off, two_fifty_six_i32);
    let inv_sum_coeff = ad_read_coeff_at_dyn(e, sum_idx);

    let logup_challenge_i32 = e.int_const(32, witness_layout.challenges_start() as u64);

    e.build_counted_loop(256, vec![], |e, i_i64, _| {
        let i_i32 = e.truncate(i_i64, 32);
        // coeff = ad_coeffs[inv_cnst_off + i] — random access, does NOT
        // advance the main AdCoeffs cursor (reserved for algebraic
        // constraints).
        let coeff_idx = e.int_arith(IntArithOp::Add, inv_cnst_off, i_i32);
        let coeff = ad_read_coeff_at_dyn(e, coeff_idx);

        // out_da[inv_wit_off + i] += coeff
        let da_idx = e.int_arith(IntArithOp::Add, inv_wit_off, i_i32);
        e.ad_write_witness(DMatrix::A, da_idx, coeff);

        // out_db[logup_challenge_off] += coeff
        e.ad_write_witness(DMatrix::B, logup_challenge_i32, coeff);

        // out_db[0] += (-i_field) * coeff. `field_neg` isn't wired up in LLVM
        // codegen — express as `0 - i` so it goes through `__field_sub`.
        let i_field = u64_as_field(e, i_i64);
        let zero_i64_f = e.int_const(64, 0);
        let zero_field = u64_as_field(e, zero_i64_f);
        let neg_i_field = e.field_arith(FieldArithOp::Sub, zero_field, i_field);
        e.ad_write_const(DMatrix::B, neg_i_field, coeff);

        // out_dc[mults_wit_off + i] += coeff
        let dc_idx = e.int_arith(IntArithOp::Add, mults_wit_off, i_i32);
        e.ad_write_witness(DMatrix::C, dc_idx, coeff);

        // Second pass: out_da[inv_wit_off + i] += inv_sum_coeff
        e.ad_write_witness(DMatrix::A, da_idx, inv_sum_coeff);

        vec![]
    });

    // After the loop: out_db[0] += inv_sum_coeff
    let one_i64 = e.int_const(64, 1);
    let one_field = u64_as_field(e, one_i64);
    e.ad_write_const(DMatrix::B, one_field, inv_sum_coeff);

    inv_cnst_off
}

fn emit_spread_ad_init_body(
    e: &mut LLBlockEmitter<'_>,
    bits: u8,
    inv_cnst_off_global: usize,
    witness_layout: WitnessLayout,
) -> ValueId {
    assert!(
        bits <= 16,
        "AD spread lookup helper currently supports bit-widths up to 16, got {}",
        bits
    );
    let length = 1usize << bits;

    let cnst_tables_slot = e.ad_vm_field_ptr(LLStruct::AD_VM_CURRENT_CNST_TABLES_OFF);
    let inv_cnst_off = e.ll_load(cnst_tables_slot, LLType::i32());
    let wit_tables_slot = e.ad_vm_field_ptr(LLStruct::AD_VM_CURRENT_WIT_TABLES_OFF);
    let inv_wit_off = e.ll_load(wit_tables_slot, LLType::i32());
    let wit_mults_slot = e.ad_vm_field_ptr(LLStruct::AD_VM_CURRENT_WIT_MULTIPLICITIES_OFF);
    let mults_wit_off = e.ll_load(wit_mults_slot, LLType::i32());

    let cnst_bump = e.int_const(32, (2 * length + 1) as u64);
    let next_cnst = e.int_arith(IntArithOp::Add, inv_cnst_off, cnst_bump);
    e.ll_store(cnst_tables_slot, next_cnst);
    let wit_bump = e.int_const(32, (2 * length) as u64);
    let next_wit = e.int_arith(IntArithOp::Add, inv_wit_off, wit_bump);
    e.ll_store(wit_tables_slot, next_wit);
    let mults_bump = e.int_const(32, length as u64);
    let next_mults = e.int_arith(IntArithOp::Add, mults_wit_off, mults_bump);
    e.ll_store(wit_mults_slot, next_mults);

    let snap_slot = e.global_addr(inv_cnst_off_global);
    let one_i32 = e.int_const(32, 1);
    let snap = e.int_arith(IntArithOp::Add, inv_cnst_off, one_i32);
    e.ll_store(snap_slot, snap);

    let two_length_i32 = e.int_const(32, (2 * length) as u64);
    let sum_idx = e.int_arith(IntArithOp::Add, inv_cnst_off, two_length_i32);
    let inv_sum_coeff = ad_read_coeff_at_dyn(e, sum_idx);
    let logup_alpha_i32 = e.int_const(32, witness_layout.challenges_start() as u64);
    let logup_beta_i32 = e.int_const(32, witness_layout.challenges_start() as u64 + 1);
    let input_ty = Type::u(bits as usize);
    let result_ty = Type::u(bits as usize * 2);

    e.build_counted_loop(length, vec![], |e, i_i64, _| {
        let i_i32 = e.truncate(i_i64, 32);
        let two_i32 = e.int_const(32, 2);
        let twice_i = e.int_arith(IntArithOp::Mul, i_i32, two_i32);
        let x_cnst_idx = e.int_arith(IntArithOp::Add, inv_cnst_off, twice_i);
        let one_i32 = e.int_const(32, 1);
        let y_cnst_idx = e.int_arith(IntArithOp::Add, x_cnst_idx, one_i32);
        let x_coeff = ad_read_coeff_at_dyn(e, x_cnst_idx);
        let y_coeff = ad_read_coeff_at_dyn(e, y_cnst_idx);

        let x_wit_idx = e.int_arith(IntArithOp::Add, inv_wit_off, twice_i);
        let y_wit_idx = e.int_arith(IntArithOp::Add, x_wit_idx, one_i32);

        let i_key = e.truncate(i_i64, bits as u32);
        let spread = lower_spread(e, i_key, &input_ty, &result_ty, bits);
        let spread_u64 = if bits as u32 * 2 == 64 {
            spread
        } else {
            e.zext(spread, 64)
        };
        let spread_field = u64_as_field(e, spread_u64);

        e.ad_write_witness(DMatrix::A, logup_beta_i32, x_coeff);
        e.ad_write_const(DMatrix::B, spread_field, x_coeff);
        let neg_x_coeff = field_neg_via_sub(e, x_coeff);
        e.ad_write_witness(DMatrix::C, x_wit_idx, neg_x_coeff);

        e.ad_write_witness(DMatrix::A, y_wit_idx, y_coeff);
        e.ad_write_witness(DMatrix::B, logup_alpha_i32, y_coeff);
        let i_field = u64_as_field(e, i_i64);
        let neg_i_field = field_neg_via_sub(e, i_field);
        e.ad_write_const(DMatrix::B, neg_i_field, y_coeff);
        let neg_y_coeff = field_neg_via_sub(e, y_coeff);
        e.ad_write_witness(DMatrix::B, x_wit_idx, neg_y_coeff);

        let mults_idx = e.int_arith(IntArithOp::Add, mults_wit_off, i_i32);
        e.ad_write_witness(DMatrix::C, mults_idx, y_coeff);
        e.ad_write_witness(DMatrix::A, y_wit_idx, inv_sum_coeff);

        vec![]
    });

    let one_i64 = e.int_const(64, 1);
    let one_field = u64_as_field(e, one_i64);
    e.ad_write_const(DMatrix::B, one_field, inv_sum_coeff);

    inv_cnst_off
}

fn generate_dspread_ad_call(
    bits: u8,
    inv_cnst_off_global: usize,
    witness_layout: WitnessLayout,
    constraints_layout: ConstraintsLayout,
    _bump_da_fn: FunctionId,
    bump_db_fn: FunctionId,
    bump_dc_fn: FunctionId,
) -> LLFunction {
    let length = 1usize << bits;
    let mut func = new_ll_function(format!("__dspread_{}_ad_call", bits));
    let entry = func.get_entry_id();

    {
        let mut e = LLBlockEmitter::new(&mut func, entry);
        let key_ptr = e.add_parameter(LLType::Ptr);
        let result_ptr = e.add_parameter(LLType::Ptr);
        let flag_ptr = e.add_parameter(LLType::Ptr);

        let snap_slot = e.global_addr(inv_cnst_off_global);
        let snap = e.ll_load(snap_slot, LLType::i32());
        let zero_i32 = e.int_const(32, 0);
        let is_unalloc = e.int_eq(snap, zero_i32);
        let merge = e.build_if_else(
            is_unalloc,
            vec![LLType::Int(32)],
            |e| {
                let inv_cnst_off =
                    emit_spread_ad_init_body(e, bits, inv_cnst_off_global, witness_layout);
                vec![inv_cnst_off]
            },
            |e| {
                let snap_slot = e.global_addr(inv_cnst_off_global);
                let snap = e.ll_load(snap_slot, LLType::i32());
                let one_i32 = e.int_const(32, 1);
                let inv_cnst_off = e.int_arith(IntArithOp::Sub, snap, one_i32);
                vec![inv_cnst_off]
            },
        );
        let inv_cnst_off = merge[0];

        let two_length_i32 = e.int_const(32, (2 * length) as u64);
        let sum_idx = e.int_arith(IntArithOp::Add, inv_cnst_off, two_length_i32);
        let inv_sum_coeff = ad_read_coeff_at_dyn(&mut e, sum_idx);

        let x_wit_off = ad_next_lookup_wit_off(&mut e);
        let y_wit_off = ad_next_lookup_wit_off(&mut e);
        let lookups_wit_start_i32 = e.int_const(32, witness_layout.lookups_data_start() as u64);
        let lookups_cnst_start_i32 =
            e.int_const(32, constraints_layout.lookups_data_start() as u64);
        let x_n = e.int_arith(IntArithOp::Sub, x_wit_off, lookups_wit_start_i32);
        let x_cnst_idx = e.int_arith(IntArithOp::Add, lookups_cnst_start_i32, x_n);
        let y_n = e.int_arith(IntArithOp::Sub, y_wit_off, lookups_wit_start_i32);
        let y_cnst_idx = e.int_arith(IntArithOp::Add, lookups_cnst_start_i32, y_n);
        let x_coeff = ad_read_coeff_at_dyn(&mut e, x_cnst_idx);
        let y_coeff = ad_read_coeff_at_dyn(&mut e, y_cnst_idx);

        let logup_alpha_i32 = e.int_const(32, witness_layout.challenges_start() as u64);
        let logup_beta_i32 = e.int_const(32, witness_layout.challenges_start() as u64 + 1);

        e.ad_write_witness(DMatrix::A, logup_beta_i32, x_coeff);
        e.call(bump_db_fn, vec![result_ptr, x_coeff], 0);
        let neg_x_coeff = field_neg_via_sub(&mut e, x_coeff);
        e.ad_write_witness(DMatrix::C, x_wit_off, neg_x_coeff);

        e.ad_write_witness(DMatrix::A, y_wit_off, y_coeff);
        e.ad_write_witness(DMatrix::B, logup_alpha_i32, y_coeff);
        let neg_y_coeff = field_neg_via_sub(&mut e, y_coeff);
        e.ad_write_witness(DMatrix::B, x_wit_off, neg_y_coeff);
        e.call(bump_db_fn, vec![key_ptr, neg_y_coeff], 0);
        e.call(bump_dc_fn, vec![flag_ptr, y_coeff], 0);
        e.ad_write_witness(DMatrix::C, y_wit_off, inv_sum_coeff);

        e.terminate_return(vec![]);
    }

    func
}

/// Generate __drngchk_8_ad_call(val: AdNode*, flag: AdNode*):
///
/// On first call (private zero-initialized helper state still 0) allocates
/// the rangecheck-8 table region from the AD VM cursors and runs the
/// per-element init bumps. On every call (first and subsequent) does the
/// per-lookup AD work: reads `inv_sum_coeff` and `inv_coeff`, allocates the
/// next lookup-witness offset via `AdCurrentLookupWitOff`, and bumps:
///   out_dc[inv_wit_off] += inv_sum_coeff
///   out_da[inv_wit_off] += inv_coeff
///   out_db[logup_challenge_off] += inv_coeff
///   val.bump_db(-inv_coeff)
///   flag.bump_dc(inv_coeff)
///
/// Matches `drngchk_8_field` in `vm/src/bytecode.rs` end-to-end. The init
/// branch reads the table region's start from runtime cursors rather than
/// baking `tables_data_start` constants.
fn generate_drngchk_8_ad_call(
    inv_cnst_off_global: usize,
    witness_layout: WitnessLayout,
    constraints_layout: ConstraintsLayout,
    bump_db_fn: FunctionId,
    bump_dc_fn: FunctionId,
) -> LLFunction {
    let mut func = new_ll_function("__drngchk_8_ad_call".to_string());
    let entry = func.get_entry_id();

    {
        let mut e = LLBlockEmitter::new(&mut func, entry);
        let val_ptr = e.add_parameter(LLType::Ptr);
        let flag_ptr = e.add_parameter(LLType::Ptr);

        // First-use check: the helper's private zero-initialized global
        // stores `inv_cnst_off + 1`. Zero means "not yet allocated".
        let snap_slot = e.global_addr(inv_cnst_off_global);
        let snap = e.ll_load(snap_slot, LLType::i32());
        let zero_i32 = e.int_const(32, 0);
        let is_unalloc = e.int_eq(snap, zero_i32);
        let merge = e.build_if_else(
            is_unalloc,
            vec![LLType::Int(32)],
            |e| {
                let inv_cnst_off =
                    emit_rngchk_8_ad_init_body(e, inv_cnst_off_global, witness_layout);
                vec![inv_cnst_off]
            },
            |e| {
                // Table already allocated — just reload the snapshot.
                let snap_slot = e.global_addr(inv_cnst_off_global);
                let snap = e.ll_load(snap_slot, LLType::i32());
                let one_i32 = e.int_const(32, 1);
                let inv_cnst_off = e.int_arith(IntArithOp::Sub, snap, one_i32);
                vec![inv_cnst_off]
            },
        );
        let inv_cnst_off = merge[0];

        // inv_sum_coeff = ad_coeffs[inv_cnst_off + 256]
        let two_fifty_six_i32 = e.int_const(32, 256);
        let sum_idx = e.int_arith(IntArithOp::Add, inv_cnst_off, two_fifty_six_i32);
        let inv_sum_coeff = ad_read_coeff_at_dyn(&mut e, sum_idx);

        let inv_wit_off = ad_next_lookup_wit_off(&mut e);

        // inv_coeff = ad_coeffs[lookups_cnst_start + (inv_wit_off - lookups_wit_start)]
        //
        // Random-access; the main AdCoeffs cursor is for the algebraic
        // constraints, so it is *not* correct to advance it here. The
        // lookups-section start offsets are layout-structural (a Lookup
        // section is one contiguous slab whose start is fixed by the layout),
        // not table-allocation dynamic, so they can stay as constants.
        let lookups_wit_start_i32 = e.int_const(32, witness_layout.lookups_data_start() as u64);
        let lookups_cnst_start_i32 =
            e.int_const(32, constraints_layout.lookups_data_start() as u64);
        let n = e.int_arith(IntArithOp::Sub, inv_wit_off, lookups_wit_start_i32);
        let cnst_idx = e.int_arith(IntArithOp::Add, lookups_cnst_start_i32, n);
        let inv_coeff = ad_read_coeff_at_dyn(&mut e, cnst_idx);

        e.ad_write_witness(DMatrix::C, inv_wit_off, inv_sum_coeff);
        e.ad_write_witness(DMatrix::A, inv_wit_off, inv_coeff);
        let logup_ch_i32 = e.int_const(32, witness_layout.challenges_start() as u64);
        e.ad_write_witness(DMatrix::B, logup_ch_i32, inv_coeff);

        // val.bump_db(-inv_coeff) — `0 - inv_coeff` via the Sub runtime
        // helper, matching the lowering of FieldArithOp::Sub.
        let zero_i64 = e.int_const(64, 0);
        let zero_field = u64_as_field(&mut e, zero_i64);
        let neg_inv = e.field_arith(FieldArithOp::Sub, zero_field, inv_coeff);
        e.call(bump_db_fn, vec![val_ptr, neg_inv], 0);

        // flag.bump_dc(inv_coeff)
        e.call(bump_dc_fn, vec![flag_ptr, inv_coeff], 0);

        e.terminate_return(vec![]);
    }

    func
}
