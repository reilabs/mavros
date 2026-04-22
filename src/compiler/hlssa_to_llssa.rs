//! HLSSA -> LLSSA lowering pass
//!
//! Translates the high-level SSA (with abstract Field/U types and a separate
//! constant map) into low-level SSA (explicit integer widths, field-as-struct,
//! constants-as-instructions).
//!
//! Array types lower to heap-allocated RC'd structs behind `Ptr`, following
//! the layout in `docs/llssa.md`. MkSeq, ArrayGet, ArraySet, and MemOp
//! (Bump/Drop) are lowered to explicit memory operations.

use std::collections::HashMap;

use crate::compiler::analysis::types::{FunctionTypeInfo, TypeInfo};
use crate::compiler::block_builder::{LLBlockEmitter, LLEmitter};
use crate::compiler::flow_analysis::FlowAnalysis;
use crate::compiler::ir::r#type::{Type, TypeExpr};
use crate::compiler::llssa::{
    FieldArithOp, IntArithOp, IntCmpOp, LLFieldType, LLFunction, LLOp, LLSSA, LLStruct, LLType,
    LookupStream, WitgenBuf,
};
use crate::compiler::ssa::{
    BinaryArithOpKind, BlockId, CmpKind, DMatrix, FunctionId, HLFunction, HLSSA, Terminator,
    ValueId,
};

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

/// Lazily-created generated helper functions for lookup operations.
/// These helpers decompose the monolithic VM lookup opcodes into LLSSA-level
/// primitives (tape writes, multiplicity bumps).
struct LookupFunctions {
    rngchk_8: Option<FunctionId>,
    drngchk_8_init: Option<FunctionId>,
    drngchk_8_call: Option<FunctionId>,
    /// Witgen-side Phase 2 helper. Registered by the witgen lowering path when
    /// the program declares a rangecheck-8 table. Hoisted to the epilogue of
    /// `mavros_main` so it runs once after all Phase 1 work is complete.
    run_phase2: Option<FunctionId>,
}

impl LookupFunctions {
    fn new() -> Self {
        Self {
            rngchk_8: None,
            drngchk_8_init: None,
            drngchk_8_call: None,
            run_phase2: None,
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

    /// Ensures both `__drngchk_8_init` and `__drngchk_8_call` are registered
    /// and returns the `_call` FunctionId (the one that per-lookup lowerings
    /// invoke). The `_init` is hoisted to the main function prologue.
    fn get_drngchk_8_call_fn(&mut self, llssa: &mut LLSSA) -> FunctionId {
        if let Some(id) = self.drngchk_8_call {
            return id;
        }
        let init = llssa.add_function("__drngchk_8_ad_init".to_string());
        let call = llssa.add_function("__drngchk_8_ad_call".to_string());
        self.drngchk_8_init = Some(init);
        self.drngchk_8_call = Some(call);
        call
    }
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
// R1CS layout info threaded to lookup helper generators
// =============================================================================

/// Layout offsets needed to generate code that reads/writes AD buffers and
/// ad_coeffs at specific positions (e.g. for the rangecheck-8 AD helper).
///
/// All offsets are in field elements (not bytes). Witness-side offsets are
/// absolute indices into the full witness buffer that `out_d{a,b,c}[idx]`
/// interprets directly.
/// Which of the two LLSSA-emitting compile paths is currently running.
/// Controls a handful of mode-specific codegen decisions — most notably
/// whether to generate and hoist the Phase 2 helper (witgen only).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LoweringMode {
    /// Witgen path: Phase 1 + generated Phase 2 finalization in the epilogue.
    Witgen,
    /// AD path: DLookup helpers + AD-side init hoist, no Phase 2.
    Ad,
}

#[derive(Clone, Copy, Debug)]
pub struct R1csLayoutInfo {
    /// `constraints_layout.tables_data_start()` — first table's y-slot in ad_coeffs.
    pub tables_cnst_start: usize,
    /// `witness_layout.tables_data_start()` — first table's y-slot (absolute witness index).
    pub tables_wit_start: usize,
    /// `witness_layout.multiplicities_start()` — first table's multiplicities base.
    pub mults_wit_start: usize,
    /// `witness_layout.challenges_start()` — absolute index of `alpha` in the witness.
    pub logup_challenge_off: usize,
    /// `constraints_layout.lookups_data_start()` — start of per-lookup entries.
    pub lookups_cnst_start: usize,
    /// `witness_layout.lookups_data_start()` — start of per-lookup witness slots.
    pub lookups_wit_start: usize,
    /// Which compile path this layout is being threaded through.
    pub mode: LoweringMode,
}

// =============================================================================
// Main entry point
// =============================================================================

/// Lower an entire HLSSA program to LLSSA, without R1CS layout info.
/// DLookup opcodes will panic if encountered. Use `lower_with_layout` if the
/// program may contain lookup-related opcodes.
pub fn lower(hlssa: &HLSSA, flow_analysis: &FlowAnalysis, type_info: &TypeInfo) -> LLSSA {
    lower_inner(hlssa, flow_analysis, type_info, None)
}

/// Lower with R1CS layout info so the DLookup helper can be generated.
pub fn lower_with_layout(
    hlssa: &HLSSA,
    flow_analysis: &FlowAnalysis,
    type_info: &TypeInfo,
    layout: R1csLayoutInfo,
) -> LLSSA {
    lower_inner(hlssa, flow_analysis, type_info, Some(layout))
}

fn lower_inner(
    hlssa: &HLSSA,
    flow_analysis: &FlowAnalysis,
    type_info: &TypeInfo,
    layout: Option<R1csLayoutInfo>,
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
            layout,
            &hlssa_global_types,
        );

        let _old = llssa.take_function(ll_fn_id);
        llssa.put_function(ll_fn_id, ll_func);
    }

    // Third pass: generate drop function bodies
    generate_all_drop_functions(&mut llssa, &drop_fns);

    // Fourth pass: generate AD dispatch function bodies
    generate_all_ad_functions(&mut llssa, &ad_fns);

    // If this is a witgen compile AND a rangecheck-8 was actually lowered
    // during the second pass, register the Phase 2 helper now. We defer this
    // registration to post-lowering so programs with no lookups don't pay the
    // cost of a generated __run_phase2 at all.
    if let Some(l) = layout {
        if l.mode == LoweringMode::Witgen && lookup_fns.rngchk_8.is_some() {
            let id = llssa.add_function("__run_phase2".to_string());
            lookup_fns.run_phase2 = Some(id);
        }
    }

    // Fifth pass: generate lookup helper function bodies
    generate_all_lookup_functions(&mut llssa, &lookup_fns, layout, &mut ad_fns);

    // Sixth pass: if DLookup was used, hoist a call to __drngchk_8_ad_init
    // into the prologue of the main function.
    if let Some(init_fn) = lookup_fns.drngchk_8_init {
        let main_ll_id = llssa.get_main_id();
        hoist_init_call_to_main_prologue(&mut llssa, main_ll_id, init_fn);
    }

    // Seventh pass: if the witgen path registered __run_phase2, hoist a call
    // to it into the epilogue of the main function so Phase 2 runs after all
    // Phase 1 work is complete.
    if let Some(phase2_fn) = lookup_fns.run_phase2 {
        let main_ll_id = llssa.get_main_id();
        hoist_call_to_main_epilogue(&mut llssa, main_ll_id, phase2_fn);
    }

    llssa
}

// =============================================================================
// Per-function lowering
// =============================================================================

fn lower_function(
    function: &HLFunction,
    fn_type_info: &FunctionTypeInfo,
    cfg: &crate::compiler::flow_analysis::CFG,
    fn_map: &HashMap<FunctionId, FunctionId>,
    llssa: &mut LLSSA,
    drop_fns: &mut Vec<DropFnEntry>,
    ad_fns: &mut AdFunctions,
    lookup_fns: &mut LookupFunctions,
    layout: Option<R1csLayoutInfo>,
    hlssa_global_types: &[Type],
) -> LLFunction {
    let mut ll_func = LLFunction::empty(function.get_name().to_string());
    let mut val_map: HashMap<ValueId, ValueId> = HashMap::new();
    let mut block_map: HashMap<BlockId, BlockId> = HashMap::new();

    let hl_entry_id = function.get_entry_id();
    let ll_entry_id = ll_func.get_entry_id();
    block_map.insert(hl_entry_id, ll_entry_id);

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
                layout,
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
    _layout: Option<R1csLayoutInfo>,
    hlssa_global_types: &[Type],
) {
    use crate::compiler::ssa::{CallTarget, CastTarget, ConstValue, MemOp, OpCode, SeqType};

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

            let ll_result = match &lhs_type.expr {
                TypeExpr::U(_) => {
                    let op = match kind {
                        CmpKind::Lt => IntCmpOp::ULt,
                        CmpKind::Eq => IntCmpOp::Eq,
                    };
                    e.int_cmp(op, ll_lhs, ll_rhs)
                }
                TypeExpr::I(_) => match kind {
                    CmpKind::Eq => e.int_cmp(IntCmpOp::Eq, ll_lhs, ll_rhs),
                    CmpKind::Lt => panic!("Signed Lt not yet implemented in LLSSA"),
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
                    // Field → U(n)/I(n): FieldToLimbs, extract limb 0, truncate
                    let limbs = e.field_to_limbs(ll_value);
                    let limb0 = e.extract_field(limbs, LLStruct::limbs(), 0);
                    let ll_result = if *target_bits < 64 {
                        e.truncate(limb0, *target_bits as u32)
                    } else {
                        limb0
                    };
                    val_map.insert(*result, ll_result);
                }
                _ => panic!(
                    "Unsupported cast target in HLSSA->LLSSA lowering: {:?}",
                    target
                ),
            }
        }

        OpCode::Spread { result, value, .. }
        | OpCode::Unspread {
            result_odd: result,
            value,
            ..
        } => {
            todo!(
                "Spread/Unspread opcodes are not handled yet in HLSSA->LLSSA lowering: {:?}",
                instruction
            )
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

            e.build_if_else(
                eq,
                vec![],
                |_| vec![],
                |te| {
                    te.trap();
                    vec![]
                },
            );
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

        OpCode::ToRadix {
            result,
            value,
            radix: crate::compiler::ssa::Radix::Bytes,
            endianness,
            count,
        } => {
            lower_to_radix_bytes(e, val_map, *result, *value, *endianness, *count);
        }

        OpCode::ToRadix { .. } => {
            panic!(
                "Unsupported ToRadix variant in HLSSA->LLSSA lowering: {:?}",
                instruction
            );
        }

        OpCode::Lookup {
            target: crate::compiler::ssa::LookupTarget::Rangecheck(8),
            keys,
            results: _,
            flag,
        } => {
            assert!(
                keys.len() == 1,
                "Rangecheck(8) lookup must have exactly one key"
            );
            let ll_val = val_map[&keys[0]];
            let ll_flag = val_map[flag];
            let fn_id = lookup_fns.get_rngchk_8_fn(llssa);
            e.call(fn_id, vec![ll_val, ll_flag], 0);
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
            results: _,
            flag,
        } => {
            assert!(
                keys.len() == 1,
                "Rangecheck(8) dlookup must have exactly one key"
            );
            let ll_val = val_map[&keys[0]];
            let ll_flag = val_map[flag];
            let fn_id = lookup_fns.get_drngchk_8_call_fn(llssa);
            e.call(fn_id, vec![ll_val, ll_flag], 0);
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

/// Lower ToRadix with Radix::Bytes to FieldToLimbs + byte extraction + array allocation.
///
/// Decomposes a field element into `count` bytes. The field is first converted to
/// 4 × u64 limbs (little-endian limb order, limb 0 = least significant). Each byte
/// is extracted via shift-right and truncate. For big-endian output the byte order
/// is reversed so that output[0] is the most significant byte.
fn lower_to_radix_bytes(
    e: &mut LLBlockEmitter<'_>,
    val_map: &mut HashMap<ValueId, ValueId>,
    result: ValueId,
    value: ValueId,
    endianness: crate::compiler::ssa::Endianness,
    count: usize,
) {
    use crate::compiler::ssa::Endianness;

    assert!(count <= 32, "ToRadix byte count must be <= 32");

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
    // Byte i lives in limb[i/8] at bit offset (i%8)*8.
    let mut bytes_le = Vec::with_capacity(count);
    for i in 0..count {
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

// =============================================================================
// Lookup generated function bodies
// =============================================================================

/// Generate all lookup helper function bodies.
fn generate_all_lookup_functions(
    llssa: &mut LLSSA,
    lookup_fns: &LookupFunctions,
    layout: Option<R1csLayoutInfo>,
    ad_fns: &mut AdFunctions,
) {
    if let Some(id) = lookup_fns.rngchk_8 {
        let layout = layout.expect("R1CS layout required to generate __rngchk_8 (witgen path)");
        let func = generate_rngchk_8_function(layout);
        let _old = llssa.take_function(id);
        llssa.put_function(id, func);
    }

    // AD rangecheck-8 helpers are a pair: init runs once (hoisted to main
    // prologue) and call runs per DLookup. They both depend on R1CS layout
    // offsets to bake in the correct absolute witness/constraint positions.
    if let (Some(init_id), Some(call_id)) = (lookup_fns.drngchk_8_init, lookup_fns.drngchk_8_call) {
        let layout = layout.expect("R1CS layout required to generate AD rangecheck-8 helpers");
        let init_fn = generate_drngchk_8_ad_init(layout);
        let _old = llssa.take_function(init_id);
        llssa.put_function(init_id, init_fn);

        let bump_db_id = ad_fns.get_bump_fn(DMatrix::B, llssa);
        let bump_dc_id = ad_fns.get_bump_fn(DMatrix::C, llssa);
        let call_fn = generate_drngchk_8_ad_call(layout, bump_db_id, bump_dc_id);
        let _old = llssa.take_function(call_id);
        llssa.put_function(call_id, call_fn);
    }

    // Witgen Phase 2 helper — runs once at the end of main, reads multiplicities
    // and lookup tape entries, rewrites out_a/b/c and the post-commit witness
    // section to finalize the LogUp argument.
    if let Some(phase2_id) = lookup_fns.run_phase2 {
        let layout = layout.expect("R1CS layout required to generate __run_phase2");
        let phase2_fn = generate_run_phase2_function(layout);
        let _old = llssa.take_function(phase2_id);
        llssa.put_function(phase2_id, phase2_fn);
    }
}

/// Generate __rngchk_8(val: FieldElem, flag: FieldElem):
///
/// Emits one entry into the LogUp lookup argument for the 8-bit rangecheck
/// table. Extracts the lowest u64 limb of both `val` and `flag`, bumps the
/// multiplicity for the indexed table slot, and appends to the lookup tape.
///
/// `layout.mults_wit_start` is the absolute witness index where the
/// rangecheck-8 table's multiplicities begin (rangecheck-8 currently occupies
/// slot 0 of the multiplicities section, so this equals `multiplicities_start`;
/// a future second table with its own multiplicities would shift).
fn generate_rngchk_8_function(layout: R1csLayoutInfo) -> LLFunction {
    let mut func = LLFunction::empty("__rngchk_8".to_string());
    let entry = func.get_entry_id();

    {
        let mut e = LLBlockEmitter::new(&mut func, entry);
        let val = e.add_parameter(LLType::Struct(LLStruct::field_elem()));
        let flag = e.add_parameter(LLType::Struct(LLStruct::field_elem()));

        // Extract lowest u64 limb of val and flag (both assumed small).
        let val_limbs = e.field_to_limbs(val);
        let key = e.extract_field(val_limbs, LLStruct::limbs(), 0);
        let flag_limbs = e.field_to_limbs(flag);
        let flag_u64 = e.extract_field(flag_limbs, LLStruct::limbs(), 0);

        // multiplicities[mults_wit_start + key].low_u64 += flag_u64
        // Layout knowledge (where the rangecheck-8 multiplicities live in the
        // pre-commit witness buffer) is baked in at codegen time via `layout`;
        // the primitive itself is a generic "add to low u64 of buf[idx]".
        let key_i32 = e.truncate(key, 32);
        let mults_base_i32 = e.int_const(32, layout.mults_wit_start as u64);
        let mults_idx = e.int_arith(IntArithOp::Add, mults_base_i32, key_i32);
        e.witgen_buf_add_low_u64(WitgenBuf::WitnessPreComm, mults_idx, flag_u64);

        // Append one tape entry: (table_id=0, key, flag_u64) to (a, b, c).
        // The rangecheck-8 table is always the first table, so its id is the
        // constant 0 — but we still emit an int_const so the value lives at
        // the LL level and the LLVM codegen is uniform.
        let table_id = e.int_const(64, 0);
        e.lookup_tape_write_u64(LookupStream::A, table_id);
        e.lookup_tape_write_u64(LookupStream::B, key);
        e.lookup_tape_write_u64(LookupStream::C, flag_u64);

        e.terminate_return(vec![]);
    }

    func
}

/// Build a Field constant with `lo` as the low u64 limb and 0 for the rest.
/// Correct for small non-negative values; do NOT use for values > 2^64 - 1.
fn emit_small_u64_as_field(e: &mut LLBlockEmitter<'_>, lo: ValueId) -> ValueId {
    let zero = e.int_const(64, 0);
    let limbs = e.mk_struct(LLStruct::limbs(), vec![lo, zero, zero, zero]);
    e.field_from_limbs(limbs)
}

/// Load a Field slot known to hold a RAW u64 in the low limb (Phase 1 writes
/// lookup tape entries and pre-fix multiplicities this way), and return that
/// low limb directly as an i64. Do NOT go through `__field_to_limbs`, which
/// would interpret the slot as Montgomery and produce garbage.
fn load_raw_low_u64(e: &mut LLBlockEmitter<'_>, buf: WitgenBuf, idx: ValueId) -> ValueId {
    let v = e.witgen_buf_load(buf, idx);
    e.extract_field(v, LLStruct::field_elem(), 0)
}

/// Generate __drngchk_8_ad_init():
///
/// Runs the one-time AD initialization for the 8-bit rangecheck LogUp table.
/// For each i in 0..256 reads one coefficient from the tables section and
/// bumps `out_da`, `out_db`, `out_dc` at the positions the LogUp argument
/// requires. After the loop it also bumps `out_db[0] += inv_sum_coeff`.
///
/// Matches the first-call branch of `drngchk_8_field` in `vm/src/bytecode.rs`.
fn generate_drngchk_8_ad_init(layout: R1csLayoutInfo) -> LLFunction {
    let mut func = LLFunction::empty("__drngchk_8_ad_init".to_string());
    let entry = func.get_entry_id();

    {
        let mut e = LLBlockEmitter::new(&mut func, entry);

        // inv_sum_coeff = ad_coeffs[tables_cnst_start + 256] — constant across loop.
        let inv_sum_coeff = e.ad_read_coeff_at(layout.tables_cnst_start + 256);

        // Constants used inside the loop.
        let tables_wit_start_i32 = e.int_const(32, layout.tables_wit_start as u64);
        let mults_wit_start_i32 = e.int_const(32, layout.mults_wit_start as u64);
        let logup_challenge_i32 = e.int_const(32, layout.logup_challenge_off as u64);
        let wit_zero_i32 = e.int_const(32, 0);

        e.build_counted_loop(256, vec![], |e, i_i64, _| {
            // i as i32 for witness-index arithmetic.
            let i_i32 = e.truncate(i_i64, 32);

            // coeff = next_d_coeff_tables()
            let coeff = e.next_d_coeff_tables();

            // out_da[tables_wit_start + i] += coeff
            let da_idx = e.int_arith(IntArithOp::Add, tables_wit_start_i32, i_i32);
            e.ad_write_witness(DMatrix::A, da_idx, coeff);

            // out_db[logup_challenge_off] += coeff
            e.ad_write_witness(DMatrix::B, logup_challenge_i32, coeff);

            // out_db[0] += (-i_field) * coeff  (i.e. `out_db[0] -= coeff * i`)
            let i_i64_val = i_i64;
            let i_field = emit_small_u64_as_field(e, i_i64_val);
            let zero_i64 = e.int_const(64, 0);
            let zero_field = emit_small_u64_as_field(e, zero_i64);
            let minus_i_field = e.field_arith(FieldArithOp::Sub, zero_field, i_field);
            e.ad_write_const(DMatrix::B, minus_i_field, coeff);

            // out_dc[mults_wit_start + i] += coeff
            let dc_idx = e.int_arith(IntArithOp::Add, mults_wit_start_i32, i_i32);
            e.ad_write_witness(DMatrix::C, dc_idx, coeff);

            // out_da[tables_wit_start + i] += inv_sum_coeff (second bump)
            e.ad_write_witness(DMatrix::A, da_idx, inv_sum_coeff);

            vec![]
        });

        // After the loop: out_db[0] += inv_sum_coeff
        // Use ad_write_const with const=1 and sensitivity=inv_sum_coeff, so the
        // runtime computes out_db[0] += 1 * inv_sum_coeff = inv_sum_coeff.
        let one_i64 = e.int_const(64, 1);
        let one_field = emit_small_u64_as_field(&mut e, one_i64);
        e.ad_write_const(DMatrix::B, one_field, inv_sum_coeff);

        e.terminate_return(vec![]);
    }

    func
}

/// Generate __drngchk_8_ad_call(val: AdNode*, flag: AdNode*):
///
/// Per-lookup AD work. Reads a fresh `inv_coeff` and `inv_sum_coeff`, allocates
/// the next lookup-witness offset, and bumps the relevant entries in out_da,
/// out_db, out_dc plus the AD-dispatched bumps on `val` and `flag`.
///
/// Matches the post-init tail of `drngchk_8_field` in `vm/src/bytecode.rs`.
fn generate_drngchk_8_ad_call(
    layout: R1csLayoutInfo,
    bump_db_fn: FunctionId,
    bump_dc_fn: FunctionId,
) -> LLFunction {
    let mut func = LLFunction::empty("__drngchk_8_ad_call".to_string());
    let entry = func.get_entry_id();

    {
        let mut e = LLBlockEmitter::new(&mut func, entry);
        let val_ptr = e.add_parameter(LLType::Ptr);
        let flag_ptr = e.add_parameter(LLType::Ptr);

        // inv_coeff: sequential read of the lookups-section cursor.
        let inv_coeff = e.next_d_coeff_lookups();
        // inv_sum_coeff: random-access read at tables_cnst_start + 256.
        let inv_sum_coeff = e.ad_read_coeff_at(layout.tables_cnst_start + 256);

        // current_inv_wit_offset: next absolute witness index for lookup slots.
        let inv_wit_off = e.ad_next_lookup_wit_off();

        // out_dc[inv_wit_off] += inv_sum_coeff
        e.ad_write_witness(DMatrix::C, inv_wit_off, inv_sum_coeff);
        // out_da[inv_wit_off] += inv_coeff
        e.ad_write_witness(DMatrix::A, inv_wit_off, inv_coeff);
        // out_db[logup_challenge_off] += inv_coeff
        let logup_ch_i32 = e.int_const(32, layout.logup_challenge_off as u64);
        e.ad_write_witness(DMatrix::B, logup_ch_i32, inv_coeff);

        // val.bump_db(-inv_coeff) — AD dispatcher
        let zero_i64 = e.int_const(64, 0);
        let zero_field = emit_small_u64_as_field(&mut e, zero_i64);
        let minus_inv_coeff = e.field_arith(FieldArithOp::Sub, zero_field, inv_coeff);
        e.call(bump_db_fn, vec![val_ptr, minus_inv_coeff], 0);

        // flag.bump_dc(inv_coeff) — AD dispatcher
        e.call(bump_dc_fn, vec![flag_ptr, inv_coeff], 0);

        e.terminate_return(vec![]);
    }

    func
}

// Generate __run_phase2():
///
/// Finalizes the LogUp lookup argument after Phase 1 has written raw values
/// into the tape and multiplicity buffers. For the rangecheck-8 table only:
///
///   1. Fix multiplicities: Phase 1 wrote raw u64 counts into the low limb of
///      each multiplicity Field slot (via `__bump_rngchk8_multiplicity`).
///      Convert each to Montgomery form by reading the low limb directly
///      (NOT via `__field_to_limbs`, which assumes Montgomery input) and
///      re-encoding via `__field_from_limbs`.
///
///   2. Forward pass over the table: for each entry i in 0..256, store
///      `denom = α - i` into out_b and the multiplicity into out_c. If the
///      multiplicity is non-zero, store the running_prod into out_a and
///      multiply running_prod by denom.
///
///   3. Do the one batch-inversion: running_inv = 1 / running_prod.
///
///   4. Backward pass: recover per-entry inverses by multiplying running_inv
///      with the stored running_prod, then updating running_inv *= denom.
///
///   5. Lookup tape walk: for each entry written during Phase 1, either zero
///      out (flag == 0) or copy the pre-inverted y-value from the table slot
///      and accumulate into the table's sum constraint (flag == 1). Tape
///      entries hold RAW u64 flags/keys so their low limbs are read directly.
///
///   6. Final consolidation: multiply each per-entry inverse by its
///      multiplicity, write to the post-commit witness, and accumulate into
///      the table's sum constraint. Finally set B on the sum constraint to 1.
fn generate_run_phase2_function(layout: R1csLayoutInfo) -> LLFunction {
    let mut func = LLFunction::empty("__run_phase2".to_string());
    let entry = func.get_entry_id();

    {
        let mut e = LLBlockEmitter::new(&mut func, entry);
        let table_len: usize = 256;
        let mults_base_i32 = e.int_const(32, layout.mults_wit_start as u64);
        let base_cnst_i32 = e.int_const(32, layout.tables_cnst_start as u64);
        let post_zero = e.int_const(32, 0);
        let zero_i64 = e.int_const(64, 0);
        let zero_field = emit_small_u64_as_field(&mut e, zero_i64);
        let one_i64 = e.int_const(64, 1);
        let one_field = emit_small_u64_as_field(&mut e, one_i64);
        let post_base: u64 = layout.logup_challenge_off as u64;
        let sum_cnst_i32 = e.int_const(32, (layout.tables_cnst_start + table_len) as u64);

        // Step 1: Phase 1 wrote raw u64 counts into the LOW LIMB of each
        // multiplicity slot. Re-encode to Montgomery via __field_from_limbs.
        e.build_counted_loop(table_len, vec![], |e, i_i64, _| {
            let i_i32 = e.truncate(i_i64, 32);
            let idx = e.int_arith(IntArithOp::Add, mults_base_i32, i_i32);
            let count = load_raw_low_u64(e, WitgenBuf::WitnessPreComm, idx);
            let fixed = emit_small_u64_as_field(e, count);
            e.witgen_buf_store(WitgenBuf::WitnessPreComm, idx, fixed);
            vec![]
        });

        // Step 2: forward pass.
        let alpha = e.witgen_buf_load(WitgenBuf::WitnessPostComm, post_zero);
        let results = e.build_counted_loop(
            table_len,
            vec![(one_field, LLType::Struct(LLStruct::field_elem()))],
            |e, i_i64, accs| {
                let running_prod = accs[0];
                let i_i32 = e.truncate(i_i64, 32);

                let m_idx = e.int_arith(IntArithOp::Add, mults_base_i32, i_i32);
                let m = e.witgen_buf_load(WitgenBuf::WitnessPreComm, m_idx);

                let i_field = emit_small_u64_as_field(e, i_i64);
                let denom = e.field_arith(FieldArithOp::Sub, alpha, i_field);

                let base_plus_i = e.int_arith(IntArithOp::Add, base_cnst_i32, i_i32);
                e.witgen_buf_store(WitgenBuf::B, base_plus_i, denom);
                e.witgen_buf_store(WitgenBuf::C, base_plus_i, m);

                let is_zero = e.field_eq(m, zero_field);
                let merged = e.build_if_else(
                    is_zero,
                    vec![LLType::Struct(LLStruct::field_elem())],
                    |_then_e| vec![running_prod],
                    |else_e| {
                        else_e.witgen_buf_store(WitgenBuf::A, base_plus_i, running_prod);
                        let new_rp = else_e.field_arith(FieldArithOp::Mul, running_prod, denom);
                        vec![new_rp]
                    },
                );
                merged
            },
        );
        let running_prod_end = results[0];

        // Step 3: inversion.
        let running_inv_initial = e.field_inverse(running_prod_end);

        // Step 4: backward pass.
        let last_i_i32 = e.int_const(32, (table_len - 1) as u64);
        let _bwd = e.build_counted_loop(
            table_len,
            vec![(running_inv_initial, LLType::Struct(LLStruct::field_elem()))],
            |e, j_i64, accs| {
                let running_inv = accs[0];
                let j_i32 = e.truncate(j_i64, 32);
                let i_i32 = e.int_arith(IntArithOp::Sub, last_i_i32, j_i32);
                let base_plus_i = e.int_arith(IntArithOp::Add, base_cnst_i32, i_i32);

                let m = e.witgen_buf_load(WitgenBuf::C, base_plus_i);
                let denom = e.witgen_buf_load(WitgenBuf::B, base_plus_i);
                let rp = e.witgen_buf_load(WitgenBuf::A, base_plus_i);

                let is_zero = e.field_eq(m, zero_field);
                let merged = e.build_if_else(
                    is_zero,
                    vec![LLType::Struct(LLStruct::field_elem())],
                    |_then_e| vec![running_inv],
                    |else_e| {
                        let elem = else_e.field_arith(FieldArithOp::Mul, rp, running_inv);
                        else_e.witgen_buf_store(WitgenBuf::A, base_plus_i, elem);
                        let new_ri = else_e.field_arith(FieldArithOp::Mul, running_inv, denom);
                        vec![new_ri]
                    },
                );
                merged
            },
        );

        // Step 5: lookup tape walk. Tape entries written by Phase 1 hold RAW
        // u64 values (flag in C, key in B) in the low limb — we read them
        // directly via `load_raw_low_u64`.
        //
        // `lookups_wit_start` and `tables_wit_start` are always ≥ `post_base`
        // (= `challenges_start`) in the witness layout — the sections are
        // strictly ordered. Use `checked_sub` + expect so a broken layout
        // surfaces at codegen time instead of silently emitting offset 0.
        let lookups_wit_rel = (layout.lookups_wit_start as u64)
            .checked_sub(post_base)
            .expect("layout invariant violated: lookups_wit_start < challenges_start");
        let lookups_wit_rel_i32 = e.int_const(32, lookups_wit_rel);
        let lookups_cnst_i32 = e.int_const(32, layout.lookups_cnst_start as u64);

        let n_lookups_i32 = e.lookup_tape_len();
        let j0_i32 = e.int_const(32, 0);
        let one_i32 = e.int_const(32, 1);
        e.build_loop(
            vec![(j0_i32, LLType::i32())],
            |b, params| b.int_ult(params[0], n_lookups_i32),
            |le, params| {
                let j_i32 = params[0];
                let cnst_off = le.int_arith(IntArithOp::Add, lookups_cnst_i32, j_i32);
                let wit_off = le.int_arith(IntArithOp::Add, lookups_wit_rel_i32, j_i32);
                let flag_u64 = load_raw_low_u64(le, WitgenBuf::C, cnst_off);
                let is_zero = le.int_eq(flag_u64, zero_i64);

                le.build_if_else(
                    is_zero,
                    vec![],
                    |then_e| {
                        let key_u64 = load_raw_low_u64(then_e, WitgenBuf::B, cnst_off);
                        let key_field = emit_small_u64_as_field(then_e, key_u64);
                        let b_val = then_e.field_arith(FieldArithOp::Sub, alpha, key_field);
                        then_e.witgen_buf_store(WitgenBuf::A, cnst_off, zero_field);
                        then_e.witgen_buf_store(WitgenBuf::B, cnst_off, b_val);
                        then_e.witgen_buf_store(WitgenBuf::C, cnst_off, zero_field);
                        then_e.witgen_buf_store(WitgenBuf::WitnessPostComm, wit_off, zero_field);
                        vec![]
                    },
                    |else_e| {
                        let ix_u64 = load_raw_low_u64(else_e, WitgenBuf::B, cnst_off);
                        let ix_i32 = else_e.truncate(ix_u64, 32);
                        let tbl_idx = else_e.int_arith(IntArithOp::Add, base_cnst_i32, ix_i32);
                        let y_a = else_e.witgen_buf_load(WitgenBuf::A, tbl_idx);
                        let y_b = else_e.witgen_buf_load(WitgenBuf::B, tbl_idx);
                        let flag_field = emit_small_u64_as_field(else_e, flag_u64);
                        else_e.witgen_buf_store(WitgenBuf::A, cnst_off, y_a);
                        else_e.witgen_buf_store(WitgenBuf::B, cnst_off, y_b);
                        else_e.witgen_buf_store(WitgenBuf::C, cnst_off, flag_field);
                        else_e.witgen_buf_store(WitgenBuf::WitnessPostComm, wit_off, y_a);
                        else_e.witgen_buf_add(WitgenBuf::C, sum_cnst_i32, y_a);
                        vec![]
                    },
                );

                let next_j = le.int_arith(IntArithOp::Add, j_i32, one_i32);
                vec![next_j]
            },
        );

        // Step 6: final consolidation — multiply each table y-value by its
        // multiplicity, write to post-commit witness, accumulate into sum slot.
        // Finally set B at sum slot to 1.
        let tables_wit_rel = (layout.tables_wit_start as u64)
            .checked_sub(post_base)
            .expect("layout invariant violated: tables_wit_start < challenges_start");
        let tables_wit_rel_i32 = e.int_const(32, tables_wit_rel);
        e.build_counted_loop(table_len, vec![], |e, i_i64, _| {
            let i_i32 = e.truncate(i_i64, 32);
            let base_plus_i = e.int_arith(IntArithOp::Add, base_cnst_i32, i_i32);
            let m = e.witgen_buf_load(WitgenBuf::C, base_plus_i);
            let is_zero = e.field_eq(m, zero_field);
            e.build_if_else(
                is_zero,
                vec![],
                |_then_e| vec![],
                |else_e| {
                    let a_val = else_e.witgen_buf_load(WitgenBuf::A, base_plus_i);
                    let elem = else_e.field_arith(FieldArithOp::Mul, a_val, m);
                    else_e.witgen_buf_store(WitgenBuf::A, base_plus_i, elem);
                    let wit_post_idx = else_e.int_arith(IntArithOp::Add, tables_wit_rel_i32, i_i32);
                    else_e.witgen_buf_store(WitgenBuf::WitnessPostComm, wit_post_idx, elem);
                    else_e.witgen_buf_add(WitgenBuf::A, sum_cnst_i32, elem);
                    vec![]
                },
            );
            vec![]
        });

        // Final: set out_b[sum_slot] = 1 (the sum constraint's B coefficient).
        e.witgen_buf_store(WitgenBuf::B, sum_cnst_i32, one_field);

        e.terminate_return(vec![]);
    }
    func
}

/// Insert a call to `init_fn` at the very start of the main function's entry
/// block. Used to hoist one-time AD lookup init before any other AD work.
fn hoist_init_call_to_main_prologue(
    llssa: &mut LLSSA,
    main_fn_id: FunctionId,
    init_fn: FunctionId,
) {
    let mut main_fn = llssa.take_function(main_fn_id);
    let entry_id = main_fn.get_entry_id();

    // Emit the init call (appended to the end of the instruction list) and
    // then rotate so it becomes the first instruction of the entry block.
    {
        let mut e = LLBlockEmitter::new(&mut main_fn, entry_id);
        let _ = e.call(init_fn, vec![], 0);
    }

    let entry_block = main_fn.get_block_mut(entry_id);
    let mut insns = entry_block.take_instructions();
    if let Some(init_call) = insns.pop() {
        insns.insert(0, init_call);
    }
    entry_block.put_instructions(insns);

    llssa.put_function(main_fn_id, main_fn);
}

/// Insert a call to `target_fn` just before every `Return` terminator in the
/// main function. This runs `target_fn` once per return path without any CFG
/// restructuring: the call is appended to the block's instruction list, which
/// already ends (terminator-wise) in `Return`.
fn hoist_call_to_main_epilogue(llssa: &mut LLSSA, main_fn_id: FunctionId, target_fn: FunctionId) {
    let mut main_fn = llssa.take_function(main_fn_id);

    let return_blocks: Vec<BlockId> = main_fn
        .get_blocks()
        .filter_map(|(bid, block)| match block.get_terminator() {
            Some(Terminator::Return(_)) => Some(*bid),
            _ => None,
        })
        .collect();

    for bid in return_blocks {
        let mut e = LLBlockEmitter::new(&mut main_fn, bid);
        let _ = e.call(target_fn, vec![], 0);
    }

    llssa.put_function(main_fn_id, main_fn);
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

    let mut func = LLFunction::empty(name.to_string());
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
    let mut func = LLFunction::empty("__ad_drop".to_string());
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

    let mut func = LLFunction::empty(format!("drop_{}", ty));
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

    let mut func = LLFunction::empty(format!("drop_{}", ty));
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
