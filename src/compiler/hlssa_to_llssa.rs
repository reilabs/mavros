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
        TypeExpr::WitnessOf(_) => LLStruct::new(vec![LLFieldType::Ptr]),
        _ => panic!("Unsupported element type: {}", ty),
    }
}

/// Get the RC'd array struct for an Array<T, N> type.
fn rc_array_struct(elem_type: &Type, count: usize) -> LLStruct {
    LLStruct::rc_array(elem_struct(elem_type), count)
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

    // Resolve or create the drop function ID
    let fn_id = match &ty.expr {
        TypeExpr::WitnessOf(_) => ad_fns.get_drop_fn(llssa),
        TypeExpr::Array(inner, _) => {
            if needs_drop(&inner.expr) {
                get_or_create_drop_fn(inner, llssa, drop_fns, ad_fns);
            }
            llssa.add_function(format!("drop_{}", ty))
        }
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
// Main entry point
// =============================================================================

/// Lower an entire HLSSA program to LLSSA.
pub fn lower(hlssa: &HLSSA, flow_analysis: &FlowAnalysis, type_info: &TypeInfo) -> LLSSA {
    let main_id = hlssa.get_main_id();
    let main_name = hlssa.get_main().get_name().to_string();
    let mut llssa = LLSSA::with_main(main_name);
    let mut fn_map: HashMap<FunctionId, FunctionId> = HashMap::new();
    let mut drop_fns: Vec<DropFnEntry> = Vec::new();
    let mut ad_fns = AdFunctions::new();

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
        );

        let _old = llssa.take_function(ll_fn_id);
        llssa.put_function(ll_fn_id, ll_func);
    }

    // Third pass: generate drop function bodies
    generate_all_drop_functions(&mut llssa, &drop_fns);

    // Fourth pass: generate AD dispatch function bodies
    generate_all_ad_functions(&mut llssa, &ad_fns);

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
            lower_ad_mul_const(e, val_map, *result, *const_val, *var);
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
                    lower_ad_const_wrap(e, val_map, *result, *value);
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
    let arr_type = fn_type_info.get_value_type(value);
    let (et, count) = array_info(arr_type);
    let rc_struct = rc_array_struct(et, count);

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

/// Allocate an ADConstNode wrapping a pure field value.
fn lower_ad_const_wrap(
    e: &mut LLBlockEmitter<'_>,
    val_map: &mut HashMap<ValueId, ValueId>,
    result: ValueId,
    value: ValueId,
) {
    let ll_val = val_map[&value];
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
) {
    let ll_coeff = val_map[&const_val];
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
            // WitnessOf points to ad_drop, whose body is generated by generate_all_ad_functions
            TypeExpr::WitnessOf(_) => continue,
            other => panic!("No drop function generator for type: {:?}", other),
        };
        let _old = llssa.take_function(entry.fn_id);
        llssa.put_function(entry.fn_id, func);
    }
}

fn needs_drop(expr: &TypeExpr) -> bool {
    matches!(expr, TypeExpr::Array(..) | TypeExpr::WitnessOf(..))
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
