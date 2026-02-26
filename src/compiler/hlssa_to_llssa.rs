//! HLSSA → LLSSA lowering pass
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
use crate::compiler::flow_analysis::FlowAnalysis;
use crate::compiler::ir::r#type::{Type, TypeExpr};
use crate::compiler::llssa::{
    FieldArithOp, IntArithOp, IntCmpOp, LLFieldType, LLFunction, LLOp, LLStruct, LLType, LLSSA,
};
use crate::compiler::ssa::{
    BinaryArithOpKind, BlockId, CmpKind, FunctionId, HLFunction, Terminator, ValueId, HLSSA,
};

// ═══════════════════════════════════════════════════════════════════════════════
// Type helpers
// ═══════════════════════════════════════════════════════════════════════════════

/// Map an HLSSA type to an LLType.
fn lower_type(ty: &Type) -> LLType {
    match &ty.expr {
        TypeExpr::Field => LLType::Struct(LLStruct::field_elem()),
        TypeExpr::U(bits) => LLType::Int(*bits as u32),
        TypeExpr::Array(..) => LLType::Ptr,
        _ => panic!("Unsupported type in HLSSA→LLSSA lowering: {}", ty),
    }
}

/// Get the LLStruct layout for a single element of the given HLSSA type,
/// for use in InlineArray fields. Scalar types become single-field structs.
fn elem_struct(ty: &Type) -> LLStruct {
    match &ty.expr {
        TypeExpr::Field => LLStruct::field_elem(),
        TypeExpr::U(bits) => LLStruct::new(vec![LLFieldType::Int(*bits as u32)]),
        TypeExpr::Array(..) => LLStruct::new(vec![LLFieldType::Ptr]),
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

// ═══════════════════════════════════════════════════════════════════════════════
// Drop function tracking
// ═══════════════════════════════════════════════════════════════════════════════

struct DropFnEntry {
    hlssa_type: Type,
    fn_id: FunctionId,
}

/// Get or create a drop function for the given array type.
/// Recursively creates drop functions for inner array elements.
fn get_or_create_drop_fn(
    ty: &Type,
    llssa: &mut LLSSA,
    drop_fns: &mut Vec<DropFnEntry>,
) -> FunctionId {
    // Check if already exists
    for entry in drop_fns.iter() {
        if entry.hlssa_type == *ty {
            return entry.fn_id;
        }
    }

    // Recursively create drop fn for inner array elements first
    if let TypeExpr::Array(inner, _) = &ty.expr {
        if matches!(inner.expr, TypeExpr::Array(..)) {
            get_or_create_drop_fn(inner, llssa, drop_fns);
        }
    }

    let fn_id = llssa.add_function(format!("drop_{}", ty));
    drop_fns.push(DropFnEntry {
        hlssa_type: ty.clone(),
        fn_id,
    });
    fn_id
}

// ═══════════════════════════════════════════════════════════════════════════════
// Main entry point
// ═══════════════════════════════════════════════════════════════════════════════

/// Lower an entire HLSSA program to LLSSA.
pub fn lower(hlssa: &HLSSA, flow_analysis: &FlowAnalysis, type_info: &TypeInfo) -> LLSSA {
    let main_id = hlssa.get_main_id();
    let main_name = hlssa.get_main().get_name().to_string();
    let mut llssa = LLSSA::with_main(main_name);
    let mut fn_map: HashMap<FunctionId, FunctionId> = HashMap::new();
    let mut drop_fns: Vec<DropFnEntry> = Vec::new();

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
        );

        let _old = llssa.take_function(ll_fn_id);
        llssa.put_function(ll_fn_id, ll_func);
    }

    // Third pass: generate drop function bodies
    generate_all_drop_functions(&mut llssa, &drop_fns);

    llssa
}

// ═══════════════════════════════════════════════════════════════════════════════
// Per-function lowering
// ═══════════════════════════════════════════════════════════════════════════════

fn lower_function(
    function: &HLFunction,
    fn_type_info: &FunctionTypeInfo,
    cfg: &crate::compiler::flow_analysis::CFG,
    fn_map: &HashMap<FunctionId, FunctionId>,
    llssa: &mut LLSSA,
    drop_fns: &mut Vec<DropFnEntry>,
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
        let mut current_block = block_map[&block_id];

        // Lower instructions
        for instruction in block.get_instructions() {
            current_block = lower_instruction(
                instruction,
                &mut ll_func,
                current_block,
                &mut val_map,
                fn_type_info,
                fn_map,
                llssa,
                drop_fns,
            );
        }

        // Lower terminator (into current_block, which may have changed due to block splits)
        if let Some(terminator) = block.get_terminator() {
            lower_terminator(
                terminator,
                &mut ll_func,
                current_block,
                &val_map,
                &block_map,
            );
        }
    }

    ll_func
}

// ═══════════════════════════════════════════════════════════════════════════════
// Instruction lowering
// ═══════════════════════════════════════════════════════════════════════════════

/// Lower a single HLSSA instruction to LLSSA ops.
/// Returns the block ID to use for subsequent instructions (may change if this
/// instruction creates new blocks, e.g. ArraySet with CoW).
#[allow(clippy::too_many_arguments)]
fn lower_instruction(
    instruction: &crate::compiler::ssa::OpCode,
    ll_func: &mut LLFunction,
    block_id: BlockId,
    val_map: &mut HashMap<ValueId, ValueId>,
    fn_type_info: &FunctionTypeInfo,
    fn_map: &HashMap<FunctionId, FunctionId>,
    llssa: &mut LLSSA,
    drop_fns: &mut Vec<DropFnEntry>,
) -> BlockId {
    use crate::compiler::ssa::{CallTarget, ConstValue, MemOp, OpCode, SeqType};

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
                    ll_func.push_field_arith(block_id, op, ll_lhs, ll_rhs)
                }
                TypeExpr::U(_) => {
                    let op = match kind {
                        BinaryArithOpKind::Add => IntArithOp::Add,
                        BinaryArithOpKind::Sub => IntArithOp::Sub,
                        BinaryArithOpKind::Mul => IntArithOp::Mul,
                        BinaryArithOpKind::And => IntArithOp::And,
                        _ => panic!("Unsupported int arith op: {:?}", kind),
                    };
                    ll_func.push_int_arith(block_id, op, ll_lhs, ll_rhs)
                }
                _ => panic!(
                    "Unsupported type for BinaryArithOp in lowering: {:?}",
                    result_type
                ),
            };
            val_map.insert(*result, ll_result);
            block_id
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
                    ll_func.push_int_cmp(block_id, op, ll_lhs, ll_rhs)
                }
                TypeExpr::Field => match kind {
                    CmpKind::Eq => ll_func.push_field_eq(block_id, ll_lhs, ll_rhs),
                    _ => panic!("Unsupported field comparison: {:?}", kind),
                },
                _ => panic!("Unsupported type for Cmp in lowering: {:?}", lhs_type),
            };
            val_map.insert(*result, ll_result);
            block_id
        }

        OpCode::Constrain { a, b, c } => {
            let ll_a = val_map[a];
            let ll_b = val_map[b];
            let ll_c = val_map[c];
            ll_func.push_constrain(block_id, ll_a, ll_b, ll_c);
            block_id
        }

        OpCode::WriteWitness { result, value, .. } => {
            let ll_value = val_map[value];
            ll_func.push_write_witness(block_id, ll_value);
            if let Some(result_id) = result {
                val_map.insert(*result_id, ll_value);
            }
            block_id
        }

        OpCode::Call {
            results,
            function: CallTarget::Static(callee),
            args,
        } => {
            let ll_callee = fn_map[callee];
            let ll_args: Vec<ValueId> = args.iter().map(|a| val_map[a]).collect();
            let ll_results = ll_func.push_call(block_id, ll_callee, ll_args, results.len());
            for (hl_result, ll_result) in results.iter().zip(ll_results.iter()) {
                val_map.insert(*hl_result, *ll_result);
            }
            block_id
        }

        OpCode::Not { result, value } => {
            let ll_value = val_map[value];
            let ll_result = ll_func.push_not(block_id, ll_value);
            val_map.insert(*result, ll_result);
            block_id
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
            let ll_result = ll_func.push_select(block_id, ll_cond, ll_if_t, ll_if_f);
            val_map.insert(*result, ll_result);
            block_id
        }

        OpCode::Const { result, value } => {
            match value {
                ConstValue::U(bits, val) => {
                    let ll_val = ll_func.push_int_const(block_id, *bits as u32, *val as u64);
                    val_map.insert(*result, ll_val);
                }
                ConstValue::Field(fr) => {
                    let field_struct = LLStruct::field_elem();
                    let limbs = fr.0.0;
                    let l0 = ll_func.push_int_const(block_id, 64, limbs[0]);
                    let l1 = ll_func.push_int_const(block_id, 64, limbs[1]);
                    let l2 = ll_func.push_int_const(block_id, 64, limbs[2]);
                    let l3 = ll_func.push_int_const(block_id, 64, limbs[3]);
                    let mk = ll_func.push_mk_struct(block_id, field_struct, vec![l0, l1, l2, l3]);
                    val_map.insert(*result, mk);
                }
                ConstValue::FnPtr(_) => {
                    panic!("FnPtr constants not supported in HLSSA→LLSSA lowering");
                }
            }
            block_id
        }

        // ── Array operations ────────────────────────────────────────────

        OpCode::MkSeq {
            result,
            elems,
            seq_type: SeqType::Array(count),
            elem_type,
        } => {
            lower_mk_array(
                ll_func, block_id, val_map, *result, elems, elem_type, *count,
            );
            block_id
        }

        OpCode::ArrayGet {
            result,
            array,
            index,
        } => {
            lower_array_get(
                ll_func, block_id, val_map, fn_type_info, *result, *array, *index,
            );
            block_id
        }

        OpCode::ArraySet {
            result,
            array,
            index,
            value,
        } => lower_array_set(
            ll_func, block_id, val_map, fn_type_info, *result, *array, *index, *value,
        ),

        // ── RC operations ───────────────────────────────────────────────

        OpCode::MemOp {
            kind: MemOp::Bump(n),
            value,
        } => {
            lower_rc_bump(ll_func, block_id, val_map, fn_type_info, *n, *value);
            block_id
        }

        OpCode::MemOp {
            kind: MemOp::Drop,
            value,
        } => {
            lower_rc_drop(
                ll_func, block_id, val_map, fn_type_info, *value, llssa, drop_fns,
            );
            block_id
        }

        _ => panic!(
            "Unsupported opcode in HLSSA→LLSSA lowering: {:?}",
            instruction
        ),
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Array lowering helpers
// ═══════════════════════════════════════════════════════════════════════════════

/// Lower MkSeq(Array) to heap allocation + element stores.
fn lower_mk_array(
    ll_func: &mut LLFunction,
    block_id: BlockId,
    val_map: &mut HashMap<ValueId, ValueId>,
    result: ValueId,
    elems: &[ValueId],
    elem_type: &Type,
    count: usize,
) {
    let rc_struct = rc_array_struct(elem_type, count);
    let es = elem_struct(elem_type);

    // Allocate
    let arr = ll_func.push_heap_alloc(block_id, rc_struct.clone(), None);

    // Init RC to 1
    let rc_hdr = ll_func.push_struct_field_ptr(block_id, arr, rc_struct.clone(), 0);
    let rc_word = ll_func.push_struct_field_ptr(block_id, rc_hdr, LLStruct::rc_header(), 0);
    let one = ll_func.push_int_const(block_id, 64, 1);
    ll_func.push_store(block_id, rc_word, one);

    // Store elements
    let data = ll_func.push_struct_field_ptr(block_id, arr, rc_struct, 1);
    for (i, elem) in elems.iter().enumerate() {
        let idx = ll_func.push_int_const(block_id, 64, i as u64);
        let elem_ptr = ll_func.push_array_elem_ptr(block_id, data, es.clone(), idx);
        let ll_elem = val_map[elem];
        ll_func.push_store(block_id, elem_ptr, ll_elem);
    }

    val_map.insert(result, arr);
}

/// Lower ArrayGet to struct_field_ptr + array_elem_ptr + load.
fn lower_array_get(
    ll_func: &mut LLFunction,
    block_id: BlockId,
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
    let idx64 = ll_func.push_zext(block_id, ll_idx, 64);

    let data = ll_func.push_struct_field_ptr(block_id, ll_arr, rc_struct, 1);
    let elem_ptr = ll_func.push_array_elem_ptr(block_id, data, es, idx64);
    let val = ll_func.push_load(block_id, elem_ptr, ll_elem_type);

    val_map.insert(result, val);
}

/// Lower ArraySet with copy-on-write semantics.
/// Creates new blocks for the RC check + conditional copy.
/// Returns the merge block ID (subsequent instructions emit there).
fn lower_array_set(
    ll_func: &mut LLFunction,
    block_id: BlockId,
    val_map: &mut HashMap<ValueId, ValueId>,
    fn_type_info: &FunctionTypeInfo,
    result: ValueId,
    array: ValueId,
    index: ValueId,
    value: ValueId,
) -> BlockId {
    let arr_type = fn_type_info.get_value_type(array);
    let (et, count) = array_info(arr_type);
    let rc_struct = rc_array_struct(et, count);
    let es = elem_struct(et);

    let ll_arr = val_map[&array];
    let ll_idx = val_map[&index];
    let ll_val = val_map[&value];

    // ZExt index
    let idx64 = ll_func.push_zext(block_id, ll_idx, 64);

    // Load RC
    let hdr = ll_func.push_struct_field_ptr(block_id, ll_arr, rc_struct.clone(), 0);
    let rc_ptr = ll_func.push_struct_field_ptr(block_id, hdr, LLStruct::rc_header(), 0);
    let rc = ll_func.push_load(block_id, rc_ptr, LLType::i64());
    let one = ll_func.push_int_const(block_id, 64, 1);
    let unique = ll_func.push_int_eq(block_id, rc, one);

    // Create blocks
    let mutate_blk = ll_func.add_block();
    let copy_blk = ll_func.add_block();
    let merge_blk = ll_func.add_block();
    let merge_param = ll_func.add_parameter(merge_blk, LLType::Ptr);

    ll_func.terminate_block_with_jmp_if(block_id, unique, mutate_blk, copy_blk);

    // ── Mutate in place ─────────────────────────────────────────────
    {
        let data = ll_func.push_struct_field_ptr(mutate_blk, ll_arr, rc_struct.clone(), 1);
        let slot = ll_func.push_array_elem_ptr(mutate_blk, data, es.clone(), idx64);
        ll_func.push_store(mutate_blk, slot, ll_val);
        ll_func.terminate_block_with_jmp(mutate_blk, merge_blk, vec![ll_arr]);
    }

    // ── Copy then mutate ────────────────────────────────────────────
    {
        // Decrement old RC
        let new_rc = ll_func.push_int_sub(copy_blk, rc, one);
        ll_func.push_store(copy_blk, rc_ptr, new_rc);

        // Allocate new array
        let new_arr = ll_func.push_heap_alloc(copy_blk, rc_struct.clone(), None);

        // Init new RC to 1
        let new_hdr = ll_func.push_struct_field_ptr(copy_blk, new_arr, rc_struct.clone(), 0);
        let new_rc_ptr =
            ll_func.push_struct_field_ptr(copy_blk, new_hdr, LLStruct::rc_header(), 0);
        ll_func.push_store(copy_blk, new_rc_ptr, one);

        // Copy all data
        let old_data = ll_func.push_struct_field_ptr(copy_blk, ll_arr, rc_struct.clone(), 1);
        let new_data = ll_func.push_struct_field_ptr(copy_blk, new_arr, rc_struct, 1);
        let count_val = ll_func.push_int_const(copy_blk, 64, count as u64);
        ll_func.push_memcpy(copy_blk, new_data, old_data, es.clone(), Some(count_val));

        // Write new value at index
        let new_slot = ll_func.push_array_elem_ptr(copy_blk, new_data, es, idx64);
        ll_func.push_store(copy_blk, new_slot, ll_val);

        ll_func.terminate_block_with_jmp(copy_blk, merge_blk, vec![new_arr]);
    }

    val_map.insert(result, merge_param);
    merge_blk
}

/// Lower MemOp::Bump(n) — increment refcount by n.
fn lower_rc_bump(
    ll_func: &mut LLFunction,
    block_id: BlockId,
    val_map: &HashMap<ValueId, ValueId>,
    fn_type_info: &FunctionTypeInfo,
    n: usize,
    value: ValueId,
) {
    let arr_type = fn_type_info.get_value_type(value);
    let (et, count) = array_info(arr_type);
    let rc_struct = rc_array_struct(et, count);

    let ll_arr = val_map[&value];

    let hdr = ll_func.push_struct_field_ptr(block_id, ll_arr, rc_struct, 0);
    let rc_ptr = ll_func.push_struct_field_ptr(block_id, hdr, LLStruct::rc_header(), 0);
    let rc = ll_func.push_load(block_id, rc_ptr, LLType::i64());
    let n_val = ll_func.push_int_const(block_id, 64, n as u64);
    let new_rc = ll_func.push_int_add(block_id, rc, n_val);
    ll_func.push_store(block_id, rc_ptr, new_rc);
}

/// Lower MemOp::Drop — call the generated drop function.
fn lower_rc_drop(
    ll_func: &mut LLFunction,
    block_id: BlockId,
    val_map: &HashMap<ValueId, ValueId>,
    fn_type_info: &FunctionTypeInfo,
    value: ValueId,
    llssa: &mut LLSSA,
    drop_fns: &mut Vec<DropFnEntry>,
) {
    let arr_type = fn_type_info.get_value_type(value);
    let drop_fn_id = get_or_create_drop_fn(arr_type, llssa, drop_fns);
    let ll_arr = val_map[&value];
    ll_func.push_call(block_id, drop_fn_id, vec![ll_arr], 0);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Drop function generation
// ═══════════════════════════════════════════════════════════════════════════════

/// Generate all drop function bodies after user functions are lowered.
fn generate_all_drop_functions(llssa: &mut LLSSA, drop_fns: &[DropFnEntry]) {
    for entry in drop_fns {
        let func = generate_drop_function(&entry.hlssa_type, drop_fns);
        let _old = llssa.take_function(entry.fn_id);
        llssa.put_function(entry.fn_id, func);
    }
}

/// Generate a type-specific drop function.
///
/// Structure:
///   entry: decrement RC, check if zero → free or done
///   free_blk: (optionally loop to drop inner elements) → Free → done
///   done: Return
fn generate_drop_function(ty: &Type, drop_fns: &[DropFnEntry]) -> LLFunction {
    let (et, count) = array_info(ty);
    let rc_struct = rc_array_struct(et, count);
    let es = elem_struct(et);
    let elem_is_ptr = matches!(et.expr, TypeExpr::Array(..));

    let mut func = LLFunction::empty(format!("drop_{}", ty));
    let entry = func.get_entry_id();

    // Parameter: ptr
    let ptr = func.add_parameter(entry, LLType::Ptr);

    // Load RC, decrement, store
    let hdr = func.push_struct_field_ptr(entry, ptr, rc_struct.clone(), 0);
    let rc_ptr = func.push_struct_field_ptr(entry, hdr, LLStruct::rc_header(), 0);
    let rc = func.push_load(entry, rc_ptr, LLType::i64());
    let one = func.push_int_const(entry, 64, 1);
    let new_rc = func.push_int_sub(entry, rc, one);
    func.push_store(entry, rc_ptr, new_rc);

    // Check if dead (new_rc == 0)
    let zero = func.push_int_const(entry, 64, 0);
    let dead = func.push_int_eq(entry, new_rc, zero);

    let free_blk = func.add_block();
    let done_blk = func.add_block();

    func.terminate_block_with_jmp_if(entry, dead, free_blk, done_blk);

    if elem_is_ptr {
        // Elements contain pointers — loop and drop each inner element
        let inner_drop_fn = drop_fns
            .iter()
            .find(|e| e.hlssa_type == *et)
            .expect("inner drop fn should exist")
            .fn_id;

        // free_blk → loop header
        let loop_header = func.add_block();
        let loop_body = func.add_block();
        let loop_exit = func.add_block();

        let init_i = func.push_int_const(free_blk, 64, 0);
        func.terminate_block_with_jmp(free_blk, loop_header, vec![init_i]);

        // loop_header: i = param, check i < count
        let i_param = func.add_parameter(loop_header, LLType::i64());
        let count_val = func.push_int_const(loop_header, 64, count as u64);
        let cmp = func.push_int_ult(loop_header, i_param, count_val);
        func.terminate_block_with_jmp_if(loop_header, cmp, loop_body, loop_exit);

        // loop_body: load element ptr, call inner drop, increment i
        let data = func.push_struct_field_ptr(loop_body, ptr, rc_struct, 1);
        let elem_ptr = func.push_array_elem_ptr(loop_body, data, es, i_param);
        let elem_val = func.push_load(loop_body, elem_ptr, LLType::Ptr);
        func.push_call(loop_body, inner_drop_fn, vec![elem_val], 0);
        let one_loop = func.push_int_const(loop_body, 64, 1);
        let next_i = func.push_int_add(loop_body, i_param, one_loop);
        func.terminate_block_with_jmp(loop_body, loop_header, vec![next_i]);

        // loop_exit: free and jump to done
        func.push_free(loop_exit, ptr);
        func.terminate_block_with_jmp(loop_exit, done_blk, vec![]);
    } else {
        // No recursive drops needed — just free
        func.push_free(free_blk, ptr);
        func.terminate_block_with_jmp(free_blk, done_blk, vec![]);
    }

    // done: return
    func.terminate_block_with_return(done_blk, vec![]);

    func
}

// ═══════════════════════════════════════════════════════════════════════════════
// Terminator lowering
// ═══════════════════════════════════════════════════════════════════════════════

fn lower_terminator(
    terminator: &Terminator,
    ll_func: &mut LLFunction,
    block_id: BlockId,
    val_map: &HashMap<ValueId, ValueId>,
    block_map: &HashMap<BlockId, BlockId>,
) {
    match terminator {
        Terminator::Jmp(target, args) => {
            let ll_target = block_map[target];
            let ll_args: Vec<ValueId> = args.iter().map(|a| val_map[a]).collect();
            ll_func.terminate_block_with_jmp(block_id, ll_target, ll_args);
        }
        Terminator::JmpIf(cond, then_target, else_target) => {
            let ll_cond = val_map[cond];
            let ll_then = block_map[then_target];
            let ll_else = block_map[else_target];
            ll_func.terminate_block_with_jmp_if(block_id, ll_cond, ll_then, ll_else);
        }
        Terminator::Return(values) => {
            let ll_values: Vec<ValueId> = values.iter().map(|v| val_map[v]).collect();
            ll_func.terminate_block_with_return(block_id, ll_values);
        }
    }
}
