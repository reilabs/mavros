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
    BinaryArithOpKind, BlockId, CmpKind, FunctionId, HLFunction, HLSSA, Terminator, ValueId,
};

// =============================================================================
// Type helpers
// =============================================================================

/// Map an HLSSA type to an LLType.
fn lower_type(ty: &Type) -> LLType {
    match &ty.expr {
        TypeExpr::Field => LLType::Struct(LLStruct::field_elem()),
        TypeExpr::U(bits) => LLType::Int(*bits as u32),
        TypeExpr::Array(..) => LLType::Ptr,
        _ => panic!("Unsupported type in HLSSA->LLSSA lowering: {}", ty),
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

// =============================================================================
// Drop function tracking
// =============================================================================

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
) {
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
                    e.field_arith(op, ll_lhs, ll_rhs)
                }
                TypeExpr::U(_) => {
                    let op = match kind {
                        BinaryArithOpKind::Add => IntArithOp::Add,
                        BinaryArithOpKind::Sub => IntArithOp::Sub,
                        BinaryArithOpKind::Mul => IntArithOp::Mul,
                        BinaryArithOpKind::And => IntArithOp::And,
                        _ => panic!("Unsupported int arith op: {:?}", kind),
                    };
                    e.int_arith(op, ll_lhs, ll_rhs)
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
            ConstValue::U(bits, val) => {
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
            lower_array_set(e, val_map, fn_type_info, *result, *array, *index, *value);
        }

        // -- RC operations --
        OpCode::MemOp {
            kind: MemOp::Bump(n),
            value,
        } => {
            lower_rc_bump(e, val_map, fn_type_info, *n, *value);
        }

        OpCode::MemOp {
            kind: MemOp::Drop,
            value,
        } => {
            lower_rc_drop(e, val_map, fn_type_info, *value, llssa, drop_fns);
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
    let data = e.struct_field_ptr(arr, rc_struct, 1);
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

    let data = e.struct_field_ptr(ll_arr, rc_struct, 1);
    let elem_ptr = e.array_elem_ptr(data, es, idx64);
    let val = e.ll_load(elem_ptr, ll_elem_type);

    val_map.insert(result, val);
}

/// Lower ArraySet with copy-on-write semantics.
/// Creates new blocks for the RC check + conditional copy.
fn lower_array_set(
    e: &mut LLBlockEmitter<'_>,
    val_map: &mut HashMap<ValueId, ValueId>,
    fn_type_info: &FunctionTypeInfo,
    result: ValueId,
    array: ValueId,
    index: ValueId,
    value: ValueId,
) {
    let arr_type = fn_type_info.get_value_type(array);
    let (et, count) = array_info(arr_type);
    let rc_struct = rc_array_struct(et, count);
    let es = elem_struct(et);

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
            let data = me.struct_field_ptr(ll_arr, rc_struct.clone(), 1);
            let slot = me.array_elem_ptr(data, es.clone(), idx64);
            me.ll_store(slot, ll_val);
            vec![ll_arr]
        },
        // -- Copy then mutate --
        |ce| {
            // Decrement old RC
            let new_rc = ce.int_sub(rc, one);
            ce.ll_store(rc_ptr, new_rc);

            // Allocate new array
            let new_arr = ce.heap_alloc(rc_struct.clone(), None);

            // Init new RC to 1
            let new_hdr = ce.struct_field_ptr(new_arr, rc_struct.clone(), 0);
            let new_rc_ptr = ce.struct_field_ptr(new_hdr, LLStruct::rc_header(), 0);
            ce.ll_store(new_rc_ptr, one);

            // Copy all data
            let old_data = ce.struct_field_ptr(ll_arr, rc_struct.clone(), 1);
            let new_data = ce.struct_field_ptr(new_arr, rc_struct.clone(), 1);
            let count_val = ce.int_const(64, count as u64);
            ce.memcpy(new_data, old_data, es.clone(), Some(count_val));

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
) {
    let arr_type = fn_type_info.get_value_type(value);
    let drop_fn_id = get_or_create_drop_fn(arr_type, llssa, drop_fns);
    let ll_arr = val_map[&value];
    e.call(drop_fn_id, vec![ll_arr], 0);
}

// =============================================================================
// Drop function generation
// =============================================================================

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
///   entry: decrement RC, check if zero -> free or done
///   free_blk: (optionally loop to drop inner elements) -> Free -> done
///   done: Return
fn generate_drop_function(ty: &Type, drop_fns: &[DropFnEntry]) -> LLFunction {
    let (et, count) = array_info(ty);
    let rc_struct = rc_array_struct(et, count);
    let es = elem_struct(et);
    let elem_is_ptr = matches!(et.expr, TypeExpr::Array(..));

    let mut func = LLFunction::empty(format!("drop_{}", ty));
    let entry = func.get_entry_id();
    let (free_blk, _) = func.add_block_mut();
    let (done_blk, _) = func.add_block_mut();

    {
        let mut e = LLBlockEmitter::new(&mut func, entry);

        // Parameter: ptr
        let ptr = e.add_parameter(LLType::Ptr);

        // Load RC, decrement, store
        let hdr = e.struct_field_ptr(ptr, rc_struct.clone(), 0);
        let rc_ptr = e.struct_field_ptr(hdr, LLStruct::rc_header(), 0);
        let rc = e.ll_load(rc_ptr, LLType::i64());
        let one = e.int_const(64, 1);
        let new_rc = e.int_sub(rc, one);
        e.ll_store(rc_ptr, new_rc);

        // Check if dead (new_rc == 0)
        let zero = e.int_const(64, 0);
        let dead = e.int_eq(new_rc, zero);

        e.terminate_jmp_if(dead, free_blk, done_blk);
    }

    if elem_is_ptr {
        // Elements contain pointers -- loop and drop each inner element
        let inner_drop_fn = drop_fns
            .iter()
            .find(|e| e.hlssa_type == *et)
            .expect("inner drop fn should exist")
            .fn_id;

        // We need the ptr value to access inside the loop body.
        // Re-read it from the entry block parameters.
        let ptr = func
            .get_block(entry)
            .get_parameter_values()
            .next()
            .copied()
            .unwrap();

        let mut e = LLBlockEmitter::new(&mut func, free_blk);
        let data = e.struct_field_ptr(ptr, rc_struct, 1);

        e.build_counted_loop(count, vec![], |e, i_val, _| {
            let elem_ptr = e.array_elem_ptr(data, es.clone(), i_val);
            let elem_val = e.ll_load(elem_ptr, LLType::Ptr);
            e.call(inner_drop_fn, vec![elem_val], 0);
            vec![]
        });

        e.free(ptr);
        e.terminate_jmp(done_blk, vec![]);
    } else {
        // No recursive drops needed -- just free
        let ptr = func
            .get_block(entry)
            .get_parameter_values()
            .next()
            .copied()
            .unwrap();

        let mut e = LLBlockEmitter::new(&mut func, free_blk);
        e.free(ptr);
        e.terminate_jmp(done_blk, vec![]);
    }

    // done: return
    func.terminate_block_with_return(done_blk, vec![]);

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
