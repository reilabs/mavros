//! HLSSA → LLSSA lowering pass
//!
//! Translates the high-level SSA (with abstract Field/U types and a separate
//! constant map) into low-level SSA (explicit integer widths, field-as-struct,
//! constants-as-instructions).
//!
//! Currently supports only the minimal subset needed for WASM witness generation:
//! BinaryArithOp, Cmp, Constrain, WriteWitness, Call (static), Not, Select,
//! plus constants (U, Field) and terminators (Jmp, JmpIf, Return).

use std::collections::HashMap;

use crate::compiler::analysis::types::TypeInfo;
use crate::compiler::flow_analysis::FlowAnalysis;
use crate::compiler::ir::r#type::TypeExpr;
use crate::compiler::llssa::{
    FieldArithOp, IntArithOp, IntCmpOp, LLFieldType, LLFunction, LLSSA, LLStruct, LLType,
};
use crate::compiler::ssa::{
    BinaryArithOpKind, BlockId, CmpKind, FunctionId, HLFunction, HLSSA, Terminator, ValueId,
};

/// Map an HLSSA type to an LLType.
fn lower_type(ty: &crate::compiler::ir::r#type::Type) -> LLType {
    match &ty.expr {
        TypeExpr::Field => LLType::Struct(LLStruct::field_elem()),
        TypeExpr::U(bits) => LLType::Int(*bits as u32),
        _ => panic!("Unsupported type in HLSSA→LLSSA lowering: {}", ty),
    }
}

/// Lower an entire HLSSA program to LLSSA.
pub fn lower(hlssa: &HLSSA, flow_analysis: &FlowAnalysis, type_info: &TypeInfo) -> LLSSA {
    let main_id = hlssa.get_main_id();
    let main_name = hlssa.get_main().get_name().to_string();

    let mut llssa = LLSSA::with_main(main_name);

    // First pass: create all functions (so we can map FunctionIds)
    let mut fn_map: HashMap<FunctionId, FunctionId> = HashMap::new();

    // Main is already created as FunctionId(0)
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

        let ll_func = lower_function(function, fn_type_info, cfg, &fn_map);

        // Replace the placeholder function with the lowered body
        let _old = llssa.take_function(ll_fn_id);
        llssa.put_function(ll_fn_id, ll_func);
    }

    llssa
}

/// Lower a single HLSSA function to an LLFunction.
fn lower_function(
    function: &HLFunction,
    fn_type_info: &crate::compiler::analysis::types::FunctionTypeInfo,
    cfg: &crate::compiler::flow_analysis::CFG,
    fn_map: &HashMap<FunctionId, FunctionId>,
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

        // Lower instructions
        for instruction in block.get_instructions() {
            lower_instruction(
                instruction,
                &mut ll_func,
                ll_block_id,
                &mut val_map,
                fn_type_info,
                fn_map,
            );
        }

        // Lower terminator
        if let Some(terminator) = block.get_terminator() {
            lower_terminator(terminator, &mut ll_func, ll_block_id, &val_map, &block_map);
        }
    }

    ll_func
}

/// Lower a single HLSSA instruction to LLSSA ops.
fn lower_instruction(
    instruction: &crate::compiler::ssa::OpCode,
    ll_func: &mut LLFunction,
    block_id: BlockId,
    val_map: &mut HashMap<ValueId, ValueId>,
    fn_type_info: &crate::compiler::analysis::types::FunctionTypeInfo,
    fn_map: &HashMap<FunctionId, FunctionId>,
) {
    use crate::compiler::ssa::{CallTarget, ConstValue, OpCode};

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
                _ => panic!("Unsupported type for Cmp in lowering: {:?}", lhs_type),
            };
            val_map.insert(*result, ll_result);
        }

        OpCode::Constrain { a, b, c } => {
            let ll_a = val_map[a];
            let ll_b = val_map[b];
            let ll_c = val_map[c];
            ll_func.push_constrain(block_id, ll_a, ll_b, ll_c);
        }

        OpCode::WriteWitness { result, value, .. } => {
            let ll_value = val_map[value];
            ll_func.push_write_witness(block_id, ll_value);
            // WriteWitness result maps to the input value (pass-through)
            if let Some(result_id) = result {
                val_map.insert(*result_id, ll_value);
            }
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
        }

        OpCode::Not { result, value } => {
            let ll_value = val_map[value];
            let ll_result = ll_func.push_not(block_id, ll_value);
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
            let ll_result = ll_func.push_select(block_id, ll_cond, ll_if_t, ll_if_f);
            val_map.insert(*result, ll_result);
        }

        OpCode::Const { result, value } => match value {
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
        },

        _ => panic!(
            "Unsupported opcode in HLSSA→LLSSA lowering: {:?}",
            instruction
        ),
    }
}

/// Lower a terminator from HLSSA to LLSSA.
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
