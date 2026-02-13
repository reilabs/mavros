use std::collections::HashMap;
use std::path::Path;

use ark_ff::BigInt;
use inkwell::context::Context;
use inkwell::memory_buffer::MemoryBuffer;
use inkwell::values::{AnyValue, AsValueRef, BasicValue, BasicValueEnum, CallSiteValue, InstructionOpcode, InstructionValue, IntValue, PhiValue};
use inkwell::{IntPredicate, types::BasicTypeEnum};

use crate::compiler::Field;
use crate::compiler::ir::r#type::{Empty, Type};
use crate::compiler::ssa::{BlockId, Function, SSA, ValueId};

/// Convert a constrained subset of LLVM IR into mavros SSA.
///
/// Supported today:
/// - Single entry function named `mavros_main`
/// - Integer add/sub/mul/div/and, integer `icmp` (eq, ult)
/// - `phi` nodes (translated to block parameters)
/// - `br`/`ret` terminators
/// - Calls to `__field_mul` and `__assert_eq`
///
/// Field values are expected to use one of:
/// - `[4 x i64]`
/// - `<4 x i64>`
/// - a single-field struct wrapping one of the above.
pub fn from_llvm_ir_file(path: &Path) -> Result<SSA<Empty>, String> {
    let ir = std::fs::read(path)
        .map_err(|e| format!("failed to read LLVM IR file {}: {e}", path.display()))?;
    from_llvm_ir_bytes(&ir)
}

pub fn from_llvm_ir_text(ir: &str) -> Result<SSA<Empty>, String> {
    from_llvm_ir_bytes(ir.as_bytes())
}

fn from_llvm_ir_bytes(ir: &[u8]) -> Result<SSA<Empty>, String> {
    let context = Context::create();
    let buffer = MemoryBuffer::create_from_memory_range_copy(ir, "mavros_import.ll");
    let module = context
        .create_module_from_ir(buffer)
        .map_err(|e| format!("failed to parse LLVM IR: {e}"))?;

    let main = module
        .get_function("mavros_main")
        .ok_or_else(|| "LLVM module must define `mavros_main`".to_string())?;

    let mut ssa = SSA::new();
    let converted = convert_function(main)?;
    *ssa.get_main_mut() = converted;
    Ok(ssa)
}

fn convert_function(main: inkwell::values::FunctionValue<'_>) -> Result<Function<Empty>, String> {
    let mut function = Function::empty("main".to_string());
    let entry_block = function.get_entry_id();

    let mut block_map: HashMap<inkwell::basic_block::BasicBlock<'_>, BlockId> = HashMap::new();
    let llvm_blocks = main.get_basic_blocks();
    if llvm_blocks.is_empty() {
        return Err("mavros_main has no basic blocks".to_string());
    }

    for (idx, bb) in llvm_blocks.iter().enumerate() {
        let id = if idx == 0 {
            entry_block
        } else {
            function.add_block()
        };
        block_map.insert(*bb, id);
    }

    let mut value_map: HashMap<usize, ValueId> = HashMap::new();

    for param in main.get_param_iter() {
        let typ = convert_llvm_type(param.get_type())?;
        let id = function.add_parameter(entry_block, typ);
        value_map.insert(param.as_value_ref() as usize, id);
    }

    // Collect phi definitions and incoming edge arguments.
    let mut edge_args: HashMap<(BlockId, BlockId), Vec<BasicValueEnum<'_>>> = HashMap::new();
    for bb in &llvm_blocks {
        let target_id = block_map[bb];
        for inst in bb.get_instructions() {
            if inst.get_opcode() != InstructionOpcode::Phi {
                break;
            }
            let phi = PhiValue::try_from(inst).map_err(|_| "internal phi conversion error")?;
            let param_type = convert_llvm_type(phi.as_basic_value().get_type())?;
            let param_id = function.add_parameter(target_id, param_type);
            value_map.insert(phi.as_value_ref() as usize, param_id);

            for (incoming_value, incoming_bb) in phi.get_incomings() {
                let pred_id = block_map[&incoming_bb];
                edge_args
                    .entry((pred_id, target_id))
                    .or_default()
                    .push(incoming_value);
            }
        }
    }

    for bb in &llvm_blocks {
        let block_id = block_map[bb];

        for inst in bb.get_instructions() {
            match inst.get_opcode() {
                InstructionOpcode::Phi => {}
                InstructionOpcode::Add
                | InstructionOpcode::Sub
                | InstructionOpcode::Mul
                | InstructionOpcode::UDiv
                | InstructionOpcode::And => {
                    let lhs = resolve_operand(&mut function, &value_map, inst, 0)?;
                    let rhs = resolve_operand(&mut function, &value_map, inst, 1)?;

                    let result = match inst.get_opcode() {
                        InstructionOpcode::Add => function.push_add(block_id, lhs, rhs),
                        InstructionOpcode::Sub => function.push_sub(block_id, lhs, rhs),
                        InstructionOpcode::Mul => function.push_mul(block_id, lhs, rhs),
                        InstructionOpcode::UDiv => function.push_div(block_id, lhs, rhs),
                        InstructionOpcode::And => function.push_and(block_id, lhs, rhs),
                        _ => unreachable!(),
                    };

                    value_map.insert(inst.as_value_ref() as usize, result);
                }
                InstructionOpcode::ICmp => {
                    let lhs = resolve_operand(&mut function, &value_map, inst, 0)?;
                    let rhs = resolve_operand(&mut function, &value_map, inst, 1)?;
                    let pred = inst
                        .get_icmp_predicate()
                        .ok_or_else(|| "icmp instruction missing predicate".to_string())?;

                    let result = match pred {
                        IntPredicate::EQ => function.push_eq(block_id, lhs, rhs),
                        IntPredicate::ULT => function.push_lt(block_id, lhs, rhs),
                        _ => {
                            return Err(format!(
                                "unsupported icmp predicate {:?}; only eq/ult are supported",
                                pred
                            ));
                        }
                    };

                    value_map.insert(inst.as_value_ref() as usize, result);
                }
                InstructionOpcode::Call => {
                    convert_call(&mut function, block_id, &mut value_map, inst)?;
                }
                InstructionOpcode::Br => {
                    convert_branch(
                        &mut function,
                        block_id,
                        &value_map,
                        &edge_args,
                        &block_map,
                        inst,
                    )?;
                }
                InstructionOpcode::Return => {
                    if inst.get_num_operands() == 0 {
                        function.terminate_block_with_return(block_id, vec![]);
                    } else {
                        let value = resolve_operand(&mut function, &value_map, inst, 0)?;
                        function.terminate_block_with_return(block_id, vec![value]);
                    }
                }
                op => {
                    return Err(format!(
                        "unsupported LLVM opcode in importer: {:?} (block {:?})",
                        op,
                        bb.get_name()
                    ));
                }
            }
        }

        if function.get_block(block_id).get_terminator().is_none() {
            return Err(format!(
                "LLVM block {:?} has no terminator after import",
                bb.get_name()
            ));
        }
    }

    Ok(function)
}

fn convert_branch(
    function: &mut Function<Empty>,
    block_id: BlockId,
    value_map: &HashMap<usize, ValueId>,
    edge_args: &HashMap<(BlockId, BlockId), Vec<BasicValueEnum<'_>>>,
    block_map: &HashMap<inkwell::basic_block::BasicBlock<'_>, BlockId>,
    inst: InstructionValue<'_>,
) -> Result<(), String> {
    match inst.get_num_operands() {
        1 => {
            let target_bb = inst
                .get_operand(0)
                .and_then(|v| v.right())
                .ok_or_else(|| "malformed unconditional branch".to_string())?;
            let target_id = block_map[&target_bb];

            let args = edge_args
                .get(&(block_id, target_id))
                .map(|values| {
                    values
                        .iter()
                        .map(|v| resolve_value(function, value_map, *v))
                        .collect::<Result<Vec<_>, _>>()
                })
                .transpose()?
                .unwrap_or_default();

            function.terminate_block_with_jmp(block_id, target_id, args);
            Ok(())
        }
        3 => {
            let cond = inst
                .get_operand(0)
                .and_then(|v| v.left())
                .ok_or_else(|| "malformed conditional branch: missing condition".to_string())?;
            let cond_id = resolve_value(function, value_map, cond)?;

            let then_bb = inst
                .get_operand(1)
                .and_then(|v| v.right())
                .ok_or_else(|| "malformed conditional branch: missing then target".to_string())?;
            let else_bb = inst
                .get_operand(2)
                .and_then(|v| v.right())
                .ok_or_else(|| "malformed conditional branch: missing else target".to_string())?;

            function.terminate_block_with_jmp_if(
                block_id,
                cond_id,
                block_map[&then_bb],
                block_map[&else_bb],
            );
            Ok(())
        }
        n => Err(format!("unsupported branch operand count: {n}")),
    }
}

fn convert_call(
    function: &mut Function<Empty>,
    block_id: BlockId,
    value_map: &mut HashMap<usize, ValueId>,
    inst: InstructionValue<'_>,
) -> Result<(), String> {
    let call = CallSiteValue::try_from(inst).map_err(|_| "internal call conversion error")?;
    let callee = call.get_called_fn_value();
    let callee_name = callee
        .get_name()
        .to_str()
        .map_err(|_| "callee name is not valid utf-8")?;

    let arg_count = inst.get_num_operands().saturating_sub(1);
    let mut args = Vec::with_capacity(arg_count as usize);
    for i in 0..arg_count {
        let operand = inst
            .get_operand(i)
            .and_then(|v| v.left())
            .ok_or_else(|| format!("malformed call operand at index {i}"))?;
        let arg = resolve_value(function, value_map, operand)?;
        args.push(arg);
    }

    match callee_name {
        "__field_mul" => {
            if args.len() != 2 {
                return Err("__field_mul expects exactly 2 arguments".to_string());
            }
            let result = function.push_mul(block_id, args[0], args[1]);
            value_map.insert(inst.as_value_ref() as usize, result);
            Ok(())
        }
        "__assert_eq" => {
            if args.len() != 2 {
                return Err("__assert_eq expects exactly 2 arguments".to_string());
            }
            function.push_assert_eq(block_id, args[0], args[1]);
            Ok(())
        }
        _ => Err(format!(
            "unsupported call target `{callee_name}`; expected __field_mul or __assert_eq"
        )),
    }
}

fn resolve_operand(
    function: &mut Function<Empty>,
    value_map: &HashMap<usize, ValueId>,
    inst: InstructionValue<'_>,
    index: u32,
) -> Result<ValueId, String> {
    let op = inst
        .get_operand(index)
        .and_then(|v| v.left())
        .ok_or_else(|| format!("malformed instruction operand at index {index}"))?;
    resolve_value(function, value_map, op)
}

fn resolve_value(
    function: &mut Function<Empty>,
    value_map: &HashMap<usize, ValueId>,
    value: BasicValueEnum<'_>,
) -> Result<ValueId, String> {
    if let Some(id) = value_map.get(&(value.as_value_ref() as usize)) {
        return Ok(*id);
    }

    if let Some(inst) = value.as_instruction_value()
        && let Some(id) = value_map.get(&(inst.as_value_ref() as usize))
    {
        return Ok(*id);
    }

    match value {
        BasicValueEnum::IntValue(int_val) => import_int_const(function, int_val),
        BasicValueEnum::ArrayValue(_) | BasicValueEnum::VectorValue(_) | BasicValueEnum::StructValue(_) => {
            import_field_const(function, value)
        }
        _ => Err(format!(
            "unsupported LLVM operand value: {}",
            value.print_to_string()
        )),
    }
}

fn import_int_const(function: &mut Function<Empty>, int_val: IntValue<'_>) -> Result<ValueId, String> {
    if !int_val.is_constant_int() {
        return Err(format!(
            "expected integer constant, got non-constant: {}",
            int_val.print_to_string()
        ));
    }
    let width = int_val.get_type().get_bit_width() as usize;
    let value = int_val
        .get_zero_extended_constant()
        .ok_or_else(|| "integer constant does not fit in u64".to_string())?;
    Ok(function.push_u_const(width, value as u128))
}

fn import_field_const(function: &mut Function<Empty>, value: BasicValueEnum<'_>) -> Result<ValueId, String> {
    let text = value.print_to_string().to_string();
    if text.contains("zeroinitializer") {
        let field = Field::new_unchecked(BigInt::new([0, 0, 0, 0]));
        return Ok(function.push_field_const(field));
    }

    let limbs = parse_i64_literals(&text);
    if limbs.len() < 4 {
        return Err(format!(
            "failed to parse Field constant from LLVM value: {text}"
        ));
    }

    let field = Field::new_unchecked(BigInt::new([
        limbs[0], limbs[1], limbs[2], limbs[3],
    ]));
    Ok(function.push_field_const(field))
}

fn parse_i64_literals(text: &str) -> Vec<u64> {
    let mut out = Vec::new();
    let mut rest = text;
    while let Some(i) = rest.find("i64 ") {
        rest = &rest[i + 4..];
        let end = rest
            .find(|c: char| !(c.is_ascii_digit() || c == '-'))
            .unwrap_or(rest.len());
        if end == 0 {
            continue;
        }
        if let Ok(v) = rest[..end].parse::<i128>() {
            out.push(v as u64);
        }
        rest = &rest[end..];
    }
    out
}

fn convert_llvm_type(ty: BasicTypeEnum<'_>) -> Result<Type<Empty>, String> {
    match ty {
        BasicTypeEnum::IntType(i) => Ok(Type::u(i.get_bit_width() as usize, Empty)),
        BasicTypeEnum::ArrayType(a) => {
            if a.len() == 4 {
                if let BasicTypeEnum::IntType(elem) = a.get_element_type()
                    && elem.get_bit_width() == 64
                {
                    return Ok(Type::field(Empty));
                }
            }
            Err(format!("unsupported LLVM array type for import: {a}"))
        }
        BasicTypeEnum::VectorType(v) => {
            if v.get_size() == 4 {
                if let BasicTypeEnum::IntType(elem) = v.get_element_type()
                    && elem.get_bit_width() == 64
                {
                    return Ok(Type::field(Empty));
                }
            }
            Err(format!("unsupported LLVM vector type for import: {v}"))
        }
        BasicTypeEnum::StructType(s) => {
            if s.count_fields() == 1 {
                let inner = s.get_field_types()[0];
                return convert_llvm_type(inner);
            }
            Err(format!("unsupported LLVM struct type for import: {s}"))
        }
        _ => Err(format!("unsupported LLVM type for import: {ty}")),
    }
}

#[cfg(test)]
mod tests {
    use crate::compiler::ssa::{BinaryArithOpKind, OpCode, Terminator};

    use super::from_llvm_ir_text;

    #[test]
    fn imports_power_style_llvm() {
        let ir = include_str!("../../../llvm_tests/power/power.ll");
        let ssa = from_llvm_ir_text(ir).expect("llvm import should succeed");
        let main = ssa.get_main();

        assert!(main.get_entry().get_parameters().count() >= 2);

        let ops = main
            .get_blocks()
            .flat_map(|(_, b)| b.get_instructions())
            .collect::<Vec<_>>();

        assert!(ops.iter().any(|op| matches!(op, OpCode::BinaryArithOp { kind: BinaryArithOpKind::Mul, .. })));
        assert!(ops.iter().any(|op| matches!(op, OpCode::AssertEq { .. })));
        assert!(main
            .get_blocks()
            .any(|(_, b)| matches!(b.get_terminator(), Some(Terminator::JmpIf(_, _, _)))));
        assert!(main
            .get_blocks()
            .any(|(_, b)| matches!(b.get_terminator(), Some(Terminator::Jmp(_, _)))));
    }
}
