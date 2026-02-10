use std::collections::HashMap;

use crate::compiler::{
    analysis::value_definitions::ValueDefinition, ir::r#type::TypeExpr, pass_manager::{DataPoint, Pass}, ssa::{CastTarget, CmpKind, OpCode}, taint_analysis::ConstantTaint
};

pub struct ArithmeticSimplifier {}

impl Pass<ConstantTaint> for ArithmeticSimplifier {
    fn run(
        &self,
        ssa: &mut crate::compiler::ssa::SSA<ConstantTaint>,
        pass_manager: &crate::compiler::pass_manager::PassManager<ConstantTaint>,
    ) {
        self.do_run(ssa, pass_manager.get_type_info(), pass_manager.get_value_definitions());
    }

    fn pass_info(&self) -> crate::compiler::pass_manager::PassInfo {
        crate::compiler::pass_manager::PassInfo {
            name: "arithmetic_simplifier",
            needs: vec![DataPoint::Types, DataPoint::ValueDefinitions],
        }
    }

    fn invalidates_cfg(&self) -> bool {
        false
    }
}

impl ArithmeticSimplifier {
    pub fn new() -> Self {
        Self {}
    }

    pub fn do_run(
        &self,
        ssa: &mut crate::compiler::ssa::SSA<ConstantTaint>,
        type_info: &crate::compiler::analysis::types::TypeInfo<ConstantTaint>,
        value_definitions: &crate::compiler::analysis::value_definitions::ValueDefinitions<
            ConstantTaint,
        >,
    ) {
        for (function_id, function) in ssa.iter_functions_mut() {
            let type_info = type_info.get_function(*function_id);
            let value_definitions = value_definitions.get_function(*function_id);
            let mut new_blocks = HashMap::new();
            for (bid, mut block) in function.take_blocks().into_iter() {
                let mut new_instructions = Vec::new();
                for instruction in block.take_instructions().into_iter() {
                    match instruction {
                        OpCode::Rangecheck { value: v, max_bits: bits } => {
                            let v_definition = value_definitions.get_definition(v);
                            match v_definition {
                                ValueDefinition::Instruction(_, _, OpCode::Cast { result: _, value: v, target: CastTarget::Field }) => {
                                    let v_type = type_info.get_value_type(*v);
                                    if !matches!(v_type.annotation, ConstantTaint::Pure) {
                                        // Witness-tainted rangechecks are handled by explicit_witness
                                        new_instructions.push(instruction);
                                        continue;
                                    }
                                    match &v_type.expr {
                                        TypeExpr::U(s) => {
                                            let cst = function.push_u_const(*s, 1 << bits);
                                            let r = function.fresh_value();
                                            let t = function.push_u_const(1,1);
                                            new_instructions.push(OpCode::Cmp {
                                                kind: CmpKind::Lt,
                                                result: r,
                                                lhs: *v,
                                                rhs: cst
                                            });
                                            new_instructions.push(OpCode::AssertEq {
                                                lhs: r,
                                                rhs: t
                                            });
                                        }
                                        _ => panic!("Rangecheck on a cast of a non-u value {}", v_type),
                                    }


                                }
                                _ => new_instructions.push(instruction),
                            }
                        }
                        _ => new_instructions.push(instruction),
                    }
                }
                block.put_instructions(new_instructions);
                new_blocks.insert(bid, block);
            }
            function.put_blocks(new_blocks);
        }
    }
}
