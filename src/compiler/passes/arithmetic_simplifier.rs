use std::collections::HashMap;

use crate::compiler::{
    analysis::{
        types::TypeInfo,
        value_definitions::{ValueDefinition, ValueDefinitions},
    },
    flow_analysis::FlowAnalysis,
    ir::r#type::TypeExpr,
    pass_manager::{Analysis, AnalysisId, AnalysisStore, Pass},
    ssa::{CastTarget, CmpKind, OpCode},
};

pub struct ArithmeticSimplifier {}

impl Pass for ArithmeticSimplifier {
    fn name(&self) -> &'static str {
        "arithmetic_simplifier"
    }

    fn needs(&self) -> Vec<AnalysisId> {
        vec![TypeInfo::id(), ValueDefinitions::id()]
    }

    fn run(&self, ssa: &mut crate::compiler::ssa::HLSSA, store: &AnalysisStore) {
        self.do_run(
            ssa,
            store.get::<TypeInfo>(),
            store.get::<ValueDefinitions>(),
        );
    }

    fn preserves(&self) -> Vec<AnalysisId> {
        vec![FlowAnalysis::id()]
    }
}

impl ArithmeticSimplifier {
    pub fn new() -> Self {
        Self {}
    }

    pub fn do_run(
        &self,
        ssa: &mut crate::compiler::ssa::HLSSA,
        type_info: &crate::compiler::analysis::types::TypeInfo,
        value_definitions: &crate::compiler::analysis::value_definitions::ValueDefinitions,
    ) {
        for (function_id, function) in ssa.iter_functions_mut() {
            let type_info = type_info.get_function(*function_id);
            let value_definitions = value_definitions.get_function(*function_id);
            let mut new_blocks = HashMap::new();
            for (bid, mut block) in function.take_blocks().into_iter() {
                let mut new_instructions = Vec::new();
                for instruction in block.take_instructions().into_iter() {
                    match instruction {
                        OpCode::Rangecheck {
                            value: v,
                            max_bits: bits,
                        } => {
                            let v_definition = value_definitions.get_definition(v);
                            match v_definition {
                                ValueDefinition::Instruction(
                                    _,
                                    _,
                                    OpCode::Cast {
                                        result: _,
                                        value: v,
                                        target: CastTarget::Field,
                                    },
                                ) => {
                                    let v_type = type_info.get_value_type(*v);
                                    if v_type.is_witness_of() {
                                        panic!("Rangecheck on impure value");
                                    }
                                    match &v_type.expr {
                                        TypeExpr::U(s) => {
                                            let cst = function.fresh_value();
                                            new_instructions.push(OpCode::mk_u_const(cst, *s, 1 << bits));
                                            let r = function.fresh_value();
                                            let t = function.fresh_value();
                                            new_instructions.push(OpCode::mk_u_const(t, 1, 1));
                                            new_instructions.push(OpCode::Cmp {
                                                kind: CmpKind::Lt,
                                                result: r,
                                                lhs: *v,
                                                rhs: cst,
                                            });
                                            new_instructions
                                                .push(OpCode::AssertEq { lhs: r, rhs: t });
                                        }
                                        _ => panic!(
                                            "Rangecheck on a cast of a non-u value {}",
                                            v_type
                                        ),
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
