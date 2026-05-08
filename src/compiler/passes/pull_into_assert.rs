use std::collections::HashMap;

use crate::compiler::{
    analysis::types::{FunctionTypeInfo, TypeInfo},
    flow_analysis::FlowAnalysis,
    ir::r#type::TypeExpr,
    pass_manager::{Analysis, AnalysisId, AnalysisStore, Pass},
    ssa::{BinaryArithOpKind, CmpKind, HLSSA, Instruction, OpCode, ValueId},
};

pub struct PullIntoAssert {}

impl Pass for PullIntoAssert {
    fn name(&self) -> &'static str {
        "pull_into_assert"
    }
    fn needs(&self) -> Vec<AnalysisId> {
        vec![TypeInfo::id()]
    }
    fn run(&self, ssa: &mut HLSSA, store: &AnalysisStore) {
        self.do_run(ssa, store.get::<TypeInfo>());
    }
    fn preserves(&self) -> Vec<AnalysisId> {
        vec![FlowAnalysis::id(), TypeInfo::id()]
    }
}

impl PullIntoAssert {
    pub fn new() -> Self {
        Self {}
    }

    pub fn do_run(&self, ssa: &mut HLSSA, type_info: &TypeInfo) {
        for (function_id, function) in ssa.iter_functions_mut() {
            let function_type_info = type_info.get_function(*function_id);

            let mut defs: HashMap<ValueId, OpCode> = HashMap::new();
            for (_, block) in function.get_blocks() {
                for instruction in block.get_instructions() {
                    for result in instruction.get_results() {
                        defs.insert(*result, instruction.clone());
                    }
                }
            }

            let mut new_blocks = HashMap::new();
            for (block_id, mut block) in function.take_blocks() {
                let mut new_instructions = Vec::new();
                for instruction in block.take_instructions().into_iter() {
                    match instruction {
                        OpCode::Assert { value } => {
                            new_instructions.extend(emit_assert(value, &defs, function_type_info));
                        }
                        OpCode::AssertCmp {
                            kind: CmpKind::Eq,
                            lhs,
                            rhs,
                        } => {
                            new_instructions.extend(emit_assert_eq(
                                lhs,
                                rhs,
                                &defs,
                                function_type_info,
                            ));
                        }
                        _ => {
                            new_instructions.push(instruction);
                        }
                    }
                }
                block.put_instructions(new_instructions);
                new_blocks.insert(block_id, block);
            }
            function.put_blocks(new_blocks);
        }
    }
}

fn emit_assert(
    value: ValueId,
    defs: &HashMap<ValueId, OpCode>,
    function_type_info: &FunctionTypeInfo,
) -> Vec<OpCode> {
    match defs.get(&value) {
        Some(OpCode::Cmp {
            kind: CmpKind::Eq,
            result: _,
            lhs,
            rhs,
        }) => emit_assert_eq(*lhs, *rhs, defs, function_type_info),

        Some(OpCode::Cmp {
            kind: CmpKind::Lt,
            result: _,
            lhs,
            rhs,
        }) => vec![OpCode::AssertCmp {
            kind: CmpKind::Lt,
            lhs: *lhs,
            rhs: *rhs,
        }],

        Some(OpCode::BinaryArithOp {
            kind: BinaryArithOpKind::And,
            result,
            lhs,
            rhs,
        }) => {
            let result_type = function_type_info.get_value_type(*result);
            match result_type.strip_witness().expr {
                TypeExpr::U(1) => {
                    let mut out = emit_assert(*lhs, defs, function_type_info);
                    out.extend(emit_assert(*rhs, defs, function_type_info));
                    out
                }
                _ => vec![OpCode::Assert { value }],
            }
        }

        _ => vec![OpCode::Assert { value }],
    }
}

fn emit_assert_eq(
    lhs: ValueId,
    rhs: ValueId,
    defs: &HashMap<ValueId, OpCode>,
    function_type_info: &FunctionTypeInfo,
) -> Vec<OpCode> {
    if let Some(OpCode::BinaryArithOp {
        kind: BinaryArithOpKind::Mul,
        result,
        lhs: a,
        rhs: b,
    }) = defs.get(&lhs)
    {
        let result_type = function_type_info.get_value_type(*result);
        if matches!(result_type.strip_witness().expr, TypeExpr::Field) {
            return vec![OpCode::AssertR1C {
                a: *a,
                b: *b,
                c: rhs,
            }];
        }
    }

    if let Some(OpCode::BinaryArithOp {
        kind: BinaryArithOpKind::Mul,
        result,
        lhs: a,
        rhs: b,
    }) = defs.get(&rhs)
    {
        let result_type = function_type_info.get_value_type(*result);
        if matches!(result_type.strip_witness().expr, TypeExpr::Field) {
            return vec![OpCode::AssertR1C {
                a: *a,
                b: *b,
                c: lhs,
            }];
        }
    }

    vec![OpCode::AssertCmp {
        kind: CmpKind::Eq,
        lhs,
        rhs,
    }]
}
