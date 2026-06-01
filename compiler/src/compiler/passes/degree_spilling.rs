use std::collections::HashMap;

use crate::compiler::{
    analysis::{
        flow_analysis::FlowAnalysis,
        types::{FunctionTypeInfo, TypeInfo},
    },
    pass_manager::{Analysis, AnalysisId, AnalysisStore, Pass},
    ssa::{
        BlockId, ValueId,
        hlssa::{
            BinaryArithOpKind, HLBlock, HLSSA, OpCode, TypeExpr,
            builder::{HLEmitter, HLInstrBuilder, HLSSABuilder},
        },
    },
};

pub struct DegreeSpilling {}

impl Pass for DegreeSpilling {
    fn name(&self) -> &'static str {
        "degree_spilling"
    }

    fn needs(&self) -> Vec<AnalysisId> {
        vec![TypeInfo::id()]
    }

    fn run(&self, ssa: &mut HLSSA, store: &AnalysisStore) {
        self.do_run(ssa, store.get::<TypeInfo>());
    }

    fn preserves(&self) -> Vec<AnalysisId> {
        vec![FlowAnalysis::id()]
    }
}

impl DegreeSpilling {
    pub fn new() -> Self {
        Self {}
    }

    pub fn do_run(&self, ssa: &mut HLSSA, type_info: &TypeInfo) {
        let fids: Vec<_> = ssa.get_function_ids().collect();
        let mut sb = HLSSABuilder::new(ssa);
        for function_id in fids {
            let function_type_info = type_info.get_function(function_id);
            sb.modify_function(function_id, |fb| {
                let mut new_blocks = HashMap::<BlockId, HLBlock>::new();
                for (bid, mut block) in fb.function.take_blocks().into_iter() {
                    let mut new_instructions = Vec::new();
                    for instruction in block.take_instructions().into_iter() {
                        let b =
                            &mut HLInstrBuilder::new(fb.function, fb.ssa, &mut new_instructions);
                        self.process_instruction(b, function_type_info, instruction);
                    }
                    block.put_instructions(new_instructions);
                    new_blocks.insert(bid, block);
                }
                fb.function.put_blocks(new_blocks);
            });
        }
    }

    fn process_instruction(
        &self,
        b: &mut HLInstrBuilder<'_>,
        function_type_info: &FunctionTypeInfo,
        instruction: OpCode,
    ) {
        match instruction {
            OpCode::BinaryArithOp {
                kind: BinaryArithOpKind::Mul,
                result,
                lhs,
                rhs,
            } => {
                if !self.lower_mul(b, function_type_info, result, lhs, rhs) {
                    b.push(OpCode::BinaryArithOp {
                        kind: BinaryArithOpKind::Mul,
                        result,
                        lhs,
                        rhs,
                    });
                }
            }
            OpCode::Guard { .. } => {
                panic!("Guard should have been lowered before degree_spilling");
            }
            instruction => b.push(instruction),
        }
    }

    fn lower_mul(
        &self,
        b: &mut HLInstrBuilder<'_>,
        function_type_info: &FunctionTypeInfo,
        result: ValueId,
        lhs: ValueId,
        rhs: ValueId,
    ) -> bool {
        let lhs_type = function_type_info.get_value_type(lhs);
        let lhs_witness = lhs_type.is_witness_of();
        let rhs_witness = function_type_info.get_value_type(rhs).is_witness_of();

        if !lhs_witness && !rhs_witness {
            return false;
        }

        match lhs_type.strip_witness().expr {
            TypeExpr::U(_) | TypeExpr::I(_) => {
                panic!(
                    "witness integer multiplication should have been lowered by instruction_lowering"
                )
            }
            _ if lhs_witness && rhs_witness => {
                let lhs_plain = b.value_of(lhs);
                let rhs_plain = b.value_of(rhs);
                let mul_hint = b.mul(lhs_plain, rhs_plain);
                b.push(OpCode::WriteWitness {
                    result: Some(result),
                    value: mul_hint,
                    pinned: false,
                });
                b.constrain(lhs, rhs, result);
                true
            }
            _ => false,
        }
    }
}
