//! This pass is intended to be the first step of the R1CS generation phase, and converts every
//! `WriteWitness` opcode into a `FreshWitness` opcode as the actual computed value cannot be known
//! in this portion of the pipeline.

use crate::compiler::util::ice_non_elided_tuple;
use crate::compiler::{
    analysis::{flow_analysis::FlowAnalysis, types::TypeInfo},
    pass_manager::{Analysis, AnalysisId, AnalysisStore, Pass},
    ssa::hlssa::{HLSSA, OpCode},
};

pub struct WitnessWriteToFresh {}

impl Pass for WitnessWriteToFresh {
    fn name(&self) -> &'static str {
        "witness_write_to_fresh"
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

impl WitnessWriteToFresh {
    pub fn new() -> Self {
        Self {}
    }

    pub fn do_run(&self, ssa: &mut HLSSA, type_info: &TypeInfo) {
        for (function_id, function) in ssa.iter_functions_mut() {
            for (_, block) in function.get_blocks_mut() {
                for instruction in block.get_instructions_mut() {
                    let new_instruction = match instruction {
                        OpCode::WriteWitness {
                            result: r,
                            value: v,
                            pinned: _,
                        } => {
                            let tp = type_info.get_function(*function_id).get_value_type(*v);
                            if tp.is_witness_of() {
                                panic!("ICE: WriteWitness input has WitnessOf type: {:?}", tp);
                            }
                            if !tp.is_numeric() {
                                panic!("Expected numeric type, got {:?}", tp);
                            }
                            OpCode::FreshWitness {
                                result: r.unwrap(),
                                result_type: tp.clone(),
                            }
                        }
                        OpCode::Cmp { .. }
                        | OpCode::Cast { .. }
                        | OpCode::MkSeq { .. }
                        | OpCode::MkRepeated { .. }
                        | OpCode::Alloc { .. }
                        | OpCode::BinaryArithOp { .. }
                        | OpCode::SExt { .. }
                        | OpCode::BitRange { .. }
                        | OpCode::Not { .. }
                        | OpCode::Store { .. }
                        | OpCode::Load { .. }
                        | OpCode::Assert { .. }
                        | OpCode::AssertCmp { .. }
                        | OpCode::AssertR1C { .. }
                        | OpCode::Call { .. }
                        | OpCode::ArrayGet { .. }
                        | OpCode::ArraySet { .. }
                        | OpCode::SlicePush { .. }
                        | OpCode::SliceLen { .. }
                        | OpCode::Select { .. }
                        | OpCode::ToBits { .. }
                        | OpCode::ToRadix { .. }
                        | OpCode::MemOp { .. }
                        | OpCode::FreshWitness { .. }
                        | OpCode::Constrain { .. }
                        | OpCode::NextDCoeff { .. }
                        | OpCode::MulConst { .. }
                        | OpCode::BumpD { .. }
                        | OpCode::Rangecheck { .. }
                        | OpCode::Lookup { .. }
                        | OpCode::DLookup { .. }
                        | OpCode::ReadGlobal { .. }
                        | OpCode::InitGlobal { .. }
                        | OpCode::DropGlobal { .. }
                        | OpCode::Todo { .. }
                        | OpCode::ValueOf { .. }
                        | OpCode::Spread { .. }
                        | OpCode::Unspread { .. }
                        | OpCode::Guard { .. } => instruction.clone(),
                        OpCode::TupleProj { .. } | OpCode::MkTuple { .. } => ice_non_elided_tuple(),
                    };
                    *instruction = new_instruction;
                }
            }
        }
    }
}
