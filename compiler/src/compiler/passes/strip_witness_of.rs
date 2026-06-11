//! Strips all `WitnessOf` type wrappers from the SSA.
//!
//! In the witgen pipeline, all computation is concrete — there's no need for the WitnessOf
//! distinction. This pass converts all `WitnessOf(X)` types back to `X` and removes casts
//! that become identity conversions once witness wrappers are gone (in particular all
//! witness-injection casts, scalar and composite alike).

use crate::compiler::util::ice_non_elided_tuple;
use crate::compiler::{
    analysis::{flow_analysis::FlowAnalysis, types::TypeInfo},
    pass_manager::{Analysis, AnalysisId, AnalysisStore, Pass},
    passes::fix_double_jumps::ValueReplacements,
    ssa::hlssa::{HLSSA, OpCode, Type},
};

pub struct StripWitnessOf {}

impl Pass for StripWitnessOf {
    fn name(&self) -> &'static str {
        "strip_witness_of"
    }
    fn needs(&self) -> Vec<AnalysisId> {
        vec![TypeInfo::id(), FlowAnalysis::id()]
    }
    fn run(&self, ssa: &mut HLSSA, store: &AnalysisStore) {
        self.do_run(ssa, store.get::<TypeInfo>());
    }
    fn preserves(&self) -> Vec<AnalysisId> {
        vec![FlowAnalysis::id()]
    }
}

impl StripWitnessOf {
    pub fn new() -> Self {
        Self {}
    }

    pub fn do_run(&self, ssa: &mut HLSSA, type_info: &TypeInfo) {
        // Strip from global types
        let new_global_types: Vec<Type> = ssa
            .get_global_types()
            .iter()
            .map(|t| t.strip_all_witness())
            .collect();
        ssa.set_global_types(new_global_types);

        let function_ids: Vec<_> = ssa.get_function_ids().collect();
        for function_id in function_ids {
            let fn_type_info = if type_info.has_function(function_id) {
                Some(type_info.get_function(function_id))
            } else {
                None
            };
            let function = ssa.get_function_mut(function_id);

            // Strip from return types
            for rtp in function.iter_returns_mut() {
                *rtp = rtp.strip_all_witness();
            }

            // Strip from block parameters and instructions
            let mut replacements = ValueReplacements::new();
            for (_, block) in function.get_blocks_mut() {
                for (_, tp) in block.get_parameters_mut() {
                    *tp = tp.strip_all_witness();
                }

                // Remove casts that become identity once WitnessOf is stripped
                // (witness injections/strips); keep real conversions with a
                // stripped target type.
                let old_instructions = block.take_instructions();
                let new_instructions: Vec<_> = old_instructions
                    .into_iter()
                    .filter_map(|mut instr| {
                        if let OpCode::Cast {
                            result,
                            value,
                            target,
                        } = &instr
                        {
                            let stripped_target = target.strip_all_witness();
                            let is_identity = fn_type_info
                                .map(|ti| {
                                    ti.get_value_type(*value).strip_all_witness() == stripped_target
                                })
                                .unwrap_or(false);
                            if is_identity {
                                replacements.insert(*result, *value);
                                return None;
                            }
                            instr = OpCode::Cast {
                                result: *result,
                                value: *value,
                                target: stripped_target,
                            };
                            return Some(instr);
                        }
                        Self::strip_instruction(&mut instr);
                        Some(instr)
                    })
                    .collect();
                block.put_instructions(new_instructions);
            }

            // Apply replacements for removed identity casts
            for (_, block) in function.get_blocks_mut() {
                for instruction in block.get_instructions_mut() {
                    replacements.replace_instruction(instruction);
                }
                replacements.replace_terminator(block.get_terminator_mut());
            }
        }
    }

    fn strip_instruction(instruction: &mut OpCode) {
        match instruction {
            OpCode::FreshWitness { result_type, .. } => {
                *result_type = result_type.strip_all_witness();
            }
            OpCode::MkSeq { elem_type, .. } => {
                *elem_type = elem_type.strip_all_witness();
            }
            OpCode::MkSeqOfBlob { element_type, .. } => {
                *element_type = element_type.strip_all_witness();
            }
            OpCode::MkRepeated { elem_type, .. } => {
                *elem_type = elem_type.strip_all_witness();
            }
            OpCode::Alloc { elem_type, .. } => {
                *elem_type = elem_type.strip_all_witness();
            }
            OpCode::Cast { .. } => {
                unreachable!("Cast is handled before strip_instruction")
            }
            OpCode::MkTuple { .. } | OpCode::TupleProj { .. } | OpCode::TupleRefProj { .. } => {
                ice_non_elided_tuple()
            }
            OpCode::ReadGlobal { result_type, .. } => {
                *result_type = result_type.strip_all_witness();
            }
            OpCode::Todo { result_types, .. } => {
                for tp in result_types.iter_mut() {
                    *tp = tp.strip_all_witness();
                }
            }
            OpCode::Cmp { .. }
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
            | OpCode::ValueOf { .. }
            | OpCode::WriteWitness { .. }
            | OpCode::NextDCoeff { .. }
            | OpCode::BumpD { .. }
            | OpCode::Constrain { .. }
            | OpCode::Lookup { .. }
            | OpCode::DLookup { .. }
            | OpCode::MulConst { .. }
            | OpCode::Rangecheck { .. }
            | OpCode::InitGlobal { .. }
            | OpCode::DropGlobal { .. }
            | OpCode::Spread { .. }
            | OpCode::Unspread { .. } => {}
            OpCode::Guard { .. } => {
                panic!("ICE: Found Guard but `LowerGuards` should have removed them")
            }
        }
    }
}
