use crate::compiler::{
    flow_analysis::FlowAnalysis,
    ir::r#type::Type,
    pass_manager::{Analysis, AnalysisId, AnalysisStore, Pass},
    passes::fix_double_jumps::ValueReplacements,
    ssa::{CastTarget, HLSSA, OpCode},
};

/// Strips all `WitnessOf` type wrappers from the SSA.
///
/// In the witgen pipeline, all computation is concrete — there's no need
/// for the WitnessOf distinction. This pass converts all `WitnessOf(X)` types
/// back to `X` and removes `Cast { target: WitnessOf }` instructions.
pub struct StripWitnessOf {}

impl Pass for StripWitnessOf {
    fn name(&self) -> &'static str {
        "strip_witness_of"
    }
    fn run(&self, ssa: &mut HLSSA, _store: &AnalysisStore) {
        self.do_run(ssa);
    }
    fn preserves(&self) -> Vec<AnalysisId> {
        vec![FlowAnalysis::id()]
    }
}

impl StripWitnessOf {
    pub fn new() -> Self {
        Self {}
    }

    pub fn do_run(&self, ssa: &mut HLSSA) {
        // Strip from global types
        let new_global_types: Vec<Type> = ssa
            .get_global_types()
            .iter()
            .map(|t| t.strip_all_witness())
            .collect();
        ssa.set_global_types(new_global_types);

        for (_, function) in ssa.iter_functions_mut() {
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

                // Remove Cast { target: WitnessOf } by replacing result → value
                let old_instructions = block.take_instructions();
                let new_instructions: Vec<_> = old_instructions
                    .into_iter()
                    .filter_map(|mut instr| {
                        if let OpCode::Cast {
                            result,
                            value,
                            target: CastTarget::WitnessOf,
                        } = &instr
                        {
                            replacements.insert(*result, *value);
                            return None;
                        }
                        Self::strip_instruction(&mut instr);
                        Some(instr)
                    })
                    .collect();
                block.put_instructions(new_instructions);
            }

            // Apply replacements for removed WitnessOf casts
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
            OpCode::Alloc { elem_type, .. } => {
                *elem_type = elem_type.strip_all_witness();
            }
            OpCode::Cast { .. } => {}
            OpCode::MkTuple { element_types, .. } => {
                for tp in element_types.iter_mut() {
                    *tp = tp.strip_all_witness();
                }
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
            | OpCode::Truncate { .. }
            | OpCode::SExt { .. }
            | OpCode::Not { .. }
            | OpCode::Store { .. }
            | OpCode::Load { .. }
            | OpCode::AssertEq { .. }
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
            | OpCode::TupleProj { .. }
            | OpCode::InitGlobal { .. }
            | OpCode::DropGlobal { .. }
            | OpCode::Const { .. } => {}
            OpCode::Guard { inner, .. } => {
                Self::strip_instruction(inner);
            }
        }
    }
}
