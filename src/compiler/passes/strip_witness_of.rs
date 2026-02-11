use crate::compiler::{
    ir::r#type::Type,
    pass_manager::{Pass, PassInfo, PassManager},
    ssa::{CastTarget, Const, OpCode, SSA},
};

/// Strips all `WitnessOf` type wrappers from the SSA.
///
/// In the witgen pipeline, all computation is concrete — there's no need
/// for the WitnessOf distinction. This pass converts all `WitnessOf(X)` types
/// back to `X`, turns `CastTarget::WitnessOf` into `Nop`, and converts
/// `Const::Witness` to `Const::Field`.
pub struct StripWitnessOf {}

impl Pass for StripWitnessOf {
    fn run(&self, ssa: &mut SSA, _pass_manager: &PassManager) {
        self.do_run(ssa);
    }

    fn pass_info(&self) -> PassInfo {
        PassInfo {
            name: "strip_witness_of",
            needs: vec![],
        }
    }

    fn invalidates_cfg(&self) -> bool {
        false
    }
}

impl StripWitnessOf {
    pub fn new() -> Self {
        Self {}
    }

    pub fn do_run(&self, ssa: &mut SSA) {
        // Strip from global types
        let new_global_types: Vec<Type> = ssa
            .get_global_types()
            .iter()
            .map(|t| t.strip_all_witness())
            .collect();
        ssa.set_global_types(new_global_types);

        for (_, function) in ssa.iter_functions_mut() {
            // Strip from constants: Const::Witness → Const::Field
            let witness_consts: Vec<_> = function
                .iter_consts()
                .filter_map(|(vid, c)| {
                    if let Const::Witness(v) = c {
                        Some((*vid, *v))
                    } else {
                        None
                    }
                })
                .collect();
            for (vid, v) in witness_consts {
                function.replace_const(vid, Const::Field(v));
            }

            // Strip from return types
            for rtp in function.iter_returns_mut() {
                *rtp = rtp.strip_all_witness();
            }

            // Strip from block parameters and instructions
            for (_, block) in function.get_blocks_mut() {
                for (_, tp) in block.get_parameters_mut() {
                    *tp = tp.strip_all_witness();
                }

                for instruction in block.get_instructions_mut() {
                    Self::strip_instruction(instruction);
                }
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
            OpCode::Cast { target, .. } => {
                if matches!(target, CastTarget::WitnessOf) {
                    *target = CastTarget::Nop;
                }
            }
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
            _ => {}
        }
    }
}
