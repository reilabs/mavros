//! Folds conditional jumps whose condition is statically known.
//!
//! Monomorphized generic dispatch (e.g. the poseidon2 permutation width dispatch in
//! `std::mavros::replacements`) compares compile-time constants. Constraint generation is
//! unaffected by such branches (symbolic execution only walks taken paths), but without folding
//! them the never-taken branches keep their callees alive through
//! `RemoveUnreachableFunctions`, bloating witgen and AD code.

use std::collections::HashMap;

use crate::compiler::{
    pass_manager::{AnalysisStore, Pass},
    ssa::{
        SSAConstantsSnapshot, Terminator, ValueId,
        hlssa::{CmpKind, Constant, HLSSA, OpCode},
    },
};

pub struct FoldConstantBranches {}

impl FoldConstantBranches {
    pub fn new() -> Self {
        Self {}
    }
}

impl Pass for FoldConstantBranches {
    fn name(&self) -> &'static str {
        "fold_constant_branches"
    }

    fn run(&self, ssa: &mut HLSSA, _store: &AnalysisStore) {
        let constants = ssa.const_snapshot();
        for (_, function) in ssa.iter_functions_mut() {
            // Conditions are Cmp results; collect the function's comparisons up front.
            let mut comparisons: HashMap<ValueId, (CmpKind, ValueId, ValueId)> = HashMap::new();
            for (_, block) in function.get_blocks() {
                for instruction in block.get_instructions() {
                    if let OpCode::Cmp {
                        kind,
                        result,
                        lhs,
                        rhs,
                    } = instruction
                    {
                        comparisons.insert(*result, (*kind, *lhs, *rhs));
                    }
                }
            }

            for (_, block) in function.get_blocks_mut() {
                let Some(Terminator::JmpIf(condition, then_target, else_target)) =
                    block.get_terminator()
                else {
                    continue;
                };
                let (then_target, else_target) = (*then_target, *else_target);
                let Some(taken) = evaluate_condition(*condition, &constants, &comparisons) else {
                    continue;
                };
                let target = if taken { then_target } else { else_target };
                block.set_terminator(Terminator::Jmp(target, Vec::new()));
            }
        }
    }
}

/// Statically evaluate a branch condition: either a boolean constant or a comparison whose
/// outcome is known (both operands constant, or `x == x`).
fn evaluate_condition(
    condition: ValueId,
    constants: &SSAConstantsSnapshot<Constant>,
    comparisons: &HashMap<ValueId, (CmpKind, ValueId, ValueId)>,
) -> Option<bool> {
    if let Some(constant) = constants.get(&condition) {
        return match **constant {
            Constant::U(_, value) => Some(value != 0),
            _ => None,
        };
    }
    let (kind, lhs, rhs) = comparisons.get(&condition)?;
    if matches!(kind, CmpKind::Eq) && lhs == rhs {
        return Some(true);
    }
    let lhs = constants.get(lhs)?;
    let rhs = constants.get(rhs)?;
    match (kind, &**lhs, &**rhs) {
        (CmpKind::Eq, Constant::U(b1, x), Constant::U(b2, y)) if b1 == b2 => Some(x == y),
        (CmpKind::Lt, Constant::U(b1, x), Constant::U(b2, y)) if b1 == b2 => Some(x < y),
        (CmpKind::Eq, Constant::Field(x), Constant::Field(y)) => Some(x == y),
        _ => None,
    }
}
