//! Splitting a `JmpIf` edge: the primitive PRE uses to place code "on" a conditional edge.
//!
//! [`Terminator::JmpIf`] carries no block arguments, so a computation destined for one arm of a
//! conditional branch needs a fresh block spliced into that edge: `pred: JmpIf(c, T, F)` becomes
//! `JmpIf(c, S, F)` with `S: Jmp(T, [])`. The split block is dominated by `pred` and dominates
//! nothing the original target did not already dominate, so facts holding on entry to the chosen
//! arm hold in it. When the split edge is the target's sole entry-side edge — splitting a loop
//! header's entry, the hoist rule's case — the split block *does* dominate the target and
//! everything below it: every first entry crosses the split, and a definition placed there is
//! available throughout the loop. The original target keeps all its other predecessors, so a split
//! never orphans a block.
//!
//! Splits are performed lazily — only when an insertion actually lands on the edge — so no empty
//! blocks are created speculatively (`FixDoubleJumps` would clean them up, but not creating them
//! is better).

use crate::compiler::ssa::{BlockId, Function, Instruction, SSAType, Terminator};

// JUMP ARMS
// ================================================================================================

/// Which arm of a [`Terminator::JmpIf`] to split. Both arms of one branch can target the same
/// block; the discriminator makes each edge individually addressable.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum JmpIfArm {
    True,
    False,
}

/// Split the `arm` edge of `pred`'s `JmpIf`: allocate a fresh block, retarget the arm at it, and
/// terminate it with an argument-less `Jmp` to the original target. Returns the new block's id.
///
/// The new jump starts with no arguments because a `JmpIf` edge cannot bind block parameters, so
/// its target has none (asserted). A caller that later adds parameters to the target must extend
/// this jump's arguments along with every other predecessor's.
///
/// Panics if `pred` does not end in a `JmpIf`.
pub fn split_jmp_if_edge<Op: Instruction, Ty: SSAType>(
    function: &mut Function<Op, Ty>,
    pred: BlockId,
    arm: JmpIfArm,
) -> BlockId {
    let split = function.add_block();
    let target = match (function.get_block_mut(pred).get_terminator_mut(), arm) {
        (Terminator::JmpIf(_, t, _), JmpIfArm::True) => std::mem::replace(t, split),
        (Terminator::JmpIf(_, _, f), JmpIfArm::False) => std::mem::replace(f, split),
        (t, _) => panic!("split_jmp_if_edge: predecessor must end in JmpIf, found {t:?}"),
    };
    assert!(
        !function.get_block(target).has_parameters(),
        "split_jmp_if_edge: JmpIf edge into a parameterized block (unbindable parameters)"
    );
    function
        .get_block_mut(split)
        .set_terminator(Terminator::Jmp(target, Vec::new()));
    split
}
