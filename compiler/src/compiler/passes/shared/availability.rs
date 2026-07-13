//! Dominance-scoped availability of program points.

use crate::compiler::{analysis::flow_analysis::CFG, ssa::ProgramPoint};

/// Whether a definition/assertion at `p1` is available at `p2`: strictly earlier in the same
/// block, or in a *different* block dominating `p2`'s.
///
/// The same-block case must be decided by index alone: [`CFG::dominates`] is reflexive, so falling
/// through to it would report a _later_ same-block point as available.
pub(crate) fn can_replace(cfg: &CFG, p1: ProgramPoint, p2: ProgramPoint) -> bool {
    if p1.block == p2.block {
        p1.index < p2.index
    } else {
        cfg.dominates(p1.block, p2.block)
    }
}
