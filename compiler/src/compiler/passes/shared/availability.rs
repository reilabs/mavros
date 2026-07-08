//! Dominance-scoped availability of program points.

use crate::compiler::{analysis::flow_analysis::CFG, ssa::ProgramPoint};

/// Whether a definition/assertion at `p1` is available at `p2`: earlier in the same block, or in
/// a block dominating `p2`'s.
pub(crate) fn can_replace(cfg: &CFG, p1: ProgramPoint, p2: ProgramPoint) -> bool {
    (p1.block == p2.block && p1.index < p2.index) || cfg.dominates(p1.block, p2.block)
}
