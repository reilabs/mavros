//! Partial redundancy elimination (PRE) driven by the Click–Cooper analysis.
//!
//! This pass subsumes the work of CSE and extends it with loop-invariant code motion and general
//! partial-redundancy code motion.
//!
//! # Design
//!
//! The algorithm is shaped like GVN-PRE, planned to run atop the converged Click-Cooper congruence
//! partition rather than using a locally-built value table:
//!
//! - **Value classes** are congruence classes keyed by their minimum member id
//!   ([`ClickCooper::class_key`]), extended by a canonicalization layer reproducing the CSE
//!   interner's equivalences that binary value numbering cannot see (n-ary flatten + sort of
//!   commutative chains, `MulConst` ≡ `Mul`).
//! - **Redundancy detection uses unconditional congruence only.** Assert-derived equalities reach
//!   PRE indirectly: SCS runs first at every site and copy-propagates asserted-equal leaders into
//!   operands, after which the recomputed analysis sees the values as structurally congruent. PRE
//!   therefore carries no conditional-fact obligations (no Gate 3); its one conditional read is
//!   the disequality channel feeding the speculation gate ([`totality`]).
//! - **Availability** (`AVAIL_OUT`) generalizes [`ClickCooper::leader`] from a value's own
//!   definition to arbitrary program points, built top-down over the domination preorder.
//! - **Down-safety** is a backward anticipability fixpoint, intersecting at `JmpIf` and taken as
//!   a greatest fixpoint so anticipation is seen through loop cycles. Phi-translation is avoided
//!   rather than performed: only binding-stable expressions (operands defined outside every CFG
//!   cycle) participate in motion, so a key names the same values at every program point. Placement
//!   at a down-safe point needs no totality gate: the op was bound to run, so a trap merely fires
//!   earlier on a run already doomed to reject (the accept/reject model).
//! - **Insertion** clones a template occurrence into the predecessors where a class is unavailable,
//!   splitting `JmpIf` edges lazily ([`edge_split`]) and joining the copies through a fresh block
//!   parameter.
//! - **Speculation** involves placement at a point that is _not_ down-safe, and is gated per
//!   insertion point by the [`totality::TotalityOracle`]. Totality is the soundness license only;
//!   profitability is a separate, structural gate: a join must eliminate strictly more occurrences
//!   than the copies it materializes, and a _speculative_ insertion is additionally admitted only
//!   at a strictly smaller loop depth than the occurrences it eliminates — work moves out of loops,
//!   never into branch arms. Every join therefore shrinks the static SSA, and the zero-trip path of
//!   a hoisted loop pays exactly one total pure op per hoisted key. Per-path evaluation counts can
//!   still grow in one narrow shape (a path pays a materialized copy and later re-evaluates at an
//!   occurrence the merge does not dominate — see the "Residue" section of `insert.rs`'s module
//!   doc).
//!
//! The kinds of code motion that the pass can perform is gated by the provided [`MotionLevel`] in
//! the pass' configuration. These levels encompass a basic elimination sweep, loop invariant
//! floating in down-safe contexts, general join insertion, and totality-gated speculative hoisting.

pub mod edge_split;
mod eliminate;
mod insert;
pub mod totality;

#[cfg(test)]
mod test;

use crate::compiler::{
    analysis::{click_cooper::ClickCooper, flow_analysis::FlowAnalysis, types::TypeInfo},
    pass_manager::{Analysis, AnalysisId, AnalysisStore, Pass},
    passes::dead_code_elimination::{Config as DceConfig, DCE},
    ssa::hlssa::HLSSA,
};

// PARTIAL REDUNDANCY ELIMINATION PASS
// ================================================================================================

/// Partial redundancy elimination (PRE) over the Click–Cooper analysis, with integrated dead-code
/// elimination.
pub struct PRE {
    config: Config,
}

impl Pass for PRE {
    fn name(&self) -> &'static str {
        "pre"
    }

    fn needs(&self) -> Vec<AnalysisId> {
        vec![ClickCooper::id(), TypeInfo::id(), FlowAnalysis::id()]
    }

    fn run(&self, ssa: &mut HLSSA, store: &AnalysisStore) {
        self.transform(
            ssa,
            store.get::<ClickCooper>(),
            store.get::<TypeInfo>(),
            store.get::<FlowAnalysis>(),
        );

        // The rewrite invalidates the store's `FlowAnalysis`; recompute it for the integrated DCE.
        let flow = FlowAnalysis::run(ssa);
        DCE::new(self.config.dce).do_run(ssa, &flow);
    }
}

impl PRE {
    pub fn with_config(config: Config) -> Self {
        Self { config }
    }

    /// The pass as wired at the pre-R1C pipeline sites.
    pub fn pre_r1c() -> Self {
        Self::with_config(Config::pre_r1c())
    }

    fn transform(&self, ssa: &mut HLSSA, cc: &ClickCooper, types: &TypeInfo, flow: &FlowAnalysis) {
        // Each function is taken out of the `ssa` for the duration of its rewrite so the motion
        // stage can mint fresh value ids through the shared `&SSA`.
        let fids: Vec<_> = ssa.get_function_ids().collect();
        for fid in fids {
            let mut function = ssa.take_function(fid);
            let function_types = types.get_function(fid);
            let cfg = flow.get_function_cfg(fid);
            let node_of = eliminate::eliminate_function(
                &mut function,
                cc,
                fid,
                function_types,
                cfg,
                self.config.deduplicate_lookups,
            );

            if self.config.motion >= MotionLevel::LoopHoist {
                let oracle = totality::TotalityOracle::new(cc, ssa, fid, function_types);
                insert::perform_code_motion(
                    ssa,
                    &mut function,
                    function_types,
                    cfg,
                    &node_of,
                    self.config.motion,
                    &oracle,
                );
            }
            ssa.put_function(fid, function);
        }
    }
}

// MOTION LEVEL
// ================================================================================================

/// How much code motion the pass may perform.
///
/// The staging/bisection lever: each level includes everything below it, and forcing a lower level
/// must reproduce that level's output exactly.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum MotionLevel {
    /// Full-redundancy elimination only.
    ///
    /// Equivalent to a basic CSE, with no code motion.
    EliminateOnly,

    /// Plus down-safe insertion at loop headers (LICM).
    LoopHoist,

    /// Plus general join-point partial-redundancy insertion.
    JoinInsert,

    /// Plus totality-gated speculative insertion ([`totality`]).
    Speculate,
}

// PASS CONFIGURATION
// ================================================================================================

#[derive(Clone, Copy)]
pub struct Config {
    /// The enabled motion stage.
    pub motion: MotionLevel,

    /// Whether `Lookup` assertions are deduplicated (pre-r1c only, mirroring the CSE config this
    /// pass subsumes).
    pub deduplicate_lookups: bool,

    /// The integrated DCE's configuration.
    pub dce: DceConfig,
}

impl Config {
    /// The configuration for the pre-R1C pipeline sites (the old `CSE::pre_r1c` positions).
    pub fn pre_r1c() -> Self {
        Self {
            motion: MotionLevel::Speculate,
            deduplicate_lookups: true,
            dce: DceConfig::pre_r1c(),
        }
    }
}
