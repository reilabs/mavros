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
//! - **Availability** generalizes [`ClickCooper::leader`] from a value's own definition to
//!   arbitrary program points. It is not a precomputed `AVAIL_OUT` set: each query is a dominance
//!   scan over the key's occurrences (collected in domination preorder) plus the values this run's
//!   motion has already minted.
//! - **Down-safety** is a backward anticipability fixpoint, intersecting at `JmpIf` and taken as
//!   a greatest fixpoint so anticipation is seen through loop cycles. Phi-translation is avoided
//!   rather than performed: only binding-stable expressions (operands single-valued per invocation,
//!   per the Click–Cooper invariance closure — see `insert.rs`'s module doc) participate in motion,
//!   so a key names the same values at every program point. Placement at a down-safe point needs no
//!   totality gate: the op was bound to run, so a trap merely fires earlier on a run already doomed
//!   to reject (the accept/reject model).
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
//!
//! # Deferred Improvements
//!
//! Improvements deliberately left for later. Each is either gated on machinery that does not exist
//! yet or expected/measured to yield little on the current circuit corpus.
//!
//! - **One-Level Phi-Translation at Joins (classic GVN-PRE):** An expression over an acyclic
//!   merge's own parameters is already collected (the parameter is stable) and anticipated; only
//!   *placement* refuses it, because no occurrence or template operand can dominate the
//!   predecessors. The classic fix translates the key through each predecessor's jump-argument
//!   vector: substitute the parameter's leaf with the leaf of that edge's argument (canonicalized
//!   through the elimination sweep's interner, which is get-or-create), look up per-predecessor
//!   leaders on the *translated* key, and materialize translated templates into the edges that do
//!   not, whose substituted operands are in scope at the predecessor's end by construction (its
//!   terminator reads them).
//!
//!   Sound because the key and each translation are individually single-valued per invocation and
//!   φ-edge semantics equate them on that edge, so anticipation at the merge licenses the per-edge
//!   insert. Translation through back edges must be refused (expressions grow without bound around
//!   a loop — the classic termination hazard); the full translated-ANTIC / virtual-value-table form
//!   of GVN-PRE stays out of scope (it duplicates the analysis's value table and needs termination
//!   caps). Main plumbing: the motion stage must receive the interner itself — only the `node_of`
//!   map crosses today.
//! - **Cluster Hoisting:** A chain fully interior to a loop is stable link-by-link but each link is
//!   placeable only after its operands are (see `insert.rs`'s module doc). Hoisting whole chains in
//!   one run needs dependence-ordered materialization with template operands rewritten to freshly
//!   minted carriers; it lifts the first-round-only hoist restriction (a carrier can unlock a
//!   previously template-less key) and must query the [`totality::TotalityOracle`] and the type
//!   info with *pristine* templates only (`get_value_type` panics on ids minted after typing). Low
//!   expected yield: the nine pipeline sites already convert pure chains link-by-link across
//!   sites — each site's hoist redirects the next link's operand to an acyclically defined value —
//!   leaving only chains rooted in call results (calls never move) and chains deeper than the
//!   remaining sites.
//! - **Break-Even Loop Joins:** The static-cost gate refuses instruction-neutral joins even when
//!   every drained occurrence sits at strictly greater loop depth than the planted copies — a
//!   per-iteration win bought with a per-entry copy, including the `(1,1)` multi-entry-header
//!   self-carry. Reclaiming them means loop-depth profitability in the join rule; the loop forest
//!   the speculation stage introduced already supplies the depths.
//! - **Speculative Join Insertion:** Speculation lives only in the single-entry-header hoist rule;
//!   the join rule is down-safe at every level, so a body-only key at a *multi-entry* loop header
//!   never moves. Closing it means porting the totality license and the strictly-smaller-loop-depth
//!   gate to the join rule's per-edge placements.
//! - **Aggregate Deduplication:** The elimination sweep currently excludes the aggregate operations
//!   (`MkSeq`, `MkRepeated`, `MkSeqOfBlob`, `ArraySet`, `SlicePush`, `SliceLen`). A dedup toggle is
//!   plausible but must be corpus-gated: aggregate rewrites have regressed witgen bytecode at scale
//!   before.
//! - **Witness-Typed `Div`/`Mod` and `Shl`/`Shr` Speculation:** [`totality`] hard-refuses any
//!   witness-typed operand regardless of facts. Lifting it needs a witness-aware totality argument:
//!   the witness gadget lowering introduces its own rejecting constraints, so the license must
//!   reason about the lowered form, not the scalar op. This is complex.
//! - **Seeing Through Noir's `!=` Lowering:** `if d != 0` lowers to a boolean-not through field
//!   arithmetic (`jmp_if(cast(1 - cast(d == 0)))`) that the analysis's disequality extraction does
//!   not see through, so idiomatic `!=` guards never produce the `known_unequal` fact the divisor
//!   gate consumes. An analysis upgrade this pass inherits for free — bounded, though: witness
//!   divisors are refused outright (above), and most real guards are witness-conditioned and
//!   untainted away.

pub mod edge_split;
mod eliminate;
mod insert;
pub mod totality;

#[cfg(test)]
mod test;

use crate::compiler::{
    analysis::{
        click_cooper::ClickCooper, flow_analysis::FlowAnalysis, types::TypeInfo,
        witness_taint_inference::ApproximateWitnessTaint,
    },
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

    /// The pass as wired at the pre-untaint pipeline site.
    pub fn pre_untaint() -> Self {
        Self::with_config(Config::pre_untaint())
    }

    fn transform(&self, ssa: &mut HLSSA, cc: &ClickCooper, types: &TypeInfo, flow: &FlowAnalysis) {
        // Pre-untaint speculation needs witness-ness the types cannot yet answer (`WitnessOf` is
        // baked in by untaint): approximate it once over the pristine program via the read-only
        // joined WTI solve. Computed before any rewrite so the recorded shapes key the same value
        // ids the oracle is queried with (elimination only redirects operands to taint-congruent
        // leaders, and the oracle only sees pristine template operands).
        let taint: Option<ApproximateWitnessTaint> = (self.config.preserve_structure
            && self.config.motion >= MotionLevel::Speculate)
            .then(|| ApproximateWitnessTaint::compute(ssa, flow, types));

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
                // With `preserve_structure` + `Speculate`, `taint` is `Some` by construction, so
                // the unsound combination (a Types-sourced oracle speculating pre-untaint) is
                // unreachable.
                let oracle = match &taint {
                    Some(t) => totality::TotalityOracle::with_witness_source(
                        cc,
                        ssa,
                        fid,
                        function_types,
                        totality::WitnessnessSource::Taint(t),
                    ),
                    None => totality::TotalityOracle::new(cc, ssa, fid, function_types),
                };
                insert::perform_code_motion(
                    ssa,
                    cc,
                    fid,
                    &mut function,
                    function_types,
                    cfg,
                    &node_of,
                    self.config.motion,
                    self.config.preserve_structure,
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

    /// Whether motion must leave the block/parameter geometry untouched — the pre-untaint
    /// contract (`untaint_control_flow` assumes a single jump into each merge from a branch
    /// side).
    ///
    /// When set, the hoist rule refuses headers whose entry predecessor ends in a `JmpIf` (an edge
    /// split would mint a block) and the join rule is off entirely (it appends merge parameters);
    /// hoisting into a `Jmp`-terminated predecessor only inserts instructions above its terminator.
    ///
    /// This flag also selects the speculation gate's witness-ness source: at
    /// [`MotionLevel::Speculate`] it switches the [`totality::TotalityOracle`] from the types to
    /// the joined WTI approximation. A pre-untaint caller must therefore never combine `Speculate`
    /// with `preserve_structure: false` — the Types-sourced oracle reads `is_witness_of` on types
    /// that carry no `WitnessOf` yet, silently answering "pure" for every value and licensing
    /// unsound speculation (besides performing structural edits untaint cannot absorb). Use the
    /// [`Config::pre_untaint`]/[`Config::pre_r1c`] constructors rather than raw literals.
    pub preserve_structure: bool,

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
            preserve_structure: false,
            deduplicate_lookups: true,
            dce: DceConfig::pre_r1c(),
        }
    }

    /// The configuration for the pre-untaint pipeline site.
    ///
    /// Full totality-gated speculation under [`Config::preserve_structure`]: the geometry
    /// restrictions that flag documents are exactly untaint's, and within them a hoist is pure
    /// instruction insertion into a `Jmp`-terminated predecessor. Because the `WitnessOf` types
    /// the [`totality::TotalityOracle`] normally reads witness-ness from do not exist yet, the
    /// oracle at this site is fed the read-only joined WTI approximation instead.
    ///
    /// `Lookup` assertions are minted only post-untaint (`LookupSpilling`, the specializer, and
    /// witness lowering), so lookup deduplication is moot here; `false` documents that. The
    /// integrated DCE preserves blocks for the same reason as every other pre-untaint DCE/SCS
    /// run.
    pub fn pre_untaint() -> Self {
        Self {
            motion: MotionLevel::Speculate,
            preserve_structure: true,
            deduplicate_lookups: false,
            dce: DceConfig::preserve_blocks(),
        }
    }
}
