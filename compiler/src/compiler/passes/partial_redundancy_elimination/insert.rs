//! Code motion: the [`MotionLevel::LoopHoist`], [`MotionLevel::JoinInsert`], and
//! [`MotionLevel::Speculate`] stages.
//!
//! The first two stages place code only where it is **anticipated** — every path onward evaluates
//! the key — so the insertion never computes anything the original run was not already bound to
//! compute (a trapping op merely traps earlier on a run already doomed to reject; no totality gate
//! is needed). Speculation lifts this restriction for loop hoisting, under a per-opcode totality
//! license.
//!
//! # Anticipability
//!
//! A backward all-paths dataflow over the canonical keys of [`super::eliminate`]:
//! `ANTIC_IN[b] = GEN[b] ∪ ⋂_{s ∈ succ(b)} ANTIC_IN[s]`, greatest fixpoint (start at the full
//! candidate set and shrink). The greatest solution counts evaluations beyond a loop's back edge as
//! anticipation, which is exactly right under the accept/reject model: a run that never gets there
//! hangs, and hang is a reject.
//!
//! No phi-translation is required, by construction: `GEN` admits an occurrence only when every
//! operand's definition lies outside every CFG cycle ([`StaticCyclicBlocks`]), so each operand is
//! bound at most once per invocation and the expression names the same values at every program
//! point. (Canonical-key leaves are congruence classes; a class containing such an operand can only
//! contain a rebindable block parameter through a constant class, where every incoming argument
//! shares the class and translation is the identity). This deliberately excludes expressions over
//! values defined in an _outer_ loop — hoisting those needs the value-stability machinery — and
//! expressions over a merge's own parameters, whose translation into a predecessor is the jump's
//! argument vector (classic GVN-PRE phi-translation, deferred): both shapes fail the operand
//! checks below.
//!
//! # Loop Hoisting
//!
//! A loop header is a join of its entry-side and back-edge predecessors; hoisting an invariant
//! expression to the entry side is the join-insertion rule restricted to that shape. Headers with a
//! single entry-side predecessor `P` take a dedicated, parameter-free form: `P` dominates `H` — a
//! first entry to `H` cannot come through a back edge — so a value materialized on the `P -> H`
//! edge dominates the header and every block it dominates: in-loop and post-loop occurrences
//! redirect to it directly. When `P` ends in a `Jmp` the computation lands at the end of `P`; when
//! it ends in a `JmpIf` (possible only for parameterless headers — a parameterized header needs
//! argument-carrying jumps) the edge is split ([`super::edge_split`]). Because frontend loops are
//! while-style, a body-only expression is *not* anticipated at the header (the zero-trip exit path
//! skips it); the down-safe form fires when the value is also demanded on the exit path.
//!
//! An insertion must eliminate: a key is hoisted only when no equivalent value is already available
//! at the header (the elimination sweep owns that case) and at least one occurrence lies inside the
//! natural loop — the shape where one entry-edge evaluation replaces a per-iteration one.
//! Post-loop-only occurrences are left alone (hoisting them moves work without removing any).
//!
//! # Speculation
//!
//! At [`MotionLevel::Speculate`] the anticipation requirement on a hoist is dropped: a body-only
//! key — precisely what the while-style zero-trip exit denies down-safety — may still move to the
//! entry edge when the [`TotalityOracle`] licenses the op there.
//!
//! Totality is queried at both ends of the edge, either of which covers it: a block-entry fact
//! ranges over immutable values, so a fact at `entry_pred` still holds at its end where the
//! insertion runs, and a fact at the header holds on every edge into it — including this one, which
//! control crossing the insertion then enters unconditionally. Querying both catches a guard
//! whether `entry_pred`'s own branch establishes it or it meets over the header's predecessors. The
//! license replaces down-safety for soundness only: on a zero-trip run the hoisted op is a wasted
//! but total evaluation, and the run's outcome is unchanged.
//!
//! Profitability tightens instead: a speculative hoist must eliminate an occurrence at strictly
//! greater loop depth than the insertion point ([`loop_forest`]), the static proxy for "runs more
//! often". Down-safe hoists keep the weaker in-loop requirement (their insertion costs nothing on
//! any path), and join insertion stays down-safe at every level — speculating into a branch arm can
//! only add evaluations to the paths that skipped it.
//!
//! # Join Insertion
//!
//! At any other merge `M` (including multi-entry loop headers), a key anticipated on entry to `M`
//! and available on at least one incoming edge is joined through a fresh block parameter. Each
//! predecessor holding a dominating evaluation passes it as a jump argument, and each predecessor
//! without one receives a materialized copy on its edge — at its own end when every path through it
//! reaches `M`, otherwise in a split block. When every edge has a leader no copy is made at all:
//! the parameter alone merges the branch-local evaluations. Occurrences `M` dominates redirect to
//! the parameter; at a loop header this includes the back edge's own leader, whose argument the
//! final rewrite folds into the parameter itself — carrying the previous iteration's value is sound
//! because binding-stable keys name one value per invocation.
//!
//! Profitability is a static-cost gate: the join must drain more live dominated occurrences than
//! the copies it materializes, so every insertion shrinks the instruction count and the shrinkage
//! pays for the parameter and its per-edge arguments. The instruction-neutral diamond (one
//! computing arm, one recomputation) is deliberately refused — statically it trades one op for one
//! op plus wiring, and dynamically it only converts an evaluation on the computing path into
//! argument moves on every path. Break-even joins whose redirects sit inside a loop (a per-iter win
//! for a per-entry copy) are sacrificed with it; reclaiming those needs the loop-depth
//! profitability the speculation stage introduces.
//!
//! The sweep runs to a fixpoint: a parameter planted at one merge is new availability that can
//! unlock a merge processed earlier in the same round (dominator-preorder position does not order
//! non-comparable merges by data dependence). Each round replants nothing (the merge/key pair set
//! is remembered), so rounds are bounded by keys × merges.
//!
//! ## Residue
//!
//! Insertion is licensed per-edge by anticipation, but elimination only reaches occurrences the
//! merge dominates. A path that enters through a copy-carrying edge and next evaluates the key at
//! an occurrence the merge does *not* dominate keeps that evaluation and pays the copy on top. The
//! static-cost gate keeps every insertion tied to a strict net elimination, and the fixpoint
//! usually covers the stragglers at their own merges, but the guarantee is per-key, not per-path;
//! the corpus row/byte gates are the empirical backstop for the residue.

use crate::{
    collections::{HashMap, HashSet},
    compiler::{
        analysis::{click_cooper::StaticCyclicBlocks, flow_analysis::CFG, types::FunctionTypeInfo},
        passes::{
            partial_redundancy_elimination::{
                MotionLevel,
                edge_split::{JmpIfArm, split_jmp_if_edge},
                eliminate::NodeId,
                totality::TotalityOracle,
            },
            shared::value_replacements::{ReplaceScope, ValueReplacements},
        },
        ssa::{
            BlockId, Instruction, Located, Location, Terminator, ValueId,
            hlssa::{HLFunction, HLSSA, OpCode, Type},
        },
    },
};

// CODE MOTION
// ================================================================================================

/// Run the enabled motion stages over `function`: loop-invariant hoisting (down-safe, plus
/// totality-licensed speculation at [`MotionLevel::Speculate`]) and general join insertion at
/// [`MotionLevel::JoinInsert`] and above.
///
/// `node_of` is the canonical-key map the elimination sweep built over this (already-rewritten)
/// function; `ssa` is borrowed only to mint fresh value ids; `oracle` answers against the same
/// pristine function the other analyses were computed over.
pub(crate) fn perform_code_motion(
    ssa: &HLSSA,
    function: &mut HLFunction,
    types: &FunctionTypeInfo,
    cfg: &CFG,
    node_of: &HashMap<ValueId, NodeId>,
    motion: MotionLevel,
    oracle: &TotalityOracle,
) {
    let state = FunctionMotionState::collect(function, cfg, node_of);
    if state.occurrences.is_empty() {
        return;
    }
    let env = MotionEnv {
        ssa,
        types,
        cfg,
        oracle,
        motion,
        antic_in: state.anticipated(function, cfg),
        forest: loop_forest(cfg),
    };

    // Domination preorder, so an outer loop's header (and any dominating merge) is processed before
    // the blocks it dominates: a key moved there is then *available* below rather than moved twice.
    // Every mutation touches only the processed block's own incoming edges, so the pristine
    // analyses stay valid for the rest of the sweep.
    let blocks: Vec<BlockId> = cfg.get_domination_pre_order().collect();

    let mut ctx = MotionContext::new();
    let mut first_round = true;
    loop {
        let mut changed = false;
        for &block in &blocks {
            if cfg.is_loop_entry(block) {
                // The single entry-side predecessor (back edges come from blocks the header
                // dominates); multi-entry headers fall through to the general join rule. So does
                // the degenerate `JmpIf` with both arms on the entry edge: the CFG holds one edge
                // per arm and `get_predecessors` does not dedup, so that predecessor shows up twice
                // here and fails the single-pred match below.
                let entry_preds: Vec<BlockId> = cfg
                    .get_predecessors(block)
                    .filter(|&p| !cfg.dominates(block, p))
                    .collect();
                if let [entry_pred] = entry_preds[..] {
                    // Hoists can fire only in the first round: every hoist gate is static across
                    // rounds (ANTIC, loop structure, totality, `def_block`) or monotone against it
                    // (carriers only accumulate, so availability only grows), so a later round
                    // could only rescan and skip. Single-entry headers stay owned by the hoist
                    // rule — they never fall through to the join rule in any round.
                    if first_round {
                        changed |=
                            state.hoist_into_header(&env, function, block, entry_pred, &mut ctx);
                    }
                    continue;
                }
            }
            if motion >= MotionLevel::JoinInsert {
                changed |= state.insert_at_join(&env, function, block, &mut ctx);
            }
        }
        first_round = false;
        if motion < MotionLevel::JoinInsert || !changed {
            break;
        }
    }

    if !ctx.replacements.is_empty() {
        // Deliberately walks *all* blocks (unlike the elimination sweep's reachable-only apply):
        // the rewrite must reach the freshly minted split blocks — invisible to the pristine
        // `cfg` — e.g. to fold a self-carried leader into its parameter on a split back edge.
        ctx.replacements
            .apply_to_function(function, ReplaceScope::Inputs);
    }

    #[cfg(debug_assertions)]
    assert_jump_argument_consistency(function);
}

// MOTION STATE
// ================================================================================================

/// One motion-eligible, binding-stable occurrence of a canonical key.
struct Occurrence {
    block: BlockId,

    /// Domination-preorder position among all occurrences (the deterministic tiebreak).
    order: usize,
    result: ValueId,

    /// The (post-elimination-rewrite) instruction, ready to clone as a materialization template.
    template: OpCode,

    /// The original instruction's source location, carried onto every materialized copy so a
    /// moved evaluation still attributes its traps.
    location: Location,
}

impl Occurrence {
    /// A relocatable copy of the occurrence: the template with its result renamed to `result`,
    /// carrying the original's source location.
    fn materialize(&self, result: ValueId) -> Located<OpCode> {
        let mut instruction = self.template.clone();
        for r in instruction.get_results_mut() {
            *r = result;
        }
        Located::new(instruction, self.location.clone())
    }
}

/// The per-function facts the motion rules plan against, collected from the post-elimination
/// function before any mutation.
struct FunctionMotionState {
    /// The defining block of every parameter and instruction result. Absent values are interned
    /// constants, available everywhere.
    def_block: HashMap<ValueId, BlockId>,

    /// Binding-stable occurrences per canonical key, in domination-preorder order.
    occurrences: HashMap<NodeId, Vec<Occurrence>>,

    /// Occurrence keys per block — the `GEN` sets of the anticipability dataflow.
    gen_keys: HashMap<BlockId, HashSet<NodeId>>,

    /// Every value some instruction or terminator reads. An occurrence outside this set is a dead
    /// duplicate the elimination sweep already drained; redirecting it eliminates nothing, so it
    /// does not count toward join profitability.
    used: HashSet<ValueId>,
}

/// The immutable planning environment shared by every motion rule across the sweep rounds.
struct MotionEnv<'a> {
    /// Borrowed only to mint fresh value ids.
    ssa: &'a HLSSA,

    /// The pristine typing of the function (never re-read after minted values exist as operands —
    /// templates and leaders are all pre-motion values).
    types: &'a FunctionTypeInfo,

    /// The CFG for the function in question.
    cfg: &'a CFG,

    /// The totality license for speculative placement, answering over the pristine function.
    oracle: &'a TotalityOracle<'a>,

    /// The kinds of code motion that are allowed.
    motion: MotionLevel,

    /// The anticipated keys on entry to each block — the down-safety proof.
    antic_in: HashMap<BlockId, HashSet<NodeId>>,

    /// The function's natural loops and per-block nesting depths ([`loop_forest`]).
    forest: LoopForest,
}

impl MotionEnv<'_> {
    /// The loop-nesting depth of `block` (zero when it sits in no loop).
    fn depth(&self, block: BlockId) -> usize {
        self.forest.depths.get(&block).copied().unwrap_or(0)
    }
}

/// The mutation state the motion rules accumulate across the sweep rounds.
struct MotionContext {
    /// Occurrence-to-moved-value redirects, applied once after the rounds converge.
    replacements: ValueReplacements,

    /// Sources already redirected — each occurrence is redirected at most once (the availability
    /// arguments make a second attempt unreachable in the hoist-only stage, but join rounds meet
    /// type-guard asymmetries where the cheap set is the simpler invariant).
    redirected: HashSet<ValueId>,

    /// Values minted by a motion rule, keyed by canonical key. Each carrier is available at every
    /// block its recorded block dominates: a hoisted definition is recorded at its header (it
    /// lives on the sole entry edge), a join parameter at its merge.
    carriers: HashMap<NodeId, Vec<Carrier>>,

    /// The (merge, key) pairs already join-inserted. Availability cannot see the materialized
    /// copies sitting in split blocks (new blocks are invisible to the pristine CFG), so this set
    /// is what stops a later round from planting the same parameter twice.
    done_joins: HashSet<(BlockId, NodeId)>,

    /// Memoized argument wiring per (predecessor, target) edge. Splitting happens on first
    /// resolution; memoization keeps later rounds consistent with it (the pristine CFG keeps
    /// naming the original predecessor, whose terminator no longer targets the merge).
    wiring: HashMap<(BlockId, BlockId), PredWiring>,
}

/// A value minted by a motion rule, available at every block dominated by `block`.
struct Carrier {
    block: BlockId,
    value: ValueId,
    typ: Type,
}

/// How [`FunctionMotionState::leader_at`] treats an occurrence sitting in the queried block itself.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Strictness {
    /// Reflexive — availability at the block's *end*, where an in-block occurrence has already run
    /// (the join rule's per-predecessor query).
    AtEnd,

    /// Strict — availability on the block's *entry*: an in-block occurrence does not count (at a
    /// loop header it runs per-iteration, exactly what a hoist would eliminate).
    StrictlyAbove,
}

/// How one predecessor's edge(s) into a target carry materialized copies and jump arguments.
#[derive(Clone)]
struct PredWiring {
    /// Where materialized copies land: the predecessor's own end when every path through it
    /// reaches the target, otherwise the split block on the edge.
    mat_block: BlockId,

    /// The blocks whose `Jmp` terminators enter the target and carry its arguments (two entries
    /// when both arms of one `JmpIf` targeted it).
    arg_blocks: Vec<BlockId>,
}

impl MotionContext {
    fn new() -> Self {
        Self {
            replacements: ValueReplacements::new(),
            redirected: HashSet::default(),
            carriers: HashMap::default(),
            done_joins: HashSet::default(),
            wiring: HashMap::default(),
        }
    }

    /// Record `from -> to`, first writer wins.
    fn redirect(&mut self, from: ValueId, to: ValueId) {
        if self.redirected.insert(from) {
            self.replacements.insert(from, to);
        }
    }

    /// Resolve — memoized — the wiring of the `pred -> target` edge(s), splitting `JmpIf` arms on
    /// first resolution.
    fn pred_wiring(
        &mut self,
        function: &mut HLFunction,
        pred: BlockId,
        target: BlockId,
    ) -> PredWiring {
        if let Some(wiring) = self.wiring.get(&(pred, target)) {
            return wiring.clone();
        }
        let wiring = match function
            .get_block(pred)
            .get_terminator()
            .expect("terminated block")
        {
            Terminator::Jmp(t, _) => {
                debug_assert_eq!(*t, target, "stale edge: predecessor jumps elsewhere");
                PredWiring {
                    mat_block: pred,
                    arg_blocks: vec![pred],
                }
            }
            Terminator::JmpIf(_, t, f) => {
                let (in_true, in_false) = (*t == target, *f == target);
                debug_assert!(
                    in_true || in_false,
                    "stale edge: predecessor branches elsewhere"
                );
                let mut arg_blocks = Vec::new();
                if in_true {
                    arg_blocks.push(split_jmp_if_edge(function, pred, JmpIfArm::True));
                }
                if in_false {
                    arg_blocks.push(split_jmp_if_edge(function, pred, JmpIfArm::False));
                }
                // With both arms on the edge the predecessor's own exit is the edge, so copies can
                // live at its end (dominating both splits); a lone arm hosts them in its split.
                let mat_block = if in_true && in_false {
                    pred
                } else {
                    arg_blocks[0]
                };
                PredWiring {
                    mat_block,
                    arg_blocks,
                }
            }
            Terminator::Return(_) => unreachable!("a Return block has no successors"),
        };
        self.wiring.insert((pred, target), wiring.clone());
        wiring
    }
}

/// One planned parameter insertion at a merge.
struct JoinPlan<'a> {
    node: NodeId,

    /// Deterministic application order: the key's first occurrence position.
    order: usize,

    /// The parameter's type; every leader, template, and redirect is guarded against it.
    expected: &'a Type,

    /// The available leader per predecessor, parallel to the sorted predecessor list.
    leaders: Vec<Option<ValueId>>,

    /// The occurrence cloned into leaderless predecessors (`None` when every edge has a leader).
    template: Option<&'a Occurrence>,

    /// The live occurrences the merge dominates, to be redirected to the parameter.
    redirects: Vec<ValueId>,
}

impl FunctionMotionState {
    fn collect(
        function: &HLFunction,
        cfg: &CFG,
        node_of: &HashMap<ValueId, NodeId>,
    ) -> FunctionMotionState {
        let cyclic = StaticCyclicBlocks::new(function);
        let mut state = FunctionMotionState {
            def_block: HashMap::default(),
            occurrences: HashMap::default(),
            gen_keys: HashMap::default(),
            used: HashSet::default(),
        };
        let mut order = 0;
        for block_id in cfg.get_domination_pre_order() {
            let block = function.get_block(block_id);
            for &(param, _) in block.get_parameters() {
                state.def_block.insert(param, block_id);
            }
            for (instruction, location) in block.get_instructions_with_source_locations() {
                state.used.extend(instruction.get_inputs().copied());
                for &result in instruction.get_results() {
                    state.def_block.insert(result, block_id);
                }
                if !is_motion_candidate(instruction) {
                    continue;
                }
                let result = *instruction.get_results().next().expect("single result");
                let Some(&node) = node_of.get(&result) else {
                    continue;
                };
                // Binding stability: every operand defined outside every cycle (or an interned
                // constant), so the expression names the same values at every program point.
                let stable = instruction.get_inputs().all(|operand| {
                    state
                        .def_block
                        .get(operand)
                        .is_none_or(|&b| !cyclic.is_in_cycle(b))
                });
                if !stable {
                    continue;
                }
                state.occurrences.entry(node).or_default().push(Occurrence {
                    block: block_id,
                    order,
                    result,
                    template: instruction.clone(),
                    location: location.cloned(),
                });
                order += 1;
                state.gen_keys.entry(block_id).or_default().insert(node);
            }
            match block.get_terminator().expect("terminated block") {
                Terminator::Jmp(_, args) => state.used.extend(args.iter().copied()),
                Terminator::JmpIf(cond, ..) => {
                    state.used.insert(*cond);
                }
                Terminator::Return(vals) => state.used.extend(vals.iter().copied()),
            }
        }
        state
    }

    /// The greatest fixpoint of `ANTIC_IN[b] = GEN[b] ∪ ⋂ ANTIC_IN[succ]` over the occurrence keys:
    /// the keys whose value every path from `b`'s entry to exit is bound to compute.
    fn anticipated(&self, function: &HLFunction, cfg: &CFG) -> HashMap<BlockId, HashSet<NodeId>> {
        let domain: HashSet<NodeId> = self.occurrences.keys().copied().collect();
        let order: Vec<BlockId> = cfg.get_domination_pre_order().collect();
        let mut antic_in: HashMap<BlockId, HashSet<NodeId>> =
            order.iter().map(|&b| (b, domain.clone())).collect();

        loop {
            let mut changed = false;
            // Roughly bottom-up (reversed preorder); iteration order affects convergence speed only
            // — the greatest fixpoint is unique.
            for &block_id in order.iter().rev() {
                let mut new_in: HashSet<NodeId> = match function
                    .get_block(block_id)
                    .get_terminator()
                    .expect("terminated block")
                {
                    Terminator::Return(_) => HashSet::default(),
                    Terminator::Jmp(target, _) => antic_in[target].clone(),
                    Terminator::JmpIf(_, t, f) => {
                        antic_in[t].intersection(&antic_in[f]).copied().collect()
                    }
                };
                if let Some(generated) = self.gen_keys.get(&block_id) {
                    new_in.extend(generated.iter().copied());
                }
                if new_in.len() != antic_in[&block_id].len() {
                    antic_in.insert(block_id, new_in);
                    changed = true;
                }
            }
            if !changed {
                return antic_in;
            }
        }
    }

    /// The dominance-legal evaluation of `node` at `block`, if any.
    ///
    /// This is an occurrence whose block dominates it ([`Strictness::AtEnd`] counts an occurrence
    /// in `block` itself; [`Strictness::StrictlyAbove`] does not), or a carrier minted this run
    /// whose recorded block dominates it (always reflexive: a carrier recorded at `block` lives on
    /// its entry edge). Type-guarded, so a mismatched evaluation (witness-wrapper asymmetry) counts
    /// as unavailable.
    fn leader_at(
        &self,
        cfg: &CFG,
        types: &FunctionTypeInfo,
        carriers: &HashMap<NodeId, Vec<Carrier>>,
        node: NodeId,
        block: BlockId,
        expected: &Type,
        strictness: Strictness,
    ) -> Option<ValueId> {
        self.occurrences[&node]
            .iter()
            .find(|occ| {
                (strictness == Strictness::AtEnd || occ.block != block)
                    && cfg.dominates(occ.block, block)
                    && types.get_value_type(occ.result) == expected
            })
            .map(|occ| occ.result)
            .or_else(|| {
                carriers.get(&node).and_then(|carriers| {
                    carriers
                        .iter()
                        .find(|c| cfg.dominates(c.block, block) && c.typ == *expected)
                        .map(|c| c.value)
                })
            })
    }

    /// The first of `occurrences` fit to clone into every block of `targets`.
    ///
    /// Each template operand's definition dominates every target (an absent definition is an
    /// interned constant, available everywhere), and the result type matches `expected` when one is
    /// given (the hoist rule selects its template *first* and derives the expected type from it).
    fn template_for<'a>(
        &self,
        cfg: &CFG,
        types: &FunctionTypeInfo,
        occurrences: &'a [Occurrence],
        targets: &[BlockId],
        expected: Option<&Type>,
    ) -> Option<&'a Occurrence> {
        occurrences.iter().find(|occ| {
            expected.is_none_or(|t| types.get_value_type(occ.result) == t)
                && occ.template.get_inputs().all(|operand| {
                    self.def_block
                        .get(operand)
                        .is_none_or(|&b| targets.iter().all(|&t| cfg.dominates(b, t)))
                })
        })
    }

    /// Hoist every eligible key of `header` onto its sole entry edge, recording the redirects and
    /// carriers into `ctx`.
    ///
    /// Anticipated keys move down-safely; at [`MotionLevel::Speculate`], body-only keys move too
    /// when the totality oracle licenses the op and an eliminated occurrence sits at strictly
    /// greater loop depth. Returns whether anything was hoisted.
    fn hoist_into_header(
        &self,
        env: &MotionEnv,
        function: &mut HLFunction,
        header: BlockId,
        entry_pred: BlockId,
        ctx: &mut MotionContext,
    ) -> bool {
        let loop_blocks = &env.forest.loops[&header];
        let mut candidates: Vec<(NodeId, &Occurrence)> = Vec::new();
        for (&node, occurrences) in &self.occurrences {
            // A key anticipated on entry to the header is down-safe here; any other key is a
            // speculative candidate, admitted only at the Speculate level (and per-op below).
            let down_safe = env.antic_in[&header].contains(&node);
            if !down_safe && env.motion < MotionLevel::Speculate {
                continue;
            }

            // The materialization template: the first occurrence whose operands are available at
            // the entry predecessor (and hence on the entry edge it dominates). Its result type
            // scopes every check below — one key can carry occurrences of distinct types (the
            // witness-wrapper asymmetry), and the hoisted value serves only the template's type.
            let template = self.template_for(env.cfg, env.types, occurrences, &[entry_pred], None);
            let Some(template) = template else {
                continue;
            };
            let expected = env.types.get_value_type(template.result);

            // Availability: a definition strictly dominating the header already carries the value —
            // either an original occurrence (whose in-loop duplicates the elimination sweep already
            // redirected) or this run's motion at a dominating block (whose redirects covered
            // everything that block dominates, this loop included). A mismatched-type evaluation
            // serves none of the occurrences the hoist would, so it does not count.
            let available = self
                .leader_at(
                    env.cfg,
                    env.types,
                    &ctx.carriers,
                    node,
                    header,
                    expected,
                    Strictness::StrictlyAbove,
                )
                .is_some();

            // Profitability: the insertion must replace a per-iteration evaluation — one of the
            // type the hoisted value can actually stand in for.
            let in_loop = occurrences.iter().any(|occ| {
                loop_blocks.contains(&occ.block) && env.types.get_value_type(occ.result) == expected
            });
            if available || !in_loop {
                continue;
            }

            if !down_safe {
                // The totality license, queried at both ends of the entry edge the op will run on
                // (both sound — see the module doc; both needed, a guard fact can live at either).
                let total = env.oracle.is_total_at(&template.template, entry_pred)
                    || env.oracle.is_total_at(&template.template, header);

                // The speculative profitability gate: the wasted zero-trip evaluation is bought
                // only by an elimination at strictly greater loop depth.
                let deeper = occurrences.iter().any(|occ| {
                    loop_blocks.contains(&occ.block)
                        && env.depth(occ.block) > env.depth(entry_pred)
                        && env.types.get_value_type(occ.result) == expected
                });
                if !total || !deeper {
                    continue;
                }
            }
            candidates.push((node, template));
        }
        if candidates.is_empty() {
            return false;
        }
        candidates.sort_unstable_by_key(|(_, occ)| occ.order);

        // Resolve the insertion block once per header: `P` itself when the edge is its sole exit,
        // a fresh split block when `P` branches (parameterless headers only — see module doc). A
        // `JmpIf` with both arms on the edge never reaches here (the caller's single-pred match
        // filters it out), and `pred_wiring` would place copies at `P`'s own end anyway.
        let insert_block = ctx.pred_wiring(function, entry_pred, header).mat_block;

        for (node, occurrence) in candidates {
            let hoisted = env.ssa.fresh_value();
            function
                .get_block_mut(insert_block)
                .push_instruction(occurrence.materialize(hoisted));
            ctx.carriers.entry(node).or_default().push(Carrier {
                block: header,
                value: hoisted,
                typ: env.types.get_value_type(occurrence.result).clone(),
            });

            // Redirect every occurrence the header dominates — the hoisted definition dominates the
            // header's entry, hence all of them (the template included: its instruction goes dead
            // and the integrated DCE sweeps it). Occurrences elsewhere (e.g. sibling branches) keep
            // their own evaluation.
            for occ in &self.occurrences[&node] {
                if env.cfg.dominates(header, occ.block)
                    && env.types.get_value_type(occ.result)
                        == env.types.get_value_type(occurrence.result)
                {
                    ctx.redirect(occ.result, hoisted);
                }
            }
        }
        true
    }

    /// Join the eligible anticipated keys of the merge `merge` through fresh block parameters.
    ///
    /// Available predecessors pass their leader, leaderless ones a copy materialized on their edge,
    /// and the live occurrences the merge dominates redirect to the parameter. Returns whether
    /// anything was inserted.
    fn insert_at_join(
        &self,
        env: &MotionEnv,
        function: &mut HLFunction,
        merge: BlockId,
        ctx: &mut MotionContext,
    ) -> bool {
        let (cfg, types) = (env.cfg, env.types);
        let mut preds: Vec<BlockId> = cfg.get_predecessors(merge).collect();
        preds.sort_unstable();
        preds.dedup();
        if preds.len() < 2 {
            return false;
        }

        // Plan against pristine facts only; mutation starts after the plan set is fixed.
        let mut plans: Vec<JoinPlan> = Vec::new();
        for &node in &env.antic_in[&merge] {
            if ctx.done_joins.contains(&(merge, node)) {
                continue;
            }
            let occurrences = &self.occurrences[&node];
            let expected = types.get_value_type(occurrences[0].result);
            let leaders: Vec<Option<ValueId>> = preds
                .iter()
                .map(|&p| {
                    self.leader_at(
                        cfg,
                        types,
                        &ctx.carriers,
                        node,
                        p,
                        expected,
                        Strictness::AtEnd,
                    )
                })
                .collect();
            let missing: Vec<BlockId> = preds
                .iter()
                .zip(&leaders)
                .filter_map(|(&p, leader)| leader.is_none().then_some(p))
                .collect();
            // Available nowhere: joining computes on every path and eliminates on none of them.
            if missing.len() == preds.len() {
                continue;
            }

            // Profitability, as a static-cost gate: the parameter and its arguments are pure
            // overhead on top of the materialized copies, so the join must drain strictly more live
            // evaluations than it plants. An instruction-count-neutral join (the classic one-arm or
            // one-recompute diamond) is refused — it trades nothing statically and only pays
            // argument moves dynamically.
            let redirects: Vec<ValueId> = occurrences
                .iter()
                .filter(|occ| {
                    cfg.dominates(merge, occ.block)
                        && types.get_value_type(occ.result) == expected
                        && self.used.contains(&occ.result)
                        && !ctx.redirected.contains(&occ.result)
                })
                .map(|occ| occ.result)
                .collect();
            if redirects.len() <= missing.len() {
                continue;
            }

            // The copy template for leaderless edges; a fully-available key needs none.
            let template = if missing.is_empty() {
                None
            } else {
                match self.template_for(cfg, types, occurrences, &missing, Some(expected)) {
                    Some(template) => Some(template),
                    None => continue,
                }
            };
            plans.push(JoinPlan {
                node,
                order: occurrences[0].order,
                expected,
                leaders,
                template,
                redirects,
            });
        }
        if plans.is_empty() {
            return false;
        }
        plans.sort_unstable_by_key(|plan| plan.order);

        let wiring: Vec<PredWiring> = preds
            .iter()
            .map(|&p| ctx.pred_wiring(function, p, merge))
            .collect();
        for plan in plans {
            let param = env.ssa.fresh_value();
            for (pred, leader) in wiring.iter().zip(&plan.leaders) {
                let arg = match leader {
                    Some(value) => *value,
                    None => {
                        let copy = env.ssa.fresh_value();
                        let instruction = plan
                            .template
                            .expect("a leaderless edge implies a template")
                            .materialize(copy);
                        function
                            .get_block_mut(pred.mat_block)
                            .push_instruction(instruction);
                        copy
                    }
                };
                for &arg_block in &pred.arg_blocks {
                    match function.get_block_mut(arg_block).get_terminator_mut() {
                        Terminator::Jmp(_, args) => args.push(arg),
                        _ => unreachable!("argument-carrying predecessors end in Jmp"),
                    }
                }
            }
            function
                .get_block_mut(merge)
                .push_parameter(param, plan.expected.clone());
            ctx.carriers.entry(plan.node).or_default().push(Carrier {
                block: merge,
                value: param,
                typ: plan.expected.clone(),
            });
            ctx.done_joins.insert((merge, plan.node));
            for result in plan.redirects {
                ctx.redirect(result, param);
            }
        }
        true
    }
}

// INTERNAL FUNCTIONALITY
// ================================================================================================

/// The natural-loop structure of one function: every header's loop block set, and the nesting
/// depth of every block (how many headers' natural loops contain it; blocks inside no loop are
/// absent).
///
/// The depth is the speculative profitability measure: strictly greater depth is the static proxy
/// for "executes more often", the license to trade one entry-edge evaluation for a per-iteration
/// one.
struct LoopForest {
    loops: HashMap<BlockId, HashSet<BlockId>>,
    depths: HashMap<BlockId, usize>,
}

/// Compute the [`LoopForest`] of `cfg`, materializing each loop header's [`natural_loop`] once
/// for the whole sweep.
fn loop_forest(cfg: &CFG) -> LoopForest {
    let mut loops: HashMap<BlockId, HashSet<BlockId>> = HashMap::default();
    let mut depths: HashMap<BlockId, usize> = HashMap::default();
    for header in cfg.get_domination_pre_order() {
        if cfg.is_loop_entry(header) {
            let blocks = natural_loop(cfg, header);
            for &block in &blocks {
                *depths.entry(block).or_default() += 1;
            }
            loops.insert(header, blocks);
        }
    }
    LoopForest { loops, depths }
}

/// The blocks of `header`'s natural loop: the header plus every block that reaches a back edge
/// without passing through the header.
fn natural_loop(cfg: &CFG, header: BlockId) -> HashSet<BlockId> {
    let mut blocks: HashSet<BlockId> = HashSet::default();
    blocks.insert(header);
    let mut work: Vec<BlockId> = cfg
        .get_predecessors(header)
        .filter(|&p| cfg.dominates(header, p))
        .collect();
    while let Some(block) = work.pop() {
        if blocks.insert(block) {
            work.extend(cfg.get_predecessors(block));
        }
    }
    blocks
}

/// The ops the motion stages may materialize at a new point: the pure scalar shapes plus
/// `ArrayGet`, matching the elimination sweep's redirect set minus the element-wise `Map` casts
/// (expanded into loops late; never worth moving).
fn is_motion_candidate(instruction: &OpCode) -> bool {
    use crate::compiler::ssa::hlssa::CastTarget;
    match instruction {
        OpCode::Cast { target, .. } => !matches!(target, CastTarget::Map(_)),
        OpCode::BinaryArithOp { .. }
        | OpCode::Cmp { .. }
        | OpCode::MulConst { .. }
        | OpCode::SExt { .. }
        | OpCode::BitRange { .. }
        | OpCode::Not { .. }
        | OpCode::Select { .. }
        | OpCode::ArrayGet { .. } => true,
        _ => false,
    }
}

/// Debug-only structural check for the argument-carrying rewrites: every jump's argument count
/// matches its target's parameter count, and no `JmpIf` arm enters a parameterized block.
#[cfg(debug_assertions)]
fn assert_jump_argument_consistency(function: &HLFunction) {
    for (&block_id, block) in function.get_blocks() {
        match block.get_terminator() {
            Some(Terminator::Jmp(target, args)) => {
                let params = function.get_block(*target).get_parameters().count();
                assert_eq!(
                    args.len(),
                    params,
                    "jump argument arity mismatch on {block_id:?} -> {target:?}"
                );
            }
            Some(Terminator::JmpIf(_, t, f)) => {
                for target in [t, f] {
                    assert!(
                        !function.get_block(*target).has_parameters(),
                        "JmpIf arm into a parameterized block on {block_id:?} -> {target:?}"
                    );
                }
            }
            _ => {}
        }
    }
}
