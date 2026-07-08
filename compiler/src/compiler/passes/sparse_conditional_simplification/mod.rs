//! Sparse Conditional Simplification (SCS): a single pass that folds the work of sparse conditional
//! constant propagation, conditional propagation, and dead-code elimination together, driven by the
//! combined [`ClickCooper`] analysis.
//!
//! It subsumes the role of SCCP in its unconditional core, handling the pruning of unreachable
//! blocks, the aliasing of unconditionally-constant values, folding constant branch/select
//! conditions, and so on. This is combined with a layer that consumes the conditional facts
//! computed by the analysis. It is then followed by a standard mark-and-sweep pass for dead-code
//! elimination.
//!
//! The pass cleans up after its own branch pruning: a block orphaned by one of its `JmpIf -> Jmp`
//! folds never survives the run.
//!
//! # Conditional Layer
//!
//! Conditional facts are those that are established by asserts or branches, and can only be used
//! under rewrites that preserve the establishing constraint.
//!
//! - **Conditional `Cmp{Eq}` Folding:** A `Cmp{Eq, a, b}` whose operands are proven equal by a
//!   dominating (or earlier same-block) `AssertCmp{Eq}` folds to `true`; one proven unequal on the
//!   false edge of an equality branch folds to `false`. The `Cmp` result is _defined at_ the point
//!   the fact holds, so the establisher dominates every use — aliasing the result function-wide is
//!   sound. When that result is the block's terminating `JmpIf` condition, the fact is conditional
//!   so `const_bool_in_block` cannot corroborate it; the terminator is folded to a `Jmp` on the
//!   taken target here, never left as a constant-fed `JmpIf`. This fold+prune covers the _defining_
//!   block only: a use of the result as the `JmpIf` condition of a _different_ dominated block is
//!   instead aliased to the folded constant function-wide, leaving a constant-fed `JmpIf` that can
//!   be handled downstream.
//! - **Per-Point Asserted-Constant Substitution:** An operand pinned to a constant by a dominating
//!   (or earlier same-block) `Assert{v}` / `AssertCmp{Eq, v, c}` is substituted _locally on that
//!   instruction only_. This is never done via the function-wide replacement map, as this would
//!   rewrite uses _before_ the assert, where the fact does not hold. All `ProgramPoint` indices are
//!   positions in the _pristine_ instruction vector (`conditional::build` recorded against those
//!   indices); the strict-`>` same-block rule then keeps an assert from folding its own operand
//!   into a tautology.
//! - **Operand-Level `asserted_equal` Copy-Propagation:** An operand proven equal to a dominating
//!   value by an assert is redirected to that value's canonical representative. This is a _local_
//!   rewrite at the use point that is type-guarded and skipped where a stronger function-wide fold
//!   already applies.
//! - **Redundant Equality-Assert Drop:** An `AssertCmp{Eq, a, b}` whose operands are proven equal
//!   _independently of the assert_ — by congruence / value-numbering / constants (`known_equal`),
//!   never by the assert's own `asserted_equal` — is dropped: the rest of the circuit already
//!   forces `a == b`, so it would only lower to a wasted `Constrain` row. This is the assert-shaped
//!   analogue of folding `Cmp{Eq, a, b}` of congruent operands to `true`.
//!
//! # Anticipated Layer
//!
//! Each assert-derived rewrite above also consumes the _anticipated_ mirror of its fact — one
//! established by an assert in a block strictly _post-dominating_ the use (or later in its block),
//! which is thus bound to run after it.
//!
//! One extra obligation shapes every consumption site: an anticipated fact is justified by its
//! assert _still checking the genuine values_, so it must never rewrite the operands of any
//! `Assert` / `AssertCmp`. Otherwise two bound-to-run checks of one fact erase each other — the
//! dominance fact from the first folds the second's operands into a droppable tautology while the
//! anticipated fact from the second folds the first's — and the constraint is silently lost.
//!
//! Concretely:
//!
//! - **Anticipated `Cmp{Eq}` Folding:** A `Cmp{Eq, a, b}` proven by `anticipated_equal` folds to
//!   `true`, but through a _separate_ replacement map applied to every input **except** assert
//!   operands, and the `Cmp` instruction is _kept_ — an excluded assert (e.g. a bare
//!   `Assert{result}` over the comparison, which `rewrite_asserts` leaves opaque) may still
//!   reference the original result. Folding a `JmpIf` condition through this map (or deciding the
//!   same-block terminator) is sound: a use of the result implies its defining block ran, and
//!   post-dominance ranges over _all_ paths from that block, so the pruned CFG still reaches the
//!   justifying assert. A _witness-typed_ result cannot alias to a bare constant (it would mistype
//!   the IR), so it takes a keep-compatible recast through a sibling map instead: its uses are
//!   redirected to a shared entry-hoisted `cast true to WitnessOf` while the kept `Cmp` still
//!   defines the result for the excluded ones.
//! - **Anticipated Constant Substitution:** The per-point substitution consults the anticipated
//!   channel on a dominance-channel miss. For every consumer _except_ `Assert`/`AssertCmp` (the
//!   dominance channel remains allowed into assert operands), it is justified by an assert that
//!   already ran, which no later rewrite can unmake.
//! - **Leader Union:** Copy-propagation redirects through `anticipated_leader`, the union of both
//!   directions' equality classes — including a use textually _before_ a same-block establishing
//!   assert (bound to run on the very same bindings); its existing skip of `Assert`/`AssertCmp`
//!   instructions is exactly Gate 3 for the redirect channel, and is the sole guard at the
//!   establishing assert's own index.
//!
//! The redundant-assert drop is deliberately **not** extended: dropping an assert because of a
//! fact the assert family itself established would be the circularity Gate 3 exists to prevent
//! (`known_equal` is congruence — independent of asserts — so the existing drop stays sound).

#[cfg(test)]
mod test;

use crate::{
    collections::{HashMap, HashSet},
    compiler::{
        analysis::{
            click_cooper::{ClickCooper, bool_constant},
            flow_analysis::FlowAnalysis,
            types::{FunctionTypeInfo, TypeInfo, Types},
        },
        pass_manager::{Analysis, AnalysisId, AnalysisStore, Pass},
        passes::{
            dead_code_elimination::{Config, DCE},
            shared::value_replacements::{ReplaceScope, ValueReplacements},
        },
        ssa::{
            BlockId, FunctionId, Instruction, Located, ProgramPoint, Terminator, ValueId,
            hlssa::{CastTarget, CmpKind, HLFunction, HLSSA, OpCode},
        },
    },
};

// SCS
// ================================================================================================

/// Sparse conditional simplification: unconditional SCCP-style folding + the conditional layer +
/// integrated dead-code elimination, in one pass.
pub struct SCS {
    config: Config,
}

impl Pass for SCS {
    fn name(&self) -> &'static str {
        "scs"
    }

    fn needs(&self) -> Vec<AnalysisId> {
        vec![ClickCooper::id(), TypeInfo::id()]
    }

    fn run(&self, ssa: &mut HLSSA, store: &AnalysisStore) {
        propagate_all(ssa, store.get::<ClickCooper>(), store.get::<TypeInfo>());
        // The rewrite invalidates the store's `FlowAnalysis`; recompute it for the integrated DCE.
        let flow = FlowAnalysis::run(ssa);
        DCE::new(self.config).do_run(ssa, &flow);
    }
}

impl SCS {
    pub fn new(config: Config) -> Self {
        Self { config }
    }

    /// Standalone entry (tests / callers without an `AnalysisStore`): recomputes `TypeInfo` from
    /// the current SSA, propagates, then runs the integrated DCE.
    pub fn do_run(&self, ssa: &mut HLSSA, cc: &ClickCooper) {
        let flow = FlowAnalysis::run(ssa);
        let type_info = Types::new().run(ssa, &flow);
        propagate_all(ssa, cc, &type_info);
        let flow = FlowAnalysis::run(ssa);
        DCE::new(self.config).do_run(ssa, &flow);
    }
}

// PROPAGATION
// ================================================================================================

/// Drive [`propagate`] over every function in the `ssa`, completing the fold-and-prune rewrite for
/// the whole module.
///
/// Each function is taken out of `ssa` for the duration of its rewrite and put back afterwards, so
/// `propagate` holds a `&mut HLFunction` while still reading the shared `ssa` (the constant pool it
/// interns into).
fn propagate_all(ssa: &mut HLSSA, cc: &ClickCooper, type_info: &TypeInfo) {
    let fids: Vec<_> = ssa.get_function_ids().collect();
    for fid in fids {
        let mut function = ssa.take_function(fid);
        propagate(&mut function, ssa, cc, type_info, fid);
        ssa.put_function(fid, function);
    }
}

/// Apply the simplifications to a single `function` in place.
///
/// No dead code is removed here, as this is left to the internal DCE with two exceptions, both
/// block-shaped. Blocks the analysis never reached are dropped eagerly (step 1), since the later
/// DCE seeds every `Return` block live and would otherwise resurrect an unreachable one. Blocks
/// orphaned by this run's own conditional `JmpIf -> Jmp` folds are reclaimed at the end (step 10)
/// for the same reason — and because they must not escape the pass at all: downstream consumers
/// (the `PointsTo` build, `InstructionLowering`'s value-range analysis) type every block's
/// instructions against reachable-only `TypeInfo`, so a leftover orphan is an ICE, not just dead
/// weight.
///
/// Edits are made through three channels, distinguished by where the underlying fact holds:
///
/// - **The function-wide replacement map**, used only for facts whose establisher dominates every
///   use of the rewritten value. unconditionally-constant values, constant-conditioned `Select`s
///   aliased to the chosen arm, and `Cmp{Eq}` results folded here, which are defined at the fold
///   point.
/// - **The anticipated replacement maps**, for `Cmp{Eq}` results folded via the anticipated
///   channel. These are applied like the function-wide map but never to `Assert`/`AssertCmp`
///   inputs, with the folded `Cmp` kept in place for those excluded uses. A witness-typed result
///   goes through the sibling map redirecting to an entry-hoisted `cast true to WitnessOf`,
///   applied only to the [`is_witness_subst_safe_target`] allowlist (witness-machinery consumers
///   keep the real witness the kept `Cmp` defines).
/// - **Local, in-place edits**, used for facts that do _not_ hold function-wide: block-entry branch
///   predicates, per-point asserted constants, and the per-block terminator folding. Values proven
///   constant that are `WitnessOf`-typed are also written in place so the IR stays well-typed.
///
/// Every `ProgramPoint` indexes the block's _pristine_ instruction vector: instructions are dropped
/// while iterating but never reindexed, and the terminator takes the index equal to the original
/// instruction count.
fn propagate(
    function: &mut HLFunction,
    ssa: &HLSSA,
    cc: &ClickCooper,
    type_info: &TypeInfo,
    fid: FunctionId,
) {
    let fn_type_info = type_info.get_function(fid);

    // 1. Drop blocks the analysis never reached. Every kept terminator only targets reachable
    // blocks: a JmpIf with a (constant or conditionally-known) condition is rewritten to a Jmp to
    // its live successor below, and all other terminators had all successor edges marked
    // executable. This must run even though DCE follows: DCE seeds every `Return`-terminated block
    // live unconditionally, so an unreachable `Return` block left here would be resurrected. It
    // also covers only solver-decided reachability (constant branch conditions): a
    // *conditionally*-decided branch keeps both edges executable in the analysis, so the blocks
    // its fold orphans are invisible here — step 10 reclaims those after the folds have run.
    let all_blocks: Vec<BlockId> = function.get_blocks().map(|(id, _)| *id).collect();
    for bid in &all_blocks {
        if !cc.is_reachable(fid, *bid) {
            let _ = function.take_block(*bid);
        }
    }

    // 2. Alias every unconditionally-constant value to the interned constant, via the function-wide
    // map. A `WitnessOf`-typed value is the exception (aliasing would mistype the IR): it is
    // collected here and redefined in place as `cast <const> to WitnessOf` in the instruction loop.
    let mut replacements = ValueReplacements::new();

    // `Cmp{Eq}` results folded via the _anticipated_ channel. Kept apart from `replacements`
    // because that map is applied to _every_ input in the function, while an anticipated fact must
    // never rewrite an assert's inputs. Applied below with the assert exclusion.
    let mut anticipated_replacements = ValueReplacements::new();

    // The witness-typed sibling of `anticipated_replacements`: `WitnessOf`-typed `Cmp{Eq}` results
    // folded via the anticipated channel, redirected to the shared entry-hoisted `cast true to
    // WitnessOf` (a bare constant would mistype the IR). Applied below under the stricter
    // `is_witness_subst_safe_target` allowlist — a witness-machinery consumer must keep referencing
    // the real witness, which the kept `Cmp` still defines.
    let mut anticipated_witness_replacements = ValueReplacements::new();

    // Witness-typed constants materialized as casts — fed by `substitute_asserted_consts`
    // (witness-typed asserted-constant operands) and the anticipated witness `Cmp{Eq}` fold: keyed
    // by the bare interned constant, valued by the fresh `cast <const> to WitnessOf` result.
    // Accumulated across the whole function and hoisted into the entry block at the end (so each
    // definition dominates every use, and identical constants share one cast).
    let mut witness_const_casts: HashMap<ValueId, ValueId> = HashMap::default();
    let const_values = cc.new_const_values(fid);
    let const_set: HashSet<ValueId> = const_values.iter().map(|(v, _)| *v).collect();

    // In practice `witness_consts` only ever holds `Cmp` instruction _results_ — the sole witnessed
    // constants the analysis produces (a vacuous witnessed comparison). A witness-typed _block
    // parameter_ proven constant would also land here, but is intentionally left as-is (neither
    // aliased nor redefined — the cast-redefine in the instruction loop runs only for instruction
    // results): sound, just unfolded.
    let mut witness_consts = HashMap::default();
    for (v, c) in &const_values {
        if fn_type_info.get_value_type(*v).is_witness_of() {
            witness_consts.insert(*v, c.clone());
        } else {
            replacements.insert(*v, ssa.add_const((**c).clone()));
        }
    }

    // The pristine index of every surviving instruction (and each block's terminator index),
    // recorded as the fold below compacts each block. The asserted-equal copy-propagation pass
    // needs these to query `ClickCooper::asserted_leader` at the right point (its same-block
    // granularity is keyed against pristine indices, which compaction would otherwise renumber).
    let mut block_points: HashMap<BlockId, (Vec<usize>, usize)> = HashMap::default();

    // Whether any conditionally-decided `JmpIf` was folded to a `Jmp` (step 7). Such a fold can
    // orphan the untaken arm's subgraph — the solver kept both edges executable, so step 1 could
    // not have dropped it — and step 10 then reclaims everything unreachable.
    let mut folded_terminator = false;

    let kept_blocks: Vec<BlockId> = function.get_blocks().map(|(id, _)| *id).collect();
    for bid in kept_blocks {
        let local_replacements = bool_fact_replacements(ssa, cc, fid, bid);

        // `Cmp{Eq}` results folded conditionally in _this_ block (value -> bool), so the terminator
        // step can prune an equality branch whose condition is such a result.
        let mut folded_bools: HashMap<ValueId, bool> = HashMap::default();
        let block = function.get_block_mut(bid);

        // The pristine vector: its length is the original instruction count (the index a terminator
        // use takes), and `enumerate` yields the original indices the conditional facts are keyed
        // against — dropping instructions below never shifts a later instruction's index.
        let instructions = block.take_instructions();
        let instr_count = instructions.len();
        let mut kept = Vec::with_capacity(instr_count);
        // The pristine index of each instruction pushed to `kept`, kept 1:1 with it.
        let mut kept_points: Vec<usize> = Vec::with_capacity(instr_count);

        for (i, located) in instructions.into_iter().enumerate() {
            // Keep the source location so every instruction we pass through — or replace with a
            // fold — retains its provenance for downstream diagnostics.
            let (instr, location) = located.take();
            // 2 (Cont). A single-result instruction whose result is unconditionally constant is a
            // pure scalar fold or pure sequence projection: drop it (uses aliased above), or recast
            // a witnessed constant to keep its `WitnessOf` type.
            //
            // The scope is to force the borrow of `results` to end.
            {
                let mut results = instr.get_results();
                if let (Some(r), None) = (results.next(), results.next()) {
                    let is_const = const_set.contains(r);
                    let foldable = instr.is_pure_scalar_fold()
                        || matches!(instr, OpCode::ArrayGet { .. } | OpCode::SliceLen { .. });

                    // An internal-consistency check on ClickCooper's output: a non-foldable result
                    // in the constant set cannot arise from well-formed SSA.
                    assert!(
                        !is_const || foldable,
                        "ICE: Result {r:?} of non-foldable instruction {instr:?} is in the constant set; ClickCooper must only fold pure scalar ops and pure sequence projections to a constant"
                    );

                    if is_const && foldable {
                        if let Some(c) = witness_consts.get(r) {
                            let bare = ssa.add_const((**c).clone());
                            kept.push(Located::new(
                                OpCode::Cast {
                                    result: *r,
                                    value: bare,
                                    target: CastTarget::WitnessOf,
                                },
                                location.clone(),
                            ));
                            kept_points.push(i);
                        }
                        continue;
                    }
                }
            }

            // 3. A `Cmp{Eq}` proven by an asserted equality folds to `true`; one proven by a known
            // disequality folds to `false`. The result is defined here, so the establisher
            // dominates every use and aliasing the result function-wide is sound.
            if let OpCode::Cmp {
                kind: CmpKind::Eq,
                result,
                lhs,
                rhs,
            } = &instr
            {
                let (result, lhs, rhs) = (*result, *lhs, *rhs);
                let point = ProgramPoint::new(bid, i);
                let folded = if cc.asserted_equal(fid, point, lhs, rhs) {
                    Some(true)
                } else if cc.known_unequal(fid, bid, lhs, rhs) {
                    Some(false)
                } else {
                    None
                };
                if let Some(b) = folded {
                    if fn_type_info.get_value_type(result).is_witness_of() {
                        let bare = ssa.add_const((*bool_constant(b)).clone());
                        kept.push(Located::new(
                            OpCode::Cast {
                                result,
                                value: bare,
                                target: CastTarget::WitnessOf,
                            },
                            location.clone(),
                        ));
                        kept_points.push(i);
                    } else {
                        replacements.insert(result, ssa.add_const((*bool_constant(b)).clone()));
                        folded_bools.insert(result, b);
                    }
                    continue;
                }

                // 3b. The anticipated direction: operands proven equal by a _bound-to-run_ assert
                // fold the comparison to `true` — but through the anticipated maps (applied with
                // the Gate-3 assert exclusion below), and the instruction is _kept_: an excluded
                // assert may still reference the original result, which deleting would leave
                // dangling. A witness-typed result takes the keep-compatible recast — its safe uses
                // are redirected to the shared entry-hoisted `cast true to WitnessOf` while the
                // kept `Cmp` still defines the result for the excluded ones (the dominance path's
                // in-place recast would not compose with keeping it).
                //
                // A scalar fold recorded here also feeds the same-block terminator step via
                // `folded_bools` — sound, because post-dominance covers all paths from this
                // defining block, so the pruned edge cannot unreach the justifying assert. The
                // witness arm records nothing there: a `JmpIf` condition is never `WitnessOf`-typed
                // in the IR SCS sees (`UntaintControlFlow` linearizes witness-conditioned branches,
                // and `LowerGuards` reintroduces them only after the last SCS run).
                if cc.anticipated_equal(fid, point, lhs, rhs) {
                    let bare = ssa.add_const((*bool_constant(true)).clone());
                    if fn_type_info.get_value_type(result).is_witness_of() {
                        let wit = *witness_const_casts
                            .entry(bare)
                            .or_insert_with(|| ssa.fresh_value());
                        anticipated_witness_replacements.insert(result, wit);
                    } else {
                        anticipated_replacements.insert(result, bare);
                        folded_bools.insert(result, true);
                    }
                }
            }

            // 4. A select with a constant condition aliases to the chosen arm.
            if let OpCode::Select {
                result,
                cond,
                if_t,
                if_f,
            } = &instr
            {
                if let Some(b) = cc.const_bool_in_block(fid, bid, *cond) {
                    replacements.insert(*result, if b { *if_t } else { *if_f });
                    continue;
                }
            }

            // 4b. A tautological equality assert is redundant: its operands are proven equal
            // _independently of this assert_ — by structural value-numbering, constants, or
            // cross-call congruence — so the rest of the circuit already forces `a == b` in every
            // admissible witness. Drop it (it would otherwise lower to a wasted `Constrain` row).
            // Gated on `known_equal` (congruence), NEVER `asserted_equal` (which this very assert
            // establishes — dropping on that would be circular and would lose the fact).
            if let OpCode::AssertCmp {
                kind: CmpKind::Eq,
                lhs,
                rhs,
            } = &instr
            {
                if lhs == rhs || cc.known_equal(fid, *lhs, *rhs) {
                    continue;
                }
            }

            // 5. Block-entry branch facts, then
            // 6. Per-point asserted-constant substitution (local to this instruction). The
            // anticipated channel is consulted for every consumer except an assert: the dominance
            // channel stays allowed into assert operands — it is justified by an assert that
            // already ran, which no later rewrite can unmake.
            let mut instr = instr;
            local_replacements.replace_inputs(&mut instr);
            let allow_witness = is_witness_subst_safe_target(&instr);
            let allow_anticipated = !instr.is_assert();
            substitute_asserted_consts(
                ssa,
                cc,
                fn_type_info,
                fid,
                ProgramPoint::new(bid, i),
                allow_witness,
                allow_anticipated,
                &mut witness_const_casts,
                instr.get_inputs_mut(),
            );
            kept.push(Located::new(instr, location));
            kept_points.push(i);
        }

        block.put_instructions(kept);

        // 7. Fold a JmpIf whose condition is a constant (`const_bool_in_block`) or a conditionally
        // folded `Cmp{Eq}` result (`folded_bools`) to a Jmp on the taken target, pruning the dead
        // edge. This fires only when the folded `Cmp` and the `JmpIf` it feeds share this block; a
        // folded result used as the condition of a dominated block's `JmpIf` is handled by the
        // function-wide alias instead.
        if let Some(Terminator::JmpIf(cond, t, f)) = block.get_terminator() {
            let (cond, t, f) = (*cond, *t, *f);

            // The pristine condition is always analysis-known (every function value and interned
            // input constant is typed), and never `WitnessOf`-typed: `UntaintControlFlow`
            // linearizes witness-conditioned branches, and `LowerGuards` reintroduces them only
            // after the last SCS run. The witness arm of the anticipated fold (step 3b) and the
            // anticipated terminator read below both lean on this invariant.
            debug_assert!(
                !fn_type_info.get_value_type(cond).is_witness_of(),
                "ICE: JmpIf condition {cond:?} is WitnessOf-typed; SCS must never see one"
            );

            // The anticipated read decides the branch by a bound-to-run pin of `cond` (e.g. a
            // post-dominating `Assert{cond}`): the fact is keyed at this block, so the justifying
            // assert post-dominates it and every path from here — through either arm, hence
            // through the kept one — still reaches it. A `JmpIf` is not an assert, so Gate 3
            // permits the fold.
            let decided = cc
                .const_bool_in_block(fid, bid, cond)
                .or_else(|| folded_bools.get(&cond).copied())
                .or_else(|| {
                    cc.anticipated_const_bool(fid, ProgramPoint::new(bid, instr_count), cond)
                });
            if let Some(b) = decided {
                let target = if b { t } else { f };
                block.set_terminator(Terminator::Jmp(target, vec![]));
                folded_terminator = true;
            }
        }
        local_replacements.replace_terminator(block.get_terminator_mut());

        // 7 (cont). Per-point asserted-constant substitution on terminator value args — `Return`
        // values and `Jmp` args only (index == original instruction count), never the `JmpIf`
        // condition (branch decisions go through `const_bool_in_block` / `folded_bools` above).
        // `Return` values and `Jmp` args are pure value consumers, so witness substitution is
        // allowed here just as for the safe-target instructions above.
        let term_point = ProgramPoint::new(bid, instr_count);
        match block.get_terminator_mut() {
            Terminator::Return(vals) => {
                substitute_asserted_consts(
                    ssa,
                    cc,
                    fn_type_info,
                    fid,
                    term_point,
                    true,
                    true,
                    &mut witness_const_casts,
                    vals.iter_mut(),
                );
            }
            Terminator::Jmp(_, params) => {
                substitute_asserted_consts(
                    ssa,
                    cc,
                    fn_type_info,
                    fid,
                    term_point,
                    true,
                    true,
                    &mut witness_const_casts,
                    params.iter_mut(),
                );
            }
            Terminator::JmpIf(..) => {}
        }

        block_points.insert(bid, (kept_points, instr_count));
    }

    // 8. Asserted-equal copy-propagation. Now that `replacements` is complete, redirect each
    // operand to its dominance-root-most asserted-equal representative. This is a second walk so
    // the outcome is independent of block-iteration order:
    for (bid, (points, instr_count)) in &block_points {
        let block = function.get_block_mut(*bid);
        for (instr, &index) in block.get_instructions_mut().zip(points) {
            // Never rewrite an establisher's own operands: it could vacuum the very equality the
            // redirect relies on (`AssertCmp{Eq, leader, leader}`) and never helps. This skip is
            // also exactly Gate 3 for the redirect channel: the leader query below unions in
            // anticipated pairs, whose facts must never reach assert inputs.
            if instr.is_assert() {
                continue;
            }
            copy_propagate_asserted_equals(
                cc,
                fn_type_info,
                [
                    &replacements,
                    &anticipated_replacements,
                    &anticipated_witness_replacements,
                ],
                fid,
                ProgramPoint::new(*bid, index),
                instr.get_inputs_mut(),
            );
        }

        // Terminator value args only (`Return` values, `Jmp` block-args) — never the `JmpIf`
        // condition.
        let term_point = ProgramPoint::new(*bid, *instr_count);
        match block.get_terminator_mut() {
            Terminator::Return(vals) => copy_propagate_asserted_equals(
                cc,
                fn_type_info,
                [
                    &replacements,
                    &anticipated_replacements,
                    &anticipated_witness_replacements,
                ],
                fid,
                term_point,
                vals.iter_mut(),
            ),
            Terminator::Jmp(_, params) => copy_propagate_asserted_equals(
                cc,
                fn_type_info,
                [
                    &replacements,
                    &anticipated_replacements,
                    &anticipated_witness_replacements,
                ],
                fid,
                term_point,
                params.iter_mut(),
            ),
            Terminator::JmpIf(..) => {}
        }
    }

    replacements.apply_to_function(function, ReplaceScope::Inputs);

    // The anticipated aliases go everywhere the function-wide map went — every instruction input
    // and every terminator — EXCEPT `Assert`/`AssertCmp` inputs (Gate 3: an anticipated fact must
    // never weaken an assert's check). Rewriting a `JmpIf` condition here is sound: a use of the
    // folded result implies its defining block ran, and post-dominance ranges over all paths from
    // that block, so the pruned CFG still reaches the justifying assert.
    //
    // The three maps' keys are disjoint by construction, so each value has one authoritative
    // rewrite. A later sweep may still rewrite a value an earlier one installed but that
    // composition is sound: it is exactly the rewrite the later sweep was licensed to make at that
    // site, the anticipated values themselves (interned constants and fresh cast results) are keys
    // of no map, and assert inputs see only the dominance-installed value (Gate 3).
    if !anticipated_replacements.is_empty() {
        anticipated_replacements
            .apply_to_function_where(function, ReplaceScope::Inputs, |instr| !instr.is_assert());
    }

    // The witness-typed anticipated aliases go through the stricter `is_witness_subst_safe_target`
    // allowlist: a witness-machinery consumer must keep referencing the real witness, which the
    // kept `Cmp` still defines — and, asserts being outside the allowlist, Gate 3 holds a fortiori.
    // The sweep always rewrites terminators, which is intended: `Return`/`Jmp` args are pure value
    // consumers (witness substitution is allowed there, as in step 7 cont), and a `JmpIf` condition
    // is never witness-typed, so no key can match it.
    if !anticipated_witness_replacements.is_empty() {
        anticipated_witness_replacements.apply_to_function_where(
            function,
            ReplaceScope::Inputs,
            is_witness_subst_safe_target,
        );
    }

    // 9. Hoist the witness-typed constants materialized above — by the asserted-constant
    // substitution and the anticipated witness `Cmp{Eq}` fold — into the entry block, whose every
    // instruction it dominates, so each `cast <const> to WitnessOf` definition dominates the uses
    // redirected to it. Emit them in result-id order for a deterministic instruction sequence.
    if !witness_const_casts.is_empty() {
        let mut casts: Vec<(ValueId, ValueId)> = witness_const_casts.into_iter().collect();
        casts.sort_by_key(|(_, wit)| wit.0);
        let mut entry_instrs: Vec<Located<OpCode>> = casts
            .into_iter()
            .map(|(bare, wit)| {
                Located::without(OpCode::Cast {
                    result: wit,
                    value: bare,
                    target: CastTarget::WitnessOf,
                })
            })
            .collect();
        let entry = function.get_entry_mut();
        entry_instrs.extend(entry.take_instructions());
        entry.put_instructions(entry_instrs);
    }

    // 10. Reclaim the blocks this run's own terminator folds orphaned. Only a conditional fold can
    // orphan (a constant-condition fold's dead arm was already analysis-unreachable and dropped in
    // step 1), so the sweep is skipped when no `JmpIf` was folded.
    if folded_terminator {
        reclaim_orphaned_blocks(function);
    }
}

/// Remove every block no longer reachable from the entry, walking the current terminators.
///
/// This is the pass's own cleanup for the blocks its conditional `JmpIf -> Jmp` folds orphan: the
/// solver marked both edges of a conditionally-decided branch executable, so neither step 1 (which
/// drops analysis-unreachable blocks) nor the integrated DCE (which seeds every `Return` block
/// live) reclaims them.
///
/// They must not survive the pass: downstream consumers walk every block and type instructions
/// against reachable-only `TypeInfo` (the `PointsTo` build, `InstructionLowering`'s value-range
/// analysis), so a leftover orphan is an ICE, not just dead weight.
fn reclaim_orphaned_blocks(function: &mut HLFunction) {
    let mut reachable: HashSet<BlockId> = HashSet::default();
    let mut worklist = vec![function.get_entry_id()];
    while let Some(bid) = worklist.pop() {
        if !reachable.insert(bid) {
            continue;
        }
        match function.get_block(bid).get_terminator() {
            Some(Terminator::Jmp(target, _)) => worklist.push(*target),
            Some(Terminator::JmpIf(_, then_b, else_b)) => {
                worklist.push(*then_b);
                worklist.push(*else_b);
            }
            Some(Terminator::Return(_)) | None => {}
        }
    }

    let all_blocks: Vec<BlockId> = function.get_blocks().map(|(id, _)| *id).collect();
    for bid in all_blocks {
        if !reachable.contains(&bid) {
            let _ = function.take_block(bid);
        }
    }
}

// UTILITIES
// ================================================================================================

/// Substitute each operand pinned to a constant by an assert at `point` with that constant, in
/// place — consulting the dominance channel (`asserted_const`) and, where `allow_anticipated`
/// permits, the anticipated channel (`anticipated_const`) on a miss.
///
/// Sound only locally (at `point`) as the function-wide map must never carry these. It first
/// queries the assert channels so the `WitnessOf` type check only runs for original value ids known
/// to the analysis — branch-fact constants introduced by an earlier `replace_inputs` are not in the
/// assert maps and have no `TypeInfo` entry.
///
/// A scalar operand is replaced with the bare interned constant directly. A `WitnessOf`-typed
/// operand needs a witness-typed constant, so — when `allow_witness` is set (the consumer is a pure
/// value op, never a witness-identity / constraint-establishing one) — it is redirected to a
/// `cast <const> to WitnessOf` recorded in `witness_casts` for the caller to hoist into the entry
/// block. The cast is well-typed: a well-formed establishing `Assert` / `AssertCmp{Eq}` gives the
/// pinned witness operand the constant's scalar type, so wrapping that constant in `WitnessOf`
/// reproduces the operand's type exactly. When `allow_witness` is clear the witness operand is left
/// intact (the constant must not displace the real witness the consumer references).
fn substitute_asserted_consts<'a>(
    ssa: &HLSSA,
    cc: &ClickCooper,
    fn_type_info: &FunctionTypeInfo,
    fid: FunctionId,
    point: ProgramPoint,
    allow_witness: bool,
    allow_anticipated: bool,
    witness_casts: &mut HashMap<ValueId, ValueId>,
    inputs: impl Iterator<Item = &'a mut ValueId>,
) {
    for input in inputs {
        // The dominance channel first; on a miss, the anticipated channel — only where the
        // consumer permits it (`allow_anticipated` is clear exactly for `Assert`/`AssertCmp`
        // inputs, per Gate 3 of the anticipated contract).
        let pinned = cc.asserted_const(fid, point, *input).or_else(|| {
            allow_anticipated
                .then(|| cc.anticipated_const(fid, point, *input))
                .flatten()
        });
        if let Some(c) = pinned {
            let bare = ssa.add_const((*c).clone());
            if !fn_type_info.get_value_type(*input).is_witness_of() {
                *input = bare;
            } else if allow_witness {
                let wit = *witness_casts
                    .entry(bare)
                    .or_insert_with(|| ssa.fresh_value());
                *input = wit;
            }
        }
    }
}

/// Whether an instruction is a safe target for _witness-typed_ asserted-constant substitution: a
/// pure value consumer where redirecting a witness operand to a constant-valued witness changes no
/// witness identity or non-deterministic advice.
///
/// It doubles as the application predicate for the anticipated witness replacement map, which
/// redirects a folded witness `Cmp{Eq}` result to a constant-valued cast under exactly the same
/// policy (and, asserts being omitted here, satisfies Gate 3 a fortiori).
///
/// This is an allowlist, so any opcode not named here defaults to _not_ substituted. The omitted
/// ones are the witness-machinery / constraint-establishing consumers — `WriteWitness` (the honest
/// hint), `Constrain` / `AssertR1C` / `Assert` / `AssertCmp` (the establishers, which must keep
/// referencing the real witness), `Lookup` / `DLookup`, `BumpD`, `FreshWitness`, `NextDCoeff`, and
/// a `ValueOf` cast feeding witness-forwarding — for which substituting a constant would be unsound
/// or pointless.
fn is_witness_subst_safe_target(op: &OpCode) -> bool {
    matches!(
        op,
        OpCode::BinaryArithOp { .. }
            | OpCode::MulConst { .. }
            | OpCode::Cmp { .. }
            | OpCode::Select { .. }
            | OpCode::ArrayGet { .. }
            | OpCode::ArraySet { .. }
            | OpCode::ToBits { .. }
            | OpCode::ToRadix { .. }
    )
}

/// Redirect each operand to the dominance-root-most member of its equality class at `point`
/// ([`ClickCooper::anticipated_leader`] — the union of the dominance-direction and anticipated
/// pairs), in place — a local copy-propagation that collapses values an assert proves equal.
///
/// Sound only locally (at `point`): the leader provably equals the operand there (on accepting
/// runs) and its definition dominates `point`, but the equality does not hold function-wide, so
/// this never goes through the function-wide `replacements` map. The establishing `Assert` /
/// `AssertCmp` is kept (it carries the fact), so the equality survives DCE. The caller's skip of
/// `Assert`/`AssertCmp` instructions doubles as Gate 3 for the anticipated members of the union.
///
/// Three guards keep it sound and non-pessimizing:
///
/// - **A** skips any operand a whole-function map will already rewrite (a folded `Cmp{Eq}` result
///   or `Select` alias that is itself an asserted-equal member — whether in the unconditional map
///   or either anticipated one): that fold is strictly stronger than a conditional value redirect.
///   This needs the _complete_ maps, so the caller runs this only after the fold loop has populated
///   every replacement.
/// - **B** queries `anticipated_leader` first, so the type lookups in **C** only run for
///   analysis-known values — an operand turned into a fresh constant earlier is in no equality
///   pair, returns `None`, and never reaches `get_value_type` (which would panic on an unknown id).
/// - **C** redirects only across _equal_ types: an `AssertCmp{Eq}`'s operand types are not enforced
///   equal, so redirecting across a mismatch would mistype the IR.
///
/// Unlike [`substitute_asserted_consts`], this needs _no_ witness-consumer allowlist
/// ([`is_witness_subst_safe_target`]). That allowlist exists because substituting a witness operand
/// with a _constant_ would displace the real witness a witness-machinery consumer must reference —
/// dropping the prover's advice. Here the operand is instead redirected to _another real witness_
/// (the leader) that the kept assert constrains equal and whose definition dominates `point`: this
/// is ordinary copy-propagation between two provably-equal SSA values, which is sound for _any_
/// consumer, witness-machinery included. Guard C confines it to redirects across equal types, so
/// the rewritten consumer still references a witness of the same type and identity class. Hence no
/// consumer is excluded.
fn copy_propagate_asserted_equals<'a>(
    cc: &ClickCooper,
    fn_type_info: &FunctionTypeInfo,
    whole_function_maps: [&ValueReplacements; 3],
    fid: FunctionId,
    point: ProgramPoint,
    inputs: impl Iterator<Item = &'a mut ValueId>,
) {
    for input in inputs {
        // Guard A: `get_replacement(v) == v` iff `v` is not slated for a whole-function rewrite,
        // in the unconditional map or either anticipated one.
        if whole_function_maps
            .iter()
            .any(|m| m.get_replacement(*input) != *input)
        {
            continue;
        }

        // Guard B, then Guard C.
        if let Some(leader) = cc.anticipated_leader(fid, point, *input) {
            if leader != *input
                && fn_type_info.get_value_type(*input) == fn_type_info.get_value_type(leader)
            {
                *input = leader;
            }
        }
    }
}

/// Within `bid`, replace every value the analysis knows is a constant boolean (a branch predicate
/// fact) with that constant.
///
/// Local and structure-preserving as the establishing branch is retained.
fn bool_fact_replacements(
    ssa: &HLSSA,
    cc: &ClickCooper,
    fid: FunctionId,
    bid: BlockId,
) -> ValueReplacements {
    let mut replacements = ValueReplacements::new();
    for (value, c) in cc.block_bool_facts(fid, bid) {
        replacements.insert(value, ssa.add_const((*c).clone()));
    }
    replacements
}
