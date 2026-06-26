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
//!   (or earlier same-block) `Assert{v}` / `AssertCmp{Eq, v, c}` is substituted *locally on that
//!   instruction only*. This is never done via the function-wide replacement map, as this would
//!   rewrite uses *before* the assert, where the fact does not hold. All `ProgramPoint` indices are
//!   positions in the *pristine* instruction vector (`conditional::build` recorded against those
//!   indices); the strict-`>` same-block rule then keeps an assert from folding its own operand
//!   into a tautology.
//! - **Operand-Level `asserted_equal` Copy-Propagation:** An operand proven equal to a dominating
//!   value by an assert is redirected to that value's canonical representative. This is a *local*
//!   rewrite at the use point that is type-guarded and skipped where a stronger function-wide fold
//!   already applies.
//! - **Redundant Equality-Assert Drop:** An `AssertCmp{Eq, a, b}` whose operands are proven equal
//!   *independently of the assert* — by congruence / value-numbering / constants (`known_equal`),
//!   never by the assert's own `asserted_equal` — is dropped: the rest of the circuit already forces
//!   `a == b`, so it would only lower to a wasted `Constrain` row. This is the assert-shaped analogue
//!   of folding `Cmp{Eq, a, b}` of congruent operands to `true`.

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
            fix_double_jumps::{ReplaceScope, ValueReplacements},
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
/// No dead code is removed here, as this is left to the internal DCE with one exception. Blocks the
/// analysis never reached are dropped eagerly, since the later DCE seeds every `Return` block live
/// and would otherwise resurrect an unreachable one.
///
/// Edits are made through two channels, distinguished by where the underlying fact holds:
///
/// - **The function-wide replacement map**, used only for facts whose establisher dominates every
///   use of the rewritten value. unconditionally-constant values, constant-conditioned `Select`s
///   aliased to the chosen arm, and `Cmp{Eq}` results folded here, which are defined at the fold
///   point.
/// - **Local, in-place edits**, used for facts that do *not* hold function-wide: block-entry branch
///   predicates, per-point asserted constants, and the per-block terminator folding. Values proven
///   constant that are `WitnessOf`-typed are also written in place so the IR stays well-typed.
///
/// Every `ProgramPoint` indexes the block's *pristine* instruction vector: instructions are dropped
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
    // live unconditionally, so an unreachable `Return` block left here would be resurrected.
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

    // Witness-typed asserted constants substituted by `substitute_asserted_consts`: keyed by the
    // bare interned constant, valued by the fresh `cast <const> to WitnessOf` result. Accumulated
    // across the whole function and hoisted into the entry block at the end (so each definition
    // dominates every use, and identical constants share one cast).
    let mut witness_const_casts: HashMap<ValueId, ValueId> = HashMap::default();
    let const_values = cc.new_const_values(fid);
    let const_set: HashSet<ValueId> = const_values.iter().map(|(v, _)| *v).collect();

    // In practice `witness_consts` only ever holds `Cmp` instruction *results* — the sole witnessed
    // constants the analysis produces (a vacuous witnessed comparison). A witness-typed *block
    // parameter* proven constant would also land here, but is intentionally left as-is (neither
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

    let kept_blocks: Vec<BlockId> = function.get_blocks().map(|(id, _)| *id).collect();
    for bid in kept_blocks {
        let local_replacements = bool_fact_replacements(ssa, cc, fid, bid);

        // `Cmp{Eq}` results folded conditionally in *this* block (value -> bool), so the terminator
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
            // *independently of this assert* — by structural value-numbering, constants, or
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
            // 6. per-point asserted-constant substitution (local to this instruction).
            let mut instr = instr;
            local_replacements.replace_inputs(&mut instr);
            let allow_witness = is_witness_subst_safe_target(&instr);
            substitute_asserted_consts(
                ssa,
                cc,
                fn_type_info,
                fid,
                ProgramPoint::new(bid, i),
                allow_witness,
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
            let decided = cc
                .const_bool_in_block(fid, bid, cond)
                .or_else(|| folded_bools.get(&cond).copied());
            if let Some(b) = decided {
                let target = if b { t } else { f };
                block.set_terminator(Terminator::Jmp(target, vec![]));
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
            // redirect relies on (`AssertCmp{Eq, leader, leader}`) and never helps.
            if matches!(instr, OpCode::Assert { .. } | OpCode::AssertCmp { .. }) {
                continue;
            }
            copy_propagate_asserted_equals(
                cc,
                fn_type_info,
                &replacements,
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
                &replacements,
                fid,
                term_point,
                vals.iter_mut(),
            ),
            Terminator::Jmp(_, params) => copy_propagate_asserted_equals(
                cc,
                fn_type_info,
                &replacements,
                fid,
                term_point,
                params.iter_mut(),
            ),
            Terminator::JmpIf(..) => {}
        }
    }

    replacements.apply_to_function(function, ReplaceScope::Inputs);

    // 9. Hoist the witness-typed asserted constants materialized above into the entry block, whose
    // every instruction it dominates, so each `cast <const> to WitnessOf` definition dominates the
    // uses redirected to it. Emit them in result-id order for a deterministic instruction sequence.
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
}

// UTILITIES
// ================================================================================================

/// Substitute each operand pinned to a constant by an assert at `point` with that constant, in
/// place.
///
/// Sound only locally (at `point`) as the function-wide map must never carry these. It first
/// queries `asserted_const` so the `WitnessOf` type check only runs for original value ids known to
/// the analysis — branch-fact constants introduced by an earlier `replace_inputs` are not in the
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
    witness_casts: &mut HashMap<ValueId, ValueId>,
    inputs: impl Iterator<Item = &'a mut ValueId>,
) {
    for input in inputs {
        if let Some(c) = cc.asserted_const(fid, point, *input) {
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

/// Whether an instruction is a safe target for *witness-typed* asserted-constant substitution: a
/// pure value consumer where redirecting a witness operand to a constant-valued witness changes no
/// witness identity or non-deterministic advice.
///
/// This is an allowlist, so any opcode not named here defaults to *not* substituted. The omitted
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

/// Redirect each operand to the dominance-root-most member of its asserted-equal class at `point`
/// ([`ClickCooper::asserted_leader`]), in place — a local copy-propagation that collapses values an
/// assert proves equal.
///
/// Sound only locally (at `point`): the leader provably equals the operand there and its definition
/// dominates `point`, but the equality does not hold function-wide, so this never goes through the
/// function-wide `replacements` map. The establishing `Assert` / `AssertCmp` is kept (it carries
/// the fact), so the equality survives DCE.
///
/// Three guards keep it sound and non-pessimizing:
///
/// - **A** skips any operand the function-wide map will already rewrite (a folded `Cmp{Eq}` result
///   or `Select` alias that is itself an asserted-equal member): that unconditional fold is
///   strictly stronger than a conditional value redirect. This needs the *complete* map, so the
///   caller runs this only after the fold loop has populated every replacement.
/// - **B** queries `asserted_leader` first, so the type lookups in **C** only run for
///   analysis-known values — an operand turned into a fresh constant earlier is in no equality
///   pair, returns `None`, and never reaches `get_value_type` (which would panic on an unknown id).
/// - **C** redirects only across *equal* types: an `AssertCmp{Eq}`'s operand types are not enforced
///   equal, so redirecting across a mismatch would mistype the IR.
///
/// Unlike [`substitute_asserted_consts`], this needs *no* witness-consumer allowlist
/// ([`is_witness_subst_safe_target`]). That allowlist exists because substituting a witness operand
/// with a *constant* would displace the real witness a witness-machinery consumer must reference —
/// dropping the prover's advice. Here the operand is instead redirected to *another real witness*
/// (the leader) that the kept assert constrains equal and whose definition dominates `point`: this
/// is ordinary copy-propagation between two provably-equal SSA values, which is sound for *any*
/// consumer, witness-machinery included. Guard C confines it to redirects across equal types, so
/// the rewritten consumer still references a witness of the same type and identity class. Hence no
/// consumer is excluded.
fn copy_propagate_asserted_equals<'a>(
    cc: &ClickCooper,
    fn_type_info: &FunctionTypeInfo,
    replacements: &ValueReplacements,
    fid: FunctionId,
    point: ProgramPoint,
    inputs: impl Iterator<Item = &'a mut ValueId>,
) {
    for input in inputs {
        // Guard A: `get_replacement(v) == v` iff `v` is not slated for a function-wide rewrite.
        if replacements.get_replacement(*input) != *input {
            continue;
        }
        // Guard B, then Guard C.
        if let Some(leader) = cc.asserted_leader(fid, point, *input) {
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

// TESTS
// ================================================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::{
        Field,
        analysis::click_cooper::test::run_in_test,
        ssa::hlssa::{BinaryArithOpKind, CastTarget, CmpKind, Constant, SequenceTargetType, Type},
    };

    /// Fold-only entry: runs the propagation rewrite without the integrated DCE, so assertions can
    /// observe the pre-DCE state exactly (the integrated DCE would otherwise prune e.g. the return
    /// slots of these caller-less test entrypoints, sweeping the values feeding them).
    fn fold(ssa: &mut HLSSA) {
        let cc = run_in_test(ssa);
        let flow = FlowAnalysis::run(ssa);
        let type_info = Types::new().run(ssa, &flow);
        propagate_all(ssa, &cc, &type_info);
    }

    /// Full pass including the integrated DCE — for the DCE-integration tests below.
    fn fold_and_dce(ssa: &mut HLSSA) {
        let cc = run_in_test(ssa);
        SCS::new(Config::preserve_blocks()).do_run(ssa, &cc);
    }

    // MIGRATED SCCP TESTS
    // --------------------------------------------------------------------------------------------

    /// `2 + 3 == 5` decides the branch: the comparison chain folds away, the `JmpIf` becomes a
    /// `Jmp` to the then-block, and the else-block is deleted.
    #[test]
    fn folds_constants_and_prunes_dead_branch() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let c2 = ssa.add_const(Constant::U(32, 2));
        let c3 = ssa.add_const(Constant::U(32, 3));
        let c5 = ssa.add_const(Constant::U(32, 5));
        let (sum, is_five) = (ssa.fresh_value(), ssa.fresh_value());

        let f = ssa.get_unique_entrypoint_mut();
        let then_b = f.add_block();
        let else_b = f.add_block();

        let entry = f.get_entry_mut();
        entry.push_instruction(OpCode::BinaryArithOp {
            kind: BinaryArithOpKind::Add,
            result: sum,
            lhs: c2,
            rhs: c3,
        });
        entry.push_instruction(OpCode::Cmp {
            kind: CmpKind::Eq,
            result: is_five,
            lhs: sum,
            rhs: c5,
        });
        entry.set_terminator(Terminator::JmpIf(is_five, then_b, else_b));
        f.get_block_mut(then_b)
            .set_terminator(Terminator::Return(vec![sum]));
        f.get_block_mut(else_b)
            .set_terminator(Terminator::Return(vec![c2]));

        fold(&mut ssa);

        let f = ssa.get_unique_entrypoint();
        // Both instructions folded away.
        assert_eq!(f.get_entry().get_instructions().count(), 0);
        // The branch is decided and the dead block is gone.
        assert!(matches!(
            f.get_entry().get_terminator(),
            Some(Terminator::Jmp(t, args)) if *t == then_b && args.is_empty()
        ));
        assert_eq!(f.get_blocks().count(), 2);
        // The surviving return now names the folded constant.
        assert!(matches!(
            f.get_block(then_b).get_terminator(),
            Some(Terminator::Return(vals)) if *vals == vec![c5]
        ));
    }

    /// A merge parameter receiving the same constant from both arms of an unknown branch folds to
    /// that constant.
    #[test]
    fn folds_phi_of_agreeing_constants() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let c7 = ssa.add_const(Constant::Field(Field::from(7u64)));
        let (cond, m_param) = (ssa.fresh_value(), ssa.fresh_value());

        let f = ssa.get_unique_entrypoint_mut();
        f.get_entry_mut().push_parameter(cond, Type::u(1));
        let a = f.add_block();
        let b = f.add_block();
        let merge = f.add_block();

        f.get_entry_mut()
            .set_terminator(Terminator::JmpIf(cond, a, b));
        f.get_block_mut(a)
            .set_terminator(Terminator::Jmp(merge, vec![c7]));
        f.get_block_mut(b)
            .set_terminator(Terminator::Jmp(merge, vec![c7]));
        let merge_block = f.get_block_mut(merge);
        merge_block.push_parameter(m_param, Type::field());
        merge_block.set_terminator(Terminator::Return(vec![m_param]));

        fold(&mut ssa);

        let f = ssa.get_unique_entrypoint();
        // All blocks reachable (unknown condition), but the merge value is the constant.
        assert_eq!(f.get_blocks().count(), 4);
        assert!(matches!(
            f.get_block(merge).get_terminator(),
            Some(Terminator::Return(vals)) if *vals == vec![c7]
        ));
    }

    /// Constants propagate around a loop with a constant trip decision: the loop body is entered,
    /// but the parameter that bottoms out (varies per iteration) is not folded while the invariant
    /// one is.
    #[test]
    fn loop_variant_value_is_not_folded() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let c0 = ssa.add_const(Constant::U(32, 0));
        let c1 = ssa.add_const(Constant::U(32, 1));
        let c10 = ssa.add_const(Constant::U(32, 10));
        let (i_param, lt, next) = (ssa.fresh_value(), ssa.fresh_value(), ssa.fresh_value());

        let f = ssa.get_unique_entrypoint_mut();
        let header = f.add_block();
        let body = f.add_block();
        let exit = f.add_block();

        f.get_entry_mut()
            .set_terminator(Terminator::Jmp(header, vec![c0]));
        let header_block = f.get_block_mut(header);
        header_block.push_parameter(i_param, Type::u(32));
        header_block.push_instruction(OpCode::Cmp {
            kind: CmpKind::Lt,
            result: lt,
            lhs: i_param,
            rhs: c10,
        });
        header_block.set_terminator(Terminator::JmpIf(lt, body, exit));
        let body_block = f.get_block_mut(body);
        body_block.push_instruction(OpCode::BinaryArithOp {
            kind: BinaryArithOpKind::Add,
            result: next,
            lhs: i_param,
            rhs: c1,
        });
        body_block.set_terminator(Terminator::Jmp(header, vec![next]));
        f.get_block_mut(exit)
            .set_terminator(Terminator::Return(vec![i_param]));

        fold(&mut ssa);

        let f = ssa.get_unique_entrypoint();
        // The loop-carried counter varies: nothing folds, all blocks survive.
        assert_eq!(f.get_blocks().count(), 4);
        assert_eq!(f.get_block(header).get_instructions().count(), 1);
        assert_eq!(f.get_block(body).get_instructions().count(), 1);
    }

    /// Overflowing integer arithmetic must not be folded: the backends' wrap behavior is not
    /// modeled, so the instruction stays.
    #[test]
    fn does_not_fold_overflow() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let c200 = ssa.add_const(Constant::U(8, 200));
        let c100 = ssa.add_const(Constant::U(8, 100));
        let sum = ssa.fresh_value();

        let f = ssa.get_unique_entrypoint_mut();
        let entry = f.get_entry_mut();
        entry.push_instruction(OpCode::BinaryArithOp {
            kind: BinaryArithOpKind::Add,
            result: sum,
            lhs: c200,
            rhs: c100,
        });
        entry.set_terminator(Terminator::Return(vec![sum]));

        fold(&mut ssa);

        let f = ssa.get_unique_entrypoint();
        assert_eq!(f.get_entry().get_instructions().count(), 1);
        assert!(matches!(
            f.get_entry().get_terminator(),
            Some(Terminator::Return(vals)) if *vals == vec![sum]
        ));
    }

    /// A cast into the witness domain is never treated as a constant: the witness chain stays
    /// intact.
    #[test]
    fn does_not_fold_witness_casts() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let c5 = ssa.add_const(Constant::Field(Field::from(5u64)));
        let (wit, doubled) = (ssa.fresh_value(), ssa.fresh_value());

        let f = ssa.get_unique_entrypoint_mut();
        let entry = f.get_entry_mut();
        entry.push_instruction(OpCode::Cast {
            result: wit,
            value: c5,
            target: CastTarget::WitnessOf,
        });
        entry.push_instruction(OpCode::BinaryArithOp {
            kind: BinaryArithOpKind::Add,
            result: doubled,
            lhs: wit,
            rhs: wit,
        });
        entry.set_terminator(Terminator::Return(vec![doubled]));

        fold(&mut ssa);

        let f = ssa.get_unique_entrypoint();
        assert_eq!(f.get_entry().get_instructions().count(), 2);
    }

    /// A degenerate loop whose branch condition converges stuck at ⊤ must ICE rather than let the
    /// rewrite delete blocks the kept `JmpIf` still targets. The ICE is raised by the ClickCooper
    /// solver (`assert_no_stuck_conditions`) while `fold` builds the analysis (`run_in_test`), i.e.
    /// before `propagate_all` runs — well ahead of the integrated DCE.
    #[test]
    #[should_panic(expected = "stuck at ⊤")]
    fn degenerate_loop_ices_on_stuck_condition() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let cond = ssa.fresh_value();

        let f = ssa.get_unique_entrypoint_mut();
        let header = f.add_block();
        let body = f.add_block();
        let exit = f.add_block();

        f.get_entry_mut()
            .set_terminator(Terminator::Jmp(header, vec![]));
        f.get_block_mut(header)
            .set_terminator(Terminator::JmpIf(cond, body, exit));
        let body_block = f.get_block_mut(body);
        body_block.push_parameter(cond, Type::u(1));
        body_block.set_terminator(Terminator::Jmp(header, vec![]));
        f.get_block_mut(exit)
            .set_terminator(Terminator::Return(vec![]));

        fold(&mut ssa);
    }

    /// A select whose condition is constant aliases to the chosen arm even when the arms are not
    /// constants.
    #[test]
    fn select_with_constant_condition_aliases_to_arm() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let c_true = ssa.add_const(Constant::U(1, 1));
        let (arm_t, arm_f, sel) = (ssa.fresh_value(), ssa.fresh_value(), ssa.fresh_value());

        let f = ssa.get_unique_entrypoint_mut();
        f.get_entry_mut().push_parameter(arm_t, Type::field());
        f.get_entry_mut().push_parameter(arm_f, Type::field());
        let entry = f.get_entry_mut();
        entry.push_instruction(OpCode::Select {
            result: sel,
            cond: c_true,
            if_t: arm_t,
            if_f: arm_f,
        });
        entry.set_terminator(Terminator::Return(vec![sel]));

        fold(&mut ssa);

        let f = ssa.get_unique_entrypoint();
        assert_eq!(f.get_entry().get_instructions().count(), 0);
        assert!(matches!(
            f.get_entry().get_terminator(),
            Some(Terminator::Return(vals)) if *vals == vec![arm_t]
        ));
    }

    /// A branch condition is path-sensitive: it is true in blocks reached only through the true
    /// edge, even when it is an unknown function argument globally.
    #[test]
    fn branch_fact_folds_dominated_uses() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let c_false = ssa.add_const(Constant::U(1, 0));
        let (cond, arm_t, arm_f, selected, not_cond) = (
            ssa.fresh_value(),
            ssa.fresh_value(),
            ssa.fresh_value(),
            ssa.fresh_value(),
            ssa.fresh_value(),
        );

        let f = ssa.get_unique_entrypoint_mut();
        let then_b = f.add_block();
        let else_b = f.add_block();
        f.get_entry_mut().push_parameter(cond, Type::u(1));
        f.get_entry_mut().push_parameter(arm_t, Type::field());
        f.get_entry_mut().push_parameter(arm_f, Type::field());
        f.get_entry_mut()
            .set_terminator(Terminator::JmpIf(cond, then_b, else_b));

        let then_block = f.get_block_mut(then_b);
        then_block.push_instruction(OpCode::Select {
            result: selected,
            cond,
            if_t: arm_t,
            if_f: arm_f,
        });
        then_block.push_instruction(OpCode::Not {
            result: not_cond,
            value: cond,
        });
        then_block.set_terminator(Terminator::Return(vec![selected, not_cond]));
        f.get_block_mut(else_b)
            .set_terminator(Terminator::Return(vec![arm_f, c_false]));

        fold(&mut ssa);

        let f = ssa.get_unique_entrypoint();
        assert_eq!(f.get_block(then_b).get_instructions().count(), 0);
        assert!(matches!(
            f.get_block(then_b).get_terminator(),
            Some(Terminator::Return(vals)) if *vals == vec![arm_t, c_false]
        ));
        assert!(matches!(
            f.get_entry().get_terminator(),
            Some(Terminator::JmpIf(c, t, e)) if *c == cond && *t == then_b && *e == else_b
        ));
    }

    /// Conflicting incoming predicate facts disappear at a join.
    #[test]
    fn branch_fact_does_not_cross_conflicting_merge() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let (cond, not_cond) = (ssa.fresh_value(), ssa.fresh_value());

        let f = ssa.get_unique_entrypoint_mut();
        let then_b = f.add_block();
        let else_b = f.add_block();
        let merge = f.add_block();
        f.get_entry_mut().push_parameter(cond, Type::u(1));
        f.get_entry_mut()
            .set_terminator(Terminator::JmpIf(cond, then_b, else_b));
        f.get_block_mut(then_b)
            .set_terminator(Terminator::Jmp(merge, vec![]));
        f.get_block_mut(else_b)
            .set_terminator(Terminator::Jmp(merge, vec![]));
        let merge_block = f.get_block_mut(merge);
        merge_block.push_instruction(OpCode::Not {
            result: not_cond,
            value: cond,
        });
        merge_block.set_terminator(Terminator::Return(vec![not_cond]));

        fold(&mut ssa);

        let f = ssa.get_unique_entrypoint();
        assert_eq!(f.get_block(merge).get_instructions().count(), 1);
        assert!(matches!(
            f.get_block(merge).get_terminator(),
            Some(Terminator::Return(vals)) if *vals == vec![not_cond]
        ));
    }

    /// A nested branch on a condition already known in the block is decided immediately.
    #[test]
    fn branch_fact_prunes_nested_same_condition_branch() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let cond = ssa.fresh_value();

        let f = ssa.get_unique_entrypoint_mut();
        let then_b = f.add_block();
        let else_b = f.add_block();
        let nested_then = f.add_block();
        let nested_else = f.add_block();
        f.get_entry_mut().push_parameter(cond, Type::u(1));
        f.get_entry_mut()
            .set_terminator(Terminator::JmpIf(cond, then_b, else_b));
        f.get_block_mut(then_b)
            .set_terminator(Terminator::JmpIf(cond, nested_then, nested_else));
        f.get_block_mut(else_b)
            .set_terminator(Terminator::Return(vec![]));
        f.get_block_mut(nested_then)
            .set_terminator(Terminator::Return(vec![]));
        f.get_block_mut(nested_else)
            .set_terminator(Terminator::Return(vec![]));

        fold(&mut ssa);

        let f = ssa.get_unique_entrypoint();
        // `nested_else` is unreachable once the nested branch is decided, and is pruned (block
        // preservation only keeps blocks the analysis still reaches).
        assert_eq!(f.get_blocks().count(), 4);
        assert!(matches!(
            f.get_block(then_b).get_terminator(),
            Some(Terminator::Jmp(t, args)) if *t == nested_then && args.is_empty()
        ));
    }

    /// If both edges go to the same block, the edge itself carries no useful predicate fact.
    #[test]
    fn branch_with_same_successor_does_not_infer_condition() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let (cond, not_cond) = (ssa.fresh_value(), ssa.fresh_value());

        let f = ssa.get_unique_entrypoint_mut();
        let join = f.add_block();
        f.get_entry_mut().push_parameter(cond, Type::u(1));
        f.get_entry_mut()
            .set_terminator(Terminator::JmpIf(cond, join, join));
        let join_block = f.get_block_mut(join);
        join_block.push_instruction(OpCode::Not {
            result: not_cond,
            value: cond,
        });
        join_block.set_terminator(Terminator::Return(vec![not_cond]));

        fold(&mut ssa);

        let f = ssa.get_unique_entrypoint();
        assert_eq!(f.get_block(join).get_instructions().count(), 1);
        assert!(matches!(
            f.get_block(join).get_terminator(),
            Some(Terminator::Return(vals)) if *vals == vec![not_cond]
        ));
    }

    /// Jump arguments are evaluated under the branch facts of the edge carrying them.
    #[test]
    fn branch_fact_folds_phi_from_one_edge() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let c_true = ssa.add_const(Constant::U(1, 1));
        let (cond, phi) = (ssa.fresh_value(), ssa.fresh_value());

        let f = ssa.get_unique_entrypoint_mut();
        let then_b = f.add_block();
        let else_b = f.add_block();
        let merge = f.add_block();
        f.get_entry_mut().push_parameter(cond, Type::u(1));
        f.get_entry_mut()
            .set_terminator(Terminator::JmpIf(cond, then_b, else_b));
        f.get_block_mut(then_b)
            .set_terminator(Terminator::Jmp(merge, vec![cond]));
        f.get_block_mut(else_b)
            .set_terminator(Terminator::Return(vec![]));
        let merge_block = f.get_block_mut(merge);
        merge_block.push_parameter(phi, Type::u(1));
        merge_block.set_terminator(Terminator::Return(vec![phi]));

        fold(&mut ssa);

        let f = ssa.get_unique_entrypoint();
        assert!(matches!(
            f.get_block(then_b).get_terminator(),
            Some(Terminator::Jmp(t, args)) if *t == merge && *args == vec![c_true]
        ));
        assert!(matches!(
            f.get_block(merge).get_terminator(),
            Some(Terminator::Return(vals)) if *vals == vec![c_true]
        ));
    }

    /// A branch decided by *congruence* — `CmpEq(a, b)` with `a` and `b` structurally equal but not
    /// constant — folds away through the combined-fixpoint writeback: the comparison is dropped, the
    /// `JmpIf` becomes a `Jmp` to the then-block, and the dead else-block is deleted. The two
    /// (non-constant) adds survive, and the purity assert does not trip on the folded `Cmp`.
    #[test]
    fn folds_congruence_decided_branch() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let c1 = ssa.add_const(Constant::U(32, 1));
        let (x, a, b, eq) = (
            ssa.fresh_value(),
            ssa.fresh_value(),
            ssa.fresh_value(),
            ssa.fresh_value(),
        );

        let f = ssa.get_unique_entrypoint_mut();
        let then_b = f.add_block();
        let else_b = f.add_block();
        f.get_entry_mut().push_parameter(x, Type::u(32));
        let entry = f.get_entry_mut();
        entry.push_instruction(OpCode::BinaryArithOp {
            kind: BinaryArithOpKind::Add,
            result: a,
            lhs: x,
            rhs: c1,
        });
        entry.push_instruction(OpCode::BinaryArithOp {
            kind: BinaryArithOpKind::Add,
            result: b,
            lhs: x,
            rhs: c1,
        });
        entry.push_instruction(OpCode::Cmp {
            kind: CmpKind::Eq,
            result: eq,
            lhs: a,
            rhs: b,
        });
        entry.set_terminator(Terminator::JmpIf(eq, then_b, else_b));
        f.get_block_mut(then_b)
            .set_terminator(Terminator::Return(vec![a]));
        f.get_block_mut(else_b)
            .set_terminator(Terminator::Return(vec![b]));

        fold(&mut ssa);

        let f = ssa.get_unique_entrypoint();
        // The two adds survive (not constant); only the `Cmp` folded away.
        assert_eq!(f.get_entry().get_instructions().count(), 2);
        // The branch is decided and the dead else-block is gone.
        assert!(matches!(
            f.get_entry().get_terminator(),
            Some(Terminator::Jmp(t, args)) if *t == then_b && args.is_empty()
        ));
        assert_eq!(f.get_blocks().count(), 2);
    }

    /// A literal `AssertCmp{Eq, v, v}` is a tautology and is dropped (it would otherwise lower to a
    /// wasted `Constrain` row).
    #[test]
    fn redundant_self_equality_assert_is_dropped() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let a = ssa.fresh_value();

        let f = ssa.get_unique_entrypoint_mut();
        f.get_entry_mut().push_parameter(a, Type::field());
        let entry = f.get_entry_mut();
        entry.push_instruction(OpCode::AssertCmp {
            kind: CmpKind::Eq,
            lhs: a,
            rhs: a,
        });
        entry.set_terminator(Terminator::Return(vec![]));

        fold(&mut ssa);

        let f = ssa.get_unique_entrypoint();
        assert!(
            !f.get_entry()
                .get_instructions()
                .any(|i| matches!(i, OpCode::AssertCmp { .. })),
            "a tautological `AssertCmp{{Eq, v, v}}` should be dropped"
        );
    }

    /// An `AssertCmp{Eq, a, b}` whose operands are proven equal by *congruence* (two identical
    /// computations) is redundant and dropped — but the computations themselves survive.
    #[test]
    fn redundant_congruent_equality_assert_is_dropped() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let c1 = ssa.add_const(Constant::U(32, 1));
        let (x, a, b) = (ssa.fresh_value(), ssa.fresh_value(), ssa.fresh_value());

        let f = ssa.get_unique_entrypoint_mut();
        f.get_entry_mut().push_parameter(x, Type::u(32));
        let entry = f.get_entry_mut();
        // Two identical adds: `a` and `b` are congruent (`known_equal`), independent of any assert.
        entry.push_instruction(OpCode::BinaryArithOp {
            kind: BinaryArithOpKind::Add,
            result: a,
            lhs: x,
            rhs: c1,
        });
        entry.push_instruction(OpCode::BinaryArithOp {
            kind: BinaryArithOpKind::Add,
            result: b,
            lhs: x,
            rhs: c1,
        });
        entry.push_instruction(OpCode::AssertCmp {
            kind: CmpKind::Eq,
            lhs: a,
            rhs: b,
        });
        entry.set_terminator(Terminator::Return(vec![a, b]));

        fold(&mut ssa);

        let f = ssa.get_unique_entrypoint();
        assert!(
            !f.get_entry()
                .get_instructions()
                .any(|i| matches!(i, OpCode::AssertCmp { .. })),
            "an `AssertCmp{{Eq}}` on congruent operands should be dropped"
        );
        // The two (non-constant) adds are not removed by the congruence-gated drop.
        assert_eq!(
            f.get_entry()
                .get_instructions()
                .filter(|i| matches!(i, OpCode::BinaryArithOp { .. }))
                .count(),
            2
        );
    }

    /// Guard: an `AssertCmp{Eq, a, b}` on *non*-congruent operands (`known_equal` false) is
    /// load-bearing and must be kept — dropping it would weaken the constraint system.
    #[test]
    fn non_congruent_equality_assert_is_kept() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let (a, b) = (ssa.fresh_value(), ssa.fresh_value());

        let f = ssa.get_unique_entrypoint_mut();
        f.get_entry_mut().push_parameter(a, Type::field());
        f.get_entry_mut().push_parameter(b, Type::field());
        let entry = f.get_entry_mut();
        entry.push_instruction(OpCode::AssertCmp {
            kind: CmpKind::Eq,
            lhs: a,
            rhs: b,
        });
        entry.set_terminator(Terminator::Return(vec![]));

        fold(&mut ssa);

        let f = ssa.get_unique_entrypoint();
        assert!(
            f.get_entry().get_instructions().any(|i| matches!(
                i,
                OpCode::AssertCmp { kind: CmpKind::Eq, lhs, rhs } if *lhs == a && *rhs == b
            )),
            "a non-congruent equality assert must be kept (it is load-bearing)"
        );
    }

    /// A *witnessed* comparison of congruent operands folds to a constant, but keeps its `WitnessOf`
    /// type: it is redefined in place as a `cast <const> to WitnessOf` rather than aliased to a bare
    /// constant, so the witnessed return slot stays correctly typed.
    #[test]
    fn witnessed_constant_is_cast_to_witness_of() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let (w, ww) = (ssa.fresh_value(), ssa.fresh_value());

        let f = ssa.get_unique_entrypoint_mut();
        let entry = f.get_entry_mut();
        // An operand is `WitnessOf`, so the comparison result is `WitnessOf(u1)`.
        entry.push_parameter(w, Type::witness_of(Type::u(32)));
        entry.push_instruction(OpCode::Cmp {
            kind: CmpKind::Eq,
            result: ww,
            lhs: w,
            rhs: w,
        });
        entry.set_terminator(Terminator::Return(vec![ww]));

        fold(&mut ssa);

        let f = ssa.get_unique_entrypoint();
        assert!(
            !f.get_entry()
                .get_instructions()
                .any(|i| matches!(i, OpCode::Cmp { .. })),
            "the witnessed comparison should have folded away"
        );
        assert!(
            f.get_entry().get_instructions().any(|i| matches!(
                i,
                OpCode::Cast { result, target: CastTarget::WitnessOf, .. } if *result == ww
            )),
            "ww should be redefined as a cast to WitnessOf"
        );
        assert!(matches!(
            f.get_entry().get_terminator(),
            Some(Terminator::Return(vals)) if vals.as_slice() == [ww]
        ));
    }

    /// A constant lookup-table projection folds: `ArrayGet`/`SliceLen` over a constant `MkSeq` are
    /// dropped and their uses aliased to the interned scalar constant, while the (never-surfaced)
    /// aggregate `MkSeq` itself is left in place by the rewrite (the integrated DCE sweeps it — see
    /// `integrated_dce_sweeps_dead_aggregate`).
    #[test]
    fn folds_constant_aggregate_projections() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let c10 = ssa.add_const(Constant::U(32, 10));
        let c20 = ssa.add_const(Constant::U(32, 20));
        let c30 = ssa.add_const(Constant::U(32, 30));
        let idx = ssa.add_const(Constant::U(32, 1));
        let c_len = ssa.add_const(Constant::U(32, 3));
        let (seq, got, len) = (ssa.fresh_value(), ssa.fresh_value(), ssa.fresh_value());

        let entry = ssa.get_unique_entrypoint_mut().get_entry_mut();
        entry.push_instruction(OpCode::MkSeq {
            result: seq,
            elems: vec![c10, c20, c30],
            seq_type: SequenceTargetType::Array(3),
            elem_type: Type::u(32),
        });
        entry.push_instruction(OpCode::ArrayGet {
            result: got,
            array: seq,
            index: idx,
        });
        entry.push_instruction(OpCode::SliceLen {
            result: len,
            slice: seq,
        });
        entry.set_terminator(Terminator::Return(vec![got, len]));

        fold(&mut ssa);

        let f = ssa.get_unique_entrypoint();
        // Both projections folded away; only the (internal, now-dead) aggregate constructor remains.
        assert!(
            !f.get_entry()
                .get_instructions()
                .any(|i| matches!(i, OpCode::ArrayGet { .. } | OpCode::SliceLen { .. })),
            "the constant projections should have folded away"
        );
        assert_eq!(f.get_entry().get_instructions().count(), 1); // the surviving MkSeq
        // Their uses are aliased to the interned scalars: element 1 == 20, length == 3.
        assert!(matches!(
            f.get_entry().get_terminator(),
            Some(Terminator::Return(vals)) if vals.as_slice() == [c20, c_len]
        ));
    }

    // CONDITIONAL-LAYER TESTS
    // --------------------------------------------------------------------------------------------

    /// A value pinned to a constant by a dominating `Assert{v}` is substituted at later uses (here,
    /// the return), while the establishing assert is kept.
    #[test]
    fn asserted_const_substituted_after_assert() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let c_true = ssa.add_const(Constant::U(1, 1));
        let b = ssa.fresh_value();

        let f = ssa.get_unique_entrypoint_mut();
        f.get_entry_mut().push_parameter(b, Type::u(1));
        let entry = f.get_entry_mut();
        entry.push_instruction(OpCode::Assert { value: b });
        entry.set_terminator(Terminator::Return(vec![b]));

        fold(&mut ssa);

        let f = ssa.get_unique_entrypoint();
        // The assert survives (it establishes the fact), and the return now names the constant.
        assert!(
            f.get_entry()
                .get_instructions()
                .any(|i| matches!(i, OpCode::Assert { .. })),
            "the establishing assert must survive"
        );
        assert!(matches!(
            f.get_entry().get_terminator(),
            Some(Terminator::Return(vals)) if *vals == vec![c_true]
        ));
    }

    /// `AssertCmp{Eq, x, c}` pins `x` to the constant `c`; later uses of `x` fold to `c`.
    #[test]
    fn asserted_const_substituted_after_assert_cmp_eq() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let c5 = ssa.add_const(Constant::Field(Field::from(5u64)));
        let x = ssa.fresh_value();

        let f = ssa.get_unique_entrypoint_mut();
        f.get_entry_mut().push_parameter(x, Type::field());
        let entry = f.get_entry_mut();
        entry.push_instruction(OpCode::AssertCmp {
            kind: CmpKind::Eq,
            lhs: x,
            rhs: c5,
        });
        entry.set_terminator(Terminator::Return(vec![x]));

        fold(&mut ssa);

        let f = ssa.get_unique_entrypoint();
        assert!(matches!(
            f.get_entry().get_terminator(),
            Some(Terminator::Return(vals)) if *vals == vec![c5]
        ));
    }

    /// Index granularity within the asserting block: a use *before* the assert is not folded (the
    /// fact does not yet hold), a use *after* it is. This is what keeps an assert from folding its
    /// own operand into a tautology.
    #[test]
    fn same_block_assert_is_index_granular() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let c_true = ssa.add_const(Constant::U(1, 1));
        let (b, before, after) = (ssa.fresh_value(), ssa.fresh_value(), ssa.fresh_value());

        let f = ssa.get_unique_entrypoint_mut();
        f.get_entry_mut().push_parameter(b, Type::u(1));
        let entry = f.get_entry_mut();
        entry.push_instruction(OpCode::Not {
            result: before,
            value: b,
        });
        entry.push_instruction(OpCode::Assert { value: b });
        entry.push_instruction(OpCode::Not {
            result: after,
            value: b,
        });
        entry.set_terminator(Terminator::Return(vec![before, after]));

        fold(&mut ssa);

        let f = ssa.get_unique_entrypoint();
        let nots: Vec<(ValueId, ValueId)> = f
            .get_entry()
            .get_instructions()
            .filter_map(|i| match i {
                OpCode::Not { result, value } => Some((*result, *value)),
                _ => None,
            })
            .collect();
        let before_input = nots.iter().find(|(r, _)| *r == before).unwrap().1;
        let after_input = nots.iter().find(|(r, _)| *r == after).unwrap().1;
        // The use before the assert still reads `b`; the use after reads the asserted constant.
        assert_eq!(before_input, b);
        assert_eq!(after_input, c_true);
    }

    /// On the false edge of an equality branch, a re-test `Cmp{Eq, a, b}` folds to `false` via the
    /// known disequality.
    #[test]
    fn known_unequal_folds_cmp_eq_to_false() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let c_false = ssa.add_const(Constant::U(1, 0));
        let (a, b, eq, eq2) = (
            ssa.fresh_value(),
            ssa.fresh_value(),
            ssa.fresh_value(),
            ssa.fresh_value(),
        );

        let f = ssa.get_unique_entrypoint_mut();
        let then_b = f.add_block();
        let else_b = f.add_block();
        f.get_entry_mut().push_parameter(a, Type::field());
        f.get_entry_mut().push_parameter(b, Type::field());
        f.get_entry_mut().push_instruction(OpCode::Cmp {
            kind: CmpKind::Eq,
            result: eq,
            lhs: a,
            rhs: b,
        });
        f.get_entry_mut()
            .set_terminator(Terminator::JmpIf(eq, then_b, else_b));
        f.get_block_mut(then_b)
            .set_terminator(Terminator::Return(vec![]));
        let else_block = f.get_block_mut(else_b);
        else_block.push_instruction(OpCode::Cmp {
            kind: CmpKind::Eq,
            result: eq2,
            lhs: a,
            rhs: b,
        });
        else_block.set_terminator(Terminator::Return(vec![eq2]));

        fold(&mut ssa);

        let f = ssa.get_unique_entrypoint();
        // The re-test folded away; its use is the constant false.
        assert_eq!(f.get_block(else_b).get_instructions().count(), 0);
        assert!(matches!(
            f.get_block(else_b).get_terminator(),
            Some(Terminator::Return(vals)) if *vals == vec![c_false]
        ));
        // The establishing equality branch survives unchanged.
        assert!(matches!(
            f.get_entry().get_terminator(),
            Some(Terminator::JmpIf(c, _, _)) if *c == eq
        ));
    }

    /// Fold + prune: when a conditionally-folded `Cmp{Eq}` result is the block's `JmpIf` condition,
    /// the terminator is folded to a `Jmp` on the taken target — never left as a constant-fed
    /// `JmpIf` (the dead edge is pruned; the now-orphaned target is cleaned up downstream).
    #[test]
    fn known_unequal_folds_and_prunes_nested_jmpif() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let c1 = ssa.add_const(Constant::U(32, 1));
        let c2 = ssa.add_const(Constant::U(32, 2));
        let (a, b, eq, eq2) = (
            ssa.fresh_value(),
            ssa.fresh_value(),
            ssa.fresh_value(),
            ssa.fresh_value(),
        );

        let f = ssa.get_unique_entrypoint_mut();
        let then_b = f.add_block();
        let else_b = f.add_block();
        let x_b = f.add_block();
        let y_b = f.add_block();
        f.get_entry_mut().push_parameter(a, Type::field());
        f.get_entry_mut().push_parameter(b, Type::field());
        f.get_entry_mut().push_instruction(OpCode::Cmp {
            kind: CmpKind::Eq,
            result: eq,
            lhs: a,
            rhs: b,
        });
        f.get_entry_mut()
            .set_terminator(Terminator::JmpIf(eq, then_b, else_b));
        f.get_block_mut(then_b)
            .set_terminator(Terminator::Return(vec![]));
        let else_block = f.get_block_mut(else_b);
        else_block.push_instruction(OpCode::Cmp {
            kind: CmpKind::Eq,
            result: eq2,
            lhs: a,
            rhs: b,
        });
        else_block.set_terminator(Terminator::JmpIf(eq2, x_b, y_b));
        f.get_block_mut(x_b)
            .set_terminator(Terminator::Return(vec![c1]));
        f.get_block_mut(y_b)
            .set_terminator(Terminator::Return(vec![c2]));

        fold(&mut ssa);

        let f = ssa.get_unique_entrypoint();
        // `eq2` is `false` on this edge, so the nested branch is folded to a `Jmp` on the false
        // target — not left as a `JmpIf` on a constant condition.
        assert_eq!(f.get_block(else_b).get_instructions().count(), 0);
        assert!(matches!(
            f.get_block(else_b).get_terminator(),
            Some(Terminator::Jmp(t, args)) if *t == y_b && args.is_empty()
        ));
    }

    /// A `Cmp{Eq, a, b}` whose operands are proven equal by an earlier same-block `AssertCmp{Eq}`
    /// folds to `true` via `asserted_equal` — the conditional-fact mirror of
    /// `known_unequal_folds_cmp_eq_to_false`. The establishing assert is kept (it carries the
    /// fact), and neither operand is constant, so the equality is conditional rather than a
    /// constant pin.
    #[test]
    fn asserted_equal_folds_cmp_eq_to_true() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let c_true = ssa.add_const(Constant::U(1, 1));
        let (a, b, eq) = (ssa.fresh_value(), ssa.fresh_value(), ssa.fresh_value());

        let f = ssa.get_unique_entrypoint_mut();
        f.get_entry_mut().push_parameter(a, Type::field());
        f.get_entry_mut().push_parameter(b, Type::field());
        let entry = f.get_entry_mut();
        entry.push_instruction(OpCode::AssertCmp {
            kind: CmpKind::Eq,
            lhs: a,
            rhs: b,
        });
        entry.push_instruction(OpCode::Cmp {
            kind: CmpKind::Eq,
            result: eq,
            lhs: a,
            rhs: b,
        });
        entry.set_terminator(Terminator::Return(vec![eq]));

        fold(&mut ssa);

        let f = ssa.get_unique_entrypoint();
        // The re-test folded away; its use is the constant true.
        assert!(
            !f.get_entry()
                .get_instructions()
                .any(|i| matches!(i, OpCode::Cmp { .. })),
            "the asserted equality should have folded the re-test away"
        );
        // The establishing equality assert survives.
        assert!(
            f.get_entry().get_instructions().any(|i| matches!(
                i,
                OpCode::AssertCmp {
                    kind: CmpKind::Eq,
                    ..
                }
            )),
            "the establishing AssertCmp must survive"
        );
        assert!(matches!(
            f.get_entry().get_terminator(),
            Some(Terminator::Return(vals)) if *vals == vec![c_true]
        ));
    }

    /// The witness-typed analog of `asserted_equal_folds_cmp_eq_to_true`: when the conditionally
    /// folded `Cmp{Eq}` result is `WitnessOf`-typed it is redefined as a `Cast` to `WitnessOf`
    /// (keeping the IR well-typed) rather than aliased to a bare constant. The operands are two
    /// distinct witnesses, so the comparison is not congruence-decidable and only folds conditionally.
    #[test]
    fn witnessed_cmp_eq_folded_conditionally_is_cast_to_witness_of() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let (a, b, ww) = (ssa.fresh_value(), ssa.fresh_value(), ssa.fresh_value());

        let f = ssa.get_unique_entrypoint_mut();
        f.get_entry_mut()
            .push_parameter(a, Type::witness_of(Type::field()));
        f.get_entry_mut()
            .push_parameter(b, Type::witness_of(Type::field()));
        let entry = f.get_entry_mut();
        entry.push_instruction(OpCode::AssertCmp {
            kind: CmpKind::Eq,
            lhs: a,
            rhs: b,
        });
        // A witness operand makes the result `WitnessOf(u1)`.
        entry.push_instruction(OpCode::Cmp {
            kind: CmpKind::Eq,
            result: ww,
            lhs: a,
            rhs: b,
        });
        entry.set_terminator(Terminator::Return(vec![ww]));

        fold(&mut ssa);

        let f = ssa.get_unique_entrypoint();
        assert!(
            !f.get_entry()
                .get_instructions()
                .any(|i| matches!(i, OpCode::Cmp { .. })),
            "the witnessed comparison should have folded away"
        );
        assert!(
            f.get_entry().get_instructions().any(|i| matches!(
                i,
                OpCode::Cast { result, target: CastTarget::WitnessOf, .. } if *result == ww
            )),
            "the conditionally folded witnessed result should be recast to WitnessOf"
        );
        assert!(matches!(
            f.get_entry().get_terminator(),
            Some(Terminator::Return(vals)) if vals.as_slice() == [ww]
        ));
    }

    /// A select whose condition is a constant `false` aliases to the else arm — symmetric to
    /// `select_with_constant_condition_aliases_to_arm`, which covers the true arm.
    #[test]
    fn select_with_constant_false_condition_aliases_to_else_arm() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let c_false = ssa.add_const(Constant::U(1, 0));
        let (arm_t, arm_f, sel) = (ssa.fresh_value(), ssa.fresh_value(), ssa.fresh_value());

        let f = ssa.get_unique_entrypoint_mut();
        f.get_entry_mut().push_parameter(arm_t, Type::field());
        f.get_entry_mut().push_parameter(arm_f, Type::field());
        let entry = f.get_entry_mut();
        entry.push_instruction(OpCode::Select {
            result: sel,
            cond: c_false,
            if_t: arm_t,
            if_f: arm_f,
        });
        entry.set_terminator(Terminator::Return(vec![sel]));

        fold(&mut ssa);

        let f = ssa.get_unique_entrypoint();
        assert_eq!(f.get_entry().get_instructions().count(), 0);
        assert!(matches!(
            f.get_entry().get_terminator(),
            Some(Terminator::Return(vals)) if *vals == vec![arm_f]
        ));
    }

    /// Per-point asserted-constant substitution reaches `Jmp` arguments, not only `Return` values:
    /// a value pinned by a dominating `Assert` is substituted where it is passed as a block
    /// argument.
    #[test]
    fn asserted_const_substituted_in_jmp_args() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let c_true = ssa.add_const(Constant::U(1, 1));
        let (b, p) = (ssa.fresh_value(), ssa.fresh_value());

        let f = ssa.get_unique_entrypoint_mut();
        let target = f.add_block();
        f.get_entry_mut().push_parameter(b, Type::u(1));
        let entry = f.get_entry_mut();
        entry.push_instruction(OpCode::Assert { value: b });
        entry.set_terminator(Terminator::Jmp(target, vec![b]));
        let target_block = f.get_block_mut(target);
        target_block.push_parameter(p, Type::u(1));
        target_block.set_terminator(Terminator::Return(vec![]));

        fold(&mut ssa);

        let f = ssa.get_unique_entrypoint();
        // The asserted constant is substituted into the jump argument.
        assert!(matches!(
            f.get_entry().get_terminator(),
            Some(Terminator::Jmp(t, args)) if *t == target && args.as_slice() == [c_true]
        ));
    }

    /// A `WitnessOf`-typed operand pinned to a constant by an `Assert`, used by a *pure value*
    /// consumer (here the return), is substituted with a witness-typed constant: a
    /// `cast <const> to WitnessOf` is hoisted into the entry block and the use is redirected to it,
    /// keeping the IR well-typed. The establishing assert is retained.
    #[test]
    fn asserted_const_substituted_for_witness_operand_via_cast() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let w = ssa.fresh_value();

        let f = ssa.get_unique_entrypoint_mut();
        f.get_entry_mut()
            .push_parameter(w, Type::witness_of(Type::u(1)));
        let entry = f.get_entry_mut();
        entry.push_instruction(OpCode::Assert { value: w });
        entry.set_terminator(Terminator::Return(vec![w]));

        fold(&mut ssa);

        let f = ssa.get_unique_entrypoint();
        let entry = f.get_entry();

        // A `cast <const> to WitnessOf` was hoisted to the front of the entry block.
        let cast_result = entry
            .get_instructions()
            .find_map(|i| match i {
                OpCode::Cast {
                    result,
                    target: CastTarget::WitnessOf,
                    ..
                } => Some(*result),
                _ => None,
            })
            .expect("a witness cast should be hoisted into the entry block");

        // The witness-typed return operand is redirected to that witness-typed constant, not `w`.
        assert!(matches!(
            entry.get_terminator(),
            Some(Terminator::Return(vals)) if *vals == vec![cast_result]
        ));

        // The establishing assert is retained (it carries the fact into R1CS).
        assert!(
            entry
                .get_instructions()
                .any(|i| matches!(i, OpCode::Assert { value } if *value == w)),
            "the establishing assert must survive"
        );
    }

    /// A `WitnessOf`-typed operand pinned to a constant by an `Assert` is *not* substituted when its
    /// consumer is an excluded witness-machinery op (here `WriteWitness`, the honest hint): such a
    /// consumer must keep referencing the real witness, so neither the operand is redirected nor a
    /// witness cast is materialized.
    #[test]
    fn asserted_const_not_substituted_for_excluded_witness_consumer() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let w = ssa.fresh_value();

        let f = ssa.get_unique_entrypoint_mut();
        f.get_entry_mut()
            .push_parameter(w, Type::witness_of(Type::u(1)));
        let entry = f.get_entry_mut();
        entry.push_instruction(OpCode::Assert { value: w });
        entry.push_instruction(OpCode::WriteWitness {
            result: None,
            value: w,
            pinned: false,
        });
        entry.set_terminator(Terminator::Return(vec![]));

        fold(&mut ssa);

        let f = ssa.get_unique_entrypoint();
        let entry = f.get_entry();
        // The excluded consumer still references the original witness `w`.
        assert!(
            entry
                .get_instructions()
                .any(|i| matches!(i, OpCode::WriteWitness { value, .. } if *value == w)),
            "the excluded consumer must keep referencing the real witness"
        );
        // No witness cast was hoisted, since the only candidate use was excluded.
        assert!(
            !entry.get_instructions().any(|i| matches!(
                i,
                OpCode::Cast {
                    target: CastTarget::WitnessOf,
                    ..
                }
            )),
            "no witness cast should be materialized for an excluded consumer"
        );
    }

    /// An operand proven equal to a dominating value by an `AssertCmp{Eq}` is redirected to that
    /// value (the class's dominance-root-most member): the return's `b` becomes the param `a`,
    /// whose definition dominates `b = a + a`. The establishing assert is kept.
    #[test]
    fn asserted_equal_copy_propagates_to_dominating_leader() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let (a, b) = (ssa.fresh_value(), ssa.fresh_value());

        let f = ssa.get_unique_entrypoint_mut();
        f.get_entry_mut().push_parameter(a, Type::field());
        let entry = f.get_entry_mut();
        entry.push_instruction(OpCode::BinaryArithOp {
            kind: BinaryArithOpKind::Add,
            result: b,
            lhs: a,
            rhs: a,
        });
        entry.push_instruction(OpCode::AssertCmp {
            kind: CmpKind::Eq,
            lhs: a,
            rhs: b,
        });
        entry.set_terminator(Terminator::Return(vec![b]));

        fold(&mut ssa);

        let f = ssa.get_unique_entrypoint();
        // `b` is redirected to its dominating asserted-equal representative `a`.
        assert!(matches!(
            f.get_entry().get_terminator(),
            Some(Terminator::Return(vals)) if *vals == vec![a]
        ));
        // The establishing equality assert survives (it carries the fact).
        assert!(
            f.get_entry().get_instructions().any(|i| matches!(
                i,
                OpCode::AssertCmp {
                    kind: CmpKind::Eq,
                    ..
                }
            )),
            "the establishing AssertCmp must survive"
        );
    }

    /// Index granularity: a use of a member *before* the establishing same-block assert is not
    /// redirected (the equality does not yet hold), a use *after* it is — the copy-prop mirror of
    /// `same_block_assert_is_index_granular`.
    #[test]
    fn asserted_equal_copy_prop_is_index_granular() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let c0 = ssa.add_const(Constant::U(32, 0));
        let (a, b, before, after) = (
            ssa.fresh_value(),
            ssa.fresh_value(),
            ssa.fresh_value(),
            ssa.fresh_value(),
        );

        let f = ssa.get_unique_entrypoint_mut();
        f.get_entry_mut().push_parameter(a, Type::u(32));
        f.get_entry_mut().push_parameter(b, Type::u(32));
        let entry = f.get_entry_mut();
        entry.push_instruction(OpCode::BinaryArithOp {
            kind: BinaryArithOpKind::Add,
            result: before,
            lhs: b,
            rhs: c0,
        });
        entry.push_instruction(OpCode::AssertCmp {
            kind: CmpKind::Eq,
            lhs: a,
            rhs: b,
        });
        entry.push_instruction(OpCode::BinaryArithOp {
            kind: BinaryArithOpKind::Add,
            result: after,
            lhs: b,
            rhs: c0,
        });
        entry.set_terminator(Terminator::Return(vec![before, after]));

        fold(&mut ssa);

        let f = ssa.get_unique_entrypoint();
        let adds: Vec<(ValueId, ValueId)> = f
            .get_entry()
            .get_instructions()
            .filter_map(|i| match i {
                OpCode::BinaryArithOp {
                    kind: BinaryArithOpKind::Add,
                    result,
                    lhs,
                    ..
                } => Some((*result, *lhs)),
                _ => None,
            })
            .collect();
        let before_lhs = adds.iter().find(|(r, _)| *r == before).unwrap().1;
        let after_lhs = adds.iter().find(|(r, _)| *r == after).unwrap().1;
        // The use before the assert still reads `b`; the use after reads the leader `a`.
        assert_eq!(before_lhs, b);
        assert_eq!(after_lhs, a);
    }

    /// Guard C: the leader is value-id/structure based, so a redirect across a type mismatch is
    /// suppressed (an `AssertCmp{Eq}`'s operand types are not enforced equal). `y: u8` is not
    /// redirected to `x: u32`, keeping the IR well-typed.
    #[test]
    fn asserted_equal_copy_prop_skips_type_mismatch() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let (x, y) = (ssa.fresh_value(), ssa.fresh_value());

        let f = ssa.get_unique_entrypoint_mut();
        f.get_entry_mut().push_parameter(x, Type::u(32));
        f.get_entry_mut().push_parameter(y, Type::u(8));
        let entry = f.get_entry_mut();
        entry.push_instruction(OpCode::AssertCmp {
            kind: CmpKind::Eq,
            lhs: x,
            rhs: y,
        });
        entry.set_terminator(Terminator::Return(vec![y]));

        fold(&mut ssa);

        let f = ssa.get_unique_entrypoint();
        assert!(matches!(
            f.get_entry().get_terminator(),
            Some(Terminator::Return(vals)) if *vals == vec![y]
        ));
    }

    /// The class is transitive: `c == b == a`, so a use of `c` redirects all the way to `a`, even
    /// though no single `AssertCmp` names both `c` and `a`.
    #[test]
    fn asserted_equal_copy_prop_is_transitive() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let (a, b, c) = (ssa.fresh_value(), ssa.fresh_value(), ssa.fresh_value());

        let f = ssa.get_unique_entrypoint_mut();
        f.get_entry_mut().push_parameter(a, Type::field());
        f.get_entry_mut().push_parameter(b, Type::field());
        f.get_entry_mut().push_parameter(c, Type::field());
        let entry = f.get_entry_mut();
        entry.push_instruction(OpCode::AssertCmp {
            kind: CmpKind::Eq,
            lhs: a,
            rhs: b,
        });
        entry.push_instruction(OpCode::AssertCmp {
            kind: CmpKind::Eq,
            lhs: b,
            rhs: c,
        });
        entry.set_terminator(Terminator::Return(vec![c]));

        fold(&mut ssa);

        let f = ssa.get_unique_entrypoint();
        assert!(matches!(
            f.get_entry().get_terminator(),
            Some(Terminator::Return(vals)) if *vals == vec![a]
        ));
    }

    /// The redirect reaches `Jmp` block-arguments, not only `Return` values — the second of the two
    /// terminator call sites.
    #[test]
    fn asserted_equal_copy_prop_reaches_jmp_args() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let (a, b, p) = (ssa.fresh_value(), ssa.fresh_value(), ssa.fresh_value());

        let f = ssa.get_unique_entrypoint_mut();
        let target = f.add_block();
        f.get_entry_mut().push_parameter(a, Type::field());
        f.get_entry_mut().push_parameter(b, Type::field());
        let entry = f.get_entry_mut();
        entry.push_instruction(OpCode::AssertCmp {
            kind: CmpKind::Eq,
            lhs: a,
            rhs: b,
        });
        entry.set_terminator(Terminator::Jmp(target, vec![b]));
        let target_block = f.get_block_mut(target);
        target_block.push_parameter(p, Type::field());
        target_block.set_terminator(Terminator::Return(vec![]));

        fold(&mut ssa);

        let f = ssa.get_unique_entrypoint();
        assert!(matches!(
            f.get_entry().get_terminator(),
            Some(Terminator::Jmp(t, args)) if *t == target && args.as_slice() == [a]
        ));
    }

    /// Guard A: a function-wide fold beats a conditional redirect. `v = Cmp{Eq, x, y}` folds to the
    /// constant `true` (via the dominating `AssertCmp{Eq, x, y}`), and is *also* placed in an
    /// asserted-equal class with `w`. The return's `v` must become the constant `true` — not its
    /// class-mate `w` — because the unconditional fold is strictly stronger. A plain
    /// unconditionally-constant operand would not exercise this: an equality pair is only filed
    /// when *both* sides are non-constant, so it would already yield no leader.
    #[test]
    fn asserted_equal_copy_prop_yields_to_function_wide_fold() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let c_true = ssa.add_const(Constant::U(1, 1));
        let (x, y, w, v) = (
            ssa.fresh_value(),
            ssa.fresh_value(),
            ssa.fresh_value(),
            ssa.fresh_value(),
        );

        let f = ssa.get_unique_entrypoint_mut();
        f.get_entry_mut().push_parameter(x, Type::field());
        f.get_entry_mut().push_parameter(y, Type::field());
        f.get_entry_mut().push_parameter(w, Type::u(1));
        let entry = f.get_entry_mut();
        // x == y pins the comparison `v` to `true` at later points.
        entry.push_instruction(OpCode::AssertCmp {
            kind: CmpKind::Eq,
            lhs: x,
            rhs: y,
        });
        entry.push_instruction(OpCode::Cmp {
            kind: CmpKind::Eq,
            result: v,
            lhs: x,
            rhs: y,
        });
        // `v` is also placed in an asserted-equal class with `w`.
        entry.push_instruction(OpCode::AssertCmp {
            kind: CmpKind::Eq,
            lhs: v,
            rhs: w,
        });
        entry.set_terminator(Terminator::Return(vec![v]));

        fold(&mut ssa);

        let f = ssa.get_unique_entrypoint();
        // The constant fold of `v` wins; `v` is not redirected to `w`.
        assert!(matches!(
            f.get_entry().get_terminator(),
            Some(Terminator::Return(vals)) if *vals == vec![c_true]
        ));
    }

    /// Equal *witness* types copy-propagate: unlike bare-constant substitution there is no
    /// `is_witness_of` skip, because Guard C already proves the redirect type-safe (both operands
    /// are the same `WitnessOf` type).
    #[test]
    fn asserted_equal_copy_prop_redirects_equal_witnesses() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let (wa, wb) = (ssa.fresh_value(), ssa.fresh_value());

        let f = ssa.get_unique_entrypoint_mut();
        f.get_entry_mut()
            .push_parameter(wa, Type::witness_of(Type::field()));
        f.get_entry_mut()
            .push_parameter(wb, Type::witness_of(Type::field()));
        let entry = f.get_entry_mut();
        entry.push_instruction(OpCode::AssertCmp {
            kind: CmpKind::Eq,
            lhs: wa,
            rhs: wb,
        });
        entry.set_terminator(Terminator::Return(vec![wb]));

        fold(&mut ssa);

        let f = ssa.get_unique_entrypoint();
        assert!(matches!(
            f.get_entry().get_terminator(),
            Some(Terminator::Return(vals)) if *vals == vec![wa]
        ));
    }

    /// Copy-propagation redirects a witness operand of a *witness-machinery* consumer (here
    /// `WriteWitness`, the honest hint) to its asserted-equal leader — it has no
    /// `is_witness_subst_safe_target` allowlist, unlike `substitute_asserted_consts`. This is sound
    /// because the operand is redirected to *another real witness* the kept assert constrains equal
    /// (not displaced by a constant, which is why the *constant* substitution excludes this very
    /// consumer — see `asserted_const_not_substituted_for_excluded_witness_consumer`).
    #[test]
    fn asserted_equal_copy_prop_redirects_witness_machinery_consumer() {
        let mut ssa = HLSSA::with_main("main".to_string());
        // `wa` first ⇒ `wa.0 < wb.0`; both are entry params (same def site), so the dominance-root-
        // most leader is the smaller-id `wa`.
        let (wa, wb) = (ssa.fresh_value(), ssa.fresh_value());

        let f = ssa.get_unique_entrypoint_mut();
        f.get_entry_mut()
            .push_parameter(wa, Type::witness_of(Type::field()));
        f.get_entry_mut()
            .push_parameter(wb, Type::witness_of(Type::field()));
        let entry = f.get_entry_mut();
        entry.push_instruction(OpCode::AssertCmp {
            kind: CmpKind::Eq,
            lhs: wa,
            rhs: wb,
        });
        // A witness-machinery consumer of `wb`: its `value` operand is exposed via `get_inputs_mut`
        // (`Lookup`/`DLookup` args are not, so copy-prop cannot reach those).
        entry.push_instruction(OpCode::WriteWitness {
            result: None,
            value: wb,
            pinned: false,
        });
        entry.set_terminator(Terminator::Return(vec![]));

        fold(&mut ssa);

        let f = ssa.get_unique_entrypoint();
        // The `WriteWitness` value operand is redirected from `wb` to the leader `wa`.
        assert!(
            f.get_entry()
                .get_instructions()
                .any(|i| matches!(i, OpCode::WriteWitness { value, .. } if *value == wa)),
            "the witness-machinery consumer should be redirected to the asserted-equal leader"
        );
        // The establishing assert survives (it carries the equality the redirect relies on).
        assert!(
            f.get_entry().get_instructions().any(|i| matches!(
                i,
                OpCode::AssertCmp {
                    kind: CmpKind::Eq,
                    ..
                }
            )),
            "the establishing AssertCmp must survive"
        );
    }

    /// The establishing assert (and the `b = a + a` def it pins live) survive the integrated DCE
    /// even after the return's `b` is redirected to `a` — so the equality the redirect relied on is
    /// never broken. `AssertCmp` is unconditionally live in DCE and marks its operands live, which is
    /// what keeps `b`'s def alive; this guards against a regression that stopped doing so.
    #[test]
    fn asserted_equal_copy_prop_keeps_establisher_under_dce() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let (a, b) = (ssa.fresh_value(), ssa.fresh_value());

        let f = ssa.get_unique_entrypoint_mut();
        f.get_entry_mut().push_parameter(a, Type::field());
        let entry = f.get_entry_mut();
        entry.push_instruction(OpCode::BinaryArithOp {
            kind: BinaryArithOpKind::Add,
            result: b,
            lhs: a,
            rhs: a,
        });
        entry.push_instruction(OpCode::AssertCmp {
            kind: CmpKind::Eq,
            lhs: a,
            rhs: b,
        });
        entry.set_terminator(Terminator::Return(vec![b]));

        fold_and_dce(&mut ssa);

        let f = ssa.get_unique_entrypoint();
        assert!(
            f.get_entry().get_instructions().any(|i| matches!(
                i,
                OpCode::AssertCmp {
                    kind: CmpKind::Eq,
                    ..
                }
            )),
            "the establishing AssertCmp must survive DCE"
        );
        assert!(
            f.get_entry()
                .get_instructions()
                .any(|i| matches!(i, OpCode::BinaryArithOp { result, .. } if *result == b)),
            "the `b` def stays live, pinned live by the assert"
        );
    }

    /// The redundant-assert drop (4b) and copy-propagation cooperate in one pass: an
    /// `AssertCmp{Eq, a, b}` on *congruent* operands is dropped (4b, `known_equal`), yet the
    /// second-walk copy-prop still redirects a use of `b` to the leader `a` — because
    /// `asserted_leader` is read from the analysis built on the pristine IR, which still has the
    /// assert. This is sound precisely because the drop is congruence-backed: `a` and `b` compute
    /// the same thing (`x + 1`), so `a == b` holds in every witness independently of the dropped
    /// assert, and the redirect is well-typed (both `u32`). Regression guard for the non-local
    /// interaction. (Fold-only: the integrated DCE would prune this caller-less entrypoint's return
    /// slot and sweep everything — sound, but it would hide the redirect.)
    #[test]
    fn congruence_dropped_assert_still_copy_propagates_soundly() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let c1 = ssa.add_const(Constant::U(32, 1));
        let (x, a, b) = (ssa.fresh_value(), ssa.fresh_value(), ssa.fresh_value());

        let f = ssa.get_unique_entrypoint_mut();
        f.get_entry_mut().push_parameter(x, Type::u(32));
        let entry = f.get_entry_mut();
        // Two identical adds ⇒ `a` and `b` are congruent (`known_equal`), `a` defined first.
        entry.push_instruction(OpCode::BinaryArithOp {
            kind: BinaryArithOpKind::Add,
            result: a,
            lhs: x,
            rhs: c1,
        });
        entry.push_instruction(OpCode::BinaryArithOp {
            kind: BinaryArithOpKind::Add,
            result: b,
            lhs: x,
            rhs: c1,
        });
        // Congruence-redundant: 4b drops this in the first walk.
        entry.push_instruction(OpCode::AssertCmp {
            kind: CmpKind::Eq,
            lhs: a,
            rhs: b,
        });
        // A use of `b` the second-walk copy-prop redirects to the leader `a`.
        entry.set_terminator(Terminator::Return(vec![b]));

        fold(&mut ssa);

        let f = ssa.get_unique_entrypoint();
        // 4b dropped the redundant assert.
        assert!(
            !f.get_entry()
                .get_instructions()
                .any(|i| matches!(i, OpCode::AssertCmp { .. })),
            "the congruence-redundant assert should be dropped"
        );
        // The use of `b` was redirected to the leader `a`, even though the connecting assert was
        // dropped in the same pass.
        assert!(matches!(
            f.get_entry().get_terminator(),
            Some(Terminator::Return(vals)) if *vals == vec![a]
        ));
        // `a`'s def survives (it now feeds the return) and is the well-typed `u32` add.
        assert!(
            f.get_entry()
                .get_instructions()
                .any(|i| matches!(i, OpCode::BinaryArithOp { result, .. } if *result == a)),
            "the leader's def must survive to feed the redirected use"
        );
    }

    /// The integrated DCE runs in the same pass: dead code (here, an unused instruction) is removed
    /// without a separate cleanup pass. Fold-only would leave the add (it is not constant); the
    /// full pass sweeps it.
    #[test]
    fn integrated_dce_removes_dead_code() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let c1 = ssa.add_const(Constant::U(32, 1));
        let (x, unused) = (ssa.fresh_value(), ssa.fresh_value());

        let f = ssa.get_unique_entrypoint_mut();
        f.get_entry_mut().push_parameter(x, Type::u(32));
        let entry = f.get_entry_mut();
        entry.push_instruction(OpCode::BinaryArithOp {
            kind: BinaryArithOpKind::Add,
            result: unused,
            lhs: x,
            rhs: c1,
        });
        entry.set_terminator(Terminator::Return(vec![x]));

        fold_and_dce(&mut ssa);

        let f = ssa.get_unique_entrypoint();
        // The unused add is swept by the integrated DCE.
        assert!(
            !f.get_entry()
                .get_instructions()
                .any(|i| matches!(i, OpCode::BinaryArithOp { .. })),
            "the unused add should be removed by the integrated DCE"
        );
    }

    /// A fold that *creates* dead code: once the constant aggregate projections fold away, the
    /// (never-surfaced) `MkSeq` is dead, and the integrated DCE sweeps it in the same pass —
    /// contrast `folds_constant_aggregate_projections`, where fold-only leaves it.
    #[test]
    fn integrated_dce_sweeps_dead_aggregate() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let c10 = ssa.add_const(Constant::U(32, 10));
        let c20 = ssa.add_const(Constant::U(32, 20));
        let c30 = ssa.add_const(Constant::U(32, 30));
        let idx = ssa.add_const(Constant::U(32, 1));
        let (seq, got, len) = (ssa.fresh_value(), ssa.fresh_value(), ssa.fresh_value());

        let entry = ssa.get_unique_entrypoint_mut().get_entry_mut();
        entry.push_instruction(OpCode::MkSeq {
            result: seq,
            elems: vec![c10, c20, c30],
            seq_type: SequenceTargetType::Array(3),
            elem_type: Type::u(32),
        });
        entry.push_instruction(OpCode::ArrayGet {
            result: got,
            array: seq,
            index: idx,
        });
        entry.push_instruction(OpCode::SliceLen {
            result: len,
            slice: seq,
        });
        entry.set_terminator(Terminator::Return(vec![got, len]));

        fold_and_dce(&mut ssa);

        let f = ssa.get_unique_entrypoint();
        // The projections fold and the now-dead aggregate is swept: nothing remains.
        assert_eq!(f.get_entry().get_instructions().count(), 0);
    }

    /// Regression guard for the transient orphan a conditional `Cmp{Eq}`->`JmpIf` fold leaves
    /// behind. `eq2 = (a == b)` is proven only by the dominating `AssertCmp{Eq, a, b}` — a
    /// *conditional* fold (`a`, `b` are distinct params, never congruent/constant), so the analysis
    /// never pruned the branch and step 1 cannot delete the block it strands. Folding
    /// `JmpIf(eq2, then_b, else_b)` to `Jmp(then_b)` therefore orphans `else_b` together with its
    /// constraint-bearing `AssertCmp{Eq, p, q}`, which the integrated DCE keeps under
    /// `preserve_blocks()` — so after one pass the orphan and its constraint both survive.
    ///
    /// The fold is nonetheless sound because the orphan is *reclaimed by the next ClickCooper-driven
    /// pass*: it is genuinely unreachable there, so that pass's step 1 deletes it. This test pins
    /// that reclamation — the invariant the downstream passes silently rely on between here and the
    /// next SCS run (WTI types every value, so untaint never ICEs on the orphan; R1CS codegen is
    /// execution-driven, so it never emits the orphan's constraint).
    #[test]
    fn conditional_jmpif_fold_orphan_is_reclaimed_by_next_pass() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let (a, b, p, q, eq2) = (
            ssa.fresh_value(),
            ssa.fresh_value(),
            ssa.fresh_value(),
            ssa.fresh_value(),
            ssa.fresh_value(),
        );

        let f = ssa.get_unique_entrypoint_mut();
        let then_b = f.add_block();
        let else_b = f.add_block();
        f.get_entry_mut().push_parameter(a, Type::field());
        f.get_entry_mut().push_parameter(b, Type::field());
        f.get_entry_mut().push_parameter(p, Type::field());
        f.get_entry_mut().push_parameter(q, Type::field());
        let entry = f.get_entry_mut();
        // Dominating assert establishes `a == b` conditionally (distinct params, so never a
        // congruence/constant fact); the re-test below folds to `true` only via `asserted_equal`.
        entry.push_instruction(OpCode::AssertCmp {
            kind: CmpKind::Eq,
            lhs: a,
            rhs: b,
        });
        entry.push_instruction(OpCode::Cmp {
            kind: CmpKind::Eq,
            result: eq2,
            lhs: a,
            rhs: b,
        });
        entry.set_terminator(Terminator::JmpIf(eq2, then_b, else_b));
        f.get_block_mut(then_b)
            .set_terminator(Terminator::Return(vec![]));
        // `else_b`'s sole predecessor is the entry, so folding the branch orphans it. Its distinct,
        // load-bearing constraint must not leak into the circuit.
        let else_block = f.get_block_mut(else_b);
        else_block.push_instruction(OpCode::AssertCmp {
            kind: CmpKind::Eq,
            lhs: p,
            rhs: q,
        });
        else_block.set_terminator(Terminator::Return(vec![]));

        // The orphan's constraint, found anywhere in the function.
        let orphan_constraint_present = |s: &HLSSA| {
            s.get_unique_entrypoint().get_blocks().any(|(_, block)| {
                block.get_instructions().any(|i| {
                    matches!(
                        i,
                        OpCode::AssertCmp { kind: CmpKind::Eq, lhs, rhs } if *lhs == p && *rhs == q
                    )
                })
            })
        };

        // Pass 1: the conditional fold fires, but leaves the orphan (and its constraint) in place.
        let cc = run_in_test(&ssa);
        SCS::new(Config::preserve_blocks()).do_run(&mut ssa, &cc);
        assert!(
            matches!(
                ssa.get_unique_entrypoint().get_entry().get_terminator(),
                Some(Terminator::Jmp(t, args)) if *t == then_b && args.is_empty()
            ),
            "the conditional `Cmp{{Eq}}`->`JmpIf` fold should rewrite the branch to a `Jmp`"
        );
        assert!(
            orphan_constraint_present(&ssa),
            "under preserve_blocks the orphaned `else_b` and its constraint survive one pass"
        );

        // Pass 2: a fresh analysis sees `else_b` as unreachable, so step 1 reclaims it — the
        // constraint never survives into a subsequent pass.
        let cc = run_in_test(&ssa);
        SCS::new(Config::preserve_blocks()).do_run(&mut ssa, &cc);
        assert!(
            !orphan_constraint_present(&ssa),
            "the orphaned block and its constraint must be reclaimed by the next ClickCooper pass"
        );
    }
}
