//! A Click–Cooper-style combined optimistic analysis.
//!
//! An analysis that handles **constants**, **reachability** and **congruence** (using optimistic
//! AWZ value numbering). It is designed to drive constant/condition propagation and PRE; a
//! context-sensitive interprocedural layer is planned (see
//! [Deferred Improvements](#deferred-improvements)). The combination is _at least_ as precise as
//! running the factors separately and alternating to a fixpoint (Click & Cooper, _Combining
//! Analyses, Combining Optimizations_, TOPLAS 1995).
//!
//! It operates over two kinds of facts:
//!
//! - **Intraprocedural facts** are those within a given function and are derived from a
//!   Wegman-Zadeck constants and reachability fixpoint with a structural-congruence partition over
//!   the reachability state.
//! - **Conditional facts** are ones derived from asserts, equalities, disequalities, and unpinned
//!   witness forwarding. They are computed post-convergence over the same reachability state plus
//!   GFG dominance, and are intentionally disjoint from the unconditional view.
//!
//! **Unconditional facts** are those which hold on any path so that any use (replacing a use,
//! deleting a pure definition, or pruning an unreachable block) maintains soundness. **Conditional
//! facts** are those established through control flow (branches, assertions, witness operations)
//! that are only correct to use in contexts where the establishing constraints are preserved.
//!
//! # Correctness
//!
//! This analysis is **sound** because each fact class is sound, with the reasoning given as
//! follows:
//!
//! - **Constants:** The transfer functions fold a value only when the result is exact for the
//!   operand widths so `value == c` holds under any advice.
//! - **Reachability:** An edge is executable only when a predecessor's terminator can take it. A
//!   block proven unreachable is thus taken in no run.
//! - **Congruence:** Two values are congruent iff computed by the same operator from operand-wise
//!   congruent inputs, hence equal in all runs. Congruence never asserts an equality an adversarial
//!   witness could validate.
//! - **Conditional Facts:** Assert-derived constants, equalities, and disequalities hold only on
//!   accepting runs that preserve their establishing constraint, so they are exposed only through
//!   dedicated queries. An assert, being a global constraint that holds at every point of an
//!   accepting run, is *also* attributed to in-scope blocks it post-dominates (the assert is then
//!   guaranteed on every continuation). Control-flow derived equality and disequality facts, being
//!   path-conditional, stay dominance-only.
//! - **Witness Forwarding:** Both readings of the one witness↔value correspondence — the
//!   `WriteWitness` hint (`r = witness_of(v)`) and the `ValueOf` projection (`v = value_of(r)`) —
//!   only _add_ constraints and hence never reject an honest run, while two free witnesses are
//!   never unified (the relation is keyed by the witness, and its members are honest values).
//!
//! This analysis **terminates** because every factor has finite height and every loop is monotone,
//! as follows:
//!
//! - **Constants:** The lattice has height 2 (`Top ⊐ Const ⊐ Bottom`) and `set_lattice` only
//!   lowers, so each value changes at most twice.
//! - **Reachability:** `exec_edges` and `reachable` grow monotonically within the finite CFG;
//!   per-block branch facts only ever shrink at joins. The two-worklist loop drains a finite number
//!   of edge/value events.
//! - **Congruence:** Partition refinement only ever *splits* classes, and the class count is
//!   bounded by the value count, so it stabilises in at most `|values|` rounds.
//!
//! # Deferred Improvements
//!
//! The following are improvements planned for the future of this analysis.
//!
//! - **Combined-Fixpoint Writebacks:** Congruence runs as a single pass over the converged
//!   reachability rather than interleaved with the constant worklist, because the reverse couplings
//!   are not wired: a congruence class with a `Const` member does not promote its peers into the
//!   (SCCP-visible) constant lattice, and a branch condition congruent to a constant does not fold
//!   an edge. These pay off only in a consumer that exploits the full combined fixpoint; until such
//!   a consumer exists, omitting them keeps SCCP's output unchanged. A future consumer can recover
//!   the promotion by composing [`ClickCooper::known_equal`] with [`ClickCooper::const_of`].
//! - **Assert Facts at Index Granularity:** Assert-derived facts are attributed at block-entry
//!   granularity via *strict* dominance/post-dominance, so a use in the asserting block itself
//!   (after the assert) is not yet claimed. Index-precise program points would recover it; soundness
//!   is unaffected.
//! - **Dominance-Aware Congruence Leader:** [`ClickCooper::leader`] returns the smallest-id class
//!   member, which is not necessarily a dominating definition; a redirecting consumer (CSE/PRE)
//!   needs the dominating member, which requires threading `FlowAnalysis` dominance into the
//!   congruence partition.
//! - **Interprocedural (1-CFA) Layer:** A context-sensitive interprocedural layer — polymorphic
//!   jump-function summaries (a `return_const` jump function holding for any arguments, a
//!   `return Param(i)` pass-through equal to argument `i`) plus 1-CFA per-`(function, k-limited
//!   call-string)` specialization, mirroring the two-phase structure of `analysis/points_to` — was
//!   prototyped and then removed pending a consumer that exploits it. SCCP, the only current
//!   consumer, reads intraprocedural facts only, so the layer was deferred to keep the analysis
//!   lean; it can be reinstated (along with its `*_in` context-keyed queries) with its first
//!   interprocedural consumer. The deferred layer also did not propagate cross-call structural
//!   congruence (e.g. `f(x); f(y)` with `x ≡ y` yielding congruent results).

mod conditional;
mod congruence;
mod lattice;
mod solver;

use std::sync::Arc;

use crate::{
    collections::HashMap,
    compiler::{
        analysis::{
            click_cooper::{
                conditional::ConditionalFacts,
                lattice::{Constness, bool_constant},
                solver::{FunctionFacts, FunctionSolver},
            },
            flow_analysis::FlowAnalysis,
            types::TypeInfo,
        },
        pass_manager::{Analysis, AnalysisId, AnalysisStore},
        ssa::{
            BlockId, FunctionId, ValueId,
            hlssa::{Constant, HLSSA, HLSSAConstantsSnapshot},
        },
    },
};

// CLICK COOPER ANALYSIS
// ================================================================================================

/// The Click-Cooper analysis pass.
///
/// See the module documentation for more details.
#[derive(Debug)]
pub struct ClickCooper {
    /// Program-wide interned-constant snapshot.
    consts: HLSSAConstantsSnapshot,

    /// Per-function **intraprocedural** converged facts (parameters and call results `Bottom`).
    functions: HashMap<FunctionId, FunctionFacts>,

    /// Per-function **conditional** side facts (assert-derived constants/equalities, branch
    /// disequalities, unpinned-witness forwarding).
    ///
    /// Held in their own fields read by their own queries, entirely disjoint from the unconditional
    /// view above.
    conditional: HashMap<FunctionId, ConditionalFacts>,
}

impl Analysis for ClickCooper {
    fn dependencies() -> Vec<AnalysisId> {
        vec![FlowAnalysis::id(), TypeInfo::id()]
    }

    fn compute(ssa: &HLSSA, store: &AnalysisStore) -> Self {
        ClickCooper::run(ssa, store.get::<FlowAnalysis>(), store.get::<TypeInfo>())
    }
}

impl ClickCooper {
    fn run(ssa: &HLSSA, flow: &FlowAnalysis, types: &TypeInfo) -> Self {
        // One snapshot serves all functions: a constant referenced from a function is always
        // interned program-wide.
        let consts = ssa.const_snapshot();

        // Per-function intraprocedural facts: parameters and call results `Bottom`. The conditional
        // side facts are computed from the same converged state plus CFG dominance, kept in a
        // disjoint map so the unconditional view is untouched.
        let mut functions = HashMap::default();
        let mut conditional = HashMap::default();
        for fid in ssa.get_function_ids() {
            let function = ssa.get_function(fid);
            let mut solver = FunctionSolver::new(function, &consts);
            solver.run();
            let facts = solver.into_facts();
            let cond = conditional::build(
                function,
                &facts,
                flow.get_function_cfg(fid),
                &consts,
                types.get_function(fid),
            );
            conditional.insert(fid, cond);
            functions.insert(fid, facts);
        }

        ClickCooper {
            consts,
            functions,
            conditional,
        }
    }
}

/// Unconditional queries — facts that hold in **every** run.
///
/// Using them to replace uses, delete pure defs, or prune unreachable blocks is accept-set
/// preserving. This impl block contains both the intraprocedural and interprocedural queries.
impl ClickCooper {
    /// The constant `v` provably holds in `f` in *every* run, or `None`.
    ///
    /// Interned constants and the global constant lattice only; path-sensitive branch facts are
    /// excluded.
    pub fn const_of(&self, f: FunctionId, v: ValueId) -> Option<Arc<Constant>> {
        Self::const_in_facts(&self.consts, self.functions.get(&f), v)
    }

    /// `true` if `bid` reachable in `f`.
    pub fn is_reachable(&self, f: FunctionId, bid: BlockId) -> bool {
        self.functions
            .get(&f)
            .is_some_and(|facts| facts.reachable.contains(&bid))
    }

    /// `true` if the CFG edge `from -> to` proven executable in `f`.
    pub fn is_executable_edge(&self, f: FunctionId, from: BlockId, to: BlockId) -> bool {
        self.functions
            .get(&f)
            .is_some_and(|facts| facts.exec_edges.contains(&(from, to)))
    }

    /// Every value proven (unconditionally) constant in `f`, as `(value, constant)` pairs.
    ///
    /// Excludes already-interned constant values and conditional branch facts.
    pub fn new_const_values(&self, f: FunctionId) -> Vec<(ValueId, Arc<Constant>)> {
        let Some(facts) = self.functions.get(&f) else {
            return Vec::new();
        };
        facts
            .lattice
            .iter()
            .filter_map(|(v, e)| match e {
                Constness::Const(c) => Some((*v, c.clone())),
                Constness::Top | Constness::Bottom => None,
            })
            .collect()
    }

    /// `true` if `a` and `b` are proven *structurally* congruent in `f`.
    pub fn known_equal(&self, f: FunctionId, a: ValueId, b: ValueId) -> bool {
        self.functions
            .get(&f)
            .is_some_and(|facts| facts.congruence.known_equal(a, b))
    }

    /// Every value structurally congruent to `v` in `f` (including `v`), sorted by value id.
    pub fn congruence_class(&self, f: FunctionId, v: ValueId) -> Vec<ValueId> {
        self.functions
            .get(&f)
            .map(|facts| facts.congruence.class_members(v))
            .unwrap_or_default()
    }

    /// A deterministic representative of `v`'s congruence class in `f` (smallest member by value
    /// id).
    ///
    /// Not yet guaranteed to dominate the class, so it is not a legal redirect target on its own;
    /// the dominance-aware leader requires threading the (already-available) `FlowAnalysis`
    /// dominance into the congruence partition.
    pub fn leader(&self, f: FunctionId, v: ValueId) -> Option<ValueId> {
        self.functions
            .get(&f)
            .and_then(|facts| facts.congruence.leader(v))
    }
}

/// Conditional queries — facts established by a branch or assert that hold only downstream of the
/// establishing control flow.
///
/// These let Mavros be stricter on adversarial inputs, so they are sound *only* under
/// constraint-preserving use: a local, structure-preserving rewrite that keeps the establishing
/// branch/assert (never folding away the constraint that proves the fact).
impl ClickCooper {
    /// The constant `v` holds on entry to `bid` in `f`, honoring path-sensitive branch facts.
    pub fn const_in_block(&self, f: FunctionId, bid: BlockId, v: ValueId) -> Option<Arc<Constant>> {
        if let Some(c) = self.consts.get(&v) {
            return Some(c.clone());
        }
        let facts = self.functions.get(&f)?;
        if let Some(known) = facts.block_facts.get(&bid).and_then(|m| m.get(&v)) {
            return Some(bool_constant(*known));
        }
        Self::const_in_facts(&self.consts, Some(facts), v)
    }

    /// `Some(...)` if `v` known true/false on entry to `bid` in `f` (a branch predicate fact).
    pub fn bool_fact(&self, f: FunctionId, bid: BlockId, v: ValueId) -> Option<bool> {
        self.functions
            .get(&f)?
            .block_facts
            .get(&bid)?
            .get(&v)
            .copied()
    }

    /// If `v` is an (in-block) constant boolean, get its value, honoring branch facts, or `None`
    /// otherwise.
    pub fn const_bool_in_block(&self, f: FunctionId, bid: BlockId, v: ValueId) -> Option<bool> {
        self.const_in_block(f, bid, v)
            .and_then(|c| lattice::const_bool(&c))
    }

    /// The branch predicate facts at entry to `bid` in `f`, as `(value, bool-constant)` pairs
    /// sorted by value id.
    pub fn block_bool_facts(&self, f: FunctionId, bid: BlockId) -> Vec<(ValueId, Arc<Constant>)> {
        let Some(facts) = self.functions.get(&f) else {
            return Vec::new();
        };
        let Some(m) = facts.block_facts.get(&bid) else {
            return Vec::new();
        };
        let mut out: Vec<(ValueId, Arc<Constant>)> =
            m.iter().map(|(v, b)| (*v, bool_constant(*b))).collect();
        out.sort_by_key(|(v, _)| v.0);
        out
    }

    /// The constant `v` is pinned to on entry to `bid` in `f` by a *dominating assert*
    /// (`Assert{v}`⇒`true`, or `AssertCmp{Eq, v, c}`).
    ///
    /// A conditional fact, deliberately disjoint from [`Self::const_of`] as a
    /// consumer must keep the establishing assert.
    pub fn asserted_const(&self, f: FunctionId, bid: BlockId, v: ValueId) -> Option<Arc<Constant>> {
        self.conditional.get(&f)?.asserted_const(bid, v)
    }

    /// `true` if a dominating `AssertCmp{Eq}` proves `a == b` on entry to `bid` in `f`.
    ///
    /// The conditional analog of [`Self::known_equal`] (structural congruence): sound only for
    /// constraint-preserving use, so kept out of the unconditional partition.
    pub fn asserted_equal(&self, f: FunctionId, bid: BlockId, a: ValueId, b: ValueId) -> bool {
        self.conditional
            .get(&f)
            .is_some_and(|c| c.asserted_equal(bid, a, b))
    }

    /// `true` if `a` and `b` are proven unequal on entry to `bid` in `f` (the false edge of an
    /// equality branch).
    ///
    /// Note that this only accounts for _disequality_ `a != b`, and not expanded linear
    /// inequalities.
    pub fn known_unequal(&self, f: FunctionId, bid: BlockId, a: ValueId, b: ValueId) -> bool {
        self.conditional
            .get(&f)
            .is_some_and(|c| c.known_disequal(bid, a, b))
    }

    /// The honest values the free witness handle `r` is equal to in `f`, containing no duplicates.
    ///
    ///
    /// It is drawn from both readings of one correspondence: the unpinned
    /// `WriteWitness{result: Some(r), value, pinned: false}` hint (`r = witness_of(value)`) and
    /// every `Cast{value: r, target: ValueOf}` read (`result = value_of(r)`).
    ///
    /// A sound *redirect* fact (each member adds a functional constraint ⇒ `A_M ⊆ A_N`, never
    /// rejecting the honest run), not a structural congruence — so it lives here, never in
    /// [`Self::known_equal`], and two free witnesses are never unified.
    pub fn witness_forward(&self, f: FunctionId, r: ValueId) -> &[ValueId] {
        self.conditional
            .get(&f)
            .map(|c| c.witness_forward(r))
            .unwrap_or_default()
    }
}

/// Private query helper shared by [`Self::const_of`] and [`Self::const_in_block`].
impl ClickCooper {
    /// The constant `v` holds in `facts`, or `None`: interned constants first, then the constant
    /// lattice.
    fn const_in_facts(
        consts: &HLSSAConstantsSnapshot,
        facts: Option<&FunctionFacts>,
        v: ValueId,
    ) -> Option<Arc<Constant>> {
        if let Some(c) = consts.get(&v) {
            return Some(c.clone());
        }
        match facts?.lattice.get(&v) {
            Some(Constness::Const(c)) => Some(c.clone()),
            Some(Constness::Top) | Some(Constness::Bottom) | None => None,
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::{ClickCooper, FlowAnalysis};
    use crate::compiler::{
        analysis::types::Types,
        ssa::{
            Terminator,
            hlssa::{BinaryArithOpKind, CastTarget, CmpKind, Constant, HLSSA, OpCode, Type},
        },
    };

    /// Build the analysis for `ssa` with freshly-computed dependencies, for test use only.
    pub(crate) fn run_in_test(ssa: &HLSSA) -> ClickCooper {
        let flow = FlowAnalysis::run(ssa);
        let types = Types::new().run(ssa, &flow);
        ClickCooper::run(ssa, &flow, &types)
    }

    /// Two values that fold to the *same* constant are congruent, even when computed differently —
    /// the const → congruence coupling.
    #[test]
    fn equal_constants_are_congruent() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let c1 = ssa.add_const(Constant::U(32, 1));
        let c2 = ssa.add_const(Constant::U(32, 2));
        let c3 = ssa.add_const(Constant::U(32, 3));
        let c4 = ssa.add_const(Constant::U(32, 4));
        let (a, b) = (ssa.fresh_value(), ssa.fresh_value());

        let f = ssa.get_unique_entrypoint_mut();
        let entry = f.get_entry_mut();
        entry.push_instruction(OpCode::BinaryArithOp {
            kind: BinaryArithOpKind::Add,
            result: a,
            lhs: c2,
            rhs: c3,
        });
        entry.push_instruction(OpCode::BinaryArithOp {
            kind: BinaryArithOpKind::Add,
            result: b,
            lhs: c1,
            rhs: c4,
        });
        entry.set_terminator(Terminator::Return(vec![a, b]));

        let fid = ssa.get_unique_entrypoint_id();
        let cc = run_in_test(&ssa);
        // 2+3 and 1+4 both equal 5.
        assert!(cc.known_equal(fid, a, b));
        // But not congruent to a different constant.
        assert!(!cc.known_equal(fid, a, c2));
    }

    /// The same expression over the same operands is congruent, commutatively; different operators
    /// or operands are not.
    #[test]
    fn structural_congruence_is_commutative() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let (x, y) = (ssa.fresh_value(), ssa.fresh_value());
        let (a, b, c, d) = (
            ssa.fresh_value(),
            ssa.fresh_value(),
            ssa.fresh_value(),
            ssa.fresh_value(),
        );

        let f = ssa.get_unique_entrypoint_mut();
        f.get_entry_mut().push_parameter(x, Type::u(32));
        f.get_entry_mut().push_parameter(y, Type::u(32));
        let entry = f.get_entry_mut();
        entry.push_instruction(OpCode::BinaryArithOp {
            kind: BinaryArithOpKind::Add,
            result: a,
            lhs: x,
            rhs: y,
        });
        entry.push_instruction(OpCode::BinaryArithOp {
            kind: BinaryArithOpKind::Add,
            result: b,
            lhs: x,
            rhs: y,
        });
        entry.push_instruction(OpCode::BinaryArithOp {
            kind: BinaryArithOpKind::Add,
            result: c,
            lhs: y,
            rhs: x,
        });
        entry.push_instruction(OpCode::BinaryArithOp {
            kind: BinaryArithOpKind::Sub,
            result: d,
            lhs: x,
            rhs: y,
        });
        entry.set_terminator(Terminator::Return(vec![a, b, c, d]));

        let fid = ssa.get_unique_entrypoint_id();
        let cc = run_in_test(&ssa);
        assert!(cc.known_equal(fid, a, b));
        assert!(cc.known_equal(fid, a, c)); // x+y ≡ y+x
        assert!(!cc.known_equal(fid, a, d)); // x+y ≢ x-y
        assert!(!cc.known_equal(fid, x, y)); // distinct opaque values

        let mut expected = vec![a, b, c];
        expected.sort();
        assert_eq!(cc.congruence_class(fid, a), expected);
    }

    /// φ-operands come from *executable* edges only: a parameter whose value differs solely on a
    /// dead in-edge is still congruent to one that agrees on the live edge.
    #[test]
    fn phi_congruence_excludes_dead_edges() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let c_true = ssa.add_const(Constant::U(1, 1));
        let (x, y, p, q) = (
            ssa.fresh_value(),
            ssa.fresh_value(),
            ssa.fresh_value(),
            ssa.fresh_value(),
        );

        let f = ssa.get_unique_entrypoint_mut();
        let live = f.add_block();
        let dead = f.add_block();
        let merge = f.add_block();
        f.get_entry_mut().push_parameter(x, Type::field());
        f.get_entry_mut().push_parameter(y, Type::field());
        f.get_entry_mut()
            .set_terminator(Terminator::JmpIf(c_true, live, dead));
        // Live edge agrees on both params; the dead edge would have forced them apart.
        f.get_block_mut(live)
            .set_terminator(Terminator::Jmp(merge, vec![x, x]));
        f.get_block_mut(dead)
            .set_terminator(Terminator::Jmp(merge, vec![x, y]));
        let merge_block = f.get_block_mut(merge);
        merge_block.push_parameter(p, Type::field());
        merge_block.push_parameter(q, Type::field());
        merge_block.set_terminator(Terminator::Return(vec![p, q]));

        let fid = ssa.get_unique_entrypoint_id();
        let cc = run_in_test(&ssa);
        assert!(cc.known_equal(fid, p, q));
    }

    /// The same merge with *both* edges live keeps the parameters apart — congruence is genuinely
    /// reachability-sensitive.
    #[test]
    fn phi_distinguished_when_both_edges_live() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let (cond, x, y, p, q) = (
            ssa.fresh_value(),
            ssa.fresh_value(),
            ssa.fresh_value(),
            ssa.fresh_value(),
            ssa.fresh_value(),
        );

        let f = ssa.get_unique_entrypoint_mut();
        let e1 = f.add_block();
        let e2 = f.add_block();
        let merge = f.add_block();
        f.get_entry_mut().push_parameter(cond, Type::u(1));
        f.get_entry_mut().push_parameter(x, Type::field());
        f.get_entry_mut().push_parameter(y, Type::field());
        f.get_entry_mut()
            .set_terminator(Terminator::JmpIf(cond, e1, e2));
        f.get_block_mut(e1)
            .set_terminator(Terminator::Jmp(merge, vec![x, x]));
        f.get_block_mut(e2)
            .set_terminator(Terminator::Jmp(merge, vec![x, y]));
        let merge_block = f.get_block_mut(merge);
        merge_block.push_parameter(p, Type::field());
        merge_block.push_parameter(q, Type::field());
        merge_block.set_terminator(Terminator::Return(vec![p, q]));

        let fid = ssa.get_unique_entrypoint_id();
        let cc = run_in_test(&ssa);
        assert!(!cc.known_equal(fid, p, q));
    }

    /// The optimistic win pessimistic value numbering cannot reach: two parallel induction variables
    /// that start equal and step identically are congruent across the loop back-edge.
    #[test]
    fn loop_carried_parallel_induction_is_congruent() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let c0 = ssa.add_const(Constant::U(32, 0));
        let c1 = ssa.add_const(Constant::U(32, 1));
        let c10 = ssa.add_const(Constant::U(32, 10));
        let (i, j, lt, i2, j2) = (
            ssa.fresh_value(),
            ssa.fresh_value(),
            ssa.fresh_value(),
            ssa.fresh_value(),
            ssa.fresh_value(),
        );

        let f = ssa.get_unique_entrypoint_mut();
        let header = f.add_block();
        let body = f.add_block();
        let exit = f.add_block();

        f.get_entry_mut()
            .set_terminator(Terminator::Jmp(header, vec![c0, c0]));
        let header_block = f.get_block_mut(header);
        header_block.push_parameter(i, Type::u(32));
        header_block.push_parameter(j, Type::u(32));
        header_block.push_instruction(OpCode::Cmp {
            kind: CmpKind::Lt,
            result: lt,
            lhs: i,
            rhs: c10,
        });
        header_block.set_terminator(Terminator::JmpIf(lt, body, exit));
        let body_block = f.get_block_mut(body);
        body_block.push_instruction(OpCode::BinaryArithOp {
            kind: BinaryArithOpKind::Add,
            result: i2,
            lhs: i,
            rhs: c1,
        });
        body_block.push_instruction(OpCode::BinaryArithOp {
            kind: BinaryArithOpKind::Add,
            result: j2,
            lhs: j,
            rhs: c1,
        });
        body_block.set_terminator(Terminator::Jmp(header, vec![i2, j2]));
        f.get_block_mut(exit)
            .set_terminator(Terminator::Return(vec![i, j]));

        let fid = ssa.get_unique_entrypoint_id();
        let cc = run_in_test(&ssa);
        assert!(cc.known_equal(fid, i, j)); // loop-carried congruence
        assert!(cc.known_equal(fid, i2, j2)); // and their parallel updates
    }

    /// Assert-vacuum soundness: `assert(x == 5)` pins `x` to 5 *conditionally* in dominated blocks,
    /// but never unconditionally — so a global fold (SCCP) can't fold `x` and vacuum the assert.
    #[test]
    fn assert_eq_const_is_conditional_not_unconditional() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let c5 = ssa.add_const(Constant::U(32, 5));
        let x = ssa.fresh_value();

        let f = ssa.get_unique_entrypoint_mut();
        let entry_id = f.get_entry_id();
        let after = f.add_block();
        f.get_entry_mut().push_parameter(x, Type::u(32));
        f.get_entry_mut().push_instruction(OpCode::AssertCmp {
            kind: CmpKind::Eq,
            lhs: x,
            rhs: c5,
        });
        f.get_entry_mut()
            .set_terminator(Terminator::Jmp(after, vec![]));
        f.get_block_mut(after)
            .set_terminator(Terminator::Return(vec![]));

        let fid = ssa.get_unique_entrypoint_id();
        let cc = run_in_test(&ssa);

        // Unconditionally `x` is unknown — the assert never enters the global lattice.
        assert_eq!(cc.const_of(fid, x), None);
        assert!(cc.new_const_values(fid).iter().all(|(v, _)| *v != x));
        // Conditionally `x == 5` at every block the assert *strictly* dominates ...
        assert_eq!(
            cc.asserted_const(fid, after, x).as_deref(),
            Some(&Constant::U(32, 5))
        );
        // ... but not in the asserting block itself (block-entry granularity).
        assert_eq!(cc.asserted_const(fid, entry_id, x), None);
    }

    /// `assert(b)` proves `b == true` conditionally in dominated blocks, never unconditionally.
    #[test]
    fn assert_bool_is_conditional() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let b = ssa.fresh_value();

        let f = ssa.get_unique_entrypoint_mut();
        let after = f.add_block();
        f.get_entry_mut().push_parameter(b, Type::u(1));
        f.get_entry_mut()
            .push_instruction(OpCode::Assert { value: b });
        f.get_entry_mut()
            .set_terminator(Terminator::Jmp(after, vec![]));
        f.get_block_mut(after)
            .set_terminator(Terminator::Return(vec![]));

        let fid = ssa.get_unique_entrypoint_id();
        let cc = run_in_test(&ssa);
        assert_eq!(cc.const_of(fid, b), None);
        assert_eq!(
            cc.asserted_const(fid, after, b).as_deref(),
            Some(&Constant::U(1, 1))
        );
    }

    /// `assert(x == y)` with neither side constant records a *conditional* equality — distinct from
    /// structural congruence, which (unconditionally) keeps the two parameters apart.
    #[test]
    fn assert_eq_pure_equality_is_conditional() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let (x, y) = (ssa.fresh_value(), ssa.fresh_value());

        let f = ssa.get_unique_entrypoint_mut();
        let entry_id = f.get_entry_id();
        let after = f.add_block();
        f.get_entry_mut().push_parameter(x, Type::u(32));
        f.get_entry_mut().push_parameter(y, Type::u(32));
        f.get_entry_mut().push_instruction(OpCode::AssertCmp {
            kind: CmpKind::Eq,
            lhs: x,
            rhs: y,
        });
        f.get_entry_mut()
            .set_terminator(Terminator::Jmp(after, vec![]));
        f.get_block_mut(after)
            .set_terminator(Terminator::Return(vec![]));

        let fid = ssa.get_unique_entrypoint_id();
        let cc = run_in_test(&ssa);
        assert!(cc.asserted_equal(fid, after, x, y));
        assert!(cc.asserted_equal(fid, after, y, x)); // symmetric
        assert!(!cc.asserted_equal(fid, entry_id, x, y)); // not in the asserting block
        // Structural congruence (unconditional) does not see the assert.
        assert!(!cc.known_equal(fid, x, y));
    }

    /// The false edge of `if x == y` proves `x != y` at its target and the blocks that target
    /// dominates — and nowhere else.
    #[test]
    fn disequality_from_false_edge() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let (x, y, eq) = (ssa.fresh_value(), ssa.fresh_value(), ssa.fresh_value());

        let f = ssa.get_unique_entrypoint_mut();
        let entry_id = f.get_entry_id();
        let then_b = f.add_block();
        let else_b = f.add_block();
        let after = f.add_block();
        f.get_entry_mut().push_parameter(x, Type::u(32));
        f.get_entry_mut().push_parameter(y, Type::u(32));
        f.get_entry_mut().push_instruction(OpCode::Cmp {
            kind: CmpKind::Eq,
            result: eq,
            lhs: x,
            rhs: y,
        });
        f.get_entry_mut()
            .set_terminator(Terminator::JmpIf(eq, then_b, else_b));
        f.get_block_mut(then_b)
            .set_terminator(Terminator::Return(vec![]));
        f.get_block_mut(else_b)
            .set_terminator(Terminator::Jmp(after, vec![]));
        f.get_block_mut(after)
            .set_terminator(Terminator::Return(vec![]));

        let fid = ssa.get_unique_entrypoint_id();
        let cc = run_in_test(&ssa);
        // x != y on the false-edge target and blocks it dominates.
        assert!(cc.known_unequal(fid, else_b, x, y));
        assert!(cc.known_unequal(fid, else_b, y, x)); // symmetric
        assert!(cc.known_unequal(fid, after, x, y));
        // Not on the true edge, nor before the branch.
        assert!(!cc.known_unequal(fid, then_b, x, y));
        assert!(!cc.known_unequal(fid, entry_id, x, y));
    }

    /// An unpinned witness write forwards `r → v`; distinct witnesses get distinct forwards and are
    /// never congruent; a *pinned* write (a real constraint) forwards nothing.
    #[test]
    fn unpinned_witness_forwards_distinct_not_merged() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let (v1, v2) = (ssa.fresh_value(), ssa.fresh_value());
        let (r1, r2, rp) = (ssa.fresh_value(), ssa.fresh_value(), ssa.fresh_value());

        let f = ssa.get_unique_entrypoint_mut();
        f.get_entry_mut().push_parameter(v1, Type::u(32));
        f.get_entry_mut().push_parameter(v2, Type::u(32));
        let entry = f.get_entry_mut();
        entry.push_instruction(OpCode::WriteWitness {
            result: Some(r1),
            value: v1,
            pinned: false,
        });
        entry.push_instruction(OpCode::WriteWitness {
            result: Some(r2),
            value: v2,
            pinned: false,
        });
        entry.push_instruction(OpCode::WriteWitness {
            result: Some(rp),
            value: v1,
            pinned: true,
        });
        entry.set_terminator(Terminator::Return(vec![]));

        let fid = ssa.get_unique_entrypoint_id();
        let cc = run_in_test(&ssa);
        assert_eq!(cc.witness_forward(fid, r1), [v1].as_slice());
        assert_eq!(cc.witness_forward(fid, r2), [v2].as_slice());
        // A pinned write carries a real `r == v` constraint — it is not a free witness, no forward.
        assert!(cc.witness_forward(fid, rp).is_empty());
        // Two distinct free witnesses are never unified (the hard non-merge prohibition).
        assert!(!cc.known_equal(fid, r1, r2));
    }

    /// `witness_forward` is the sorted union of *both* readings of the one witness↔value
    /// correspondence: the `WriteWitness` hint (`r → v`) and every `value_of(r)` read (`r → w`).
    /// Distinct witnesses keep disjoint sets — no cross-witness union.
    #[test]
    fn witness_forward_unions_hint_and_value_of_reads() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let (v, v2) = (ssa.fresh_value(), ssa.fresh_value()); // honest hint payloads
        let (r, r2) = (ssa.fresh_value(), ssa.fresh_value()); // witness handles
        let (w1, w2) = (ssa.fresh_value(), ssa.fresh_value()); // two distinct `value_of(r)` reads

        let f = ssa.get_unique_entrypoint_mut();
        f.get_entry_mut().push_parameter(v, Type::u(32));
        f.get_entry_mut().push_parameter(v2, Type::u(32));
        let entry = f.get_entry_mut();
        entry.push_instruction(OpCode::WriteWitness {
            result: Some(r),
            value: v,
            pinned: false,
        });
        // `value_of(r)` strips r's witness wrapper: w1, w2 are honestly equal to r. Two separate
        // reads stay distinct (ValueOf is excluded from value-numbering), so both join r's set.
        entry.push_instruction(OpCode::Cast {
            result: w1,
            value: r,
            target: CastTarget::ValueOf,
        });
        entry.push_instruction(OpCode::Cast {
            result: w2,
            value: r,
            target: CastTarget::ValueOf,
        });
        entry.push_instruction(OpCode::WriteWitness {
            result: Some(r2),
            value: v2,
            pinned: false,
        });
        entry.set_terminator(Terminator::Return(vec![]));

        let fid = ssa.get_unique_entrypoint_id();
        let cc = run_in_test(&ssa);

        // r's honest-value set is the sorted union of the hint and both `value_of` reads.
        let mut expected = vec![v, w1, w2];
        expected.sort_unstable();
        assert_eq!(cc.witness_forward(fid, r), expected.as_slice());

        // The unrelated witness keeps a disjoint, single-element set — no cross-witness union.
        assert_eq!(cc.witness_forward(fid, r2), [v2].as_slice());
        assert!(!cc.known_equal(fid, r, r2));
    }

    /// An assert placed in the *last* block proves nothing by dominance (nothing follows it), but it
    /// post-dominates everything upstream — so on accepting runs its fact holds at those earlier
    /// blocks. The asserted value (`x`, a parameter) is in scope throughout.
    #[test]
    fn assert_below_use_holds_via_post_dominance() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let c5 = ssa.add_const(Constant::U(32, 5));
        let x = ssa.fresh_value();

        let f = ssa.get_unique_entrypoint_mut();
        let entry_id = f.get_entry_id();
        let mid = f.add_block();
        let tail = f.add_block();
        f.get_entry_mut().push_parameter(x, Type::u(32));
        f.get_entry_mut()
            .set_terminator(Terminator::Jmp(mid, vec![]));
        f.get_block_mut(mid)
            .set_terminator(Terminator::Jmp(tail, vec![]));
        f.get_block_mut(tail).push_instruction(OpCode::AssertCmp {
            kind: CmpKind::Eq,
            lhs: x,
            rhs: c5,
        });
        f.get_block_mut(tail)
            .set_terminator(Terminator::Return(vec![]));

        let fid = ssa.get_unique_entrypoint_id();
        let cc = run_in_test(&ssa);
        // `tail` post-dominates `mid` and `entry`, so `x == 5` holds at both — a fact pure dominance
        // (the assert is last) would miss entirely.
        assert_eq!(
            cc.asserted_const(fid, mid, x).as_deref(),
            Some(&Constant::U(32, 5))
        );
        assert_eq!(
            cc.asserted_const(fid, entry_id, x).as_deref(),
            Some(&Constant::U(32, 5))
        );
        // Still never recorded at the asserting block's own entry (block-entry granularity).
        assert_eq!(cc.asserted_const(fid, tail, x), None);
    }

    /// Post-dominance fan-out is gated on the asserted value being in scope: a value defined *below*
    /// the target block must not have the fact attributed to that target (the guard dominance
    /// otherwise supplies for free).
    #[test]
    fn post_dominance_respects_value_scope() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let c10 = ssa.add_const(Constant::U(32, 10));
        let p = ssa.fresh_value();
        let x = ssa.fresh_value();

        let f = ssa.get_unique_entrypoint_mut();
        let entry_id = f.get_entry_id();
        let mid = f.add_block();
        let tail = f.add_block();
        f.get_entry_mut().push_parameter(p, Type::u(32));
        f.get_entry_mut()
            .set_terminator(Terminator::Jmp(mid, vec![]));
        // `x` is defined in `mid`, *below* `entry`.
        f.get_block_mut(mid)
            .push_instruction(OpCode::BinaryArithOp {
                kind: BinaryArithOpKind::Add,
                result: x,
                lhs: p,
                rhs: p,
            });
        f.get_block_mut(mid)
            .set_terminator(Terminator::Jmp(tail, vec![]));
        f.get_block_mut(tail).push_instruction(OpCode::AssertCmp {
            kind: CmpKind::Eq,
            lhs: x,
            rhs: c10,
        });
        f.get_block_mut(tail)
            .set_terminator(Terminator::Return(vec![]));

        let fid = ssa.get_unique_entrypoint_id();
        let cc = run_in_test(&ssa);
        // At `mid`, `x` is in scope and `tail` post-dominates it ⇒ the fact holds.
        assert_eq!(
            cc.asserted_const(fid, mid, x).as_deref(),
            Some(&Constant::U(32, 10))
        );
        // At `entry`, `x` is not yet defined — the in-scope guard withholds the fact even though
        // `tail` post-dominates `entry`.
        assert_eq!(cc.asserted_const(fid, entry_id, x), None);
    }

    /// An assert on only one arm of a branch neither dominates nor post-dominates the blocks around
    /// the branch, so its fact must not leak there — the other arm reaches `exit` without it.
    #[test]
    fn assert_on_one_branch_does_not_post_dominate() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let c5 = ssa.add_const(Constant::U(32, 5));
        let cond = ssa.fresh_value();
        let x = ssa.fresh_value();

        let f = ssa.get_unique_entrypoint_mut();
        let entry_id = f.get_entry_id();
        let then_b = f.add_block();
        let else_b = f.add_block();
        let merge = f.add_block();
        f.get_entry_mut().push_parameter(cond, Type::u(1));
        f.get_entry_mut().push_parameter(x, Type::u(32));
        f.get_entry_mut()
            .set_terminator(Terminator::JmpIf(cond, then_b, else_b));
        // The assert lives only on the `then` branch.
        f.get_block_mut(then_b).push_instruction(OpCode::AssertCmp {
            kind: CmpKind::Eq,
            lhs: x,
            rhs: c5,
        });
        f.get_block_mut(then_b)
            .set_terminator(Terminator::Jmp(merge, vec![]));
        f.get_block_mut(else_b)
            .set_terminator(Terminator::Jmp(merge, vec![]));
        f.get_block_mut(merge)
            .set_terminator(Terminator::Return(vec![]));

        let fid = ssa.get_unique_entrypoint_id();
        let cc = run_in_test(&ssa);
        // `x` is in scope everywhere, so only the missing dominance/post-dominance keeps the fact
        // out of `entry` and `merge` (the `else` path skips the assert).
        assert_eq!(cc.asserted_const(fid, entry_id, x), None);
        assert_eq!(cc.asserted_const(fid, merge, x), None);
        // And never at the asserting block's own entry.
        assert_eq!(cc.asserted_const(fid, then_b, x), None);
    }

    /// A post-dominating *pure* equality (`assert(x == y)`, neither side constant) requires *both*
    /// sides in scope at the target.
    #[test]
    fn post_dominating_assert_eq_needs_both_sides_in_scope() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let x = ssa.fresh_value();
        let y = ssa.fresh_value();
        let p = ssa.fresh_value();

        let f = ssa.get_unique_entrypoint_mut();
        let entry_id = f.get_entry_id();
        let mid = f.add_block();
        let tail = f.add_block();
        f.get_entry_mut().push_parameter(x, Type::u(32));
        f.get_entry_mut().push_parameter(p, Type::u(32));
        f.get_entry_mut()
            .set_terminator(Terminator::Jmp(mid, vec![]));
        // `y` is defined in `mid`, below `entry`.
        f.get_block_mut(mid)
            .push_instruction(OpCode::BinaryArithOp {
                kind: BinaryArithOpKind::Add,
                result: y,
                lhs: p,
                rhs: p,
            });
        f.get_block_mut(mid)
            .set_terminator(Terminator::Jmp(tail, vec![]));
        f.get_block_mut(tail).push_instruction(OpCode::AssertCmp {
            kind: CmpKind::Eq,
            lhs: x,
            rhs: y,
        });
        f.get_block_mut(tail)
            .set_terminator(Terminator::Return(vec![]));

        let fid = ssa.get_unique_entrypoint_id();
        let cc = run_in_test(&ssa);
        // Both sides live at `mid` ⇒ the post-dominating equality holds, symmetrically.
        assert!(cc.asserted_equal(fid, mid, x, y));
        assert!(cc.asserted_equal(fid, mid, y, x));
        // `y` is undefined at `entry`, so the equality is withheld there.
        assert!(!cc.asserted_equal(fid, entry_id, x, y));
    }
}
