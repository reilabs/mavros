//! The per-function combined optimistic fixpoint.
//!
//! This is the classic Wegman-Zadeck two-worklist algorithm. Every value starts at ⊤, is lowered to
//! a concrete constant when a transfer function folds it, and bottoms out at ⊥ otherwise; block
//! reachability and value lattices are computed together so constants propagate through
//! constant-decided branches.

use std::sync::Arc;

use crate::{
    collections::{HashMap, HashSet},
    compiler::{
        analysis::click_cooper::{
            congruence::Congruence,
            lattice::{
                Constness, bool_constant, bool_constness, const_bool, const_join, eval_array_get,
                eval_array_set, eval_binary, eval_bit_range, eval_cast, eval_cmp, eval_mk_repeated,
                eval_mk_seq, eval_not, eval_sext, eval_slice_len, eval_slice_push,
            },
            summary::{DetSummaries, FnSummary, ReturnJump, SymSummaries},
        },
        ssa::{
            BlockId, FunctionId, Instruction, Terminator, ValueId,
            hlssa::{
                CallTarget, CmpKind, Constant, HLFunction, HLSSAConstantsSnapshot, OpCode,
                ScalarFold,
            },
        },
    },
};

// TYPES
// ================================================================================================

/// Per-block branch predicate facts: a value known true/false on entry to the block, recorded only
/// when every executable incoming edge proves the same boolean.
pub(crate) type BoolFacts = HashMap<ValueId, bool>;

// CONSTANTS
// ================================================================================================

/// The maximum number of per-function solver rounds in the combined-fixpoint writeback (round 0 is
/// the base solve; the rest refine with congruence-derived promotions).
///
/// The writeback is monotone and converges well before this, so it is only a backstop in case of an
/// algorithmic bug. Real cascades are one or two rounds deep at most.
const MAX_WRITEBACK_ROUNDS: usize = 8;

// FUNCTION FACTS
// ================================================================================================

/// The converged per-function analysis state.
#[derive(Debug, Default)]
pub(crate) struct FunctionFacts {
    /// Each value's converged constness (`⊤` / `Const` / `⊥`).
    ///
    /// A value absent from the map is implicitly `⊤` (never lowered); already-interned constants
    /// are resolved from the snapshot, so they are not stored here.
    pub lattice: HashMap<ValueId, Constness>,

    /// Blocks proven reachable by the combined reachability+constant fixpoint — those whose
    /// instructions were visited at least once.
    pub reachable: HashSet<BlockId>,

    /// CFG edges `(from, to)` proven executable. A constant-decided `JmpIf` marks only the taken
    /// edge, so its untaken successor can stay unreachable.
    pub exec_edges: HashSet<(BlockId, BlockId)>,

    /// Per-block branch predicate facts: values known true/false on entry to a block, recorded only
    /// where every executable in-edge agrees on the same boolean (path-sensitive).
    pub block_facts: HashMap<BlockId, BoolFacts>,

    /// The converged global value-numbering partition (structural congruence classes), built once
    /// over the final reachability/lattice state.
    pub congruence: Congruence,
}

// FUNCTION SOLVER
// ================================================================================================

/// The fact solver for a single function.
pub(crate) struct FunctionSolver<'f, 'c, 's> {
    /// The function being solved.
    function: &'f HLFunction,

    /// A snapshot of the constants that are currently available.
    consts: &'c HLSSAConstantsSnapshot,

    /// Each value's constness in the in-progress fixpoint; a value absent from the map is
    /// implicitly `⊤` (not yet lowered).
    lattice: HashMap<ValueId, Constness>,

    /// CFG edges `(from, to)` proven executable so far.
    exec_edges: HashSet<(BlockId, BlockId)>,

    /// Executable predecessors per block — the inverse of `exec_edges`, kept in step with it (and
    /// so deduplicated by that edge set) so a block's live in-edges can be iterated directly.
    exec_preds: HashMap<BlockId, Vec<BlockId>>,

    /// Blocks whose instructions have been visited at least once.
    reachable: HashSet<BlockId>,

    /// Predicate facts known at block entry (intersected over executable in-edges).
    block_facts: HashMap<BlockId, BoolFacts>,

    /// Newly-executable CFG edges awaiting processing — the flow (reachability) worklist.
    edge_worklist: Vec<(BlockId, BlockId)>,

    /// Values whose lattice cell just changed, awaiting re-evaluation of their uses — the SSA
    /// worklist.
    value_worklist: Vec<ValueId>,

    /// For each value: the sites that read it. `None` marks the block's terminator.
    uses: HashMap<ValueId, Vec<(BlockId, Option<usize>)>>,

    /// Interprocedural jump-function summaries, used to fold the results of constrained static
    /// `Call`s into the constant lattice.
    ///
    /// `None` (the default) is the intraprocedural mode where every `Call` result is `Bottom`.
    summaries: Option<&'s HashMap<FunctionId, FnSummary>>,

    /// Per-`(callee, return-index)` determinism bits, used when building the congruence partition
    /// to number deterministic static-call results cross-call.
    ///
    /// `None` (the default) leaves every call result opaque, identical to the prior behavior.
    det: Option<&'s DetSummaries>,

    /// Per-`(callee, return-index)` symbolic congruence jump functions, used when building the
    /// congruence partition to graft a callee's return expression into a constrained static call.
    ///
    /// A *separate* channel from `summaries`: this refines only congruence, never the constant
    /// lattice (`eval_call`), so it can be enabled in the summary-free intraprocedural solve
    /// without breaking the contract that keeps `Call` results `Bottom`. `None` (the default) falls
    /// back to the `det`-gated `CallDet` numbering.
    sym_summaries: Option<&'s SymSummaries>,

    /// Entry-parameter seeds for a context-sensitive solve: an entry parameter maps to the constant
    /// the calling context proved for the matching argument. Absent params default to `Bottom`.
    param_seeds: HashMap<ValueId, Constness>,

    /// Values promoted to a constant by the combined-fixpoint writeback — congruence-derived
    /// must-equal facts (e.g. `CmpEq(x, y)` with `known_equal(x, y)` folds to `true`).
    ///
    /// `None` (the default) is the base solve. When present, a promoted value takes precedence
    /// over its own (weaker) transfer result in `lattice_of`/`lattice_in_block` — exactly as the
    /// interned-const snapshot does — and `set_lattice` refuses to lower it, so the value's own
    /// `Bottom` transfer can never `const_join` the promotion back down.
    promotions: Option<&'s HashMap<ValueId, Arc<Constant>>>,
}

impl<'f, 'c, 's> FunctionSolver<'f, 'c, 's> {
    pub(crate) fn new(function: &'f HLFunction, consts: &'c HLSSAConstantsSnapshot) -> Self {
        let mut uses: HashMap<ValueId, Vec<(BlockId, Option<usize>)>> = HashMap::default();
        for (bid, block) in function.get_blocks() {
            for (idx, instr) in block.get_instructions().enumerate() {
                for input in instr.get_inputs() {
                    uses.entry(*input).or_default().push((*bid, Some(idx)));
                }
            }
            if let Some(term) = block.get_terminator() {
                let inputs: Vec<ValueId> = match term {
                    Terminator::Jmp(_, args) => args.clone(),
                    Terminator::JmpIf(cond, _, _) => vec![*cond],
                    Terminator::Return(vals) => vals.clone(),
                };
                for v in inputs {
                    uses.entry(v).or_default().push((*bid, None));
                }
            }
        }

        Self {
            function,
            consts,
            lattice: HashMap::default(),
            exec_edges: HashSet::default(),
            exec_preds: HashMap::default(),
            reachable: HashSet::default(),
            block_facts: HashMap::default(),
            edge_worklist: Vec::new(),
            value_worklist: Vec::new(),
            uses,
            summaries: None,
            det: None,
            sym_summaries: None,
            param_seeds: HashMap::default(),
            promotions: None,
        }
    }

    /// Fold constrained static `Call` results via these interprocedural summaries.
    pub(crate) fn with_summaries(mut self, summaries: &'s HashMap<FunctionId, FnSummary>) -> Self {
        self.summaries = Some(summaries);
        self
    }

    /// Number deterministic static-call results cross-call using these determinism bits.
    pub(crate) fn with_determinism(mut self, det: &'s DetSummaries) -> Self {
        self.det = Some(det);
        self
    }

    /// Graft callees' symbolic return expressions into constrained static-call results when
    /// building the congruence partition (a refinement of the `det`-gated `CallDet` numbering).
    pub(crate) fn with_sym_summaries(mut self, sym_summaries: &'s SymSummaries) -> Self {
        self.sym_summaries = Some(sym_summaries);
        self
    }

    /// Seed entry parameters with the calling context's argument constants.
    pub(crate) fn with_param_seeds(mut self, seeds: HashMap<ValueId, Constness>) -> Self {
        self.param_seeds = seeds;
        self
    }

    /// Seed congruence-derived must-equal constants (the combined-fixpoint writeback).
    ///
    /// A promoted value takes precedence over its own transfer, so the worklist propagates the
    /// stronger fact.
    pub(crate) fn with_promotions(
        mut self,
        promotions: &'s HashMap<ValueId, Arc<Constant>>,
    ) -> Self {
        self.promotions = Some(promotions);
        self
    }

    pub(crate) fn run(&mut self) {
        let entry = self.function.get_entry_id();

        // Entry parameters are the function's arguments. Intraprocedurally they are unknown
        // (`Bottom`); a context-sensitive solve seeds them with the calling context's argument
        // constants via `param_seeds`.
        for (p, _) in self.function.get_entry().get_parameters() {
            let seed = self
                .param_seeds
                .get(p)
                .cloned()
                .unwrap_or(Constness::Bottom);
            self.lattice.insert(*p, seed);
        }
        self.reachable.insert(entry);
        self.block_facts.insert(entry, BoolFacts::default());
        self.visit_block(entry);

        loop {
            if let Some((p, b)) = self.edge_worklist.pop() {
                self.process_edge(p, b);
                continue;
            }
            if let Some(v) = self.value_worklist.pop() {
                self.process_value(v);
                continue;
            }
            break;
        }
        self.assert_no_stuck_conditions();
    }

    /// Convert the solver into facts.
    pub(crate) fn into_facts(self) -> FunctionFacts {
        let det = self.det;
        let promotions = self.promotions;

        // `lattice_of` already consults the promotions, so the rebuilt congruence labels a promoted
        // value by its constant (folding it together with structurally-equal constants).
        let congruence = Congruence::build(
            self.function,
            &self.reachable,
            &self.exec_edges,
            |v| match self.lattice_of(v) {
                Constness::Const(c) => Some(c),
                Constness::Top | Constness::Bottom => None,
            },
            |g, j| {
                det.is_some_and(|d| d.get(&g).and_then(|rets| rets.get(j)).copied() == Some(true))
            },
            self.sym_summaries,
        );

        // Materialise the promotions into the returned lattice: they live in the borrowed
        // `promotions` map, not in `self.lattice`, so `const_of`/`new_const_values` would not see
        // them otherwise. A promoted value is an instruction result, never interned, so there is no
        // collision with the snapshot.
        let mut lattice = self.lattice;
        if let Some(promotions) = promotions {
            for (v, c) in promotions {
                lattice.insert(*v, Constness::Const(c.clone()));
            }
        }

        FunctionFacts {
            lattice,
            reachable: self.reachable,
            exec_edges: self.exec_edges,
            block_facts: self.block_facts,
            congruence,
        }
    }

    /// A `JmpIf` condition can converge at ⊤ only if it (transitively) bottoms out at a block
    /// parameter fed solely by itself along executable edges, so we sanity check.
    fn assert_no_stuck_conditions(&self) {
        for bid in &self.reachable {
            if let Some(Terminator::JmpIf(cond, _, _)) =
                self.function.get_block(*bid).get_terminator()
            {
                assert!(
                    self.lattice_of(*cond) != Constness::Top,
                    "ICE: ClickCooper converged with JmpIf condition v{} in `{}` stuck at ⊤; \
                     the condition is only defined through a self-referential block-parameter \
                     cycle, so a use is not dominated by its definition (malformed SSA)",
                    cond.0,
                    self.function.get_name(),
                );
            }
        }
    }

    fn lattice_of(&self, v: ValueId) -> Constness {
        if let Some(c) = self.consts.get(&v) {
            return Constness::Const(c.clone());
        }
        if let Some(c) = self.promotions.and_then(|p| p.get(&v)) {
            return Constness::Const(c.clone());
        }
        self.lattice.get(&v).cloned().unwrap_or(Constness::Top)
    }

    fn lattice_in_block(&self, bid: BlockId, v: ValueId) -> Constness {
        if let Some(c) = self.consts.get(&v) {
            return Constness::Const(c.clone());
        }

        // A promotion is an *unconditional* must-equal, so it dominates the path-sensitive branch
        // facts just as the interned constant above does.
        if let Some(c) = self.promotions.and_then(|p| p.get(&v)) {
            return Constness::Const(c.clone());
        }
        if let Some(value) = self.block_facts.get(&bid).and_then(|facts| facts.get(&v)) {
            return bool_constness(*value);
        }
        self.lattice.get(&v).cloned().unwrap_or(Constness::Top)
    }

    /// Lower `v` to `join(current, new)`, scheduling its users if the value changed.
    fn set_lattice(&mut self, v: ValueId, new: Constness) {
        // A promoted value is pinned to its congruence-derived constant: its own transfer (which
        // does not see the congruence) must never lower it. Interned consts are never instruction
        // results so never reach here; promoted values are, hence this explicit guard.
        if self.promotions.is_some_and(|p| p.contains_key(&v)) {
            return;
        }

        let old = self.lattice_of(v);
        let joined = const_join(old.clone(), new);
        if joined != old {
            self.lattice.insert(v, joined);
            self.value_worklist.push(v);
        }
    }

    fn process_edge(&mut self, p: BlockId, b: BlockId) {
        if !self.exec_edges.insert((p, b)) {
            return;
        }
        self.exec_preds.entry(b).or_default().push(p);

        // Snapshot the executable predecessors once and share it across both recomputations: both
        // helpers take `&mut self`, so they need an owned copy detached from `self.exec_preds`.
        let preds = self.exec_preds.get(&b).cloned().unwrap_or_default();
        self.recompute_params_at(b, &preds, None);

        let facts_changed = self.recompute_facts(b, &preds);
        let first_visit = self.reachable.insert(b);
        if first_visit || facts_changed {
            self.visit_block(b);
        }
    }

    fn process_value(&mut self, v: ValueId) {
        let sites = self.uses.get(&v).cloned().unwrap_or_default();
        for (bid, site) in sites {
            if !self.reachable.contains(&bid) {
                continue;
            }
            match site {
                Some(idx) => self.visit_instruction(bid, idx),
                None => self.revisit_terminator_for(bid, v),
            }
        }
    }

    /// Re-examine `bid`'s terminator because its operand `v` changed.
    ///
    /// A `Jmp` along an already-executable edge re-joins only the target parameters that actually
    /// consume `v`.
    fn revisit_terminator_for(&mut self, bid: BlockId, v: ValueId) {
        match self.function.get_block(bid).get_terminator() {
            Some(Terminator::Jmp(t, args)) => {
                let t = *t;
                if self.exec_edges.contains(&(bid, t)) {
                    let indices: Vec<usize> = args
                        .iter()
                        .enumerate()
                        .filter_map(|(i, a)| (*a == v).then_some(i))
                        .collect();
                    let preds = self.exec_preds.get(&t).cloned().unwrap_or_default();
                    self.recompute_params_at(t, &preds, Some(&indices));
                }
            }
            Some(Terminator::JmpIf(..)) | Some(Terminator::Return(..)) | None => {
                self.visit_terminator(bid)
            }
        }
    }

    fn visit_block(&mut self, bid: BlockId) {
        let n = self.function.get_block(bid).get_instructions().count();
        for idx in 0..n {
            self.visit_instruction(bid, idx);
        }
        self.visit_terminator(bid);
    }

    fn visit_instruction(&mut self, bid: BlockId, idx: usize) {
        let updates = self.transfer(bid, self.function.get_block(bid).get_instruction(idx));
        for (v, lat) in updates {
            self.set_lattice(v, lat);
        }
    }

    fn visit_terminator(&mut self, bid: BlockId) {
        let Some(term) = self.function.get_block(bid).get_terminator().cloned() else {
            return;
        };
        match term {
            Terminator::Jmp(t, _) => self.notify_edge(bid, t),
            Terminator::JmpIf(cond, t, f) => match self.lattice_in_block(bid, cond) {
                Constness::Top => {}
                Constness::Const(c) => match const_bool(&c) {
                    Some(true) => self.notify_edge(bid, t),
                    Some(false) => self.notify_edge(bid, f),
                    None => {
                        self.notify_edge(bid, t);
                        self.notify_edge(bid, f);
                    }
                },
                Constness::Bottom => {
                    self.notify_edge(bid, t);
                    self.notify_edge(bid, f);
                }
            },
            Terminator::Return(_) => {}
        }
    }

    /// Mark edge `(p, t)` executable; if it already is, the jump arguments may have changed, so
    /// re-join the target's parameters.
    fn notify_edge(&mut self, p: BlockId, t: BlockId) {
        if self.exec_edges.contains(&(p, t)) {
            let preds = self.exec_preds.get(&t).cloned().unwrap_or_default();
            self.recompute_params_at(t, &preds, None);
            if self.recompute_facts(t, &preds) {
                self.visit_block(t);
            }
        } else {
            self.edge_worklist.push((p, t));
        }
    }

    /// Re-join target parameters from `preds`.
    ///
    /// `indices` selects which parameters to recompute; `None` recomputes them all (the common
    /// full-recompute path) without materialising a `0..n` index vector. `preds` is the caller's
    /// executable-predecessor snapshot for `b`.
    fn recompute_params_at(&mut self, b: BlockId, preds: &[BlockId], indices: Option<&[usize]>) {
        if b == self.function.get_entry_id() {
            return;
        }

        let block = self.function.get_block(b);
        if !block.has_parameters() {
            return;
        }

        let params: Vec<ValueId> = block.get_parameter_values().copied().collect();

        // Recompute the chosen parameters in place: `indices` selects which, and `None` recomputes
        // them all (the common full-recompute path).
        let recompute = |this: &mut Self, i: usize| {
            let lat = this.join_param(b, preds, i);
            this.set_lattice(params[i], lat);
        };
        match indices {
            Some(indices) => indices.iter().for_each(|&i| recompute(self, i)),
            None => (0..params.len()).for_each(|i| recompute(self, i)),
        }
    }

    /// The constness contributed to block `b`'s `i`-th parameter, joined over the `i`-th jump
    /// argument of every executable predecessor in `preds`.
    fn join_param(&self, b: BlockId, preds: &[BlockId], i: usize) -> Constness {
        let mut acc = Constness::Top;
        for pred in preds {
            match self.function.get_block(*pred).get_terminator() {
                // A well-formed `Jmp` to `b` supplies one arg per parameter; a missing operand (an
                // ill-formed jump) is defensively treated as unknown rather than panicking.
                Some(Terminator::Jmp(t, args)) if *t == b => {
                    acc = match args.get(i) {
                        Some(arg) => const_join(acc, self.lattice_in_block(*pred, *arg)),
                        None => Constness::Bottom,
                    };
                }

                // A parameterized block is only ever entered via `Jmp`; treat anything else
                // (including a `Jmp` to a different target) as unknown in case the invariant ever
                // fails.
                Some(Terminator::Jmp(..))
                | Some(Terminator::JmpIf(..))
                | Some(Terminator::Return(..))
                | None => acc = Constness::Bottom,
            }
        }
        acc
    }

    fn recompute_facts(&mut self, b: BlockId, preds: &[BlockId]) -> bool {
        if b == self.function.get_entry_id() {
            return false;
        }
        let mut iter = preds.iter().copied();
        let Some(first_pred) = iter.next() else {
            return false;
        };

        let mut facts = self.edge_facts(first_pred, b);
        for pred in iter {
            let next = self.edge_facts(pred, b);
            facts.retain(|value, known| next.get(value) == Some(known));
            if facts.is_empty() {
                break;
            }
        }

        let old = self.block_facts.get(&b).cloned().unwrap_or_default();
        if old == facts {
            return false;
        }

        if facts.is_empty() {
            self.block_facts.remove(&b);
        } else {
            self.block_facts.insert(b, facts);
        }
        true
    }

    fn edge_facts(&self, pred: BlockId, target: BlockId) -> BoolFacts {
        let mut facts = self.block_facts.get(&pred).cloned().unwrap_or_default();
        if let Some((cond, value)) = self.branch_fact_for_edge(pred, target) {
            facts.insert(cond, value);
        }
        facts
    }

    fn branch_fact_for_edge(&self, pred: BlockId, target: BlockId) -> Option<(ValueId, bool)> {
        let Some(Terminator::JmpIf(cond, then_b, else_b)) =
            self.function.get_block(pred).get_terminator()
        else {
            return None;
        };
        if then_b == else_b {
            return None;
        }
        if *then_b == target {
            Some((*cond, true))
        } else if *else_b == target {
            Some((*cond, false))
        } else {
            None
        }
    }

    /// The transfer function: new lattice values for the instruction's results.
    fn transfer(&self, bid: BlockId, instr: &OpCode) -> Vec<(ValueId, Constness)> {
        // A constrained static `Call` folds its results via the callee's interprocedural jump
        // summary (when summaries are supplied). Without summaries every result is `Bottom`. It is
        // not a scalar fold, so it is handled before the `scalar_fold` projection.
        if let OpCode::Call {
            results,
            function,
            args,
            unconstrained,
        } = instr
        {
            return self.eval_call(bid, function, args, results, *unconstrained);
        }

        // The pure, value-semantic sequence ops are not scalar folds but *are* constant-folded over
        // the aggregate (`Blob`) constants kept in the lattice — e.g. indexing a constant lookup
        // table with a constant index yields a scalar constant. Like `Call`, they are handled
        // before the `scalar_fold` projection (which would otherwise bottom them out).
        if let Some(value) = self.eval_aggregate(bid, instr) {
            return instr.get_results().map(|r| (*r, value.clone())).collect();
        }

        // Foldable scalar ops are evaluated; everything else (memory, witness ops, asserts,
        // sequences, ...) is overdefined. `scalar_fold` is the single source of truth for which
        // opcodes are foldable, so this can never disagree with `OpCode::is_pure_scalar_fold` or
        // `op_signature` — no cross-classifier assertion is needed.
        let Some(fold) = instr.scalar_fold() else {
            return instr
                .get_results()
                .map(|r| (*r, Constness::Bottom))
                .collect();
        };

        let value = match fold {
            ScalarFold::Bin { kind, lhs, rhs } => {
                self.eval2(bid, lhs, rhs, |a, b| eval_binary(kind, a, b))
            }
            ScalarFold::Cmp { kind, lhs, rhs } => {
                self.eval2(bid, lhs, rhs, |a, b| eval_cmp(kind, a, b))
            }

            // `MulConst` is `const_val(field) * var(WitnessOf<…>)`. `var` is always a witness, so
            // it is never a lattice `Const` — the fold could never fire (and field-domain
            // multiplication must never inherit integer `Mul` width/overflow rules).
            ScalarFold::MulConst { .. } => Constness::Bottom,
            ScalarFold::Cast { target, value } => self.eval1(bid, value, |v| eval_cast(target, v)),
            ScalarFold::SExt {
                value,
                from_bits,
                to_bits,
            } => self.eval1(bid, value, |v| eval_sext(v, from_bits, to_bits)),
            ScalarFold::BitRange {
                value,
                offset,
                width,
            } => self.eval1(bid, value, |v| eval_bit_range(v, offset, width)),
            ScalarFold::Not { value } => self.eval1(bid, value, eval_not),
            ScalarFold::Select { cond, if_t, if_f } => self.eval_select(bid, cond, if_t, if_f),
        };

        // Every scalar fold is single-result; map keeps that contract without an `unwrap`.
        instr.get_results().map(|r| (*r, value.clone())).collect()
    }

    /// Fold a `Call`'s results through the callee's jump-function summary.
    ///
    /// Only constrained static calls to a summarized function fold; an unconstrained call (results
    /// are advice, not circuit-constrained), a dynamic call (unknown target), or a summary-less /
    /// non-interprocedural solve leaves every result `Bottom`. A `Param(i)` jump function resolves
    /// to the *in-context* constant of argument `i`, so a passthrough callee yields per-call-site
    /// constant results without re-solving the callee.
    fn eval_call(
        &self,
        bid: BlockId,
        function: &CallTarget,
        args: &[ValueId],
        results: &[ValueId],
        unconstrained: bool,
    ) -> Vec<(ValueId, Constness)> {
        let all_bottom = || results.iter().map(|r| (*r, Constness::Bottom)).collect();
        if unconstrained {
            return all_bottom();
        }
        let (Some(summaries), CallTarget::Static(g)) = (self.summaries, function) else {
            return all_bottom();
        };
        let Some(summary) = summaries.get(g) else {
            return all_bottom();
        };
        results
            .iter()
            .enumerate()
            .map(|(j, r)| {
                let c = match summary.returns.get(j) {
                    Some(ReturnJump::Const(c)) => Constness::Const(c.clone()),
                    Some(ReturnJump::Param(i)) if *i < args.len() => {
                        self.lattice_in_block(bid, args[*i])
                    }
                    _ => Constness::Bottom,
                };
                (*r, c)
            })
            .collect()
    }

    /// Evaluate a 1-argument function `f` on `v` in the context of `bid`.
    fn eval1(
        &self,
        bid: BlockId,
        v: ValueId,
        f: impl FnOnce(&Constant) -> Option<Constant>,
    ) -> Constness {
        match self.lattice_in_block(bid, v) {
            Constness::Top => Constness::Top,
            Constness::Const(c) => f(&c)
                .map(|c| Constness::Const(Arc::new(c)))
                .unwrap_or(Constness::Bottom),
            Constness::Bottom => Constness::Bottom,
        }
    }

    /// Evaluate a 2-argument function `f` on `l, r` in the context of `bid`.
    fn eval2(
        &self,
        bid: BlockId,
        l: ValueId,
        r: ValueId,
        f: impl FnOnce(&Constant, &Constant) -> Option<Constant>,
    ) -> Constness {
        match (self.lattice_in_block(bid, l), self.lattice_in_block(bid, r)) {
            (Constness::Top, _) | (_, Constness::Top) => Constness::Top,
            (Constness::Const(a), Constness::Const(b)) => f(&a, &b)
                .map(|c| Constness::Const(Arc::new(c)))
                .unwrap_or(Constness::Bottom),
            (Constness::Bottom, _) | (_, Constness::Bottom) => Constness::Bottom,
        }
    }

    /// Evaluate a variadic aggregate op `f` over `operands` in the context of `bid`.
    ///
    /// `f` receives the operands' constants by value (already collected), so a constructor can move
    /// them into the result blob rather than cloning them a second time. Any number of operands is
    /// accepted, so this also serves the fixed-arity ops (e.g. `ArraySet`'s three).
    fn eval_n(
        &self,
        bid: BlockId,
        operands: &[ValueId],
        f: impl FnOnce(Vec<Constant>) -> Option<Constant>,
    ) -> Constness {
        let mut consts = Vec::with_capacity(operands.len());
        let mut saw_bottom = false;
        for v in operands {
            match self.lattice_in_block(bid, *v) {
                Constness::Top => return Constness::Top,
                Constness::Bottom => saw_bottom = true,
                Constness::Const(c) => consts.push((*c).clone()),
            }
        }
        if saw_bottom {
            return Constness::Bottom;
        }
        f(consts)
            .map(|c| Constness::Const(Arc::new(c)))
            .unwrap_or(Constness::Bottom)
    }

    /// The transfer for the pure, value-semantic sequence ops, folded over aggregate (`Blob`)
    /// constants in the lattice, or `None` if `instr` is not one of those ops.
    ///
    /// The result is a single value. An aggregate constructor folds to a `Const(Blob)` only when
    /// every element is constant; a projection (`ArrayGet`/`SliceLen`) folds to a scalar. Operands
    /// that are not yet constant propagate `Top`/`Bottom` exactly as the scalar folds do.
    fn eval_aggregate(&self, bid: BlockId, instr: &OpCode) -> Option<Constness> {
        let value = match instr {
            OpCode::MkSeq {
                elems, elem_type, ..
            } => self.eval_n(bid, elems, |cs| eval_mk_seq(elem_type, cs)),
            OpCode::MkRepeated {
                element,
                count,
                elem_type,
                ..
            } => self.eval1(bid, *element, |e| eval_mk_repeated(elem_type, e, *count)),

            // `MkSeqOfBlob` views its blob operand as a sequence; the operand already *is* the
            // aggregate constant so we can share the Arc cheaply.
            OpCode::MkSeqOfBlob { blob, .. } => match self.lattice_in_block(bid, *blob) {
                Constness::Const(c) if matches!(&*c, Constant::Blob(_)) => Constness::Const(c),
                Constness::Const(_) => Constness::Bottom,
                other => other,
            },
            OpCode::ArrayGet { array, index, .. } => {
                self.eval2(bid, *array, *index, eval_array_get)
            }
            OpCode::ArraySet {
                array,
                index,
                value,
                ..
            } => self.eval_n(bid, &[*array, *index, *value], |cs| {
                let [array, index, value]: [Constant; 3] = cs
                    .try_into()
                    .expect("ArraySet folds exactly three constant operands");
                eval_array_set(array, &index, value)
            }),
            OpCode::SlicePush {
                dir, slice, values, ..
            } => {
                let operands: Vec<ValueId> = std::iter::once(*slice)
                    .chain(values.iter().copied())
                    .collect();

                self.eval_n(bid, &operands, |mut cs| {
                    let values = cs.split_off(1);
                    let slice = cs.pop().expect("SlicePush has a slice operand");
                    eval_slice_push(*dir, slice, values)
                })
            }
            OpCode::SliceLen { slice, .. } => self.eval1(bid, *slice, eval_slice_len),
            _ => return None,
        };
        Some(value)
    }

    fn eval_select(&self, bid: BlockId, cond: ValueId, if_t: ValueId, if_f: ValueId) -> Constness {
        match self.lattice_in_block(bid, cond) {
            Constness::Top => Constness::Top,
            Constness::Const(c) => match const_bool(&c) {
                Some(true) => self.lattice_in_block(bid, if_t),
                Some(false) => self.lattice_in_block(bid, if_f),
                None => Constness::Bottom,
            },
            Constness::Bottom => const_join(
                self.lattice_in_block(bid, if_t),
                self.lattice_in_block(bid, if_f),
            ),
        }
    }
}

// COMBINED-FIXPOINT WRITEBACK
// ================================================================================================

/// Solve `function` to the combined-fixpoint writeback.
///
/// Each round solves the function, then folds every comparison of congruent operands to a constant
/// (see [`derive_promotions`]) and accumulates the new must-equal facts into `promotions`. If that
/// set grew, the next round re-solves with them — which prunes branches and cascades through
/// reachability and congruence — otherwise the combined fixpoint is reached.
///
/// The first round runs with an empty `promotions`, so it is a plain base solve. The loop is
/// monotone (promotions only grow, executable edges only shrink, congruence only coarsens) over
/// finite domains, so it converges; [`MAX_WRITEBACK_ROUNDS`] is only a backstop.
///
/// `summaries`/`param_seeds` select the solve mode and are layered with the accumulating promotions
/// identically every round:
///
/// - **Intraprocedural:** `summaries = None`, empty `param_seeds`.
/// - **Polymorphic Summary Solve:** `summaries = Some`, empty `param_seeds`;
/// - **Context-Specialized Solve:** `summaries = Some`, the context's seeds.
pub(crate) fn solve_with_writeback(
    function: &HLFunction,
    consts: &HLSSAConstantsSnapshot,
    det: &DetSummaries,
    summaries: Option<&HashMap<FunctionId, FnSummary>>,
    sym_summaries: Option<&SymSummaries>,
    param_seeds: &HashMap<ValueId, Constness>,
) -> FunctionFacts {
    // The accumulator starts empty, so the first iteration is a plain base solve:
    // `with_promotions(&{})` is byte-identical to passing no promotions (every query short-circuits
    // to `None`, `set_lattice`'s guard is `false`, and `into_facts` materialises nothing).
    let mut promotions: HashMap<ValueId, Arc<Constant>> = HashMap::default();
    let mut facts = None;
    let mut converged = false;
    for _ in 0..MAX_WRITEBACK_ROUNDS {
        let mut solver = FunctionSolver::new(function, consts)
            .with_determinism(det)
            .with_param_seeds(param_seeds.clone())
            .with_promotions(&promotions);

        if let Some(summaries) = summaries {
            solver = solver.with_summaries(summaries);
        }

        if let Some(sym_summaries) = sym_summaries {
            solver = solver.with_sym_summaries(sym_summaries);
        }

        solver.run();
        let solved = solver.into_facts();

        // Fold every newly-foldable comparison of congruent operands; re-solve only if the set
        // grew, otherwise the combined fixpoint is reached.
        let before = promotions.len();
        for (v, c) in derive_promotions(function, &solved) {
            promotions.entry(v).or_insert(c);
        }
        converged = promotions.len() == before;
        facts = Some(solved);
        if converged {
            break;
        }
    }

    debug_assert!(
        converged,
        "ClickCooper combined-fixpoint writeback did not converge within {MAX_WRITEBACK_ROUNDS} \
         rounds for `{}` (expected 1–2)",
        function.get_name(),
    );
    facts.expect("the writeback loop always runs at least one round")
}

/// The congruence-derived must-equal constants in `facts`: a comparison of two provably-congruent
/// operands folds to a constant the constants lattice alone cannot derive.
///
/// - `CmpEq(x, y)` with `known_equal(x, y)` is unconditionally `true` (`x == y` in every run).
/// - `CmpLt(x, y)` with `known_equal(x, y)` is unconditionally `false` (`x < x` is never true).
///
/// Both are *must-equal* facts of the same strength as congruence itself — congruence never unifies
/// two free witnesses — so feeding them back into the lattice (where they fold branches and
/// cascade) keeps the analysis sound. Only the comparison *result* is promoted; it is a pure scalar
/// `Cmp` fold, so a consumer may alias and delete it. Results already proven constant are skipped,
/// so the outer fixpoint converges once no new fact appears.
///
/// A `WitnessOf`-typed comparison result (a witnessed comparison from witness spilling / AD
/// lowering) is promoted just like any other when vacuous — the must-equal fact holds either way —
/// but the consumer needs to keep the substituted constant witness-typed.
fn derive_promotions(
    function: &HLFunction,
    facts: &FunctionFacts,
) -> HashMap<ValueId, Arc<Constant>> {
    let mut out: HashMap<ValueId, Arc<Constant>> = HashMap::default();
    for (bid, block) in function.get_blocks() {
        if !facts.reachable.contains(bid) {
            continue;
        }
        for instr in block.get_instructions() {
            let folded = match instr.scalar_fold() {
                Some(ScalarFold::Cmp {
                    kind: CmpKind::Eq,
                    lhs,
                    rhs,
                }) if facts.congruence.known_equal(lhs, rhs) => true,
                Some(ScalarFold::Cmp {
                    kind: CmpKind::Lt,
                    lhs,
                    rhs,
                }) if facts.congruence.known_equal(lhs, rhs) => false,
                _ => continue,
            };

            // A `Cmp` is single-result; skip any result the lattice already proved constant so the
            // promotion set only ever grows with genuinely new facts.
            for r in instr.get_results() {
                if matches!(facts.lattice.get(r), Some(Constness::Const(_))) {
                    continue;
                }
                out.insert(*r, bool_constant(folded));
            }
        }
    }
    out
}
