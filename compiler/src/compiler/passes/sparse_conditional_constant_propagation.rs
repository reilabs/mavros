//! Sparse conditional constant propagation (Wegman–Zadeck) over the HLSSA.
//!
//! Runs the classic optimistic worklist algorithm per function: every value starts at ⊤
//! (unvisited), is lowered to a concrete constant when a transfer function can fold it, and bottoms
//! out at ⊥ (overdefined) otherwise. Block reachability and value lattices are computed together,
//! so constants are propagated through branches that are themselves decided by constants — strictly
//! stronger than folding and branch pruning as separate passes.
//!
//! In addition to the global SCCP value lattice, the analysis tracks branch-edge constant facts: on
//! the true edge of `JmpIf(c, t, f)`, `c` is true; on the false edge, `c` is false. On the true edge
//! of `JmpIf(c, t, f)` where `c` is `x == K`, it also records `x = K`. Facts are intersected at
//! joins, so they only survive while every executable incoming edge agrees. This captures
//! path-sensitive propagation without pretending an edge-local fact is globally constant.
//!
//! After the lattice converges, the rewrite:
//!
//! - replaces every use of a constant-valued instruction result or block parameter with the
//!   interned constant, dropping the (pure, by construction) defining instruction,
//! - aliases `Select`s with a constant condition to the chosen arm even when the arms are not
//!   constants,
//! - rewrites `JmpIf`s with a constant condition into `Jmp`s to the live successor,
//! - deletes blocks the analysis proved unreachable.
//!
//! Constant-valued block parameters are aliased but left in place; DCE will prune the now-unused
//! parameters and their jump arguments.
//!
//! # Folding Semantics
//!
//! Only *pure scalar* values are folded, and only when every operand is a pure interned constant —
//! a `WitnessOf`-typed value is never an interned constant, so witness arithmetic is never touched
//! and result types never change. Integer folds are deliberately conservative: an op is folded
//! only when the mathematical result fits the operand bit width. An overflowing pure integer op is
//! an erroneous evaluation, not a value: where overflow is observable the lowering guards it with
//! explicit checks (see `pure_guards`), and the residue an op leaves behind past those checks is
//! backend-specific (machine integers wrap at a register width, R1CS computes in the field) and
//! deliberately not part of the IR semantics. Folding an overflowed result would bake one
//! backend's residue into all of them, so the op is instead left in place to fail (or wrap)
//! exactly as it would have at runtime. Division by a constant zero is likewise left alone.

use std::sync::{Arc, OnceLock};

use ark_ff::{PrimeField, Zero};

use crate::{
    collections::{HashMap, HashSet},
    compiler::{
        Field,
        pass_manager::{AnalysisStore, Pass},
        passes::fix_double_jumps::ValueReplacements,
        ssa::{
            BlockId, Instruction, Terminator, ValueId,
            hlssa::{
                BinaryArithOpKind, CastTarget, CmpKind, Constant, HLFunction, HLSSA,
                HLSSAConstantsSnapshot, MAX_SUPPORTED_SIGNED_BITS, OpCode,
            },
        },
        util::{bit_mask, decode_signed, encode_signed, fits_signed},
    },
};

pub struct SCCP {}

impl Pass for SCCP {
    fn name(&self) -> &'static str {
        "sccp"
    }

    fn run(&self, ssa: &mut HLSSA, _store: &AnalysisStore) {
        self.do_run(ssa);
    }
}

impl SCCP {
    pub fn new() -> Self {
        Self {}
    }

    pub fn do_run(&self, ssa: &mut HLSSA) {
        // One snapshot serves all functions: a constant interned by the rewrite of an earlier
        // function is only ever referenced from that (already rewritten) function, so a later
        // function's analysis never needs to see it.
        let consts = ssa.const_snapshot();
        let fids: Vec<_> = ssa.get_function_ids().collect();
        for fid in fids {
            let mut function = ssa.take_function(fid);
            let result = {
                let mut analysis = FunctionLattice::new(&function, &consts);
                analysis.run();
                analysis.into_result()
            };
            rewrite(&mut function, ssa, &result);
            ssa.put_function(fid, function);
        }
    }
}

// LATTICE
// ================================================================================================

/// The abstract value of a single SSA value: one element of the constant-propagation lattice.
/// Constants are `Arc`-shared with the SSA's interning table, so cloning an element is cheap.
#[derive(Clone, Debug, PartialEq)]
enum LatticeElement {
    /// Not (yet) known to be reachable with any value.
    Top,
    /// Proven to always hold this constant.
    Const(Arc<Constant>),
    /// Overdefined: holds a runtime-dependent (or non-foldable) value.
    Bottom,
}

fn join(a: LatticeElement, b: LatticeElement) -> LatticeElement {
    match (a, b) {
        (LatticeElement::Top, x) | (x, LatticeElement::Top) => x,
        (LatticeElement::Bottom, _) | (_, LatticeElement::Bottom) => LatticeElement::Bottom,
        (LatticeElement::Const(c1), LatticeElement::Const(c2)) => {
            if c1 == c2 {
                LatticeElement::Const(c1)
            } else {
                LatticeElement::Bottom
            }
        }
    }
}

/// The converged per-function analysis state handed to the rewrite.
struct LatticeResult<'c> {
    consts: &'c HLSSAConstantsSnapshot,
    lattice: HashMap<ValueId, LatticeElement>,
    reachable: HashSet<BlockId>,
    instruction_facts: HashMap<(BlockId, usize), Facts>,
    block_exit_facts: HashMap<BlockId, Facts>,
}

impl LatticeResult<'_> {
    fn lookup_at_instruction(&self, bid: BlockId, idx: usize, v: ValueId) -> LatticeElement {
        lookup_value(
            self.consts,
            &self.lattice,
            self.instruction_facts.get(&(bid, idx)),
            v,
        )
    }

    fn lookup_at_exit(&self, bid: BlockId, v: ValueId) -> LatticeElement {
        lookup_value(
            self.consts,
            &self.lattice,
            self.block_exit_facts.get(&bid),
            v,
        )
    }
}

type Facts = HashMap<ValueId, Fact>;

#[derive(Clone, Debug, PartialEq)]
struct Fact {
    constant: Arc<Constant>,
    source: FactSource,
}

impl Fact {
    fn new(constant: Arc<Constant>, source: FactSource) -> Self {
        Self { constant, source }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct FactSource(u8);

impl FactSource {
    const BRANCH: Self = Self(1 << 0);
    const ASSERT: Self = Self(1 << 1);

    fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    fn intersection(self, other: Self) -> Self {
        Self(self.0 & other.0)
    }

    fn contains_branch(self) -> bool {
        self.0 & Self::BRANCH.0 != 0
    }
}

fn lookup_value(
    consts: &HLSSAConstantsSnapshot,
    lattice: &HashMap<ValueId, LatticeElement>,
    facts: Option<&Facts>,
    v: ValueId,
) -> LatticeElement {
    if let Some(c) = consts.get(&v) {
        return LatticeElement::Const(c.clone());
    }
    if let Some(fact) = facts.and_then(|facts| facts.get(&v)) {
        return LatticeElement::Const(fact.constant.clone());
    }
    lattice.get(&v).cloned().unwrap_or(LatticeElement::Top)
}

fn set_facts<K: Eq + std::hash::Hash>(map: &mut HashMap<K, Facts>, key: K, facts: Facts) -> bool {
    let changed = if facts.is_empty() {
        map.contains_key(&key)
    } else {
        map.get(&key) != Some(&facts)
    };
    if !changed {
        return false;
    }

    if facts.is_empty() {
        map.remove(&key);
    } else {
        map.insert(key, facts);
    }
    true
}

fn insert_fact(facts: &mut Facts, value: ValueId, constant: Arc<Constant>, source: FactSource) {
    match facts.get_mut(&value) {
        Some(fact) if fact.constant == constant => {
            fact.source = fact.source.union(source);
        }
        _ => {
            facts.insert(value, Fact::new(constant, source));
        }
    }
}

fn branch_propagated_facts(facts: Option<&Facts>) -> Facts {
    facts
        .into_iter()
        .flat_map(|facts| facts.iter())
        .filter_map(|(value, fact)| {
            fact.source
                .contains_branch()
                .then(|| (*value, Fact::new(fact.constant.clone(), FactSource::BRANCH)))
        })
        .collect()
}

// ANALYSIS
// ================================================================================================

struct FunctionLattice<'f, 'c> {
    function: &'f HLFunction,
    consts: &'c HLSSAConstantsSnapshot,
    lattice: HashMap<ValueId, LatticeElement>,

    /// CFG edges proven executable, and the executable predecessors per block (deduplicated by the
    /// edge set).
    exec_edges: HashSet<(BlockId, BlockId)>,
    exec_preds: HashMap<BlockId, Vec<BlockId>>,

    /// Blocks whose instructions have been visited at least once.
    reachable: HashSet<BlockId>,

    /// Constant facts known at block entry. A fact appears here only when every executable
    /// incoming edge proves the same value.
    block_facts: HashMap<BlockId, Facts>,

    /// Constant facts known before a given instruction.
    instruction_facts: HashMap<(BlockId, usize), Facts>,

    /// Constant facts known after a block's instructions, before its terminator.
    block_exit_facts: HashMap<BlockId, Facts>,

    edge_worklist: Vec<(BlockId, BlockId)>,
    value_worklist: Vec<ValueId>,
    fact_worklist: Vec<BlockId>,
    queued_fact_blocks: HashSet<BlockId>,

    /// For each value: the sites that read it. `None` marks the block's terminator.
    uses: HashMap<ValueId, Vec<(BlockId, Option<usize>)>>,

    /// For instruction results: the instruction that defines the value.
    defs: HashMap<ValueId, (BlockId, usize)>,
}

impl<'f, 'c> FunctionLattice<'f, 'c> {
    fn new(function: &'f HLFunction, consts: &'c HLSSAConstantsSnapshot) -> Self {
        let mut uses: HashMap<ValueId, Vec<(BlockId, Option<usize>)>> = HashMap::default();
        let mut defs: HashMap<ValueId, (BlockId, usize)> = HashMap::default();
        for (bid, block) in function.get_blocks() {
            for (idx, instr) in block.get_instructions().enumerate() {
                for result in instr.get_results() {
                    defs.insert(*result, (*bid, idx));
                }
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
        let mut this = Self {
            function,
            consts,
            lattice: HashMap::default(),
            exec_edges: HashSet::default(),
            exec_preds: HashMap::default(),
            reachable: HashSet::default(),
            block_facts: HashMap::default(),
            instruction_facts: HashMap::default(),
            block_exit_facts: HashMap::default(),
            edge_worklist: Vec::new(),
            value_worklist: Vec::new(),
            fact_worklist: Vec::new(),
            queued_fact_blocks: HashSet::default(),
            uses,
            defs,
        };
        this.add_fact_dependency_uses();
        this
    }

    fn add_fact_dependency_uses(&mut self) {
        for (bid, block) in self.function.get_blocks() {
            for (idx, instr) in block.get_instructions().enumerate() {
                if let OpCode::Assert { value } = instr {
                    for dep in self.bool_fact_dependencies(*value) {
                        self.uses.entry(dep).or_default().push((*bid, Some(idx)));
                    }
                }
            }
            if let Some(Terminator::JmpIf(cond, _, _)) = block.get_terminator() {
                for dep in self.bool_fact_dependencies(*cond) {
                    self.uses.entry(dep).or_default().push((*bid, None));
                }
            }
        }
    }

    fn bool_fact_dependencies(&self, value: ValueId) -> Vec<ValueId> {
        let mut deps = Vec::new();
        let mut seen = HashSet::default();
        self.collect_bool_fact_dependencies(value, &mut seen, &mut deps);
        deps
    }

    fn collect_bool_fact_dependencies(
        &self,
        value: ValueId,
        seen: &mut HashSet<ValueId>,
        deps: &mut Vec<ValueId>,
    ) {
        if !seen.insert(value) {
            return;
        }
        let Some(&(def_bid, idx)) = self.defs.get(&value) else {
            return;
        };
        match self.function.get_block(def_bid).get_instruction(idx) {
            OpCode::Not { value, .. } => {
                deps.push(*value);
                self.collect_bool_fact_dependencies(*value, seen, deps);
            }
            OpCode::Cmp {
                kind: CmpKind::Eq,
                lhs,
                rhs,
                ..
            } => {
                deps.push(*lhs);
                deps.push(*rhs);
            }
            _ => {}
        }
    }

    fn into_result(self) -> LatticeResult<'c> {
        LatticeResult {
            consts: self.consts,
            lattice: self.lattice,
            reachable: self.reachable,
            instruction_facts: self.instruction_facts,
            block_exit_facts: self.block_exit_facts,
        }
    }

    fn run(&mut self) {
        let entry = self.function.get_entry_id();
        // Entry parameters are the function's arguments: unknown at this (intraprocedural) level.
        for (p, _) in self.function.get_entry().get_parameters() {
            self.lattice.insert(*p, LatticeElement::Bottom);
        }
        self.reachable.insert(entry);
        self.block_facts.insert(entry, Facts::default());
        self.visit_block(entry);

        loop {
            if let Some((p, b)) = self.edge_worklist.pop() {
                self.process_edge(p, b);
                continue;
            }
            if let Some(b) = self.fact_worklist.pop() {
                self.queued_fact_blocks.remove(&b);
                self.process_facts(b);
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

    /// A `JmpIf` condition can converge at ⊤ only if it (transitively) bottoms out at a block
    /// parameter fed solely by itself along executable edges — a use its definition does not
    /// dominate, i.e. malformed SSA. The rewrite would silently delete blocks the kept `JmpIf`
    /// still targets, so ICE here instead.
    fn assert_no_stuck_conditions(&self) {
        for bid in &self.reachable {
            if let Some(Terminator::JmpIf(cond, _, _)) =
                self.function.get_block(*bid).get_terminator()
            {
                assert!(
                    self.lattice_at_block_exit(*bid, *cond) != LatticeElement::Top,
                    "ICE: SCCP converged with JmpIf condition v{} in `{}` stuck at ⊤; \
                     the condition is only defined through a self-referential block-parameter \
                     cycle, so a use is not dominated by its definition (malformed SSA)",
                    cond.0,
                    self.function.get_name(),
                );
            }
        }
    }

    fn lattice_of(&self, v: ValueId) -> LatticeElement {
        self.lookup_with_facts(None, v)
    }

    fn lattice_at_instruction(&self, bid: BlockId, idx: usize, v: ValueId) -> LatticeElement {
        self.lookup_with_facts(self.instruction_facts.get(&(bid, idx)), v)
    }

    fn lattice_at_block_exit(&self, pred: BlockId, v: ValueId) -> LatticeElement {
        self.lookup_with_facts(self.block_exit_facts.get(&pred), v)
    }

    fn lookup_with_facts(&self, facts: Option<&Facts>, v: ValueId) -> LatticeElement {
        lookup_value(self.consts, &self.lattice, facts, v)
    }

    /// Lower `v` to `join(current, new)`, scheduling its users if the value changed.
    fn set_lattice(&mut self, v: ValueId, new: LatticeElement) {
        let old = self.lattice_of(v);
        let joined = join(old.clone(), new);
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
        self.recompute_params(b);
        let facts_changed = self.recompute_facts(b);
        let first_visit = self.reachable.insert(b);
        if first_visit || facts_changed {
            self.visit_block(b);
        }
    }

    fn process_facts(&mut self, b: BlockId) {
        if self.recompute_facts(b) && self.reachable.contains(&b) {
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
                Some(idx) => {
                    if self.instruction_updates_facts(bid, idx) {
                        self.visit_block(bid);
                    } else {
                        self.visit_instruction(bid, idx);
                    }
                }
                None => self.revisit_terminator_for(bid, v),
            }
        }
    }

    /// Re-examine `bid`'s terminator because its operand `v` changed. A `Jmp` along an
    /// already-executable edge re-joins only the target parameters that actually consume `v`
    /// (per-phi), not every parameter of the target; other terminators take the generic path.
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
                    self.recompute_params_at(t, &indices);
                }
                // Otherwise the edge is still queued and `process_edge` will join all parameters
                // when it runs.
            }
            Some(Terminator::JmpIf(_, _, _)) => {
                self.visit_terminator(bid);
                self.enqueue_successor_facts(bid);
            }
            _ => self.visit_terminator(bid),
        }
    }

    fn visit_block(&mut self, bid: BlockId) {
        let n = self.function.get_block(bid).get_instructions().count();
        let mut facts = self.block_facts.get(&bid).cloned().unwrap_or_default();
        for idx in 0..n {
            self.set_instruction_facts(bid, idx, facts.clone());
            self.visit_instruction(bid, idx);
            let instr = self.function.get_block(bid).get_instruction(idx).clone();
            self.add_instruction_facts(&mut facts, &instr);
        }
        if self.set_block_exit_facts(bid, facts) {
            self.enqueue_successor_facts(bid);
        }
        self.visit_terminator(bid);
    }

    fn visit_instruction(&mut self, bid: BlockId, idx: usize) {
        let updates = self.transfer(bid, idx, self.function.get_block(bid).get_instruction(idx));
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
            Terminator::JmpIf(cond, t, f) => match self.lattice_at_block_exit(bid, cond) {
                LatticeElement::Top => {}
                LatticeElement::Const(c) => match const_bool(&c) {
                    Some(true) => self.notify_edge(bid, t),
                    Some(false) => self.notify_edge(bid, f),
                    None => {
                        self.notify_edge(bid, t);
                        self.notify_edge(bid, f);
                    }
                },
                LatticeElement::Bottom => {
                    self.notify_edge(bid, t);
                    self.notify_edge(bid, f);
                }
            },
            Terminator::Return(_) => {}
        }
    }

    /// Mark edge `(p, t)` executable; if it already is, the jump arguments may have changed, so
    /// re-join the target's parameters. Fact propagation is driven separately by block-exit and
    /// branch-fact dependency changes.
    fn notify_edge(&mut self, p: BlockId, t: BlockId) {
        if self.exec_edges.contains(&(p, t)) {
            self.recompute_params(t);
        } else {
            self.edge_worklist.push((p, t));
        }
    }

    /// Join each parameter of `b` over the arguments along all executable in-edges.
    fn recompute_params(&mut self, b: BlockId) {
        let n = self.function.get_block(b).get_parameter_values().count();
        let all: Vec<usize> = (0..n).collect();
        self.recompute_params_at(b, &all);
    }

    /// Join the parameters of `b` at `indices` over the arguments along all executable in-edges.
    /// Restricting the join to the touched indices keeps a single jump-argument change from
    /// re-joining every parameter of the target (the classic per-phi formulation).
    fn recompute_params_at(&mut self, b: BlockId, indices: &[usize]) {
        if indices.is_empty() || b == self.function.get_entry_id() {
            return;
        }
        let block = self.function.get_block(b);
        if !block.has_parameters() {
            return;
        }
        let params: Vec<ValueId> = block.get_parameter_values().copied().collect();
        let preds = self.exec_preds.get(&b).cloned().unwrap_or_default();
        let pred_edge_facts: Vec<_> = preds
            .iter()
            .map(|pred| (*pred, self.edge_facts(*pred, b)))
            .collect();
        let mut updates = Vec::with_capacity(indices.len());
        for &i in indices {
            let mut acc = LatticeElement::Top;
            for (pred, facts) in &pred_edge_facts {
                match self.function.get_block(*pred).get_terminator() {
                    Some(Terminator::Jmp(t, args)) if *t == b => {
                        acc = join(acc, self.lookup_with_facts(Some(facts), args[i]));
                    }
                    // A parameterized block is only ever entered via `Jmp`; treat anything else
                    // as unknown rather than trusting the invariant.
                    _ => acc = LatticeElement::Bottom,
                }
            }
            updates.push((params[i], acc));
        }
        for (p, lat) in updates {
            self.set_lattice(p, lat);
        }
    }

    fn instruction_updates_facts(&self, bid: BlockId, idx: usize) -> bool {
        matches!(
            self.function.get_block(bid).get_instruction(idx),
            OpCode::Assert { .. }
                | OpCode::AssertCmp {
                    kind: CmpKind::Eq,
                    ..
                }
        )
    }

    fn set_instruction_facts(&mut self, bid: BlockId, idx: usize, facts: Facts) {
        set_facts(&mut self.instruction_facts, (bid, idx), facts);
    }

    fn set_block_exit_facts(&mut self, bid: BlockId, facts: Facts) -> bool {
        set_facts(&mut self.block_exit_facts, bid, facts)
    }

    fn set_block_facts(&mut self, bid: BlockId, facts: Facts) -> bool {
        set_facts(&mut self.block_facts, bid, facts)
    }

    fn enqueue_fact_recompute(&mut self, bid: BlockId) {
        if bid == self.function.get_entry_id() {
            return;
        }
        if self.queued_fact_blocks.insert(bid) {
            self.fact_worklist.push(bid);
        }
    }

    fn enqueue_successor_facts(&mut self, bid: BlockId) {
        let Some(term) = self.function.get_block(bid).get_terminator() else {
            return;
        };
        let mut successors = Vec::new();
        match term {
            Terminator::Jmp(t, _) => successors.push(*t),
            Terminator::JmpIf(_, t, f) => {
                successors.push(*t);
                if f != t {
                    successors.push(*f);
                }
            }
            Terminator::Return(_) => {}
        }
        for succ in successors {
            if self.exec_edges.contains(&(bid, succ)) {
                self.enqueue_fact_recompute(succ);
            }
        }
    }

    fn recompute_facts(&mut self, b: BlockId) -> bool {
        if b == self.function.get_entry_id() {
            return false;
        }
        let Some(preds) = self.exec_preds.get(&b) else {
            return false;
        };
        let facts = match preds.as_slice() {
            [] => return false,
            [pred] => self.edge_facts(*pred, b),
            [first_pred, rest @ ..] => {
                let mut facts = self.edge_facts(*first_pred, b);
                for pred in rest {
                    let next = self.edge_facts(*pred, b);
                    facts.retain(|value, known| {
                        let Some(next) = next.get(value) else {
                            return false;
                        };
                        if next.constant != known.constant {
                            return false;
                        }
                        known.source = known.source.intersection(next.source);
                        true
                    });
                    if facts.is_empty() {
                        break;
                    }
                }
                facts
            }
        };

        self.set_block_facts(b, facts)
    }

    fn add_instruction_facts(&self, facts: &mut Facts, instr: &OpCode) {
        match instr {
            OpCode::Assert { value } => {
                self.add_bool_value_facts(facts, *value, true, FactSource::ASSERT);
            }
            OpCode::AssertCmp {
                kind: CmpKind::Eq,
                lhs,
                rhs,
            } => self.add_equality_operand_facts(facts, *lhs, *rhs, FactSource::ASSERT),
            OpCode::AssertCmp {
                kind: CmpKind::Lt, ..
            } => {}
            _ => {}
        }
    }

    fn edge_facts(&self, pred: BlockId, target: BlockId) -> Facts {
        let mut facts = branch_propagated_facts(self.block_exit_facts.get(&pred));
        self.add_branch_facts_for_edge(pred, target, &mut facts);
        facts
    }

    fn add_branch_facts_for_edge(&self, pred: BlockId, target: BlockId, facts: &mut Facts) {
        let Some(Terminator::JmpIf(cond, then_b, else_b)) =
            self.function.get_block(pred).get_terminator()
        else {
            return;
        };
        if then_b == else_b {
            return;
        }
        let Some(taken) = (if *then_b == target {
            Some(true)
        } else if *else_b == target {
            Some(false)
        } else {
            None
        }) else {
            return;
        };

        self.add_bool_value_facts(facts, *cond, taken, FactSource::BRANCH);
    }

    fn add_bool_value_facts(
        &self,
        facts: &mut Facts,
        value: ValueId,
        truth: bool,
        source: FactSource,
    ) {
        let mut seen = HashSet::default();
        self.add_bool_value_facts_inner(facts, value, truth, source, &mut seen);
    }

    fn add_bool_value_facts_inner(
        &self,
        facts: &mut Facts,
        value: ValueId,
        truth: bool,
        source: FactSource,
        seen: &mut HashSet<ValueId>,
    ) {
        if !seen.insert(value) {
            return;
        }
        if self.consts.get(&value).is_none() {
            insert_fact(facts, value, bool_constant(truth), source);
        }

        let Some(&(def_bid, idx)) = self.defs.get(&value) else {
            return;
        };
        match self.function.get_block(def_bid).get_instruction(idx) {
            OpCode::Not { value, .. } => {
                self.add_bool_value_facts_inner(facts, *value, !truth, source, seen);
            }
            OpCode::Cmp {
                kind: CmpKind::Eq,
                lhs,
                rhs,
                ..
            } if truth => {
                self.add_equality_operand_facts(facts, *lhs, *rhs, source);
            }
            _ => {}
        }
    }

    fn add_equality_operand_facts(
        &self,
        facts: &mut Facts,
        lhs: ValueId,
        rhs: ValueId,
        source: FactSource,
    ) {
        if let Some((value, known)) = self
            .constant_side_refinement(facts, lhs, rhs)
            .or_else(|| self.constant_side_refinement(facts, rhs, lhs))
        {
            insert_fact(facts, value, known, source);
        }
    }

    fn constant_side_refinement(
        &self,
        facts: &Facts,
        maybe_const: ValueId,
        value: ValueId,
    ) -> Option<(ValueId, Arc<Constant>)> {
        if self.consts.get(&value).is_some() {
            return None;
        }
        match self.lookup_with_facts(Some(facts), maybe_const) {
            LatticeElement::Const(c) => Some((value, c)),
            LatticeElement::Top | LatticeElement::Bottom => None,
        }
    }

    /// The transfer function: new lattice values for the instruction's results.
    fn transfer(&self, bid: BlockId, idx: usize, instr: &OpCode) -> Vec<(ValueId, LatticeElement)> {
        match instr {
            OpCode::BinaryArithOp {
                kind,
                result,
                lhs,
                rhs,
            } => vec![(
                *result,
                self.eval2(bid, idx, *lhs, *rhs, |a, b| eval_binary(*kind, a, b)),
            )],
            OpCode::Cmp {
                kind,
                result,
                lhs,
                rhs,
            } => vec![(
                *result,
                self.eval2(bid, idx, *lhs, *rhs, |a, b| eval_cmp(*kind, a, b)),
            )],
            OpCode::MulConst {
                result,
                const_val,
                var,
            } => vec![(
                *result,
                self.eval2(bid, idx, *const_val, *var, eval_field_mul),
            )],
            OpCode::Cast {
                result,
                value,
                target,
            } => vec![(
                *result,
                self.eval1(bid, idx, *value, |v| eval_cast(target, v)),
            )],
            OpCode::SExt {
                result,
                value,
                from_bits,
                to_bits,
            } => vec![(
                *result,
                self.eval1(bid, idx, *value, |v| eval_sext(v, *from_bits, *to_bits)),
            )],
            OpCode::BitRange {
                result,
                value,
                offset,
                width,
            } => vec![(
                *result,
                self.eval1(bid, idx, *value, |v| eval_bit_range(v, *offset, *width)),
            )],
            OpCode::Not { result, value } => {
                vec![(*result, self.eval1(bid, idx, *value, eval_not))]
            }
            OpCode::Select {
                result,
                cond,
                if_t,
                if_f,
            } => vec![(*result, self.eval_select(bid, idx, *cond, *if_t, *if_f))],
            // Everything else (memory, witness ops, calls, asserts, sequences, ...) is
            // overdefined. The rewrite relies on this: a constant-valued instruction result is
            // always the output of one of the pure scalar ops above. Spelled out variant by
            // variant (no catch-all) so that adding an opcode forces a decision here.
            OpCode::MkSeq { .. }
            | OpCode::MkSeqOfBlob { .. }
            | OpCode::MkRepeated { .. }
            | OpCode::Alloc { .. }
            | OpCode::Store { .. }
            | OpCode::Load { .. }
            | OpCode::Assert { .. }
            | OpCode::AssertCmp { .. }
            | OpCode::AssertR1C { .. }
            | OpCode::Call { .. }
            | OpCode::ArrayGet { .. }
            | OpCode::ArraySet { .. }
            | OpCode::SlicePush { .. }
            | OpCode::SliceLen { .. }
            | OpCode::ToBits { .. }
            | OpCode::ToRadix { .. }
            | OpCode::MemOp { .. }
            | OpCode::WriteWitness { .. }
            | OpCode::FreshWitness { .. }
            | OpCode::NextDCoeff { .. }
            | OpCode::BumpD { .. }
            | OpCode::Constrain { .. }
            | OpCode::Lookup { .. }
            | OpCode::DLookup { .. }
            | OpCode::Rangecheck { .. }
            | OpCode::ReadGlobal { .. }
            | OpCode::TupleProj { .. }
            | OpCode::TupleRefProj { .. }
            | OpCode::MkTuple { .. }
            | OpCode::Todo { .. }
            | OpCode::InitGlobal { .. }
            | OpCode::DropGlobal { .. }
            | OpCode::Spread { .. }
            | OpCode::Unspread { .. }
            | OpCode::Guard { .. } => instr
                .get_results()
                .map(|r| (*r, LatticeElement::Bottom))
                .collect(),
        }
    }

    fn eval1(
        &self,
        bid: BlockId,
        idx: usize,
        v: ValueId,
        f: impl FnOnce(&Constant) -> Option<Constant>,
    ) -> LatticeElement {
        match self.lattice_at_instruction(bid, idx, v) {
            LatticeElement::Top => LatticeElement::Top,
            LatticeElement::Const(c) => f(&c)
                .map(|c| LatticeElement::Const(Arc::new(c)))
                .unwrap_or(LatticeElement::Bottom),
            LatticeElement::Bottom => LatticeElement::Bottom,
        }
    }

    fn eval2(
        &self,
        bid: BlockId,
        idx: usize,
        l: ValueId,
        r: ValueId,
        f: impl FnOnce(&Constant, &Constant) -> Option<Constant>,
    ) -> LatticeElement {
        match (
            self.lattice_at_instruction(bid, idx, l),
            self.lattice_at_instruction(bid, idx, r),
        ) {
            (LatticeElement::Top, _) | (_, LatticeElement::Top) => LatticeElement::Top,
            (LatticeElement::Const(a), LatticeElement::Const(b)) => f(&a, &b)
                .map(|c| LatticeElement::Const(Arc::new(c)))
                .unwrap_or(LatticeElement::Bottom),
            _ => LatticeElement::Bottom,
        }
    }

    fn eval_select(
        &self,
        bid: BlockId,
        idx: usize,
        cond: ValueId,
        if_t: ValueId,
        if_f: ValueId,
    ) -> LatticeElement {
        match self.lattice_at_instruction(bid, idx, cond) {
            LatticeElement::Top => LatticeElement::Top,
            LatticeElement::Const(c) => match const_bool(&c) {
                Some(true) => self.lattice_at_instruction(bid, idx, if_t),
                Some(false) => self.lattice_at_instruction(bid, idx, if_f),
                None => LatticeElement::Bottom,
            },
            // Unknown condition: the select still folds if both arms agree.
            LatticeElement::Bottom => join(
                self.lattice_at_instruction(bid, idx, if_t),
                self.lattice_at_instruction(bid, idx, if_f),
            ),
        }
    }
}

// CONSTANT EVALUATION
// ================================================================================================

fn const_bool(c: &Constant) -> Option<bool> {
    match c {
        Constant::U(1, 0) => Some(false),
        Constant::U(1, 1) => Some(true),
        _ => None,
    }
}

fn bool_constant(value: bool) -> Arc<Constant> {
    static FALSE: OnceLock<Arc<Constant>> = OnceLock::new();
    static TRUE: OnceLock<Arc<Constant>> = OnceLock::new();
    let slot = if value { &TRUE } else { &FALSE };
    slot.get_or_init(|| Arc::new(Constant::U(1, value as u128)))
        .clone()
}

/// Fold a binary arithmetic op. Integer results must fit the operand width: an overflowing pure op
/// is an erroneous evaluation with a backend-specific residue (see the module docs), so an
/// overflowing fold is refused rather than guessed at.
fn eval_binary(kind: BinaryArithOpKind, a: &Constant, b: &Constant) -> Option<Constant> {
    use BinaryArithOpKind::*;
    match (a, b) {
        (Constant::U(s1, x), Constant::U(s2, y)) => {
            match kind {
                // Shifts are the only ops with legitimately mixed operand widths (the amount is
                // typically a narrow integer), but the type analysis types the result as
                // `U(max(s1, s2))`. A fold to a `U(s1)` constant is therefore only
                // width-preserving when the amount is no wider than the value; refuse the
                // degenerate wider-amount case rather than silently changing the result's width.
                Shl | Shr => {
                    if s2 > s1 {
                        return None;
                    }
                }
                Add | Sub | Mul | Div | Mod | And | Or | Xor => {
                    if s1 != s2 {
                        return None;
                    }
                }
            }
            let s = *s1;
            let v = match kind {
                Add => x.checked_add(*y)?,
                Sub => x.checked_sub(*y)?,
                Mul => x.checked_mul(*y)?,
                Div => {
                    if *y == 0 {
                        return None;
                    }
                    x / y
                }
                Mod => {
                    if *y == 0 {
                        return None;
                    }
                    x % y
                }
                And => x & y,
                Or => x | y,
                Xor => x ^ y,
                // A shift by >= the operand width is not defined consistently across backends.
                Shl | Shr => {
                    if *y >= s as u128 {
                        return None;
                    }
                    match kind {
                        Shl => x.checked_shl(*y as u32)?,
                        _ => x >> (*y as u32),
                    }
                }
            };
            if v > bit_mask(s) {
                return None;
            }
            Some(Constant::U(s, v))
        }
        (Constant::I(s1, x), Constant::I(s2, y)) => {
            if s1 != s2 {
                return None;
            }
            let s = *s1;
            if s == 0 || s > MAX_SUPPORTED_SIGNED_BITS {
                return None;
            }
            match kind {
                And => Some(Constant::I(s, x & y)),
                Or => Some(Constant::I(s, x | y)),
                Xor => Some(Constant::I(s, x ^ y)),
                Add | Sub | Mul | Div | Mod => {
                    let xa = decode_signed(s, *x);
                    let ya = decode_signed(s, *y);
                    let v = match kind {
                        Add => xa.checked_add(ya)?,
                        Sub => xa.checked_sub(ya)?,
                        Mul => xa.checked_mul(ya)?,
                        Div => {
                            if ya == 0 {
                                return None;
                            }
                            xa.checked_div(ya)?
                        }
                        Mod => {
                            if ya == 0 {
                                return None;
                            }
                            xa.checked_rem(ya)?
                        }
                        _ => unreachable!(),
                    };
                    if !fits_signed(s, v) {
                        return None;
                    }
                    Some(Constant::I(s, encode_signed(s, v)))
                }
                Shl | Shr => None,
            }
        }
        (Constant::Field(x), Constant::Field(y)) => match kind {
            Add => Some(Constant::Field(*x + *y)),
            Sub => Some(Constant::Field(*x - *y)),
            Mul => Some(Constant::Field(*x * *y)),
            Div => {
                if y.is_zero() {
                    None
                } else {
                    Some(Constant::Field(*x / *y))
                }
            }
            Mod | And | Or | Xor | Shl | Shr => None,
        },
        // Mixed-kind pairs and non-scalar constants do not fold. Spelled out (no catch-all) so
        // that adding a `Constant` variant forces a decision here.
        (
            Constant::U(..)
            | Constant::I(..)
            | Constant::Field(_)
            | Constant::FnPtr(_)
            | Constant::Blob(_),
            Constant::U(..)
            | Constant::I(..)
            | Constant::Field(_)
            | Constant::FnPtr(_)
            | Constant::Blob(_),
        ) => None,
    }
}

fn eval_cmp(kind: CmpKind, a: &Constant, b: &Constant) -> Option<Constant> {
    let res = |v: bool| Some(Constant::U(1, v as u128));
    match (kind, a, b) {
        (CmpKind::Eq, Constant::U(s1, x), Constant::U(s2, y)) if s1 == s2 => res(x == y),
        (CmpKind::Eq, Constant::I(s1, x), Constant::I(s2, y)) if s1 == s2 => res(x == y),
        (CmpKind::Eq, Constant::Field(x), Constant::Field(y)) => res(x == y),
        (CmpKind::Lt, Constant::U(s1, x), Constant::U(s2, y)) if s1 == s2 => res(x < y),
        (CmpKind::Lt, Constant::I(s1, x), Constant::I(s2, y))
            if s1 == s2 && *s1 >= 1 && *s1 <= MAX_SUPPORTED_SIGNED_BITS =>
        {
            res(decode_signed(*s1, *x) < decode_signed(*s1, *y))
        }
        // Width-mismatched, mixed-kind, and non-scalar comparisons do not fold. Spelled out (no
        // catch-all) so that adding a `Constant` variant forces a decision here.
        (
            CmpKind::Eq | CmpKind::Lt,
            Constant::U(..)
            | Constant::I(..)
            | Constant::Field(_)
            | Constant::FnPtr(_)
            | Constant::Blob(_),
            Constant::U(..)
            | Constant::I(..)
            | Constant::Field(_)
            | Constant::FnPtr(_)
            | Constant::Blob(_),
        ) => None,
    }
}

fn eval_field_mul(a: &Constant, b: &Constant) -> Option<Constant> {
    match (a, b) {
        (Constant::Field(x), Constant::Field(y)) => Some(Constant::Field(*x * *y)),
        (
            Constant::U(..)
            | Constant::I(..)
            | Constant::Field(_)
            | Constant::FnPtr(_)
            | Constant::Blob(_),
            Constant::U(..)
            | Constant::I(..)
            | Constant::Field(_)
            | Constant::FnPtr(_)
            | Constant::Blob(_),
        ) => None,
    }
}

/// HLSSA casts are raw-bits conversions (sign extension is the separate `SExt` op): integers
/// zero-extend into fields, fields truncate to their low bits, and integer-to-integer casts
/// zero-extend or truncate.
fn eval_cast(target: &CastTarget, v: &Constant) -> Option<Constant> {
    match target {
        CastTarget::Nop => Some(v.clone()),
        // WitnessOf wraps the value into a witness type: not a pure constant.
        // ArrayToSlice transfers ownership of a runtime object: nothing to fold.
        // ValueOf/Map operate on witness wrappers and runtime sequences.
        CastTarget::WitnessOf
        | CastTarget::ArrayToSlice
        | CastTarget::ValueOf
        | CastTarget::Map(_) => None,
        CastTarget::Field => match v {
            Constant::U(_, x) | Constant::I(_, x) => Some(Constant::Field(Field::from(*x))),
            Constant::Field(_) => Some(v.clone()),
            Constant::FnPtr(_) | Constant::Blob(_) => None,
        },
        CastTarget::U(n) => int_cast_bits(v, *n).map(|bits| Constant::U(*n, bits)),
        CastTarget::I(n) => {
            if *n > MAX_SUPPORTED_SIGNED_BITS {
                return None;
            }
            int_cast_bits(v, *n).map(|bits| Constant::I(*n, bits))
        }
    }
}

fn int_cast_bits(v: &Constant, n: usize) -> Option<u128> {
    let mask = bit_mask(n);
    match v {
        Constant::U(_, x) | Constant::I(_, x) => Some(x & mask),
        Constant::Field(f) => {
            // Matches the lowering: combine the low limbs of the canonical representation and
            // truncate to the target width.
            let limbs = f.into_bigint().0;
            let low = (limbs[0] as u128) | ((limbs[1] as u128) << 64);
            Some(low & mask)
        }
        Constant::FnPtr(_) | Constant::Blob(_) => None,
    }
}

fn eval_sext(v: &Constant, from_bits: usize, to_bits: usize) -> Option<Constant> {
    if from_bits == 0 || from_bits > to_bits || to_bits > 128 {
        return None;
    }
    let ext = |x: u128| {
        if (x >> (from_bits - 1)) & 1 == 1 {
            x | (bit_mask(to_bits) & !bit_mask(from_bits))
        } else {
            x
        }
    };
    match v {
        Constant::U(_, x) => Some(Constant::U(to_bits, ext(*x))),
        Constant::I(_, x) => Some(Constant::I(to_bits, ext(*x))),
        Constant::Field(_) | Constant::FnPtr(_) | Constant::Blob(_) => None,
    }
}

/// `BitRange` keeps the source type (it is the IR's truncation primitive), so only the payload
/// bits change.
fn eval_bit_range(v: &Constant, offset: usize, width: usize) -> Option<Constant> {
    if offset >= 128 {
        return None;
    }
    match v {
        Constant::U(s, x) => Some(Constant::U(*s, (x >> offset) & bit_mask(width))),
        Constant::I(s, x) => Some(Constant::I(*s, (x >> offset) & bit_mask(width))),
        Constant::Field(_) | Constant::FnPtr(_) | Constant::Blob(_) => None,
    }
}

fn eval_not(v: &Constant) -> Option<Constant> {
    match v {
        Constant::U(s, x) => Some(Constant::U(*s, !x & bit_mask(*s))),
        Constant::I(s, x) => Some(Constant::I(*s, !x & bit_mask(*s))),
        Constant::Field(_) | Constant::FnPtr(_) | Constant::Blob(_) => None,
    }
}

// REWRITE
// ================================================================================================

fn rewrite(function: &mut HLFunction, ssa: &HLSSA, res: &LatticeResult) {
    // Drop blocks the analysis never reached. Every kept terminator only targets reachable
    // blocks: a JmpIf with a constant condition is rewritten to a Jmp to its (reachable) live
    // successor below, and all other terminators had all successor edges marked executable.
    let all_blocks: Vec<BlockId> = function.get_blocks().map(|(id, _)| *id).collect();
    for bid in &all_blocks {
        if !res.reachable.contains(bid) {
            let _ = function.take_block(*bid);
        }
    }
    let assertion_dependencies = assertion_dependencies(function);

    // Alias every constant-valued value (instruction results and block parameters) to the
    // interned constant. Interning a missing constant allocates a fresh value id, so the entries
    // are sorted first: iterating the hash map directly would make the output value ids differ
    // between runs.
    let mut replacements = ValueReplacements::new();
    let mut folded: Vec<(ValueId, &Arc<Constant>)> = res
        .lattice
        .iter()
        .filter_map(|(v, lat)| match lat {
            LatticeElement::Const(c) => Some((*v, c)),
            LatticeElement::Top | LatticeElement::Bottom => None,
        })
        .collect();
    folded.sort_by_key(|(v, _)| v.0);
    for (v, c) in folded {
        replacements.insert(v, ssa.add_const((**c).clone()));
    }

    let kept_blocks: Vec<BlockId> = function.get_blocks().map(|(id, _)| *id).collect();
    for bid in kept_blocks {
        let block = function.get_block_mut(bid);
        let instructions = block.take_instructions();
        let mut kept = Vec::with_capacity(instructions.len());
        for (idx, instr) in instructions.into_iter().enumerate() {
            let local_replacements =
                fact_replacements(ssa, res.instruction_facts.get(&(bid, idx)), FactUse::All);
            // A single-result instruction whose result the lattice proved constant is one of the
            // pure scalar folds (see `transfer`): its uses are aliased, so drop it.
            {
                let mut results = instr.get_results();
                if let (Some(r), None) = (results.next(), results.next()) {
                    if !assertion_dependencies.contains(r)
                        && matches!(res.lattice.get(r), Some(LatticeElement::Const(_)))
                    {
                        continue;
                    }
                }
            }
            // A select with a constant condition aliases to the chosen arm even when the arms are
            // not constants.
            if let OpCode::Select {
                result,
                cond,
                if_t,
                if_f,
            } = &instr
                && !assertion_dependencies.contains(result)
            {
                if let LatticeElement::Const(c) = res.lookup_at_instruction(bid, idx, *cond) {
                    if let Some(b) = const_bool(&c) {
                        replacements.insert(*result, if b { *if_t } else { *if_f });
                        continue;
                    }
                }
            }
            let mut instr = instr;
            if instruction_accepts_fact_replacements(&instr)
                && !instruction_defines_assertion_dependency(&instr, &assertion_dependencies)
            {
                local_replacements.replace_inputs(&mut instr);
            }
            kept.push(instr);
        }
        block.put_instructions(kept);

        if let Some(Terminator::JmpIf(cond, t, f)) = block.get_terminator() {
            let (cond, t, f) = (*cond, *t, *f);
            if let LatticeElement::Const(c) = res.lookup_at_exit(bid, cond) {
                if let Some(b) = const_bool(&c) {
                    let target = if b { t } else { f };
                    block.set_terminator(Terminator::Jmp(target, vec![]));
                }
            }
        }

        let terminator_replacements =
            fact_replacements(ssa, res.block_exit_facts.get(&bid), FactUse::BranchOnly);
        terminator_replacements.replace_terminator(block.get_terminator_mut());
    }

    apply_replacements(function, &replacements, &assertion_dependencies);
}

#[derive(Clone, Copy)]
enum FactUse {
    All,
    BranchOnly,
}

fn fact_replacements(ssa: &HLSSA, facts: Option<&Facts>, fact_use: FactUse) -> ValueReplacements {
    let mut replacements = ValueReplacements::new();
    let Some(facts) = facts else {
        return replacements;
    };

    let mut facts: Vec<_> = facts.iter().collect();
    facts.sort_by_key(|(value, _)| value.0);
    for (value, fact) in facts {
        if matches!(fact_use, FactUse::BranchOnly) && !fact.source.contains_branch() {
            continue;
        }
        replacements.insert(*value, ssa.add_const((*fact.constant).clone()));
    }
    replacements
}

fn instruction_accepts_fact_replacements(instr: &OpCode) -> bool {
    !matches!(instr, OpCode::Assert { .. } | OpCode::AssertCmp { .. })
}

fn instruction_defines_assertion_dependency(
    instr: &OpCode,
    assertion_dependencies: &HashSet<ValueId>,
) -> bool {
    instr
        .get_results()
        .any(|result| assertion_dependencies.contains(result))
}

fn assertion_dependencies(function: &HLFunction) -> HashSet<ValueId> {
    let mut defs: HashMap<ValueId, Vec<ValueId>> = HashMap::default();
    let mut values = HashSet::default();
    for (_, block) in function.get_blocks() {
        for instr in block.get_instructions() {
            let inputs: Vec<ValueId> = instr.get_inputs().copied().collect();
            for result in instr.get_results() {
                defs.insert(*result, inputs.clone());
            }
            match instr {
                OpCode::Assert { value } => {
                    values.insert(*value);
                }
                OpCode::AssertCmp { lhs, rhs, .. } => {
                    values.insert(*lhs);
                    values.insert(*rhs);
                }
                _ => {}
            }
        }
    }

    let mut stack: Vec<ValueId> = values.iter().copied().collect();
    while let Some(value) = stack.pop() {
        let Some(inputs) = defs.get(&value) else {
            continue;
        };
        for input in inputs {
            if values.insert(*input) {
                stack.push(*input);
            }
        }
    }

    values
}

fn apply_replacements(
    function: &mut HLFunction,
    replacements: &ValueReplacements,
    assertion_dependencies: &HashSet<ValueId>,
) {
    for (_, block) in function.get_blocks_mut() {
        for instr in block.get_instructions_mut() {
            if instruction_accepts_fact_replacements(instr)
                && !instruction_defines_assertion_dependency(instr, assertion_dependencies)
            {
                replacements.replace_inputs(instr);
            }
        }
        replacements.replace_terminator(block.get_terminator_mut());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::ssa::hlssa::Type;

    /// `2 + 3 == 5` decides the branch: the comparison chain folds away, the `JmpIf` becomes a
    /// `Jmp` to the then-block, and the else-block is deleted.
    #[test]
    fn folds_constants_and_prunes_dead_branch() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let c2 = ssa.add_const(Constant::U(32, 2));
        let c3 = ssa.add_const(Constant::U(32, 3));
        let c5 = ssa.add_const(Constant::U(32, 5));
        let (sum, is_five) = (ssa.fresh_value(), ssa.fresh_value());

        let f = ssa.get_main_mut();
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

        SCCP::new().do_run(&mut ssa);

        let f = ssa.get_main();
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

        let f = ssa.get_main_mut();
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

        SCCP::new().do_run(&mut ssa);

        let f = ssa.get_main();
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

        let f = ssa.get_main_mut();
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

        SCCP::new().do_run(&mut ssa);

        let f = ssa.get_main();
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

        let f = ssa.get_main_mut();
        let entry = f.get_entry_mut();
        entry.push_instruction(OpCode::BinaryArithOp {
            kind: BinaryArithOpKind::Add,
            result: sum,
            lhs: c200,
            rhs: c100,
        });
        entry.set_terminator(Terminator::Return(vec![sum]));

        SCCP::new().do_run(&mut ssa);

        let f = ssa.get_main();
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

        let f = ssa.get_main_mut();
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

        SCCP::new().do_run(&mut ssa);

        let f = ssa.get_main();
        assert_eq!(f.get_entry().get_instructions().count(), 2);
    }

    /// A degenerate loop: the branch condition is a parameter of a block that is itself only
    /// reachable through that branch, so no executable jump ever supplies the parameter and the
    /// condition converges stuck at ⊤. Such a condition has a use its definition does not
    /// dominate — malformed SSA — and must ICE rather than let the rewrite delete blocks the kept
    /// `JmpIf` still targets.
    #[test]
    #[should_panic(expected = "stuck at ⊤")]
    fn degenerate_loop_ices_on_stuck_condition() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let cond = ssa.fresh_value();

        let f = ssa.get_main_mut();
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

        SCCP::new().do_run(&mut ssa);
    }

    /// A select whose condition is constant aliases to the chosen arm even when the arms are not
    /// constants.
    #[test]
    fn select_with_constant_condition_aliases_to_arm() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let c_true = ssa.add_const(Constant::U(1, 1));
        let (arm_t, arm_f, sel) = (ssa.fresh_value(), ssa.fresh_value(), ssa.fresh_value());

        let f = ssa.get_main_mut();
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

        SCCP::new().do_run(&mut ssa);

        let f = ssa.get_main();
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

        let f = ssa.get_main_mut();
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

        SCCP::new().do_run(&mut ssa);

        let f = ssa.get_main();
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

    /// On the true edge of `x == K`, the compared value is known to be `K`.
    #[test]
    fn branch_equality_fact_folds_compared_value_uses() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let c1 = ssa.add_const(Constant::U(32, 1));
        let c7 = ssa.add_const(Constant::U(32, 7));
        let c8 = ssa.add_const(Constant::U(32, 8));
        let (x, is_seven, sum) = (ssa.fresh_value(), ssa.fresh_value(), ssa.fresh_value());

        let f = ssa.get_main_mut();
        let then_b = f.add_block();
        let else_b = f.add_block();
        let entry = f.get_entry_mut();
        entry.push_parameter(x, Type::u(32));
        entry.push_instruction(OpCode::Cmp {
            kind: CmpKind::Eq,
            result: is_seven,
            lhs: x,
            rhs: c7,
        });
        entry.set_terminator(Terminator::JmpIf(is_seven, then_b, else_b));

        let then_block = f.get_block_mut(then_b);
        then_block.push_instruction(OpCode::BinaryArithOp {
            kind: BinaryArithOpKind::Add,
            result: sum,
            lhs: x,
            rhs: c1,
        });
        then_block.set_terminator(Terminator::Return(vec![x, sum]));
        f.get_block_mut(else_b)
            .set_terminator(Terminator::Return(vec![x]));

        SCCP::new().do_run(&mut ssa);

        let f = ssa.get_main();
        assert_eq!(f.get_block(then_b).get_instructions().count(), 0);
        assert!(matches!(
            f.get_block(then_b).get_terminator(),
            Some(Terminator::Return(vals)) if *vals == vec![c7, c8]
        ));
        assert!(matches!(
            f.get_entry().get_terminator(),
            Some(Terminator::JmpIf(c, t, e)) if *c == is_seven && *t == then_b && *e == else_b
        ));
    }

    /// `assert x == K` proves `x = K` for ordinary following instructions in the same block.
    #[test]
    fn assert_cmp_equality_fact_folds_following_instruction_uses() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let c1 = ssa.add_const(Constant::U(32, 1));
        let c7 = ssa.add_const(Constant::U(32, 7));
        let c8 = ssa.add_const(Constant::U(32, 8));
        let (x, sum) = (ssa.fresh_value(), ssa.fresh_value());

        let f = ssa.get_main_mut();
        let entry = f.get_entry_mut();
        entry.push_parameter(x, Type::u(32));
        entry.push_instruction(OpCode::AssertCmp {
            kind: CmpKind::Eq,
            lhs: x,
            rhs: c7,
        });
        entry.push_instruction(OpCode::BinaryArithOp {
            kind: BinaryArithOpKind::Add,
            result: sum,
            lhs: x,
            rhs: c1,
        });
        entry.set_terminator(Terminator::Return(vec![x, sum]));

        SCCP::new().do_run(&mut ssa);

        let f = ssa.get_main();
        assert_eq!(f.get_entry().get_instructions().count(), 1);
        assert!(matches!(
            f.get_entry().get_terminator(),
            Some(Terminator::Return(vals)) if *vals == vec![x, c8]
        ));
    }

    /// Assertion-derived facts must not fold a later assertion predicate into a compile-time
    /// failing assert.
    #[test]
    fn assert_cmp_equality_keeps_later_assertion_predicate_runtime() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let c_true = ssa.add_const(Constant::U(1, 1));
        let c_false = ssa.add_const(Constant::U(1, 0));
        let (x, is_false) = (ssa.fresh_value(), ssa.fresh_value());

        let f = ssa.get_main_mut();
        let entry = f.get_entry_mut();
        entry.push_parameter(x, Type::u(1));
        entry.push_instruction(OpCode::AssertCmp {
            kind: CmpKind::Eq,
            lhs: x,
            rhs: c_true,
        });
        entry.push_instruction(OpCode::Cmp {
            kind: CmpKind::Eq,
            result: is_false,
            lhs: x,
            rhs: c_false,
        });
        entry.push_instruction(OpCode::Assert { value: is_false });
        entry.set_terminator(Terminator::Return(vec![]));

        SCCP::new().do_run(&mut ssa);

        let f = ssa.get_main();
        assert_eq!(f.get_entry().get_instructions().count(), 3);
        let mut instructions = f.get_entry().get_instructions();
        assert!(matches!(
            instructions.next(),
            Some(OpCode::AssertCmp { .. })
        ));
        assert!(matches!(
            instructions.next(),
            Some(OpCode::Cmp { lhs, rhs, .. }) if *lhs == x && *rhs == c_false
        ));
        assert!(matches!(
            instructions.next(),
            Some(OpCode::Assert { value }) if *value == is_false
        ));
    }

    /// Assertion-derived facts stay local to the block and do not replace terminator values.
    #[test]
    fn assert_cmp_equality_does_not_replace_return_operand() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let c7 = ssa.add_const(Constant::U(32, 7));
        let x = ssa.fresh_value();

        let f = ssa.get_main_mut();
        let entry = f.get_entry_mut();
        entry.push_parameter(x, Type::u(32));
        entry.push_instruction(OpCode::AssertCmp {
            kind: CmpKind::Eq,
            lhs: x,
            rhs: c7,
        });
        entry.set_terminator(Terminator::Return(vec![x]));

        SCCP::new().do_run(&mut ssa);

        let f = ssa.get_main();
        assert_eq!(f.get_entry().get_instructions().count(), 1);
        assert!(matches!(
            f.get_entry().get_terminator(),
            Some(Terminator::Return(vals)) if *vals == vec![x]
        ));
    }

    /// `assert c` proves the condition itself for following condition uses.
    #[test]
    fn assert_condition_fact_folds_following_condition_use() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let (cond, arm_t, arm_f, selected) = (
            ssa.fresh_value(),
            ssa.fresh_value(),
            ssa.fresh_value(),
            ssa.fresh_value(),
        );

        let f = ssa.get_main_mut();
        let entry = f.get_entry_mut();
        entry.push_parameter(cond, Type::u(1));
        entry.push_parameter(arm_t, Type::field());
        entry.push_parameter(arm_f, Type::field());
        entry.push_instruction(OpCode::Assert { value: cond });
        entry.push_instruction(OpCode::Select {
            result: selected,
            cond,
            if_t: arm_t,
            if_f: arm_f,
        });
        entry.set_terminator(Terminator::Return(vec![selected]));

        SCCP::new().do_run(&mut ssa);

        let f = ssa.get_main();
        assert_eq!(f.get_entry().get_instructions().count(), 1);
        assert!(matches!(
            f.get_entry().get_terminator(),
            Some(Terminator::Return(vals)) if *vals == vec![arm_t]
        ));
    }

    /// Branching on `not b` refines `b` on each edge.
    #[test]
    fn branch_not_fact_refines_inner_condition() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let (cond, not_cond, arm_t, arm_f, selected) = (
            ssa.fresh_value(),
            ssa.fresh_value(),
            ssa.fresh_value(),
            ssa.fresh_value(),
            ssa.fresh_value(),
        );

        let f = ssa.get_main_mut();
        let then_b = f.add_block();
        let else_b = f.add_block();
        let entry = f.get_entry_mut();
        entry.push_parameter(cond, Type::u(1));
        entry.push_parameter(arm_t, Type::field());
        entry.push_parameter(arm_f, Type::field());
        entry.push_instruction(OpCode::Not {
            result: not_cond,
            value: cond,
        });
        entry.set_terminator(Terminator::JmpIf(not_cond, then_b, else_b));

        let then_block = f.get_block_mut(then_b);
        then_block.push_instruction(OpCode::Select {
            result: selected,
            cond,
            if_t: arm_t,
            if_f: arm_f,
        });
        then_block.set_terminator(Terminator::Return(vec![selected]));
        f.get_block_mut(else_b)
            .set_terminator(Terminator::Return(vec![arm_t]));

        SCCP::new().do_run(&mut ssa);

        let f = ssa.get_main();
        assert_eq!(f.get_block(then_b).get_instructions().count(), 0);
        assert!(matches!(
            f.get_block(then_b).get_terminator(),
            Some(Terminator::Return(vals)) if *vals == vec![arm_f]
        ));
    }

    /// Conflicting incoming predicate facts disappear at a join.
    #[test]
    fn branch_fact_does_not_cross_conflicting_merge() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let (cond, not_cond) = (ssa.fresh_value(), ssa.fresh_value());

        let f = ssa.get_main_mut();
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

        SCCP::new().do_run(&mut ssa);

        let f = ssa.get_main();
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

        let f = ssa.get_main_mut();
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

        SCCP::new().do_run(&mut ssa);

        let f = ssa.get_main();
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

        let f = ssa.get_main_mut();
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

        SCCP::new().do_run(&mut ssa);

        let f = ssa.get_main();
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

        let f = ssa.get_main_mut();
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

        SCCP::new().do_run(&mut ssa);

        let f = ssa.get_main();
        assert!(matches!(
            f.get_block(then_b).get_terminator(),
            Some(Terminator::Jmp(t, args)) if *t == merge && *args == vec![c_true]
        ));
        assert!(matches!(
            f.get_block(merge).get_terminator(),
            Some(Terminator::Return(vals)) if *vals == vec![c_true]
        ));
    }
}
