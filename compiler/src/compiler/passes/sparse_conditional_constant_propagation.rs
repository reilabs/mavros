//! Sparse conditional constant propagation (Wegman–Zadeck) over the HLSSA.
//!
//! Runs the classic optimistic worklist algorithm per function: every value starts at ⊤
//! (unvisited), is lowered to a concrete constant when a transfer function can fold it, and bottoms
//! out at ⊥ (overdefined) otherwise. Block reachability and value lattices are computed together,
//! so constants are propagated through branches that are themselves decided by constants — strictly
//! stronger than folding and branch pruning as separate passes.
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

use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};

use ark_ff::{PrimeField, Zero};

use crate::compiler::{
    Field,
    pass_manager::{AnalysisStore, Pass},
    passes::fix_double_jumps::{ReplaceScope, ValueReplacements},
    ssa::{
        BlockId, Instruction, Terminator, ValueId,
        hlssa::{
            BinaryArithOpKind, CmpKind, Constant, HLFunction, HLSSA, HLSSAConstantsSnapshot,
            MAX_SUPPORTED_SIGNED_BITS, OpCode, Type, TypeExpr,
        },
    },
    util::{bit_mask, decode_signed, encode_signed, fits_signed},
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
}

impl LatticeResult<'_> {
    fn lookup(&self, v: ValueId) -> LatticeElement {
        if let Some(c) = self.consts.get(&v) {
            return LatticeElement::Const(c.clone());
        }
        self.lattice.get(&v).cloned().unwrap_or(LatticeElement::Top)
    }
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

    edge_worklist: Vec<(BlockId, BlockId)>,
    value_worklist: Vec<ValueId>,

    /// For each value: the sites that read it. `None` marks the block's terminator.
    uses: HashMap<ValueId, Vec<(BlockId, Option<usize>)>>,
}

impl<'f, 'c> FunctionLattice<'f, 'c> {
    fn new(function: &'f HLFunction, consts: &'c HLSSAConstantsSnapshot) -> Self {
        let mut uses: HashMap<ValueId, Vec<(BlockId, Option<usize>)>> = HashMap::new();
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
            lattice: HashMap::new(),
            exec_edges: HashSet::new(),
            exec_preds: HashMap::new(),
            reachable: HashSet::new(),
            edge_worklist: Vec::new(),
            value_worklist: Vec::new(),
            uses,
        }
    }

    fn into_result(self) -> LatticeResult<'c> {
        LatticeResult {
            consts: self.consts,
            lattice: self.lattice,
            reachable: self.reachable,
        }
    }

    fn run(&mut self) {
        let entry = self.function.get_entry_id();
        // Entry parameters are the function's arguments: unknown at this (intraprocedural) level.
        for (p, _) in self.function.get_entry().get_parameters() {
            self.lattice.insert(*p, LatticeElement::Bottom);
        }
        self.reachable.insert(entry);
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
                    self.lattice_of(*cond) != LatticeElement::Top,
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
        if let Some(c) = self.consts.get(&v) {
            return LatticeElement::Const(c.clone());
        }
        self.lattice.get(&v).cloned().unwrap_or(LatticeElement::Top)
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
        if self.reachable.insert(b) {
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
            _ => self.visit_terminator(bid),
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
        let updates = self.transfer(self.function.get_block(bid).get_instruction(idx));
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
            Terminator::JmpIf(cond, t, f) => match self.lattice_of(cond) {
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
    /// re-join the target's parameters.
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
        let mut updates = Vec::with_capacity(indices.len());
        for &i in indices {
            let mut acc = LatticeElement::Top;
            for pred in &preds {
                match self.function.get_block(*pred).get_terminator() {
                    Some(Terminator::Jmp(t, args)) if *t == b => {
                        acc = join(acc, self.lattice_of(args[i]));
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

    /// The transfer function: new lattice values for the instruction's results.
    fn transfer(&self, instr: &OpCode) -> Vec<(ValueId, LatticeElement)> {
        match instr {
            OpCode::BinaryArithOp {
                kind,
                result,
                lhs,
                rhs,
            } => vec![(
                *result,
                self.eval2(*lhs, *rhs, |a, b| eval_binary(*kind, a, b)),
            )],
            OpCode::Cmp {
                kind,
                result,
                lhs,
                rhs,
            } => vec![(
                *result,
                self.eval2(*lhs, *rhs, |a, b| eval_cmp(*kind, a, b)),
            )],
            OpCode::MulConst {
                result,
                const_val,
                var,
            } => vec![(*result, self.eval2(*const_val, *var, eval_field_mul))],
            OpCode::Cast {
                result,
                value,
                target,
            } => vec![(*result, self.eval1(*value, |v| eval_cast(target, v)))],
            OpCode::SExt {
                result,
                value,
                from_bits,
                to_bits,
            } => vec![(
                *result,
                self.eval1(*value, |v| eval_sext(v, *from_bits, *to_bits)),
            )],
            OpCode::BitRange {
                result,
                value,
                offset,
                width,
            } => vec![(
                *result,
                self.eval1(*value, |v| eval_bit_range(v, *offset, *width)),
            )],
            OpCode::Not { result, value } => vec![(*result, self.eval1(*value, eval_not))],
            OpCode::Select {
                result,
                cond,
                if_t,
                if_f,
            } => vec![(*result, self.eval_select(*cond, *if_t, *if_f))],
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
            | OpCode::ValueOf { .. }
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

    fn eval1(&self, v: ValueId, f: impl FnOnce(&Constant) -> Option<Constant>) -> LatticeElement {
        match self.lattice_of(v) {
            LatticeElement::Top => LatticeElement::Top,
            LatticeElement::Const(c) => f(&c)
                .map(|c| LatticeElement::Const(Arc::new(c)))
                .unwrap_or(LatticeElement::Bottom),
            LatticeElement::Bottom => LatticeElement::Bottom,
        }
    }

    fn eval2(
        &self,
        l: ValueId,
        r: ValueId,
        f: impl FnOnce(&Constant, &Constant) -> Option<Constant>,
    ) -> LatticeElement {
        match (self.lattice_of(l), self.lattice_of(r)) {
            (LatticeElement::Top, _) | (_, LatticeElement::Top) => LatticeElement::Top,
            (LatticeElement::Const(a), LatticeElement::Const(b)) => f(&a, &b)
                .map(|c| LatticeElement::Const(Arc::new(c)))
                .unwrap_or(LatticeElement::Bottom),
            _ => LatticeElement::Bottom,
        }
    }

    fn eval_select(&self, cond: ValueId, if_t: ValueId, if_f: ValueId) -> LatticeElement {
        match self.lattice_of(cond) {
            LatticeElement::Top => LatticeElement::Top,
            LatticeElement::Const(c) => match const_bool(&c) {
                Some(true) => self.lattice_of(if_t),
                Some(false) => self.lattice_of(if_f),
                None => LatticeElement::Bottom,
            },
            // Unknown condition: the select still folds if both arms agree.
            LatticeElement::Bottom => join(self.lattice_of(if_t), self.lattice_of(if_f)),
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
fn eval_cast(target: &Type, v: &Constant) -> Option<Constant> {
    // A cast into the witness domain is never a pure constant: the witness
    // chain stays intact.
    if target.is_witness_of() {
        return None;
    }
    match &target.expr {
        TypeExpr::Field => match v {
            Constant::U(_, x) | Constant::I(_, x) => Some(Constant::Field(Field::from(*x))),
            Constant::Field(_) => Some(v.clone()),
            Constant::FnPtr(_) | Constant::Blob(_) => None,
        },
        TypeExpr::U(n) => int_cast_bits(v, *n).map(|bits| Constant::U(*n, bits)),
        TypeExpr::I(n) => {
            if *n > MAX_SUPPORTED_SIGNED_BITS {
                return None;
            }
            int_cast_bits(v, *n).map(|bits| Constant::I(*n, bits))
        }
        // Composite targets (e.g. array→slice or element-wise witness
        // conversions) transfer or transform runtime objects: nothing to fold.
        _ => None,
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
        for instr in instructions {
            // A single-result instruction whose result the lattice proved constant is one of the
            // pure scalar folds (see `transfer`): its uses are aliased, so drop it.
            {
                let mut results = instr.get_results();
                if let (Some(r), None) = (results.next(), results.next()) {
                    if matches!(res.lattice.get(r), Some(LatticeElement::Const(_))) {
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
            {
                if let LatticeElement::Const(c) = res.lookup(*cond) {
                    if let Some(b) = const_bool(&c) {
                        replacements.insert(*result, if b { *if_t } else { *if_f });
                        continue;
                    }
                }
            }
            kept.push(instr);
        }
        block.put_instructions(kept);

        if let Some(Terminator::JmpIf(cond, t, f)) = block.get_terminator() {
            let (cond, t, f) = (*cond, *t, *f);
            if let LatticeElement::Const(c) = res.lookup(cond) {
                if let Some(b) = const_bool(&c) {
                    let target = if b { t } else { f };
                    block.set_terminator(Terminator::Jmp(target, vec![]));
                }
            }
        }
    }

    replacements.apply_to_function(function, ReplaceScope::Inputs);
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
            target: Type::witness_of(Type::field()),
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
}
