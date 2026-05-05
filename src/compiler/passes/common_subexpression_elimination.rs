use std::{
    collections::{HashMap, HashSet},
    fmt::{Debug, Display},
};

use ark_ff::{AdditiveGroup, Field as _};
use num_traits::{One, Zero};

use crate::compiler::{
    flow_analysis::{CFG, FlowAnalysis},
    ssa::{
        BinaryArithOpKind, BlockId, CastTarget, CmpKind, ConstValue, Endianness, HLFunction, HLSSA,
        OpCode, Radix, ValueId,
    },
};
use crate::compiler::{
    pass_manager::{Analysis, AnalysisId, AnalysisStore, Pass},
    passes::fix_double_jumps::ValueReplacements,
};

#[derive(Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
enum Expr {
    Add(Vec<Expr>),
    Mul(Vec<Expr>),
    Div(Box<Expr>, Box<Expr>),
    Mod(Box<Expr>, Box<Expr>),
    Sub(Box<Expr>, Box<Expr>),
    FConst(ark_bn254::Fr),
    UConst(usize, u128),
    IConst(usize, u128),
    Variable(u64),
    Eq(Box<Expr>, Box<Expr>),
    Lt(Box<Expr>, Box<Expr>),
    And(Vec<Expr>),
    Or(Vec<Expr>),
    Xor(Vec<Expr>),
    Shl(Box<Expr>, Box<Expr>),
    Shr(Box<Expr>, Box<Expr>),
    Select(Box<Expr>, Box<Expr>, Box<Expr>),
    ArrayGet(Box<Expr>, Box<Expr>),
    TupleGet(Box<Expr>, Box<Expr>),
    Not(Box<Expr>),
    ReadGlobal(u64),
    Cast(Box<Expr>, CastTarget),
    Truncate(Box<Expr>, usize /* to_bits */, usize /* from_bits */),
    SExt(Box<Expr>, usize /* from_bits */, usize /* to_bits */),
    ValueOf(Box<Expr>),
    /// Byte decomposition of a Field-typed expression. The whole array result;
    /// element accesses are represented via `ArrayGet`. Endianness and count
    /// are part of the key, so two decompositions only merge when they agree.
    BytesOf(Box<Expr>, Endianness, usize /* count */),
    BitsOf(Box<Expr>, Endianness, usize /* count */),
    /// Witness slot whose hint is `expr`. Two non-pinned `write_witness` ops
    /// with the same hint expression resolve to the same slot under CSE.
    /// (Pinned writes and `FreshWitness` aren't given an `Expr` at all — they
    /// fall back to the default `Expr::Variable(value_id)` keyed by their
    /// unique result ValueId, which never collides.)
    Witness(Box<Expr>),
}

/// Side-effect keys for opcodes that don't produce values but do emit
/// constraints — duplicates of these are redundant and can be dropped.
#[derive(Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
enum Effect {
    /// `Rangecheck { value, max_bits }` — high-level rangecheck.
    Rangecheck(Expr, usize),
    /// `Lookup { target: Rangecheck(8), keys: [k], flag }` — the byte-lookup
    /// constraint. Duplicates with the same key + flag are redundant.
    ByteLookup(Expr, Expr),
}

impl Expr {
    pub fn variable(value_id: ValueId) -> Self {
        Self::Variable(value_id.0)
    }

    /// True if this expression is the additive identity in any scalar type
    /// we care about. Used by smart constructors to fold `x + 0 → x` etc.
    fn is_zero(&self) -> bool {
        match self {
            Self::FConst(v) => v.is_zero(),
            Self::UConst(_, 0) => true,
            Self::IConst(_, 0) => true,
            _ => false,
        }
    }

    /// Multiplicative identity.
    fn is_one(&self) -> bool {
        match self {
            Self::FConst(v) => v.is_one(),
            Self::UConst(_, 1) => true,
            Self::IConst(_, 1) => true,
            _ => false,
        }
    }

    fn get_adds(&self) -> Vec<Self> {
        match self {
            Self::Add(exprs) => exprs.iter().cloned().collect(),
            _ => vec![self.clone()],
        }
    }

    fn get_muls(&self) -> Vec<Self> {
        match self {
            Self::Mul(exprs) => exprs.iter().cloned().collect(),
            _ => vec![self.clone()],
        }
    }

    fn get_ands(&self) -> Vec<Self> {
        match self {
            Self::And(exprs) => exprs.iter().cloned().collect(),
            _ => vec![self.clone()],
        }
    }

    /// Smart constructor: drop literal zeros (`x + 0 → x`); flatten and sort
    /// for canonical form. Does NOT fold `c1 + c2 → const` because the
    /// result wouldn't have an SSA opcode of its own to materialise — and CSE
    /// picking it as a canonical for some unrelated `Const(c1+c2)` opcode
    /// would let downstream consumers see a non-Const opcode where they
    /// expect one.
    pub fn add(&self, other: &Self) -> Self {
        if self.is_zero() {
            return other.clone();
        }
        if other.is_zero() {
            return self.clone();
        }
        let mut adds: Vec<Self> = self
            .get_adds()
            .into_iter()
            .chain(other.get_adds().into_iter())
            .filter(|e| !e.is_zero())
            .collect();
        match adds.len() {
            0 => unreachable!("zero-zero filtered above"),
            1 => adds.pop().unwrap(),
            _ => {
                adds.sort();
                Self::Add(adds)
            }
        }
    }

    /// `x · 1 → x`, `0 · y → 0` (returning the existing zero operand). Same
    /// no-introduce-constants rule as `add`.
    pub fn mul(&self, other: &Self) -> Self {
        // Annihilator: return whichever operand is the literal zero so we
        // preserve its variant and SSA identity.
        if self.is_zero() {
            return self.clone();
        }
        if other.is_zero() {
            return other.clone();
        }
        let mut muls: Vec<Self> = self
            .get_muls()
            .into_iter()
            .chain(other.get_muls().into_iter())
            .filter(|e| !e.is_one())
            .collect();
        match muls.len() {
            0 => unreachable!("one-one filtered above"),
            1 => muls.pop().unwrap(),
            _ => {
                muls.sort();
                Self::Mul(muls)
            }
        }
    }

    pub fn div(&self, other: &Self) -> Self {
        // x / 1 → x.
        if other.is_one() {
            return self.clone();
        }
        Self::Div(Box::new(self.clone()), Box::new(other.clone()))
    }

    pub fn modulo(&self, other: &Self) -> Self {
        Self::Mod(Box::new(self.clone()), Box::new(other.clone()))
    }

    /// `x − 0 → x`. Does not fold `x − x → 0` (would introduce an SSA-less
    /// constant; see `add`).
    pub fn sub(&self, other: &Self) -> Self {
        if other.is_zero() {
            return self.clone();
        }
        Self::Sub(Box::new(self.clone()), Box::new(other.clone()))
    }

    /// Bitwise AND: `x & 0 → 0` (returning the literal zero operand).
    pub fn and(&self, other: &Self) -> Self {
        if self.is_zero() {
            return self.clone();
        }
        if other.is_zero() {
            return other.clone();
        }
        let mut ands = self.get_ands();
        ands.extend(other.get_ands());
        ands.sort();
        ands.dedup();
        if ands.len() == 1 {
            return ands.pop().unwrap();
        }
        Self::And(ands)
    }

    /// Bitwise OR: `x | 0 → x`.
    pub fn or(&self, other: &Self) -> Self {
        if self.is_zero() {
            return other.clone();
        }
        if other.is_zero() {
            return self.clone();
        }
        let mut ors: Vec<Self> = match self {
            Self::Or(exprs) => exprs.iter().cloned().collect(),
            _ => vec![self.clone()],
        };
        ors.extend(match other {
            Self::Or(exprs) => exprs.iter().cloned().collect(),
            _ => vec![other.clone()],
        });
        ors.sort();
        ors.dedup();
        if ors.len() == 1 {
            return ors.pop().unwrap();
        }
        Self::Or(ors)
    }

    /// Bitwise XOR: `x ^ 0 → x`. Does not fold `x ^ x → 0` (constant
    /// introduction).
    pub fn xor(&self, other: &Self) -> Self {
        if self.is_zero() {
            return other.clone();
        }
        if other.is_zero() {
            return self.clone();
        }
        let mut xors: Vec<Self> = match self {
            Self::Xor(exprs) => exprs.iter().cloned().collect(),
            _ => vec![self.clone()],
        };
        xors.extend(match other {
            Self::Xor(exprs) => exprs.iter().cloned().collect(),
            _ => vec![other.clone()],
        });
        xors.sort();
        Self::Xor(xors)
    }

    pub fn shl(&self, other: &Self) -> Self {
        if other.is_zero() {
            return self.clone();
        }
        Self::Shl(Box::new(self.clone()), Box::new(other.clone()))
    }

    pub fn shr(&self, other: &Self) -> Self {
        if other.is_zero() {
            return self.clone();
        }
        Self::Shr(Box::new(self.clone()), Box::new(other.clone()))
    }

    pub fn fconst(value: ark_bn254::Fr) -> Self {
        Self::FConst(value)
    }

    pub fn eq(&self, other: &Self) -> Self {
        Self::Eq(Box::new(self.clone()), Box::new(other.clone()))
    }

    pub fn lt(&self, other: &Self) -> Self {
        Self::Lt(Box::new(self.clone()), Box::new(other.clone()))
    }

    pub fn array_get(&self, index: &Self) -> Self {
        Self::ArrayGet(Box::new(self.clone()), Box::new(index.clone()))
    }

    pub fn tuple_get(&self, index: &Self) -> Self {
        Self::TupleGet(Box::new(self.clone()), Box::new(index.clone()))
    }

    pub fn select(&self, then: &Self, otherwise: &Self) -> Self {
        // select(_, x, x) → x. Same alternatives — condition irrelevant.
        if then == otherwise {
            return then.clone();
        }
        Self::Select(
            Box::new(self.clone()),
            Box::new(then.clone()),
            Box::new(otherwise.clone()),
        )
    }

    pub fn not(&self) -> Self {
        // ~~x → x.
        if let Self::Not(inner) = self {
            return (**inner).clone();
        }
        Self::Not(Box::new(self.clone()))
    }

    /// `Cast(x, Nop) → x`. Two casts to the same target collapse to one.
    pub fn cast(&self, target: CastTarget) -> Self {
        if matches!(target, CastTarget::Nop) {
            return self.clone();
        }
        if let Self::Cast(_, t) = self {
            if *t == target {
                return self.clone();
            }
        }
        Self::Cast(Box::new(self.clone()), target)
    }

    pub fn truncate(&self, to_bits: usize, from_bits: usize) -> Self {
        Self::Truncate(Box::new(self.clone()), to_bits, from_bits)
    }

    pub fn sext(&self, from_bits: usize, to_bits: usize) -> Self {
        Self::SExt(Box::new(self.clone()), from_bits, to_bits)
    }

    /// `ValueOf(ValueOf(x)) → ValueOf(x)` (idempotent).
    /// `ValueOf(Witness(h)) → h` (witgen identity).
    pub fn value_of(&self) -> Self {
        match self {
            Self::ValueOf(_) => self.clone(),
            Self::Witness(hint) => (**hint).clone(),
            _ => Self::ValueOf(Box::new(self.clone())),
        }
    }

    pub fn bytes_of(&self, endianness: Endianness, count: usize) -> Self {
        Self::BytesOf(Box::new(self.clone()), endianness, count)
    }

    pub fn bits_of(&self, endianness: Endianness, count: usize) -> Self {
        Self::BitsOf(Box::new(self.clone()), endianness, count)
    }

    /// `Witness(ValueOf(x)) → x` — dual to `ValueOf(Witness(h)) → h`. The new
    /// witness slot's hint is `x`'s value, so an honest prover fills both
    /// identically; merging is equivalent to adding the always-true
    /// constraint `new_slot == x`. Sound by construction.
    ///
    /// For this rewrite to be safe in our pipeline, hint chains in
    /// `explicit_witness` gadgets MUST be structured so that `ValueOf` only
    /// appears at hint-chain boundaries (right after the gadget input
    /// operands), not in the middle of compute chains. Otherwise the rewrite
    /// can keep witness-typed `Cmp`/`Div`/etc. opcodes alive past
    /// `witness_write_to_fresh`'s DCE, and R1CS gen panics on them.
    pub fn witness(&self) -> Self {
        if let Self::ValueOf(inner) = self {
            return (**inner).clone();
        }
        Self::Witness(Box::new(self.clone()))
    }
}

impl Display for Expr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Add(exprs) => write!(
                f,
                "({})",
                exprs
                    .iter()
                    .map(|e| e.to_string())
                    .collect::<Vec<_>>()
                    .join(" + ")
            ),
            Self::Mul(exprs) => write!(
                f,
                "({})",
                exprs
                    .iter()
                    .map(|e| e.to_string())
                    .collect::<Vec<_>>()
                    .join(" * ")
            ),
            Self::Div(lhs, rhs) => write!(f, "({} / {})", lhs, rhs),
            Self::Mod(lhs, rhs) => write!(f, "({} % {})", lhs, rhs),
            Self::Sub(lhs, rhs) => write!(f, "({} - {})", lhs, rhs),
            Self::FConst(value) => write!(f, "{}", value),
            Self::UConst(size, value) => write!(f, "u{}({})", size, value),
            Self::IConst(size, value) => write!(f, "i{}({})", size, value),
            Self::Variable(value) => write!(f, "v{}", value),
            Self::Eq(lhs, rhs) => write!(f, "({} == {})", lhs, rhs),
            Self::Lt(lhs, rhs) => write!(f, "({} < {})", lhs, rhs),
            Self::And(exprs) => write!(
                f,
                "({})",
                exprs
                    .iter()
                    .map(|e| e.to_string())
                    .collect::<Vec<_>>()
                    .join(" & ")
            ),
            Self::Or(exprs) => write!(
                f,
                "({})",
                exprs
                    .iter()
                    .map(|e| e.to_string())
                    .collect::<Vec<_>>()
                    .join(" | ")
            ),
            Self::Xor(exprs) => write!(
                f,
                "({})",
                exprs
                    .iter()
                    .map(|e| e.to_string())
                    .collect::<Vec<_>>()
                    .join(" ^ ")
            ),
            Self::Shl(lhs, rhs) => write!(f, "({} << {})", lhs, rhs),
            Self::Shr(lhs, rhs) => write!(f, "({} >> {})", lhs, rhs),
            Self::Select(cond, then, otherwise) => {
                write!(f, "({} ? {} : {})", cond, then, otherwise)
            }
            Self::ArrayGet(array, index) => write!(f, "{}[{}]", array, index),
            Self::TupleGet(tuple, index) => write!(f, "{}.{}", tuple, index),
            Self::Not(value) => write!(f, "(~{})", value),
            Self::ReadGlobal(index) => write!(f, "g{}", index),
            Self::Cast(value, target) => write!(f, "cast({}, {})", value, target),
            Self::Truncate(value, to, from) => {
                write!(f, "trunc({}, {}, {})", value, to, from)
            }
            Self::SExt(value, from, to) => write!(f, "sext({}, {}, {})", value, from, to),
            Self::ValueOf(value) => write!(f, "value_of({})", value),
            Self::BytesOf(value, end, count) => {
                write!(f, "bytes_of({}, {:?}, {})", value, end, count)
            }
            Self::BitsOf(value, end, count) => {
                write!(f, "bits_of({}, {:?}, {})", value, end, count)
            }
            Self::Witness(hint) => write!(f, "witness({})", hint),
        }
    }
}

impl Debug for Expr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self)
    }
}

pub struct CSE {}

impl Pass for CSE {
    fn name(&self) -> &'static str {
        "cse"
    }

    fn needs(&self) -> Vec<AnalysisId> {
        vec![FlowAnalysis::id()]
    }

    fn run(&self, ssa: &mut HLSSA, store: &AnalysisStore) {
        self.do_run(ssa, store.get::<FlowAnalysis>());
    }

    fn preserves(&self) -> Vec<AnalysisId> {
        vec![FlowAnalysis::id()]
    }
}

impl CSE {
    pub fn new() -> Self {
        Self {}
    }

    pub fn do_run(&self, ssa: &mut HLSSA, cfg: &FlowAnalysis) {
        for (function_id, function) in ssa.iter_functions_mut() {
            let cfg = cfg.get_function_cfg(*function_id);
            let (exprs, effects) = self.gather_expressions(function, cfg);
            let mut value_replacements = ValueReplacements::new();
            for (_, occurrences) in exprs {
                if occurrences.len() <= 1 {
                    continue;
                }
                let mut replacement_groups: Vec<((BlockId, usize, ValueId), Vec<ValueId>)> = vec![];
                for (block_id, instruction_idx, value_id) in occurrences {
                    let mut found = false;
                    for ((candidate_block, candidate_instruction, candidate_value_id), others) in
                        replacement_groups.iter_mut()
                    {
                        if self.can_replace(
                            cfg,
                            *candidate_block,
                            *candidate_instruction,
                            block_id,
                            instruction_idx,
                        ) {
                            found = true;
                            others.push(value_id);
                            break;
                        } else if self.can_replace(
                            cfg,
                            block_id,
                            instruction_idx,
                            *candidate_block,
                            *candidate_instruction,
                        ) {
                            found = true;
                            others.push(*candidate_value_id);
                            *candidate_block = block_id;
                            *candidate_instruction = instruction_idx;
                            *candidate_value_id = value_id;
                            break;
                        }
                    }
                    if !found {
                        replacement_groups.push(((block_id, instruction_idx, value_id), vec![]));
                    }
                }
                for ((_, _, value_id), others) in replacement_groups {
                    for other in others {
                        value_replacements.insert(other, value_id);
                    }
                }
            }

            // Side-effect dedup: for each Effect with multiple occurrences,
            // keep the dominator-most canonical and mark the others for
            // removal. Same dominance grouping logic as the value-replacement
            // loop above; we just discard duplicates instead of redirecting
            // their results.
            let mut to_remove: HashSet<(BlockId, usize)> = HashSet::new();
            for (_, occurrences) in effects {
                if occurrences.len() <= 1 {
                    continue;
                }
                let mut groups: Vec<(BlockId, usize, Vec<(BlockId, usize)>)> = vec![];
                for (block_id, instruction_idx) in occurrences {
                    let mut found = false;
                    for (candidate_block, candidate_instruction, others) in groups.iter_mut() {
                        if self.can_replace(
                            cfg,
                            *candidate_block,
                            *candidate_instruction,
                            block_id,
                            instruction_idx,
                        ) {
                            found = true;
                            others.push((block_id, instruction_idx));
                            break;
                        } else if self.can_replace(
                            cfg,
                            block_id,
                            instruction_idx,
                            *candidate_block,
                            *candidate_instruction,
                        ) {
                            found = true;
                            others.push((*candidate_block, *candidate_instruction));
                            *candidate_block = block_id;
                            *candidate_instruction = instruction_idx;
                            break;
                        }
                    }
                    if !found {
                        groups.push((block_id, instruction_idx, vec![]));
                    }
                }
                for (_, _, others) in groups {
                    for pos in others {
                        to_remove.insert(pos);
                    }
                }
            }

            for (block_id, block) in function.get_blocks_mut() {
                let bid = *block_id;
                let old_instructions = block.take_instructions();
                let mut new_instructions = Vec::with_capacity(old_instructions.len());
                for (idx, mut instruction) in old_instructions.into_iter().enumerate() {
                    if to_remove.contains(&(bid, idx)) {
                        continue;
                    }
                    value_replacements.replace_inputs(&mut instruction);
                    new_instructions.push(instruction);
                }
                block.put_instructions(new_instructions);
                value_replacements.replace_terminator(block.get_terminator_mut());
            }
        }
    }

    fn can_replace(
        &self,
        cfg: &CFG,
        block1: BlockId,
        instruction1: usize,
        block2: BlockId,
        instruction2: usize,
    ) -> bool {
        if block1 == block2 && instruction1 < instruction2 {
            return true;
        }
        if cfg.dominates(block1, block2) {
            return true;
        }
        false
    }

    fn gather_expressions(
        &self,
        ssa: &HLFunction,
        cfg: &CFG,
    ) -> (
        HashMap<Expr, Vec<(BlockId, usize, ValueId)>>,
        HashMap<Effect, Vec<(BlockId, usize)>>,
    ) {
        let mut result: HashMap<Expr, Vec<(BlockId, usize, ValueId)>> = HashMap::new();
        let mut effects: HashMap<Effect, Vec<(BlockId, usize)>> = HashMap::new();
        let mut exprs = HashMap::<ValueId, Expr>::new();

        fn get_expr(exprs: &HashMap<ValueId, Expr>, value_id: &ValueId) -> Expr {
            exprs
                .get(&value_id)
                .cloned()
                .unwrap_or(Expr::variable(*value_id))
        }

        for block_id in cfg.get_domination_pre_order() {
            let block = ssa.get_block(block_id);

            for (instruction_idx, instruction) in block.get_instructions().enumerate() {
                match instruction {
                    OpCode::BinaryArithOp { kind: BinaryArithOpKind::Add, result: r, lhs, rhs } => {
                        let lhs_expr = get_expr(&exprs, lhs);
                        let rhs_expr = get_expr(&exprs, rhs);
                        let result_expr = lhs_expr.add(&rhs_expr);
                        exprs.insert(*r, result_expr.clone());
                        result.entry(result_expr).or_default().push((
                            block_id,
                            instruction_idx,
                            *r,
                        ));
                    }
                    OpCode::BinaryArithOp { kind: BinaryArithOpKind::Mul, result: r, lhs, rhs } => {
                        let lhs_expr = get_expr(&exprs, lhs);
                        let rhs_expr = get_expr(&exprs, rhs);
                        let result_expr = lhs_expr.mul(&rhs_expr);
                        exprs.insert(*r, result_expr.clone());
                        result.entry(result_expr).or_default().push((
                            block_id,
                            instruction_idx,
                            *r,
                        ));
                    }
                    OpCode::BinaryArithOp { kind: BinaryArithOpKind::Div, result: r, lhs, rhs } => {
                        let lhs_expr = get_expr(&exprs, lhs);
                        let rhs_expr = get_expr(&exprs, rhs);
                        let result_expr = lhs_expr.div(&rhs_expr);
                        exprs.insert(*r, result_expr.clone());
                        result.entry(result_expr).or_default().push((
                            block_id,
                            instruction_idx,
                            *r,
                        ));
                    }
                    OpCode::BinaryArithOp { kind: BinaryArithOpKind::Sub, result: r, lhs, rhs } => {
                        let lhs_expr = get_expr(&exprs, lhs);
                        let rhs_expr = get_expr(&exprs, rhs);
                        let result_expr = lhs_expr.sub(&rhs_expr);
                        exprs.insert(*r, result_expr.clone());
                        result.entry(result_expr).or_default().push((
                            block_id,
                            instruction_idx,
                            *r,
                        ));
                    }
                    OpCode::Cmp { kind: CmpKind::Eq, result: r, lhs, rhs } => {
                        let lhs_expr = get_expr(&exprs, lhs);
                        let rhs_expr = get_expr(&exprs, rhs);
                        let result_expr = lhs_expr.eq(&rhs_expr);
                        exprs.insert(*r, result_expr.clone());
                        result.entry(result_expr).or_default().push((
                            block_id,
                            instruction_idx,
                            *r,
                        ));
                    }
                    OpCode::Cmp { kind: CmpKind::Lt, result: r, lhs, rhs } => {
                        let lhs_expr = get_expr(&exprs, lhs);
                        let rhs_expr = get_expr(&exprs, rhs);
                        let result_expr = lhs_expr.lt(&rhs_expr);
                        exprs.insert(*r, result_expr.clone());
                        result.entry(result_expr).or_default().push((
                            block_id,
                            instruction_idx,
                            *r,
                        ));
                    }
                    OpCode::BinaryArithOp { kind: BinaryArithOpKind::Mod, result: r, lhs, rhs } => {
                        let lhs_expr = get_expr(&exprs, lhs);
                        let rhs_expr = get_expr(&exprs, rhs);
                        let result_expr = lhs_expr.modulo(&rhs_expr);
                        exprs.insert(*r, result_expr.clone());
                        result.entry(result_expr).or_default().push((
                            block_id,
                            instruction_idx,
                            *r,
                        ));
                    }
                    OpCode::BinaryArithOp { kind: BinaryArithOpKind::And, result: r, lhs, rhs } => {
                        let lhs_expr = get_expr(&exprs, lhs);
                        let rhs_expr = get_expr(&exprs, rhs);
                        let result_expr = lhs_expr.and(&rhs_expr);
                        exprs.insert(*r, result_expr.clone());
                        result.entry(result_expr).or_default().push((
                            block_id,
                            instruction_idx,
                            *r,
                        ));
                    }
                    OpCode::BinaryArithOp { kind: BinaryArithOpKind::Or, result: r, lhs, rhs } => {
                        let lhs_expr = get_expr(&exprs, lhs);
                        let rhs_expr = get_expr(&exprs, rhs);
                        let result_expr = lhs_expr.or(&rhs_expr);
                        exprs.insert(*r, result_expr.clone());
                        result.entry(result_expr).or_default().push((
                            block_id,
                            instruction_idx,
                            *r,
                        ));
                    }
                    OpCode::BinaryArithOp { kind: BinaryArithOpKind::Xor, result: r, lhs, rhs } => {
                        let lhs_expr = get_expr(&exprs, lhs);
                        let rhs_expr = get_expr(&exprs, rhs);
                        let result_expr = lhs_expr.xor(&rhs_expr);
                        exprs.insert(*r, result_expr.clone());
                        result.entry(result_expr).or_default().push((
                            block_id,
                            instruction_idx,
                            *r,
                        ));
                    }
                    OpCode::BinaryArithOp { kind: BinaryArithOpKind::Shl, result: r, lhs, rhs } => {
                        let lhs_expr = get_expr(&exprs, lhs);
                        let rhs_expr = get_expr(&exprs, rhs);
                        let result_expr = lhs_expr.shl(&rhs_expr);
                        exprs.insert(*r, result_expr.clone());
                        result.entry(result_expr).or_default().push((
                            block_id,
                            instruction_idx,
                            *r,
                        ));
                    }
                    OpCode::BinaryArithOp { kind: BinaryArithOpKind::Shr, result: r, lhs, rhs } => {
                        let lhs_expr = get_expr(&exprs, lhs);
                        let rhs_expr = get_expr(&exprs, rhs);
                        let result_expr = lhs_expr.shr(&rhs_expr);
                        exprs.insert(*r, result_expr.clone());
                        result.entry(result_expr).or_default().push((
                            block_id,
                            instruction_idx,
                            *r,
                        ));
                    }
                    OpCode::ArrayGet { result: r, array, index } => {
                        let array_expr = get_expr(&exprs, array);
                        let index_expr = get_expr(&exprs, index);
                        let result_expr = array_expr.array_get(&index_expr);
                        exprs.insert(*r, result_expr.clone());
                        result.entry(result_expr).or_default().push((
                            block_id,
                            instruction_idx,
                            *r,
                        ));
                    }
                    OpCode::Select { result: r, cond, if_t: then, if_f: otherwise } => {
                        let cond_expr = get_expr(&exprs, cond);
                        let then_expr = get_expr(&exprs, then);
                        let otherwise_expr = get_expr(&exprs, otherwise);
                        let result_expr = cond_expr.select(&then_expr, &otherwise_expr);
                        exprs.insert(*r, result_expr.clone());
                        result.entry(result_expr).or_default().push((
                            block_id,
                            instruction_idx,
                            *r,
                        ));
                    }
                    OpCode::ReadGlobal { result: r, offset: index, result_type: _ } => {
                        let result_expr = Expr::ReadGlobal(*index);
                        exprs.insert(*r, result_expr.clone());
                        result.entry(result_expr).or_default().push((
                            block_id,
                            instruction_idx,
                            *r,
                        ));
                    }
                    OpCode::Cast {
                        result: r,
                        value,
                        target,
                    } => {
                        let value_expr = get_expr(&exprs, value);
                        let result_expr = value_expr.cast(*target);
                        exprs.insert(*r, result_expr.clone());
                        result
                            .entry(result_expr)
                            .or_default()
                            .push((block_id, instruction_idx, *r));
                    }
                    OpCode::Truncate {
                        result: r,
                        value,
                        to_bits,
                        from_bits,
                    } => {
                        let value_expr = get_expr(&exprs, value);
                        let result_expr = value_expr.truncate(*to_bits, *from_bits);
                        exprs.insert(*r, result_expr.clone());
                        result
                            .entry(result_expr)
                            .or_default()
                            .push((block_id, instruction_idx, *r));
                    }
                    OpCode::SExt {
                        result: r,
                        value,
                        from_bits,
                        to_bits,
                    } => {
                        let value_expr = get_expr(&exprs, value);
                        let result_expr = value_expr.sext(*from_bits, *to_bits);
                        exprs.insert(*r, result_expr.clone());
                        result
                            .entry(result_expr)
                            .or_default()
                            .push((block_id, instruction_idx, *r));
                    }
                    OpCode::ValueOf { result: r, value } => {
                        let value_expr = get_expr(&exprs, value);
                        let result_expr = value_expr.value_of();
                        exprs.insert(*r, result_expr.clone());
                        result
                            .entry(result_expr)
                            .or_default()
                            .push((block_id, instruction_idx, *r));
                    }
                    OpCode::MulConst {
                        result: r,
                        const_val,
                        var,
                    } => {
                        // Semantically equivalent to a regular `Mul` on the same
                        // operands; folding into Expr::Mul lets it CSE with both
                        // BinaryArithOp::Mul and other MulConst occurrences.
                        let lhs_expr = get_expr(&exprs, const_val);
                        let rhs_expr = get_expr(&exprs, var);
                        let result_expr = lhs_expr.mul(&rhs_expr);
                        exprs.insert(*r, result_expr.clone());
                        result
                            .entry(result_expr)
                            .or_default()
                            .push((block_id, instruction_idx, *r));
                    }
                    OpCode::ToBits {
                        result: r,
                        value,
                        endianness,
                        count,
                    } => {
                        let value_expr = get_expr(&exprs, value);
                        let result_expr = value_expr.bits_of(*endianness, *count);
                        exprs.insert(*r, result_expr.clone());
                        result
                            .entry(result_expr)
                            .or_default()
                            .push((block_id, instruction_idx, *r));
                    }
                    OpCode::ToRadix {
                        result: r,
                        value,
                        radix,
                        endianness,
                        count,
                    } => {
                        // We only fold the `Bytes` case; `Dyn(_)` references
                        // a runtime ValueId we don't currently encode in Expr.
                        match radix {
                            Radix::Bytes => {
                                let value_expr = get_expr(&exprs, value);
                                let result_expr =
                                    value_expr.bytes_of(*endianness, *count);
                                exprs.insert(*r, result_expr.clone());
                                result
                                    .entry(result_expr)
                                    .or_default()
                                    .push((block_id, instruction_idx, *r));
                            }
                            Radix::Dyn(_) => {}
                        }
                    }
                    OpCode::WriteWitness {
                        result: Some(r),
                        value,
                        pinned: false,
                    } => {
                        // Non-pinned: two writes with the same hint expression
                        // can share a slot. The constraint system is unchanged
                        // because every constraint that referenced either now
                        // references the survivor.
                        let hint_expr = get_expr(&exprs, value);
                        let result_expr = hint_expr.witness();
                        exprs.insert(*r, result_expr.clone());
                        result
                            .entry(result_expr)
                            .or_default()
                            .push((block_id, instruction_idx, *r));
                    }
                    // Pinned `WriteWitness` and `FreshWitness` produce slots
                    // that must NOT be merged with anything. We deliberately
                    // skip inserting an Expr — downstream `get_expr` lookups
                    // fall back to `Expr::Variable(value_id)`, which is unique
                    // per ValueId by construction.
                    OpCode::WriteWitness { result: Some(_), pinned: true, .. } => {}
                    OpCode::FreshWitness { .. } => {}
                    OpCode::Rangecheck { value, max_bits } => {
                        let value_expr = get_expr(&exprs, value);
                        effects
                            .entry(Effect::Rangecheck(value_expr, *max_bits))
                            .or_default()
                            .push((block_id, instruction_idx));
                    }
                    OpCode::Lookup {
                        target: crate::compiler::ssa::LookupTarget::Rangecheck(8),
                        keys,
                        results: _,
                        flag,
                    } if keys.len() == 1 => {
                        let key_expr = get_expr(&exprs, &keys[0]);
                        let flag_expr = get_expr(&exprs, flag);
                        effects
                            .entry(Effect::ByteLookup(key_expr, flag_expr))
                            .or_default()
                            .push((block_id, instruction_idx));
                    }
                    OpCode::WriteWitness { result: None, .. }
                    | OpCode::Constrain { .. }
                    | OpCode::NextDCoeff { result: _ }
                    | OpCode::BumpD { matrix: _, variable: _, sensitivity: _ }
                    | OpCode::Alloc { .. }
                    | OpCode::Store { .. }
                    | OpCode::Load { .. }
                    | OpCode::Assert { .. }
                    | OpCode::AssertCmp { .. }
                    | OpCode::AssertR1C { .. }
                    | OpCode::Call { .. }
                    | OpCode::MkSeq { .. }
                    | OpCode::MkTuple { .. }
                    | OpCode::ArraySet { .. }
                    | OpCode::SlicePush { .. }
                    | OpCode::SliceLen { .. }
                    | OpCode::MemOp { kind: _, value: _ }
                    | OpCode::Lookup { .. }
                    | OpCode::DLookup { target: _, keys: _, results: _, flag: _ }
                    | OpCode::Todo { .. }
                    | OpCode::InitGlobal { .. }
                    | OpCode::DropGlobal { .. }
                    | OpCode::Spread { .. }
                    | OpCode::Unspread { .. } => {}
                    OpCode::Not { result: r, value } => {
                        let value_expr = get_expr(&exprs, value);
                        let result_expr = value_expr.not();
                        exprs.insert(*r, result_expr.clone());
                        result.entry(result_expr).or_default().push((
                            block_id,
                            instruction_idx,
                            *r,
                        ));
                    }
                    OpCode::TupleProj {
                        result: r,
                        tuple,
                        idx,
                    } => {
                        let tuple_expr = get_expr(&exprs, tuple);
                        let index_expr = Expr::UConst(64, *idx as u128);
                        let result_expr = tuple_expr.tuple_get(&index_expr);
                        exprs.insert(*r, result_expr.clone());
                        result.entry(result_expr).or_default().push((
                            block_id,
                            instruction_idx,
                            *r,
                        ));
                    },
                    OpCode::Const {
                        result: r,
                        value: cv,
                    } => match cv {
                        ConstValue::U(size, val) => {
                            let expr = Expr::UConst(*size, *val);
                            exprs.insert(*r, expr.clone());
                            result
                                .entry(expr)
                                .or_default()
                                .push((block_id, instruction_idx, *r));
                        }
                        ConstValue::I(size, val) => {
                            let expr = Expr::IConst(*size, *val);
                            exprs.insert(*r, expr.clone());
                            result
                                .entry(expr)
                                .or_default()
                                .push((block_id, instruction_idx, *r));
                        }
                        ConstValue::Field(val) => {
                            let expr = Expr::fconst(*val);
                            exprs.insert(*r, expr.clone());
                            result
                                .entry(expr)
                                .or_default()
                                .push((block_id, instruction_idx, *r));
                        }
                        ConstValue::FnPtr(_) => {}
                    }
                    OpCode::Guard { .. } => {
                        // Guards are opaque to CSE
                    }
                }
            }
        }
        (result, effects)
    }
}
