use std::{
    collections::{HashMap, HashSet},
    fmt::{Debug, Display},
};

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

    /// Pure normalization: flatten nested Adds and sort for a canonical form.
    /// Algebraic folds (`x + 0 → x`) live in the simplifier pass — by the time
    /// we hash for CSE, those have already fired.
    pub fn add(&self, other: &Self) -> Self {
        let mut adds: Vec<Self> = self
            .get_adds()
            .into_iter()
            .chain(other.get_adds().into_iter())
            .collect();
        adds.sort();
        Self::Add(adds)
    }

    pub fn mul(&self, other: &Self) -> Self {
        let mut muls: Vec<Self> = self
            .get_muls()
            .into_iter()
            .chain(other.get_muls().into_iter())
            .collect();
        muls.sort();
        Self::Mul(muls)
    }

    pub fn div(&self, other: &Self) -> Self {
        Self::Div(Box::new(self.clone()), Box::new(other.clone()))
    }

    pub fn modulo(&self, other: &Self) -> Self {
        Self::Mod(Box::new(self.clone()), Box::new(other.clone()))
    }

    pub fn sub(&self, other: &Self) -> Self {
        Self::Sub(Box::new(self.clone()), Box::new(other.clone()))
    }

    pub fn and(&self, other: &Self) -> Self {
        let mut ands = self.get_ands();
        ands.extend(other.get_ands());
        ands.sort();
        ands.dedup();
        Self::And(ands)
    }

    pub fn or(&self, other: &Self) -> Self {
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
        Self::Or(ors)
    }

    pub fn xor(&self, other: &Self) -> Self {
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
        Self::Shl(Box::new(self.clone()), Box::new(other.clone()))
    }

    pub fn shr(&self, other: &Self) -> Self {
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
        Self::Select(
            Box::new(self.clone()),
            Box::new(then.clone()),
            Box::new(otherwise.clone()),
        )
    }

    pub fn not(&self) -> Self {
        Self::Not(Box::new(self.clone()))
    }

    pub fn cast(&self, target: CastTarget) -> Self {
        Self::Cast(Box::new(self.clone()), target)
    }

    pub fn truncate(&self, to_bits: usize, from_bits: usize) -> Self {
        Self::Truncate(Box::new(self.clone()), to_bits, from_bits)
    }

    pub fn sext(&self, from_bits: usize, to_bits: usize) -> Self {
        Self::SExt(Box::new(self.clone()), from_bits, to_bits)
    }

    pub fn value_of(&self) -> Self {
        Self::ValueOf(Box::new(self.clone()))
    }

    pub fn bytes_of(&self, endianness: Endianness, count: usize) -> Self {
        Self::BytesOf(Box::new(self.clone()), endianness, count)
    }

    pub fn bits_of(&self, endianness: Endianness, count: usize) -> Self {
        Self::BitsOf(Box::new(self.clone()), endianness, count)
    }

    pub fn witness(&self) -> Self {
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
