//! Deduplicates expressions when one occurrence dominates the other.
//!
//! Does not float expressions across branches or otherwise move them outside the block in which
//! they appear (#172).

use std::{
    collections::{HashMap, HashSet},
    fmt::{Debug, Display},
};

use crate::compiler::{
    analysis::flow_analysis::{CFG, FlowAnalysis},
    ssa::{
        BlockId, ValueId,
        hlssa::{
            BinaryArithOpKind, CastTarget, CmpKind, ConstValue, Endianness, HLFunction, HLSSA,
            OpCode, Radix,
        },
    },
};
use crate::compiler::{
    pass_manager::{AnalysisId, AnalysisStore, Pass},
    passes::fix_double_jumps::ValueReplacements,
};

// Lowered bit-range gadgets can build very deep arithmetic trees. Keeping those
// trees as CSE keys makes hashing/cloning dominate compilation, while matching
// across such large expressions is not worth the cost.
const MAX_CSE_EXPR_NODES: usize = 256;

#[derive(Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
enum Expr {
    Add(Vec<Expr>),
    Mul(Vec<Expr>),
    Div {
        lhs: Box<Expr>,
        rhs: Box<Expr>,
    },
    Mod {
        lhs: Box<Expr>,
        rhs: Box<Expr>,
    },
    Sub {
        lhs: Box<Expr>,
        rhs: Box<Expr>,
    },
    FConst(ark_bn254::Fr),
    UConst {
        bits: usize,
        value: u128,
    },
    IConst {
        bits: usize,
        value: u128,
    },
    Variable(u64),
    Eq {
        lhs: Box<Expr>,
        rhs: Box<Expr>,
    },
    Lt {
        lhs: Box<Expr>,
        rhs: Box<Expr>,
    },
    And(Vec<Expr>),
    Or(Vec<Expr>),
    Xor(Vec<Expr>),
    Shl {
        lhs: Box<Expr>,
        rhs: Box<Expr>,
    },
    Shr {
        lhs: Box<Expr>,
        rhs: Box<Expr>,
    },
    BitRange {
        value: Box<Expr>,
        offset: usize,
        width: usize,
        source_width: Option<usize>,
    },
    Select {
        condition: Box<Expr>,
        then: Box<Expr>,
        otherwise: Box<Expr>,
    },
    ArrayGet {
        array: Box<Expr>,
        index: Box<Expr>,
    },
    TupleGet {
        tuple: Box<Expr>,
        index: Box<Expr>,
    },
    Not(Box<Expr>),
    ReadGlobal(u64),
    Cast {
        value: Box<Expr>,
        target: CastTarget,
    },
    Truncate {
        value: Box<Expr>,
        to_bits: usize,
        from_bits: usize,
    },
    SExt {
        value: Box<Expr>,
        from_bits: usize,
        to_bits: usize,
    },
    ValueOf(Box<Expr>),
    BytesOf {
        value: Box<Expr>,
        endianness: Endianness,
        count: usize,
    },
    BitsOf {
        value: Box<Expr>,
        endianness: Endianness,
        count: usize,
    },
    Witness(Box<Expr>),
}

#[derive(Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
enum Assertion {
    Rangecheck { value: Expr, max_bits: usize },
    ByteLookup { key: Expr, flag: Expr },
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

    /// Flatten nested Adds and sort for a canonical form.
    pub fn add(&self, other: &Self) -> Self {
        let mut adds: Vec<Self> = self
            .get_adds()
            .into_iter()
            .chain(other.get_adds())
            .collect();
        adds.sort();
        Self::Add(adds)
    }

    pub fn mul(&self, other: &Self) -> Self {
        let mut muls: Vec<Self> = self
            .get_muls()
            .into_iter()
            .chain(other.get_muls())
            .collect();
        muls.sort();
        Self::Mul(muls)
    }

    pub fn div(&self, other: &Self) -> Self {
        Self::Div {
            lhs: Box::new(self.clone()),
            rhs: Box::new(other.clone()),
        }
    }

    pub fn modulo(&self, other: &Self) -> Self {
        Self::Mod {
            lhs: Box::new(self.clone()),
            rhs: Box::new(other.clone()),
        }
    }

    pub fn sub(&self, other: &Self) -> Self {
        Self::Sub {
            lhs: Box::new(self.clone()),
            rhs: Box::new(other.clone()),
        }
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
        Self::Shl {
            lhs: Box::new(self.clone()),
            rhs: Box::new(other.clone()),
        }
    }

    pub fn shr(&self, other: &Self) -> Self {
        Self::Shr {
            lhs: Box::new(self.clone()),
            rhs: Box::new(other.clone()),
        }
    }

    pub fn bit_range(&self, offset: usize, width: usize, source_width: Option<usize>) -> Self {
        Self::BitRange {
            value: Box::new(self.clone()),
            offset,
            width,
            source_width,
        }
    }

    pub fn fconst(value: ark_bn254::Fr) -> Self {
        Self::FConst(value)
    }

    pub fn eq(&self, other: &Self) -> Self {
        Self::Eq {
            lhs: Box::new(self.clone()),
            rhs: Box::new(other.clone()),
        }
    }

    pub fn lt(&self, other: &Self) -> Self {
        Self::Lt {
            lhs: Box::new(self.clone()),
            rhs: Box::new(other.clone()),
        }
    }

    pub fn array_get(&self, index: &Self) -> Self {
        Self::ArrayGet {
            array: Box::new(self.clone()),
            index: Box::new(index.clone()),
        }
    }

    pub fn tuple_get(&self, index: &Self) -> Self {
        Self::TupleGet {
            tuple: Box::new(self.clone()),
            index: Box::new(index.clone()),
        }
    }

    pub fn select(&self, then: &Self, otherwise: &Self) -> Self {
        Self::Select {
            condition: Box::new(self.clone()),
            then: Box::new(then.clone()),
            otherwise: Box::new(otherwise.clone()),
        }
    }

    pub fn not(&self) -> Self {
        Self::Not(Box::new(self.clone()))
    }

    pub fn cast(&self, target: CastTarget) -> Self {
        Self::Cast {
            value: Box::new(self.clone()),
            target,
        }
    }

    pub fn truncate(&self, to_bits: usize, from_bits: usize) -> Self {
        Self::Truncate {
            value: Box::new(self.clone()),
            to_bits,
            from_bits,
        }
    }

    pub fn sext(&self, from_bits: usize, to_bits: usize) -> Self {
        Self::SExt {
            value: Box::new(self.clone()),
            from_bits,
            to_bits,
        }
    }

    pub fn value_of(&self) -> Self {
        Self::ValueOf(Box::new(self.clone()))
    }

    pub fn bytes_of(&self, endianness: Endianness, count: usize) -> Self {
        Self::BytesOf {
            value: Box::new(self.clone()),
            endianness,
            count,
        }
    }

    pub fn bits_of(&self, endianness: Endianness, count: usize) -> Self {
        Self::BitsOf {
            value: Box::new(self.clone()),
            endianness,
            count,
        }
    }

    pub fn witness(&self) -> Self {
        Self::Witness(Box::new(self.clone()))
    }

    fn exceeds_node_budget(&self, budget: usize) -> bool {
        fn visit(expr: &Expr, remaining: &mut usize) -> bool {
            if *remaining == 0 {
                return true;
            }
            *remaining -= 1;

            match expr {
                Expr::Add(exprs)
                | Expr::Mul(exprs)
                | Expr::And(exprs)
                | Expr::Or(exprs)
                | Expr::Xor(exprs) => exprs.iter().any(|expr| visit(expr, remaining)),
                Expr::Div { lhs, rhs }
                | Expr::Mod { lhs, rhs }
                | Expr::Sub { lhs, rhs }
                | Expr::Eq { lhs, rhs }
                | Expr::Lt { lhs, rhs }
                | Expr::Shl { lhs, rhs }
                | Expr::Shr { lhs, rhs }
                | Expr::ArrayGet {
                    array: lhs,
                    index: rhs,
                } => visit(lhs, remaining) || visit(rhs, remaining),
                Expr::BitRange { value, .. }
                | Expr::Not(value)
                | Expr::Cast { value, .. }
                | Expr::Truncate { value, .. }
                | Expr::SExt { value, .. }
                | Expr::ValueOf(value)
                | Expr::BytesOf { value, .. }
                | Expr::BitsOf { value, .. }
                | Expr::Witness(value) => visit(value, remaining),
                Expr::Select {
                    condition,
                    then,
                    otherwise,
                } => {
                    visit(condition, remaining)
                        || visit(then, remaining)
                        || visit(otherwise, remaining)
                }
                Expr::TupleGet { tuple, index } => {
                    visit(tuple, remaining) || visit(index, remaining)
                }
                Expr::FConst(_)
                | Expr::UConst { .. }
                | Expr::IConst { .. }
                | Expr::Variable(_)
                | Expr::ReadGlobal(_) => false,
            }
        }

        let mut remaining = budget;
        visit(self, &mut remaining)
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
            Self::Div { lhs, rhs } => write!(f, "({} / {})", lhs, rhs),
            Self::Mod { lhs, rhs } => write!(f, "({} % {})", lhs, rhs),
            Self::Sub { lhs, rhs } => write!(f, "({} - {})", lhs, rhs),
            Self::FConst(value) => write!(f, "{}", value),
            Self::UConst { bits, value } => write!(f, "u{}({})", bits, value),
            Self::IConst { bits, value } => write!(f, "i{}({})", bits, value),
            Self::Variable(value) => write!(f, "v{}", value),
            Self::Eq { lhs, rhs } => write!(f, "({} == {})", lhs, rhs),
            Self::Lt { lhs, rhs } => write!(f, "({} < {})", lhs, rhs),
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
            Self::Shl { lhs, rhs } => write!(f, "({} << {})", lhs, rhs),
            Self::Shr { lhs, rhs } => write!(f, "({} >> {})", lhs, rhs),
            Self::BitRange {
                value,
                offset,
                width,
                source_width,
            } => {
                let source_width = source_width
                    .map(|source_width| format!(", source_width={source_width}"))
                    .unwrap_or_default();
                write!(
                    f,
                    "bit_range({}, {}, {}){}",
                    value, offset, width, source_width
                )
            }
            Self::Select {
                condition,
                then,
                otherwise,
            } => {
                write!(f, "({} ? {} : {})", condition, then, otherwise)
            }
            Self::ArrayGet { array, index } => write!(f, "{}[{}]", array, index),
            Self::TupleGet { tuple, index } => write!(f, "{}.{}", tuple, index),
            Self::Not(value) => write!(f, "(~{})", value),
            Self::ReadGlobal(index) => write!(f, "g{}", index),
            Self::Cast { value, target } => write!(f, "cast({}, {})", value, target),
            Self::Truncate {
                value,
                to_bits,
                from_bits,
            } => {
                write!(f, "trunc({}, {}, {})", value, to_bits, from_bits)
            }
            Self::SExt {
                value,
                from_bits,
                to_bits,
            } => {
                write!(f, "sext({}, {}, {})", value, from_bits, to_bits)
            }
            Self::ValueOf(value) => write!(f, "value_of({})", value),
            Self::BytesOf {
                value,
                endianness,
                count,
            } => {
                write!(f, "bytes_of({}, {:?}, {})", value, endianness, count)
            }
            Self::BitsOf {
                value,
                endianness,
                count,
            } => {
                write!(f, "bits_of({}, {:?}, {})", value, endianness, count)
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
            let (exprs, assertions) = self.gather_expressions(function, cfg);
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

            // Side-effect dedup: same dominance grouping as the value loop,
            // but duplicates are dropped rather than redirected.
            let mut to_remove: HashSet<(BlockId, usize)> = HashSet::new();
            for (_, occurrences) in assertions {
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
        HashMap<Assertion, Vec<(BlockId, usize)>>,
    ) {
        let mut result: HashMap<Expr, Vec<(BlockId, usize, ValueId)>> = HashMap::new();
        let mut assertions: HashMap<Assertion, Vec<(BlockId, usize)>> = HashMap::new();
        let mut exprs = HashMap::<ValueId, Expr>::new();

        fn get_expr(exprs: &HashMap<ValueId, Expr>, value_id: &ValueId) -> Expr {
            exprs
                .get(&value_id)
                .cloned()
                .unwrap_or(Expr::variable(*value_id))
        }

        fn record_expr(
            exprs: &mut HashMap<ValueId, Expr>,
            result: &mut HashMap<Expr, Vec<(BlockId, usize, ValueId)>>,
            block_id: BlockId,
            instruction_idx: usize,
            value_id: ValueId,
            expr: Expr,
        ) {
            if expr.exceeds_node_budget(MAX_CSE_EXPR_NODES) {
                exprs.insert(value_id, Expr::variable(value_id));
                return;
            }

            exprs.insert(value_id, expr.clone());
            result
                .entry(expr)
                .or_default()
                .push((block_id, instruction_idx, value_id));
        }

        fn record_assertion(
            assertions: &mut HashMap<Assertion, Vec<(BlockId, usize)>>,
            block_id: BlockId,
            instruction_idx: usize,
            assertion: Assertion,
        ) {
            let exceeds_budget = match &assertion {
                Assertion::Rangecheck { value, .. } => {
                    value.exceeds_node_budget(MAX_CSE_EXPR_NODES)
                }
                Assertion::ByteLookup { key, flag } => {
                    key.exceeds_node_budget(MAX_CSE_EXPR_NODES)
                        || flag.exceeds_node_budget(MAX_CSE_EXPR_NODES)
                }
            };
            if exceeds_budget {
                return;
            }

            assertions
                .entry(assertion)
                .or_default()
                .push((block_id, instruction_idx));
        }

        for block_id in cfg.get_domination_pre_order() {
            let block = ssa.get_block(block_id);

            for (instruction_idx, instruction) in block.get_instructions().enumerate() {
                match instruction {
                    OpCode::BinaryArithOp {
                        kind: BinaryArithOpKind::Add,
                        result: r,
                        lhs,
                        rhs,
                    } => {
                        let lhs_expr = get_expr(&exprs, lhs);
                        let rhs_expr = get_expr(&exprs, rhs);
                        let result_expr = lhs_expr.add(&rhs_expr);
                        record_expr(
                            &mut exprs,
                            &mut result,
                            block_id,
                            instruction_idx,
                            *r,
                            result_expr,
                        );
                    }
                    OpCode::BinaryArithOp {
                        kind: BinaryArithOpKind::Mul,
                        result: r,
                        lhs,
                        rhs,
                    } => {
                        let lhs_expr = get_expr(&exprs, lhs);
                        let rhs_expr = get_expr(&exprs, rhs);
                        let result_expr = lhs_expr.mul(&rhs_expr);
                        record_expr(
                            &mut exprs,
                            &mut result,
                            block_id,
                            instruction_idx,
                            *r,
                            result_expr,
                        );
                    }
                    OpCode::BinaryArithOp {
                        kind: BinaryArithOpKind::Div,
                        result: r,
                        lhs,
                        rhs,
                    } => {
                        let lhs_expr = get_expr(&exprs, lhs);
                        let rhs_expr = get_expr(&exprs, rhs);
                        let result_expr = lhs_expr.div(&rhs_expr);
                        record_expr(
                            &mut exprs,
                            &mut result,
                            block_id,
                            instruction_idx,
                            *r,
                            result_expr,
                        );
                    }
                    OpCode::BinaryArithOp {
                        kind: BinaryArithOpKind::Sub,
                        result: r,
                        lhs,
                        rhs,
                    } => {
                        let lhs_expr = get_expr(&exprs, lhs);
                        let rhs_expr = get_expr(&exprs, rhs);
                        let result_expr = lhs_expr.sub(&rhs_expr);
                        record_expr(
                            &mut exprs,
                            &mut result,
                            block_id,
                            instruction_idx,
                            *r,
                            result_expr,
                        );
                    }
                    OpCode::Cmp {
                        kind: CmpKind::Eq,
                        result: r,
                        lhs,
                        rhs,
                    } => {
                        let lhs_expr = get_expr(&exprs, lhs);
                        let rhs_expr = get_expr(&exprs, rhs);
                        let result_expr = lhs_expr.eq(&rhs_expr);
                        record_expr(
                            &mut exprs,
                            &mut result,
                            block_id,
                            instruction_idx,
                            *r,
                            result_expr,
                        );
                    }
                    OpCode::Cmp {
                        kind: CmpKind::Lt,
                        result: r,
                        lhs,
                        rhs,
                    } => {
                        let lhs_expr = get_expr(&exprs, lhs);
                        let rhs_expr = get_expr(&exprs, rhs);
                        let result_expr = lhs_expr.lt(&rhs_expr);
                        record_expr(
                            &mut exprs,
                            &mut result,
                            block_id,
                            instruction_idx,
                            *r,
                            result_expr,
                        );
                    }
                    OpCode::BinaryArithOp {
                        kind: BinaryArithOpKind::Mod,
                        result: r,
                        lhs,
                        rhs,
                    } => {
                        let lhs_expr = get_expr(&exprs, lhs);
                        let rhs_expr = get_expr(&exprs, rhs);
                        let result_expr = lhs_expr.modulo(&rhs_expr);
                        record_expr(
                            &mut exprs,
                            &mut result,
                            block_id,
                            instruction_idx,
                            *r,
                            result_expr,
                        );
                    }
                    OpCode::BinaryArithOp {
                        kind: BinaryArithOpKind::And,
                        result: r,
                        lhs,
                        rhs,
                    } => {
                        let lhs_expr = get_expr(&exprs, lhs);
                        let rhs_expr = get_expr(&exprs, rhs);
                        let result_expr = lhs_expr.and(&rhs_expr);
                        record_expr(
                            &mut exprs,
                            &mut result,
                            block_id,
                            instruction_idx,
                            *r,
                            result_expr,
                        );
                    }
                    OpCode::BinaryArithOp {
                        kind: BinaryArithOpKind::Or,
                        result: r,
                        lhs,
                        rhs,
                    } => {
                        let lhs_expr = get_expr(&exprs, lhs);
                        let rhs_expr = get_expr(&exprs, rhs);
                        let result_expr = lhs_expr.or(&rhs_expr);
                        record_expr(
                            &mut exprs,
                            &mut result,
                            block_id,
                            instruction_idx,
                            *r,
                            result_expr,
                        );
                    }
                    OpCode::BinaryArithOp {
                        kind: BinaryArithOpKind::Xor,
                        result: r,
                        lhs,
                        rhs,
                    } => {
                        let lhs_expr = get_expr(&exprs, lhs);
                        let rhs_expr = get_expr(&exprs, rhs);
                        let result_expr = lhs_expr.xor(&rhs_expr);
                        record_expr(
                            &mut exprs,
                            &mut result,
                            block_id,
                            instruction_idx,
                            *r,
                            result_expr,
                        );
                    }
                    OpCode::BinaryArithOp {
                        kind: BinaryArithOpKind::Shl,
                        result: r,
                        lhs,
                        rhs,
                    } => {
                        let lhs_expr = get_expr(&exprs, lhs);
                        let rhs_expr = get_expr(&exprs, rhs);
                        let result_expr = lhs_expr.shl(&rhs_expr);
                        record_expr(
                            &mut exprs,
                            &mut result,
                            block_id,
                            instruction_idx,
                            *r,
                            result_expr,
                        );
                    }
                    OpCode::BinaryArithOp {
                        kind: BinaryArithOpKind::Shr,
                        result: r,
                        lhs,
                        rhs,
                    } => {
                        let lhs_expr = get_expr(&exprs, lhs);
                        let rhs_expr = get_expr(&exprs, rhs);
                        let result_expr = lhs_expr.shr(&rhs_expr);
                        record_expr(
                            &mut exprs,
                            &mut result,
                            block_id,
                            instruction_idx,
                            *r,
                            result_expr,
                        );
                    }
                    OpCode::ArrayGet {
                        result: r,
                        array,
                        index,
                    } => {
                        let array_expr = get_expr(&exprs, array);
                        let index_expr = get_expr(&exprs, index);
                        let result_expr = array_expr.array_get(&index_expr);
                        record_expr(
                            &mut exprs,
                            &mut result,
                            block_id,
                            instruction_idx,
                            *r,
                            result_expr,
                        );
                    }
                    OpCode::Select {
                        result: r,
                        cond,
                        if_t: then,
                        if_f: otherwise,
                    } => {
                        let cond_expr = get_expr(&exprs, cond);
                        let then_expr = get_expr(&exprs, then);
                        let otherwise_expr = get_expr(&exprs, otherwise);
                        let result_expr = cond_expr.select(&then_expr, &otherwise_expr);
                        record_expr(
                            &mut exprs,
                            &mut result,
                            block_id,
                            instruction_idx,
                            *r,
                            result_expr,
                        );
                    }
                    OpCode::ReadGlobal {
                        result: r,
                        offset: index,
                        result_type: _,
                    } => {
                        let result_expr = Expr::ReadGlobal(*index);
                        record_expr(
                            &mut exprs,
                            &mut result,
                            block_id,
                            instruction_idx,
                            *r,
                            result_expr,
                        );
                    }
                    OpCode::Cast {
                        result: r,
                        value,
                        target,
                    } => {
                        let value_expr = get_expr(&exprs, value);
                        let result_expr = value_expr.cast(*target);
                        record_expr(
                            &mut exprs,
                            &mut result,
                            block_id,
                            instruction_idx,
                            *r,
                            result_expr,
                        );
                    }
                    OpCode::Truncate {
                        result: r,
                        value,
                        to_bits,
                        from_bits,
                    } => {
                        let value_expr = get_expr(&exprs, value);
                        let result_expr = value_expr.truncate(*to_bits, *from_bits);
                        record_expr(
                            &mut exprs,
                            &mut result,
                            block_id,
                            instruction_idx,
                            *r,
                            result_expr,
                        );
                    }
                    OpCode::SExt {
                        result: r,
                        value,
                        from_bits,
                        to_bits,
                    } => {
                        let value_expr = get_expr(&exprs, value);
                        let result_expr = value_expr.sext(*from_bits, *to_bits);
                        record_expr(
                            &mut exprs,
                            &mut result,
                            block_id,
                            instruction_idx,
                            *r,
                            result_expr,
                        );
                    }
                    OpCode::BitRange {
                        result: r,
                        value,
                        offset,
                        width,
                        source_width,
                    } => {
                        let value_expr = get_expr(&exprs, value);
                        let result_expr = value_expr.bit_range(*offset, *width, *source_width);
                        record_expr(
                            &mut exprs,
                            &mut result,
                            block_id,
                            instruction_idx,
                            *r,
                            result_expr,
                        );
                    }
                    OpCode::ValueOf { result: r, value } => {
                        let value_expr = get_expr(&exprs, value);
                        let result_expr = value_expr.value_of();
                        record_expr(
                            &mut exprs,
                            &mut result,
                            block_id,
                            instruction_idx,
                            *r,
                            result_expr,
                        );
                    }
                    OpCode::MulConst {
                        result: r,
                        const_val,
                        var,
                    } => {
                        // Fold into Expr::Mul so MulConst dedups with BinaryArithOp::Mul.
                        let lhs_expr = get_expr(&exprs, const_val);
                        let rhs_expr = get_expr(&exprs, var);
                        let result_expr = lhs_expr.mul(&rhs_expr);
                        record_expr(
                            &mut exprs,
                            &mut result,
                            block_id,
                            instruction_idx,
                            *r,
                            result_expr,
                        );
                    }
                    OpCode::ToBits {
                        result: r,
                        value,
                        endianness,
                        count,
                    } => {
                        let value_expr = get_expr(&exprs, value);
                        let result_expr = value_expr.bits_of(*endianness, *count);
                        record_expr(
                            &mut exprs,
                            &mut result,
                            block_id,
                            instruction_idx,
                            *r,
                            result_expr,
                        );
                    }
                    OpCode::ToRadix {
                        result: r,
                        value,
                        radix,
                        endianness,
                        count,
                    } => {
                        // `Dyn(_)` carries a runtime ValueId we don't encode in Expr,
                        // so only the static `Bytes` case is keyed.
                        match radix {
                            Radix::Bytes => {
                                let value_expr = get_expr(&exprs, value);
                                let result_expr = value_expr.bytes_of(*endianness, *count);
                                record_expr(
                                    &mut exprs,
                                    &mut result,
                                    block_id,
                                    instruction_idx,
                                    *r,
                                    result_expr,
                                );
                            }
                            Radix::Dyn(_) => {}
                        }
                    }
                    OpCode::WriteWitness {
                        result: Some(r),
                        value,
                        pinned: false,
                    } => {
                        // Two non-pinned writes with the same hint can share a slot.
                        let hint_expr = get_expr(&exprs, value);
                        let result_expr = hint_expr.witness();
                        record_expr(
                            &mut exprs,
                            &mut result,
                            block_id,
                            instruction_idx,
                            *r,
                            result_expr,
                        );
                    }
                    // Pinned WriteWitness and FreshWitness must not merge with anything;
                    // skipping the Expr insert leaves `get_expr` to fall back to a
                    // unique-per-ValueId `Expr::Variable`.
                    OpCode::WriteWitness {
                        result: Some(_),
                        pinned: true,
                        ..
                    } => {}
                    OpCode::FreshWitness { .. } => {}
                    OpCode::Rangecheck { value, max_bits } => {
                        let value_expr = get_expr(&exprs, value);
                        record_assertion(
                            &mut assertions,
                            block_id,
                            instruction_idx,
                            Assertion::Rangecheck {
                                value: value_expr,
                                max_bits: *max_bits,
                            },
                        );
                    }
                    OpCode::Lookup {
                        target: crate::compiler::ssa::hlssa::LookupTarget::Rangecheck(8),
                        args,
                        flag,
                    } if args.len() == 1 => {
                        let key_expr = get_expr(&exprs, &args[0]);
                        let flag_expr = get_expr(&exprs, flag);
                        record_assertion(
                            &mut assertions,
                            block_id,
                            instruction_idx,
                            Assertion::ByteLookup {
                                key: key_expr,
                                flag: flag_expr,
                            },
                        );
                    }
                    OpCode::WriteWitness { result: None, .. }
                    | OpCode::Constrain { .. }
                    | OpCode::NextDCoeff { result: _ }
                    | OpCode::BumpD {
                        matrix: _,
                        variable: _,
                        sensitivity: _,
                    }
                    | OpCode::Alloc { .. }
                    | OpCode::Store { .. }
                    | OpCode::Load { .. }
                    | OpCode::Assert { .. }
                    | OpCode::AssertCmp { .. }
                    | OpCode::AssertR1C { .. }
                    | OpCode::Call { .. }
                    | OpCode::MkSeq { .. }
                    | OpCode::MkRepeated { .. }
                    | OpCode::MkTuple { .. }
                    | OpCode::ArraySet { .. }
                    | OpCode::SlicePush { .. }
                    | OpCode::SliceLen { .. }
                    | OpCode::MemOp { kind: _, value: _ }
                    | OpCode::Lookup { .. }
                    | OpCode::DLookup {
                        target: _,
                        args: _,
                        flag: _,
                    }
                    | OpCode::Todo { .. }
                    | OpCode::InitGlobal { .. }
                    | OpCode::DropGlobal { .. }
                    | OpCode::Spread { .. }
                    | OpCode::Unspread { .. } => {}
                    OpCode::Not { result: r, value } => {
                        let value_expr = get_expr(&exprs, value);
                        let result_expr = value_expr.not();
                        record_expr(
                            &mut exprs,
                            &mut result,
                            block_id,
                            instruction_idx,
                            *r,
                            result_expr,
                        );
                    }
                    OpCode::TupleProj {
                        result: r,
                        tuple,
                        idx,
                    } => {
                        let tuple_expr = get_expr(&exprs, tuple);
                        let index_expr = Expr::UConst {
                            bits: 64,
                            value: *idx as u128,
                        };
                        let result_expr = tuple_expr.tuple_get(&index_expr);
                        record_expr(
                            &mut exprs,
                            &mut result,
                            block_id,
                            instruction_idx,
                            *r,
                            result_expr,
                        );
                    }
                    OpCode::Const {
                        result: r,
                        value: cv,
                    } => match cv {
                        ConstValue::U(size, val) => {
                            let expr = Expr::UConst {
                                bits: *size,
                                value: *val,
                            };
                            record_expr(
                                &mut exprs,
                                &mut result,
                                block_id,
                                instruction_idx,
                                *r,
                                expr,
                            );
                        }
                        ConstValue::I(size, val) => {
                            let expr = Expr::IConst {
                                bits: *size,
                                value: *val,
                            };
                            record_expr(
                                &mut exprs,
                                &mut result,
                                block_id,
                                instruction_idx,
                                *r,
                                expr,
                            );
                        }
                        ConstValue::Field(val) => {
                            let expr = Expr::fconst(*val);
                            record_expr(
                                &mut exprs,
                                &mut result,
                                block_id,
                                instruction_idx,
                                *r,
                                expr,
                            );
                        }
                        ConstValue::FnPtr(_) => {}
                    },
                    OpCode::Guard { .. } => {
                        // Guards are opaque to CSE
                    }
                }
            }
        }
        (result, assertions)
    }
}
