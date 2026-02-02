use std::{
    collections::HashMap,
    fmt::{Debug, Display},
};

use crate::compiler::{
    flow_analysis::{CFG, FlowAnalysis},
    ssa::{BinaryArithOpKind, BlockId, CmpKind, Const, Function, OpCode, SSA, TupleIdx, ValueId},
};
use crate::compiler::{pass_manager::Pass, passes::fix_double_jumps::ValueReplacements};

#[derive(Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
enum Expr {
    Add(Vec<Expr>),
    Mul(Vec<Expr>),
    Div(Box<Expr>, Box<Expr>),
    Sub(Box<Expr>, Box<Expr>),
    FConst(ark_bn254::Fr),
    UConst(usize, u128),
    Variable(u64),
    Eq(Box<Expr>, Box<Expr>),
    Lt(Box<Expr>, Box<Expr>),
    And(Vec<Expr>),
    Select(Box<Expr>, Box<Expr>, Box<Expr>),
    ArrayGet(Box<Expr>, Box<Expr>),
    TupleGet(Box<Expr>, Box<Expr>),
    Not(Box<Expr>),
    ReadGlobal(u64),
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

    pub fn add(&self, other: &Self) -> Self {
        let mut adds = self.get_adds();
        adds.extend(other.get_adds());
        adds.sort();
        Self::Add(adds)
    }

    pub fn mul(&self, other: &Self) -> Self {
        let mut muls = self.get_muls();
        muls.extend(other.get_muls());
        muls.sort();
        Self::Mul(muls)
    }

    pub fn div(&self, other: &Self) -> Self {
        Self::Div(Box::new(self.clone()), Box::new(other.clone()))
    }

    pub fn sub(&self, other: &Self) -> Self {
        Self::Sub(Box::new(self.clone()), Box::new(other.clone()))
    }

    pub fn and(&self, other: &Self) -> Self {
        let mut ands = self.get_ands();
        ands.extend(other.get_ands());
        ands.sort();
        Self::And(ands)
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
            Self::Sub(lhs, rhs) => write!(f, "({} - {})", lhs, rhs),
            Self::FConst(value) => write!(f, "{}", value),
            Self::UConst(size, value) => write!(f, "u{}({})", size, value),
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
            Self::Select(cond, then, otherwise) => {
                write!(f, "({} ? {} : {})", cond, then, otherwise)
            }
            Self::ArrayGet(array, index) => write!(f, "{}[{}]", array, index),
            Self::TupleGet(tuple, index) => write!(f, "{}.{}", tuple, index),
            Self::Not(value) => write!(f, "(~{})", value),
            Self::ReadGlobal(index) => write!(f, "g{}", index),
        }
    }
}

impl Debug for Expr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self)
    }
}

pub struct CSE {}
 
impl<V: Clone> Pass<V> for CSE {
    fn run(&self, ssa: &mut SSA<V>, pass_manager: &crate::compiler::pass_manager::PassManager<V>) {
        self.do_run(ssa, pass_manager.get_cfg());
    }

    fn pass_info(&self) -> crate::compiler::pass_manager::PassInfo {
        crate::compiler::pass_manager::PassInfo {
            name: "cse",
            needs: vec![crate::compiler::pass_manager::DataPoint::CFG],
        }
    }

    fn invalidates_cfg(&self) -> bool {
        false
    }
}

impl CSE {
    pub fn new() -> Self {
        Self {}
    }

    pub fn do_run<V: Clone>(&self, ssa: &mut SSA<V>, cfg: &FlowAnalysis) {
        for (function_id, function) in ssa.iter_functions_mut() {
            let cfg = cfg.get_function_cfg(*function_id);
            let exprs = self.gather_expressions(function, cfg);
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

            for (_, block) in function.get_blocks_mut() {
                for instruction in block.get_instructions_mut() {
                    value_replacements.replace_inputs(instruction);
                }
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

    fn gather_expressions<V: Clone>(
        &self,
        ssa: &Function<V>,
        cfg: &CFG,
    ) -> HashMap<Expr, Vec<(BlockId, usize, ValueId)>> {
        let mut result: HashMap<Expr, Vec<(BlockId, usize, ValueId)>> = HashMap::new();
        let mut exprs = HashMap::<ValueId, Expr>::new();

        for (value_id, const_) in ssa.iter_consts() {
            match const_ {
                Const::U(size, value) => exprs.insert(*value_id, Expr::UConst(*size, *value)),
                Const::Field(value) => exprs.insert(*value_id, Expr::fconst(*value)),
                Const::WitnessRef(_) => todo!(),
            };
        }

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
                    OpCode::WriteWitness { .. } // TODO: is witness store a subexpression to be optimized?
                    | OpCode::FreshWitness { result: _, result_type: _ }
                    | OpCode::Constrain { .. }
                    | OpCode::NextDCoeff { result: _ }
                    | OpCode::BumpD { matrix: _, variable: _, sensitivity: _ }
                    | OpCode::Alloc { .. }
                    | OpCode::Store { .. }
                    | OpCode::Load { .. }
                    | OpCode::AssertEq { .. }
                    | OpCode::AssertR1C { .. }
                    | OpCode::Call { .. }
                    | OpCode::MkSeq { .. }
                    | OpCode::MkTuple { .. }
                    | OpCode::ArraySet { .. }
                    | OpCode::SlicePush { .. }
                    | OpCode::SliceLen { .. }
                    | OpCode::Cast { .. }
                    | OpCode::Truncate { .. }
                    | OpCode::MemOp { kind: _, value: _ }
                    | OpCode::Rangecheck { value: _, max_bits: _ }
                    | OpCode::ToBits { .. }
                    | OpCode::ToRadix { .. }
                    | OpCode::Lookup { target: _, keys: _, results: _ }
                    | OpCode::DLookup { target: _, keys: _, results: _ }
                    | OpCode::Todo { .. } => {}
                     OpCode::PureToWitnessRef { result: _, value: _, result_annotation: _ }
                    | OpCode::UnboxField { result: _, value: _ }
                    | OpCode::MulConst { result: _, const_val: _, var: _ } => { todo!() }
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
                        let index_expr = match idx {
                            TupleIdx::Static(i) => Expr::UConst(64, *i as u128),
                            TupleIdx::Dynamic(idx_value, _) => get_expr(&exprs, idx_value),
                        };
                        let result_expr = tuple_expr.tuple_get(&index_expr);
                        exprs.insert(*r, result_expr.clone());
                        result.entry(result_expr).or_default().push((
                            block_id,
                            instruction_idx,
                            *r,
                        ));
                    },
                }
            }
        }
        result
    }
}
