//! Deduplicates expressions when one occurrence dominates the other.
//!
//! Does not float expressions across branches or otherwise move them outside the block in which
//! they appear (#172).

use crate::{
    collections::{HashMap, HashSet},
    compiler::{
        analysis::flow_analysis::{CFG, FlowAnalysis},
        pass_manager::{AnalysisId, AnalysisStore, Pass},
        passes::fix_double_jumps::ValueReplacements,
        ssa::{
            BlockId, SSAConstantsSnapshot, ValueId,
            hlssa::{
                BinaryArithOpKind, CmpKind, Constant, Endianness, HLFunction, HLSSA,
                LookupTarget, OpCode, Radix, Type,
            },
        },
        util::ice_non_elided_tuple,
    },
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
struct ExprId(u32);

#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
enum ExprNode {
    Add(Vec<ExprId>),
    Mul(Vec<ExprId>),
    Div { lhs: ExprId, rhs: ExprId },
    Mod { lhs: ExprId, rhs: ExprId },
    Sub { lhs: ExprId, rhs: ExprId },
    FConst(ark_bn254::Fr),
    UConst { bits: usize, value: u128 },
    IConst { bits: usize, value: u128 },
    Variable(u64),
    Eq { lhs: ExprId, rhs: ExprId },
    Lt { lhs: ExprId, rhs: ExprId },
    And(Vec<ExprId>),
    Or(Vec<ExprId>),
    Xor(Vec<ExprId>),
    Shl { lhs: ExprId, rhs: ExprId },
    Shr { lhs: ExprId, rhs: ExprId },
    BitRange { value: ExprId, offset: usize, width: usize },
    Select { condition: ExprId, then: ExprId, otherwise: ExprId },
    ArrayGet { array: ExprId, index: ExprId },
    Not(ExprId),
    ReadGlobal(u64),
    Cast { value: ExprId, target: Type },
    SExt { value: ExprId, from_bits: usize, to_bits: usize },
    ValueOf(ExprId),
    BytesOf { value: ExprId, endianness: Endianness, count: usize },
    BitsOf { value: ExprId, endianness: Endianness, count: usize },
    Witness(ExprId),
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
enum Assertion {
    Rangecheck { value: ExprId, max_bits: usize },
    Lookup { target: LookupAssertionTarget, args: Vec<ExprId>, flag: ExprId },
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
enum LookupAssertionTarget {
    Rangecheck(u8),
    DynRangecheck(ExprId),
    Array(ExprId),
    Spread(u8),
}

#[derive(Clone, Copy, Debug)]
pub struct Config {
    deduplicate_lookups: bool,
}

impl Config {
    pub fn pre_r1c() -> Self {
        Self {
            deduplicate_lookups: true,
        }
    }

    pub fn post_r1c() -> Self {
        Self {
            deduplicate_lookups: false,
        }
    }
}

#[derive(Default)]
struct ExprInterner {
    nodes: Vec<ExprNode>,
    ids: HashMap<ExprNode, ExprId>,
}

impl ExprInterner {
    fn intern(&mut self, node: ExprNode) -> ExprId {
        if let Some(id) = self.ids.get(&node) {
            return *id;
        }

        let id = ExprId(self.nodes.len() as u32);
        self.nodes.push(node.clone());
        self.ids.insert(node, id);
        id
    }

    fn node(&self, id: ExprId) -> &ExprNode {
        &self.nodes[id.0 as usize]
    }

    fn variable(&mut self, value_id: ValueId) -> ExprId {
        self.intern(ExprNode::Variable(value_id.0))
    }

    fn fconst(&mut self, value: ark_bn254::Fr) -> ExprId {
        self.intern(ExprNode::FConst(value))
    }

    fn uconst(&mut self, bits: usize, value: u128) -> ExprId {
        self.intern(ExprNode::UConst { bits, value })
    }

    fn iconst(&mut self, bits: usize, value: u128) -> ExprId {
        self.intern(ExprNode::IConst { bits, value })
    }

    fn extend_adds(&self, expr: ExprId, out: &mut Vec<ExprId>) {
        match self.node(expr) {
            ExprNode::Add(exprs) => out.extend(exprs.iter().copied()),
            _ => out.push(expr),
        }
    }

    fn extend_muls(&self, expr: ExprId, out: &mut Vec<ExprId>) {
        match self.node(expr) {
            ExprNode::Mul(exprs) => out.extend(exprs.iter().copied()),
            _ => out.push(expr),
        }
    }

    fn extend_ands(&self, expr: ExprId, out: &mut Vec<ExprId>) {
        match self.node(expr) {
            ExprNode::And(exprs) => out.extend(exprs.iter().copied()),
            _ => out.push(expr),
        }
    }

    fn extend_ors(&self, expr: ExprId, out: &mut Vec<ExprId>) {
        match self.node(expr) {
            ExprNode::Or(exprs) => out.extend(exprs.iter().copied()),
            _ => out.push(expr),
        }
    }

    fn extend_xors(&self, expr: ExprId, out: &mut Vec<ExprId>) {
        match self.node(expr) {
            ExprNode::Xor(exprs) => out.extend(exprs.iter().copied()),
            _ => out.push(expr),
        }
    }

    fn add(&mut self, lhs: ExprId, rhs: ExprId) -> ExprId {
        let mut adds = Vec::new();
        self.extend_adds(lhs, &mut adds);
        self.extend_adds(rhs, &mut adds);
        adds.sort();
        self.intern(ExprNode::Add(adds))
    }

    fn mul(&mut self, lhs: ExprId, rhs: ExprId) -> ExprId {
        let mut muls = Vec::new();
        self.extend_muls(lhs, &mut muls);
        self.extend_muls(rhs, &mut muls);
        muls.sort();
        self.intern(ExprNode::Mul(muls))
    }

    fn div(&mut self, lhs: ExprId, rhs: ExprId) -> ExprId {
        self.intern(ExprNode::Div { lhs, rhs })
    }

    fn modulo(&mut self, lhs: ExprId, rhs: ExprId) -> ExprId {
        self.intern(ExprNode::Mod { lhs, rhs })
    }

    fn sub(&mut self, lhs: ExprId, rhs: ExprId) -> ExprId {
        self.intern(ExprNode::Sub { lhs, rhs })
    }

    fn and(&mut self, lhs: ExprId, rhs: ExprId) -> ExprId {
        let mut ands = Vec::new();
        self.extend_ands(lhs, &mut ands);
        self.extend_ands(rhs, &mut ands);
        ands.sort();
        ands.dedup();
        self.intern(ExprNode::And(ands))
    }

    fn or(&mut self, lhs: ExprId, rhs: ExprId) -> ExprId {
        let mut ors = Vec::new();
        self.extend_ors(lhs, &mut ors);
        self.extend_ors(rhs, &mut ors);
        ors.sort();
        ors.dedup();
        self.intern(ExprNode::Or(ors))
    }

    fn xor(&mut self, lhs: ExprId, rhs: ExprId) -> ExprId {
        let mut xors = Vec::new();
        self.extend_xors(lhs, &mut xors);
        self.extend_xors(rhs, &mut xors);
        xors.sort();
        self.intern(ExprNode::Xor(xors))
    }

    fn shl(&mut self, lhs: ExprId, rhs: ExprId) -> ExprId {
        self.intern(ExprNode::Shl { lhs, rhs })
    }

    fn shr(&mut self, lhs: ExprId, rhs: ExprId) -> ExprId {
        self.intern(ExprNode::Shr { lhs, rhs })
    }

    fn bit_range(&mut self, value: ExprId, offset: usize, width: usize) -> ExprId {
        self.intern(ExprNode::BitRange {
            value,
            offset,
            width,
        })
    }

    fn eq(&mut self, lhs: ExprId, rhs: ExprId) -> ExprId {
        self.intern(ExprNode::Eq { lhs, rhs })
    }

    fn lt(&mut self, lhs: ExprId, rhs: ExprId) -> ExprId {
        self.intern(ExprNode::Lt { lhs, rhs })
    }

    fn array_get(&mut self, array: ExprId, index: ExprId) -> ExprId {
        self.intern(ExprNode::ArrayGet { array, index })
    }

    fn select(&mut self, condition: ExprId, then: ExprId, otherwise: ExprId) -> ExprId {
        self.intern(ExprNode::Select {
            condition,
            then,
            otherwise,
        })
    }

    fn not(&mut self, value: ExprId) -> ExprId {
        self.intern(ExprNode::Not(value))
    }

    fn read_global(&mut self, index: u64) -> ExprId {
        self.intern(ExprNode::ReadGlobal(index))
    }

    fn cast(&mut self, value: ExprId, target: Type) -> ExprId {
        self.intern(ExprNode::Cast { value, target })
    }

    fn sext(&mut self, value: ExprId, from_bits: usize, to_bits: usize) -> ExprId {
        self.intern(ExprNode::SExt {
            value,
            from_bits,
            to_bits,
        })
    }

    fn value_of(&mut self, value: ExprId) -> ExprId {
        self.intern(ExprNode::ValueOf(value))
    }

    fn bytes_of(&mut self, value: ExprId, endianness: Endianness, count: usize) -> ExprId {
        self.intern(ExprNode::BytesOf {
            value,
            endianness,
            count,
        })
    }

    fn bits_of(&mut self, value: ExprId, endianness: Endianness, count: usize) -> ExprId {
        self.intern(ExprNode::BitsOf {
            value,
            endianness,
            count,
        })
    }

    fn witness(&mut self, value: ExprId) -> ExprId {
        self.intern(ExprNode::Witness(value))
    }
}
pub struct CSE {
    config: Config,
}

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
    pub fn with_config(config: Config) -> Self {
        Self { config }
    }

    pub fn pre_r1c() -> Self {
        Self::with_config(Config::pre_r1c())
    }

    pub fn post_r1c() -> Self {
        Self::with_config(Config::post_r1c())
    }

    pub fn do_run(&self, ssa: &mut HLSSA, cfg: &FlowAnalysis) {
        let constants = ssa.const_snapshot();

        for (function_id, function) in ssa.iter_functions_mut() {
            let cfg = cfg.get_function_cfg(*function_id);
            let (exprs, assertions) = self.gather_expressions(function, cfg, &constants);
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
            let mut to_remove: HashSet<(BlockId, usize)> = HashSet::default();
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
        constants: &SSAConstantsSnapshot<Constant>,
    ) -> (
        HashMap<ExprId, Vec<(BlockId, usize, ValueId)>>,
        HashMap<Assertion, Vec<(BlockId, usize)>>,
    ) {
        let mut interner = ExprInterner::default();
        let mut result: HashMap<ExprId, Vec<(BlockId, usize, ValueId)>> = HashMap::default();
        let mut assertions: HashMap<Assertion, Vec<(BlockId, usize)>> = HashMap::default();

        // Seed the value->expr map with the SSA's constants so they can be referenced as operands.
        // They are not recorded into `result`: the constant store already dedups them, so CSE must
        // not try to dedup the constants themselves.
        let mut exprs: HashMap<ValueId, ExprId> = HashMap::default();
        for (vid, cv) in constants {
            let id = match cv.as_ref() {
                Constant::U(bits, value) => interner.uconst(*bits, *value),
                Constant::I(bits, value) => interner.iconst(*bits, *value),
                Constant::Field(value) => interner.fconst(*value),
                Constant::FnPtr(_) | Constant::Blob(_) => continue,
            };
            exprs.insert(*vid, id);
        }

        fn get_expr(
            exprs: &HashMap<ValueId, ExprId>,
            interner: &mut ExprInterner,
            value_id: &ValueId,
        ) -> ExprId {
            exprs
                .get(value_id)
                .copied()
                .unwrap_or_else(|| interner.variable(*value_id))
        }

        fn record_expr(
            exprs: &mut HashMap<ValueId, ExprId>,
            result: &mut HashMap<ExprId, Vec<(BlockId, usize, ValueId)>>,
            block_id: BlockId,
            instruction_idx: usize,
            value_id: ValueId,
            expr: ExprId,
        ) {
            exprs.insert(value_id, expr);
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
            assertions
                .entry(assertion)
                .or_default()
                .push((block_id, instruction_idx));
        }

        fn lookup_target_expr(
            target: &LookupTarget<ValueId>,
            exprs: &HashMap<ValueId, ExprId>,
            interner: &mut ExprInterner,
        ) -> LookupAssertionTarget {
            match target {
                LookupTarget::Rangecheck(bits) => LookupAssertionTarget::Rangecheck(*bits),
                LookupTarget::DynRangecheck(bound) => {
                    LookupAssertionTarget::DynRangecheck(get_expr(exprs, interner, bound))
                }
                LookupTarget::Array(array) => {
                    LookupAssertionTarget::Array(get_expr(exprs, interner, array))
                }
                LookupTarget::Spread(bits) => LookupAssertionTarget::Spread(*bits),
            }
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
                        let lhs_expr = get_expr(&exprs, &mut interner, lhs);
                        let rhs_expr = get_expr(&exprs, &mut interner, rhs);
                        let result_expr = interner.add(lhs_expr, rhs_expr);
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
                        let lhs_expr = get_expr(&exprs, &mut interner, lhs);
                        let rhs_expr = get_expr(&exprs, &mut interner, rhs);
                        let result_expr = interner.mul(lhs_expr, rhs_expr);
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
                        let lhs_expr = get_expr(&exprs, &mut interner, lhs);
                        let rhs_expr = get_expr(&exprs, &mut interner, rhs);
                        let result_expr = interner.div(lhs_expr, rhs_expr);
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
                        let lhs_expr = get_expr(&exprs, &mut interner, lhs);
                        let rhs_expr = get_expr(&exprs, &mut interner, rhs);
                        let result_expr = interner.sub(lhs_expr, rhs_expr);
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
                        let lhs_expr = get_expr(&exprs, &mut interner, lhs);
                        let rhs_expr = get_expr(&exprs, &mut interner, rhs);
                        let result_expr = interner.eq(lhs_expr, rhs_expr);
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
                        let lhs_expr = get_expr(&exprs, &mut interner, lhs);
                        let rhs_expr = get_expr(&exprs, &mut interner, rhs);
                        let result_expr = interner.lt(lhs_expr, rhs_expr);
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
                        let lhs_expr = get_expr(&exprs, &mut interner, lhs);
                        let rhs_expr = get_expr(&exprs, &mut interner, rhs);
                        let result_expr = interner.modulo(lhs_expr, rhs_expr);
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
                        let lhs_expr = get_expr(&exprs, &mut interner, lhs);
                        let rhs_expr = get_expr(&exprs, &mut interner, rhs);
                        let result_expr = interner.and(lhs_expr, rhs_expr);
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
                        let lhs_expr = get_expr(&exprs, &mut interner, lhs);
                        let rhs_expr = get_expr(&exprs, &mut interner, rhs);
                        let result_expr = interner.or(lhs_expr, rhs_expr);
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
                        let lhs_expr = get_expr(&exprs, &mut interner, lhs);
                        let rhs_expr = get_expr(&exprs, &mut interner, rhs);
                        let result_expr = interner.xor(lhs_expr, rhs_expr);
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
                        let lhs_expr = get_expr(&exprs, &mut interner, lhs);
                        let rhs_expr = get_expr(&exprs, &mut interner, rhs);
                        let result_expr = interner.shl(lhs_expr, rhs_expr);
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
                        let lhs_expr = get_expr(&exprs, &mut interner, lhs);
                        let rhs_expr = get_expr(&exprs, &mut interner, rhs);
                        let result_expr = interner.shr(lhs_expr, rhs_expr);
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
                        let array_expr = get_expr(&exprs, &mut interner, array);
                        let index_expr = get_expr(&exprs, &mut interner, index);
                        let result_expr = interner.array_get(array_expr, index_expr);
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
                        let cond_expr = get_expr(&exprs, &mut interner, cond);
                        let then_expr = get_expr(&exprs, &mut interner, then);
                        let otherwise_expr = get_expr(&exprs, &mut interner, otherwise);
                        let result_expr = interner.select(cond_expr, then_expr, otherwise_expr);
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
                        let result_expr = interner.read_global(*index);
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
                        let value_expr = get_expr(&exprs, &mut interner, value);
                        let result_expr = interner.cast(value_expr, target.clone());
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
                        let value_expr = get_expr(&exprs, &mut interner, value);
                        let result_expr = interner.sext(value_expr, *from_bits, *to_bits);
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
                    } => {
                        let value_expr = get_expr(&exprs, &mut interner, value);
                        let result_expr = interner.bit_range(value_expr, *offset, *width);
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
                        let value_expr = get_expr(&exprs, &mut interner, value);
                        let result_expr = interner.value_of(value_expr);
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
                        let lhs_expr = get_expr(&exprs, &mut interner, const_val);
                        let rhs_expr = get_expr(&exprs, &mut interner, var);
                        let result_expr = interner.mul(lhs_expr, rhs_expr);
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
                        let value_expr = get_expr(&exprs, &mut interner, value);
                        let result_expr = interner.bits_of(value_expr, *endianness, *count);
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
                                let value_expr = get_expr(&exprs, &mut interner, value);
                                let result_expr =
                                    interner.bytes_of(value_expr, *endianness, *count);
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
                        let hint_expr = get_expr(&exprs, &mut interner, value);
                        let result_expr = interner.witness(hint_expr);
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
                        let value_expr = get_expr(&exprs, &mut interner, value);
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
                    OpCode::Lookup { target, args, flag } if self.config.deduplicate_lookups => {
                        let target_expr = lookup_target_expr(target, &exprs, &mut interner);
                        let arg_exprs = args
                            .iter()
                            .map(|arg| get_expr(&exprs, &mut interner, arg))
                            .collect();
                        let flag_expr = get_expr(&exprs, &mut interner, flag);
                        record_assertion(
                            &mut assertions,
                            block_id,
                            instruction_idx,
                            Assertion::Lookup {
                                target: target_expr,
                                args: arg_exprs,
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
                    | OpCode::MkSeqOfBlob { .. }
                    | OpCode::MkRepeated { .. }
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
                        let value_expr = get_expr(&exprs, &mut interner, value);
                        let result_expr = interner.not(value_expr);
                        record_expr(
                            &mut exprs,
                            &mut result,
                            block_id,
                            instruction_idx,
                            *r,
                            result_expr,
                        );
                    }
                    OpCode::TupleProj { .. }
                    | OpCode::TupleRefProj { .. }
                    | OpCode::MkTuple { .. } => ice_non_elided_tuple(),
                    OpCode::Guard { .. } => {
                        // Guards are opaque to CSE
                    }
                }
            }
        }
        (result, assertions)
    }
}
