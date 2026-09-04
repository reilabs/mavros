//! Deduplicates expressions when one occurrence dominates the other, without floating expressions
//! across branches or moving them outside the block in which they appear.
//!
//! This is a very simple CSE in comparison to PRE, and does elimination only to ensure that the
//! witness shape remains frozen between the witness generator and AD programs.

use mavros_int_semantics::IntBits;

use crate::{
    collections::{HashMap, HashSet},
    compiler::{
        Field,
        analysis::flow_analysis::{CFG, FlowAnalysis},
        pass_manager::{AnalysisId, AnalysisStore, Pass},
        passes::shared::{availability::can_replace, value_replacements::ValueReplacements},
        ssa::{
            BlockId, ProgramPoint, SSAConstantsSnapshot, ValueId,
            hlssa::{
                BinaryArithOpKind, CastTarget, CmpKind, Constant, Endianness, HLFunction, HLSSA,
                OpCode, Radix,
            },
        },
        util::{ice_non_elided_tuple, ice_unvalidated_assert_constant},
    },
};

// COMMON SUBEXPRESSION ELIMINATION
// ================================================================================================

/// A basic, deduplicating CSE that only deals with the elimination of redundant expressions.
///
/// More aggressive code motion is a non-goal, as that is instead handled by PRE, which owns all
/// pre-R1C deduplication; this pass remains wired only at the post-R1C sites.
pub struct CSE;

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
    /// The pass as wired at the post-R1C pipeline sites (its only remaining placement).
    pub fn post_r1c() -> Self {
        Self
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
                // Occurrences arrive in domination-preorder walk order, so an existing group
                // leader is never dominated by a later occurrence — the forward availability
                // query is the only direction that can match.
                let mut replacement_groups: Vec<((ProgramPoint, ValueId), Vec<ValueId>)> = vec![];
                for (point, value_id) in occurrences {
                    let mut found = false;
                    for ((candidate_point, _), others) in replacement_groups.iter_mut() {
                        if can_replace(cfg, *candidate_point, point) {
                            found = true;
                            others.push(value_id);
                            break;
                        }
                    }
                    if !found {
                        replacement_groups.push(((point, value_id), vec![]));
                    }
                }
                for ((_, value_id), others) in replacement_groups {
                    for other in others {
                        value_replacements.insert(other, value_id);
                    }
                }
            }

            // Side-effect dedup: same dominance grouping as the value loop,
            // but duplicates are dropped rather than redirected.
            let mut to_remove: HashSet<ProgramPoint> = HashSet::default();
            for (_, occurrences) in assertions {
                if occurrences.len() <= 1 {
                    continue;
                }
                let mut groups: Vec<(ProgramPoint, Vec<ProgramPoint>)> = vec![];
                for point in occurrences {
                    let mut found = false;
                    for (candidate_point, others) in groups.iter_mut() {
                        if can_replace(cfg, *candidate_point, point) {
                            found = true;
                            others.push(point);
                            break;
                        }
                    }
                    if !found {
                        groups.push((point, vec![]));
                    }
                }
                for (_, others) in groups {
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
                    if to_remove.contains(&ProgramPoint::new(bid, idx)) {
                        continue;
                    }
                    value_replacements.replace_inputs(&mut *instruction);
                    new_instructions.push(instruction);
                }
                block.put_instructions(new_instructions);
                value_replacements.replace_terminator(block.get_terminator_mut());
            }
        }
    }

    fn gather_expressions(
        &self,
        ssa: &HLFunction,
        cfg: &CFG,
        constants: &SSAConstantsSnapshot<Constant>,
    ) -> (
        HashMap<ExprId, Vec<(ProgramPoint, ValueId)>>,
        HashMap<Assertion, Vec<ProgramPoint>>,
    ) {
        let mut interner = ExprInterner::default();
        let mut result: HashMap<ExprId, Vec<(ProgramPoint, ValueId)>> = HashMap::default();
        let mut assertions: HashMap<Assertion, Vec<ProgramPoint>> = HashMap::default();

        // Seed the value->expr map with the SSA's constants so they can be referenced as operands.
        // They are not recorded into `result`: the constant store already dedups them, so CSE must
        // not try to dedup the constants themselves.
        let mut exprs: HashMap<ValueId, ExprId> = HashMap::default();
        for (vid, cv) in constants {
            let id = match cv.as_ref() {
                Constant::Int(v) => interner.int_const(v.clone()),
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
            result: &mut HashMap<ExprId, Vec<(ProgramPoint, ValueId)>>,
            block_id: BlockId,
            instruction_idx: usize,
            value_id: ValueId,
            expr: ExprId,
        ) {
            exprs.insert(value_id, expr);
            result
                .entry(expr)
                .or_default()
                .push((ProgramPoint::new(block_id, instruction_idx), value_id));
        }

        fn record_assertion(
            assertions: &mut HashMap<Assertion, Vec<ProgramPoint>>,
            block_id: BlockId,
            instruction_idx: usize,
            assertion: Assertion,
        ) {
            assertions
                .entry(assertion)
                .or_default()
                .push(ProgramPoint::new(block_id, instruction_idx));
        }

        // The sign participates in every key below. `UDiv(x, y)` and `SDiv(x, y)` are two different
        // computations over the same operands; `UAdd(x, y)` and `SAdd(x, y)` agree on every bit but
        // owe different rejections, and a merge picks one of them to emit. Both are wrong answers,
        // so no arm is exempt.
        //
        // Each arm still matches an operation's two forms together and then carries the sign into
        // the interner node. The commutative pair take it as part of the flattened chain, which is
        // also what stops `extend_adds`/`extend_muls` splicing a signed chain into an unsigned one.
        // Only the genuinely sign-free operations pass nothing: `And`/`Or`/`Xor` have one form, and
        // `Eq` compares patterns under either reading.
        for block_id in cfg.get_domination_pre_order() {
            let block = ssa.get_block(block_id);

            for (instruction_idx, instruction) in block.get_instructions().enumerate() {
                match instruction {
                    OpCode::BinaryArithOp {
                        kind: kind @ (BinaryArithOpKind::UAdd | BinaryArithOpKind::SAdd),
                        result: r,
                        lhs,
                        rhs,
                    } => {
                        let lhs_expr = get_expr(&exprs, &mut interner, lhs);
                        let rhs_expr = get_expr(&exprs, &mut interner, rhs);
                        let result_expr = interner.add(lhs_expr, rhs_expr, kind.is_signed());
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
                        kind: kind @ (BinaryArithOpKind::UMul | BinaryArithOpKind::SMul),
                        result: r,
                        lhs,
                        rhs,
                    } => {
                        let lhs_expr = get_expr(&exprs, &mut interner, lhs);
                        let rhs_expr = get_expr(&exprs, &mut interner, rhs);
                        let result_expr = interner.mul(lhs_expr, rhs_expr, kind.is_signed());
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
                        kind: kind @ (BinaryArithOpKind::UDiv | BinaryArithOpKind::SDiv),
                        result: r,
                        lhs,
                        rhs,
                    } => {
                        let lhs_expr = get_expr(&exprs, &mut interner, lhs);
                        let rhs_expr = get_expr(&exprs, &mut interner, rhs);
                        let sign = kind.signedness();
                        let result_expr = interner.div(lhs_expr, rhs_expr, sign);
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
                        kind: kind @ (BinaryArithOpKind::USub | BinaryArithOpKind::SSub),
                        result: r,
                        lhs,
                        rhs,
                    } => {
                        let lhs_expr = get_expr(&exprs, &mut interner, lhs);
                        let rhs_expr = get_expr(&exprs, &mut interner, rhs);
                        let sign = kind.signedness();
                        let result_expr = interner.sub(lhs_expr, rhs_expr, sign);
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
                        kind: kind @ (CmpKind::ULt | CmpKind::SLt),
                        result: r,
                        lhs,
                        rhs,
                    } => {
                        let lhs_expr = get_expr(&exprs, &mut interner, lhs);
                        let rhs_expr = get_expr(&exprs, &mut interner, rhs);
                        let result_expr = interner.lt(lhs_expr, rhs_expr, kind.is_signed());
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
                        kind: kind @ (BinaryArithOpKind::URem | BinaryArithOpKind::SRem),
                        result: r,
                        lhs,
                        rhs,
                    } => {
                        let lhs_expr = get_expr(&exprs, &mut interner, lhs);
                        let rhs_expr = get_expr(&exprs, &mut interner, rhs);
                        let sign = kind.signedness();
                        let result_expr = interner.modulo(lhs_expr, rhs_expr, sign);
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
                        kind: kind @ (BinaryArithOpKind::UShl | BinaryArithOpKind::SShl),
                        result: r,
                        lhs,
                        rhs,
                    } => {
                        let lhs_expr = get_expr(&exprs, &mut interner, lhs);
                        let rhs_expr = get_expr(&exprs, &mut interner, rhs);
                        let sign = kind.signedness();
                        let result_expr = interner.shl(lhs_expr, rhs_expr, sign);
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
                        kind: kind @ (BinaryArithOpKind::UShr | BinaryArithOpKind::SShr),
                        result: r,
                        lhs,
                        rhs,
                    } => {
                        let lhs_expr = get_expr(&exprs, &mut interner, lhs);
                        let rhs_expr = get_expr(&exprs, &mut interner, rhs);
                        let sign = kind.signedness();
                        let result_expr = interner.shr(lhs_expr, rhs_expr, sign);
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
                    OpCode::MulConst {
                        result: r,
                        const_val,
                        var,
                    } => {
                        // Fold into an unsigned Expr::Mul so MulConst dedups with `UMul`.
                        // Unsigned is not a default: `MulConst` is a field multiply, which
                        // `witness_lowering` only ever builds from `UMul` or field arithmetic.
                        let lhs_expr = get_expr(&exprs, &mut interner, const_val);
                        let rhs_expr = get_expr(&exprs, &mut interner, var);
                        let result_expr = interner.mul(lhs_expr, rhs_expr, false);
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
                    // `Lookup`s are never deduplicated here: that dedup is pre-R1C-only and now
                    // lives in the PRE pass.
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
                    | OpCode::SlicePop { .. }
                    | OpCode::SliceInsert { .. }
                    | OpCode::SliceRemove { .. }
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
                    OpCode::AssertConstant { .. } => ice_unvalidated_assert_constant(),
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

// EXPRESSION KEYING
// ================================================================================================

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
struct ExprId(u32);

// No `Ord` as this is an interning key, only ever hashed and compared for equality.
//
// The commutative chains sort their `ExprId`s, but never the nodes. Not having it matters because
// `IntConst`'s payload deliberately has no ordering as it is an uninterpreted bit pattern with no
// particular reading.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum ExprNode {
    /// The flattened commutative chains, carrying the sign of the operation that built them.
    ///
    /// The sign is part of the key, so a signed chain never merges with an unsigned one, and
    /// `extend_adds`/`extend_muls` will not splice the parts of one into the other.
    Add(Vec<ExprId>, bool),
    Mul(Vec<ExprId>, bool),

    /// The non-commutative binary operations.
    ///
    /// `sign` is the reading the operation applies to its operands, taken from
    /// `BinaryArithOpKind::signedness`, so a signed and an unsigned form never share a node. It is
    /// `None` only for operations that have no signed form at all — none of which appear here,
    /// since the bitwise trio are chains.
    Div {
        lhs: ExprId,
        rhs: ExprId,
        sign: Option<bool>,
    },
    Mod {
        lhs: ExprId,
        rhs: ExprId,
        sign: Option<bool>,
    },
    Sub {
        lhs: ExprId,
        rhs: ExprId,
        sign: Option<bool>,
    },
    FConst(Field),
    IntConst(IntBits),
    Variable(u64),
    Eq {
        lhs: ExprId,
        rhs: ExprId,
    },
    Lt {
        lhs: ExprId,
        rhs: ExprId,
        signed: bool,
    },
    And(Vec<ExprId>),
    Or(Vec<ExprId>),
    Xor(Vec<ExprId>),
    Shl {
        lhs: ExprId,
        rhs: ExprId,
        sign: Option<bool>,
    },
    Shr {
        lhs: ExprId,
        rhs: ExprId,
        sign: Option<bool>,
    },
    BitRange {
        value: ExprId,
        offset: usize,
        width: usize,
    },
    Select {
        condition: ExprId,
        then: ExprId,
        otherwise: ExprId,
    },
    ArrayGet {
        array: ExprId,
        index: ExprId,
    },
    Not(ExprId),
    ReadGlobal(u64),
    Cast {
        value: ExprId,
        target: CastTarget,
    },
    SExt {
        value: ExprId,
        from_bits: usize,
        to_bits: usize,
    },
    BytesOf {
        value: ExprId,
        endianness: Endianness,
        count: usize,
    },
    BitsOf {
        value: ExprId,
        endianness: Endianness,
        count: usize,
    },
    Witness(ExprId),
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
enum Assertion {
    Rangecheck { value: ExprId, max_bits: usize },
}

// EXPRESSION INTERNING
// ================================================================================================

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

    fn fconst(&mut self, value: Field) -> ExprId {
        self.intern(ExprNode::FConst(value))
    }

    fn int_const(&mut self, pattern: IntBits) -> ExprId {
        self.intern(ExprNode::IntConst(pattern))
    }

    fn extend_adds(&self, expr: ExprId, signed: bool, out: &mut Vec<ExprId>) {
        match self.node(expr) {
            ExprNode::Add(exprs, s) if *s == signed => out.extend(exprs.iter().copied()),
            _ => out.push(expr),
        }
    }

    fn extend_muls(&self, expr: ExprId, signed: bool, out: &mut Vec<ExprId>) {
        match self.node(expr) {
            ExprNode::Mul(exprs, s) if *s == signed => out.extend(exprs.iter().copied()),
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

    fn add(&mut self, lhs: ExprId, rhs: ExprId, signed: bool) -> ExprId {
        let mut adds = Vec::new();
        self.extend_adds(lhs, signed, &mut adds);
        self.extend_adds(rhs, signed, &mut adds);
        adds.sort();
        self.intern(ExprNode::Add(adds, signed))
    }

    fn mul(&mut self, lhs: ExprId, rhs: ExprId, signed: bool) -> ExprId {
        let mut muls = Vec::new();
        self.extend_muls(lhs, signed, &mut muls);
        self.extend_muls(rhs, signed, &mut muls);
        muls.sort();
        self.intern(ExprNode::Mul(muls, signed))
    }

    fn div(&mut self, lhs: ExprId, rhs: ExprId, sign: Option<bool>) -> ExprId {
        self.intern(ExprNode::Div { lhs, rhs, sign })
    }

    fn modulo(&mut self, lhs: ExprId, rhs: ExprId, sign: Option<bool>) -> ExprId {
        self.intern(ExprNode::Mod { lhs, rhs, sign })
    }

    fn sub(&mut self, lhs: ExprId, rhs: ExprId, sign: Option<bool>) -> ExprId {
        self.intern(ExprNode::Sub { lhs, rhs, sign })
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

    fn shl(&mut self, lhs: ExprId, rhs: ExprId, sign: Option<bool>) -> ExprId {
        self.intern(ExprNode::Shl { lhs, rhs, sign })
    }

    fn shr(&mut self, lhs: ExprId, rhs: ExprId, sign: Option<bool>) -> ExprId {
        self.intern(ExprNode::Shr { lhs, rhs, sign })
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

    fn lt(&mut self, lhs: ExprId, rhs: ExprId, signed: bool) -> ExprId {
        self.intern(ExprNode::Lt { lhs, rhs, signed })
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

    fn cast(&mut self, value: ExprId, target: CastTarget) -> ExprId {
        self.intern(ExprNode::Cast { value, target })
    }

    fn sext(&mut self, value: ExprId, from_bits: usize, to_bits: usize) -> ExprId {
        self.intern(ExprNode::SExt {
            value,
            from_bits,
            to_bits,
        })
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

// TESTS
// ================================================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::ssa::{Terminator, hlssa::Type};

    #[test]
    fn an_interned_constant_is_keyed_by_its_width_as_well_as_its_value() {
        // The interner is what decides two expressions are the same expression, so a constant's key
        // has to be the whole pattern. Keying on the value alone would make `1u8` and `1u32` one
        // node, and any two expressions differing only there would merge.
        let mut interner = ExprInterner::default();
        let narrow = interner.int_const(IntBits::from(1u8));
        let wide = interner.int_const(IntBits::from(1u32));
        assert_ne!(narrow, wide);

        // And the same pattern twice is the same node, which is the property that makes it an
        // interner at all.
        assert_eq!(narrow, interner.int_const(IntBits::from(1u8)));
    }

    /// `main(x, y) { a = x op1 y; b = x op2 y; return (a, b) }`, run through CSE.
    ///
    /// Returns the two values the terminator carries afterwards. If the pass merged the second
    /// expression into the first they are equal, because the redirect rewrites the `Return`.
    fn cse_two_ops(op1: BinaryArithOpKind, op2: BinaryArithOpKind) -> (ValueId, ValueId) {
        let mut ssa = HLSSA::with_main("main".to_string());
        let (x, y) = (ssa.fresh_value(), ssa.fresh_value());
        let (a, b) = (ssa.fresh_value(), ssa.fresh_value());

        let f = ssa.get_unique_entrypoint_mut();
        let entry = f.get_entry_mut();
        entry.push_parameter(x, Type::int(64));
        entry.push_parameter(y, Type::int(64));
        for (kind, result) in [(op1, a), (op2, b)] {
            entry.push_test_instruction(OpCode::BinaryArithOp {
                kind,
                result,
                lhs: x,
                rhs: y,
            });
        }
        entry.set_terminator(Terminator::Return(vec![a, b]));

        let cfg = FlowAnalysis::run(&ssa);
        CSE::post_r1c().do_run(&mut ssa, &cfg);

        let f = ssa.get_unique_entrypoint();
        match f.get_entry().get_terminator() {
            Some(Terminator::Return(values)) => (values[0], values[1]),
            other => panic!("expected the entry block to still return two values, got {other:?}"),
        }
    }

    #[test]
    fn two_occurrences_of_one_expression_are_deduplicated() {
        // The positive control. Without it, every assertion below would pass just as well on a pass
        // that had stopped merging anything at all.
        let (a, b) = cse_two_ops(BinaryArithOpKind::UDiv, BinaryArithOpKind::UDiv);
        assert_eq!(a, b, "two identical divisions should share one value");
    }

    #[test]
    fn the_two_signed_forms_of_a_division_are_not_one_expression() {
        // The split. `UDiv` and `SDiv` compute different values from the same operands, so merging
        // them is a wrong answer -- whichever survives is lowered for the sign it carries, and the
        // other operation silently becomes it.
        let (a, b) = cse_two_ops(BinaryArithOpKind::UDiv, BinaryArithOpKind::SDiv);
        assert_ne!(
            a, b,
            "an unsigned and a signed division are different expressions"
        );
    }

    #[test]
    fn no_operation_merges_with_its_opposite_sign() {
        // Every group with two forms, not just the ones whose forms compute different values.
        // `UAdd`/`SAdd` agree on every bit, but this is not a licence to merge them: the survivor's
        // opcode is what picks the overflow check emitted downstream.
        use BinaryArithOpKind::*;
        for (unsigned, signed) in [
            (UAdd, SAdd),
            (USub, SSub),
            (UMul, SMul),
            (UDiv, SDiv),
            (URem, SRem),
            (UShl, SShl),
            (UShr, SShr),
        ] {
            let (a, b) = cse_two_ops(unsigned, signed);
            assert_ne!(a, b, "{unsigned:?} and {signed:?} must not merge");
        }
    }

    #[test]
    fn a_signed_chain_does_not_absorb_an_unsigned_one() {
        // `Add` and `Mul` are flattened chains, so the sign has to live on the chain rather than on
        // the instruction: `(x + y) s+ z` must not splice the unsigned pair into a signed chain and
        // come out as the same expression as `(x s+ y) s+ z`.
        use BinaryArithOpKind::*;
        let build = |inner: BinaryArithOpKind| {
            let mut ssa = HLSSA::with_main("main".to_string());
            let (x, y, z) = (ssa.fresh_value(), ssa.fresh_value(), ssa.fresh_value());
            let (inner_r, outer_r) = (ssa.fresh_value(), ssa.fresh_value());
            let (signed_inner, other_r) = (ssa.fresh_value(), ssa.fresh_value());

            let f = ssa.get_unique_entrypoint_mut();
            let entry = f.get_entry_mut();
            for v in [x, y, z] {
                entry.push_parameter(v, Type::int(64));
            }
            for (kind, result, lhs, rhs) in [
                (inner, inner_r, x, y),
                (SAdd, outer_r, inner_r, z),
                // A wholly signed chain over the same leaves. It may merge with `outer_r` only
                // when `inner` was itself signed.
                (SAdd, signed_inner, x, y),
                (SAdd, other_r, signed_inner, z),
            ] {
                entry.push_test_instruction(OpCode::BinaryArithOp {
                    kind,
                    result,
                    lhs,
                    rhs,
                });
            }
            entry.set_terminator(Terminator::Return(vec![outer_r, other_r]));

            let cfg = FlowAnalysis::run(&ssa);
            CSE::post_r1c().do_run(&mut ssa, &cfg);

            let f = ssa.get_unique_entrypoint();
            match f.get_entry().get_terminator() {
                Some(Terminator::Return(values)) => (values[0], values[1]),
                other => {
                    panic!("expected the entry block to still return two values, got {other:?}")
                }
            }
        };

        let (a, b) = build(SAdd);
        assert_eq!(a, b, "two identical signed chains should share one value");

        let (a, b) = build(UAdd);
        assert_ne!(
            a, b,
            "an unsigned sub-chain must stay a leaf inside a signed chain"
        );
    }

    #[test]
    fn the_two_signed_forms_of_a_comparison_are_not_one_expression() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let (x, y) = (ssa.fresh_value(), ssa.fresh_value());
        let (a, b) = (ssa.fresh_value(), ssa.fresh_value());

        let f = ssa.get_unique_entrypoint_mut();
        let entry = f.get_entry_mut();
        entry.push_parameter(x, Type::int(64));
        entry.push_parameter(y, Type::int(64));
        for (kind, result) in [(CmpKind::ULt, a), (CmpKind::SLt, b)] {
            entry.push_test_instruction(OpCode::Cmp {
                kind,
                result,
                lhs: x,
                rhs: y,
            });
        }
        entry.set_terminator(Terminator::Return(vec![a, b]));

        let cfg = FlowAnalysis::run(&ssa);
        CSE::post_r1c().do_run(&mut ssa, &cfg);

        let f = ssa.get_unique_entrypoint();
        let Some(Terminator::Return(values)) = f.get_entry().get_terminator() else {
            panic!("expected the entry block to still return two values");
        };
        assert_ne!(
            values[0], values[1],
            "`<` and `s<` are different comparisons of the same operands"
        );
    }
}
