use crate::compiler::ssa::{
    ValueId,
    hlssa::{
        CmpKind, OpCode, TypeExpr,
        builder::{HLBlockEmitter, HLEmitter},
    },
};

use super::{InstructionLoweringRule, LoweringContext};

pub struct LowerWitnessAssertOps {}

impl InstructionLoweringRule for LowerWitnessAssertOps {
    fn lower_instruction(
        &self,
        b: &mut HLBlockEmitter<'_>,
        context: &LoweringContext<'_>,
        instruction: &OpCode,
    ) -> bool {
        if let OpCode::Guard { condition, inner } = instruction {
            self.process_assert(b, context, Some(*condition), inner.as_ref())
        } else {
            self.process_assert(b, context, None, instruction)
        }
    }
}

impl LowerWitnessAssertOps {
    pub fn new() -> Self {
        Self {}
    }

    fn process_assert(
        &self,
        b: &mut HLBlockEmitter<'_>,
        context: &LoweringContext<'_>,
        guard: Option<ValueId>,
        op: &OpCode,
    ) -> bool {
        match op {
            OpCode::Assert { value } => {
                if guard.is_none() && !context.types().get_value_type(*value).is_witness_of() {
                    return false;
                }
                self.lower_assert_value(b, context, guard, *value);
                true
            }
            // `assert(x == x)` is a tautology for every `x`, under any guard, so it lowers to
            // nothing at all. Only `Eq` may be dropped this way: `assert(x < x)` on the same value
            // is the _opposite_ — always false — and must keep its rejecting constraint.
            //
            // This is not a hypothetical tidy-up. `LowerPureGuards` emits a divide-by-zero check
            // in front of every `Div`/`Mod`, including the ones whose divisor is a nonzero
            // constant. For those, SCS folds the failure condition to the interned `u1 0` — the
            // very value id the check compares against — leaving `AssertCmp(Eq, c, c)`. Lowered as
            // an ordinary guarded assert that becomes `constrain(cond, 0, 0)`: an R1CS row that
            // constrains nothing.
            OpCode::AssertCmp {
                kind: CmpKind::Eq,
                lhs,
                rhs,
            } if lhs == rhs => true,

            OpCode::AssertCmp { kind, lhs, rhs } => {
                let lhs_witness = context.types().get_value_type(*lhs).is_witness_of();
                let rhs_witness = context.types().get_value_type(*rhs).is_witness_of();
                if guard.is_none() && !lhs_witness && !rhs_witness {
                    return false;
                }
                match kind {
                    CmpKind::Eq => self.lower_assert_eq(b, context, guard, *lhs, *rhs),
                    CmpKind::ULt | CmpKind::SLt => self.lower_assert_lt(
                        b,
                        context,
                        guard,
                        *kind,
                        *lhs,
                        *rhs,
                        lhs_witness,
                        rhs_witness,
                    ),
                }
                true
            }
            OpCode::AssertR1C { a, b: r1c_b, c } => {
                let a_witness = context.types().get_value_type(*a).is_witness_of();
                let b_witness = context.types().get_value_type(*r1c_b).is_witness_of();
                let c_witness = context.types().get_value_type(*c).is_witness_of();
                if guard.is_none() && !a_witness && !b_witness && !c_witness {
                    return false;
                }
                self.lower_assert_r1c(b, context, guard, *a, *r1c_b, *c);
                true
            }
            _ => false,
        }
    }

    fn lower_assert_value(
        &self,
        b: &mut HLBlockEmitter<'_>,
        context: &LoweringContext<'_>,
        guard: Option<ValueId>,
        value: ValueId,
    ) {
        let value_type = context.types().get_value_type(value);
        let value_field = b.ensure_field(value, &value_type.strip_witness());
        self.lower_assert_field(b, context, guard, value_field);
    }

    fn lower_assert_field(
        &self,
        b: &mut HLBlockEmitter<'_>,
        context: &LoweringContext<'_>,
        guard: Option<ValueId>,
        value_field: ValueId,
    ) {
        let cond_field = guard
            .map(|condition| b.ensure_field(condition, context.types().get_value_type(condition)))
            .unwrap_or_else(|| b.field_const(b.field().one()));
        b.constrain(cond_field, value_field, cond_field);
    }

    fn lower_assert_eq(
        &self,
        b: &mut HLBlockEmitter<'_>,
        context: &LoweringContext<'_>,
        guard: Option<ValueId>,
        lhs: ValueId,
        rhs: ValueId,
    ) {
        let lhs_type = context.types().get_value_type(lhs);
        let rhs_type = context.types().get_value_type(rhs);
        let lhs_field = b.ensure_field(lhs, &lhs_type.strip_witness());
        let rhs_field = b.ensure_field(rhs, &rhs_type.strip_witness());
        if let Some(condition) = guard {
            let cond_field = b.ensure_field(condition, context.types().get_value_type(condition));
            let diff = b.usub(lhs_field, rhs_field);
            let zero = b.field_const(b.field().zero());
            b.constrain(cond_field, diff, zero);
        } else {
            let one = b.field_const(b.field().one());
            b.constrain(lhs_field, one, rhs_field);
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn lower_assert_lt(
        &self,
        b: &mut HLBlockEmitter<'_>,
        context: &LoweringContext<'_>,
        guard: Option<ValueId>,
        kind: CmpKind,
        lhs: ValueId,
        rhs: ValueId,
        lhs_witness: bool,
        rhs_witness: bool,
    ) {
        // The width is read from `rhs`, as it always has been here, while every other consumer
        // reads `lhs` (`symbolic_executor::cmp_operand_bits`, `hlssa_to_llssa`). Nothing enforces
        // that the two agree: `Types` looks at both operands of a comparison only to decide whether
        // the `bool` result is witnessed, and never compares their widths. So the agreement is a
        // property of what the frontend emits (Noir unifies the operands of a comparison) rather
        // than a checked invariant, so we check it in debug builds.
        let lhs_type = context.types().get_value_type(lhs);
        let rhs_type = context.types().get_value_type(rhs);
        let signed = kind.is_signed();

        let TypeExpr::Int(bits) = rhs_type.strip_witness().expr else {
            panic!("ICE: AssertCmp Lt rhs is not an integer type");
        };
        debug_assert!(
            matches!(lhs_type.strip_witness().expr, TypeExpr::Int(lhs_bits) if lhs_bits == bits),
            "ICE: AssertCmp Lt operands disagree on width: {lhs_type} vs {rhs_type}"
        );

        // A signed assertion always goes via the comparison, which is the only lowering that reads
        // the operands as two's complement. An unsigned one prefers the direct form, except behind
        // a guard on two pure operands, where the comparison is cheaper.
        if signed || (guard.is_some() && !lhs_witness && !rhs_witness) {
            self.lower_assert_lt_via_cmp(b, context, guard, kind, lhs, rhs);
        } else {
            self.lower_unsigned_assert_lt(b, context, guard, lhs, rhs, bits);
        }
    }

    /// Re-express `assert(lhs < rhs)` as `assert(lhs < rhs == true)`, leaving the comparison for
    /// `LowerWitnessCompareOps` to lower.
    ///
    /// The comparison it emits carries the assertion's own sign, as resolved by the caller. It
    /// reaches here on the signed operands themselves — not on a raw-bit stand-in the way
    /// `lower_unsigned_assert_lt` does — so emitting an unsigned comparison would be a genuinely
    /// different test.
    fn lower_assert_lt_via_cmp(
        &self,
        b: &mut HLBlockEmitter<'_>,
        context: &LoweringContext<'_>,
        guard: Option<ValueId>,
        kind: CmpKind,
        lhs: ValueId,
        rhs: ValueId,
    ) {
        let cmp = b.cmp(lhs, rhs, kind);
        let cmp_field = b.cast_to_field(cmp);
        self.lower_assert_field(b, context, guard, cmp_field);
    }

    fn lower_unsigned_assert_lt(
        &self,
        b: &mut HLBlockEmitter<'_>,
        context: &LoweringContext<'_>,
        guard: Option<ValueId>,
        lhs: ValueId,
        rhs: ValueId,
        bits: usize,
    ) {
        assert!(bits > 0, "rangecheck width must be at least 1 bit");
        let lhs_type = context.types().get_value_type(lhs);
        let rhs_type = context.types().get_value_type(rhs);
        let lhs_field = b.ensure_field(lhs, &lhs_type.strip_witness());
        let rhs_field = b.ensure_field(rhs, &rhs_type.strip_witness());
        let diff = b.usub(rhs_field, lhs_field);
        let one = b.field_const(b.field().one());
        let diff_minus_one = b.usub(diff, one);
        self.emit_rangecheck(b, guard, diff_minus_one, bits);
    }

    fn emit_rangecheck(
        &self,
        b: &mut HLBlockEmitter<'_>,
        guard: Option<ValueId>,
        value: ValueId,
        bits: usize,
    ) {
        let rangecheck = OpCode::Rangecheck {
            value,
            max_bits: bits,
        };
        if let Some(condition) = guard {
            b.emit(OpCode::Guard {
                condition,
                inner: Box::new(rangecheck),
            });
        } else {
            b.emit(rangecheck);
        }
    }

    fn lower_assert_r1c(
        &self,
        b: &mut HLBlockEmitter<'_>,
        context: &LoweringContext<'_>,
        guard: Option<ValueId>,
        a: ValueId,
        r1c_b: ValueId,
        c: ValueId,
    ) {
        let a_type = context.types().get_value_type(a);
        let b_type = context.types().get_value_type(r1c_b);
        let c_type = context.types().get_value_type(c);
        let a_field = b.ensure_field(a, &a_type.strip_witness());
        let b_field = b.ensure_field(r1c_b, &b_type.strip_witness());
        let c_field = b.ensure_field(c, &c_type.strip_witness());

        if let Some(condition) = guard {
            let product = b.umul(a_field, b_field);
            let diff = b.usub(product, c_field);
            let cond_field = b.ensure_field(condition, context.types().get_value_type(condition));
            let zero = b.field_const(b.field().zero());
            b.constrain(cond_field, diff, zero);
        } else {
            b.constrain(a_field, b_field, c_field);
        }
    }
}
