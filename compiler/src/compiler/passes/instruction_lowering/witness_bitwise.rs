//! Lowers integer bitwise, bit-selection, and sign-extension operations before the main
//! explicit-witness pass.
//!
//! This pass emits `Spread`/`Unspread` operations, except for `u64` bitwise ops where it keeps a
//! two-limb `u32` decomposition. It also canonicalizes witness integer casts/shifts into the shared
//! `BitRange` representation where possible.

use crate::compiler::{
    analysis::{
        types::FunctionTypeInfo,
        value_range_analysis::{Interval, field_modulus},
    },
    passes::instruction_lowering::{InstructionLoweringRule, LoweringContext, integer_bits},
    ssa::{
        ValueId,
        hlssa::{
            ArithGroup, BinaryArithOpKind, CastTarget, CmpKind, MAX_SUPPORTED_UNSIGNED_BITS,
            OpCode, Type, TypeExpr, assert_signed_op_width,
            builder::{HLBlockEmitter, HLEmitter},
        },
    },
};

use mavros_artifacts::FieldConfig;
use num_bigint::BigInt;
use num_traits::{One, ToPrimitive};

pub struct LowerWitnessBitwiseOps {}

impl InstructionLoweringRule for LowerWitnessBitwiseOps {
    fn lower_instruction(
        &self,
        b: &mut HLBlockEmitter<'_>,
        context: &LoweringContext<'_>,
        instruction: &OpCode,
    ) -> bool {
        if let OpCode::Guard { condition, inner } = instruction {
            self.process_guarded_shift(b, context, *condition, inner.as_ref())
        } else {
            self.process_op(b, context, instruction)
        }
    }
}

impl LowerWitnessBitwiseOps {
    pub fn new() -> Self {
        Self {}
    }

    fn process_op(
        &self,
        b: &mut HLBlockEmitter<'_>,
        context: &LoweringContext<'_>,
        op: &OpCode,
    ) -> bool {
        let function_type_info = context.types();
        match op {
            OpCode::BinaryArithOp {
                kind:
                    kind @ (BinaryArithOpKind::And | BinaryArithOpKind::Or | BinaryArithOpKind::Xor),
                result,
                lhs,
                rhs,
            } => {
                let lhs_witness = function_type_info.get_value_type(*lhs).is_witness_of();
                let rhs_witness = function_type_info.get_value_type(*rhs).is_witness_of();
                if lhs_witness || rhs_witness {
                    self.lower_binary_bitwise(
                        b,
                        function_type_info,
                        *kind,
                        *result,
                        *lhs,
                        *rhs,
                        lhs_witness,
                        rhs_witness,
                    );
                    true
                } else {
                    false
                }
            }
            OpCode::Not { result, value } => {
                self.lower_not(b, function_type_info, *result, *value);
                true
            }
            OpCode::SExt {
                result,
                value,
                from_bits,
                to_bits,
            } if integer_bits(context.types().get_value_type(*value)).is_some() => {
                self.lower_integer_sext(b, context, *result, *value, *from_bits, *to_bits);
                true
            }
            OpCode::BinaryArithOp {
                kind,
                result,
                lhs,
                rhs,
            } if matches!(kind.group(), ArithGroup::Shl | ArithGroup::Shr)
                && (context.types().get_value_type(*lhs).is_witness_of()
                    || context.types().get_value_type(*rhs).is_witness_of()) =>
            {
                self.lower_shift(b, context, None, *kind, *result, *lhs, *rhs);
                true
            }
            _ => false,
        }
    }

    fn process_guarded_shift(
        &self,
        b: &mut HLBlockEmitter<'_>,
        context: &LoweringContext<'_>,
        condition: ValueId,
        op: &OpCode,
    ) -> bool {
        match op {
            OpCode::BinaryArithOp {
                kind,
                result,
                lhs,
                rhs,
            } if matches!(kind.group(), ArithGroup::Shl | ArithGroup::Shr)
                && (context.types().get_value_type(*lhs).is_witness_of()
                    || context.types().get_value_type(*rhs).is_witness_of()) =>
            {
                self.lower_shift(b, context, Some(condition), *kind, *result, *lhs, *rhs);
                true
            }
            _ => false,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn lower_binary_bitwise(
        &self,
        b: &mut HLBlockEmitter<'_>,
        function_type_info: &FunctionTypeInfo,
        kind: BinaryArithOpKind,
        result: ValueId,
        lhs: ValueId,
        rhs: ValueId,
        lhs_witness: bool,
        rhs_witness: bool,
    ) {
        let (bits, result_cast) =
            integer_bits_and_cast(function_type_info, result, "bitwise result");
        assert!(
            bits <= MAX_SUPPORTED_UNSIGNED_BITS,
            "bitwise spread width too large for natural-width Spread lowering: {bits}"
        );

        let lhs = b.cast_to(CastTarget::Int(bits), lhs);
        let rhs = b.cast_to(CastTarget::Int(bits), rhs);

        if bits == 1 {
            self.lower_u1_bitwise(b, kind, result, lhs, rhs);
            return;
        }

        let result_word = if bits == 64 {
            let lhs_limbs = decompose_u64_input(b, lhs, lhs_witness);
            let rhs_limbs = decompose_u64_input(b, rhs, rhs_witness);
            let result_limbs = lower_u64_limb_bitwise(b, kind, lhs_limbs, rhs_limbs);
            combine_u32_limbs(b, result_limbs)
        } else if bits == 128 {
            let lhs_limbs = extract_u128_limbs(b, lhs);
            let rhs_limbs = extract_u128_limbs(b, rhs);
            let lhs_lo = decompose_u64_input(b, lhs_limbs.lo, lhs_witness);
            let rhs_lo = decompose_u64_input(b, rhs_limbs.lo, rhs_witness);
            let lo = lower_u64_limb_bitwise(b, kind, lhs_lo, rhs_lo);
            let lhs_hi = decompose_u64_input(b, lhs_limbs.hi, lhs_witness);
            let rhs_hi = decompose_u64_input(b, rhs_limbs.hi, rhs_witness);
            let hi = lower_u64_limb_bitwise(b, kind, lhs_hi, rhs_hi);
            let lo = combine_u32_limbs(b, lo);
            let hi = combine_u32_limbs(b, hi);
            combine_u64_fields(b, lo, hi)
        } else {
            lower_word_bitwise(b, kind, lhs, rhs, bits as u8)
        };

        b.emit(OpCode::Cast {
            result,
            value: result_word,
            target: result_cast,
        });
    }

    fn lower_u1_bitwise(
        &self,
        b: &mut HLBlockEmitter<'_>,
        kind: BinaryArithOpKind,
        result: ValueId,
        lhs: ValueId,
        rhs: ValueId,
    ) {
        let target = CastTarget::Int(1);
        let lhs_field = b.cast_to_field(lhs);
        let rhs_field = b.cast_to_field(rhs);

        let result_field = match kind {
            BinaryArithOpKind::And => b.umul(lhs_field, rhs_field),
            BinaryArithOpKind::Or => {
                let sum = b.uadd(lhs_field, rhs_field);
                let product = b.umul(lhs_field, rhs_field);
                b.usub(sum, product)
            }
            BinaryArithOpKind::Xor => {
                let sum = b.uadd(lhs_field, rhs_field);
                let two = b.field_const(b.field().constant(2));
                let product = b.umul(lhs_field, rhs_field);
                let two_product = b.umul(two, product);
                b.usub(sum, two_product)
            }
            _ => unreachable!(),
        };

        b.emit(OpCode::Cast {
            result,
            value: result_field,
            target,
        });
    }

    // FIELD-ASSUMPTION: L6-int-op-strategy
    // `not = (2^bits - 1) - value`. The all-ones mask `2^bits - 1` exceeds p at bits=64 on a
    // small field, so u64/u128 `not` must be done per-limb.
    fn lower_not(
        &self,
        b: &mut HLBlockEmitter<'_>,
        function_type_info: &FunctionTypeInfo,
        result: ValueId,
        value: ValueId,
    ) {
        let (bits, cast_target) = integer_bits_and_cast(function_type_info, value, "bitwise not");
        // FIELD-ASSUMPTION: L4-decompose
        let ones = b.field_const(b.field().two_pow(bits) - b.field().one());
        let value_field = b.cast_to_field(value);
        let not_value = b.usub(ones, value_field);
        b.emit(OpCode::Cast {
            result,
            value: not_value,
            target: cast_target,
        });
    }

    // FIELD-ASSUMPTION: L6-int-op-strategy
    // Sign-extends via `value + sign * (field.two_pow(to_bits) - field.two_pow(from_bits))`. The
    // `field.two_pow(to_bits)` shift wraps mod p once `to_bits` reaches the field width.
    fn lower_integer_sext(
        &self,
        b: &mut HLBlockEmitter<'_>,
        context: &LoweringContext<'_>,
        result: ValueId,
        value: ValueId,
        from_bits: usize,
        to_bits: usize,
    ) {
        // The bound belongs on the **source**, not on the target. A signed source is capped at
        // `MAX_SUPPORTED_SIGNED_BITS` for now, but the target is just a wider integer to deposit
        // the result in, and widening an `i32` into a `u128` is exactly what `x as u128` asks for.
        assert_signed_op_width(from_bits, "sign extension source");
        assert!(
            from_bits < to_bits && to_bits <= MAX_SUPPORTED_UNSIGNED_BITS,
            "sign extension must widen within the integer type cap: {from_bits} -> {to_bits}"
        );

        // FIELD-ASSUMPTION: L4-modulus-query. The extension term below is
        // `two_pow(to_bits) - two_pow(from_bits)`, which is only the value it is meant to be while
        // `two_pow(to_bits)` has not wrapped. On bn254 the 128-bit cap leaves ample room; on a
        // narrower field this refuses rather than silently extending by a wrapped constant.
        assert!(
            to_bits < b.field().field_bit_size() as usize,
            "sign extension to {to_bits} bits needs a field wider than {} bits",
            b.field().field_bit_size()
        );

        // The question is whether bit `from_bits - 1` of the encoding is provably clear, so it is
        // asked of the range record rather than of one chosen reading — `SExt`'s source may be
        // either signed or unsigned, and each carries its information in a different component.
        let sign = if context.range(value).is_non_negative_at_width(from_bits) {
            b.field_const(b.field().zero())
        } else {
            let sign_bits = b.bit_range(value, from_bits - 1, 1);
            b.cast_to_field(sign_bits)
        };
        let value_field = b.cast_to_field(value);
        // FIELD-ASSUMPTION: L4-decompose
        let extension = b.field_const(b.field().two_pow(to_bits) - b.field().two_pow(from_bits));
        let offset = b.umul(sign, extension);
        let extended = b.uadd(value_field, offset);
        b.emit(OpCode::Cast {
            result,
            value: extended,
            target: cast_target_for_integer_type(context.types().get_value_type(result)),
        });
    }

    /// Lowers a shift with at least one witness operand.
    ///
    /// An unsigned left-hand side shifted by a _pure_ amount keeps its own lowering, which folds
    /// the amount into a constant and is what every shift in the corpus takes today. A signed
    /// left-hand side or a witness amount goes to [`Self::lower_general_shift`], which pays for a
    /// runtime shift-amount decomposition.
    ///
    /// The amount check is emitted **here**, above the split, because it is owed by both of them
    /// and only one of them used to pay it. [`Self::lower_general_shift`] needs it to bound the
    /// decomposition it is about to build, which made it easy to read as part of that lowering;
    /// [`Self::lower_constant_amount_shift`] hands the raw amount to a backend shift instead, so
    /// its need for the check is just as real and far less visible. Without it an out-of-range
    /// amount reached `1 << rhs`, where the VM's `shl_int` masks it to `bits - 1` and the program
    /// quietly computes a shift Noir rejects. Hoisting it is what makes the check a property of
    /// "a shift" rather than of one of the two ways to lower one.
    #[allow(clippy::too_many_arguments)]
    fn lower_shift(
        &self,
        b: &mut HLBlockEmitter<'_>,
        context: &LoweringContext<'_>,
        guard: Option<ValueId>,
        kind: BinaryArithOpKind,
        result: ValueId,
        lhs: ValueId,
        rhs: ValueId,
    ) {
        let lhs_type = context.types().get_value_type(lhs);
        let bits = integer_bits(lhs_type)
            .unwrap_or_else(|| panic!("witness shift on non-integer lhs type {lhs_type:?}"));
        // The shift's own sign decides which lowering runs.
        let lhs_signed = kind.is_signed();
        let rhs_witness = context.types().get_value_type(rhs).is_witness_of();

        // The check below indexes bit `log2(bits)` upwards as "too large", which is only the right
        // test when `bits` is a power of two. Every Noir integer width is, but the lowering would
        // be silently wrong rather than merely unsupported if that ever changed.
        assert!(
            bits.is_power_of_two(),
            "the shift-amount check assumes a power-of-two integer width, got {bits}"
        );
        if lhs_signed {
            assert_signed_op_width(bits, "shift");
        }

        let widths = shift_amount_bits(context, rhs, bits);
        emit_shift_amount_check(b, context, guard, rhs, widths);

        if !rhs_witness && !lhs_signed {
            self.lower_constant_amount_shift(b, context, guard, kind, result, lhs, rhs, bits);
        } else {
            self.lower_general_shift(
                b, context, guard, kind, result, lhs, rhs, bits, lhs_signed, widths,
            );
        }
    }

    /// The pre-existing lowering: an unsigned left-hand side shifted by a pure amount.
    ///
    /// The `1 << rhs` here is a _pure_ `Shl`, which constant-folds later. It has to: `Shl` on a
    /// `U(bits)` reaches `hlssa_to_r1cs` only with both operands constant, so this shape is viable
    /// precisely because the amount is not a witness.
    ///
    /// That `Shl` is emitted **after** `LowerPureGuards` has run, so nothing checks its amount
    /// downstream of here and nothing can: it is the last chance. [`Self::lower_shift`] has already
    /// taken it. Do not move that check into the sibling lowering on the grounds that it is where
    /// the decomposition needs it — an unchecked amount is harmless-looking on this path and is
    /// not harmless, because the backend shift masks it to `bits - 1` and answers rather than
    /// failing.
    #[allow(clippy::too_many_arguments)]
    fn lower_constant_amount_shift(
        &self,
        b: &mut HLBlockEmitter<'_>,
        context: &LoweringContext<'_>,
        guard: Option<ValueId>,
        kind: BinaryArithOpKind,
        result: ValueId,
        lhs: ValueId,
        rhs: ValueId,
        bits: usize,
    ) {
        let one_u = b.int_const(bits, 1);
        let factor = b.fresh_value();
        b.emit(OpCode::BinaryArithOp {
            kind: BinaryArithOpKind::UShl,
            result: factor,
            lhs: one_u,
            rhs,
        });

        match kind.group() {
            ArithGroup::Shl => {
                let lhs_field = b.cast_to_field(lhs);
                let factor_field = b.cast_to_field(factor);
                let shifted = b.umul(lhs_field, factor_field);
                let value = wrap_shifted_product(b, context, shifted, rhs, bits, guard);
                b.emit(OpCode::Cast {
                    result,
                    value,
                    target: CastTarget::Int(bits),
                });
            }
            ArithGroup::Shr => {
                b.emit_guarded(
                    guard,
                    OpCode::BinaryArithOp {
                        // The value being divided is `U(bits)` on this path — the signed
                        // left-hand side goes to `lower_general_shift` — so this is an unsigned
                        // division, not a re-tagging of the shift's own sign.
                        kind: BinaryArithOpKind::UDiv,
                        result,
                        lhs,
                        rhs: factor,
                    },
                );
            }
            _ => unreachable!("lower_shift only dispatches Shl and Shr"),
        }
    }

    /// Lowers a shift whose amount is not a compile-time constant, whose left-hand side is signed,
    /// or both.
    ///
    /// The amount is decomposed into bits and `2^amount` rebuilt as a product, because nothing
    /// below HLSSA can shift by a variable. Noir treats a shift by `bits` or more as a runtime
    /// error, so the amount's high bits are asserted clear rather than masked — the backends mask
    /// and the cost model saturates, and neither answer is Noir's.
    #[allow(clippy::too_many_arguments)]
    fn lower_general_shift(
        &self,
        b: &mut HLBlockEmitter<'_>,
        context: &LoweringContext<'_>,
        guard: Option<ValueId>,
        kind: BinaryArithOpKind,
        result: ValueId,
        lhs: ValueId,
        rhs: ValueId,
        bits: usize,
        lhs_signed: bool,
        widths: ShiftAmountWidths,
    ) {
        let ShiftAmountWidths {
            rhs_bits,
            amount_bits,
        } = widths;
        let amount = extract_amount_bits(b, rhs, rhs_bits, amount_bits);
        let factor = build_shift_factor(b, &amount);

        match (kind.group(), lhs_signed) {
            // `Shl` is the one shift that takes no sign: the shifted product is wrapped and then
            // reinterpreted at `bits`, which is the same bit pattern under either reading. The
            // match arm was already sign-agnostic; now the callee is too.
            (ArithGroup::Shl, _) => {
                self.lower_shl(b, context, guard, result, lhs, rhs, factor, bits)
            }
            (ArithGroup::Shr, false) => {
                self.lower_unsigned_shr(b, guard, result, lhs, factor, bits)
            }
            (ArithGroup::Shr, true) => self.lower_signed_shr(
                b,
                context,
                guard,
                result,
                lhs,
                factor,
                &amount,
                amount_bits,
                bits,
            ),
            _ => unreachable!("lower_shift only dispatches Shl and Shr"),
        }
    }

    /// `lhs * 2^n`, wrapped to the declared width.
    ///
    /// Signedness only picks the result's cast target: a left shift is the same operation on the
    /// bit pattern either way, because `raw` and the mathematical value are congruent mod `2^bits`
    /// and so are their products with `2^n`. On `i8` that gives `64 << 1 == -128`, which is what
    /// Noir reports.
    #[allow(clippy::too_many_arguments)]
    fn lower_shl(
        &self,
        b: &mut HLBlockEmitter<'_>,
        context: &LoweringContext<'_>,
        guard: Option<ValueId>,
        result: ValueId,
        lhs: ValueId,
        rhs: ValueId,
        factor: ValueId,
        bits: usize,
    ) {
        let lhs_field = b.cast_to_field(lhs);
        let shifted = b.umul(lhs_field, factor);
        let value = wrap_shifted_product(b, context, shifted, rhs, bits, guard);
        b.emit(OpCode::Cast {
            result,
            value,
            target: CastTarget::Int(bits),
        });
    }

    /// `lhs / 2^n` on the raw bits. The divisor is a power of two in `[1, 2^(bits-1)]`, so the
    /// division is total whatever the amount turns out to be.
    fn lower_unsigned_shr(
        &self,
        b: &mut HLBlockEmitter<'_>,
        guard: Option<ValueId>,
        result: ValueId,
        lhs: ValueId,
        factor: ValueId,
        bits: usize,
    ) {
        let factor_u = b.cast_to(CastTarget::Int(bits), factor);
        b.emit_guarded(
            guard,
            OpCode::BinaryArithOp {
                kind: BinaryArithOpKind::UDiv,
                result,
                lhs,
                rhs: factor_u,
            },
        );
    }

    /// An arithmetic right shift, as `q + sign * (2^bits - 2^(bits-n))`.
    ///
    /// `q` is the _unsigned_ division of the raw bits, which is the right answer for a non-negative
    /// value and `2^(bits-n)` too small for a negative one — because `floor((raw - 2^bits) / 2^n) =
    /// q - 2^(bits-n)`, and re-encoding that adds `2^bits` back. So the correction is exactly
    /// `2^bits - 2^(bits-n)`, and it sign-fills as `>>` must: on `i8`, `-4 >> 1` is `126 + 128 =
    /// 254`, and `-1 >> 7` is `1 + 254 = 255`, saturating at `-1` rather than becoming a large
    /// positive number.
    #[allow(clippy::too_many_arguments)]
    fn lower_signed_shr(
        &self,
        b: &mut HLBlockEmitter<'_>,
        context: &LoweringContext<'_>,
        guard: Option<ValueId>,
        result: ValueId,
        lhs: ValueId,
        factor: ValueId,
        amount: &[ValueId],
        amount_bits: usize,
        bits: usize,
    ) {
        let raw = b.cast_to(CastTarget::Int(bits), lhs);
        let factor_u = b.cast_to(CastTarget::Int(bits), factor);
        let quotient = b.fresh_value();
        b.emit_guarded(
            guard,
            OpCode::BinaryArithOp {
                kind: BinaryArithOpKind::UDiv,
                result: quotient,
                lhs: raw,
                rhs: factor_u,
            },
        );

        let quotient_field = b.cast_to_field(quotient);
        let value = match sign_bit_of(b, context, lhs, bits) {
            None => quotient_field,
            Some(sign) => {
                let cofactor = build_shift_cofactor(b, amount, amount_bits);
                // FIELD-ASSUMPTION: L4-decompose
                let two_pow_bits = b.field_const(b.field().two_pow(bits));
                let fill = b.usub(two_pow_bits, cofactor);
                let offset = b.umul(sign, fill);
                b.uadd(quotient_field, offset)
            }
        };

        b.emit(OpCode::Cast {
            result,
            value,
            target: CastTarget::Int(bits),
        });
    }
}

/// The two widths a shift-amount check and decomposition are cut against.
#[derive(Clone, Copy)]
struct ShiftAmountWidths {
    /// The declared width of the amount operand.
    rhs_bits: usize,
    /// `log2(bits)`: how many bits of the amount a valid shift can use.
    amount_bits: usize,
}

/// The widths for a shift of a `bits`-wide value by `rhs`.
fn shift_amount_bits(
    context: &LoweringContext<'_>,
    rhs: ValueId,
    bits: usize,
) -> ShiftAmountWidths {
    let rhs_type = context.types().get_value_type(rhs);
    let rhs_bits = integer_bits(rhs_type)
        .unwrap_or_else(|| panic!("witness shift by a non-integer amount type {rhs_type:?}"));
    ShiftAmountWidths {
        rhs_bits,
        amount_bits: bits.trailing_zeros() as usize,
    }
}

/// Asserts that the shift amount is smaller than the width being shifted.
///
/// Since the width is a power of two, "too large" is just "some bit at or above `log2(bits)` is
/// set" — and that one test also catches a _negative_ amount, whose raw encoding always has its
/// top bit set and so is at least `2^(rhs_bits-1) >= bits`.
///
/// Guarded, so an inactive guard around an out-of-range shift is vacuous rather than a failure.
fn emit_shift_amount_check(
    b: &mut HLBlockEmitter<'_>,
    context: &LoweringContext<'_>,
    guard: Option<ValueId>,
    rhs: ValueId,
    widths: ShiftAmountWidths,
) {
    let ShiftAmountWidths {
        rhs_bits,
        amount_bits,
    } = widths;
    // No bit that high exists, so every amount this type can hold is in range.
    //
    // This also drops the negative-amount rejection that the doc above relies on, so it must only
    // ever fire where a negative amount cannot be represented either. It does: Noir's integer widths
    // are 1/8/16/32/64/128, and the narrowest of those that can hold a negative number is `i8`,
    // against `amount_bits <= 7`. The only way in is a one-bit amount, which has no negative reading
    // a shift could be given.
    //
    // Asserted rather than `debug_assert`ed: this is the whole justification for emitting no check,
    // so a release build must not be the one that skips it.
    if rhs_bits <= amount_bits {
        assert!(
            rhs_bits <= 1,
            "a {rhs_bits}-bit shift amount skipped the range check, so a negative amount would \
             read as a small positive one"
        );
        return;
    }

    // The range domain already proves it. This is the payoff the dual-interval domain was for: it
    // removes the check, and with it the only reason the factor ever needs neutralising.
    if context
        .urange(rhs)
        .proves_fits_in_unsigned_bits(amount_bits)
    {
        return;
    }

    let high = b.bit_range(rhs, amount_bits, rhs_bits - amount_bits);
    let high_field = b.cast_to_field(high);
    let zero = b.field_const(b.field().zero());

    b.emit_guarded(
        guard,
        OpCode::AssertCmp {
            kind: CmpKind::Eq,
            lhs: high_field,
            rhs: zero,
        },
    );
}

/// The low `log2(bits)` bits of the shift amount, as field elements.
fn extract_amount_bits(
    b: &mut HLBlockEmitter<'_>,
    rhs: ValueId,
    rhs_bits: usize,
    amount_bits: usize,
) -> Vec<ValueId> {
    (0..amount_bits.min(rhs_bits))
        .map(|i| {
            let bit = b.bit_range(rhs, i, 1);
            let bit_u1 = b.cast_to(CastTarget::Int(1), bit);
            b.cast_to_field(bit_u1)
        })
        .collect()
}

/// `2^n` from the bits of `n`, as `prod_i (1 + b_i * (2^(2^i) - 1))`.
///
/// Each term is linear in its bit, so this is `amount_bits - 1` multiplications. The widest
/// constant is `2^64 - 1`, at `i = 6` for a 128-bit shift.
fn build_shift_factor(b: &mut impl HLEmitter, amount: &[ValueId]) -> ValueId {
    let one = b.field_const(b.field().one());

    let mut acc: Option<ValueId> = None;
    for (i, bit) in amount.iter().enumerate() {
        // FIELD-ASSUMPTION: L4-decompose
        let step = b.field_const(b.field().two_pow(1 << i) - b.field().one());
        let scaled = b.umul(*bit, step);
        let term = b.uadd(one, scaled);

        acc = Some(match acc {
            None => term,
            Some(acc) => b.umul(acc, term),
        });
    }

    acc.unwrap_or(one)
}

/// `2^bits / 2^n`, built from the same bits rather than by dividing.
///
/// A field division would need a nonzero check on the divisor that nothing here can discharge.
/// Instead note that `2^bits = 2 * prod_{i<k} 2^(2^i)` where `k = log2(bits)`, so the quotient is
/// `2 * prod_i (2^(2^i) / f_i)` with the same per-bit factors `f_i` — and each term is once again
/// linear in the bit, as `2^(2^i) - b_i * (2^(2^i) - 1)`.
///
/// Bits the amount's own type is too narrow to hold are zero, so their terms fold into the leading
/// constant: `2^(1 + bits - 2^len)`.
fn build_shift_cofactor(b: &mut impl HLEmitter, amount: &[ValueId], amount_bits: usize) -> ValueId {
    debug_assert!(amount.len() <= amount_bits);

    // FIELD-ASSUMPTION: L4-decompose
    let leading = b
        .field()
        .two_pow(1 + (1 << amount_bits) - (1 << amount.len()));

    let mut acc = b.field_const(leading);
    for (i, bit) in amount.iter().enumerate() {
        let full = b.field().two_pow(1 << i);
        let step = b.field_const(full - b.field().one());
        let scaled = b.umul(*bit, step);
        let full_const = b.field_const(full);
        let term = b.usub(full_const, scaled);
        acc = b.umul(acc, term);
    }
    acc
}

/// The value's sign bit as a field element, or `None` when the range domain proves it clear.
fn sign_bit_of(
    b: &mut HLBlockEmitter<'_>,
    context: &LoweringContext<'_>,
    value: ValueId,
    bits: usize,
) -> Option<ValueId> {
    if context.range(value).is_non_negative_at_width(bits) {
        return None;
    }

    let sign_bits = b.bit_range(value, bits - 1, 1);
    let sign_u1 = b.cast_to(CastTarget::Int(1), sign_bits);

    Some(b.cast_to_field(sign_u1))
}

/// The low `bits` bits of `lhs * 2^n`, which is what Noir's `<<` evaluates to.
///
/// **A left shift wraps.** Noir reports a runtime error when the _amount_ reaches the width, but a
/// shift that merely pushes bits off the top truncates: `x << 63` is `0` for `x = 64: u64`
/// (`execution_success/bit_shifts_comptime`), and `64: i8 << 1` is `-128`
/// (`execution_success/bit_shifts_runtime`). Mavros' own interpreter already agreed with that; only
/// this lowering did not, because it rangechecked the product and so rejected the overflow instead
/// of discarding it.
///
/// The prover supplies `product >> bits` as a hint and we subtract it back off. **Both halves have
/// to be bounded.** It is tempting to argue that `discarded` needs no rangecheck of its own,
/// because `product - high * 2^bits` lands in `[0, 2^bits)` for exactly one integer `high` — but
/// `discarded` is a field element, not an integer. `2^bits` is invertible mod `p`, so without a
/// bound a prover can pick _any_ `wrapped` in `[0, 2^bits)` and solve
/// `discarded = (product - wrapped) * (2^bits)^-1`, leaving the shift result entirely unconstrained.
/// Bounding both is what makes the field identity lift to the integers, and hence unique. This is
/// the same discipline `bit_range.rs::lower_witness_bit_range` follows for every piece it splits
/// out.
///
/// The bound on `discarded` comes from the amount rather than from the width: `product` is
/// `raw * 2^n` with `raw < 2^bits`, so at most `n` bits can be pushed out, and the range domain
/// usually pins `n` exactly. A shift by a small constant — which is nearly all of them — therefore
/// pays a correspondingly small rangecheck, and an amount provably zero pays nothing at all.
///
/// FIELD-ASSUMPTION: L4-decompose. This needs `lhs * 2^n` not to wrap mod `p` — see
/// [`product_headroom_or_bail`], which is the precondition _both_ paths below are held to — and it
/// reads the discarded half through a `U(2 * bits)` intermediate. The second requirement fails at
/// `bits = 128`, where there is no `U(256)` to decompose the product with; that width therefore
/// keeps the old trapping rangecheck, which rejects a shift Noir would have wrapped and is wrong in
/// the same way it has always been wrong. Correcting _that_ needs a limb-wise lowering rather than a
/// single field product.
fn wrap_shifted_product(
    b: &mut HLBlockEmitter<'_>,
    context: &LoweringContext<'_>,
    product: ValueId,
    rhs: ValueId,
    bits: usize,
    guard: Option<ValueId>,
) -> ValueId {
    // `discarded_width` is the bound both paths reason against: the effective amount is the low
    // `log2(bits)` bits of `rhs`, so it never exceeds `bits - 1`, and ⊥ answers with that cap.
    let discarded_bits = discarded_width(&context.urange(rhs), bits);
    product_headroom_or_bail(bits, discarded_bits, b.field());

    let wide_bits = 2 * bits;
    if wide_bits > MAX_SUPPORTED_UNSIGNED_BITS {
        guarded_rangecheck(b, product, bits, guard);
        return product;
    }

    // Nothing can be shifted out of a shift by zero, so the product is already the answer.
    if discarded_bits == 0 {
        return product;
    }

    let pure_product = b.value_of(product);
    let wide = b.cast_to(CastTarget::Int(wide_bits), pure_product);
    let discarded_hint = b.bit_range(wide, bits, bits);
    let discarded_hint = b.cast_to_field(discarded_hint);
    let discarded = b.write_witness(discarded_hint);
    // Deliberately _not_ `guarded_rangecheck`. Both halves are bounded structurally rather than by
    // anything the guard controls: `factor` is at most `2^(bits - 1)` however the amount is built,
    // and every guarded failable lowering routes its result through
    // `witness_integer_arith::guarded_or_zero_field`, so `lhs` is inside its declared width even on
    // an inactive path. `product` is therefore below `2^(bits + discarded_bits)` unconditionally and
    // both checks are satisfiable whatever the guard does — which is what lets the result be bounded
    // on every path rather than only on the live one. The `bits = 128` fallback above is the one
    // lowering that does _not_ bound its result this way.
    b.rangecheck(discarded, discarded_bits);

    // FIELD-ASSUMPTION: L4-decompose
    let two_pow_bits = b.field_const(b.field().two_pow(bits));
    let overflow = b.umul(discarded, two_pow_bits);
    let wrapped = b.usub(product, overflow);
    b.rangecheck(wrapped, bits);

    wrapped
}

/// Refuse a `<<` whose product `lhs * 2^n` could wrap modulo the field.
///
/// This is the shared precondition of both halves of [`wrap_shifted_product`], and neither of them
/// means anything without it. `raw * 2^n` reaches `2^(bits + n)`, and once that can exceed the
/// modulus the product wraps: there are `raw < 2^bits` whose product lands in `[p, p + 2^bits)`,
/// leaving a residue no constraint on the product can tell apart from an honest one.
///
/// - On the **truncating** path the identity `wrapped = product - discarded * 2^bits` still has a
///   satisfying assignment with both halves in range, but `wrapped` is the low bits of the residue
///   rather than of the shift. The uniqueness argument that makes the field identity lift to the
///   integers needs `discarded * 2^bits + wrapped < p`, which is exactly this bound.
/// - On the **trapping** fallback the rangecheck simply accepts the residue.
///
/// Either way the circuit constrains a value with no relation to the shift while the VM computes the
/// truncated answer — a wrong answer rather than a rejection, and one no test can see without a
/// witness that hits the window. On bn254 at `bits = 128` that is `n >= 126`; every narrower width
/// has room to spare, which is why this has never fired there.
///
/// So the headroom is a precondition rather than an assumption, and a program that cannot meet it
/// fails loudly at compile time. That is deliberately _not_ how the fallback's other defect is
/// handled: at `bits = 128` it also _rejects_ a shift that should have wrapped, which needs the
/// limb-wise lowering of wide integer operations (Layer 6, `L6-int-op-strategy` in
/// `docs/field-agnosticism.md`) and is deferred. A rejection is visible; a wrong answer is not.
///
/// FIELD-ASSUMPTION: L4-modulus-query. Read off the configured field rather than a fixed prime, so a
/// narrower field simply refuses more shifts instead of losing the check. This is the reason the
/// bound is checked on both paths rather than only on the wide one: on bn254 the truncating path is
/// capped at `bits <= 64` and so always has headroom, but that is a fact about this modulus, not
/// about the lowering.
fn product_headroom_or_bail(bits: usize, discarded_bits: usize, field: FieldConfig) {
    if !product_fits_field(bits, discarded_bits, field) {
        unimplemented!(
            "a witness `<<` at {bits} bits by up to {discarded_bits} needs a limb-wise lowering: the single field product `lhs * 2^n` reaches 2^{} and so wraps modulo the field, which no rangecheck on it can detect. See docs/field-agnosticism.md, L6-int-op-strategy.",
            bits + discarded_bits
        );
    }
}

/// Whether `raw * 2^n` stays below the modulus for every `raw < 2^bits` and every
/// `n <= discarded_bits`, which is what makes a rangecheck on that product meaningful.
///
/// The bound on the shift amount and the bound on the discarded half are the same number (a shift
/// by `n` pushes exactly `n` bits past the top) so this carries [`discarded_width`]'s name all the
/// way through rather than renaming it to `max_shift` at each hop.
fn product_fits_field(bits: usize, discarded_bits: usize, field: FieldConfig) -> bool {
    // The product is at most `(2^bits - 1) * 2^discarded_bits`, so `2^(bits + discarded_bits)`
    // bounds it.
    (BigInt::one() << (bits + discarded_bits)) <= field_modulus(field)
}

/// How many bits of `lhs * 2^n` can be pushed past the top, as a bound on the discarded half.
///
/// `lhs` is below `2^bits` by its own type, so the product is below `2^(bits + n)` and the discarded
/// half below `2^n`. The amount is capped at `bits - 1` regardless of what the range domain says:
/// `lower_general_shift` builds its factor from only the low `log2(bits)` bits of the amount, so the
/// _effective_ shift is always in range even when the declared range is not — and when it is not,
/// the guarded amount check is what rejects the program.
///
/// ⊥ falls back to the cap rather than measuring as a zero-bit amount. `Interval::empty` is `[1, 0]`,
/// so its `hi` is a perfectly plausible-looking `0` — and answering `0` here does not merely narrow
/// a check, it makes [`wrap_shifted_product`] skip the truncation _and both_ of its rangechecks,
/// leaving a product that the following `Cast` reinterprets for free. That the analysis believes the
/// amount unreachable is no evidence: it believes it on the strength of constraints elsewhere in
/// this same circuit. See the `proves_*` predicates on `Interval`.
fn discarded_width(amount: &Interval, bits: usize) -> usize {
    let cap = bits.saturating_sub(1);
    if amount.is_empty() {
        return cap;
    }
    match amount.hi() {
        Some(hi) => hi.to_usize().unwrap_or(cap).min(cap),
        None => cap,
    }
}

#[derive(Clone, Copy)]
struct U64Limbs {
    lo: ValueId,
    hi: ValueId,
}

#[derive(Clone, Copy)]
struct U128Limbs {
    lo: ValueId,
    hi: ValueId,
}

fn guarded_rangecheck(
    b: &mut HLBlockEmitter<'_>,
    value: ValueId,
    bits: usize,
    guard: Option<ValueId>,
) {
    assert!(bits >= 1, "rangecheck width must be at least 1 bit");
    b.emit_guarded(
        guard,
        OpCode::Rangecheck {
            value,
            max_bits: bits,
        },
    );
}

fn cast_target_for_integer_type(ty: &Type) -> CastTarget {
    match ty.strip_witness().expr {
        // A `CastTarget` is a raw-bits conversion, so there is one target per width and no sign to
        // choose. Sign extension is the separate `SExt` opcode.
        TypeExpr::Int(bits) => CastTarget::Int(bits),
        other => panic!("expected integer type, got {:?}", other),
    }
}

fn integer_bits_and_cast(
    function_type_info: &FunctionTypeInfo,
    value: ValueId,
    context: &str,
) -> (usize, CastTarget) {
    match function_type_info
        .get_value_type(value)
        .strip_witness()
        .expr
    {
        // One target per width, no sign to choose; see `cast_target_for_integer_type`.
        TypeExpr::Int(bits) => (bits, CastTarget::Int(bits)),
        other => panic!("{context}: expected integer type, got {:?}", other),
    }
}

fn spread_as_field(b: &mut impl HLEmitter, value: ValueId, bits: u8) -> ValueId {
    let spread = b.spread(value, bits);
    b.cast_to_field(spread)
}

// FIELD-ASSUMPTION: L6-int-op-strategy
// Bitwise via spread-then-add: the spread of a `bits`-wide value occupies ~2*bits bits (cast
// to `U(bits*2)`), so on a ~64-bit field even a 32-bit spread nearly saturates p. Small fields
// need narrower spread limbs (why u64/u128 are already split into 32-bit limbs).
fn lower_word_bitwise(
    b: &mut impl HLEmitter,
    kind: BinaryArithOpKind,
    lhs: ValueId,
    rhs: ValueId,
    bits: u8,
) -> ValueId {
    let lhs_spread = spread_as_field(b, lhs, bits);
    let rhs_spread = spread_as_field(b, rhs, bits);
    let input_spread_sum = b.uadd(lhs_spread, rhs_spread);
    let input_spread_sum = b.cast_to(CastTarget::Int(bits as usize * 2), input_spread_sum);
    let (and_word, xor_word) = b.unspread(input_spread_sum, bits);

    match kind {
        BinaryArithOpKind::And => and_word,
        BinaryArithOpKind::Xor => xor_word,
        BinaryArithOpKind::Or => b.uadd(and_word, xor_word),
        _ => unreachable!(),
    }
}

fn lower_u64_limb_bitwise(
    b: &mut impl HLEmitter,
    kind: BinaryArithOpKind,
    lhs: U64Limbs,
    rhs: U64Limbs,
) -> U64Limbs {
    U64Limbs {
        lo: lower_word_bitwise(b, kind, lhs.lo, rhs.lo, 32),
        hi: lower_word_bitwise(b, kind, lhs.hi, rhs.hi, 32),
    }
}

// FIELD-ASSUMPTION: L6-int-representation (combine_u32_limbs + combine_u64_fields)
// These recombine limbs into a single field cell (`lo + hi * 2^32` / `lo + hi * 2^64`). The
// 2^64 recombination exceeds p on a ~64-bit field, so wide results cannot live in one cell and
// must stay multi-cell; the shift width must derive from the field size.
fn combine_u32_limbs(b: &mut impl HLEmitter, limbs: U64Limbs) -> ValueId {
    let lo = b.cast_to_field(limbs.lo);
    let hi = b.cast_to_field(limbs.hi);
    let shift = b.field_const(b.field().constant(1u128 << 32));
    let shifted_hi = b.umul(hi, shift);
    b.uadd(lo, shifted_hi)
}

fn combine_u64_fields(b: &mut impl HLEmitter, lo: ValueId, hi: ValueId) -> ValueId {
    let lo = b.cast_to_field(lo);
    let hi = b.cast_to_field(hi);
    // FIELD-ASSUMPTION: L4-decompose
    let shift = b.field_const(b.field().two_pow(64));
    let shifted_hi = b.umul(hi, shift);
    b.uadd(lo, shifted_hi)
}

fn extract_u128_limbs(b: &mut impl HLEmitter, value: ValueId) -> U128Limbs {
    U128Limbs {
        lo: extract_u128_limb(b, value, 0),
        hi: extract_u128_limb(b, value, 64),
    }
}

fn extract_u128_limb(b: &mut impl HLEmitter, value: ValueId, offset: usize) -> ValueId {
    let limb = b.bit_range(value, offset, 64);
    b.cast_to(CastTarget::Int(64), limb)
}

fn decompose_u64_input(b: &mut impl HLEmitter, value: ValueId, is_witness: bool) -> U64Limbs {
    if !is_witness {
        return extract_u64_limbs(b, value);
    }

    let pure_value = b.value_of(value);
    let hi_hint = extract_u64_limb(b, pure_value, 32);
    let hi_field = b.cast_to_field(hi_hint);
    let hi_wit = b.write_witness(hi_field);
    let lo = derive_low_u32_limb(b, value, hi_wit);

    U64Limbs {
        lo,
        hi: b.cast_to(CastTarget::Int(32), hi_wit),
    }
}

fn extract_u64_limbs(b: &mut impl HLEmitter, value: ValueId) -> U64Limbs {
    U64Limbs {
        lo: extract_u64_limb(b, value, 0),
        hi: extract_u64_limb(b, value, 32),
    }
}

fn extract_u64_limb(b: &mut impl HLEmitter, value: ValueId, offset: usize) -> ValueId {
    let limb = b.bit_range(value, offset, 32);
    b.cast_to(CastTarget::Int(32), limb)
}

fn derive_low_u32_limb(b: &mut impl HLEmitter, value: ValueId, hi_field: ValueId) -> ValueId {
    let value_field = b.cast_to_field(value);
    let shift = b.field_const(b.field().constant(1u128 << 32));
    let shifted_hi = b.umul(hi_field, shift);
    let lo_field = b.usub(value_field, shifted_hi);
    b.cast_to(CastTarget::Int(32), lo_field)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_wide_shift_is_refused_exactly_when_its_product_can_wrap_the_field() {
        // The precondition _both_ halves of `wrap_shifted_product` depend on: once `raw * 2^n` can
        // pass the modulus, neither a rangecheck on the product nor the truncation identity can
        // tell the residue apart from an honest one. On bn254 (~2^253.5) the boundary sits at
        // `n = 125`.
        let bn254 = FieldConfig::bn254();
        assert!(product_fits_field(128, 125, bn254));
        assert!(!product_fits_field(128, 126, bn254));
        assert!(!product_fits_field(128, 127, bn254));

        // The truncating path is capped at `bits <= 64` by `2 * bits <= MAX_SUPPORTED_UNSIGNED_BITS`,
        // and on bn254 its worst case has room to spare — which is why checking it there is free
        // today, and a statement about this modulus rather than about the lowering.
        assert!(product_fits_field(64, 63, bn254));
        assert!(product_fits_field(128, 0, bn254));
    }

    #[test]
    fn bottom_does_not_shrink_the_discarded_half() {
        // ⊥ is `[1, 0]`, whose `hi` reads as a plausible zero -- and a zero here is not a narrower
        // check but _no_ check: `wrap_shifted_product` returns the raw product untruncated and the
        // following `Cast` reinterprets it for free. The analysis only believes the amount
        // unreachable because of constraints this same circuit emits, so it is not evidence.
        assert_eq!(discarded_width(&Interval::empty(), 32), 31);
        assert_eq!(discarded_width(&Interval::empty(), 8), 7);

        // A genuinely zero amount still skips the truncation -- the product is `lhs * 1`, already
        // within the width by the operand's own type.
        assert_eq!(discarded_width(&Interval::closed(0, 0), 32), 0);

        // And the ordinary cases are unchanged: the amount bounds the discarded half, capped at
        // `bits - 1` because the factor is built from the low `log2(bits)` bits regardless.
        assert_eq!(discarded_width(&Interval::closed(0, 5), 32), 5);
        assert_eq!(discarded_width(&Interval::closed(0, 200), 32), 31);
        assert_eq!(discarded_width(&Interval::top(), 32), 31);
        assert_eq!(discarded_width(&Interval::closed(0, 0), 1), 0);
    }
}
