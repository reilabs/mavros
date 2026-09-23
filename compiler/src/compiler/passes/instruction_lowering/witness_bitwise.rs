//! Lowers integer bitwise, bit-selection, and sign-extension operations before the main
//! explicit-witness pass.
//!
//! This pass emits `Spread`/`Unspread` operations. A width the spread instruction takes whole is
//! spread directly; anything wider is decomposed into half-limbs at half the field's witness limb
//! width (`shared::limbs`) — 32 bits on bn254 — with the top half-limb carrying only the bits that
//! are left, which is what lets a bitwise operation reach **any** width. It also canonicalizes
//! witness integer casts/shifts into the shared `BitRange` representation where possible.

use crate::compiler::codegen::bytecode::layout::SPREAD_MAX_BITS;
use crate::compiler::{
    analysis::{
        types::FunctionTypeInfo,
        value_range_analysis::{Interval, field_modulus},
    },
    passes::{
        instruction_lowering::{
            InstructionLoweringRule, LoweringContext, integer_bits,
            witness_integer_arith::guarded_or_zero_field,
        },
        shared::{
            limbs::{
                WitnessLimbs, combine_limbs_of_value, extract_limb, max_pow2_table_size,
                narrow_int_bits, spread_sum_fits_field, widest_injective_int_bits,
                witness_half_limb_bits,
            },
            shift_guard::shift_amount_pinned_to,
            unsupported::unsupported_on_this_field,
        },
    },
    ssa::{
        ValueId,
        hlssa::{
            ArithGroup, BinaryArithOpKind, CastTarget, CmpKind, OpCode, Type, TypeExpr,
            assert_signed_op_width,
            builder::{HLBlockEmitter, HLEmitter, two_pow_pattern},
        },
    },
};

use mavros_artifacts::FieldConfig;
use mavros_int_semantics::IntBits;
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
            // Gated on witness-ness exactly as the binary arm above is, and for the same reason:
            // outside the witness domain a complement is a value the interpreter and the compiled
            // WASM each compute with an opcode of their own, at every width. Rewriting it into
            // field arithmetic constrains a hint, and leaves those opcodes unreachable from a
            // compiled program.
            //
            // Measured rather than assumed: the gate moves **no constraint at all** — rows and
            // columns are identical on every corpus test — and shrinks bytecode by 0.30% and WASM
            // by 0.17%, `passport_08` by 7584 bytes.
            OpCode::Not { result, value }
                if function_type_info.get_value_type(*value).is_witness_of() =>
            {
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

        // The bound is the narrow threshold, not the integer type cap: what this lowering needs
        // is a value that spreads into a host word and a field cell, which is a representational
        // question and not a question about what widths the type system admits.
        // The recombination is one field element, so what bounds this is the widest value the
        // modulus carries injectively — above that a value is limbs before it reaches here, and
        // each limb is a width this lowering holds.
        let injective = widest_injective_int_bits(b.field());
        assert!(
            bits <= injective,
            "a bitwise operation recombines its limbs into one field element, so int{bits} is past what this field carries"
        );

        let lhs = b.cast_to(CastTarget::Int(bits), lhs);
        let rhs = b.cast_to(CastTarget::Int(bits), rhs);

        if bits == 1 {
            self.lower_u1_bitwise(b, kind, result, lhs, rhs);
            return;
        }

        let result_word = if bits <= SPREAD_MAX_BITS && spread_sum_fits_field(bits, b.field()) {
            // A width the spread instruction takes whole. Decomposing it would be one half-limb
            // plus a recombination, which is **measurably** worse: routing these through the
            // general path below costs +1.88% of corpus rows.
            //
            // The spread bound is a bytecode-layout constant and the sum bound is the field's, so
            // both are asked here: a field that cannot hold the sum of two spreads this wide has a
            // width the arm below reaches perfectly well, and taking this one would refuse it.
            let width = u8::try_from(bits).expect("a width the spread takes is under 256");
            lower_word_bitwise(b, kind, lhs, rhs, width)
        } else {
            // Every other width, including the ones between the arms above and the ragged top limb
            // the multi-cell representation leaves at a width the limb count does not divide. The
            // half-limbs are each at most what the spread instruction takes, so this reaches any
            // width the recombination still fits a field element at.
            let lhs_limbs = decompose_into_spread_limbs(b, lhs, bits, lhs_witness);
            let rhs_limbs = decompose_into_spread_limbs(b, rhs, bits, rhs_witness);
            let result_limbs = lower_limb_bitwise(b, kind, &lhs_limbs, &rhs_limbs);
            // The value's own width, not the limbs' nominal span: a bitwise operation cannot set a
            // bit neither operand had, so the result is bounded where the operands were.
            combine_limbs_of_value(b, &result_limbs, bits)
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
            _ => ice_unreachable!(),
        };

        b.emit(OpCode::Cast {
            result,
            value: result_field,
            target,
        });
    }

    /// `not = (2^bits - 1) - value`, for a witnessed operand.
    ///
    /// A wider one is limbs before this pass runs and each limb is complemented at its own width.
    /// That is what ensures this one field subtraction sound on **any** field.
    fn lower_not(
        &self,
        b: &mut HLBlockEmitter<'_>,
        function_type_info: &FunctionTypeInfo,
        result: ValueId,
        value: ValueId,
    ) {
        let (bits, cast_target) = integer_bits_and_cast(function_type_info, value, "bitwise not");

        let injective = widest_injective_int_bits(b.field());
        assert!(
            bits <= injective,
            "a complement subtracts from an all-ones mask in one field element, so int{bits} is past what this field carries"
        );

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
        // `MAX_LOWERED_SIGNED_BITS` for now, but the target is just a wider integer to deposit the
        // result in, and widening an `i32` into a `u128` is exactly what `x as u128` asks for.
        assert_signed_op_width(from_bits, "sign extension source");
        let narrow_bits = narrow_int_bits(b.field());
        assert!(
            from_bits < to_bits && to_bits <= narrow_bits,
            "sign extension must widen within the narrow integer range: {from_bits} -> {to_bits}"
        );

        // FIELD-ASSUMPTION: L4-modulus-query. The extension term below is
        // `two_pow(to_bits) - two_pow(from_bits)`, which is only the value it is meant to be while
        // `two_pow(to_bits)` has not wrapped. On bn254 the 128-bit cap leaves ample room; on a
        // narrower field this refuses rather than silently extending by a wrapped constant.
        if to_bits >= b.field().field_bit_size() as usize {
            unsupported_on_this_field(
                format_args!(
                    "sign extension from {from_bits} to {to_bits} bits builds its extension term out of `two_pow({to_bits})`, which has itself wrapped, so the widened value would be a residue rather than the sign-extended one"
                ),
                b.field(),
            );
        }

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
    /// An unsigned left-hand side shifted by an amount that is _known_ keeps its own lowering,
    /// which folds the amount into a constant. Known covers two cases: a pure amount, and a witness
    /// one the range domain pins to a single legal value. A signed left-hand side, or an amount
    /// that is genuinely unknown, goes to [`Self::lower_general_shift`], which pays for a runtime
    /// factor (a table lookup where there is a table, a bit decomposition otherwise).
    ///
    /// The amount check is emitted **here**, above the split, rather than inside either lowering,
    /// because both need to provide it. [`Self::lower_general_shift`] needed it to bound the
    /// decomposition it built, which made it easy to read as part of that lowering;
    /// [`Self::lower_constant_amount_shift`] hands the raw amount to a backend shift instead, so
    /// its need for the check is just as real and far less visible.
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
            .unwrap_or_else(|| ice!("witness shift on non-integer lhs type {lhs_type:?}"));
        // The shift's own sign decides which lowering runs.
        let lhs_signed = kind.is_signed();
        let rhs_witness = context.types().get_value_type(rhs).is_witness_of();

        // Everything below packs the value, its `2^n` factor and their product into single field
        // elements, so a value too wide to live in one is refused.
        let narrow_bits = narrow_int_bits(b.field());
        if bits > narrow_bits {
            unsupported_on_this_field(
                format_args!(
                    "a witness shift of a {bits}-bit value, which is wider than the {narrow_bits} \
                     bits one field element and one host word can carry"
                ),
                b.field(),
            );
        }

        // The check below indexes bit `log2(bits)` upwards as "too large", and `emit_pow2_factor`'s
        // table is keyed by `log2(bits)` so that membership is the amount bound. Both are only
        // right when `bits` is a power of two.
        //
        // This is the guard IR's own requirement: the _total_ evaluators (the VM, LLVM and the
        // model) reduce an amount modulo the width and so agree at every width. Admitting a non
        // power-of-two shift means rebuilding the check as a real `amount < bits` comparison and
        // splitting the `2^n` factor, not deleting this assert.
        assert!(
            bits.is_power_of_two(),
            "the shift-amount check assumes a power-of-two integer width, got {bits}"
        );
        if lhs_signed {
            assert_signed_op_width(bits, "shift");
        }

        let widths = shift_amount_bits(context, rhs, bits);

        // A witness amount the range domain pins to one legal value is lowered as the constant it
        // provably is. This is where the domain's transparency to `WitnessOf` finally buys
        // something: a value SCS proved constant keeps its witness type all the way here, so the
        // type says "runtime amount" long after the analysis stopped believing it.
        let pinned = if rhs_witness {
            shift_amount_pinned_to(&context.range(rhs), bits)
        } else {
            None
        };

        // [`Self::lower_constant_amount_shift`] is unsigned-only by construction, so a signed
        // left-hand side keeps the general lowering however well known its amount is.
        let constant_amount = !lhs_signed && (!rhs_witness || pinned.is_some());

        // A witness amount narrow enough to have a table takes the lookup, and the lookup _is_ the
        // bound: its keys are exactly the legal amounts, so membership rejects an amount at or past
        // the width, and a negative one too. Every other route still performs the explicit check —
        // where a pinned amount discharges it for free, since the range that pinned it also proves
        // it in range.
        let use_pow2_table =
            !constant_amount && rhs_witness && widths.amount_bits <= max_pow2_table_size(b.field());
        if !use_pow2_table {
            emit_shift_amount_check(b, context, guard, rhs, widths);
        }

        if constant_amount {
            // The literal stands in for `rhs` only where the factor is built. Everything that reads
            // the amount to _reason_ about it keeps the original value, which is the one the range
            // domain has an entry for.
            let amount = match pinned {
                Some(v) => b.int_const(IntBits::from_u128(widths.rhs_bits, v)),
                None => rhs,
            };
            self.lower_constant_amount_shift(
                b, context, guard, kind, result, lhs, rhs, amount, bits,
            );
        } else {
            self.lower_general_shift(
                b,
                context,
                guard,
                kind,
                result,
                lhs,
                rhs,
                bits,
                lhs_signed,
                widths,
                use_pow2_table,
            );
        }
    }

    /// The pre-existing lowering: an unsigned left-hand side shifted by an amount that folds.
    ///
    /// The `1 << amount` is a _pure_ `Shl`, which constant-folds later. `Shl` on a `U(bits)`
    /// reaches `hlssa_to_r1cs` only with both operands constant, so this shape is viable precisely
    /// as far as the amount folds.
    ///
    /// `amount` and `rhs` are the same value except where the caller has replaced a pinned witness
    /// amount with the literal it provably equals. The two are kept apart because only the factor
    /// wants the literal: [`wrap_shifted_product`] reads the amount to size a range check, and the
    /// range domain has a record for the operand as written, not for a value minted after it ran.
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
        amount: ValueId,
        bits: usize,
    ) {
        let one_u = b.int_const(IntBits::one(bits));
        let factor = b.fresh_value();
        b.emit(OpCode::BinaryArithOp {
            kind: BinaryArithOpKind::UShl,
            result: factor,
            lhs: one_u,
            rhs: amount,
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
            _ => ice_unreachable!("lower_shift only dispatches Shl and Shr"),
        }
    }

    /// Lowers a shift whose amount is not a compile-time constant, whose left-hand side is signed,
    /// or both.
    ///
    /// `2^amount` cannot be built by shifting, because nothing below HLSSA can shift by a variable.
    /// A witness amount reads it out of the powers-of-two table in one lookup, which also supplies
    /// the rejection. Otherwise the amount is decomposed into bits and the factor rebuilt as a
    /// product of per-bit linear terms, and the caller has already planted the bound as an explicit
    /// check.
    ///
    /// "Otherwise" is narrower than it sounds: since the table covers every width, the only amount
    /// that reaches the decomposition is a **pure** one, which arrives here when the left-hand side
    /// is signed. Every bit of that decomposition then constant-folds, so it costs nothing at
    /// runtime — the per-bit product is the shape, not the price. It is not dead code, but no
    /// witness amount can take it.
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
        use_pow2_table: bool,
    ) {
        let ShiftAmountWidths {
            rhs_bits,
            amount_bits,
        } = widths;

        // The route that builds the factor is also the route that decides where a cofactor could
        // come from, so the two are chosen together and travel as one value. Only the signed `>>`
        // correction ever asks for the cofactor.
        let (factor, cofactor) = if use_pow2_table {
            (
                emit_pow2_factor(b, guard, rhs, bits, amount_bits),
                CofactorSource::Table,
            )
        } else {
            let amount = extract_amount_bits(b, rhs, rhs_bits, amount_bits);
            (build_shift_factor(b, &amount), CofactorSource::Bits(amount))
        };

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
                &cofactor,
                amount_bits,
                bits,
            ),
            _ => ice_unreachable!("lower_shift only dispatches Shl and Shr"),
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
        cofactor: &CofactorSource,
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
                let cofactor = match cofactor {
                    CofactorSource::Table => emit_pow2_cofactor(b, factor, bits),
                    CofactorSource::Bits(amount) => build_shift_cofactor(b, amount, amount_bits),
                };

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

/// Where the signed `>>` correction's `2^(bits - n)` comes from.
///
/// Not a free choice at the use site: it is fixed by whichever route built the factor, so the two
/// are produced together in [`LowerWitnessBitwiseOps::lower_general_shift`] and travel as one
/// value. Carrying the decomposition in the variant that needs it is what keeps a bit list and a
/// "use the table" flag from disagreeing.
enum CofactorSource {
    /// The cofactor is pinned algebraically against the table-supplied factor by
    /// [`emit_pow2_cofactor`], which never needs the amount's bits.
    Table,

    /// The cofactor is a second product over the same bits the factor was built from, by
    /// [`build_shift_cofactor`].
    Bits(Vec<ValueId>),
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
        .unwrap_or_else(|| ice!("witness shift by a non-integer amount type {rhs_type:?}"));
    ShiftAmountWidths {
        rhs_bits,
        amount_bits: bits.trailing_zeros() as usize,
    }
}

/// Asserts that the shift amount is smaller than the width being shifted.
///
/// Since the width is a power of two, "too large" is just "some bit at or above `log2(bits)` is
/// set" — and wherever that test is emitted it also catches a _negative_ amount, whose raw
/// encoding always has its top bit set. The second reading needs `2^(rhs_bits-1) >= bits`, i.e.
/// `rhs_bits > amount_bits`, which is precisely the condition under which the body emits anything
/// at all; the branch that skips the check is the branch where a negative amount must instead be
/// unrepresentable.
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

    // No bit that high exists, so every amount this type can hold is in range. That also drops the
    // negative-amount rejection, so it may only ever fire where a negative amount cannot be
    // represented either.
    //
    // Asserted rather than `debug_assert`ed: this is the whole justification for emitting no check,
    // so a release build must not be the one that skips it.
    if rhs_bits <= amount_bits {
        assert!(
            rhs_bits <= 1,
            "every value a {rhs_bits}-bit shift amount can hold is already below the width, so no \
             range check is emitted — but a {rhs_bits}-bit amount also has a negative reading, \
             which would then go unrejected and read as a small positive one"
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

/// `2^amount` for a witness amount, read out of the powers-of-two table.
///
/// The lookup is emitted **unguarded**, with the amount neutralized to zero on an inactive path
/// instead. Gating the lookup itself would be wrong: a vacuous row leaves `factor` unconstrained,
/// and [`wrap_shifted_product`] depends on the factor being at most `2^(bits - 1)` _on every path_
/// to keep its two range checks satisfiable.
///
/// Neutralizing the amount keeps the row live, so the factor is pinned to `1` where the guard is
/// off, and an inactive out-of-range shift is vacuous rather than a failure.
fn emit_pow2_factor(
    b: &mut HLBlockEmitter<'_>,
    guard: Option<ValueId>,
    rhs: ValueId,
    bits: usize,
    amount_bits: usize,
) -> ValueId {
    let rhs_field = b.cast_to_field(rhs);
    let amount = guarded_or_zero_field(b, rhs_field, guard);

    // The hint. An out-of-range amount masks here exactly as the backends' shifts do, which is
    // harmless: the lookup below rejects that amount whatever this computed.
    let amount_pure = b.value_of(amount);
    let amount_int = b.cast_to(CastTarget::Int(bits), amount_pure);
    let one = b.int_const(IntBits::one(bits));
    let factor_int = b.fresh_value();
    b.emit(OpCode::BinaryArithOp {
        kind: BinaryArithOpKind::UShl,
        result: factor_int,
        lhs: one,
        rhs: amount_int,
    });
    let factor_hint = b.cast_to_field(factor_int);
    let factor = b.write_witness(factor_hint);

    let one_flag = b.field_const(b.field().one());
    b.lookup_pow2(amount_bits as u8, amount, factor, one_flag);

    factor
}

/// `2^bits / 2^n`, pinned by a single multiplication against the table-supplied factor.
///
/// `factor * cofactor == 2^bits` determines `cofactor` uniquely.
fn emit_pow2_cofactor(b: &mut HLBlockEmitter<'_>, factor: ValueId, bits: usize) -> ValueId {
    // Only the signed `>>` correction wants a cofactor, and `assert_signed_op_width` caps a signed
    // operand at 64 bits, so the double-width hint below stays inside the widest unsigned type
    // there is.
    assert!(
        2 * bits <= narrow_int_bits(b.field()),
        "a {bits}-bit shift cofactor needs an Int({}) this lowering cannot mint",
        2 * bits
    );

    // FIELD-ASSUMPTION: L4-decompose
    let two_pow_bits = b.field_const(b.field().two_pow(bits));

    // The hint is an exact integer division, computed at double width because `2^bits` itself
    // does not fit the shifted width -- an amount of zero makes the cofactor `2^bits`.
    let wide_bits = 2 * bits;
    let factor_pure = b.value_of(factor);
    let factor_wide = b.cast_to(CastTarget::Int(wide_bits), factor_pure);
    let two_pow_bits_wide = b.int_const(two_pow_pattern(wide_bits, bits));
    let cofactor_int = b.fresh_value();
    b.emit(OpCode::BinaryArithOp {
        kind: BinaryArithOpKind::UDiv,
        result: cofactor_int,
        lhs: two_pow_bits_wide,
        rhs: factor_wide,
    });
    let cofactor_hint = b.cast_to_field(cofactor_int);
    let cofactor = b.write_witness(cofactor_hint);

    b.constrain(factor, cofactor, two_pow_bits);

    cofactor
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
/// falls back to a trapping rangecheck, which rejects a shift Noir would have wrapped. Correcting
/// _that_ needs a limb-wise lowering rather than a single field product.
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

    // This fallback is deliberately **not** an `unsupported_on_this_field` site. Its effect is a
    // trapping rangecheck rather than a refusal: a shift that should have wrapped is rejected by
    // the circuit at proving time, and every shift that does not overflow still lowers. The funnel
    // would trade that for a compile-time refusal of every witness `<<` at the width, including the
    // overwhelming majority that never overflow.
    //
    // Its condition _is_ field-sensitive — the threshold is derived, so a narrower field lowers it
    // and this fires at more widths — which strengthens the case rather than weakening it: the
    // narrower the field, the more programs a funnel refusal would reject outright.
    //
    // **This branch is live at an existing width and must stay where it is.** At `bits == 128` it
    // reads `256 > 128` and takes the trapping path. Against the integer type cap it would read
    // `256 <= 16384` instead and fall through to the truncating path below, minting an `Int(256)`
    // intermediate and silently changing the circuit for a width the corpus already compiles.
    //
    // TODO Remove once an `Int(2 * bits)` intermediate is expressible — at `bits == 128` that is
    // an `Int(256)`, and with it the rejection below becomes an honest wrapping shift. What blocks
    // it is measured rather than assumed: forcing the truncating path here fails in
    // `bit_range::lower_pure_bit_range_value`, whose `bits <= narrow_int_bits` assert is there
    // because `window_mask` returns a host word and the divisor beside it is `1u128 << offset`.
    // Both are width-generic constants minted through a host word, so both are expressible as
    // patterns.
    //
    // This is the narrower of the two limits at this width and the only one a wider intermediate
    // reaches. The other is `product_headroom_or_bail`: past `n >= 126` on this field the product
    // itself wraps, leaving no honest value to truncate, and no amount of intermediate width helps.
    let wide_bits = 2 * bits;
    if wide_bits > narrow_int_bits(b.field()) {
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
    // an inactive path. `product` is therefore below `2^(bits + discarded_bits)` unconditionally.
    // The trapping fallback above is the one lowering that does _not_ bound its result this way.
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
        unsupported_on_this_field(
            format_args!(
                "a witness `<<` at {bits} bits by up to {discarded_bits} needs a limb-wise lowering: the single field product `lhs * 2^n` reaches 2^{} and so wraps modulo the field, which no rangecheck on it can detect",
                bits + discarded_bits
            ),
            field,
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
/// `lhs` is below `2^bits` by its own type, so the product is below `2^(bits + n)` and the
/// discarded half below `2^n`. The amount is capped at `bits - 1` regardless of what the range
/// domain says, and every route that builds a factor holds that cap by a distinct mechanism:
///
/// - The **table** route reads `2^n` out of a table whose only rows are the amounts `0..bits`, so
///   an amount at or past the width has no row and the program is rejected. It is never masked.
/// - The **decomposition** route builds its factor from only the low `log2(bits)` bits of the
///   amount, so the _effective_ shift is in range even when the declared range is not. When it is
///   not, the guarded amount check hoisted above the lowering rejects the program.
/// - The **constant-amount** route folds `1 << amount` at an amount the same check has already
///   proved in range.
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
        other => ice!("expected integer type, got {:?}", other),
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
        other => ice!("{context}: expected integer type, got {:?}", other),
    }
}

fn spread_as_field(b: &mut impl HLEmitter, value: ValueId, bits: u8) -> ValueId {
    let spread = b.spread(value, bits);
    b.cast_to_field(spread)
}

/// Bitwise on a whole `bits`-wide word, via spread-then-add.
///
/// The two callers reach here by different routes and only one of them has already been sized by
/// the field: `lower_limb_bitwise` arrives at [`witness_half_limb_bits`], while the direct arm of
/// [`LowerWitnessBitwiseOps::lower_binary_bitwise`] arrives at the operand's own **type** width,
/// which that arm caps at [`SPREAD_MAX_BITS`].
// FIELD-ASSUMPTION: L6-int-op-strategy
// Bitwise via spread-then-add: the spread of a `bits`-wide value occupies ~2*bits bits (cast
// to `U(bits*2)`), so on a ~64-bit field even a 32-bit spread nearly saturates p. A half-limb
// operand satisfies `2*bits <= h` by construction, which is why every width past the whole-width
// arm goes through a half-limb decomposition. The whole-width arm asks this same predicate before
// taking itself, so what the refusal below covers is a field whose own half-limb two spreads of do
// not fit — which no decomposition below it can step around.
fn lower_word_bitwise(
    b: &mut impl HLEmitter,
    kind: BinaryArithOpKind,
    lhs: ValueId,
    rhs: ValueId,
    bits: u8,
) -> ValueId {
    if !spread_sum_fits_field(bits as usize, b.field()) {
        unsupported_on_this_field(
            format_args!(
                "a {bits}-bit bitwise op spreads each operand to {} bits, and the sum of the two spreads no longer fits one field element, so `Unspread` would read a residue rather than the interleaved bits",
                2 * bits as usize
            ),
            b.field(),
        );
    }

    let lhs_spread = spread_as_field(b, lhs, bits);
    let rhs_spread = spread_as_field(b, rhs, bits);
    let input_spread_sum = b.uadd(lhs_spread, rhs_spread);
    let input_spread_sum = b.cast_to(CastTarget::Int(bits as usize * 2), input_spread_sum);
    let (and_word, xor_word) = b.unspread(input_spread_sum, bits);

    match kind {
        BinaryArithOpKind::And => and_word,
        BinaryArithOpKind::Xor => xor_word,
        BinaryArithOpKind::Or => b.uadd(and_word, xor_word),
        _ => ice_unreachable!(),
    }
}

// SPREAD LIMBS
// ================================================================================================

/// A decomposition into limbs the spread instruction takes whole, each with the width it carries.
///
/// [`WitnessLimbs`] is uniform-width by contract The **spread** wants the other reading as its cost
/// falls as the width does (`LookupSizing::decompose_spread`), so a ragged top limb spread at its
/// own four or thirty bits is cheaper than the same limb padded to a full half-limb first.
struct SpreadLimbs {
    /// The place-value stride, which is the half-limb whatever the top limb carries.
    limb_bits: usize,

    /// The limbs, least significant first.
    limbs: Vec<ValueId>,

    /// The bits each limb carries, equal to `limb_bits` for all but a ragged top.
    widths: Vec<usize>,
}

/// Bitwise limb by limb, at the width each limb was decomposed to.
///
/// The result is uniform-width as a place-value sum reads it: limb `i` of the answer is bounded by
/// limb `i` of the operands.
fn lower_limb_bitwise(
    b: &mut impl HLEmitter,
    kind: BinaryArithOpKind,
    lhs: &SpreadLimbs,
    rhs: &SpreadLimbs,
) -> WitnessLimbs {
    assert_eq!(
        lhs.limb_bits, rhs.limb_bits,
        "bitwise operands were decomposed at different limb widths"
    );
    assert_eq!(
        lhs.limbs.len(),
        rhs.limbs.len(),
        "bitwise operands were decomposed into different limb counts"
    );
    assert_eq!(
        lhs.widths, rhs.widths,
        "bitwise operands were decomposed with differently ragged top limbs"
    );
    let limb_bits = lhs.limb_bits;
    let limbs = lhs
        .limbs
        .iter()
        .zip(&rhs.limbs)
        .zip(&lhs.widths)
        .map(|((&lhs_limb, &rhs_limb), &limb_width)| {
            let width = u8::try_from(limb_width).expect("a limb is at most one host word wide");
            lower_word_bitwise(b, kind, lhs_limb, rhs_limb, width)
        })
        .collect();
    WitnessLimbs { limb_bits, limbs }
}

/// Split `value` into half-limbs at every width, the top one carrying only the bits that are left.
///
/// The only decomposition a bitwise operation needs. Every limb is at most a half-limb wide, which
/// is what the spread instruction bounds, so what this reaches is every width past the one arm
/// above it.
///
/// The high half-limbs are witnessed from the pure shadow and the lowest is **derived** as
/// `value - sum(higher limbs at their places)`, so the reconstruction holds by construction and
/// needs no constraint of its own. Otherwise, a dishonest prover could inflate a high limb and let
/// the derived one absorb it, but the range check on **every** limb at **its own** width is what
/// stops that: the top limb is bounded at the bits actually left over, not at a full half-limb, so
/// the limbs cannot between them carry a value the operand's width does not.
fn decompose_into_spread_limbs(
    b: &mut impl HLEmitter,
    value: ValueId,
    value_bits: usize,
    is_witness: bool,
) -> SpreadLimbs {
    let half_bits = witness_half_limb_bits(b.field());
    let count = value_bits.div_ceil(half_bits);
    let width_of = |index: usize| (value_bits - index * half_bits).min(half_bits);
    let widths: Vec<usize> = (0..count).map(width_of).collect();

    if !is_witness {
        // A hint, so the limbs are read straight out of it and bounded by where they came from.
        let limbs = (0..count)
            .map(|index| extract_limb(b, value, index * half_bits, width_of(index)))
            .collect();
        return SpreadLimbs {
            limb_bits: half_bits,
            limbs,
            widths,
        };
    }

    let pure_value = b.value_of(value);
    let mut high = Vec::with_capacity(count - 1);
    for index in 1..count {
        let hint = extract_limb(b, pure_value, index * half_bits, width_of(index));
        let hint_field = b.cast_to_field(hint);
        let written = b.write_witness(hint_field);
        // At its **own** width: a full half-limb here would let the top limb carry bits the
        // operand's width does not have, and the derived limb below would absorb the difference.
        let bounded = b.cast_to(CastTarget::Int(width_of(index)), written);
        high.push(bounded);
    }

    // The lowest limb is what is left once the others are taken out at their place values, so the
    // reconstruction is an identity rather than a constraint.
    let value_field = b.cast_to_field(value);
    let mut remainder = value_field;
    for (index, &limb) in high.iter().enumerate() {
        let place = b.field_const(b.field().two_pow((index + 1) * half_bits));
        let limb_field = b.cast_to_field(limb);
        let shifted = b.umul(limb_field, place);
        remainder = b.usub(remainder, shifted);
    }
    let low = b.cast_to(CastTarget::Int(width_of(0)), remainder);

    let mut limbs = Vec::with_capacity(count);
    limbs.push(low);
    limbs.extend(high);
    SpreadLimbs {
        limb_bits: half_bits,
        limbs,
        widths,
    }
}

// TESTS
// ================================================================================================

#[cfg(test)]
mod tests {
    use super::*;

    use crate::compiler::ssa::hlssa::{HLSSA, MAX_POW2_TABLE_SIZE, builder::HLSSABuilder};

    /// The two routes into [`lower_word_bitwise`], both cleared on bn254.
    #[test]
    fn no_width_this_lowering_spreads_at_can_wrap_bn254() {
        let bn254 = FieldConfig::bn254();

        // The half-limb route, which is every width past the direct arm.
        assert!(spread_sum_fits_field(witness_half_limb_bits(bn254), bn254));

        // And the whole-width route. The dispatch asks this predicate before taking that arm, so
        // what the loop pins is that on bn254 it never diverts: every width up to the cap goes
        // whole. It starts at 2 because a single bit takes the arm above it.
        for bits in 2..=SPREAD_MAX_BITS {
            assert!(spread_sum_fits_field(bits, bn254), "{bits} bits");
        }

        // The half-limb is itself a width the spread instruction takes, which is what lets the
        // general arm reach any width at all: a wider one would meet the `todo!` in bytecode
        // codegen rather than the predicate above. It holds because a limb width is a power of two
        // capped at one host word, so its half is at most half a host word.
        assert!(witness_half_limb_bits(bn254) <= SPREAD_MAX_BITS);
    }

    /// A ragged decomposition bounds its top limb at **its own** width, not a full half-limb.
    ///
    /// This is the whole soundness argument of the general bitwise path, and **no honest witness
    /// can see it**: the limbs of a real value are inside their true widths either way, so widening
    /// the top bound leaves every end-to-end test in this repository green. What it would allow is
    /// a prover inflating the top limb and letting the derived low limb absorb the difference —
    /// the decomposition would then represent a value the operand's width cannot hold.
    ///
    /// So it is checked by what the lowering emits. 100 bits over 32-bit half-limbs is
    /// `32 + 32 + 32 + 4`, and the `4` is the assertion: under a full-width bound there is no cast
    /// to `int4` anywhere in the output.
    #[test]
    fn a_ragged_decomposition_bounds_its_top_limb_at_its_own_width() {
        let bits = 100usize;
        let mut ssa = HLSSA::with_main("main".to_string());
        let main_id = ssa.get_unique_entrypoint_id();
        let value = ssa.fresh_value();
        {
            let mut sb = HLSSABuilder::new(&mut ssa);
            sb.modify_function(main_id, |b| {
                let entry = b.function.get_entry_id();
                b.function
                    .get_block_mut(entry)
                    .push_parameter(value, Type::witness_of(Type::int(bits)));
                let mut e = b.test_block(entry);
                let limbs = decompose_into_spread_limbs(&mut e, value, bits, true);
                assert_eq!(limbs.limbs.len(), bits.div_ceil(32));
                e.terminate_return(vec![]);
            });
        }

        // The bound is the cast applied to each **witness column**, which has to be read by
        // following the column rather than by looking for a width: the hint extraction casts to the
        // same widths, so a test that only asked whether some `int4` cast exists passes under the
        // very ablation it is written for.
        let function = ssa.get_unique_entrypoint();
        let ops: Vec<&OpCode> = function
            .get_block(function.get_entry_id())
            .get_instructions()
            .collect();
        let columns: Vec<ValueId> = ops
            .iter()
            .filter_map(|op| match op {
                OpCode::WriteWitness {
                    result: Some(result),
                    ..
                } => Some(*result),
                _ => None,
            })
            .collect();
        let mut bounds: Vec<usize> = columns
            .iter()
            .map(|column| {
                ops.iter()
                    .find_map(|op| match op {
                        OpCode::Cast {
                            value,
                            target: CastTarget::Int(width),
                            ..
                        } if value == column => Some(*width),
                        _ => None,
                    })
                    .unwrap_or_else(|| panic!("witness column {column:?} is never bounded"))
            })
            .collect();
        bounds.sort_unstable();

        // `100 = 32 + 32 + 32 + 4`, and the lowest limb is derived rather than witnessed.
        assert_eq!(
            bounds,
            vec![4, 32, 32],
            "every witnessed limb is bounded at its own width, the top one included"
        );
    }

    /// A decomposition of `bits` at `limb_bits`, every limb full, built the way a test wants one.
    fn uniform_limbs(
        e: &mut HLBlockEmitter<'_>,
        value: ValueId,
        limb_bits: usize,
        count: usize,
    ) -> SpreadLimbs {
        SpreadLimbs {
            limb_bits,
            limbs: (0..count)
                .map(|index| extract_limb(e, value, index * limb_bits, limb_bits))
                .collect(),
            widths: vec![limb_bits; count],
        }
    }

    /// `lower_limb_bitwise` lowers _every_ limb, not the first two.
    ///
    /// Its one caller reaches three limbs and more from `int96` upward, but the shortest widths it
    /// dispatches are a pair, and a zip that silently truncated a mismatched decomposition would
    /// still pass every one of those. The length past the pair is pinned here, with the check that
    /// refuses a mismatch rather than truncating it in the test below.
    #[test]
    fn a_bitwise_decomposition_is_lowered_limb_by_limb_at_any_length() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let main_id = ssa.get_unique_entrypoint_id();
        {
            let mut sb = HLSSABuilder::new(&mut ssa);
            sb.modify_function(main_id, |b| {
                let entry = b.function.get_entry_id();
                let mut e = b.test_block(entry);
                let lhs = e.int_const(IntBits::from_u128(24, 0x00_AB_CD));
                let rhs = e.int_const(IntBits::from_u128(24, 0x00_12_34));
                let lhs_limbs = uniform_limbs(&mut e, lhs, 8, 3);
                let rhs_limbs = uniform_limbs(&mut e, rhs, 8, 3);
                let result =
                    lower_limb_bitwise(&mut e, BinaryArithOpKind::And, &lhs_limbs, &rhs_limbs);
                assert_eq!(result.limb_bits, 8);
                assert_eq!(result.limbs.len(), 3, "one result limb per operand limb");
                e.terminate_return(vec![]);
            });
        }

        // One spread per operand limb and one unspread per result limb: six and three, not the four
        // and two a pair-shaped lowering would have emitted.
        let function = ssa.get_unique_entrypoint();
        let ops: Vec<&OpCode> = function
            .get_block(function.get_entry_id())
            .get_instructions()
            .collect();
        let spreads = ops
            .iter()
            .filter(|op| matches!(op, OpCode::Spread { bits: 8, .. }))
            .count();
        let unspreads = ops
            .iter()
            .filter(|op| matches!(op, OpCode::Unspread { bits: 8, .. }))
            .count();
        assert_eq!((spreads, unspreads), (6, 3));
    }

    #[test]
    #[should_panic(expected = "decomposed into different limb counts")]
    fn a_bitwise_pair_of_unequal_length_is_refused_rather_than_truncated() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let main_id = ssa.get_unique_entrypoint_id();
        let mut sb = HLSSABuilder::new(&mut ssa);
        sb.modify_function(main_id, |b| {
            let entry = b.function.get_entry_id();
            let mut e = b.test_block(entry);
            let lhs = e.int_const(IntBits::zero(24));
            let rhs = e.int_const(IntBits::zero(16));
            let lhs_limbs = uniform_limbs(&mut e, lhs, 8, 3);
            let rhs_limbs = uniform_limbs(&mut e, rhs, 8, 2);
            lower_limb_bitwise(&mut e, BinaryArithOpKind::And, &lhs_limbs, &rhs_limbs);
        });
    }

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

        // The truncating path is capped at `bits <= 64` by `2 * bits <= narrow_int_bits(field)`,
        // and on bn254 its worst case has room to spare — which is why checking it there is free
        // today, and a statement about this modulus rather than about the lowering.
        assert!(product_fits_field(64, 63, bn254));
        assert!(product_fits_field(128, 0, bn254));
    }

    #[test]
    fn every_table_backed_width_has_exactly_as_many_rows_as_amounts() {
        // The table is keyed by `log2(bits)` so that its row count is `1 << size`, the convention
        // every other width-keyed table follows. That works only because the legal amounts are
        // `0..bits` and `bits` is a power of two: `lower_shift` asserts the latter. If the
        // two ever drift, membership stops being the amount bound and the lowering silently accepts
        // or rejects the wrong amounts.
        for bits in [8usize, 16, 32, 64, 128] {
            let size = bits.trailing_zeros() as usize;
            assert!(size <= MAX_POW2_TABLE_SIZE, "{bits}-bit shift has no table");
            assert_eq!(
                1usize << size,
                bits,
                "{bits}-bit shift: rows must be amounts"
            );
        }

        // Every width Noir can name is covered, and the ceiling sits exactly at the widest of them
        // rather than above it. There is deliberately no headroom: the bound is the _field's_, not
        // the host's — row `n` carries the value `2^n`, so a size-`s` table's widest row is
        // `2^(2^s - 1)`, and one size further would put that row past the bn254 modulus, where
        // every evaluator wraps identically and the table stops holding powers of two at all.
        assert!(128usize.trailing_zeros() as usize <= MAX_POW2_TABLE_SIZE);
        assert_eq!(
            1usize << MAX_POW2_TABLE_SIZE,
            narrow_int_bits(FieldConfig::bn254())
        );

        // The bound stated as the field question it is, at the ceiling and one past it.
        let modulus = field_modulus(FieldConfig::bn254());
        let widest_row = |size: usize| BigInt::one() << ((1usize << size) - 1);
        assert!(widest_row(MAX_POW2_TABLE_SIZE) < modulus);
        assert!(widest_row(MAX_POW2_TABLE_SIZE + 1) > modulus);
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
