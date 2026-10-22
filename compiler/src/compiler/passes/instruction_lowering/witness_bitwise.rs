//! Lowers integer bitwise, bit-selection and sign-extension operations before the main
//! explicit-witness pass.
//!
//! Bitwise operations become `Spread`/`Unspread`. A width the spread opcode takes whole is spread
//! directly; a wider one is cut into half-limbs (half the witness limb, `shared::limbs`; 32 bits on
//! bn254), the top one ragged, so any width works. Witness integer casts and shifts are
//! canonicalized into `BitRange` where possible.

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
                WitnessLimbs, ceil_log2, combine_limbs_of_value, extract_limb, max_pow2_table_size,
                narrow_int_bits, single_cell_shift_fits, spread_sum_fits_field,
                widest_injective_int_bits, witness_half_limb_bits,
            },
            shift_guard::{
                emit_pure_shift_amount_check, shift_amount_pinned_to,
                shift_amount_provably_in_range,
            },
            unsupported::unsupported_on_this_field,
        },
    },
    ssa::{
        ValueId,
        hlssa::{
            ArithGroup, BinaryArithOpKind, CastTarget, OpCode, Type, TypeExpr,
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
            // Gated on witness-ness like the binary arm above, for the same reason: outside the
            // witness domain the interpreter and compiled WASM each compute a complement with a
            // dedicated opcode at every width. Rewriting it into field arithmetic would constrain a
            // hint and leave those opcodes unreachable from a compiled program.
            //
            // Measured: the gate moves **no constraint** (rows and columns are identical on every
            // corpus test) and shrinks bytecode by 0.30% and WASM by 0.17%, `passport_08` by 7584
            // bytes.
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

        // The recombination is one field element, so the bound is the widest injective width, not
        // the type cap. A wider value is limbs before it reaches here.
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
            lower_word_bitwise(b, kind, lhs, rhs, bits)
        } else {
            let lhs_limbs = decompose_into_spread_limbs(b, lhs, bits, lhs_witness);
            let rhs_limbs = decompose_into_spread_limbs(b, rhs, bits, rhs_witness);
            let result_limbs = lower_limb_bitwise(b, kind, &lhs_limbs, &rhs_limbs);

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
    /// A wider operand is already limbs when this pass runs, and each limb is complemented at its
    /// width, which keeps this single field subtraction sound on **any** field.
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

    /// Sign-extends via `value + sign * (2^to_bits - 2^from_bits)` in one field element.
    ///
    /// Sound while `2^to_bits <= p`. A wider witnessed target is limbs, which `WideWitnessInts`
    /// sign-fills before this pass runs.
    fn lower_integer_sext(
        &self,
        b: &mut HLBlockEmitter<'_>,
        context: &LoweringContext<'_>,
        result: ValueId,
        value: ValueId,
        from_bits: usize,
        to_bits: usize,
    ) {
        let narrow_bits = narrow_int_bits(b.field());
        let witnessed = context.types().get_value_type(value).is_witness_of();
        if !witnessed && to_bits > narrow_bits {
            self.lower_wide_pure_sext(b, context, result, value, from_bits, to_bits);
            return;
        }

        // Wider pure targets took the integer route above and wider witnessed ones are limbs, so
        // `2^to_bits - 2^from_bits` does not wrap.
        let widest = if witnessed {
            widest_injective_int_bits(b.field())
        } else {
            narrow_bits
        };
        assert!(
            from_bits < to_bits && to_bits <= widest,
            "ICE: a sign extension from {from_bits} to {to_bits} bits reached the single-element lowering"
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

    /// Sign-extends a pure value to a target past the narrow range, where the field formula's
    /// place value does not fit one element.
    fn lower_wide_pure_sext(
        &self,
        b: &mut HLBlockEmitter<'_>,
        context: &LoweringContext<'_>,
        result: ValueId,
        value: ValueId,
        from_bits: usize,
        to_bits: usize,
    ) {
        assert!(
            from_bits < to_bits,
            "sign extension must widen: {from_bits} -> {to_bits}"
        );
        // As in the field formula, a sign bit the range record proves clear costs nothing: the
        // extension is then the widening alone.
        if context.range(value).is_non_negative_at_width(from_bits) {
            b.emit(OpCode::Cast {
                result,
                value,
                target: CastTarget::Int(to_bits),
            });
            return;
        }

        let widened = b.cast_to(CastTarget::Int(to_bits), value);
        let gap = b.int_const(IntBits::from_u128(to_bits, (to_bits - from_bits) as u128));
        let raised = b.bin(BinaryArithOpKind::UShl, widened, gap);
        b.emit(OpCode::BinaryArithOp {
            kind: BinaryArithOpKind::SShr,
            result,
            lhs: raised,
            rhs: gap,
        });
    }

    /// Lowers a shift with at least one witness operand.
    ///
    /// An unsigned left-hand side shifted by a _known_ amount (pure, or a witness the range domain
    /// pins to one legal value) goes to [`Self::lower_constant_amount_shift`]; anything else to
    /// [`Self::lower_general_shift`].
    ///
    /// The amount check is emitted here, above the split, because both need it: the constant route
    /// hands the raw amount to a backend shift.
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
        // elements. `WideWitnessInts` runs first and lowers every shift those elements cannot hold
        // limb-wise, so a shift reaching here fits by construction. A signed right shift's
        // correction adds a multiple of `2^bits`, which the same bound keeps inside the modulus.
        assert!(
            single_cell_shift_fits(b.field(), bits, kind.group() == ArithGroup::Shl),
            "ICE: an int{bits} witness {kind:?} reached the single-cell shift, whose field element \
             cannot hold it"
        );

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

        // A witness amount narrow enough to have a table takes the lookup, and at a power-of-two
        // width the lookup _is_ the bound.
        //
        // A pure amount builds its factor on the pure side instead, which reduces the amount modulo
        // the width as the backends do, so the factor is a power of two below `2^bits` on every
        // path, a guard's included, and needs no table.
        let use_pow2_table = !constant_amount && rhs_witness;
        assert!(
            !use_pow2_table || ceil_log2(bits) <= max_pow2_table_size(b.field()),
            "ICE: an int{bits} shift by a witness amount has no table to read its factor from"
        );
        let table_is_the_bound = use_pow2_table && bits.is_power_of_two();
        if !table_is_the_bound {
            emit_shift_amount_check(b, context, guard, rhs, bits);
        }

        if constant_amount {
            // The literal stands in for `rhs` only where the factor is built. Everything that reads
            // the amount to _reason_ about it keeps the original value, which is the one the range
            // domain has an entry for.
            let amount = match pinned {
                Some(v) => b.int_const(IntBits::from_u128(bits, v)),
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
                use_pow2_table,
            );
        }
    }

    /// An unsigned left-hand side shifted by an amount that folds.
    ///
    /// The factor `1 << amount` is a pure `Shl`, which `hlssa_to_r1cs` accepts only once both
    /// operands are constant, so this route works exactly as far as the amount folds.
    ///
    /// `amount` is `rhs`, or the literal a pinned witness `rhs` provably equals. Only the factor
    /// takes the literal: [`wrap_shifted_product`] sizes its range check from the range domain,
    /// which knows `rhs` and not a value minted afterwards.
    ///
    /// That `Shl` is emitted after `LowerPureGuards`, so nothing downstream checks its amount;
    /// [`Self::lower_shift`] does. Do not move that check into [`Self::lower_general_shift`]: here
    /// an unchecked amount is masked to `bits - 1` by the backend shift, which answers instead of
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
        let factor = pure_factor(b, amount, bits);

        match kind.group() {
            ArithGroup::Shl => {
                let lhs_field = b.cast_to_field(lhs);
                let factor_field = b.cast_to_field(factor);
                let shifted = b.umul(lhs_field, factor_field);
                let value = wrap_shifted_product(b, context, shifted, rhs, bits);
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
    /// Nothing below HLSSA can shift a witness, so a witness amount reads `2^amount` from the
    /// powers-of-two table; at a power-of-two width that lookup is also the rejection, and at any
    /// other the caller emits an explicit check. A pure amount (only with a signed left-hand side)
    /// builds `2^amount` on the pure side, and the caller checks it.
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
        use_pow2_table: bool,
    ) {
        let amount_bits = ceil_log2(bits);

        // The route that builds the factor is also the route that decides where a cofactor could
        // come from, so the two are chosen together and travel as one value. Only the signed `>>`
        // correction ever asks for the cofactor.
        let (factor, cofactor) = if use_pow2_table {
            (
                emit_pow2_factor(b, guard, rhs, bits, amount_bits),
                CofactorSource::Table,
            )
        } else {
            let factor = pure_factor(b, rhs, bits);
            (b.cast_to_field(factor), CofactorSource::Pure(factor))
        };

        match (kind.group(), lhs_signed) {
            // `Shl` is the one shift that takes no sign: the shifted product is wrapped and then
            // reinterpreted at `bits`, which is the same bit pattern under either reading, so the
            // match arm and its callee are both sign-agnostic.
            (ArithGroup::Shl, _) => self.lower_shl(b, context, result, lhs, rhs, factor, bits),
            (ArithGroup::Shr, false) => {
                self.lower_unsigned_shr(b, guard, result, lhs, factor, bits)
            }
            (ArithGroup::Shr, true) => {
                self.lower_signed_shr(b, context, guard, result, lhs, factor, &cofactor, bits)
            }
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
        result: ValueId,
        lhs: ValueId,
        rhs: ValueId,
        factor: ValueId,
        bits: usize,
    ) {
        let lhs_field = b.cast_to_field(lhs);
        let shifted = b.umul(lhs_field, factor);
        let value = wrap_shifted_product(b, context, shifted, rhs, bits);
        b.emit(OpCode::Cast {
            result,
            value,
            target: CastTarget::Int(bits),
        });
    }

    /// `lhs / 2^n` on the raw bits.
    ///
    /// For every amount the check admits, the divisor is a power of two in `[1, 2^(bits-1)]`, so
    /// the division's zero-divisor refusal never fires on a run that passes the check. At a width
    /// that is not a power of two the table has rows past the width, so an amount the check
    /// refuses can make the divisor zero, which only refuses that run a second time.
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
    /// `q` is the _unsigned_ division of the raw bits, `2^(bits-n)` too small for a negative value:
    /// `floor((raw - 2^bits) / 2^n) = q - 2^(bits-n)`, and re-encoding adds `2^bits` back. The
    /// correction sign-fills: on `i8`, `-4 >> 1` is `126 + 128 = 254`.
    ///
    /// Where the guard is off the answer is zero: the guarded division answers zero, and a table
    /// factor is one, which zeroes the correction. A pure factor does not, so its correction is
    /// dropped there.
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
                let power = match cofactor {
                    CofactorSource::Table => emit_pow2_cofactor(b, factor, bits),
                    CofactorSource::Pure(factor) => pure_cofactor(b, *factor, bits),
                };

                // FIELD-ASSUMPTION: L4-decompose
                let two_pow_bits = b.field_const(b.field().two_pow(bits));
                let fill = b.usub(two_pow_bits, power);
                let offset = b.umul(sign, fill);
                let value = b.uadd(quotient_field, offset);
                match cofactor {
                    CofactorSource::Table => value,
                    CofactorSource::Pure(_) => guarded_or_zero_field(b, value, guard),
                }
            }
        };

        b.emit(OpCode::Cast {
            result,
            value,
            target: CastTarget::Int(bits),
        });
    }
}

/// Where the signed `>>` correction's `2^(bits - n)` comes from: the route that built the factor,
/// chosen with it in [`LowerWitnessBitwiseOps::lower_general_shift`] so the two cannot disagree.
enum CofactorSource {
    /// The cofactor is pinned algebraically against the table-supplied factor by
    /// [`emit_pow2_cofactor`], which never needs the amount's bits.
    Table,

    /// The cofactor is divided out of the pure factor, `2^n` as an `Int(bits)`, by
    /// [`pure_cofactor`]. It is known wherever the constraints are built, as the factor is.
    Pure(ValueId),
}

/// Asserts that the shift amount is below the width being shifted.
///
/// The check is a true `amount < bits` at every width. A pure amount is compared by
/// [`emit_pure_shift_amount_check`]. A witness amount has `bits - 1 - amount` range-checked at
/// [`ceil_log2`]`(bits)`: the amount is below its declared width, far inside the modulus, so an
/// amount at or past `bits` makes the difference an element near `p`, which so narrow a range
/// check refuses. Both read the raw pattern, so they also reject a _negative_ amount, whose
/// smallest pattern, `2^(bits - 1)`, is at least `bits`.
///
/// Guarded, so an out-of-range shift under an inactive guard is vacuous rather than a failure.
fn emit_shift_amount_check(
    b: &mut HLBlockEmitter<'_>,
    context: &LoweringContext<'_>,
    guard: Option<ValueId>,
    rhs: ValueId,
    bits: usize,
) {
    // The range domain already proves it. This is the payoff the dual-interval domain was for: it
    // removes the check, and with it the only reason the factor ever needs neutralising.
    if shift_amount_provably_in_range(&context.range(rhs), bits) {
        return;
    }

    // A pure amount is a comparison, which folds wherever the amount does and costs nothing at
    // runtime; a witness one is the range check below.
    if !context.types().get_value_type(rhs).is_witness_of() {
        emit_pure_shift_amount_check(b, guard, rhs, bits);
        return;
    }

    // A witness amount reaches here only at a width that is not a power of two, as the table is
    // its bound at every other, so there are at least two bits to range-check the difference at.
    let amount_bits = ceil_log2(bits);
    assert!(
        amount_bits > 0,
        "ICE: a one-bit shift's witness amount reached the explicit check, which its table is"
    );
    let rhs_field = b.cast_to_field(rhs);
    let largest = b.field_const(b.field().constant((bits - 1) as u64));
    let headroom = b.usub(largest, rhs_field);
    guarded_rangecheck(b, headroom, amount_bits, guard);
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

    // The hint. An out-of-range amount masks here, harmlessly: the lookup below (at a power-of-two
    // width) or the caller's check (at any other) rejects it.
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
    // Only the signed `>>` correction needs a cofactor, at a width `single_cell_shift_fits`
    // limits it to. The pure side divides at any width, so the double-width hint below needs no
    // separate bound.
    //
    // FIELD-ASSUMPTION: L4-decompose
    let two_pow_bits = b.field_const(b.field().two_pow(bits));

    let factor_pure = b.value_of(factor);
    let cofactor_hint = pure_cofactor(b, factor_pure, bits);
    let cofactor = b.write_witness(cofactor_hint);

    b.constrain(factor, cofactor, two_pow_bits);

    cofactor
}

/// `2^amount` as a pure `Int(bits)`, which the backends compute with the amount reduced modulo
/// `bits`, so it is a power of two below `2^bits` whatever the amount.
fn pure_factor(b: &mut HLBlockEmitter<'_>, amount: ValueId, bits: usize) -> ValueId {
    let one = b.int_const(IntBits::one(bits));
    let factor = b.fresh_value();
    b.emit(OpCode::BinaryArithOp {
        kind: BinaryArithOpKind::UShl,
        result: factor,
        lhs: one,
        rhs: amount,
    });
    factor
}

/// `2^bits / factor` as a field element, for a pure `factor` that is a power of two up to
/// `2^(bits - 1)`: [`pure_factor`], or the pure side of a table factor.
///
/// An exact integer division, computed at double width because `2^bits` itself does not fit the
/// shifted width: an amount of zero makes the cofactor `2^bits`.
fn pure_cofactor(b: &mut HLBlockEmitter<'_>, factor: ValueId, bits: usize) -> ValueId {
    let wide_bits = 2 * bits;
    let factor_wide = b.cast_to(CastTarget::Int(wide_bits), factor);
    let two_pow_bits_wide = b.int_const(two_pow_pattern(wide_bits, bits));
    let cofactor = b.fresh_value();
    b.emit(OpCode::BinaryArithOp {
        kind: BinaryArithOpKind::UDiv,
        result: cofactor,
        lhs: two_pow_bits_wide,
        rhs: factor_wide,
    });
    b.cast_to_field(cofactor)
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

/// The low `bits` bits of `lhs * 2^n`, which is Noir's `<<`: a left shift wraps, so bits pushed
/// off the top are discarded, not rejected (`execution_success/bit_shifts_comptime`).
///
/// The prover hints `discarded = product >> bits`, which is subtracted off. **Both halves must be
/// bounded**: `2^bits` is invertible mod `p`, so with `discarded` unbounded a prover can pick any
/// `wrapped` and solve for `discarded`. Bounding both lifts the identity to the integers, as
/// `bit_range.rs::lower_witness_bit_range` does.
///
/// `discarded` is bounded at the amount's width, not `bits`, since at most `n` bits are pushed out;
/// a small constant shift pays a small check, and a provably zero amount none.
///
/// FIELD-ASSUMPTION: L4-decompose. This needs `lhs * 2^n` not to wrap mod `p`, which is
/// [`assert_product_headroom`], and it reads the discarded half through an `Int(2 * bits)`
/// intermediate on the pure side. A shift whose product could wrap is lowered limb-wise by
/// `WideWitnessInts` first.
fn wrap_shifted_product(
    b: &mut HLBlockEmitter<'_>,
    context: &LoweringContext<'_>,
    product: ValueId,
    rhs: ValueId,
    bits: usize,
) -> ValueId {
    // `discarded_width` is the bound the truncation reasons against: the effective amount never
    // exceeds `bits - 1`, and ⊥ answers with that cap.
    let discarded_bits = discarded_width(&context.urange(rhs), bits);
    assert_product_headroom(bits, discarded_bits, b.field());

    // Nothing can be shifted out of a shift by zero, so the product is already the answer.
    if discarded_bits == 0 {
        return product;
    }

    let wide_bits = 2 * bits;
    let pure_product = b.value_of(product);
    let wide = b.cast_to(CastTarget::Int(wide_bits), pure_product);
    let discarded_hint = b.bit_range(wide, bits, bits);
    let discarded_hint = b.cast_to_field(discarded_hint);
    let discarded = b.write_witness(discarded_hint);

    // Unguarded on purpose: an honest product is in range on every path. `factor` is at most
    // `2^(bits - 1)`, and an honest `lhs` is within its width even where its guard is off, since a
    // guarded lowering either zeroes its result (`guarded_or_zero_field`) or answers with an
    // in-range hint, as a guarded spread or limbed division does.
    b.rangecheck(discarded, discarded_bits);

    // FIELD-ASSUMPTION: L4-decompose
    let two_pow_bits = b.field_const(b.field().two_pow(bits));
    let overflow = b.umul(discarded, two_pow_bits);
    let wrapped = b.usub(product, overflow);
    b.rangecheck(wrapped, bits);

    wrapped
}

/// Assert that a `<<` has no product `lhs * 2^n` that could wrap modulo the field.
///
/// The precondition of [`wrap_shifted_product`]: its identity lifts to the integers only while
/// `2^(bits + n) <= p`. Past that, `wrapped` is the low bits of a residue, a wrong answer rather
/// than a rejection. The routing already ensures it ([`single_cell_shift_fits`]; `WideWitnessInts`
/// takes every other shift), so this states the coupling and refuses nothing.
///
/// FIELD-ASSUMPTION: L4-modulus-query. Read off the configured field rather than a fixed prime.
fn assert_product_headroom(bits: usize, discarded_bits: usize, field: FieldConfig) {
    assert!(
        product_fits_field(bits, discarded_bits, field),
        "ICE: a witness `<<` at {bits} bits by up to {discarded_bits} reached the single field product, which reaches 2^{} and so wraps modulo the field",
        bits + discarded_bits
    );
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

/// How many bits of `lhs * 2^n` can be pushed past the top: the amount's upper bound, capped at
/// `bits - 1`. Every factor route keeps the effective amount within that cap:
///
/// - **table**: an amount at or past the width has no row (at a power-of-two width) or fails the
///   hoisted check (at any other); it is never masked.
/// - **pure**: the backends reduce `1 << amount` modulo `bits`, and the hoisted check rejects an
///   out-of-range amount.
/// - **constant-amount**: the hoisted check has already proved the amount in range.
///
/// ⊥ answers the cap, not zero. `Interval::empty`'s `hi` is `0`, and a zero here would skip the
/// truncation and both range checks; the analysis calling the amount unreachable rests on other
/// constraints in this same circuit, so it proves nothing here.
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

fn spread_as_field(b: &mut impl HLEmitter, value: ValueId, bits: usize) -> ValueId {
    let spread = b.spread(value, bits);
    b.cast_to_field(spread)
}

/// Bitwise on a whole `bits`-wide word, via spread-then-add.
///
/// The two callers arrive by different routes, and only one is already sized by the field:
/// `lower_limb_bitwise` arrives at [`witness_half_limb_bits`], while the direct arm of
/// [`LowerWitnessBitwiseOps::lower_binary_bitwise`] arrives at the operand's **type** width, which
/// that arm caps at [`SPREAD_MAX_BITS`].
// FIELD-ASSUMPTION: L6-int-op-strategy
// Bitwise via spread-then-add: the spread of a `bits`-wide value occupies ~2*bits bits (cast to
// `U(bits*2)`), so on a ~64-bit field even a 32-bit spread nearly saturates p. A half-limb operand
// satisfies `2*bits <= h` by construction, which is why every width past the whole-width arm goes
// through a half-limb decomposition. The whole-width arm checks this same predicate before taking
// an operand, so the refusal below could fire only on two half-limb spreads. Their sum is below
// `2^h`, inside the `2h` bits the limb rule in `shared/limbs.rs` keeps injective, so the refusal
// guards no reachable case.
fn lower_word_bitwise(
    b: &mut impl HLEmitter,
    kind: BinaryArithOpKind,
    lhs: ValueId,
    rhs: ValueId,
    bits: usize,
) -> ValueId {
    if !spread_sum_fits_field(bits, b.field()) {
        unsupported_on_this_field(
            format_args!(
                "a {bits}-bit bitwise op spreads each operand to {} bits, and the sum of the two spreads no longer fits one field element, so `Unspread` would read a residue rather than the interleaved bits",
                2 * bits
            ),
            b.field(),
        );
    }

    let lhs_spread = spread_as_field(b, lhs, bits);
    let rhs_spread = spread_as_field(b, rhs, bits);
    let input_spread_sum = b.uadd(lhs_spread, rhs_spread);
    let input_spread_sum = b.cast_to(CastTarget::Int(2 * bits), input_spread_sum);
    let (and_word, xor_word) = b.unspread(input_spread_sum, 2 * bits);

    match kind {
        BinaryArithOpKind::And => and_word,
        BinaryArithOpKind::Xor => xor_word,
        BinaryArithOpKind::Or => b.uadd(and_word, xor_word),
        _ => ice_unreachable!(),
    }
}

// SPREAD LIMBS
// ================================================================================================

/// A decomposition into half-limbs, each with the width it carries.
///
/// [`WitnessLimbs`] is uniform-width by contract. The **spread** wants ragged widths, as its cost
/// falls with the width (`LookupSizing::decompose_spread`): a ragged top limb spread at the four or
/// thirty bits it carries is cheaper than the same limb padded to a full half-limb.
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
            lower_word_bitwise(b, kind, lhs_limb, rhs_limb, limb_width)
        })
        .collect();
    WitnessLimbs { limb_bits, limbs }
}

/// Split `value` into half-limbs at every width, the top one carrying only the bits left over.
///
/// The only decomposition of a bitwise operation. Every limb is at most a half-limb wide, and one
/// field element holds the sum of two such limbs' spreads, so this reaches every width past the
/// one arm above it.
///
/// The high half-limbs are witnessed from the pure shadow, and the lowest is **derived** as
/// `value - sum(higher limbs at their places)`, so the reconstruction holds by construction and
/// needs no separate constraint. A dishonest prover could otherwise inflate a high limb and let the
/// derived one absorb it; the range check on **every** limb at the width it carries stops that. The
/// top limb is bounded at the bits left over, not a full half-limb, so the limbs together cannot
/// carry a value wider than the operand.
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
        // At the width this limb carries: a full half-limb would let the top limb carry bits the
        // operand does not have, and the derived limb below would absorb the difference.
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

    use crate::compiler::{
        pass_manager::{AnalysisStore, Pass},
        passes::instruction_lowering::InstructionLowering,
        ssa::{
            Instruction, Terminator,
            hlssa::{HLSSA, MAX_POW2_TABLE_SIZE, builder::HLSSABuilder},
        },
    };

    /// The two routes into [`lower_word_bitwise`], both cleared on bn254.
    #[test]
    fn no_width_this_lowering_spreads_at_can_wrap_bn254() {
        let bn254 = FieldConfig::bn254();

        // The half-limb route, which is every width past the direct arm.
        assert!(spread_sum_fits_field(witness_half_limb_bits(bn254), bn254));

        // The whole-width route: on bn254 every width from 2 (1 takes the arm above) up to the cap
        // goes whole.
        for bits in 2..=SPREAD_MAX_BITS {
            assert!(spread_sum_fits_field(bits, bn254), "{bits} bits");
        }

        // The half-limb is a width the narrow spread opcode takes, so every spread this lowering
        // makes keeps that opcode's encoding. It holds on every field, because a limb width is a
        // power of two capped at one 64-bit host limb, so its half is at most 32 bits.
        assert!(witness_half_limb_bits(bn254) <= SPREAD_MAX_BITS);
    }

    /// A ragged decomposition bounds its top limb at the width it carries, not a full half-limb.
    ///
    /// This is the whole soundness argument of the general bitwise path, and **no honest witness
    /// can see it**: a real value's limbs are within their true widths either way, so widening the
    /// top bound leaves every end-to-end test green. It would let a prover inflate the top limb and
    /// have the derived low limb absorb the difference, so the decomposition represents a value too
    /// wide for the operand.
    ///
    /// So the test checks what the lowering emits. 100 bits over 32-bit half-limbs is
    /// `32 + 32 + 32 + 4`, and the `4` is the assertion: under a full-width bound there is no cast
    /// to `int4` in the output.
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

        // Follow each witness column to its cast: the hint extraction casts to the same widths, so
        // merely finding an `int4` cast would pass under the ablation this test targets.
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

    /// `lower_limb_bitwise` lowers _every_ limb, not just the first two, which a truncating zip
    /// would still pass at the shortest widths. The test below pins the mismatch check.
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
            .filter(|op| matches!(op, OpCode::Spread { value_bits: 8, .. }))
            .count();
        let unspreads = ops
            .iter()
            .filter(|op| matches!(op, OpCode::Unspread { value_bits: 16, .. }))
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
    fn a_shifted_product_can_wrap_the_field_only_past_the_single_cell() {
        // The precondition _both_ halves of `wrap_shifted_product` depend on: once `raw * 2^n` can
        // pass the modulus, neither a rangecheck on the product nor the truncation identity can
        // tell the residue apart from an honest one. On bn254 (~2^253.5) the boundary sits at
        // `n = 125`.
        let bn254 = FieldConfig::bn254();
        assert!(product_fits_field(128, 125, bn254));
        assert!(!product_fits_field(128, 126, bn254));
        assert!(!product_fits_field(128, 127, bn254));

        // The routing sends only shifts `single_cell_shift_fits` accepts, so on bn254 this always
        // passes, at 127 bits exactly.
        assert!(product_fits_field(64, 63, bn254));
        assert!(product_fits_field(127, 126, bn254));
        assert!(single_cell_shift_fits(bn254, 127, true));
        assert!(!single_cell_shift_fits(bn254, 128, true));
        assert!(product_fits_field(128, 0, bn254));
    }

    #[test]
    fn every_table_backed_width_has_a_row_for_every_amount() {
        // The table is keyed by `ceil(log2(bits))` so that its row count is `1 << size`, the
        // convention every other width-keyed table follows. At a power-of-two width the rows are
        // exactly the legal amounts `0..bits`, which is what lets membership be the amount bound;
        // at any other there are more rows than amounts, and `lower_shift` checks the amount
        // explicitly instead. Either way every legal amount has a row.
        for bits in [8usize, 16, 32, 64] {
            let size = ceil_log2(bits);
            assert!(size <= MAX_POW2_TABLE_SIZE, "{bits}-bit shift has no table");
            assert_eq!(
                1usize << size,
                bits,
                "{bits}-bit shift: rows must be amounts"
            );
        }
        for bits in [1usize, 2, 3, 5, 37, 96, 127] {
            let size = ceil_log2(bits);
            assert!(
                1usize << size >= bits,
                "{bits}-bit shift: an amount with no row"
            );
            assert!(
                bits == 1 || 1usize << (size - 1) < bits,
                "{bits}-bit shift: a table twice the size it needs"
            );
        }

        // The ceiling is exactly the widest single-cell shift (127 bits), with no headroom: one
        // size more puts the widest row, `2^(2^s - 1)`, past the bn254 modulus, where the table
        // stops holding powers of two.
        assert!(single_cell_shift_fits(FieldConfig::bn254(), 127, true));
        assert!(!single_cell_shift_fits(FieldConfig::bn254(), 128, true));
        assert_eq!(ceil_log2(127), MAX_POW2_TABLE_SIZE);
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

    /// `main(c: WitnessOf(u1), x: WitnessOf(i8), n) { Guard(c) { x s>> n } }`, with `n` witnessed
    /// or pure, lowered by this rule alone, so that what it emits is not lowered further, and the
    /// value its result is cast from.
    fn guarded_signed_shr(witnessed_amount: bool) -> (HLSSA, ValueId, ValueId) {
        let mut ssa = HLSSA::with_main("main".to_string());
        let (condition, lhs, rhs, result) = (
            ssa.fresh_value(),
            ssa.fresh_value(),
            ssa.fresh_value(),
            ssa.fresh_value(),
        );
        let function = ssa.get_unique_entrypoint_mut();
        let entry = function.get_entry_mut();
        entry.push_parameter(condition, Type::witness_of(Type::int(1)));
        entry.push_parameter(lhs, Type::witness_of(Type::int(8)));
        let amount = if witnessed_amount {
            Type::witness_of(Type::int(8))
        } else {
            Type::int(8)
        };
        entry.push_parameter(rhs, amount);
        entry.push_test_instruction(OpCode::Guard {
            condition,
            inner: Box::new(OpCode::BinaryArithOp {
                kind: BinaryArithOpKind::SShr,
                result,
                lhs,
                rhs,
            }),
        });
        function.add_return_type(Type::witness_of(Type::int(8)));
        function
            .get_entry_mut()
            .set_terminator(Terminator::Return(vec![result]));
        InstructionLowering::with_lowerers(
            "witness_bitwise_alone",
            vec![Box::new(LowerWitnessBitwiseOps::new())],
            false,
        )
        .run(&mut ssa, &AnalysisStore::new());

        let answer = ssa
            .get_unique_entrypoint()
            .get_entry()
            .get_instructions()
            .find_map(|op| match op {
                OpCode::Cast {
                    result: cast,
                    value,
                    ..
                } if *cast == result => Some(*value),
                _ => None,
            })
            .expect("the shift's result is cast from its answer");
        (ssa, condition, answer)
    }

    /// A guarded signed `>>` answers zero where its guard is off, as the range domain's `Guard`
    /// arm assumes. The quotient is zero there, but the sign correction is zero only for a table
    /// factor; with a pure factor it is selected away.
    #[test]
    fn a_guarded_signed_right_shift_answers_zero_where_its_guard_is_off() {
        for witnessed_amount in [false, true] {
            let (ssa, condition, answer) = guarded_signed_shr(witnessed_amount);
            let ops: Vec<&OpCode> = ssa
                .get_unique_entrypoint()
                .get_entry()
                .get_instructions()
                .collect();
            let definition = ops
                .iter()
                .find(|op| op.get_results().any(|result| *result == answer))
                .unwrap_or_else(|| panic!("the answer is defined: {ops:#?}"));
            let selected = matches!(definition, OpCode::Select { cond, .. } if *cond == condition);
            assert_eq!(
                selected, !witnessed_amount,
                "witnessed amount: {witnessed_amount}; the answer is {definition:?}"
            );
        }
    }
}
