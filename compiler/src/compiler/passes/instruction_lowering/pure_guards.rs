//! Lowers failable pure Guard instructions into explicit checks.
//!
//! After UntaintControlFlow, Guards wrap operations in witness-conditional blocks. Side-effect-free
//! guarded operations are handled by `LowerSideEffectFreeGuards`; this rule keeps only operations whose
//! inactive branch needs special handling to avoid evaluating a failing operation.
//!
//! Classification:
//!
//! - **Lower with OOB check** (can fail if given an out-of-bounds index): ArrayGet — if OOB, assert
//!   !cond and produce default; else array_get.
//! - **Lower with OOB check + passthrough** (RC-tracked allocation): ArraySet — if OOB, assert
//!   !cond and pass through array; else array_set.
//! - **Lower with overflow check** (pure inputs only, can fail): Integer Add/Sub/Mul — compute,
//!   check overflow with native-width predicates, if overflow assert !cond and produce 0.
//! - **Lower with shift check** (pure inputs only, can fail): Integer Shl/Shr — validate the shift
//!   *amount* before shifting. The value is not checked: a shift that pushes bits off the top
//!   wraps, it does not fail.
//! - **Lower with div-zero check** (pure inputs only, can fail): Div/Mod — if divisor==0 assert
//!   !cond and produce 0; else compute. A division the range domain proves defined skips this
//!   entirely: guarded, the guard is dropped and the bare operation emitted; unguarded, the
//!   instruction is left alone with no assertion attached.
//! - **Leave untouched here** (side-effectful, constraint-generating, or handled by witness rules):
//!   Store, Call, Assert, AssertCmp, AssertR1C, Constrain, witness Rangecheck, and failable ops with
//!   witness inputs.

use crate::compiler::{
    analysis::types::FunctionTypeInfo,
    passes::{
        instruction_lowering::{InstructionLoweringRule, LoweringContext},
        shared::{
            divmod_guard::{
                divmod_can_fail, divmod_provably_defined, emit_divmod_failure_cond,
                emit_divmod_is_defined_assert,
            },
            seq_bounds::seq_bounds_operands,
        },
    },
    ssa::{
        Instruction, ValueId,
        hlssa::{
            BinaryArithOpKind, CastTarget, CmpKind, MAX_SUPPORTED_SIGNED_BITS,
            MAX_SUPPORTED_UNSIGNED_BITS, OpCode, Type, TypeExpr,
            builder::{HLBlockEmitter, HLEmitter},
        },
    },
    util::bit_mask,
};

pub struct LowerPureGuards {}

impl InstructionLoweringRule for LowerPureGuards {
    fn lower_instruction(
        &self,
        emitter: &mut HLBlockEmitter<'_>,
        context: &LoweringContext<'_>,
        instruction: &OpCode,
    ) -> bool {
        let type_info = context.types();
        match instruction {
            OpCode::Guard { condition, inner } => {
                self.lower_guard(emitter, context, *condition, inner.as_ref().clone())
            }

            // An unguarded div/mod still has to be checked as `Guard` only covers the ops that sit
            // under witness-dependent control flow. Witness operands are included: the comparisons
            // become ordinary constraints, lowered by the later witness passes just like any
            // other, so one implementation covers both.
            //
            // `Field` is in scope for the same reason the integers are, and needs it most, because
            // there the missing check is a *soundness* hole rather than a wrong answer. `div_field`
            // answers `0` instead of trapping, and for a witness divisor `lower_field_div`
            // constrains only `result * rhs == lhs`. At `rhs == lhs == 0` that degenerates to
            // `q * 0 == 0`, which holds for **every** `q`: witgen fills in `0`, but the quotient is
            // pinned by nothing, so a verifier would accept a proof carrying any value at all in
            // that slot. This check closes that, and covers pure operands, which reach no
            // constraint at all.
            //
            // This agrees with Noir, which rejects a zero field divisor in every runtime — the
            // check is not a strictness increase over it. Noir's ACIR does it structurally rather
            // than with a separate assert: `div_var` on `NativeField` goes through `inv_var`
            // (`acir/acir_context/mod.rs:277`), constraining
            // `predicate * (inv(rhs) * rhs) == predicate` and then forming `lhs * inv(rhs)`. Since
            // `FieldElement::inverse` yields zero when no inverse exists, `rhs == 0` reduces that
            // to `0 == predicate` — unsatisfiable under an active predicate whatever `lhs` is.
            // Brillig and comptime raise "attempt to divide by zero" outright.
            //
            // Note that constraining the inverse is *stronger* than mavros's `result * rhs == lhs`,
            // and gets the nonzero-ness for free. Moving `lower_field_div` onto that encoding would
            // make this `Field` arm redundant; it is left for separate work because it changes the
            // witness shape and so moves R1CS layout everywhere.
            OpCode::BinaryArithOp {
                kind: kind @ (BinaryArithOpKind::Div | BinaryArithOpKind::Mod),
                result,
                lhs,
                rhs,
            } if divmod_can_fail(type_info.get_value_type(*lhs))
                && !self.divmod_discharged(context, *lhs, *rhs) =>
            {
                let lhs_type = type_info.get_value_type(*lhs).strip_witness().clone();
                self.lower_unguarded_divmod(emitter, *kind, *result, *lhs, *rhs, &lhs_type);
                true
            }
            _ => false,
        }
    }
}

impl LowerPureGuards {
    pub fn new() -> Self {
        Self {}
    }

    /// Check whether all inputs to an opcode are pure (not WitnessOf-typed).
    fn all_inputs_pure(&self, op: &OpCode, type_info: &FunctionTypeInfo) -> bool {
        op.get_inputs().all(|id| {
            let ty = type_info.get_value_type(*id);
            !ty.is_witness_of()
        })
    }

    /// Whether the range domain discharges this division's failure check, making both the check and
    /// — under a guard — the branch built around it dead.
    ///
    /// The single query behind both divmod sites in this rule — the guarded one and the unguarded
    /// one — so the two cannot drift apart.
    fn divmod_discharged(&self, context: &LoweringContext<'_>, lhs: ValueId, rhs: ValueId) -> bool {
        let lhs_type = context.types().get_value_type(lhs).peel_witness();
        divmod_provably_defined(&context.range(lhs), &context.range(rhs), lhs_type)
    }

    /// Lower a single Guard instruction.
    fn lower_guard(
        &self,
        emitter: &mut HLBlockEmitter<'_>,
        context: &LoweringContext<'_>,
        condition: ValueId,
        inner: OpCode,
    ) -> bool {
        let type_info = context.types();
        match inner {
            // -- Side-effectful / constraint-generating: always keep as Guard --
            OpCode::Store { .. }
            | OpCode::Call { .. }
            | OpCode::Assert { .. }
            | OpCode::AssertCmp { .. }
            | OpCode::AssertR1C { .. }
            | OpCode::Constrain { .. }
            | OpCode::MemOp { .. }
            | OpCode::Lookup { .. }
            | OpCode::DLookup { .. } => false,

            OpCode::Rangecheck { value, max_bits }
                if !type_info.get_value_type(value).is_witness_of() =>
            {
                self.lower_rangecheck_guard(emitter, condition, value, max_bits, type_info);
                true
            }
            OpCode::Rangecheck { .. } => false,

            // -- Integer arith that can overflow: lower only if all inputs pure --
            OpCode::BinaryArithOp {
                kind:
                    kind @ (BinaryArithOpKind::Add | BinaryArithOpKind::Sub | BinaryArithOpKind::Mul),
                result,
                lhs,
                rhs,
            } => {
                let lhs_type = type_info.get_value_type(lhs);
                match &lhs_type.strip_witness().expr {
                    TypeExpr::U(bits) if self.all_inputs_pure(&inner, type_info) => {
                        self.lower_overflow_guard(
                            emitter, condition, kind, result, lhs, rhs, *bits, false,
                        );
                        true
                    }
                    TypeExpr::I(bits) if self.all_inputs_pure(&inner, type_info) => {
                        self.lower_overflow_guard(
                            emitter, condition, kind, result, lhs, rhs, *bits, true,
                        );
                        true
                    }
                    _ => false,
                }
            }

            // -- Shifts can fail when the shift amount is out of range.  In guarded
            // code, check that before emitting the shift so inactive bad shifts do
            // not become LLVM poison.
            OpCode::BinaryArithOp {
                kind: kind @ (BinaryArithOpKind::Shl | BinaryArithOpKind::Shr),
                result,
                lhs,
                rhs,
            } => {
                let lhs_type = type_info.get_value_type(lhs);
                match &lhs_type.strip_witness().expr {
                    TypeExpr::U(bits) if self.all_inputs_pure(&inner, type_info) => {
                        self.lower_shift_guard(
                            emitter, condition, kind, result, lhs, rhs, *bits, false,
                        );
                        true
                    }
                    TypeExpr::I(bits) if self.all_inputs_pure(&inner, type_info) => {
                        self.lower_shift_guard(
                            emitter, condition, kind, result, lhs, rhs, *bits, true,
                        );
                        true
                    }
                    _ => false,
                }
            }

            // -- Div/Mod: can fail on division by zero, lower only if pure inputs --
            OpCode::BinaryArithOp {
                kind: kind @ (BinaryArithOpKind::Div | BinaryArithOpKind::Mod),
                result,
                lhs,
                rhs,
            } => {
                // Provably defined: the operation is total, so its guard carries no information
                // about it and can simply be dropped -- the same rewrite `LowerSideEffectFreeGuards`
                // applies to every op it accepts, and the reason it refuses `Div`/`Mod` outright is
                // exactly the failure this discharges.
                //
                // Purity is beside the point here. It is required below because that path *builds*
                // a branch out of the operands and needs them evaluable outside the witness; this
                // path builds nothing, so a witness-operand division is dropped on the same terms.
                if self.divmod_discharged(context, lhs, rhs) {
                    emitter.emit(OpCode::BinaryArithOp {
                        kind,
                        result,
                        lhs,
                        rhs,
                    });
                    return true;
                }

                let lhs_type = type_info.get_value_type(lhs);
                match &lhs_type.strip_witness().expr {
                    TypeExpr::U(_) | TypeExpr::I(_) | TypeExpr::Field
                        if self.all_inputs_pure(&inner, type_info) =>
                    {
                        self.lower_divmod_guard(
                            emitter, condition, kind, result, lhs, rhs, lhs_type,
                        );
                        true
                    }
                    // Witness inputs: keep as Guard
                    _ => false,
                }
            }

            // -- ArraySet: lower with OOB check if index is pure.
            OpCode::ArraySet {
                result,
                array,
                index,
                value,
            } if !type_info.get_value_type(index).is_witness_of() => {
                self.lower_array_set_guard(
                    emitter, condition, result, array, index, value, type_info,
                );
                true
            }

            // -- ArrayGet: lower with OOB check if index is pure.
            OpCode::ArrayGet {
                result,
                array,
                index,
            } if !type_info.get_value_type(index).is_witness_of() => {
                self.lower_array_get_guard(emitter, condition, result, array, index, type_info);
                true
            }

            // ArrayGet/ArraySet with witness index: keep as Guard
            OpCode::ArraySet { .. } | OpCode::ArrayGet { .. } => false,

            // Guard-within-Guard should not happen
            OpCode::Guard { .. } => {
                panic!("LowerPureGuards: nested Guard not expected");
            }
            _ => false,
        }
    }

    /// Lower `Guard(cond, arith_op(lhs, rhs) -> result)` for integer overflow.
    ///
    /// Computes at the original width, checks overflow with native-width predicates, and on
    /// overflow constrains !cond and produces a default 0.
    fn lower_overflow_guard(
        &self,
        emitter: &mut HLBlockEmitter<'_>,
        condition: ValueId,
        kind: BinaryArithOpKind,
        original_result: ValueId,
        lhs: ValueId,
        rhs: ValueId,
        bits: usize,
        signed: bool,
    ) {
        if signed && bits > MAX_SUPPORTED_SIGNED_BITS {
            panic!("signed integers wider than i{MAX_SUPPORTED_SIGNED_BITS} are unsupported");
        }

        match (signed, kind) {
            (false, BinaryArithOpKind::Add | BinaryArithOpKind::Sub) => self
                .lower_unsigned_add_sub_guard(
                    emitter,
                    condition,
                    kind,
                    original_result,
                    lhs,
                    rhs,
                    bits,
                ),
            (false, BinaryArithOpKind::Mul) => {
                self.lower_unsigned_mul_guard(emitter, condition, original_result, lhs, rhs, bits);
            }
            (true, BinaryArithOpKind::Add | BinaryArithOpKind::Sub) => self
                .lower_signed_add_sub_guard(
                    emitter,
                    condition,
                    kind,
                    original_result,
                    lhs,
                    rhs,
                    bits,
                ),
            (true, BinaryArithOpKind::Mul) => {
                self.lower_signed_mul_guard(emitter, condition, original_result, lhs, rhs, bits);
            }
            _ => unreachable!("lower_overflow_guard called for {:?}", kind),
        }
    }

    fn lower_unsigned_add_sub_guard(
        &self,
        emitter: &mut HLBlockEmitter<'_>,
        condition: ValueId,
        kind: BinaryArithOpKind,
        original_result: ValueId,
        lhs: ValueId,
        rhs: ValueId,
        bits: usize,
    ) {
        let result_type = Type {
            expr: TypeExpr::U(bits),
        };

        let native_result = emitter.fresh_value();
        emitter.emit(OpCode::BinaryArithOp {
            kind,
            result: native_result,
            lhs,
            rhs,
        });

        let overflow = match kind {
            BinaryArithOpKind::Add => emitter.lt(native_result, lhs),
            BinaryArithOpKind::Sub => emitter.lt(lhs, native_result),
            _ => unreachable!("lower_unsigned_add_sub_guard called for {:?}", kind),
        };

        self.emit_guarded_branch(
            emitter,
            condition,
            overflow,
            original_result,
            &result_type,
            |_| native_result,
            false,
            bits,
        );
    }

    fn lower_signed_add_sub_guard(
        &self,
        emitter: &mut HLBlockEmitter<'_>,
        condition: ValueId,
        kind: BinaryArithOpKind,
        original_result: ValueId,
        lhs: ValueId,
        rhs: ValueId,
        bits: usize,
    ) {
        let result_type = Type {
            expr: TypeExpr::I(bits),
        };
        let native_result = emitter.fresh_value();
        emitter.emit(OpCode::BinaryArithOp {
            kind,
            result: native_result,
            lhs,
            rhs,
        });

        let sign_l = self.sign_bit(emitter, lhs, bits);
        let sign_r = self.sign_bit(emitter, rhs, bits);
        let sign_result = self.sign_bit(emitter, native_result, bits);
        let sign_l_xor_r = emitter.xor(sign_l, sign_r);
        let signs_same = emitter.not(sign_l_xor_r);
        let sign_l_xor_result = emitter.xor(sign_l, sign_result);
        let signs_differ = sign_l_xor_r;
        let overflow = match kind {
            BinaryArithOpKind::Add => emitter.and(signs_same, sign_l_xor_result),
            BinaryArithOpKind::Sub => emitter.and(signs_differ, sign_l_xor_result),
            _ => unreachable!("signed add/sub guard called for {:?}", kind),
        };

        self.emit_guarded_branch(
            emitter,
            condition,
            overflow,
            original_result,
            &result_type,
            |_| native_result,
            true,
            bits,
        );
    }

    fn lower_unsigned_mul_guard(
        &self,
        emitter: &mut HLBlockEmitter<'_>,
        condition: ValueId,
        original_result: ValueId,
        lhs: ValueId,
        rhs: ValueId,
        bits: usize,
    ) {
        let result_type = Type {
            expr: TypeExpr::U(bits),
        };
        let zero = emitter.u_const(bits, 0);
        let rhs_zero = emitter.eq(rhs, zero);
        emitter.build_if_else_into(
            rhs_zero,
            vec![(original_result, result_type.clone())],
            |e| vec![e.mul(lhs, rhs)],
            |e| {
                let max = e.u_const(bits, bit_mask(bits));
                let limit = e.div(max, rhs);
                let overflow = e.lt(limit, lhs);
                e.build_if_else(
                    overflow,
                    vec![result_type.clone()],
                    |e| vec![self.emit_guard_failure_default(e, condition, false, bits)],
                    |e| vec![e.mul(lhs, rhs)],
                )
            },
        );
    }

    fn lower_signed_mul_guard(
        &self,
        emitter: &mut HLBlockEmitter<'_>,
        condition: ValueId,
        original_result: ValueId,
        lhs: ValueId,
        rhs: ValueId,
        bits: usize,
    ) {
        let result_type = Type {
            expr: TypeExpr::I(bits),
        };
        let sign_l = self.sign_bit(emitter, lhs, bits);
        let sign_r = self.sign_bit(emitter, rhs, bits);
        let result_sign = emitter.xor(sign_l, sign_r);
        let abs_l = self.abs_as_u(emitter, lhs, sign_l, bits);
        let abs_r = self.abs_as_u(emitter, rhs, sign_r, bits);
        let zero = emitter.u_const(bits, 0);
        let abs_r_zero = emitter.eq(abs_r, zero);
        emitter.build_if_else_into(
            abs_r_zero,
            vec![(original_result, result_type.clone())],
            |e| vec![e.mul(lhs, rhs)],
            |e| {
                let positive_max = e.u_const(bits, (1u128 << (bits - 1)) - 1);
                let result_sign = e.cast_to(CastTarget::U(bits), result_sign);
                let max_mag = e.add(positive_max, result_sign);
                let limit = e.div(max_mag, abs_r);
                let overflow = e.lt(limit, abs_l);
                e.build_if_else(
                    overflow,
                    vec![result_type.clone()],
                    |e| vec![self.emit_guard_failure_default(e, condition, true, bits)],
                    |e| vec![e.mul(lhs, rhs)],
                )
            },
        );
    }

    fn sign_bit(&self, emitter: &mut HLBlockEmitter<'_>, value: ValueId, bits: usize) -> ValueId {
        let sign = emitter.bit_range(value, bits - 1, 1);
        emitter.cast_to(CastTarget::U(1), sign)
    }

    fn abs_as_u(
        &self,
        emitter: &mut HLBlockEmitter<'_>,
        value: ValueId,
        sign_u1: ValueId,
        bits: usize,
    ) -> ValueId {
        let value_field = emitter.cast_to_field(value);
        let sign = emitter.cast_to_field(sign_u1);
        let sign_shift = emitter.field_const(emitter.field().two_pow(bits));
        let sign_shifted = emitter.mul(sign, sign_shift);
        let signed_value = emitter.sub(value_field, sign_shifted);
        let two = emitter.field_const(emitter.field().constant(2));
        let two_sign = emitter.mul(two, sign);
        let one = emitter.field_const(emitter.field().constant(1));
        let factor = emitter.sub(one, two_sign);
        let abs = emitter.mul(signed_value, factor);
        emitter.cast_to(CastTarget::U(bits), abs)
    }

    /// Lower `Guard(cond, shift(lhs, rhs) -> result)`.
    ///
    /// Shifts are only valid for amounts in `[0, bits)`. The range check must dominate the shift
    /// itself; otherwise LLVM can treat an out-of-range shift in an inactive guarded branch as
    /// poison.
    ///
    /// The *amount* is the only failure mode. A `<<` whose result leaves the width wraps, with Noir
    /// reporting an error only for the amount, so the valid path is the bare operation for both
    /// kinds, and the backends all truncate it to `bits` (`shl_u64` masks, LLVM's `shl` is already
    /// at the operand width, and `hlssa_to_r1cs`'s constant fold wraps).
    fn lower_shift_guard(
        &self,
        emitter: &mut HLBlockEmitter<'_>,
        condition: ValueId,
        kind: BinaryArithOpKind,
        original_result: ValueId,
        lhs: ValueId,
        rhs: ValueId,
        bits: usize,
        signed: bool,
    ) {
        debug_assert!(
            matches!(kind, BinaryArithOpKind::Shl | BinaryArithOpKind::Shr),
            "ICE: lower_shift_guard called for non-shift op"
        );

        let result_type = if signed {
            Type {
                expr: TypeExpr::I(bits),
            }
        } else {
            Type {
                expr: TypeExpr::U(bits),
            }
        };
        let invalid_shift = self.emit_invalid_shift_cond(emitter, rhs, bits, signed);

        emitter.build_if_else_into(
            invalid_shift,
            vec![(original_result, result_type.clone())],
            |e| vec![self.emit_guard_failure_default(e, condition, signed, bits)],
            |e| {
                let result = e.fresh_value();
                e.emit(OpCode::BinaryArithOp {
                    kind,
                    result,
                    lhs,
                    rhs,
                });
                vec![result]
            },
        );
    }

    fn emit_invalid_shift_cond(
        &self,
        emitter: &mut HLBlockEmitter<'_>,
        rhs: ValueId,
        bits: usize,
        signed: bool,
    ) -> ValueId {
        let cmp_bits = bits.max(64);
        let cmp_target = if signed {
            CastTarget::I(cmp_bits)
        } else {
            CastTarget::U(cmp_bits)
        };
        let rhs_cmp = emitter.cast_to(cmp_target, rhs);
        let rhs_bound = if signed {
            emitter.i_const(cmp_bits, bits as u128)
        } else {
            emitter.u_const(cmp_bits, bits as u128)
        };
        let rhs_lt_bits = emitter.lt(rhs_cmp, rhs_bound);
        let rhs_too_large = emitter.not(rhs_lt_bits);

        if signed {
            let zero = emitter.i_const(cmp_bits, 0);
            let rhs_negative = emitter.lt(rhs_cmp, zero);
            emitter.or(rhs_negative, rhs_too_large)
        } else {
            rhs_too_large
        }
    }

    /// Lower an *unguarded* pure `Div`/`Mod` by asserting it is defined, then performing it
    /// unchanged.
    ///
    /// Without this, an undefined division reaches the backends unchecked and each one disagrees
    /// about what it means:
    ///
    /// - **Zero divisor, integer:** The VM's `div_u64`/`div_u128`/`div_s64` are plain Rust `/`, so
    ///   witness generation aborts the process with an arithmetic panic instead of reporting a
    ///   failed execution. LLVM's `udiv`/`sdiv` by zero is undefined behavior outright.
    /// - **Zero divisor, field:** `div_field` answers `0`, so nothing traps — and at `0 / 0` the
    ///   `result * rhs == lhs` constraint is satisfied by any quotient, so the value is not pinned
    ///   at all. The worst of the three: silent, and unsound rather than merely wrong.
    /// - **`INT_MIN / -1`:** `div_s64` sign-extends to `i64` and wraps on the way back down; LLVM
    ///   calls signed-division overflow undefined behaviour.
    ///
    /// Noir treats all of these as execution failures, so we check in the IR so all backends can
    /// inherit the same, correct answer.
    fn lower_unguarded_divmod(
        &self,
        emitter: &mut HLBlockEmitter<'_>,
        kind: BinaryArithOpKind,
        result: ValueId,
        lhs: ValueId,
        rhs: ValueId,
        lhs_type: &Type,
    ) {
        emit_divmod_is_defined_assert(emitter, lhs, rhs, lhs_type);
        emitter.emit(OpCode::BinaryArithOp {
            kind,
            result,
            lhs,
            rhs,
        });
    }

    fn lower_divmod_guard(
        &self,
        emitter: &mut HLBlockEmitter<'_>,
        condition: ValueId,
        kind: BinaryArithOpKind,
        original_result: ValueId,
        lhs: ValueId,
        rhs: ValueId,
        lhs_type: &Type,
    ) {
        let failure = emit_divmod_failure_cond(emitter, lhs, rhs, lhs_type);

        emitter.build_if_else_into(
            failure,
            vec![(original_result, lhs_type.clone())],
            // Divisor is zero: assert condition is false, produce default
            |e| {
                let zero_u1 = e.u_const(1, 0);
                e.emit(OpCode::AssertCmp {
                    kind: CmpKind::Eq,
                    lhs: condition,
                    rhs: zero_u1,
                });
                let default_val = match &lhs_type.expr {
                    TypeExpr::U(b) => e.u_const(*b, 0),
                    TypeExpr::I(b) => e.i_const(*b, 0),
                    TypeExpr::Field => e.field_const(e.field().constant(0u64)),
                    _ => unreachable!(),
                };
                vec![default_val]
            },
            // Divisor is non-zero: perform the div/mod
            |e| {
                let r = e.fresh_value();
                e.emit(OpCode::BinaryArithOp {
                    kind,
                    result: r,
                    lhs,
                    rhs,
                });
                vec![r]
            },
        );
    }

    /// Lower `Guard(cond, ArraySet(array, idx, val) -> result)`.
    ///
    /// Pattern:
    ///   oob = idx >= len(array)
    ///   if oob { assert !cond; result = array } else { result = array_set(array, idx, value) }
    fn lower_array_set_guard(
        &self,
        emitter: &mut HLBlockEmitter<'_>,
        condition: ValueId,
        original_result: ValueId,
        array: ValueId,
        index: ValueId,
        value: ValueId,
        type_info: &FunctionTypeInfo,
    ) {
        let array_type = type_info.get_value_type(array).strip_witness().clone();
        let oob = self.emit_oob_cond(emitter, array, index, type_info);

        emitter.build_if_else_into(
            oob,
            vec![(original_result, array_type)],
            // OOB: assert condition is false, pass through original array
            |e| {
                let zero = e.u_const(1, 0);
                e.emit(OpCode::AssertCmp {
                    kind: CmpKind::Eq,
                    lhs: condition,
                    rhs: zero,
                });
                vec![array]
            },
            // In-bounds: do the set
            |e| vec![e.array_set(array, index, value)],
        );
    }

    /// Lower `Guard(cond, ArrayGet(array, idx) -> result)`.
    ///
    /// Pattern:
    ///   if oob { assert !cond; result = default } else { result = array_get(array, idx) }
    fn lower_array_get_guard(
        &self,
        emitter: &mut HLBlockEmitter<'_>,
        condition: ValueId,
        original_result: ValueId,
        array: ValueId,
        index: ValueId,
        type_info: &FunctionTypeInfo,
    ) {
        let array_type = type_info.get_value_type(array);
        let elem_type = match &array_type.strip_witness().expr {
            TypeExpr::Array(elem, _) | TypeExpr::Slice(elem) => (**elem).clone(),
            other => panic!("LowerPureGuards: ArrayGet on non-seq type: {:?}", other),
        };
        let oob = self.emit_oob_cond(emitter, array, index, type_info);

        emitter.build_if_else_into(
            oob,
            vec![(original_result, elem_type.clone())],
            // OOB: assert condition is false, produce default value
            |e| {
                let zero = e.u_const(1, 0);
                e.emit(OpCode::AssertCmp {
                    kind: CmpKind::Eq,
                    lhs: condition,
                    rhs: zero,
                });
                vec![e.default_value(&elem_type)]
            },
            // In-bounds: do the get
            |e| vec![e.array_get(array, index)],
        );
    }

    /// Lower `Guard(cond, Rangecheck(v, max_bits))` for a pure `v` into
    /// `if v >= 2^max_bits { assert(cond == 0) }`. When the type bound on
    /// `v` already implies the rangecheck holds, the lowering collapses to
    /// a no-op.
    fn lower_rangecheck_guard(
        &self,
        emitter: &mut HLBlockEmitter<'_>,
        condition: ValueId,
        value: ValueId,
        max_bits: usize,
        type_info: &FunctionTypeInfo,
    ) {
        let val_type = type_info.get_value_type(value);
        match &val_type.expr {
            TypeExpr::U(n) | TypeExpr::I(n) => {
                let val_bits = *n;
                if val_bits <= max_bits {
                    return;
                }
                assert!(
                    val_bits <= MAX_SUPPORTED_UNSIGNED_BITS
                        && max_bits < MAX_SUPPORTED_UNSIGNED_BITS,
                    "LowerPureGuards: pure rangecheck on {val_type} with max_bits = \
                     {max_bits} needs wider-than-u128 comparison; not yet supported"
                );
                let cmp_bits = val_bits.max(max_bits + 1);
                let v_cmp = emitter.widen_u(value, val_bits, cmp_bits);
                let bound = emitter.u_const(cmp_bits, 1u128 << max_bits);
                let in_range = emitter.lt(v_cmp, bound);
                let oob = emitter.not(in_range);

                emitter.build_if_else_into(
                    oob,
                    vec![],
                    |e| {
                        let zero = e.u_const(1, 0);
                        e.emit(OpCode::AssertCmp {
                            kind: CmpKind::Eq,
                            lhs: condition,
                            rhs: zero,
                        });
                        vec![]
                    },
                    |_| vec![],
                );
            }
            TypeExpr::Field => {
                if max_bits >= emitter.field().field_bit_size() as usize {
                    return;
                }

                let bound = emitter.field_const(emitter.field().two_pow(max_bits));
                let in_range = emitter.lt(value, bound);
                let oob = emitter.not(in_range);

                emitter.build_if_else_into(
                    oob,
                    vec![],
                    |e| {
                        let zero = e.u_const(1, 0);
                        e.emit(OpCode::AssertCmp {
                            kind: CmpKind::Eq,
                            lhs: condition,
                            rhs: zero,
                        });
                        vec![]
                    },
                    |_| vec![],
                );
            }
            other => panic!(
                "LowerPureGuards: pure rangecheck on unsupported type {:?}; \
                 add a comparison strategy for this type",
                other
            ),
        }
    }

    /// Compute the OOB condition `idx >= len(seq)`, returning a bool ValueId, for both arrays and
    /// slices.
    ///
    /// The length lookup and the widening rule come from [`seq_bounds_operands`], which is also
    /// what DCE's dead-access rewrite builds its `AssertCmp` from. This rule needs the condition as
    /// a *value* to branch on rather than as an assert, but the comparison itself must be the same
    /// one or the two disagree about what "out of bounds" means.
    fn emit_oob_cond(
        &self,
        emitter: &mut HLBlockEmitter<'_>,
        seq: ValueId,
        index: ValueId,
        type_info: &FunctionTypeInfo,
    ) -> ValueId {
        let seq_type = type_info.get_value_type(seq).clone();
        let idx_type = type_info.get_value_type(index).clone();
        let (_, len_cmp, idx_cmp, _) =
            seq_bounds_operands(emitter, seq, index, &seq_type, &idx_type);
        let in_bounds = emitter.lt(idx_cmp, len_cmp);
        emitter.not(in_bounds)
    }

    /// Common pattern: branch on a failure condition.
    ///
    /// In the fail branch, assert condition==false and produce a default value. In the ok branch,
    /// execute the actual computation.
    fn emit_guarded_branch(
        &self,
        emitter: &mut HLBlockEmitter<'_>,
        condition: ValueId,
        failure: ValueId,
        original_result: ValueId,
        result_type: &Type,
        ok_path: impl FnOnce(&mut HLBlockEmitter<'_>) -> ValueId,
        signed: bool,
        bits: usize,
    ) {
        emitter.build_if_else_into(
            failure,
            vec![(original_result, result_type.clone())],
            // Failure: assert condition is false, produce default value
            |e| vec![self.emit_guard_failure_default(e, condition, signed, bits)],
            // Ok: compute the result
            |e| vec![ok_path(e)],
        );
    }

    fn emit_guard_failure_default(
        &self,
        emitter: &mut HLBlockEmitter<'_>,
        condition: ValueId,
        signed: bool,
        bits: usize,
    ) -> ValueId {
        let zero = emitter.u_const(1, 0);
        emitter.emit(OpCode::AssertCmp {
            kind: CmpKind::Eq,
            lhs: condition,
            rhs: zero,
        });
        if signed {
            emitter.i_const(bits, 0)
        } else {
            emitter.u_const(bits, 0)
        }
    }
}
