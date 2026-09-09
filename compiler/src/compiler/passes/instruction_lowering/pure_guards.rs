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
//!   Unguarded, the operation is performed and the check simply asserted. An operation the range
//!   domain proves stays inside its width skips this entirely, on the same terms as the shift and
//!   division below: guarded, the guard is dropped and the bare operation emitted; unguarded, the
//!   instruction is left alone with no assertion attached.
//! - **Lower with shift check** (pure inputs only, can fail): Integer Shl/Shr — validate the shift
//!   _amount_ before shifting. The value is not checked: a shift that pushes bits off the top
//!   wraps, it does not fail. A shift the range domain proves in range skips this entirely, on the
//!   same terms as the division below.
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
            overflow_guard::{
                emit_no_overflow_assert, emit_overflow_cond, mul_overflows_nonzero,
                overflow_operand_bits, overflow_provably_impossible,
                signed_mul_magnitude_overflows, signed_mul_operands,
            },
            seq_bounds::seq_bounds_operands,
            shift_guard::{
                emit_invalid_shift_cond, emit_shift_amount_is_valid_assert,
                shift_amount_provably_in_range, shift_operand_bits,
            },
        },
    },
    ssa::{
        Instruction, ValueId,
        hlssa::{
            ArithGroup, BinaryArithOpKind, CmpKind, MAX_SUPPORTED_UNSIGNED_BITS, OpCode, Type,
            TypeExpr, assert_signed_op_width,
            builder::{HLBlockEmitter, HLEmitter},
        },
    },
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
            // there the missing check is a _soundness_ hole rather than a wrong answer. `div_field`
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
            // Note that constraining the inverse is _stronger_ than mavros's `result * rhs == lhs`,
            // and gets the nonzero-ness for free. Moving `lower_field_div` onto that encoding would
            // make this `Field` arm redundant; it is left for separate work because it changes the
            // witness shape and so moves R1CS layout everywhere.
            OpCode::BinaryArithOp {
                kind,
                result,
                lhs,
                rhs,
            } if matches!(kind.group(), ArithGroup::Div | ArithGroup::Rem)
                && divmod_can_fail(type_info.get_value_type(*lhs))
                && !self.divmod_discharged(context, *kind, *lhs, *rhs) =>
            {
                let lhs_type = type_info.get_value_type(*lhs).strip_witness().clone();

                // As in the guarded arm: the operation decides, and both the check and the division
                // are built from that one answer.
                self.lower_unguarded_divmod(
                    emitter,
                    *kind,
                    *result,
                    *lhs,
                    *rhs,
                    &lhs_type,
                    kind.is_signed(),
                );
                true
            }

            // An unguarded shift needs its amount checked for exactly the reason the unguarded
            // div/mod above does: `Guard` only wraps the ops under witness-dependent control flow,
            // so an unconditional shift reached `lower_shift_guard` never.
            //
            // Pure inputs only, matching the guarded arm. A shift with a witness operand is left
            // for `LowerWitnessBitwiseOps::lower_shift`, which emits the equivalent check itself
            // (`emit_shift_amount_check`, on both of its lowerings) because it has to be able to
            // build that check out of constraints rather than out of a pure comparison.
            OpCode::BinaryArithOp {
                kind,
                result,
                lhs,
                rhs,
            } if matches!(kind.group(), ArithGroup::Shl | ArithGroup::Shr)
                && self.all_inputs_pure(instruction, type_info) =>
            {
                match shift_operand_bits(type_info.get_value_type(*lhs)) {
                    Some(bits) => {
                        // Discharged where the domain already pins the amount in range, exactly as
                        // the guarded arm below discharges it.
                        if self.shift_discharged(context, bits, *rhs) {
                            return false;
                        }
                        self.lower_unguarded_shift(emitter, *kind, *result, *lhs, *rhs, bits);
                        true
                    }
                    None => false,
                }
            }

            // An unguarded `Add`/`Sub`/`Mul` needs its overflow checked for exactly the reason the
            // unguarded div/mod and shift above do: `Guard` only wraps the operations under
            // witness-dependent control flow, so an unconditional one never reached
            // `lower_overflow_guard`.
            //
            // Noir rejects an overflowing `+`, `-` or `*` in both constrained and unconstrained
            // code, so we have to ensure we match that behavior here.
            //
            // Pure inputs only, matching the shift arm. A witness operand belongs to
            // `LowerWitnessIntegerArithOps`, which rejects the same executions by computing in the
            // field and range-checking the result back down to the width; emitting this check as
            // well would be a second, differently encoded copy of the same condition.
            OpCode::BinaryArithOp {
                kind,
                result,
                lhs,
                rhs,
            } if matches!(
                kind.group(),
                ArithGroup::Add | ArithGroup::Sub | ArithGroup::Mul
            ) && self.all_inputs_pure(instruction, type_info) =>
            {
                match overflow_operand_bits(type_info.get_value_type(*lhs)) {
                    Some(bits) => {
                        // Discharged where the domain already pins the result inside the width,
                        // exactly as the two arms above discharge theirs. This is what keeps an
                        // ordinary loop counter's `i + 1` from paying for a check.
                        if self.overflow_discharged(context, *kind, bits, *lhs, *rhs) {
                            return false;
                        }
                        self.lower_unguarded_overflow(emitter, *kind, *result, *lhs, *rhs, bits);
                        true
                    }
                    None => false,
                }
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
    fn divmod_discharged(
        &self,
        context: &LoweringContext<'_>,
        kind: BinaryArithOpKind,
        lhs: ValueId,
        rhs: ValueId,
    ) -> bool {
        let signed = kind.is_signed();
        let lhs_type = context.types().get_value_type(lhs).peel_witness();
        divmod_provably_defined(&context.range(lhs), &context.range(rhs), lhs_type, signed)
    }

    /// Whether the range domain discharges this shift's amount check, making both the check and —
    /// under a guard — the branch built around it dead.
    ///
    /// The single query behind both shift sites in this rule, for the reason
    /// [`Self::divmod_discharged`] is the single query behind both division sites.
    ///
    /// Asked of the raw pattern rather than of a chosen reading, which is why it takes no sign: an
    /// amount below the width is non-negative under the signed reading too, so one answer serves
    /// both shift kinds. See [`ValueRange::proves_shift_amount_below`].
    ///
    /// It does not fire nearly as often as it could. `LowerPureGuards` runs long before
    /// `Specializer`, so an amount that only becomes a literal after inlining still reads as
    /// full-width here; measured over the local corpus, 100 of 212 unguarded sites discharge. That
    /// is the same blind spot the divmod discharge has, and closing it would mean re-checking later
    /// rather than weakening this.
    ///
    /// [`ValueRange::proves_shift_amount_below`]: crate::compiler::analysis::value_range_analysis::ValueRange::proves_shift_amount_below
    fn shift_discharged(&self, context: &LoweringContext<'_>, bits: usize, rhs: ValueId) -> bool {
        shift_amount_provably_in_range(&context.range(rhs), bits)
    }

    /// Whether the range domain discharges this operation's overflow check, making both the check
    /// and (under a guard) the branch built around it dead.
    ///
    /// Unlike the shift discharge this one _does_ take a sign, because the two readings ask
    /// different questions of the same operands: `100 + 100` fits a `u8` and overflows an `i8`. It
    /// comes from the opcode, which is also what selects the check being discharged.
    fn overflow_discharged(
        &self,
        context: &LoweringContext<'_>,
        kind: BinaryArithOpKind,
        bits: usize,
        lhs: ValueId,
        rhs: ValueId,
    ) -> bool {
        overflow_provably_impossible(
            &context.range(lhs),
            &context.range(rhs),
            kind.group(),
            bits,
            kind.is_signed(),
        )
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
                kind,
                result,
                lhs,
                rhs,
            } if matches!(
                kind.group(),
                ArithGroup::Add | ArithGroup::Sub | ArithGroup::Mul
            ) =>
            {
                let lhs_type = type_info.get_value_type(lhs);

                // The operation decides; the arm below matches the type only to read `bits`.
                let signed = kind.is_signed();
                match &lhs_type.strip_witness().expr {
                    TypeExpr::Int(bits) if self.all_inputs_pure(&inner, type_info) => {
                        // Provably inside the width: the operation is total, so its guard carries
                        // no information about it and the bare operation can simply replace the
                        // whole diamond. Consulted from both overflow sites.
                        if self.overflow_discharged(context, kind, *bits, lhs, rhs) {
                            emitter.emit(OpCode::BinaryArithOp {
                                kind,
                                result,
                                lhs,
                                rhs,
                            });
                            return true;
                        }
                        self.lower_overflow_guard(
                            emitter, condition, kind, result, lhs, rhs, *bits, signed,
                        );
                        true
                    }
                    _ => false,
                }
            }

            // Shifts can fail when the shift amount is out of range. In guarded code, check that
            // BEFORE emitting the shift so inactive bad shifts do not become LLVM poison.
            OpCode::BinaryArithOp {
                kind,
                result,
                lhs,
                rhs,
            } if matches!(kind.group(), ArithGroup::Shl | ArithGroup::Shr) => {
                let lhs_type = type_info.get_value_type(lhs);
                match shift_operand_bits(lhs_type) {
                    Some(bits) if self.all_inputs_pure(&inner, type_info) => {
                        // Provably in range: the shift is total, so its guard carries no
                        // information about it and the bare operation can simply replace the whole
                        // diamond. This is the divmod discharge's argument, on the same query, and
                        // it is consulted from both shift sites so the two cannot drift apart.
                        if self.shift_discharged(context, bits, rhs) {
                            emitter.emit(OpCode::BinaryArithOp {
                                kind,
                                result,
                                lhs,
                                rhs,
                            });
                            return true;
                        }
                        self.lower_shift_guard(emitter, condition, kind, result, lhs, rhs, bits);
                        true
                    }
                    _ => false,
                }
            }

            // -- Div/Mod: can fail on division by zero, lower only if pure inputs --
            OpCode::BinaryArithOp {
                kind,
                result,
                lhs,
                rhs,
            } if matches!(kind.group(), ArithGroup::Div | ArithGroup::Rem) => {
                // Provably defined: the operation is total, so its guard carries no information
                // about it and can simply be dropped -- the same rewrite `LowerSideEffectFreeGuards`
                // applies to every op it accepts, and the reason it refuses `Div`/`Mod` outright is
                // exactly the failure this discharges.
                //
                // Purity is beside the point here. It is required below because that path _builds_
                // a branch out of the operands and needs them evaluable outside the witness; this
                // path builds nothing, so a witness-operand division is dropped on the same terms.
                if self.divmod_discharged(context, kind, lhs, rhs) {
                    emitter.emit(OpCode::BinaryArithOp {
                        kind,
                        result,
                        lhs,
                        rhs,
                    });
                    return true;
                }

                // As for the overflow and shift guards above: the operation decides. The division
                // re-planted inside the guard keeps the opcode it arrived with, and
                // `emit_divmod_failure_cond` builds the check from the same flag, so the two halves
                // of this lowering cannot read different sources.
                let signed = kind.is_signed();

                let lhs_type = type_info.get_value_type(lhs);
                match &lhs_type.strip_witness().expr {
                    TypeExpr::Int(_) | TypeExpr::Field
                        if self.all_inputs_pure(&inner, type_info) =>
                    {
                        self.lower_divmod_guard(
                            emitter, condition, kind, result, lhs, rhs, lhs_type, signed,
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
        if signed {
            assert_signed_op_width(bits, "guarded overflow check");
        }

        match kind.group() {
            // The operation is performed first and unconditionally. It wraps rather than trapping,
            // so an overflowing one in an inactive branch is harmless, and the check needs the
            // wrapped result anyway, which is why both polarities read it rather than recomputing
            // anything. Only the multiplies below need the operation kept off the failing path.
            ArithGroup::Add | ArithGroup::Sub => {
                let wrapped = emitter.bin(kind, lhs, rhs);
                let overflow = emit_overflow_cond(emitter, kind, lhs, rhs, Some(wrapped), bits);
                self.emit_guarded_branch(
                    emitter,
                    condition,
                    overflow,
                    original_result,
                    &Type::int(bits),
                    |_| wrapped,
                    bits,
                );
            }
            ArithGroup::Mul if signed => {
                self.lower_signed_mul_guard(emitter, condition, original_result, lhs, rhs, bits);
            }
            ArithGroup::Mul => {
                self.lower_unsigned_mul_guard(emitter, condition, original_result, lhs, rhs, bits);
            }
            _ => unreachable!("lower_overflow_guard called for {:?}", kind),
        }
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
        let result_type = Type::int(bits);
        let zero = emitter.int_const(bits, 0);
        let rhs_zero = emitter.eq(rhs, zero);
        emitter.build_if_else_into(
            rhs_zero,
            vec![(original_result, result_type.clone())],
            |e| vec![e.umul(lhs, rhs)],
            |e| {
                // Reached only where `rhs != 0`, which is the precondition
                // `mul_overflows_nonzero` is stated under. The unguarded lowering selects a safe
                // divisor instead of branching; both ask this one predicate.
                let overflow = mul_overflows_nonzero(e, lhs, rhs, bits);
                e.build_if_else(
                    overflow,
                    vec![result_type.clone()],
                    |e| vec![self.emit_guard_failure_default(e, condition, bits)],
                    |e| vec![e.umul(lhs, rhs)],
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
        let result_type = Type::int(bits);
        let operands = signed_mul_operands(emitter, lhs, rhs, bits);
        let zero = emitter.int_const(bits, 0);
        let abs_r_zero = emitter.eq(operands.abs_rhs, zero);
        emitter.build_if_else_into(
            abs_r_zero,
            vec![(original_result, result_type.clone())],
            // The product itself is a _signed_ multiply, which `smul` is what states: the operands'
            // type carries no sign to read it from. The magnitude arithmetic below runs on absolute
            // values under unsigned opcodes and stays unsigned.
            |e| vec![e.smul(lhs, rhs)],
            |e| {
                // As above: this arm has already established `|rhs| != 0`.
                let overflow = signed_mul_magnitude_overflows(
                    e,
                    operands.abs_lhs,
                    operands.abs_rhs,
                    operands.result_sign,
                    bits,
                );
                e.build_if_else(
                    overflow,
                    vec![result_type.clone()],
                    |e| vec![self.emit_guard_failure_default(e, condition, bits)],
                    |e| vec![e.smul(lhs, rhs)],
                )
            },
        );
    }

    /// Lower `Guard(cond, shift(lhs, rhs) -> result)`.
    ///
    /// Shifts are only valid for amounts in `[0, bits)`. The range check must dominate the shift
    /// itself; otherwise LLVM can treat an out-of-range shift in an inactive guarded branch as
    /// poison.
    ///
    /// The _amount_ is the only failure mode. A `<<` whose result leaves the width wraps, with Noir
    /// reporting an error only for the amount, so the valid path is the bare operation for both
    /// kinds, and the backends all truncate it to `bits`: `shl_int` masks, LLVM's `shl` is already
    /// at the operand width, and `hlssa_to_r1cs`'s constant fold delegates to
    /// [`mavros_int_semantics::residue`].
    #[allow(clippy::too_many_arguments)]
    fn lower_shift_guard(
        &self,
        emitter: &mut HLBlockEmitter<'_>,
        condition: ValueId,
        kind: BinaryArithOpKind,
        original_result: ValueId,
        lhs: ValueId,
        rhs: ValueId,
        bits: usize,
    ) {
        debug_assert!(
            matches!(kind.group(), ArithGroup::Shl | ArithGroup::Shr),
            "ICE: lower_shift_guard called for non-shift op"
        );

        let result_type = Type::int(bits);
        let invalid_shift = emit_invalid_shift_cond(emitter, rhs, bits, kind.is_signed());

        emitter.build_if_else_into(
            invalid_shift,
            vec![(original_result, result_type.clone())],
            |e| vec![self.emit_guard_failure_default(e, condition, bits)],
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

    /// Lower an _unguarded_ pure `Div`/`Mod` by asserting it is defined, then performing it
    /// unchanged.
    ///
    /// Without this, an undefined division reaches the backends unchecked and each one disagrees
    /// about what it means:
    ///
    /// - **Zero divisor, integer:** LLVM's `udiv`/`sdiv` by zero is undefined behavior outright.
    ///   The VM answers zero as a backstop, not a definition: `cell_udiv` picks it so a program
    ///   that got here reports a failed execution rather than aborting witness generation with an
    ///   arithmetic panic, and [`mavros_int_semantics::residue`] declines to specify these inputs
    ///   precisely because the two do not agree.
    /// - **Zero divisor, field:** `div_field` answers `0`, so nothing traps — and at `0 / 0` the
    ///   `result * rhs == lhs` constraint is satisfied by any quotient, so the value is not pinned
    ///   at all. The worst of the three: silent, and unsound rather than merely wrong.
    /// - **`INT_MIN / -1`:** `sdiv_int` sign-extends to `i64` and wraps on the way back down; LLVM
    ///   calls signed-division overflow undefined behaviour.
    ///
    /// Noir treats all of these as execution failures, so we check in the IR so all backends can
    /// inherit the same, correct answer.
    #[allow(clippy::too_many_arguments)]
    fn lower_unguarded_divmod(
        &self,
        emitter: &mut HLBlockEmitter<'_>,
        kind: BinaryArithOpKind,
        result: ValueId,
        lhs: ValueId,
        rhs: ValueId,
        lhs_type: &Type,
        signed: bool,
    ) {
        emit_divmod_is_defined_assert(emitter, lhs, rhs, lhs_type, signed);
        emitter.emit(OpCode::BinaryArithOp {
            kind,
            result,
            lhs,
            rhs,
        });
    }

    /// Assert the shift amount is in range, then emit the bare shift.
    ///
    /// The unguarded counterpart of [`Self::lower_shift_guard`]. Both build their condition from
    /// the same two comparisons in [`crate::compiler::passes::shared::shift_guard`], so they cannot
    /// disagree about what "in range" means. The guarded form substitutes a default value on the
    /// invalid path because the shift may be inactive; here there is no guard, so an invalid amount
    /// is simply a failure.
    fn lower_unguarded_shift(
        &self,
        emitter: &mut HLBlockEmitter<'_>,
        kind: BinaryArithOpKind,
        result: ValueId,
        lhs: ValueId,
        rhs: ValueId,
        bits: usize,
    ) {
        emit_shift_amount_is_valid_assert(emitter, rhs, bits, kind.is_signed());
        emitter.emit(OpCode::BinaryArithOp {
            kind,
            result,
            lhs,
            rhs,
        });
    }

    /// Perform an _unguarded_ pure `Add`/`Sub`/`Mul`, then assert it did not overflow.
    ///
    /// The operation comes **first**, which is the opposite of [`Self::lower_unguarded_divmod`]'s
    /// order and deliberate. A division has to be dominated by its check because an undefined one
    /// is undefined behavior in LLVM; an overflowing add is neither. It wraps, every backend wraps
    /// it identically, and the wrapped value is exactly what the check compares; emitting it first
    /// lets the check read the result rather than build a second copy of the arithmetic.
    ///
    /// A multiply needs no such result, and passes it only because one entry point serves all three
    /// groups; see [`emit_overflow_cond`].
    fn lower_unguarded_overflow(
        &self,
        emitter: &mut HLBlockEmitter<'_>,
        kind: BinaryArithOpKind,
        result: ValueId,
        lhs: ValueId,
        rhs: ValueId,
        bits: usize,
    ) {
        emitter.emit(OpCode::BinaryArithOp {
            kind,
            result,
            lhs,
            rhs,
        });
        emit_no_overflow_assert(emitter, kind, lhs, rhs, Some(result), bits);
    }

    #[allow(clippy::too_many_arguments)]
    fn lower_divmod_guard(
        &self,
        emitter: &mut HLBlockEmitter<'_>,
        condition: ValueId,
        kind: BinaryArithOpKind,
        original_result: ValueId,
        lhs: ValueId,
        rhs: ValueId,
        lhs_type: &Type,
        signed: bool,
    ) {
        let failure = emit_divmod_failure_cond(emitter, lhs, rhs, lhs_type, signed);

        emitter.build_if_else_into(
            failure,
            vec![(original_result, lhs_type.clone())],
            // Divisor is zero: assert condition is false, produce default
            |e| {
                let zero_u1 = e.int_const(1, 0);
                e.emit(OpCode::AssertCmp {
                    kind: CmpKind::Eq,
                    lhs: condition,
                    rhs: zero_u1,
                });
                let default_val = match &lhs_type.expr {
                    TypeExpr::Int(b) => e.int_const(*b, 0),
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
                let zero = e.int_const(1, 0);
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
                let zero = e.int_const(1, 0);
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
            TypeExpr::Int(n) => {
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
                let bound = emitter.int_const(cmp_bits, 1u128 << max_bits);
                let in_range = emitter.ult(v_cmp, bound);
                let oob = emitter.not(in_range);

                emitter.build_if_else_into(
                    oob,
                    vec![],
                    |e| {
                        let zero = e.int_const(1, 0);
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
                let in_range = emitter.ult(value, bound);
                let oob = emitter.not(in_range);

                emitter.build_if_else_into(
                    oob,
                    vec![],
                    |e| {
                        let zero = e.int_const(1, 0);
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
        let in_bounds = emitter.ult(idx_cmp, len_cmp);
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
        bits: usize,
    ) {
        emitter.build_if_else_into(
            failure,
            vec![(original_result, result_type.clone())],
            // Failure: assert condition is false, produce default value
            |e| vec![self.emit_guard_failure_default(e, condition, bits)],
            // Ok: compute the result
            |e| vec![ok_path(e)],
        );
    }

    /// The value a failed guarded operation yields: assert the guard is inactive, then produce the
    /// result type's zero.
    ///
    /// The zero takes no sign, so neither does this. It used to be chosen between `i_const` and
    /// `u_const`, which is why every caller still resolves a signedness of its own — that one
    /// selects the lowering, and never reached anything here but the tag on a zero.
    fn emit_guard_failure_default(
        &self,
        emitter: &mut HLBlockEmitter<'_>,
        condition: ValueId,
        bits: usize,
    ) -> ValueId {
        let zero = emitter.int_const(1, 0);
        emitter.emit(OpCode::AssertCmp {
            kind: CmpKind::Eq,
            lhs: condition,
            rhs: zero,
        });
        emitter.int_const(bits, 0)
    }
}
