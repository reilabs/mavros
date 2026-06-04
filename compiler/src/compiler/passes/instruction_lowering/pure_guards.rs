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
//! - **Lower with shift check** (pure inputs only, can fail): Integer Shl/Shr — validate shift
//!   amount before shifting; Shl also checks overflow so we fail there too.
//! - **Lower with div-zero check** (pure inputs only, can fail): Div/Mod — if divisor==0 assert
//!   !cond and produce 0; else compute.
//! - **Leave untouched here** (side-effectful, constraint-generating, or handled by witness rules):
//!   Store, Call, Assert, AssertCmp, AssertR1C, Constrain, witness Rangecheck, and failable ops with
//!   witness inputs.

use ark_ff::Field as _;

use crate::compiler::{
    analysis::types::FunctionTypeInfo,
    ssa::{
        Instruction, ValueId,
        hlssa::{
            BinaryArithOpKind, CastTarget, CmpKind, MAX_SUPPORTED_SIGNED_BITS,
            MAX_SUPPORTED_UNSIGNED_BITS, OpCode, Type, TypeExpr,
            builder::{HLBlockEmitter, HLEmitter},
        },
    },
};

use super::{InstructionLoweringRule, LoweringContext};

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
                self.lower_guard(emitter, *condition, inner.as_ref().clone(), type_info)
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

    /// Lower a single Guard instruction.
    fn lower_guard(
        &self,
        emitter: &mut HLBlockEmitter<'_>,
        condition: ValueId,
        inner: OpCode,
        type_info: &FunctionTypeInfo,
    ) -> bool {
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
        let sign_shift = emitter.field_const(two_pow(bits));
        let sign_shifted = emitter.mul(sign, sign_shift);
        let signed_value = emitter.sub(value_field, sign_shifted);
        let two = emitter.field_const(ark_bn254::Fr::from(2));
        let two_sign = emitter.mul(two, sign);
        let one = emitter.field_const(ark_bn254::Fr::from(1));
        let factor = emitter.sub(one, two_sign);
        let abs = emitter.mul(signed_value, factor);
        emitter.cast_to(CastTarget::U(bits), abs)
    }

    /// Lower `Guard(cond, shift(lhs, rhs) -> result)`.
    ///
    /// Shifts are only valid for amounts in `[0, bits)`.  The range check must
    /// dominate the shift itself; otherwise LLVM can treat an out-of-range shift
    /// in an inactive guarded branch as poison.
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
            |e| match kind {
                BinaryArithOpKind::Shr => {
                    let result = e.fresh_value();
                    e.emit(OpCode::BinaryArithOp {
                        kind,
                        result,
                        lhs,
                        rhs,
                    });
                    vec![result]
                }
                BinaryArithOpKind::Shl => {
                    vec![self.emit_checked_shl_ok_path(e, condition, lhs, rhs, bits, signed)]
                }
                _ => unreachable!("lower_shift_guard called for non-shift op"),
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

    fn emit_checked_shl_ok_path(
        &self,
        emitter: &mut HLBlockEmitter<'_>,
        condition: ValueId,
        lhs: ValueId,
        rhs: ValueId,
        bits: usize,
        signed: bool,
    ) -> ValueId {
        let result_type = if signed {
            Type {
                expr: TypeExpr::I(bits),
            }
        } else {
            Type {
                expr: TypeExpr::U(bits),
            }
        };

        let shifted = emitter.fresh_value();
        emitter.emit(OpCode::BinaryArithOp {
            kind: BinaryArithOpKind::Shl,
            result: shifted,
            lhs,
            rhs,
        });
        let back = emitter.fresh_value();
        emitter.emit(OpCode::BinaryArithOp {
            kind: BinaryArithOpKind::Shr,
            result: back,
            lhs: shifted,
            rhs,
        });
        let identity_eq = emitter.eq(back, lhs);
        let overflow = emitter.not(identity_eq);

        let result = emitter.build_if_else(
            overflow,
            vec![result_type],
            |e| vec![self.emit_guard_failure_default(e, condition, signed, bits)],
            |_| vec![shifted],
        );
        result[0]
    }

    /// Lower `Guard(cond, div/mod(lhs, rhs) -> result)` for division by zero.
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
        let zero_val = match &lhs_type.expr {
            TypeExpr::U(b) => emitter.u_const(*b, 0),
            TypeExpr::I(b) => emitter.i_const(*b, 0),
            TypeExpr::Field => emitter.field_const(ark_bn254::Fr::from(0u64)),
            _ => unreachable!(),
        };
        let is_zero = emitter.eq(rhs, zero_val);
        let failure = match &lhs_type.expr {
            TypeExpr::I(bits) => {
                let min_val = emitter.i_const(*bits, 1u128 << (*bits - 1));
                let minus_one = emitter.i_const(*bits, bit_mask(*bits));
                let lhs_is_min = emitter.eq(lhs, min_val);
                let rhs_is_minus_one = emitter.eq(rhs, minus_one);
                let signed_overflow = emitter.and(lhs_is_min, rhs_is_minus_one);
                emitter.or(is_zero, signed_overflow)
            }
            _ => is_zero,
        };

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
                    TypeExpr::Field => e.field_const(ark_bn254::Fr::from(0u64)),
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
                let v_cmp = if val_bits == cmp_bits {
                    value
                } else {
                    emitter.cast_to(CastTarget::U(cmp_bits), value)
                };
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
                if max_bits >= <ark_bn254::Fr as ark_ff::PrimeField>::MODULUS_BIT_SIZE as usize {
                    return;
                }

                let bound = emitter.field_const(two_pow(max_bits));
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

    /// Compute the OOB condition: idx >= len(seq). Returns a bool ValueId.
    /// Works for both arrays (static length) and slices (runtime SliceLen).
    fn emit_oob_cond(
        &self,
        emitter: &mut HLBlockEmitter<'_>,
        seq: ValueId,
        index: ValueId,
        type_info: &FunctionTypeInfo,
    ) -> ValueId {
        let seq_type = type_info.get_value_type(seq);
        let len_val = match &seq_type.strip_witness().expr {
            TypeExpr::Array(_, len) => emitter.u_const(32, *len as u128),
            TypeExpr::Slice(_) => emitter.slice_len(seq),
            other => panic!("LowerPureGuards: seq op on non-seq type: {:?}", other),
        };

        let idx_type = type_info.get_value_type(index);
        let idx_as_u32 = if matches!(idx_type.strip_witness().expr, TypeExpr::U(32)) {
            index
        } else {
            emitter.cast_to(CastTarget::U(32), index)
        };
        let in_bounds = emitter.lt(idx_as_u32, len_val);
        emitter.not(in_bounds)
    }

    /// Common pattern: branch on a failure condition. In the fail branch,
    /// assert condition==false and produce a default value. In the ok branch,
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

fn bit_mask(bits: usize) -> u128 {
    if bits == 128 {
        u128::MAX
    } else {
        (1u128 << bits) - 1
    }
}

fn two_pow(exponent: usize) -> ark_bn254::Fr {
    ark_bn254::Fr::from(2).pow([exponent as u64])
}
