use std::collections::HashMap;

use ark_ff::{AdditiveGroup, Field as _};

use crate::compiler::{
    Field,
    analysis::{
        types::{FunctionTypeInfo, TypeInfo},
        value_range::{FunctionValueRanges, IntInterval, ValueRanges},
    },
    block_builder::{HLEmitter, HLInstrBuilder},
    flow_analysis::FlowAnalysis,
    ir::r#type::{Type, TypeExpr},
    pass_manager::{Analysis, AnalysisId, AnalysisStore, Pass},
    ssa::{
        BinaryArithOpKind, BlockId, CastTarget, CmpKind, Endianness, HLBlock, HLSSA, LookupTarget,
        OpCode, Radix, SeqType, ValueId,
    },
};

use num_bigint::BigInt;
use num_traits::{One, Signed};

/// Number of bits needed to represent every value in the interval as a
/// non-negative integer. Returns `None` if the interval may contain a negative
/// value or has no upper bound — in that case the analyzer can't tell us how
/// wide the byte decomposition needs to be.
fn value_range_unsigned_bit_width(r: &IntInterval) -> Option<usize> {
    let lo = r.lo.as_ref()?;
    let hi = r.hi.as_ref()?;
    if lo.is_negative() {
        return None;
    }
    Some(hi.bits() as usize)
}

/// Byte-aligned rangecheck width that's still no wider than `default_bits`.
/// Picks the smallest multiple of 8 that contains the analyzer's proven bound,
/// so the rangecheck stays byte-aligned (no gap-on-MSB lookup) while never
/// exceeding the original width.
fn narrow_rangecheck_width(value_range: &IntInterval, default_bits: usize) -> usize {
    let Some(bound_bits) = value_range_unsigned_bit_width(value_range) else {
        return default_bits;
    };
    if bound_bits == 0 {
        return 1;
    }
    let byte_aligned = ((bound_bits + 7) / 8) * 8;
    byte_aligned.min(default_bits)
}

/// Bound on `q` for `a / b` with `b` strictly positive. Returns the analyzer's
/// derived bound, or `None` if we can't say anything (e.g. divisor's lower
/// bound is 0).
fn quotient_bound(a_range: &IntInterval, b_range: &IntInterval) -> IntInterval {
    let (Some(a_hi), Some(b_lo)) = (a_range.hi.as_ref(), b_range.lo.as_ref()) else {
        return IntInterval::top();
    };
    if !a_range.is_non_negative() || !b_lo.is_positive() {
        return IntInterval::top();
    }
    IntInterval::closed(BigInt::from(0), a_hi / b_lo)
}

/// Bound on `r` for `a % b` (and on `b − r − 1`). With `b ∈ [b.lo, b.hi]`,
/// `r ∈ [0, b.hi − 1]`.
fn remainder_bound(b_range: &IntInterval) -> IntInterval {
    let Some(b_hi) = b_range.hi.as_ref() else {
        return IntInterval::top();
    };
    if !b_hi.is_positive() {
        return IntInterval::top();
    }
    IntInterval::closed(BigInt::from(0), b_hi - BigInt::one())
}

/// Byte-aligned width sufficient to hold every value in `r`'s range as its
/// non-negative absolute value. Used for rangechecks on differences whose
/// magnitude (not signed value) is what we're proving.
fn abs_max_byte_width(r: &IntInterval, default_bits: usize) -> usize {
    let (Some(lo), Some(hi)) = (r.lo.as_ref(), r.hi.as_ref()) else {
        return default_bits;
    };
    let m = lo.abs().max(hi.abs());
    if m.bits() == 0 {
        return 1;
    }
    let byte_aligned = (((m.bits() as usize) + 7) / 8) * 8;
    byte_aligned.min(default_bits)
}

pub struct ExplicitWitness {}

impl Pass for ExplicitWitness {
    fn name(&self) -> &'static str {
        "explicit_witness"
    }

    fn needs(&self) -> Vec<AnalysisId> {
        vec![TypeInfo::id(), ValueRanges::id()]
    }

    fn run(&self, ssa: &mut HLSSA, store: &AnalysisStore) {
        self.do_run(
            ssa,
            store.get::<TypeInfo>(),
            store.get::<ValueRanges>(),
        );
    }

    fn preserves(&self) -> Vec<AnalysisId> {
        vec![FlowAnalysis::id()]
    }
}

impl ExplicitWitness {
    pub fn new() -> Self {
        Self {}
    }

    pub fn do_run(&self, ssa: &mut HLSSA, type_info: &TypeInfo, value_ranges: &ValueRanges) {
        for (function_id, function) in ssa.iter_functions_mut() {
            let function_type_info = type_info.get_function(*function_id);
            let function_value_ranges = value_ranges.get_function(*function_id);
            let mut new_blocks = HashMap::<BlockId, HLBlock>::new();
            for (bid, mut block) in function.take_blocks().into_iter() {
                let mut new_instructions = Vec::new();
                for instruction in block.take_instructions().into_iter() {
                    let b = &mut HLInstrBuilder::new(function, &mut new_instructions);
                    self.process_instruction(
                        b,
                        function_type_info,
                        function_value_ranges,
                        instruction,
                    );
                }
                block.put_instructions(new_instructions);
                new_blocks.insert(bid, block);
            }
            function.put_blocks(new_blocks);
        }
    }

    fn process_instruction(
        &self,
        b: &mut HLInstrBuilder<'_>,
        function_type_info: &FunctionTypeInfo,
        function_value_ranges: &FunctionValueRanges,
        instruction: OpCode,
    ) {
        match instruction {
            OpCode::BinaryArithOp {
                kind: kind @ (BinaryArithOpKind::Add | BinaryArithOpKind::Sub),
                result,
                lhs: l,
                rhs: r,
            } => {
                let l_taint = function_type_info.get_value_type(l).is_witness_of();
                let r_taint = function_type_info.get_value_type(r).is_witness_of();
                if !l_taint && !r_taint {
                    b.push(instruction);
                    return;
                }
                let l_type = function_type_info.get_value_type(l);
                match l_type.strip_witness().expr {
                    TypeExpr::U(bits) => {
                        // Uint add/sub: linear, but needs rangecheck
                        let l_field = b.cast_to_field(l);
                        let r_field = b.cast_to_field(r);
                        let arith_result = match kind {
                            BinaryArithOpKind::Add => b.add(l_field, r_field),
                            BinaryArithOpKind::Sub => b.sub(l_field, r_field),
                            _ => unreachable!(),
                        };
                        let one = b.field_const(Field::ONE);
                        // For Add: result range is l+r (non-negative), can narrow.
                        // For Sub: result range may go negative (field wrap); the helper
                        // returns the default width when lo is negative, so we stay safe.
                        let l_range = function_value_ranges.get(l);
                        let r_range = function_value_ranges.get(r);
                        let result_range = match kind {
                            BinaryArithOpKind::Add => l_range.add(&r_range),
                            BinaryArithOpKind::Sub => l_range.sub(&r_range),
                            _ => unreachable!(),
                        };
                        let rc_bits = narrow_rangecheck_width(&result_range, bits);
                        self.gen_witness_rangecheck_bits(b, arith_result, rc_bits, one);
                        b.push(OpCode::Cast {
                            result,
                            value: arith_result,
                            target: CastTarget::U(bits),
                        });
                    }
                    TypeExpr::I(bits) => {
                        let l_field = b.cast_to_field(l);
                        let r_field = b.cast_to_field(r);
                        let one = b.field_const(Field::ONE);
                        let l_range = function_value_ranges.get(l);
                        let r_range = function_value_ranges.get(r);
                        self.gen_witness_signed_addsub(
                            b, l_field, r_field, bits, kind, one, result, l_taint, r_taint,
                            &l_range, &r_range,
                        );
                    }
                    _ => {
                        // Field add/sub: linear, pass through
                        b.push(instruction);
                    }
                }
            }
            OpCode::Alloc { .. }
            | OpCode::Call { .. }
            | OpCode::Constrain { .. }
            | OpCode::WriteWitness { .. }
            | OpCode::FreshWitness {
                result: _,
                result_type: _,
            } => {
                b.push(instruction);
            }
            OpCode::Cmp {
                kind,
                result,
                lhs,
                rhs,
            } => {
                let l_taint = function_type_info.get_value_type(lhs).is_witness_of();
                let r_taint = function_type_info.get_value_type(rhs).is_witness_of();
                if !(!l_taint && !r_taint) {
                    assert!(function_type_info.get_value_type(rhs).is_numeric());
                    assert!(function_type_info.get_value_type(lhs).is_numeric());
                    match kind {
                        CmpKind::Eq => {
                            let u1 = CastTarget::U(1);
                            // Conditionally cast operands to Field (skip if already Field)
                            let l_field = if function_type_info
                                .get_value_type(lhs)
                                .strip_witness()
                                .is_field()
                            {
                                lhs
                            } else {
                                b.cast_to_field(lhs)
                            };
                            let r_field = if function_type_info
                                .get_value_type(rhs)
                                .strip_witness()
                                .is_field()
                            {
                                rhs
                            } else {
                                b.cast_to_field(rhs)
                            };
                            let lr_diff = b.sub(l_field, r_field);

                            let field_one_for_div = b.field_const(Field::ONE);
                            let div_hint = b.div(field_one_for_div, lr_diff);
                            let div_hint_plain = b.value_of(div_hint);
                            let div_hint_witness = b.write_witness(div_hint_plain);

                            let out_hint = b.eq(lhs, rhs);
                            let out_hint_field = b.cast_to_field(out_hint);
                            let out_hint_plain = b.value_of(out_hint_field);
                            let out_hint_witness = b.write_witness(out_hint_plain);
                            b.push(OpCode::Cast {
                                result,
                                value: out_hint_witness,
                                target: u1,
                            });

                            let result_field = b.cast_to_field(result);
                            let field_one = b.field_const(Field::ONE);
                            let not_res = b.sub(field_one, result_field);

                            b.constrain(lr_diff, div_hint_witness, not_res);
                            let field_zero = b.field_const(Field::ZERO);
                            b.constrain(lr_diff, result_field, field_zero);
                        }
                        CmpKind::Lt => {
                            let l_range = function_value_ranges.get(lhs);
                            let r_range = function_value_ranges.get(rhs);
                            self.lower_witness_lt(
                                b,
                                function_type_info,
                                lhs,
                                rhs,
                                result,
                                l_taint,
                                r_taint,
                                &l_range,
                                &r_range,
                            );
                        }
                    }
                } else {
                    b.push(instruction);
                }
            }
            OpCode::BinaryArithOp {
                kind: BinaryArithOpKind::Mul,
                result: res,
                lhs: l,
                rhs: r,
            } => {
                let l_taint = function_type_info.get_value_type(l).is_witness_of();
                let r_taint = function_type_info.get_value_type(r).is_witness_of();

                if !l_taint && !r_taint {
                    b.push(instruction);
                    return;
                }

                let l_type = function_type_info.get_value_type(l);
                match l_type.strip_witness().expr {
                    TypeExpr::U(bits) => {
                        let l_field = b.cast_to_field(l);
                        let r_field = b.cast_to_field(r);
                        let arith_result = if l_taint && r_taint {
                            let l_pure = b.value_of(l_field);
                            let r_pure = b.value_of(r_field);
                            let hint = b.mul(l_pure, r_pure);
                            let w = b.write_witness(hint);
                            b.constrain(l_field, r_field, w);
                            w
                        } else {
                            b.mul(l_field, r_field)
                        };
                        let one = b.field_const(Field::ONE);
                        let l_range = function_value_ranges.get(l);
                        let r_range = function_value_ranges.get(r);
                        let result_range = l_range.mul(&r_range);
                        let rc_bits = narrow_rangecheck_width(&result_range, bits);
                        self.gen_witness_rangecheck_bits(b, arith_result, rc_bits, one);
                        b.push(OpCode::Cast {
                            result: res,
                            value: arith_result,
                            target: CastTarget::U(bits),
                        });
                    }
                    TypeExpr::I(bits) => {
                        let l_field = b.cast_to_field(l);
                        let r_field = b.cast_to_field(r);
                        let one = b.field_const(Field::ONE);
                        let l_range = function_value_ranges.get(l);
                        let r_range = function_value_ranges.get(r);
                        self.gen_witness_signed_mul(
                            b, l_field, r_field, bits, one, res, l_taint, r_taint,
                            &l_range, &r_range,
                        );
                    }
                    _ => {
                        if l_taint && r_taint {
                            // Field witness*witness: non-linear
                            let l_plain = b.value_of(l);
                            let r_plain = b.value_of(r);
                            let mul_hint = b.mul(l_plain, r_plain);
                            b.push(OpCode::WriteWitness {
                                result: Some(res),
                                value: mul_hint,
                                pinned: false,
                            });
                            b.constrain(l, r, res);
                        } else {
                            // Field witness*pure: linear, pass through
                            b.push(instruction);
                        }
                    }
                }
            }
            OpCode::BinaryArithOp {
                kind: kind @ (BinaryArithOpKind::Div | BinaryArithOpKind::Mod),
                result: res,
                lhs: l,
                rhs: r,
            } => {
                let l_taint = function_type_info.get_value_type(l).is_witness_of();
                let r_taint = function_type_info.get_value_type(r).is_witness_of();

                if !l_taint && !r_taint {
                    b.push(instruction);
                    return;
                }

                let l_type = function_type_info.get_value_type(l);
                match l_type.strip_witness().expr {
                    TypeExpr::I(_) => panic!("Signed div/mod not yet implemented"),
                    TypeExpr::U(bits) => {
                        let one = b.field_const(Field::ONE);
                        let l_range = function_value_ranges.get(l);
                        let r_range = function_value_ranges.get(r);
                        self.gen_witness_divmod(
                            b, l, r, bits, kind, one, res, l_taint, r_taint, &l_range, &r_range,
                        );
                    }
                    TypeExpr::Field => {
                        assert!(
                            kind == BinaryArithOpKind::Div,
                            "Modulo is not defined on field elements"
                        );
                        self.gen_witness_field_div(b, l, r, l_taint, r_taint, res);
                    }
                    _ => unreachable!("DivMod on non-numeric type: {:?}", l_type),
                }
            }
            OpCode::BinaryArithOp {
                kind:
                    kind @ (BinaryArithOpKind::And | BinaryArithOpKind::Or | BinaryArithOpKind::Xor),
                result,
                lhs: l,
                rhs: r,
            } => {
                let l_taint = function_type_info.get_value_type(l).is_witness_of();
                let r_taint = function_type_info.get_value_type(r).is_witness_of();
                if !l_taint && !r_taint {
                    b.push(instruction);
                    return;
                }
                let l_type = function_type_info.get_value_type(l);
                match l_type.strip_witness().expr {
                    TypeExpr::U(1) => {
                        // Bool (u1) witness case: And = mul, Or = a+b - a*b, Xor = a+b - 2*a*b
                        let u1 = CastTarget::U(1);
                        let l_field = b.cast_to_field(l);
                        let r_field = b.cast_to_field(r);
                        match kind {
                            BinaryArithOpKind::And => {
                                // a AND b = a * b
                                if l_taint && r_taint {
                                    let res_hint = b.and(l, r);
                                    let res_hint_field = b.cast_to_field(res_hint);
                                    let res_hint_plain = b.value_of(res_hint_field);
                                    let res_witness = b.write_witness(res_hint_plain);
                                    b.constrain(l_field, r_field, res_witness);
                                    b.push(OpCode::Cast {
                                        result,
                                        value: res_witness,
                                        target: u1,
                                    });
                                } else {
                                    // One pure, one witness: a * b is linear
                                    let product = b.mul(l_field, r_field);
                                    b.push(OpCode::Cast {
                                        result,
                                        value: product,
                                        target: u1,
                                    });
                                }
                            }
                            BinaryArithOpKind::Or => {
                                // a OR b = a + b - a*b
                                let sum = b.add(l_field, r_field);
                                if l_taint && r_taint {
                                    let l_pure = b.value_of(l_field);
                                    let r_pure = b.value_of(r_field);
                                    let prod_hint = b.mul(l_pure, r_pure);
                                    let prod_wit = b.write_witness(prod_hint);
                                    b.constrain(l_field, r_field, prod_wit);
                                    let res = b.sub(sum, prod_wit);
                                    b.push(OpCode::Cast {
                                        result,
                                        value: res,
                                        target: u1,
                                    });
                                } else {
                                    let prod = b.mul(l_field, r_field);
                                    let res = b.sub(sum, prod);
                                    b.push(OpCode::Cast {
                                        result,
                                        value: res,
                                        target: u1,
                                    });
                                }
                            }
                            BinaryArithOpKind::Xor => {
                                // a XOR b = a + b - 2*a*b
                                let sum = b.add(l_field, r_field);
                                let two = b.field_const(Field::from(2));
                                if l_taint && r_taint {
                                    let l_pure = b.value_of(l_field);
                                    let r_pure = b.value_of(r_field);
                                    let prod_hint = b.mul(l_pure, r_pure);
                                    let prod_wit = b.write_witness(prod_hint);
                                    b.constrain(l_field, r_field, prod_wit);
                                    let two_prod = b.mul(two, prod_wit);
                                    let res = b.sub(sum, two_prod);
                                    b.push(OpCode::Cast {
                                        result,
                                        value: res,
                                        target: u1,
                                    });
                                } else {
                                    let prod = b.mul(l_field, r_field);
                                    let two_prod = b.mul(two, prod);
                                    let res = b.sub(sum, two_prod);
                                    b.push(OpCode::Cast {
                                        result,
                                        value: res,
                                        target: u1,
                                    });
                                }
                            }
                            _ => unreachable!(),
                        }
                    }
                    TypeExpr::U(n) => {
                        self.gen_witness_bitwise(b, kind, l, r, l_taint, r_taint, n, result);
                    }
                    _ => {
                        panic!(
                            "Bitwise {:?} on witness operands is only supported for unsigned int, got type {:?}",
                            kind,
                            l_type.strip_witness()
                        );
                    }
                }
            }
            OpCode::BinaryArithOp {
                kind: kind @ (BinaryArithOpKind::Shl | BinaryArithOpKind::Shr),
                lhs: l,
                rhs: r,
                ..
            } => {
                let l_taint = function_type_info.get_value_type(l).is_witness_of();
                let r_taint = function_type_info.get_value_type(r).is_witness_of();
                if l_taint || r_taint {
                    panic!("Shift {:?} on witness operands is not supported", kind);
                }
                b.push(instruction);
            }
            OpCode::Store { ptr, value: _ } => {
                let ptr_taint = function_type_info.get_value_type(ptr).is_witness_of();
                assert!(!ptr_taint);
                b.push(instruction);
            }
            OpCode::Load { result: _, ptr } => {
                let ptr_taint = function_type_info.get_value_type(ptr).is_witness_of();
                assert!(!ptr_taint);
                b.push(instruction);
            }
            OpCode::Assert { value } => {
                let v_type = function_type_info.get_value_type(value);
                let v_taint = v_type.is_witness_of();
                if !v_taint {
                    b.push(instruction);
                    return;
                }
                // assert(v) with witness → constrain(v_field, 1, 1)
                // i.e. v * 1 = 1, meaning v = 1
                let one_field = b.field_const(Field::ONE);
                let v_field = if v_type.strip_witness().is_field() {
                    value
                } else {
                    b.cast_to(CastTarget::Field, value)
                };
                b.constrain(v_field, one_field, one_field);
            }
            OpCode::AssertCmp { kind, lhs: l, rhs: r } => {
                let l_type = function_type_info.get_value_type(l);
                let r_type = function_type_info.get_value_type(r);
                let l_taint = l_type.is_witness_of();
                let r_taint = r_type.is_witness_of();
                if !l_taint && !r_taint {
                    b.push(instruction);
                    return;
                }
                match kind {
                    CmpKind::Eq => {
                        let one = b.field_const(Field::ONE);
                        let l = if l_type.strip_witness().is_field() {
                            l
                        } else {
                            b.cast_to(CastTarget::Field, l)
                        };
                        let r = if r_type.strip_witness().is_field() {
                            r
                        } else {
                            b.cast_to(CastTarget::Field, r)
                        };
                        b.constrain(l, one, r);
                    }
                    CmpKind::Lt => {
                        let r_stripped = function_type_info.get_value_type(r).strip_witness().expr;
                        let (s, is_signed) = match r_stripped {
                            TypeExpr::U(s) => (s, false),
                            TypeExpr::I(s) => (s, true),
                            _ => panic!("ICE: AssertCmp Lt rhs is not an integer type"),
                        };
                        if is_signed {
                            // Signed: fall back to lower_witness_lt + assert result == 1
                            let result = b.fresh_value();
                            let l_range = function_value_ranges.get(l);
                            let r_range = function_value_ranges.get(r);
                            self.lower_witness_lt(
                                b,
                                function_type_info,
                                l,
                                r,
                                result,
                                l_taint,
                                r_taint,
                                &l_range,
                                &r_range,
                            );
                            let result_field = b.cast_to_field(result);
                            let one = b.field_const(Field::ONE);
                            b.constrain(result_field, one, one);
                        } else {
                            // Unsigned: assert lhs < rhs by proving rhs - lhs - 1 ∈ [0, 2^n).
                            // Saves 2 R1C constraints vs computing the boolean result.
                            let l_field = b.cast_to_field(l);
                            let r_field = b.cast_to_field(r);
                            let diff = b.sub(r_field, l_field);
                            let one = b.field_const(Field::ONE);
                            let diff_minus_one = b.sub(diff, one);
                            let diff_plain = b.value_of(diff_minus_one);
                            let diff_wit = b.write_witness(diff_plain);
                            let flag = b.field_const(Field::ONE);
                            self.gen_witness_rangecheck_bits(b, diff_wit, s, flag);
                        }
                    }
                }
            }
            OpCode::AssertR1C { a, b: r1c_b, c } => {
                let a_taint = function_type_info.get_value_type(a).is_witness_of();
                let b_taint = function_type_info.get_value_type(r1c_b).is_witness_of();
                let c_taint = function_type_info.get_value_type(c).is_witness_of();
                if !a_taint && !b_taint && !c_taint {
                    b.push(instruction);
                    return;
                }
                b.constrain(a, r1c_b, c);
            }
            OpCode::NextDCoeff { result: _ } => {
                panic!("ICE: should not be present at this stage");
            }
            OpCode::BumpD {
                matrix: _,
                variable: _,
                sensitivity: _,
            } => {
                panic!("ICE: should not be present at this stage");
            }
            OpCode::ArrayGet {
                result,
                array: arr,
                index: idx,
            } => {
                let arr_taint = function_type_info.get_value_type(arr).is_witness_of();
                let idx_taint = function_type_info.get_value_type(idx).is_witness_of();
                assert!(!arr_taint);
                if !idx_taint {
                    b.push(instruction);
                } else {
                    let flag = b.field_const(Field::from(1));
                    self.gen_witness_array_get(b, function_type_info, arr, idx, result, flag);
                }
            }
            OpCode::ArraySet {
                result: _,
                array: arr,
                index: idx,
                value: _,
            } => {
                let arr_taint = function_type_info.get_value_type(arr).is_witness_of();
                let idx_taint = function_type_info.get_value_type(idx).is_witness_of();
                assert!(!arr_taint);
                assert!(!idx_taint);
                b.push(instruction);
            }
            OpCode::SlicePush {
                dir: _,
                result: _,
                slice: sl,
                values: _,
            } => {
                let slice_taint = function_type_info.get_value_type(sl).is_witness_of();
                assert!(!slice_taint);
                b.push(instruction);
            }
            OpCode::SliceLen {
                result: _,
                slice: sl,
            } => {
                let slice_taint = function_type_info.get_value_type(sl).is_witness_of();
                assert!(!slice_taint);
                b.push(instruction);
            }
            OpCode::Select {
                result: res,
                cond,
                if_t: l,
                if_f: r,
            } => {
                let cond_taint = function_type_info.get_value_type(cond).is_witness_of();
                let l_taint = function_type_info.get_value_type(l).is_witness_of();
                let r_taint = function_type_info.get_value_type(r).is_witness_of();
                // The result is cond * l + (1 - cond) * r = cond * (l - r) + r
                // If cond is pure, this is just a conditional move
                if !cond_taint {
                    b.push(instruction);
                    return;
                }
                // Cast branches to field for arithmetic (they may be U(n)/I(n))
                let l_type = function_type_info.get_value_type(l);
                let r_type = function_type_info.get_value_type(r);
                let l_field = if l_type.strip_witness().is_field() {
                    l
                } else {
                    b.cast_to_field(l)
                };
                let r_field = if r_type.strip_witness().is_field() {
                    r
                } else {
                    b.cast_to_field(r)
                };

                // If both branches are pure, result is a linear combination
                // of cond and constants: cond * (l - r) + r. No constraint needed.
                if !l_taint && !r_taint {
                    let l_sub_r = b.sub(l_field, r_field);
                    // cond * (l - r): l_sub_r is a constant, cond is witness
                    let cond_field = if function_type_info
                        .get_value_type(cond)
                        .strip_witness()
                        .is_field()
                    {
                        cond
                    } else {
                        b.cast_to_field(cond)
                    };
                    let cond_times_diff = b.mul(l_sub_r, cond_field);
                    // result = cond * (l - r) + r
                    b.push(OpCode::BinaryArithOp {
                        kind: BinaryArithOpKind::Add,
                        result: res,
                        lhs: cond_times_diff,
                        rhs: r_field,
                    });
                    return;
                }
                // At least one branch is witness: full lowering with constraint
                let select_witness = b.select(cond, l, r);
                let select_plain = b.value_of(select_witness);
                let is_field = l_type.strip_witness().is_field();
                let select_hint = if is_field {
                    select_plain
                } else {
                    b.cast_to_field(select_plain)
                };
                if is_field {
                    b.push(OpCode::WriteWitness {
                        result: Some(res),
                        value: select_hint,
                        pinned: false,
                    });
                } else {
                    // WriteWitness value is Field, but res must keep its original
                    // type (e.g. u32) for downstream consumers like MkSeq.
                    // Use a fresh ID for the WriteWitness result, then cast back.
                    let ww_res = b.fresh_value();
                    b.push(OpCode::WriteWitness {
                        result: Some(ww_res),
                        value: select_hint,
                        pinned: false,
                    });
                    let cast_target = match l_type.strip_witness().expr {
                        TypeExpr::U(n) => CastTarget::U(n),
                        TypeExpr::I(n) => CastTarget::I(n),
                        _ => unreachable!("non-field scalar must be U or I"),
                    };
                    b.push(OpCode::Cast {
                        result: res,
                        value: ww_res,
                        target: cast_target,
                    });
                }
                // Goal is to assert 0 = cond * l + (1 - cond) * r - res
                // This is equivalent to 0 = cond * (l - r) + r - res = cond * (l - r) - (res - r)
                let l_sub_r = b.sub(l_field, r_field);
                let res_field = b.cast_to_field(res);
                let res_sub_r = b.sub(res_field, r_field);
                let cond_type = function_type_info.get_value_type(cond);
                let cond_field = if cond_type.strip_witness().is_field() {
                    cond
                } else {
                    b.cast_to_field(cond)
                };
                b.constrain(cond_field, l_sub_r, res_sub_r);
            }

            OpCode::MkSeq {
                result: _,
                elems: _,
                seq_type: _,
                elem_type: _,
            } => {
                b.push(instruction);
            }
            OpCode::MkTuple { .. } => {
                b.push(instruction);
            }
            OpCode::Cast {
                result: _,
                value: _,
                target: _,
            } => {
                b.push(instruction);
            }
            OpCode::Truncate {
                result,
                value,
                to_bits,
                from_bits: _,
            } => {
                let i_taint = function_type_info.get_value_type(value).is_witness_of();
                if !i_taint {
                    b.push(instruction);
                } else {
                    let one = b.field_const(Field::ONE);
                    let value_type = function_type_info.get_value_type(value);
                    let value_field = if value_type.strip_witness().is_field() {
                        value
                    } else {
                        b.cast_to_field(value)
                    };
                    let value_range = function_value_ranges.get(value);
                    self.gen_witness_truncate(
                        b, value_field, to_bits, one, result, &value_range,
                    );
                }
            }
            OpCode::SExt {
                result,
                value,
                from_bits,
                to_bits,
            } => {
                let value_type = function_type_info.get_value_type(value);
                let i_taint = value_type.is_witness_of();
                let one = b.field_const(Field::ONE);
                let value_field = if value_type.strip_witness().is_field() {
                    value
                } else {
                    b.cast_to_field(value)
                };
                let value_range = function_value_ranges.get(value);
                self.gen_sext(
                    b, value_field, from_bits, to_bits, one, i_taint, result, &value_range,
                );
            }
            OpCode::Not { result, value } => {
                let value_type = function_type_info.get_value_type(value);
                let (s, cast_target) = match &value_type.strip_witness().expr {
                    TypeExpr::U(s) => (*s, CastTarget::U(*s)),
                    TypeExpr::I(s) => (*s, CastTarget::I(*s)),
                    e => todo!("Unsupported type for negation: {:?}", e),
                };
                let ones = b.field_const(Field::from((1u128 << s) - 1));
                let casted = b.cast_to(CastTarget::Field, value);
                let subbed = b.sub(ones, casted);
                b.push(OpCode::Cast {
                    result,
                    value: subbed,
                    target: cast_target,
                });
            }
            OpCode::ToBits {
                result: _,
                value: i,
                endianness: _,
                count: _,
            } => {
                let i_taint = function_type_info.get_value_type(i).is_witness_of();
                assert!(!i_taint); // Only handle pure input case for now
                b.push(instruction);
            }
            OpCode::ToRadix {
                result,
                value,
                radix,
                endianness,
                count,
            } => {
                let value_taint = function_type_info.get_value_type(value).is_witness_of();
                if !value_taint {
                    b.push(instruction);
                } else {
                    assert!(endianness == Endianness::Little);
                    let pure_value = b.value_of(value);
                    let hint = b.fresh_value();
                    b.push(OpCode::ToRadix {
                        result: hint,
                        value: pure_value,
                        radix,
                        endianness: Endianness::Little,
                        count,
                    });
                    let mut witnesses = vec![ValueId(0); count];
                    let mut current_sum = b.field_const(Field::ZERO);
                    let radix_val = match radix {
                        Radix::Bytes => b.field_const(Field::from(256)),
                        Radix::Dyn(radix) => b.cast_to(CastTarget::Field, radix),
                    };
                    let rangecheck_type = match radix {
                        Radix::Bytes => LookupTarget::Rangecheck(8),
                        Radix::Dyn(radix) => LookupTarget::DynRangecheck(radix),
                    };
                    // TODO this should probably be an SSA loop for codesize reasons.
                    for i in (0..count).rev() {
                        let idx = b.u_const(32, i as u128);
                        let byte = b.array_get(hint, idx);
                        let byte_field = b.cast_to_field(byte);
                        let byte_wit = b.write_witness(byte_field);
                        let one = b.field_const(Field::from(1));
                        b.lookup_rngchk(rangecheck_type, byte_wit, one);
                        let shift_prev_res = b.mul(current_sum, radix_val);
                        current_sum = b.add(shift_prev_res, byte_wit);
                        witnesses[i] = byte_wit;
                    }
                    let constrain_one = b.field_const(Field::from(1));
                    b.constrain(current_sum, constrain_one, value);
                    b.push(OpCode::MkSeq {
                        result,
                        elems: witnesses,
                        seq_type: SeqType::Array(count),
                        elem_type: Type::witness_of(Type::field()),
                    });
                }
            }

            OpCode::MemOp { kind: _, value: _ } => {
                b.push(instruction);
            }
            OpCode::MulConst {
                result: _,
                const_val: _,
                var: _,
            } => {
                b.push(instruction);
            }
            OpCode::Rangecheck { value, max_bits } => {
                let v_taint = function_type_info.get_value_type(value).is_witness_of();
                if !v_taint {
                    b.push(instruction);
                } else {
                    let one = b.field_const(Field::from(1));
                    let value_range = function_value_ranges.get(value);
                    let rc_bits = narrow_rangecheck_width(&value_range, max_bits);
                    self.gen_witness_rangecheck_bits(b, value, rc_bits, one);
                }
            }
            OpCode::ReadGlobal {
                result: _,
                offset: _,
                result_type: _,
            } => {
                b.push(instruction);
            }
            OpCode::Lookup { .. } => {
                b.push(instruction);
            }
            OpCode::DLookup { .. } => {
                b.push(instruction);
            }
            OpCode::Todo { .. } => {
                b.push(instruction);
            }
            OpCode::TupleProj {
                result: _,
                tuple,
                idx: _,
            } => {
                let tuple_taint = function_type_info.get_value_type(tuple).is_witness_of();
                assert!(!tuple_taint);
                b.push(instruction);
            }
            OpCode::InitGlobal { .. }
            | OpCode::DropGlobal { .. }
            | OpCode::ValueOf { .. }
            | OpCode::Const { .. } => {
                b.push(instruction);
            }
            OpCode::Spread {
                result,
                value,
                bits,
            } => {
                let is_witness = function_type_info.get_value_type(value).is_witness_of();
                if !is_witness {
                    b.push(instruction);
                } else {
                    let one = b.field_const(Field::ONE);
                    let value_pure = b.value_of(value);
                    let input_field = b.cast_to_field(value);
                    // Compute spread hint (pure) and write as witness
                    let spread_hint = b.spread(value_pure, bits);
                    let spread_hint_field = b.cast_to_field(spread_hint);
                    let spread_wit = b.write_witness(spread_hint_field);
                    // Constrain via lookup table
                    b.lookup_spread(bits, input_field, spread_wit, one);
                    // Bind the original result to the spread witness
                    b.push(OpCode::Cast {
                        result,
                        value: spread_wit,
                        target: CastTarget::U(bits as usize * 2),
                    });
                }
            }
            OpCode::Unspread {
                result_odd,
                result_even,
                value,
                bits,
            } => {
                let is_witness = function_type_info.get_value_type(value).is_witness_of();
                if !is_witness {
                    b.push(instruction);
                } else {
                    let one = b.field_const(Field::ONE);
                    let two = b.field_const(Field::from(2));
                    let value_pure = b.value_of(value);
                    let value_field = b.cast_to_field(value);
                    let (odd_hint, even_hint) = b.unspread(value_pure, bits);
                    // Write odd as witness with spread lookup
                    let odd_field = b.cast_to_field(odd_hint);
                    let odd_wit = b.write_witness(odd_field);
                    let odd_spread_hint = b.spread(odd_hint, bits);
                    let odd_spread_field = b.cast_to_field(odd_spread_hint);
                    let odd_spread_wit = b.write_witness(odd_spread_field);
                    b.lookup_spread(bits, odd_wit, odd_spread_wit, one);
                    // Write even as witness with spread lookup
                    let even_field = b.cast_to_field(even_hint);
                    let even_wit = b.write_witness(even_field);
                    // Derive even_spread algebraically: value - 2 * spread(odd)
                    let two_odd_spread = b.mul(two, odd_spread_wit);
                    let even_spread = b.sub(value_field, two_odd_spread);
                    b.lookup_spread(bits, even_wit, even_spread, one);
                    // Bind results
                    b.push(OpCode::Cast {
                        result: result_odd,
                        value: odd_wit,
                        target: CastTarget::U(bits as usize),
                    });
                    b.push(OpCode::Cast {
                        result: result_even,
                        value: even_wit,
                        target: CastTarget::U(bits as usize),
                    });
                }
            }
            OpCode::Guard { condition, inner } => {
                self.lower_guard(b, function_type_info, function_value_ranges, condition, *inner);
            }
        }
    }

    fn lower_guard(
        &self,
        b: &mut HLInstrBuilder<'_>,
        function_type_info: &FunctionTypeInfo,
        function_value_ranges: &FunctionValueRanges,
        condition: ValueId,
        inner: OpCode,
    ) {
        match inner {
            OpCode::Assert { value: v } => {
                // Guard(cond, Assert(v)) → constrain(cond_field, v_field, cond_field)
                // i.e., cond * v == cond, meaning if cond then v == 1
                let v_type = function_type_info.get_value_type(v);
                let cond_type = function_type_info.get_value_type(condition);
                let cond_field = if cond_type.strip_witness().is_field() {
                    condition
                } else {
                    b.cast_to_field(condition)
                };
                let v_field = if v_type.strip_witness().is_field() {
                    v
                } else {
                    b.cast_to(CastTarget::Field, v)
                };
                b.constrain(cond_field, v_field, cond_field);
            }
            OpCode::AssertCmp {
                kind: CmpKind::Eq,
                lhs: l,
                rhs: r,
            } => {
                // Guard(cond, AssertCmp(Eq, l, r)) → constrain(cond_field, l_sub_r, zero)
                // This asserts cond * (l - r) = 0, i.e., if cond then l == r
                let l_type = function_type_info.get_value_type(l);
                let r_type = function_type_info.get_value_type(r);
                let cond_type = function_type_info.get_value_type(condition);
                let cond_field = if cond_type.strip_witness().is_field() {
                    condition
                } else {
                    b.cast_to_field(condition)
                };
                let l_field = if l_type.strip_witness().is_field() {
                    l
                } else {
                    b.cast_to(CastTarget::Field, l)
                };
                let r_field = if r_type.strip_witness().is_field() {
                    r
                } else {
                    b.cast_to(CastTarget::Field, r)
                };
                let diff = b.sub(l_field, r_field);
                let zero = b.field_const(Field::ZERO);
                b.constrain(cond_field, diff, zero);
            }
            OpCode::Store { ptr, value: v } => {
                // Guard(cond, Store(ptr, v)) → old = Load(ptr); new = Select(cond, v, old); Store(ptr, new)
                let old_value = b.load(ptr);
                let new_value = b.select(condition, v, old_value);
                b.store(ptr, new_value);
            }
            OpCode::Truncate {
                result,
                value,
                to_bits,
                from_bits: _,
            } => {
                let cond_type = function_type_info.get_value_type(condition);
                let cond_field = if cond_type.strip_witness().is_field() {
                    condition
                } else {
                    b.cast_to_field(condition)
                };
                let value_range = function_value_ranges.get(value);
                self.gen_witness_truncate(b, value, to_bits, cond_field, result, &value_range);
            }
            OpCode::SExt {
                result,
                value,
                from_bits,
                to_bits,
            } => {
                let cond_type = function_type_info.get_value_type(condition);
                let cond_field = if cond_type.strip_witness().is_field() {
                    condition
                } else {
                    b.cast_to_field(condition)
                };
                let value_type = function_type_info.get_value_type(value);
                let value_field = if value_type.strip_witness().is_field() {
                    value
                } else {
                    b.cast_to_field(value)
                };
                let value_range = function_value_ranges.get(value);
                self.gen_sext(
                    b, value_field, from_bits, to_bits, cond_field, true, result, &value_range,
                );
            }
            OpCode::BinaryArithOp {
                kind: kind @ (BinaryArithOpKind::Add | BinaryArithOpKind::Sub),
                result,
                lhs: l,
                rhs: r,
            } => {
                let l_taint = function_type_info.get_value_type(l).is_witness_of();
                let r_taint = function_type_info.get_value_type(r).is_witness_of();
                let l_type = function_type_info.get_value_type(l);
                match l_type.strip_witness().expr {
                    TypeExpr::U(bits) if l_taint || r_taint => {
                        let cond_field = self.ensure_field(b, function_type_info, condition);
                        let l_field = b.cast_to_field(l);
                        let r_field = b.cast_to_field(r);
                        let arith_result = match kind {
                            BinaryArithOpKind::Add => b.add(l_field, r_field),
                            BinaryArithOpKind::Sub => b.sub(l_field, r_field),
                            _ => unreachable!(),
                        };
                        let l_range = function_value_ranges.get(l);
                        let r_range = function_value_ranges.get(r);
                        let result_range = match kind {
                            BinaryArithOpKind::Add => l_range.add(&r_range),
                            BinaryArithOpKind::Sub => l_range.sub(&r_range),
                            _ => unreachable!(),
                        };
                        let rc_bits = narrow_rangecheck_width(&result_range, bits);
                        self.gen_witness_rangecheck_bits(b, arith_result, rc_bits, cond_field);
                        b.push(OpCode::Cast {
                            result,
                            value: arith_result,
                            target: CastTarget::U(bits),
                        });
                    }
                    TypeExpr::I(bits) if l_taint || r_taint => {
                        let cond_field = self.ensure_field(b, function_type_info, condition);
                        let l_field = b.cast_to_field(l);
                        let r_field = b.cast_to_field(r);
                        let l_range = function_value_ranges.get(l);
                        let r_range = function_value_ranges.get(r);
                        self.gen_witness_signed_addsub(
                            b, l_field, r_field, bits, kind, cond_field, result, l_taint, r_taint,
                            &l_range, &r_range,
                        );
                    }
                    _ => {
                        // Pure or field: linear, emit unconditionally
                        b.push(inner);
                    }
                }
            }
            OpCode::BinaryArithOp {
                kind: BinaryArithOpKind::Mul,
                result: res,
                lhs: l,
                rhs: r,
            } => {
                let l_taint = function_type_info.get_value_type(l).is_witness_of();
                let r_taint = function_type_info.get_value_type(r).is_witness_of();
                let l_type = function_type_info.get_value_type(l);
                match l_type.strip_witness().expr {
                    TypeExpr::U(bits) if l_taint || r_taint => {
                        let cond_field = self.ensure_field(b, function_type_info, condition);
                        let l_field = b.cast_to_field(l);
                        let r_field = b.cast_to_field(r);
                        let arith_result = if l_taint && r_taint {
                            let l_pure = b.value_of(l_field);
                            let r_pure = b.value_of(r_field);
                            let hint = b.mul(l_pure, r_pure);
                            let w = b.write_witness(hint);
                            b.constrain(l_field, r_field, w);
                            w
                        } else {
                            b.mul(l_field, r_field)
                        };
                        let l_range = function_value_ranges.get(l);
                        let r_range = function_value_ranges.get(r);
                        let result_range = l_range.mul(&r_range);
                        let rc_bits = narrow_rangecheck_width(&result_range, bits);
                        self.gen_witness_rangecheck_bits(b, arith_result, rc_bits, cond_field);
                        b.push(OpCode::Cast {
                            result: res,
                            value: arith_result,
                            target: CastTarget::U(bits),
                        });
                    }
                    TypeExpr::I(bits) if l_taint || r_taint => {
                        let cond_field = self.ensure_field(b, function_type_info, condition);
                        let l_field = b.cast_to_field(l);
                        let r_field = b.cast_to_field(r);
                        let l_range = function_value_ranges.get(l);
                        let r_range = function_value_ranges.get(r);
                        self.gen_witness_signed_mul(
                            b, l_field, r_field, bits, cond_field, res, l_taint, r_taint,
                            &l_range, &r_range,
                        );
                    }
                    _ if l_taint && r_taint => {
                        // Field witness*witness under guard: non-linear, lower it
                        let l_plain = b.value_of(l);
                        let r_plain = b.value_of(r);
                        let mul_hint = b.mul(l_plain, r_plain);
                        b.push(OpCode::WriteWitness {
                            result: Some(res),
                            value: mul_hint,
                            pinned: false,
                        });
                        b.constrain(l, r, res);
                    }
                    _ => {
                        // Pure or field witness*pure: linear, emit unconditionally
                        b.push(inner);
                    }
                }
            }
            OpCode::BinaryArithOp {
                kind: kind @ (BinaryArithOpKind::Div | BinaryArithOpKind::Mod),
                result: res,
                lhs: l,
                rhs: r,
            } => {
                let l_taint = function_type_info.get_value_type(l).is_witness_of();
                let r_taint = function_type_info.get_value_type(r).is_witness_of();
                let l_type = function_type_info.get_value_type(l);
                match l_type.strip_witness().expr {
                    TypeExpr::I(_) => panic!("Signed div/mod not yet implemented"),
                    TypeExpr::U(bits) if l_taint || r_taint => {
                        let cond_field = self.ensure_field(b, function_type_info, condition);
                        let l_range = function_value_ranges.get(l);
                        let r_range = function_value_ranges.get(r);
                        self.gen_witness_divmod(
                            b, l, r, bits, kind, cond_field, res, l_taint, r_taint,
                            &l_range, &r_range,
                        );
                    }
                    TypeExpr::U(_) => {
                        // Both pure: emit unconditionally
                        b.push(inner);
                    }
                    TypeExpr::Field => {
                        assert!(
                            kind == BinaryArithOpKind::Div,
                            "Modulo is not defined on field elements"
                        );
                        if l_taint || r_taint {
                            // Field div under guard with witnesses — not yet supported
                            panic!("Guarded field division with witnesses not yet supported");
                        }
                        b.push(inner);
                    }
                    _ => unreachable!("DivMod on non-numeric type: {:?}", l_type),
                }
            }
            OpCode::Cast { .. }
            | OpCode::Const { .. }
            | OpCode::MkSeq { .. }
            | OpCode::MkTuple { .. }
            | OpCode::TupleProj { .. } => {
                // Pure data construction/access — no constraints. Emit unconditionally.
                b.push(inner);
            }
            OpCode::ArrayGet {
                result,
                array: arr,
                index: idx,
            } => {
                let arr_taint = function_type_info.get_value_type(arr).is_witness_of();
                let idx_taint = function_type_info.get_value_type(idx).is_witness_of();
                assert!(!arr_taint);
                if !idx_taint {
                    b.push(inner);
                } else {
                    let flag = self.ensure_field(b, function_type_info, condition);
                    self.gen_witness_array_get(b, function_type_info, arr, idx, result, flag);
                }
            }
            OpCode::ArraySet { .. } => {
                panic!("ArraySet inside Guard not supported yet: {:?}", inner);
            }
            OpCode::Rangecheck { value, max_bits } => {
                // Guard(cond, Rangecheck(value, max_bits)) → conditional rangecheck
                // The condition becomes the lookup flag
                let cond_type = function_type_info.get_value_type(condition);
                let cond_field = if cond_type.strip_witness().is_field() {
                    condition
                } else {
                    b.cast_to_field(condition)
                };
                let value_range = function_value_ranges.get(value);
                let rc_bits = narrow_rangecheck_width(&value_range, max_bits);
                self.gen_witness_rangecheck_bits(b, value, rc_bits, cond_field);
            }
            inner => {
                panic!("unrecognized op inside Guard: {:?}", inner);
            }
        }
    }

    /// Generates a canonical byte decomposition of a field element and returns
    /// the truncated value recovered from the low bytes.
    ///
    /// Algorithm:
    /// 1. Decompose `value` into 32 bytes (big-endian), range-check each byte
    /// 2. Constrain reconstruction equals original value
    /// 3. Reconstruct hi (upper 128 bits) and lo (lower 128 bits)
    /// 4. Compute borrow hint: borrow = 1 if lo > modulusLoMinusOne
    /// 5. Constrain borrow is boolean
    /// 6. Compute resultLo = modulusLoMinusOne - lo + borrow * 2^128 + 1
    /// 7. Compute resultHi = modulusHi - hi - borrow
    /// 8. Range-check resultHi and resultLo to 128 bits (proves value < modulus)
    /// 9. Return truncated value from low `to_bits/8` bytes
    fn gen_witness_truncate(
        &self,
        b: &mut HLInstrBuilder<'_>,
        value: ValueId,
        to_bits: usize,
        flag: ValueId,
        result: ValueId,
        value_range: &IntInterval,
    ) {
        assert!(to_bits <= 256);

        // Tier 1: value already fits in `to_bits`, so `value mod 2^to_bits = value`.
        // Emit a no-op cast and let downstream consumers see `value` directly.
        if value_range.fits_in_unsigned_bits(to_bits) {
            b.push(OpCode::Cast {
                result,
                value,
                target: CastTarget::Nop,
            });
            return;
        }

        // Tier 2: value provably fits in `bound_bits ≤ 248` bits → its byte
        // decomposition is unique below `256^m < p`, so we can skip the
        // canonical-form check. Cost: m byte rangechecks (vs 32 + 32 lookups
        // for the canonical path).
        if let Some(bound_bits) = value_range_unsigned_bit_width(value_range) {
            if bound_bits <= 248 {
                let m_from_bound = (bound_bits + 7) / 8;
                // Need at least enough bytes to cover the truncation output.
                let m_from_to_bits = (to_bits + 7) / 8;
                let m = m_from_bound.max(m_from_to_bits).max(1);
                self.gen_witness_truncate_short(b, value, to_bits, flag, result, m);
                return;
            }
        }

        // Tier 3: value could be a full-width Field element. Use the canonical
        // form check.

        // BN254 modulus: p = 0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000001
        // modulusHi = upper 128 bits
        let modulus_hi = b.field_const(Field::from(0x30644e72e131a029b85045b68181585du128));
        // modulusLoMinusOne = lower 128 bits - 1
        let modulus_lo_m1 = b.field_const(Field::from(0x2833e84879b9709143e1f593f0000000u128));
        let two_to_8 = b.field_const(Field::from(256u128));
        let two_to_64 = b.field_const(Field::from(2u128).pow([64u64]));
        let two_to_128 = b.field_const(Field::from(2u128).pow([128u64]));
        let zero = b.field_const(Field::ZERO);
        let one = b.field_const(Field::ONE);

        // Step 1: Decompose value into 32 bytes (big-endian)
        let pure_value = b.value_of(value);
        let bytes_arr = b.fresh_value();
        b.push(OpCode::ToRadix {
            result: bytes_arr,
            value: pure_value,
            radix: Radix::Bytes,
            endianness: Endianness::Big,
            count: 32,
        });

        // Extract each byte, write as witness, range-check.
        // Simultaneously accumulate 4 x 64-bit limbs and the full 256-bit sum.
        // Witness only 31 bytes; compute byte[31] = value - partial*256 (saves 1 witness + 1 constraint).
        let mut bytes = Vec::with_capacity(32);
        let mut limbs = [zero; 4]; // 64-bit limbs, big-endian
        let mut full_sum = zero;
        for i in 0..31 {
            let idx = b.u_const(32, i as u128);
            let byte = b.array_get(bytes_arr, idx);
            let byte_field = b.cast_to_field(byte);
            let byte_wit = b.write_witness(byte_field);
            b.lookup_rngchk_8(byte_wit, flag);
            bytes.push(byte_wit);

            let limb_idx = i / 8;
            let shifted_limb = b.mul(limbs[limb_idx], two_to_8);
            limbs[limb_idx] = b.add(shifted_limb, byte_wit);

            let shifted_full = b.mul(full_sum, two_to_8);
            full_sum = b.add(shifted_full, byte_wit);
        }

        // Compute last byte as value - partial*256; rangecheck proves it's a byte,
        // implicitly constraining reconstruction (no separate constraint needed).
        let full_sum_shifted = b.mul(full_sum, two_to_8);
        let lsb = b.sub(value, full_sum_shifted);
        b.lookup_rngchk_8(lsb, flag);
        bytes.push(lsb);

        let shifted_limb = b.mul(limbs[3], two_to_8);
        limbs[3] = b.add(shifted_limb, lsb);

        // full_sum is no longer needed (reconstruction is implicit)

        // Step 3: Reconstruct hi (limbs[0..2]) and lo (limbs[2..4]) as 128-bit values
        let hi_upper = b.mul(limbs[0], two_to_64);
        let hi = b.add(hi_upper, limbs[1]);
        let lo_upper = b.mul(limbs[2], two_to_64);
        let lo = b.add(lo_upper, limbs[3]);

        // Step 4: Compute borrow hint via 2-limb comparison
        // borrow = 1 if lo > modulusLoMinusOne, i.e. (limb2, limb3) > (mod_limb2, mod_limb3)
        // = mod_limb2 < limb2 || (mod_limb2 == limb2 && mod_limb3 < limb3)
        let limb2_pure = b.value_of(limbs[2]);
        let limb3_pure = b.value_of(limbs[3]);
        let limb2_u64 = b.cast_to(CastTarget::U(64), limb2_pure);
        let limb3_u64 = b.cast_to(CastTarget::U(64), limb3_pure);
        let mod_limb2 = b.u_const(64, 0x2833e84879b97091u64 as u128);
        let mod_limb3 = b.u_const(64, 0x43e1f593f0000000u64 as u128);
        let hi_lt = b.lt(mod_limb2, limb2_u64);
        let hi_eq = b.eq(mod_limb2, limb2_u64);
        let lo_lt = b.lt(mod_limb3, limb3_u64);
        let hi_eq_f = b.cast_to_field(hi_eq);
        let lo_lt_f = b.cast_to_field(lo_lt);
        let hi_eq_and_lo_lt = b.mul(hi_eq_f, lo_lt_f);
        let hi_lt_f = b.cast_to_field(hi_lt);
        let borrow_hint = b.add(hi_lt_f, hi_eq_and_lo_lt);
        let borrow_wit = b.write_witness(borrow_hint);

        // Step 5: Constrain borrow is boolean: borrow * borrow = borrow
        b.constrain(borrow_wit, borrow_wit, borrow_wit);

        // Step 6: Compute resultLo = modulusLoMinusOne - lo + borrow * 2^128
        let borrow_shift = b.mul(borrow_wit, two_to_128);
        let tmp1 = b.sub(modulus_lo_m1, lo);
        let result_lo = b.add(tmp1, borrow_shift);

        // Step 7: Compute resultHi = modulusHi - hi - borrow
        let tmp3 = b.sub(modulus_hi, hi);
        let result_hi = b.sub(tmp3, borrow_wit);

        // Step 8: Range-check resultHi and resultLo to 128 bits
        // This proves the decomposition is canonical (value < field modulus)
        self.gen_witness_rangecheck_bits(b, result_hi, 128, flag);
        self.gen_witness_rangecheck_bits(b, result_lo, 128, flag);

        // Step 9: Recover truncated value from low bytes of the decomposition
        // In big-endian, the low bytes are bytes[start..32], where the first may be partial.
        let full_bytes = to_bits / 8;
        let partial_bits = to_bits % 8;
        let start = 32 - full_bytes - if partial_bits > 0 { 1 } else { 0 };
        let mut trunc_val = zero;
        for i in start..32 {
            let elem = if i == start && partial_bits > 0 {
                // Non-full byte: split into hi/lo and use lo
                self.split_partial_byte(b, bytes[i], partial_bits, flag)
            } else {
                bytes[i]
            };
            let shifted = b.mul(trunc_val, two_to_8);
            if i == 31 {
                // Last iteration: use the caller's result ValueId
                b.push(OpCode::BinaryArithOp {
                    kind: BinaryArithOpKind::Add,
                    result,
                    lhs: shifted,
                    rhs: elem,
                });
            } else {
                trunc_val = b.add(shifted, elem);
            }
        }
    }

    /// Truncate when the analyzer has proven `value` fits in `m` bytes for
    /// `m ≤ 31` (so `256^m < p`). The byte decomposition is automatically
    /// canonical — no modulus comparison needed.
    fn gen_witness_truncate_short(
        &self,
        b: &mut HLInstrBuilder<'_>,
        value: ValueId,
        to_bits: usize,
        flag: ValueId,
        result: ValueId,
        m: usize,
    ) {
        debug_assert!(m >= 1 && m <= 31);
        let two_to_8 = b.field_const(Field::from(256u128));
        let zero = b.field_const(Field::ZERO);

        // Decompose value into m bytes (big-endian).
        let pure_value = b.value_of(value);
        let bytes_arr = b.fresh_value();
        b.push(OpCode::ToRadix {
            result: bytes_arr,
            value: pure_value,
            radix: Radix::Bytes,
            endianness: Endianness::Big,
            count: m,
        });

        // Witness and rangecheck the upper m-1 bytes; derive the LSB byte
        // from `value − partial * 256` so the reconstruction is implicit.
        let mut bytes = Vec::with_capacity(m);
        let mut partial = zero;
        for i in 0..m - 1 {
            let idx = b.u_const(32, i as u128);
            let byte = b.array_get(bytes_arr, idx);
            let byte_field = b.cast_to_field(byte);
            let byte_wit = b.write_witness(byte_field);
            b.lookup_rngchk_8(byte_wit, flag);
            bytes.push(byte_wit);

            let shifted = b.mul(partial, two_to_8);
            partial = b.add(shifted, byte_wit);
        }
        let partial_shifted = b.mul(partial, two_to_8);
        let lsb = b.sub(value, partial_shifted);
        b.lookup_rngchk_8(lsb, flag);
        bytes.push(lsb);

        // Recover the truncated value from the bottom bytes.
        let full_bytes = to_bits / 8;
        let partial_bits = to_bits % 8;
        let start = m - full_bytes - if partial_bits > 0 { 1 } else { 0 };
        let mut trunc_val = zero;
        for i in start..m {
            let elem = if i == start && partial_bits > 0 {
                self.split_partial_byte(b, bytes[i], partial_bits, flag)
            } else {
                bytes[i]
            };
            let shifted = b.mul(trunc_val, two_to_8);
            if i == m - 1 {
                b.push(OpCode::BinaryArithOp {
                    kind: BinaryArithOpKind::Add,
                    result,
                    lhs: shifted,
                    rhs: elem,
                });
            } else {
                trunc_val = b.add(shifted, elem);
            }
        }
    }

    /// Split a byte witness into hi (`8 - lo_size` bits) and lo (`lo_size` bits).
    /// Rangechecks both parts via gap checks to 8 bits, constrains the split
    /// correctness (`hi * 2^lo_size + lo = byte_wit`), and returns `lo_wit`
    /// for use in reconstruction.
    fn split_partial_byte(
        &self,
        b: &mut HLInstrBuilder<'_>,
        byte_wit: ValueId,
        lo_size: usize,
        flag: ValueId,
    ) -> ValueId {
        let hi_size = 8 - lo_size;
        let two_to_lo = b.field_const(Field::from(1u128 << lo_size));
        let zero = b.field_const(Field::ZERO);

        // Compute lo and hi hints from the byte value
        let byte_pure = b.value_of(byte_wit);
        let lo_hint = b.truncate(byte_pure, lo_size, 254);
        let lo_hint_field = b.cast_to_field(lo_hint);
        let hi_pre = b.sub(byte_pure, lo_hint_field);
        let hi_hint = b.div(hi_pre, two_to_lo);

        let hi_wit = b.write_witness(hi_hint);

        // Rangecheck hi: prove 2^hi_size - 1 - hi fits in 8 bits
        let hi_bound = b.field_const(Field::from((1u128 << hi_size) - 1));
        let hi_gap = b.sub(hi_bound, hi_wit);
        b.lookup_rngchk_8(hi_gap, flag);

        // Compute lo = byte_wit - hi * 2^lo_size (saves 1 witness + 1 constraint)
        let hi_shifted = b.mul(hi_wit, two_to_lo);
        let lo = b.sub(byte_wit, hi_shifted);

        // Rangecheck lo: prove 2^lo_size - 1 - lo fits in 8 bits
        let lo_bound = b.field_const(Field::from((1u128 << lo_size) - 1));
        let lo_gap = b.sub(lo_bound, lo);
        b.lookup_rngchk_8(lo_gap, flag);

        lo
    }

    /// Rangecheck `value ∈ [0, 2^bits)` for any `bits ≥ 1`.
    ///
    /// Cost:
    ///   bits == 1                     → 0 lookups (boolean check, 2 algebraic constraints)
    ///   bits == 8q (byte-aligned)     → q lookups
    ///   bits == 8q + r, r ∈ (0, 8)    → q + 2 lookups
    ///
    /// The non-byte-aligned case extends the byte-decomposition trick: rangecheck all
    /// q+1 byte chunks normally, then add a single byte rangecheck of
    /// `(2^r - 1) - top_chunk` to prove `top_chunk < 2^r`.
    fn gen_witness_rangecheck_bits(
        &self,
        b: &mut HLInstrBuilder<'_>,
        value: ValueId,
        bits: usize,
        flag: ValueId,
    ) {
        assert!(bits >= 1, "rangecheck width must be at least 1 bit");

        if bits == 1 {
            // Boolean check: flag * (value² - value) = 0
            // Split into: t = value * value (constrained), then flag * (t - value) = 0
            let v_plain = b.value_of(value);
            let t_hint = b.mul(v_plain, v_plain);
            let t = b.write_witness(t_hint);
            b.constrain(value, value, t);
            let diff = b.sub(t, value);
            let zero = b.field_const(Field::ZERO);
            b.constrain(flag, diff, zero);
            return;
        }

        let full_bytes = bits / 8;
        let leftover_bits = bits % 8;
        let total_chunks = full_bytes + if leftover_bits > 0 { 1 } else { 0 };
        // total_chunks ≥ 1 since bits ≥ 2 here.

        let pure_value = b.value_of(value);
        let bytes_val = b.fresh_value();
        b.push(OpCode::ToRadix {
            result: bytes_val,
            value: pure_value,
            radix: Radix::Bytes,
            endianness: Endianness::Big,
            count: total_chunks,
        });
        let two_to_8 = b.field_const(Field::from(256));

        // Witness and rangecheck upper total_chunks-1 chunks, accumulate partial sum.
        // The top chunk is the first iteration when leftover_bits > 0 (big-endian).
        let mut partial = b.field_const(Field::ZERO);
        let mut top_chunk: Option<ValueId> = None;
        for i in 0..total_chunks - 1 {
            let idx = b.u_const(32, i as u128);
            let byte = b.array_get(bytes_val, idx);
            let byte_field = b.cast_to_field(byte);
            let byte_wit = b.write_witness(byte_field);
            b.lookup_rngchk_8(byte_wit, flag);
            if i == 0 {
                top_chunk = Some(byte_wit);
            }
            let shift_prev = b.mul(partial, two_to_8);
            partial = b.add(shift_prev, byte_wit);
        }
        // Compute the LSB chunk as value - partial*256; rangecheck proves it's a byte,
        // which implicitly constrains the reconstruction (no separate constraint needed).
        let partial_shifted = b.mul(partial, two_to_8);
        let lsb = b.sub(value, partial_shifted);
        b.lookup_rngchk_8(lsb, flag);
        if total_chunks == 1 {
            top_chunk = Some(lsb);
        }

        // For non-byte-aligned widths, constrain the top chunk to leftover_bits bits
        // by rangechecking (2^r - 1) - top_chunk to 8 bits.
        if leftover_bits > 0 {
            let top = top_chunk.expect("top_chunk set when total_chunks ≥ 1");
            let bound = b.field_const(Field::from((1u128 << leftover_bits) - 1));
            let gap = b.sub(bound, top);
            b.lookup_rngchk_8(gap, flag);
        }
    }

    /// Lower a witness-tainted Lt comparison, emitting the result into `result`.
    /// Generates range-check constraints to prove the comparison.
    fn lower_witness_lt(
        &self,
        b: &mut HLInstrBuilder<'_>,
        function_type_info: &FunctionTypeInfo,
        lhs: ValueId,
        rhs: ValueId,
        result: ValueId,
        l_taint: bool,
        r_taint: bool,
        l_range: &IntInterval,
        r_range: &IntInterval,
    ) {
        let rhs_stripped = function_type_info.get_value_type(rhs).strip_witness().expr;
        let (s, is_signed) = match rhs_stripped {
            TypeExpr::U(s) => (s, false),
            TypeExpr::I(s) => (s, true),
            _ => panic!("ICE: rhs is not an integer type"),
        };
        let u1 = CastTarget::U(1);
        let res_hint = b.lt(lhs, rhs);
        let res_hint_field = b.cast_to_field(res_hint);
        let res_hint_plain = b.value_of(res_hint_field);
        let res_witness = b.write_witness(res_hint_plain);
        b.push(OpCode::Cast {
            result,
            value: res_witness,
            target: u1,
        });

        let l_field = b.cast_to_field(lhs);
        let r_field = b.cast_to_field(rhs);
        let lr_diff = b.sub(l_field, r_field);

        let two = b.field_const(Field::from(2));
        let result_field = b.cast_to_field(result);
        let two_res = b.mul(result_field, two);
        let one = b.field_const(Field::ONE);
        let adjustment = b.sub(one, two_res);

        let adjusted_diff = b.mul(lr_diff, adjustment);
        let adjusted_diff_plain = b.value_of(adjusted_diff);
        let adjusted_diff_wit = b.write_witness(adjusted_diff_plain);
        b.constrain(lr_diff, adjustment, adjusted_diff_wit);

        // Width of |lhs - rhs|, byte-rounded. Tight when both operands are
        // bounded narrowly (e.g. u8 widened to i64): |diff| ≤ 255 → 8-bit
        // rangecheck instead of 64.
        let diff_range = l_range.sub(r_range);
        let diff_bits = abs_max_byte_width(&diff_range, s);

        if is_signed {
            let always_flag = b.field_const(Field::ONE);

            let sign_a = self.extract_sign_bit(b, l_field, s, always_flag, l_taint, &l_range);
            let sign_b = self.extract_sign_bit(b, r_field, s, always_flag, r_taint, &r_range);

            let sa_pure = if l_taint { b.value_of(sign_a) } else { sign_a };
            let sb_pure = if r_taint { b.value_of(sign_b) } else { sign_b };
            let sa_sb_hint = b.mul(sa_pure, sb_pure);
            let sa_sb = b.write_witness(sa_sb_hint);
            b.constrain(sign_a, sign_b, sa_sb);

            let two_sa_sb = b.mul(sa_sb, two);
            let sa_plus_sb = b.add(sign_a, sign_b);
            let signs_differ = b.sub(sa_plus_sb, two_sa_sb);
            let signs_same = b.sub(one, signs_differ);

            self.gen_witness_rangecheck_bits(b, adjusted_diff_wit, diff_bits, signs_same);

            let zero = b.field_const(Field::ZERO);
            let diff_r_sa = b.sub(result_field, sign_a);
            b.constrain(signs_differ, diff_r_sa, zero);
        } else {
            let rc_flag = b.field_const(Field::from(1));
            self.gen_witness_rangecheck_bits(b, adjusted_diff_wit, diff_bits, rc_flag);
        }
    }

    /// Extract the sign bit (MSB) of an n-bit value.
    /// Requires: `value` is already rangechecked to n bits.
    /// Returns: sign ∈ {0,1} such that value = sign * 2^(n-1) + low, low ∈ [0, 2^(n-1)).
    ///
    /// `value_range` is the analyzer's bound on the underlying integer; if it
    /// proves the sign bit is 0 (i.e. value ∈ [0, 2^(n-1))), this short-circuits
    /// to the field constant 0 and emits no witness, rangecheck, or constraint.
    ///
    /// For witness inputs: writes sign as a witness with a boolean check, then
    /// computes low = value - sign * 2^(n-1) and rangechecks low ∈ [0, 2^(n-1))
    /// directly via `gen_witness_rangecheck_bits`. For byte-aligned `bits` this
    /// costs `bits/8 + 1` lookups (e.g. n=64 → 9, n=32 → 5, n=16 → 3).
    ///
    /// For pure inputs: computes sign as a pure hint (no witnesses or constraints).
    fn extract_sign_bit(
        &self,
        b: &mut HLInstrBuilder<'_>,
        value: ValueId,
        bits: usize,
        flag: ValueId,
        is_witness: bool,
        value_range: &IntInterval,
    ) -> ValueId {
        if value_range.is_non_negative_in_signed(bits) {
            return b.field_const(Field::ZERO);
        }

        let two_n_minus_1 = b.field_const(Field::from(1u128 << (bits - 1)));
        let pure_val = if is_witness { b.value_of(value) } else { value };
        let low_hint = b.truncate(pure_val, bits - 1, 254);
        let high_hint = b.sub(pure_val, low_hint);
        let sign_hint = b.div(high_hint, two_n_minus_1);

        if !is_witness {
            return sign_hint;
        }

        let sign_wit = b.write_witness(sign_hint);

        // sign ∈ {0, 1}
        self.gen_witness_rangecheck_bits(b, sign_wit, 1, flag);

        // Compute low = value - sign * 2^(n-1) (saves 1 witness + 1 constraint)
        let sign_shifted = b.mul(sign_wit, two_n_minus_1);
        let low = b.sub(value, sign_shifted);

        // low ∈ [0, 2^(n-1)) — proves the sign bit is correctly extracted.
        // For byte-aligned `bits`, this is bits/8 lookups for low's bytes plus
        // 1 lookup of (127 - top_byte) to constrain the top byte to 7 bits.
        self.gen_witness_rangecheck_bits(b, low, bits - 1, flag);

        sign_wit
    }

    /// Sign-extend a value from `from_bits` to `to_bits`.
    ///
    /// result = value + sign_bit * (2^to_bits - 2^from_bits)
    fn gen_sext(
        &self,
        b: &mut HLInstrBuilder<'_>,
        value: ValueId,
        from_bits: usize,
        to_bits: usize,
        flag: ValueId,
        is_witness: bool,
        result: ValueId,
        value_range: &IntInterval,
    ) {
        let sign_bit = self.extract_sign_bit(b, value, from_bits, flag, is_witness, value_range);
        let extension =
            b.field_const(Field::from(1u128 << to_bits) - Field::from(1u128 << from_bits));
        let offset = b.mul(sign_bit, extension);
        let extended = b.add(value, offset);

        // Cast back to target integer type
        b.push(OpCode::Cast {
            result,
            value: extended,
            target: CastTarget::I(to_bits),
        });
    }

    /// Signed n-bit add/sub gadget for witness variables.
    ///
    /// Modular reduction:
    ///   ADD: a + b = result + carry * 2^n      (carry ∈ {0,1}, result ∈ [0, 2^n))
    ///   SUB: a - b + 2^n = result + borrow * 2^n  (borrow ∈ {0,1}, result ∈ [0, 2^n))
    ///
    /// Overflow rejection (Noir arithmetic is non-wrapping):
    ///   ADD: carry + sign(result) = sign(a) + sign(b)
    ///   SUB: borrow + sign(result) + sign(b) = 1 + sign(a)
    fn gen_witness_signed_addsub(
        &self,
        b: &mut HLInstrBuilder<'_>,
        l_field: ValueId,
        r_field: ValueId,
        bits: usize,
        kind: BinaryArithOpKind,
        flag: ValueId,
        result: ValueId,
        l_taint: bool,
        r_taint: bool,
        l_range: &IntInterval,
        r_range: &IntInterval,
    ) {
        let two_n = b.field_const(Field::from(1u128 << bits));
        let zero = b.field_const(Field::ZERO);

        // Extract sign bits of inputs (short-circuited to const 0 when range
        // proves the sign bit is 0).
        let sign_a = self.extract_sign_bit(b, l_field, bits, flag, l_taint, l_range);
        let sign_b = self.extract_sign_bit(b, r_field, bits, flag, r_taint, r_range);

        // Bound on the result. For Add of non-negatives the integer bound is
        // l + r; for Sub of non-negatives it's l - r (which can be negative).
        // The gadget consumer (`extract_sign_bit` for `sign_r`) checks
        // `is_non_negative_in_signed(bits)` to decide whether to elide.
        let result_range = match kind {
            BinaryArithOpKind::Add => l_range.add(r_range),
            BinaryArithOpKind::Sub => l_range.sub(r_range),
            _ => IntInterval::top(),
        };

        match kind {
            BinaryArithOpKind::Add => {
                // raw_sum = l + r is always linear (Field add). Define
                //   result = raw_sum - carry * 2^n
                // as an LC: this folds the reconstruction equality into the
                // definition, saving 1 witness write + 1 R1C constraint.
                let raw_sum = b.add(l_field, r_field);
                let raw_pure = b.value_of(raw_sum);
                let hint_result = b.truncate(raw_pure, bits, 254);
                let hint_diff = b.sub(raw_pure, hint_result);
                let hint_carry = b.div(hint_diff, two_n);

                let carry_wit = b.write_witness(hint_carry);

                let carry_shifted = b.mul(carry_wit, two_n);
                let result_lc = b.sub(raw_sum, carry_shifted);

                // Narrow result rangecheck when the analyzer proves the integer
                // result fits in fewer bytes. A malicious carry-flip would push
                // result_lc into a huge value; any narrower-than-bits rangecheck
                // still catches that.
                let result_rc_bits = narrow_rangecheck_width(&result_range, bits);
                self.gen_witness_rangecheck_bits(b, result_lc, result_rc_bits, flag);
                self.gen_witness_rangecheck_bits(b, carry_wit, 1, flag);

                // Extract sign bit of result for overflow check (skipped when
                // range proves result is non-negative).
                let sign_r =
                    self.extract_sign_bit(b, result_lc, bits, flag, true, &result_range);

                // Signed overflow iff carry_into_MSB != carry_out_of_MSB.
                // The MSB full-adder gives: s_a + s_b + carry_in = s_r + 2*carry_out.
                // So carry_in = carry_out is equivalent to: s_a + s_b = s_r + carry_out,
                // i.e. carry_out + s_r = s_a + s_b.
                let lhs_ov = b.add(carry_wit, sign_r);
                let rhs_ov = b.add(sign_a, sign_b);
                let diff_ov = b.sub(lhs_ov, rhs_ov);
                b.constrain(flag, diff_ov, zero);

                b.push(OpCode::Cast {
                    result,
                    value: result_lc,
                    target: CastTarget::I(bits),
                });
            }
            BinaryArithOpKind::Sub => {
                // raw_diff = l - r is linear, so raw_diff + 2^n is too. Define
                //   result = (raw_diff + 2^n) - borrow * 2^n
                // as an LC, folding the reconstruction equality into the definition.
                let raw_diff = b.sub(l_field, r_field);
                let raw_pure = b.value_of(raw_diff);
                let shifted = b.add(raw_pure, two_n);
                let hint_result = b.truncate(shifted, bits, 254);
                let hint_rem = b.sub(shifted, hint_result);
                let hint_borrow = b.div(hint_rem, two_n);

                let borrow_wit = b.write_witness(hint_borrow);

                let lhs_full = b.add(raw_diff, two_n);
                let borrow_shifted = b.mul(borrow_wit, two_n);
                let result_lc = b.sub(lhs_full, borrow_shifted);

                // Narrow only when result_range proves non-negative: the honest
                // result_lc then equals l - r and fits in result_range.hi bytes.
                // (When result_range is negative or straddles zero, the honest
                // result lives near 2^n and we need the full width.)
                let result_rc_bits = narrow_rangecheck_width(&result_range, bits);
                self.gen_witness_rangecheck_bits(b, result_lc, result_rc_bits, flag);
                self.gen_witness_rangecheck_bits(b, borrow_wit, 1, flag);

                // Extract sign bit of result for overflow check (skipped when
                // range proves result is non-negative).
                let sign_r =
                    self.extract_sign_bit(b, result_lc, bits, flag, true, &result_range);

                // Subtraction a - b is computed as a + ~b + 1 (two's complement).
                // At the MSB column the full-adder sees:
                //   s_a + ~s_b + carry_in = s_r + 2*borrow
                // where ~s_b = 1 - s_b (bitwise NOT), carry_in is whatever
                // rippled up from the lower bits, and borrow is the carry out.
                // Signed overflow iff carry_in != borrow. Substituting
                // carry_in = borrow into the equation eliminates carry_in:
                //   s_a + 1 - s_b = s_r + borrow
                // i.e. borrow + s_r + s_b = 1 + s_a.
                let one = b.field_const(Field::ONE);
                let lhs_ov = b.add(borrow_wit, sign_r);
                let lhs_ov = b.add(lhs_ov, sign_b);
                let rhs_ov = b.add(one, sign_a);
                let diff_ov = b.sub(lhs_ov, rhs_ov);
                b.constrain(flag, diff_ov, zero);

                b.push(OpCode::Cast {
                    result,
                    value: result_lc,
                    target: CastTarget::I(bits),
                });
            }
            _ => unreachable!(),
        }
    }

    /// Signed n-bit multiplication gadget for witness variables.
    ///
    /// Modular reduction (field product can be up to ~2^(2n)):
    ///   a * b = result + q * 2^n   (result ∈ [0, 2^n), q ∈ [0, 2^n))
    ///
    /// Overflow rejection:
    ///   result * (sign(a) ⊕ sign(b) − sign(result)) = 0
    /// Non-zero results must have sign matching XOR of input signs.
    /// Zero results satisfy the constraint trivially.
    fn gen_witness_signed_mul(
        &self,
        b: &mut HLInstrBuilder<'_>,
        l_field: ValueId,
        r_field: ValueId,
        bits: usize,
        flag: ValueId,
        result: ValueId,
        l_taint: bool,
        r_taint: bool,
        l_range: &IntInterval,
        r_range: &IntInterval,
    ) {
        let two_n = b.field_const(Field::from(1u128 << bits));
        let zero = b.field_const(Field::ZERO);

        // Extract sign bits of inputs (short-circuited when range proves non-negative).
        // Integer bound on the product. Interval arithmetic handles both
        // non-negative and mixed-sign cases.
        let result_range = l_range.mul(r_range);
        let result_rc_bits = narrow_rangecheck_width(&result_range, bits);

        // Tier A: when both operands and the product all fit in non-negative
        // i_n (their integer values are in [0, 2^(n-1))), the modular `q`
        // witness is always zero, both input sign bits are zero, and the
        // overflow check is a tautology. Skip everything and emit a single
        // multiplication constraint plus the result rangecheck.
        if l_range.is_non_negative_in_signed(bits)
            && r_range.is_non_negative_in_signed(bits)
            && result_range.is_non_negative_in_signed(bits)
        {
            let result_wit = if l_taint && r_taint {
                let l_pure = b.value_of(l_field);
                let r_pure = b.value_of(r_field);
                let prod_hint = b.mul(l_pure, r_pure);
                let result_wit = b.write_witness(prod_hint);
                b.constrain(l_field, r_field, result_wit);
                result_wit
            } else {
                // One operand is pure → product is linear; emit as LC.
                b.mul(l_field, r_field)
            };
            self.gen_witness_rangecheck_bits(b, result_wit, result_rc_bits, flag);
            b.push(OpCode::Cast {
                result,
                value: result_wit,
                target: CastTarget::I(bits),
            });
            return;
        }

        let sign_a = self.extract_sign_bit(b, l_field, bits, flag, l_taint, l_range);
        let sign_b = self.extract_sign_bit(b, r_field, bits, flag, r_taint, r_range);

        let result_wit = if l_taint && r_taint {
            // Non-linear: compute hint from pure values
            let l_pure = b.value_of(l_field);
            let r_pure = b.value_of(r_field);
            let raw_product = b.mul(l_pure, r_pure);
            let hint_result = b.truncate(raw_product, bits, 254);
            let hint_rem = b.sub(raw_product, hint_result);
            let hint_q = b.div(hint_rem, two_n);

            let result_wit = b.write_witness(hint_result);
            let q_wit = b.write_witness(hint_q);

            self.gen_witness_rangecheck_bits(b, result_wit, result_rc_bits, flag);
            self.gen_witness_rangecheck_bits(b, q_wit, bits, flag);

            // Constrain: l * r = result + q * 2^n
            let q_shifted = b.mul(q_wit, two_n);
            let rhs = b.add(result_wit, q_shifted);
            b.constrain(l_field, r_field, rhs);
            result_wit
        } else {
            // Linear: one operand is pure → raw_product is an LC.
            // Define result = raw_product - q * 2^n as an LC, folding the
            // reconstruction equality into the definition. Saves 1 witness
            // + 1 R1C constraint vs the witness form.
            let raw_product = b.mul(l_field, r_field);
            let raw_pure = b.value_of(raw_product);
            let hint_result = b.truncate(raw_pure, bits, 254);
            let hint_rem = b.sub(raw_pure, hint_result);
            let hint_q = b.div(hint_rem, two_n);

            let q_wit = b.write_witness(hint_q);

            let q_shifted = b.mul(q_wit, two_n);
            let result_lc = b.sub(raw_product, q_shifted);

            self.gen_witness_rangecheck_bits(b, result_lc, result_rc_bits, flag);
            self.gen_witness_rangecheck_bits(b, q_wit, bits, flag);

            result_lc
        };

        let sign_r = self.extract_sign_bit(b, result_wit, bits, flag, true, &result_range);

        // sa_sb = sign_a * sign_b (R1CS multiplication)
        let sa_pure = if l_taint { b.value_of(sign_a) } else { sign_a };
        let sb_pure = if r_taint { b.value_of(sign_b) } else { sign_b };
        let sa_sb_val = b.mul(sa_pure, sb_pure);
        let sa_sb = b.write_witness(sa_sb_val);
        b.constrain(sign_a, sign_b, sa_sb);
        // xor = sign_a + sign_b - 2*sa_sb (linear combination)
        let two = b.field_const(Field::from(2u64));
        let two_sa_sb = b.mul(two, sa_sb);
        let xor_val = b.add(sign_a, sign_b);
        let xor_val = b.sub(xor_val, two_sa_sb);
        let expected_diff = b.sub(xor_val, sign_r);
        // result * expected_diff = 0 (degree-2, needs auxiliary witness h)
        let r_pure = b.value_of(result_wit);
        let ed_pure = b.value_of(expected_diff);
        let h_val = b.mul(r_pure, ed_pure);
        let h_wit = b.write_witness(h_val);
        b.constrain(result_wit, expected_diff, h_wit); // h = result * expected_diff
        b.constrain(flag, h_wit, zero); // flag * h = 0

        b.push(OpCode::Cast {
            result,
            value: result_wit,
            target: CastTarget::I(bits),
        });
    }

    /// Integer div/mod gadget for witness variables.
    ///
    /// Given `a / b` or `a % b` on U(bits) with at least one witness operand:
    /// - Compute q and r as hints (pure VM ops)
    /// - Write q and r as witnesses
    /// - Constrain: q * b = a - r
    /// - Rangecheck q, r to `bits`
    /// - Rangecheck (b - r - 1) to `bits` (proves r < b)
    /// - Return q (for Div) or r (for Mod)
    fn gen_witness_divmod(
        &self,
        b: &mut HLInstrBuilder<'_>,
        a: ValueId,
        divisor: ValueId,
        bits: usize,
        kind: BinaryArithOpKind,
        flag: ValueId,
        result: ValueId,
        l_taint: bool,
        r_taint: bool,
        a_range: &IntInterval,
        b_range: &IntInterval,
    ) {
        // Get pure (non-witness) versions for hint computation — integer arithmetic
        let a_pure = if l_taint { b.value_of(a) } else { a };
        let b_pure = if r_taint {
            b.value_of(divisor)
        } else {
            divisor
        };
        let q_hint = b.div(a_pure, b_pure);
        let qb = b.mul(q_hint, b_pure);
        let r_hint = b.sub(a_pure, qb);

        // Write witnesses (cast to field first)
        let q_hint_field = b.cast_to_field(q_hint);
        let r_hint_field = b.cast_to_field(r_hint);
        let q_wit = b.write_witness(q_hint_field);
        let r_wit = b.write_witness(r_hint_field);

        // Cast operands to field for constraints
        let a_field = b.cast_to_field(a);
        let b_field = b.cast_to_field(divisor);

        // Constraint: q * b = a - r
        let a_minus_r = b.sub(a_field, r_wit);
        b.constrain(q_wit, b_field, a_minus_r);

        // Rangecheck q and r — narrow widths when the analyzer's bound proves
        // they fit in fewer bytes than the operand width.
        let q_bits = narrow_rangecheck_width(&quotient_bound(a_range, b_range), bits);
        let r_bound = remainder_bound(b_range);
        let r_bits = narrow_rangecheck_width(&r_bound, bits);
        self.gen_witness_rangecheck_bits(b, q_wit, q_bits, flag);
        self.gen_witness_rangecheck_bits(b, r_wit, r_bits, flag);

        // Assert r < b via rangecheck(b - r - 1, bits). Same width as r since
        // b - r - 1 ∈ [0, b.hi - 1] when r ∈ [0, b - 1].
        let one = b.field_const(Field::ONE);
        let b_minus_r = b.sub(b_field, r_wit);
        let b_minus_r_minus_1 = b.sub(b_minus_r, one);
        self.gen_witness_rangecheck_bits(b, b_minus_r_minus_1, r_bits, flag);

        // Cast result
        let wit_val = match kind {
            BinaryArithOpKind::Div => q_wit,
            BinaryArithOpKind::Mod => r_wit,
            _ => unreachable!(),
        };
        b.push(OpCode::Cast {
            result,
            value: wit_val,
            target: CastTarget::U(bits),
        });
    }

    /// Field division with witnesses: q * b = a
    fn gen_witness_field_div(
        &self,
        b: &mut HLInstrBuilder<'_>,
        a: ValueId,
        divisor: ValueId,
        l_taint: bool,
        r_taint: bool,
        result: ValueId,
    ) {
        if !l_taint && r_taint {
            // Only rhs is witness: q = a / b is non-linear
            let a_pure = b.value_of(a);
            let b_pure = b.value_of(divisor);
            let q_hint = b.div(a_pure, b_pure);
            let q_hint_field = b.cast_to_field(q_hint);
            b.push(OpCode::WriteWitness {
                result: Some(result),
                value: q_hint_field,
                pinned: false,
            });
            b.constrain(result, divisor, a);
        } else if l_taint && !r_taint {
            // Only lhs is witness: q = a * (1/b) is linear, pass through
            b.push(OpCode::BinaryArithOp {
                kind: BinaryArithOpKind::Div,
                result,
                lhs: a,
                rhs: divisor,
            });
        } else {
            // Both witness: q * b = a
            let a_pure = b.value_of(a);
            let b_pure = b.value_of(divisor);
            let q_hint = b.div(a_pure, b_pure);
            let q_hint_field = b.cast_to_field(q_hint);
            b.push(OpCode::WriteWitness {
                result: Some(result),
                value: q_hint_field,
                pinned: false,
            });
            b.constrain(result, divisor, a);
        }
    }

    /// Multi-bit witness And/Or/Xor via spread tables.
    ///
    /// Uses the identity: spread(a) + spread(b) = 2*spread(a&b) + spread(a^b)
    /// Each operand is decomposed into bytes, each byte spread via lookup table,
    /// then per-byte and/xor are extracted and verified.
    fn gen_witness_bitwise(
        &self,
        b: &mut HLInstrBuilder<'_>,
        kind: BinaryArithOpKind,
        l: ValueId,
        r: ValueId,
        l_taint: bool,
        r_taint: bool,
        n: usize,
        result: ValueId,
    ) {
        assert!(n % 8 == 0 && n >= 8 && n <= 256);
        let chunks = n / 8;
        let one = b.field_const(Field::ONE);
        let zero = b.field_const(Field::ZERO);
        let two = b.field_const(Field::from(2));
        let two_to_8 = b.field_const(Field::from(256u128));
        let two_to_16 = b.field_const(Field::from(1u128 << 16));

        // Get pure values for hint computation
        let l_pure = if l_taint { b.value_of(l) } else { l };
        let r_pure = if r_taint { b.value_of(r) } else { r };

        // Cast operands to field for arithmetic
        let l_field = b.cast_to_field(l);
        let r_field = b.cast_to_field(r);

        // Step 1: Byte decomposition + spread for each operand.
        let (a_pure_bytes, a_spread_word) =
            self.spread_decompose(b, l_pure, l_field, chunks, one, l_taint);
        let (b_pure_bytes, b_spread_word) =
            self.spread_decompose(b, r_pure, r_field, chunks, one, r_taint);

        // Step 2: Per-byte witness generation, while reconstructing full-word values.
        // For the last xor byte, compute its spread from the bitwise identity
        // spread(a)+spread(b) = 2*spread(and)+spread(xor), saving 1 witness + 1 constraint.
        let mut and_word = zero;
        let mut xor_word = zero;
        let mut and_spread_word = zero;
        let mut xor_spread_word = zero;
        for i in 0..chunks {
            let a_byte_u8 = b.cast_to(CastTarget::U(8), a_pure_bytes[i]);
            let b_byte_u8 = b.cast_to(CastTarget::U(8), b_pure_bytes[i]);
            let and_hint = b.and(a_byte_u8, b_byte_u8);
            let xor_hint = b.xor(a_byte_u8, b_byte_u8);

            // And byte: always fully witnessed
            let (and_byte_wit, and_spread_wit) = self.write_spread_byte_witness(b, and_hint, one);
            let shifted_and_word = b.mul(and_word, two_to_8);
            and_word = b.add(shifted_and_word, and_byte_wit);
            let shifted_and_spread = b.mul(and_spread_word, two_to_16);
            and_spread_word = b.add(shifted_and_spread, and_spread_wit);

            if i < chunks - 1 {
                // Xor byte: fully witnessed for non-last bytes
                let (xor_byte_wit, xor_spread_wit) =
                    self.write_spread_byte_witness(b, xor_hint, one);
                let shifted_xor_word = b.mul(xor_word, two_to_8);
                xor_word = b.add(shifted_xor_word, xor_byte_wit);
                let shifted_xor_spread = b.mul(xor_spread_word, two_to_16);
                xor_spread_word = b.add(shifted_xor_spread, xor_spread_wit);
            } else {
                // Last xor byte: witness byte only, derive spread from identity
                let xor_byte_field = b.cast_to_field(xor_hint);
                let xor_byte_wit = b.write_witness(xor_byte_field);
                let shifted_xor_word = b.mul(xor_word, two_to_8);
                xor_word = b.add(shifted_xor_word, xor_byte_wit);

                let input_spread_sum = b.add(a_spread_word, b_spread_word);
                let two_and = b.mul(two, and_spread_word);
                let remainder = b.sub(input_spread_sum, two_and);
                let shifted_xor_spread = b.mul(xor_spread_word, two_to_16);
                let xor_spread_last = b.sub(remainder, shifted_xor_spread);
                b.lookup_spread(8, xor_byte_wit, xor_spread_last, one);
            }
        }

        let result_word = match kind {
            BinaryArithOpKind::And => and_word,
            BinaryArithOpKind::Xor => xor_word,
            BinaryArithOpKind::Or => b.add(and_word, xor_word),
            _ => unreachable!(),
        };

        b.push(OpCode::Cast {
            result,
            value: result_word,
            target: CastTarget::U(n),
        });
    }

    /// Decompose an operand into bytes and their spreads.
    /// Returns (pure_bytes, reconstructed_spread) in big-endian order.
    /// pure_bytes: for use in hint computations (never witness-typed).
    ///
    /// When `is_witness` is false, the operand is pure (e.g. a constant in `a & 0xff`),
    /// so we compute spreads purely without witnesses or lookups.
    fn spread_decompose(
        &self,
        b: &mut HLInstrBuilder<'_>,
        pure_val: ValueId,
        field_val: ValueId,
        chunks: usize,
        flag: ValueId,
        is_witness: bool,
    ) -> (Vec<ValueId>, ValueId) {
        let zero = b.field_const(Field::ZERO);
        let two_to_8 = b.field_const(Field::from(256u128));
        let two_to_16 = b.field_const(Field::from(1u128 << 16));

        // Decompose into bytes via ToRadix (big-endian) using pure value
        let pure_field = b.cast_to_field(pure_val);
        let bytes_arr = b.to_radix(pure_field, Radix::Bytes, Endianness::Big, chunks);

        let mut pure_bytes = Vec::with_capacity(chunks);

        if !is_witness {
            // Pure operand: no per-byte lookups needed.
            // Compute per-byte spreads as hints and accumulate, then witness
            // the accumulated spread. The identity constraint implicitly
            // constrains it; in R1CS the hint chain gets DCE'd behind
            // write_witness→FreshWitness.
            let mut spread_acc = zero;
            for i in 0..chunks {
                let idx = b.u_const(32, i as u128);
                let byte_i = b.array_get(bytes_arr, idx);
                pure_bytes.push(byte_i);

                let byte_spread = b.spread(byte_i, 8);
                let byte_spread_field = b.cast_to_field(byte_spread);
                let shifted_spread = b.mul(spread_acc, two_to_16);
                spread_acc = b.add(shifted_spread, byte_spread_field);
            }
            let spread_wit = b.write_witness(spread_acc);
            return (pure_bytes, spread_wit);
        }

        // Witness path: witness chunks-1 bytes with spread lookups, then compute
        // last byte as field_val - partial*256 (saves 1 witness + 1 constraint).
        let mut recon_value = zero;
        let mut recon_spread = zero;
        for i in 0..chunks - 1 {
            let idx = b.u_const(32, i as u128);
            let byte_i = b.array_get(bytes_arr, idx);
            let (byte_wit, spread_wit) = self.write_spread_byte_witness(b, byte_i, flag);

            let shifted_value = b.mul(recon_value, two_to_8);
            recon_value = b.add(shifted_value, byte_wit);

            let shifted_spread = b.mul(recon_spread, two_to_16);
            recon_spread = b.add(shifted_spread, spread_wit);

            pure_bytes.push(byte_i);
        }

        // Last byte: compute from reconstruction remainder, only witness spread
        let last_idx = b.u_const(32, (chunks - 1) as u128);
        let last_byte_pure = b.array_get(bytes_arr, last_idx);
        let recon_shifted = b.mul(recon_value, two_to_8);
        let last_byte = b.sub(field_val, recon_shifted);

        let spread_hint = b.spread(last_byte_pure, 8);
        let spread_hint_field = b.cast_to_field(spread_hint);
        let last_spread_wit = b.write_witness(spread_hint_field);
        b.lookup_spread(8, last_byte, last_spread_wit, flag);

        let shifted_spread = b.mul(recon_spread, two_to_16);
        recon_spread = b.add(shifted_spread, last_spread_wit);

        pure_bytes.push(last_byte_pure);

        (pure_bytes, recon_spread)
    }

    fn write_spread_byte_witness(
        &self,
        b: &mut HLInstrBuilder<'_>,
        byte_value: ValueId,
        flag: ValueId,
    ) -> (ValueId, ValueId) {
        let byte_value_field = b.cast_to_field(byte_value);
        let byte_wit = b.write_witness(byte_value_field);

        let spread_hint = b.spread(byte_value, 8);
        let spread_hint_field = b.cast_to_field(spread_hint);
        let spread_wit = b.write_witness(spread_hint_field);
        b.lookup_spread(8, byte_wit, spread_wit, flag);

        (byte_wit, spread_wit)
    }

    /// Lower a witness-indexed ArrayGet into a hint + lookup constraint.
    /// `flag` is the lookup flag: `1` unconditionally, or the guard condition.
    fn gen_witness_array_get(
        &self,
        b: &mut HLInstrBuilder<'_>,
        function_type_info: &FunctionTypeInfo,
        arr: ValueId,
        idx: ValueId,
        result: ValueId,
        flag: ValueId,
    ) {
        let result_type = function_type_info
            .get_value_type(result)
            .strip_all_witness();
        let back_cast_target = match &result_type.expr {
            TypeExpr::U(s) => CastTarget::U(*s),
            TypeExpr::I(s) => CastTarget::I(*s),
            TypeExpr::Field => CastTarget::Field,
            TypeExpr::WitnessOf(_) => CastTarget::Field,
            TypeExpr::Array(_, _) => {
                todo!("array types in witnessed array reads")
            }
            TypeExpr::Slice(_) => {
                todo!("slice types in witnessed array reads")
            }
            TypeExpr::Ref(_) => {
                todo!("ref types in witnessed array reads")
            }
            TypeExpr::Tuple(_elements) => {
                todo!("Tuples not supported yet")
            }
            TypeExpr::Function => {
                panic!("Function type not expected in witnessed array reads")
            }
        };

        let pure_idx = b.value_of(idx);
        let mut r_pure_val = b.array_get(arr, pure_idx);

        let elem_is_witness = function_type_info
            .get_value_type(arr)
            .get_array_element()
            .is_witness_of();
        if elem_is_witness {
            r_pure_val = b.value_of(r_pure_val);
        }

        let idx_field = b.cast_to_field(idx);
        let r_wit_field = b.cast_to_field(r_pure_val);
        let r_wit = b.write_witness(r_wit_field);
        b.push(OpCode::Cast {
            result,
            value: r_wit,
            target: back_cast_target,
        });
        b.lookup_arr(arr, idx_field, r_wit, flag);
    }

    fn ensure_field(
        &self,
        b: &mut HLInstrBuilder<'_>,
        function_type_info: &FunctionTypeInfo,
        value: ValueId,
    ) -> ValueId {
        if function_type_info
            .get_value_type(value)
            .strip_witness()
            .is_field()
        {
            value
        } else {
            b.cast_to_field(value)
        }
    }
}
