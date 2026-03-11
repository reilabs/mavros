use std::collections::HashMap;

use ark_ff::{AdditiveGroup, Field as _};

use crate::compiler::{
    Field,
    analysis::types::TypeInfo,
    block_builder::{HLEmitter, HLInstrBuilder},
    flow_analysis::FlowAnalysis,
    ir::r#type::{Type, TypeExpr},
    pass_manager::{Analysis, AnalysisId, AnalysisStore, Pass},
    ssa::{
        BinaryArithOpKind, BlockId, CastTarget, CmpKind, Endianness, HLBlock, HLFunction, HLSSA,
        Instruction, LookupTarget, OpCode, Radix, SeqType, ValueId,
    },
};

pub struct ExplicitWitness {}

impl Pass for ExplicitWitness {
    fn name(&self) -> &'static str {
        "explicit_witness"
    }

    fn needs(&self) -> Vec<AnalysisId> {
        vec![TypeInfo::id()]
    }

    fn run(&self, ssa: &mut HLSSA, store: &AnalysisStore) {
        self.do_run(ssa, store.get::<TypeInfo>());
    }

    fn preserves(&self) -> Vec<AnalysisId> {
        vec![FlowAnalysis::id()]
    }
}

impl ExplicitWitness {
    pub fn new() -> Self {
        Self {}
    }

    pub fn do_run(&self, ssa: &mut HLSSA, type_info: &TypeInfo) {
        for (function_id, function) in ssa.iter_functions_mut() {
            let function_type_info = type_info.get_function(*function_id);
            let mut new_blocks = HashMap::<BlockId, HLBlock>::new();
            for (bid, mut block) in function.take_blocks().into_iter() {
                let mut new_instructions = Vec::new();
                for instruction in block.take_instructions().into_iter() {
                    let b = &mut HLInstrBuilder::new(function, &mut new_instructions);
                    self.process_instruction(b, function_type_info, instruction);
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
        function_type_info: &crate::compiler::analysis::types::FunctionTypeInfo,
        instruction: OpCode,
    ) {
        match instruction {
            OpCode::BinaryArithOp {
                kind: BinaryArithOpKind::Add,
                ..
            } => {
                b.push(instruction);
            }
            OpCode::BinaryArithOp {
                kind: BinaryArithOpKind::Sub,
                ..
            } => {
                b.push(instruction);
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
                            let TypeExpr::U(s) =
                                function_type_info.get_value_type(rhs).strip_witness().expr
                            else {
                                panic!("ICE: rhs is not a U type");
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
                            let two_res = b.mul(result, two);
                            let one = b.field_const(Field::ONE);
                            let adjustment = b.sub(one, two_res);

                            let adjusted_diff = b.mul(lr_diff, adjustment);
                            let adjusted_diff_plain = b.value_of(adjusted_diff);
                            let adjusted_diff_wit = b.write_witness(adjusted_diff_plain);
                            b.constrain(lr_diff, adjustment, adjusted_diff_wit);

                            let rc_flag = b.field_const(Field::from(1));
                            self.gen_witness_rangecheck(b, adjusted_diff_wit, s, rc_flag);
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

                if !l_taint || !r_taint {
                    b.push(instruction);
                    return;
                }

                // witness-witness mul
                let mul_witness = b.mul(l, r);
                let mul_plain = b.value_of(mul_witness);
                b.push(OpCode::WriteWitness {
                    result: Some(res),
                    value: mul_plain,
                    pinned: false,
                });
                b.constrain(l, r, res);
            }
            OpCode::BinaryArithOp {
                kind: BinaryArithOpKind::Div,
                result: _,
                lhs: l,
                rhs: r,
            } => {
                let l_taint = function_type_info.get_value_type(l).is_witness_of();
                let r_taint = function_type_info.get_value_type(r).is_witness_of();
                assert!(!l_taint);
                assert!(!r_taint);
                b.push(instruction);
            }
            OpCode::BinaryArithOp {
                kind: BinaryArithOpKind::And,
                result,
                lhs: l,
                rhs: r,
            } => {
                let l_taint = function_type_info.get_value_type(l).is_witness_of();
                let r_taint = function_type_info.get_value_type(r).is_witness_of();
                match (l_taint, r_taint) {
                    (false, false) => {
                        b.push(instruction);
                    }
                    (true, true) => {
                        let u1 = CastTarget::U(1);
                        let l_field = b.cast_to_field(l);
                        let r_field = b.cast_to_field(r);
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
                    }
                    _ => {
                        b.push(OpCode::Todo {
                            payload: format!("witness AND {} {}", l_taint, r_taint),
                            results: vec![result],
                            result_types: vec![function_type_info.get_value_type(r).clone()],
                        });
                    }
                }
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
            OpCode::AssertEq { lhs: l, rhs: r } => {
                let l_type = function_type_info.get_value_type(l);
                let r_type = function_type_info.get_value_type(r);
                let l_taint = l_type.is_witness_of();
                let r_taint = r_type.is_witness_of();
                if !l_taint && !r_taint {
                    b.push(instruction);
                    return;
                }
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
                let idx_type = function_type_info.get_value_type(idx);
                let idx_taint = idx_type.is_witness_of();
                assert!(!arr_taint);
                match idx_taint {
                    false => {
                        b.push(instruction);
                    }
                    true => {
                        let back_cast_target = match &idx_type.expr {
                            TypeExpr::U(s) => CastTarget::U(*s),
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
                        let one = b.field_const(Field::from(1));
                        b.lookup_arr(arr, idx_field, r_wit, one);
                    }
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
                // If both branches are pure, result is a linear combination
                // of cond and constants: cond * (l - r) + r. No constraint needed.
                if !l_taint && !r_taint {
                    let neg_one = b.field_const(ark_ff::Fp::from(-1));
                    let neg_r = b.mul(r, neg_one);
                    let l_sub_r = b.add(l, neg_r);
                    // cond * (l - r): l_sub_r is a constant, cond is witness
                    let cond_times_diff = b.mul(l_sub_r, cond);
                    // result = cond * (l - r) + r
                    b.push(OpCode::BinaryArithOp {
                        kind: BinaryArithOpKind::Add,
                        result: res,
                        lhs: cond_times_diff,
                        rhs: r,
                    });
                    return;
                }
                // At least one branch is witness: full lowering with constraint
                let select_witness = b.select(cond, l, r);
                let select_plain = b.value_of(select_witness);
                b.push(OpCode::WriteWitness {
                    result: Some(res),
                    value: select_plain,
                    pinned: false,
                });
                // Goal is to assert 0 = cond * l + (1 - cond) * r - res
                // This is equivalent to 0 = cond * (l - r) + r - res = cond * (l - r) - (res - r)
                let neg_one = b.field_const(ark_ff::Fp::from(-1));
                let neg_r = b.mul(r, neg_one);
                let l_sub_r = b.add(l, neg_r);
                let res_sub_r = b.add(res, neg_r);
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
                    self.gen_witness_truncate(b, value, to_bits, one, result);
                }
            }
            OpCode::Not { result, value } => {
                let value_type = function_type_info.get_value_type(value);
                let s = match &value_type.strip_witness().expr {
                    TypeExpr::U(s) => *s,
                    e => todo!("Unsupported type for negation: {:?}", e),
                };
                let ones = b.field_const(Field::from((1u128 << s) - 1));
                let casted = b.cast_to(CastTarget::Field, value);
                let subbed = b.sub(ones, casted);
                b.push(OpCode::Cast {
                    result,
                    value: subbed,
                    target: CastTarget::U(s),
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
                    self.gen_witness_rangecheck(b, value, max_bits, one);
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
            OpCode::Guard { condition, inner } => {
                self.lower_guard(b, function_type_info, condition, *inner);
            }
        }
    }

    fn lower_guard(
        &self,
        b: &mut HLInstrBuilder<'_>,
        function_type_info: &crate::compiler::analysis::types::FunctionTypeInfo,
        condition: ValueId,
        inner: OpCode,
    ) {
        match inner {
            OpCode::AssertEq { lhs: l, rhs: r } => {
                // Guard(cond, AssertEq(l, r)) → constrain(cond_field, l_sub_r, zero)
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
                // Witness truncate: full byte decomposition with canonical check.
                // 1) Decompose value into 32 bytes, range-check each, constrain reconstruction
                // 2) Prove decomposition is canonical (< field modulus) via 128-bit digital check
                // 3) Recover truncated value from low bytes
                let cond_type = function_type_info.get_value_type(condition);
                let cond_field = if cond_type.strip_witness().is_field() {
                    condition
                } else {
                    b.cast_to_field(condition)
                };
                self.gen_witness_truncate(b, value, to_bits, cond_field, result);
            }
            OpCode::Cast { .. }
            | OpCode::Const { .. }
            | OpCode::Cmp { .. }
            | OpCode::BinaryArithOp { .. } => {
                // Pure computations — no constraints. Emit unconditionally.
                b.push(inner);
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
                self.gen_witness_rangecheck(b, value, max_bits, cond_field);
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
    ) {
        assert!(to_bits % 8 == 0);
        let to_bytes = to_bits / 8;
        assert!(to_bytes <= 32);

        // BN254 modulus: p = 0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000001
        // modulusHi = upper 128 bits
        let modulus_hi = b.field_const(Field::from(0x30644e72e131a029b85045b68181585du128));
        // modulusLoMinusOne = lower 128 bits - 1
        let modulus_lo_m1 =
            b.field_const(Field::from(0x2833e84879b9709143e1f593f0000000u128));
        let two_to_8 = b.field_const(Field::from(256u128));
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

        // Extract each byte, write as witness, range-check
        let mut bytes = Vec::with_capacity(32);
        for i in 0..32 {
            let idx = b.u_const(32, i as u128);
            let byte = b.array_get(bytes_arr, idx);
            let byte_field = b.cast_to_field(byte);
            let byte_wit = b.write_witness(byte_field);
            b.lookup_rngchk_8(byte_wit, flag);
            bytes.push(byte_wit);
        }

        // Step 2: Reconstruct full value and constrain == input value (conditional)
        let mut full_sum = zero;
        for i in 0..32 {
            let shifted = b.mul(full_sum, two_to_8);
            full_sum = b.add(shifted, bytes[i]);
        }
        // flag * (full_sum - value) = 0
        let recon_diff = b.sub(full_sum, value);
        b.constrain(flag, recon_diff, zero);

        // Step 3: Reconstruct hi (bytes[0..16]) and lo (bytes[16..32])
        let mut hi = zero;
        for i in 0..16 {
            let shifted = b.mul(hi, two_to_8);
            hi = b.add(shifted, bytes[i]);
        }
        let mut lo = zero;
        for i in 16..32 {
            let shifted = b.mul(lo, two_to_8);
            lo = b.add(shifted, bytes[i]);
        }

        // Step 4: Compute borrow hint (for witgen only; DCE'd in R1CS pipeline)
        // borrow = 1 if lo > modulusLoMinusOne, else 0
        // Strategy: compute (modulusLoMinusOne - lo) as field subtraction.
        // If lo <= modulusLoMinusOne, result fits in 128 bits (byte[0] == 0).
        // If lo > modulusLoMinusOne, result wraps to p + modulusLoMinusOne - lo (byte[0] != 0).
        let diff_for_borrow = b.sub(modulus_lo_m1, lo);
        let diff_pure = b.value_of(diff_for_borrow);
        let diff_bytes_arr = b.fresh_value();
        b.push(OpCode::ToRadix {
            result: diff_bytes_arr,
            value: diff_pure,
            radix: Radix::Bytes,
            endianness: Endianness::Big,
            count: 32,
        });
        // If byte[0] > 0, the value wrapped (lo > modulusLoMinusOne), so borrow = 1
        let idx_0 = b.u_const(32, 0);
        let high_byte = b.array_get(diff_bytes_arr, idx_0);
        let zero_u8 = b.u_const(8, 0);
        let borrow_hint = b.lt(zero_u8, high_byte); // U(1): 1 if wrapped
        let borrow_field = b.cast_to_field(borrow_hint);
        let borrow_wit = b.write_witness(borrow_field);

        // Step 5: Constrain borrow is boolean: borrow * borrow = borrow
        b.constrain(borrow_wit, borrow_wit, borrow_wit);

        // Step 6: Compute resultLo = modulusLoMinusOne - lo + borrow * 2^128 + 1
        let borrow_shift = b.mul(borrow_wit, two_to_128);
        let tmp1 = b.sub(modulus_lo_m1, lo);
        let tmp2 = b.add(tmp1, borrow_shift);
        let result_lo = b.add(tmp2, one);

        // Step 7: Compute resultHi = modulusHi - hi - borrow
        let tmp3 = b.sub(modulus_hi, hi);
        let result_hi = b.sub(tmp3, borrow_wit);

        // Step 8: Range-check resultHi and resultLo to 128 bits
        // This proves the decomposition is canonical (value < field modulus)
        self.gen_witness_rangecheck(b, result_hi, 128, flag);
        self.gen_witness_rangecheck(b, result_lo, 128, flag);

        // Step 9: Recover truncated value from low bytes of the decomposition
        // In big-endian, the low `to_bytes` bytes are bytes[32 - to_bytes .. 32]
        let mut trunc_val = zero;
        let start = 32 - to_bytes;
        for i in start..32 {
            let shifted = b.mul(trunc_val, two_to_8);
            if i == 31 {
                // Last iteration: use the caller's result ValueId
                b.push(OpCode::BinaryArithOp {
                    kind: BinaryArithOpKind::Add,
                    result,
                    lhs: shifted,
                    rhs: bytes[i],
                });
            } else {
                trunc_val = b.add(shifted, bytes[i]);
            }
        }
    }

    fn gen_witness_rangecheck(
        &self,
        b: &mut HLInstrBuilder<'_>,
        value: ValueId,
        max_bits: usize,
        flag: ValueId,
    ) {
        assert!(max_bits % 8 == 0); // TODO
        let chunks = max_bits / 8;
        let pure_value = b.value_of(value);
        let bytes_val = b.fresh_value();
        b.push(OpCode::ToRadix {
            result: bytes_val,
            value: pure_value,
            radix: Radix::Bytes,
            endianness: Endianness::Big,
            count: chunks,
        });
        let mut result = b.field_const(Field::ZERO);
        let two_to_8 = b.field_const(Field::from(256));
        let one = b.field_const(Field::from(1));
        for i in 0..chunks {
            let idx = b.u_const(32, i as u128);
            let byte = b.array_get(bytes_val, idx);
            let byte_field = b.cast_to_field(byte);
            let byte_wit = b.write_witness(byte_field);
            b.lookup_rngchk_8(byte_wit, flag);
            let shift_prev_res = b.mul(result, two_to_8);
            result = b.add(shift_prev_res, byte_wit);
        }
        // Conditional reconstruction constraint: flag * (result - value) = 0
        let diff = b.sub(result, value);
        let zero = b.field_const(Field::ZERO);
        b.constrain(flag, diff, zero);
    }
}
