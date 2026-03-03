use std::collections::HashMap;

use ark_ff::{AdditiveGroup, Field as _};

use crate::compiler::{
    Field,
    analysis::types::TypeInfo,
    block_builder::InstrBuilder,
    flow_analysis::FlowAnalysis,
    ir::r#type::{Type, TypeExpr},
    pass_manager::{Analysis, AnalysisId, AnalysisStore, Pass},
    ssa::{
        BinaryArithOpKind, BlockId, CastTarget, CmpKind, Endianness, HLBlock, HLFunction, HLSSA,
        LookupTarget, OpCode, Radix, SeqType, ValueId,
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
                    match instruction {
                        OpCode::BinaryArithOp {
                            kind: BinaryArithOpKind::Add,
                            ..
                        } => {
                            new_instructions.push(instruction);
                        }
                        OpCode::BinaryArithOp {
                            kind: BinaryArithOpKind::Sub,
                            ..
                        } => {
                            new_instructions.push(instruction);
                        }
                        OpCode::Alloc { .. }
                        | OpCode::Call { .. }
                        | OpCode::Constrain { .. }
                        | OpCode::WriteWitness { .. }
                        | OpCode::FreshWitness {
                            result: _,
                            result_type: _,
                        } => {
                            new_instructions.push(instruction);
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
                                            let v = function.fresh_value();
                                            new_instructions.push(OpCode::mk_cast_to_field(v, lhs));
                                            v
                                        };
                                        let r_field = if function_type_info
                                            .get_value_type(rhs)
                                            .strip_witness()
                                            .is_field()
                                        {
                                            rhs
                                        } else {
                                            let v = function.fresh_value();
                                            new_instructions.push(OpCode::mk_cast_to_field(v, rhs));
                                            v
                                        };
                                        let b =
                                            &mut InstrBuilder::new(function, &mut new_instructions);
                                        let lr_diff = b.sub(l_field, r_field);

                                        let div_hint = b.div(lr_diff, lr_diff);
                                        let div_hint_plain = b.value_of(div_hint);
                                        let div_hint_witness = b.write_witness(div_hint_plain);

                                        let out_hint = b.eq(lhs, rhs);
                                        let out_hint_field = b.cast_to_field(out_hint);
                                        let out_hint_plain = b.value_of(out_hint_field);
                                        let out_hint_witness = b.write_witness(out_hint_plain);
                                        b.push(OpCode::mk_cast_to(result, u1, out_hint_witness));

                                        let result_field = b.cast_to_field(result);
                                        let field_one = b.field_const(Field::ONE);
                                        let not_res = b.sub(field_one, result_field);

                                        b.constrain(lr_diff, div_hint_witness, not_res);
                                        let field_zero = b.field_const(Field::ZERO);
                                        b.constrain(lr_diff, result_field, field_zero);
                                    }
                                    CmpKind::Lt => {
                                        let TypeExpr::U(s) = function_type_info
                                            .get_value_type(rhs)
                                            .strip_witness()
                                            .expr
                                        else {
                                            panic!("ICE: rhs is not a U type");
                                        };
                                        let u1 = CastTarget::U(1);
                                        let b =
                                            &mut InstrBuilder::new(function, &mut new_instructions);
                                        let res_hint = b.lt(lhs, rhs);
                                        let res_hint_field = b.cast_to_field(res_hint);
                                        let res_hint_plain = b.value_of(res_hint_field);
                                        let res_witness = b.write_witness(res_hint_plain);
                                        b.push(OpCode::mk_cast_to(result, u1, res_witness));

                                        let l_field = b.cast_to_field(lhs);
                                        let r_field = b.cast_to_field(rhs);
                                        let lr_diff = b.sub(l_field, r_field);

                                        let two = b.field_const(Field::from(2));
                                        let two_res = b.mul(result, two);
                                        let one = b.field_const(Field::ONE);
                                        let adjustment = b.sub(one, two_res);

                                        let adjusted_diff = b.mul(lr_diff, adjustment);
                                        let adjusted_diff_plain = b.value_of(adjusted_diff);
                                        let adjusted_diff_wit =
                                            b.write_witness(adjusted_diff_plain);
                                        b.constrain(lr_diff, adjustment, adjusted_diff_wit);

                                        self.gen_witness_rangecheck(
                                            function,
                                            &mut new_instructions,
                                            adjusted_diff_wit,
                                            s,
                                        );
                                    }
                                }
                            } else {
                                new_instructions.push(instruction);
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
                                new_instructions.push(instruction);
                                continue;
                            }

                            // witness-witness mul
                            let b = &mut InstrBuilder::new(function, &mut new_instructions);
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
                            new_instructions.push(instruction);
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
                                    new_instructions.push(instruction);
                                }
                                (true, true) => {
                                    let u1 = CastTarget::U(1);
                                    let b = &mut InstrBuilder::new(function, &mut new_instructions);
                                    let l_field = b.cast_to_field(l);
                                    let r_field = b.cast_to_field(r);
                                    let res_hint = b.and(l, r);
                                    let res_hint_field = b.cast_to_field(res_hint);
                                    let res_hint_plain = b.value_of(res_hint_field);
                                    let res_witness = b.write_witness(res_hint_plain);
                                    b.constrain(l_field, r_field, res_witness);
                                    b.push(OpCode::mk_cast_to(result, u1, res_witness));
                                }
                                _ => {
                                    new_instructions.push(OpCode::Todo {
                                        payload: format!("witness AND {} {}", l_taint, r_taint),
                                        results: vec![result],
                                        result_types: vec![
                                            function_type_info.get_value_type(r).clone(),
                                        ],
                                    });
                                }
                            }
                        }
                        OpCode::Store { ptr, value: _ } => {
                            let ptr_taint = function_type_info.get_value_type(ptr).is_witness_of();
                            assert!(!ptr_taint);
                            new_instructions.push(instruction);
                        }
                        OpCode::Load { result: _, ptr } => {
                            let ptr_taint = function_type_info.get_value_type(ptr).is_witness_of();
                            assert!(!ptr_taint);
                            new_instructions.push(instruction);
                        }
                        OpCode::AssertEq { lhs: l, rhs: r } => {
                            let l_type = function_type_info.get_value_type(l);
                            let r_type = function_type_info.get_value_type(r);
                            let l_taint = l_type.is_witness_of();
                            let r_taint = r_type.is_witness_of();
                            if !l_taint && !r_taint {
                                new_instructions.push(instruction);
                                continue;
                            }
                            let one = function.fresh_value();
                            new_instructions.push(OpCode::mk_field_const(one, ark_ff::Fp::from(1)));
                            let l = if l_type.strip_witness().is_field() {
                                l
                            } else {
                                let casted = function.fresh_value();
                                new_instructions.push(OpCode::Cast {
                                    result: casted,
                                    value: l,
                                    target: CastTarget::Field,
                                });
                                casted
                            };
                            let r = if r_type.strip_witness().is_field() {
                                r
                            } else {
                                let casted = function.fresh_value();
                                new_instructions.push(OpCode::Cast {
                                    result: casted,
                                    value: r,
                                    target: CastTarget::Field,
                                });
                                casted
                            };
                            new_instructions.push(OpCode::Constrain { a: l, b: one, c: r });
                        }
                        OpCode::AssertR1C { a, b, c } => {
                            let a_taint = function_type_info.get_value_type(a).is_witness_of();
                            let b_taint = function_type_info.get_value_type(b).is_witness_of();
                            let c_taint = function_type_info.get_value_type(c).is_witness_of();
                            if !a_taint && !b_taint && !c_taint {
                                new_instructions.push(instruction);
                                continue;
                            }
                            new_instructions.push(OpCode::Constrain { a, b, c });
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
                                    new_instructions.push(instruction);
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
                                        TypeExpr::Function => panic!(
                                            "Function type not expected in witnessed array reads"
                                        ),
                                    };

                                    let b = &mut InstrBuilder::new(function, &mut new_instructions);
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
                                    b.push(OpCode::mk_cast_to(result, back_cast_target, r_wit));
                                    b.lookup_arr(arr, idx_field, r_wit);
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
                            new_instructions.push(instruction);
                        }
                        OpCode::SlicePush {
                            dir: _,
                            result: _,
                            slice: sl,
                            values: _,
                        } => {
                            let slice_taint = function_type_info.get_value_type(sl).is_witness_of();
                            assert!(!slice_taint);
                            new_instructions.push(instruction);
                        }
                        OpCode::SliceLen {
                            result: _,
                            slice: sl,
                        } => {
                            let slice_taint = function_type_info.get_value_type(sl).is_witness_of();
                            assert!(!slice_taint);
                            new_instructions.push(instruction);
                        }
                        OpCode::Select {
                            result: res,
                            cond,
                            if_t: l,
                            if_f: r,
                        } => {
                            let cond_taint =
                                function_type_info.get_value_type(cond).is_witness_of();
                            let l_taint = function_type_info.get_value_type(l).is_witness_of();
                            let r_taint = function_type_info.get_value_type(r).is_witness_of();
                            // The result is cond * l + (1 - cond) * r = cond * (l - r) + r
                            // If cond is pure, this is just a conditional move
                            if !cond_taint {
                                new_instructions.push(instruction);
                                continue;
                            }
                            // If both branches are pure, result is a linear combination
                            // of cond and constants: cond * (l - r) + r. No constraint needed.
                            if !l_taint && !r_taint {
                                let b = &mut InstrBuilder::new(function, &mut new_instructions);
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
                                continue;
                            }
                            // At least one branch is witness: full lowering with constraint
                            let b = &mut InstrBuilder::new(function, &mut new_instructions);
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
                            b.constrain(cond, l_sub_r, res_sub_r);
                        }

                        OpCode::MkSeq {
                            result: _,
                            elems: _,
                            seq_type: _,
                            elem_type: _,
                        } => {
                            new_instructions.push(instruction);
                        }
                        OpCode::MkTuple { .. } => {
                            new_instructions.push(instruction);
                        }
                        OpCode::Cast {
                            result: _,
                            value: _,
                            target: _,
                        } => {
                            new_instructions.push(instruction);
                        }
                        OpCode::Truncate {
                            result: _,
                            value: i,
                            to_bits: _,
                            from_bits: _,
                        } => {
                            let i_taint = function_type_info.get_value_type(i).is_witness_of();
                            assert!(!i_taint); // TODO: witness versions
                            new_instructions.push(instruction);
                        }
                        OpCode::Not { result, value } => {
                            let value_type = function_type_info.get_value_type(value);
                            let s = match &value_type.strip_witness().expr {
                                TypeExpr::U(s) => *s,
                                e => todo!("Unsupported type for negation: {:?}", e),
                            };
                            let b = &mut InstrBuilder::new(function, &mut new_instructions);
                            let ones = b.field_const(Field::from((1u128 << s) - 1));
                            let casted = b.cast_to(CastTarget::Field, value);
                            let subbed = b.sub(ones, casted);
                            b.push(OpCode::mk_cast_to(result, CastTarget::U(s), subbed));
                        }
                        OpCode::ToBits {
                            result: _,
                            value: i,
                            endianness: _,
                            count: _,
                        } => {
                            let i_taint = function_type_info.get_value_type(i).is_witness_of();
                            assert!(!i_taint); // Only handle pure input case for now
                            new_instructions.push(instruction);
                        }
                        OpCode::ToRadix {
                            result,
                            value,
                            radix,
                            endianness,
                            count,
                        } => {
                            let value_taint =
                                function_type_info.get_value_type(value).is_witness_of();
                            if !value_taint {
                                new_instructions.push(instruction);
                            } else {
                                assert!(endianness == Endianness::Little);
                                let b = &mut InstrBuilder::new(function, &mut new_instructions);
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
                                    b.lookup_rngchk(rangecheck_type, byte_wit);
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
                            new_instructions.push(instruction);
                        }
                        OpCode::MulConst {
                            result: _,
                            const_val: _,
                            var: _,
                        } => {
                            new_instructions.push(instruction);
                        }
                        OpCode::Rangecheck { value, max_bits } => {
                            let v_taint = function_type_info.get_value_type(value).is_witness_of();
                            if !v_taint {
                                new_instructions.push(instruction);
                            } else {
                                self.gen_witness_rangecheck(
                                    function,
                                    &mut new_instructions,
                                    value,
                                    max_bits,
                                );
                            }
                        }
                        OpCode::ReadGlobal {
                            result: _,
                            offset: _,
                            result_type: _,
                        } => {
                            new_instructions.push(instruction);
                        }
                        OpCode::Lookup { .. } => {
                            new_instructions.push(instruction);
                        }
                        OpCode::DLookup { .. } => {
                            new_instructions.push(instruction);
                        }
                        OpCode::Todo {
                            payload,
                            results,
                            result_types,
                        } => {
                            new_instructions.push(OpCode::Todo {
                                payload,
                                results,
                                result_types,
                            });
                        }
                        OpCode::TupleProj { result, tuple, idx } => {
                            if let crate::compiler::ssa::TupleIdx::Static(index) = idx {
                                let tuple_taint =
                                    function_type_info.get_value_type(tuple).is_witness_of();
                                assert!(!tuple_taint);
                                new_instructions.push(OpCode::TupleProj {
                                    result,
                                    tuple,
                                    idx: crate::compiler::ssa::TupleIdx::Static(index),
                                });
                            } else {
                                panic!("Dynamic tuple indexing should not appear here");
                            }
                        }
                        OpCode::InitGlobal { .. }
                        | OpCode::DropGlobal { .. }
                        | OpCode::ValueOf { .. }
                        | OpCode::Const { .. } => {
                            new_instructions.push(instruction);
                        }
                    }
                }
                block.put_instructions(new_instructions);
                new_blocks.insert(bid, block);
            }
            function.put_blocks(new_blocks);
        }
    }

    fn gen_witness_rangecheck(
        &self,
        function: &mut HLFunction,
        new_instructions: &mut Vec<OpCode>,
        value: ValueId,
        max_bits: usize,
    ) {
        assert!(max_bits % 8 == 0); // TODO
        let chunks = max_bits / 8;
        let b = &mut InstrBuilder::new(function, new_instructions);
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
            b.lookup_rngchk_8(byte_wit);
            let shift_prev_res = b.mul(result, two_to_8);
            result = b.add(shift_prev_res, byte_wit);
        }
        b.constrain(result, one, value);
    }
}
