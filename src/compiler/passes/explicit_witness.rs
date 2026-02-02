use std::collections::HashMap;

use ark_ff::{AdditiveGroup, Field as _};
use ssa_builder::{ssa_append};

use crate::compiler::{
    Field,
    analysis::types::TypeInfo,
    ir::r#type::{Type, TypeExpr},
    pass_manager::{DataPoint, Pass},
    ssa::{
        BinaryArithOpKind, Block, BlockId, CastTarget, CmpKind, Endianness, Function, LookupTarget, OpCode, Radix, SSA, SeqType, ValueId
    },
    taint_analysis::ConstantTaint,
};

pub struct ExplicitWitness {}

impl Pass<ConstantTaint> for ExplicitWitness {
    fn run(
        &self,
        ssa: &mut SSA<ConstantTaint>,
        pass_manager: &crate::compiler::pass_manager::PassManager<ConstantTaint>,
    ) {
        self.do_run(ssa, pass_manager.get_type_info());
    }

    fn pass_info(&self) -> crate::compiler::pass_manager::PassInfo {
        crate::compiler::pass_manager::PassInfo {
            name: "explicit_witness",
            needs: vec![DataPoint::Types],
        }
    }

    fn invalidates_cfg(&self) -> bool {
        false
    }
}

impl ExplicitWitness {
    pub fn new() -> Self {
        Self {}
    }

    pub fn do_run(&self, ssa: &mut SSA<ConstantTaint>, type_info: &TypeInfo<ConstantTaint>) {
        for (function_id, function) in ssa.iter_functions_mut() {
            let function_type_info = type_info.get_function(*function_id);
            let mut new_blocks = HashMap::<BlockId, Block<ConstantTaint>>::new();
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
                            let l_taint = function_type_info.get_value_type(lhs).get_annotation();
                            let r_taint = function_type_info.get_value_type(rhs).get_annotation();
                            if !(l_taint.is_pure() && r_taint.is_pure()) {
                                assert!(function_type_info.get_value_type(rhs).is_numeric());
                                assert!(function_type_info.get_value_type(lhs).is_numeric());
                                match kind {
                                    CmpKind::Eq => {
                                        let u1 = CastTarget::U(1);
                                        ssa_append!(function, new_instructions, {
                                            l_field := cast_to_field(lhs);
                                            r_field := cast_to_field(rhs);
                                            lr_diff := sub(l_field, r_field);

                                            div_hint := div(lr_diff, lr_diff);
                                            div_hint_witness := write_witness(div_hint);

                                            out_hint := eq(lhs, rhs);
                                            out_hint_field := cast_to_field(out_hint);
                                            out_hint_witness := write_witness(out_hint_field);
                                            #result := cast_to(u1, out_hint_witness);

                                            not_res := sub(! Field::ONE : Field, result);

                                            constrain(lr_diff, div_hint_witness, not_res);
                                            constrain(lr_diff, result, ! Field::ZERO : Field);


                                        } ->);
                                    }
                                    CmpKind::Lt => {
                                        let TypeExpr::U(s) = function_type_info.get_value_type(rhs).expr else {
                                            panic!("ICE: rhs is not a U type");
                                        };
                                        let u1 = CastTarget::U(1);
                                        let r = ssa_append!(function, new_instructions, {
                                            res_hint := lt(lhs, rhs);
                                            res_hint_field := cast_to_field(res_hint);
                                            res_witness := write_witness(res_hint_field);
                                            #result := cast_to(u1, res_witness);

                                            l_field := cast_to_field(lhs);
                                            r_field := cast_to_field(rhs);
                                            lr_diff := sub(l_field, r_field);

                                            two_res := mul(result, ! Field::from(2) : Field);
                                            adjustment := sub(! Field::ONE : Field, two_res);
                                            
                                            adjusted_diff := mul(lr_diff, adjustment);
                                            adjusted_diff_wit := write_witness(adjusted_diff);
                                            constrain(lr_diff, adjustment, adjusted_diff_wit);
                                        } -> adjusted_diff_wit);
                                        self.gen_witness_rangecheck(function, &mut new_instructions, r.adjusted_diff_wit, s);
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
                            let l_taint = function_type_info.get_value_type(l).get_annotation();
                            let r_taint = function_type_info.get_value_type(r).get_annotation();

                            if l_taint.is_pure() || r_taint.is_pure() {
                                new_instructions.push(instruction);
                                continue;
                            }

                            // witness-witness mul
                            let mul_witness = function.fresh_value();
                            new_instructions.push(OpCode::BinaryArithOp {
                                kind: BinaryArithOpKind::Mul,
                                result: mul_witness,
                                lhs: l,
                                rhs: r,
                            });
                            new_instructions.push(OpCode::WriteWitness {
                                result: Some(res),
                                value: mul_witness,
                                witness_annotation: ConstantTaint::Witness,
                            });
                            new_instructions.push(OpCode::Constrain { a: l, b: r, c: res });
                        }
                        OpCode::BinaryArithOp {
                            kind: BinaryArithOpKind::Div,
                            result: _,
                            lhs: l,
                            rhs: r,
                        } => {
                            let l_taint = function_type_info.get_value_type(l).get_annotation();
                            let r_taint = function_type_info.get_value_type(r).get_annotation();
                            assert!(l_taint.is_pure());
                            assert!(r_taint.is_pure());
                            new_instructions.push(instruction);
                        }
                        OpCode::BinaryArithOp {
                            kind: BinaryArithOpKind::And,
                            result,
                            lhs: l,
                            rhs: r,
                        } => {
                            let l_taint = function_type_info.get_value_type(l).get_annotation();
                            let r_taint = function_type_info.get_value_type(r).get_annotation();
                            match (l_taint, r_taint) {
                                (ConstantTaint::Pure, ConstantTaint::Pure) => {
                                    new_instructions.push(instruction);
                                }
                                (ConstantTaint::Witness, ConstantTaint::Witness) => {
                                    let u1 = CastTarget::U(1);
                                    ssa_append!(function, new_instructions, {
                                        l_field := cast_to_field(l);
                                        r_field := cast_to_field(r);
                                        res_hint := and(l, r);
                                        res_hint_field := cast_to_field(res_hint);
                                        res_witness := write_witness(res_hint_field);
                                        constrain(l_field, r_field, res_witness);
                                        #result := cast_to(u1, res_witness);
                                    } ->);
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
                            let ptr_taint = function_type_info.get_value_type(ptr).get_annotation();
                            assert!(ptr_taint.is_pure());
                            new_instructions.push(instruction);
                        }
                        OpCode::Load { result: _, ptr } => {
                            let ptr_taint = function_type_info.get_value_type(ptr).get_annotation();
                            assert!(ptr_taint.is_pure());
                            new_instructions.push(instruction);
                        }
                        OpCode::AssertEq { lhs: l, rhs: r } => {
                            let l_type = function_type_info.get_value_type(l);
                            let r_type = function_type_info.get_value_type(r);
                            let l_taint = l_type.get_annotation();
                            let r_taint = r_type.get_annotation();
                            if l_taint.is_pure() && r_taint.is_pure() {
                                new_instructions.push(instruction);
                                continue;
                            }
                            let one = function.push_field_const(ark_ff::Fp::from(1));
                            let l = if l_type.is_field() {
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
                            let r = if r_type.is_field() {
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
                            let a_taint = function_type_info.get_value_type(a).get_annotation();
                            let b_taint = function_type_info.get_value_type(b).get_annotation();
                            let c_taint = function_type_info.get_value_type(c).get_annotation();
                            if a_taint.is_pure() && b_taint.is_pure() && c_taint.is_pure() {
                                new_instructions.push(instruction);
                                continue;
                            }
                            new_instructions.push(OpCode::Constrain { a: a, b: b, c: c });
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
                            let arr_taint = function_type_info.get_value_type(arr).get_annotation();
                            let idx_type = function_type_info.get_value_type(idx);
                            let idx_taint = idx_type.get_annotation();
                            assert!(arr_taint.is_pure());
                            match idx_taint {
                                ConstantTaint::Pure => {
                                    new_instructions.push(instruction);
                                }
                                ConstantTaint::Witness => {
                                    let back_cast_target = match &idx_type.expr {
                                        TypeExpr::U(s) => CastTarget::U(*s),
                                        TypeExpr::Field => CastTarget::Field,
                                        TypeExpr::WitnessRef => CastTarget::Field,
                                        TypeExpr::Array(_, _) => {
                                            todo!("array types in witnessed array reads")
                                        }
                                        TypeExpr::Slice(_) => {
                                            todo!("slice types in witnessed array reads")
                                        }
                                        TypeExpr::Ref(_) => {
                                            todo!("ref types in witnessed array reads")
                                        }
                                        TypeExpr::Tuple(_elements) => {todo!("Tuples not supported yet")}
                                    };

                                    ssa_append!(function, new_instructions, {
                                        idx_field := cast_to_field(idx);
                                        r_wit_val := array_get(arr, idx);
                                        r_wit_field := cast_to_field(r_wit_val);
                                        r_wit := write_witness(r_wit_field);
                                        #result := cast_to(back_cast_target, r_wit);
                                        lookup_arr(arr, idx_field, r_wit);
                                    } ->);
                                }
                            }
                        }
                        OpCode::ArraySet {
                            result: _,
                            array: arr,
                            index: idx,
                            value: _,
                        } => {
                            let arr_taint = function_type_info.get_value_type(arr).get_annotation();
                            let idx_taint = function_type_info.get_value_type(idx).get_annotation();
                            assert!(arr_taint.is_pure());
                            assert!(idx_taint.is_pure());
                            new_instructions.push(instruction);
                        }
                        OpCode::SlicePush {
                            dir: _,
                            result: _,
                            slice: sl,
                            values: _,
                        } => {
                            let slice_taint = function_type_info.get_value_type(sl).get_annotation();
                            assert!(slice_taint.is_pure());
                            new_instructions.push(instruction);
                        }
                        OpCode::SliceLen {
                            result: _,
                            slice: sl,
                        } => {
                            let slice_taint = function_type_info.get_value_type(sl).get_annotation();
                            assert!(slice_taint.is_pure());
                            new_instructions.push(instruction);
                        }
                        OpCode::Select {
                            result: res,
                            cond,
                            if_t: l,
                            if_f: r,
                        } => {
                            let cond_taint =
                                function_type_info.get_value_type(cond).get_annotation();
                            let l_taint = function_type_info.get_value_type(l).get_annotation();
                            let r_taint = function_type_info.get_value_type(r).get_annotation();
                            // The result is cond * l + (1 - cond) * r
                            // If either cond or both l and r and pure, this becomes a linear combination
                            // and as such doesn't need a witness
                            if cond_taint.is_pure() || (l_taint.is_pure() && r_taint.is_pure()) {
                                new_instructions.push(instruction);
                                continue;
                            }
                            let select_witness = function.fresh_value();
                            new_instructions.push(OpCode::Select {
                                result: select_witness,
                                cond: cond,
                                if_t: l,
                                if_f: r,
                            });
                            new_instructions.push(OpCode::WriteWitness {
                                result: Some(res),
                                value: select_witness,
                                witness_annotation: ConstantTaint::Witness,
                            });
                            // Goal is to assert 0 = cond * l + (1 - cond) * r - res
                            // This is equivalent to 0 = cond * (l - r) + r - res = cond * (l - r) - (res - r)
                            let neg_one = function.push_field_const(ark_ff::Fp::from(-1));
                            let neg_r = function.fresh_value();
                            new_instructions.push(OpCode::BinaryArithOp {
                                kind: BinaryArithOpKind::Mul,
                                result: neg_r,
                                lhs: r,
                                rhs: neg_one,
                            });
                            let l_sub_r = function.fresh_value();
                            new_instructions.push(OpCode::BinaryArithOp {
                                kind: BinaryArithOpKind::Add,
                                result: l_sub_r,
                                lhs: l,
                                rhs: neg_r,
                            });
                            let res_sub_r = function.fresh_value();
                            new_instructions.push(OpCode::BinaryArithOp {
                                kind: BinaryArithOpKind::Add,
                                result: res_sub_r,
                                lhs: res,
                                rhs: neg_r,
                            });
                            new_instructions.push(OpCode::Constrain {
                                a: cond,
                                b: l_sub_r,
                                c: res_sub_r,
                            });
                        }

                        OpCode::MkSeq {
                            result: _,
                            elems: _,
                            seq_type: _,
                            elem_type: _,
                        } => {
                            new_instructions.push(instruction);
                        }
                        OpCode::MkTuple {..} => {
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
                            let i_taint = function_type_info.get_value_type(i).get_annotation();
                            assert!(i_taint.is_pure()); // TODO: witness versions
                            new_instructions.push(instruction);
                        }
                        OpCode::Not { result, value } => {
                            match &function_type_info.get_value_type(value).expr {
                                TypeExpr::U(s) => {
                                    let ones = function.push_field_const(Field::from((1u128 << *s) - 1));
                                    let casted = function.fresh_value();
                                    new_instructions.push(OpCode::Cast {
                                        result: casted,
                                        value: value,
                                        target: CastTarget::Field,
                                    });
                                    let subbed = function.fresh_value();
                                    new_instructions.push(OpCode::BinaryArithOp {
                                        kind: BinaryArithOpKind::Sub,
                                        result: subbed,
                                        lhs: ones,
                                        rhs: casted,
                                    });
                                    new_instructions.push(OpCode::Cast {
                                        result: result,
                                        value: subbed,
                                        target: CastTarget::U(*s),
                                    });
                                }
                                e => todo!("Unsupported type for negation: {:?}", e),
                            }
                        }
                        OpCode::ToBits {
                            result: _,
                            value: i,
                            endianness: _,
                            count: _,
                        } => {
                            let i_taint = function_type_info.get_value_type(i).get_annotation();
                            assert!(i_taint.is_pure()); // Only handle pure input case for now
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
                                function_type_info.get_value_type(value).get_annotation();
                            if value_taint.is_pure() {
                                new_instructions.push(instruction);
                            } else {
                                assert!(endianness == Endianness::Little);
                                let hint = function.fresh_value();
                                new_instructions.push(OpCode::ToRadix {
                                    result: hint,
                                    value,
                                    radix,
                                    endianness: Endianness::Little,
                                    count,
                                });
                                let mut witnesses = vec![ValueId(0); count];
                                let mut current_sum = function.push_field_const(Field::ZERO);
                                let radix_val = match radix {
                                    Radix::Bytes => function.push_field_const(Field::from(256)),
                                    Radix::Dyn(radix) => {
                                        let casted = function.fresh_value();
                                        new_instructions.push(OpCode::Cast {
                                            result: casted,
                                            value: radix,
                                            target: CastTarget::Field,
                                        });
                                        casted
                                    }
                                };
                                let rangecheck_type = match radix {
                                    Radix::Bytes => LookupTarget::Rangecheck(8),
                                    Radix::Dyn(radix) => LookupTarget::DynRangecheck(radix),
                                };
                                // TODO this should probably be an SSA loop for codesize reasons.
                                for i in (0..count).rev() {
                                    let r = ssa_append!(function, new_instructions, {
                                        byte := array_get(hint, ! i as u128 : u32);
                                        byte_field := cast_to_field(byte);
                                        byte_wit := write_witness(byte_field);
                                        lookup_rngchk(rangecheck_type, byte_wit);
                                        shift_prev_res := mul(current_sum, radix_val);
                                        new_result := add(shift_prev_res, byte_wit);
                                    } -> new_result, byte_wit);
                                    current_sum = r.new_result;
                                    witnesses[i] = r.byte_wit;
                                }

                                new_instructions.push(OpCode::Constrain {
                                    a: current_sum,
                                    b: function.push_field_const(Field::from(1)),
                                    c: value,
                                });

                                new_instructions.push(OpCode::MkSeq {
                                    result: result,
                                    elems: witnesses,
                                    seq_type: SeqType::Array(count),
                                    elem_type: Type::field(ConstantTaint::Witness),
                                });
                            }
                        }

                        OpCode::MemOp { kind: _, value: _ } => {
                            new_instructions.push(instruction);
                        }
                        OpCode::PureToWitnessRef {
                            result: _,
                            value: _,
                            result_annotation: _,
                        } => {
                            new_instructions.push(instruction);
                        }
                        OpCode::UnboxField {
                            result: _,
                            value: _,
                        } => {
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
                            let v_taint = function_type_info.get_value_type(value).get_annotation();
                            if v_taint.is_pure() {
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
                        OpCode::TupleProj {
                            result,
                            tuple,
                            idx,
                        } => {
                            if let crate::compiler::ssa::TupleIdx::Static(index) = idx {
                                let tuple_taint =
                                    function_type_info.get_value_type(tuple).get_annotation();
                                assert!(tuple_taint.is_pure());
                                new_instructions.push(OpCode::TupleProj {
                                    result,
                                    tuple,
                                    idx: crate::compiler::ssa::TupleIdx::Static(index),
                                });
                            } else {
                                panic!("Dynamic tuple indexing should not appear here");
                            }
                        },
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
        function: &mut Function<ConstantTaint>,
        new_instructions: &mut Vec<OpCode<ConstantTaint>>,
        value: ValueId,
        max_bits: usize,
    ) {
        assert!(max_bits % 8 == 0); // TODO
        let bytes_val = function.fresh_value();
        new_instructions.push(OpCode::ToRadix {
            result: bytes_val,
            value: value,
            radix: Radix::Bytes,
            endianness: Endianness::Big,
            count: max_bits / 8,
        });
        let chunks = max_bits / 8;
        let mut result = function.push_field_const(Field::ZERO);
        let two_to_8 = function.push_field_const(Field::from(256));
        let one = function.push_field_const(Field::from(1));
        for i in 0..chunks {
            result = ssa_append!(function, new_instructions, {
                byte := array_get(bytes_val, ! i as u128 : u32);
                byte_field := cast_to_field(byte);
                byte_wit := write_witness(byte_field);
                lookup_rngchk_8(byte_wit);
                shift_prev_res := mul(result, two_to_8);
                new_result := add(shift_prev_res, byte_wit);
            } -> new_result)
            .new_result;
        }
        new_instructions.push(OpCode::Constrain {
            a: result,
            b: one,
            c: value,
        });
    }
}
