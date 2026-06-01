//! Lowers high-level operations on witness-tainted values into the explicit primitives that can be
//! used by the R1CS backend for constraints.
//!
//! It runs late in the pipeline right before R1CS generation, and is the main tool that makes the
//! subsequent R1CS lowering possible.

use std::collections::HashMap;

use ark_ff::{AdditiveGroup, Field as _};

use crate::compiler::{
    Field,
    analysis::{
        flow_analysis::FlowAnalysis,
        types::{FunctionTypeInfo, TypeInfo},
        value_range_analysis::{FunctionValueRanges, IntInterval, ValueRanges},
    },
    pass_manager::{Analysis, AnalysisId, AnalysisStore, Pass},
    ssa::{
        BlockId, ValueId,
        hlssa::{
            BinaryArithOpKind, CastTarget, Endianness, HLBlock, HLSSA, LookupTarget, OpCode, Radix,
            SequenceTargetType, Type, TypeExpr,
            builder::{HLEmitter, HLInstrBuilder, HLSSABuilder},
        },
    },
};

const SPREAD_SPILL_THRESHOLD_BITS: u8 = 16;

pub struct ExplicitWitness {}

impl Pass for ExplicitWitness {
    fn name(&self) -> &'static str {
        "explicit_witness"
    }

    fn needs(&self) -> Vec<AnalysisId> {
        vec![TypeInfo::id(), ValueRanges::id()]
    }

    fn run(&self, ssa: &mut HLSSA, store: &AnalysisStore) {
        self.do_run(ssa, store.get::<TypeInfo>(), store.get::<ValueRanges>());
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
        let fids: Vec<_> = ssa.get_function_ids().collect();
        let mut sb = HLSSABuilder::new(ssa);
        for function_id in fids {
            let function_type_info = type_info.get_function(function_id);
            let function_value_ranges = value_ranges.get_function(function_id);
            sb.modify_function(function_id, |fb| {
                let mut new_blocks = HashMap::<BlockId, HLBlock>::new();
                for (bid, mut block) in fb.function.take_blocks().into_iter() {
                    let mut new_instructions = Vec::new();
                    for instruction in block.take_instructions().into_iter() {
                        let b =
                            &mut HLInstrBuilder::new(fb.function, fb.ssa, &mut new_instructions);
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
                fb.function.put_blocks(new_blocks);
            });
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
                result: _,
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
                    TypeExpr::U(_) | TypeExpr::I(_) => {
                        panic!(
                            "witness integer {:?} should have been lowered by \
                             instruction_lowering",
                            kind
                        );
                    }
                    _ => {
                        // Field add/sub: linear, pass through
                        b.push(instruction);
                    }
                }
            }
            OpCode::Alloc { .. }
            | OpCode::ArrayGet { .. }
            | OpCode::ArraySet { .. }
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
                assert!(
                    !l_taint && !r_taint,
                    "witness cmp {:?} should have been lowered by instruction_lowering",
                    kind
                );
                b.push(OpCode::Cmp {
                    kind,
                    result,
                    lhs,
                    rhs,
                });
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
                    TypeExpr::U(_) | TypeExpr::I(_) => {
                        panic!(
                            "witness integer multiplication should have been lowered by \
                             instruction_lowering"
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
                    TypeExpr::U(_) | TypeExpr::I(_) => {
                        panic!(
                            "witness integer {:?} should have been lowered by \
                             instruction_lowering",
                            kind
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
                b.push(OpCode::BinaryArithOp {
                    kind,
                    result,
                    lhs: l,
                    rhs: r,
                });
            }
            OpCode::BinaryArithOp {
                kind: kind @ (BinaryArithOpKind::Shl | BinaryArithOpKind::Shr),
                result: _,
                lhs: l,
                rhs: r,
            } => {
                let l_taint = function_type_info.get_value_type(l).is_witness_of();
                let r_taint = function_type_info.get_value_type(r).is_witness_of();
                if !l_taint && !r_taint {
                    b.push(instruction);
                    return;
                }
                panic!(
                    "witness integer shift {:?} should have been lowered by \
                     instruction_lowering",
                    kind
                );
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
                assert!(
                    !v_taint,
                    "witness assert should have been lowered by instruction_lowering"
                );
                b.push(instruction);
            }
            OpCode::AssertCmp {
                kind,
                lhs: l,
                rhs: r,
            } => {
                let l_type = function_type_info.get_value_type(l);
                let r_type = function_type_info.get_value_type(r);
                let l_taint = l_type.is_witness_of();
                let r_taint = r_type.is_witness_of();
                assert!(
                    !l_taint && !r_taint,
                    "witness assert_cmp {:?} should have been lowered by instruction_lowering",
                    kind
                );
                b.push(instruction);
            }
            OpCode::AssertR1C { a, b: r1c_b, c } => {
                let a_taint = function_type_info.get_value_type(a).is_witness_of();
                let b_taint = function_type_info.get_value_type(r1c_b).is_witness_of();
                let c_taint = function_type_info.get_value_type(c).is_witness_of();
                assert!(
                    !a_taint && !b_taint && !c_taint,
                    "witness assert_r1c should have been lowered by instruction_lowering"
                );
                b.push(instruction);
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
                // At least one branch is witness: full lowering with constraint.
                let cond_pure = b.value_of(cond);
                let l_pure = if l_taint { b.value_of(l) } else { l };
                let r_pure = if r_taint { b.value_of(r) } else { r };
                let select_hint_value = b.select(cond_pure, l_pure, r_pure);
                let is_field = l_type.strip_witness().is_field();
                let select_hint = if is_field {
                    select_hint_value
                } else {
                    b.cast_to_field(select_hint_value)
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
            OpCode::MkRepeated { .. } => {
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
            OpCode::SExt {
                result,
                value,
                from_bits,
                to_bits,
            } => {
                let value_type = function_type_info.get_value_type(value);
                assert!(
                    !value_type.is_witness_of(),
                    "witness sign-extension should have been lowered by instruction_lowering"
                );
                let one = b.field_const(Field::ONE);
                let value_field = if value_type.is_field() {
                    value
                } else {
                    b.cast_to_field(value)
                };
                let value_range = function_value_ranges.get(value);
                self.gen_sext(
                    b,
                    value_field,
                    from_bits,
                    to_bits,
                    one,
                    false,
                    result,
                    &value_range,
                );
            }
            OpCode::Not { result, value } => {
                b.push(OpCode::Not { result, value });
            }
            OpCode::BitRange { .. } => {
                panic!("BitRange should have been lowered by instruction_lowering");
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
                // `Radix::Dyn(rv)` only reaches mavros via Noir's
                // `to_be_radix` / `to_le_radix` builtins, which Noir doesn't
                // currently expose to user code; the stdlib only ever calls
                // them with `256`. Codegen also only has a specialised
                // bytecode op for `Radix::Bytes`. Pin the assumption with a
                // runtime `assert(rv == 256)` and proceed as Bytes.
                let radix = match radix {
                    Radix::Dyn(rv) => {
                        let const_256 = b.u_const(32, 256);
                        b.assert_eq(rv, const_256);
                        Radix::Bytes
                    }
                    Radix::Bytes => Radix::Bytes,
                };
                let value_taint = function_type_info.get_value_type(value).is_witness_of();
                if !value_taint {
                    b.push(OpCode::ToRadix {
                        result,
                        value,
                        radix,
                        endianness,
                        count,
                    });
                } else {
                    let pure_value = b.value_of(value);
                    // Compute digit hints in the caller's requested endianness;
                    // the reconstruction below visits them MSB-first either way.
                    let hint = b.fresh_value();
                    b.push(OpCode::ToRadix {
                        result: hint,
                        value: pure_value,
                        radix,
                        endianness,
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
                    // Walk hint indices MSB-first so the Horner accumulator
                    // ends up equal to the reconstructed value.
                    // TODO this should probably be an SSA loop for codesize reasons.
                    let visit_order: Box<dyn Iterator<Item = usize>> = match endianness {
                        Endianness::Little => Box::new((0..count).rev()),
                        Endianness::Big => Box::new(0..count),
                    };
                    for i in visit_order {
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
                    let byte_elems: Vec<ValueId> = witnesses
                        .iter()
                        .map(|&w| b.cast_to(CastTarget::U(8), w))
                        .collect();
                    b.push(OpCode::MkSeq {
                        result,
                        elems: byte_elems,
                        seq_type: SequenceTargetType::Array(count),
                        elem_type: Type::witness_of(Type::u(8)),
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
                    self.gen_witness_rangecheck_bits(b, value, max_bits, one);
                }
            }
            OpCode::ReadGlobal {
                result: _,
                offset: _,
                result_type: _,
            } => {
                b.push(instruction);
            }
            OpCode::Lookup {
                target: LookupTarget::Spread(bits),
                args,
                flag,
            } if bits >= SPREAD_SPILL_THRESHOLD_BITS => {
                assert_eq!(
                    args.len(),
                    2,
                    "Spread lookup must have exactly one key and one result"
                );
                self.lower_wide_spread_lookup(b, function_type_info, args[0], args[1], flag, bits);
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
            OpCode::InitGlobal { .. } | OpCode::DropGlobal { .. } | OpCode::ValueOf { .. } => {
                b.push(instruction);
            }
            OpCode::Spread {
                result,
                value,
                bits,
            } => {
                let is_witness = function_type_info.get_value_type(value).is_witness_of();
                assert!(
                    !is_witness,
                    "witness Spread should have been lowered by instruction_lowering"
                );
                b.push(OpCode::Spread {
                    result,
                    value,
                    bits,
                });
            }
            OpCode::Unspread {
                result_odd,
                result_even,
                value,
                bits,
            } => {
                let is_witness = function_type_info.get_value_type(value).is_witness_of();
                assert!(
                    !is_witness,
                    "witness Unspread should have been lowered by instruction_lowering"
                );
                b.push(OpCode::Unspread {
                    result_odd,
                    result_even,
                    value,
                    bits,
                });
            }
            OpCode::Guard { condition, inner } => {
                self.lower_guard(b, function_type_info, condition, *inner);
            }
        }
    }

    fn lower_guard(
        &self,
        b: &mut HLInstrBuilder<'_>,
        function_type_info: &FunctionTypeInfo,
        condition: ValueId,
        inner: OpCode,
    ) {
        match inner {
            OpCode::Assert { .. } | OpCode::AssertCmp { .. } | OpCode::AssertR1C { .. } => {
                panic!("guarded assert should have been lowered by instruction_lowering");
            }
            OpCode::Cmp {
                kind,
                result,
                lhs,
                rhs,
            } => {
                let l_taint = function_type_info.get_value_type(lhs).is_witness_of();
                let r_taint = function_type_info.get_value_type(rhs).is_witness_of();
                assert!(
                    !l_taint && !r_taint,
                    "guarded witness cmp {:?} should have been lowered by instruction_lowering",
                    kind
                );
                b.push(OpCode::Guard {
                    condition,
                    inner: Box::new(OpCode::Cmp {
                        kind,
                        result,
                        lhs,
                        rhs,
                    }),
                });
            }
            OpCode::Store { ptr, value: v } => {
                // Guard(cond, Store(ptr, v)) → old = Load(ptr); new = Select(cond, v, old); Store(ptr, new)
                let old_value = b.load(ptr);
                let new_value = b.select(condition, v, old_value);
                b.store(ptr, new_value);
            }
            OpCode::SExt {
                result,
                value,
                from_bits,
                to_bits,
            } => {
                assert!(
                    !function_type_info.get_value_type(value).is_witness_of(),
                    "guarded witness sign-extension should have been lowered by \
                     instruction_lowering"
                );
                b.push(OpCode::SExt {
                    result,
                    value,
                    from_bits,
                    to_bits,
                });
            }
            OpCode::BinaryArithOp {
                kind: kind @ (BinaryArithOpKind::Add | BinaryArithOpKind::Sub),
                result: _,
                lhs: l,
                rhs: r,
            } => {
                let l_taint = function_type_info.get_value_type(l).is_witness_of();
                let r_taint = function_type_info.get_value_type(r).is_witness_of();
                let l_type = function_type_info.get_value_type(l);
                match l_type.strip_witness().expr {
                    TypeExpr::U(_) | TypeExpr::I(_) if l_taint || r_taint => {
                        panic!(
                            "guarded witness integer {:?} should have been lowered by \
                             instruction_lowering",
                            kind
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
                    TypeExpr::U(_) | TypeExpr::I(_) if l_taint || r_taint => {
                        panic!(
                            "guarded witness integer multiplication should have been lowered by \
                             instruction_lowering"
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
                    TypeExpr::U(_) | TypeExpr::I(_) if l_taint || r_taint => {
                        panic!(
                            "guarded witness integer {:?} should have been lowered by \
                             instruction_lowering",
                            kind
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
                            let cond_field = b.ensure_field(
                                condition,
                                function_type_info.get_value_type(condition),
                            );
                            self.gen_witness_field_div_guarded(
                                b,
                                l,
                                r,
                                l_taint,
                                r_taint,
                                function_type_info.get_value_type(condition).is_witness_of(),
                                cond_field,
                                res,
                            );
                        } else {
                            b.push(inner);
                        }
                    }
                    _ => unreachable!("DivMod on non-numeric type: {:?}", l_type),
                }
            }
            OpCode::BinaryArithOp {
                kind: kind @ (BinaryArithOpKind::Shl | BinaryArithOpKind::Shr),
                result: _,
                lhs: l,
                rhs: r,
            } => {
                let l_taint = function_type_info.get_value_type(l).is_witness_of();
                let r_taint = function_type_info.get_value_type(r).is_witness_of();
                if !l_taint && !r_taint {
                    b.push(inner);
                    return;
                }
                panic!(
                    "guarded witness integer shift {:?} should have been lowered by \
                     instruction_lowering",
                    kind
                );
            }
            OpCode::BitRange { .. } => {
                panic!("guarded BitRange should have been lowered by instruction_lowering");
            }
            OpCode::Cast { .. }
            | OpCode::MkSeq { .. }
            | OpCode::MkRepeated { .. }
            | OpCode::MkTuple { .. }
            | OpCode::TupleProj { .. } => {
                // Pure data construction/access — no constraints. Emit unconditionally.
                b.push(inner);
            }
            OpCode::WriteWitness {
                result,
                value,
                pinned,
            } => {
                b.push(OpCode::WriteWitness {
                    result,
                    value,
                    pinned,
                });
            }
            OpCode::Rangecheck { value, max_bits } => {
                assert!(
                    function_type_info.get_value_type(value).is_witness_of(),
                    "pure Rangecheck inside Guard should have been lowered by \
                     instruction_lowering"
                );
                let cond_field =
                    b.ensure_field(condition, function_type_info.get_value_type(condition));
                self.gen_witness_rangecheck_bits(b, value, max_bits, cond_field);
            }
            inner => b.push(OpCode::Guard {
                condition,
                inner: Box::new(inner),
            }),
        }
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
        let low_hint = self.pure_low_bits_hint(b, pure_val, bits - 1);
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

    fn pure_low_bits_hint(
        &self,
        b: &mut HLInstrBuilder<'_>,
        value: ValueId,
        bits: usize,
    ) -> ValueId {
        if bits == 0 {
            return b.field_const(Field::ZERO);
        }

        let full_bytes = bits / 8;
        let partial_bits = bits % 8;
        let byte_count = full_bytes + usize::from(partial_bits > 0);
        let bytes = b.to_radix(value, Radix::Bytes, Endianness::Big, byte_count);
        let two_to_8 = b.field_const(Field::from(256u128));
        let mut result = b.field_const(Field::ZERO);

        for i in 0..byte_count {
            let idx = b.u_const(32, i as u128);
            let mut byte = b.array_get(bytes, idx);
            if i == 0 && partial_bits > 0 {
                let mask = b.u_const(8, (1u128 << partial_bits) - 1);
                byte = b.and(byte, mask);
            }
            let byte_field = b.cast_to_field(byte);
            let shifted = b.mul(result, two_to_8);
            result = b.add(shifted, byte_field);
        }

        result
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

    fn gen_witness_field_div_guarded(
        &self,
        b: &mut HLInstrBuilder<'_>,
        a: ValueId,
        divisor: ValueId,
        l_taint: bool,
        r_taint: bool,
        cond_taint: bool,
        cond_field: ValueId,
        result: ValueId,
    ) {
        if l_taint && !r_taint {
            b.push(OpCode::BinaryArithOp {
                kind: BinaryArithOpKind::Div,
                result,
                lhs: a,
                rhs: divisor,
            });
            return;
        }
        let a_pure = if l_taint { b.value_of(a) } else { a };
        let b_pure = b.value_of(divisor);
        let q_hint = b.div(a_pure, b_pure);
        let q_hint_field = b.cast_to_field(q_hint);
        b.push(OpCode::WriteWitness {
            result: Some(result),
            value: q_hint_field,
            pinned: false,
        });
        let a_gated = if l_taint && cond_taint {
            let cond_pure = b.value_of(cond_field);
            let a_gated_hint = b.mul(a_pure, cond_pure);
            let a_gated_wit = b.write_witness(a_gated_hint);
            b.constrain(a, cond_field, a_gated_wit);
            a_gated_wit
        } else {
            b.mul(a, cond_field)
        };
        b.constrain(result, divisor, a_gated);
    }

    fn lower_wide_spread_lookup(
        &self,
        b: &mut HLInstrBuilder<'_>,
        function_type_info: &FunctionTypeInfo,
        key: ValueId,
        expected_spread: ValueId,
        flag: ValueId,
        bits: u8,
    ) {
        assert!(
            bits <= 128,
            "wide Spread lookup spilling currently supports widths up to 128 bits, got {bits}"
        );

        let key_type = function_type_info.get_value_type(key);
        let key_is_witness = key_type.is_witness_of();
        let key_inner = key_type.strip_witness();
        let mut pure_key = if key_is_witness { b.value_of(key) } else { key };
        if key_inner.is_field() {
            pure_key = b.cast_to(CastTarget::U(bits as usize), pure_key);
        }

        let flag_field = b.ensure_field(flag, function_type_info.get_value_type(flag));
        let zero = b.field_const(Field::ZERO);
        let mut reconstructed_key = zero;
        let mut reconstructed_spread = zero;
        let mut offset = 0usize;
        let bits = bits as usize;

        while offset < bits {
            let chunk_bits = (bits - offset).min(8);
            let chunk = extract_low_chunk(b, pure_key, bits, offset, chunk_bits);
            let chunk_field = b.cast_to_field(chunk);
            let chunk_key = if key_is_witness {
                b.write_witness(chunk_field)
            } else {
                chunk_field
            };
            let is_last = offset + chunk_bits == bits;

            let chunk_spread = if is_last {
                let remaining_spread = b.sub(expected_spread, reconstructed_spread);
                let inv_spread_shift = two_pow(offset * 2)
                    .inverse()
                    .expect("non-zero power of two must be invertible");
                let inv_spread_shift = b.field_const(inv_spread_shift);
                b.mul(remaining_spread, inv_spread_shift)
            } else if key_is_witness {
                let spread_hint = b.spread(chunk, chunk_bits as u8);
                let spread_hint_field = b.cast_to_field(spread_hint);
                b.write_witness(spread_hint_field)
            } else {
                let spread = b.spread(chunk, chunk_bits as u8);
                b.cast_to_field(spread)
            };

            b.lookup_spread(chunk_bits as u8, chunk_key, chunk_spread, flag_field);

            let value_shift = b.field_const(two_pow(offset));
            let shifted_key = b.mul(chunk_key, value_shift);
            reconstructed_key = b.add(reconstructed_key, shifted_key);

            if !is_last {
                let spread_shift = b.field_const(two_pow(offset * 2));
                let shifted_spread = b.mul(chunk_spread, spread_shift);
                reconstructed_spread = b.add(reconstructed_spread, shifted_spread);
            }

            offset += chunk_bits;
        }

        let key_field = if key_inner.is_field() {
            key
        } else {
            b.cast_to_field(key)
        };
        let key_diff = b.sub(reconstructed_key, key_field);
        b.constrain(key_diff, flag_field, zero);
    }
}

fn extract_low_chunk(
    b: &mut HLInstrBuilder<'_>,
    value: ValueId,
    value_bits: usize,
    offset: usize,
    chunk_bits: usize,
) -> ValueId {
    let shifted = if offset == 0 {
        value
    } else {
        let divisor = b.u_const(value_bits, two_pow_u128(offset));
        b.div(value, divisor)
    };
    let modulus = b.u_const(value_bits, two_pow_u128(chunk_bits));
    let chunk = b.modulo(shifted, modulus);
    b.cast_to(CastTarget::U(chunk_bits), chunk)
}

fn two_pow(exponent: usize) -> Field {
    Field::from(2).pow([exponent as u64])
}

fn two_pow_u128(exponent: usize) -> u128 {
    assert!(
        exponent < 128,
        "u128 constant shift out of range for exponent {exponent}"
    );
    1u128 << exponent
}
