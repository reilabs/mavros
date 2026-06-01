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
    },
    pass_manager::{Analysis, AnalysisId, AnalysisStore, Pass},
    ssa::{
        BlockId, ValueId,
        hlssa::{
            BinaryArithOpKind, CastTarget, Endianness, HLBlock, HLSSA, LookupTarget, OpCode, Radix,
            TypeExpr,
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
        let fids: Vec<_> = ssa.get_function_ids().collect();
        let mut sb = HLSSABuilder::new(ssa);
        for function_id in fids {
            let function_type_info = type_info.get_function(function_id);
            sb.modify_function(function_id, |fb| {
                let mut new_blocks = HashMap::<BlockId, HLBlock>::new();
                for (bid, mut block) in fb.function.take_blocks().into_iter() {
                    let mut new_instructions = Vec::new();
                    for instruction in block.take_instructions().into_iter() {
                        let b =
                            &mut HLInstrBuilder::new(fb.function, fb.ssa, &mut new_instructions);
                        self.process_instruction(b, function_type_info, instruction);
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
                        if r_taint {
                            panic!(
                                "witness field division should have been lowered by \
                                 instruction_lowering"
                            );
                        }
                        b.push(OpCode::BinaryArithOp {
                            kind,
                            result: res,
                            lhs: l,
                            rhs: r,
                        });
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
                result: _,
                cond,
                if_t: _,
                if_f: _,
            } => {
                let cond_taint = function_type_info.get_value_type(cond).is_witness_of();
                if cond_taint {
                    panic!("witness select should have been lowered by instruction_lowering");
                }
                b.push(instruction);
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
                result: _,
                value: _,
                from_bits: _,
                to_bits: _,
            } => {
                panic!("SExt should have been lowered by instruction_lowering");
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
                if function_type_info.get_value_type(value).is_witness_of()
                    || matches!(radix, Radix::Dyn(_))
                {
                    panic!("ToRadix should have been lowered by instruction_lowering");
                }
                b.push(OpCode::ToRadix {
                    result,
                    value,
                    radix,
                    endianness,
                    count,
                });
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
                if v_taint {
                    panic!("witness rangecheck should have been lowered by instruction_lowering");
                }
                b.push(OpCode::Rangecheck { value, max_bits });
            }
            OpCode::ReadGlobal {
                result: _,
                offset: _,
                result_type: _,
            } => {
                b.push(instruction);
            }
            OpCode::Lookup {
                target: LookupTarget::Rangecheck(bits),
                args,
                flag,
            } => {
                assert_eq!(args.len(), 1, "Rangecheck lookup must have exactly one key");
                self.gen_witness_rangecheck_bits(b, args[0], bits as usize, flag);
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
            OpCode::Store { ptr: _, value: _ } => {
                panic!("guarded store should have been lowered by instruction_lowering");
            }
            OpCode::SExt {
                result: _,
                value: _,
                from_bits: _,
                to_bits: _,
            } => {
                panic!("guarded SExt should have been lowered by instruction_lowering");
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
                            panic!(
                                "guarded witness field division should have been lowered by \
                                 instruction_lowering"
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
            OpCode::Rangecheck {
                value: _,
                max_bits: _,
            } => {
                panic!("guarded rangecheck should have been lowered by instruction_lowering");
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
        if bits == 8 {
            b.lookup_rngchk_8(value, flag);
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
