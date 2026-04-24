use crate::compiler::{
    analysis::types::TypeInfo,
    block_builder::{HLBlockEmitter, HLEmitter},
    ir::r#type::{Type, TypeExpr},
    pass_manager::{Analysis, AnalysisId, AnalysisStore, Pass},
    ssa::{
        BinaryArithOpKind, BlockId, CastTarget, HLFunction, HLSSA, Instruction, OpCode, ValueId,
    },
};

/// Lowers pure Guard instructions into plain control flow where possible.
///
/// After UntaintControlFlow, Guards wrap operations in witness-conditional
/// blocks. Many of these Guards are unnecessary because the inner operation
/// is a pure computation that doesn't generate constraints or have side effects.
///
/// Classification:
/// - **Always unwrap** (no constraints, no side effects, can't fail):
///   Const, Cmp, Not, And, Or, Xor, Shr, Cast, Truncate, ExtractTupleField,
///   MkTuple, MkSeq, Load, Select, Field Add/Sub/Mul, etc.
/// - **Lower with OOB check** (can fail on out-of-bounds index):
///   ArrayGet — bounds-check index, assert !cond on OOB, then execute unconditionally.
/// - **Lower with OOB check + value select** (RC-tracked allocation):
///   ArraySet — bounds-check index, select(cond, new_val, old_val), always execute ArraySet.
/// - **Lower with overflow check** (pure inputs only, can fail):
///   Integer Add/Sub/Mul/Shl — widen, compute, check overflow, constrain !cond on fail.
/// - **Lower with div-zero check** (pure inputs only, can fail):
///   Div/Mod (integer and Field).
/// - **Keep as Guard** (side-effectful or generates constraints):
///   Store, Call, WriteWitness, AssertEq, AssertR1C, Constrain, Rangecheck,
///   and failable ops with witness inputs (SExt, integer arith).
pub struct LowerPureGuards {}

impl Pass for LowerPureGuards {
    fn name(&self) -> &'static str {
        "lower_pure_guards"
    }

    fn needs(&self) -> Vec<AnalysisId> {
        vec![TypeInfo::id()]
    }

    fn run(&self, ssa: &mut HLSSA, store: &AnalysisStore) {
        self.do_run(ssa, store.get::<TypeInfo>());
    }

    fn preserves(&self) -> Vec<AnalysisId> {
        vec![]
    }
}

impl LowerPureGuards {
    pub fn new() -> Self {
        Self {}
    }

    fn do_run(&self, ssa: &mut HLSSA, type_info: &TypeInfo) {
        let function_ids: Vec<_> = ssa.get_function_ids().collect();
        for function_id in &function_ids {
            let mut function = ssa.take_function(*function_id);
            let func_types = type_info.get_function(*function_id);
            self.run_function(&mut function, func_types);
            ssa.put_function(*function_id, function);
        }
    }

    fn run_function(
        &self,
        function: &mut HLFunction,
        type_info: &crate::compiler::analysis::types::FunctionTypeInfo,
    ) {
        let block_ids: Vec<_> = function.get_blocks().map(|(bid, _)| *bid).collect();
        for block_id in block_ids {
            self.lower_block(function, block_id, type_info);
        }
    }

    fn lower_block(
        &self,
        function: &mut HLFunction,
        block_id: BlockId,
        type_info: &crate::compiler::analysis::types::FunctionTypeInfo,
    ) {
        let (instructions, terminator) = {
            let mut block = function.take_block(block_id);
            let instructions = block.take_instructions();
            let terminator = block.take_terminator();
            function.put_block(block_id, block);
            (instructions, terminator)
        };

        let mut emitter = HLBlockEmitter::new(function, block_id);

        for instruction in instructions {
            match instruction {
                OpCode::Guard { condition, inner } => {
                    self.lower_guard(&mut emitter, condition, *inner, type_info);
                }
                other => {
                    emitter.emit(other);
                }
            }
        }

        if let Some(term) = terminator {
            emitter.set_terminator(term);
        }
    }

    /// Check whether all inputs to an opcode are pure (not WitnessOf-typed).
    fn all_inputs_pure(
        &self,
        op: &OpCode,
        type_info: &crate::compiler::analysis::types::FunctionTypeInfo,
    ) -> bool {
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
        type_info: &crate::compiler::analysis::types::FunctionTypeInfo,
    ) {
        match inner {
            // -- Side-effectful / constraint-generating: always keep as Guard --
            OpCode::Store { .. }
            | OpCode::Call { .. }
            | OpCode::WriteWitness { .. }
            | OpCode::AssertEq { .. }
            | OpCode::AssertR1C { .. }
            | OpCode::Constrain { .. }
            | OpCode::Rangecheck { .. }
            | OpCode::MemOp { .. }
            | OpCode::Lookup { .. }
            | OpCode::DLookup { .. } => {
                emitter.emit(OpCode::Guard {
                    condition,
                    inner: Box::new(inner),
                });
            }

            // -- Integer arith that can overflow: lower only if all inputs pure --
            OpCode::BinaryArithOp {
                kind:
                    kind @ (BinaryArithOpKind::Add
                    | BinaryArithOpKind::Sub
                    | BinaryArithOpKind::Mul
                    | BinaryArithOpKind::Shl),
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
                    }
                    TypeExpr::I(bits) if self.all_inputs_pure(&inner, type_info) => {
                        self.lower_overflow_guard(
                            emitter, condition, kind, result, lhs, rhs, *bits, true,
                        );
                    }
                    // Field arith can't overflow — always unwrap
                    TypeExpr::Field => {
                        emitter.emit(OpCode::BinaryArithOp {
                            kind,
                            result,
                            lhs,
                            rhs,
                        });
                    }
                    // Witness inputs on integer arith: keep as Guard for ExplicitWitness
                    _ => {
                        emitter.emit(OpCode::Guard {
                            condition,
                            inner: Box::new(OpCode::BinaryArithOp {
                                kind,
                                result,
                                lhs,
                                rhs,
                            }),
                        });
                    }
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
                    }
                    // Witness inputs: keep as Guard
                    _ => {
                        emitter.emit(OpCode::Guard {
                            condition,
                            inner: Box::new(OpCode::BinaryArithOp {
                                kind,
                                result,
                                lhs,
                                rhs,
                            }),
                        });
                    }
                }
            }

            // -- Truncate: pure bit-narrowing, can't fail → always unwrap --
            OpCode::Truncate { .. } => {
                emitter.emit(inner);
            }

            // -- SExt: keep as Guard (ExplicitWitness handles it) --
            OpCode::SExt { .. } => {
                emitter.emit(OpCode::Guard {
                    condition,
                    inner: Box::new(inner),
                });
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
            }

            // -- ArrayGet: lower with OOB check if index is pure.
            OpCode::ArrayGet {
                result,
                array,
                index,
            } if !type_info.get_value_type(index).is_witness_of() => {
                self.lower_array_get_guard(emitter, condition, result, array, index, type_info);
            }

            // ArrayGet/ArraySet with witness index: keep as Guard
            OpCode::ArraySet { .. } | OpCode::ArrayGet { .. } => {
                emitter.emit(OpCode::Guard {
                    condition,
                    inner: Box::new(inner),
                });
            }

            // -- Pure computation, no constraints, can't fail → unwrap --
            OpCode::Const { .. }
            | OpCode::Cmp { .. }
            | OpCode::Not { .. }
            | OpCode::BinaryArithOp {
                kind:
                    BinaryArithOpKind::And
                    | BinaryArithOpKind::Or
                    | BinaryArithOpKind::Xor
                    | BinaryArithOpKind::Shr,
                ..
            }
            | OpCode::Cast { .. }
            | OpCode::MkSeq { .. }
            | OpCode::MkTuple { .. }
            | OpCode::TupleProj { .. }
            | OpCode::Load { .. }
            | OpCode::Select { .. }
            | OpCode::SlicePush { .. }
            | OpCode::SliceLen { .. }
            | OpCode::ToBits { .. }
            | OpCode::ToRadix { .. }
            | OpCode::ValueOf { .. }
            | OpCode::MulConst { .. }
            | OpCode::Alloc { .. }
            | OpCode::ReadGlobal { .. }
            | OpCode::InitGlobal { .. }
            | OpCode::DropGlobal { .. }
            | OpCode::Spread { .. }
            | OpCode::Unspread { .. }
            | OpCode::FreshWitness { .. }
            | OpCode::NextDCoeff { .. }
            | OpCode::BumpD { .. }
            | OpCode::Todo { .. } => {
                emitter.emit(inner);
            }

            // Guard-within-Guard should not happen
            OpCode::Guard { .. } => {
                panic!("LowerPureGuards: nested Guard not expected");
            }
        }
    }

    /// Lower `Guard(cond, arith_op(lhs, rhs) -> result)` for integer overflow.
    ///
    /// Widens to double-width, performs the op, checks if it fits in the original width.
    /// On overflow: constrains !cond, produces a default 0.
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
        let wide_bits = wider_bits(bits);

        // Widen operands
        let wide_target = if signed {
            CastTarget::I(wide_bits)
        } else {
            CastTarget::U(wide_bits)
        };
        let lhs_wide = emitter.cast_to(wide_target, lhs);
        let rhs_wide = emitter.cast_to(wide_target, rhs);

        // Perform the op in wider type
        let wide_result = emitter.fresh_value();
        emitter.emit(OpCode::BinaryArithOp {
            kind,
            result: wide_result,
            lhs: lhs_wide,
            rhs: rhs_wide,
        });

        // Check overflow: does the result fit in the original type?
        let overflow = if signed {
            // Signed: check result < -(2^(bits-1)) || result >= 2^(bits-1)
            let min_val = emitter.i_const(wide_bits, (-(1i128 << (bits - 1))) as u128);
            let max_val = emitter.i_const(wide_bits, 1u128 << (bits - 1));
            let too_low = emitter.lt(wide_result, min_val);
            let too_high = emitter.cmp(max_val, wide_result, crate::compiler::ssa::CmpKind::Lt);
            emitter.or(too_low, too_high)
        } else {
            // Unsigned: check result >= 2^bits
            let max_val = emitter.u_const(wide_bits, 1u128 << bits);
            let fits = emitter.lt(wide_result, max_val);
            emitter.not(fits)
        };

        let result_type = if signed {
            Type {
                expr: TypeExpr::I(bits),
            }
        } else {
            Type {
                expr: TypeExpr::U(bits),
            }
        };

        self.emit_guarded_branch(
            emitter,
            condition,
            overflow,
            original_result,
            &result_type,
            |e| {
                let narrow_target = if signed {
                    CastTarget::I(bits)
                } else {
                    CastTarget::U(bits)
                };
                e.cast_to(narrow_target, wide_result)
            },
            signed,
            bits,
        );
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

        emitter.build_if_else_into(
            is_zero,
            vec![(original_result, lhs_type.clone())],
            // Divisor is zero: assert condition is false, produce default
            |e| {
                let zero_u1 = e.u_const(1, 0);
                e.emit(OpCode::AssertEq {
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
    ///   assert !(oob && cond)
    ///   if oob { result = array } else { result = array_set(array, idx, value) }
    fn lower_array_set_guard(
        &self,
        emitter: &mut HLBlockEmitter<'_>,
        condition: ValueId,
        original_result: ValueId,
        array: ValueId,
        index: ValueId,
        value: ValueId,
        type_info: &crate::compiler::analysis::types::FunctionTypeInfo,
    ) {
        let array_type = type_info.get_value_type(array).strip_witness().clone();
        let oob = self.emit_oob_cond(emitter, array, index, type_info);

        emitter.build_if_else_into(
            oob,
            vec![(original_result, array_type)],
            // OOB: assert condition is false, pass through original array
            |e| {
                let zero = e.u_const(1, 0);
                e.emit(OpCode::AssertEq {
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
    ///   assert !(oob && cond)
    ///   if oob { result = default } else { result = array_get(array, idx) }
    fn lower_array_get_guard(
        &self,
        emitter: &mut HLBlockEmitter<'_>,
        condition: ValueId,
        original_result: ValueId,
        array: ValueId,
        index: ValueId,
        type_info: &crate::compiler::analysis::types::FunctionTypeInfo,
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
                e.emit(OpCode::AssertEq {
                    lhs: condition,
                    rhs: zero,
                });
                vec![self.default_scalar(e, &elem_type)]
            },
            // In-bounds: do the get
            |e| vec![e.array_get(array, index)],
        );
    }

    /// Compute the OOB condition: idx >= len(seq). Returns a bool ValueId.
    /// Works for both arrays (static length) and slices (runtime SliceLen).
    fn emit_oob_cond(
        &self,
        emitter: &mut HLBlockEmitter<'_>,
        seq: ValueId,
        index: ValueId,
        type_info: &crate::compiler::analysis::types::FunctionTypeInfo,
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

    /// Produce a default zero value for a scalar type.
    fn default_scalar(&self, emitter: &mut HLBlockEmitter<'_>, ty: &Type) -> ValueId {
        match &ty.expr {
            TypeExpr::Field => emitter.field_const(ark_bn254::Fr::from(0u64)),
            TypeExpr::U(bits) => emitter.u_const(*bits, 0),
            TypeExpr::I(bits) => emitter.i_const(*bits, 0),
            TypeExpr::WitnessOf(inner) => {
                let inner_val = self.default_scalar(emitter, inner);
                emitter.cast_to(CastTarget::WitnessOf, inner_val)
            }
            other => panic!(
                "LowerPureGuards: cannot produce default for non-scalar element type: {:?}",
                other
            ),
        }
    }

    /// Common pattern: branch on a failure condition, asserting condition==false
    /// in the fail block and producing a default value, or executing the ok path.
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
            |e| {
                let zero = e.u_const(1, 0);
                e.emit(OpCode::AssertEq {
                    lhs: condition,
                    rhs: zero,
                });
                vec![if signed {
                    e.i_const(bits, 0)
                } else {
                    e.u_const(bits, 0)
                }]
            },
            // Ok: compute the result
            |e| vec![ok_path(e)],
        );
    }
}

/// Pick the next wider integer size that can hold overflow results.
fn wider_bits(bits: usize) -> usize {
    match bits {
        1..=8 => 16,
        9..=16 => 32,
        17..=32 => 64,
        33..=64 => 128,
        _ => bits * 2,
    }
}
