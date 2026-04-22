use crate::compiler::{
    analysis::types::TypeInfo,
    block_builder::{HLBlockEmitter, HLEmitter},
    ir::r#type::{Type, TypeExpr},
    pass_manager::{Analysis, AnalysisId, AnalysisStore, Pass},
    ssa::{
        BinaryArithOpKind, BlockId, CastTarget, HLFunction, HLSSA, Instruction, OpCode, Terminator,
        ValueId,
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
///   Const, Cmp, Not, And, Or, Xor, Shr, Cast, ExtractTupleField, MkTuple,
///   MkSeq, Load, Select, Field arith, etc.
/// - **Desugar with OOB check** (can fail on out-of-bounds index):
///   ArrayGet — bounds-check index, assert !cond on OOB, then execute unconditionally.
/// - **Desugar with OOB check + value select** (RC-tracked allocation):
///   ArraySet — bounds-check index, select(cond, new_val, old_val), always execute ArraySet.
/// - **Desugar with overflow check** (pure inputs only, can fail):
///   Integer Add/Sub/Mul/Shl — widen, compute, check overflow, constrain !cond on fail.
/// - **Desugar with div-zero check** (pure inputs only, can fail):
///   Integer Div/Mod.
/// - **Desugar with range check** (pure inputs only, can fail):
///   Truncate.
/// - **Keep as Guard** (side-effectful or generates constraints):
///   Store, Call, WriteWitness, AssertEq, AssertR1C, Constrain, Rangecheck,
///   and failable ops with witness inputs (Truncate, SExt, integer arith).
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
            self.desugar_block(function, block_id, type_info);
        }
    }

    fn desugar_block(
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
                    self.desugar_guard(&mut emitter, condition, *inner, type_info);
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

    /// Desugar a single Guard instruction.
    fn desugar_guard(
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

            // -- Integer arith that can overflow: desugar only if all inputs pure --
            OpCode::BinaryArithOp {
                kind: kind @ (BinaryArithOpKind::Add | BinaryArithOpKind::Sub | BinaryArithOpKind::Mul | BinaryArithOpKind::Shl),
                result,
                lhs,
                rhs,
            } => {
                let lhs_type = type_info.get_value_type(lhs);
                match &lhs_type.strip_witness().expr {
                    TypeExpr::U(bits) if self.all_inputs_pure(&inner, type_info) => {
                        self.desugar_overflow_guard(
                            emitter, condition, kind, result, lhs, rhs, *bits, false,
                        );
                    }
                    TypeExpr::I(bits) if self.all_inputs_pure(&inner, type_info) => {
                        self.desugar_overflow_guard(
                            emitter, condition, kind, result, lhs, rhs, *bits, true,
                        );
                    }
                    // Field arith can't overflow — always unwrap
                    TypeExpr::Field => {
                        emitter.emit(OpCode::BinaryArithOp { kind, result, lhs, rhs });
                    }
                    // Witness inputs on integer arith: keep as Guard for ExplicitWitness
                    _ => {
                        emitter.emit(OpCode::Guard {
                            condition,
                            inner: Box::new(OpCode::BinaryArithOp { kind, result, lhs, rhs }),
                        });
                    }
                }
            }

            // -- Div/Mod: can fail on division by zero, desugar only if pure inputs --
            OpCode::BinaryArithOp {
                kind: kind @ (BinaryArithOpKind::Div | BinaryArithOpKind::Mod),
                result,
                lhs,
                rhs,
            } => {
                let lhs_type = type_info.get_value_type(lhs);
                match &lhs_type.strip_witness().expr {
                    TypeExpr::U(_) | TypeExpr::I(_) if self.all_inputs_pure(&inner, type_info) => {
                        self.desugar_divmod_guard(emitter, condition, kind, result, lhs, rhs, lhs_type);
                    }
                    // Field div can't fail — always unwrap
                    TypeExpr::Field => {
                        emitter.emit(OpCode::BinaryArithOp { kind, result, lhs, rhs });
                    }
                    // Witness inputs: keep as Guard
                    _ => {
                        emitter.emit(OpCode::Guard {
                            condition,
                            inner: Box::new(OpCode::BinaryArithOp { kind, result, lhs, rhs }),
                        });
                    }
                }
            }

            // -- Truncate: lossy narrowing, desugar only if pure inputs --
            OpCode::Truncate { result, value, to_bits, from_bits }
                if self.all_inputs_pure(&inner, type_info) =>
            {
                self.desugar_truncate_guard(emitter, condition, result, value, to_bits, from_bits);
            }

            // -- SExt: keep as Guard (ExplicitWitness handles it) --
            OpCode::SExt { .. } => {
                emitter.emit(OpCode::Guard {
                    condition,
                    inner: Box::new(inner),
                });
            }

            // Truncate with witness inputs: keep as Guard
            OpCode::Truncate { .. } => {
                emitter.emit(OpCode::Guard {
                    condition,
                    inner: Box::new(inner),
                });
            }

            // -- ArraySet: RC-tracked allocation. Desugar with OOB check + value select
            // so that ArraySet always runs exactly once (RC balanced).
            OpCode::ArraySet {
                result,
                array,
                index,
                value,
            } => {
                self.desugar_array_set_guard(emitter, condition, result, array, index, value, type_info);
            }

            // -- ArrayGet: can fail with OOB. Desugar with OOB check.
            OpCode::ArrayGet {
                result,
                array,
                index,
            } => {
                self.desugar_array_get_guard(emitter, condition, result, array, index, type_info);
            }

            // -- Everything else: pure computation, no constraints, can't fail → unwrap --
            // This covers: Const, Cmp, Not, And, Or, Xor, Shr, Cast, ExtractTupleField,
            // MkTuple, MkSeq, Load, Select, SlicePush, SliceLen,
            // ToBits, ToRadix, ValueOf, etc.
            other => {
                emitter.emit(other);
            }
        }
    }

    /// Desugar `Guard(cond, arith_op(lhs, rhs) -> result)` for integer overflow.
    ///
    /// Widens to double-width, performs the op, checks if it fits in the original width.
    /// On overflow: constrains !cond, produces a default 0.
    fn desugar_overflow_guard(
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
            Type { expr: TypeExpr::I(bits) }
        } else {
            Type { expr: TypeExpr::U(bits) }
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

    /// Desugar `Guard(cond, div/mod(lhs, rhs) -> result)` for division by zero.
    fn desugar_divmod_guard(
        &self,
        emitter: &mut HLBlockEmitter<'_>,
        condition: ValueId,
        kind: BinaryArithOpKind,
        original_result: ValueId,
        lhs: ValueId,
        rhs: ValueId,
        lhs_type: &Type,
    ) {
        let (bits, signed) = match &lhs_type.strip_witness().expr {
            TypeExpr::U(b) => (*b, false),
            TypeExpr::I(b) => (*b, true),
            _ => unreachable!(),
        };

        let zero = if signed {
            emitter.i_const(bits, 0)
        } else {
            emitter.u_const(bits, 0)
        };
        let is_zero = emitter.eq(rhs, zero);

        let result_type = lhs_type.strip_witness().clone();

        self.emit_guarded_branch(
            emitter,
            condition,
            is_zero,
            original_result,
            &result_type,
            |e| {
                let r = e.fresh_value();
                e.emit(OpCode::BinaryArithOp {
                    kind,
                    result: r,
                    lhs,
                    rhs,
                });
                r
            },
            signed,
            bits,
        );
    }

    /// Desugar `Guard(cond, Truncate(value, to_bits, from_bits) -> result)`.
    fn desugar_truncate_guard(
        &self,
        emitter: &mut HLBlockEmitter<'_>,
        condition: ValueId,
        original_result: ValueId,
        value: ValueId,
        to_bits: usize,
        from_bits: usize,
    ) {
        let max_val = emitter.u_const(from_bits, 1u128 << to_bits);
        let fits = emitter.lt(value, max_val);
        let overflow = emitter.not(fits);

        let result_type = Type { expr: TypeExpr::U(to_bits) };

        self.emit_guarded_branch(
            emitter,
            condition,
            overflow,
            original_result,
            &result_type,
            |e| e.truncate(value, to_bits, from_bits),
            false,
            to_bits,
        );
    }

    /// Desugar `Guard(cond, ArraySet(array, idx, val) -> result)`.
    ///
    /// Pattern:
    ///   oob = idx >= len(array)
    ///   if oob { assert !cond }
    ///   old_val = array_get(array, idx)
    ///   selected_val = select(cond, val, old_val)
    ///   result = array_set(array, idx, selected_val)
    ///
    /// ArraySet always runs exactly once → RC balanced. If OOB, the assert fires
    /// first; any subsequent crash from the OOB access is irrelevant.
    fn desugar_array_set_guard(
        &self,
        emitter: &mut HLBlockEmitter<'_>,
        condition: ValueId,
        original_result: ValueId,
        array: ValueId,
        index: ValueId,
        value: ValueId,
        type_info: &crate::compiler::analysis::types::FunctionTypeInfo,
    ) {
        self.emit_oob_check(emitter, condition, array, index, type_info);

        // old_val = array_get(array, idx)
        let old_val = emitter.array_get(array, index);

        // selected_val = select(cond, val, old_val)
        let selected_val = emitter.select(condition, value, old_val);

        // result = array_set(array, idx, selected_val)
        emitter.emit(OpCode::ArraySet {
            result: original_result,
            array,
            index,
            value: selected_val,
        });
    }

    /// Desugar `Guard(cond, ArrayGet(array, idx) -> result)`.
    ///
    /// Pattern:
    ///   oob = idx >= len(array)
    ///   if oob { assert !cond }
    ///   result = array_get(array, idx)
    fn desugar_array_get_guard(
        &self,
        emitter: &mut HLBlockEmitter<'_>,
        condition: ValueId,
        original_result: ValueId,
        array: ValueId,
        index: ValueId,
        type_info: &crate::compiler::analysis::types::FunctionTypeInfo,
    ) {
        self.emit_oob_check(emitter, condition, array, index, type_info);

        // Execute the get unconditionally (if OOB, assert already failed)
        emitter.emit(OpCode::ArrayGet {
            result: original_result,
            array,
            index,
        });
    }

    /// Emit an OOB bounds check: if idx >= len(array), assert !cond.
    fn emit_oob_check(
        &self,
        emitter: &mut HLBlockEmitter<'_>,
        condition: ValueId,
        array: ValueId,
        index: ValueId,
        type_info: &crate::compiler::analysis::types::FunctionTypeInfo,
    ) {
        let array_type = type_info.get_value_type(array);
        let arr_len = match &array_type.strip_witness().expr {
            TypeExpr::Array(_, len) => *len,
            other => panic!("LowerPureGuards: array op on non-array type: {:?}", other),
        };

        let len_val = emitter.u_const(32, arr_len as u128);
        let idx_type = type_info.get_value_type(index);
        let idx_as_u32 = if matches!(idx_type.strip_witness().expr, TypeExpr::U(32)) {
            index
        } else {
            emitter.cast_to(CastTarget::U(32), index)
        };
        let in_bounds = emitter.lt(idx_as_u32, len_val);
        let oob = emitter.not(in_bounds);

        let (fail_block, _) = emitter.add_block();
        let (continue_block, _) = emitter.add_block();

        emitter.seal_and_switch(Terminator::JmpIf(oob, fail_block, continue_block), fail_block);

        let not_cond = emitter.not(condition);
        let one = emitter.u_const(1, 1);
        emitter.emit(OpCode::AssertEq { lhs: not_cond, rhs: one });
        emitter.seal_and_switch(Terminator::Jmp(continue_block, vec![]), continue_block);
    }

    /// Common pattern: branch on a failure condition, constraining !cond in the fail
    /// block and producing a default value, or executing the ok path.
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
        let (fail_block, _) = emitter.add_block();
        let (ok_block, _) = emitter.add_block();
        let (merge_block, _) = emitter.add_block();

        emitter
            .function
            .get_block_mut(merge_block)
            .push_parameter(original_result, result_type.clone());

        emitter.seal_and_switch(Terminator::JmpIf(failure, fail_block, ok_block), fail_block);

        // Fail block: constrain !cond, produce default
        let not_cond = emitter.not(condition);
        let one = emitter.u_const(1, 1);
        emitter.emit(OpCode::AssertEq { lhs: not_cond, rhs: one });
        let default_val = if signed {
            emitter.i_const(bits, 0)
        } else {
            emitter.u_const(bits, 0)
        };
        emitter.seal_and_switch(
            Terminator::Jmp(merge_block, vec![default_val]),
            ok_block,
        );

        // Ok block: execute the operation
        let ok_val = ok_path(emitter);
        emitter.seal_and_switch(Terminator::Jmp(merge_block, vec![ok_val]), merge_block);
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
