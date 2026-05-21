//! Lowers array operations whose index is witness-tainted before the main explicit-witness pass.
//!
//! This pass deliberately emits ordinary arithmetic/comparison/rangecheck operations and leaves
//! their constraint-level lowering to `ExplicitWitness`.

use ark_ff::Field as _;

use crate::compiler::{
    Field,
    analysis::types::{FunctionTypeInfo, TypeInfo},
    pass_manager::{Analysis, AnalysisId, AnalysisStore, Pass},
    ssa::{
        ValueId,
        hlssa::{
            CastTarget, HLSSA, OpCode, Type, TypeExpr,
            builder::{HLBlockEmitter, HLEmitter, HLFunctionBuilder, HLSSABuilder},
        },
    },
};

fn leaf_scalar_count(t: &Type) -> usize {
    match &t.expr {
        TypeExpr::Array(inner, n) => n * leaf_scalar_count(inner),
        TypeExpr::Field | TypeExpr::U(_) | TypeExpr::I(_) => 1,
        TypeExpr::WitnessOf(inner) => leaf_scalar_count(inner),
        TypeExpr::Slice(_) | TypeExpr::Ref(_) | TypeExpr::Tuple(_) | TypeExpr::Function => {
            panic!("leaf_scalar_count: unsupported type {}", t)
        }
    }
}

fn scalar_cast_target(ty: &Type, context: &str) -> CastTarget {
    match &ty.strip_all_witness().expr {
        TypeExpr::U(s) => CastTarget::U(*s),
        TypeExpr::I(s) => CastTarget::I(*s),
        TypeExpr::Field => CastTarget::Field,
        other => panic!("{context}: unsupported scalar type {:?}", other),
    }
}

fn array_len(ty: &Type, context: &str) -> usize {
    match &ty.strip_witness().expr {
        TypeExpr::Array(_, n) => *n,
        TypeExpr::Slice(_) => panic!("{context}: slice is not supported"),
        other => panic!("{context}: expected array type, got {:?}", other),
    }
}

fn uint_bits(ty: &Type, context: &str) -> usize {
    match ty.strip_witness().expr {
        TypeExpr::U(n) => n,
        _ => panic!("{context}: expected unsigned integer type, got {ty}"),
    }
}

pub struct LowerWitnessArrayOps {}

impl LowerWitnessArrayOps {
    pub fn new() -> Self {
        Self {}
    }

    fn do_run(&self, ssa: &mut HLSSA, type_info: &TypeInfo) {
        let fids: Vec<_> = ssa.get_function_ids().collect();
        let mut sb = HLSSABuilder::new(ssa);
        for function_id in fids {
            let function_type_info = type_info.get_function(function_id);
            sb.modify_function(function_id, |fb| {
                self.run_function(fb, function_type_info);
            });
        }
    }

    fn run_function(&self, fb: &mut HLFunctionBuilder<'_>, function_type_info: &FunctionTypeInfo) {
        let block_ids: Vec<_> = fb.function.get_blocks().map(|(bid, _)| *bid).collect();
        for block_id in block_ids {
            let (instructions, terminator) = {
                let mut block = fb.function.take_block(block_id);
                let instructions = block.take_instructions();
                let terminator = block.take_terminator();
                fb.function.put_block(block_id, block);
                (instructions, terminator)
            };

            let mut b = fb.block(block_id);
            for instruction in instructions {
                self.process_instruction(&mut b, function_type_info, instruction);
            }
            if let Some(terminator) = terminator {
                b.set_terminator(terminator);
            }
        }
    }

    fn process_instruction(
        &self,
        b: &mut HLBlockEmitter<'_>,
        function_type_info: &FunctionTypeInfo,
        instruction: OpCode,
    ) {
        if let OpCode::Guard { condition, inner } = instruction {
            self.process_array_op(b, function_type_info, Some(condition), *inner);
        } else {
            self.process_array_op(b, function_type_info, None, instruction);
        }
    }

    fn process_array_op(
        &self,
        b: &mut HLBlockEmitter<'_>,
        function_type_info: &FunctionTypeInfo,
        guard: Option<ValueId>,
        op: OpCode,
    ) {
        match op {
            OpCode::ArrayGet {
                result,
                array: arr,
                index: idx,
            } => {
                if self.has_witness_index(function_type_info, arr, idx) {
                    let flag = self.lookup_flag(b, function_type_info, guard);
                    self.gen_witness_array_get(
                        b,
                        function_type_info,
                        arr,
                        idx,
                        result,
                        flag,
                        guard,
                    );
                } else {
                    self.emit_guarded(
                        b,
                        guard,
                        OpCode::ArrayGet {
                            result,
                            array: arr,
                            index: idx,
                        },
                    );
                }
            }
            OpCode::ArraySet {
                result,
                array: arr,
                index: idx,
                value,
            } => {
                if guard.is_some() {
                    panic!(
                        "ArraySet inside Guard not supported yet: {:?}",
                        OpCode::ArraySet {
                            result,
                            array: arr,
                            index: idx,
                            value,
                        }
                    );
                }
                if self.has_witness_index(function_type_info, arr, idx) {
                    self.gen_witness_array_set(b, function_type_info, arr, idx, value, result);
                } else {
                    b.emit(OpCode::ArraySet {
                        result,
                        array: arr,
                        index: idx,
                        value,
                    });
                }
            }
            _ => self.emit_guarded(b, guard, op),
        }
    }

    fn has_witness_index(
        &self,
        function_type_info: &FunctionTypeInfo,
        arr: ValueId,
        idx: ValueId,
    ) -> bool {
        assert!(!function_type_info.get_value_type(arr).is_witness_of());
        function_type_info.get_value_type(idx).is_witness_of()
    }

    fn emit_guarded(&self, b: &mut HLBlockEmitter<'_>, guard: Option<ValueId>, op: OpCode) {
        if let Some(condition) = guard {
            b.emit(OpCode::Guard {
                condition,
                inner: Box::new(op),
            });
        } else {
            b.emit(op);
        }
    }

    fn lookup_flag(
        &self,
        b: &mut HLBlockEmitter<'_>,
        function_type_info: &FunctionTypeInfo,
        guard: Option<ValueId>,
    ) -> ValueId {
        match guard {
            Some(condition) => {
                b.ensure_field(condition, function_type_info.get_value_type(condition))
            }
            None => b.field_const(Field::ONE),
        }
    }

    /// Lower a witness-indexed ArrayGet into a hint + lookup constraint.
    /// `flag` is the lookup flag: `1` unconditionally, or the guard condition.
    fn gen_witness_array_get(
        &self,
        b: &mut HLBlockEmitter<'_>,
        function_type_info: &FunctionTypeInfo,
        arr: ValueId,
        idx: ValueId,
        result: ValueId,
        flag: ValueId,
        cond: Option<ValueId>,
    ) {
        let result_type_full = function_type_info.get_value_type(result).clone();
        let result_type = result_type_full.strip_all_witness();
        let arr_elem_type = function_type_info.get_value_type(arr).get_array_element();

        let pure_idx = b.value_of(idx);

        if matches!(&result_type.expr, TypeExpr::Array(..)) {
            let idx_field = b.cast_to_field(idx);
            let inner_hint = self.emit_array_get_hint(b, arr, pure_idx, cond);
            let outer_stride = leaf_scalar_count(&result_type);
            let stride_const = b.field_const(Field::from(outer_stride as u128));
            let base_key = b.mul(idx_field, stride_const);
            self.gen_witness_array_get_multidim(
                b,
                arr,
                base_key,
                inner_hint,
                &arr_elem_type,
                &result_type_full,
                Some(result),
                flag,
            );
            return;
        }

        let back_cast_target = scalar_cast_target(&result_type, "witnessed array read");

        let hint = self.emit_array_get_hint(b, arr, pure_idx, cond);
        let mut r_pure_val = hint;

        if arr_elem_type.is_witness_of() {
            r_pure_val = b.value_of(r_pure_val);
        }

        let idx_field = b.cast_to_field(idx);
        let r_wit_field = b.cast_to_field(r_pure_val);
        let r_wit = b.write_witness(r_wit_field);
        b.emit(OpCode::Cast {
            result,
            value: r_wit,
            target: back_cast_target,
        });
        b.lookup_arr(arr, idx_field, r_wit, flag);
    }

    fn emit_array_get_hint(
        &self,
        b: &mut HLBlockEmitter<'_>,
        arr: ValueId,
        pure_idx: ValueId,
        guard: Option<ValueId>,
    ) -> ValueId {
        let hint = b.fresh_value();
        self.emit_guarded(
            b,
            guard,
            OpCode::ArrayGet {
                result: hint,
                array: arr,
                index: pure_idx,
            },
        );
        hint
    }

    fn gen_witness_array_set(
        &self,
        b: &mut HLBlockEmitter<'_>,
        function_type_info: &FunctionTypeInfo,
        arr: ValueId,
        idx: ValueId,
        value: ValueId,
        result: ValueId,
    ) {
        let result_type = function_type_info.get_value_type(result);
        let length = array_len(result_type, "ArraySet result");
        let result_elem_type = result_type.get_array_element();
        let result_elem_back_cast = match &result_elem_type.strip_witness().expr {
            TypeExpr::Field => None,
            TypeExpr::U(s) => Some((CastTarget::U(*s), *s)),
            TypeExpr::I(s) => Some((CastTarget::I(*s), *s)),
            other => panic!(
                "ArraySet with witness idx: unsupported element type {:?}",
                other
            ),
        };

        let value_type = function_type_info.get_value_type(value);
        let value_field = b.ensure_field(value, value_type);
        let idx_bits = uint_bits(function_type_info.get_value_type(idx), "ArraySet index");

        let updated_array = b.build_array_loop(length, result_elem_type.clone(), |b, i| {
            let cmp_index = if idx_bits == 32 {
                i
            } else {
                b.cast_to(CastTarget::U(idx_bits), i)
            };
            let eq = b.eq(idx, cmp_index);
            let arr_i = b.array_get(arr, i);
            let arr_i_field = b.cast_to_field(arr_i);

            let new_i_field = b.select(eq, value_field, arr_i_field);
            if let Some((target, bits)) = result_elem_back_cast {
                b.rangecheck(new_i_field, bits);
                b.cast_to(target, new_i_field)
            } else {
                new_i_field
            }
        });
        b.emit(OpCode::Cast {
            result,
            value: updated_array,
            target: CastTarget::Nop,
        });
    }

    #[allow(clippy::too_many_arguments)]
    fn gen_witness_array_get_multidim(
        &self,
        b: &mut HLBlockEmitter<'_>,
        arr: ValueId,
        base_key: ValueId,
        hint: ValueId,
        arr_elem_type: &Type,
        target_type: &Type,
        result_override: Option<ValueId>,
        flag: ValueId,
    ) -> ValueId {
        let stripped = target_type.strip_all_witness();
        match &stripped.expr {
            TypeExpr::Array(inner_stripped, n) => {
                assert!(
                    !target_type.is_witness_of(),
                    "array containers should not be witness-typed here: {target_type}"
                );
                let inner_target = target_type.get_array_element();
                let inner_arr_type = arr_elem_type.get_array_element();
                let inner_leaves = leaf_scalar_count(inner_stripped.as_ref()) as u128;
                let built_array = b.build_array_loop(*n, inner_target.clone(), |b, i| {
                    let child_hint = b.array_get(hint, i);
                    let i_field = b.cast_to_field(i);
                    let stride_const = b.field_const(Field::from(inner_leaves));
                    let child_offset = b.mul(i_field, stride_const);
                    let child_base_key = b.add(base_key, child_offset);
                    self.gen_witness_array_get_multidim(
                        b,
                        arr,
                        child_base_key,
                        child_hint,
                        &inner_arr_type,
                        &inner_target,
                        None,
                        flag,
                    )
                });
                if let Some(result) = result_override {
                    b.emit(OpCode::Cast {
                        result,
                        value: built_array,
                        target: CastTarget::Nop,
                    });
                    result
                } else {
                    built_array
                }
            }
            TypeExpr::Slice(_) => {
                panic!("multidimensional witness array read: slice element types not supported")
            }
            TypeExpr::Tuple(_) | TypeExpr::Ref(_) | TypeExpr::Function => {
                panic!(
                    "multidimensional witness array read: unsupported element type {}",
                    target_type
                )
            }
            TypeExpr::Field | TypeExpr::U(_) | TypeExpr::I(_) => {
                let leaf_pure = if arr_elem_type.is_witness_of() {
                    b.value_of(hint)
                } else {
                    hint
                };
                let leaf_field = b.cast_to_field(leaf_pure);
                let leaf_wit = b.write_witness(leaf_field);
                b.lookup_arr(arr, base_key, leaf_wit, flag);
                let cast_target = match &stripped.expr {
                    TypeExpr::U(s) => CastTarget::U(*s),
                    TypeExpr::I(s) => CastTarget::I(*s),
                    TypeExpr::Field => CastTarget::Field,
                    _ => unreachable!(),
                };
                let id = result_override.unwrap_or_else(|| b.fresh_value());
                b.emit(OpCode::Cast {
                    result: id,
                    value: leaf_wit,
                    target: cast_target,
                });
                id
            }
            TypeExpr::WitnessOf(_) => {
                unreachable!("strip_all_witness should remove all WitnessOf wrappers")
            }
        }
    }
}

impl Pass for LowerWitnessArrayOps {
    fn name(&self) -> &'static str {
        "lower_witness_array_ops"
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
