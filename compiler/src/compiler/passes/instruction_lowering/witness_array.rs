//! Lowers array operations whose index is witness-tainted before witness spilling.
//!
//! This pass deliberately emits ordinary arithmetic/comparison/rangecheck operations and leaves
//! their constraint-level lowering to the later spilling passes.

use crate::compiler::util::ice_non_elided_tuple;
use ark_ff::Field as _;

use crate::compiler::{
    Field,
    analysis::types::FunctionTypeInfo,
    ssa::{
        ValueId,
        hlssa::{
            CastTarget, OpCode, Type, TypeExpr,
            builder::{HLBlockEmitter, HLEmitter},
        },
    },
};

use super::{InstructionLoweringRule, LoweringContext};

pub struct LowerWitnessArrayOps {}

impl InstructionLoweringRule for LowerWitnessArrayOps {
    fn lower_instruction(
        &self,
        b: &mut HLBlockEmitter<'_>,
        context: &LoweringContext<'_>,
        instruction: &OpCode,
    ) -> bool {
        let function_type_info = context.types();
        if let OpCode::Guard { condition, inner } = instruction {
            self.process_array_op(b, function_type_info, Some(*condition), inner.as_ref())
        } else {
            self.process_array_op(b, function_type_info, None, instruction)
        }
    }
}

impl LowerWitnessArrayOps {
    pub fn new() -> Self {
        Self {}
    }

    fn process_array_op(
        &self,
        b: &mut HLBlockEmitter<'_>,
        function_type_info: &FunctionTypeInfo,
        guard: Option<ValueId>,
        op: &OpCode,
    ) -> bool {
        match op {
            OpCode::ArrayGet {
                result,
                array: arr,
                index: idx,
            } => {
                if self.has_witness_index(function_type_info, *arr, *idx) {
                    let flag = self.lookup_flag(b, function_type_info, guard);
                    self.gen_witness_array_get(
                        b,
                        function_type_info,
                        *arr,
                        *idx,
                        *result,
                        flag,
                        guard,
                    );
                    true
                } else {
                    false
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
                            result: *result,
                            array: *arr,
                            index: *idx,
                            value: *value,
                        }
                    );
                }
                if self.has_witness_index(function_type_info, *arr, *idx) {
                    self.gen_witness_array_set(b, function_type_info, *arr, *idx, *value, *result);
                    true
                } else {
                    false
                }
            }
            _ => false,
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
        let hint = self.emit_array_get_hint(b, arr, pure_idx, cond);
        let idx_field = b.cast_to_field(idx);
        let stride = leaf_scalar_count(&result_type);
        let base_key = if stride == 1 {
            idx_field
        } else {
            let stride_const = b.field_const(Field::from(stride as u128));
            b.mul(idx_field, stride_const)
        };
        self.gen_witness_array_get_from_hint(
            b,
            arr,
            base_key,
            hint,
            &arr_elem_type,
            &result_type_full,
            Some(result),
            flag,
        );
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
            TypeExpr::U(s) => Some(CastTarget::U(*s)),
            TypeExpr::I(s) => Some(CastTarget::I(*s)),
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
            if let Some(target) = result_elem_back_cast {
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
    fn gen_witness_array_get_from_hint(
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
                    self.gen_witness_array_get_from_hint(
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
            TypeExpr::Tuple(_) => ice_non_elided_tuple(),
            TypeExpr::Ref(_) | TypeExpr::Function | TypeExpr::Blob(_) => {
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
                let cast_target = scalar_cast_target(&stripped, "witnessed array read");
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

fn leaf_scalar_count(t: &Type) -> usize {
    match &t.expr {
        TypeExpr::Array(inner, n) => n * leaf_scalar_count(inner),
        TypeExpr::Field | TypeExpr::U(_) | TypeExpr::I(_) => 1,
        TypeExpr::WitnessOf(inner) => leaf_scalar_count(inner),
        TypeExpr::Tuple(_) => ice_non_elided_tuple(),
        TypeExpr::Slice(_) | TypeExpr::Ref(_) | TypeExpr::Function | TypeExpr::Blob(_) => {
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
