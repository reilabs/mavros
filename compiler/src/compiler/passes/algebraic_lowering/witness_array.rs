//! Lowers array operations whose index is witness-tainted before the main explicit-witness pass.
//!
//! This pass deliberately emits ordinary arithmetic/comparison/rangecheck operations and leaves
//! their constraint-level lowering to `ExplicitWitness`.

use ark_ff::Field as _;

use crate::compiler::{
    Field,
    analysis::types::FunctionTypeInfo,
    ssa::{
        ValueId,
        hlssa::{
            CastTarget, OpCode, SequenceTargetType, Type, TypeExpr,
            builder::{HLBlockEmitter, HLEmitter},
        },
    },
};

use super::{AlgebraicLoweringRule, LoweringContext};

pub struct LowerWitnessArrayOps {}

impl AlgebraicLoweringRule for LowerWitnessArrayOps {
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
                    self.gen_witness_array_get(b, function_type_info, *arr, *idx, *result, flag);
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
    ) {
        let result_type_full = function_type_info.get_value_type(result).clone();
        let result_type = result_type_full.strip_all_witness();
        let arr_type = function_type_info.get_value_type(arr);
        let arr_elem_type = function_type_info.get_value_type(arr).get_array_element();
        let lookup_arr = self.lookup_array(b, arr, arr_type);

        let pure_idx = b.value_of(idx);
        let hint = self.emit_array_get_hint(b, function_type_info, arr, idx, pure_idx);
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
            lookup_arr,
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
        function_type_info: &FunctionTypeInfo,
        arr: ValueId,
        idx: ValueId,
        pure_idx: ValueId,
    ) -> ValueId {
        let length = array_len(function_type_info.get_value_type(arr), "ArrayGet input");
        let idx_bits = uint_bits(function_type_info.get_value_type(idx), "ArrayGet index");
        let len = b.u_const(idx_bits, length as u128);
        let in_bounds = b.lt(pure_idx, len);
        let oob = b.not(in_bounds);
        let zero = b.u_const(idx_bits, 0);
        let safe_idx = b.select(oob, zero, pure_idx);
        b.array_get(arr, safe_idx)
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

    fn lookup_array(&self, b: &mut HLBlockEmitter<'_>, arr: ValueId, arr_type: &Type) -> ValueId {
        if leaf_scalar_count(&arr_type.get_array_element()) == 1 {
            return arr;
        }

        let mut leaves = Vec::new();
        self.collect_array_leaves(b, arr, arr_type, &mut leaves);
        let elem_type = scalar_leaf_type(arr_type);
        b.mk_seq(
            leaves,
            SequenceTargetType::Array(leaf_scalar_count(arr_type)),
            elem_type,
        )
    }

    fn collect_array_leaves(
        &self,
        b: &mut HLBlockEmitter<'_>,
        value: ValueId,
        value_type: &Type,
        leaves: &mut Vec<ValueId>,
    ) {
        match &value_type.strip_witness().expr {
            TypeExpr::Array(elem, len) => {
                for i in 0..*len {
                    let idx = b.u_const(32, i as u128);
                    let child = b.array_get(value, idx);
                    self.collect_array_leaves(b, child, elem, leaves);
                }
            }
            TypeExpr::Field | TypeExpr::U(_) | TypeExpr::I(_) | TypeExpr::WitnessOf(_) => {
                leaves.push(value);
            }
            TypeExpr::Slice(_) | TypeExpr::Ref(_) | TypeExpr::Tuple(_) | TypeExpr::Function => {
                panic!("lookup array flattening: unsupported type {}", value_type)
            }
        }
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
        TypeExpr::Slice(_) | TypeExpr::Ref(_) | TypeExpr::Tuple(_) | TypeExpr::Function => {
            panic!("leaf_scalar_count: unsupported type {}", t)
        }
    }
}

fn scalar_leaf_type(t: &Type) -> Type {
    match &t.expr {
        TypeExpr::Array(inner, _) => scalar_leaf_type(inner),
        TypeExpr::Field | TypeExpr::U(_) | TypeExpr::I(_) | TypeExpr::WitnessOf(_) => t.clone(),
        TypeExpr::Slice(_) | TypeExpr::Ref(_) | TypeExpr::Tuple(_) | TypeExpr::Function => {
            panic!("scalar_leaf_type: unsupported type {}", t)
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
