//! Lowers array operations whose index is witness-tainted before witness spilling.
//!
//! This pass deliberately emits ordinary arithmetic/comparison/rangecheck operations and leaves
//! their constraint-level lowering to the later spilling passes.

use crate::compiler::util::ice_non_elided_tuple;

use crate::compiler::{
    analysis::types::FunctionTypeInfo,
    ssa::{
        ValueId,
        hlssa::{
            CastTarget, CmpKind, OpCode, SequenceTargetType, Type, TypeExpr,
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
                    if function_type_info.get_value_type(*arr).is_slice() {
                        self.gen_witness_slice_get(
                            b,
                            function_type_info,
                            *arr,
                            *idx,
                            *result,
                            guard,
                        );
                    } else {
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
                    }
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
                if self.has_witness_index(function_type_info, *arr, *idx) {
                    self.gen_witness_array_set(
                        b,
                        function_type_info,
                        *arr,
                        *idx,
                        *value,
                        *result,
                        guard,
                    );
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
            None => b.field_const(b.field().one()),
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
            let stride_const = b.field_const(b.field().constant(stride as u128));
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

    fn gen_witness_slice_get(
        &self,
        b: &mut HLBlockEmitter<'_>,
        function_type_info: &FunctionTypeInfo,
        arr: ValueId,
        idx: ValueId,
        result: ValueId,
        guard: Option<ValueId>,
    ) {
        let elem_type = function_type_info.get_value_type(arr).get_array_element();
        let acc_type = push_witness_of_to_leaves_for_slice_children(&elem_type);
        let idx_bits = uint_bits(
            function_type_info.get_value_type(idx),
            "witness slice get index",
        );

        let slice_len = b.slice_len(arr);
        self.emit_guarded(
            b,
            guard,
            OpCode::AssertCmp {
                kind: CmpKind::Lt,
                lhs: idx,
                rhs: slice_len,
            },
        );
        let zero = b.u_const(32, 0);
        let one = b.u_const(32, 1);
        let init = b.default_value(&acc_type);
        let results = b.build_loop(
            vec![(zero, Type::u(32)), (init, acc_type)],
            |hb, p| hb.lt(p[0], slice_len),
            |bb, p| {
                let i = p[0];
                let acc = p[1];
                let cmp_index = if idx_bits == 32 {
                    i
                } else {
                    bb.cast_to(CastTarget::U(idx_bits), i)
                };
                let hit = bb.eq(idx, cmp_index);
                let arr_i = bb.array_get(arr, i);
                let acc2 = merge_select_for_slice_leaves(bb, hit, arr_i, acc, &elem_type);
                let i2 = bb.add(i, one);
                vec![i2, acc2]
            },
        );
        b.emit(OpCode::Cast {
            result,
            value: results[1],
            target: CastTarget::Nop,
        });
    }

    fn gen_witness_array_set(
        &self,
        b: &mut HLBlockEmitter<'_>,
        function_type_info: &FunctionTypeInfo,
        arr: ValueId,
        idx: ValueId,
        value: ValueId,
        result: ValueId,
        guard: Option<ValueId>,
    ) {
        let result_type = function_type_info.get_value_type(result).clone();
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

        let elem_at = |b: &mut HLBlockEmitter<'_>, i: ValueId| -> ValueId {
            let cmp_index = if idx_bits == 32 {
                i
            } else {
                b.cast_to(CastTarget::U(idx_bits), i)
            };
            let hit = b.eq(idx, cmp_index);
            let write = match guard {
                Some(g) => b.and(g, hit),
                None => hit,
            };
            let arr_i = b.array_get(arr, i);
            let arr_i_field = b.cast_to_field(arr_i);
            let new_i_field = b.select(write, value_field, arr_i_field);
            match result_elem_back_cast {
                Some(target) => b.cast_to(target, new_i_field),
                None => new_i_field,
            }
        };

        let updated = if result_type.is_slice() {
            let slice_len = b.slice_len(arr); // Will fold to const
            let empty = b.mk_seq(vec![], SequenceTargetType::Slice, result_elem_type.clone());
            b.build_slice_extend_loop(slice_len, (empty, result_type.clone()), |b, i| {
                elem_at(b, i)
            })
        } else {
            let length = array_len(&result_type, "ArraySet result");
            b.build_array_loop(length, result_elem_type.clone(), |b, i| elem_at(b, i))
        };

        b.emit(OpCode::Cast {
            result,
            value: updated,
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
                    let stride_const = b.field_const(b.field().constant(inner_leaves));
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
            TypeExpr::Slice { .. } => {
                panic!("multidimensional witness array read: slice element types not supported")
            }
            TypeExpr::Tuple(_) => ice_non_elided_tuple(),
            TypeExpr::Ref(_) | TypeExpr::Function | TypeExpr::Blob(..) => {
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

pub(super) fn push_witness_of_to_leaves_for_slice_children(ty: &Type) -> Type {
    match &ty.expr {
        TypeExpr::WitnessOf(_) => ty.clone(),
        TypeExpr::Field | TypeExpr::U(_) | TypeExpr::I(_) => Type::witness_of(ty.clone()),
        TypeExpr::Array(inner, n) => {
            push_witness_of_to_leaves_for_slice_children(inner).array_of(*n)
        }
        other => panic!(
            "witness-indexed slice get: unsupported element type {:?}",
            other
        ),
    }
}

pub(super) fn merge_select_for_slice_leaves(
    b: &mut HLBlockEmitter<'_>,
    hit: ValueId,
    new_v: ValueId,
    acc: ValueId,
    ty: &Type,
) -> ValueId {
    match &ty.expr {
        TypeExpr::Field | TypeExpr::U(_) | TypeExpr::I(_) | TypeExpr::WitnessOf(_) => {
            b.select(hit, new_v, acc)
        }
        TypeExpr::Array(inner, n) => {
            let inner = (**inner).clone();
            b.build_array_loop(
                *n,
                push_witness_of_to_leaves_for_slice_children(&inner),
                |b, j| {
                    let nj = b.array_get(new_v, j);
                    let aj = b.array_get(acc, j);
                    merge_select_for_slice_leaves(b, hit, nj, aj, &inner)
                },
            )
        }
        other => panic!(
            "witness-indexed slice get: unsupported element type {:?}",
            other
        ),
    }
}

fn leaf_scalar_count(t: &Type) -> usize {
    match &t.expr {
        TypeExpr::Array(inner, n) => n * leaf_scalar_count(inner),
        TypeExpr::Field | TypeExpr::U(_) | TypeExpr::I(_) => 1,
        TypeExpr::WitnessOf(inner) => leaf_scalar_count(inner),
        TypeExpr::Tuple(_) => ice_non_elided_tuple(),
        TypeExpr::Slice { .. } | TypeExpr::Ref(_) | TypeExpr::Function | TypeExpr::Blob(..) => {
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
        TypeExpr::Slice { .. } => panic!("{context}: slice is not supported"),
        other => panic!("{context}: expected array type, got {:?}", other),
    }
}

pub(super) fn uint_bits(ty: &Type, context: &str) -> usize {
    match ty.strip_witness().expr {
        TypeExpr::U(n) => n,
        _ => panic!("{context}: expected unsigned integer type, got {ty}"),
    }
}
