//! Lowers array operations whose index is witness-tainted before witness spilling.
//!
//! This pass deliberately emits ordinary arithmetic/comparison/rangecheck operations and leaves
//! their constraint-level lowering to the later spilling passes.

use crate::compiler::{
    analysis::types::{FunctionTypeInfo, push_witness_of_to_leaves},
    passes::instruction_lowering::{InstructionLoweringRule, LoweringContext},
    ssa::{
        ValueId,
        hlssa::{
            CastTarget, CmpKind, OpCode, SequenceTargetType, Type, TypeExpr,
            builder::{HLBlockEmitter, HLEmitter},
        },
    },
    util::ice_non_elided_tuple,
};

pub struct LowerWitnessArrayOps {}

impl InstructionLoweringRule for LowerWitnessArrayOps {
    fn lower_instruction(
        &self,
        b: &mut HLBlockEmitter<'_>,
        context: &LoweringContext<'_>,
        instruction: &OpCode,
    ) -> bool {
        let function_type_info = context.types();
        let (guard, op) = HLBlockEmitter::unwrap_guard(instruction);
        self.process_array_op(b, function_type_info, guard, op)
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
                    // A slice cannot take the lookup path: the LLSSA lowering sizes the table from
                    // the type, which for a slice carries no length, so the lookup proves fine and
                    // then dies in the WASM lane. It scans instead, at +n rows per read — see
                    // [`Self::gen_witness_slice_get`] before "simplifying" this fork away.
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
        b.emit_guarded(
            guard,
            OpCode::ArrayGet {
                result: hint,
                array: arr,
                index: pure_idx,
            },
        );
        hint
    }

    /// Lower a witness-indexed `ArrayGet` on a **slice** into a bounds assert plus a linear scan:
    /// one `idx == i` per slot, selecting the element where it hits.
    ///
    /// # Why not the lookup
    ///
    /// [`Self::gen_witness_array_get`] is cheaper and is what the array arm uses — one lookup
    /// argument instead of `n` selects — and it works for slices as far as R1CS generation is
    /// concerned, since `LookupTarget::Array` builds its table from the concrete array *object*. It
    /// is unavailable here because it cannot be **compiled to WASM**: the LLSSA lowering sizes the
    /// table from the *type*, via `lookup_array_len` -> `array_info`, which panics "Expected array
    /// type" on anything that is not a fixed-size `Array`. A slice type carries no length, and the
    /// physical length is only recoverable from the value's provenance. Routing slices through the
    /// lookup therefore compiles and proves fine while dying in the WASM lane, which is what the
    /// scan buys.
    ///
    /// The cost is real and worth knowing: measured against the lookup, the scan is **+n rows per
    /// read** (n = 64: 153 -> 226; n = 256: 537 -> 802), and it does not amortize the way a lookup
    /// does across several reads of the same container. A fixed-size array is the way to get
    /// witness-indexed random access cheaply; only slices pay this.
    ///
    /// Closing the gap means giving the LLSSA lowering a length it can use — either by teaching
    /// `lookup_array_len` about slices, or by materializing the physical slice as an array at the
    /// read (post-purify physical lengths are compile-time constants, but the constant is not
    /// available *at lowering time*, only after the later fold).
    ///
    /// # Soundness
    ///
    /// The assert is what replaces the bound the lookup gave for free — a key outside the table
    /// cannot satisfy a lookup, whereas a scan that finds no hit would happily return the
    /// accumulator's default. With `idx < slice_len` constrained and the scan covering exactly
    /// `0..slice_len`, precisely one slot hits. It is emitted under the op's guard so an inactive
    /// branch still cannot fail.
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
        let acc_type = push_witness_of_to_leaves(elem_type.clone());
        let idx_bits = uint_bits(
            function_type_info.get_value_type(idx),
            "witness slice get index",
        );

        let slice_len = b.slice_len(arr);
        let cmp_bits = idx_bits.max(32);
        let idx_cmp = b.widen_u(idx, idx_bits, cmp_bits);
        let len_cmp = b.widen_u(slice_len, 32, cmp_bits);
        b.emit_guarded(
            guard,
            OpCode::AssertCmp {
                kind: CmpKind::Lt,
                lhs: idx_cmp,
                rhs: len_cmp,
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
                let i_cmp = bb.widen_u(i, 32, cmp_bits);
                let hit = bb.eq(idx_cmp, i_cmp);
                let arr_i = bb.array_get(arr, i);
                let acc2 = select_leaves(bb, hit, arr_i, acc, &elem_type);
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

        // `select_leaves` selects the written value against the old cell *at the element type*.
        // This avoids absorbing a width mismatch so the two operands must already share a
        // witness-free skeleton. `witness_cast_insertion` guarantees that as a Noir store into
        // a container is type-checked and any narrowing is an explicit cast, but we rely on it so
        // pin it here for safety.
        debug_assert_eq!(
            function_type_info.get_value_type(value).strip_all_witness(),
            result_elem_type.strip_all_witness(),
            "ArraySet with witness idx: value type does not match the element type; \
             `select_leaves` cannot reconcile them"
        );

        let idx_bits = uint_bits(function_type_info.get_value_type(idx), "ArraySet index");
        let cmp_bits = idx_bits.max(32);
        let idx_cmp = b.widen_u(idx, idx_bits, cmp_bits);

        // `hit` is the slot's own "this is the write target" bit, before the guard is folded in.
        // The array arm sums it, so it must not be pre-masked.
        let elem_at = |b: &mut HLBlockEmitter<'_>, i: ValueId| -> (ValueId, ValueId) {
            let i_cmp = b.widen_u(i, 32, cmp_bits);
            let hit = b.eq(idx_cmp, i_cmp);
            let write = match guard {
                Some(g) => b.and(g, hit),
                None => hit,
            };
            let arr_i = b.array_get(arr, i);
            (
                select_leaves(b, write, value, arr_i, &result_elem_type),
                hit,
            )
        };

        // The rebuild below only ever writes at a slot it VISITS, so an out-of-range index would
        // silently no-op. Noir, however, treats an out-of-bounds write as an execution failure, so
        // both container kinds constrain the index — under the op's guard, so an inactive branch
        // still cannot fail. The two arms use different encodings of the same condition, because
        // only one of them can afford the cheap one:
        //
        // - **Array.** The trip count is static, so every `hit` is reachable and exactly one of
        //   them is set iff the index is in range. `sum(hit) == 1` is one linear row, and needs no
        //   range decomposition. It is also *stronger* than an ordering comparison, since it pins
        //   the index to a specific slot rather than to an interval.
        //
        //   This leans on `hit` being pinned rather than merely hinted, which it is:
        //   `LowerWitnessCompareOps` emits `d * (q + r) == 1 - r` for `r = (lhs == rhs)` over
        //   `d = lhs - rhs`, and the `q = (1 - r) / d` it feeds that from is itself constrained
        //   by the field-division lowering as `q * d == 1 - r`. The two together give `d * r == 0`,
        //   so `d != 0` forces `r == 0` and `d == 0` forces `r == 1`. A prover cannot forge a hit
        //   at a slot the index does not name, which is what makes the sum a real bound.
        // - **Slice.** The trip count is `slice_len`, so there is no static set of hits to sum;
        //   it keeps the ordering comparison, which is the check `gen_witness_slice_get` emits.
        //   Note every slice reaching here has a *pure* length: `PurifyWitnessSlices` runs long
        //   before this pass and rewrites each witness-length slice into a `(physical, log_len,
        //   start)` tuple, emitting `index < log_len` itself. What arrives here is a set on the
        //   pure-length `physical` at `start + index`, so `slice_len` is the physical capacity and
        //   this bound holds because purify maintains `start + log_len <= physical.len()`.
        let is_slice = result_type.is_slice();

        let updated = if is_slice {
            let slice_len = b.slice_len(arr); // Pure length, so this folds to a constant.
            let len_cmp = b.widen_u(slice_len, 32, cmp_bits);
            b.emit_guarded(
                guard,
                OpCode::AssertCmp {
                    kind: CmpKind::Lt,
                    lhs: idx_cmp,
                    rhs: len_cmp,
                },
            );
            let empty = b.mk_seq(vec![], SequenceTargetType::Slice, result_elem_type.clone());
            b.build_slice_extend_loop(slice_len, (empty, result_type.clone()), |b, i| {
                elem_at(b, i).0
            })
        } else {
            let length = array_len(&result_type, "ArraySet result");
            // The sum is witness-derived (every `hit` is because the index is), so the loop-carried
            // accumulator has to be typed that way too. Typed pure, the assert below would look
            // constant-foldable to `LowerWitnessAssertOps`, which would leave it alone and let a
            // non-constant `AssertCmp` reach the R1CS generator.
            let sum_type = push_witness_of_to_leaves(Type::field());
            let zero = b.default_value(&sum_type);

            // A zero-length array runs no loop, leaving the sum at `0`, so the assert below reads
            // `0 == 1` and rejects — which is right, every index into it is out of range.
            let (updated, hits) = b.build_array_loop_with_acc(
                length,
                result_elem_type.clone(),
                (zero, sum_type),
                |b, i, acc| {
                    let (elem, hit) = elem_at(b, i);
                    let hit_field = b.cast_to_field(hit);
                    (elem, b.add(acc, hit_field))
                },
            );
            let one = b.field_const(b.field().one());
            b.emit_guarded(
                guard,
                OpCode::AssertCmp {
                    kind: CmpKind::Eq,
                    lhs: hits,
                    rhs: one,
                },
            );
            updated
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
            TypeExpr::Slice(_) => {
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

/// `hit ? new_v : acc`, pushed down to the scalar leaves of `ty`.
///
/// A `Select` is only defined on scalars, so an array-typed element is rebuilt cell by cell with
/// one select per leaf. Used by every witness-indexed rebuild scan in this phase — the array and
/// slice arms of `ArraySet`, `gen_witness_slice_get`, and the `SliceInsert`/`SliceRemove`
/// lowerings — hence the type-agnostic name.
pub(super) fn select_leaves(
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
            b.build_array_loop(*n, push_witness_of_to_leaves(inner.clone()), |b, j| {
                let nj = b.array_get(new_v, j);
                let aj = b.array_get(acc, j);
                select_leaves(b, hit, nj, aj, &inner)
            })
        }
        other => panic!(
            "witness-indexed array/slice rebuild: unsupported element type {:?}",
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
        TypeExpr::Slice(_) | TypeExpr::Ref(_) | TypeExpr::Function | TypeExpr::Blob(..) => {
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

pub(super) fn uint_bits(ty: &Type, context: &str) -> usize {
    match ty.strip_witness().expr {
        TypeExpr::U(n) => n,
        _ => panic!("{context}: expected unsigned integer type, got {ty}"),
    }
}
