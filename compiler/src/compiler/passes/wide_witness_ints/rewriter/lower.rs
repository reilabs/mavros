use super::{
    super::types::{element_types, flat_limbs, limb_types, limb_widths, paired},
    FunctionTypeInfo, Rewriter,
};

use crate::compiler::ssa::{
    Instruction, ValueId,
    hlssa::{
        BinaryArithOpKind, Blob, CastTarget, CmpKind, Constant, LookupTarget, OpCode,
        builder::HLEmitter,
    },
};
use mavros_int_semantics::IntBits;
use num_bigint::BigUint;

impl Rewriter<'_> {
    /// Rewrite one instruction into the limb-wise instructions that replace it.
    pub(super) fn lower(&mut self, op: &OpCode) {
        if self.lower_global(op) {
            return;
        }
        if self.lower_through_chain(op)
            || self.lower_product(op)
            || self.lower_division(op)
            || self.lower_shift(op)
        {
            return;
        }

        let touches_wide = op
            .get_inputs()
            .chain(op.get_results())
            .any(|value| self.limbs(*value).len() > 1);
        if !touches_wide {
            self.push(op.clone());
            return;
        }

        match op {
            OpCode::Cast {
                result,
                value,
                target,
            } => self.lower_cast(*result, *value, target),

            OpCode::SExt {
                result,
                value,
                from_bits,
                to_bits,
            } => self.lower_sext(*result, *value, *from_bits, *to_bits),

            OpCode::Cmp {
                kind,
                result,
                lhs,
                rhs,
            } => self.lower_compare(*kind, *result, *lhs, *rhs),

            OpCode::AssertCmp { kind, lhs, rhs } => {
                self.lower_assert_compare(*kind, *lhs, *rhs, None);
            }

            // A witnessed spread or unspread, under its guard where it has one, whose lookups and
            // zero checks hold only where the guard does.
            OpCode::Spread { value, .. } | OpCode::Unspread { value, .. }
                if self.is_witness(*value) =>
            {
                self.lower_spread_family(None, op);
            }
            OpCode::Guard { condition, inner }
                if matches!(
                    **inner,
                    OpCode::Spread { value, .. } | OpCode::Unspread { value, .. }
                        if self.is_witness(value)
                ) =>
            {
                let guard = self.one(*condition);
                self.lower_spread_family(Some(guard), inner);
            }

            // An assertion in a branch on a witness, which holds only where the branch is taken.
            OpCode::Guard { condition, inner } if matches!(**inner, OpCode::AssertCmp { .. }) => {
                let OpCode::AssertCmp { kind, lhs, rhs } = **inner else {
                    ice_unreachable!("matched above");
                };
                let guard = self.one(*condition);
                self.lower_assert_compare(kind, lhs, rhs, Some(guard));
            }

            // Everything below moves limbs without reading them, so each becomes its narrow form
            // once per limb.
            OpCode::Select {
                result,
                cond,
                if_t,
                if_f,
            } => {
                let cond = self.one(*cond);
                let results = self.limbs(*result);
                let then = self.operand_limbs(*if_t, results.len());
                let otherwise = self.operand_limbs(*if_f, results.len());
                for ((result, if_t), if_f) in results.into_iter().zip(then).zip(otherwise) {
                    self.push(OpCode::Select {
                        result,
                        cond,
                        if_t,
                        if_f,
                    });
                }
            }

            // The transpose. An element-major sequence of `k`-limb values becomes `k` limb-major
            // sequences, each holding one limb position of every element.
            //
            // A pure element beside a witnessed one arrives as a single value which `operand_limbs`
            // splits: the same mixed-domain case a wide comparison meets.
            OpCode::MkSeq {
                result,
                elems,
                seq_type,
                elem_type,
            } => {
                let results = self.limbs(*result);
                let count = results.len();
                let elem_types = element_types(elem_type, self.limb_config(), count);
                let per_element: Vec<Vec<ValueId>> = elems
                    .iter()
                    .map(|elem| self.operand_limbs(*elem, count))
                    .collect();
                for (index, (result, elem_type)) in results.into_iter().zip(elem_types).enumerate()
                {
                    let elems = per_element.iter().map(|limbs| limbs[index]).collect();
                    self.push(OpCode::MkSeq {
                        result,
                        elems,
                        seq_type: *seq_type,
                        elem_type,
                    });
                }
            }

            // The witness-indexed read: `witness_array_access` built its lookup before this pass,
            // so the index and flag are narrow and pinned. One lookup per limb sequence against
            // them selects each limb by the same `sum(hit) == 1` argument.
            OpCode::Lookup {
                target: LookupTarget::Array(array),
                args,
                flag,
            }
            | OpCode::DLookup {
                target: LookupTarget::Array(array),
                args,
                flag,
            } => {
                assert_eq!(
                    args.len(),
                    2,
                    "ICE: an array lookup takes an index and a result"
                );
                let dynamic = matches!(op, OpCode::DLookup { .. });
                let index = self.one(args[0]);
                let flag = self.one(*flag);
                let results = self.limbs(args[1]);
                let arrays = self.limbs(*array);
                // `paired` rather than a `zip`: a limb short here is a limb the tape never pins,
                // which an honest witness still satisfies. It is the one mismatch in this pass
                // that would be unsound rather than merely wrong.
                for (result, array) in paired(results, arrays) {
                    let target = LookupTarget::Array(array);
                    let args = vec![index, result];
                    self.push(if dynamic {
                        OpCode::DLookup { target, args, flag }
                    } else {
                        OpCode::Lookup { target, args, flag }
                    });
                }
            }

            // A blob-backed sequence: one blob per limb. The elements are constants, so the split
            // is arithmetic on their patterns, which lets a constant sequence hold elements too
            // wide for one field element.
            OpCode::MkSeqOfBlob {
                result,
                element_type,
                blob,
            } => {
                let results = self.limbs(*result);
                let Some(Constant::Blob(blob)) = self.ssa.get_const(*blob).map(|c| (*c).clone())
                else {
                    ice!("a blob-backed sequence without a blob constant")
                };
                let bits = element_type
                    .int_width()
                    .unwrap_or_else(|| ice!("a wide blob sequence of {element_type}"));
                let widths = limb_widths(bits, self.limb_bits());
                let elem_types = element_types(element_type, self.limb_config(), results.len());
                for (index, ((result, width), element_type)) in
                    results.into_iter().zip(&widths).zip(elem_types).enumerate()
                {
                    let low = index * self.limb_bits();
                    let elements = blob
                        .elements
                        .iter()
                        .map(|element| {
                            let Constant::Int(pattern) = element else {
                                ice!("a wide blob element that is not an integer")
                            };
                            Constant::Int(pattern.bit_range(low, *width))
                        })
                        .collect();
                    let limb_blob = self
                        .ssa
                        .add_const(Constant::Blob(Blob::new(element_type.clone(), elements)));
                    self.push(OpCode::MkSeqOfBlob {
                        result,
                        element_type,
                        blob: limb_blob,
                    });
                }
            }

            OpCode::MkRepeated {
                result,
                element,
                seq_type,
                count,
                elem_type,
            } => {
                let results = self.limbs(*result);
                let elem_types = element_types(elem_type, self.limb_config(), results.len());
                let elements = self.operand_limbs(*element, results.len());
                for ((result, element), elem_type) in
                    results.into_iter().zip(elements).zip(elem_types)
                {
                    self.push(OpCode::MkRepeated {
                        result,
                        element,
                        seq_type: *seq_type,
                        count: *count,
                        elem_type,
                    });
                }
            }

            // A witness-indexed read inside a branch on a witness, whose hint the array lowering
            // guards so that an index past the end on the branch not taken does not trap: one
            // guarded read per limb sequence, under the same condition.
            OpCode::Guard { condition, inner } if matches!(**inner, OpCode::ArrayGet { .. }) => {
                let OpCode::ArrayGet {
                    result,
                    array,
                    index,
                } = **inner
                else {
                    ice_unreachable!("matched above");
                };
                let condition = self.one(*condition);
                self.lower_array_get(Some(condition), result, array, index);
            }

            // One read per limb sequence, at the caller's index. The index is narrow and shared
            // rather than re-derived, so a witnessed one is looked up once per limb against the
            // value the array lowering pinned.
            OpCode::ArrayGet {
                result,
                array,
                index,
            } => self.lower_array_get(None, *result, *array, *index),

            OpCode::ArraySet {
                result,
                array,
                index,
                value,
            } => {
                let index = self.one(*index);
                let results = self.limbs(*result);
                let arrays = self.limbs(*array);
                let values = self.operand_limbs(*value, results.len());
                for ((result, array), value) in paired(results, arrays).zip(values) {
                    self.push(OpCode::ArraySet {
                        result,
                        array,
                        index,
                        value,
                    });
                }
            }

            // The slice operations that survive this far. Every operation that changes a limb
            // sequence's length is applied to all `k`, so the length is read off any one, and the
            // index is shared.
            //
            // `SlicePop`, `SliceInsert` and `SliceRemove` have no arm:
            // `InstructionLowering::slice_ops` replaces each with an assert, a get and a copy loop
            // before this pass.
            OpCode::SliceLen { result, slice } => {
                let slice = self.limbs(*slice);
                self.push(OpCode::SliceLen {
                    result: self.one(*result),
                    slice: slice[0],
                });
            }

            OpCode::SlicePush {
                dir,
                result,
                slice,
                values,
            } => {
                let results = self.limbs(*result);
                let slices = self.limbs(*slice);
                let per_value: Vec<Vec<ValueId>> = values
                    .iter()
                    .map(|value| self.operand_limbs(*value, results.len()))
                    .collect();
                for (index, (result, slice)) in paired(results, slices).enumerate() {
                    let values = per_value.iter().map(|limbs| limbs[index]).collect();
                    self.push(OpCode::SlicePush {
                        dir: *dir,
                        result,
                        slice,
                        values,
                    });
                }
            }

            // The bitwise three, the one arithmetic family with **no cross-limb interaction**:
            // bit `i` of the answer depends only on bit `i` of each operand, so each answer limb is
            // the same operation on the matching operand limbs. There is no carry and nothing to
            // reconstruct: each limb is already held to its width, and an operation that sets no
            // bit absent from both operands cannot break that.
            OpCode::BinaryArithOp {
                kind:
                    kind @ (BinaryArithOpKind::And | BinaryArithOpKind::Or | BinaryArithOpKind::Xor),
                result,
                lhs,
                rhs,
            } => {
                let results = self.limbs(*result);
                let pairs = self.operand_pair(*lhs, *rhs);
                assert_eq!(
                    results.len(),
                    pairs.len(),
                    "ICE: a bitwise result of {} limbs met operands of {}",
                    results.len(),
                    pairs.len()
                );
                for (result, (lhs, rhs)) in results.into_iter().zip(pairs) {
                    self.push(OpCode::BinaryArithOp {
                        kind: *kind,
                        result,
                        lhs,
                        rhs,
                    });
                }
            }

            // The complement, the unary member of the same family and limb-wise for the same
            // reason: bit `i` of the answer depends only on bit `i` of the operand. Each limb is
            // complemented at its width, so the top one, which may be narrower than a full limb,
            // gains no bits beyond the value's width.
            OpCode::Not { result, value } => {
                let results = self.limbs(*result);
                let values = self.operand_limbs(*value, results.len());
                for (result, value) in paired(results, values) {
                    self.push(OpCode::Not { result, value });
                }
            }

            OpCode::Alloc { result, value } => {
                for (result, value) in paired(self.limbs(*result), self.limbs(*value)) {
                    self.push(OpCode::Alloc { result, value });
                }
            }

            OpCode::Load { result, ptr } => {
                for (result, ptr) in paired(self.limbs(*result), self.limbs(*ptr)) {
                    self.push(OpCode::Load { result, ptr });
                }
            }

            OpCode::Store { ptr, value } => {
                for (ptr, value) in paired(self.limbs(*ptr), self.limbs(*value)) {
                    self.push(OpCode::Store { ptr, value });
                }
            }

            // A whole pure integer written as a witnessed one held as limbs, such as an
            // unconstrained call's answer: each limb is a new column, written from the value and
            // bounded at its width, as nothing else ties the limbs to it.
            OpCode::WriteWitness {
                result: Some(result),
                value,
                pinned,
            } if self.limbs(*value).len() == 1 && self.limbs(*result).len() > 1 => {
                // A pinned write is an entry point's input, which `prepare_entry_point` keeps
                // within one element.
                debug_assert!(!pinned, "ICE: a pinned write of a wide value");
                let bits = self
                    .wide_width(*result)
                    .expect("ICE: a write into the representation without a wide result");
                let widths = limb_widths(bits, self.limb_bits());
                for (index, (result, width)) in
                    self.limbs(*result).into_iter().zip(widths).enumerate()
                {
                    let (limb, _) = self.witness_limb(*value, bits, index, width, width);
                    self.push(OpCode::Cast {
                        result,
                        value: limb,
                        target: CastTarget::Nop,
                    });
                }
            }

            OpCode::WriteWitness {
                result,
                value,
                pinned,
            } => {
                let values = self.limbs(*value);
                let results: Vec<Option<ValueId>> = match result {
                    Some(result) => self.limbs(*result).into_iter().map(Some).collect(),
                    None => vec![None; values.len()],
                };
                assert_eq!(
                    results.len(),
                    values.len(),
                    "ICE: {} witness columns were written from {} limbs",
                    results.len(),
                    values.len()
                );
                for (result, value) in results.into_iter().zip(values) {
                    self.push(OpCode::WriteWitness {
                        result,
                        value,
                        pinned: *pinned,
                    });
                }
            }

            OpCode::FreshWitness {
                result,
                result_type,
            } => {
                let types = limb_types(result_type, self.limb_config());
                let results = self.limbs(*result);
                assert_eq!(
                    results.len(),
                    types.len(),
                    "ICE: {} fresh witnesses were minted for a type of {} limbs",
                    results.len(),
                    types.len()
                );
                for (result, result_type) in results.into_iter().zip(types) {
                    self.push(OpCode::FreshWitness {
                        result,
                        result_type,
                    });
                }
            }

            OpCode::Call {
                results,
                function,
                args,
                unconstrained,
            } => self.push(OpCode::Call {
                results: flat_limbs(self.value_map, results),
                function: function.clone(),
                args: flat_limbs(self.value_map, args),
                unconstrained: *unconstrained,
            }),

            // A wide witnessed value reaching anything else is a shape this pass does not
            // represent.
            other => ice!(
                "{other:?} reached the multi-cell representation with a wide operand, which is a shape it does not represent"
            ),
        }
    }

    /// A global opcode, renumbered to its slots' new place and fanned out over its limbs, a limb
    /// per slot. A drop drops only the slots that hold something on the heap.
    fn lower_global(&mut self, op: &OpCode) -> bool {
        if !self.globals.moved {
            return false;
        }
        match op {
            OpCode::ReadGlobal {
                result,
                offset,
                result_type,
            } => {
                let base = self.globals.offsets[*offset as usize];
                let results = self.limbs(*result);
                let types = limb_types(result_type, self.limb_config());
                assert_eq!(
                    results.len(),
                    types.len(),
                    "ICE: a global read of {} slots answers {} limbs",
                    types.len(),
                    results.len()
                );
                for (slot, (result, result_type)) in results.into_iter().zip(types).enumerate() {
                    self.push(OpCode::ReadGlobal {
                        result,
                        offset: (base + slot) as u64,
                        result_type,
                    });
                }
            }
            OpCode::InitGlobal { global, value } => {
                let base = self.globals.offsets[*global];
                let values = self.limbs(*value);
                let slots = limb_types(&self.globals.old_types[*global], self.limb_config()).len();
                assert_eq!(
                    values.len(),
                    slots,
                    "ICE: a global of {slots} slots is initialised with {} limbs",
                    values.len()
                );
                for (slot, value) in values.into_iter().enumerate() {
                    self.push(OpCode::InitGlobal {
                        global: base + slot,
                        value,
                    });
                }
            }
            OpCode::DropGlobal { global } => {
                let base = self.globals.offsets[*global];
                for (slot, ty) in limb_types(&self.globals.old_types[*global], self.limb_config())
                    .into_iter()
                    .enumerate()
                {
                    if ty.is_heap_allocated() {
                        self.push(OpCode::DropGlobal {
                            global: base + slot,
                        });
                    }
                }
            }
            _ => return false,
        }
        true
    }

    /// One read per limb sequence of `array`, at the caller's index and under `guard` if any.
    ///
    /// A pure element is recombined into `result` at once for users that expect a whole integer.
    /// The limbs remain in `read_limbs` for any later operand consumer in this block, including
    /// witness-indexed lookups, without splitting the whole value again. Recombination costs
    /// interpreter work, not constraints, as a pure value is known wherever the constraints are built.
    fn lower_array_get(
        &mut self,
        guard: Option<ValueId>,
        result: ValueId,
        array: ValueId,
        index: ValueId,
    ) {
        let index = self.one(index);
        let arrays = self.limbs(array);
        let results = self.limbs(result);
        let whole = results.len() == 1 && arrays.len() > 1;
        let reads: Vec<ValueId> = if whole {
            arrays.iter().map(|_| self.fresh()).collect()
        } else {
            results
        };
        for (read, array) in paired(reads.clone(), arrays) {
            let get = OpCode::ArrayGet {
                result: read,
                array,
                index,
            };
            self.push(match guard {
                Some(condition) => OpCode::Guard {
                    condition,
                    inner: Box::new(get),
                },
                None => get,
            });
        }
        if whole {
            let bits = self
                .types
                .get_value_type(result)
                .int_width()
                .unwrap_or_else(|| {
                    ice!("a whole element read out of limb sequences is not an integer")
                });
            let combined = self.recombine_pure(&reads, bits);
            self.push(OpCode::Cast {
                result,
                value: combined,
                target: CastTarget::Nop,
            });
            self.read_limbs.insert(result, reads);
        }
    }
    /// A spread or an unspread, under `guard` where there is one.
    fn lower_spread_family(&mut self, guard: Option<ValueId>, op: &OpCode) {
        match op {
            OpCode::Spread {
                result,
                value,
                value_bits,
            } => self.lower_spread(guard, *result, *value, *value_bits),
            OpCode::Unspread {
                result_odd,
                result_even,
                value,
                value_bits,
            } => self.lower_unspread(guard, *result_odd, *result_even, *value, *value_bits),
            other => ice_unreachable!("{other:?} is not a spread or an unspread"),
        }
    }

    /// Equality: the conjunction of per-limb equalities.
    ///
    /// A pair of limbs both known at compile time is decided here rather than compared: the
    /// comparison's lowering would spend a gadget on a witnessed constant. One pair known to differ
    /// decides the whole equality, and if every pair is known equal, so is the value.
    fn lower_compare(&mut self, kind: CmpKind, result: ValueId, lhs: ValueId, rhs: ValueId) {
        assert!(
            matches!(kind, CmpKind::Eq),
            "ICE: a {kind:?} of a wide witnessed integer reached the multi-cell representation; the carry chain takes every ordering"
        );

        let mut pairs = Vec::new();
        let mut decided = true;
        for (lhs, rhs) in self.operand_pair(lhs, rhs) {
            match self.known_equal(lhs, rhs) {
                Some(true) => {}
                Some(false) => decided = false,
                None => pairs.push((lhs, rhs)),
            }
        }
        if !decided || pairs.is_empty() {
            let answer = self.int_const(IntBits::from_u128(1, u128::from(decided)));
            let target = if self.is_witness(result) {
                CastTarget::WitnessOf
            } else {
                CastTarget::Nop
            };
            self.push(OpCode::Cast {
                result,
                value: answer,
                target,
            });
            return;
        }

        let mut all = None;
        for (lhs, rhs) in pairs {
            let equal = self.fresh();
            self.push(OpCode::Cmp {
                kind: CmpKind::Eq,
                result: equal,
                lhs,
                rhs,
            });
            all = Some(match all {
                None => equal,
                Some(acc) => self.bin(BinaryArithOpKind::And, acc, equal),
            });
        }
        self.push(OpCode::Cast {
            result,
            value: all.expect("an integer has at least one limb"),
            target: CastTarget::Nop,
        });
    }

    /// Whether two limbs are equal, where both are known at compile time.
    pub(super) fn known_equal(&self, lhs: ValueId, rhs: ValueId) -> Option<bool> {
        let lhs = self.known(lhs)?;
        let rhs = self.known(rhs)?;
        Some(BigUint::from(&lhs) == BigUint::from(&rhs))
    }
}

/// The operation a `Guard` wraps, or `op` itself where there is none.
pub(super) fn unguarded(op: &OpCode) -> &OpCode {
    match op {
        OpCode::Guard { inner, .. } => inner.as_ref(),
        other => other,
    }
}

/// The width of an integer operation with a witnessed operand, read off its left one, or [`None`]
/// where both are pure or it is not an integer operation.
pub(super) fn witnessed_width(
    types: &FunctionTypeInfo,
    lhs: ValueId,
    rhs: ValueId,
) -> Option<usize> {
    let bits = types.get_value_type(lhs).int_width()?;
    (types.get_value_type(lhs).is_witness_of() || types.get_value_type(rhs).is_witness_of())
        .then_some(bits)
}
