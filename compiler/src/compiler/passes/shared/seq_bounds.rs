//! The single definition of "this sequence op is out of bounds", shared by the slice lowerings,
//! `LowerZstSlices`, `LowerPureGuards`, `side_effect_free_guards`, and DCE's dead-op rewrite so they
//! cannot drift.
//!
//! Two shapes are needed because the consumers want different things from the same comparison: the
//! lowerings and DCE want an `AssertCmp` to emit, while `LowerPureGuards` wants the *condition* as
//! a value so it can branch on it. [`seq_bounds_operands`] is the piece they all agree on: which
//! value is the length, compared with an `int32` index.

use crate::compiler::ssa::{
    ValueId,
    hlssa::{CmpKind, HLSSA, OpCode, Type, TypeExpr, builder::HLEmitter},
};
use mavros_int_semantics::IntBits;

/// A failable sequence op's bounds condition.
pub enum SeqBoundsCheck {
    /// Requires `0 < len`.
    Pop { slice: ValueId },

    /// Requires `index < len + 1`.
    Insert { slice: ValueId, index: ValueId },

    /// Requires `index < len`.
    Remove { slice: ValueId, index: ValueId },

    /// An `ArrayGet` or `ArraySet`: requires `index < len`.
    ///
    /// DCE emits this only before witness-slice purification, while a vector's `slice_len`
    /// is still its logical length. A live witness-indexed read gets its bounds constraint
    /// from lowering; an unused read must retain a check when its lookup is removed.
    SeqAccess { seq: ValueId, index: ValueId },
}

/// Matches only at the top level: a `Guard`-wrapped op is required *not* to fail inside an inactive
/// branch, and the lowerings encode that by emitting the assert under the same guard.
pub fn failable_bounds(instruction: &OpCode) -> Option<SeqBoundsCheck> {
    match instruction {
        OpCode::SlicePop { slice, .. } => Some(SeqBoundsCheck::Pop { slice: *slice }),
        OpCode::SliceInsert { slice, index, .. } => Some(SeqBoundsCheck::Insert {
            slice: *slice,
            index: *index,
        }),
        OpCode::SliceRemove { slice, index, .. } => Some(SeqBoundsCheck::Remove {
            slice: *slice,
            index: *index,
        }),
        OpCode::ArrayGet { array, index, .. } | OpCode::ArraySet { array, index, .. } => {
            Some(SeqBoundsCheck::SeqAccess {
                seq: *array,
                index: *index,
            })
        }
        _ => None,
    }
}

impl SeqBoundsCheck {
    pub fn operands(&self) -> (ValueId, Option<ValueId>) {
        match self {
            Self::Pop { slice } => (*slice, None),
            Self::Insert { slice, index }
            | Self::Remove { slice, index }
            | Self::SeqAccess { seq: slice, index } => (*slice, Some(*index)),
        }
    }
}

/// The length of `seq` and the index, as the operands of their comparison.
///
/// Returns `(len, index_cmp, len_cmp)`; `len` is also what the insert lowering needs for the slice
/// it builds.
pub fn seq_bounds_operands(
    emitter: &mut impl HLEmitter,
    seq: ValueId,
    index: ValueId,
    seq_ty: &Type,
    index_ty: &Type,
) -> (ValueId, ValueId, ValueId) {
    let len = seq_len(emitter, seq, seq_ty);
    let (idx_cmp, len_cmp) = index_bounds_operands(index, index_ty, len);
    (len, idx_cmp, len_cmp)
}

/// An array's length is in its type; a slice carries it as a value.
pub fn seq_len(emitter: &mut impl HLEmitter, seq: ValueId, seq_ty: &Type) -> ValueId {
    match &seq_ty.strip_witness().expr {
        TypeExpr::Array(_, n) => emitter.int_const(IntBits::from_u128(32, *n as u128)),
        TypeExpr::Slice(_) => emitter.slice_len(seq),
        other => ice!("seq bounds check on non-sequence type: {other:?}"),
    }
}

/// The index and the length, compared as they are: the type analysis admits only an `int32` index,
/// and a length is one.
pub fn index_bounds_operands(index: ValueId, index_ty: &Type, len: ValueId) -> (ValueId, ValueId) {
    let TypeExpr::Int(32) = index_ty.strip_witness().expr else {
        ice!("a sequence index must be an int32, not {index_ty}")
    };
    (index, len)
}

/// Returns `(assert, len)`; the caller emits the assert — bare, or under the op's guard.
pub fn build_pop_bounds_assert(emitter: &mut impl HLEmitter, slice: ValueId) -> (OpCode, ValueId) {
    let len = emitter.slice_len(slice);
    (build_pop_bounds_assert_on_len(emitter, len), len)
}

/// Returns `assert`.
pub fn build_pop_bounds_assert_on_len(emitter: &mut impl HLEmitter, len: ValueId) -> OpCode {
    let zero = emitter.int_const(IntBits::zero(32));
    OpCode::AssertCmp {
        kind: CmpKind::ULt,
        lhs: zero,
        rhs: len,
    }
}

/// Returns `(assert, len, new_len, idx_cmp)`; the insert lowering's rebuild scan reuses the
/// intermediates instead of re-emitting them.
pub fn build_insert_bounds_assert(
    emitter: &mut impl HLEmitter,
    slice: ValueId,
    index: ValueId,
    index_ty: &Type,
) -> (OpCode, ValueId, ValueId, ValueId) {
    let len = emitter.slice_len(slice);
    let one = emitter.int_const(IntBits::one(32));
    let new_len = emitter.uadd(len, one);
    let (assert, idx_cmp) = build_lt_bounds_assert_on_len(new_len, index, index_ty);
    (assert, len, new_len, idx_cmp)
}

/// Returns `(assert, len, idx_cmp)`.
pub fn build_remove_bounds_assert(
    emitter: &mut impl HLEmitter,
    slice: ValueId,
    index: ValueId,
    index_ty: &Type,
) -> (OpCode, ValueId, ValueId) {
    let len = emitter.slice_len(slice);
    let (assert, idx_cmp) = build_lt_bounds_assert_on_len(len, index, index_ty);
    (assert, len, idx_cmp)
}

/// Returns `(assert, idx_cmp)`.
pub fn build_lt_bounds_assert_on_len(
    len: ValueId,
    index: ValueId,
    index_ty: &Type,
) -> (OpCode, ValueId) {
    let (idx_cmp, len_cmp) = index_bounds_operands(index, index_ty, len);
    let assert = OpCode::AssertCmp {
        kind: CmpKind::ULt,
        lhs: idx_cmp,
        rhs: len_cmp,
    };
    (assert, idx_cmp)
}

/// `index < len(seq)` for a user array/vector access, before witness-slice purification.
pub fn build_seq_access_bounds_assert(
    ssa: &HLSSA,
    emitter: &mut impl HLEmitter,
    seq: ValueId,
    index: ValueId,
    seq_ty: &Type,
    index_ty: &Type,
) -> Option<OpCode> {
    if !matches!(
        seq_ty.strip_witness().expr,
        TypeExpr::Array(_, _) | TypeExpr::Slice(_)
    ) {
        return None;
    }
    assert!(
        !matches!(seq_ty.strip_witness().expr, TypeExpr::Slice(_))
            || !ssa.witness_slices_purified(),
        "cannot derive logical slice bounds after witness-slice purification"
    );
    let (_, idx_cmp, len_cmp) = seq_bounds_operands(emitter, seq, index, seq_ty, index_ty);
    Some(OpCode::AssertCmp {
        kind: CmpKind::ULt,
        lhs: idx_cmp,
        rhs: len_cmp,
    })
}

/// The check alone — DCE's form, when the op itself is dead. Returns whether one was emitted.
pub fn emit_bounds_assert(
    ssa: &HLSSA,
    emitter: &mut impl HLEmitter,
    check: &SeqBoundsCheck,
    seq_ty: Option<&Type>,
    index_ty: Option<&Type>,
) -> bool {
    let index_ty = || index_ty.expect("an indexed sequence bounds check requires the index type");
    let assert = match check {
        SeqBoundsCheck::Pop { slice } => build_pop_bounds_assert(emitter, *slice).0,
        SeqBoundsCheck::Insert { slice, index } => {
            build_insert_bounds_assert(emitter, *slice, *index, index_ty()).0
        }
        SeqBoundsCheck::Remove { slice, index } => {
            build_remove_bounds_assert(emitter, *slice, *index, index_ty()).0
        }
        SeqBoundsCheck::SeqAccess { seq, index } => {
            let seq_ty = seq_ty.expect("a sequence access bounds check requires the sequence type");
            match build_seq_access_bounds_assert(ssa, emitter, *seq, *index, seq_ty, index_ty()) {
                Some(assert) => assert,
                None => return false,
            }
        }
    };
    emitter.emit(assert);
    true
}
