//! The single definition of "this sequence op is out of bounds", shared by the slice lowerings,
//! `LowerPureGuards`, `side_effect_free_guards`, and DCE's dead-op rewrite so they cannot drift.
//!
//! Two shapes are needed because the consumers want different things from the same comparison: the
//! lowerings and DCE want an `AssertCmp` to emit, while `LowerPureGuards` wants the *condition* as
//! a value so it can branch on it. [`seq_bounds_operands`] is the piece they all agree on — which
//! value is the length, and at what width the two are compared.

use crate::compiler::ssa::{
    ValueId,
    hlssa::{CastTarget, CmpKind, OpCode, Type, TypeExpr, builder::HLEmitter},
};

/// A failable sequence op's bounds condition.
pub enum SeqBoundsCheck {
    /// Requires `0 < len`.
    Pop { slice: ValueId },

    /// Requires `index < len + 1`.
    Insert { slice: ValueId, index: ValueId },

    /// Requires `index < len`.
    Remove { slice: ValueId, index: ValueId },

    /// An `ArraySet`: requires `index < len`.
    ///
    /// Only *arrays* are checked through this variant. A witness-length slice access already
    /// carries `index < log_len` from `PurifyWitnessSlices`, and a pure-length one is constrained
    /// by its own lowering, so re-deriving a bound from the physical container here would be both
    /// redundant and — for a purified slice, whose `slice_len` is the capacity rather than the
    /// logical length — weaker than the check already present. Noir draws the same line: see
    /// `should_insert_oob_check` in `noirc_evaluator/src/ssa/opt/die/array_oob_checks.rs`, which
    /// notes that "vectors are expected to have explicit checks laid down in the initial SSA".
    ///
    /// `ArrayGet` is deliberately **not** matched, which is the one place this diverges from
    /// Noir's `should_insert_oob_check`. A live witness-indexed read already gets its bound for
    /// free from the lookup argument `gen_witness_array_get` emits, so the only thing at stake is
    /// a read whose result nothing uses.
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
        OpCode::ArraySet { array, index, .. } => Some(SeqBoundsCheck::SeqAccess {
            seq: *array,
            index: *index,
        }),
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

/// The length of `seq` and the index, brought to a common comparison width.
///
/// Returns `(len, len_cmp, index_cmp, cmp_bits)`; `len` is the un-widened length, which the insert
/// lowering needs for the slice it builds.
///
/// Compare at the wider of the two widths. Narrowing the index to u32 instead would alias an
/// out-of-range wide index onto its low limb — `1 << 32` would read as in-bounds — so the narrow
/// operand is always the one brought up. A non-integer index (a `Field` subscript) has no width to
/// widen to; it keeps the historical narrowing cast rather than change behaviour for a case none of
/// the consumers model.
pub fn seq_bounds_operands(
    emitter: &mut impl HLEmitter,
    seq: ValueId,
    index: ValueId,
    seq_ty: &Type,
    index_ty: &Type,
) -> (ValueId, ValueId, ValueId, usize) {
    let len = match &seq_ty.strip_witness().expr {
        TypeExpr::Array(_, n) => emitter.int_const(32, *n as u128),
        TypeExpr::Slice(_) => emitter.slice_len(seq),
        other => panic!("seq bounds check on non-sequence type: {other:?}"),
    };
    match index_ty.strip_witness().expr {
        TypeExpr::Int(idx_bits) => {
            let cmp_bits = idx_bits.max(32);
            let idx_cmp = emitter.widen_u(index, idx_bits, cmp_bits);
            let len_cmp = emitter.widen_u(len, 32, cmp_bits);
            (len, len_cmp, idx_cmp, cmp_bits)
        }
        _ => {
            let idx_cmp = emitter.cast_to(CastTarget::Int(32), index);
            (len, len, idx_cmp, 32)
        }
    }
}

/// Returns `(assert, len)`; the caller emits the assert — bare, or under the op's guard.
pub fn build_pop_bounds_assert(emitter: &mut impl HLEmitter, slice: ValueId) -> (OpCode, ValueId) {
    let len = emitter.slice_len(slice);
    let zero = emitter.int_const(32, 0);
    let assert = OpCode::AssertCmp {
        kind: CmpKind::ULt,
        lhs: zero,
        rhs: len,
    };
    (assert, len)
}

/// Returns `(assert, len, new_len, idx_cmp, cmp_bits)`; the insert lowering's rebuild scan reuses
/// the intermediates instead of re-emitting them.
pub fn build_insert_bounds_assert(
    emitter: &mut impl HLEmitter,
    slice: ValueId,
    index: ValueId,
    index_ty: &Type,
) -> (OpCode, ValueId, ValueId, ValueId, usize) {
    let idx_bits = index_bits(index_ty, "slice insert");
    let len = emitter.slice_len(slice);
    let one = emitter.int_const(32, 1);
    let new_len = emitter.uadd(len, one);
    let cmp_bits = idx_bits.max(32);
    let idx_cmp = emitter.widen_u(index, idx_bits, cmp_bits);
    let new_len_cmp = emitter.widen_u(new_len, 32, cmp_bits);
    let assert = OpCode::AssertCmp {
        kind: CmpKind::ULt,
        lhs: idx_cmp,
        rhs: new_len_cmp,
    };
    (assert, len, new_len, idx_cmp, cmp_bits)
}

/// Returns `(assert, len, idx_cmp, cmp_bits)`.
pub fn build_remove_bounds_assert(
    emitter: &mut impl HLEmitter,
    slice: ValueId,
    index: ValueId,
    index_ty: &Type,
) -> (OpCode, ValueId, ValueId, usize) {
    let idx_bits = index_bits(index_ty, "slice remove");
    let len = emitter.slice_len(slice);
    let cmp_bits = idx_bits.max(32);
    let idx_cmp = emitter.widen_u(index, idx_bits, cmp_bits);
    let len_cmp = emitter.widen_u(len, 32, cmp_bits);
    let assert = OpCode::AssertCmp {
        kind: CmpKind::ULt,
        lhs: idx_cmp,
        rhs: len_cmp,
    };
    (assert, len, idx_cmp, cmp_bits)
}

/// `index < len(seq)` for an array element access, or `None` when `seq` is not a fixed-length array
/// — see [`SeqBoundsCheck::SeqAccess`] for why slices are excluded here.
pub fn build_seq_access_bounds_assert(
    emitter: &mut impl HLEmitter,
    seq: ValueId,
    index: ValueId,
    seq_ty: &Type,
    index_ty: &Type,
) -> Option<OpCode> {
    if !matches!(seq_ty.strip_witness().expr, TypeExpr::Array(_, _)) {
        return None;
    }
    let (_, len_cmp, idx_cmp, _) = seq_bounds_operands(emitter, seq, index, seq_ty, index_ty);
    Some(OpCode::AssertCmp {
        kind: CmpKind::ULt,
        lhs: idx_cmp,
        rhs: len_cmp,
    })
}

/// The check alone — DCE's form, when the op itself is dead. Returns whether one was emitted.
pub fn emit_bounds_assert(
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
            match build_seq_access_bounds_assert(emitter, *seq, *index, seq_ty, index_ty()) {
                Some(assert) => assert,
                None => return false,
            }
        }
    };
    emitter.emit(assert);
    true
}

fn index_bits(ty: &Type, context: &str) -> usize {
    match ty.strip_witness().expr {
        TypeExpr::Int(n) => n,
        _ => panic!("{context}: index must be an integer, got {ty}"),
    }
}
