//! The constant-propagation lattice element and its constant-evaluation transfer functions.

use std::sync::{Arc, OnceLock};

use mavros_artifacts::FieldConfig;

use mavros_int_semantics::{self as semantics, CmpOp, Sign};

use crate::compiler::ssa::hlssa::{
    ArithGroup, BinaryArithOpKind, Blob, CastTarget, CmpKind, Constant, MAX_SUPPORTED_SIGNED_BITS,
    MAX_SUPPORTED_UNSIGNED_BITS, SliceOpDir, Type,
};

// CONSTNESS
// ================================================================================================

/// A value's _constness_: where it sits in the constant-propagation lattice.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Constness {
    /// Not (yet) known to be reachable with any value.
    Top,

    /// Proven to always hold this constant.
    Const(Arc<Constant>),

    /// Overdefined: holds a runtime-dependent (or non-foldable) value.
    Bottom,
}

// CONSTANT EVALUATION
// ================================================================================================

pub(crate) fn const_join(a: Constness, b: Constness) -> Constness {
    match (a, b) {
        (Constness::Top, x) | (x, Constness::Top) => x,
        (Constness::Bottom, _) | (_, Constness::Bottom) => Constness::Bottom,
        (Constness::Const(c1), Constness::Const(c2)) => {
            if c1 == c2 {
                Constness::Const(c1)
            } else {
                Constness::Bottom
            }
        }
    }
}

pub(crate) fn const_bool(c: &Constant) -> Option<bool> {
    match c {
        Constant::Int(1, 0) => Some(false),
        Constant::Int(1, 1) => Some(true),
        Constant::Int(..) | Constant::Field(_) | Constant::FnPtr(_) | Constant::Blob(_) => None,
    }
}

pub(crate) fn bool_constness(value: bool) -> Constness {
    Constness::Const(bool_constant(value))
}

pub(crate) fn bool_constant(value: bool) -> Arc<Constant> {
    static FALSE: OnceLock<Arc<Constant>> = OnceLock::new();
    static TRUE: OnceLock<Arc<Constant>> = OnceLock::new();
    let slot = if value { &TRUE } else { &FALSE };
    slot.get_or_init(|| Arc::new(Constant::Int(1, value as u128)))
        .clone()
}

/// Fold a binary arithmetic op.
///
/// Integer results must fit the operand width: an overflowing pure op is an erroneous evaluation
/// with a backend-specific residue, so an overflowing fold is refused rather than guessed at.
///
/// `Shl` is the exception, because for it overflow is not an error. A left shift _wraps_ — Noir
/// reports a runtime error only when the _amount_ reaches the width — so it folds to the truncated
/// value rather than declining, agreeing with every backend and with the witness lowering in
/// `witness_bitwise::wrap_shifted_product`.
pub(crate) fn eval_binary(
    kind: BinaryArithOpKind,
    a: &Constant,
    b: &Constant,
    field: FieldConfig,
) -> Option<Constant> {
    use ArithGroup::*;

    let group = kind.group();

    match (a, b) {
        // A `Constant::Int` is a raw bit pattern and says nothing about how to read itself; the
        // _operation_ decides, and these two folds are the two readings of the same patterns.
        (Constant::Int(s1, x), Constant::Int(s2, y)) => {
            fold_int(group, kind.sign(), *s1, *x, *s2, *y)
        }
        // FIELD-ASSUMPTION: L4-eval
        (Constant::Field(x), Constant::Field(y)) => {
            match group {
                Add => Some(Constant::Field(*x + *y)),
                Sub => Some(Constant::Field(*x - *y)),
                Mul => Some(Constant::Field(*x * *y)),
                Div => {
                    if *y == field.zero() {
                        None
                    } else {
                        // FIELD-ASSUMPTION: L4-inverse
                        Some(Constant::Field(*x / *y))
                    }
                }
                Rem | And | Or | Xor | Shl | Shr => None,
            }
        }
        // Mixed-kind pairs and non-scalar constants do not fold.
        (
            Constant::Int(..) | Constant::Field(_) | Constant::FnPtr(_) | Constant::Blob(_),
            Constant::Int(..) | Constant::Field(_) | Constant::FnPtr(_) | Constant::Blob(_),
        ) => None,
    }
}

/// The widest pattern an operation of this reading may act on.
///
/// A signed opcode tops out well below the integer type cap because the signed lowerings and the
/// VM's `sdiv_int`/`slt_int` are 64-bit for the moment.
fn width_cap(sign: Sign) -> usize {
    match sign {
        Sign::Signed => MAX_SUPPORTED_SIGNED_BITS,
        Sign::Unsigned => MAX_SUPPORTED_UNSIGNED_BITS,
    }
}

/// The width a fold may produce a constant at, or `None` if it must decline.
///
/// The two operands must already be at one width, shifts included. That is not a restriction
/// invented by this analysis: `hlssa_to_llssa::assert_int_arith_widths` requires _every_ integer
/// `BinaryArithOp`'s operands to be exactly the width of its result, so a mixed-width pair is IR
/// that panics downstream, and folding one would mint a constant for a shape nothing may build.
fn fold_width(sign: Sign, s1: usize, s2: usize) -> Option<usize> {
    ((1..=width_cap(sign)).contains(&s1) && s1 == s2).then_some(s1)
}

/// Fold a pair of raw patterns under one reading.
///
/// The arithmetic is [`mavros_int_semantics::eval`]'s. Everything this analysis adds is the refusal
/// to fold what the model _rejects_: an overflowing `Add`, a zero divisor, `INT_MIN / -1`, an
/// out-of-range shift amount. These are runtime errors in Noir, and folding one would delete the
/// rejection along with the operation.
///
/// `Shl` is not an exception though it may look like one. A left shift that pushes bits off the top
/// **wraps**, and Noir reports an error only when the _amount_ reaches the width, so the model
/// accepts it and returns the truncated value.
fn fold_int(
    group: ArithGroup,
    sign: Sign,
    s1: usize,
    x: u128,
    s2: usize,
    y: u128,
) -> Option<Constant> {
    let bits = fold_width(sign, s1, s2)?;
    let v = semantics::eval(group.into(), sign, bits, x, bits, y).value()?;
    Some(Constant::Int(bits, v))
}

/// Folds a constant comparison operation.
///
/// As in [`eval_binary`], the _comparison_ decides how the patterns are read: `ULt` compares them
/// as magnitudes and `SLt` as two's complement. `Eq` compares the patterns themselves, so it needs
/// no reading at all — only equal widths.
pub(crate) fn eval_cmp(kind: CmpKind, a: &Constant, b: &Constant) -> Option<Constant> {
    let res = |v: bool| Some(Constant::Int(1, v as u128));

    match (kind, a, b) {
        // `Eq` needs no reading at all, but it is routed through the model with the rest so that
        // "how are two patterns compared" has exactly one answer in this compiler.
        (kind, Constant::Int(s1, x), Constant::Int(s2, y)) if s1 == s2 => {
            let (op, sign) = cmp_reading(kind);
            (1..=width_cap(sign))
                .contains(s1)
                .then(|| semantics::cmp(op, sign, *s1, *x, *y))
                .and_then(res)
        }
        (CmpKind::Eq, Constant::Field(x), Constant::Field(y)) => res(x == y),

        // Width-mismatched, mixed-kind, and non-scalar comparisons do not fold
        (
            CmpKind::Eq | CmpKind::ULt | CmpKind::SLt,
            Constant::Int(..) | Constant::Field(_) | Constant::FnPtr(_) | Constant::Blob(_),
            Constant::Int(..) | Constant::Field(_) | Constant::FnPtr(_) | Constant::Blob(_),
        ) => None,
    }
}

/// How a comparison reads its operands, in the model's vocabulary.
///
/// Spelled out rather than defaulted so that a new [`CmpKind`] is a compile error here instead of
/// silently acquiring whichever reading the fallback happened to name.
fn cmp_reading(kind: CmpKind) -> (CmpOp, Sign) {
    match kind {
        // `Eq` compares the patterns themselves, so its reading is inert. `Unsigned` is the one
        // that leaves them alone, and it keeps `Eq` out of the signed width cap it need not obey.
        CmpKind::Eq => (CmpOp::Eq, Sign::Unsigned),
        CmpKind::ULt => (CmpOp::Lt, Sign::Unsigned),
        CmpKind::SLt => (CmpOp::Lt, Sign::Signed),
    }
}

/// Folds a constant cast operation.
///
/// HLSSA casts are raw-bits conversions (sign extension is the separate `SExt` op). Integers
/// zero-extend into fields, fields truncate to their low bits, and integer-to-integer casts
/// zero-extend or truncate.
pub(crate) fn eval_cast(target: &CastTarget, v: &Constant, field: FieldConfig) -> Option<Constant> {
    match target {
        CastTarget::Nop => Some(v.clone()),
        CastTarget::WitnessOf
        | CastTarget::ArrayToSlice
        | CastTarget::ValueOf
        | CastTarget::Map(_) => None,
        CastTarget::Field => match v {
            // FIELD-ASSUMPTION: L4-eval
            Constant::Int(_, x) => Some(Constant::Field(field.constant(*x))),
            Constant::Field(_) => Some(v.clone()),
            Constant::FnPtr(_) | Constant::Blob(_) => None,
        },
        CastTarget::Int(n) => int_cast_bits(v, *n).map(|bits| Constant::Int(*n, bits)),
    }
}

/// Extracts the low `n` bits of a constant's value and returns them as a raw u128 magnitude, or
/// `None` if the constant is not numeric.
fn int_cast_bits(v: &Constant, n: usize) -> Option<u128> {
    if !(1..=MAX_SUPPORTED_UNSIGNED_BITS).contains(&n) {
        return None;
    }
    match v {
        Constant::Int(_, x) => Some(semantics::cast_int(*x, n)),
        // FIELD-ASSUMPTION: L4-decompose
        Constant::Field(f) => {
            let limbs = f.into_bigint().0;
            Some(semantics::field_limbs_to_int(&limbs, n))
        }
        Constant::FnPtr(_) | Constant::Blob(_) => None,
    }
}

/// Folds a constant sign extension operation.
pub(crate) fn eval_sext(v: &Constant, from_bits: usize, to_bits: usize) -> Option<Constant> {
    if from_bits == 0 || from_bits > to_bits || to_bits > MAX_SUPPORTED_UNSIGNED_BITS {
        return None;
    }
    match v {
        Constant::Int(_, x) => Some(Constant::Int(
            to_bits,
            semantics::sign_extend(*x, from_bits, to_bits),
        )),
        Constant::Field(_) | Constant::FnPtr(_) | Constant::Blob(_) => None,
    }
}

/// Folds a constant `BitRange` operation.
///
/// `BitRange` keeps the source type (it is the IR's truncation primitive), so only the payload
/// bits change and a field source folds to a field.
///
/// A **field** source is read through [`semantics::field_limbs_to_int`], the same canonical LE
/// decomposition that `int_cast_bits` uses for the `Field -> Int` cast, and so is bounded
/// by what that can express: a window reaching past bit [`MAX_SUPPORTED_UNSIGNED_BITS`] declines
/// rather than answering the low bits of one that does fit.
///
/// [`semantics::bit_range`] is itself total and answers `0` for an offset past the width. Declining
/// there instead is this analysis keeping out of minting a constant for a shape nothing builds, on
/// the same terms as every other refusal here.
pub(crate) fn eval_bit_range(
    v: &Constant,
    offset: usize,
    width: usize,
    field: FieldConfig,
) -> Option<Constant> {
    if offset >= MAX_SUPPORTED_UNSIGNED_BITS {
        return None;
    }
    match v {
        Constant::Int(s, x) => Some(Constant::Int(*s, semantics::bit_range(*x, offset, width))),

        // FIELD-ASSUMPTION: L4-decompose
        Constant::Field(f) => {
            let read = offset.checked_add(width)?;
            if !(1..=MAX_SUPPORTED_UNSIGNED_BITS).contains(&read) {
                return None;
            }
            let limbs = f.into_bigint().0;
            let low = semantics::field_limbs_to_int(&limbs, read);
            let extracted = semantics::bit_range(low, offset, width);
            Some(Constant::Field(field.constant(extracted)))
        }

        Constant::FnPtr(_) | Constant::Blob(_) => None,
    }
}

/// Folds a constant binary negation.
pub(crate) fn eval_not(v: &Constant) -> Option<Constant> {
    match v {
        Constant::Int(s, x) => Some(Constant::Int(*s, semantics::not(*x, *s))),
        Constant::Field(_) | Constant::FnPtr(_) | Constant::Blob(_) => None,
    }
}

// AGGREGATE CONSTANT EVALUATION
// ================================================================================================

/// The maximum element count of an aggregate the analysis will materialise as a constant.
///
/// Aggregate folding keeps the whole `Vec<Constant>` in the lattice, so an unbounded `MkRepeated`
/// count (or a very long `MkSeq` / `SlicePush`) could blow up memory for no analysis benefit.
/// Constant lookup tables of interest are far smaller than this, so the cap only rejects
/// pathological sizes — which stay `Bottom`, hence sound.
const AGGREGATE_FOLD_CAP: usize = 1 << 12;

/// Reads a constant integer index as a `usize`, or `None` if it is non-integer or too large.
fn const_index(index: &Constant) -> Option<usize> {
    match index {
        Constant::Int(_, x) => usize::try_from(*x).ok(),
        Constant::Field(_) | Constant::FnPtr(_) | Constant::Blob(_) => None,
    }
}

/// Folds an `ArrayGet`: projects element `index` out of a constant aggregate.
///
/// `None` (→ `Bottom`) when the array is not an aggregate constant or the index is out of bounds —
/// an out-of-bounds constant index is an erroneous program, so refusing the fold is sound.
pub(crate) fn eval_array_get(array: &Constant, index: &Constant) -> Option<Constant> {
    let Constant::Blob(blob) = array else {
        return None;
    };
    blob.elements.get(const_index(index)?).cloned()
}

/// Folds an `ArraySet`: a constant aggregate with element `index` replaced by `value`.
pub(crate) fn eval_array_set(
    array: Constant,
    index: &Constant,
    value: Constant,
) -> Option<Constant> {
    let Constant::Blob(mut blob) = array else {
        return None;
    };

    let idx = const_index(index)?;
    if idx >= blob.elements.len() {
        return None;
    }

    blob.elements[idx] = value;
    Some(Constant::Blob(blob))
}

/// Folds a `SliceLen`: the element count of a constant aggregate.
pub(crate) fn eval_slice_len(slice: &Constant) -> Option<Constant> {
    let Constant::Blob(blob) = slice else {
        return None;
    };

    // The result is always a u32 according to the type system.
    Some(Constant::Int(32, blob.len() as u128))
}

/// Folds a `SlicePush`: a constant aggregate extended by `values` at the front or back.
///
/// `Front` prepends the pushed values (in order) before the original elements; `Back` appends them
/// after — matching the backends' slice-push semantics.
pub(crate) fn eval_slice_push(
    dir: SliceOpDir,
    slice: Constant,
    values: Vec<Constant>,
) -> Option<Constant> {
    let Constant::Blob(blob) = slice else {
        return None;
    };
    if blob.len() + values.len() > AGGREGATE_FOLD_CAP {
        return None;
    }
    let elements: Vec<Constant> = match dir {
        SliceOpDir::Front => values.into_iter().chain(blob.elements).collect(),
        SliceOpDir::Back => blob.elements.into_iter().chain(values).collect(),
    };
    Some(Constant::Blob(Blob::new(blob.elem_type, elements)))
}

/// Folds a `SlicePop`: a non-empty constant aggregate split into the shrunk slice and the popped
/// element. A statically empty aggregate is an erroneous program.
pub(crate) fn eval_slice_pop(dir: SliceOpDir, slice: Constant) -> Option<(Constant, Constant)> {
    let Constant::Blob(mut blob) = slice else {
        return None;
    };
    let elem = match dir {
        SliceOpDir::Back => blob.elements.pop()?,
        SliceOpDir::Front => {
            if blob.is_empty() {
                return None;
            }
            blob.elements.remove(0)
        }
    };
    Some((Constant::Blob(blob), elem))
}

/// Folds a `SliceInsert`: a constant aggregate with `value` inserted at a constant `index`.
/// An index larger than `len` is erroneous and refuses the fold.
pub(crate) fn eval_slice_insert(
    slice: Constant,
    index: &Constant,
    value: Constant,
) -> Option<Constant> {
    let Constant::Blob(mut blob) = slice else {
        return None;
    };
    let idx = const_index(index)?;
    if idx > blob.len() || blob.len() + 1 > AGGREGATE_FOLD_CAP {
        return None;
    }
    blob.elements.insert(idx, value);
    Some(Constant::Blob(blob))
}

/// Folds a `SliceRemove`: a constant aggregate split into the slice without element `index` and
/// the removed element.
pub(crate) fn eval_slice_remove(slice: Constant, index: &Constant) -> Option<(Constant, Constant)> {
    let Constant::Blob(mut blob) = slice else {
        return None;
    };
    let idx = const_index(index)?;
    // Also covers the empty aggregate: every index is out of bounds when `len` is 0.
    if idx >= blob.len() {
        return None;
    }
    let elem = blob.elements.remove(idx);
    Some((Constant::Blob(blob), elem))
}

/// Folds a `MkSeq`: an aggregate constant from constant elements.
pub(crate) fn eval_mk_seq(elem_type: &Type, elems: Vec<Constant>) -> Option<Constant> {
    if elems.len() > AGGREGATE_FOLD_CAP {
        return None;
    }
    Some(Constant::Blob(Blob::new(elem_type.clone(), elems)))
}

/// Folds a `MkRepeated`: an aggregate constant of `count` copies of a constant element.
pub(crate) fn eval_mk_repeated(
    elem_type: &Type,
    element: &Constant,
    count: usize,
) -> Option<Constant> {
    if count > AGGREGATE_FOLD_CAP {
        return None;
    }
    Some(Constant::Blob(Blob::new(
        elem_type.clone(),
        vec![element.clone(); count],
    )))
}

#[cfg(test)]
mod tests {
    use mavros_int_semantics::{Outcome, corners, decode_signed, encode_signed};

    use super::*;

    /// An `i8` constant, written as the value it denotes rather than its raw bits.
    fn i8c(v: i128) -> Constant {
        Constant::Int(8, encode_signed(8, v))
    }

    /// Every arithmetic group, so a new one is a compile error here rather than a silent gap in
    /// the sweep below.
    const ALL_GROUPS: [ArithGroup; 10] = [
        ArithGroup::Add,
        ArithGroup::Sub,
        ArithGroup::Mul,
        ArithGroup::Div,
        ArithGroup::Rem,
        ArithGroup::Shl,
        ArithGroup::Shr,
        ArithGroup::And,
        ArithGroup::Or,
        ArithGroup::Xor,
    ];

    /// The refinement relation this analysis holds with respect to the reference model.
    ///
    /// Folding runs long before the guard IR exists, so declining is always allowed. However, the
    /// following are not:
    ///
    /// 1. Answering a value the model does not, as this would result in a quiet miscompile.
    /// 2. Folding an input the model **rejects**, as Noir turns those into a runtime constrain
    ///    failure, so folding one deletes a rejection the program was required to have.
    ///
    /// The counter at the end is what stops the whole sweep passing vacuously: an implementation
    /// that declined everything would satisfy both rules and optimize nothing.
    #[test]
    fn folding_refines_the_reference_model() {
        let field = FieldConfig::bn254();
        let mut folds = 0usize;

        for group in ALL_GROUPS {
            for signed in [false, true] {
                let kind = BinaryArithOpKind::with_sign(group, signed);
                let sign = kind.sign();

                for &bits in corners::widths_for(sign.is_signed()) {
                    for &x in &corners::values(bits) {
                        for &y in &corners::values(bits) {
                            let want = semantics::eval(group.into(), sign, bits, x, bits, y);
                            let got = eval_binary(
                                kind,
                                &Constant::Int(bits, x),
                                &Constant::Int(bits, y),
                                field,
                            );

                            let ctx = format!("{kind:?} {bits} {x} {y}");
                            match (want, got) {
                                (_, None) => {}
                                (Outcome::Rejected(why), Some(folded)) => panic!(
                                    "{ctx}: folded to {folded:?} an input the model rejects ({why:?}), deleting a rejection Noir requires"
                                ),
                                (Outcome::Value(v), Some(Constant::Int(s, raw))) => {
                                    assert_eq!((s, raw), (bits, v), "{ctx}: wrong fold");
                                    folds += 1;
                                }
                                (Outcome::Value(_), Some(other)) => {
                                    panic!("{ctx}: folded an integer pair to {other:?}")
                                }
                            }
                        }
                    }

                    // A mixed-width pair has no fold at all, shifts included: it is IR that
                    // `assert_int_arith_widths` would panic on, so there is no answer to give. It
                    // is checked here rather than left implicit because the model _would_ answer
                    // for a shift, and following it there is exactly the mistake.
                    let other = if bits == 8 { 16 } else { 8 };
                    assert!(
                        eval_binary(
                            kind,
                            &Constant::Int(bits, 1),
                            &Constant::Int(other, 1),
                            field
                        )
                        .is_none(),
                        "{kind:?} folded a {bits}-by-{other} pair"
                    );
                }
            }
        }

        assert!(
            folds > 10_000,
            "only {folds} folds: the sweep is passing vacuously"
        );
    }

    /// Fold two `i8` constants, decoding the result back to the value it denotes.
    /// `None` means the fold was refused.
    fn fold(kind: BinaryArithOpKind, a: i128, b: i128) -> Option<i128> {
        match eval_binary(kind, &i8c(a), &i8c(b), FieldConfig::bn254()) {
            Some(Constant::Int(s, raw)) => Some(decode_signed(s, raw)),
            Some(other) => panic!("expected a signed constant, got {other:?}"),
            None => None,
        }
    }

    #[test]
    fn signed_shr_sign_fills() {
        use BinaryArithOpKind::SShr;
        assert_eq!(fold(SShr, -8, 1), Some(-4));
        assert_eq!(fold(SShr, -8, 2), Some(-2));
        assert_eq!(fold(SShr, -8, 3), Some(-1));
        // Sign-fill saturates at -1 however far it goes, within range.
        assert_eq!(fold(SShr, -8, 7), Some(-1));
        // -1 is all-ones, so it is a fixed point.
        assert_eq!(fold(SShr, -1, 5), Some(-1));
        // Non-negative values behave like a logical shift.
        assert_eq!(fold(SShr, 8, 1), Some(4));
        assert_eq!(fold(SShr, 0, 5), Some(0));
        // Rounds toward negative infinity, unlike signed division which truncates
        // toward zero: -7 >> 1 is -4 but -7 / 2 is -3.
        assert_eq!(fold(SShr, -7, 1), Some(-4));
        assert_eq!(fold(BinaryArithOpKind::SDiv, -7, 2), Some(-3));
    }

    #[test]
    fn shifts_refuse_out_of_range_amounts() {
        use BinaryArithOpKind::{SShl, SShr};
        // At or past the width.
        assert_eq!(fold(SShr, -8, 8), None);
        assert_eq!(fold(SShl, 1, 8), None);
        // A negative amount. This is the case that matters: as raw bits `-1` is
        // 0xFF, which would read as a shift by 255 if it were not decoded, and
        // masking that to the width would produce a plausible-looking shift by 7.
        assert_eq!(fold(SShr, -8, -1), None);
        assert_eq!(fold(SShl, 1, -1), None);
    }

    #[test]
    fn signed_shl_wraps_rather_than_refusing() {
        use BinaryArithOpKind::SShl;
        assert_eq!(fold(SShl, -8, 1), Some(-16));
        assert_eq!(fold(SShl, 1, 6), Some(64));
        // 1 << 7 is 128, one past `i8::MAX`, so it wraps to `i8::MIN`. Two positive operands
        // producing a negative result is the case a `fits_signed` gate used to refuse.
        assert_eq!(fold(SShl, 1, 7), Some(-128));
        assert_eq!(fold(SShl, 2, 6), Some(-128));
        // Everything shifted out: `-16` is 0xF0, and 0xF0 << 4 keeps nothing.
        assert_eq!(fold(SShl, -16, 4), Some(0));
        assert_eq!(fold(SShl, -16, 3), Some(-128));
        // These two are pinned against Noir's own execution by `noir_tests/signed_shift`
        // (`e << 4 == 0` and `e << 3 == -128` for `e: i8 = -16`).
    }

    #[test]
    fn unsigned_shl_wraps_rather_than_refusing() {
        use BinaryArithOpKind::UShl;
        let fold_u8 = |a: u128, b: u128| {
            eval_binary(
                UShl,
                &Constant::Int(8, a),
                &Constant::Int(8, b),
                FieldConfig::bn254(),
            )
        };
        assert_eq!(fold_u8(40, 1), Some(Constant::Int(8, 80)));
        // 40 << 3 is 320, which keeps only its low eight bits.
        assert_eq!(fold_u8(40, 3), Some(Constant::Int(8, 64)));
        assert_eq!(fold_u8(255, 7), Some(Constant::Int(8, 128)));
        // The amount is still a hard refusal.
        assert_eq!(fold_u8(1, 8), None);
    }

    #[test]
    fn int_min_over_minus_one_never_folds() {
        use BinaryArithOpKind::{SDiv, SRem};
        // The quotient overflows, and the remainder is defined in terms of it —
        // so neither may fold, even though the remainder itself would be 0.
        assert_eq!(fold(SDiv, -128, -1), None);
        assert_eq!(fold(SRem, -128, -1), None);
        // The same operands are fine anywhere off that exact pair.
        assert_eq!(fold(SDiv, -127, -1), Some(127));
        assert_eq!(fold(SRem, -127, -1), Some(0));
        assert_eq!(fold(SDiv, -128, 1), Some(-128));
        assert_eq!(fold(SRem, -128, 2), Some(0));
    }

    #[test]
    fn division_by_zero_never_folds() {
        use BinaryArithOpKind::{SDiv, SRem};
        assert_eq!(fold(SDiv, 5, 0), None);
        assert_eq!(fold(SRem, 5, 0), None);
        assert_eq!(fold(SDiv, -5, 0), None);
        assert_eq!(fold(SRem, -5, 0), None);
    }
    #[test]
    fn one_pair_of_patterns_folds_the_way_the_operation_says() {
        // A `Constant::Int` is a bit pattern with no reading attached, so the _only_ thing that can
        // say how to fold a pair of them is the opcode. This pins that: identical operands, two
        // opcodes, two different correct answers. It is also what makes the collapse safe -- while
        // the tag existed, this pair was "mixed" and did not fold at all, which cost the constant
        // propagation that collapses whole chains.
        let f = FieldConfig::bn254();
        let pair = |kind| eval_binary(kind, &Constant::Int(8, 0xFB), &Constant::Int(8, 0x02), f);
        // 0xFB is -5 read as two's complement and 251 read as a magnitude.
        // -5 / 2 == -2 (0xFE); 251 / 2 == 125 (0x7D).
        assert_eq!(pair(BinaryArithOpKind::SDiv), Some(Constant::Int(8, 0xFE)));
        assert_eq!(pair(BinaryArithOpKind::UDiv), Some(Constant::Int(8, 0x7D)));

        // The same for a comparison: -5 < 2 but 251 > 2.
        let cmp = |kind| eval_cmp(kind, &Constant::Int(8, 0xFB), &Constant::Int(8, 0x02));
        assert_eq!(cmp(CmpKind::SLt), Some(Constant::Int(1, 1)));
        assert_eq!(cmp(CmpKind::ULt), Some(Constant::Int(1, 0)));
    }
}
