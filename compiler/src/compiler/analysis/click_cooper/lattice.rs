//! The constant-propagation lattice element and its constant-evaluation transfer functions.

use std::sync::{Arc, OnceLock};

use ark_ff::{PrimeField, Zero};

use crate::compiler::{
    Field,
    ssa::hlssa::{BinaryArithOpKind, CastTarget, CmpKind, Constant, MAX_SUPPORTED_SIGNED_BITS},
    util::{bit_mask, decode_signed, encode_signed, fits_signed},
};

// CONSTNESS
// ================================================================================================

/// A value's *constness*: where it sits in the constant-propagation lattice.
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
        Constant::U(1, 0) => Some(false),
        Constant::U(1, 1) => Some(true),
        Constant::U(..)
        | Constant::I(..)
        | Constant::Field(_)
        | Constant::FnPtr(_)
        | Constant::Blob(_) => None,
    }
}

pub(crate) fn bool_constness(value: bool) -> Constness {
    Constness::Const(bool_constant(value))
}

pub(crate) fn bool_constant(value: bool) -> Arc<Constant> {
    static FALSE: OnceLock<Arc<Constant>> = OnceLock::new();
    static TRUE: OnceLock<Arc<Constant>> = OnceLock::new();
    let slot = if value { &TRUE } else { &FALSE };
    slot.get_or_init(|| Arc::new(Constant::U(1, value as u128)))
        .clone()
}

/// Fold a binary arithmetic op.
///
/// Integer results must fit the operand width: an overflowing pure op is an erroneous evaluation
/// with a backend-specific residue, so an overflowing fold is refused rather than guessed at.
pub(crate) fn eval_binary(kind: BinaryArithOpKind, a: &Constant, b: &Constant) -> Option<Constant> {
    use BinaryArithOpKind::*;
    match (a, b) {
        (Constant::U(s1, x), Constant::U(s2, y)) => {
            match kind {
                // Shifts are the only ops with legitimately mixed operand widths (the amount is
                // typically a narrow integer), but the type analysis types the result as
                // `U(max(s1, s2))`. A fold to a `U(s1)` constant is therefore only width-preserving
                // when the amount is no wider than the value; refuse the degenerate wider-amount
                // case rather than silently changing the result's width.
                Shl | Shr => {
                    if s2 > s1 {
                        return None;
                    }
                }
                Add | Sub | Mul | Div | Mod | And | Or | Xor => {
                    if s1 != s2 {
                        return None;
                    }
                }
            }

            let s = *s1;
            let v = match kind {
                Add => x.checked_add(*y)?,
                Sub => x.checked_sub(*y)?,
                Mul => x.checked_mul(*y)?,
                Div => {
                    if *y == 0 {
                        return None;
                    }
                    x / y
                }
                Mod => {
                    if *y == 0 {
                        return None;
                    }
                    x % y
                }
                And => x & y,
                Or => x | y,
                Xor => x ^ y,
                Shl | Shr => {
                    if *y >= s as u128 {
                        return None;
                    }
                    match kind {
                        Shl => x.checked_shl(*y as u32)?,
                        Shr => x >> (*y as u32),
                        Add | Sub | Mul | Div | Mod | And | Or | Xor => unreachable!(),
                    }
                }
            };

            if v > bit_mask(s) {
                return None;
            }
            Some(Constant::U(s, v))
        }
        (Constant::I(s1, x), Constant::I(s2, y)) => {
            if s1 != s2 {
                return None;
            }
            let s = *s1;
            if s == 0 || s > MAX_SUPPORTED_SIGNED_BITS {
                return None;
            }
            match kind {
                And => Some(Constant::I(s, x & y)),
                Or => Some(Constant::I(s, x | y)),
                Xor => Some(Constant::I(s, x ^ y)),
                Add | Sub | Mul | Div | Mod => {
                    let xa = decode_signed(s, *x);
                    let ya = decode_signed(s, *y);
                    let v = match kind {
                        Add => xa.checked_add(ya)?,
                        Sub => xa.checked_sub(ya)?,
                        Mul => xa.checked_mul(ya)?,
                        Div => {
                            if ya == 0 {
                                return None;
                            }
                            xa.checked_div(ya)?
                        }
                        Mod => {
                            if ya == 0 {
                                return None;
                            }
                            xa.checked_rem(ya)?
                        }
                        And | Or | Xor | Shl | Shr => unreachable!(),
                    };
                    if !fits_signed(s, v) {
                        return None;
                    }
                    Some(Constant::I(s, encode_signed(s, v)))
                }
                Shl | Shr => None,
            }
        }
        (Constant::Field(x), Constant::Field(y)) => match kind {
            Add => Some(Constant::Field(*x + *y)),
            Sub => Some(Constant::Field(*x - *y)),
            Mul => Some(Constant::Field(*x * *y)),
            Div => {
                if y.is_zero() {
                    None
                } else {
                    Some(Constant::Field(*x / *y))
                }
            }
            Mod | And | Or | Xor | Shl | Shr => None,
        },

        // Mixed-kind pairs and non-scalar constants do not fold.
        (
            Constant::U(..)
            | Constant::I(..)
            | Constant::Field(_)
            | Constant::FnPtr(_)
            | Constant::Blob(_),
            Constant::U(..)
            | Constant::I(..)
            | Constant::Field(_)
            | Constant::FnPtr(_)
            | Constant::Blob(_),
        ) => None,
    }
}

/// Folds a constant comparison operation.
pub(crate) fn eval_cmp(kind: CmpKind, a: &Constant, b: &Constant) -> Option<Constant> {
    let res = |v: bool| Some(Constant::U(1, v as u128));
    match (kind, a, b) {
        (CmpKind::Eq, Constant::U(s1, x), Constant::U(s2, y)) if s1 == s2 => res(x == y),
        (CmpKind::Eq, Constant::I(s1, x), Constant::I(s2, y)) if s1 == s2 => res(x == y),
        (CmpKind::Eq, Constant::Field(x), Constant::Field(y)) => res(x == y),
        (CmpKind::Lt, Constant::U(s1, x), Constant::U(s2, y)) if s1 == s2 => res(x < y),
        (CmpKind::Lt, Constant::I(s1, x), Constant::I(s2, y))
            if s1 == s2 && *s1 >= 1 && *s1 <= MAX_SUPPORTED_SIGNED_BITS =>
        {
            res(decode_signed(*s1, *x) < decode_signed(*s1, *y))
        }

        // Width-mismatched, mixed-kind, and non-scalar comparisons do not fold
        (
            CmpKind::Eq | CmpKind::Lt,
            Constant::U(..)
            | Constant::I(..)
            | Constant::Field(_)
            | Constant::FnPtr(_)
            | Constant::Blob(_),
            Constant::U(..)
            | Constant::I(..)
            | Constant::Field(_)
            | Constant::FnPtr(_)
            | Constant::Blob(_),
        ) => None,
    }
}

/// Folds a field multiplication operation (the transfer for `MulConst`).
///
/// This is deliberately *not* `eval_binary(Mul, ..)`: `MulConst` is field-domain, where
/// multiplication is modular and total, so it must not inherit the integer width/overflow rules
/// `eval_binary` applies to `U`/`I` operands. Non-field operands therefore do not fold.
pub(crate) fn eval_field_mul(a: &Constant, b: &Constant) -> Option<Constant> {
    match (a, b) {
        (Constant::Field(x), Constant::Field(y)) => Some(Constant::Field(*x * *y)),
        (
            Constant::U(..)
            | Constant::I(..)
            | Constant::Field(_)
            | Constant::FnPtr(_)
            | Constant::Blob(_),
            Constant::U(..)
            | Constant::I(..)
            | Constant::Field(_)
            | Constant::FnPtr(_)
            | Constant::Blob(_),
        ) => None,
    }
}

/// Folds a constant cast operation.
///
/// HLSSA casts are raw-bits conversions (sign extension is the separate `SExt` op). Integers
/// zero-extend into fields, fields truncate to their low bits, and integer-to-integer casts
/// zero-extend or truncate.
pub(crate) fn eval_cast(target: &CastTarget, v: &Constant) -> Option<Constant> {
    match target {
        CastTarget::Nop => Some(v.clone()),
        CastTarget::WitnessOf
        | CastTarget::ArrayToSlice
        | CastTarget::ValueOf
        | CastTarget::Map(_) => None,
        CastTarget::Field => match v {
            Constant::U(_, x) | Constant::I(_, x) => Some(Constant::Field(Field::from(*x))),
            Constant::Field(_) => Some(v.clone()),
            Constant::FnPtr(_) | Constant::Blob(_) => None,
        },
        CastTarget::U(n) => int_cast_bits(v, *n).map(|bits| Constant::U(*n, bits)),
        CastTarget::I(n) => {
            if *n > MAX_SUPPORTED_SIGNED_BITS {
                return None;
            }
            int_cast_bits(v, *n).map(|bits| Constant::I(*n, bits))
        }
    }
}

/// Extracts the low `n` bits of a constant's value and returns them as a raw u128 magnitude, or
/// `None` if the constant is not numeric.
fn int_cast_bits(v: &Constant, n: usize) -> Option<u128> {
    let mask = bit_mask(n);
    match v {
        Constant::U(_, x) | Constant::I(_, x) => Some(x & mask),
        Constant::Field(f) => {
            let limbs = f.into_bigint().0;
            let low = (limbs[0] as u128) | ((limbs[1] as u128) << 64);
            Some(low & mask)
        }
        Constant::FnPtr(_) | Constant::Blob(_) => None,
    }
}

/// Folds a constant sign extension operation.
pub(crate) fn eval_sext(v: &Constant, from_bits: usize, to_bits: usize) -> Option<Constant> {
    if from_bits == 0 || from_bits > to_bits || to_bits > 128 {
        return None;
    }
    let ext = |x: u128| {
        if (x >> (from_bits - 1)) & 1 == 1 {
            x | (bit_mask(to_bits) & !bit_mask(from_bits))
        } else {
            x
        }
    };
    match v {
        Constant::U(_, x) => Some(Constant::U(to_bits, ext(*x))),
        Constant::I(_, x) => Some(Constant::I(to_bits, ext(*x))),
        Constant::Field(_) | Constant::FnPtr(_) | Constant::Blob(_) => None,
    }
}

/// Folds a constant `BitRange` operation.
///
/// `BitRange` keeps the source type (it is the IR's truncation primitive), so only the payload
/// bits change.
pub(crate) fn eval_bit_range(v: &Constant, offset: usize, width: usize) -> Option<Constant> {
    if offset >= 128 {
        return None;
    }
    match v {
        Constant::U(s, x) => Some(Constant::U(*s, (x >> offset) & bit_mask(width))),
        Constant::I(s, x) => Some(Constant::I(*s, (x >> offset) & bit_mask(width))),
        Constant::Field(_) | Constant::FnPtr(_) | Constant::Blob(_) => None,
    }
}

/// Folds a constant binary negation.
pub(crate) fn eval_not(v: &Constant) -> Option<Constant> {
    match v {
        Constant::U(s, x) => Some(Constant::U(*s, !x & bit_mask(*s))),
        Constant::I(s, x) => Some(Constant::I(*s, !x & bit_mask(*s))),
        Constant::Field(_) | Constant::FnPtr(_) | Constant::Blob(_) => None,
    }
}
