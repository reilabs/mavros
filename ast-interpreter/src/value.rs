use acvm::{AcirField, FieldElement};
use num_bigint::{BigInt, Sign};
use num_traits::{One, Zero};

use noirc_frontend::monomorphization::ast::FuncId;

use super::error::InterpretError;

/// A runtime value produced while interpreting the monomorphized AST.
///
/// Integers carry their own width + signedness and a `BigInt` of the canonical mathematical
/// value, so beyond-field magnitudes (a `u64` ≥ the source field's modulus) are represented
/// faithfully — the property the SSA interpreter could not provide. `Field` values use the
/// compiled-in [`FieldElement`] (bn254 by default, Goldilocks under `--features goldilocks`),
/// so field arithmetic is correct for whichever field the frontend was built with.
#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    Field(FieldElement),
    Int(IntValue),
    Bool(bool),
    Unit,
    Array(Vec<Value>),
    Tuple(Vec<Value>),
    Str(String),
    Function(FuncId),
}

/// A machine integer: width, signedness, and the canonical value.
///
/// Invariant: `value` is the mathematical integer in the type's range — unsigned in
/// `[0, 2^bits)`, signed in `[-2^(bits-1), 2^(bits-1))`.
#[derive(Clone, Debug, PartialEq)]
pub struct IntValue {
    pub signed: bool,
    pub bits: u8,
    pub value: BigInt,
}

fn pow2(bits: u8) -> BigInt {
    BigInt::one() << bits as usize
}

/// The non-negative integer value of a field element. Unlike `FieldElement::to_u128`, this does
/// not panic for values `>= 2^128` (bn254 elements are up to ~254 bits).
pub fn field_to_bigint(field: &FieldElement) -> BigInt {
    BigInt::from_bytes_be(Sign::Plus, &field.to_be_bytes())
}

/// Reduce `raw` into the canonical two's-complement representative for `(signed, bits)`.
pub fn wrap(signed: bool, bits: u8, raw: BigInt) -> BigInt {
    let modulus = pow2(bits);
    let mut u = raw % &modulus;
    if u.sign() == Sign::Minus {
        u += &modulus;
    }
    if signed && u >= pow2(bits - 1) {
        u -= &modulus;
    }
    u
}

impl IntValue {
    /// Construct by *wrapping* `raw` into the type's range (two's complement) — the truncating
    /// constructor used for casts, `!`, and wrapping shifts. For checked arithmetic, where an
    /// out-of-range value must be an overflow error instead, use [`IntValue::checked`].
    pub fn canonical(signed: bool, bits: u8, raw: BigInt) -> Self {
        IntValue {
            signed,
            bits,
            value: wrap(signed, bits, raw),
        }
    }

    /// Inclusive `[min, max]` range for the type.
    pub fn range(signed: bool, bits: u8) -> (BigInt, BigInt) {
        if signed {
            let half = pow2(bits - 1);
            (-half.clone(), half - BigInt::one())
        } else {
            (BigInt::zero(), pow2(bits) - BigInt::one())
        }
    }

    /// Construct from a result of checked arithmetic; error if it does not fit the type.
    pub fn checked(signed: bool, bits: u8, raw: BigInt, op: &str) -> Result<Self, InterpretError> {
        let (min, max) = Self::range(signed, bits);
        if raw < min || raw > max {
            return Err(InterpretError::Overflow(op.to_string()));
        }
        Ok(IntValue {
            signed,
            bits,
            value: raw,
        })
    }

    /// The value as a non-negative integer in `[0, 2^bits)` (two's-complement bit pattern),
    /// used for bitwise ops and conversion to a field element.
    pub fn unsigned_repr(&self) -> BigInt {
        if self.value.sign() == Sign::Minus {
            &self.value + pow2(self.bits)
        } else {
            self.value.clone()
        }
    }

    /// Encode as a field element (the value's bit pattern reduced into the field).
    pub fn to_field(&self) -> FieldElement {
        let (_, bytes) = self.unsigned_repr().to_bytes_be();
        FieldElement::from_be_bytes_reduce(&bytes)
    }
}

impl Value {
    pub fn as_bool(&self) -> Result<bool, InterpretError> {
        match self {
            Value::Bool(b) => Ok(*b),
            other => Err(InterpretError::Type(format!(
                "expected bool, got {other:?}"
            ))),
        }
    }

    pub fn as_int(&self) -> Result<&IntValue, InterpretError> {
        match self {
            Value::Int(i) => Ok(i),
            other => Err(InterpretError::Type(format!(
                "expected integer, got {other:?}"
            ))),
        }
    }

    /// Coerce an integer value to a `usize` index.
    pub fn as_index(&self) -> Result<usize, InterpretError> {
        let int = self.as_int()?;
        let repr = int.unsigned_repr();
        let (sign, digits) = repr.to_u64_digits();
        if sign == Sign::Minus || digits.len() > 1 {
            return Err(InterpretError::Type(format!(
                "index out of usize range: {repr}"
            )));
        }
        Ok(digits.first().copied().unwrap_or(0) as usize)
    }
}
