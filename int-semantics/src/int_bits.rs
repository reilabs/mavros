//! [`IntBits`] is a width-sized two's-complement bit pattern, used for storing and evaluating
//! arbitrary-sized integers at compile time. It uses exactly [`IntBits::limbs_for_bits`] limbs with
//! any additional bits set to zero.

use std::fmt;

use num_bigint::BigUint;
use thiserror::Error;

use crate::{CmpOp, MAX_BITS, MAX_SIGNED_BITS, SignedValue, check_widths, mask};

// INTEGER BIT PATTERN
// ================================================================================================

/// A `bits`-wide two's-complement pattern: exactly [`IntBits::limbs_for_bits`] little-endian
/// 64-bit limbs, with every bit at or above `bits` set to zero.
///
/// It has no [`Ord`] implementation because it is a raw bit interpretation and there are two valid
/// readings of ordering for this type. [`IntBits::compare`] should be used to choose a reading.
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct IntBits {
    /// The width of the integer.
    bits: usize,

    /// Exactly `IntBits::limbs_for_bits(bits)` of them, little-endian, with the top limb masked.
    limbs: Box<[u64]>,
}

// CONSTRUCTORS
// ================================================================================================

/// Constructors for the type, all of which perform normalization during construction.
///
/// **None of these caps the width at [`MAX_BITS`]** as that is the _model's_ bound on the
/// operations it will evaluate, not the type's on the patterns it can hold. What does assert it is
/// the field-limb group below — [`IntBits::from_packed_limbs`], [`IntBits::from_field_limbs`] and
/// their companion predicate [`IntBits::field_limbs_fit`] — because those are reading a _field
/// element_, which the model does own an opinion about.
///
/// The **lower** bound is a different matter and is enforced through [`IntBits::limbs_for_bits`]: a
/// width of zero describes no pattern at all, so it panics rather than producing an empty one.
/// Everything that takes a width downstream — [`IntBits::cast`], [`IntBits::bit_range`],
/// [`IntBits::sign_extend`] — inherits that panic, and a caller holding a width it did not choose
/// itself is the one that has to refuse first.
impl IntBits {
    /// Zero at `bits` wide.
    #[must_use]
    pub fn zero(bits: usize) -> Self {
        Self::normalized(bits, vec![0; Self::limbs_for_bits(bits)])
    }

    /// A `bits`-wide pattern carrying `value`, discarding any bits at or above `bits`.
    #[must_use]
    pub fn from_u128(bits: usize, value: u128) -> Self {
        let count = Self::limbs_for_bits(bits);
        let mut limbs = vec![0u64; count];
        limbs[0] = value as u64;
        if count > 1 {
            limbs[1] = (value >> 64) as u64;
        }
        Self::normalized(bits, limbs)
    }

    /// Every bit below `bits` set, and the companion of [`IntBits::is_all_ones`].
    ///
    /// Distinct from [`crate::mask`], which answers the same thing as a _host_ word and so stops
    /// being expressible at a width the host has no corresponding type for.
    #[must_use]
    pub fn all_ones(bits: usize) -> Self {
        Self::normalized(bits, vec![u64::MAX; Self::limbs_for_bits(bits)])
    }

    /// A `bits`-wide pattern from little-endian limbs, zero-extending a slice too short to fill the
    /// width and truncating one too long for it.
    #[must_use]
    pub fn from_limbs(bits: usize, limbs: &[u64]) -> Self {
        let count = Self::limbs_for_bits(bits);
        let mut out = vec![0u64; count];
        let taken = limbs.len().min(count);
        out[..taken].copy_from_slice(&limbs[..taken]);
        Self::normalized(bits, out)
    }
}

// FIELD LIMB INTEROP
// ================================================================================================

/// The width of one limb of a **canonical field representation**.
///
/// This is not a property of any particular field: `ark_ff::BigInt<const N: usize>` is `[u64; N]`,
/// so the limb _count_ `N` varies from field to field while the limb _width_ does not. That is
/// **arkworks'** `BigInt` and not the `num-bigint` one this crate computes with, whose own digit
/// width is `pub(crate)` and irrelevant here: [`IntBits::from_biguint`] reads a magnitude back
/// through `iter_u64_digits`, which is a `u64` view of it however it is stored.
///
/// The claim is therefore about a type this crate deliberately cannot see, so it is held to
/// arkworks one crate up, by a `const` assertion in the compiler's `util.rs`.
pub const FIELD_LIMB_BITS: usize = 64;

/// Reading a field element's limbs as a pattern, which needs no reading of its own: a canonical
/// field representation is an unsigned magnitude, so only the truncation is in question.
impl IntBits {
    /// Recombine little-endian limbs of `limb_bits` each into a `bits`-wide pattern.
    ///
    /// The limb width is a parameter because Mavros has two unrelated kinds of limb, and conflating
    /// them risks a bug: a canonical field representation is always [`FIELD_LIMB_BITS`] wide, while
    /// the witness decompositions in `witness_bitwise` use a width that is _derived from the field
    /// size_. Both are little-endian limb vectors and neither knows the other's width.
    ///
    /// Each limb is masked to `limb_bits` first, so a caller packing a narrow limb into a wider
    /// container need not clear the space above it.
    ///
    /// # Panics
    ///
    /// If `limb_bits` is zero, which describes no representation at all, or if `bits` is outside
    /// the model's range.
    #[must_use]
    pub fn from_packed_limbs(limbs: &[u64], limb_bits: usize, bits: usize) -> Self {
        assert!(
            (1..=MAX_BITS).contains(&bits),
            "pattern width {bits} is outside 1..={MAX_BITS}"
        );
        assert!(limb_bits > 0, "a limb must be at least one bit wide");

        // Limb `i` starts at bit `i * limb_bits`, so the ones from here up contribute nothing below
        // `bits` and the final truncation would discard them anyway.
        let usable = bits.div_ceil(limb_bits);
        let limb_mask = mask(limb_bits);

        let low = limbs
            .iter()
            .take(usable)
            .enumerate()
            .fold(BigUint::ZERO, |acc, (i, &limb)| {
                acc | (BigUint::from(u128::from(limb) & limb_mask) << (i * limb_bits))
            });

        Self::from_biguint(bits, &low)
    }

    /// Read a field element's canonical little-endian limbs as a `bits`-wide pattern.
    ///
    /// The `Field -> Int` cast, and [`IntBits::from_packed_limbs`] at [`FIELD_LIMB_BITS`]. It takes
    /// the low `bits` bits of the **canonical** representation, so a caller holding a **Montgomery
    /// form must convert first**.
    ///
    /// The slice is unsized because the limb count is a property of the field. A slice shorter than
    /// `bits` implies is not an error either: it simply describes a smaller element.
    ///
    /// # Panics
    ///
    /// If `bits` is outside the model's range, as [`IntBits::from_packed_limbs`] does.
    #[must_use]
    pub fn from_field_limbs(limbs: &[u64], bits: usize) -> Self {
        assert!(
            (1..=MAX_BITS).contains(&bits),
            "pattern width {bits} is outside 1..={MAX_BITS}"
        );

        // Not routed through `from_packed_limbs`: at the host limb width the packed limbs already
        // **are** the pattern's own little-endian words, so there's nothing to unpack. The
        // congruence is enforced in tests.
        Self::from_limbs(bits, limbs)
    }

    /// Whether a field element's canonical limbs are entirely inside a `bits`-wide pattern.
    ///
    /// The companion of [`IntBits::from_field_limbs`], which **truncates**: this is how a caller
    /// asks whether the truncation would lose anything, so that it can refuse if needed.
    ///
    /// # Panics
    ///
    /// If `bits` is outside the model's range.
    #[must_use]
    pub fn field_limbs_fit(limbs: &[u64], bits: usize) -> bool {
        assert!(
            (1..=MAX_BITS).contains(&bits),
            "pattern width {bits} is outside 1..={MAX_BITS}"
        );

        // Limbs below `whole` are covered outright. The one at `whole` is covered only up to
        // `spare` bits, and where `spare` is zero that shift asks for the whole limb: at a width
        // that ends on a limb boundary the limb above it is entirely outside.
        let whole = bits / FIELD_LIMB_BITS;
        let spare = bits % FIELD_LIMB_BITS;

        limbs
            .iter()
            .enumerate()
            .all(|(i, &limb)| match i.cmp(&whole) {
                std::cmp::Ordering::Less => true,
                std::cmp::Ordering::Equal => limb >> spare == 0,
                std::cmp::Ordering::Greater => limb == 0,
            })
    }
}

// ACCESSORS
// ================================================================================================

/// Reading a pattern back out.
impl IntBits {
    /// The declared width.
    #[must_use]
    pub fn bits(&self) -> usize {
        self.bits
    }

    /// The [`IntBits::limb_count`] limbs in little-endian order.
    #[must_use]
    pub fn limbs(&self) -> &[u64] {
        &self.limbs
    }

    /// The number of 64-bit limbs a `bits`-wide pattern occupies.
    ///
    /// The host limb is 64 bits and that is fixed by the machine, not by the field: this is the
    /// quantity `int_cell_count` reserves in a VM frame and `for_each_constant_word` emits. The
    /// _field_ limb is a different and narrower thing, derived from the field's own width.
    ///
    /// # Panics
    ///
    /// On a width of zero, which describes no pattern at all.
    #[must_use]
    pub fn limbs_for_bits(bits: usize) -> usize {
        assert!(bits > 0, "an integer pattern is at least one bit wide");
        bits.div_ceil(64)
    }

    /// How many limbs this pattern occupies.
    #[must_use]
    pub fn limb_count(&self) -> usize {
        self.limbs.len()
    }

    /// Whether every bit is zero.
    #[must_use]
    pub fn is_zero(&self) -> bool {
        self.limbs.iter().all(|&limb| limb == 0)
    }

    /// Whether this is the pattern `1`, with only the low bit set.
    #[must_use]
    pub fn is_one(&self) -> bool {
        self.limbs[0] == 1 && self.limbs[1..].iter().all(|&limb| limb == 0)
    }

    /// Whether every bit below the width is set.
    ///
    /// Read as two's complement that is `-1`, but this only says the pattern is saturated at its
    /// own width. It is a property of the limbs alone precisely _because_ the pattern is normalized
    /// — the top limb's own maximum is [`top_limb_mask`].
    #[must_use]
    pub fn is_all_ones(&self) -> bool {
        let top = self.limbs.len() - 1;
        self.limbs[..top].iter().all(|&limb| limb == u64::MAX)
            && self.limbs[top] == top_limb_mask(self.bits)
    }

    /// Bit `index`, counting from the least significant, or [`None`] at or above the width.
    ///
    /// [`None`] rather than `false`, because the two are not the same thing: a pattern has no bit
    /// at or above its own width, and answering `false` there would make an out-of-range read
    /// indistinguishable from a zero bit that is genuinely part of the value.
    #[must_use]
    pub fn bit(&self, index: usize) -> Option<bool> {
        if index >= self.bits {
            return None;
        }
        Some((self.limbs[index / 64] >> (index % 64)) & 1 == 1)
    }
}

// PATTERN OPERATIONS
// ================================================================================================

/// Operations a pattern supports _without_ a reading being named.
///
/// Every one of them is a rearrangement of bits: the answer for a given input is the same whether
/// the operands are meant as signed or unsigned.
impl IntBits {
    /// Bitwise `and` with another pattern of the same width.
    ///
    /// # Panics
    ///
    /// If the widths differ. Two patterns of different widths have no common bit positions to
    /// combine, and every caller here has already checked that they agree.
    #[must_use]
    pub fn and(&self, other: &Self) -> Self {
        self.zip_with(other, |a, b| a & b)
    }

    /// Bitwise `or` with another pattern of the same width.
    ///
    /// # Panics
    ///
    /// As [`IntBits::and`] does, if the widths differ.
    #[must_use]
    pub fn or(&self, other: &Self) -> Self {
        self.zip_with(other, |a, b| a | b)
    }

    /// Bitwise `xor` with another pattern of the same width.
    ///
    /// # Panics
    ///
    /// As [`IntBits::and`] does, if the widths differ.
    #[must_use]
    pub fn xor(&self, other: &Self) -> Self {
        self.zip_with(other, |a, b| a ^ b)
    }

    /// Every bit flipped, held to the width.
    #[must_use]
    pub fn complement(&self) -> Self {
        Self::normalized(self.bits, self.limbs.iter().map(|limb| !limb).collect())
    }

    /// Reinterpret at a new width: truncate when narrowing, zero-extend when widening.
    ///
    /// # Panics
    ///
    /// On a `to_bits` of zero, which describes no pattern to reinterpret as.
    #[must_use]
    pub fn cast(&self, to_bits: usize) -> Self {
        // A cast to the width already held is the identity, and it is the common case: both
        // constant folders spell the `BitRange` transfer as `bit_range(..).cast(self.bits())`.
        if to_bits == self.bits {
            return self.clone();
        }
        Self::from_limbs(to_bits, &self.limbs)
    }

    /// Extract `width` bits starting at `offset`, right-aligned, akin to HLSSA's `BitRange`.
    ///
    /// The truncation primitive: `v.bit_range(0, n)` is the low `n` bits, and a logical `>>` by a
    /// constant amount is `v.bit_range(k, v.bits() - k)`.
    ///
    /// There is no bound on `offset`, because a pattern shifted past its own width is empty where
    /// a host word shifted past its own width is undefined. `width` is bounded below, though, for
    /// the reason [`IntBits::cast`] is: an empty window is not a pattern.
    ///
    /// # Panics
    ///
    /// On a `width` of zero. HLSSA's `BitRange` refuses one too (`analysis::types` types it as an
    /// error), so a folder handed one is looking at IR that does not type-check.
    #[must_use]
    pub fn bit_range(&self, offset: usize, width: usize) -> Self {
        self.shifted_right(offset).cast(width)
    }

    /// The pattern moved `amount` places toward the most significant end, zero-filling behind it.
    ///
    /// Named for the movement rather than for `<<`, because an operation's shift carries additional
    /// rules: at or above the width Noir rejects, and a total backend reduces the amount modulo the
    /// width first. Here an amount at or above the width simply leaves nothing.
    #[must_use]
    pub fn shifted_left(&self, amount: usize) -> Self {
        if amount >= self.bits {
            return Self::zero(self.bits);
        }
        let (whole, part) = (amount / 64, amount % 64);
        let mut out = vec![0u64; self.limbs.len()];
        for (i, &limb) in self.limbs.iter().enumerate() {
            let Some(target) = out.get_mut(i + whole) else {
                break;
            };
            *target |= limb << part;
            // A shift by a whole limb is `limb << 64`, which is undefined rather than zero, so the
            // carry into the next limb is guarded rather than written as `limb >> (64 - part)`.
            if part > 0
                && let Some(next) = out.get_mut(i + whole + 1)
            {
                *next |= limb >> (64 - part);
            }
        }
        Self::normalized(self.bits, out)
    }

    /// The pattern moved `amount` places toward the least significant end, zero-filling behind it.
    #[must_use]
    pub fn shifted_right(&self, amount: usize) -> Self {
        if amount >= self.bits {
            return Self::zero(self.bits);
        }
        let (whole, part) = (amount / 64, amount % 64);
        let mut out = vec![0u64; self.limbs.len()];
        for i in 0..self.limbs.len() {
            let Some(&limb) = self.limbs.get(i + whole) else {
                break;
            };
            out[i] |= limb >> part;
            if part > 0
                && let Some(&next) = self.limbs.get(i + whole + 1)
            {
                out[i] |= next << (64 - part);
            }
        }
        Self::normalized(self.bits, out)
    }

    /// Combine two same-width patterns limb by limb.
    ///
    /// # Panics
    ///
    /// If the widths differ.
    fn zip_with(&self, other: &Self, f: impl Fn(u64, u64) -> u64) -> Self {
        assert_eq!(
            self.bits, other.bits,
            "a bitwise operation needs two patterns of one width"
        );
        Self::normalized(
            self.bits,
            self.limbs
                .iter()
                .zip(other.limbs.iter())
                .map(|(&a, &b)| f(a, b))
                .collect(),
        )
    }
}

// OPERATIONS THAT NAME A READING
// ================================================================================================

/// The operations that name a **reading** of a pattern.
///
/// The block above answers only what is true of the bits (`and`, `cast`, `bit_range`, the shifts)
/// and the type deliberately has no [`Ord`], because two patterns have two orderings and it cannot
/// pick one. Everything here picks a signedness, encoded in its name.
impl IntBits {
    /// Read this pattern as two's complement.
    ///
    /// # Panics
    ///
    /// If the pattern is wider than [`MAX_SIGNED_BITS`].
    #[must_use]
    pub fn to_signed(&self) -> SignedValue {
        let bits = self.bits();
        assert!(
            (1..=MAX_SIGNED_BITS).contains(&bits),
            "a signed reading of a {bits}-bit pattern is outside 1..={MAX_SIGNED_BITS}"
        );
        let magnitude = SignedValue::from(BigUint::from(self));
        if self.bit(self.bits() - 1) == Some(true) {
            magnitude - two_pow(self.bits())
        } else {
            magnitude
        }
    }

    /// Encode a signed value as a `bits`-wide pattern.
    ///
    /// Total for any `v`, however far outside the width it sits: a value that does not fit is
    /// reduced modulo `2^bits`, as required by [`residue`](crate::residue).
    #[must_use]
    pub fn from_signed(bits: usize, v: &SignedValue) -> Self {
        // A [`SignedValue`]'s bitwise operators read a negative value as two's complement extended
        // indefinitely: masking to `bits` takes the low `bits` of that and lands in `0..2^bits`
        // whatever the sign and magnitude of `v`.
        let masked = v & (two_pow(bits) - SignedValue::from(1u8));
        Self::from_biguint(bits, masked.magnitude())
    }

    /// Whether `v` is representable in `bits`-bit two's complement.
    ///
    /// # Panics
    ///
    /// If `bits` is above [`MAX_SIGNED_BITS`].
    #[must_use]
    pub fn fits_signed(bits: usize, v: &SignedValue) -> bool {
        assert!(
            (1..=MAX_SIGNED_BITS).contains(&bits),
            "a signed reading at {bits} bits is outside 1..={MAX_SIGNED_BITS}"
        );
        *v >= Self::signed_min(bits) && *v <= Self::signed_max(bits)
    }

    /// The largest value a `bits`-wide signed pattern represents.
    ///
    /// # Panics
    ///
    /// On a width of zero: `bits - 1` would wrap in a release build and ask for a two-power with
    /// no memory to hold it.
    #[must_use]
    pub fn signed_max(bits: usize) -> SignedValue {
        assert!(bits >= 1, "a signed reading needs at least a sign bit");
        two_pow(bits - 1) - SignedValue::from(1u8)
    }

    /// The smallest value a `bits`-wide signed pattern represents.
    ///
    /// # Panics
    ///
    /// On a width of zero, for the reason [`IntBits::signed_max`] gives.
    #[must_use]
    pub fn signed_min(bits: usize) -> SignedValue {
        assert!(bits >= 1, "a signed reading needs at least a sign bit");
        -two_pow(bits - 1)
    }

    /// Widen from the pattern's width to `to` bits, replicating the sign bit.
    ///
    /// # Panics
    ///
    /// On a narrowing. The width extended _from_ needs no check of its own as it is the receiver's.
    #[must_use]
    pub fn sign_extend(&self, to: usize) -> Self {
        let from = self.bits();
        assert!(to >= from, "cannot sign-extend {from} to {to}");
        let widened = self.cast(to);
        if self.bit(from - 1) == Some(true) {
            // The fill is every bit from `from` up to `to`. At `to == from` the shift moves the
            // ones clean out of the width and leaves nothing, which is the right answer: there is
            // no room above the sign bit to fill.
            widened.or(&Self::all_ones(to).shifted_left(from))
        } else {
            widened
        }
    }

    /// The amount this pattern actually shifts a `operand_bits`-wide value by, where the guard IR
    /// failed to reject it.
    ///
    /// The receiver is the **amount** and the parameter is the width of the value being shifted: an
    /// amount carries a width of its own and it is not the width the reduction is against.
    ///
    /// `amount % operand_bits`: a genuine **modulo at every width**, rather than a mask. At a
    /// power-of-two width the two coincide, but at a width that is not a power of two they differ,
    /// and a mask would be incorrect.
    ///
    /// # Panics
    ///
    /// On an `operand_bits` of zero. The VM's counterpart answers `0` there, because its width
    /// arrives from a frame cell and a zero-width cell holds nothing; this one's arrives from the
    /// operand [`eval`](crate::eval) was handed, whose width check has already held it at one bit
    /// or more.
    #[must_use]
    pub fn reduced_shift_amount(&self, operand_bits: usize) -> u32 {
        assert!(
            operand_bits > 0,
            "a shift amount reduces against a width of at least one bit"
        );
        let reduced = BigUint::from(self) % BigUint::from(operand_bits);
        u32::try_from(&reduced).expect("a value below a supported width fits a u32")
    }

    /// Compare against another pattern of the same width, under the reading `op` names.
    ///
    /// Spelled `compare` rather than `cmp` because [`IntBits`] deliberately has no [`Ord`] for an
    /// inherent `cmp` to shadow, but one answering a `bool` beside the `Ord::cmp` everyone knows
    /// answers an [`Ordering`](std::cmp::Ordering) is a readability trap for no gain.
    ///
    /// # Panics
    ///
    /// As [`eval`](crate::eval) does, on invalid or unequal widths.
    #[must_use]
    pub fn compare(&self, op: CmpOp, rhs: &Self) -> bool {
        check_widths(matches!(op, CmpOp::SLt), false, self.bits(), rhs.bits());
        match op {
            // Equality is the one comparison a pattern answers on its own: two patterns of one
            // width are equal under either reading exactly when their bits are.
            CmpOp::Eq => self == rhs,
            CmpOp::SLt => self.to_signed() < rhs.to_signed(),
            CmpOp::ULt => BigUint::from(self) < BigUint::from(rhs),
        }
    }
}

/// `2^n` as a [`SignedValue`]: the constant every two's-complement boundary above is built from.
fn two_pow(n: usize) -> SignedValue {
    SignedValue::from(1u8) << n
}

// INVARIANT
// ================================================================================================

/// The private machinery that establishes the width/limb agreement.
impl IntBits {
    /// Establish the invariant: clear every bit at or above `bits` in the top limb.
    fn normalized(bits: usize, mut limbs: Vec<u64>) -> Self {
        assert_eq!(
            limbs.len(),
            Self::limbs_for_bits(bits),
            "a {bits}-bit pattern needs exactly {} limbs",
            Self::limbs_for_bits(bits)
        );
        let top = limbs.len() - 1;
        limbs[top] &= top_limb_mask(bits);
        Self {
            bits,
            limbs: limbs.into_boxed_slice(),
        }
    }
}

// CONVERSIONS FROM HOST TYPES
// ================================================================================================

// These all take their widths and interpretations from the host type.

impl From<u8> for IntBits {
    /// An 8-bit pattern: the width comes from the host type, so there is nothing to pass.
    fn from(value: u8) -> Self {
        Self::from_u128(8, u128::from(value))
    }
}

impl From<i8> for IntBits {
    /// An 8-bit pattern holding `value` in two's complement, which at the host type's own width is
    /// simply its bit pattern: `-1` is every bit set, `i8::MIN` is the sign bit alone.
    fn from(value: i8) -> Self {
        Self::from_u128(8, u128::from(value as u8))
    }
}

impl From<u16> for IntBits {
    /// A 16-bit pattern: the width comes from the host type, so there is nothing to pass.
    fn from(value: u16) -> Self {
        Self::from_u128(16, u128::from(value))
    }
}

impl From<i16> for IntBits {
    /// A 16-bit pattern holding `value` in two's complement, which at the host type's own width
    /// is simply its bit pattern: `-1` is every bit set, `i16::MIN` is the sign bit alone.
    fn from(value: i16) -> Self {
        Self::from_u128(16, u128::from(value as u16))
    }
}

impl From<u32> for IntBits {
    /// A 32-bit pattern: the width comes from the host type, so there is nothing to pass.
    fn from(value: u32) -> Self {
        Self::from_u128(32, u128::from(value))
    }
}

impl From<i32> for IntBits {
    /// A 32-bit pattern holding `value` in two's complement, which at the host type's own width
    /// is simply its bit pattern: `-1` is every bit set, `i32::MIN` is the sign bit alone.
    fn from(value: i32) -> Self {
        Self::from_u128(32, u128::from(value as u32))
    }
}

impl From<u64> for IntBits {
    /// A 64-bit pattern: the width comes from the host type, so there is nothing to pass.
    fn from(value: u64) -> Self {
        Self::from_u128(64, u128::from(value))
    }
}

impl From<i64> for IntBits {
    /// A 64-bit pattern holding `value` in two's complement, which at the host type's own width
    /// is simply its bit pattern: `-1` is every bit set, `i64::MIN` is the sign bit alone.
    fn from(value: i64) -> Self {
        Self::from_u128(64, u128::from(value as u64))
    }
}

// CONVERSIONS TO HOST TYPES
// ================================================================================================

/// Reading a pattern back out, refusing rather than truncating if it will not fit.
///
/// Deliberately unsigned targets only as this type contains a bit pattern, not a signed or unsigned
/// interpretation.
impl TryFrom<&IntBits> for u8 {
    type Error = DoesNotFit;

    fn try_from(value: &IntBits) -> Result<Self, Self::Error> {
        Self::try_from(value.as_u128()?).map_err(|_| DoesNotFit { bits: value.bits })
    }
}

impl TryFrom<IntBits> for u8 {
    type Error = DoesNotFit;

    fn try_from(value: IntBits) -> Result<Self, Self::Error> {
        Self::try_from(&value)
    }
}

impl TryFrom<&IntBits> for u16 {
    type Error = DoesNotFit;

    fn try_from(value: &IntBits) -> Result<Self, Self::Error> {
        Self::try_from(value.as_u128()?).map_err(|_| DoesNotFit { bits: value.bits })
    }
}

impl TryFrom<IntBits> for u16 {
    type Error = DoesNotFit;

    fn try_from(value: IntBits) -> Result<Self, Self::Error> {
        Self::try_from(&value)
    }
}

impl TryFrom<&IntBits> for u32 {
    type Error = DoesNotFit;

    fn try_from(value: &IntBits) -> Result<Self, Self::Error> {
        Self::try_from(value.as_u128()?).map_err(|_| DoesNotFit { bits: value.bits })
    }
}

impl TryFrom<IntBits> for u32 {
    type Error = DoesNotFit;

    fn try_from(value: IntBits) -> Result<Self, Self::Error> {
        Self::try_from(&value)
    }
}

impl TryFrom<&IntBits> for u64 {
    type Error = DoesNotFit;

    fn try_from(value: &IntBits) -> Result<Self, Self::Error> {
        Self::try_from(value.as_u128()?).map_err(|_| DoesNotFit { bits: value.bits })
    }
}

impl TryFrom<IntBits> for u64 {
    type Error = DoesNotFit;

    fn try_from(value: IntBits) -> Result<Self, Self::Error> {
        Self::try_from(&value)
    }
}

impl TryFrom<&IntBits> for u128 {
    type Error = DoesNotFit;

    fn try_from(value: &IntBits) -> Result<Self, Self::Error> {
        value.as_u128()
    }
}

impl TryFrom<IntBits> for u128 {
    type Error = DoesNotFit;

    fn try_from(value: IntBits) -> Result<Self, Self::Error> {
        value.as_u128()
    }
}

/// The one target whose width is the _machine's_ rather than a fixed number of bits.
impl TryFrom<&IntBits> for usize {
    type Error = DoesNotFit;

    fn try_from(value: &IntBits) -> Result<Self, Self::Error> {
        Self::try_from(value.as_u128()?).map_err(|_| DoesNotFit { bits: value.bits })
    }
}

impl TryFrom<IntBits> for usize {
    type Error = DoesNotFit;

    fn try_from(value: IntBits) -> Result<Self, Self::Error> {
        Self::try_from(&value)
    }
}

/// The one conversion the [`TryFrom`] impls above are all written in terms of.
///
/// It is private, and deliberately: a `u128` is the widest host word, so every narrower target
/// checks its own range against the value this produces rather than re-deriving one from the limbs.
/// Widening past 128 bits will change this function and nothing else.
impl IntBits {
    /// The pattern as a `u128`, unsigned, or [`DoesNotFit`] if any bit above 127 is set.
    fn as_u128(&self) -> Result<u128, DoesNotFit> {
        if self.limbs.iter().skip(2).any(|&limb| limb != 0) {
            return Err(DoesNotFit { bits: self.bits });
        }
        let lo = self.limbs.first().copied().unwrap_or(0);
        let hi = self.limbs.get(1).copied().unwrap_or(0);
        Ok(u128::from(lo) | (u128::from(hi) << 64))
    }
}

// FORMATTING
// ================================================================================================

/// Verilog's sized-literal notation.
///
/// Hand-written rather than derived because a derived one prints the limb vector, which is
/// unreadable at any width worth having this type for: `int16384` would be 256 decimal limbs.
impl fmt::Debug for IntBits {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}'h", self.bits)?;

        // Most significant limb first, and the digits have to read as one number rather than as a
        // per-limb rendering. That takes both halves: everything below the leading limb is padded
        // to its full sixteen digits, so a zero limb cannot swallow the magnitude above it, and the
        // leading limb is the most significant _non-zero_ one, so a wide pattern holding a small
        // value does not acquire a run of unnecessary zeroes.
        let Some(top) = self.limbs.iter().rposition(|&limb| limb != 0) else {
            // Every limb is zero, so there is no leading digit to find and the value is `0`.
            return write!(f, "0");
        };

        write!(f, "{:x}", self.limbs[top])?;
        for limb in self.limbs[..top].iter().rev() {
            write!(f, "{limb:016x}")?;
        }
        Ok(())
    }
}

// CONVERSIONS TO AND FROM A BIGNUM
// ================================================================================================

/// The width-free view of a pattern, for the arithmetic the model does on it.
///
/// Unlike the host conversions above this one cannot fail, and unlike them it is not a reading: a
/// [`BigUint`] is the unsigned magnitude of the bits, which is what they are before anyone decides
/// what they mean.
impl From<&IntBits> for BigUint {
    fn from(value: &IntBits) -> Self {
        let bytes: Vec<u8> = value
            .limbs
            .iter()
            .flat_map(|limb| limb.to_le_bytes())
            .collect();
        Self::from_bytes_le(&bytes)
    }
}

impl From<IntBits> for BigUint {
    fn from(value: IntBits) -> Self {
        Self::from(&value)
    }
}

/// The direction the [`From`] impls above cannot take, because it needs a width.
///
/// A [`BigUint`] is a magnitude and carries none, so coming back from one is a choice about how
/// many bits to keep rather than a conversion, which is why it is a named constructor here and not
/// a `From`.
impl IntBits {
    /// A `bits`-wide pattern carrying `value`, discarding any bits at or above `bits`.
    ///
    /// The counterpart of the [`BigUint`] conversion above, and the way an arithmetic result comes
    /// back to being a pattern. It truncates rather than refusing, which is what makes it the
    /// wrapping every total evaluator owes.
    #[must_use]
    pub fn from_biguint(bits: usize, value: &BigUint) -> Self {
        let digits: Vec<u64> = value.iter_u64_digits().collect();
        Self::from_limbs(bits, &digits)
    }
}

// CONVERSION ERROR TYPE
// ================================================================================================

/// A pattern that will not fit the host type asked for, carrying the declared width.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Error)]
#[error("a {bits}-bit pattern does not fit the host type")]
pub struct DoesNotFit {
    /// The width of the pattern that would not convert.
    pub bits: usize,
}

// UTILITY FUNCTIONS
// ================================================================================================

/// The mask the top limb of a `bits`-wide pattern carries.
///
/// A width that is a whole number of limbs has a full top limb; `bits % 64 == 0` is that case and
/// not an empty one, which is why it cannot be written as `(1 << (bits % 64)) - 1`.
const fn top_limb_mask(bits: usize) -> u64 {
    match bits % 64 {
        0 => u64::MAX,
        rest => (1u64 << rest) - 1,
    }
}

// TESTS
// ================================================================================================

#[cfg(test)]
mod tests {
    use super::*;

    use proptest::prelude::*;

    use std::collections::BTreeSet;
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    /// Widths worth constructing at, chosen for their top limbs.
    ///
    /// `16383` and `16384` are the pair that matters most and the reason both are here: `16384` is
    /// exactly 256 limbs, so its top limb is full and every `bits % 64` special case vanishes,
    /// while `16383` is the widest width whose top limb is partial. They occupy the same number of
    /// limbs, so a test that carries only one of them is testing half the boundary.
    const WIDTHS: &[usize] = &[
        1, 2, 3, 7, 8, 63, 64, 65, 96, 127, 128, 129, 191, 192, 193, 1000, 16383, 16384,
    ];

    fn hash_of(v: &IntBits) -> u64 {
        let mut hasher = DefaultHasher::new();
        v.hash(&mut hasher);
        hasher.finish()
    }

    /// The reference the round-trip is held to, for the widths a `u128` can express.
    fn mask_u128(bits: usize) -> u128 {
        if bits >= 128 {
            u128::MAX
        } else {
            (1u128 << bits) - 1
        }
    }

    #[test]
    fn a_pattern_prints_as_one_sized_literal() {
        assert_eq!(format!("{:?}", IntBits::from(42u64)), "64'h2a");
        assert_eq!(format!("{:?}", IntBits::zero(1)), "1'h0");
        assert_eq!(format!("{:?}", IntBits::from(-1i8)), "8'hff");

        // Across a limb boundary the digits have to read as one number: the low limb is padded to
        // its full sixteen digits, so a zero low limb cannot swallow the high limb's magnitude.
        assert_eq!(
            format!("{:?}", IntBits::from_u128(128, 1u128 << 64)),
            "128'h10000000000000000"
        );
        assert_eq!(
            format!("{:?}", IntBits::from_limbs(129, &[0, 0, 1])),
            "129'h100000000000000000000000000000000"
        );

        // The other half of "reads as one number": a value that does not reach the top limb is
        // written the way a number is, without the run of leading zeroes the limb vector holds. The
        // width prefix is what says how much room there was.
        assert_eq!(format!("{:?}", IntBits::from_u128(128, 1)), "128'h1");
        assert_eq!(format!("{:?}", IntBits::from_u128(16384, 42)), "16384'h2a");

        // Zero has no significant limb at all, at any width.
        assert_eq!(format!("{:?}", IntBits::zero(128)), "128'h0");
        assert_eq!(format!("{:?}", IntBits::zero(16384)), "16384'h0");

        // And the rendering stays injective.
        let mut rendered: Vec<(String, IntBits)> = Vec::new();
        for &bits in WIDTHS {
            for value in [0u128, 1, 2, 42, 1 << 64, u128::MAX] {
                let v = IntBits::from_u128(bits, value);
                let text = format!("{v:?}");
                if let Some((_, other)) = rendered.iter().find(|(seen, _)| *seen == text) {
                    assert_eq!(*other, v, "{text} is the rendering of two patterns");
                }
                rendered.push((text, v));
            }
        }
    }

    #[test]
    fn a_pattern_occupies_one_limb_per_64_bits_of_its_width() {
        for &bits in WIDTHS {
            assert_eq!(
                IntBits::zero(bits).limb_count(),
                bits.div_ceil(64),
                "at {bits} bits"
            );
        }
        // The 16383/16384 pair shares a limb count and differs only in the top limb's mask, which
        // is exactly why the count alone is not enough to pin the representation down.
        assert_eq!(IntBits::zero(16383).limb_count(), 256);
        assert_eq!(IntBits::zero(16384).limb_count(), 256);
    }

    #[test]
    fn two_widths_sharing_a_limb_count_are_different_patterns() {
        // The reason the width is stored rather than inferred. Both of these pairs occupy the same
        // number of limbs and hold the same limbs, so a pattern that did not carry its own width
        // could not tell them apart -- and `Eq` would have meant "one spelling per value per limb
        // count" where the interning bimap needs "per width".
        for &(narrow, wide) in &[(63usize, 64usize), (16383, 16384), (1, 64), (65, 128)] {
            let a = IntBits::from_u128(narrow, 1);
            let b = IntBits::from_u128(wide, 1);
            assert_eq!(
                a.limbs(),
                b.limbs(),
                "{narrow} and {wide} hold the same limbs"
            );
            assert_eq!(a.limb_count(), b.limb_count(), "and occupy the same count");
            assert_ne!(a, b, "but {narrow} and {wide} are not the same pattern");
        }
    }

    #[test]
    fn the_stored_width_and_the_limb_count_always_agree() {
        for &bits in WIDTHS {
            let v = IntBits::from_u128(bits, 12345);
            assert_eq!(v.bits(), bits);
            assert_eq!(
                v.limb_count(),
                IntBits::limbs_for_bits(v.bits()),
                "at {bits} bits"
            );
        }
    }

    #[test]
    fn a_host_conversion_takes_its_width_from_the_host_type() {
        for (got, want_bits) in [
            (IntBits::from(1u8), 8),
            (IntBits::from(1u16), 16),
            (IntBits::from(1u32), 32),
            (IntBits::from(1u64), 64),
            (IntBits::from(1i8), 8),
            (IntBits::from(1i16), 16),
            (IntBits::from(1i32), 32),
            (IntBits::from(1i64), 64),
        ] {
            assert_eq!(got.bits(), want_bits);
            assert_eq!(u128::try_from(&got), Ok(1));
        }
    }

    #[test]
    fn a_negative_reads_back_as_the_pattern_it_is() {
        // A pattern has no sign, so reading one out gives the unsigned reading of the bits and
        // nothing else: `-1i8` went in, `255u8` comes back. Recovering `-1` means naming the
        // reading, which is `to_signed`'s job and not a conversion's.
        let minus_one = IntBits::from(-1i8);
        assert_eq!(u8::try_from(&minus_one), Ok(255));
        assert_eq!(u32::try_from(&minus_one), Ok(255));
        assert_eq!(minus_one.to_signed(), crate::SignedValue::from(-1));
    }

    #[test]
    fn a_value_that_will_not_fit_the_target_is_refused() {
        // Too wide for the target, though it fits a `u128` perfectly well.
        let big = IntBits::from(300u16);
        assert_eq!(u8::try_from(&big), Err(DoesNotFit { bits: 16 }));
        assert_eq!(u16::try_from(&big), Ok(300));

        // A negative value has no unsigned reading at all.
        assert_eq!(
            u8::try_from(&IntBits::from(-1i16)),
            Err(DoesNotFit { bits: 16 })
        );

        // And the width the error carries is the pattern's, which is the thing the call site
        // cannot see for itself.
        assert_eq!(
            u64::try_from(&IntBits::from_limbs(256, &[0, 0, 0, 1])),
            Err(DoesNotFit { bits: 256 })
        );
    }

    #[test]
    fn the_error_names_the_width_that_would_not_convert() {
        // The width is the whole content of the message: the call site already knows which host
        // type it asked for, and cannot see what it asked of.
        let err = u8::try_from(&IntBits::from(300u16)).expect_err("300 does not fit a u8");
        assert_eq!(
            err.to_string(),
            "a 16-bit pattern does not fit the host type"
        );
    }

    #[test]
    fn a_by_value_conversion_agrees_with_the_borrowed_one() {
        let v = IntBits::from(-2i32);
        assert_eq!(u32::try_from(v.clone()), u32::try_from(&v));
        assert_eq!(u8::try_from(v.clone()), u8::try_from(&v));
    }

    #[test]
    fn a_signed_host_constructor_stores_the_twos_complement_pattern() {
        // `-1` is every bit set, at the width the name gives and no wider. What this actually
        // catches is a lost sign or a wrong width -- _not_ an over-wide sign extension, which
        // `from_u128` masks away, so `value as i128 as u128` here would be equally correct.
        assert_eq!(IntBits::from(-1i8), IntBits::from(u8::MAX));
        assert_eq!(IntBits::from(-1i16), IntBits::from(u16::MAX));
        assert_eq!(IntBits::from(-1i32), IntBits::from(u32::MAX));
        assert_eq!(IntBits::from(-1i64), IntBits::from(u64::MAX));

        // `INT_MIN` is the sign bit alone.
        assert_eq!(IntBits::from(i8::MIN), IntBits::from(1u8 << 7));
        assert_eq!(IntBits::from(i16::MIN), IntBits::from(1u16 << 15));
        assert_eq!(IntBits::from(i32::MIN), IntBits::from(1u32 << 31));
        assert_eq!(IntBits::from(i64::MIN), IntBits::from(1u64 << 63));

        // Spelled out as limbs too, so the assertion does not rest on `From<u8>` being right.
        assert_eq!(IntBits::from(-1i8).limbs(), &[0xFF]);
        assert_eq!(IntBits::from(-2i8).limbs(), &[0xFE]);
    }

    #[test]
    #[should_panic(expected = "at least one bit wide")]
    fn a_zero_width_pattern_is_rejected() {
        let _ = IntBits::zero(0);
    }

    #[test]
    fn no_bit_at_or_above_the_width_survives_construction() {
        for &bits in WIDTHS {
            // All-ones in every limb the width reserves, which is the worst case for the mask.
            let saturated = vec![u64::MAX; bits.div_ceil(64)];
            let v = IntBits::from_limbs(bits, &saturated);

            for i in 0..bits {
                assert_eq!(v.bit(i), Some(true), "bit {i} of {bits} was cleared");
            }

            // Checked against the limbs directly, NOT through `bit`. `bit` answers `None` at or
            // above the width whatever the limbs hold, so asking it here would prove nothing --
            // the top limb could be entirely dirty and this test would still pass.
            let top = v.limbs().last().expect("a width has at least one limb");
            assert_eq!(
                top & !top_limb_mask(bits),
                0,
                "the top limb of {bits} kept bits above the width"
            );

            // And `bit` does refuse to read there, which is the separate claim.
            for i in bits..bits.div_ceil(64) * 64 {
                assert_eq!(v.bit(i), None, "bit {i} is outside a {bits}-bit pattern");
            }
        }
    }

    #[test]
    fn dirt_above_the_width_does_not_change_the_value() {
        // The canonicity property, stated the way it actually gets violated: a caller hands over a
        // word that some earlier step left dirty above the declared width, and the result must be
        // indistinguishable from the clean one. Anything less splits one value into two
        // `ValueId`s at the interning bimap.
        for &bits in &[1usize, 3, 8, 63, 64, 65, 127, 128] {
            let clean = IntBits::from_u128(bits, 1);
            let dirty = IntBits::from_u128(bits, 1 | !mask_u128(bits));
            assert_eq!(clean, dirty, "at {bits} bits");
            assert_eq!(hash_of(&clean), hash_of(&dirty), "at {bits} bits");
        }

        // And through the wide door too, where `from_u128` cannot reach.
        for &bits in &[129usize, 1000, 16383] {
            let mut dirty = vec![0u64; bits.div_ceil(64)];
            dirty[0] = 1;
            *dirty.last_mut().expect("a width has at least one limb") = u64::MAX;
            let mut clean = vec![0u64; bits.div_ceil(64)];
            clean[0] = 1;
            *clean.last_mut().expect("a width has at least one limb") = top_limb_mask(bits);
            assert_eq!(
                IntBits::from_limbs(bits, &dirty),
                IntBits::from_limbs(bits, &clean),
                "at {bits} bits"
            );
        }
    }

    #[test]
    fn a_full_top_limb_is_not_masked_away() {
        // `bits % 64 == 0` is the case a naive `(1 << (bits % 64)) - 1` turns into a zero mask,
        // which would silently delete the entire top limb of every whole-limb width.
        for &bits in &[64usize, 128, 192, 16384] {
            let saturated = vec![u64::MAX; bits / 64];
            let v = IntBits::from_limbs(bits, &saturated);
            assert_eq!(
                v.limbs().last(),
                Some(&u64::MAX),
                "the top limb of {bits} was masked"
            );
        }
    }

    #[test]
    fn a_limb_slice_is_zero_extended_and_truncated_to_the_width() {
        // Short: the missing limbs read as zero rather than as garbage.
        let wide = IntBits::from_limbs(256, &[7]);
        assert_eq!(wide.limbs(), &[7, 0, 0, 0]);

        // Long: the limbs past the width go, on the same terms as bits past the width within a
        // limb. Truncating rather than rejecting is what keeps this consistent with `from_u128`.
        let narrow = IntBits::from_limbs(64, &[7, u64::MAX, u64::MAX]);
        assert_eq!(narrow.limbs(), &[7]);
    }

    #[test]
    fn the_field_read_is_the_packed_one_at_the_host_limb_width() {
        // `from_field_limbs` takes the shortcut `from_packed_limbs` cannot: at the host limb width
        // the packed limbs already are the pattern's own words. The two must answer alike, so we
        // pin it here.
        //
        // The slice lengths straddle the width on purpose -- shorter, exact and longer -- because
        // that is where the two spellings' truncation and zero-extension could part company.
        for &bits in &[1usize, 8, 63, 64, 65, 96, 127, 128] {
            for limbs in [
                &[][..],
                &[0][..],
                &[7][..],
                &[u64::MAX][..],
                &[7, 1][..],
                &[u64::MAX, u64::MAX][..],
                &[0, 0, 1][..],
                &[u64::MAX, u64::MAX, u64::MAX, u64::MAX][..],
            ] {
                assert_eq!(
                    IntBits::from_field_limbs(limbs, bits),
                    IntBits::from_packed_limbs(limbs, FIELD_LIMB_BITS, bits),
                    "{limbs:?} at {bits} bits"
                );
            }
        }
    }

    #[test]
    fn a_value_round_trips_through_a_host_word_when_it_fits() {
        for &bits in &[1usize, 3, 8, 63, 64, 65, 127, 128] {
            for value in [0u128, 1, 2, 0x5555_5555_5555_5555, u128::MAX, u128::MAX - 1] {
                assert_eq!(
                    u128::try_from(&IntBits::from_u128(bits, value)),
                    Ok(value & mask_u128(bits)),
                    "{value:#x} at {bits} bits"
                );
            }
        }
    }

    #[test]
    fn a_value_too_wide_for_a_host_word_declines_rather_than_truncating() {
        // The whole reason the conversion is fallible: a caller wanting a host word out of a
        // pattern has to decide what a wider one means, and refusing is what makes it decide.
        let mut limbs = vec![0u64; 4];
        limbs[3] = 1;
        assert!(u128::try_from(&IntBits::from_limbs(256, &limbs)).is_err());

        // A wide *width* with a narrow *value* still fits, because the question is about the
        // value and not about how much room it was given.
        assert_eq!(u128::try_from(&IntBits::from_u128(16384, 5)), Ok(5));
        assert_eq!(u128::try_from(&IntBits::zero(16384)), Ok(0));
    }

    #[test]
    fn is_zero_and_bit_agree_with_the_limbs() {
        assert!(IntBits::zero(16384).is_zero());
        assert!(IntBits::from_u128(128, 0).is_zero());
        assert!(!IntBits::from_u128(128, 1 << 100).is_zero());

        // A one-bit width has exactly one bit to read, and nothing above it.
        let one = IntBits::from_u128(1, 1);
        assert_eq!(one.bit(0), Some(true));
        assert_eq!(one.bit(1), None);
        assert_eq!(one.bit(63), None);
        // Reading far past the last limb is out of range, not a panic.
        assert_eq!(one.bit(100_000), None);
    }

    #[test]
    fn the_distinguished_patterns_are_recognised_at_every_width() {
        for &bits in WIDTHS {
            let zero = IntBits::zero(bits);
            let one = IntBits::from_u128(bits, 1);
            let limbs = vec![u64::MAX; IntBits::limbs_for_bits(bits)];
            let all_ones = IntBits::from_limbs(bits, &limbs);

            assert!(zero.is_zero() && !zero.is_one(), "zero at {bits}");
            assert!(one.is_one() && !one.is_zero(), "one at {bits}");
            assert!(all_ones.is_all_ones(), "all ones at {bits}");
            assert!(!zero.is_all_ones(), "zero is not saturated at {bits}");

            // At one bit these three patterns are two, not three, and the predicates have to say
            // so rather than each claiming its own value.
            assert_eq!(one.is_all_ones(), bits == 1, "one vs all ones at {bits}");
            assert_eq!(all_ones.is_one(), bits == 1, "all ones vs one at {bits}");
        }

        // Saturation is per-width, so the same limbs are all-ones at one width and not at another.
        assert!(IntBits::from_limbs(64, &[u64::MAX]).is_all_ones());
        assert!(!IntBits::from_limbs(128, &[u64::MAX, 0]).is_all_ones());
        // And it must read every limb, not just the top one.
        assert!(!IntBits::from_limbs(128, &[0, u64::MAX]).is_all_ones());
        // Likewise `is_one` has to look past the limb the one is in, or any value whose low limb
        // happens to be 1 answers yes.
        assert!(!IntBits::from_limbs(128, &[1, 1]).is_one());
    }

    #[test]
    fn an_index_too_wide_for_the_machine_is_refused_rather_than_truncated() {
        // The whole point of the `usize` conversion: a pattern carrying a value above the host's
        // address width must not come back as some unrelated small number that passes a bounds
        // check it should have failed.
        let huge = IntBits::from_u128(128, (u64::MAX as u128) + 1);
        assert_eq!(usize::try_from(&huge), Err(DoesNotFit { bits: 128 }));

        assert_eq!(usize::try_from(&IntBits::from_u128(128, 7)), Ok(7));
        assert_eq!(usize::try_from(IntBits::from(3u8)), Ok(3));
    }

    #[test]
    fn distinct_values_at_one_width_stay_distinct() {
        // The other half of canonicity: masking must not collapse values that differ below the
        // width. A mask applied at the wrong width would show up here and nowhere else.
        for &bits in &[3usize, 8, 64, 65, 128] {
            let m = mask_u128(bits);
            let values = [0u128, 1, 2, 3, m, m - 1, m ^ 1];

            // A `Vec` and a linear scan rather than a set: `IntBits` has no `Ord` for a `BTreeSet`
            // and a `HashSet` is disallowed workspace-wide for its iteration order. Seven values.
            let mut seen: Vec<IntBits> = Vec::new();
            for value in values {
                let v = IntBits::from_u128(bits, value);
                if !seen.contains(&v) {
                    seen.push(v);
                }
            }

            let distinct = values.iter().map(|v| v & m).collect::<BTreeSet<_>>().len();
            assert_eq!(seen.len(), distinct, "at {bits} bits");
        }
    }

    proptest! {
        /// Canonicity over the whole space rather than the corners: one value at one width has
        /// exactly one spelling, however it is reached.
        #[test]
        fn one_value_at_one_width_has_one_spelling(
            width_index in 0..WIDTHS.len(),
            value in any::<u128>(),
            dirt in any::<u128>(),
        ) {
            let bits = WIDTHS[width_index];
            let direct = IntBits::from_u128(bits, value);

            // Reached again through the limb door, carrying arbitrary dirt above the width.
            let mut limbs = direct.limbs().to_vec();
            let top = limbs.len() - 1;
            limbs[top] |= (dirt as u64) & !top_limb_mask(bits);
            let indirect = IntBits::from_limbs(bits, &limbs);

            prop_assert_eq!(&direct, &indirect);
            prop_assert_eq!(hash_of(&direct), hash_of(&indirect));
        }

        /// Construction is idempotent: feeding a pattern's own limbs back in changes nothing.
        #[test]
        fn rebuilding_from_its_own_limbs_is_the_identity(
            width_index in 0..WIDTHS.len(),
            value in any::<u128>(),
        ) {
            let bits = WIDTHS[width_index];
            let v = IntBits::from_u128(bits, value);
            prop_assert_eq!(IntBits::from_limbs(bits, v.limbs()), v);
        }

        /// The bits a pattern reports are exactly the bits its limbs hold, and none above.
        #[test]
        fn every_bit_reads_back_from_the_limb_that_holds_it(
            width_index in 0..WIDTHS.len(),
            value in any::<u128>(),
        ) {
            let bits = WIDTHS[width_index];
            let v = IntBits::from_u128(bits, value);
            for i in 0..bits.min(128) {
                let want = (value >> i) & 1 == 1;
                prop_assert_eq!(v.bit(i), Some(want), "bit {} at {} bits", i, bits);
            }
            for i in bits..bits + 64 {
                prop_assert_eq!(v.bit(i), None, "bit {} is outside {} bits", i, bits);
            }
        }
    }

    // PATTERN OPERATIONS
    // --------------------------------------------------------------------------------------

    /// Widths that straddle every limb boundary a 128-bit cap can reach: a sub-limb width, the two
    /// either side of the first boundary, an odd one, and the two either side of the second.
    const CORNERS: [usize; 8] = [1, 5, 63, 64, 65, 96, 127, 128];

    /// The `u128` a pattern denotes, for holding the limb code to host semantics.
    fn as_u128(v: &IntBits) -> u128 {
        u128::try_from(v).expect("the corner widths all fit a u128")
    }

    #[test]
    fn all_ones_saturates_exactly_the_declared_width() {
        for bits in CORNERS {
            let ones = IntBits::all_ones(bits);
            assert!(ones.is_all_ones(), "{bits}");
            assert_eq!(ones.bit(bits - 1), Some(true), "{bits}");
            assert_eq!(ones.bit(bits), None, "{bits} has no bit at its own width");
            assert!(ones.complement().is_zero(), "{bits}");
            assert_eq!(as_u128(&ones), crate::mask(bits), "{bits}");
        }
    }

    #[test]
    fn the_bitwise_operations_are_limbwise_across_a_boundary() {
        // Values chosen so that both limbs differ and the top limb is partial at the odd widths,
        // which is where a mask applied to the wrong limb would show.
        for bits in CORNERS {
            let a = IntBits::from_u128(bits, 0x0F0F_0F0F_0F0F_0F0F_FFFF_0000_FFFF_0000);
            let b = IntBits::from_u128(bits, 0x00FF_00FF_00FF_00FF_0F0F_0F0F_0F0F_0F0F);
            let (x, y) = (as_u128(&a), as_u128(&b));
            assert_eq!(as_u128(&a.and(&b)), x & y, "and at {bits}");
            assert_eq!(
                as_u128(&a.or(&b)),
                (x | y) & crate::mask(bits),
                "or at {bits}"
            );
            assert_eq!(
                as_u128(&a.xor(&b)),
                (x ^ y) & crate::mask(bits),
                "xor at {bits}"
            );
            assert_eq!(
                as_u128(&a.complement()),
                !x & crate::mask(bits),
                "complement at {bits}"
            );
            assert_eq!(a.complement().complement(), a, "an involution at {bits}");
        }
    }

    #[test]
    #[should_panic(expected = "cannot sign-extend 32 to 8")]
    fn sign_extension_refuses_to_narrow() {
        // The gate is a full `assert!` rather than a `debug_assert!` because a release build that
        // skipped it would not merely truncate: the fill is taken from the source's sign bit, so a
        // narrowing would answer a value that is neither the source nor its low bits.
        let _ = IntBits::from(-1i32).sign_extend(8);
    }

    #[test]
    #[should_panic(expected = "is outside 1..=64")]
    fn a_signed_reading_refuses_a_pattern_above_the_signed_cap() {
        // A caller that reaches here has skipped `assert_signed_op_width`, and answering would
        // be worse than stopping.
        let _ = IntBits::from_u128(128, 1).to_signed();
    }

    #[test]
    #[should_panic(expected = "at least a sign bit")]
    fn the_signed_bounds_refuse_a_zero_width() {
        // `bits - 1` wraps in a release build, so this gate is the difference between a panic and
        // asking for a two-power of `usize::MAX`.
        let _ = IntBits::signed_max(0);
    }

    #[test]
    #[should_panic(expected = "a width of at least one bit")]
    fn a_shift_amount_refuses_to_reduce_against_no_width() {
        // The last width-taking method to answer instead of refusing. The VM's counterpart is total
        // there because its width comes off a frame cell; this one's comes off the operand `eval`
        // was handed, so a zero is a caller that skipped the model's own width check.
        let _ = IntBits::from(1u8).reduced_shift_amount(0);
    }

    #[test]
    fn a_shift_amount_reduces_by_the_width_it_is_given() {
        // The neighboring claim, so the refusal above is not the only thing pinning this method: an
        // in-range amount survives and an out-of-range one wraps by a modulo rather than a mask. At
        // seven bits a mask by `bits - 1 == 6` would answer `0` for both of the first two.
        assert_eq!(IntBits::from(1u8).reduced_shift_amount(7), 1);
        assert_eq!(IntBits::from(8u8).reduced_shift_amount(7), 1);
        assert_eq!(IntBits::from(9u8).reduced_shift_amount(8), 1);
        assert_eq!(IntBits::from(1u8).reduced_shift_amount(1), 0);
    }

    #[test]
    #[should_panic(expected = "two patterns of one width")]
    fn a_bitwise_operation_refuses_two_widths() {
        // 32 and 64 share a limb, so nothing about the storage stops this: it is refused because
        // the bit positions of two widths do not correspond, not because the limbs disagree.
        let _ = IntBits::from(1u32).and(&IntBits::from(1u64));
    }

    #[test]
    #[should_panic(expected = "at least one bit wide")]
    fn a_cast_to_no_width_is_refused() {
        // The width-taking operations inherit the constructors' lower bound rather than answering
        // an empty pattern.
        let _ = IntBits::from(1u8).cast(0);
    }

    #[test]
    #[should_panic(expected = "at least one bit wide")]
    fn an_empty_bit_range_is_refused() {
        let _ = IntBits::from(0xABCDu16).bit_range(4, 0);
    }

    #[test]
    fn a_cast_to_the_width_already_held_is_the_identity() {
        // The fast path, and the property it has to preserve: `bit_range(..).cast(self.bits())` is
        // how both constant folders spell the `BitRange` transfer, so the common case must answer
        // exactly what the general one does.
        for bits in CORNERS {
            for value in [0u128, 1, 0x5A5A_5A5A, u128::MAX] {
                let v = IntBits::from_u128(bits, value);
                assert_eq!(v.cast(bits), v, "{value:#x} at {bits} bits");
                assert_eq!(v.cast(bits), IntBits::from_limbs(bits, v.limbs()));
            }
        }
    }

    #[test]
    fn shifting_moves_bits_and_discards_what_leaves_the_width() {
        for bits in CORNERS {
            let v = IntBits::from_u128(bits, 0x1234_5678_9ABC_DEF0_0FED_CBA9_8765_4321);
            let x = as_u128(&v);
            for amount in [0usize, 1, 31, 63, 64, 65, 96, 127] {
                let expected_left = x.checked_shl(amount as u32).unwrap_or(0) & crate::mask(bits);
                let expected_right = x.checked_shr(amount as u32).unwrap_or(0);
                assert_eq!(
                    as_u128(&v.shifted_left(amount)),
                    if amount >= bits { 0 } else { expected_left },
                    "{bits} << {amount}"
                );
                assert_eq!(
                    as_u128(&v.shifted_right(amount)),
                    if amount >= bits { 0 } else { expected_right },
                    "{bits} >> {amount}"
                );
            }
            assert_eq!(v.shifted_left(0), v, "{bits}");
            assert_eq!(v.shifted_right(0), v, "{bits}");
        }
    }

    #[test]
    fn a_shift_by_a_whole_limb_keeps_every_bit_it_should() {
        // The case the guarded carry exists for: `limb << 64` is undefined rather than zero, so a
        // shift that is an exact multiple of the limb width must take the other path.
        let v = IntBits::from_u128(128, u128::MAX);
        assert_eq!(as_u128(&v.shifted_left(64)), u128::MAX << 64);
        assert_eq!(as_u128(&v.shifted_right(64)), u128::MAX >> 64);
    }

    // BIGNUM CONVERSIONS
    // --------------------------------------------------------------------------------------

    #[test]
    fn a_pattern_round_trips_through_a_bignum() {
        for bits in CORNERS {
            for value in [0u128, 1, 0xDEAD_BEEF, u128::MAX, u128::MAX >> 1] {
                let v = IntBits::from_u128(bits, value);
                let big = BigUint::from(&v);
                assert_eq!(
                    big,
                    BigUint::from(as_u128(&v)),
                    "the magnitude at {bits} of {value:#x}"
                );
                assert_eq!(IntBits::from_biguint(bits, &big), v, "{bits}, {value:#x}");
            }
        }
    }

    #[test]
    fn a_bignum_too_wide_for_the_pattern_is_truncated() {
        // The same discipline as `from_limbs`: a constructor that takes a width discards above it
        // rather than complaining, which is what makes it the wrapping a total evaluator owes.
        let wide = BigUint::from(u128::MAX);
        assert_eq!(IntBits::from_biguint(8, &wide), IntBits::from(0xFFu8));
        assert_eq!(IntBits::from_biguint(64, &wide), IntBits::from(u64::MAX));
    }
}
