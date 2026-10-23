//! [`IntBits`] is a width-sized two's-complement bit pattern that stores and evaluates
//! arbitrary-width integers at compile time. It has exactly [`IntBits::limbs_for_bits`] limbs, with
//! every bit above the width set to zero.

use std::fmt;

use num_bigint::BigUint;
use smallvec::SmallVec;
use thiserror::Error;

use crate::{CmpOp, MAX_BITS, SignedValue, check_widths, mask};

// HOST LIMB
// ================================================================================================

/// The width of one limb of an [`IntBits`] pattern, and of one integer cell in a VM frame.
pub const HOST_LIMB_BITS: usize = u64::BITS as usize;

/// The width of the widest integer the host has a type for.
///
/// One definition for the many sites that need it: a mask computed in a `u128`, a constant built
/// with `1u128 << n`, a generator returning `Vec<u128>`, the widest [`crate::mask`] can express.
///
/// **Not** [`MAX_BITS`], which is far higher: a bound written as the model's cap would reach a
/// `1u128 << bits` at a width no `u128` holds, which panics in a debug build and, worse, silently
/// gives a wrong answer in release, where the shift amount is masked.
pub const HOST_WORD_BITS: usize = u128::BITS as usize;

// Two inline limbs cover every host and every Noir width; a wider pattern spills to the heap. So
// cloning the common case, including the elements of arrays compiler analysis copies, never
// allocates.
type Limbs = SmallVec<[u64; 2]>;

// INTEGER BIT PATTERN
// ================================================================================================

/// A `bits`-wide two's-complement pattern: exactly [`IntBits::limbs_for_bits`] little-endian
/// [`HOST_LIMB_BITS`]-wide limbs, with every bit at or above `bits` set to zero.
///
/// It has no [`Ord`]: a bit pattern has two valid orderings, signed and unsigned. Use
/// [`IntBits::compare`], which names the reading.
///
/// # Performance
///
/// Up to 128 bits the limbs are inline, so cloning a small integer never allocates. Storage does
/// not affect width, normalization, equality, hashing or arithmetic.
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct IntBits {
    /// The width of the integer.
    bits: usize,

    /// Exactly `IntBits::limbs_for_bits(bits)` of them, little-endian, with the top limb masked.
    limbs: Limbs,
}

// CONSTRUCTORS
// ================================================================================================

/// Constructors for the type, all of which perform normalization during construction.
///
/// **None of these caps the width at [`MAX_BITS`]**: that bounds the operations the model
/// evaluates, not the patterns the type holds. The field-limb group below does check it
/// ([`IntBits::from_packed_limbs`], [`IntBits::from_field_limbs`] and their predicate
/// [`IntBits::field_limbs_fit`]), because those read a _field element_, which the model does
/// constrain.
///
/// The **lower** bound is enforced by [`IntBits::limbs_for_bits`]: a width of zero describes no
/// pattern, so it panics rather than building an empty one. Everything that takes a width
/// ([`IntBits::cast`], [`IntBits::bit_range`], [`IntBits::sign_extend`]) inherits that panic, so a
/// caller holding a width it did not choose must refuse a zero first.
impl IntBits {
    /// Zero at `bits` wide.
    #[must_use]
    pub fn zero(bits: usize) -> Self {
        Self::normalized(bits, Limbs::from_elem(0, Self::limbs_for_bits(bits)))
    }

    /// The pattern `1` at `bits` wide, and the companion of [`IntBits::is_one`].
    #[must_use]
    pub fn one(bits: usize) -> Self {
        Self::from_u128(bits, 1)
    }

    /// A `bits`-wide pattern carrying `value`, discarding any bits at or above `bits`.
    #[must_use]
    pub fn from_u128(bits: usize, value: u128) -> Self {
        let count = Self::limbs_for_bits(bits);
        let mut limbs = Limbs::from_elem(0, count);
        limbs[0] = value as u64;
        if count > 1 {
            limbs[1] = (value >> HOST_LIMB_BITS) as u64;
        }
        Self::normalized(bits, limbs)
    }

    /// Every bit below `bits` set, and the companion of [`IntBits::is_all_ones`].
    ///
    /// It works at any width, unlike [`crate::mask`], which answers as a _host_ word and so cannot
    /// express a width the host has no type for.
    #[must_use]
    pub fn all_ones(bits: usize) -> Self {
        Self::normalized(bits, Limbs::from_elem(u64::MAX, Self::limbs_for_bits(bits)))
    }

    /// A `bits`-wide pattern from little-endian limbs, zero-extending a slice too short to fill the
    /// width and truncating one too long for it.
    #[must_use]
    pub fn from_limbs(bits: usize, limbs: &[u64]) -> Self {
        let count = Self::limbs_for_bits(bits);
        let mut out = Limbs::from_elem(0, count);
        let taken = limbs.len().min(count);
        out[..taken].copy_from_slice(&limbs[..taken]);
        Self::normalized(bits, out)
    }
}

// FIELD LIMB INTEROP
// ================================================================================================

/// The width of one limb of a **canonical field representation**.
///
/// The same for every field: arkworks' `ark_ff::BigInt<const N: usize>` is `[u64; N]`, so the limb
/// _count_ varies by field and the limb _width_ does not. (This crate computes with `num-bigint`,
/// whose digit width does not matter here: [`IntBits::from_biguint`] reads through
/// `iter_u64_digits`.)
///
/// This crate cannot see arkworks, so a `const` assertion in the compiler's `util.rs` checks the
/// value.
pub const FIELD_LIMB_BITS: usize = 64;

/// Reading a field element's limbs as a pattern. A canonical field representation is an unsigned
/// magnitude, so no signed or unsigned reading is needed, only a truncation.
impl IntBits {
    /// Recombine little-endian limbs of `limb_bits` each into a `bits`-wide pattern.
    ///
    /// The limb width is a parameter because Mavros has two unrelated kinds of limb: a canonical
    /// field representation's are [`FIELD_LIMB_BITS`] wide, while the witness decompositions in
    /// `witness_bitwise` use a width derived from the field size.
    ///
    /// Each limb is masked to `limb_bits` first, so a caller packing a narrow limb into a wider
    /// container need not clear the bits above it.
    ///
    /// # Panics
    ///
    /// If `limb_bits` is zero, or if `bits` is outside the model's range.
    #[must_use]
    pub fn from_packed_limbs(limbs: &[u64], limb_bits: usize, bits: usize) -> Self {
        assert!(
            (1..=MAX_BITS).contains(&bits),
            "pattern width {bits} is outside 1..={MAX_BITS}"
        );
        assert!(limb_bits > 0, "a limb must be at least one bit wide");

        // Limb `i` starts at bit `i * limb_bits`, so limbs from here up contribute nothing below
        // `bits`; the final truncation would discard them anyway.
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

    /// Read a field element's canonical little-endian limbs as a `bits`-wide pattern: the
    /// `Field -> Int` cast, [`IntBits::from_packed_limbs`] at [`FIELD_LIMB_BITS`].
    ///
    /// It keeps the low `bits` bits of the **canonical** form, so a caller holding a Montgomery
    /// form must convert first. The limb count depends on the field; limbs missing below `bits`
    /// read as zero, as they do for a smaller element.
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

        // Not through `from_packed_limbs`: at the host limb width the packed limbs **are** the
        // pattern's little-endian words. Tests check the two agree.
        Self::from_limbs(bits, limbs)
    }

    /// Whether a field element's canonical limbs are entirely inside a `bits`-wide pattern.
    ///
    /// [`IntBits::from_field_limbs`] **truncates**; this asks whether that truncation would lose
    /// anything, so a caller can refuse.
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

        // Limbs below `whole` are covered outright, and the one at `whole` up to `spare` bits.
        // Where `spare` is zero the width ends on a limb boundary, and that shift asks for the
        // whole limb above to be zero.
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

    /// The number of [`HOST_LIMB_BITS`]-wide limbs a `bits`-wide pattern occupies.
    ///
    /// [`HOST_LIMB_BITS`] comes from the machine, not the field: this is what `int_cell_count`
    /// reserves in a VM frame and `for_each_constant_word` emits. The _field_ limb is a separate,
    /// narrower width derived from the field.
    ///
    /// # Panics
    ///
    /// On a width of zero, which describes no pattern at all.
    #[must_use]
    pub fn limbs_for_bits(bits: usize) -> usize {
        assert!(bits > 0, "an integer pattern is at least one bit wide");
        bits.div_ceil(HOST_LIMB_BITS)
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
    /// That is `-1` as two's complement, but this asks only whether the pattern is saturated. The
    /// limbs alone answer it _because_ the pattern is normalized: the top limb's maximum is
    /// [`top_limb_mask`].
    #[must_use]
    pub fn is_all_ones(&self) -> bool {
        let top = self.limbs.len() - 1;
        self.limbs[..top].iter().all(|&limb| limb == u64::MAX)
            && self.limbs[top] == top_limb_mask(self.bits)
    }

    /// Bit `index`, counting from the least significant, or [`None`] at or above the width.
    ///
    /// [`None`] rather than `false`: a pattern has no bit at or above its width, and `false` would
    /// make an out-of-range read look like a zero bit of the value.
    #[must_use]
    pub fn bit(&self, index: usize) -> Option<bool> {
        if index >= self.bits {
            return None;
        }
        Some((self.limbs[index / HOST_LIMB_BITS] >> (index % HOST_LIMB_BITS)) & 1 == 1)
    }
}

// PATTERN OPERATIONS
// ================================================================================================

/// Operations a pattern supports _without_ a reading being named.
///
/// Each one rearranges bits, so its answer is the same whether the operands are read as signed
/// or unsigned.
impl IntBits {
    /// Bitwise `and` with another pattern of the same width.
    ///
    /// # Panics
    ///
    /// If the widths differ. Every caller checks that they agree first.
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
        // A cast to the same width is the identity, and the common case: both constant folders
        // spell the `BitRange` transfer as `bit_range(..).cast(self.bits())`.
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
    /// `offset` is unbounded: a pattern shifted past its width is simply empty (unlike a host word,
    /// where that is undefined). `width` must be at least one, as for [`IntBits::cast`]: an empty
    /// window is not a pattern.
    ///
    /// # Panics
    ///
    /// On a `width` of zero. `analysis::types` refuses such a `BitRange` too, so a folder that
    /// meets one has IR that does not type-check.
    #[must_use]
    pub fn bit_range(&self, offset: usize, width: usize) -> Self {
        self.shifted_right(offset).cast(width)
    }

    /// The pattern moved `amount` places toward the most significant end, zero-filling behind it.
    ///
    /// Not `<<`, which has more rules: Noir rejects an amount at or past the width, and a total
    /// backend reduces it modulo the width first. Here such an amount leaves zero.
    #[must_use]
    pub fn shifted_left(&self, amount: usize) -> Self {
        if amount >= self.bits {
            return Self::zero(self.bits);
        }
        let (whole, part) = (amount / HOST_LIMB_BITS, amount % HOST_LIMB_BITS);
        let mut out = Limbs::from_elem(0, self.limbs.len());
        for (i, &limb) in self.limbs.iter().enumerate() {
            let Some(target) = out.get_mut(i + whole) else {
                break;
            };
            *target |= limb << part;
            // A carry by a whole limb would be `limb << HOST_LIMB_BITS`, which overflows rather
            // than giving zero, so the carry into the next limb is guarded.
            if part > 0
                && let Some(next) = out.get_mut(i + whole + 1)
            {
                *next |= limb >> (HOST_LIMB_BITS - part);
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
        let (whole, part) = (amount / HOST_LIMB_BITS, amount % HOST_LIMB_BITS);
        let mut out = Limbs::from_elem(0, self.limbs.len());
        for i in 0..self.limbs.len() {
            let Some(&limb) = self.limbs.get(i + whole) else {
                break;
            };
            out[i] |= limb >> part;
            if part > 0
                && let Some(&next) = self.limbs.get(i + whole + 1)
            {
                out[i] |= next << (HOST_LIMB_BITS - part);
            }
        }
        Self::normalized(self.bits, out)
    }

    /// Interleave the low `bits` bits with zeros, so that bit `i` lands on bit `2i` of a pattern
    /// twice this one's width.
    ///
    /// Only the low `bits` bits are read. A witnessed spread rejects a wider value through its
    /// lookup, but a total evaluator still needs a pattern. The result is twice the pattern's
    /// width, whatever `bits` is.
    ///
    /// Written bit by bit, as a specification.
    ///
    /// # Panics
    ///
    /// If `bits` is zero or wider than the pattern.
    #[must_use]
    pub fn spread(&self, bits: usize) -> Self {
        assert!(
            (1..=self.bits).contains(&bits),
            "a spread reads 1..={} bits of an int{}, not {bits}",
            self.bits,
            self.bits
        );
        let mut out = Self::zero(2 * self.bits);
        for i in (0..bits).filter(|&i| self.bit(i) == Some(true)) {
            out.set_bit(2 * i);
        }
        out
    }

    /// Separate the low `bits` bits into their odd- and even-indexed streams, the inverse of
    /// [`IntBits::spread`], answering `(odd, even)`.
    ///
    /// Bit `2i + 1` becomes bit `i` of the odd stream and bit `2i` bit `i` of the even one, so the
    /// odd answer is `floor(width / 2)` wide and the even one `ceil(width / 2)`. Bits at or above
    /// `bits` are discarded. The read is at least two bits, as HLSSA's `Unspread` requires.
    ///
    /// # Panics
    ///
    /// If `bits` is below two or wider than the pattern, or if the pattern is a single bit, which
    /// has no odd stream.
    #[must_use]
    pub fn unspread(&self, bits: usize) -> (Self, Self) {
        assert!(self.bits >= 2, "an int1 has no odd stream to unspread into");
        assert!(
            (2..=self.bits).contains(&bits),
            "an unspread reads 2..={} bits of an int{}, not {bits}",
            self.bits,
            self.bits
        );
        let mut odd = Self::zero(self.bits / 2);
        let mut even = Self::zero(self.bits.div_ceil(2));
        for i in (0..bits).filter(|&i| self.bit(i) == Some(true)) {
            let stream = if i % 2 == 0 { &mut even } else { &mut odd };
            stream.set_bit(i / 2);
        }
        (odd, even)
    }

    /// Set bit `index`, which must be below the width for the pattern to stay normalized.
    fn set_bit(&mut self, index: usize) {
        debug_assert!(index < self.bits, "bit {index} is outside int{}", self.bits);
        self.limbs[index / HOST_LIMB_BITS] |= 1 << (index % HOST_LIMB_BITS);
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

/// The operations that name a **reading** of a pattern: each picks a signedness, named in the
/// method. The block above holds what is true of the bits alone.
impl IntBits {
    /// Read this pattern as two's complement.
    ///
    /// Total at every width: the magnitude is a [`BigUint`] and the correction a power of two, so
    /// no host type can overflow.
    #[must_use]
    pub fn to_signed(&self) -> SignedValue {
        let magnitude = SignedValue::from(BigUint::from(self));
        if self.bit(self.bits() - 1) == Some(true) {
            magnitude - two_pow(self.bits())
        } else {
            magnitude
        }
    }

    /// Encode a signed value as a `bits`-wide pattern.
    ///
    /// Total for any `v`: a value that does not fit is reduced modulo `2^bits`, as
    /// [`residue`](crate::residue) requires.
    #[must_use]
    pub fn from_signed(bits: usize, v: &SignedValue) -> Self {
        // A [`SignedValue`]'s bitwise operators treat a negative value as infinitely
        // sign-extended two's complement, so masking to `bits` lands in `0..2^bits` for any `v`.
        let masked = v & (two_pow(bits) - SignedValue::from(1u8));
        Self::from_biguint(bits, masked.magnitude())
    }

    /// Whether `v` is representable in `bits`-bit two's complement.
    ///
    /// # Panics
    ///
    /// On a width of zero, through the boundaries it compares against.
    #[must_use]
    pub fn fits_signed(bits: usize, v: &SignedValue) -> bool {
        *v >= Self::signed_min(bits) && *v <= Self::signed_max(bits)
    }

    /// The largest value a `bits`-wide signed pattern represents.
    ///
    /// # Panics
    ///
    /// On a width of zero: `bits - 1` would wrap in a release build and ask for an unrepresentably
    /// large power of two.
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
    /// On a narrowing. The width extended _from_ is the receiver's, so it needs no check.
    #[must_use]
    pub fn sign_extend(&self, to: usize) -> Self {
        let from = self.bits();
        assert!(to >= from, "cannot sign-extend {from} to {to}");
        let widened = self.cast(to);
        if self.bit(from - 1) == Some(true) {
            // The fill is every bit from `from` up to `to`. At `to == from` the shift moves the
            // ones out of the width and leaves nothing to fill, which is right.
            widened.or(&Self::all_ones(to).shifted_left(from))
        } else {
            widened
        }
    }

    /// The amount this pattern (the receiver) actually shifts an `operand_bits`-wide value by,
    /// where the guard IR failed to reject it.
    ///
    /// `amount % operand_bits`: a **modulo at every width**, not a mask. The two agree only at a
    /// power-of-two width.
    ///
    /// # Panics
    ///
    /// On an `operand_bits` of zero. The VM's version answers `0` there, since its width comes from
    /// a frame cell and a zero-width cell holds nothing; here the width comes from an operand that
    /// [`eval`](crate::eval) has already checked is at least one bit.
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
    /// Not named `cmp`, which would read like `Ord::cmp` returning an
    /// [`Ordering`](std::cmp::Ordering).
    ///
    /// # Panics
    ///
    /// As [`eval`](crate::eval) does, on invalid or unequal widths.
    #[must_use]
    pub fn compare(&self, op: CmpOp, rhs: &Self) -> bool {
        check_widths(self.bits(), rhs.bits());
        match op {
            // Equality needs no reading: two patterns of one width are equal under either reading
            // exactly when their bits are.
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
    fn normalized(bits: usize, mut limbs: Limbs) -> Self {
        assert_eq!(
            limbs.len(),
            Self::limbs_for_bits(bits),
            "a {bits}-bit pattern needs exactly {} limbs",
            Self::limbs_for_bits(bits)
        );
        let top = limbs.len() - 1;
        limbs[top] &= top_limb_mask(bits);
        Self { bits, limbs }
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
    /// An 8-bit pattern holding `value` in two's complement, which is `value`'s bit pattern: `-1`
    /// is every bit set, `i8::MIN` is the sign bit alone.
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
    /// A 16-bit pattern holding `value` in two's complement, which is `value`'s bit pattern: `-1`
    /// is every bit set, `i16::MIN` is the sign bit alone.
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
    /// A 32-bit pattern holding `value` in two's complement, which is `value`'s bit pattern: `-1`
    /// is every bit set, `i32::MIN` is the sign bit alone.
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
    /// A 64-bit pattern holding `value` in two's complement, which is `value`'s bit pattern: `-1`
    /// is every bit set, `i64::MIN` is the sign bit alone.
    fn from(value: i64) -> Self {
        Self::from_u128(64, u128::from(value as u64))
    }
}

// CONVERSIONS TO HOST TYPES
// ================================================================================================

/// Reading a pattern back out, refusing rather than truncating if it does not fit.
///
/// Unsigned targets only: this type holds a bit pattern, not a signed or unsigned reading.
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
/// Private on purpose: `u128` is the widest host word, so every narrower target range-checks the
/// value this returns instead of re-reading the limbs. Supporting a wider host type would change
/// only this function.
impl IntBits {
    /// The pattern as a `u128`, unsigned, or [`DoesNotFit`] if any bit above 127 is set.
    fn as_u128(&self) -> Result<u128, DoesNotFit> {
        if self.limbs.iter().skip(2).any(|&limb| limb != 0) {
            return Err(DoesNotFit { bits: self.bits });
        }
        let lo = self.limbs.first().copied().unwrap_or(0);
        let hi = self.limbs.get(1).copied().unwrap_or(0);
        Ok(u128::from(lo) | (u128::from(hi) << HOST_LIMB_BITS))
    }
}

// FORMATTING
// ================================================================================================

/// Verilog's sized-literal notation.
///
/// Hand-written because a derived one prints the limb vector, which is unreadable at the widths
/// this type exists for: `int16384` would be 256 decimal limbs.
impl fmt::Debug for IntBits {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}'h", self.bits)?;

        // Most significant limb first, as one number: limbs below the leading non-zero one are
        // padded to sixteen digits, so a zero limb cannot swallow the magnitude above it, and a
        // small value in a wide pattern has no leading zeroes.
        let Some(top) = self.limbs.iter().rposition(|&limb| limb != 0) else {
            // Every limb is zero.
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

/// The width-free view of a pattern: its unsigned magnitude, before any reading. It cannot fail.
///
/// Whether it converts to a field element depends on the modulus, which this model does not know.
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

/// The reverse direction needs a width, so it is a named constructor, not a `From`.
impl IntBits {
    /// A `bits`-wide pattern carrying `value`, discarding any bits at or above `bits`: the wrapping
    /// every total evaluator needs.
    #[must_use]
    pub fn from_biguint(bits: usize, value: &BigUint) -> Self {
        let digits: Limbs = value.iter_u64_digits().collect();
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
const fn top_limb_mask(bits: usize) -> u64 {
    match bits % HOST_LIMB_BITS {
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

    #[test]
    fn the_boundary_widths_straddle_the_top_limb_as_documented() {
        // `16384` is exactly 256 limbs, so its top limb is full ...
        assert_eq!(IntBits::limbs_for_bits(16384), 256);
        assert_eq!(top_limb_mask(16384), u64::MAX);

        // ... and `16383` is the widest width with that limb count and a partial top limb.
        assert_eq!(IntBits::limbs_for_bits(16383), 256);
        assert_ne!(top_limb_mask(16383), u64::MAX);
        assert_eq!(IntBits::limbs_for_bits(16385), 257);
    }

    /// Widths worth constructing at, chosen for their top limbs. `16384` is exactly 256 full limbs,
    /// where every `bits % HOST_LIMB_BITS` case vanishes, and `16383` the widest with a partial top
    /// limb; with the same limb count, either alone covers half the boundary.
    const WIDTHS: &[usize] = &[
        1, 2, 3, 7, 8, 63, 64, 65, 96, 127, 128, 129, 191, 192, 193, 1000, 16383, 16384,
    ];

    fn hash_of(v: &IntBits) -> u64 {
        let mut hasher = DefaultHasher::new();
        v.hash(&mut hasher);
        hasher.finish()
    }

    #[test]
    fn small_patterns_stay_inline_across_construction_and_bit_operations() {
        for bits in [1, 63, 64, 65, 127, 128] {
            let ones = IntBits::all_ones(bits);
            for value in [
                IntBits::zero(bits),
                IntBits::from_u128(bits, u128::MAX),
                IntBits::from_limbs(bits, &[u64::MAX; 2]),
                IntBits::from_biguint(bits, &BigUint::from(u128::MAX)),
                ones.clone(),
                ones.and(&ones),
                ones.or(&ones),
                ones.xor(&ones),
                ones.complement(),
                ones.shifted_left(1),
                ones.shifted_right(1),
                ones.bit_range(1, bits),
                IntBits::all_ones(129).cast(bits),
            ] {
                assert!(
                    !value.limbs.spilled(),
                    "{value:?} should use inline storage"
                );
                assert_eq!(value.limb_count(), bits.div_ceil(HOST_LIMB_BITS));
            }
        }
    }

    #[test]
    fn storage_boundary_preserves_values_and_the_slice_hash() {
        for bits in [127, 128, 129, 192, 256, 16384] {
            let value = IntBits::all_ones(bits);
            assert_eq!(value.limbs.spilled(), bits > 128);
            assert_eq!(value.clone(), value);
            assert_eq!(value.cast(128), IntBits::all_ones(bits.min(128)).cast(128));
            assert_eq!(IntBits::from_biguint(bits, &BigUint::from(&value)), value);

            // Keep the same hash input as the original boxed-slice representation. Integer
            // constants are interned, so storage strategy must not change their identity.
            let mut hasher = DefaultHasher::new();
            bits.hash(&mut hasher);
            value.limbs().hash(&mut hasher);
            assert_eq!(hash_of(&value), hasher.finish());
        }
        let narrow = IntBits::all_ones(128);
        let wide = narrow.cast(129);
        assert_eq!(wide.limbs(), &[u64::MAX, u64::MAX, 0]);
        assert_eq!(wide.cast(128), narrow);
        assert_ne!(wide, narrow);
    }

    /// The round-trip's reference, at the widths a `u128` can express.
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

        // Across a limb boundary the digits read as one number: the low limb is padded to sixteen
        // digits, so a zero low limb cannot swallow the high limb's magnitude.
        assert_eq!(
            format!("{:?}", IntBits::from_u128(128, 1u128 << 64)),
            "128'h10000000000000000"
        );
        assert_eq!(
            format!("{:?}", IntBits::from_limbs(129, &[0, 0, 1])),
            "129'h100000000000000000000000000000000"
        );

        // A value below the top limb is written without the leading zeroes the limb vector holds;
        // the width prefix gives the width.
        assert_eq!(format!("{:?}", IntBits::from_u128(128, 1)), "128'h1");
        assert_eq!(format!("{:?}", IntBits::from_u128(16384, 42)), "16384'h2a");

        // Zero has no significant limb, at any width.
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
        // 16383 and 16384 share a limb count and differ only in the top limb's mask, so the count
        // alone does not pin the representation down.
        assert_eq!(IntBits::zero(16383).limb_count(), 256);
        assert_eq!(IntBits::zero(16384).limb_count(), 256);
    }

    #[test]
    fn two_widths_sharing_a_limb_count_are_different_patterns() {
        // Why the width is stored rather than inferred: each pair has the same limbs, so without
        // the width they would compare equal, and the interning bimap needs one value per width,
        // not per limb count.
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
        // A pattern has no sign, so reading one out gives the unsigned reading: `-1i8` goes in,
        // `255u8` comes back. Recovering `-1` means naming the reading, which is `to_signed`'s job.
        let minus_one = IntBits::from(-1i8);
        assert_eq!(u8::try_from(&minus_one), Ok(255));
        assert_eq!(u32::try_from(&minus_one), Ok(255));
        assert_eq!(minus_one.to_signed(), crate::SignedValue::from(-1));
    }

    #[test]
    fn a_value_that_will_not_fit_the_target_is_refused() {
        // Too wide for the target, though it fits a `u128`.
        let big = IntBits::from(300u16);
        assert_eq!(u8::try_from(&big), Err(DoesNotFit { bits: 16 }));
        assert_eq!(u16::try_from(&big), Ok(300));

        // A negative value has no unsigned reading.
        assert_eq!(
            u8::try_from(&IntBits::from(-1i16)),
            Err(DoesNotFit { bits: 16 })
        );

        // The error carries the pattern's width, which the call site cannot see.
        assert_eq!(
            u64::try_from(&IntBits::from_limbs(256, &[0, 0, 0, 1])),
            Err(DoesNotFit { bits: 256 })
        );
    }

    #[test]
    fn the_error_names_the_width_that_would_not_convert() {
        // The width is the whole message: the call site knows which host type it asked for, but
        // not the width of the pattern it asked.
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
        // `-1` is every bit set, at the type's width and no wider. This catches a lost sign or a
        // wrong width, not an over-wide sign extension, which `from_u128` masks away.
        assert_eq!(IntBits::from(-1i8), IntBits::from(u8::MAX));
        assert_eq!(IntBits::from(-1i16), IntBits::from(u16::MAX));
        assert_eq!(IntBits::from(-1i32), IntBits::from(u32::MAX));
        assert_eq!(IntBits::from(-1i64), IntBits::from(u64::MAX));

        // `INT_MIN` is the sign bit alone.
        assert_eq!(IntBits::from(i8::MIN), IntBits::from(1u8 << 7));
        assert_eq!(IntBits::from(i16::MIN), IntBits::from(1u16 << 15));
        assert_eq!(IntBits::from(i32::MIN), IntBits::from(1u32 << 31));
        assert_eq!(IntBits::from(i64::MIN), IntBits::from(1u64 << 63));

        // As limbs too, so the assertion does not rest on `From<u8>`.
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
            // All ones in every limb the width reserves: the worst case for the mask.
            let saturated = vec![u64::MAX; bits.div_ceil(64)];
            let v = IntBits::from_limbs(bits, &saturated);

            for i in 0..bits {
                assert_eq!(v.bit(i), Some(true), "bit {i} of {bits} was cleared");
            }

            // Checked against the limbs directly, NOT through `bit`, which answers `None` at or
            // above the width whatever the limbs hold, and so would pass on a dirty top limb.
            let top = v.limbs().last().expect("a width has at least one limb");
            assert_eq!(
                top & !top_limb_mask(bits),
                0,
                "the top limb of {bits} kept bits above the width"
            );

            // Separately, `bit` refuses to read there.
            for i in bits..bits.div_ceil(64) * 64 {
                assert_eq!(v.bit(i), None, "bit {i} is outside a {bits}-bit pattern");
            }
        }
    }

    #[test]
    fn dirt_above_the_width_does_not_change_the_value() {
        // Canonicity, as it gets violated: a caller passes a word left dirty above the width, and
        // the result must be indistinguishable from the clean one. Otherwise the interning bimap
        // splits one value into two `ValueId`s.
        for &bits in &[1usize, 3, 8, 63, 64, 65, 127, 128] {
            let clean = IntBits::from_u128(bits, 1);
            let dirty = IntBits::from_u128(bits, 1 | !mask_u128(bits));
            assert_eq!(clean, dirty, "at {bits} bits");
            assert_eq!(hash_of(&clean), hash_of(&dirty), "at {bits} bits");
        }

        // Also through `from_limbs`, at widths `from_u128` cannot reach.
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
        // At `bits % 64 == 0`, a naive `(1 << (bits % 64)) - 1` is a zero mask, which would
        // silently clear the top limb of every whole-limb width.
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
        // Short: the missing limbs read as zero.
        let wide = IntBits::from_limbs(256, &[7]);
        assert_eq!(wide.limbs(), &[7, 0, 0, 0]);

        // Long: limbs past the width are dropped, like bits past the width within a limb.
        // Truncating rather than rejecting matches `from_u128`.
        let narrow = IntBits::from_limbs(64, &[7, u64::MAX, u64::MAX]);
        assert_eq!(narrow.limbs(), &[7]);
    }

    #[test]
    fn the_field_read_is_the_packed_one_at_the_host_limb_width() {
        // `from_field_limbs` skips `from_packed_limbs`, since at the host limb width the packed
        // limbs are the pattern's words. This checks the two agree.
        //
        // The slice lengths are shorter than, equal to and longer than the width, where the two
        // routes' truncation and zero-extension could differ.
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
        // Why the conversion is fallible: a caller taking a host word out of a wider pattern must
        // decide what that means, and the refusal forces the decision.
        let mut limbs = vec![0u64; 4];
        limbs[3] = 1;
        assert!(u128::try_from(&IntBits::from_limbs(256, &limbs)).is_err());

        // A wide _width_ with a narrow _value_ still fits: what matters is the value, not the
        // width.
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

            // At one bit, one and all ones are the same pattern, and the predicates must agree.
            assert_eq!(one.is_all_ones(), bits == 1, "one vs all ones at {bits}");
            assert_eq!(all_ones.is_one(), bits == 1, "all ones vs one at {bits}");
        }

        // All-ones is per width: the same limbs are all ones at one width and not at another.
        assert!(IntBits::from_limbs(64, &[u64::MAX]).is_all_ones());
        assert!(!IntBits::from_limbs(128, &[u64::MAX, 0]).is_all_ones());
        // And it must read every limb, not just the top one.
        assert!(!IntBits::from_limbs(128, &[0, u64::MAX]).is_all_ones());
        // Likewise `is_one` must look past the low limb, or any value whose low limb is 1 passes.
        assert!(!IntBits::from_limbs(128, &[1, 1]).is_one());
    }

    #[test]
    fn an_index_too_wide_for_the_machine_is_refused_rather_than_truncated() {
        // The point of the `usize` conversion: a value above the host's address width must not
        // come back as an unrelated small number that passes a bounds check it should fail.
        let huge = IntBits::from_u128(128, (u64::MAX as u128) + 1);
        assert_eq!(usize::try_from(&huge), Err(DoesNotFit { bits: 128 }));

        assert_eq!(usize::try_from(&IntBits::from_u128(128, 7)), Ok(7));
        assert_eq!(usize::try_from(IntBits::from(3u8)), Ok(3));
    }

    #[test]
    fn distinct_values_at_one_width_stay_distinct() {
        // The other half of canonicity: masking must not merge values that differ below the
        // width. Only this catches a mask applied at the wrong width.
        for &bits in &[3usize, 8, 64, 65, 128] {
            let m = mask_u128(bits);
            let values = [0u128, 1, 2, 3, m, m - 1, m ^ 1];

            // A `Vec`: `IntBits` has no `Ord` for a `BTreeSet`, and `HashSet` is disallowed
            // workspace-wide for its iteration order.
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
        /// Canonicity over the whole space, not just the corners: one value at one width has
        /// exactly one representation, however it is reached.
        #[test]
        fn one_value_at_one_width_has_one_spelling(
            width_index in 0..WIDTHS.len(),
            value in any::<u128>(),
            dirt in any::<u128>(),
        ) {
            let bits = WIDTHS[width_index];
            let direct = IntBits::from_u128(bits, value);

            // Reached again through `from_limbs`, with arbitrary dirt above the width.
            let mut limbs = direct.limbs().to_vec();
            let top = limbs.len() - 1;
            limbs[top] |= (dirt as u64) & !top_limb_mask(bits);
            let indirect = IntBits::from_limbs(bits, &limbs);

            prop_assert_eq!(&direct, &indirect);
            prop_assert_eq!(hash_of(&direct), hash_of(&indirect));
        }

        /// Construction is idempotent: feeding a pattern's limbs back in changes nothing.
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

    /// Widths around every limb boundary up to 128 bits: a sub-limb width, both sides of the first
    /// boundary, an odd one, and both sides of the second.
    const CORNERS: [usize; 8] = [1, 5, 63, 64, 65, 96, 127, 128];

    /// The `u128` a pattern denotes, to check the limb code against host semantics.
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
        // Both limbs differ, and the top limb is partial at the odd widths, so a mask applied to
        // the wrong limb shows.
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
        // A full `assert!`: in release a narrowing would fill from the source's sign bit and give
        // neither the source nor its low bits.
        let _ = IntBits::from(-1i32).sign_extend(8);
    }

    #[test]
    fn a_signed_reading_is_total_above_every_lowering() {
        // A signed reading is total: the model caps no width.
        assert_eq!(
            IntBits::from_u128(128, 1).to_signed(),
            SignedValue::from(1u8)
        );
        assert_eq!(
            IntBits::from_u128(128, u128::MAX).to_signed(),
            SignedValue::from(-1i8),
            "the top bit is the sign at 128 bits, not a large magnitude"
        );

        // Past any host type, where only the `BigInt` reading answers: bit 999 alone is `INT_MIN`
        // at 1000 bits.
        let min = IntBits::from_signed(1000, &IntBits::signed_min(1000));
        assert_eq!(min.to_signed(), IntBits::signed_min(1000));
        assert!(min.to_signed() < SignedValue::from(0u8));
    }

    #[test]
    #[should_panic(expected = "at least a sign bit")]
    fn the_signed_bounds_refuse_a_zero_width() {
        // `bits - 1` wraps in a release build, so without this gate it would ask for
        // `2^usize::MAX` instead of panicking.
        let _ = IntBits::signed_max(0);
    }

    #[test]
    #[should_panic(expected = "a width of at least one bit")]
    fn a_shift_amount_refuses_to_reduce_against_no_width() {
        // The VM's version is total here because its width comes from a frame cell. This width
        // comes from an operand `eval` checked, so a zero means a caller skipped that check.
        let _ = IntBits::from(1u8).reduced_shift_amount(0);
    }

    #[test]
    fn a_shift_amount_reduces_by_the_width_it_is_given() {
        // An in-range amount survives, and an out-of-range one is reduced modulo the width, not
        // masked: at seven bits a mask by 6 would give `0` for both of the first two.
        assert_eq!(IntBits::from(1u8).reduced_shift_amount(7), 1);
        assert_eq!(IntBits::from(8u8).reduced_shift_amount(7), 1);
        assert_eq!(IntBits::from(9u8).reduced_shift_amount(8), 1);
        assert_eq!(IntBits::from(1u8).reduced_shift_amount(1), 0);
    }

    #[test]
    #[should_panic(expected = "two patterns of one width")]
    fn a_bitwise_operation_refuses_two_widths() {
        // 32 and 64 share a limb, but their bit positions do not correspond, so this is refused.
        let _ = IntBits::from(1u32).and(&IntBits::from(1u64));
    }

    #[test]
    #[should_panic(expected = "at least one bit wide")]
    fn a_cast_to_no_width_is_refused() {
        // Operations that take a width share the constructors' lower bound rather than returning an
        // empty pattern.
        let _ = IntBits::from(1u8).cast(0);
    }

    #[test]
    #[should_panic(expected = "at least one bit wide")]
    fn an_empty_bit_range_is_refused() {
        let _ = IntBits::from(0xABCDu16).bit_range(4, 0);
    }

    #[test]
    fn a_cast_to_the_width_already_held_is_the_identity() {
        // The fast path must agree with the general one. It is the common case: both constant
        // folders spell `BitRange` as `bit_range(..).cast(self.bits())`.
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
        // The case the guarded carry exists for: `limb << 64` overflows rather than giving zero,
        // so a shift by an exact multiple of the limb width must take the other path.
        let v = IntBits::from_u128(128, u128::MAX);
        assert_eq!(as_u128(&v.shifted_left(64)), u128::MAX << 64);
        assert_eq!(as_u128(&v.shifted_right(64)), u128::MAX >> 64);
    }

    // SPREAD AND UNSPREAD
    // --------------------------------------------------------------------------------------

    /// The mask ladder the backends run: an oracle for the bit-by-bit definition at host-word
    /// widths.
    fn ladder_spread(v: u64) -> u128 {
        let mut x = u128::from(v);
        x = (x | (x << 32)) & 0x0000_0000_FFFF_FFFF_0000_0000_FFFF_FFFF;
        x = (x | (x << 16)) & 0x0000_FFFF_0000_FFFF_0000_FFFF_0000_FFFF;
        x = (x | (x << 8)) & 0x00FF_00FF_00FF_00FF_00FF_00FF_00FF_00FF;
        x = (x | (x << 4)) & 0x0F0F_0F0F_0F0F_0F0F_0F0F_0F0F_0F0F_0F0F;
        x = (x | (x << 2)) & 0x3333_3333_3333_3333_3333_3333_3333_3333;
        (x | (x << 1)) & 0x5555_5555_5555_5555_5555_5555_5555_5555
    }

    #[test]
    fn a_spread_gives_the_values_the_noir_corpus_asserts() {
        // `noir_tests/spread_and_cast` asserts these through `spread::<N>` on a `u32`.
        for (value, bits, want) in [(5, 5, 17), (13, 5, 81), (100, 7, 5136), (77, 7, 4177)] {
            let got = IntBits::from_u128(32, value).spread(bits);
            assert_eq!(
                got,
                IntBits::from_u128(64, want),
                "spread::<{bits}>({value})"
            );
        }
    }

    #[test]
    fn bits_above_the_read_width_are_discarded() {
        let dirty = IntBits::from_u128(8, 0xFF);
        let clean = IntBits::from_u128(8, 0x0F);
        assert_eq!(dirty.spread(4), clean.spread(4));
        assert_eq!(dirty.spread(4), IntBits::from_u128(16, 0x55));

        // The unspread discards bits above its read too: a spread `0x0F` plus two dirty bits.
        let spread_dirty = IntBits::from_u128(16, 0xC055);
        assert_eq!(
            spread_dirty.unspread(8),
            (IntBits::zero(8), IntBits::from_u128(8, 0x0F))
        );
    }

    #[test]
    fn an_odd_width_unspreads_into_a_wider_even_stream() {
        // Seven bits with the four even ones set and the three odd ones clear.
        let (odd, even) = IntBits::from_u128(7, 0b101_0101).unspread(7);
        assert_eq!(odd, IntBits::zero(3));
        assert_eq!(even, IntBits::from_u128(4, 0b1111));

        // The odd ones set, which fill the narrower stream exactly.
        let (odd, even) = IntBits::from_u128(7, 0b010_1010).unspread(7);
        assert_eq!(odd, IntBits::from_u128(3, 0b111));
        assert_eq!(even, IntBits::zero(4));
    }

    #[test]
    #[should_panic(expected = "a spread reads 1..=8 bits of an int8, not 0")]
    fn a_spread_of_no_bits_is_refused() {
        let _ = IntBits::from_u128(8, 1).spread(0);
    }

    #[test]
    #[should_panic(expected = "a spread reads 1..=8 bits of an int8, not 9")]
    fn a_spread_wider_than_its_container_is_refused() {
        let _ = IntBits::from_u128(8, 1).spread(9);
    }

    #[test]
    #[should_panic(expected = "an unspread reads 2..=8 bits of an int8, not 9")]
    fn an_unspread_wider_than_its_container_is_refused() {
        let _ = IntBits::from_u128(8, 1).unspread(9);
    }

    #[test]
    #[should_panic(expected = "an unspread reads 2..=8 bits of an int8, not 1")]
    fn an_unspread_reading_no_odd_bit_is_refused() {
        let _ = IntBits::from_u128(8, 1).unspread(1);
    }

    #[test]
    #[should_panic(expected = "an int1 has no odd stream")]
    fn a_single_bit_has_no_unspread() {
        let _ = IntBits::from_u128(1, 1).unspread(1);
    }

    proptest! {
        /// At every width a host word holds, the definition is the ladder, reading only `bits`.
        #[test]
        fn a_spread_is_the_mask_ladder_on_a_host_word(
            width in 1usize..=64,
            value in any::<u64>(),
            bits_seed in any::<usize>(),
        ) {
            let bits = 1 + bits_seed % width;
            let got = IntBits::from_u128(width, u128::from(value)).spread(bits);
            let read = value & (u64::MAX >> (64 - bits));
            prop_assert_eq!(got, IntBits::from_u128(2 * width, ladder_spread(read)));
        }

        /// Two spreads, one moved onto the odd positions, separate again at every width: the
        /// identity a bitwise lowering relies on.
        #[test]
        fn an_unspread_separates_two_interleaved_spreads(
            width_index in 0..WIDTHS.len(),
            odd_limbs in prop::collection::vec(any::<u64>(), 256),
            even_limbs in prop::collection::vec(any::<u64>(), 256),
        ) {
            let bits = WIDTHS[width_index];
            let odd = IntBits::from_limbs(bits, &odd_limbs);
            let even = IntBits::from_limbs(bits, &even_limbs);
            let woven = odd.spread(bits).shifted_left(1).or(&even.spread(bits));
            prop_assert_eq!(woven.unspread(2 * bits), (odd, even));
        }

        /// The streams put back together give the original value, at odd and even widths.
        #[test]
        fn an_unspread_at_any_width_loses_nothing(
            width_index in 0..WIDTHS.len(),
            limbs in prop::collection::vec(any::<u64>(), 256),
        ) {
            let bits = WIDTHS[width_index];
            // A single bit has no odd stream.
            prop_assume!(bits >= 2);
            let value = IntBits::from_limbs(bits, &limbs);
            let (odd, even) = value.unspread(bits);
            prop_assert_eq!(odd.bits(), bits / 2);
            prop_assert_eq!(even.bits(), bits.div_ceil(2));

            // At the original width, the even stream's spread sits on the even positions, and the
            // odd stream's moves one place up onto the odd ones.
            let rebuilt = odd
                .spread(odd.bits())
                .cast(2 * bits)
                .shifted_left(1)
                .or(&even.spread(even.bits()).cast(2 * bits))
                .cast(bits);
            prop_assert_eq!(rebuilt, value);
        }
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
        // As with `from_limbs`, a constructor that takes a width discards bits above it rather
        // than refusing, which makes it the wrapping a total evaluator needs.
        let wide = BigUint::from(u128::MAX);
        assert_eq!(IntBits::from_biguint(8, &wide), IntBits::from(0xFFu8));
        assert_eq!(IntBits::from_biguint(64, &wide), IntBits::from(u64::MAX));
    }
}
