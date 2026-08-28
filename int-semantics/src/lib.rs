//! What an integer operation _means_ in Mavros.
//!
//! Mavros evaluates integer arithmetic in lots of places, spanning from compile-time evaluation to
//! the VM and the WASM backend. This crate exists to avoid the risk of all of those sites drifting
//! from the agreed-upon set of semantics that match Noir.
//!
//! We split this executable specification into two portions:
//!
//! - [`eval`] is the answer to **what Noir specifies**, including that the program may be rejected.
//! - [`residue`] is the answer to what a total evaluator must produce anyway, because most backends
//!   have no way to reject in the middle of an expression. When it returns [`None`] it is calling
//!   the input undefined in the LLVM sense.
//!
//! This distinction exists because the guard IR is supposed to have rejected the program before the
//! opcode runs, but if it _is_ running it still has to put a bit pattern in a register somewhere.
//!
//! The two are tied together by the invariant that **an accepted evaluation and its residue never disagree**.
//!
//! ```
//! # use mavros_int_semantics::{eval, residue, IntOp, Outcome, Sign};
//! // Accepted, so both agree.
//! assert_eq!(
//!     eval(IntOp::Add, Sign::Unsigned, 8, 1, 8, 2),
//!     Outcome::Value(3)
//! );
//! assert_eq!(residue(IntOp::Add, Sign::Unsigned, 8, 1, 8, 2), Some(3));
//!
//! // Rejected by Noir, but every backend wraps, so the residue is pinned.
//! assert!(matches!(
//!     eval(IntOp::Add, Sign::Unsigned, 8, 200, 8, 100),
//!     Outcome::Rejected(_)
//! ));
//! assert_eq!(
//!     residue(IntOp::Add, Sign::Unsigned, 8, 200, 8, 100),
//!     Some(44)
//! );
//!
//! // Rejected, and the backends do not agree on what to do anyway.
//! assert!(matches!(
//!     eval(IntOp::Div, Sign::Unsigned, 8, 1, 8, 0),
//!     Outcome::Rejected(_)
//! ));
//! assert_eq!(residue(IntOp::Div, Sign::Unsigned, 8, 1, 8, 0), None);
//! ```
//!
//! # Values are Raw
//!
//! Every value in and out is a **raw bit pattern** masked to its width, never a signed integer. The
//! value carries no sign, so it is up to the _operation_ to enforce interpretation instead.
//!
//! That pattern is spelled [`Raw`] rather than `u128`, and its two's-complement reading is spelled
//! [`SignedValue`] rather than `i128`, so that the host types stay an implementation detail of this
//! crate rather than something every conformance test and delegating folder has written into its
//! own signatures.
//!
//! # The Source of the Rules
//!
//! The rules are drawn from the semantics encoded in the currently-pinned version of Noir, and are
//! found in the following source locations in the Noir tree:
//!
//! - `noirc_evaluator/src/ssa/ir/instruction/binary.rs::eval_constant_binary_op`, which defines the
//!   constant-folded operations that are a `Failure`.
//! - `noirc_evaluator/src/ssa/opt/remove_bit_shifts.rs`, which handles shift lowering. Its module
//!   doc states the rule this crate encodes: "In all cases, if the shift amount is equal to or
//!   exceeds the operand's number of bits, the result will be a constrain failure".
//! - `noirc_evaluator/src/ssa/opt/expand_signed_math.rs`, which handles the fact that signed
//!   division truncates toward zero, the remainder takes the dividend's sign, and that
//!   `INT_MIN / -1` is rejected.
//! - `noirc_evaluator/src/brillig/brillig_gen/brillig_instructions/brillig_binary.rs`, which
//!   describes the behavior in unconstrained code and that it gets `add_overflow_check` and
//!   `bit_shift_overflow` too.
//!
//! A rejection is a **runtime** failure in Noir, not a compile error:
//! `remove_unreachable_instructions.rs` replaces a statically-failing operation with a failing
//! constrain. So [`Outcome::Rejected`] means "this program must not produce a proof", not "this
//! must not compile".

#![forbid(unsafe_code)]

pub mod corners;
pub mod corpus;
pub mod register;

// CONSTANTS
// ================================================================================================

/// The widest integer any operation may act on.
pub const MAX_BITS: usize = 128;

/// The widest integer a _signed_ operation may act on.
///
/// A bound on operations, not on types: a 128-bit pattern is perfectly legal, it currently has no
/// signed reading here, because the signed lowerings and the VM's `sdiv_int`/`slt_int` are 64-bit.
/// Mirrors `MAX_SUPPORTED_SIGNED_BITS` in `hlssa::type_system`.
pub const MAX_SIGNED_BITS: usize = 64;

// PAYLOAD TYPES
// ================================================================================================

/// A raw integer bit pattern, always masked to its operand's width.
///
/// An alias rather than a spelled-out host type because the host type is an implementation detail:
/// `u128` covers every width Mavros supports today ([`MAX_BITS`]), and the day an integer type
/// exceeds that, this becomes a bignum without a single consumer's signature changing.
///
/// The alias is deliberately confined to _signatures_. Expressions that genuinely depend on the
/// host — `1u128 << k`, `checked_mul`, a `proptest` strategy over `u128` — stay spelled out, so
/// that widening the alias produces compile errors at exactly the places that need real thought
/// rather than silently appearing to be free.
pub type Raw = u128;

/// The two's-complement _reading_ of a [`Raw`], as a mathematical integer.
///
/// A distinct alias because it is a number rather than a pattern: it goes negative, and it needs a
/// bit more room than the width it decodes so that [`signed_min`] is representable at that width.
/// Signed operations are capped at [`MAX_SIGNED_BITS`], well inside `i128`, so this one has far
/// more headroom than [`Raw`] does.
pub type SignedValue = i128;

// INTEGER OPERATIONS MODEL
// ================================================================================================

/// A sign-agnostic binary integer operation.
///
/// The sign is [`Sign`], carried separately, as not all operations have a signed reading.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum IntOp {
    Add,
    Sub,
    Mul,
    Div,
    Rem,
    And,
    Or,
    Xor,
    Shl,
    Shr,
}

impl IntOp {
    /// Every operation, for exhaustive sweeps.
    pub const ALL: [IntOp; 10] = [
        IntOp::Add,
        IntOp::Sub,
        IntOp::Mul,
        IntOp::Div,
        IntOp::Rem,
        IntOp::And,
        IntOp::Or,
        IntOp::Xor,
        IntOp::Shl,
        IntOp::Shr,
    ];

    /// Whether the signed and unsigned readings of this operation can give different answers.
    #[must_use]
    pub const fn reading_matters(self) -> bool {
        !matches!(self, IntOp::And | IntOp::Or | IntOp::Xor)
    }

    /// Whether this operation reads its right operand at a width of its **own**.
    ///
    /// True for the two shifts, whose amount carries its own type, and false for everything else,
    /// where the two operands are one width and a caller offering two widths is offering a
    /// contradiction. The width checks every entry point performs are what act on the answer.
    #[must_use]
    pub const fn has_own_amount_width(self) -> bool {
        matches!(self, IntOp::Shl | IntOp::Shr)
    }
}

/// How to read the operand patterns.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum Sign {
    Unsigned,
    Signed,
}

impl Sign {
    pub const ALL: [Sign; 2] = [Sign::Unsigned, Sign::Signed];

    #[must_use]
    pub const fn is_signed(self) -> bool {
        matches!(self, Sign::Signed)
    }
}

// REJECTION REASONS
// ================================================================================================

/// Why Noir rejects an evaluation.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum Reject {
    /// The mathematical result does not fit the operand width.
    ///
    /// This covers underflow too: a difference below zero does not fit either.
    Overflow,

    /// A zero divisor, for either `Div` or `Rem`.
    DivByZero,

    /// Signed `INT_MIN / -1`, whose quotient is one past the top of the type.
    ///
    /// `INT_MIN % -1` is rejected for the same reason even though the remainder it would produce
    /// (`0`) is representable as Noir defines the remainder in terms of that same quotient.
    DivOverflow,

    /// A shift amount at or above the operand width, or a negative one.
    ///
    /// Noir types a shift's amount as the _value's_ own type (its elaborator unifies the right
    /// operand with the left), and `remove_bit_shifts::enforce_bitshift_rhs_lt_bit_size` casts that
    /// amount to unsigned before comparing it against the width. A negative amount therefore reads
    /// as a very large one and fails the same test.
    ShiftAmount,
}

// EVALUATION OUTCOMES
// ================================================================================================

/// What an evaluation does.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum Outcome {
    /// Noir computes this raw pattern, masked to the operand width.
    Value(Raw),

    /// Noir rejects the program at runtime.
    Rejected(Reject),
}

impl Outcome {
    /// The value, if this evaluation is accepted.
    #[must_use]
    pub const fn value(self) -> Option<Raw> {
        match self {
            Outcome::Value(v) => Some(v),
            Outcome::Rejected(_) => None,
        }
    }

    #[must_use]
    pub const fn is_rejected(self) -> bool {
        matches!(self, Outcome::Rejected(_))
    }
}

// BIT-PATTERN HELPERS
// ================================================================================================

/// All-ones mask for the low `bits` bits, total at the edges.
#[must_use]
pub fn mask(bits: usize) -> Raw {
    if bits == 0 {
        0
    } else if bits >= MAX_BITS {
        u128::MAX
    } else {
        (1u128 << bits) - 1
    }
}

/// Read a `bits`-wide raw pattern as two's complement.
///
/// Bits above `bits` are discarded first, so this is correct on a cell that some earlier step left
/// dirty above its declared width.
#[must_use]
pub fn decode_signed(bits: usize, raw: Raw) -> SignedValue {
    debug_assert!((1..=MAX_SIGNED_BITS).contains(&bits));
    let raw = raw & mask(bits);
    if (raw >> (bits - 1)) & 1 == 1 {
        (raw as i128) - (1i128 << bits)
    } else {
        raw as i128
    }
}

/// Encode a signed value as a `bits`-wide raw pattern.
#[must_use]
pub fn encode_signed(bits: usize, v: SignedValue) -> Raw {
    (v as u128) & mask(bits)
}

/// Whether `v` is representable in `bits`-bit two's complement.
#[must_use]
pub fn fits_signed(bits: usize, v: SignedValue) -> bool {
    debug_assert!((1..=MAX_SIGNED_BITS).contains(&bits));
    v >= -(1i128 << (bits - 1)) && v < (1i128 << (bits - 1))
}

/// The largest value a `bits`-wide signed pattern represents.
#[must_use]
pub fn signed_max(bits: usize) -> SignedValue {
    (1i128 << (bits - 1)) - 1
}

/// The smallest value a `bits`-wide signed pattern represents.
#[must_use]
pub fn signed_min(bits: usize) -> SignedValue {
    -(1i128 << (bits - 1))
}

/// Widen a `from`-bit two's-complement pattern to `to` bits, replicating the sign bit.
///
/// Operates on the pattern. Narrowing is not sign extension and is rejected rather than silently
/// truncating.
#[must_use]
pub fn sign_extend(v: Raw, from: usize, to: usize) -> Raw {
    debug_assert!(from > 0 && to >= from, "cannot sign-extend {from} to {to}");
    let v = v & mask(from);
    if (v >> (from - 1)) & 1 == 1 {
        v | (mask(to) ^ mask(from))
    } else {
        v
    }
}

/// The amount a `bits`-wide shift actually applies to a pattern the guard IR failed to reject.
///
/// `amount & (bits - 1)`. Note it is a **mask, not a modulo**, which is fine and deliberate as the
/// result is a submask of `bits - 1` and so always below `bits`, which is all a backstop needs to
/// guarantee.
#[must_use]
pub fn masked_shift_amount(amount: Raw, bits: usize) -> u32 {
    (amount & (bits as u128).saturating_sub(1)) as u32
}

// THE MODEL
// ================================================================================================

/// Check the width arguments every entry point shares.
///
/// `op` is [`None`] for the entry points that have no [`IntOp`] at all, which is [`cmp`]; those
/// read both operands at one width by construction and pass `bits` twice.
fn check_widths(op: Option<IntOp>, sign: Sign, bits: usize, rhs_bits: usize) {
    assert!(
        (1..=MAX_BITS).contains(&bits),
        "operand width {bits} is outside 1..={MAX_BITS}"
    );
    assert!(
        (1..=MAX_BITS).contains(&rhs_bits),
        "right-operand width {rhs_bits} is outside 1..={MAX_BITS}"
    );
    assert!(
        !sign.is_signed() || bits <= MAX_SIGNED_BITS,
        "signed operation on a {bits}-bit value exceeds {MAX_SIGNED_BITS}"
    );
    assert!(
        rhs_bits == bits || op.is_some_and(IntOp::has_own_amount_width),
        "{op:?} was given operand widths {bits} and {rhs_bits}; only a shift reads its right operand at a width of its own"
    );
}

/// Noir's specification for one integer operation.
///
/// `lhs` and `rhs` are raw patterns and anything above their declared width is discarded, so a
/// caller need not pre-mask. `rhs_bits` is separate from `bits` because a shift amount legitimately
/// has its own width.
///
/// # Panics
///
/// If the widths are outside `1..=128`, or a signed op is requested above [`MAX_SIGNED_BITS`].
/// Those are compiler bugs rather than program errors, so they are not [`Reject`] variants.
#[must_use]
pub fn eval(op: IntOp, sign: Sign, bits: usize, lhs: Raw, rhs_bits: usize, rhs: Raw) -> Outcome {
    check_widths(Some(op), sign, bits, rhs_bits);
    let lhs = lhs & mask(bits);
    let rhs = rhs & mask(rhs_bits);

    // Bitwise operations have no reading and cannot fail, so they short-circuit both arms below.
    match op {
        IntOp::And => return Outcome::Value(lhs & rhs),
        IntOp::Or => return Outcome::Value(lhs | rhs),
        IntOp::Xor => return Outcome::Value(lhs ^ rhs),
        _ => {}
    }

    if matches!(op, IntOp::Shl | IntOp::Shr) {
        return eval_shift(op, sign, bits, lhs, rhs);
    }

    if sign.is_signed() {
        eval_signed_arith(op, bits, decode_signed(bits, lhs), decode_signed(bits, rhs))
    } else {
        eval_unsigned_arith(op, bits, lhs, rhs)
    }
}

fn eval_unsigned_arith(op: IntOp, bits: usize, lhs: Raw, rhs: Raw) -> Outcome {
    let checked = match op {
        IntOp::Add => lhs.checked_add(rhs),
        IntOp::Sub => lhs.checked_sub(rhs),
        IntOp::Mul => lhs.checked_mul(rhs),
        IntOp::Div | IntOp::Rem => {
            if rhs == 0 {
                return Outcome::Rejected(Reject::DivByZero);
            }
            // A quotient or remainder of two in-range values is always in range, so neither of
            // these can fail the width check below.
            Some(if op == IntOp::Div {
                lhs / rhs
            } else {
                lhs % rhs
            })
        }
        IntOp::And | IntOp::Or | IntOp::Xor | IntOp::Shl | IntOp::Shr => {
            unreachable!("handled before this function")
        }
    };

    // `checked_*` catches only a 128-bit overflow; the operand width is usually much narrower, so
    // the width test is what actually rejects. Underflow arrives here as `checked_sub` failing.
    match checked {
        Some(v) if v <= mask(bits) => Outcome::Value(v),
        _ => Outcome::Rejected(Reject::Overflow),
    }
}

fn eval_signed_arith(op: IntOp, bits: usize, lhs: SignedValue, rhs: SignedValue) -> Outcome {
    let checked = match op {
        IntOp::Add => lhs.checked_add(rhs),
        IntOp::Sub => lhs.checked_sub(rhs),
        IntOp::Mul => lhs.checked_mul(rhs),
        IntOp::Div | IntOp::Rem => {
            if rhs == 0 {
                return Outcome::Rejected(Reject::DivByZero);
            }
            if lhs == signed_min(bits) && rhs == -1 {
                return Outcome::Rejected(Reject::DivOverflow);
            }

            // Rust's `/` and `%` are Noir's: truncation toward zero, remainder signed like the
            // dividend. `expand_signed_math` builds the same thing out of a magnitude division
            // plus a sign fix-up.
            Some(if op == IntOp::Div {
                lhs / rhs
            } else {
                lhs % rhs
            })
        }
        IntOp::And | IntOp::Or | IntOp::Xor | IntOp::Shl | IntOp::Shr => {
            unreachable!("handled before this function")
        }
    };

    match checked {
        Some(v) if fits_signed(bits, v) => Outcome::Value(encode_signed(bits, v)),
        _ => Outcome::Rejected(Reject::Overflow),
    }
}

/// Shifts, whose rules differ from the standard arithmetic ones in both directions.
///
/// Stricter about the amount: at or above the width is a rejection, where an arithmetic operand can
/// hold any pattern its width allows. Laxer about the result: a `<<` that pushes bits off the top
/// **wraps** and is not an error at all. `remove_bit_shifts` lowers it as a multiply in the field
/// followed by a truncation, so the discarded bits are discarded by construction.
fn eval_shift(op: IntOp, sign: Sign, bits: usize, lhs: Raw, rhs: Raw) -> Outcome {
    // The amount is read as an unsigned magnitude whatever the operation's sign.
    if rhs >= bits as u128 {
        return Outcome::Rejected(Reject::ShiftAmount);
    }
    let amount = rhs as u32;

    let value = match (op, sign) {
        // One map on the bit pattern whatever the reading, then truncate. The result can change
        // sign (`64i8 << 1` is `-128`), which is why this stays on the raw pattern rather than
        // asking whether the mathematical product fits.
        (IntOp::Shl, _) => lhs.wrapping_shl(amount) & mask(bits),
        (IntOp::Shr, Sign::Unsigned) => lhs >> amount,
        // Sign-filling, so it saturates at `-1` rather than reaching zero. The magnitude only
        // shrinks, so it can never leave the width.
        (IntOp::Shr, Sign::Signed) => encode_signed(bits, decode_signed(bits, lhs) >> amount),
        _ => unreachable!("only shifts reach this function"),
    };
    Outcome::Value(value)
}

/// The bit pattern a **total** evaluator must produce, including where [`eval`] rejects.
///
/// A backend cannot reject mid-expression: by the time an opcode runs, the guard IR either proved
/// the input fine or the program was already rejected, and either way the opcode has to write
/// something.
///
/// [`None`] means _deliberately unspecified_, where the backends do not agree and there is no right
/// answer to hold them to. In essence, this is the model recording that the whole obligation for
/// those inputs sits on the guard IR. Today it is exactly the two division cases, where LLVM's
/// `udiv`/`sdiv` are undefined outright while the VM answers zero.
///
/// # Panics
///
/// As [`eval`] does, on invalid widths.
#[must_use]
pub fn residue(
    op: IntOp,
    sign: Sign,
    bits: usize,
    lhs: Raw,
    rhs_bits: usize,
    rhs: Raw,
) -> Option<Raw> {
    check_widths(Some(op), sign, bits, rhs_bits);

    match eval(op, sign, bits, lhs, rhs_bits, rhs) {
        Outcome::Value(v) => Some(v),
        Outcome::Rejected(Reject::Overflow) => {
            let lhs = lhs & mask(bits);
            let rhs = rhs & mask(rhs_bits);
            Some(
                match op {
                    IntOp::Add => lhs.wrapping_add(rhs),
                    IntOp::Sub => lhs.wrapping_sub(rhs),
                    IntOp::Mul => lhs.wrapping_mul(rhs),
                    _ => unreachable!("only add/sub/mul overflow"),
                } & mask(bits),
            )
        }
        Outcome::Rejected(Reject::ShiftAmount) => {
            // Re-entering `eval` keeps one definition of what a shift computes. The masked amount
            // is a submask of the original, so it still fits `rhs_bits` and the re-entry cannot
            // mask it a second time into something else, and it is below `bits`, so the re-entry
            // cannot reject again and the `value()` is always `Some`.
            let amount = masked_shift_amount(rhs & mask(rhs_bits), bits) as u128;
            eval(op, sign, bits, lhs, rhs_bits, amount).value()
        }

        // LLVM calls both undefined; the VM answers zero and wraps respectively. No agreed answer,
        // so none is specified.
        Outcome::Rejected(Reject::DivByZero | Reject::DivOverflow) => None,
    }
}

// COMPARISON AND BIT-LEVEL OPERATIONS
// ================================================================================================

/// A comparison, sign-agnostic.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum CmpOp {
    /// Pattern equality
    Eq,

    /// Less-than, read as the [`Sign`] says.
    Lt,
}

/// Compare two `bits`-wide patterns.
///
/// # Panics
///
/// As [`eval`] does, on invalid widths.
#[must_use]
pub fn cmp(op: CmpOp, sign: Sign, bits: usize, lhs: Raw, rhs: Raw) -> bool {
    check_widths(None, sign, bits, bits);
    let (lhs, rhs) = (lhs & mask(bits), rhs & mask(bits));
    match op {
        CmpOp::Eq => lhs == rhs,
        CmpOp::Lt if sign.is_signed() => decode_signed(bits, lhs) < decode_signed(bits, rhs),
        CmpOp::Lt => lhs < rhs,
    }
}

/// Reinterpret a pattern at a new width: truncate when narrowing, zero-extend when widening.
///
/// A cast in HLSSA is a raw-bit conversion and nothing more. Widening a _signed_ value is
/// [`sign_extend`], a separate operation, because the frontend emits a separate opcode for it.
#[must_use]
pub fn cast_int(v: Raw, to_bits: usize) -> Raw {
    v & mask(to_bits)
}

/// Extract `width` bits starting at `offset`, right-aligned. HLSSA's `BitRange`.
///
/// The truncation primitive: `BitRange(v, 0, n)` is the low `n` bits, and a logical `>>` by a
/// constant is `BitRange(v, k, bits - k)`.
#[must_use]
pub fn bit_range(v: Raw, offset: usize, width: usize) -> Raw {
    if offset >= MAX_BITS {
        return 0;
    }
    (v >> offset) & mask(width)
}

/// Bitwise complement, held to the operand width.
#[must_use]
pub fn not(v: Raw, bits: usize) -> Raw {
    !v & mask(bits)
}

/// The width of one limb of a **canonical field representation**.
///
/// Not a property of any particular field: `ark_ff`'s `BigInt<const N: usize>` is `[u64; N]`, so
/// the limb _count_ `N` varies from field to field while the limb _width_ does not. It is named
/// here so that [`field_limbs_to_int`]'s callers state the contract rather than each rediscovering
/// the 64.
pub const FIELD_LIMB_BITS: usize = 64;

/// Recombine little-endian limbs of `limb_bits` each into a `bits`-wide integer.
///
/// The limb width is a parameter because Mavros has two unrelated kinds of limb, and conflating
/// them is exactly the sort of bug this crate exists to make impossible: a canonical field
/// representation is always [`FIELD_LIMB_BITS`] wide, while the witness decompositions in
/// `witness_bitwise` use a width that is _derived from the field size_. Both are little-endian limb
/// vectors and neither knows the other's width.
///
/// Only [`field_limbs_to_int`] calls this today, and the second kind is not about to: those limbs
/// are values in the emitted IR rather than numbers the host holds, so they are recombined by
/// instructions and never reach a function like this one. The parameter is therefore here to stop
/// [`FIELD_LIMB_BITS`] being baked into the recombination rule, not because a caller is coming —
/// which is why the field-shaped entry point below is the one everything is expected to use.
///
/// Each limb is masked to `limb_bits` first, so a caller packing a narrow limb into a wider
/// container need not clear the space above it.
///
/// # Panics
///
/// If `limb_bits` is zero, which describes no representation at all.
#[must_use]
pub fn limbs_to_int(limbs: &[u64], limb_bits: usize, bits: usize) -> Raw {
    debug_assert!((1..=MAX_BITS).contains(&bits));
    assert!(limb_bits > 0, "a limb must be at least one bit wide");

    // Limbs from here up contribute only bits the final mask discards. The bound also keeps the
    // shift below in range: the largest offset it permits is `((MAX_BITS - 1) / limb_bits) *
    // limb_bits`, which is at most `MAX_BITS - 1`.
    let usable = MAX_BITS.div_ceil(limb_bits);
    let limb_mask = mask(limb_bits);

    let low = limbs
        .iter()
        .take(usable)
        .enumerate()
        .fold(0, |acc, (i, &limb)| {
            acc | ((Raw::from(limb) & limb_mask) << (i * limb_bits))
        });

    low & mask(bits)
}

/// Read a field element's canonical little-endian limbs as a `bits`-wide integer.
///
/// The `Field -> Int` cast, and [`limbs_to_int`] at [`FIELD_LIMB_BITS`]. It takes the low `bits`
/// bits of the **canonical** representation, so a caller holding a Montgomery form must convert
/// first.
///
/// The slice is unsized because the limb count is a property of the field. A slice shorter than
/// `bits` implies is not an error either: it simply describes a smaller element.
#[must_use]
pub fn field_limbs_to_int(limbs: &[u64], bits: usize) -> Raw {
    limbs_to_int(limbs, FIELD_LIMB_BITS, bits)
}

/// Whether a field element's canonical limbs are entirely inside a `bits`-wide integer.
///
/// The companion of [`field_limbs_to_int`], which **truncates**: this is how a caller asks whether
/// the truncation would lose anything, so that it can refuse rather than answer a number the
/// element is not.
///
/// It lives here rather than at the call site because the two have to agree about which bits the
/// read covers, and a caller working that boundary out a second time is how they come apart. The
/// obvious spelling, "every limb from `bits / FIELD_LIMB_BITS` up is zero", is the same test as
/// this one only while `bits` is a whole number of limbs — which is true of the one width that
/// asks today, and is not a property of the question.
#[must_use]
pub fn field_limbs_fit(limbs: &[u64], bits: usize) -> bool {
    debug_assert!((1..=MAX_BITS).contains(&bits));

    // Limbs below `whole` are covered outright. The one at `whole` is covered only up to `spare`
    // bits, and where `spare` is zero that shift asks for the whole limb, which is the answer
    // wanted: at a width that ends on a limb boundary the limb above it is entirely outside.
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

// TESTS
// ================================================================================================

#[cfg(test)]
mod tests;
