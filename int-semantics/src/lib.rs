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
//! # use mavros_int_semantics::{eval, residue, IntBits, IntOp, Outcome};
//! let u8 = |v| IntBits::from_u128(8, v);
//!
//! // Accepted, so both agree.
//! assert_eq!(eval(IntOp::UAdd, &u8(1), &u8(2)), Outcome::Value(u8(3)));
//! assert_eq!(residue(IntOp::UAdd, &u8(1), &u8(2)), Some(u8(3)));
//!
//! // Rejected by Noir, but every backend wraps, so the residue is pinned.
//! assert!(matches!(
//!     eval(IntOp::UAdd, &u8(200), &u8(100)),
//!     Outcome::Rejected(_)
//! ));
//! assert_eq!(residue(IntOp::UAdd, &u8(200), &u8(100)), Some(u8(44)));
//!
//! // The same two patterns read as signed are `-56 + 100`, which fits. The operation is what
//! // picks the reading, so these are two operations rather than one with a flag.
//! assert_eq!(
//!     eval(IntOp::SAdd, &u8(200), &u8(100)),
//!     Outcome::Value(u8(44))
//! );
//!
//! // Rejected, and the backends do not agree on what to do anyway.
//! assert!(matches!(
//!     eval(IntOp::UDiv, &u8(1), &u8(0)),
//!     Outcome::Rejected(_)
//! ));
//! assert_eq!(residue(IntOp::UDiv, &u8(1), &u8(0)), None);
//! ```
//!
//! # Values are Patterns
//!
//! Every value in and out is an [`IntBits`], which is a **raw bit pattern** that carries its own
//! width. With no sign, it's up to the operation to name the reading much like in the SSA, and its
//! corresponding 2's-complement reading is [`SignedValue`].
//!
//! Everything an [`IntBits`] can *do* lives on [`IntBits`], in `int_bits.rs`. What is left here is
//! the part of the model with **no receiver**: [`eval`] and [`residue`], whose subject is the
//! _operation_ rather than the left operand, plus the vocabulary they take and the width rule they
//! enforce.
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
pub mod int_bits;
pub mod register;

pub use int_bits::IntBits;
use num_bigint::{BigInt, BigUint};

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

/// The two's-complement _reading_ of an [`IntBits`], as a mathematical integer.
///
/// Named rather than spelled `BigInt` at every site so that what backs a signed reading stays this
/// crate's business. [`IntBits`] itself is the payload type and is named directly: it is Mavros's
/// own, so there is nothing to hide behind an alias.
pub type SignedValue = BigInt;

// INTEGER OPERATIONS MODEL
// ================================================================================================

/// One binary integer operation, with the signedness reading it uses.
///
/// # Which Operations Come in Pairs
///
/// Every operation whose _answer or acceptance_ depends on its choice of signedness comes in a
/// pair:
///
/// - `Div`, `Rem` and `Shr` compute **different values**.
/// - `Add`, `Sub` and `Mul` compute the same bits whenever they succeed, and differ in **when they
///   fail**.
/// - `And`, `Or` and `Xor` have no reading at all, and neither does `Shl`: it wraps under both
///   readings and its amount is read as an unsigned magnitude either way.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum IntOp {
    UAdd,
    SAdd,
    USub,
    SSub,
    UMul,
    SMul,
    UDiv,
    SDiv,
    URem,
    SRem,
    And,
    Or,
    Xor,
    Shl,
    UShr,
    SShr,
}

impl IntOp {
    /// Every operation, for exhaustive sweeps.
    ///
    /// Ordered unsigned-then-signed within each group, and the groups in the order the generated
    /// corpus renders them, so that grouping by [`IntOp::name`] reproduces the file and cell order
    /// the checked-in `noir_tests/int_semantics_*` directories were blessed at.
    pub const ALL: [IntOp; 16] = [
        IntOp::UAdd,
        IntOp::SAdd,
        IntOp::USub,
        IntOp::SSub,
        IntOp::UMul,
        IntOp::SMul,
        IntOp::UDiv,
        IntOp::SDiv,
        IntOp::URem,
        IntOp::SRem,
        IntOp::And,
        IntOp::Or,
        IntOp::Xor,
        IntOp::Shl,
        IntOp::UShr,
        IntOp::SShr,
    ];

    /// The reading this operation names, or [`None`] where it names none.
    ///
    /// `None` is not "unsigned": it is the statement that this operation gives one answer whatever
    /// the operands were meant as. Only the width checks distinguish the two as a signed cap has
    /// nothing to say about an operation with no signed form.
    #[must_use]
    pub const fn sign(self) -> Option<Sign> {
        match self {
            IntOp::SAdd | IntOp::SSub | IntOp::SMul | IntOp::SDiv | IntOp::SRem | IntOp::SShr => {
                Some(Sign::Signed)
            }
            IntOp::UAdd | IntOp::USub | IntOp::UMul | IntOp::UDiv | IntOp::URem | IntOp::UShr => {
                Some(Sign::Unsigned)
            }
            IntOp::And | IntOp::Or | IntOp::Xor | IntOp::Shl => None,
        }
    }

    /// Whether this operation reads its operands as two's complement.
    ///
    /// The reading-free operations answer `false`, which describes what they do to the raw pattern.
    /// A caller that needs to tell "unsigned" from "the question does not arise" wants
    /// [`IntOp::sign`].
    #[must_use]
    pub const fn is_signed(self) -> bool {
        matches!(self.sign(), Some(Sign::Signed))
    }

    /// The name this operation's group carries in the generated corpus and in a diagnostic.
    ///
    /// Sign-erased, because a Noir program has one `+` and the type of its operands is what picks
    /// the reading. It is also how [`corpus`] groups the split variants back into one test file per
    /// operator.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            IntOp::UAdd | IntOp::SAdd => "add",
            IntOp::USub | IntOp::SSub => "sub",
            IntOp::UMul | IntOp::SMul => "mul",
            IntOp::UDiv | IntOp::SDiv => "div",
            IntOp::URem | IntOp::SRem => "rem",
            IntOp::And => "and",
            IntOp::Or => "or",
            IntOp::Xor => "xor",
            IntOp::Shl => "shl",
            IntOp::UShr | IntOp::SShr => "shr",
        }
    }

    /// The Noir operator this lowers from.
    #[must_use]
    pub const fn symbol(self) -> &'static str {
        match self {
            IntOp::UAdd | IntOp::SAdd => "+",
            IntOp::USub | IntOp::SSub => "-",
            IntOp::UMul | IntOp::SMul => "*",
            IntOp::UDiv | IntOp::SDiv => "/",
            IntOp::URem | IntOp::SRem => "%",
            IntOp::And => "&",
            IntOp::Or => "|",
            IntOp::Xor => "^",
            IntOp::Shl => "<<",
            IntOp::UShr | IntOp::SShr => ">>",
        }
    }

    /// Whether the two readings of this operation take **different routes through the lowering**.
    ///
    /// Not "give different answers", which would be wrong about `Shl`: a left shift wraps
    /// identically under both readings, and what its signed form additionally owns is a rejecting
    /// constraint on a negative _amount_. The corpus generator
    /// reads this to decide how many operand pairs a cell is worth, so the set it picks out is
    /// load-bearing on the blessed test files — it is exactly the operations with a signed form
    /// plus `Shl`, or equivalently everything but the three bitwise ones.
    #[must_use]
    pub const fn reading_matters(self) -> bool {
        !matches!(self, IntOp::And | IntOp::Or | IntOp::Xor)
    }

    /// Whether this operation reads its right operand at a width of its **own**.
    ///
    /// True for the shifts, whose amount carries its own type, and false for everything else,
    /// where the two operands are one width and a caller offering two widths is offering a
    /// contradiction. The width checks every entry point performs are what act on the answer.
    #[must_use]
    pub const fn is_shift(self) -> bool {
        matches!(self, IntOp::Shl | IntOp::UShr | IntOp::SShr)
    }
}

/// A comparison, with the reading it names.
///
/// The same shape as HLSSA's `CmpKind` and LLSSA's `IntCmpOp`, which is the point: equality needs
/// no reading — two patterns of one width are equal under either — so only the ordering comes in
/// a pair.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum CmpOp {
    /// Pattern equality.
    Eq,

    /// Less-than as unsigned magnitudes.
    ULt,

    /// Less-than as two's complement.
    SLt,
}

impl CmpOp {
    pub const ALL: [CmpOp; 3] = [CmpOp::Eq, CmpOp::ULt, CmpOp::SLt];

    /// The reading this comparison names, or [`None`] for [`CmpOp::Eq`].
    #[must_use]
    pub const fn sign(self) -> Option<Sign> {
        match self {
            CmpOp::Eq => None,
            CmpOp::ULt => Some(Sign::Unsigned),
            CmpOp::SLt => Some(Sign::Signed),
        }
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
///
/// No [`Copy`], because a [`IntBits`] is a heap-backed pattern rather than a host word.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub enum Outcome {
    /// Noir computes this raw pattern, masked to the operand width.
    Value(IntBits),

    /// Noir rejects the program at runtime.
    Rejected(Reject),
}

impl Outcome {
    /// The value, if this evaluation is accepted.
    #[must_use]
    pub fn value(self) -> Option<IntBits> {
        match self {
            Outcome::Value(v) => Some(v),
            Outcome::Rejected(_) => None,
        }
    }

    #[must_use]
    pub const fn is_rejected(&self) -> bool {
        matches!(self, Outcome::Rejected(_))
    }
}

// HOST-WORD HELPER
// ================================================================================================

/// All-ones for the low `bits` bits, as a **host word**, total at the edges.
///
/// The one free function here that is not part of the model, and a host word rather than a method
/// on [`IntBits`] because it takes no operand and answers no value: there is no integer for it to
/// be an operation _on_. Its callers are the
/// ones genuinely working in `u64`s and `u128`s — the limb recombination in [`IntBits`], and the
/// compiler's remaining host-word arithmetic — so it stops being expressible the moment a width
/// outruns the host, which is the intended failure. The pattern of the same shape is
/// [`IntBits::all_ones`].
///
/// The saturation bound is the **host's** width and deliberately not [`MAX_BITS`], even though the
/// two are the same number today. The claim being made is that a `u128` cannot hold more than 128
/// bits, which is a fact about the host; writing it as the model's cap would leave `mask` correct
/// only for as long as that cap stays at 128, and the arm below is `1u128 << bits` — a debug panic
/// and, worse, a release build that masks the shift amount and answers a plausible wrong number.
#[must_use]
pub fn mask(bits: usize) -> u128 {
    if bits == 0 {
        0
    } else if bits >= u128::BITS as usize {
        u128::MAX
    } else {
        (1u128 << bits) - 1
    }
}

// THE MODEL
// ================================================================================================

/// Check the widths the operands arrived carrying.
pub(crate) fn check_widths(signed: bool, is_shift: bool, bits: usize, rhs_bits: usize) {
    assert!(
        (1..=MAX_BITS).contains(&bits),
        "operand width {bits} is outside 1..={MAX_BITS}"
    );
    assert!(
        (1..=MAX_BITS).contains(&rhs_bits),
        "right-operand width {rhs_bits} is outside 1..={MAX_BITS}"
    );
    assert!(
        !signed || bits <= MAX_SIGNED_BITS,
        "signed operation on a {bits}-bit value exceeds {MAX_SIGNED_BITS}"
    );
    assert!(
        rhs_bits == bits || is_shift,
        "operand widths {bits} and {rhs_bits}; only a shift reads its right operand at a width of its own"
    );
}

/// Noir's specification for one integer operation.
///
/// # Panics
///
/// If either width is outside `1..=128`, if a signed operation is asked of a pattern above
/// [`MAX_SIGNED_BITS`], or if a non-shift is given two operands of different widths. Those are
/// compiler bugs rather than program errors, so they are not [`Reject`] variants.
#[must_use]
pub fn eval(op: IntOp, lhs: &IntBits, rhs: &IntBits) -> Outcome {
    let bits = lhs.bits();
    check_widths(op.is_signed(), op.is_shift(), bits, rhs.bits());

    match op {
        // Bitwise operations have no reading and cannot fail, so they answer here. They are also
        // the only arms that stay on the pattern throughout: everything past this point needs a
        // number, and a number is what a pattern is not.
        IntOp::And => Outcome::Value(lhs.and(rhs)),
        IntOp::Or => Outcome::Value(lhs.or(rhs)),
        IntOp::Xor => Outcome::Value(lhs.xor(rhs)),

        IntOp::Shl | IntOp::UShr | IntOp::SShr => eval_shift(op, lhs, rhs),

        IntOp::SAdd | IntOp::SSub | IntOp::SMul | IntOp::SDiv | IntOp::SRem => {
            eval_signed_arith(op, bits, &lhs.to_signed(), &rhs.to_signed())
        }
        IntOp::UAdd | IntOp::USub | IntOp::UMul | IntOp::UDiv | IntOp::URem => {
            eval_unsigned_arith(op, lhs, rhs)
        }
    }
}

fn eval_unsigned_arith(op: IntOp, lhs: &IntBits, rhs: &IntBits) -> Outcome {
    // Both operands are one width here — a shift never reaches this function — so either says what
    // the result is held to. The signed arm below still takes it as a parameter, because by then
    // the operands are `SignedValue`s and a mathematical integer has no width to read.
    let bits = lhs.bits();
    let (lhs, rhs) = (BigUint::from(lhs), BigUint::from(rhs));

    let result = match op {
        IntOp::UAdd => lhs + rhs,
        // A `BigUint` has no negatives to fall into, so underflow is a test rather than a failed
        // subtraction. It is the same rejection either way: a difference below zero does not fit.
        IntOp::USub => {
            if lhs < rhs {
                return Outcome::Rejected(Reject::Overflow);
            }
            lhs - rhs
        }
        IntOp::UMul => lhs * rhs,
        IntOp::UDiv | IntOp::URem => {
            if rhs == BigUint::ZERO {
                return Outcome::Rejected(Reject::DivByZero);
            }
            // A quotient or remainder of two in-range values is always in range, so neither of
            // these can fail the width check below.
            if op == IntOp::UDiv {
                lhs / rhs
            } else {
                lhs % rhs
            }
        }
        _ => unreachable!("{op:?} does not reach the unsigned arithmetic arm"),
    };

    // The width test, stated as the number of significant bits rather than against a mask, so that
    // it says what it means at a width no host word could hold.
    if result.bits() <= bits as u64 {
        Outcome::Value(IntBits::from_biguint(bits, &result))
    } else {
        Outcome::Rejected(Reject::Overflow)
    }
}

fn eval_signed_arith(op: IntOp, bits: usize, lhs: &SignedValue, rhs: &SignedValue) -> Outcome {
    // No `checked_*` layer on either arm: these are mathematical integers, so there is no host
    // range to fall out of, and the width test below is the only test there is.
    let result = match op {
        IntOp::SAdd => lhs + rhs,
        IntOp::SSub => lhs - rhs,
        IntOp::SMul => lhs * rhs,
        IntOp::SDiv | IntOp::SRem => {
            if *rhs == SignedValue::from(0u8) {
                return Outcome::Rejected(Reject::DivByZero);
            }
            if *lhs == IntBits::signed_min(bits) && *rhs == SignedValue::from(-1i8) {
                return Outcome::Rejected(Reject::DivOverflow);
            }

            // `BigInt`'s `/` and `%` are Rust's, which are Noir's: truncation toward zero,
            // remainder signed like the dividend. `expand_signed_math` builds the same thing out
            // of a magnitude division plus a sign fix-up.
            if op == IntOp::SDiv {
                lhs / rhs
            } else {
                lhs % rhs
            }
        }
        _ => unreachable!("{op:?} does not reach the signed arithmetic arm"),
    };

    if IntBits::fits_signed(bits, &result) {
        Outcome::Value(IntBits::from_signed(bits, &result))
    } else {
        Outcome::Rejected(Reject::Overflow)
    }
}

/// Shifts, whose rules differ from the standard arithmetic ones in both directions.
///
/// Stricter about the amount: at or above the width is a rejection, where an arithmetic operand can
/// hold any pattern its width allows. Laxer about the result: a `<<` that pushes bits off the top
/// **wraps** and is not an error at all. `remove_bit_shifts` lowers it as a multiply in the field
/// followed by a truncation, so the discarded bits are discarded by construction.
fn eval_shift(op: IntOp, lhs: &IntBits, rhs: &IntBits) -> Outcome {
    let bits = lhs.bits();

    // The amount is read as an unsigned magnitude whatever the operation's sign.
    let magnitude = BigUint::from(rhs);
    if magnitude >= BigUint::from(bits) {
        return Outcome::Rejected(Reject::ShiftAmount);
    }
    let amount = usize::try_from(&magnitude).expect("an amount below the width fits a usize");

    let value = match op {
        // One map on the bit pattern whatever the reading, which is why `Shl` is a single variant.
        // The result can change sign (`64i8 << 1` is `-128`), so this stays on the raw pattern
        // rather than asking whether the mathematical product fits.
        IntOp::Shl => lhs.shifted_left(amount),
        IntOp::UShr => lhs.shifted_right(amount),
        // Sign-filling, so it saturates at `-1` rather than reaching zero. The magnitude only
        // shrinks, so it can never leave the width.
        IntOp::SShr => IntBits::from_signed(bits, &(lhs.to_signed() >> amount)),
        _ => unreachable!("{op:?} is not a shift"),
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
pub fn residue(op: IntOp, lhs: &IntBits, rhs: &IntBits) -> Option<IntBits> {
    let bits = lhs.bits();

    match eval(op, lhs, rhs) {
        Outcome::Value(v) => Some(v),
        Outcome::Rejected(Reject::Overflow) => {
            // One formula for both readings. Wrapping _is_ arithmetic modulo `2^bits`, and which
            // reading the operands were meant as does not change the bits of a sum, difference or
            // product — so the exact result
            // taken through `IntBits::from_signed`, which reduces modulo the width, is the answer
            // for signed and unsigned alike.
            let lhs = SignedValue::from(BigUint::from(lhs));
            let rhs = SignedValue::from(BigUint::from(rhs));
            let exact = match op {
                IntOp::UAdd | IntOp::SAdd => lhs + rhs,
                IntOp::USub | IntOp::SSub => lhs - rhs,
                IntOp::UMul | IntOp::SMul => lhs * rhs,
                _ => unreachable!("{op:?} cannot overflow"),
            };
            Some(IntBits::from_signed(bits, &exact))
        }
        Outcome::Rejected(Reject::ShiftAmount) => {
            // Re-entering `eval` keeps one definition of what a shift computes. The reduced amount
            // is below `bits`, so the re-entry cannot reject again; it is carried at the amount's
            // own width, which is where it fits by construction — the reduction is modulo `bits`,
            // and the original amount was already representable at that width.
            //
            // The `expect` is what states that. A bare `value()` would turn a broken re-entry into
            // a `None`, which this function documents as meaning something quite different: that
            // the backends have no agreed answer. Widening that set silently is the failure to
            // avoid, so the invariant is asserted rather than propagated.
            let amount = IntBits::from_u128(rhs.bits(), u128::from(rhs.reduced_shift_amount(bits)));
            Some(
                eval(op, lhs, &amount)
                    .value()
                    .expect("a shift by an amount reduced below the width cannot reject"),
            )
        }

        // LLVM calls both undefined; the VM answers zero and wraps respectively. No agreed answer,
        // so none is specified.
        Outcome::Rejected(Reject::DivByZero | Reject::DivOverflow) => None,
    }
}

// TESTS
// ================================================================================================

#[cfg(test)]
mod tests;
