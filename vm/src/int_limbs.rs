//! Multi-limb integer arithmetic, the engine behind the `_intn` lane.
//!
//! An `int(N)` value is `ceil(N/64)` consecutive frame cells, little-endian; the top cell holds the
//! `N - 64*(k-1)` leftover bits. No host type is this wide, so nothing here can defer to one.
//!
//! Every function here assumes:
//!
//! - **Operands arrive masked**, so they are not masked again. Results are returned masked.
//! - **Each function is total.** The guards are separate bytecode, so a function must answer every
//!   input.
//! - **A slice is exactly the width's cells.** A function reads a slice's length as the value's
//!   extent, so a slice that disagrees with the width beside it does not fail: it masks the wrong
//!   cell or accumulates into a neighbor.
//!
//! Every operation is checked against the model at every wide width. That does not show each is
//! reachable from its opcode, so `bytecode.rs` also runs every `_intn` opcode through real
//! dispatch, in `every_wide_opcode_agrees_with_the_model_through_dispatch` and the round trips
//! beside it.
//!
//! This is a second implementation of the model in `IntBits`: that one stays legible, this one is
//! tuned for speed, and the tests hold them together.

use mavros_limb_arith::top_limb_mask;

use crate::interpreter::Frame;

// UTILITIES
// ================================================================================================

/// Debug-check that `value` is exactly the cells an `int(bits)` occupies.
#[inline(always)]
fn debug_extent(value: &[u64], bits: u64) {
    debug_assert_eq!(
        value.len(),
        Frame::int_cells(bits),
        "an int{bits} is {} cells, not the {} given",
        Frame::int_cells(bits),
        value.len()
    );
}

/// Debug-check that a result and its two operands cover the same cells.
#[inline(always)]
fn debug_same_extent(res: &[u64], a: &[u64], b: &[u64]) {
    debug_assert!(
        res.len() == a.len() && res.len() == b.len(),
        "an operation over {} cells was given operands of {} and {}",
        res.len(),
        a.len(),
        b.len()
    );
}

/// Clear every bit above the width.
#[inline(always)]
fn normalize(value: &mut [u64], bits: u64) {
    debug_extent(value, bits);
    mavros_limb_arith::mask_top(value, bits as usize);
}

/// The value's sign bit, under a two's complement reading at `bits`.
#[inline(always)]
fn is_negative(value: &[u64], bits: u64) -> bool {
    debug_extent(value, bits);
    mavros_limb_arith::is_negative(value, bits as usize)
}

// ADDITION AND SUBTRACTION
// ================================================================================================

/// `a + b`, wrapping at the width.
pub fn add(res: &mut [u64], a: &[u64], b: &[u64], bits: u64) {
    debug_extent(a, bits);
    debug_extent(b, bits);
    mavros_limb_arith::add(res, a, b);
    normalize(res, bits);
}

/// `a - b`, wrapping at the width.
pub fn sub(res: &mut [u64], a: &[u64], b: &[u64], bits: u64) {
    debug_extent(a, bits);
    debug_extent(b, bits);
    mavros_limb_arith::sub(res, a, b);
    normalize(res, bits);
}

// BITWISE
// ================================================================================================

/// `a & b`, limb by limb.
pub fn and(res: &mut [u64], a: &[u64], b: &[u64]) {
    debug_same_extent(res, a, b);
    for i in 0..res.len() {
        res[i] = a[i] & b[i];
    }
}

/// `a | b`, limb by limb.
pub fn or(res: &mut [u64], a: &[u64], b: &[u64]) {
    debug_same_extent(res, a, b);
    for i in 0..res.len() {
        res[i] = a[i] | b[i];
    }
}

/// `a ^ b`, limb by limb.
pub fn xor(res: &mut [u64], a: &[u64], b: &[u64]) {
    debug_same_extent(res, a, b);
    for i in 0..res.len() {
        res[i] = a[i] ^ b[i];
    }
}

/// `!a`, masked back into the width.
///
/// The only bitwise operation that needs the width: complementing sets every bit above it.
pub fn not(res: &mut [u64], a: &[u64], bits: u64) {
    debug_extent(a, bits);
    for i in 0..res.len() {
        res[i] = !a[i];
    }
    normalize(res, bits);
}

// COMPARISON
// ================================================================================================

/// Whether the two values have the same pattern.
pub fn eq(a: &[u64], b: &[u64]) -> bool {
    debug_assert_eq!(a.len(), b.len(), "a comparison across two extents");
    a == b
}

/// Whether `a < b`, reading both as unsigned.
pub fn ult(a: &[u64], b: &[u64]) -> bool {
    mavros_limb_arith::ult(a, b)
}

/// Whether `a < b`, reading both as two's complement at `bits`.
pub fn slt(a: &[u64], b: &[u64], bits: u64) -> bool {
    debug_extent(a, bits);
    debug_extent(b, bits);
    mavros_limb_arith::slt(a, b, bits as usize)
}

// SHIFTS
// ================================================================================================

/// The amount a shift by `b` applies to a `bits`-wide operand: `b` read as unsigned, mod `bits`.
pub fn shift_amount(b: &[u64], bits: u64) -> u64 {
    mavros_limb_arith::shift_amount(b, bits as usize) as u64
}

/// `a << amount`, wrapping at the width, with `amount` already reduced.
pub fn shl_by(res: &mut [u64], a: &[u64], amount: u64, bits: u64) {
    debug_extent(a, bits);
    debug_extent(res, bits);
    mavros_limb_arith::shl(res, a, amount as usize, bits as usize);
}

/// `a >> amount`, zero-filling, with `amount` already reduced.
pub fn ushr_by(res: &mut [u64], a: &[u64], amount: u64, bits: u64) {
    debug_extent(a, bits);
    debug_extent(res, bits);
    mavros_limb_arith::ushr(res, a, amount as usize, bits as usize);
}

/// `a >> amount`, sign-filling under a two's complement reading at `bits`.
pub fn ashr_by(res: &mut [u64], a: &[u64], amount: u64, bits: u64) {
    debug_extent(a, bits);
    debug_extent(res, bits);
    mavros_limb_arith::ashr(res, a, amount as usize, bits as usize);
}

// WIDTH
// ================================================================================================

/// Write `a`, read at `from_bits`, into `res` at `to_bits`, filling above the source with zeros.
///
/// Both narrowing and widening: a narrowing cast keeps the low `to_bits` bits, which is what
/// `normalize` leaves behind, and a widening one zero-fills.
pub fn zero_extend(res: &mut [u64], a: &[u64], from_bits: u64, to_bits: u64) {
    extend_by(res, a, from_bits, to_bits, 0);
}

/// As [`zero_extend`], filling above the source with its sign bit instead.
pub fn sign_extend(res: &mut [u64], a: &[u64], from_bits: u64, to_bits: u64) {
    let fill = if is_negative(a, from_bits) {
        u64::MAX
    } else {
        0
    };
    extend_by(res, a, from_bits, to_bits, fill);
}

/// The width-cast body both casts share. They differ only in `fill`, the bits written above the
/// source.
fn extend_by(res: &mut [u64], a: &[u64], from_bits: u64, to_bits: u64, fill: u64) {
    debug_extent(a, from_bits);
    debug_extent(res, to_bits);
    let source_top = a.len() - 1;
    let source_mask = top_limb_mask(from_bits as usize);
    for i in 0..res.len() {
        res[i] = match i {
            _ if i > source_top => fill,
            // The source's top cell is half value and half fill, split at the source width.
            _ if i == source_top => (a[i] & source_mask) | (fill & !source_mask),
            _ => a[i],
        };
    }
    normalize(res, to_bits);
}

// MULTIPLICATION
// ================================================================================================

/// `a * b`, wrapping at the width: [`mavros_limb_arith::mul`], which needs no scratch buffer,
/// masked back to the width.
pub fn mul(res: &mut [u64], a: &[u64], b: &[u64], bits: u64) {
    debug_extent(a, bits);
    debug_extent(b, bits);
    mavros_limb_arith::mul(res, a, b);
    normalize(res, bits);
}

// DIVISION
// ================================================================================================

/// `a / b` and `a % b`, both read as unsigned, written into `quotient` and `remainder`.
///
/// Total: a zero divisor answers zero for both, which is the convention the `_int` lane's
/// `cell_udiv` records for all four division helpers. Nothing here traps, because the VM reports a
/// failed execution through `trap` rather than by aborting the host.
pub fn udivrem(quotient: &mut [u64], remainder: &mut [u64], a: &[u64], b: &[u64], bits: u64) {
    debug_extent(quotient, bits);
    debug_extent(remainder, bits);
    debug_extent(a, bits);
    debug_extent(b, bits);
    let mut scratch = vec![0u64; 2 * a.len() + 1];
    mavros_limb_arith::udivrem(quotient, remainder, a, b, &mut scratch);
}

/// `a / b` and `a % b`, both read as two's complement at `bits`, by
/// [`mavros_limb_arith::sdivrem`].
pub fn sdivrem(quotient: &mut [u64], remainder: &mut [u64], a: &[u64], b: &[u64], bits: u64) {
    debug_extent(quotient, bits);
    debug_extent(remainder, bits);
    debug_extent(a, bits);
    debug_extent(b, bits);
    let mut scratch = vec![0u64; 4 * a.len() + 1];
    mavros_limb_arith::sdivrem(quotient, remainder, a, b, bits as usize, &mut scratch);
}

// TESTS
// ================================================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use mavros_int_semantics::{CmpOp, IntBits, IntOp, corners, residue};

    /// Run one operation through this lane's bodies, at the width the operands carry.
    fn intn_lane(op: IntOp, bits: u64, a: &[u64], b: &[u64]) -> Vec<u64> {
        let mut res = vec![0u64; a.len()];
        let mut scratch = vec![0u64; a.len()];
        match op {
            // The signed and unsigned members of a pair differ only in _when they fail_, which the
            // guard IR decides before an opcode runs. So this lane is held to `residue`, not
            // `eval`.
            IntOp::UAdd | IntOp::SAdd => add(&mut res, a, b, bits),
            IntOp::USub | IntOp::SSub => sub(&mut res, a, b, bits),
            IntOp::UMul | IntOp::SMul => mul(&mut res, a, b, bits),
            IntOp::UDiv => udivrem(&mut res, &mut scratch, a, b, bits),
            IntOp::URem => udivrem(&mut scratch, &mut res, a, b, bits),
            IntOp::SDiv => sdivrem(&mut res, &mut scratch, a, b, bits),
            IntOp::SRem => sdivrem(&mut scratch, &mut res, a, b, bits),
            IntOp::And => and(&mut res, a, b),
            IntOp::Or => or(&mut res, a, b),
            IntOp::Xor => xor(&mut res, a, b),
            IntOp::Shl => shl_by(&mut res, a, shift_amount(b, bits), bits),
            IntOp::UShr => ushr_by(&mut res, a, shift_amount(b, bits), bits),
            IntOp::SShr => ashr_by(&mut res, a, shift_amount(b, bits), bits),
        }
        res
    }

    /// The widths this engine is swept at, which include widths it is never dispatched for.
    ///
    /// [`corners::WIDE_WIDTHS`] starts at 129, past every two-limb width: where a carry chain first
    /// carries, a Knuth divisor first has two limbs and a shift first crosses a limb boundary. The
    /// interpreter sends those widths to its double lane, so they are added here.
    fn swept_widths() -> Vec<usize> {
        let mut widths = vec![65, 96, 127];
        widths.extend(corners::WIDE_WIDTHS);
        widths
    }

    /// The operand pairs to sweep, both read at `bits` as the frontend's unification guarantees.
    fn operand_pairs(op: IntOp, bits: usize) -> Vec<(IntBits, IntBits)> {
        let (values, rhs) = corners::wide_operands(op, bits);
        values
            .into_iter()
            .flat_map(|a| rhs.iter().map(move |b| (a.clone(), b.clone())))
            .collect()
    }

    #[test]
    fn the_intn_lane_agrees_with_the_model() {
        let mut checked = 0usize;

        for op in IntOp::ALL {
            for bits in swept_widths() {
                for (a, b) in operand_pairs(op, bits) {
                    let got = intn_lane(op, bits as u64, a.limbs(), b.limbs());

                    // Rule 3, which binds on every input including the unspecified ones: whatever
                    // the answer is, it is inside the width.
                    assert_eq!(
                        got[got.len() - 1] & !top_limb_mask(bits),
                        0,
                        "{op:?} at {bits} bits left a result outside the width"
                    );

                    // Rule 1, where the model has an opinion.
                    if let Some(want) = residue(op, &a, &b) {
                        assert_eq!(
                            got,
                            want.limbs(),
                            "{op:?} at {bits} bits disagreed with the model"
                        );
                        checked += 1;
                    }
                }
            }
        }

        // A sweep that agreed with the model on nothing would satisfy every assertion above, so
        // the count is part of the test rather than a diagnostic.
        assert!(
            checked > 10_000,
            "the sweep only reached {checked} specified points"
        );
    }

    #[test]
    fn the_intn_comparisons_agree_with_the_model() {
        for bits in swept_widths() {
            let values = corners::wide_values(bits);
            for a in &values {
                for b in &values {
                    assert_eq!(
                        eq(a.limbs(), b.limbs()),
                        a.compare(CmpOp::Eq, b),
                        "eq disagreed at {bits} bits"
                    );
                    assert_eq!(
                        ult(a.limbs(), b.limbs()),
                        a.compare(CmpOp::ULt, b),
                        "ult disagreed at {bits} bits"
                    );
                    assert_eq!(
                        slt(a.limbs(), b.limbs(), bits as u64),
                        a.compare(CmpOp::SLt, b),
                        "slt disagreed at {bits} bits"
                    );
                }
            }
        }
    }

    #[test]
    fn the_intn_complement_agrees_with_the_model() {
        for bits in swept_widths() {
            for a in corners::wide_values(bits) {
                let mut got = vec![0u64; a.limbs().len()];
                not(&mut got, a.limbs(), bits as u64);
                assert_eq!(got, a.complement().limbs(), "not disagreed at {bits} bits");
            }
        }
    }

    #[test]
    fn the_intn_width_casts_agree_with_the_model() {
        for from_bits in swept_widths() {
            for to_bits in swept_widths() {
                for a in corners::wide_values(from_bits) {
                    let cells = IntBits::limbs_for_bits(to_bits);

                    let mut widened = vec![0u64; cells];
                    zero_extend(&mut widened, a.limbs(), from_bits as u64, to_bits as u64);
                    assert_eq!(
                        widened,
                        a.cast(to_bits).limbs(),
                        "a cast from {from_bits} to {to_bits} bits disagreed"
                    );

                    let mut extended = vec![0u64; cells];
                    sign_extend(&mut extended, a.limbs(), from_bits as u64, to_bits as u64);
                    // The model's `sign_extend` only widens, so a narrowing pair is checked
                    // against the cast it degenerates to.
                    let want = if to_bits >= from_bits {
                        a.sign_extend(to_bits)
                    } else {
                        a.cast(to_bits)
                    };
                    assert_eq!(
                        extended,
                        want.limbs(),
                        "a sign extension from {from_bits} to {to_bits} bits disagreed"
                    );
                }
            }
        }
    }
}
