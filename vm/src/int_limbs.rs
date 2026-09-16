//! Multi-limb integer arithmetic, the backing engine for the `_intn` lane.
//!
//! An `int(N)` value is `ceil(N/64)` consecutive frame cells, little-endian, with the top cell
//! holding the `N - 64*(k-1)` bits that are left over. Unlike its cousins it cannot rely on host
//! implementations.
//!
//! These functions all assume the following:
//!
//! - **Operands are provided masked**, avoiding the need to perform their own masking which may be
//!   unnecessary. Results are, however, returned with the necessary masking performed.
//! - **They are responsible for guaranteeing totality**, as the guards are emitted into the
//!   bytecode external to these implementations.
//! - **A slice is exactly the width's cells,** so every function reads a slice's length as the
//!   value's extent, so one that disagrees with the width beside it does not fail: it masks the
//!   wrong cell or accumulates into a neighbor.
//!
//! Every operation the lane could need is here and checked against the model at every wide width.
//! This does not check whether they are reachable from their corresponding opcodes, however, so
//! this is instead checked in `bytecode.rs`.
//!
//! This is a separate implementation of the model described by `IntBits` as that is intended to
//! stay legible, while this needs to be optimized for performance wherever possible. It is
//! nevertheless checked against that model of the semantics.

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

/// The number of bits the top cell of a `bits`-wide value actually holds, in `1..=64`.
#[inline(always)]
fn top_cell_bits(bits: u64) -> u64 {
    debug_assert!(bits > 0, "a zero-width integer has no top cell");
    ((bits - 1) % 64) + 1
}

/// The mask of the top cell of a `bits`-wide value.
#[inline(always)]
fn top_cell_mask(bits: u64) -> u64 {
    let held = top_cell_bits(bits);
    if held == 64 {
        u64::MAX
    } else {
        (1u64 << held) - 1
    }
}

/// Clear every bit above the width to ensure validity.
#[inline(always)]
fn normalize(value: &mut [u64], bits: u64) {
    debug_extent(value, bits);
    let top = value.len() - 1;
    value[top] &= top_cell_mask(bits);
}

/// The value's sign bit, under a two's complement reading at `bits`.
#[inline(always)]
fn is_negative(value: &[u64], bits: u64) -> bool {
    debug_extent(value, bits);
    let held = top_cell_bits(bits);
    (value[value.len() - 1] >> (held - 1)) & 1 == 1
}

// ADDITION AND SUBTRACTION
// ================================================================================================

/// `a + b`, wrapping at the width.
pub fn add(res: &mut [u64], a: &[u64], b: &[u64], bits: u64) {
    debug_extent(a, bits);
    debug_extent(b, bits);
    let mut carry = false;
    for i in 0..res.len() {
        let (sum, overflowed) = a[i].overflowing_add(b[i]);
        let (sum, carried) = sum.overflowing_add(u64::from(carry));
        res[i] = sum;
        // Two additions, so two chances to overflow — but never both at once: the second addend is
        // at most one, and a sum that already wrapped is small enough to absorb it.
        carry = overflowed | carried;
    }
    normalize(res, bits);
}

/// `a - b`, wrapping at the width.
pub fn sub(res: &mut [u64], a: &[u64], b: &[u64], bits: u64) {
    debug_extent(a, bits);
    debug_extent(b, bits);
    let mut borrow = false;
    for i in 0..res.len() {
        let (difference, underflowed) = a[i].overflowing_sub(b[i]);
        let (difference, borrowed) = difference.overflowing_sub(u64::from(borrow));
        res[i] = difference;
        borrow = underflowed | borrowed;
    }
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
    debug_assert_eq!(a.len(), b.len(), "a comparison across two extents");
    for i in (0..a.len()).rev() {
        if a[i] != b[i] {
            return a[i] < b[i];
        }
    }
    false
}

/// Whether `a < b`, reading both as two's complement at `bits`.
pub fn slt(a: &[u64], b: &[u64], bits: u64) -> bool {
    match (is_negative(a, bits), is_negative(b, bits)) {
        (true, false) => true,
        (false, true) => false,
        _ => ult(a, b),
    }
}

// SHIFTS
// ================================================================================================

/// The amount a shift by `b` actually applies to a `bits`-wide operand: the magnitude mod `bits`.
pub fn shift_amount(b: &[u64], bits: u64) -> u64 {
    if bits == 0 {
        return 0;
    }
    if bits.is_power_of_two() {
        return b[0] & (bits - 1);
    }
    // A running remainder below `bits`, so `(rest << 64) | limb` cannot overflow a `u128` for any
    // width this VM supports.
    let modulus = u128::from(bits);
    let mut rest = 0u128;
    for i in (0..b.len()).rev() {
        rest = ((rest << 64) | u128::from(b[i])) % modulus;
    }
    rest as u64
}

/// `a << amount`, wrapping at the width, with `amount` already reduced.
pub fn shl_by(res: &mut [u64], a: &[u64], amount: u64, bits: u64) {
    debug_extent(a, bits);
    let k = res.len();
    let cells = (amount / 64) as usize;
    let within = (amount % 64) as u32;

    for i in (0..k).rev() {
        // The limb landing at `i` is the one `cells` below it, with the bits carried up out of the
        // limb below that. `within == 0` is the case a `64 - within` shift would answer wrongly.
        let low = if i >= cells { a[i - cells] } else { 0 };
        let carried = if within > 0 && i > cells {
            a[i - cells - 1] >> (64 - within)
        } else {
            0
        };
        res[i] = (low << within) | carried;
    }
    normalize(res, bits);
}

/// `a >> amount`, zero-filling, with `amount` already reduced.
pub fn ushr_by(res: &mut [u64], a: &[u64], amount: u64, bits: u64) {
    shr_by(res, a, amount, bits, 0);
}

/// `a >> amount`, sign-filling under a two's complement reading at `bits`.
pub fn ashr_by(res: &mut [u64], a: &[u64], amount: u64, bits: u64) {
    let fill = if is_negative(a, bits) { u64::MAX } else { 0 };
    shr_by(res, a, amount, bits, fill);
}

/// The shared right-shift body, `fill` being the bits shifted in at the top.
fn shr_by(res: &mut [u64], a: &[u64], amount: u64, bits: u64, fill: u64) {
    debug_extent(a, bits);
    let k = res.len();
    let cells = (amount / 64) as usize;
    let within = (amount % 64) as u32;
    let top_mask = top_cell_mask(bits);

    // The operand's own top cell, sign-extended above the width where the fill asks for it.
    let cell = |i: usize| -> u64 {
        if i >= k {
            fill
        } else if i == k - 1 {
            (a[i] & top_mask) | (fill & !top_mask)
        } else {
            a[i]
        }
    };

    for i in 0..k {
        let high = cell(i + cells);
        let carried = if within > 0 {
            cell(i + cells + 1) << (64 - within)
        } else {
            0
        };
        res[i] = (high >> within) | carried;
    }
    normalize(res, bits);
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

/// The shared width-cast body, `fill` being the bits written above the source.
///
/// The two casts differ in that one value, as the two right shifts do, so they share a body for
/// the same reason [`shr_by`] is shared.
fn extend_by(res: &mut [u64], a: &[u64], from_bits: u64, to_bits: u64, fill: u64) {
    debug_extent(a, from_bits);
    debug_extent(res, to_bits);
    let source_top = a.len() - 1;
    let source_mask = top_cell_mask(from_bits);
    for i in 0..res.len() {
        res[i] = match i {
            _ if i > source_top => fill,
            // The source's own top cell is half value and half fill, split at the source width.
            _ if i == source_top => (a[i] & source_mask) | (fill & !source_mask),
            _ => a[i],
        };
    }
    normalize(res, to_bits);
}

// MULTIPLICATION
// ================================================================================================

/// `a * b`, wrapping at the width.
///
/// Schoolbook, accumulating straight into the result rather than into a `2k`-limb product and
/// truncating after: the columns at or above `k` are exactly the ones the wrap discards, so the
/// high half is never computed and the lane needs no scratch buffer. The carry leaving column
/// `k-1` is dropped for the same reason.
pub fn mul(res: &mut [u64], a: &[u64], b: &[u64], bits: u64) {
    debug_extent(a, bits);
    debug_extent(b, bits);
    let k = res.len();
    res.fill(0);
    for i in 0..k {
        let mut carry = 0u64;
        for j in 0..(k - i) {
            // A limb product plus two limb-sized addends is exactly a `u128` at its widest:
            // `(2^64-1)^2 + 2*(2^64-1)` is `2^128 - 1`.
            let column =
                u128::from(a[i]) * u128::from(b[j]) + u128::from(res[i + j]) + u128::from(carry);
            res[i + j] = column as u64;
            carry = (column >> 64) as u64;
        }
    }
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
    // All four up front rather than through `normalize`, which the zero-divisor path returns
    // before reaching.
    debug_extent(quotient, bits);
    debug_extent(remainder, bits);
    debug_extent(a, bits);
    debug_extent(b, bits);

    quotient.fill(0);
    remainder.fill(0);

    let n = significant_limbs(b);
    if n == 0 {
        return;
    }
    if n == 1 {
        let divisor = u128::from(b[0]);
        let mut rest = 0u128;
        for i in (0..a.len()).rev() {
            let dividend = (rest << 64) | u128::from(a[i]);
            quotient[i] = (dividend / divisor) as u64;
            rest = dividend % divisor;
        }
        remainder[0] = rest as u64;
        normalize(quotient, bits);
        normalize(remainder, bits);
        return;
    }

    knuth_divide(quotient, remainder, a, b, n);
    normalize(quotient, bits);
    normalize(remainder, bits);
}

/// `a / b` and `a % b`, both read as two's complement at `bits`.
///
/// Sign-magnitude around [`udivrem`]: the quotient takes the operands' xor and the remainder takes
/// the dividend's sign.
pub fn sdivrem(quotient: &mut [u64], remainder: &mut [u64], a: &[u64], b: &[u64], bits: u64) {
    debug_extent(a, bits);
    debug_extent(b, bits);
    let (a_negative, b_negative) = (is_negative(a, bits), is_negative(b, bits));

    let mut a_magnitude = a.to_vec();
    let mut b_magnitude = b.to_vec();
    if a_negative {
        negate(&mut a_magnitude, bits);
    }
    if b_negative {
        negate(&mut b_magnitude, bits);
    }

    udivrem(quotient, remainder, &a_magnitude, &b_magnitude, bits);

    if a_negative != b_negative {
        negate(quotient, bits);
    }
    if a_negative {
        negate(remainder, bits);
    }
}

/// Replace `value` with its two's complement negation at `bits`.
fn negate(value: &mut [u64], bits: u64) {
    let mut carry = true;
    for limb in value.iter_mut() {
        let (complemented, carried) = (!*limb).overflowing_add(u64::from(carry));
        *limb = complemented;
        carry = carried;
    }
    normalize(value, bits);
}

/// The index one past the highest non-zero limb, so zero answers zero.
fn significant_limbs(value: &[u64]) -> usize {
    value
        .iter()
        .rposition(|&limb| limb != 0)
        .map_or(0, |i| i + 1)
}

/// Knuth's algorithm D, for a divisor of two limbs or more.
///
/// The estimate `qhat` for each quotient limb comes from dividing the top two limbs of the running
/// dividend by the top limb of the divisor, which is why the divisor is first shifted left until
/// its top bit is set: the estimate is then within one of the truth, and the correction below
/// closes that gap. `u128` division supplies the two-limb-by-one-limb step a 64-bit host does not
/// have as an operator.
fn knuth_divide(quotient: &mut [u64], remainder: &mut [u64], a: &[u64], b: &[u64], n: usize) {
    const BASE: u128 = 1 << 64;

    let m = significant_limbs(a);
    if m < n {
        remainder[..a.len()].copy_from_slice(a);
        return;
    }

    // Normalise so the divisor's top bit is set. The same shift on the dividend leaves the
    // quotient unchanged and scales the remainder, which is undone at the end.
    let shift = b[n - 1].leading_zeros();
    let mut divisor = vec![0u64; n];
    shift_left_into(&mut divisor, &b[..n], shift);

    // One limb wider than the dividend: the shift can carry out of the top, and the algorithm
    // reads `dividend[j + n]` at the highest `j`.
    let mut dividend = vec![0u64; m + 1];
    shift_left_into(&mut dividend, &a[..m], shift);

    for j in (0..=(m - n)).rev() {
        let top = (u128::from(dividend[j + n]) << 64) | u128::from(dividend[j + n - 1]);
        let mut estimate = top / u128::from(divisor[n - 1]);
        let mut rest = top % u128::from(divisor[n - 1]);

        // Bring the estimate down to at most one above the true limb. The first disjunct is
        // checked first so the product below is only ever formed for an estimate that fits a limb.
        while estimate >= BASE
            || estimate * u128::from(divisor[n - 2])
                > (rest << 64) | u128::from(dividend[j + n - 2])
        {
            estimate -= 1;
            rest += u128::from(divisor[n - 1]);
            if rest >= BASE {
                break;
            }
        }

        // Subtract `estimate * divisor` from the window, keeping a signed borrow: the estimate may
        // still be one too large, and that is the case the add-back below repairs.
        let mut borrow = 0i128;
        for i in 0..n {
            let product = estimate * u128::from(divisor[i]);
            let column = i128::from(dividend[i + j]) - borrow - i128::from(product as u64);
            dividend[i + j] = column as u64;
            borrow = (product >> 64) as i128 - (column >> 64);
        }
        let column = i128::from(dividend[j + n]) - borrow;
        dividend[j + n] = column as u64;

        quotient[j] = estimate as u64;
        if column < 0 {
            quotient[j] -= 1;
            let mut carry = false;
            for i in 0..n {
                let (sum, overflowed) = dividend[i + j].overflowing_add(divisor[i]);
                let (sum, carried) = sum.overflowing_add(u64::from(carry));
                dividend[i + j] = sum;
                carry = overflowed | carried;
            }
            dividend[j + n] = dividend[j + n].wrapping_add(u64::from(carry));
        }
    }

    // Undo the normalising shift to recover the true remainder.
    shift_right_into(&mut remainder[..n], &dividend[..n], shift);
}

/// `source << shift` written into `target`, where `shift` is under 64 and `target` may be longer.
fn shift_left_into(target: &mut [u64], source: &[u64], shift: u32) {
    for i in (0..target.len()).rev() {
        let low = source.get(i).copied().unwrap_or(0);
        let carried = if shift > 0 && i > 0 {
            source.get(i - 1).copied().unwrap_or(0) >> (64 - shift)
        } else {
            0
        };
        target[i] = (low << shift) | carried;
    }
}

/// `source >> shift` written into `target`, where `shift` is under 64 and the two are equal length.
fn shift_right_into(target: &mut [u64], source: &[u64], shift: u32) {
    for i in 0..target.len() {
        let high = source[i];
        let carried = if shift > 0 && i + 1 < source.len() {
            source[i + 1] << (64 - shift)
        } else {
            0
        };
        target[i] = (high >> shift) | carried;
    }
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
            // The signed and unsigned members of a pair differ only in _when they fail_, and by
            // the time an opcode runs the guard IR has already decided. That is the same reason
            // this lane is held to `residue` rather than to `eval`.
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

    /// The widths this engine is swept at, which is more than the widths it is dispatched for.
    ///
    /// [`corners::wide_widths_for`] starts at 129, so nothing in the shared set has **two** limbs
    /// — the shape where a carry chain first has a carry, a Knuth divisor first has two limbs, and
    /// a shift first crosses a limb boundary. The interpreter sends those widths to its double
    /// lane, so this engine never sees one in production.
    ///
    /// Swept here regardless, for the reason the signed operations are: a body is either right at
    /// a width or it is not, and which widths a _dispatch_ sends here is a separate question that
    /// can change under it. It changed once already — the double lane grew from one width to a
    /// range — and an engine tested only at the widths that reached it that day would have had to
    /// be re-argued rather than re-run.
    fn swept_widths() -> Vec<usize> {
        let mut widths = vec![65, 96, 127];
        widths.extend(corners::wide_widths_for(false));
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
            // Every operation is swept at every wide width, the signed ones included, even though
            // `corners::wide_widths_for(true)` is empty and no lowering will reach a signed wide
            // opcode until `MAX_LOWERED_SIGNED_BITS` moves. That constant gates which widths a
            // _lowering_ may emit; a body is either right at a width or it is not, and these are
            // written here. Sweeping only what is currently reachable would leave `sdivrem` and
            // `ashr_by` unchecked at every width they exist for.
            for bits in swept_widths() {
                for (a, b) in operand_pairs(op, bits) {
                    let got = intn_lane(op, bits as u64, a.limbs(), b.limbs());

                    // Rule 3, which binds on every input including the unspecified ones: whatever
                    // the answer is, it is inside the width.
                    assert_eq!(
                        got[got.len() - 1] & !top_cell_mask(bits as u64),
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
                    // `slt` is swept at every wide width even though no lowering reaches one yet:
                    // the reading is this lane's own, and `corners::signed_width_ok` gates which
                    // widths a _lowering_ may use rather than which ones a body must be right at.
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
