// A shift by an amount at or above the operand width must be rejected, on the path where the shift
// is *unguarded* and both its operands are pure.
//
// That is `LowerPureGuards`' own arm, and it is a different lowering from the one
// `pure_guarded_shift` covers: there the shift sits under a witness-conditional guard, so an
// out-of-range amount asserts the guard is inactive rather than failing outright. Here there is no
// guard, so the amount is simply asserted in range.
//
// The shift is put behind a function taking both operands as parameters, which is what keeps it
// out of the constant folder's reach -- `LowerPureGuards` runs long before `Specializer`, so an
// amount that only becomes a literal after inlining still reads as full-width when the check is
// emitted. Written as a bare `200 << 9` the fold happens first and this covers nothing.
//
// Noir rejects the same program, so this is agreement rather than added strictness: a constant
// amount at or above the width is `Failure("attempt to bit-shift with overflow")` in
// `noirc_evaluator`'s `binary.rs`, and a variable one gets a `rhs < bit_size` constraint from
// `remove_bit_shifts::enforce_bitshift_rhs_lt_bit_size`.
//
// Before the check existed this program ran clean and answered `0` -- the VM's `shl_int` masks the
// amount to `bits - 1` -- which is a wrong answer rather than a rejection, and the worst of the
// two ways to be wrong.

fn shl_u8(v: u8, n: u8) -> u8 {
    v << n
}

fn main(w: Field) {
    // Keeps `main` from folding away entirely; the shift below is what is under test.
    assert(w != 0);
    // `9` is above the eight bits of a `u8`.
    assert(shl_u8(200, 9) == 0);
}
