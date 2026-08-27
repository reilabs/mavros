// An unguarded pure `+` that overflows its width must be rejected.
//
// Noir rejects it in both directions: `eval_constant_binary_op` reports
// `Failure("attempt to add with overflow")` for a constant pair, and `brillig_gen` emits an
// `add_overflow_check` for a runtime one. Mavros checked neither. `LowerPureGuards` had unguarded
// arms for `Div`/`Mod` and `Shl`/`Shr` but none for the arithmetic trio, so an overflowing pure add
// simply _wrapped_ -- and it wrapped the same way everywhere, `hlssa_to_r1cs::arith` folding with
// `wrapping_add` while the VM's `cell_add` masks, so R1CS and witness generation agreed on the
// wrong answer and no oracle in the test runner could see the difference.
//
// The operands are behind a function so the fold is not the thing under test: `LowerPureGuards`
// runs long before `Specializer`, so a value that only becomes a literal after inlining still
// reads as full-width when the check is emitted. Written as a bare `200 + 100` the fold happens
// first and this covers nothing.
//
// The equality is written against the _wrapped_ answer, which is what the program produced before
// the check existed. So this test states the old behaviour exactly, and the only thing that can
// reject it is the new overflow check.

fn add_u8(a: u8, b: u8) -> u8 {
    a + b
}

fn main(w: Field) {
    // Keeps `main` from folding away entirely; the addition below is what is under test.
    assert(w != 0);
    // 200 + 100 is 300, which is above the 255 a `u8` holds. `44` is the wrapped answer.
    assert(add_u8(200, 100) == 44);
}
