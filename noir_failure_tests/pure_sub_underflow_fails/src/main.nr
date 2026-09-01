// An unguarded pure `-` that borrows below zero must be rejected.
//
// The subtraction half of `pure_add_overflow_fails`, and a distinct lowering: the unsigned wrap
// test is `lhs < wrapped` rather than `wrapped < lhs`, since a difference that borrows lands
// _above_ the operand it was subtracted from. Getting the two comparisons the same way round is
// exactly the mistake this pins.
//
// Noir calls this "attempt to subtract with overflow" and rejects it identically.

fn sub_u8(a: u8, b: u8) -> u8 {
    a - b
}

fn main(w: Field) {
    // Keeps `main` from folding away entirely; the subtraction below is what is under test.
    assert(w != 0);
    // 1 - 2 borrows. `255` is the wrapped answer, which is what ran clean before the check.
    assert(sub_u8(1, 2) == 255);
}
