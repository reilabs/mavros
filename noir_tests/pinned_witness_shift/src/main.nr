// Shifts whose amount is a *witness* that the range domain nonetheless pins to
// a single value.
//
// `LowerWitnessBitwiseOps::lower_shift` used to split on the amount's type: a
// pure amount folded into a constant factor, a witness one paid for a runtime
// factor. Type is the wrong question. What the constant-amount lowering needs
// is that `1 << amount` folds, and an amount the range domain has pinned folds
// whether or not it is still called a witness -- `WitnessOf` is transparent to
// the domain, so a value can be provably 3 and witness-typed at once.
//
// `(w as u32) / 256` is the shape that separates the two. `w` is a witness, so
// `n` is one all the way to lowering, but `w as u32` cannot exceed 255 and the
// division therefore has exactly one answer. The range domain proves that; the
// constant folder cannot see it, because neither operand of the division is a
// constant the folder can substitute.
//
// Nothing else in the corpus reaches this path today, which is the reason this
// test exists: the dispatch is otherwise exercised only by unit tests, and a
// change that quietly stopped consulting the domain would move no number.
fn main(a: u32, b: u32, s: i32, w: u8) {
    // Pinned to 3, and witness-typed. Both of these take the constant-amount
    // lowering, so neither builds a powers-of-two table.
    let n = (w as u32) / 256 + 3;
    assert(a << n == 40);
    assert(b >> n == 4);

    // A signed left-hand side keeps the general lowering however well known its
    // amount is, because the constant-amount lowering is unsigned-only. This is
    // the control: it must not move when the dispatch above changes.
    //
    // It also allocates a powers-of-two table that the two shifts above no
    // longer need, so the row count for this package understates what pinning
    // saves per site -- the table is paid for either way.
    let m = (w as i32) / 256 + 3;
    assert(s << m == -32);
}
