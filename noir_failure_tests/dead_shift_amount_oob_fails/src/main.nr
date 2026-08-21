// A shift by an out-of-range amount must be rejected even when nothing reads its result.
//
// This is the shift's half of what `unguarded_divmod_operands` has always done for a division.
// `DCE(Config::preserve_blocks())` runs at `driver.rs:302` and `:365`, well before
// `InstructionLowering::pure_guards()` at `:463` emits the amount check -- so a dead shift used to
// be deleted outright and the failure Noir promises went with it. The live sibling
// `pure_shift_amount_oob_fails` was rejected while this one ran clean, which is the worst kind of
// disagreement: the same expression accepted or rejected according to whether anything happened to
// read it.
//
// DCE now replaces the dead shift with its amount check alone, exactly as it replaces a dead
// division with `assert(!undefined(lhs, rhs))`. The arithmetic still goes; only the rejection
// stays.
//
// The shift is inside a callee whose *return value* is used, so the call itself is live and only
// the shift is dead. Both operands are parameters, which keeps the amount out of the constant
// folder's reach at the point DCE decides -- `Specializer` runs at `:502`, long after.

fn shl_and_ignore(v: u8, n: u8) -> u8 {
    // Dead: nothing reads `_dead`, so DCE would delete this shift if it were not a partial op.
    let _dead = v << n;
    v
}

fn main(w: Field) {
    // Keeps `main` from folding away entirely; the shift inside the call is what is under test.
    assert(w != 0);
    // `9` is above the eight bits of a `u8`. The equality itself holds -- `shl_and_ignore` returns
    // its first argument -- so the amount check is the only thing here that can reject.
    assert(shl_and_ignore(200, 9) == 200);
}
