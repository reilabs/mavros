// An overflowing `+` must be rejected even when nothing reads its result.
//
// This is the arithmetic trio's half of what `unguarded_divmod_operands` has always done for a
// division and `unguarded_shift_operands` for a shift. `DCE(Config::preserve_blocks())` runs well
// before `InstructionLowering::pure_guards()` emits the overflow check, so a dead overflowing add
// used to be deleted outright and the failure Noir promises went with it. That is the worst kind of
// disagreement: the same expression accepted or rejected according to whether anything happened to
// read it.
//
// Noir authorises keeping it. `die.rs:456` makes a `Binary` eliminable only when
// `!requires_acir_gen_predicate`, and that is `true` for a checked `Add`/`Sub`/`Mul`, so Noir keeps
// a dead overflowing add too.
//
// Unlike the division and shift rewrites, this one does not save the arithmetic: the sum has to be
// recomputed for the check to compare against. What goes is the _use_ of it.
//
// The add is inside a callee whose return value is used, so the call itself is live and only the
// add is dead. Both operands are parameters, which keeps them out of the constant folder's reach at
// the point DCE decides -- `Specializer` runs much later.

fn add_and_ignore(a: u8, b: u8) -> u8 {
    // Dead: nothing reads `_dead`, so DCE would delete this add if it were not a partial op.
    let _dead = a + b;
    a
}

fn main(w: Field) {
    // Keeps `main` from folding away entirely; the add inside the call is what is under test.
    assert(w != 0);
    // 200 + 100 overflows a `u8`. The equality itself holds -- `add_and_ignore` returns its first
    // argument -- so the overflow check is the only thing here that can reject.
    assert(add_and_ignore(200, 100) == 200);
}
