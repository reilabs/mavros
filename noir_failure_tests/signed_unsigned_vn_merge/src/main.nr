// A signed and an unsigned operation over congruent operands must not be value-numbered
// together, even where the two forms agree on every bit of the result.
//
// `Add`, `Sub` and `Mul` produce the same bits under either reading; the sign selects only
// which operand values make them fail. That is not a difference a value-numbering pass may
// discard, because the merge picks one survivor and the survivor's opcode is what
// `LowerWitnessIntegerArithOps` reads to decide which rejecting constraints to emit. Merging
// the unsigned form into the signed one deletes the unsigned overflow check outright.
//
// Each pair below is a signed operation and an unsigned one over operands the analyses see as
// equal: `x as i8` and `x as u8` are the same expression (both are a `bit_range` and a cast to
// `int8` over the same input, since an HLSSA integer type carries no sign), and the literal
// `10` is one interned `Constant::Int(8, 10)` whichever way it is read. Only the unsigned half
// of each pair fails; the signed half is there purely to be congruent with it.
//
// This is an execution_failure test, so the assertion that matters is the one the harness
// makes: the program must be rejected. What makes it load-bearing is that it was *accepted*
// while `PRE::pre_untaint` keyed these operations sign-free -- all three failures vanished.
fn main(x: Field, y: Field) {
    // 200 + 200 overflows a `u8`; read as `i8` the same bits are -56, and -56 + -56 = -112 fits.
    let xs = x as i8;
    let xu = x as u8;
    let add_s = xs + xs;
    let add_u = xu + xu;

    // 200 * 2 overflows a `u8`; -56 * 2 = -112 fits.
    let mul_s = xs * 2;
    let mul_u = xu * 2;

    // 5 - 10 underflows a `u8`; as `i8` it is -5.
    let ys = y as i8;
    let yu = y as u8;
    let sub_s = ys - 10;
    let sub_u = yu - 10;

    // Keep every result live so none of the operations is eliminated as dead. The comparisons
    // are all true, so they are not what rejects the program.
    assert(add_s != 111);
    assert(add_u != 111);
    assert(mul_s != 111);
    assert(mul_u != 111);
    assert(sub_s != 111);
    assert(sub_u != 111);
}
