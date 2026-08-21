// A left shift whose operands are both *pure* -- no witness anywhere in the shift
// itself -- sitting under a witness-conditional guard. That is the shape
// `LowerPureGuards::lower_shift_guard` handles, and it is a different lowering
// from the one `signed_shift` and `inactive_witness_shift` cover: those two have
// a witness operand, so they go through `witness_bitwise` instead.
//
// The shifts are put behind a function taking both operands as parameters,
// which is what keeps them out of the constant folder's reach. The one live
// instance of this shape in the corpus today has exactly that form --
// `mavros_stdlib/replacements/sha256_compression.nr`'s `fn pow2(n: u64)`, whose
// `1u64 << n` cannot overflow `u64` and so never exercised the interesting case.
//
// What is being pinned: `<<` **wraps**. Bits pushed off the top are discarded,
// and only an out-of-range *amount* is a runtime error. The pure path used to
// treat a value that left the width as a guard failure, which made an active
// guard around a wrapping shift unsatisfiable. Every expected value below is
// checked against Noir's own execution by the `Witgen Correct` lane, so this is
// a test of agreement with Noir rather than of arithmetic done here.

fn shl_u8(v: u8, n: u8) -> u8 {
    v << n
}

fn shl_i8(v: i8, n: i8) -> i8 {
    v << n
}

fn main(cond: bool) {
    if cond {
        // Unsigned, wrapping. 200 << 1 is 400, which keeps only its low eight bits.
        assert(shl_u8(200, 1) == 144);
        assert(shl_u8(200, 2) == 32);
        // Everything shifted out.
        assert(shl_u8(200, 8 - 1) == 0);
        // Controls: the same lowering on shifts that do not overflow.
        assert(shl_u8(1, 7) == 128);
        assert(shl_u8(1, 0) == 1);

        // Signed, where wrapping also changes the sign. -16 is 0xF0, so 0xF0 << 3
        // is 0x80 (-128) and 0xF0 << 4 keeps nothing.
        assert(shl_i8(-16, 3) == -128);
        assert(shl_i8(-16, 4) == 0);
        // Two positive operands producing a negative result, which is the case a
        // signed-fits check rejects.
        assert(shl_i8(64, 1) == -128);
        assert(shl_i8(-8, 1) == -16);
        assert(shl_i8(-8, 0) == -8);
    }
}
