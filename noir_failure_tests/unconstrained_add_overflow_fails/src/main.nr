// An overflowing `+` in _unconstrained_ code must be rejected.
//
// This is the route that made the missing check a silent wrong answer rather than merely a missing
// rejection. `witness_taint_inference` records unconstrained results as Pure, so every integer
// `+ - *` inside an unconstrained function is a pure but _runtime-varying_ operation: the range
// domain cannot discharge it, no constraint is emitted for it, and both the VM and the WASM backend
// simply wrap it.
//
// Noir applies the same rule to unconstrained code as to constrained: `brillig_gen` emits an
// `add_overflow_check` ahead of the operation, so brillig's own wrapping `add` is unreachable in
// generated code.
//
// `x` is a witness, so the sum genuinely varies at run time and cannot be folded away.

unconstrained fn add_u8(a: u8, b: u8) -> u8 {
    a + b
}

fn main(x: u8) {
    // Safety: the sum is asserted below; the overflow it commits is the point of the test.
    let r = unsafe { add_u8(x, 100) };
    // 200 + 100 overflows a `u8`; `44` is the wrapped answer the VM and WASM both produced.
    assert(r == 44);
}
