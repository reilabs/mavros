// Replacement for the `field_less_than` builtin.
//
// Noir's stdlib exposes `field_less_than` as an unconstrained builtin used as
// a hint for field comparisons. Mavros has no native field-less-than opcode,
// so we implement it by decomposing both inputs into little-endian bits and
// comparing them from the most significant bit down. We use bits (rather than
// bytes) because mavros' witgen codegen has bytecode for `ToBitsLe` but not
// for the byte-decomposition wrapper `to_be_bytes`. Because this runs in
// unconstrained context we can use a normal early-`break` loop.

pub unconstrained fn field_less_than(x: Field, y: Field) -> bool {
    // BN254 elements fit in 254 bits, so a 254-bit decomposition is enough.
    let x_bits: [u1; 254] = x.to_le_bits();
    let y_bits: [u1; 254] = y.to_le_bits();
    let mut result = false;
    for i in 0..254 {
        let idx = 254 - 1 - i;
        let xb = x_bits[idx];
        let yb = y_bits[idx];
        if xb != yb {
            // Big-endian first-differing bit decides the order.
            result = xb < yb;
            break;
        }
    }
    result
}
