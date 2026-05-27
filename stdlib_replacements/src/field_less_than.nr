// Replacement for the `field_less_than` builtin.
//
// Noir's stdlib exposes `field_less_than` as an unconstrained builtin used as
// a hint for field comparisons. Mavros has no native field-less-than opcode,
// so we implement it by decomposing both inputs into big-endian bytes and
// comparing them from the most significant byte down. Because this runs in
// unconstrained context we can use a normal early-`break` loop.

pub unconstrained fn field_less_than(x: Field, y: Field) -> bool {
    let x_bytes: [u8; 32] = x.to_be_bytes();
    let y_bytes: [u8; 32] = y.to_be_bytes();
    let mut result = false;
    for i in 0..32 {
        let xb = x_bytes[i];
        let yb = y_bytes[i];
        if xb != yb {
            result = xb < yb;
            break;
        }
    }
    result
}
