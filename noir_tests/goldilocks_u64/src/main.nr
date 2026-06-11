// Integer program whose constants exceed the Goldilocks modulus (p = 2^64-2^32+1).
// The mono AST carries these as field-agnostic SignedField; Mavros lowers them natively
// (to_u128 -> Constant::U(64, ..)), so they round-trip even when the source field is Goldilocks.
fn main(x: u64) -> pub u64 {
    let big: u64 = 18446744069414584322; // p + 1, a legal u64
    let prod = x * 2;
    prod + big
}
