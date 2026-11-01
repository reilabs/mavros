// The vendored gadgets use 120-bit limbs and products up to 246 bits in one Field.
#[field(bn254)]
pub mod bignum;
#[field(bn254)]
pub mod bigcurve;
