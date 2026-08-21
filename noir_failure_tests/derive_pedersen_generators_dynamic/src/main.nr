use std::embedded_curve_ops::EmbeddedCurvePoint;
use std::hash::derive_generators;

fn derive_from_input(offset: u32) -> [EmbeddedCurvePoint; 1] {
    derive_generators("DEFAULT_DOMAIN_SEPARATOR".as_bytes(), offset)
}

fn main(offset: u32) {
    let _ = derive_from_input(offset);
}
