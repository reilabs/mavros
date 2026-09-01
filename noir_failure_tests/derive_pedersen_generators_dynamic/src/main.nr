use std::embedded_curve_ops::EmbeddedCurvePoint;
use std::hash::derive_generators;

fn main(starting_index: u32) {
    let _: [EmbeddedCurvePoint; 1] =
        derive_generators("DEFAULT_DOMAIN_SEPARATOR".as_bytes(), starting_index);
}
