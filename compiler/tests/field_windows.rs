mod common;

use mavros_artifacts::{Bn254, Goldilocks, MavrosField};

/// `u8`, `u16` and `u32` windows of the witnessed element `f = x + y`, and a `u16` window of the
/// `u32` one.
const WINDOWS_OF_ONE_ELEMENT: &str = "fn main(x: u32, y: u32) -> pub (u8, u16, u32, u16) {
        let f = (x as Field) + (y as Field);
        (f as u8, f as u16, f as u32, (f as u32 >> 16) as u16)
    }";

/// Several windows of one witnessed element are taken from one decomposition of it. Each still
/// reads the bits it names, constrained and unconstrained, on both fields.
fn several_windows_of_one_element_each_read_their_bits<F: MavrosField>() {
    for (x, y) in [
        (0u64, 0u64),
        (0x1234_5678, 0x9abc_def0),
        (u32::MAX.into(), u32::MAX.into()),
        (0xff, 1),
    ] {
        let sum = x + y;
        let inputs = format!(
            "x = {x}\ny = {y}\nreturn = [{}, {}, {}, {}]",
            sum & 0xff,
            sum & 0xffff,
            sum & 0xffff_ffff,
            (sum & 0xffff_ffff) >> 16
        );
        common::check::<F>(WINDOWS_OF_ONE_ELEMENT, &inputs);
    }
}

#[test]
fn several_windows_of_one_element_each_read_their_bits_on_bn254() {
    several_windows_of_one_element_each_read_their_bits::<Bn254>();
}

#[test]
fn several_windows_of_one_element_each_read_their_bits_on_goldilocks() {
    several_windows_of_one_element_each_read_their_bits::<Goldilocks>();
}

/// The same windows of three different elements pay for three decompositions, and so for more
/// rows than the windows of one element, which share one.
fn the_windows_of_one_element_share_a_decomposition<F: MavrosField>() {
    let apart = "fn main(x: u32, y: u32) -> pub (u8, u16, u32) {
            let f = (x as Field) + (y as Field);
            (f as u8, (f + 1) as u16, (f + 2) as u32)
        }";
    let together = "fn main(x: u32, y: u32) -> pub (u8, u16, u32) {
            let f = (x as Field) + (y as Field);
            (f as u8, f as u16, f as u32)
        }";
    let rows = |source| common::compile::<F>(source).r1cs.constraints.len();
    let (apart, together) = (rows(apart), rows(together));
    assert!(
        together < apart,
        "{:?}: {together} rows for the windows of one element, {apart} for three",
        F::ID
    );
}

#[test]
fn the_windows_of_one_element_share_a_decomposition_on_bn254() {
    the_windows_of_one_element_share_a_decomposition::<Bn254>();
}

#[test]
fn the_windows_of_one_element_share_a_decomposition_on_goldilocks() {
    the_windows_of_one_element_share_a_decomposition::<Goldilocks>();
}
