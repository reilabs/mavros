mod common;

use mavros_artifacts::{Field, Goldilocks, InputValueOrdered, MavrosField};
use mavros_compiler::{
    api,
    compiler::{
        passes::{shared::MIN_LOOPED_RUN, wide_witness_ints::int_limb_widths},
        util::biguint_of_limbs,
    },
};
use num_bigint::{BigInt, BigUint};

fn pow2(bits: usize) -> BigInt {
    BigInt::from(1u8) << bits
}

#[test]
fn wide_parameters_cross_as_limbs_on_both_fields() {
    // On bn254 `x`, `y`, `p.1` and `s.hi` cross as limbs, every one with a value past the modulus;
    // on Goldilocks every integer but `p.0` and `s.lo` does.
    let main = "struct S { lo: u8, hi: u260 }
        fn main(x: pub u256, y: [i300; 2], z: pub u128, p: (u8, u512), s: S) -> pub u32 {
            assert(y[0] == -y[1]);
            assert(x + (z as u256) == 0xffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff);
            ((x >> 248) as u32) + ((p.1 >> 504) as u32) + (p.0 as u32) + ((y[1] >> 290) as u32)
                + ((s.hi >> 252) as u32) + (s.lo as u32)
        }";
    let inputs = format!(
        "x = '{}'\ny = ['{}', '{}']\nz = 5\np = [7, '{}']\nreturn = 1030\n[s]\nlo = 1\nhi = '{}'",
        pow2(256) - 6u8,
        -(pow2(299) - 1u8),
        pow2(299) - 1u8,
        pow2(511) + 3u8,
        pow2(259),
    );
    common::check_both_fields(main, &inputs);
}

/// A wide parameter read at a witness index goes through the array lookup, limb by limb.
#[test]
fn a_wide_parameter_array_is_read_at_a_witness_index() {
    let main = "fn main(xs: [u256; 3], i: u32) -> pub u32 { (xs[i] >> 224) as u32 }";
    let inputs = format!(
        "xs = [1, '{}', 3]\ni = 1\nreturn = 4294967295",
        pow2(256) - 1u8
    );
    common::check_both_fields(main, &inputs);
}

/// Goldilocks holds no `u64` in one element: a `u64` past its modulus crosses as limbs.
#[test]
fn a_u64_past_the_goldilocks_modulus_crosses() {
    common::check::<Goldilocks>(
        "fn main(a: u64, b: pub u64) -> pub u32 { ((a + b) >> 40) as u32 }",
        "a = '18446744073709551000'\nb = 615\nreturn = 16777215",
    );
}

/// Each wide parameter and return takes one column per limb, and the manifest names them.
#[test]
fn the_manifest_gives_a_wide_integer_one_column_per_limb() {
    fn widths<F: MavrosField>() -> (Vec<usize>, usize) {
        let compiled = common::compile::<F>(
            "fn main(x: u256, y: pub [u128; 2], z: u8) -> pub (u8, u256) {
                (z + (y[1] as u8), x)
            }",
        );
        let manifest = compiled.driver.witness_manifest(&compiled.r1cs).unwrap();
        let params = manifest
            .parameters
            .iter()
            .map(|param| param.width)
            .collect();
        (params, manifest.returns.unwrap().len)
    }
    // 64-bit limbs on bn254, which holds a `u128` in one element; 16-bit limbs on Goldilocks.
    assert_eq!(widths::<Field>(), (vec![4, 2, 1], 5));
    assert_eq!(widths::<Goldilocks>(), (vec![16, 16, 1], 17));
}

/// A wide return is published as its limbs, from a parameter, a constant or arithmetic, wherever
/// it sits in the type. The `return` the inputs declare is the plain integer.
#[test]
fn wide_returns_are_published_as_limbs_on_both_fields() {
    // On Goldilocks the `u64` is wide too, and `z + 1` is past its modulus.
    let main = "fn main(x: u256, y: pub i300, z: u64) -> pub (u256, [i300; 2], u512, u8, u64) {
            (x + 1, [y, -y], 0xff << 504, 3, z + 1)
        }";
    let goldilocks = biguint_of_limbs(Goldilocks::CONFIG.modulus_limbs());
    // `x + 1` is past bn254's modulus, and its limbs differ, so a limb out of place shows.
    let x = pow2(255) + pow2(130) + 4u8;
    let inputs = format!(
        "x = '{x}'\ny = '{}'\nz = '{}'\nreturn = ['{}', ['{}', '{}'], '{}', 3, '{}']",
        -(pow2(299) - 1u8),
        &goldilocks + 3u8,
        &x + 1u8,
        -(pow2(299) - 1u8),
        pow2(299) - 1u8,
        pow2(512) - pow2(504),
        &goldilocks + 4u8,
    );
    common::check_both_fields(main, &inputs);
}

/// A return's limbs tell `v` from `v + p`, which one column holding its residue would not.
#[test]
fn a_wide_return_is_not_taken_for_its_residue() {
    fn check<F: MavrosField>(typ: &str) {
        let modulus = biguint_of_limbs(F::CONFIG.modulus_limbs());
        for mode in ["", "unconstrained "] {
            let mut compiled =
                common::compile::<F>(&format!("{mode}fn main(x: {typ}) -> pub {typ} {{ x }}"));
            let (params, declared_return) = compiled.inputs("x = 5\nreturn = 5");
            compiled.run(&params, declared_return.as_ref()).unwrap();
            let (params, declared_return) =
                compiled.inputs(&format!("x = 5\nreturn = '{}'", &modulus + 5u8));
            let error = compiled
                .run(&params, declared_return.as_ref())
                .expect_err("a declared return of 5 + p is refused");
            assert!(error.starts_with("declared return"), "{}: {error}", F::ID);
        }
    }
    check::<Field>("u256");
    check::<Goldilocks>("u64");
}

/// A prover that adds `2^w` to one limb and takes one from the next keeps the integer the limbs
/// spell. Only the limb's range check refuses it, so each limb's check is load-bearing.
///
/// A limb past its width has no entry in its range check's table, so witness generation traps
/// there; a witness built around the trap would break the check's constraints instead.
fn a_limb_out_of_range_is_refused<F: MavrosField>() {
    let source = "fn main(x: u300) -> pub u8 { (x >> 292) as u8 }";
    // Every limb is all ones, so each forgery leaves only the limb under test out of range.
    let value: BigUint = (BigUint::from(1u8) << 300) - 1u8;
    let widths = int_limb_widths(300, F::CONFIG);
    for mode in ["", "unconstrained "] {
        let mut compiled = common::compile::<F>(&format!("{mode}{source}"));
        let (params, declared_return) = compiled.inputs(&format!("x = '{value}'\nreturn = 255"));
        compiled.run(&params, declared_return.as_ref()).unwrap();

        let InputValueOrdered::Vec(limbs) = &params[0] else {
            panic!("a u300 crosses as limbs on {}", F::ID);
        };
        assert_eq!(limbs.len(), widths.len());
        for (index, width) in widths.iter().enumerate() {
            let mut forged = limbs.clone();
            let InputValueOrdered::Field(limb) = &forged[index] else { unreachable!() };
            forged[index] = InputValueOrdered::Field(*limb + F::from(1u128 << width));
            if let Some(InputValueOrdered::Field(next)) = forged.get(index + 1).cloned() {
                forged[index + 1] = InputValueOrdered::Field(next - F::ONE);
            }
            let error = compiled
                .run(&[InputValueOrdered::Vec(forged)], None)
                .expect_err(&format!("limb {index} of {} past 2^{width}", widths.len()));
            assert!(
                error.starts_with("witness generation trapped")
                    || error == "the witness breaks a constraint",
                "{error}"
            );
        }
    }
}

#[test]
fn a_limb_out_of_range_is_refused_on_bn254() {
    a_limb_out_of_range_is_refused::<Field>();
}

#[test]
fn a_limb_out_of_range_is_refused_on_goldilocks() {
    a_limb_out_of_range_is_refused::<Goldilocks>();
}

/// A returned array of `MIN_LOOPED_RUN` or more elements is bound in a counted loop, nested arrays
/// in nested loops. Its elements, wide limbs and tuple fields included, take the return's columns in
/// order, and every column is held to its element: a witness with any one of them off by one breaks
/// a constraint.
fn a_looped_return_is_published_in_order_and_bound<F: MavrosField>() {
    let main =
        "fn main(x: u8, y: u256) -> pub ([u8; 6], [[u16; 4]; 4], [u256; 4], [(u8, u32); 4]) {
            let mut a = [0; 6];
            for i in 0..6 { a[i] = x + i as u8; }
            let mut b = [[0; 4]; 4];
            for i in 0..4 { for j in 0..4 { b[i][j] = (x as u16) * 16 + (i * 4 + j) as u16; } }
            let mut c = [0; 4];
            for i in 0..4 { c[i] = y + i as u256; }
            let mut d = [(0, 0); 4];
            for i in 0..4 { d[i] = (x + i as u8, (x as u32) << (8 * i)); }
            (a, b, c, d)
        }";
    let (x, y) = (7u32, pow2(255) + pow2(130) + 4u8);
    let a: Vec<_> = (0..6).map(|i| (x + i).to_string()).collect();
    let b: Vec<_> = (0..4)
        .map(|i| {
            let row: Vec<_> = (0..4).map(|j| (x * 16 + i * 4 + j).to_string()).collect();
            format!("[{}]", row.join(", "))
        })
        .collect();
    let c: Vec<_> = (0..4u8).map(|i| format!("'{}'", &y + i)).collect();
    let d: Vec<_> = (0..4)
        .map(|i| format!("[{}, {}]", x + i, x << (8 * i)))
        .collect();
    let declared = |a: &[String]| {
        format!(
            "x = {x}\ny = '{y}'\nreturn = [[{}], [{}], [{}], [{}]]",
            a.join(", "),
            b.join(", "),
            c.join(", "),
            d.join(", ")
        )
    };
    let mut swapped = a.clone();
    swapped.swap(2, 3);
    let width = 6 + 16 + 4 * int_limb_widths(256, F::CONFIG).len() + 8;

    for mode in ["", "unconstrained "] {
        let source = main.replace("fn main", &format!("{mode}fn main"));
        let mut compiled = common::compile::<F>(&source);
        let (params, declared_return) = compiled.inputs(&declared(&a));
        compiled
            .run(&params, declared_return.as_ref())
            .unwrap_or_else(|error| panic!("{source}: {error}"));
        let (_, out_of_order) = compiled.inputs(&declared(&swapped));
        let error = compiled
            .run(&params, out_of_order.as_ref())
            .expect_err("two elements of the return swapped");
        assert!(error.starts_with("declared return"), "{}: {error}", F::ID);

        let result =
            api::run_witgen_from_binary(&mut compiled.binary, &compiled.r1cs, &params, None)
                .unwrap();
        let witness = [
            result.out_wit_pre_comm.as_slice(),
            result.out_wit_post_comm.as_slice(),
        ]
        .concat();
        assert!(compiled.r1cs.unsatisfied_constraint(&witness).is_none());
        let start = compiled
            .driver
            .entry_columns()
            .and_then(|columns| columns.returns)
            .expect("the return has columns");
        for column in start..start + width {
            let mut forged = witness.clone();
            forged[column] += F::ONE;
            assert!(
                compiled.r1cs.unsatisfied_constraint(&forged).is_some(),
                "{:?} {mode}: return column {} of {width} is not bound",
                F::ID,
                column - start
            );
        }
    }
}

#[test]
fn a_looped_return_is_published_in_order_and_bound_on_bn254() {
    a_looped_return_is_published_in_order_and_bound::<Field>();
}

#[test]
fn a_looped_return_is_published_in_order_and_bound_on_goldilocks() {
    a_looped_return_is_published_in_order_and_bound::<Goldilocks>();
}

/// A constant wide return is published as its constant limbs, and each of its columns is held to
/// its limb: a witness with any one of them off by one breaks a constraint. On Goldilocks the
/// `u64` is wide too.
fn a_constant_wide_return_is_bound<F: MavrosField>() {
    let main = "fn main(x: u8) -> pub (u8, u512, u64) { (x, 0xff << 504, 0xffffffffffffffff) }";
    let inputs = format!(
        "x = 3\nreturn = [3, '{}', '{}']",
        pow2(512) - pow2(504),
        u64::MAX
    );
    let width = 1 + int_limb_widths(512, F::CONFIG).len() + int_limb_widths(64, F::CONFIG).len();
    for source in common::in_both_modes(main) {
        let mut compiled = common::compile::<F>(&source);
        let (params, declared_return) = compiled.inputs(&inputs);
        compiled
            .run(&params, declared_return.as_ref())
            .unwrap_or_else(|error| panic!("{source}: {error}"));
        let result =
            api::run_witgen_from_binary(&mut compiled.binary, &compiled.r1cs, &params, None)
                .unwrap();
        let witness = [
            result.out_wit_pre_comm.as_slice(),
            result.out_wit_post_comm.as_slice(),
        ]
        .concat();
        let start = compiled
            .driver
            .entry_columns()
            .and_then(|columns| columns.returns)
            .expect("the return has columns");
        for column in start..start + width {
            let mut forged = witness.clone();
            forged[column] += F::ONE;
            assert!(
                compiled.r1cs.unsatisfied_constraint(&forged).is_some(),
                "{:?} {source}: return column {} of {width} is not bound",
                F::ID,
                column - start
            );
        }
    }
}

#[test]
fn a_constant_wide_return_is_bound_on_bn254() {
    a_constant_wide_return_is_bound::<Field>();
}

#[test]
fn a_constant_wide_return_is_bound_on_goldilocks() {
    a_constant_wide_return_is_bound::<Goldilocks>();
}

/// A returned array of `MIN_LOOPED_RUN` or more elements is bound in a loop, so a longer one adds
/// no code to the entry point.
#[test]
fn a_long_returned_array_is_bound_in_a_loop() {
    let words = |len: usize| {
        common::compile::<Field>(&format!(
            "fn main(x: u8) -> pub [u8; {len}] {{ [x; {len}] }}"
        ))
        .binary
        .len()
    };
    let (short, long) = (words(MIN_LOOPED_RUN), words(64));
    assert!(
        long.abs_diff(short) < 64,
        "{short} words for {MIN_LOOPED_RUN} elements, {long} for 64"
    );
}

/// A looped return of zero-width elements takes no columns.
#[test]
fn a_looped_return_of_zero_width_elements_takes_no_columns() {
    common::check_both_fields(
        "struct Empty {}
        fn main(x: u8) -> pub [Empty; 4] { let _ = x; [Empty {}; 4] }",
        "x = 1\nreturn = [{}, {}, {}, {}]",
    );
    common::check_both_fields(
        "fn main(x: u8) -> pub [[u8; 0]; 4] { let _ = x; [[]; 4] }",
        "x = 1\nreturn = [[], [], [], []]",
    );
}
