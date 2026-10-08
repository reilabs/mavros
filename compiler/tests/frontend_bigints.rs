mod common;

use mavros_compiler::{abi_helpers, api, compiler::codegen::CodeGenOptions};

/// Exercise source parsing, monomorphization, input decoding, witness generation and R1CS.
/// Check a declared return as well so a missing high limb cannot pass unnoticed.
fn check(source: &str, inputs: &str) {
    let dir = common::noir_project(source);
    if source.contains("match ") {
        let manifest = dir.path().join("Nargo.toml");
        let contents = std::fs::read_to_string(&manifest).unwrap();
        std::fs::write(
            manifest,
            contents + "compiler_unstable_features = ['enums']\n",
        )
        .unwrap();
    }
    let (mut driver, r1cs) = api::compile_to_r1cs(dir.path().to_path_buf(), false)
        .unwrap_or_else(|error| panic!("{source}: {error:?}"));
    let mut binary = api::compile_bytecode(
        &mut driver,
        CodeGenOptions {
            check_constraints: true,
            ..Default::default()
        },
    )
    .unwrap();
    std::fs::write(dir.path().join("Prover.toml"), inputs).unwrap();
    let params = api::read_prover_inputs(dir.path(), driver.abi()).unwrap();
    let result = api::run_witgen_from_binary(&mut binary, &r1cs, &params, None).unwrap();
    assert!(api::check_witgen(&r1cs, &result));
    abi_helpers::check_return_guard(
        driver.abi(),
        &r1cs.witness_layout,
        &params,
        &result.out_wit_pre_comm,
    )
    .unwrap();
}

fn check_both_modes(source: &str, inputs: &str) {
    check(source, inputs);
    check(&source.replace("fn main", "unconstrained fn main"), inputs);
}

#[test]
fn wide_arithmetic_from_source() {
    check_both_modes(
        include_str!("../../noir_tests/bigint_arithmetic/src/main.nr"),
        include_str!("../../noir_tests/bigint_arithmetic/Prover.toml"),
    );
}

#[test]
fn wide_signed_arithmetic_from_source() {
    check_both_modes(
        r#"fn main(x: i64) -> pub [i64; 3] {
            let high: i256 = 1 << 200;
            let negative: i256 = -0x100000000000000000000000000000000000000000000000000;
            assert(negative == -high);
            let value = negative + (x as i256);
            assert(value < 0);
            let product = value * -3;
            assert(product / -3 == value);
            [(value >> 200) as i64, ((value as i512) >> 256) as i64,
             (value + high) as i64]
        }"#,
        "x = 9\nreturn = ['-1', '-1', 9]",
    );
}

#[test]
fn generic_wide_integers_in_arrays_and_vectors() {
    check_both_modes(
        include_str!("../../noir_tests/bigint_sequences/src/main.nr"),
        include_str!("../../noir_tests/bigint_sequences/Prover.toml"),
    );
}

#[test]
fn wide_abi_from_source() {
    check_both_modes(
        include_str!("../../noir_tests/bigint_abi/src/main.nr"),
        include_str!("../../noir_tests/bigint_abi/Prover.toml"),
    );
}

#[test]
fn wide_match_literals_and_signed_literals_keep_their_value() {
    check(
        r#"fn main(x: Field, n: i64) -> pub [u64; 2] {
            let tag = match x as u200 {
                0x1000000000000000000000000000000000000000000000 => 7,
                _ => 9,
            };
            assert(n == -42);
            assert((-7 as Field) + 7 == 0);
            [tag, (n as u64)]
        }"#,
        "x = '0x1000000000000000000000000000000000000000000000'\nn = '-42'\nreturn = [7, '18446744073709551574']",
    );
}

#[test]
fn the_maximum_width_and_nonstandard_signed_width_reach_source_lowering() {
    check(
        r#"fn main(n: i64) -> pub [u64; 2] {
            let n = n as i33;
            let high: u16384 = 1 << 16383;
            assert(high >> 16383 == 1);
            let low: u3 = 7;
            assert(low == 7);
            [(high >> 16383) as u64, ((n + 1) as i64) as u64]
        }"#,
        "n = '-2'\nreturn = [1, '18446744073709551615']",
    );
}

#[test]
fn wide_arithmetic_still_rejects_overflow_and_zero_divisors() {
    for (body, valid, invalid) in [
        (
            "let max: u256 = !0; let sum = max + (x as u256); assert(sum >= max);",
            "x = 0",
            "x = 1",
        ),
        (
            "let high: u256 = 1 << 200; let q = high / (x as u256); assert(q > 0);",
            "x = 1",
            "x = 0",
        ),
    ] {
        let dir = common::noir_project(&format!("fn main(x: u64) {{ {body} }}"));
        let (mut driver, r1cs) = api::compile_to_r1cs(dir.path().to_path_buf(), false).unwrap();
        let binary = api::compile_bytecode(
            &mut driver,
            CodeGenOptions {
                check_constraints: true,
                ..Default::default()
            },
        )
        .unwrap();
        for (input, accepted) in [(valid, true), (invalid, false)] {
            std::fs::write(dir.path().join("Prover.toml"), input).unwrap();
            let params = api::read_prover_inputs(dir.path(), driver.abi()).unwrap();
            let result = api::run_witgen_from_binary(&mut binary.clone(), &r1cs, &params, None);
            if accepted {
                assert!(api::check_witgen(&r1cs, &result.unwrap()));
            } else {
                assert!(result.is_err(), "{body}: invalid operands were accepted");
            }
        }
    }
}

#[test]
fn wide_abi_checks_input_ranges_and_declared_returns() {
    let dir = common::noir_project(include_str!("../../noir_tests/bigint_abi/src/main.nr"));
    let (mut driver, r1cs) = api::compile_to_r1cs(dir.path().to_path_buf(), false).unwrap();
    let mut binary = api::compile_bytecode(
        &mut driver,
        CodeGenOptions {
            check_constraints: true,
            ..Default::default()
        },
    )
    .unwrap();
    // A wrong return differing only above bit 128 must fail the return constraint.
    let correct_high = 1u128 << 52;
    let wrong_high = 2u128 << 52;
    std::fs::write(
        dir.path().join("Prover.toml"),
        format!("a = [16, '{correct_high}']\nb = [0, 0]\nreturn = [16, '{wrong_high}']"),
    )
    .unwrap();
    let params = api::read_prover_inputs(dir.path(), driver.abi()).unwrap();
    assert!(api::run_witgen_from_binary(&mut binary, &r1cs, &params, None).is_err());

    let out_of_range = num_bigint::BigUint::from(1u8) << 128;
    std::fs::write(
        dir.path().join("Prover.toml"),
        format!("a = ['{out_of_range}', 0]\nb = [0, 0]\nreturn = [0, 0]"),
    )
    .unwrap();
    assert!(api::read_prover_inputs(dir.path(), driver.abi()).is_err());
}

/// Each refusal is matched by the diagnostic the rule under test emits, so that a program that
/// fails for some other reason, a typo say, does not pass for the rule.
#[test]
fn frontend_rejects_widths_and_entry_points_outside_the_supported_domain() {
    const NOT_LOWERABLE: &str =
        "Integers the circuit backend does not lower are not valid entry point types. Found: ";
    for (source, expected) in [
        ("fn main(x: u24) {}", format!("{NOT_LOWERABLE}u24")),
        ("fn main(x: u200) {}", format!("{NOT_LOWERABLE}u200")),
        ("fn main(x: u253) {}", format!("{NOT_LOWERABLE}u253")),
        ("fn main(x: i33) {}", format!("{NOT_LOWERABLE}i33")),
        ("fn main(x: i65) {}", format!("{NOT_LOWERABLE}i65")),
        ("fn main(x: [u200; 2]) {}", format!("{NOT_LOWERABLE}u200")),
        (
            "fn main() -> pub u200 { 0 }",
            format!("{NOT_LOWERABLE}u200"),
        ),
        ("fn main(x: u256) {}", format!("{NOT_LOWERABLE}u256")),
        (
            "fn main() -> pub u256 { 0 }",
            format!("{NOT_LOWERABLE}u256"),
        ),
        (
            "fn main(x: u64) { let _: u16385 = x as u16385; }",
            "`u16385` is not a supported integer type".to_string(),
        ),
        (
            "fn main(x: u64) { let _: u1 = x as u1; }",
            "`u1` is not a supported integer type".to_string(),
        ),
        // These errors arise only after the generic width is bound in monomorphization.
        (
            "fn bad<let N: u32>(x: u64) { let _: u<N> = x as u<N>; } fn main(x: u64) { bad::<16385>(x); }",
            "`u16385` is not a supported integer type".to_string(),
        ),
        (
            "fn bad<let N: u32>() -> u<N> { 256 } fn main() { assert(bad::<8>() == 0); }",
            "Integer literal does not fit its type".to_string(),
        ),
    ] {
        common::assert_frontend_refused(source, &expected);
    }
}
