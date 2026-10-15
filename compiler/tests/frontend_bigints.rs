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
fn arithmetic_and_casts_at_multiple_limb_widths() {
    for (source, inputs) in [
        (
            include_str!("../../noir_tests/bigint_arithmetic_u64/src/main.nr"),
            include_str!("../../noir_tests/bigint_arithmetic_u64/Prover.toml"),
        ),
        (
            include_str!("../../noir_tests/bigint_arithmetic_u256/src/main.nr"),
            include_str!("../../noir_tests/bigint_arithmetic_u256/Prover.toml"),
        ),
        (
            include_str!("../../noir_tests/bigint_arithmetic_u384/src/main.nr"),
            include_str!("../../noir_tests/bigint_arithmetic_u384/Prover.toml"),
        ),
    ] {
        check_both_modes(source, inputs);
    }
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
    // Keep the original no-carry case alongside the fixture's cross-limb carry.
    check_both_modes(
        include_str!("../../noir_tests/bigint_abi/src/main.nr"),
        r#"a = ["7", "4503599627370496"]
b = ["9", "4294967296"]
return = ["16", "4503603922337792"]"#,
    );
}

#[test]
fn field_identity_casts_preserve_the_top_field_bit() {
    check_both_modes(
        "fn main(x: Field) -> pub Field { x as Field }",
        "x = '21888242871839275222246405745257275088548364400416034343698204186575808495616'\nreturn = '21888242871839275222246405745257275088548364400416034343698204186575808495616'",
    );
}

#[test]
fn constant_field_expressions_can_be_cast_to_wide_integers() {
    check_both_modes(
        r#"fn main(x: u64) -> pub [u64; 3] {
            let constant: Field = 1 + 2;
            let literal = (1 as u512) << 300;
            let computed = (constant as u512) << 300;
            let negative: Field = -1;
            let top = (negative as u512) >> 253;
            [((literal >> 300) as u64) + x,
             ((computed >> 300) as u64) + x, top as u64]
        }"#,
        "x = 4\nreturn = [5, 7, 1]",
    );
}

#[test]
fn field_casts_to_wide_integers_preserve_the_canonical_value() {
    for target in ["u254", "u256", "i512"] {
        for decimal in [
            "0",
            "14474011154664524427946373126085988481658748083205070504932198000989141204993",
            "21888242871839275222246405745257275088548364400416034343698204186575808495616",
        ] {
            let value: num_bigint::BigUint = decimal.parse().unwrap();
            let mask = (num_bigint::BigUint::from(1u8) << 128) - 1u8;
            let low = &value & &mask;
            let high = &value >> 128;
            check_both_modes(
                &format!(
                    "fn main(x: Field) -> pub [u128; 2] {{
                        let wide = x as {target};
                        [wide as u128, (wide >> 128) as u128]
                    }}"
                ),
                &format!("x = '{decimal}'\nreturn = ['{low}', '{high}']"),
            );
        }
    }
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
    // A one-bit top limb is checked by a Boolean constraint. Wider top limbs can stop in
    // the VM's lookup machinery before we can independently check the invalid witness.
    // Invalid inputs omit the declared return so rejection cannot come from a return mismatch.
    for (body, valid, invalid) in [
        (
            "let max: u257 = !0; let result = max + (x as u257);",
            "x = 0\nreturn = ['340282366920938463463374607431768211455', '340282366920938463463374607431768211455', 1]",
            "x = 1",
        ),
        (
            "let high: u257 = 1 << 256; let divisor = (x as u257) << 256; let result = high / divisor;",
            "x = 1\nreturn = [1, 0, 0]",
            "x = 0",
        ),
    ] {
        let dir = common::noir_project(&format!(
            "fn main(x: u64) -> pub [u128; 3] {{ {body}
                [result as u128, (result >> 128) as u128, (result >> 256) as u128]
            }}"
        ));
        let (mut driver, r1cs) = api::compile_to_r1cs(dir.path().to_path_buf(), false).unwrap();
        let binary = api::compile_bytecode(
            &mut driver,
            CodeGenOptions {
                check_constraints: false,
                ..Default::default()
            },
        )
        .unwrap();
        for (input, accepted) in [(valid, true), (invalid, false)] {
            std::fs::write(dir.path().join("Prover.toml"), input).unwrap();
            let params = api::read_prover_inputs(dir.path(), driver.abi()).unwrap();
            // Require witness generation to finish with honest-prover checks disabled:
            // a VM trap would not demonstrate that the constraints reject the witness.
            let result = api::run_witgen_from_binary(&mut binary.clone(), &r1cs, &params, None)
                .unwrap_or_else(|error| panic!("{body}: {input}: {error:?}"));
            let witness = [
                result.out_wit_pre_comm.as_slice(),
                result.out_wit_post_comm.as_slice(),
            ]
            .concat();
            assert_eq!(
                r1cs.unsatisfied_constraint(&witness).is_none(),
                accepted,
                "{body}: {input}"
            );
            if accepted {
                assert!(api::check_witgen(&r1cs, &result));
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
            check_constraints: false,
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
    let result = api::run_witgen_from_binary(&mut binary, &r1cs, &params, None)
        .expect("unchecked witness generation must complete");
    let witness = [
        result.out_wit_pre_comm.as_slice(),
        result.out_wit_post_comm.as_slice(),
    ]
    .concat();
    assert!(
        r1cs.unsatisfied_constraint(&witness).is_some(),
        "wrong high return limb must violate R1CS"
    );

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
            "`u1` has been removed, use `bool` instead".to_string(),
        ),
        // These errors arise only after the generic width is bound in monomorphization.
        (
            "fn bad<let N: u32>() -> u<N> { 0 } fn main() { assert(bad::<16385>() == 0); }",
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

/// Returned errors keep all source diagnostics even after the project and its files are gone.
#[test]
fn frontend_errors_render_all_source_diagnostics() {
    use mavros_compiler::{
        Project,
        driver::{Driver, Error},
    };

    let dir = common::noir_project("fn main() {\n    let a: bool = 1;\n    let b: bool = 2;\n}");
    let mut driver = Driver::new(Project::new(dir.path().to_path_buf()).unwrap(), false);
    let error = driver.run_noir_compiler().unwrap_err();
    let Error::NoirCompilerError { diagnostics, .. } = &error else {
        panic!("expected Noir diagnostics: {error}");
    };
    assert!(diagnostics.len() >= 2, "{diagnostics:?}");
    drop(driver);
    drop(dir);
    let rendered = error.to_string();
    for diagnostic in diagnostics {
        assert!(rendered.contains(&diagnostic.message), "{rendered}");
    }
    for snippet in [
        "let a: bool = 1;",
        "let b: bool = 2;",
        "src/main.nr:2:",
        "src/main.nr:3:",
    ] {
        assert!(rendered.contains(snippet), "missing {snippet}: {rendered}");
    }
}
