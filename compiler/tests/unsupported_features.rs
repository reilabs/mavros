mod common;

use mavros_compiler::{
    Project,
    driver::{Driver, Error},
};

// The fork rejects unsupported features with source diagnostics before Mavros lowering.
use common::assert_frontend_refused as assert_refused;

#[test]
fn folded_recursion_is_refused_before_cost_analysis() {
    assert_refused(
        r#"fn main(x: u32) {
    assert(fibonacci(x) == 55);
}

#[fold]
fn fibonacci(x: u32) -> u32 {
    if x <= 1 { x } else { fibonacci(x - 1) + fibonacci(x - 2) }
}
"#,
        "#[fold] attribute on function fibonacci is not supported",
    );
}

#[test]
fn nonrecursive_fold_is_also_refused() {
    assert_refused(
        r#"fn main(x: Field) { assert(folded(x) == x); }
#[fold]
fn folded(x: Field) -> Field { x }
"#,
        "#[fold] attribute on function folded is not supported",
    );
}

#[test]
fn oracle_calls_and_function_values_are_refused() {
    for attribute in ["", "#[pure]"] {
        for body in ["oracle(x)", "let f = oracle; f(x)"] {
            assert_refused(
                &format!(
                    "#[oracle(barnacle)]\n{attribute}\nunconstrained fn oracle(x: Field) -> Field {{}}\n\
                     unconstrained fn main(x: Field) -> pub Field {{ {body} }}\n"
                ),
                "Oracle `barnacle` is not supported",
            );
        }
    }
}

#[test]
fn print_oracles_remain_ignored() {
    let dir = common::noir_project("fn main(x: Field) { print(x); println(x); assert(x == 1); }");
    let mut driver = Driver::new(Project::new(dir.path().to_path_buf()).unwrap(), false);
    driver.run_noir_compiler().unwrap();
    driver.make_struct_access_static().unwrap();
}

/// Mavros lowering refuses `source` with [`Error::LoweringRefused`]
fn assert_lowering_refused(source: &str, messages: &[&str]) -> String {
    let dir = common::noir_project(source);
    let mut driver = Driver::new(Project::new(dir.path().to_path_buf()).unwrap(), false);
    let error = match driver.run_noir_compiler() {
        Err(error @ Error::LoweringRefused(_)) => error,
        Err(error) => panic!("expected a lowering refusal, got: {error}"),
        Ok(()) => panic!("expected a lowering refusal"),
    };
    let rendered = error.to_string();
    for message in messages {
        assert!(
            rendered.contains(message),
            "missing {message:?}: {rendered}"
        );
    }
    rendered
}

/// `spread::<N>` reads `N` bits of a `u32` and `unspread::<N>` takes two `N`-bit halves out of a
/// `u64`, so each takes `N` from 1 to 32 and refuses the rest with a diagnostic.
#[test]
fn spread_widths_past_their_containers_are_refused() {
    for (call, message) in [
        (
            "spread::<0>(x as u32)",
            "`spread::<0>` must read 1..=32 bits of its `u32`",
        ),
        (
            "spread::<33>(x as u32)",
            "`spread::<33>` must read 1..=32 bits of its `u32`",
        ),
        (
            "unspread::<0>(x).0",
            "`unspread::<0>` must take halves of 1..=32 bits of its `u64`",
        ),
        (
            "unspread::<33>(x).0",
            "`unspread::<33>` must take halves of 1..=32 bits of its `u64`",
        ),
    ] {
        assert_lowering_refused(
            &format!(
                "use std::mavros::{{spread, unspread}};\n\
                 fn main(x: u64) {{ let _ = {call}; }}\n"
            ),
            &[message],
        );
    }
}

/// Lowering carries on past a refusal
#[test]
fn every_refusal_in_a_program_is_reported_together() {
    let rendered = assert_lowering_refused(
        "use std::mavros::spread;\n\
         fn main(x: u32) { let _ = spread::<0>(x); let _ = spread::<33>(x); }\n",
        &[
            "`spread::<0>` must read 1..=32 bits of its `u32`",
            "`spread::<33>` must read 1..=32 bits of its `u32`",
        ],
    );
    assert_eq!(rendered.matches("error:").count(), 2, "{rendered}");
}

/// `unsafe_cast` handles scalars only
#[test]
fn unsafe_casts_to_non_scalars_are_refused() {
    let rendered = assert_lowering_refused(
        "use std::mavros::unsafe_cast;\n\
         fn main(x: Field) -> pub [Field; 2] { unsafe_cast::<[Field; 2]>(x) }\n",
        &["unsafe_cast: unsupported target type Array(2, Field)"],
    );
    assert!(rendered.contains("src/main.nr:2:"), "{rendered}");
}

/// Compiles `source`, whose `main` takes `x: u32`, and runs it on each value, answering whether its
/// witness satisfies the constraints.
fn witness_holds_for_each(source: &str, values: &[u32]) -> Vec<bool> {
    use mavros_compiler::{api, compiler::codegen::CodeGenOptions};

    let dir = common::noir_project(source);
    let (mut driver, r1cs) = api::compile_to_r1cs(dir.path().to_path_buf(), false).unwrap();
    let mut binary = api::compile_bytecode(
        &mut driver,
        CodeGenOptions {
            check_constraints: true,
            ..Default::default()
        },
    )
    .unwrap();
    values
        .iter()
        .map(|x| {
            std::fs::write(dir.path().join("Prover.toml"), format!("x = {x}")).unwrap();
            let (params, _) = api::read_prover_inputs(dir.path(), driver.abi()).unwrap();
            api::run_witgen_from_binary(&mut binary, &r1cs, &params, None)
                .is_ok_and(|result| api::check_witgen(&r1cs, &result))
        })
        .collect()
}

/// The widest `N` each takes compiles, and its witness satisfies the constraints.
#[test]
fn spread_widths_up_to_their_containers_run() {
    let values = [0u32, 1, 0x8000_0000, 0x1234_5678, u32::MAX];
    let holds = witness_holds_for_each(
        r#"use std::mavros::{spread, unspread};
fn main(x: u32) {
    let s = spread::<32>(x);
    assert_eq(s & 0xAAAAAAAAAAAAAAAA, 0);
    let (odd, even) = unspread::<32>(s);
    assert_eq(odd, 0);
    assert_eq(even, x);
}
"#,
        &values,
    );
    for (x, holds) in values.iter().zip(holds) {
        assert!(holds, "x = {x}");
    }
}

/// A witnessed `spread::<N>` of a `u32` reads `N` bits and has no witness for a value with any bit
/// above them.
#[test]
fn a_witnessed_spread_has_no_witness_past_its_read() {
    let values = [0u32, 0x5A, 0xFF, 0x100, 0x1_0000, u32::MAX];
    let holds = witness_holds_for_each(
        r#"use std::mavros::spread;
fn main(x: u32) {
    let s = spread::<8>(x);
    assert_eq(s & 0xAAAAAAAAAAAAAAAA, 0);
}
"#,
        &values,
    );
    for (x, holds) in values.iter().zip(holds) {
        assert_eq!(holds, *x < 0x100, "x = {x:#x}");
    }
}
