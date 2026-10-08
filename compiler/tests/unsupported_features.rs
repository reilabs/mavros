mod common;

use mavros_compiler::{Project, driver::Driver};

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

/// Mavros lowering still reports these errors through `ice_usr!`.
fn assert_lowering_refused(source: &str, message: &str) {
    use std::panic::{AssertUnwindSafe, catch_unwind};

    let dir = common::noir_project(source);
    let mut driver = Driver::new(Project::new(dir.path().to_path_buf()).unwrap(), false);
    let panic = catch_unwind(AssertUnwindSafe(|| driver.run_noir_compiler()))
        .expect_err("expected a lowering diagnostic");
    let rendered = panic
        .downcast_ref::<String>()
        .map(String::as_str)
        .or_else(|| panic.downcast_ref::<&str>().copied())
        .expect("a rendered user diagnostic");
    assert!(
        rendered.contains("Unhandled error from user input:") && rendered.contains(message),
        "missing {message:?}: {rendered}"
    );
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
            message,
        );
    }
}

/// The widest `N` each takes compiles, and its witness satisfies the constraints.
#[test]
fn spread_widths_up_to_their_containers_run() {
    use mavros_compiler::{api, compiler::codegen::CodeGenOptions};

    let dir = common::noir_project(
        r#"use std::mavros::{spread, unspread};
fn main(x: u32) {
    let s = spread::<32>(x);
    assert_eq(s & 0xAAAAAAAAAAAAAAAA, 0);
    let (odd, even) = unspread::<32>(s);
    assert_eq(odd, 0);
    assert_eq(even, x);
}
"#,
    );
    let (mut driver, r1cs) = api::compile_to_r1cs(dir.path().to_path_buf(), false).unwrap();
    let mut binary = api::compile_bytecode(
        &mut driver,
        CodeGenOptions {
            check_constraints: true,
            ..Default::default()
        },
    )
    .unwrap();
    for x in [0u32, 1, 0x8000_0000, 0x1234_5678, u32::MAX] {
        std::fs::write(dir.path().join("Prover.toml"), format!("x = {x}")).unwrap();
        let params = api::read_prover_inputs(dir.path(), driver.abi()).unwrap();
        let result = api::run_witgen_from_binary(&mut binary, &r1cs, &params, None).unwrap();
        assert!(api::check_witgen(&r1cs, &result), "x = {x}");
    }
}

/// A `Field` can be read back as an integer no wider than one element carries injectively. The
/// fork allows any width, but the wide integer pass reads a non-integer source back one limb per
/// limb and an element is one limb, so a wider target is refused at the cast rather than reached
/// as an ICE in the pass.
#[test]
fn a_field_cast_to_a_wide_integer_is_refused_at_the_cast() {
    for (target, body) in [
        ("u256", "let y = x as u256; [y as u128, (y >> 128) as u128]"),
        ("u254", "let y = x as u254; [y as u128, (y >> 128) as u128]"),
        ("i512", "let y = x as i512; [y as u128, (y >> 128) as u128]"),
    ] {
        assert_lowering_refused(
            &format!("fn main(x: Field) -> pub [u128; 2] {{ {body} }}"),
            &format!(
                "casting a `Field` to `{target}` is not supported: a `Field` can be cast to an \
                 integer of at most 253 bits on this field"
            ),
        );
    }
}
