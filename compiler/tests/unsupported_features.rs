mod common;

use mavros_compiler::{Project, driver::Driver};
use std::panic::{AssertUnwindSafe, catch_unwind};

/// These refusals deliberately use ice_usr! until lowering propagates diagnostics as errors.
fn assert_refused(source: &str, message: &str, source_checks: &[&str]) {
    let dir = common::noir_project(source);
    let mut driver = Driver::new(Project::new(dir.path().to_path_buf()).unwrap(), false);
    let panic = catch_unwind(AssertUnwindSafe(|| driver.run_noir_compiler()))
        .expect_err("unsupported features must be refused during lowering");
    let rendered = panic
        .downcast_ref::<String>()
        .map(String::as_str)
        .or_else(|| panic.downcast_ref::<&str>().copied())
        .expect("a rendered user diagnostic");
    for expected in ["Unhandled error from user input:", message]
        .into_iter()
        .chain(source_checks.iter().copied())
    {
        assert!(
            rendered.contains(expected),
            "missing {expected:?}: {rendered}"
        );
    }
}

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
        "#[fold] functions are not supported by Mavros",
        &["src/main.nr:", "if x <= 1", "^"],
    );
}

#[test]
fn nonrecursive_fold_is_also_refused() {
    assert_refused(
        r#"fn main(x: Field) { assert(folded(x) == x); }
#[fold]
fn folded(x: Field) -> Field { x }
"#,
        "#[fold] functions are not supported by Mavros",
        &["src/main.nr:", "fn folded(x: Field) -> Field { x }", "^"],
    );
}

#[test]
fn oracle_calls_and_function_values_are_refused() {
    for attribute in ["", "#[pure]"] {
        for body in ["oracle(x)", "let f = oracle; f(x)"] {
            // Noir synthesizes a wrapper for the function value with no source location.
            let source_checks: &[&str] = if body == "oracle(x)" {
                &["src/main.nr:", body, "^"]
            } else {
                &["<Noir generated>:1:1"]
            };
            assert_refused(
                &format!(
                    "#[oracle(barnacle)]\n{attribute}\nunconstrained fn oracle(x: Field) -> Field {{}}\n\
                     unconstrained fn main(x: Field) -> pub Field {{ {body} }}\n"
                ),
                "oracle functions are not supported by Mavros: `barnacle`",
                source_checks,
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
