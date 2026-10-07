mod common;

use mavros_compiler::{
    Project,
    driver::{Driver, Error},
};

/// The fork rejects unsupported features with source diagnostics before Mavros lowering.
fn assert_refused(source: &str, message: &str) {
    let dir = common::noir_project(source);
    let mut driver = Driver::new(Project::new(dir.path().to_path_buf()).unwrap(), false);
    let Err(Error::NoirCompilerError(diagnostics)) = driver.run_noir_compiler() else {
        panic!("expected a frontend diagnostic for {source}");
    };
    assert!(
        diagnostics.iter().any(|diagnostic| {
            diagnostic.message.contains(message) && !diagnostic.secondaries.is_empty()
        }),
        "missing located diagnostic {message:?}: {diagnostics:?}"
    );
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
