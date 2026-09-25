use mavros_compiler::{
    Project,
    driver::{Driver, Error},
};

fn compile(source: &str) -> Result<(), Error> {
    let dir = tempfile::tempdir().unwrap();
    std::fs::create_dir(dir.path().join("src")).unwrap();
    std::fs::write(
        dir.path().join("Nargo.toml"),
        "[package]\nname = 'recursive_cost_analysis'\ntype = 'bin'\nauthors = []\n",
    )
    .unwrap();
    std::fs::write(dir.path().join("src/main.nr"), source).unwrap();
    let mut driver = Driver::new(Project::new(dir.path().to_path_buf()).unwrap(), false);
    driver.run_noir_compiler()?;
    driver.make_struct_access_static()?;
    driver.monomorphize()?;
    driver.spill_witness()?;
    driver.generate_r1cs()?;
    Ok(())
}

#[test]
fn witness_dependent_recursion_reports_a_source_diagnostic() {
    // The first case is Noir's execution_success/fold_fibonacci (issue #375).
    for source in [
        "fn main(x: u32) { assert(fibonacci(x) == 55); }
         #[fold]
         fn fibonacci(x: u32) -> u32 {
             if x <= 1 { x } else { fibonacci(x - 1) + fibonacci(x - 2) }
         }",
        "fn main(x: u32) { assert(even(x)); }
         #[fold]
         fn even(x: u32) -> bool { if x == 0 { true } else { odd(x - 1) } }
         #[fold]
         fn odd(x: u32) -> bool { if x == 0 { false } else { even(x - 1) } }",
    ] {
        let error = compile(source).unwrap_err();
        let Error::Refused(diagnostics) = &error else {
            panic!("expected unsupported recursion diagnostic, got {error:?}");
        };
        assert_eq!(diagnostics.len(), 1);
        assert_eq!(
            diagnostics[0].message(),
            "cannot bound constrained recursion at compile time"
        );
        assert!(diagnostics[0].location().file.ends_with("src/main.nr"));
        assert!(diagnostics[0].location().start.line > 1);
        // The source is retained even after the temporary project is removed.
        assert!(
            error
                .to_string()
                .contains("#[fold] circuits are not supported")
        );
        assert!(error.to_string().contains("else"));
    }
}

#[test]
fn bounded_and_unconstrained_recursion_still_compile() {
    for source in [
        "fn main(x: Field) { assert(sum(5, x) == 5 * x); }
         fn sum(n: u32, x: Field) -> Field {
             if n == 0 { 0 } else { x + sum(n - 1, x) }
         }",
        "fn main(x: Field) { assert(ping(5, x) == 5 * x); }
         fn ping(n: u32, x: Field) -> Field {
             if n == 0 { 0 } else { x + pong(n - 1, x) }
         }
         fn pong(n: u32, x: Field) -> Field {
             if n == 0 { 0 } else { x + ping(n - 1, x) }
         }",
        "fn main(x: u32) { assert(unsafe { down(x) } == 7); }
         unconstrained fn down(n: u32) -> u32 {
             if n == 0 { 7 } else { down(n - 1) }
         }",
        "fn main() { assert(fibonacci(10) == 55); }
         #[fold]
         fn fibonacci(x: u32) -> u32 {
             if x <= 1 { x } else { fibonacci(x - 1) + fibonacci(x - 2) }
         }",
    ] {
        compile(source).unwrap_or_else(|error| panic!("{source}: {error}"));
    }
}
