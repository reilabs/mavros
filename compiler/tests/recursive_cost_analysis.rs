use mavros_compiler::{Project, driver::Driver};

fn with_source(source: &str, check: impl FnOnce(&mut Driver)) {
    let dir = tempfile::tempdir().unwrap();
    std::fs::create_dir(dir.path().join("src")).unwrap();
    std::fs::write(
        dir.path().join("Nargo.toml"),
        "[package]\nname = 'recursive_cost_analysis'\ntype = 'bin'\nauthors = []\n",
    )
    .unwrap();
    std::fs::write(dir.path().join("src/main.nr"), source).unwrap();
    let mut driver = Driver::new(Project::new(dir.path().to_path_buf()).unwrap(), false);
    driver.run_noir_compiler().unwrap();
    driver.make_struct_access_static().unwrap();
    driver.monomorphize().unwrap();
    check(&mut driver);
}

#[test]
fn recursive_fibonacci_completes_cost_analysis_and_lookup_spilling() {
    // Upstream execution_success/fold_fibonacci (#375). This regression covers the optimizer
    // stage; runtime-dependent recursion in R1CS generation is a separate limitation.
    with_source(
        "fn main(x: u32) { assert(fibonacci(x) == 55); }
         #[fold]
         fn fibonacci(x: u32) -> u32 {
             if x <= 1 { x } else { fibonacci(x - 1) + fibonacci(x - 2) }
         }",
        |driver| driver.spill_witness().unwrap(),
    );
}

#[test]
fn bounded_and_unconstrained_recursion_still_generate_constraints() {
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
    ] {
        with_source(source, |driver| {
            driver.spill_witness().unwrap();
            driver.generate_r1cs().unwrap();
        });
    }
}
