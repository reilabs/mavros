use mavros_compiler::{abi_helpers, api, compiler::codegen::CodeGenOptions};

/// Check actual execution and the declared return, not just successful SSA construction.
fn check(source: &str, inputs: &[&str]) {
    let dir = tempfile::tempdir().unwrap();
    std::fs::create_dir(dir.path().join("src")).unwrap();
    std::fs::write(
        dir.path().join("Nargo.toml"),
        "[package]\nname = 'while_conditions'\ntype = 'bin'\nauthors = []\n",
    )
    .unwrap();
    std::fs::write(dir.path().join("src/main.nr"), source).unwrap();
    let (mut driver, r1cs) = api::compile_to_r1cs(dir.path().to_path_buf(), false).unwrap();
    let mut binary = api::compile_bytecode(
        &mut driver,
        CodeGenOptions {
            check_constraints: true,
            ..Default::default()
        },
    )
    .unwrap();
    for input in inputs {
        std::fs::write(dir.path().join("Prover.toml"), input).unwrap();
        let params = api::read_prover_inputs(dir.path(), driver.abi()).unwrap();
        let result = api::run_witgen_from_binary(&mut binary, &r1cs, &params, None).unwrap();
        assert!(api::check_witgen(&r1cs, &result), "{input}");
        abi_helpers::check_return_guard(
            driver.abi(),
            &r1cs.witness_layout,
            &params,
            &result.out_wit_pre_comm,
        )
        .unwrap();
    }
}

#[test]
fn break_in_while_condition_preserves_the_enclosing_loops_live_values() {
    // Issue #376: breaking before the assignment commits must preserve x, even after
    // mutating a copy of it. The break belongs to the outer while.
    check(
        r#"
unconstrained fn main(cond: bool) -> pub [Field; 2] {
    let mut x = [1, 2];
    let mut z = [0, 0];
    let mut i = 0;
    while i <= 3 {
        x = if cond {
            let mut y = x;
            y[0] = 10;
            z = y;
            let mut j = 0;
            while { break; j < 3 } {
                j = j + 1;
            }
            [4, 5]
        } else {
            x
        };
        i = i + 1;
    }
    [x[0], z[0]]
}
"#,
        &[
            "cond = true\nreturn = [1, 10]",
            "cond = false\nreturn = [1, 0]",
        ],
    );
}

#[test]
fn while_condition_branches_are_evaluated_on_every_iteration() {
    check(
        r#"
unconstrained fn main(limit: u32) -> pub u32 {
    let mut i = 0;
    while if i < limit { true } else { false } {
        i += 1;
    }
    i
}
"#,
        &["limit = 0\nreturn = 0", "limit = 3\nreturn = 3"],
    );
}

#[test]
fn conditional_break_in_while_condition_exits_the_enclosing_loop() {
    check(
        r#"
unconstrained fn main(stop: u32) -> pub u32 {
    let mut i = 0;
    let mut count = 0;
    while i < 3 {
        let mut j = 0;
        while { if i == stop { break; } j < 2 } {
            count += 1;
            j += 1;
        }
        i += 1;
    }
    count
}
"#,
        &[
            "stop = 0\nreturn = 0",
            "stop = 2\nreturn = 4",
            "stop = 3\nreturn = 6",
        ],
    );
}

#[test]
fn continue_in_while_condition_advances_the_enclosing_for_loop() {
    check(
        r#"
unconstrained fn main(skip: u32) -> pub u32 {
    let mut count = 0;
    for i in 0..3 {
        let mut j = 0;
        while { if i == skip { continue; } j < 2 } {
            count += 1;
            j += 1;
        }
        count += 10;
    }
    count
}
"#,
        &[
            "skip = 0\nreturn = 24",
            "skip = 2\nreturn = 24",
            "skip = 3\nreturn = 36",
        ],
    );
}
