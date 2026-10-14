mod common;

use ark_ff::{AdditiveGroup, UniformRand};
use mavros_artifacts::{Field, R1CS};
use mavros_compiler::{api, compiler::codegen::CodeGenOptions};
use rand::{SeedableRng, rngs::StdRng};

/// Independent oracle: multiply each materialized matrix row by w, then by r.
fn reference(r1cs: &R1CS, r: &[Field], w: &[Field]) -> [Field; 3] {
    let mut result = [Field::ZERO; 3];
    for (row, weight) in r1cs.constraints.iter().zip(r) {
        for (sum, terms) in result.iter_mut().zip([&row.a, &row.b, &row.c]) {
            let row_value: Field = terms
                .iter()
                .map(|(column, coeff)| *coeff * w[*column])
                .sum();
            *sum += *weight * row_value;
        }
    }
    result
}

fn check(source: &str) {
    let project = common::noir_project(source);
    let (mut driver, r1cs) = api::compile_to_r1cs(project.path().to_path_buf(), false).unwrap();
    let mut binary = api::compile_bytecode(&mut driver, CodeGenOptions::default()).unwrap();
    let original = binary.clone();
    let mut rng = StdRng::seed_from_u64(411);
    for case in 0..6 {
        let mut r: Vec<Field> = (0..r1cs.constraints.len())
            .map(|_| Field::rand(&mut rng))
            .collect();
        let mut w: Vec<Field> = (0..r1cs.witness_layout.size())
            .map(|_| Field::rand(&mut rng))
            .collect();
        match case {
            0 => r.fill(Field::ZERO),
            1 => w.fill(Field::ZERO),
            2 => {
                // Constants must multiply w[0], even when it is not one.
                w.fill(Field::ZERO);
                w[0] = Field::from(7u64);
            }
            _ => {}
        }
        let result = api::run_verifier_from_binary(&binary, &r1cs, &r, &w, None).unwrap();
        assert_eq!(
            [result.r_aw, result.r_bw, result.r_cw],
            reference(&r1cs, &r, &w),
            "case {case}"
        );
        assert_eq!(result.instrumenter.final_memory_usage(), 0);

        // Both execution modes can still use the same compiled artifact repeatedly.
        let (a, b, c, memory) = api::run_ad_from_binary(&mut binary, &r1cs, &r, None).unwrap();
        assert!(api::check_ad(&r1cs, &r, &a, &b, &c));
        assert_eq!(memory.final_memory_usage(), 0);
    }
    assert_eq!(binary, original);
    assert!(!project.path().join("Prover.toml").exists());
}

#[test]
fn algebraic_constraints_and_shared_linear_combinations() {
    check(
        r#"
        fn main(x: Field, y: pub Field) -> pub Field {
            let z = 3 * x - y + 7;
            let mut result = z * z;
            for i in 0..4 { result = result * z + i as Field; }
            assert(result != 0);
            result
        }
    "#,
    );
}

#[test]
fn range_checks_and_guarded_constraints() {
    check(include_str!("../../noir_tests/cond_rangecheck/src/main.nr"));
}

#[test]
fn constant_and_witness_array_tables() {
    check(include_str!("../../noir_tests/array_lookup/src/main.nr"));
}

#[test]
fn spread_tables() {
    check(include_str!("../../noir_tests/spread_and_cast/src/main.nr"));
}

#[test]
fn powers_of_two_tables() {
    check("fn main(x: u8, amount: u8) -> pub u8 { x << amount }");
}

#[test]
fn empty_circuit() {
    check("fn main() {}");
}

#[test]
fn vector_lengths_are_checked_before_execution() {
    let project = common::noir_project("fn main(x: Field) { assert(x * x == 9); }");
    let (mut driver, r1cs) = api::compile_to_r1cs(project.path().to_path_buf(), false).unwrap();
    let binary = api::compile_bytecode(&mut driver, CodeGenOptions::default()).unwrap();
    let r = vec![Field::from(1u64); r1cs.constraints.len()];
    let w = vec![Field::from(2u64); r1cs.witness_layout.size()];
    for len in [0, r.len() - 1, r.len() + 1] {
        let invalid_r = vec![Field::ZERO; len];
        let err = api::run_verifier_from_binary(&binary, &r1cs, &invalid_r, &w, None)
            .err()
            .expect("incorrect r length must fail");
        assert!(err.message().unwrap().contains("coefficient length"));
    }
    for len in [0, w.len() - 1, w.len() + 1] {
        let invalid_w = vec![Field::ZERO; len];
        let err = api::run_verifier_from_binary(&binary, &r1cs, &r, &invalid_w, None)
            .err()
            .expect("incorrect w length must fail");
        assert!(err.message().unwrap().contains("w length"));
    }
}

#[test]
fn cli_verifies_without_prover_inputs_and_prints_json() {
    let project = common::noir_project("fn main(x: Field, y: pub Field) { assert(x * x == y); }");
    let (_, r1cs) = api::compile_to_r1cs(project.path().to_path_buf(), false).unwrap();
    let r = vec![Field::from(3u64); r1cs.constraints.len()];
    let w = vec![Field::from(7u64); r1cs.witness_layout.size()];
    for (name, vector) in [("r.json", &r), ("w.json", &w)] {
        std::fs::write(
            project.path().join(name),
            serde_json::to_vec(&vector.iter().map(ToString::to_string).collect::<Vec<_>>())
                .unwrap(),
        )
        .unwrap();
    }
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_mavros"))
        .current_dir(project.path())
        .args(["--verify", "--r", "r.json", "--w", "w.json"])
        .env("RUST_LOG", "info")
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let result: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    let expected = reference(&r1cs, &r, &w);
    assert_eq!(
        result,
        serde_json::json!({
            "r_aw": expected[0].to_string(),
            "r_bw": expected[1].to_string(),
            "r_cw": expected[2].to_string(),
        })
    );
    assert!(!project.path().join("Prover.toml").exists());
}

#[test]
fn cli_requires_one_workspace_binary_and_honors_member_selection() {
    let workspace = tempfile::tempdir().unwrap();
    let manifest = workspace.path().join("Nargo.toml");
    let members = "[workspace]\nmembers = ['a', 'b']\n";
    std::fs::write(&manifest, members).unwrap();
    for member in ["a", "b"] {
        let root = workspace.path().join(member);
        std::fs::create_dir_all(root.join("src")).unwrap();
        std::fs::write(
            root.join("Nargo.toml"),
            format!("[package]\nname = '{member}'\ntype = 'bin'\nauthors = []\n"),
        )
        .unwrap();
        std::fs::write(
            root.join("src/main.nr"),
            "fn main(x: Field) { assert(x * x == 9); }",
        )
        .unwrap();
    }
    let invoke = |root: &str| {
        std::process::Command::new(env!("CARGO_BIN_EXE_mavros"))
            .current_dir(workspace.path())
            .args(["--root", root, "--verify", "--r", "r.json", "--w", "w.json"])
            .env("RUST_LOG", "info")
            .output()
            .unwrap()
    };

    // Rejection happens even before reading inputs or compiling either member.
    let rejected = invoke(".");
    assert!(!rejected.status.success());
    assert!(rejected.stdout.is_empty());
    assert!(String::from_utf8_lossy(&rejected.stderr).contains("requires a single binary package"));
    for member in ["a", "b"] {
        assert!(!workspace.path().join(member).join("mavros_debug").exists());
    }

    let (_, r1cs) = api::compile_to_r1cs(workspace.path().join("a"), false).unwrap();
    for (file, len) in [
        ("r.json", r1cs.constraints.len()),
        ("w.json", r1cs.witness_layout.size()),
    ] {
        std::fs::write(
            workspace.path().join(file),
            serde_json::to_vec(&vec!["0"; len]).unwrap(),
        )
        .unwrap();
    }
    let expected = serde_json::json!({"r_aw": "0", "r_bw": "0", "r_cw": "0"});
    let explicit = invoke("a");
    assert!(
        explicit.status.success(),
        "{}",
        String::from_utf8_lossy(&explicit.stderr)
    );
    assert_eq!(
        serde_json::from_slice::<serde_json::Value>(&explicit.stdout).unwrap(),
        expected
    );

    std::fs::write(&manifest, format!("{members}default-member = 'a'\n")).unwrap();
    let default = invoke(".");
    assert!(
        default.status.success(),
        "{}",
        String::from_utf8_lossy(&default.stderr)
    );
    assert_eq!(
        serde_json::from_slice::<serde_json::Value>(&default.stdout).unwrap(),
        expected
    );
    assert!(!workspace.path().join("b/mavros_debug").exists());
}
