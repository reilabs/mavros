mod common;

use ark_ff::{AdditiveGroup, UniformRand};
use mavros_artifacts::{Field, R1CS};
use mavros_compiler::{api, compiler::codegen::CodeGenOptions};
use rand::{SeedableRng, rngs::StdRng};

/// Independent oracle: multiply each materialized matrix row by w, then by r.
fn reference<F: mavros_artifacts::MavrosField>(r1cs: &R1CS<F>, r: &[F], w: &[F]) -> [F; 3] {
    let mut result = [F::ZERO; 3];
    for (row, weight) in r1cs.constraints.iter().zip(r) {
        for (sum, terms) in result.iter_mut().zip([&row.a, &row.b, &row.c]) {
            let row_value = terms
                .iter()
                .fold(F::ZERO, |sum, (column, coeff)| sum + *coeff * w[*column]);
            *sum += *weight * row_value;
        }
    }
    result
}

/// The verifier artifact must never execute witgen-only or AD operations.
fn assert_forward_bytecode(binary: &[u64]) {
    use mavros_vm::bytecode::{FieldOpCode as Op, FramePosition, OpCode, parse_program_header};
    let header = parse_program_header(binary);
    let p = FramePosition(0);
    let forbidden: Vec<u64> = [
        Op::PureToWitnessRef { res: p, v: p },
        Op::MulConst {
            res: p,
            coeff: p,
            v: p,
        },
        Op::AddBoxed { res: p, a: p, b: p },
        Op::BumpDa { v: p, coeff: p },
        Op::BumpDb { v: p, coeff: p },
        Op::BumpDc { v: p, coeff: p },
        Op::NextDCoeff { v: p },
        Op::WriteWitness { val: p },
        Op::R1C { a: p, b: p, c: p },
        Op::Rangecheck {
            val: p,
            max_bits: 8,
        },
        Op::RngchkField {
            val: p,
            flag: p,
            bits: 8,
        },
        Op::DrngchkField {
            val: p,
            flag: p,
            bits: 8,
        },
        Op::SpreadLookupField {
            val: p,
            result: p,
            flag: p,
            bits: 8,
        },
        Op::DspreadLookupField {
            val: p,
            result: p,
            flag: p,
            bits: 8,
        },
        Op::Pow2LookupField {
            amount: p,
            factor: p,
            flag: p,
            size: 8,
        },
        Op::Dpow2LookupField {
            amount: p,
            factor: p,
            flag: p,
            size: 8,
        },
        Op::ArrayLookupField {
            array: p,
            index: p,
            result: p,
            flag: p,
            stride: 1,
            elem_kind: 0,
        },
        Op::DarrayLookupField {
            array: p,
            index: p,
            result: p,
            flag: p,
            stride: 1,
            elem_kind: 0,
        },
    ]
    .into_iter()
    .map(|op| op.for_field(header.field))
    .chain([OpCode::FreshWitness { res: p }])
    .map(|op| {
        let mut words = vec![];
        op.to_binary(&mut words, &mut vec![]);
        words[0]
    })
    .collect();
    let mut cursor = header.code_start;
    while cursor < binary.len() {
        if binary[cursor] == u64::MAX {
            cursor += 2;
            continue;
        }
        assert!(
            !forbidden.contains(&binary[cursor]),
            "witgen-only or AD opcode in verifier at {cursor}"
        );
        cursor = OpCode::next_opcode(binary, cursor);
    }
}

#[test]
fn goldilocks_verifier_matches_materialized_matrices() {
    use mavros_artifacts::{FieldConfig, Goldilocks};
    use mavros_compiler::{
        compiler::ssa::{
            SourceLocation,
            hlssa::{
                BinaryArithOpKind, HLSSA, Type,
                builder::{HLEmitter as _, HLSSABuilder},
            },
        },
        driver::Driver,
    };
    use mavros_int_semantics::IntBits;

    let mut ssa = HLSSA::with_main_in(FieldConfig::goldilocks(), "main".to_string());
    let entry = ssa.get_unique_entrypoint_id();
    HLSSABuilder::new(&mut ssa).modify_function(entry, |b| {
        let entry = b.function.get_entry_id();
        let mut e = b
            .block(entry)
            .with_source_location(SourceLocation::synthetic("main"));
        let x = e.add_parameter(Type::field());
        let y = e.add_parameter(Type::int(8));
        let amount = e.add_parameter(Type::int(8));
        let square = e.umul(x, x);
        let nine = e.field_const(9u64);
        e.assert_eq(square, nine);
        let shifted = e.bin(BinaryArithOpKind::UShl, y, amount);
        let nine = e.int_const(IntBits::from_u128(8, 9));
        e.assert_eq(shifted, nine);
        e.terminate_return(vec![]);
    });
    let scratch = tempfile::tempdir().unwrap();
    let mut driver = Driver::from_ssa(ssa, scratch.path().to_path_buf(), false);
    // Arithmetic test only: Goldilocks needs a lower target for the single-challenge LogUp.
    driver.set_logup_soundness(40);
    driver.make_struct_access_static().unwrap();
    driver.monomorphize().unwrap();
    driver.spill_witness().unwrap();
    let r1cs = driver.generate_r1cs::<Goldilocks>().unwrap();
    let binary = api::compile_verifier_bytecode(&mut driver, CodeGenOptions::default()).unwrap();
    assert_forward_bytecode(&binary);
    assert!(r1cs.constraints_layout.tables_data_size > 0);
    let r: Vec<_> = (0..r1cs.constraints.len())
        .map(|i| Goldilocks::from(i as u64 + 2))
        .collect();
    let w: Vec<_> = (0..r1cs.witness_layout.size())
        .map(|i| Goldilocks::from(i as u64 + 5))
        .collect();
    let result = api::run_verifier_from_binary(
        &binary,
        &r,
        &w,
        r1cs.witness_layout,
        r1cs.constraints_layout,
        None,
    )
    .unwrap();
    assert_eq!(
        [result.r_aw, result.r_bw, result.r_cw],
        reference(&r1cs, &r, &w)
    );
    assert_eq!(result.instrumenter.final_memory_usage(), 0);
}

fn check(source: &str) -> (usize, usize) {
    use mavros_vm::bytecode::{AllocationType, AlocationEvent};
    let heap_allocations = |memory: &mavros_vm::bytecode::AllocationInstrumenter| {
        memory
            .events
            .iter()
            .filter(|event| matches!(event, AlocationEvent::Alloc(AllocationType::Heap, _)))
            .count()
    };
    let mut heaps = (0, 0);
    let project = common::noir_project(source);
    let (mut driver, r1cs) = api::compile_to_r1cs(project.path().to_path_buf(), false).unwrap();
    let mut prover = api::compile_bytecode(&mut driver, CodeGenOptions::default()).unwrap();
    let original_prover = prover.clone();
    let binary = api::compile_verifier_bytecode(&mut driver, CodeGenOptions::default()).unwrap();
    assert_eq!(
        api::compile_bytecode(&mut driver, CodeGenOptions::default()).unwrap(),
        original_prover
    );
    let original = binary.clone();
    let verifier_header = mavros_vm::bytecode::parse_program_header(&binary);
    assert_eq!(
        &verifier_header.entry_points[..2],
        &[usize::MAX, usize::MAX]
    );
    assert_eq!(
        mavros_vm::bytecode::parse_program_header(&prover)
            .entry_points
            .len(),
        2
    );
    assert_forward_bytecode(&binary);
    assert_eq!(
        mavros_vm::bytecode::parse_program_header(&binary)
            .entry_points
            .len(),
        3
    );
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
        let result = api::run_verifier_from_binary(
            &binary,
            &r,
            &w,
            r1cs.witness_layout,
            r1cs.constraints_layout,
            None,
        )
        .unwrap();
        assert_eq!(
            [result.r_aw, result.r_bw, result.r_cw],
            reference(&r1cs, &r, &w),
            "case {case}"
        );
        assert_eq!(result.instrumenter.final_memory_usage(), 0);
        heaps.0 = heaps.0.max(heap_allocations(&result.instrumenter));

        // Each mode repeatedly executes its own artifact.
        let (a, b, c, memory) = api::run_ad_from_binary(&mut prover, &r1cs, &r, None).unwrap();
        assert!(api::check_ad(&r1cs, &r, &a, &b, &c));
        assert_eq!(memory.final_memory_usage(), 0);
        heaps.1 = heaps.1.max(heap_allocations(&memory));
    }
    assert_eq!(binary, original);
    assert_eq!(prover, original_prover);
    assert!(!project.path().join("Prover.toml").exists());
    heaps
}

#[test]
fn algebraic_constraints_and_shared_linear_combinations() {
    let (verifier_heap, ad_heap) = check(
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
    assert!(
        verifier_heap < ad_heap,
        "forward evaluation removes AD allocations; ordinary program containers remain"
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
fn constraint_layouts_must_match_the_compiled_artifact() {
    let project =
        common::noir_project("fn main(x: u8, i: u32) { let a = [x, 2, 3]; assert(a[i] == x); }");
    let (mut driver, r1cs) = api::compile_to_r1cs(project.path().to_path_buf(), false).unwrap();
    let verifier = api::compile_verifier_bytecode(&mut driver, CodeGenOptions::default()).unwrap();
    let mut prover = api::compile_bytecode(&mut driver, CodeGenOptions::default()).unwrap();
    let other = common::noir_project("fn main(x: Field) { assert(x == 1); }");
    let (_, other_r1cs) = api::compile_to_r1cs(other.path().to_path_buf(), false).unwrap();
    let mut shifted_witness = r1cs.witness_layout;
    shifted_witness.algebraic_size -= 1;
    shifted_witness.multiplicities_size += 1;
    let mut shifted_constraints = r1cs.constraints_layout;
    shifted_constraints.algebraic_size -= 1;
    shifted_constraints.tables_data_size += 1;
    for (wl, cl) in [
        (other_r1cs.witness_layout, other_r1cs.constraints_layout),
        (shifted_witness, r1cs.constraints_layout),
        (r1cs.witness_layout, shifted_constraints),
    ] {
        // The vectors fit the supplied layouts, including mismatches with unchanged total lengths.
        let r = vec![Field::from(3u64); cl.size()];
        let w = vec![Field::from(5u64); wl.size()];
        let error = api::run_verifier_from_binary(&verifier, &r, &w, wl, cl, None)
            .err()
            .unwrap();
        assert_eq!(
            error.message(),
            Some("constraint layouts do not match bytecode")
        );
        let error = mavros_vm::interpreter::run_ad(&prover, &r, wl, cl, None)
            .err()
            .unwrap();
        assert_eq!(
            error.message(),
            Some("constraint layouts do not match bytecode")
        );
    }
    let r = vec![Field::from(3u64); r1cs.constraints_layout.size()];
    let w = vec![Field::from(5u64); r1cs.witness_layout.size()];
    let result = api::run_verifier_from_binary(
        &verifier,
        &r,
        &w,
        r1cs.witness_layout,
        r1cs.constraints_layout,
        None,
    )
    .unwrap();
    assert_eq!(
        [result.r_aw, result.r_bw, result.r_cw],
        reference(&r1cs, &r, &w)
    );
    api::run_ad_from_binary(&mut prover, &r1cs, &r, None).unwrap();
}

#[test]
fn vector_lengths_are_checked_before_execution() {
    let project = common::noir_project("fn main(x: Field) { assert(x * x == 9); }");
    let (mut driver, r1cs) = api::compile_to_r1cs(project.path().to_path_buf(), false).unwrap();
    let binary = api::compile_verifier_bytecode(&mut driver, CodeGenOptions::default()).unwrap();
    let r = vec![Field::from(1u64); r1cs.constraints.len()];
    let w = vec![Field::from(2u64); r1cs.witness_layout.size()];
    for len in [0, r.len() - 1, r.len() + 1] {
        let invalid_r = vec![Field::ZERO; len];
        let err = api::run_verifier_from_binary(
            &binary,
            &invalid_r,
            &w,
            r1cs.witness_layout,
            r1cs.constraints_layout,
            None,
        )
        .err()
        .expect("incorrect r length must fail");
        assert!(err.message().unwrap().contains("coefficient length"));
    }
    for len in [0, w.len() - 1, w.len() + 1] {
        let invalid_w = vec![Field::ZERO; len];
        let err = api::run_verifier_from_binary(
            &binary,
            &r,
            &invalid_w,
            r1cs.witness_layout,
            r1cs.constraints_layout,
            None,
        )
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

#[test]
fn witness_slices_across_calls_and_nested_arrays() {
    check(include_str!(
        "../../noir_tests/slice_wl_call_return/src/main.nr"
    ));
    check(include_str!(
        "../../noir_tests/slice_witness_set_nested/src/main.nr"
    ));
    check(include_str!(
        "../../noir_tests/ref_array_witness_write_one_cell/src/main.nr"
    ));
    check(include_str!(
        "../../noir_tests/argpromo_loop_accum/src/main.nr"
    ));
}

#[test]
fn prover_artifacts_are_rejected_by_the_verifier() {
    use mavros_vm::bytecode::{Function, OpCode, Program};
    let project = common::noir_project("fn main() {}");
    let (_, r1cs) = api::compile_to_r1cs(project.path().to_path_buf(), false).unwrap();
    let binary = Program {
        field: mavros_artifacts::FieldId::Bn254,
        constraint_layouts: None,
        functions: (0..2)
            .map(|i| Function {
                name: format!("entry_{i}"),
                frame_size: 2,
                code: vec![OpCode::Ret {}],
                source_locations: vec![],
            })
            .collect(),
        entry_points: vec![0, 1],
        entry_blob_field_count: 0,
        global_frame_size: 0,
        constant_pool: vec![],
    }
    .to_binary();
    let r = vec![Field::ZERO; r1cs.constraints_layout.size()];
    let w = vec![Field::ZERO; r1cs.witness_layout.size()];
    let error = api::run_verifier_from_binary(
        &binary,
        &r,
        &w,
        r1cs.witness_layout,
        r1cs.constraints_layout,
        None,
    )
    .err()
    .unwrap();
    assert!(error.to_string().contains("separate verifier artifact"));
}

#[test]
fn verifier_artifact_cannot_run_as_a_prover() {
    let project = common::noir_project("fn main(x: Field) { assert(x * x == 9); }");
    let (mut driver, r1cs) = api::compile_to_r1cs(project.path().to_path_buf(), false).unwrap();
    let mut binary =
        api::compile_verifier_bytecode(&mut driver, CodeGenOptions::default()).unwrap();
    let r = vec![Field::ZERO; r1cs.constraints_layout.size()];
    let error = api::run_ad_from_binary(&mut binary, &r1cs, &r, None)
        .err()
        .unwrap();
    assert!(error.to_string().contains("no AD entry point"));
    let error = api::run_witgen_from_binary(&mut binary, &r1cs, &[], None)
        .err()
        .unwrap();
    assert!(error.to_string().contains("no witgen entry point"));
}

#[test]
fn cli_exports_verifier_separately_without_changing_prover_binary() {
    let project = common::noir_project("fn main(x: Field) -> pub Field { x * x + 3 }");
    let invoke = |extra: &[&str]| {
        let output = std::process::Command::new(env!("CARGO_BIN_EXE_mavros"))
            .current_dir(project.path())
            .arg("compile")
            .args(extra)
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
    };
    invoke(&[]);
    let program_path = project.path().join("target/program.mavros");
    let original_program = std::fs::read(&program_path).unwrap();
    assert!(
        !project
            .path()
            .join("mavros_debug/verifier_bytecode.txt")
            .exists()
    );
    invoke(&[
        "--verifier-output",
        "target/verifier.json",
        "--include-debug-info",
    ]);
    assert_eq!(std::fs::read(&program_path).unwrap(), original_program);
    let artifact: serde_json::Value = serde_json::from_slice(
        &std::fs::read(project.path().join("target/verifier.json")).unwrap(),
    )
    .unwrap();
    assert!(artifact.get("constraints").is_none());
    let binary: Vec<u64> = serde_json::from_value(artifact["binary"].clone()).unwrap();
    let wl: mavros_vm::WitnessLayout =
        serde_json::from_value(artifact["witness_layout"].clone()).unwrap();
    let cl: mavros_vm::ConstraintsLayout =
        serde_json::from_value(artifact["constraints_layout"].clone()).unwrap();
    let debug: mavros_vm::bytecode::DebugInfo = serde_json::from_slice(
        &std::fs::read(project.path().join("target/verifier.debug.json")).unwrap(),
    )
    .unwrap();
    assert!(
        debug
            .functions
            .iter()
            .all(|f| f.name.starts_with("verifier_"))
    );
    assert_forward_bytecode(&binary);
    let r = vec![Field::from(3u64); cl.size()];
    let w = vec![Field::from(7u64); wl.size()];
    let result =
        mavros_vm::interpreter::run_verifier(&binary, &r, &w, wl, cl, Some(debug)).unwrap();
    let (_, r1cs) = api::compile_to_r1cs(project.path().to_path_buf(), false).unwrap();
    assert_eq!(
        [result.r_aw, result.r_bw, result.r_cw],
        reference(&r1cs, &r, &w)
    );
}
