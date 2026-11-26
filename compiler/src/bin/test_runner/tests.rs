use super::*;

fn lines(input: &[&str]) -> Vec<String> {
    input.iter().map(|line| line.to_string()).collect()
}

#[test]
fn corpus_field_selection_defaults_to_every_field() {
    assert_eq!(parse_fields_arg(&lines(&["test-runner"])), FieldId::ALL);
    for field in FieldId::ALL {
        assert_eq!(
            parse_fields_arg(&lines(&["test-runner", "--field", field.name()])),
            vec![field]
        );
    }
}

#[test]
fn determinism_tests_skip_the_field_selector() {
    let args = lines(&[
        "test-runner",
        "--check-determinism",
        "--field",
        "goldilocks",
        "--runs",
        "2",
        "sha256",
    ]);
    assert_eq!(parse_determinism_tests(&args), ["sha256"]);
    assert_eq!(parse_field_arg(&args), Some(FieldId::Goldilocks));
}

#[test]
fn separate_field_reports_merge_to_the_combined_report() {
    let mut results = Vec::new();
    for name in ["suite/z", "suite/a"] {
        for field in FieldId::ALL {
            let mut result = parse_child_output(
                name,
                field,
                TestExpectation::ExecutionSuccess,
                &lines(&[
                    "START:COMPILED",
                    "END:COMPILED:ok",
                    "START:R1CS",
                    "END:R1CS:ok:10:20",
                    "START:COMPILE",
                    "END:COMPILE:ok:30:40",
                    "START:WITGEN_RUN",
                    "END:WITGEN_RUN:error",
                    "START:AD_RUN",
                    "START:WASM_COMPILE",
                    "END:WASM_COMPILE:ok:50:60",
                ]),
            );
            result
                .steps
                .insert("AD_CORRECT".into(), Status::NotApplicable);
            results.push(result);
        }
    }
    let expected = render_markdown(&results);
    let (bn254, mut goldilocks): (Vec<_>, Vec<_>) = results
        .into_iter()
        .partition(|result| result.field == FieldId::Bn254);
    goldilocks.reverse();
    let bn254 = render_markdown(&bn254);
    assert!(!bn254.contains("goldilocks success rate"));
    let goldilocks = render_markdown(&goldilocks);
    assert!(!goldilocks.contains("bn254 success rate"));
    assert_eq!(
        merge_status_reports(&[bn254, goldilocks]).unwrap(),
        expected
    );
}

#[test]
fn merging_refuses_incomplete_or_ambiguous_field_reports() {
    let bn254 = single_row_status("suite/test", "✅");
    let goldilocks = bn254.replace("bn254", "goldilocks");
    assert!(merge_status_reports(&[]).is_err());
    assert!(merge_status_reports(&[bn254.clone()]).is_err());
    assert!(merge_status_reports(&[bn254.clone(), bn254.clone()]).is_err());
    assert!(
        merge_status_reports(&[bn254.clone(), goldilocks.replace("suite/test", "other")]).is_err()
    );
    assert!(merge_status_reports(&[bn254.clone(), goldilocks.replace("✅", "unknown")]).is_err());
    let unknown =
        merge_status_reports(&[bn254.clone(), goldilocks.replace("goldilocks", "babybear")])
            .unwrap_err();
    assert!(unknown.contains("unsupported report field"), "{unknown}");
    let duplicate = format!("{goldilocks}{}\n", goldilocks.lines().nth(2).unwrap());
    assert!(merge_status_reports(&[bn254.clone(), duplicate]).is_err());
    let mixed = format!("{goldilocks}{}\n", bn254.lines().nth(2).unwrap());
    assert!(merge_status_reports(&[bn254, mixed]).is_err());
}

#[test]
fn fields_have_independent_rates_and_round_trip_without_collisions() {
    let mut bn254 = ignored_test_result("same/test", FieldId::Bn254);
    bn254
        .steps
        .values_mut()
        .for_each(|status| *status = Status::Pass);
    let mut goldilocks = ignored_test_result("same/test", FieldId::Goldilocks);
    goldilocks.steps.insert("COMPILED".into(), Status::Fail);
    let ignored = ignored_test_result("ignored/test", FieldId::Goldilocks);
    let results = [bn254, goldilocks, ignored];
    assert_eq!(success_counts(&results, FieldId::Bn254), (16, 16));
    assert_eq!(success_counts(&results, FieldId::Goldilocks), (0, 16));
    let markdown = render_markdown(&results);
    assert!(markdown.contains("bn254 success rate: 100.0%"));
    assert!(markdown.contains("goldilocks success rate: 0.0%"));
    let table = parse_status_content(&markdown);
    assert_eq!(table.rows.len(), 3);
    assert_eq!(table.rows[0].field, "bn254");
    assert_eq!(table.rows[1].field, "goldilocks");
    assert_eq!(table.rows[0].name, table.rows[1].name);
}

#[test]
fn regression_checks_match_fields_and_accept_legacy_bn254_baselines() {
    let baseline = tempfile::NamedTempFile::new().unwrap();
    let current = tempfile::NamedTempFile::new().unwrap();
    let bn_pass = single_row_status("same/test", "✅");
    let gold_pass = bn_pass.replace("bn254", "goldilocks");
    let gold_fail = single_row_status("same/test", "❌").replace("bn254", "goldilocks");
    let append_row =
        |table: &str, other: &str| format!("{table}{}\n", other.lines().nth(2).unwrap());
    fs::write(baseline.path(), append_row(&bn_pass, &gold_fail)).unwrap();
    // Reversing the row order must not compare Goldilocks failures against BN254 passes.
    fs::write(current.path(), append_row(&gold_fail, &bn_pass)).unwrap();
    assert_eq!(check_regression(baseline.path(), current.path()), 0);
    fs::write(baseline.path(), append_row(&bn_pass, &gold_pass)).unwrap();
    assert_eq!(check_regression(baseline.path(), current.path()), 1);
    let legacy: String = bn_pass
        .lines()
        .map(|line| {
            let mut cells = markdown_cells(line);
            cells.pop();
            format!("| {} |\n", cells.join(" | "))
        })
        .collect();
    fs::write(baseline.path(), legacy).unwrap();
    assert_eq!(check_regression(baseline.path(), current.path()), 0);
}

#[test]
fn host_failure_fixtures_accept_only_their_intended_rejection() {
    for (name, expected_kind) in [
        ("prover_file_missing", "input-error"),
        ("prover_missing_param", "input-error"),
        ("return_wrong_value", "return-mismatch"),
    ] {
        let display_name = format!("noir_failure_tests/{name}");
        let mut entries = vec![TestEntry {
            path: PathBuf::from(&display_name),
            fixture_name: display_name.clone(),
            display_name: display_name.clone(),
            expectation: TestExpectation::ExecutionFailure,
        }];
        apply_test_expectation_overrides(&mut entries);
        let expectation = entries[0].expectation;
        assert_ne!(expectation, TestExpectation::ExecutionFailure);
        for kind in ["input-error", "return-mismatch", "error", "fail", "ok"] {
            let result = parse_child_output(
                &display_name,
                FieldId::Bn254,
                expectation,
                &lines(&[
                    "START:COMPILED",
                    "END:COMPILED:ok",
                    "START:WITGEN_RUN",
                    &format!("END:WITGEN_RUN:{kind}"),
                    "START:WITGEN_WASM_RUN",
                    &format!("END:WITGEN_WASM_RUN:{kind}"),
                ]),
            );
            let status = if kind == expected_kind {
                Status::Pass
            } else {
                Status::Fail
            };
            assert_eq!(result.steps["WITGEN_RUN"], status, "{name}: {kind}");
            assert_eq!(result.steps["WITGEN_WASM_RUN"], status, "{name}: {kind}");
            assert_eq!(result.steps["AD_RUN"], Status::NotApplicable);
        }
        let rejected = parse_child_output(
            &display_name,
            FieldId::Bn254,
            expectation,
            &lines(&["START:COMPILED", "END:COMPILED:reject"]),
        );
        assert_eq!(rejected.steps["COMPILED"], Status::Fail);
        let crashed = parse_child_output(
            &display_name,
            FieldId::Bn254,
            expectation,
            &lines(&["START:WITGEN_RUN", "START:WITGEN_WASM_RUN"]),
        );
        assert_eq!(crashed.steps["WITGEN_RUN"], Status::Crash);
        assert_eq!(crashed.steps["WITGEN_WASM_RUN"], Status::Crash);
    }
}

/// A fixture expected to trap passes on a trap only: a crash of the VM or of the host stays a
/// crash, where an expected execution failure takes either.
#[test]
fn an_expected_trap_is_not_satisfied_by_a_crash() {
    for (expectation, crash) in [
        (TestExpectation::WitgenTrap, Status::Crash),
        (TestExpectation::ExecutionFailure, Status::Pass),
    ] {
        let trapped = parse_child_output(
            "trap",
            FieldId::Bn254,
            expectation,
            &lines(&["START:WITGEN_RUN", "END:WITGEN_RUN:fail"]),
        );
        assert_eq!(trapped.steps["WITGEN_RUN"], Status::Pass, "{expectation:?}");
        let ran = parse_child_output(
            "trap",
            FieldId::Bn254,
            expectation,
            &lines(&["START:WITGEN_RUN", "END:WITGEN_RUN:ok"]),
        );
        assert_eq!(ran.steps["WITGEN_RUN"], Status::Fail, "{expectation:?}");
        let crashed = parse_child_output(
            "trap",
            FieldId::Bn254,
            expectation,
            &lines(&["START:WITGEN_RUN", "START:WITGEN_WASM_RUN"]),
        );
        assert_eq!(crashed.steps["WITGEN_RUN"], crash, "{expectation:?}");
        assert_eq!(crashed.steps["WITGEN_WASM_RUN"], crash, "{expectation:?}");
    }
}

#[test]
fn input_errors_do_not_satisfy_expected_runtime_failures() {
    for kind in ["error", "input-error", "return-mismatch"] {
        let result = parse_child_output(
            "invalid_input",
            FieldId::Bn254,
            TestExpectation::ExecutionFailure,
            &lines(&["START:WITGEN_RUN", &format!("END:WITGEN_RUN:{kind}")]),
        );
        assert_eq!(result.steps["WITGEN_RUN"], Status::Fail);
    }
}

#[test]
fn frontend_field_selection_reaches_the_noir_compiler() {
    fn run<F: MavrosField>() {
        let tmp = tempfile::tempdir().unwrap();
        fs::create_dir(tmp.path().join("src")).unwrap();
        fs::write(
            tmp.path().join("Nargo.toml"),
            "[package]\nname = 'field_test'\ntype = 'bin'\nauthors = []\n",
        )
        .unwrap();
        let modulus_minus_one = match F::ID {
            FieldId::Bn254 => {
                "21888242871839275222246405745257275088548364400416034343698204186575808495616"
            }
            FieldId::Goldilocks => "18446744069414584320",
        };
        fs::write(
            tmp.path().join("src/main.nr"),
            format!(
                "fn main() {{ let x: Field = -1; assert(x == {modulus_minus_one}); \
             let wide: u256 = 0xffffffffffffffffffffffffffffffff; \
             assert(wide + 1 == 0x100000000000000000000000000000000); }}"
            ),
        )
        .unwrap();
        let mut run = run_single_vm::<F>(tmp.path().to_path_buf(), false, None, false).unwrap();
        let r1cs = run.r1cs.as_ref().expect("R1CS generation must succeed");
        let artifact = run
            .driver
            .compile_bytecode_artifact(CodeGenOptions {
                check_constraints: true,
                include_debug_info: false,
            })
            .unwrap();
        let result = interpreter::run::<F>(
            &artifact.binary,
            r1cs.witness_layout,
            r1cs.constraints_layout,
            &[],
            None,
        )
        .unwrap();
        assert!(r1cs.check_witgen_output(
            &result.out_wit_pre_comm,
            &result.out_wit_post_comm,
            &result.out_a,
            &result.out_b,
            &result.out_c
        ));
    }
    run::<Bn254>();
    run::<Goldilocks>();
}

#[test]
#[should_panic(expected = "--run-single requires --field, one of bn254, goldilocks")]
fn a_child_without_a_field_is_refused() {
    ChildArgs::parse(&lines(&["runner", "--run-single", "suite/fixture"]));
}

#[test]
fn a_determinism_check_fails_when_any_field_fails() {
    assert_eq!(first_failure([0, 0]), 0);
    assert_eq!(first_failure([0, 1]), 1);
    assert_eq!(first_failure([1, 0]), 1);
    assert_eq!(first_failure([]), 0);
}

/// The text a fixture expects reaches its child, which is told to expect a failure; a fixture
/// expected to succeed passes neither.
#[test]
fn an_expected_error_reaches_the_child() {
    let path = Path::new("suite/fixture");
    let parsed = |field, expectation| {
        let args: Vec<String> = std::iter::once("runner".to_string())
            .chain(
                child_args(path, field, expectation, true)
                    .iter()
                    .map(|arg| arg.to_string_lossy().into_owned()),
            )
            .collect();
        let child = ChildArgs::parse(&args).unwrap();
        assert_eq!((child.path.as_path(), child.field), (path, field));
        assert!(child.analyze);
        (
            child.expect_failure,
            child.expected_error.map(str::to_string),
        )
    };
    for field in FieldId::ALL {
        assert_eq!(
            parsed(field, TestExpectation::CompileRejection(DATA_BUS_REFUSAL)),
            (true, Some(DATA_BUS_REFUSAL.to_string()))
        );
        assert_eq!(
            parsed(field, TestExpectation::ExecutionSuccess),
            (false, None)
        );
    }
}

/// The child holds a refusal to the text its fixture expects, on every field: the frontend's
/// data bus refusal is a `reject` with its text and a `fail` with any other.
#[test]
fn a_child_holds_a_refusal_to_its_expected_error() {
    fn outcomes<F: MavrosField>() -> [&'static str; 2] {
        let tmp = tempfile::tempdir().unwrap();
        fs::create_dir(tmp.path().join("src")).unwrap();
        fs::write(
            tmp.path().join("Nargo.toml"),
            "[package]\nname = 'data_bus'\ntype = 'bin'\nauthors = []\n",
        )
        .unwrap();
        fs::write(
            tmp.path().join("src/main.nr"),
            "fn main(x: call_data(0) u32) -> return_data u32 { x }\n",
        )
        .unwrap();
        let mut driver = Driver::new(Project::new(tmp.path().to_path_buf()).unwrap(), false);
        let result = driver.run_noir_compiler(F::CONFIG);
        [Some(DATA_BUS_REFUSAL), Some("another refusal")]
            .map(|expected| compile_outcome(F::ID, &result, expected))
    }
    assert_eq!(outcomes::<Bn254>(), ["reject", "fail"]);
    assert_eq!(outcomes::<Goldilocks>(), ["reject", "fail"]);
}

fn write_discovery_workspace(base: &Path, name: &str) -> PathBuf {
    let root = base.join(name);
    fs::create_dir_all(&root).unwrap();
    fs::write(
        root.join("Nargo.toml"),
        "[workspace]\nmembers = ['crates/a', 'crates/b', 'lib']\n",
    )
    .unwrap();
    for (member, name, kind) in [
        ("crates/a", "a", "bin"),
        ("crates/b", "b", "bin"),
        ("lib", "shared", "lib"),
    ] {
        let dir = root.join(member);
        fs::create_dir_all(dir.join("src")).unwrap();
        fs::write(
            dir.join("Nargo.toml"),
            format!("[package]\nname = '{name}'\ntype = '{kind}'\nauthors = []\n"),
        )
        .unwrap();
        let source = if kind == "bin" { "main.nr" } else { "lib.nr" };
        fs::write(dir.join("src").join(source), "fn main() {}\n").unwrap();
    }
    root
}

#[test]
fn workspace_discovery_reports_binary_members_and_honors_default() {
    let tmp = tempfile::tempdir().unwrap();
    let root = write_discovery_workspace(tmp.path(), "workspace");
    let discover = || collect_test_dirs(tmp.path(), "tests/", TestExpectation::ExecutionSuccess);
    let entries = discover();
    assert_eq!(
        entries
            .iter()
            .map(|e| e.display_name.as_str())
            .collect::<Vec<_>>(),
        ["tests/workspace/crates/a", "tests/workspace/crates/b"]
    );
    assert_eq!(
        entries[0].path,
        fs::canonicalize(root.join("crates/a")).unwrap()
    );
    assert!(
        entries
            .iter()
            .all(|e| e.expectation == TestExpectation::ExecutionSuccess)
    );

    fs::write(
        root.join("Nargo.toml"),
        "[workspace]\nmembers = ['crates/a', 'crates/b', 'lib']\ndefault-member = 'crates/b'\n",
    )
    .unwrap();
    let entries = discover();
    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0].display_name, "tests/workspace/crates/b");

    // Ordinary packages use their directory names; invalid manifests produce test entries.
    let entries = collect_test_dirs(
        &root.join("crates"),
        "tests/",
        TestExpectation::ExecutionSuccess,
    );
    assert_eq!(
        entries
            .iter()
            .map(|e| e.display_name.as_str())
            .collect::<Vec<_>>(),
        ["tests/a", "tests/b"]
    );
    fs::write(root.join("Nargo.toml"), "invalid manifest").unwrap();
    let entries = discover();
    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0].display_name, "tests/workspace");
}

#[test]
fn workspace_discovery_assigns_mixed_member_expectations() {
    let tmp = tempfile::tempdir().unwrap();
    write_discovery_workspace(tmp.path(), "workspace_fail");
    let mut entries = collect_test_dirs(
        tmp.path(),
        "noir/test_programs/execution_failure/",
        TestExpectation::ExecutionFailure,
    );
    // The parent applies suite-specific expectations after discovery.
    assert!(
        entries
            .iter()
            .all(|e| e.expectation == TestExpectation::ExecutionFailure)
    );
    apply_test_expectation_overrides(&mut entries);
    assert_eq!(entries.len(), 2);
    assert_eq!(
        entries[0].display_name,
        "noir/test_programs/execution_failure/workspace_fail/crates/a"
    );
    assert_eq!(entries[0].expectation, TestExpectation::ExecutionFailure);
    assert_eq!(
        entries[1].display_name,
        "noir/test_programs/execution_failure/workspace_fail/crates/b"
    );
    assert_eq!(entries[1].expectation, TestExpectation::ExecutionSuccess);
    // The passing-member exception belongs only to this specific upstream fixture.
    let mut entries = collect_test_dirs(tmp.path(), "local/", TestExpectation::ExecutionFailure);
    apply_test_expectation_overrides(&mut entries);
    assert!(
        entries
            .iter()
            .all(|e| e.expectation == TestExpectation::ExecutionFailure)
    );
}

/// A frontend error whose diagnostics are `errors`, each with `note`, and a warning saying
/// `warning`.
fn frontend_error(errors: &[&str], note: &str, warning: &str) -> DriverError {
    use noirc_errors::reporter::{CustomDiagnostic, DiagnosticKind};

    let mut diagnostics: Vec<_> = errors
        .iter()
        .map(|message| {
            let mut diagnostic = CustomDiagnostic::from_message(message, fm::FileId::dummy());
            diagnostic.add_note(note.to_string());
            diagnostic
        })
        .collect();
    diagnostics.push(CustomDiagnostic {
        kind: DiagnosticKind::Warning,
        ..CustomDiagnostic::from_message(warning, fm::FileId::dummy())
    });
    let rendered = errors
        .iter()
        .chain([&warning])
        .map(|message| format!("{message}\n  = {note}"))
        .collect();
    DriverError::NoirCompilerError {
        diagnostics,
        rendered,
    }
}

/// Without an expected text a frontend refusal counts on bn254 only: off bn254 it can mean a
/// missing builtin.
#[test]
fn an_unnamed_frontend_refusal_counts_on_bn254_only() {
    let error = frontend_error(
        &["break is only allowed in unconstrained functions"],
        "",
        "",
    );
    assert!(frontend_rejection_expected(FieldId::Bn254, &error, None));
    assert!(!frontend_rejection_expected(
        FieldId::Goldilocks,
        &error,
        None
    ));
}

/// An expected text holds a frontend refusal to it on every field, and it must be said by an
/// error: a warning or another error saying something else does not count.
#[test]
fn a_compile_rejection_text_holds_a_frontend_refusal_on_every_field() {
    let refused = frontend_error(
        &["call_data(0) is not supported"],
        "A program's inputs and return are read as bytes, so the data bus is not supported",
        "unused variable",
    );
    let warned = frontend_error(
        &["break is only allowed in unconstrained functions"],
        "",
        "the data bus is not supported here",
    );
    for field in FieldId::ALL {
        assert!(frontend_rejection_expected(
            field,
            &refused,
            Some(DATA_BUS_REFUSAL)
        ));
        assert!(
            !frontend_rejection_expected(field, &warned, Some(DATA_BUS_REFUSAL)),
            "a warning saying the text does not make the refusal expected"
        );
    }
}

/// A Mavros refusal with an expected text is held to it. Without one, a statically false
/// assertion is an expected failure, and a lowering refusal is an unsupported program.
#[test]
fn a_mavros_refusal_is_expected_only_as_its_fixture_says() {
    use mavros_compiler::compiler::{diagnostic::Diagnostic, ssa::SourceLocation};

    let refusal = |message: &str| {
        DriverError::LoweringRefused(vec![Diagnostic::error(
            message,
            SourceLocation::synthetic("fixture"),
        )])
    };
    let refused = refusal("a witnessed decomposition into 32 bytes a field element occupies");
    let assertion = DriverError::AssertConstantFailed(vec![SourceLocation::synthetic("fixture")]);

    assert!(!mavros_rejection_expected(&refused, None));
    assert!(mavros_rejection_expected(
        &refused,
        Some("bytes a field element occupies")
    ));
    assert!(!mavros_rejection_expected(&refused, Some("something else")));
    assert!(mavros_rejection_expected(&assertion, None));
    assert!(mavros_rejection_expected(
        &assertion,
        Some("value may depend on witness data")
    ));
    assert!(!mavros_rejection_expected(
        &assertion,
        Some("something else")
    ));
}

/// An expected-error file makes its fixture a compile rejection of the file's text, and an
/// empty one is no expectation at all.
#[test]
fn an_expected_error_file_names_a_compile_rejection() {
    let tmp = tempfile::tempdir().unwrap();
    for (name, text) in [("named", "break is only allowed\n"), ("blank", " \n")] {
        let dir = tmp.path().join(name);
        fs::create_dir_all(dir.join("src")).unwrap();
        fs::write(
            dir.join("Nargo.toml"),
            format!("[package]\nname = \"{name}\"\ntype = \"bin\"\nauthors = []\n"),
        )
        .unwrap();
        fs::write(dir.join("src/main.nr"), "fn main() {}\n").unwrap();
        fs::write(dir.join(EXPECTED_ERROR_FILE), text).unwrap();
    }
    let entries = collect_test_dirs(tmp.path(), "local/", TestExpectation::ExecutionFailure);
    let expectation = |name: &str| {
        entries
            .iter()
            .find(|entry| entry.display_name == format!("local/{name}"))
            .unwrap()
            .expectation
    };
    assert_eq!(
        expectation("named"),
        TestExpectation::CompileRejection("break is only allowed")
    );
    assert_eq!(expectation("blank"), TestExpectation::ExecutionFailure);
}

#[test]
fn workspace_ignore_matches_whole_fixtures_or_individual_members() {
    let tmp = tempfile::tempdir().unwrap();
    write_discovery_workspace(tmp.path(), "workspace");
    write_discovery_workspace(tmp.path(), "workspace_extra");
    let entries = collect_test_dirs(tmp.path(), "tests/", TestExpectation::ExecutionSuccess);
    for ignored in ["workspace", "tests/workspace", "/tests/workspace/"] {
        assert_eq!(
            entries
                .iter()
                .filter(|e| is_ignored_test(e, &[ignored]))
                .count(),
            2
        );
    }
    for ignored in ["workspace/crates/a", "tests/workspace/crates/a"] {
        let ignored_entries: Vec<_> = entries
            .iter()
            .filter(|e| is_ignored_test(e, &[ignored]))
            .collect();
        assert_eq!(ignored_entries.len(), 1);
        assert_eq!(ignored_entries[0].display_name, "tests/workspace/crates/a");
    }
    for ignored in ["work", "space", "", "/"] {
        assert!(!entries.iter().any(|e| is_ignored_test(e, &[ignored])));
    }
}

#[test]
fn only_keeps_the_tests_it_names() {
    let args: Vec<String> = ["runner", "--only", "a", "--jobs", "2", "--only", "b"]
        .map(String::from)
        .to_vec();
    assert_eq!(parse_only_args(&args), ["a", "b"]);
    assert!(parse_only_args(&args[..1]).is_empty());

    let tmp = tempfile::tempdir().unwrap();
    write_discovery_workspace(tmp.path(), "workspace");
    write_discovery_workspace(tmp.path(), "workspace_extra");
    let discover = || collect_test_dirs(tmp.path(), "tests/", TestExpectation::ExecutionSuccess);
    let names = |entries: &[TestEntry]| {
        entries
            .iter()
            .map(|e| e.display_name.clone())
            .collect::<Vec<_>>()
    };

    let mut entries = discover();
    let all = names(&entries);
    select_tests(&mut entries, &[]);
    assert_eq!(names(&entries), all);

    let mut entries = discover();
    select_tests(&mut entries, &["workspace/crates/a".to_string()]);
    assert_eq!(names(&entries), ["tests/workspace/crates/a"]);

    let mut entries = discover();
    select_tests(
        &mut entries,
        &[
            "workspace_extra".to_string(),
            "workspace/crates/b".to_string(),
        ],
    );
    assert_eq!(
        names(&entries),
        [
            "tests/workspace/crates/b",
            "tests/workspace_extra/crates/a",
            "tests/workspace_extra/crates/b"
        ]
    );
}

#[test]
#[should_panic(expected = "--only work: no test has this name")]
fn only_refuses_a_name_that_names_no_test() {
    let tmp = tempfile::tempdir().unwrap();
    write_discovery_workspace(tmp.path(), "workspace");
    let mut entries = collect_test_dirs(tmp.path(), "tests/", TestExpectation::ExecutionSuccess);
    select_tests(&mut entries, &["work".to_string()]);
}

#[test]
fn discovery_keeps_tests_whose_directory_cannot_be_resolved() {
    let tmp = tempfile::tempdir().unwrap();
    let missing = tmp.path().join("missing");
    let entries = discover_test_packages(
        missing.clone(),
        "tests/missing".into(),
        TestExpectation::ExecutionSuccess,
    );
    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0].path, missing);
    assert_eq!(entries[0].display_name, "tests/missing");
    assert!(is_ignored_test(&entries[0], &["missing"]));
}

#[test]
fn workspace_discovery_accepts_members_outside_the_test_directory() {
    let tmp = tempfile::tempdir().unwrap();
    let external = write_discovery_workspace(tmp.path(), "external");
    let root = tmp.path().join("suite/workspace");
    fs::create_dir_all(&root).unwrap();
    let member = fs::canonicalize(external.join("crates/a")).unwrap();
    fs::write(
        root.join("Nargo.toml"),
        format!("[workspace]\nmembers = ['{}']\n", member.display()),
    )
    .unwrap();
    let entries = collect_test_dirs(
        &tmp.path().join("suite"),
        "tests/",
        TestExpectation::ExecutionSuccess,
    );
    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0].path, member);
    assert!(entries[0].display_name.contains(member.to_str().unwrap()));
    assert!(is_ignored_test(&entries[0], &["workspace"]));
}

#[test]
fn execution_success_uses_raw_statuses() {
    let result = parse_child_output(
        "ok",
        FieldId::Bn254,
        TestExpectation::ExecutionSuccess,
        &lines(&["START:COMPILED", "END:COMPILED:ok"]),
    );

    assert_eq!(result.steps["COMPILED"], Status::Pass);
    assert_eq!(result.steps["R1CS"], Status::Skip);
}

#[test]
fn parses_executable_and_debug_sidecar_sizes() {
    let result = parse_child_output(
        "sizes",
        FieldId::Bn254,
        TestExpectation::ExecutionSuccess,
        &lines(&[
            "START:COMPILE",
            "END:COMPILE:ok:128:456",
            "START:WASM_COMPILE",
            "END:WASM_COMPILE:ok:789:1234",
        ]),
    );

    assert_eq!(result.binary_bytes, Some(128));
    assert_eq!(result.vm_debug_bytes, Some(456));
    assert_eq!(result.wasm_bytes, Some(789));
    assert_eq!(result.wasm_debug_bytes, Some(1234));

    let markdown = render_markdown(&[result]);
    let mut rendered_lines = markdown.lines();
    assert_eq!(
        markdown_cells(rendered_lines.next().unwrap()),
        STATUS_COLUMNS
    );
    rendered_lines.next();
    let row = markdown_cells(rendered_lines.next().unwrap());
    assert_eq!(row.len(), STATUS_COLUMNS.len());
    // Bytecode / VM-debug / WASM / WASM-debug sizes, immediately after the Cols column.
    assert_eq!(&row[5..9], ["128", "456", "789", "1234"]);
}

#[test]
fn legacy_status_rows_are_normalized_for_baseline_comparisons() {
    // A baseline predating the debug-sidecar columns. Its stage columns must still resolve by
    // name, and the size columns it lacks read as `None`, so it can be compared against a
    // current-format run across the schema transition.
    const LEGACY_STATUS_COLUMNS: &[&str] = &[
        "Test",
        "Compiled",
        "R1CS",
        "Rows",
        "Cols",
        "Bytecode Size",
        "WASM Size",
        "Compile",
        "Witgen Run VM",
        "Witgen Correct",
        "Witgen No Leak",
        "AD Run VM",
        "AD Correct",
        "AD No Leak",
        "WASM Compile",
        "Witgen WASM Run",
        "Witgen WASM Correct",
        "Witgen WASM No Leak",
        "AD WASM Run",
        "AD WASM Correct",
        "AD WASM No Leak",
    ];

    let mut status = tempfile::NamedTempFile::new().unwrap();
    writeln!(status, "| {} |", LEGACY_STATUS_COLUMNS.join(" | ")).unwrap();
    writeln!(
        status,
        "| {} |",
        vec!["---"; LEGACY_STATUS_COLUMNS.len()].join(" | ")
    )
    .unwrap();
    writeln!(
        status,
        "| example | ✅ | ✅ | 10 | 20 | 30 | 40 | ✅ | ➖ | ➖ | ➖ | ➖ | ➖ | ➖ | ✅ | ➖ | ➖ | ➖ | ➖ | ➖ | ➖ |"
    )
    .unwrap();

    let table = parse_status_table(status.path());
    validate_layout(&table, status.path());
    assert_eq!(table.rows.len(), 1);
    let row = &table.rows[0];
    assert_eq!(row.size_bytes, Some(30));
    assert_eq!(row.vm_debug_size_bytes, None);
    assert_eq!(row.wasm_size_bytes, Some(40));
    assert_eq!(row.wasm_debug_size_bytes, None);
    assert_eq!(table.cell(row, "Compile"), Some("✅"));
    assert_eq!(table.cell(row, "WASM Compile"), Some("✅"));
}

/// The expected refusal passes `COMPILED` and preempts every later stage.
#[test]
fn a_compile_rejection_passes_when_compilation_refuses_the_program() {
    let result = parse_child_output(
        "refused",
        FieldId::Bn254,
        TestExpectation::CompileRejection(DATA_BUS_REFUSAL),
        &lines(&["START:COMPILED", "END:COMPILED:reject"]),
    );

    assert_eq!(result.steps["COMPILED"], Status::Pass);
    for key in STEP_KEYS.iter().filter(|key| **key != "COMPILED") {
        assert_eq!(result.steps[*key], Status::NotApplicable, "{key}");
    }
}

/// A program that compiles has lost the refusal: `COMPILED` fails, whatever the later stages
/// do, and an unexpected compile failure fails too.
#[test]
fn a_compile_rejection_fails_when_the_program_compiles_or_fails_otherwise() {
    let compiled = parse_child_output(
        "refused",
        FieldId::Bn254,
        TestExpectation::CompileRejection(DATA_BUS_REFUSAL),
        &lines(&[
            "START:COMPILED",
            "END:COMPILED:ok",
            "START:R1CS",
            "END:R1CS:ok:10:20",
        ]),
    );
    assert_eq!(compiled.steps["COMPILED"], Status::Fail);
    assert_eq!(compiled.steps["R1CS"], Status::Pass);

    let failed = parse_child_output(
        "refused",
        FieldId::Bn254,
        TestExpectation::CompileRejection(DATA_BUS_REFUSAL),
        &lines(&["START:COMPILED", "END:COMPILED:fail"]),
    );
    assert_eq!(failed.steps["COMPILED"], Status::Fail);
}

#[test]
fn execution_failure_passes_when_witgen_runs_fail() {
    let result = parse_child_output(
        "exec_fail",
        FieldId::Bn254,
        TestExpectation::ExecutionFailure,
        &lines(&[
            "START:COMPILED",
            "END:COMPILED:ok",
            "START:WITGEN_RUN",
            "END:WITGEN_RUN:fail",
            "START:WITGEN_WASM_RUN",
            "END:WITGEN_WASM_RUN:fail",
        ]),
    );

    assert_eq!(result.steps["COMPILED"], Status::Pass);
    assert_eq!(result.steps["WITGEN_RUN"], Status::Pass);
    assert_eq!(result.steps["WITGEN_CORRECT"], Status::NotApplicable);
    assert_eq!(result.steps["WITGEN_WASM_RUN"], Status::Pass);
    assert_eq!(result.steps["WITGEN_WASM_CORRECT"], Status::NotApplicable);
}

#[test]
fn execution_failure_reports_native_ad_steps_as_not_applicable() {
    // The child skips the native AD pipeline for expected-failure tests, so
    // no native AD START/END markers are emitted; those columns must read
    // N/A, not ➖.
    let result = parse_child_output(
        "exec_fail",
        FieldId::Bn254,
        TestExpectation::ExecutionFailure,
        &lines(&[
            "START:COMPILED",
            "END:COMPILED:ok",
            "START:WITGEN_RUN",
            "END:WITGEN_RUN:fail",
        ]),
    );

    for key in ["AD_RUN", "AD_CORRECT", "AD_NOLEAK"] {
        assert_eq!(result.steps[key], Status::NotApplicable, "{key}");
    }
}

#[test]
fn execution_failure_uses_raw_ad_wasm_statuses() {
    let result = parse_child_output(
        "exec_fail",
        FieldId::Bn254,
        TestExpectation::ExecutionFailure,
        &lines(&[
            "START:AD_WASM_RUN",
            "END:AD_WASM_RUN:ok",
            "START:AD_WASM_CORRECT",
            "END:AD_WASM_CORRECT:ok",
            "START:AD_WASM_NOLEAK",
            "END:AD_WASM_NOLEAK:ok",
        ]),
    );

    assert_eq!(result.steps["AD_WASM_RUN"], Status::Pass);
    assert_eq!(result.steps["AD_WASM_CORRECT"], Status::Pass);
    assert_eq!(result.steps["AD_WASM_NOLEAK"], Status::Pass);
}

#[test]
fn execution_failure_passes_when_r1cs_rejects_program() {
    // A statically-unsatisfiable program is rejected during R1CS generation. The clean
    // rejection is the expected failure, so R1CS reads ✅. Every later stage is preempted
    // by that rejection and must read N/A — not ➖ Skip — so it neither counts as a failure
    // nor dilutes the success rate (whose denominator counts every non-N/A cell).
    let result = parse_child_output(
        "exec_fail",
        FieldId::Bn254,
        TestExpectation::ExecutionFailure,
        &lines(&[
            "START:COMPILED",
            "END:COMPILED:ok",
            "START:R1CS",
            "END:R1CS:reject",
        ]),
    );

    assert_eq!(result.steps["COMPILED"], Status::Pass);
    assert_eq!(result.steps["R1CS"], Status::Pass);
    for key in [
        "COMPILE",
        "WITGEN_RUN",
        "WITGEN_CORRECT",
        "WITGEN_NOLEAK",
        "WASM_COMPILE",
        "WITGEN_WASM_RUN",
        "AD_WASM_RUN",
    ] {
        assert_eq!(result.steps[key], Status::NotApplicable, "{key}");
    }
}

#[test]
fn execution_failure_passes_when_noir_frontend_rejects_program() {
    // The Noir frontend itself can reject a program that will never execute (e.g. the
    // comptime bit-shift overflow in `comptime_bitshift_failure`). That clean rejection at
    // COMPILED is the expected failure: COMPILED reads ✅ and every later stage — which
    // never ran — reads N/A rather than dragging the success rate down.
    let result = parse_child_output(
        "exec_fail",
        FieldId::Bn254,
        TestExpectation::ExecutionFailure,
        &lines(&["START:COMPILED", "END:COMPILED:reject"]),
    );

    assert_eq!(result.steps["COMPILED"], Status::Pass);
    for key in [
        "R1CS",
        "COMPILE",
        "WITGEN_RUN",
        "WASM_COMPILE",
        "WITGEN_WASM_RUN",
        "AD_WASM_RUN",
    ] {
        assert_eq!(result.steps[key], Status::NotApplicable, "{key}");
    }
}

#[test]
fn execution_failure_does_not_accept_plain_r1cs_failure() {
    // Only a clean rejection counts. A plain R1CS `fail` (some other error) or a crash is a
    // real problem and must not be silently turned green by the expected-failure flag.
    let failed = parse_child_output(
        "exec_fail",
        FieldId::Bn254,
        TestExpectation::ExecutionFailure,
        &lines(&[
            "START:COMPILED",
            "END:COMPILED:ok",
            "START:R1CS",
            "END:R1CS:fail",
        ]),
    );
    assert_eq!(failed.steps["R1CS"], Status::Fail);

    let crashed = parse_child_output(
        "exec_fail",
        FieldId::Bn254,
        TestExpectation::ExecutionFailure,
        &lines(&["START:COMPILED", "END:COMPILED:ok", "START:R1CS"]),
    );
    assert_eq!(crashed.steps["R1CS"], Status::Crash);
}

#[test]
fn execution_success_treats_r1cs_rejection_as_failure() {
    // A program that is supposed to execute should never be rejected as unsatisfiable.
    let result = parse_child_output(
        "ok",
        FieldId::Bn254,
        TestExpectation::ExecutionSuccess,
        &lines(&[
            "START:COMPILED",
            "END:COMPILED:ok",
            "START:R1CS",
            "END:R1CS:reject",
        ]),
    );
    assert_eq!(result.steps["R1CS"], Status::Fail);
}

#[test]
fn execution_failure_fails_when_witgen_runs_succeed() {
    let result = parse_child_output(
        "exec_fail",
        FieldId::Bn254,
        TestExpectation::ExecutionFailure,
        &lines(&[
            "START:WITGEN_RUN",
            "END:WITGEN_RUN:ok",
            "START:WITGEN_WASM_RUN",
            "END:WITGEN_WASM_RUN:ok",
        ]),
    );

    assert_eq!(result.steps["WITGEN_RUN"], Status::Fail);
    assert_eq!(result.steps["WITGEN_WASM_RUN"], Status::Fail);
}

#[test]
fn r1cs_line_parses_rows_and_cols() {
    let result = parse_child_output(
        "ok",
        FieldId::Bn254,
        TestExpectation::ExecutionSuccess,
        &lines(&["START:R1CS", "END:R1CS:ok:100:50"]),
    );
    assert_eq!(result.rows, Some(100));
    assert_eq!(result.cols, Some(50));
    // `END:R1CS` never carries the analysis breakdown; that arrives (opt-in) on a separate line.
    assert_eq!(result.analysis_stats, None);
}

/// A generic INFO line is invisible to the stage parser; the opt-in `INFO:R1CS_ANALYSIS` line
/// is captured for stdout echoing but likewise does not affect stage parsing.
#[test]
fn info_lines_are_ignored_but_analysis_is_captured() {
    let result = parse_child_output(
        "ok",
        FieldId::Bn254,
        TestExpectation::ExecutionSuccess,
        &lines(&[
            "START:R1CS",
            "INFO:SOMETHING:noise",
            "INFO:R1CS_ANALYSIS:algebraic_rows=10 removable_rows=2 rounds=3",
            "END:R1CS:ok:10:5",
        ]),
    );
    assert_eq!(result.steps["R1CS"], Status::Pass);
    assert_eq!(result.rows, Some(10));
    assert_eq!(
        result.analysis_stats.as_deref(),
        Some("algebraic_rows=10 removable_rows=2 rounds=3")
    );
}

fn single_row_status(name: &str, stage_cell: &str) -> String {
    let row: Vec<_> = STATUS_COLUMNS
        .iter()
        .map(|column| match *column {
            "Test" => name,
            "Field" => "bn254",
            "Rows"
            | "Cols"
            | "Bytecode Size"
            | "VM Debug Sidecar Size"
            | "WASM Size"
            | "WASM Debug Sidecar Size" => "-",
            _ => stage_cell,
        })
        .collect();
    format!(
        "| {} |\n| {} |\n| {} |\n",
        STATUS_COLUMNS.join(" | "),
        vec!["---"; STATUS_COLUMNS.len()].join(" | "),
        row.join(" | "),
    )
}

#[test]
fn regression_check_excludes_intentionally_ignored_rows() {
    let baseline = tempfile::NamedTempFile::new().unwrap();
    let current = tempfile::NamedTempFile::new().unwrap();
    fs::write(
        baseline.path(),
        single_row_status("noir/test/ignored", "✅"),
    )
    .unwrap();
    fs::write(current.path(), single_row_status("noir/test/ignored", "➖")).unwrap();

    assert_eq!(check_regression(baseline.path(), current.path()), 0);

    let table = parse_status_table(current.path());
    assert!(row_is_ignored(&table, &table.rows[0]));
}

/// Current-format table: columns resolve by name.
#[test]
fn status_table_parses_by_header_name() {
    let table = parse_status_content(
        "| Test | Compiled | R1CS | Rows | Cols | Bytecode Size | WASM Size | Compile |\n\
         |------|----------|------|------|------|---------------|-----------|---------|\n\
         | a/b | ✅ | ✅ | 100 | 50 | 2488 | 186940 | ✅ |\n\
         | c/d | ✅ | ✅ | 10 | 5 | 100 | 200 | ✅ |\n",
    );
    assert_eq!(table.rows.len(), 2);
    let row = &table.rows[0];
    assert_eq!(row.name, "a/b");
    assert_eq!(row.rows, Some(100));
    assert_eq!(row.cols, Some(50));
    assert_eq!(row.size_bytes, Some(2488));
    assert_eq!(table.cell(row, "R1CS"), Some("✅"));
}

/// A baseline that still carries the now-removed `Shrink` column must parse fine: unknown
/// columns are resolved by name and simply ignored, so `--check-regression`/`--check-growth`
/// keep working against pre-transition baselines.
#[test]
fn status_table_tolerates_extra_shrink_column() {
    let table = parse_status_content(
        "| Test | Compiled | R1CS | Rows | Cols | Shrink | Bytecode Size | WASM Size | Compile |\n\
         |------|----------|------|------|------|--------|---------------|-----------|---------|\n\
         | a/b | ✅ | ❌ | 100 | 50 | 8.0% | 2488 | 186940 | ✅ |\n",
    );
    let row = &table.rows[0];
    assert_eq!(row.rows, Some(100));
    assert_eq!(row.size_bytes, Some(2488));
    assert_eq!(table.cell(row, "R1CS"), Some("❌"));
    // The retired column is no longer part of `ParsedRow`, but by-name access still finds its
    // raw cell when a legacy baseline provides it.
    assert_eq!(table.cell(row, "Shrink"), Some("8.0%"));
}

/// Rendered output must round-trip through the parser with columns intact.
#[test]
fn render_markdown_round_trips_through_parser() {
    let steps: HashMap<String, Status> = STEP_KEYS
        .iter()
        .map(|&key| (key.to_string(), Status::Pass))
        .collect();
    let results = vec![TestResult {
        name: "suite/case".to_string(),
        field: FieldId::Bn254,
        steps,
        rows: Some(200),
        cols: Some(80),
        analysis_stats: None,
        binary_bytes: Some(1234),
        vm_debug_bytes: None,
        wasm_bytes: None,
        wasm_debug_bytes: None,
    }];
    let md = render_markdown(&results);
    let table = parse_status_content(&md);
    for name in REGRESSION_COLS {
        assert!(table.columns.contains_key(*name), "{name} missing");
    }
    let row = &table.rows[0];
    assert_eq!(row.name, "suite/case");
    assert_eq!(row.rows, Some(200));
    assert_eq!(row.wasm_size_bytes, None);
    assert_eq!(table.cell(row, "AD WASM No Leak"), Some("✅"));
}

/// A growth report with one field, a sentence, and a table of `rows` rows.
fn growth_report(rows: usize) -> GrowthReport {
    let mut report = GrowthReport::default();
    report
        .blocks
        .push(ReportBlock::Heading("bn254".to_string()));
    report.text("**bn254 success rate on test cases: 100.0%**\n\n");
    push_change_table(
        &mut report,
        "WASM Size (bytes) grew",
        &(0..rows)
            .map(|row| format!("| noir_tests/test_{row} | 100 | 140 | +40 (+40.0%) |"))
            .collect::<Vec<_>>(),
    );
    report
}

/// The report's list rows, in order, wherever they are.
fn table_rows(text: &str) -> Vec<&str> {
    text.lines()
        .filter(|line| line.starts_with("| noir_tests/"))
        .collect()
}

#[test]
fn a_growth_report_that_fits_is_one_unnumbered_comment() {
    let report = growth_report(5);
    let whole = report.render();
    assert_eq!(report.into_comments(GROWTH_COMMENT_BYTES), vec![whole]);
}

/// Splitting loses nothing: every row arrives once, in order, and each comment is a well-formed
/// piece of the report under the limit.
#[test]
fn a_growth_report_too_long_for_one_comment_is_split_without_losing_a_row() {
    let limit = 2_000;
    let report = growth_report(100);
    let whole = report.render();
    let comments = report.into_comments(limit);

    assert!(comments.len() > 1, "{} comment(s)", comments.len());
    let rows: Vec<&str> = comments
        .iter()
        .flat_map(|comment| table_rows(comment))
        .collect();
    assert_eq!(rows, table_rows(&whole));
    for (index, comment) in comments.iter().enumerate() {
        assert!(
            comment.len() <= limit,
            "comment {index} is {} bytes",
            comment.len()
        );
        assert!(comment.starts_with(&format!(
            "_Growth report, part {} of {}._\n\n",
            index + 1,
            comments.len()
        )));
        assert_eq!(
            comment.matches("<details>").count(),
            comment.matches("</details>").count()
        );
    }
    // A comment that continues the table reopens its section, summary and header.
    let last = comments.last().unwrap();
    assert!(last.contains("## bn254 (continued)\n\n<details>"));
    assert!(
        last.contains("<summary><b>WASM Size (bytes) grew (100 test(s)), continued</b></summary>")
    );
    assert!(
        last.contains("| Test | Before | After | Change |\n|------|--------|-------|--------|\n")
    );
}

/// A refusal anywhere but `COMPILED` is not the expected one, a crash at `COMPILED` stays a
/// crash, and a later stage keeps its raw status.
#[test]
fn a_compile_rejection_scores_other_outcomes_by_what_ran() {
    let expectation = TestExpectation::CompileRejection(DATA_BUS_REFUSAL);
    let refused_late = parse_child_output(
        "refused",
        FieldId::Bn254,
        expectation,
        &lines(&[
            "START:COMPILED",
            "END:COMPILED:ok",
            "START:R1CS",
            "END:R1CS:reject",
        ]),
    );
    assert_eq!(refused_late.steps["COMPILED"], Status::Fail);
    assert_eq!(refused_late.steps["R1CS"], Status::Fail);

    let crashed = parse_child_output(
        "refused",
        FieldId::Bn254,
        expectation,
        &lines(&["START:COMPILED"]),
    );
    assert_eq!(crashed.steps["COMPILED"], Status::Crash);

    let ran = parse_child_output(
        "refused",
        FieldId::Bn254,
        expectation,
        &lines(&[
            "START:COMPILED",
            "END:COMPILED:ok",
            "START:R1CS",
            "END:R1CS:ok:10:20",
            "START:COMPILE",
            "END:COMPILE:ok:80:40",
            "START:WITGEN_RUN",
            "END:WITGEN_RUN:input-error",
        ]),
    );
    assert_eq!(ran.steps["COMPILED"], Status::Fail);
    assert_eq!(ran.steps["WITGEN_RUN"], Status::Fail);
}

/// `--expected-error` takes the next argument as the text.
#[test]
fn the_expected_error_is_the_argument_after_its_flag() {
    let args = lines(&[
        "--expect-failure",
        "--expected-error",
        "two words",
        "--field",
    ]);
    assert_eq!(parse_expected_error_arg(&args), Some("two words"));
    assert_eq!(
        parse_expected_error_arg(&lines(&["--expect-failure"])),
        None
    );
}

#[test]
#[should_panic(expected = "--expected-error requires the error's text")]
fn an_expected_error_flag_without_its_text_is_refused() {
    parse_expected_error_arg(&lines(&["--expected-error"]));
}

/// A full run takes each of its flags, with a value where one needs it.
#[test]
fn a_full_run_takes_its_flags() {
    validate_parent_args(&lines(&[
        "runner",
        "--field",
        "bn254",
        "--output",
        "partial.md",
        "--jobs",
        "4",
        "--only",
        "a",
        "--analyze-r1cs",
    ]));
    validate_parent_args(&lines(&["runner"]));
}

/// A full run writes `STATUS.md` by default, so a mistyped flag, a flag missing its value, or a
/// mode given too few arguments stops it before it starts.
#[test]
fn a_full_run_refuses_what_it_does_not_take() {
    for args in [
        &["runner", "--onyl", "a"][..],
        &["runner", "--only=a", "--output", "partial.md"],
        &["runner", "--check-regression", "baseline.md"],
        &["runner", "--check-growth", "baseline.md"],
        &["runner", "--output"],
        &["runner", "--only", "a", "--jobs"],
    ] {
        let refused = std::panic::catch_unwind(|| validate_parent_args(&lines(args)));
        assert!(refused.is_err(), "{args:?} was taken");
    }
}

#[test]
#[should_panic(expected = "--output requires a path")]
fn an_output_flag_without_its_path_is_refused() {
    parse_output_arg(&lines(&["runner", "--only", "a", "--output"]));
}

fn entry_named(name: &str, expectation: TestExpectation) -> TestEntry {
    TestEntry {
        path: PathBuf::from(name),
        fixture_name: name.to_string(),
        display_name: name.to_string(),
        expectation,
    }
}

/// A field override replaces the entry's expectation on its field only, and the entry's
/// expectation, an override from the field-agnostic table included, holds elsewhere.
#[test]
fn a_field_override_takes_precedence_on_its_field() {
    let overrides = [
        ("a", FieldId::Goldilocks, TestExpectation::ExecutionFailure),
        ("b", FieldId::Bn254, TestExpectation::ReturnMismatch),
    ];
    let a = entry_named("a", TestExpectation::CompileRejection(DATA_BUS_REFUSAL));
    assert_eq!(
        expectation_in(&overrides, &a, FieldId::Goldilocks),
        TestExpectation::ExecutionFailure
    );
    assert_eq!(
        expectation_in(&overrides, &a, FieldId::Bn254),
        TestExpectation::CompileRejection(DATA_BUS_REFUSAL)
    );
    let b = entry_named("b", TestExpectation::ExecutionSuccess);
    assert_eq!(
        expectation_in(&overrides, &b, FieldId::Bn254),
        TestExpectation::ReturnMismatch
    );
    assert_eq!(
        expectation_in(&overrides, &b, FieldId::Goldilocks),
        TestExpectation::ExecutionSuccess
    );
    let c = entry_named("c", TestExpectation::ExecutionSuccess);
    for field in FieldId::ALL {
        assert_eq!(
            expectation_in(&overrides, &c, field),
            TestExpectation::ExecutionSuccess
        );
    }
}

#[test]
fn overrides_naming_discovered_tests_once_are_accepted() {
    let entries = [entry_named("a", TestExpectation::ExecutionSuccess)];
    validate_overrides(
        &entries,
        &[("a", TestExpectation::ExecutionFailure)],
        &[("a", FieldId::Goldilocks, TestExpectation::ExecutionFailure)],
        true,
    );
}

#[test]
#[should_panic(expected = "the expectation override for `gone` names no test")]
fn an_override_naming_no_test_is_refused() {
    let entries = [entry_named("a", TestExpectation::ExecutionSuccess)];
    validate_overrides(
        &entries,
        &[("gone", TestExpectation::ExecutionFailure)],
        &[],
        true,
    );
}

/// Without Noir's programs, an override naming one is left unchecked; one naming a local fixture
/// is still checked.
#[test]
fn only_overrides_of_noir_programs_wait_for_them() {
    let entries = [entry_named("local", TestExpectation::ExecutionSuccess)];
    let noir = ("noir/test_programs/gone", TestExpectation::ExecutionFailure);
    validate_overrides(&entries, &[noir], &[], false);
    let refused = std::panic::catch_unwind(|| {
        validate_overrides(
            &entries,
            &[("gone", TestExpectation::ExecutionFailure)],
            &[],
            false,
        )
    });
    assert!(refused.is_err());
}

/// An override would silently replace the error a fixture's `expected_error.txt` names.
#[test]
#[should_panic(expected = "`a` has an expectation override and an expected_error.txt")]
fn an_override_of_a_fixture_naming_its_error_is_refused() {
    let entries = [entry_named("a", TestExpectation::CompileRejection("text"))];
    validate_overrides(
        &entries,
        &[],
        &[("a", FieldId::Goldilocks, TestExpectation::ExecutionFailure)],
        true,
    );
}

#[test]
#[should_panic(expected = "`a` has two expectation overrides on goldilocks")]
fn a_test_overridden_twice_on_one_field_is_refused() {
    let entries = [entry_named("a", TestExpectation::ExecutionSuccess)];
    let twice = ("a", FieldId::Goldilocks, TestExpectation::ExecutionFailure);
    validate_overrides(&entries, &[], &[twice, twice], true);
}

/// A test that compiled at the baseline and is now refused as expected is a warning, even when
/// its stages score as handled and nothing grew. A test that compiles but fails at R1CS, or
/// whose expected failure is caught there, is not refused.
#[test]
fn a_test_newly_refused_is_a_growth_warning() {
    let result = |rows: Option<usize>, r1cs: Status| {
        let steps = STEP_KEYS
            .iter()
            .map(|&key| {
                let status = match key {
                    "COMPILED" => Status::Pass,
                    "R1CS" => r1cs,
                    _ if rows.is_some() => Status::Pass,
                    _ => Status::NotApplicable,
                };
                (key.to_string(), status)
            })
            .collect();
        TestResult {
            name: "suite/case".to_string(),
            field: FieldId::Bn254,
            steps,
            rows,
            cols: rows,
            analysis_stats: None,
            binary_bytes: None,
            vm_debug_bytes: None,
            wasm_bytes: None,
            wasm_debug_bytes: None,
        }
    };
    let baseline = parse_status_content(&render_markdown(&[result(Some(10), Status::Pass)]));
    let growth_against = |current: TestResult| {
        let current = parse_status_content(&render_markdown(&[current]));
        let mut report = GrowthReport::default();
        check_growth_for_field(&baseline, &current, "bn254", &mut report);
        report.render()
    };
    const WARNING: &str = "compiled at the baseline and are now refused";

    let text = growth_against(result(None, Status::NotApplicable));
    assert!(text.contains(WARNING), "{text}");
    assert!(text.contains("suite/case"), "{text}");
    for r1cs in [Status::Fail, Status::Pass] {
        let text = growth_against(result(None, r1cs));
        assert!(!text.contains(WARNING), "R1CS {r1cs:?}: {text}");
    }
}
