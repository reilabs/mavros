use std::panic::{self, AssertUnwindSafe};
use std::path::{Path, PathBuf};

#[cfg(not(feature = "goldilocks"))]
use mavros_compiler::driver::Driver;
use mavros_compiler::project::Project;

use super::validation_frontend::compile_for_validation;
use super::{
    DiffOutcome, DiffValue, IntValue, Value, inputs_from_prover_toml, interpret_with_inputs,
    outcomes_equivalent,
};
#[cfg(not(feature = "goldilocks"))]
use super::{InterpretError, expected_return_from_prover_toml, interpret};
use num_bigint::BigInt;

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../noir_tests")
        .join(name)
}

/// Negative fixtures (programs expected to fail to compile or assert) live in this crate, not the
/// shared `noir_tests/` corpus, which the test runner treats as positive-only.
#[cfg(not(feature = "goldilocks"))]
fn negative_fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("fixtures")
        .join(name)
}

/// Compile a fixture through Noir's frontend + monomorphizer and interpret the resulting AST.
#[cfg(not(feature = "goldilocks"))]
fn interpret_fixture(name: &str) -> Result<Value, Box<dyn std::error::Error>> {
    let project = Project::new(fixture(name))?;
    let mut driver = Driver::new(project, false);
    driver.run_noir_compiler()?;
    let program = driver.monomorphized_program();
    Ok(interpret(program)?)
}

/// Baseline differential oracle under bn254: the self-checking corpus program interprets to a
/// clean `Unit` with every `assert` holding. This proves the interpreter agrees with Noir's
/// own semantics on real monomorphized output. Under `--features goldilocks` the same source
/// cannot yet be compiled (the auto-injected bn254 stdlib blocks it — CRY-9), so the Goldilocks
/// side of the differential is gated until the stdlib port lands; the interpreter itself is
/// already field-agnostic and will validate the Goldilocks AST unchanged once it does.
#[cfg(not(feature = "goldilocks"))]
#[test]
fn interprets_basic_corpus_program() {
    let result = interpret_fixture("interp_basic").expect("interpretation should succeed");
    assert_eq!(result, Value::Unit, "main returns unit");
}

/// The oracle must bite: a program whose (computed, non-const-folded) assertion is false
/// interprets to an `AssertionFailed`, not a clean pass. Without this the green result above
/// would be meaningless.
#[cfg(not(feature = "goldilocks"))]
#[test]
fn detects_false_assertion() {
    let project = Project::new(negative_fixture("interp_assert_fail")).expect("project");
    let mut driver = Driver::new(project, false);
    driver.run_noir_compiler().expect("frontend compile");
    match interpret(driver.monomorphized_program()) {
        Err(InterpretError::AssertionFailed { .. }) => {}
        other => panic!("expected AssertionFailed, got {other:?}"),
    }
}

/// The validation frontend's error boundary must hold: `run_noir_frontend_for_validation`
/// tolerates elaboration errors only in code `main` does not reach (the bn254 crypto stdlib
/// under Goldilocks). A type error in the *reachable* set must still be rejected — otherwise
/// the oracle would validate a silently-miscompiled AST (false confidence). The fixture's `main`
/// binds a `u64` to a `bool` in its own body, so the error is unambiguously reachable. A clean
/// `Ok(())` here is the failure mode we are guarding against; a monomorphizer panic on the
/// elaborator's `Error` node is also a rejection (not silent acceptance), so both count as pass.
#[cfg(not(feature = "goldilocks"))]
#[test]
fn rejects_reachable_type_error() {
    let outcome = panic::catch_unwind(AssertUnwindSafe(|| {
        let project = Project::new(negative_fixture("interp_reachable_error")).expect("project");
        compile_for_validation(&project).map(|_| ())
    }));
    match outcome {
        Ok(Err(_)) => {} // rejected cleanly — desired
        Err(_) => {}     // monomorphizer panicked on the Error node — also a rejection
        Ok(Ok(())) => panic!(
            "reachable type error was silently accepted: the validation frontend produced a \
             mono-AST for un-type-checkable reachable code — oracle false-confidence hole"
        ),
    }
}

/// The tolerated-error file invariant must hold: a program that *reaches* code from a file
/// whose elaboration errors were tolerated (the stdlib's `ops/arith.nr` under Goldilocks — its
/// `wrapping_sub_hlp` carries a 2^128 Field constant) is rejected loudly. Without this, the
/// wrongly-typed HIR left behind by the tolerated error could monomorphize *silently* into a
/// wrong mono-AST and the oracle would "validate" it. The bn254 counterpart below proves the
/// fixture itself is sound, so this rejection is the invariant firing, not a broken fixture.
#[cfg(feature = "goldilocks")]
#[test]
fn rejects_reached_dependency_error() {
    let project = Project::new(fixture("interp_reached_dep_error")).expect("project");
    let err = match compile_for_validation(&project) {
        Ok(_) => panic!(
            "a program reaching code from a tolerated-error file must be rejected, not validated"
        ),
        Err(e) => format!("{e:?}"),
    };
    assert!(
        err.contains("failed elaboration"),
        "expected the tolerated-file invariant rejection, got: {err}"
    );
}

/// bn254 positive control for [`rejects_reached_dependency_error`]: the same fixture compiles
/// and interprets cleanly when the stdlib elaborates (proving the Goldilocks rejection above is
/// the invariant firing, not a fixture defect). `3.wrapping_sub(1) == 2`.
#[cfg(not(feature = "goldilocks"))]
#[test]
fn interprets_reached_dep_fixture_on_bn254() {
    let project = Project::new(fixture("interp_reached_dep_error")).expect("project");
    let validated = compile_for_validation(&project).expect("clean compile under bn254");
    let x = Value::Int(IntValue {
        signed: false,
        bits: 64,
        value: BigInt::from(3u64),
    });
    let result = interpret_with_inputs(&validated.program, vec![x]).expect("interpret");
    let expected = Value::Int(IntValue {
        signed: false,
        bits: 64,
        value: BigInt::from(2u64),
    });
    assert_eq!(result, expected);
}

/// The `Prover.toml` input bridge, exercised from an always-present fixture (the corpus-based
/// tests below skip when the Noir checkout is absent, so without this the bridge would be
/// untested in such environments). `interp_inputs_u64/Prover.toml` sets `x = 3`; main computes
/// `x * 2 + (p + 1)` natively in u64, the same value under either build field.
#[test]
fn interprets_fixture_inputs_from_prover_toml() {
    let root = fixture("interp_inputs_u64");
    let project = Project::new(root.clone()).expect("project");
    let validated = compile_for_validation(&project).expect("frontend");
    let toml = std::fs::read_to_string(root.join("Prover.toml")).expect("Prover.toml");
    let inputs =
        inputs_from_prover_toml(&validated.program, &validated.abi, &toml).expect("inputs");

    let result = interpret_with_inputs(&validated.program, inputs).expect("interpret");
    let expected = Value::Int(IntValue {
        signed: false,
        bits: 64,
        value: BigInt::from(18446744069414584328u64),
    });
    assert_eq!(result, expected, "input bridge must feed x = 3");
}

/// A program whose `main` takes inputs interprets correctly when those inputs are supplied from
/// `Prover.toml`. `assert_statement` is `main(x: Field, y: pub Field)` with `x == y == 3`, so a
/// clean `Unit` proves the input bridge feeds the right values.
#[cfg(not(feature = "goldilocks"))]
#[test]
fn interprets_program_with_inputs() {
    let program_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../noir/test_programs/execution_success/assert_statement");
    if !program_dir.is_dir() {
        eprintln!("SKIPPED (vacuous pass): noir corpus not checked out at ../../noir");
        return;
    }
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path().join("pkg");
    copy_dir(&program_dir, &root);

    let project = Project::new(root.clone()).unwrap();
    let mut driver = Driver::new(project, false);
    driver.run_noir_compiler().unwrap();
    let program = driver.monomorphized_program();
    let toml = std::fs::read_to_string(root.join("Prover.toml")).unwrap();
    let inputs = inputs_from_prover_toml(program, driver.abi(), &toml).unwrap();

    let result = interpret_with_inputs(program, inputs).unwrap();
    assert_eq!(result, Value::Unit);
}

/// Differential correctness: the interpreter's computed return value matches the expected output
/// Noir's corpus records in `Prover.toml`. `arithmetic_binary_operations` returns 10 (a u64),
/// so this verifies the actual value, not merely that interpretation didn't error.
#[cfg(not(feature = "goldilocks"))]
#[test]
fn interpreter_return_matches_recorded_expected() {
    let program_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../noir/test_programs/execution_success/arithmetic_binary_operations");
    if !program_dir.is_dir() {
        eprintln!("SKIPPED (vacuous pass): noir corpus not checked out at ../../noir");
        return;
    }
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path().join("pkg");
    copy_dir(&program_dir, &root);

    let project = Project::new(root.clone()).unwrap();
    let mut driver = Driver::new(project, false);
    driver.run_noir_compiler().unwrap();
    let program = driver.monomorphized_program();
    let toml = std::fs::read_to_string(root.join("Prover.toml")).unwrap();

    let inputs = inputs_from_prover_toml(program, driver.abi(), &toml).unwrap();
    let value = interpret_with_inputs(program, inputs).unwrap();
    let expected = expected_return_from_prover_toml(program, driver.abi(), &toml)
        .unwrap()
        .expect("this program records a return value");

    assert_eq!(
        value, expected,
        "interpreter output must match Noir's recorded return"
    );
}

/// Tier-1 Goldilocks validation: a `u64` program whose constant exceeds the Goldilocks modulus
/// (`big = p + 1`) compiles to a monomorphized AST *under Goldilocks* via the reachability-only
/// frontend (the bn254 crypto stdlib it never touches does not block it), and the interpreter
/// computes the correct **native** `u64` result — proving the Goldilocks frontend did not corrupt
/// the integer. The expected value is field-independent (bn254 carries the same `u64` natively),
/// so this assertion is itself the cross-field equivalence claim for this program; the corpus-wide
/// version is the `dump_corpus_outcomes` + `cross_field_diff` pair.
#[cfg(feature = "goldilocks")]
#[test]
fn validates_goldilocks_mono_ast_u64() {
    let project = Project::new(fixture("goldilocks_u64")).expect("project");
    let validated = compile_for_validation(&project)
        .expect("goldilocks frontend should produce a mono-AST for a stdlib-free u64 program");

    // main(x: u64) -> u64 = x * 2 + (p + 1). With x = 3: 6 + 18446744069414584322.
    let x = Value::Int(IntValue {
        signed: false,
        bits: 64,
        value: BigInt::from(3u64),
    });
    let result = interpret_with_inputs(&validated.program, vec![x]).expect("interpret");
    let expected = Value::Int(IntValue {
        signed: false,
        bits: 64,
        value: BigInt::from(18446744069414584328u64),
    });
    assert_eq!(
        result, expected,
        "Goldilocks mono-AST must carry p+1 exactly and compute the native u64 result"
    );
}

// ---------------------------------------------------------------------------
// Corpus survey (manual): run Noir's own `execution_success` programs through the
// interpreter and bucket the outcomes, to map the coverage frontier. Run with:
//   cargo test -p mavros-compiler --no-default-features --lib \
//       ast_interpreter::tests::survey -- --ignored --nocapture
// It is `#[ignore]`d because it compiles hundreds of external programs (slow) and is a
// reporting tool, not a pass/fail gate. Programs with a `Prover.toml` have their inputs fed in
// via `inputs_from_prover_toml`; multi-member workspaces are skipped.
// ---------------------------------------------------------------------------

fn copy_dir(src: &Path, dst: &Path) {
    std::fs::create_dir_all(dst).unwrap();
    for entry in std::fs::read_dir(src).unwrap() {
        let entry = entry.unwrap();
        let from = entry.path();
        let to = dst.join(entry.file_name());
        if from.is_dir() {
            copy_dir(&from, &to);
        } else {
            std::fs::copy(&from, &to).unwrap();
        }
    }
}

/// Outcome bucket for one program. `Unsupported` keeps the message so the report shows which
/// constructs block coverage; `AssertFailed` on a known-passing Noir program is a red flag
/// (interpreter bug or a semantics mismatch), not expected.
#[cfg(not(feature = "goldilocks"))]
fn classify(program_dir: &Path) -> String {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path().join("pkg");
    copy_dir(program_dir, &root);

    let outcome = panic::catch_unwind(AssertUnwindSafe(|| {
        let project = Project::new(root.clone()).map_err(|e| format!("project: {e}"))?;
        let mut driver = Driver::new(project, false);
        driver
            .run_noir_compiler()
            .map_err(|e| format!("compile: {e}"))?;
        let program = driver.monomorphized_program();
        let prover_src = std::fs::read_to_string(root.join("Prover.toml")).ok();

        let inputs = match &prover_src {
            Some(src) => inputs_from_prover_toml(program, driver.abi(), src)
                .map_err(|e| format!("inputs: {e}"))?,
            None => Vec::new(),
        };
        let value =
            interpret_with_inputs(program, inputs).map_err(|e| format!("interpret: {e}"))?;

        // Differential check: when the corpus records an expected return value, compare the
        // interpreter's computed output against it. This is the real correctness signal — a
        // genuine value mismatch (not just "did it error") surfaces here as MISMATCH.
        match &prover_src {
            Some(src) => match expected_return_from_prover_toml(program, driver.abi(), src)
                .map_err(|e| format!("expected: {e}"))?
            {
                Some(expected) if expected == value => Ok("pass: return verified".to_string()),
                Some(_) => Err("MISMATCH: return value disagrees with Noir".to_string()),
                None => Ok("pass: no recorded return".to_string()),
            },
            None => Ok("pass: no recorded return".to_string()),
        }
    }));

    match outcome {
        Err(payload) => {
            // A panic is a failure, not a neutral outcome: capture the message so interpreter
            // panics (e.g. an arithmetic helper that should have returned an error) are visible
            // in the report rather than lumped into one opaque bucket.
            let msg = payload
                .downcast_ref::<&str>()
                .map(|s| s.to_string())
                .or_else(|| payload.downcast_ref::<String>().cloned())
                .unwrap_or_default();
            let first_line = msg.lines().next().unwrap_or("").trim();
            if first_line.is_empty() {
                "PANIC".to_string()
            } else {
                format!("PANIC: {first_line}")
            }
        }
        Ok(Ok(bucket)) => bucket,
        Ok(Err(msg)) => {
            if msg.starts_with("MISMATCH") {
                msg
            } else if let Some(rest) = msg.strip_prefix("interpret: unsupported construct: ") {
                // Normalise to the construct kind (drop any trailing detail like a name).
                let kind = rest.split([':', '\'']).next().unwrap_or(rest).trim();
                format!("unsupported: {kind}")
            } else if msg.starts_with("interpret: assertion failed") {
                "ASSERT_FAILED (unexpected!)".to_string()
            } else if msg.starts_with("compile:") {
                "compile_error".to_string()
            } else if let Some(rest) = msg.strip_prefix("expected: ") {
                let kind = rest.split(['(', ':']).next().unwrap_or(rest).trim();
                format!("expected_return_error: {kind}")
            } else if let Some(rest) = msg.strip_prefix("inputs: ") {
                let kind = rest.split(['(', ':']).next().unwrap_or(rest).trim();
                format!("input_error: {kind}")
            } else if let Some(rest) = msg.strip_prefix("interpret: ") {
                let kind = rest.split([':', '(']).next().unwrap_or(&rest).trim();
                format!("interpret_error: {kind}")
            } else {
                msg
            }
        }
    }
}

/// Run one program through compile + interpret and capture a serializable, field-independent
/// outcome for the cross-field differential. Mirrors `classify` but returns structured data.
fn run_outcome(program_dir: &Path) -> DiffOutcome {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path().join("pkg");
    copy_dir(program_dir, &root);

    let result = panic::catch_unwind(AssertUnwindSafe(|| {
        let project = Project::new(root.clone()).map_err(|e| format!("project: {e}"))?;
        // Use the reachability-only validation frontend so the two field arms are extracted
        // identically and the Goldilocks arm is not blocked by the unreached bn254 crypto stdlib
        // (Tier-2).
        let validated = compile_for_validation(&project).map_err(|e| format!("compile: {e}"))?;
        let inputs = match std::fs::read_to_string(root.join("Prover.toml")) {
            Ok(src) => inputs_from_prover_toml(&validated.program, &validated.abi, &src)
                .map_err(|e| format!("inputs: {e}"))?,
            Err(_) => Vec::new(),
        };
        interpret_with_inputs(&validated.program, inputs).map_err(|e| format!("interpret: {e}"))
    }));

    match result {
        Ok(Ok(value)) => DiffOutcome::Returned(DiffValue::from_value(&value)),
        Ok(Err(reason)) => DiffOutcome::Errored(reason),
        Err(payload) => {
            let msg = payload
                .downcast_ref::<&str>()
                .map(|s| s.to_string())
                .or_else(|| payload.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "panic".to_string());
            DiffOutcome::Errored(format!(
                "panic: {}",
                msg.lines().next().unwrap_or("").trim()
            ))
        }
    }
}

fn corpus_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../noir/test_programs/execution_success")
}

/// The field this build targets, used to tag the dump file. The two fields are separate builds
/// (the field is selected at compile time), so each writes its own file for the diff to consume.
fn field_tag() -> &'static str {
    if cfg!(feature = "goldilocks") {
        "goldilocks"
    } else {
        "bn254"
    }
}

fn dump_path(tag: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(format!("../target/cross_field_{tag}.json"))
}

/// Cross-field differential, step 1: dump this build's per-program outcomes to a field-tagged
/// JSON file. Run once per field:
///   cargo test ... --lib ast_interpreter::tests::dump_corpus_outcomes -- --ignored
///   cargo test ... --features goldilocks --lib ast_interpreter::tests::dump_corpus_outcomes -- --ignored
/// (The Goldilocks dump is mostly `Errored` until the stdlib port lands — CRY-9.)
#[test]
#[ignore = "cross-field differential: dump this field's interpreter outcomes"]
fn dump_corpus_outcomes() {
    let corpus = corpus_dir();
    assert!(corpus.is_dir(), "corpus not found at {}", corpus.display());

    let mut outcomes: Vec<(String, DiffOutcome)> = Vec::new();
    for entry in std::fs::read_dir(&corpus).unwrap() {
        let dir = entry.unwrap().path();
        let manifest = dir.join("Nargo.toml");
        if !dir.is_dir() || !manifest.exists() {
            continue;
        }
        if std::fs::read_to_string(&manifest)
            .map(|s| s.contains("[workspace]"))
            .unwrap_or(false)
        {
            continue;
        }
        let name = dir.file_name().unwrap().to_string_lossy().into_owned();
        outcomes.push((name, run_outcome(&dir)));
    }

    // Round-trip through JSON to validate serialization (plumbing check, no second field needed).
    let json = serde_json::to_string_pretty(&outcomes).unwrap();
    let restored: Vec<(String, DiffOutcome)> = serde_json::from_str(&json).unwrap();
    assert_eq!(restored, outcomes, "dump must round-trip through JSON");

    let path = dump_path(field_tag());
    std::fs::write(&path, &json).unwrap();
    let returned = outcomes
        .iter()
        .filter(|(_, o)| matches!(o, DiffOutcome::Returned(_)))
        .count();
    println!(
        "wrote {} outcomes ({returned} returned a value) for field '{}' to {}",
        outcomes.len(),
        field_tag(),
        path.display()
    );
}

/// Cross-field differential, step 2: diff the two field dumps. Integer/struct values must match;
/// `Field` values may differ. A divergence on an integer is the corruption the Goldilocks work
/// risks. Requires both dumps to exist (run `dump_corpus_outcomes` under each field first).
#[test]
#[ignore = "cross-field differential: diff the bn254 and goldilocks dumps"]
fn cross_field_diff() {
    use std::collections::BTreeMap;

    let load = |tag: &str| -> Option<BTreeMap<String, DiffOutcome>> {
        let text = std::fs::read_to_string(dump_path(tag)).ok()?;
        Some(
            serde_json::from_str::<Vec<(String, DiffOutcome)>>(&text)
                .unwrap()
                .into_iter()
                .collect(),
        )
    };

    let (Some(bn254), Some(goldilocks)) = (load("bn254"), load("goldilocks")) else {
        panic!(
            "missing a dump; run dump_corpus_outcomes under each field first \
             (the goldilocks dump needs the CRY-9 stdlib port to produce real values)"
        );
    };

    let mut compared = 0;
    let mut divergences: Vec<String> = Vec::new();
    for (name, bn) in &bn254 {
        let Some(gl) = goldilocks.get(name) else {
            continue;
        };
        compared += 1;
        if let Err(reason) = outcomes_equivalent(bn, gl) {
            divergences.push(format!("{name}: {reason}"));
        }
    }

    println!("compared {compared} programs present in both fields");
    for d in &divergences {
        println!("  DIVERGENCE: {d}");
    }
    assert!(
        divergences.is_empty(),
        "{} cross-field divergence(s) found",
        divergences.len()
    );
}

#[cfg(not(feature = "goldilocks"))]
#[test]
#[ignore = "manual coverage survey over Noir's execution_success corpus"]
fn survey_execution_success_corpus() {
    use std::collections::BTreeMap;

    let corpus = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../noir/test_programs/execution_success");
    assert!(corpus.is_dir(), "corpus not found at {}", corpus.display());

    let mut buckets: BTreeMap<String, Vec<String>> = BTreeMap::new();
    let mut total = 0;
    for entry in std::fs::read_dir(&corpus).unwrap() {
        let dir = entry.unwrap().path();
        let manifest = dir.join("Nargo.toml");
        if !dir.is_dir() || !manifest.exists() {
            continue;
        }
        // Workspaces have multiple members; the driver picks one arbitrarily, which doesn't match
        // the workspace's default-member `main`, so skip them rather than report a false failure.
        if std::fs::read_to_string(&manifest)
            .map(|s| s.contains("[workspace]"))
            .unwrap_or(false)
        {
            continue;
        }
        let name = dir.file_name().unwrap().to_string_lossy().into_owned();
        total += 1;
        buckets.entry(classify(&dir)).or_default().push(name);
    }

    println!("\n=== interpreter coverage over {total} execution_success programs ===");
    for (bucket, names) in &buckets {
        println!("\n[{}]  {}", names.len(), bucket);
        for name in names {
            println!("    {name}");
        }
    }
}
