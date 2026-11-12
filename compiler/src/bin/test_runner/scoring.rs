//! A child's stage results, scored against what its test is expected to do.

use super::*;

/// The step keys in display order.
pub(crate) const STEP_KEYS: &[&str] = &[
    "COMPILED",
    "R1CS",
    "COMPILE",
    "WITGEN_RUN",
    "WITGEN_CORRECT",
    "WITGEN_NOLEAK",
    "AD_RUN",
    "AD_CORRECT",
    "AD_NOLEAK",
    "WASM_COMPILE",
    "WITGEN_WASM_RUN",
    "WITGEN_WASM_CORRECT",
    "WITGEN_WASM_NOLEAK",
    "AD_WASM_RUN",
    "AD_WASM_CORRECT",
    "AD_WASM_NOLEAK",
];

pub(crate) struct TestResult {
    pub(crate) name: String,
    pub(crate) field: FieldId,
    pub(crate) steps: HashMap<String, Status>,
    pub(crate) rows: Option<usize>,
    pub(crate) cols: Option<usize>,
    /// The report-only R1CS compaction breakdown captured from the child's `INFO:R1CS_ANALYSIS`
    /// line, printed to stdout under `--analyze-r1cs`. `None` on a normal run (the analysis did not
    /// run) — it never feeds STATUS.md.
    pub(crate) analysis_stats: Option<String>,
    pub(crate) binary_bytes: Option<usize>,
    pub(crate) vm_debug_bytes: Option<usize>,
    pub(crate) wasm_bytes: Option<usize>,
    pub(crate) wasm_debug_bytes: Option<usize>,
}

/// Determined purely from child output:
/// - `started && ended ok` → Pass
/// - `started && ended reject` → Rejected (program proven unsatisfiable — a legit failing assert)
/// - `started && ended fail` → Fail
/// - `started && no end` → Crash
/// - `never started` → Skip
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Status {
    Pass,
    Fail,
    Crash,
    Skip,
    NotApplicable,
    /// The compiler cleanly rejected the program (for example, a dynamic `assert_constant` or an
    /// assertion that can never hold). For an execution_failure test this is the expected outcome;
    /// for an execution_success test it is a genuine failure.
    Rejected,
}

impl Status {
    pub(crate) fn emoji(self) -> &'static str {
        match self {
            Status::Pass => "✅",
            Status::Fail => "❌",
            Status::Crash => "💥",
            Status::Skip => "➖",
            Status::NotApplicable => "N/A",
            // Only surfaces in the table if it escapes `expected_step_status` unmapped, which
            // would itself be a bug; render it distinctly rather than hide it.
            Status::Rejected => "🚫",
        }
    }
}

pub(crate) fn parse_child_output(
    name: &str,
    expectation: TestExpectation,
    lines: &[String],
) -> TestResult {
    let mut started = HashMap::<String, bool>::default();
    let mut ended = HashMap::<String, bool>::default();
    let mut rejected = HashMap::<String, bool>::default();
    let mut errors = HashMap::<String, String>::default();
    let mut rows = None;
    let mut cols = None;
    let mut analysis_stats = None;
    let mut binary_bytes = None;
    let mut vm_debug_bytes = None;
    let mut wasm_bytes = None;
    let mut wasm_debug_bytes = None;

    for line in lines {
        // Opt-in R1CS compaction breakdown (only emitted under `--analyze-r1cs`). Captured whole so
        // the parent can echo it to stdout; it never feeds STATUS.md. Handled before the `:`-split
        // dispatch because its free-form payload is not a `START:`/`END:` record.
        if let Some(rest) = line.strip_prefix("INFO:R1CS_ANALYSIS:") {
            analysis_stats = Some(rest.to_string());
            continue;
        }
        let parts: Vec<&str> = line.split(':').collect();
        match parts.as_slice() {
            ["START", key] => {
                started.insert(key.to_string(), true);
            }
            ["END", key, "ok", ..] => {
                ended.insert(key.to_string(), true);
                if *key == "R1CS" && parts.len() >= 5 {
                    rows = parts[3].parse().ok();
                    cols = parts[4].parse().ok();
                }
                if *key == "COMPILE" && parts.len() >= 4 {
                    binary_bytes = parts[3].parse().ok();
                    vm_debug_bytes = parts.get(4).and_then(|value| value.parse().ok());
                }
                if *key == "WASM_COMPILE" && parts.len() >= 4 {
                    wasm_bytes = parts[3].parse().ok();
                    wasm_debug_bytes = parts.get(4).and_then(|value| value.parse().ok());
                }
            }
            ["END", key, "reject"] => {
                rejected.insert(key.to_string(), true);
            }
            [
                "END",
                key,
                kind @ ("error" | "input-error" | "return-mismatch"),
            ] => {
                errors.insert(key.to_string(), kind.to_string());
            }
            ["END", key, "fail"] => {
                ended.insert(key.to_string(), false);
            }
            _ => {}
        }
    }

    // A program can be cleanly rejected at the first stage where it is proven it will never
    // execute — compilation (COMPILED) or R1CS generation. That rejection preempts every
    // later stage, so we track which stage it happened at (by position in the pipeline) and
    // report all later stages N/A rather than ➖ Skip — otherwise the success-rate denominator
    // (which counts every non-N/A cell) would count them against an otherwise handled test.
    let reject_idx = STEP_KEYS.iter().position(|key| rejected.contains_key(*key));

    let steps = STEP_KEYS
        .iter()
        .enumerate()
        .map(|(idx, &key)| {
            let raw_status = raw_step_status(key, &started, &ended, &rejected);
            if let Some(kind) = errors.get(key) {
                // Only fixtures specifically testing this host rejection may accept it.
                let expected = matches!(key, "WITGEN_RUN" | "WITGEN_WASM_RUN")
                    && matches!(
                        (expectation, kind.as_str()),
                        (TestExpectation::InputFailure, "input-error")
                            | (TestExpectation::ReturnMismatch, "return-mismatch")
                    );
                return (
                    key.to_string(),
                    if expected { Status::Pass } else { Status::Fail },
                );
            }
            (
                key.to_string(),
                expected_step_status(idx, key, raw_status, expectation, reject_idx),
            )
        })
        .collect();

    TestResult {
        name: name.to_string(),
        field: FieldId::Bn254,
        steps,
        rows,
        cols,
        analysis_stats,
        binary_bytes,
        vm_debug_bytes,
        wasm_bytes,
        wasm_debug_bytes,
    }
}

pub(crate) fn raw_step_status(
    key: &str,
    started: &HashMap<String, bool>,
    ended: &HashMap<String, bool>,
    rejected: &HashMap<String, bool>,
) -> Status {
    if rejected.contains_key(key) {
        Status::Rejected
    } else if let Some(&ok) = ended.get(key) {
        if ok { Status::Pass } else { Status::Fail }
    } else if started.contains_key(key) {
        Status::Crash
    } else {
        Status::Skip
    }
}

pub(crate) fn expected_step_status(
    idx: usize,
    key: &str,
    raw_status: Status,
    expectation: TestExpectation,
    reject_idx: Option<usize>,
) -> Status {
    match expectation {
        // Only the expected refusal at `COMPILED` passes, and it preempts every later stage. A
        // program that compiles has lost the refusal, so `COMPILED` fails, and the stages it then
        // ran keep their raw status.
        TestExpectation::CompileRejection(_) => {
            let refused = reject_idx.is_some_and(|reject| STEP_KEYS[reject] == "COMPILED");
            match (refused, key) {
                (true, "COMPILED") => Status::Pass,
                (true, _) => Status::NotApplicable,
                (false, "COMPILED") => match raw_status {
                    Status::Pass | Status::Rejected => Status::Fail,
                    other => other,
                },
                (false, _) => match raw_status {
                    Status::Rejected => Status::Fail,
                    other => other,
                },
            }
        }
        // A program that is supposed to execute must not be rejected as unsatisfiable; if it
        // was, that is a real failure. Every other stage uses its raw status unchanged.
        TestExpectation::ExecutionSuccess => match raw_status {
            Status::Rejected => Status::Fail,
            other => other,
        },
        // A program proven to never execute is rejected cleanly at the first stage that can
        // prove it: compilation (`COMPILED`, e.g. a comptime overflow or dynamic
        // `assert_constant`) or R1CS generation (`R1CS`, a statically-unsatisfiable assertion —
        // `Err(UnsatisfiableProgram)`). The
        // child reports that with a `reject` marker. For an expected-failure test that
        // rejection is exactly the success, so: stages before it keep their real status (they
        // genuinely ran), the rejecting stage itself reads ✅, and every later stage is
        // preempted and reads N/A — so it neither looks like a failure nor dilutes the success
        // rate. A plain `Fail` or a `Crash` has no `reject` marker, so a real regression is
        // never masked by the expected-failure flag.
        TestExpectation::ExecutionFailure if reject_idx.is_some() => {
            match idx.cmp(&reject_idx.unwrap()) {
                std::cmp::Ordering::Less => raw_status,
                std::cmp::Ordering::Equal => Status::Pass,
                std::cmp::Ordering::Greater => Status::NotApplicable,
            }
        }
        TestExpectation::ExecutionFailure
        | TestExpectation::WitgenTrap
        | TestExpectation::InputFailure
        | TestExpectation::ReturnMismatch => match key {
            "WITGEN_RUN" | "WITGEN_WASM_RUN" => match (expectation, raw_status) {
                (TestExpectation::ExecutionFailure, Status::Fail | Status::Crash)
                | (TestExpectation::WitgenTrap, Status::Fail) => Status::Pass,
                (_, Status::Pass | Status::Rejected) => Status::Fail,
                (_, other) => other,
            },
            // Correctness/leak checks need a witness, and the native AD
            // pipeline only feeds proving; neither is meaningful for a program
            // whose witgen is expected to trap. WASM AD still runs as a
            // compile/runtime sanity check and uses raw status.
            "WITGEN_CORRECT"
            | "WITGEN_NOLEAK"
            | "WITGEN_WASM_CORRECT"
            | "WITGEN_WASM_NOLEAK"
            | "AD_RUN"
            | "AD_CORRECT"
            | "AD_NOLEAK" => Status::NotApplicable,
            _ => match raw_status {
                Status::Rejected => Status::Fail,
                other => other,
            },
        },
    }
}

/// The result of a test the run skips: every stage ➖.
pub(crate) fn ignored_test_result(name: &str) -> TestResult {
    let steps = STEP_KEYS
        .iter()
        .map(|&key| (key.to_string(), Status::Skip))
        .collect();

    TestResult {
        name: name.to_string(),
        field: FieldId::Bn254,
        steps,
        rows: None,
        cols: None,
        analysis_stats: None,
        binary_bytes: None,
        vm_debug_bytes: None,
        wasm_bytes: None,
        wasm_debug_bytes: None,
    }
}
