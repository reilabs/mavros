//! What each corpus test is expected to do, and the overrides that say so.

use super::*;

pub(crate) const DEFAULT_IGNORED_TESTS: &[&str] = &[
    // Passing an intrinsic such as `std::as_witness` as a function value is unsupported.
    "regression_10156",
    // `func_1` recurses without ever decrementing `ctx_limit`, so it never terminates. The witgen
    // VM has no recursion/step/memory guard (frames are heap-allocated per call in `Frame::push`),
    // so running it grows memory without bound. Depending on the compiled circuit shape it either
    // SIGSEGVs fast (alloc returns null → unchecked deref) or thrashes for hours before the OOM
    // killer fires — the latter froze CI for 6h. Skip until the VM gains an execution budget.
    "brillig_mem_layout_regression",
];

/// Expectations that hold on every field, for fixtures whose directory gives the wrong one. A local
/// fixture the compiler refuses with a known text takes an `expected_error.txt` instead, and an
/// expectation that holds on one field goes in [`FIELD_EXPECTATION_OVERRIDES`]. A
/// [`TestExpectation::CompileRejection`] text is matched against the error's diagnostics.
pub(crate) const TEST_EXPECTATION_OVERRIDES: &[(&str, TestExpectation)] = &[
    // This upstream fixture has mixed outcomes: a asserts 1 == 2, while b asserts 1 != 0.
    (
        "noir/test_programs/execution_failure/workspace_fail/crates/b",
        TestExpectation::ExecutionSuccess,
    ),
    // These fixtures deliberately exercise host-side validation rather than a VM trap.
    (
        "noir_failure_tests/prover_file_missing",
        TestExpectation::InputFailure,
    ),
    (
        "noir_failure_tests/prover_missing_param",
        TestExpectation::InputFailure,
    ),
    (
        "noir_failure_tests/return_wrong_value",
        TestExpectation::ReturnMismatch,
    ),
];

/// The fixtures using the data bus, which the frontend refuses on every function with
/// [`DATA_BUS_REFUSAL`], whatever the field.
pub(crate) const DATA_BUS_FIXTURES: &[&str] = &[
    "noir/test_programs/execution_success/databus",
    "noir/test_programs/execution_success/databus_composite_calldata",
    "noir/test_programs/execution_success/databus_two_calldata",
    "noir/test_programs/execution_success/databus_two_calldata_simple",
    "noir/test_programs/execution_success/regression_7143",
    "noir/test_programs/execution_success/regression_7612",
    "noir/test_programs/execution_success/regression_claude_1124",
    "noir/test_programs/execution_failure/regression_databus_out_of_bounds",
];

/// The frontend's note refusing the data bus.
pub(crate) const DATA_BUS_REFUSAL: &str = "the data bus is not supported";

/// The frontend's refusal of a cast to `Field` from an integer type wider than a field element
/// holds injectively.
pub(crate) const WIDE_FIELD_CAST_REFUSAL: &str = "cannot be cast to Field";

/// Mavros' refusal of a witnessed decomposition into more bytes than a field element has.
pub(crate) const WIDE_DECOMPOSITION_REFUSAL: &str = "bytes a field element occupies";

/// Every field-agnostic override: [`TEST_EXPECTATION_OVERRIDES`] and the data-bus fixtures.
pub(crate) fn test_expectation_overrides() -> impl Iterator<Item = (&'static str, TestExpectation)>
{
    TEST_EXPECTATION_OVERRIDES.iter().copied().chain(
        DATA_BUS_FIXTURES
            .iter()
            .map(|name| (*name, TestExpectation::CompileRejection(DATA_BUS_REFUSAL))),
    )
}

/// Expectations that hold on one field only. Each overrides the entry's expectation on its field.
pub(crate) const FIELD_EXPECTATION_OVERRIDES: &[(&str, FieldId, TestExpectation)] = &[
    // `(0 - 1) as u64` depends on the field, and the fixture's `return` holds bn254's value.
    (
        "noir/test_programs/execution_success/cast_regression_7776",
        FieldId::Goldilocks,
        TestExpectation::ReturnMismatch,
    ),
    // These cast a `u64` or `u128` to a `Field`, which Goldilocks cannot hold injectively.
    (
        "noir/test_programs/execution_success/regression_brillig_const_fold_self_dedup",
        FieldId::Goldilocks,
        TestExpectation::CompileRejection(WIDE_FIELD_CAST_REFUSAL),
    ),
    (
        "noir/test_programs/execution_success/regression_struct_array_conditional",
        FieldId::Goldilocks,
        TestExpectation::CompileRejection(WIDE_FIELD_CAST_REFUSAL),
    ),
    (
        "noir/test_programs/execution_success/unsafe_range_constraint",
        FieldId::Goldilocks,
        TestExpectation::CompileRejection(WIDE_FIELD_CAST_REFUSAL),
    ),
    // These decompose a `Field` into 32 bytes, where a Goldilocks element has 8. A witnessed
    // decomposition is refused at compile time; in unconstrained code the standard library's
    // assertion traps in witness generation.
    (
        "noir/test_programs/execution_success/regression_9208",
        FieldId::Goldilocks,
        TestExpectation::CompileRejection(WIDE_DECOMPOSITION_REFUSAL),
    ),
    (
        "noir/test_programs/execution_success/regression_9303",
        FieldId::Goldilocks,
        TestExpectation::WitgenTrap,
    ),
];

/// The fixture file naming the error the compiler refuses the fixture with, which makes the
/// fixture's expectation a [`TestExpectation::CompileRejection`] of that text.
pub(crate) const EXPECTED_ERROR_FILE: &str = "expected_error.txt";

/// The text of the [`EXPECTED_ERROR_FILE`] in the package at `root`, if it has a nonempty one. It
/// lives as long as the run, as a [`TestExpectation`] holds it.
pub(crate) fn expected_error_of(root: &Path) -> Option<&'static str> {
    let text = fs::read_to_string(root.join(EXPECTED_ERROR_FILE)).ok()?;
    let text = text.trim();
    (!text.is_empty()).then(|| &*Box::leak(text.to_string().into_boxed_str()))
}

/// Whether `error` says `text`: in one of its diagnostics, where it has them, or in its rendering
/// otherwise. A frontend rendering also holds warnings and source lines, either of which could
/// contain the text by chance, so an error with diagnostics is held to them.
pub(crate) fn error_says(error: &DriverError, text: &str) -> bool {
    match error {
        DriverError::NoirCompilerError { diagnostics, .. } => diagnostics
            .iter()
            .filter(|d| d.is_error())
            .any(|diagnostic| {
                std::iter::once(diagnostic.message.as_str())
                    .chain(
                        diagnostic
                            .secondaries
                            .iter()
                            .map(|label| label.message.as_str()),
                    )
                    .chain(diagnostic.notes.iter().map(String::as_str))
                    .any(|said| said.contains(text))
            }),
        DriverError::LoweringRefused(diagnostics)
        | DriverError::UnsatisfiableProgram(diagnostics) => {
            diagnostics.iter().any(|diagnostic| diagnostic.says(text))
        }
        other => other.to_string().contains(text),
    }
}

/// Whether the frontend's refusal `error` of a fixture on `field` is the one the fixture expects.
/// A fixture with an `expected` text ([`TestExpectation::CompileRejection`]) is held to that text
/// on every field. On any other fixture a frontend refusal counts on bn254 only: off bn254 it can
/// mean a missing builtin.
pub(crate) fn frontend_rejection_expected(
    field: FieldId,
    error: &DriverError,
    expected: Option<&str>,
) -> bool {
    match expected {
        Some(expected) => error_says(error, expected),
        None => field == FieldId::Bn254,
    }
}

/// Whether Mavros' compile-time refusal `error` of a fixture is the one the fixture expects. With
/// an `expected` text it is held to that text. Without one, a statically false assertion is an
/// expected failure, while a lowering refusal means a program Mavros does not support.
pub(crate) fn mavros_rejection_expected(error: &DriverError, expected: Option<&str>) -> bool {
    match expected {
        Some(expected) => error_says(error, expected),
        None => !matches!(error, DriverError::LoweringRefused(_)),
    }
}

/// The expectation for `entry` on `field`.
pub(crate) fn expectation_on(entry: &TestEntry, field: FieldId) -> TestExpectation {
    expectation_in(FIELD_EXPECTATION_OVERRIDES, entry, field)
}

/// The expectation for `entry` on `field`, where `overrides` gives the field-specific ones.
pub(crate) fn expectation_in(
    overrides: &[(&str, FieldId, TestExpectation)],
    entry: &TestEntry,
    field: FieldId,
) -> TestExpectation {
    overrides
        .iter()
        .find(|(name, on, _)| *name == entry.display_name && *on == field)
        .map_or(entry.expectation, |(_, _, expectation)| *expectation)
}

/// Every override names a discovered test, and none is given twice or given to a fixture that
/// names its expected error itself. An override naming one of Noir's programs is checked only
/// when `noir_programs_found`, since every one would look stale without them.
///
/// # Panics
///
/// On an override naming no test in `entries`, as a renamed or deleted fixture leaves, on a name
/// given twice in the same table, or on an override of a fixture with an [`EXPECTED_ERROR_FILE`],
/// which the override would silently replace.
pub(crate) fn validate_overrides(
    entries: &[TestEntry],
    overrides: &[(&str, TestExpectation)],
    field_overrides: &[(&str, FieldId, TestExpectation)],
    noir_programs_found: bool,
) {
    let names = overrides.iter().map(|(name, _)| (*name, None)).chain(
        field_overrides
            .iter()
            .map(|(name, field, _)| (*name, Some(*field))),
    );
    let mut seen = HashSet::default();
    for (name, field) in names {
        assert!(
            seen.insert((name, field)),
            "`{name}` has two expectation overrides{}",
            field.map_or(String::new(), |field| format!(" on {}", field.name()))
        );
        if !noir_programs_found && name.starts_with("noir/") {
            continue;
        }
        let entry = entries
            .iter()
            .find(|entry| entry.display_name == name)
            .unwrap_or_else(|| panic!("the expectation override for `{name}` names no test"));
        assert!(
            !matches!(entry.expectation, TestExpectation::CompileRejection(_)),
            "`{name}` has an expectation override and an {EXPECTED_ERROR_FILE}"
        );
    }
}

/// Applies the field-agnostic overrides; [`expectation_on`] applies the per-field ones.
pub(crate) fn apply_test_expectation_overrides(entries: &mut [TestEntry]) {
    for entry in entries {
        if let Some((_, expectation)) =
            test_expectation_overrides().find(|(name, _)| *name == entry.display_name)
        {
            entry.expectation = expectation;
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum TestExpectation {
    ExecutionSuccess,
    ExecutionFailure,
    /// Witness generation traps, cleanly: a crash of the VM or of the host is still a crash. For a
    /// fixture expected to succeed on one field that traps on another, where
    /// [`TestExpectation::ExecutionFailure`] would also take a crash as its failure.
    WitgenTrap,
    InputFailure,
    ReturnMismatch,
    /// The compiler refuses the program before R1CS generation (the `COMPILED` stage), in the
    /// frontend or in a Mavros refusal, with an error saying the text.
    CompileRejection(&'static str),
}
