use mavros_compiler::{
    Project,
    driver::{Driver, Error},
};
use tempfile::TempDir;

/// Create an isolated Noir binary package from a complete source program.
pub fn noir_project(source: &str) -> TempDir {
    let dir = tempfile::tempdir().unwrap();
    std::fs::create_dir(dir.path().join("src")).unwrap();
    std::fs::write(
        dir.path().join("Nargo.toml"),
        "[package]\nname = 'test_program'\ntype = 'bin'\nauthors = []\n",
    )
    .unwrap();
    std::fs::write(dir.path().join("src/main.nr"), source).unwrap();
    dir
}

/// The frontend refuses `source` before Mavros lowering, with a located diagnostic that says
/// `expected` in its message, one of its secondaries or one of its notes.
///
/// Matching the text is what makes a refusal test meaningful: a bare `is_err()` is satisfied by a
/// typo in the program as readily as by the rule under test.
#[allow(dead_code)]
pub fn assert_frontend_refused(source: &str, expected: &str) {
    let dir = noir_project(source);
    let mut driver = Driver::new(Project::new(dir.path().to_path_buf()).unwrap(), false);
    let Err(Error::NoirCompilerError(diagnostics)) = driver.run_noir_compiler() else {
        panic!("expected a frontend diagnostic for {source}");
    };
    assert!(
        diagnostics.iter().any(|diagnostic| {
            let located = !diagnostic.secondaries.is_empty();
            let says = std::iter::once(diagnostic.message.as_str())
                .chain(
                    diagnostic
                        .secondaries
                        .iter()
                        .map(|label| label.message.as_str()),
                )
                .chain(diagnostic.notes.iter().map(String::as_str))
                .any(|text| text.contains(expected));
            located && says
        }),
        "missing located diagnostic {expected:?} for {source}: {diagnostics:?}"
    );
}
