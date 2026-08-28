//! Keeps the generated Noir corpus on disk in step with the model that renders it.
//!
//! The corpus is _committed_, not built: the test runner discovers tests by scanning `noir_tests/`
//! and `noir_failure_tests/`, and `STATUS.md` is a table of the result. So the files have to exist
//! in the tree, and this test is what stops them drifting from [`mavros_int_semantics::corpus`].
//!
//! Run it normally and it **verifies**. Run it with `MAVROS_BLESS=1` and it **writes**, following
//! the same override convention the `Makefile` uses for `STATUS`:
//!
//! ```text
//! MAVROS_BLESS=1 cargo test -p mavros-int-semantics --test generated_corpus
//! ```
//!
//! A semantic change therefore arrives as a reviewable diff of Noir programs — which of them
//! changed answer, and to what — rather than as a red test with no context.

use mavros_int_semantics::corpus::{self, GeneratedTest};

use std::{
    collections::BTreeSet,
    fs,
    path::{Path, PathBuf},
};

/// Every generated directory carries this prefix, which is what makes it safe to delete a stale one.
const GENERATED_PREFIX: &str = "int_semantics_";

/// The corpus directory a test belongs in, by whether it must be rejected.
fn corpus_dir(root: &Path, test: &GeneratedTest) -> PathBuf {
    let dir = if test.expect_failure {
        "noir_failure_tests"
    } else {
        "noir_tests"
    };
    root.join(dir).join(&test.name)
}

/// The workspace root, which is the crate's parent.
fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("ICE: the crate directory has no parent")
        .to_path_buf()
}

/// The three files a Noir package needs, as `(relative path, contents)`.
fn files(test: &GeneratedTest) -> Vec<(PathBuf, String)> {
    vec![
        (PathBuf::from("Nargo.toml"), test.nargo_toml()),
        (PathBuf::from("Prover.toml"), test.prover_toml.clone()),
        (PathBuf::from("src").join("main.nr"), test.main_nr.clone()),
    ]
}

#[test]
fn the_generated_corpus_matches_the_model() {
    let root = repo_root();
    let tests: Vec<GeneratedTest> = corpus::accepting_tests()
        .into_iter()
        .chain(corpus::rejecting_tests())
        .collect();
    assert!(!tests.is_empty(), "the generator produced no tests at all");

    if std::env::var_os("MAVROS_BLESS").is_some() {
        bless(&root, &tests);
        return;
    }

    let mut wrong = Vec::new();
    for test in &tests {
        let dir = corpus_dir(&root, test);
        for (relative, expected) in files(test) {
            let path = dir.join(&relative);
            match fs::read_to_string(&path) {
                Ok(actual) if actual == expected => {}
                Ok(_) => wrong.push(format!("{} differs", path.display())),
                Err(_) => wrong.push(format!("{} is missing", path.display())),
            }
        }
    }

    for stale in stale_directories(&root, &tests) {
        wrong.push(format!(
            "{} is generated but no longer produced",
            stale.display()
        ));
    }

    assert!(
        wrong.is_empty(),
        "the generated corpus is out of date:\n  {}\n\nRegenerate it with \
         `MAVROS_BLESS=1 cargo test -p mavros-int-semantics --test generated_corpus`, and read the \
         diff: a program whose expected answer moved is a program whose meaning moved.",
        wrong.join("\n  ")
    );
}

/// Write the corpus, and delete any generated directory the model no longer produces.
///
/// The deletion is bounded to directories whose name carries [`GENERATED_PREFIX`], which nothing
/// hand-written uses. Without it a narrowed model would leave orphan tests behind that no longer
/// correspond to anything and that nothing would ever regenerate.
fn bless(root: &Path, tests: &[GeneratedTest]) {
    for stale in stale_directories(root, tests) {
        fs::remove_dir_all(&stale).expect("could not remove a stale generated test");
        println!("removed {}", stale.display());
    }

    for test in tests {
        let dir = corpus_dir(root, test);
        for (relative, contents) in files(test) {
            let path = dir.join(&relative);
            fs::create_dir_all(path.parent().expect("ICE: a file with no directory"))
                .expect("could not create a generated test directory");
            let unchanged = fs::read_to_string(&path).is_ok_and(|old| old == contents);
            if !unchanged {
                fs::write(&path, contents).expect("could not write a generated test file");
                println!("wrote {}", path.display());
            }
        }
    }
}

/// Generated directories present on disk that the model does not produce.
fn stale_directories(root: &Path, tests: &[GeneratedTest]) -> Vec<PathBuf> {
    let expected: BTreeSet<PathBuf> = tests.iter().map(|t| corpus_dir(root, t)).collect();

    let mut stale = Vec::new();
    for corpus in ["noir_tests", "noir_failure_tests"] {
        let Ok(entries) = fs::read_dir(root.join(corpus)) else { continue };
        for entry in entries.flatten() {
            let path = entry.path();
            let is_generated = path
                .file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.starts_with(GENERATED_PREFIX));
            if is_generated && path.is_dir() && !expected.contains(&path) {
                stale.push(path);
            }
        }
    }
    stale.sort();
    stale
}
