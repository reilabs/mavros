//! Holds the tree to [`mavros_int_semantics::register::EVALUATORS`], the register of everything
//! that must conform to this crate.
//!
//! The register is a hand-maintained list, and these tests are what make failing to maintain it
//! noisy. Four things are checked:
//!
//! 1. every registered file still exists;
//! 2. every registered conformance test still exists, by name, in the file that claims it;
//! 3. `docs/int-semantics.md` has exactly the registered tags as sections, in the register's order;
//! 4. no **unregistered** file defines integer arithmetic of its own (on a best-effort basis).

use mavros_int_semantics::register::EVALUATORS;

use std::{
    collections::BTreeSet,
    fs,
    path::{Path, PathBuf},
};

/// Wrapping arithmetic that is not an integer operation of the language being compiled.
///
/// Every entry is a host-side computation that happens to want modular arithmetic — a counter, a
/// PRNG, an address. Keyed by the line's own text rather than by line number, so editing the line
/// brings it back here to be justified again, and by file as well so that a second use in the same
/// file is not waved through by the first.
const NOT_INTEGER_SEMANTICS: &[(&str, &str, &str)] = &[
    (
        "opcode-gen/src/lib.rs",
        "read_cycles().wrapping_sub(__prof_start)",
        "a cycle-counter delta, which wraps at the counter's width and not at any Noir type's",
    ),
    (
        "mavros-artifacts/src/lib.rs",
        "self.sampler_state.wrapping_mul(",
        "an LCG step; wrapping is the generator, not an overflow",
    ),
    (
        "compiler/src/bin/test_runner.rs",
        "host_witness_base.wrapping_add(mults_off as usize)",
        "host pointer arithmetic over a witness buffer",
    ),
];

/// The signature check 4 looks for.
///
/// The leading dot matters: it is a method _call_ that evaluates something, where a bare
/// `wrapping_` also matches a test named after the behavior it pins.
const HAND_ROLLED: &[&str] = &[".wrapping_", ".overflowing_"];

/// The workspace root, which is this crate's parent.
fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("ICE: the crate directory has no parent")
        .to_path_buf()
}

/// The crate roots check 4 sweeps: every workspace member's `src`, less this crate's own.
fn swept_roots(root: &Path) -> Vec<PathBuf> {
    let manifest =
        fs::read_to_string(root.join("Cargo.toml")).expect("the workspace manifest is missing");
    let members = manifest
        .split_once("members = [")
        .expect("the workspace manifest declares no members")
        .1
        .split_once(']')
        .expect("the workspace manifest's member list is unterminated")
        .0;

    let mut roots = Vec::new();
    for member in members.split(',') {
        // Each entry is a quoted path; anything without a quote is the trailing whitespace.
        let Some(name) = member.split('"').nth(1) else {
            continue;
        };
        let src = root.join(name).join("src");

        // This crate is the model, and so is the one place licensed to define an integer operation
        // without registering it. Compared by path rather than by name, which would be asserting
        // that a package is named after its directory.
        if src == Path::new(env!("CARGO_MANIFEST_DIR")).join("src") {
            continue;
        }
        assert!(
            src.is_dir(),
            "workspace member `{name}` has no `src` directory to sweep"
        );
        roots.push(src);
    }

    assert!(
        roots.len() > 3,
        "only {} crates to sweep, which cannot be the whole workspace",
        roots.len()
    );
    roots
}

/// Every `.rs` file under `dir`, sorted, so a failure lists the same files in the same order twice
/// running.
fn rust_files(dir: &Path) -> Vec<PathBuf> {
    let mut found = Vec::new();
    let Ok(entries) = fs::read_dir(dir) else {
        return found;
    };

    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            found.extend(rust_files(&path));
        } else if path.extension().is_some_and(|e| e == "rs") {
            found.push(path);
        }
    }

    found.sort();
    found
}

/// A path as the register spells it.
fn relative(root: &Path, path: &Path) -> String {
    path.strip_prefix(root)
        .expect("ICE: a swept file outside the workspace")
        .to_string_lossy()
        .replace('\\', "/")
}

/// Whether `line` is nothing but a comment.
///
/// Check 4 skips these: prose that mentions `wrapping_add` is describing an evaluation, not
/// performing one, and several of the doc comments around these evaluators do exactly that. A
/// trailing comment on a line of code is not stripped, which only ever makes the check stricter.
fn is_comment(line: &str) -> bool {
    let trimmed = line.trim_start();
    if trimmed.starts_with("//") || trimmed.starts_with("/*") {
        return true;
    }

    trimmed == "*" || trimmed.starts_with("* ") || trimmed.starts_with("*/")
}

#[test]
fn every_registered_evaluator_still_exists() {
    let root = repo_root();

    let mut wrong = Vec::new();
    for evaluator in EVALUATORS {
        assert!(
            !evaluator.files.is_empty(),
            "`{}` is registered with no file at all",
            evaluator.tag
        );
        for file in evaluator.files {
            if !root.join(file).is_file() {
                wrong.push(format!("`{}`: {file} does not exist", evaluator.tag));
            }
        }
    }

    assert!(
        wrong.is_empty(),
        "a registered evaluator names a file that is not there:\n  {}\n\nThe register is also the \
         sweep's exemption list, so a path that has moved does not merely read wrong — it stops \
         exempting the file it meant to.",
        wrong.join("\n  ")
    );
}

#[test]
fn every_registered_relation_is_held_by_a_test_that_exists() {
    let root = repo_root();

    let mut wrong = Vec::new();
    for evaluator in EVALUATORS {
        assert!(
            !evaluator.conformance.is_empty(),
            "`{}` claims a relation that nothing checks",
            evaluator.tag
        );
        for conformance in evaluator.conformance {
            let path = root.join(conformance.path);
            let Ok(text) = fs::read_to_string(&path) else {
                wrong.push(format!(
                    "`{}`: {} does not exist",
                    evaluator.tag, conformance.path
                ));
                continue;
            };
            let signature = format!("fn {}(", conformance.test);
            if !text.contains(&signature) {
                wrong.push(format!(
                    "`{}`: {} has no `{}`",
                    evaluator.tag, conformance.path, conformance.test
                ));
            }
        }
    }

    assert!(
        wrong.is_empty(),
        "a registered conformance test has gone missing:\n  {}\n\nA renamed test is a rename in \
         `int-semantics/src/register.rs` too. A deleted one is a relation nothing checks any more, \
         which is the state this register exists to make impossible.",
        wrong.join("\n  ")
    );
}

#[test]
fn the_document_and_the_register_describe_the_same_evaluators() {
    let root = repo_root();
    let doc = root.join("docs").join("int-semantics.md");
    let text = fs::read_to_string(&doc).expect("docs/int-semantics.md is missing");

    // A section heading is ``### `tag` — ...``; the tag is what sits between the first backticks.
    let documented: Vec<&str> = text
        .lines()
        .filter_map(|line| line.strip_prefix("### `"))
        .filter_map(|rest| rest.split_once('`'))
        .map(|(tag, _)| tag)
        .collect();
    let registered: Vec<&str> = EVALUATORS.iter().map(|e| e.tag).collect();

    assert_eq!(
        documented, registered,
        "docs/int-semantics.md and the register disagree about the evaluators, or about their \
         order. The document is the normative prose and the register is its machine-readable \
         half; they are meant to be two views of one list."
    );
}

#[test]
fn no_unregistered_file_defines_integer_arithmetic_of_its_own() {
    let root = repo_root();
    let registered: BTreeSet<&str> = EVALUATORS.iter().flat_map(|e| e.files).copied().collect();

    let mut wrong = Vec::new();
    let mut swept = 0usize;
    for root_dir in swept_roots(&root) {
        for path in rust_files(&root_dir) {
            swept += 1;
            let relative = relative(&root, &path);
            if registered.contains(relative.as_str()) {
                continue;
            }
            let text = fs::read_to_string(&path).expect("could not read a swept file");
            for (number, line) in text.lines().enumerate() {
                if is_comment(line) || !HAND_ROLLED.iter().any(|sig| line.contains(sig)) {
                    continue;
                }
                let waived = NOT_INTEGER_SEMANTICS
                    .iter()
                    .any(|(file, snippet, _)| *file == relative && line.contains(snippet));
                if !waived {
                    wrong.push(format!("{relative}:{}: {}", number + 1, line.trim()));
                }
            }
        }
    }

    // A sweep that found nothing because it looked nowhere would pass silently otherwise.
    assert!(
        swept > 100,
        "the sweep read only {swept} files, which is too few to have covered the workspace"
    );

    assert!(
        wrong.is_empty(),
        "wrapping arithmetic outside every registered evaluator:\n  {}\n\nIf this is an integer \
         operation of the language being compiled, it belongs in \
         `int-semantics/src/register.rs` with a relation and a conformance test — see \
         `docs/CONTRIBUTING.md`. If it is host arithmetic that merely wants a modulus, add it to \
         `NOT_INTEGER_SEMANTICS` in this file with the reason.",
        wrong.join("\n  ")
    );
}

#[test]
fn every_waiver_still_describes_a_line_that_is_there() {
    let root = repo_root();

    let mut stale = Vec::new();
    for (file, snippet, reason) in NOT_INTEGER_SEMANTICS {
        let text = fs::read_to_string(root.join(file)).unwrap_or_default();
        if !text.lines().any(|line| line.contains(snippet)) {
            stale.push(format!("{file}: `{snippet}` ({reason})"));
        }
    }

    assert!(
        stale.is_empty(),
        "a waiver outlived the line it excused:\n  {}\n\nDelete it. A waiver nobody can see the \
         subject of is how an exception list stops meaning anything.",
        stale.join("\n  ")
    );
}
