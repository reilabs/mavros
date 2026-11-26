use std::{
    cell::OnceCell,
    env, fs,
    io::{BufRead, BufReader, Write},
    path::{Path, PathBuf},
    process::{Command, Stdio},
    sync::{
        Mutex,
        atomic::{AtomicUsize, Ordering},
    },
};

use cargo_metadata::MetadataCommand;

use mavros_artifacts::{Bn254, FieldId, Goldilocks, MavrosField, WitnessManifest};
use mavros_compiler::{
    Project, abi_helpers, api,
    collections::{HashMap, HashSet},
    compiler::codegen::{
        CodeGenOptions,
        hlssa_to_r1cs::SealedR1CS,
        llssa_to_llvm::{WasmCompileOpts, wasm_debug_info_path},
        r1cs_compact,
    },
    driver::{Driver, Error as DriverError},
    ice_usr,
    vm::interpreter,
    wasm_host, wasm_runtime,
};
use rand::SeedableRng;

mod child;
mod determinism;
mod expectations;
mod report;
mod scoring;
#[cfg(test)]
mod tests;

use child::*;
use determinism::*;
use expectations::*;
use report::*;
use scoring::*;

fn main() {
    let args: Vec<String> = env::args().collect();

    // Child mode: --run-single <path> --field F [--expect-failure] [--expected-error TEXT]
    // [--analyze-r1cs], as `child_args` spells it.
    if let Some(child) = ChildArgs::parse(&args) {
        run_single(
            child.path,
            child.expect_failure,
            child.expected_error,
            child.analyze,
            child.field,
        );
        return;
    }

    // Combine the independently executed field reports without rerunning the corpus.
    if args.get(1).is_some_and(|arg| arg == "--merge-status") {
        let paths: Vec<_> = args[2..]
            .iter()
            .take_while(|arg| *arg != "--output")
            .collect();
        let reports: Vec<_> = paths
            .iter()
            .map(|path| {
                fs::read_to_string(path)
                    .unwrap_or_else(|error| panic!("Cannot read {path}: {error}"))
            })
            .collect();
        let markdown = merge_status_reports(&reports).expect("Cannot merge field reports");
        fs::write(parse_output_arg(&args), markdown).expect("Cannot write merged report");
        return;
    }

    // Regression check mode: --check-regression <baseline> <current>
    if args.len() >= 4 && args[1] == "--check-regression" {
        let baseline = PathBuf::from(&args[2]);
        let current = PathBuf::from(&args[3]);
        std::process::exit(check_regression(&baseline, &current));
    }

    // Growth check mode: --check-growth <baseline> <current> [--comments <dir>]
    // Prints markdown to stdout if any rows/cols grew; exits 0 always. With `--comments`, also
    // writes the report split into PR comments that each fit GitHub's limit, as
    // `<dir>/comment-NN.md`.
    if args.len() >= 4 && args[1] == "--check-growth" {
        let baseline = PathBuf::from(&args[2]);
        let current = PathBuf::from(&args[3]);
        let report = check_growth(&baseline, &current);
        print!("{}", report.render());
        if args.len() >= 6 && args[4] == "--comments" {
            let dir = PathBuf::from(&args[5]);
            fs::create_dir_all(&dir).expect("Cannot create the comment directory");
            for (index, comment) in report
                .into_comments(GROWTH_COMMENT_BYTES)
                .iter()
                .enumerate()
            {
                fs::write(dir.join(format!("comment-{:02}.md", index + 1)), comment)
                    .expect("Cannot write a growth report comment");
            }
        }
        return;
    }

    // Determinism check mode: --check-determinism [--runs N] [--jobs N] [--field F] [test names...]
    // Compiles each test several times in fresh processes and verifies the compiler
    // output (SSA dumps, bytecode, R1CS shape) is byte-identical across runs, over `--field` or
    // over every field.
    if args.len() >= 2 && args[1] == "--check-determinism" {
        let runs = parse_runs_arg(&args);
        let jobs = parse_jobs_arg(&args);
        let tests = parse_determinism_tests(&args);
        let codes: Vec<i32> = parse_fields_arg(&args)
            .into_iter()
            .map(|field| check_determinism(&tests, runs, jobs, field))
            .collect();
        std::process::exit(first_failure(codes));
    }

    // Parent mode
    validate_parent_args(&args);
    let output_path = parse_output_arg(&args);
    let jobs = parse_jobs_arg(&args);
    // Opt-in, developer-only: run the report-only R1CS compaction analysis for every test and print
    // its per-test stats to stdout. Off by default so CI (and the STATUS.md it produces) never pays
    // for it.
    let analyze = args.iter().any(|a| a == "--analyze-r1cs");
    let fields = parse_fields_arg(&args);
    // `--only <test>`, repeatable: run only the tests each names, as `DEFAULT_IGNORED_TESTS` names
    // them. The report then covers those tests alone, so it may not land in the default STATUS.md.
    let only = parse_only_args(&args);
    assert!(
        only.is_empty() || args.iter().any(|arg| arg == "--output"),
        "--only writes a partial report, so it needs --output"
    );
    run_parent(
        &output_path,
        jobs,
        DEFAULT_IGNORED_TESTS,
        &only,
        analyze,
        &fields,
    );
}

/// What a child is asked to run, as `child_args` spells it.
struct ChildArgs<'a> {
    path: PathBuf,
    expect_failure: bool,
    expected_error: Option<&'a str>,
    analyze: bool,
    field: FieldId,
}

impl<'a> ChildArgs<'a> {
    /// The child's arguments, if `args` runs one.
    fn parse(args: &'a [String]) -> Option<Self> {
        if args.len() < 3 || args[1] != "--run-single" {
            return None;
        }
        let flags = &args[3..];
        Some(Self {
            path: PathBuf::from(&args[2]),
            expect_failure: flags.iter().any(|a| a == "--expect-failure"),
            expected_error: parse_expected_error_arg(flags),
            analyze: flags.iter().any(|a| a == "--analyze-r1cs"),
            field: parse_field_arg(args).unwrap_or_else(|| {
                panic!(
                    "--run-single requires --field, one of {}",
                    FieldId::ALL.map(FieldId::name).join(", ")
                )
            }),
        })
    }
}

/// The text `--expected-error TEXT` names among a child's `args`, if any.
///
/// # Panics
///
/// If `--expected-error` is the last argument.
fn parse_expected_error_arg(args: &[String]) -> Option<&str> {
    let flag = args.iter().position(|arg| arg == "--expected-error")?;
    Some(
        args.get(flag + 1)
            .unwrap_or_else(|| panic!("--expected-error requires the error's text"))
            .as_str(),
    )
}

/// The flags a full run takes that are followed by a value.
const PARENT_VALUE_FLAGS: &[&str] = &["--output", "--jobs", "--field", "--only"];

/// The flags a full run takes alone.
const PARENT_SWITCHES: &[&str] = &["--analyze-r1cs"];

/// Refuses any argument a full run does not take. A full run writes `STATUS.md` unless told
/// otherwise, so a mistyped flag, or a mode given too few arguments, must not start one.
///
/// # Panics
///
/// If an argument is unknown, or a flag is missing its value.
fn validate_parent_args(args: &[String]) {
    let mut rest = args.iter().skip(1);
    while let Some(arg) = rest.next() {
        if PARENT_VALUE_FLAGS.contains(&arg.as_str()) {
            assert!(rest.next().is_some(), "{arg} requires a value");
        } else {
            assert!(
                PARENT_SWITCHES.contains(&arg.as_str()),
                "unknown argument {arg:?}: a full run takes {PARENT_VALUE_FLAGS:?} with a value \
                 and {PARENT_SWITCHES:?}"
            );
        }
    }
}

/// The path `--output PATH` names, or `STATUS.md`.
///
/// # Panics
///
/// If `--output` is the last argument.
fn parse_output_arg(args: &[String]) -> PathBuf {
    let Some(flag) = args.iter().position(|arg| arg == "--output") else {
        return PathBuf::from("STATUS.md");
    };
    PathBuf::from(
        args.get(flag + 1)
            .unwrap_or_else(|| panic!("--output requires a path")),
    )
}

fn parse_jobs_arg(args: &[String]) -> usize {
    let mut i = 1;
    while i < args.len() {
        if args[i] == "--jobs" && i + 1 < args.len() {
            let n: usize = args[i + 1]
                .parse()
                .expect("--jobs requires a positive integer");
            assert!(n >= 1, "--jobs must be >= 1");
            return n;
        }
        i += 1;
    }
    1
}

fn parse_only_args(args: &[String]) -> Vec<String> {
    args.iter()
        .enumerate()
        .filter(|(_, arg)| *arg == "--only")
        .map(|(i, _)| {
            args.get(i + 1)
                .unwrap_or_else(|| panic!("--only requires a test name"))
                .clone()
        })
        .collect()
}

/// The first nonzero exit code among `codes`, or 0 when every one succeeded.
fn first_failure(codes: impl IntoIterator<Item = i32>) -> i32 {
    codes.into_iter().find(|&code| code != 0).unwrap_or(0)
}

/// The field `--field` names, or `None` without the flag.
fn parse_field_arg(args: &[String]) -> Option<FieldId> {
    let index = args.iter().position(|arg| arg == "--field")?;
    let name = args.get(index + 1).map_or("", String::as_str);
    Some(
        name.parse()
            .unwrap_or_else(|error| panic!("--field: {error}")),
    )
}

/// The field `--field` names, or every field without the flag.
fn parse_fields_arg(args: &[String]) -> Vec<FieldId> {
    parse_field_arg(args).map_or_else(|| FieldId::ALL.to_vec(), |field| vec![field])
}

// ── Parent: discover & run all tests ──────────────────────────────────

/// Use `cargo metadata` to find the root of the noir git dependency, then
/// return the path to `test_programs` inside it.
fn find_noir_test_programs_dir() -> Option<PathBuf> {
    let metadata = MetadataCommand::new().exec().ok()?;
    // Find any package from the noir git repo (e.g. "nargo").
    let noir_pkg = metadata.packages.iter().find(|p| {
        p.source
            .as_ref()
            .is_some_and(|s| s.repr.contains("worldfnd/noir"))
    })?;
    // Walk up from the package manifest to find the repo root containing
    // `test_programs`.
    let mut dir: &Path = noir_pkg.manifest_path.as_std_path();
    loop {
        dir = dir.parent()?;
        let candidate = dir.join("test_programs");
        if candidate.join("execution_success").is_dir() {
            return Some(candidate);
        }
    }
}

/// A package test with its fixture name for workspace-level ignores.
struct TestEntry {
    path: PathBuf,
    fixture_name: String,
    display_name: String,
    expectation: TestExpectation,
}

fn discover_test_packages(
    path: PathBuf,
    fixture_name: String,
    expectation: TestExpectation,
) -> Vec<TestEntry> {
    let root = match fs::canonicalize(&path) {
        Ok(root) => root,
        Err(error) => {
            eprintln!("Cannot resolve test directory {}: {error}", path.display());
            // Let the child report the directory error as a test result.
            return vec![TestEntry {
                path,
                display_name: fixture_name.clone(),
                fixture_name,
                expectation,
            }];
        }
    };
    let packages = Project::binary_package_roots(&root).unwrap_or_else(|error| {
        eprintln!("Cannot discover packages in {}: {error}", root.display());
        vec![root.clone()]
    });
    packages
        .into_iter()
        .map(|path| {
            // Use full paths to label workspace members outside the test directory.
            let member = path.strip_prefix(&root).unwrap_or(&path);
            let display_name = if member.as_os_str().is_empty() {
                fixture_name.clone()
            } else {
                format!("{fixture_name}/{}", member.display())
            };
            let expectation =
                expected_error_of(&path).map_or(expectation, TestExpectation::CompileRejection);
            TestEntry {
                path,
                fixture_name: fixture_name.clone(),
                display_name,
                expectation,
            }
        })
        .collect()
}

fn collect_test_dirs(base: &Path, prefix: &str, expectation: TestExpectation) -> Vec<TestEntry> {
    let Ok(entries) = fs::read_dir(base) else {
        return Vec::new();
    };
    let mut dirs: Vec<TestEntry> = entries
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.is_dir() && p.join("Nargo.toml").is_file())
        .flat_map(|p| {
            let test_name = p.file_name().unwrap().to_string_lossy().into_owned();
            discover_test_packages(p, format!("{prefix}{test_name}"), expectation)
        })
        .collect();
    dirs.sort_by(|a, b| a.display_name.cmp(&b.display_name));
    dirs
}

fn run_parent(
    output_path: &Path,
    jobs: usize,
    ignored_tests: &[&str],
    only: &[String],
    analyze: bool,
    fields: &[FieldId],
) {
    let mut entries: Vec<TestEntry> = Vec::new();

    // 1. Local noir_tests/ directory
    let local_tests = PathBuf::from("noir_tests");
    if local_tests.is_dir() {
        entries.extend(collect_test_dirs(
            &local_tests,
            "noir_tests/",
            TestExpectation::ExecutionSuccess,
        ));
    }

    // 2. Local programs expected to be rejected — at compile time (frontend or Mavros-specific)
    // or at witness generation (a bounds check that only a real input can violate). Keeping these
    // separate from `noir_tests` lets the same child runner exercise them without treating a
    // clean rejection as a regression.
    let local_failure_tests = PathBuf::from("noir_failure_tests");
    if local_failure_tests.is_dir() {
        entries.extend(collect_test_dirs(
            &local_failure_tests,
            "noir_failure_tests/",
            TestExpectation::ExecutionFailure,
        ));
    }

    // 3. Noir repo test_programs/* (discovered via cargo-metadata)
    let noir_test_programs = find_noir_test_programs_dir();
    if let Some(test_programs) = &noir_test_programs {
        eprintln!("Found noir test_programs at: {}", test_programs.display());
        entries.extend(collect_test_dirs(
            &test_programs.join("execution_success"),
            "noir/test_programs/execution_success/",
            TestExpectation::ExecutionSuccess,
        ));
        entries.extend(collect_test_dirs(
            &test_programs.join("execution_failure"),
            "noir/test_programs/execution_failure/",
            TestExpectation::ExecutionFailure,
        ));
        entries.extend(collect_test_dirs(
            &test_programs.join("execution_panic"),
            "noir/test_programs/execution_panic/",
            TestExpectation::ExecutionFailure,
        ));
    } else {
        eprintln!("Warning: could not locate noir test_programs via cargo-metadata");
    }

    assert!(!entries.is_empty(), "No test directories found");
    let overrides: Vec<_> = test_expectation_overrides().collect();
    validate_overrides(
        &entries,
        &overrides,
        FIELD_EXPECTATION_OVERRIDES,
        noir_test_programs.is_some(),
    );
    apply_test_expectation_overrides(&mut entries);
    select_tests(&mut entries, only);

    // Build the wasm runtime once before spawning workers and hand children the artifact path.
    // If children built it themselves, their `cargo metadata`/`cargo build` invocations would
    // walk the noir git checkout while other children create and delete `mavros_debug` dirs for
    // tests living inside it, crashing whichever child loses the race.
    let wasm_runtime_lib =
        wasm_runtime::locate_or_build().unwrap_or_else(|error| panic!("{error}"));
    eprintln!("Built wasm runtime at: {}", wasm_runtime_lib.display());

    let exe = env::current_exe().expect("Cannot determine own exe path");
    let total = entries.len() * fields.len();
    let ignored_count = entries
        .iter()
        .filter(|entry| is_ignored_test(entry, ignored_tests))
        .count()
        * fields.len();
    eprintln!("Running {total} test(s) with {jobs} parallel job(s) ({ignored_count} ignored)");

    let next_idx = AtomicUsize::new(0);
    let completed = AtomicUsize::new(0);
    let slots: Vec<Mutex<Option<TestResult>>> = (0..total).map(|_| Mutex::new(None)).collect();

    std::thread::scope(|scope| {
        for _ in 0..jobs {
            scope.spawn(|| {
                loop {
                    let idx = next_idx.fetch_add(1, Ordering::SeqCst);
                    if idx >= entries.len() {
                        break;
                    }
                    let entry = &entries[idx];
                    // Each driver clears its package's mavros_debug directory. Run fields for
                    // one package sequentially so they never delete each other's dumps.
                    for (field_idx, &field) in fields.iter().enumerate() {
                        let ignored = is_ignored_test(entry, ignored_tests);
                        let expectation = expectation_on(entry, field);
                        let result = if ignored {
                            ignored_test_result(&entry.display_name, field)
                        } else {
                            let mut cmd = Command::new(&exe);
                            cmd.args(child_args(&entry.path, field, expectation, analyze));
                            cmd.env(wasm_runtime::WASM_RUNTIME_LIB_ENV, &wasm_runtime_lib);
                            let mut child = cmd
                                .stdout(Stdio::piped())
                                .stderr(Stdio::inherit())
                                .spawn()
                                .expect("Failed to spawn child");
                            let stdout = child.stdout.take().unwrap();
                            let lines: Vec<String> = BufReader::new(stdout)
                                .lines()
                                .map_while(Result::ok)
                                .collect();
                            let _ = child.wait();
                            parse_child_output(&entry.display_name, field, expectation, &lines)
                        };
                        let n = completed.fetch_add(1, Ordering::SeqCst) + 1;
                        let suffix = if ignored { " (ignored)" } else { "" };
                        eprintln!(
                            "[{n}/{total}] {} [{}]{suffix}",
                            entry.display_name,
                            field.name()
                        );
                        *slots[idx * fields.len() + field_idx].lock().unwrap() = Some(result);
                    }
                }
            });
        }
    });

    let results: Vec<TestResult> = slots
        .into_iter()
        .map(|m| m.into_inner().unwrap().expect("test slot unfilled"))
        .collect();

    // Developer-only: echo the per-test R1CS compaction breakdown to stdout, in test order. Only
    // present under `--analyze-r1cs`; never written to STATUS.md.
    if analyze {
        for r in &results {
            if let Some(stats) = &r.analysis_stats {
                println!("{} [{}]: {stats}", r.name, r.field.name());
            }
        }
    }

    let md = render_markdown(&results);
    fs::write(output_path, &md).expect("Cannot write output file");
    eprintln!("Wrote {}", output_path.display());
    print!("{md}");
}

/// The arguments that run the test at `path` on `field` in a child, which `main` reads back.
fn child_args(
    path: &Path,
    field: FieldId,
    expectation: TestExpectation,
    analyze: bool,
) -> Vec<std::ffi::OsString> {
    let mut args: Vec<std::ffi::OsString> = vec!["--run-single".into(), path.into()];
    args.extend(["--field", field.name()].map(Into::into));
    if expectation != TestExpectation::ExecutionSuccess {
        args.push("--expect-failure".into());
    }
    if let TestExpectation::CompileRejection(text) = expectation {
        args.extend(["--expected-error", text].map(Into::into));
    }
    if analyze {
        args.push("--analyze-r1cs".into());
    }
    args
}

fn is_ignored_test(entry: &TestEntry, ignored_tests: &[&str]) -> bool {
    ignored_tests
        .iter()
        .any(|ignored| names_test(entry, ignored))
}

/// Whether `name` names `entry`: it is a trailing run of path components of the entry's fixture
/// or display name, or of one of their ancestors, so a workspace's name names each member.
fn names_test(entry: &TestEntry, name: &str) -> bool {
    let name = Path::new(name.trim_matches('/'));
    !name.as_os_str().is_empty()
        && [&entry.fixture_name, &entry.display_name]
            .iter()
            .any(|entry_name| {
                Path::new(entry_name)
                    .ancestors()
                    .any(|ancestor| ancestor.ends_with(name))
            })
}

/// Keeps only the entries some name in `only` names, unless `only` is empty.
///
/// # Panics
///
/// If a name in `only` names no entry.
fn select_tests(entries: &mut Vec<TestEntry>, only: &[String]) {
    if only.is_empty() {
        return;
    }
    for name in only {
        assert!(
            entries.iter().any(|entry| names_test(entry, name)),
            "--only {name}: no test has this name"
        );
    }
    entries.retain(|entry| only.iter().any(|name| names_test(entry, name)));
}
