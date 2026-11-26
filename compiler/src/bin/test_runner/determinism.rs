//! The determinism check: one test compiled several times, its dumps compared.

use super::*;

/// Default subset for `--check-determinism`: small tests chosen to cover the passes where iteration
/// order historically leaked into output (defunctionalize/fn-pointer interning, the specializer,
/// globals lowering, tuple elision), plus one large composite (sha256) and the PRE code-motion
/// shapes (speculative hoisting, the guarded-division gate, cross-branch join insertion, the
/// nested-loop det-call hoist).
pub(crate) const DEFAULT_DETERMINISM_TESTS: &[&str] = &[
    "higher_order_fns",
    "lambda_array",
    "recursion",
    "power",
    "array_lookup",
    "struct_literals",
    "sha256",
    "pre_licm_speculation",
    "pre_guarded_div_hoist",
    "pre_cross_branch_join",
    "pre_nested_loop_hoist",
    "pre_untaint_chain_speculation",
];

/// The dump files the driver writes under `mavros_debug/`, in pipeline-stage order, so the first
/// divergent file names the stage that introduced the divergence.
pub(crate) const DUMP_STAGE_ORDER: &[&str] = &[
    "initial_ssa.txt",
    "monomorphized_ssa.txt",
    "untainted_ssa.txt",
    "program_ssa.txt",
    "hlssa_before_lowering.txt",
    "llssa_after_lowering.txt",
    "program_bytecode.txt",
];

pub(crate) fn parse_runs_arg(args: &[String]) -> usize {
    let mut i = 1;
    while i < args.len() {
        if args[i] == "--runs" && i + 1 < args.len() {
            let n: usize = args[i + 1]
                .parse()
                .expect("--runs requires a positive integer");
            assert!(n >= 2, "--runs must be >= 2");
            return n;
        }
        i += 1;
    }
    3
}

/// Positional arguments after `--check-determinism` (skipping flags and their values) name
/// the tests to check; each is a directory name under `noir_tests/` or a path to a test.
pub(crate) fn parse_determinism_tests(args: &[String]) -> Vec<String> {
    let mut tests = Vec::new();
    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--runs" | "--jobs" | "--field" => i += 2,
            name => {
                tests.push(name.to_string());
                i += 1;
            }
        }
    }
    if tests.is_empty() {
        DEFAULT_DETERMINISM_TESTS
            .iter()
            .map(|s| s.to_string())
            .collect()
    } else {
        tests
    }
}

/// What one `--run-single` execution produced, reduced to comparable form: a content hash
/// per `mavros_debug` dump file, plus the non-WASM `START:`/`END:` lines from the child's
/// stdout (which carry R1CS rows/cols, bytecode sizes, and pass/fail per stage).
///
/// WASM-lane outcomes are deliberately excluded: `llssa_to_llvm` has known pre-existing
/// nondeterministic failures that are tracked separately, and including them here would
/// mask regressions in the deterministic part of the pipeline behind known noise.
pub(crate) struct RunFingerprint {
    pub(crate) dump_hashes: std::collections::BTreeMap<String, u64>,
    pub(crate) output_lines: Vec<String>,
}

/// Recursively copy a test project, skipping artifacts from previous runs.
pub(crate) fn copy_dir_recursive(src: &Path, dst: &Path) {
    fs::create_dir_all(dst).expect("Cannot create destination dir");
    for entry in fs::read_dir(src).expect("Cannot read source dir") {
        let entry = entry.expect("Cannot read dir entry");
        let name = entry.file_name();
        if name == "mavros_debug" || name == "target" {
            continue;
        }
        let src_path = entry.path();
        let dst_path = dst.join(&name);
        if src_path.is_dir() {
            copy_dir_recursive(&src_path, &dst_path);
        } else {
            fs::copy(&src_path, &dst_path).expect("Cannot copy file");
        }
    }
}

/// Compile the test at `root` in a fresh child process and fingerprint the result.
pub(crate) fn determinism_run(
    exe: &Path,
    wasm_runtime_lib: &Path,
    root: &Path,
    field: FieldId,
) -> RunFingerprint {
    use std::hash::{DefaultHasher, Hash, Hasher};

    let mut child = Command::new(exe)
        .args(["--run-single", root.to_str().unwrap()])
        .args(["--field", field.name()])
        .env(wasm_runtime::WASM_RUNTIME_LIB_ENV, wasm_runtime_lib)
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .expect("Failed to spawn child");
    let stdout = child.stdout.take().unwrap();
    let output_lines: Vec<String> = BufReader::new(stdout)
        .lines()
        .map_while(Result::ok)
        .filter(|line| {
            (line.starts_with("START:") || line.starts_with("END:")) && !line.contains("WASM")
        })
        .collect();
    let _ = child.wait();

    let mut dump_hashes = std::collections::BTreeMap::new();
    let debug_dir = root.join("mavros_debug");
    if let Ok(entries) = fs::read_dir(&debug_dir) {
        for entry in entries.filter_map(|e| e.ok()) {
            let path = entry.path();
            if path.extension().is_some_and(|ext| ext == "txt") {
                let contents = fs::read(&path).expect("Cannot read dump file");
                let mut hasher = DefaultHasher::new();
                contents.hash(&mut hasher);
                dump_hashes.insert(
                    path.file_name().unwrap().to_string_lossy().into_owned(),
                    hasher.finish(),
                );
            }
        }
    }

    RunFingerprint {
        dump_hashes,
        output_lines,
    }
}

/// Compare `fp` against the reference run and name what diverged, in stage order.
pub(crate) fn fingerprint_divergences(
    reference: &RunFingerprint,
    fp: &RunFingerprint,
) -> Vec<String> {
    let mut divergent = Vec::new();
    // Stage-ordered known dumps first, then anything unexpected.
    let known = DUMP_STAGE_ORDER.iter().copied();
    let extra = reference
        .dump_hashes
        .keys()
        .chain(fp.dump_hashes.keys())
        .map(|s| s.as_str())
        .filter(|name| !DUMP_STAGE_ORDER.contains(name));
    for name in known.chain(extra) {
        if reference.dump_hashes.get(name) != fp.dump_hashes.get(name)
            && !divergent.iter().any(|d| d == name)
        {
            divergent.push(name.to_string());
        }
    }
    if reference.output_lines != fp.output_lines {
        divergent.push("child stage output (R1CS shape / bytecode size / stage status)".into());
    }
    divergent
}

/// Run each test `runs` times in fresh processes (fresh process = fresh hash seeds, which is
/// the property under test) and check all compiler output is byte-identical across runs.
/// Returns the process exit code: 0 if everything is deterministic, 1 otherwise.
pub(crate) fn check_determinism(tests: &[String], runs: usize, jobs: usize, field: FieldId) -> i32 {
    let resolved: Vec<TestEntry> = tests
        .iter()
        .map(|name| {
            let path = if name.contains('/') {
                PathBuf::from(name)
            } else {
                PathBuf::from("noir_tests").join(name)
            };
            assert!(
                path.is_dir(),
                "Test directory not found: {}",
                path.display()
            );
            TestEntry {
                path: fs::canonicalize(&path).unwrap(),
                fixture_name: name.clone(),
                display_name: name.clone(),
                expectation: TestExpectation::ExecutionSuccess,
            }
        })
        .collect();

    // Build the wasm runtime once and hand children the artifact path (see `run_parent`).
    let wasm_runtime_lib =
        wasm_runtime::locate_or_build().unwrap_or_else(|error| panic!("{error}"));
    let exe = env::current_exe().expect("Cannot determine own exe path");

    let scratch = tempfile::tempdir().expect("Cannot create scratch dir");
    let total = resolved.len() * runs;
    eprintln!(
        "Checking determinism over {field}: {} test(s) x {runs} run(s), {jobs} parallel job(s)",
        resolved.len()
    );

    // Copy every test x run into its own directory up front, then fingerprint them in
    // parallel with the same worker-pool shape as `run_parent`.
    let mut work: Vec<(usize, usize, PathBuf)> = Vec::new();
    for (t, entry) in resolved.iter().enumerate() {
        let test_dir_name = entry
            .path
            .file_name()
            .unwrap()
            .to_string_lossy()
            .into_owned();
        for r in 0..runs {
            let dst = scratch.path().join(format!("{test_dir_name}-run{r}"));
            copy_dir_recursive(&entry.path, &dst);
            work.push((t, r, dst));
        }
    }

    let next_idx = AtomicUsize::new(0);
    let completed = AtomicUsize::new(0);
    let slots: Vec<Mutex<Option<RunFingerprint>>> = (0..total).map(|_| Mutex::new(None)).collect();
    std::thread::scope(|scope| {
        for _ in 0..jobs {
            scope.spawn(|| {
                loop {
                    let idx = next_idx.fetch_add(1, Ordering::SeqCst);
                    if idx >= total {
                        break;
                    }
                    let (t, r, dst) = &work[idx];
                    let fp = determinism_run(&exe, &wasm_runtime_lib, dst, field);
                    let n = completed.fetch_add(1, Ordering::SeqCst) + 1;
                    eprintln!("[{n}/{total}] {} run {}", resolved[*t].display_name, r + 1);
                    *slots[idx].lock().unwrap() = Some(fp);
                }
            });
        }
    });
    let fingerprints: Vec<RunFingerprint> = slots
        .into_iter()
        .map(|m| m.into_inner().unwrap().expect("run slot unfilled"))
        .collect();

    let mut all_deterministic = true;
    println!("## {field}\n");
    println!("| Test | Deterministic | First divergence |");
    println!("|------|---------------|------------------|");
    for (t, entry) in resolved.iter().enumerate() {
        let base = t * runs;
        let reference = &fingerprints[base];
        let divergent: Vec<String> = (1..runs)
            .flat_map(|r| fingerprint_divergences(reference, &fingerprints[base + r]))
            .fold(Vec::new(), |mut acc, d| {
                if !acc.contains(&d) {
                    acc.push(d);
                }
                acc
            });
        if divergent.is_empty() {
            println!("| {} | ✅ | |", entry.display_name);
        } else {
            all_deterministic = false;
            println!("| {} | ❌ | {} |", entry.display_name, divergent.join(", "));
        }
    }
    println!();

    if all_deterministic {
        eprintln!("All outputs deterministic across {runs} runs");
        0
    } else {
        // Keep the scratch dir around so the divergent dumps can be diffed directly.
        let kept = scratch.keep();
        eprintln!("NONDETERMINISM DETECTED — dumps kept at {}", kept.display());
        1
    }
}
