use std::{
    collections::HashMap,
    env,
    fs,
    io::{BufRead, BufReader, Write},
    path::{Path, PathBuf},
    process::{Command, Stdio},
};

use cargo_metadata::MetadataCommand;

use ark_ff::{Field as ArkField, UniformRand as _};
use noirc_abi::input_parser::Format;
use rand::SeedableRng;
use mavros::{Project, abi_helpers, compiler::Field, compiler::r1cs_gen::R1CS, driver::Driver, vm::interpreter};
use wasmtime::{Engine, Linker, Memory, Module, Store};

fn main() {
    let args: Vec<String> = env::args().collect();

    // Child mode: --run-single <path>
    if args.len() >= 3 && args[1] == "--run-single" {
        run_single(PathBuf::from(&args[2]));
        return;
    }

    // Regression check mode: --check-regression <baseline> <current>
    if args.len() >= 4 && args[1] == "--check-regression" {
        let baseline = PathBuf::from(&args[2]);
        let current = PathBuf::from(&args[3]);
        std::process::exit(check_regression(&baseline, &current));
    }

    // Growth check mode: --check-growth <baseline> <current>
    // Prints markdown to stdout if any rows/cols grew; exits 0 always.
    if args.len() >= 4 && args[1] == "--check-growth" {
        let baseline = PathBuf::from(&args[2]);
        let current = PathBuf::from(&args[3]);
        check_growth(&baseline, &current);
        return;
    }

    // Parent mode
    let output_path = parse_output_arg(&args);
    run_parent(&output_path);
}

fn parse_output_arg(args: &[String]) -> PathBuf {
    let mut i = 1;
    while i < args.len() {
        if args[i] == "--output" && i + 1 < args.len() {
            return PathBuf::from(&args[i + 1]);
        }
        i += 1;
    }
    PathBuf::from("STATUS.md")
}

// ── Child: run a single test ──────────────────────────────────────────

fn emit(line: &str) {
    let stdout = std::io::stdout();
    let mut out = stdout.lock();
    let _ = writeln!(out, "{line}");
    let _ = out.flush();
}

fn run_single(root: PathBuf) {
    // 1. Compile
    emit("START:COMPILED");
    let driver = (|| {
        let project = Project::new(root.clone()).ok()?;
        let mut driver = Driver::new(project, false);
        driver.run_noir_compiler().ok()?;
        driver.make_struct_access_static().ok()?;
        driver.monomorphize().ok()?;
        driver.explictize_witness().ok()?;
        Some(driver)
    })();
    let mut driver = match driver {
        Some(d) => { emit("END:COMPILED:ok"); d }
        None => { emit("END:COMPILED:fail"); return; }
    };

    // 2. R1CS
    emit("START:R1CS");
    let r1cs = match driver.generate_r1cs() {
        Ok(r) => {
            let rows = r.constraints.len();
            let cols = r.witness_layout.size();
            emit(&format!("END:R1CS:ok:{rows}:{cols}"));
            Some(r)
        }
        Err(_) => { emit("END:R1CS:fail"); None }
    };

    // 3. Compile witgen  (depends on R1CS)
    let witgen_binary = r1cs.as_ref().and_then(|_| {
        emit("START:WITGEN_COMPILE");
        match driver.compile_witgen() {
            Ok(b) => { emit("END:WITGEN_COMPILE:ok"); Some(b) }
            Err(_) => { emit("END:WITGEN_COMPILE:fail"); None }
        }
    });

    // 4. Compile AD  (depends on R1CS, independent of witgen)
    let ad_binary = r1cs.as_ref().and_then(|_| {
        emit("START:AD_COMPILE");
        match driver.compile_ad() {
            Ok(b) => { emit("END:AD_COMPILE:ok"); Some(b) }
            Err(_) => { emit("END:AD_COMPILE:fail"); None }
        }
    });

    // Load inputs (needed for witgen run)
    let ordered_params = load_inputs(&root.join("Prover.toml"), &driver);

    // 5. Run witgen  (depends on WITGEN_COMPILE)
    let had_witgen_binary = witgen_binary.is_some();
    let witgen_result = witgen_binary.and_then(|mut binary| {
        emit("START:WITGEN_RUN");
        let r1cs = r1cs.as_ref().unwrap();
        let params = ordered_params.as_ref()?;
        let result = interpreter::run(
            &mut binary,
            r1cs.witness_layout,
            r1cs.constraints_layout,
            params,
        );
        emit("END:WITGEN_RUN:ok");
        Some(result)
    });
    if had_witgen_binary && witgen_result.is_none() {
        emit("START:WITGEN_RUN");
        emit("END:WITGEN_RUN:fail");
    }

    // 6. Check witgen correctness  (depends on WITGEN_RUN)
    if let (Some(result), Some(r1cs)) = (&witgen_result, &r1cs) {
        emit("START:WITGEN_CORRECT");
        let correct = r1cs.check_witgen_output(
            &result.out_wit_pre_comm,
            &result.out_wit_post_comm,
            &result.out_a,
            &result.out_b,
            &result.out_c,
        );
        emit(if correct { "END:WITGEN_CORRECT:ok" } else { "END:WITGEN_CORRECT:fail" });
    }

    // 7. Witgen leak check  (depends on WITGEN_RUN)
    if let Some(result) = &witgen_result {
        emit("START:WITGEN_NOLEAK");
        let tmpdir = tempfile::tempdir().unwrap();
        let leftover = result.instrumenter.plot(&tmpdir.path().join("wt.png"));
        emit(if leftover == 0 { "END:WITGEN_NOLEAK:ok" } else { "END:WITGEN_NOLEAK:fail" });
    }

    // 8. Run AD  (depends on AD_COMPILE, independent of witgen)
    let ad_result = ad_binary.and_then(|mut binary| {
        emit("START:AD_RUN");
        let r1cs = r1cs.as_ref().unwrap();
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let ad_coeffs: Vec<Field> = (0..r1cs.constraints.len())
            .map(|_| ark_bn254::Fr::rand(&mut rng))
            .collect();
        let (ad_a, ad_b, ad_c, ad_instrumenter) = interpreter::run_ad(
            &mut binary,
            &ad_coeffs,
            r1cs.witness_layout,
            r1cs.constraints_layout,
        );
        emit("END:AD_RUN:ok");
        Some((ad_coeffs, ad_a, ad_b, ad_c, ad_instrumenter))
    });

    // 9. Check AD correctness  (depends on AD_RUN)
    if let (Some((coeffs, ad_a, ad_b, ad_c, _)), Some(r1cs)) = (&ad_result, &r1cs) {
        emit("START:AD_CORRECT");
        let correct = r1cs.check_ad_output(coeffs, ad_a, ad_b, ad_c);
        emit(if correct { "END:AD_CORRECT:ok" } else { "END:AD_CORRECT:fail" });
    }

    // 10. AD leak check  (depends on AD_RUN)
    if let Some((_, _, _, _, instrumenter)) = &ad_result {
        emit("START:AD_NOLEAK");
        let tmpdir = tempfile::tempdir().unwrap();
        let leftover = instrumenter.plot(&tmpdir.path().join("ad.png"));
        emit(if leftover == 0 { "END:AD_NOLEAK:ok" } else { "END:AD_NOLEAK:fail" });
    }

    // 11. Compile WASM  (depends on R1CS)
    let wasm_path = r1cs.as_ref().and_then(|r1cs| {
        emit("START:WITGEN_WASM_COMPILE");
        let tmpdir = tempfile::tempdir().ok()?;
        let wasm_path = tmpdir.into_path().join("witgen.wasm");
        match driver.compile_llvm_targets(false, Some((wasm_path.clone(), r1cs))) {
            Ok(_) if wasm_path.exists() => {
                emit("END:WITGEN_WASM_COMPILE:ok");
                Some(wasm_path)
            }
            Ok(_) => {
                eprintln!("WASM compile succeeded but output file not found at {:?}", wasm_path);
                emit("END:WITGEN_WASM_COMPILE:fail");
                None
            }
            Err(e) => {
                eprintln!("WASM compile error: {:?}", e);
                emit("END:WITGEN_WASM_COMPILE:fail");
                None
            }
        }
    });

    // 12. Run WASM  (depends on WITGEN_WASM_COMPILE)
    let wasm_result = wasm_path.as_ref().and_then(|wasm_path| {
        emit("START:WITGEN_WASM_RUN");
        let r1cs = r1cs.as_ref().unwrap();
        let params = ordered_params.as_ref()?;
        match run_wasm(wasm_path, r1cs, params) {
            Ok(result) => {
                emit("END:WITGEN_WASM_RUN:ok");
                Some(result)
            }
            Err(_) => {
                emit("END:WITGEN_WASM_RUN:fail");
                None
            }
        }
    });

    // 13. Check WASM correctness  (depends on WITGEN_WASM_RUN)
    if let (Some(result), Some(r1cs)) = (&wasm_result, &r1cs) {
        emit("START:WITGEN_WASM_CORRECT");


        let correct = r1cs.check_witgen_output(
            &result.out_wit_pre_comm,
            &result.out_wit_post_comm,
            &result.out_a,
            &result.out_b,
            &result.out_c,
        );
        emit(if correct { "END:WITGEN_WASM_CORRECT:ok" } else { "END:WITGEN_WASM_CORRECT:fail" });
    }

    // 14. AD WASM Compile  (depends on R1CS, not yet implemented)
    let ad_wasm_path: Option<std::path::PathBuf> = r1cs.as_ref().and_then(|_r1cs| {
        emit("START:AD_WASM_COMPILE");
        panic!("AD WASM Compile is not yet implemented");
        #[allow(unreachable_code)]
        {
            emit("END:AD_WASM_COMPILE:fail");
            None
        }
    });

    // 15. AD WASM Run  (depends on AD_WASM_COMPILE, not yet implemented)
    let ad_wasm_result: Option<()> = ad_wasm_path.as_ref().and_then(|_wasm_path| {
        emit("START:AD_WASM_RUN");
        panic!("AD WASM Run is not yet implemented");
        #[allow(unreachable_code)]
        {
            emit("END:AD_WASM_RUN:fail");
            None
        }
    });

    // 16. AD WASM Correct  (depends on AD_WASM_RUN, not yet implemented)
    if let (Some(_result), Some(_r1cs)) = (&ad_wasm_result, &r1cs) {
        emit("START:AD_WASM_CORRECT");
        panic!("AD WASM Correct is not yet implemented");
        #[allow(unreachable_code)]
        emit("END:AD_WASM_CORRECT:fail");
    }
}

fn load_inputs(file_path: &Path, driver: &Driver) -> Option<Vec<interpreter::InputValueOrdered>> {
    let ext = file_path.extension().and_then(|e| e.to_str())?;
    let format = Format::from_ext(ext)?;
    let contents = fs::read_to_string(file_path).ok()?;
    let params = format.parse(&contents, driver.abi()).ok()?;
    Some(abi_helpers::ordered_params_from_btreemap(driver.abi(), &params))
}

// ── WASM Runner ──────────────────────────────────────────────────────

const FIELD_SIZE: usize = 32; // 4 x i64 = 32 bytes

/// Output from running WASM witgen
struct WasmResult {
    out_wit_pre_comm: Vec<Field>,
    out_wit_post_comm: Vec<Field>,
    out_a: Vec<Field>,
    out_b: Vec<Field>,
    out_c: Vec<Field>,
}

/// Read a field element from WASM memory
fn read_field_from_memory(memory: &Memory, store: impl wasmtime::AsContext, ptr: u32) -> Field {
    use ark_ff::BigInt;
    let data = memory.data(&store);
    let offset = ptr as usize;
    let l0 = u64::from_le_bytes(data[offset..offset + 8].try_into().unwrap());
    let l1 = u64::from_le_bytes(data[offset + 8..offset + 16].try_into().unwrap());
    let l2 = u64::from_le_bytes(data[offset + 16..offset + 24].try_into().unwrap());
    let l3 = u64::from_le_bytes(data[offset + 24..offset + 32].try_into().unwrap());
    ark_bn254::Fr::new_unchecked(BigInt::new([l0, l1, l2, l3]))
}

/// Write a field element to WASM memory (writes Montgomery form)
fn write_field_to_memory(memory: &Memory, store: &mut Store<()>, ptr: u32, field: Field) {
    // Access the internal Montgomery representation directly
    // field.0 is the BigInt, field.0.0 is the [u64; 4] limbs in Montgomery form
    let limbs = field.0.0;
    let offset = ptr as usize;
    let data = memory.data_mut(store);
    data[offset..offset + 8].copy_from_slice(&limbs[0].to_le_bytes());
    data[offset + 8..offset + 16].copy_from_slice(&limbs[1].to_le_bytes());
    data[offset + 16..offset + 24].copy_from_slice(&limbs[2].to_le_bytes());
    data[offset + 24..offset + 32].copy_from_slice(&limbs[3].to_le_bytes());
}

/// Flatten an InputValueOrdered into a list of Field elements
fn flatten_input_value(value: &interpreter::InputValueOrdered) -> Vec<Field> {
    let mut result = Vec::new();
    match value {
        interpreter::InputValueOrdered::Field(elem) => result.push(*elem),
        interpreter::InputValueOrdered::Vec(vec_elements) => {
            for elem in vec_elements {
                result.extend(flatten_input_value(elem));
            }
        }
        interpreter::InputValueOrdered::Struct(fields) => {
            for (_field_name, field_value) in fields {
                result.extend(flatten_input_value(field_value));
            }
        }
        interpreter::InputValueOrdered::String(_) => {
            panic!("Strings are not supported in WASM runner")
        }
    }
    result
}

fn run_wasm(
    wasm_path: &Path,
    r1cs: &R1CS,
    params: &[interpreter::InputValueOrdered],
) -> Result<WasmResult, Box<dyn std::error::Error>> {
    let witness_count = r1cs.witness_layout.size();
    let constraint_count = r1cs.constraints.len();

    // Calculate memory layout
    // Reserve space at the start for the WASM module's stack (default 64KB from wasm-ld)
    let data_offset: u32 = 65536; // Skip past WASM stack area
    let vm_struct_size: u32 = 16; // 4 x u32 pointers
    let witness_bytes = (witness_count * FIELD_SIZE) as u32;
    let constraint_bytes = (constraint_count * FIELD_SIZE) as u32;

    let vm_struct_ptr = data_offset;
    let witness_ptr = vm_struct_ptr + vm_struct_size;
    let a_ptr = witness_ptr + witness_bytes;
    let b_ptr = a_ptr + constraint_bytes;
    let c_ptr = b_ptr + constraint_bytes;
    let total_bytes = c_ptr + constraint_bytes;

    // Create wasmtime engine and store
    let engine = Engine::default();
    let mut store = Store::new(&engine, ());

    // Create memory (enough pages for our data)
    let pages = ((total_bytes as usize + 65535) / 65536) as u32;
    let memory_type = wasmtime::MemoryType::new(pages, None);
    let memory = Memory::new(&mut store, memory_type)?;

    // Pre-initialize witness buffer like the VM interpreter does:
    // - witness[0] = Field::ONE (the constant 1)
    // - witness[1..1+num_inputs] = flattened input values
    let flat_inputs: Vec<Field> = params.iter().flat_map(flatten_input_value).collect();
    let num_pre_init = 1 + flat_inputs.len();

    // Write constant ONE at witness[0]
    write_field_to_memory(&memory, &mut store, witness_ptr, <Field as ArkField>::ONE);

    // Write inputs at witness[1..]
    for (i, &field) in flat_inputs.iter().enumerate() {
        let ptr = witness_ptr + ((1 + i) * FIELD_SIZE) as u32;
        write_field_to_memory(&memory, &mut store, ptr, field);
    }

    // Compute the write pointer that skips the pre-initialized entries
    let witness_write_ptr = witness_ptr + (num_pre_init * FIELD_SIZE) as u32;

    // Initialize VM struct with buffer pointers
    // Note: witness_ptr in struct points to where WASM should START writing (after pre-init)
    {
        let data = memory.data_mut(&mut store);
        let off = vm_struct_ptr as usize;
        data[off..off+4].copy_from_slice(&witness_write_ptr.to_le_bytes());
        data[off+4..off+8].copy_from_slice(&a_ptr.to_le_bytes());
        data[off+8..off+12].copy_from_slice(&b_ptr.to_le_bytes());
        data[off+12..off+16].copy_from_slice(&c_ptr.to_le_bytes());
    }

    // Create linker and register imported memory
    let mut linker = Linker::new(&engine);
    linker.define(&store, "env", "memory", memory)?;

    // Load the WASM module (runtime functions are linked in, no host imports needed)
    let wasm_bytes = fs::read(wasm_path)?;
    let module = Module::new(&engine, &wasm_bytes)?;

    // Instantiate and get the main function
    let instance = linker.instantiate(&mut store, &module)?;

    // Build the call arguments: vmPtr followed by flattened input limbs
    // The function signature varies based on inputs, so we need dynamic dispatch
    let func = instance.get_func(&mut store, "mavros_main")
        .ok_or("mavros_main not found")?;

    // Prepare arguments: vmPtr followed by all input field element limbs
    let mut args: Vec<wasmtime::Val> = Vec::new();
    args.push(wasmtime::Val::I32(vm_struct_ptr as i32)); // vmPtr

    for param in params {
        for field in flatten_input_value(param) {
            let limbs = field.0 .0;
            args.push(wasmtime::Val::I64(limbs[0] as i64));
            args.push(wasmtime::Val::I64(limbs[1] as i64));
            args.push(wasmtime::Val::I64(limbs[2] as i64));
            args.push(wasmtime::Val::I64(limbs[3] as i64));
        }
    }

    // Call the function
    let mut results = vec![];
    func.call(&mut store, &args, &mut results)?;

    // Read outputs from memory
    let mut out_witness = Vec::with_capacity(witness_count);
    let mut out_a = Vec::with_capacity(constraint_count);
    let mut out_b = Vec::with_capacity(constraint_count);
    let mut out_c = Vec::with_capacity(constraint_count);

    for i in 0..witness_count {
        let ptr = witness_ptr + (i * FIELD_SIZE) as u32;
        out_witness.push(read_field_from_memory(&memory, &store, ptr));
    }
    for i in 0..constraint_count {
        out_a.push(read_field_from_memory(&memory, &store, a_ptr + (i * FIELD_SIZE) as u32));
        out_b.push(read_field_from_memory(&memory, &store, b_ptr + (i * FIELD_SIZE) as u32));
        out_c.push(read_field_from_memory(&memory, &store, c_ptr + (i * FIELD_SIZE) as u32));
    }

    // Split witness into pre-commit and post-commit sections
    let pre_comm_count = r1cs.witness_layout.pre_commitment_size();
    let out_wit_pre_comm = out_witness[..pre_comm_count].to_vec();
    let out_wit_post_comm = out_witness[pre_comm_count..].to_vec();

    Ok(WasmResult {
        out_wit_pre_comm,
        out_wit_post_comm,
        out_a,
        out_b,
        out_c,
    })
}

// ── Parent: discover & run all tests ──────────────────────────────────

/// The step keys in display order.
const STEP_KEYS: &[&str] = &[
    "COMPILED", "R1CS", "WITGEN_COMPILE", "AD_COMPILE",
    "WITGEN_RUN", "WITGEN_CORRECT", "WITGEN_NOLEAK",
    "AD_RUN", "AD_CORRECT", "AD_NOLEAK",
    "WITGEN_WASM_COMPILE", "WITGEN_WASM_RUN", "WITGEN_WASM_CORRECT",
    "AD_WASM_COMPILE", "AD_WASM_RUN", "AD_WASM_CORRECT",
];

struct TestResult {
    name: String,
    steps: HashMap<String, Status>,
    rows: Option<usize>,
    cols: Option<usize>,
}

/// Determined purely from child output:
/// - `started && ended ok` → Pass
/// - `started && ended fail` → Fail
/// - `started && no end` → Crash
/// - `never started` → Skip
#[derive(Clone, Copy, PartialEq)]
enum Status {
    Pass,
    Fail,
    Crash,
    Skip,
}

impl Status {
    fn emoji(self) -> &'static str {
        match self {
            Status::Pass => "✅",
            Status::Fail => "❌",
            Status::Crash => "💥",
            Status::Skip => "➖",
        }
    }
}

/// Use `cargo metadata` to find the root of the noir git dependency, then
/// return the path to `test_programs/execution_success` inside it.
fn find_noir_execution_success_dir() -> Option<PathBuf> {
    let metadata = MetadataCommand::new().exec().ok()?;
    // Find any package from the noir git repo (e.g. "nargo").
    let noir_pkg = metadata.packages.iter().find(|p| {
        p.source
            .as_ref()
            .is_some_and(|s| s.repr.contains("reilabs/noir"))
    })?;
    // Walk up from the package manifest to find the repo root containing
    // `test_programs/execution_success`.
    let mut dir: &Path = noir_pkg.manifest_path.as_std_path();
    loop {
        dir = dir.parent()?;
        let candidate = dir.join("test_programs").join("execution_success");
        if candidate.is_dir() {
            return Some(candidate);
        }
    }
}

/// A test entry with its absolute path and display name.
struct TestEntry {
    path: PathBuf,
    display_name: String,
}

fn collect_test_dirs(base: &Path, prefix: &str) -> Vec<TestEntry> {
    let Ok(entries) = fs::read_dir(base) else {
        return Vec::new();
    };
    let mut dirs: Vec<TestEntry> = entries
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.is_dir())
        .map(|p| {
            let test_name = p.file_name().unwrap().to_string_lossy().into_owned();
            TestEntry {
                path: p,
                display_name: format!("{prefix}{test_name}"),
            }
        })
        .collect();
    dirs.sort_by(|a, b| a.display_name.cmp(&b.display_name));
    dirs
}

fn run_parent(output_path: &Path) {
    let mut entries: Vec<TestEntry> = Vec::new();

    // 1. Local noir_tests/ directory
    let local_tests = PathBuf::from("noir_tests");
    if local_tests.is_dir() {
        entries.extend(collect_test_dirs(&local_tests, "noir_tests/"));
    }

    // 2. Noir repo test_programs/execution_success (discovered via cargo-metadata)
    if let Some(exec_success) = find_noir_execution_success_dir() {
        eprintln!(
            "Found noir execution_success tests at: {}",
            exec_success.display()
        );
        entries.extend(collect_test_dirs(
            &exec_success,
            "noir/test_programs/execution_success/",
        ));
    } else {
        eprintln!("Warning: could not locate noir test_programs/execution_success via cargo-metadata");
    }

    assert!(!entries.is_empty(), "No test directories found");

    let exe = env::current_exe().expect("Cannot determine own exe path");
    let mut results = Vec::new();

    for entry in &entries {
        let abs = fs::canonicalize(&entry.path).unwrap();
        eprintln!("Running: {}", entry.display_name);

        let mut child = Command::new(&exe)
            .args(["--run-single", abs.to_str().unwrap()])
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
        results.push(parse_child_output(&entry.display_name, &lines));
    }

    let md = render_markdown(&results);
    fs::write(output_path, &md).expect("Cannot write output file");
    eprintln!("Wrote {}", output_path.display());
    print!("{md}");
}

fn parse_child_output(name: &str, lines: &[String]) -> TestResult {
    let mut started = HashMap::<String, bool>::new();
    let mut ended = HashMap::<String, bool>::new();
    let mut rows = None;
    let mut cols = None;

    for line in lines {
        let parts: Vec<&str> = line.split(':').collect();
        match parts.as_slice() {
            ["START", key] => { started.insert(key.to_string(), true); }
            ["END", key, "ok", ..] => {
                ended.insert(key.to_string(), true);
                if *key == "R1CS" && parts.len() >= 5 {
                    rows = parts[3].parse().ok();
                    cols = parts[4].parse().ok();
                }
            }
            ["END", key, "fail"] => { ended.insert(key.to_string(), false); }
            _ => {}
        }
    }

    let steps = STEP_KEYS
        .iter()
        .map(|&key| {
            let status = if let Some(&ok) = ended.get(key) {
                if ok { Status::Pass } else { Status::Fail }
            } else if started.contains_key(key) {
                Status::Crash
            } else {
                Status::Skip
            };
            (key.to_string(), status)
        })
        .collect();

    TestResult { name: name.to_string(), steps, rows, cols }
}

// ── Regression & growth checks ───────────────────────────────────────

struct ParsedRow {
    name: String,
    cells: Vec<String>,
    rows: Option<usize>,
    cols: Option<usize>,
}

fn parse_status_rows(path: &Path) -> Vec<ParsedRow> {
    let content = fs::read_to_string(path)
        .unwrap_or_else(|_| panic!("Cannot read {}", path.display()));
    let mut result = Vec::new();
    for line in content.lines().skip(2) {
        let cells: Vec<String> = line.split('|')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect();
        if cells.len() < 19 { continue; }
        let rows = cells[3].parse().ok();
        let cols = cells[4].parse().ok();
        result.push(ParsedRow {
            name: cells[0].clone(),
            cells,
            rows,
            cols,
        });
    }
    result
}

const REGRESSION_COLS: &[(usize, &str)] = &[
    (1, "Compiled"), (2, "R1CS"),
    (5, "Witgen Compile"), (6, "Witgen Run VM"), (7, "Witgen Correct"), (8, "Witgen No Leak"),
    (9, "AD Compile"), (10, "AD Run VM"), (11, "AD Correct"), (12, "AD No Leak"),
    (13, "Witgen WASM Compile"), (14, "Witgen WASM Run"), (15, "Witgen WASM Correct"),
    (16, "AD WASM Compile"), (17, "AD WASM Run"), (18, "AD WASM Correct"),
];

fn check_regression(baseline_path: &Path, current_path: &Path) -> i32 {
    let baseline = parse_status_rows(baseline_path);
    let current = parse_status_rows(current_path);

    let base_map: HashMap<&str, &ParsedRow> = baseline.iter()
        .map(|r| (r.name.as_str(), r))
        .collect();

    let mut regressions = Vec::new();
    for cur in &current {
        let Some(base) = base_map.get(cur.name.as_str()) else { continue };
        for &(col, col_name) in REGRESSION_COLS {
            let base_val = &base.cells[col];
            let cur_val = &cur.cells[col];
            if base_val == "✅" && cur_val != "✅" {
                regressions.push(format!("  {} / {}: ✅ → {}", cur.name, col_name, cur_val));
            }
        }
    }

    if regressions.is_empty() {
        eprintln!("No regressions detected.");
        0
    } else {
        eprintln!("REGRESSIONS DETECTED:");
        for r in &regressions {
            eprintln!("{r}");
        }
        1
    }
}

fn check_growth(baseline_path: &Path, current_path: &Path) {
    let baseline = parse_status_rows(baseline_path);
    let current = parse_status_rows(current_path);

    let base_map: HashMap<&str, &ParsedRow> = baseline.iter()
        .map(|r| (r.name.as_str(), r))
        .collect();

    // Track stats for existing tests (tests in both baseline and current)
    let mut new_checkmarks = 0usize;
    let mut existing_baseline_checkmarks = 0usize;
    let mut existing_current_checkmarks = 0usize;
    let mut existing_total = 0usize;

    // Track stats for all current tests (including new ones)
    let mut total_current_checkmarks = 0usize;
    let mut total_current_cells = 0usize;

    // Track constraint/witness decreases (good news)
    let mut improvements = Vec::new();

    // Track constraint/witness increases (warnings)
    let mut warnings = Vec::new();

    for cur in &current {
        // Count checkmarkable cells in current (all tests)
        for &(col, _) in REGRESSION_COLS {
            total_current_cells += 1;
            if cur.cells[col] == "✅" {
                total_current_checkmarks += 1;
            }
        }

        let Some(base) = base_map.get(cur.name.as_str()) else { continue };

        // For existing tests: count baseline/current checkmarks and new checkmarks
        for &(col, _) in REGRESSION_COLS {
            existing_total += 1;
            let base_pass = base.cells[col] == "✅";
            let cur_pass = cur.cells[col] == "✅";
            if base_pass {
                existing_baseline_checkmarks += 1;
            }
            if cur_pass {
                existing_current_checkmarks += 1;
            }
            if !base_pass && cur_pass {
                new_checkmarks += 1;
            }
        }

        // Check constraint changes
        if let (Some(br), Some(cr)) = (base.rows, cur.rows) {
            if cr > br {
                warnings.push(format!(
                    "| {} | Constraints | {} | {} | +{} ({:+.1}%) |",
                    cur.name, br, cr, cr - br, (cr as f64 - br as f64) / br as f64 * 100.0
                ));
            } else if cr < br {
                improvements.push(format!(
                    "| {} | Constraints | {} | {} | {} ({:.1}%) |",
                    cur.name, br, cr, cr as i64 - br as i64, (cr as f64 - br as f64) / br as f64 * 100.0
                ));
            }
        }

        // Check witness changes
        if let (Some(bc), Some(cc)) = (base.cols, cur.cols) {
            if cc > bc {
                warnings.push(format!(
                    "| {} | Witnesses | {} | {} | +{} ({:+.1}%) |",
                    cur.name, bc, cc, cc - bc, (cc as f64 - bc as f64) / bc as f64 * 100.0
                ));
            } else if cc < bc {
                improvements.push(format!(
                    "| {} | Witnesses | {} | {} | {} ({:.1}%) |",
                    cur.name, bc, cc, cc as i64 - bc as i64, (cc as f64 - bc as f64) / bc as f64 * 100.0
                ));
            }
        }
    }

    // Calculate completion percentages
    let existing_baseline_pct = if existing_total > 0 {
        existing_baseline_checkmarks as f64 / existing_total as f64 * 100.0
    } else {
        0.0
    };
    let existing_current_pct = if existing_total > 0 {
        existing_current_checkmarks as f64 / existing_total as f64 * 100.0
    } else {
        0.0
    };
    let existing_pct_change = existing_current_pct - existing_baseline_pct;

    let total_current_pct = if total_current_cells > 0 {
        total_current_checkmarks as f64 / total_current_cells as f64 * 100.0
    } else {
        0.0
    };

    // Always print overall success rate
    println!("**Overall success rate on test cases: {:.1}%**\n", total_current_pct);

    // Print positive news section
    let has_positive_news = new_checkmarks > 0 || existing_pct_change > 0.0 || !improvements.is_empty();
    if has_positive_news {
        println!("### Positive Changes\n");

        if new_checkmarks > 0 || existing_pct_change.abs() > 0.001 {
            if new_checkmarks > 0 {
                println!("- {} cell(s) turned into checkmarks ✅", new_checkmarks);
            }
            if existing_pct_change.abs() > 0.001 {
                println!(
                    "- Existing tests: {:.1}% → {:.1}% ({:+.1}%)",
                    existing_baseline_pct, existing_current_pct, existing_pct_change
                );
            }
            println!();
        }

        if !improvements.is_empty() {
            println!("**R1CS constraint or witness count decreased:**\n");
            println!("| Test | Metric | Before | After | Change |");
            println!("|------|--------|--------|-------|--------|");
            for imp in &improvements {
                println!("{imp}");
            }
            println!();
        }
    }

    if warnings.is_empty() {
        if !has_positive_news {
            println!("No test improvements or R1CS constraint/witness count changes detected.");
        } else {
            println!("No R1CS constraint or witness count growth detected.");
        }
        return;
    }

    // Print warnings section
    println!("### Warnings\n");
    println!("**R1CS constraint or witness count growth detected:**\n");
    println!("| Test | Metric | Before | After | Change |");
    println!("|------|--------|--------|-------|--------|");
    for w in &warnings {
        println!("{w}");
    }
}

fn render_markdown(results: &[TestResult]) -> String {
    let mut md = String::new();
    md.push_str("| Test | Compiled | R1CS | Rows | Cols | Witgen Compile | Witgen Run VM | Witgen Correct | Witgen No Leak | AD Compile | AD Run VM | AD Correct | AD No Leak | Witgen WASM Compile | Witgen WASM Run | Witgen WASM Correct | AD WASM Compile | AD WASM Run | AD WASM Correct |\n");
    md.push_str("|------|----------|------|------|------|----------------|---------------|----------------|----------------|------------|-----------|------------|------------|---------------------|-----------------|---------------------|-----------------|-------------|---------------------|\n");

    for r in results {
        let s = |key: &str| r.steps.get(key).copied().unwrap_or(Status::Skip).emoji();
        let rows = r.rows.map_or("-".to_string(), |v| v.to_string());
        let cols = r.cols.map_or("-".to_string(), |v| v.to_string());
        md.push_str(&format!(
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |\n",
            r.name,
            s("COMPILED"),
            s("R1CS"),
            rows,
            cols,
            s("WITGEN_COMPILE"),
            s("WITGEN_RUN"),
            s("WITGEN_CORRECT"),
            s("WITGEN_NOLEAK"),
            s("AD_COMPILE"),
            s("AD_RUN"),
            s("AD_CORRECT"),
            s("AD_NOLEAK"),
            s("WITGEN_WASM_COMPILE"),
            s("WITGEN_WASM_RUN"),
            s("WITGEN_WASM_CORRECT"),
            s("AD_WASM_COMPILE"),
            s("AD_WASM_RUN"),
            s("AD_WASM_CORRECT"),
        ));
    }

    md
}
