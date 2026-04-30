use std::{
    collections::HashMap,
    env, fs,
    io::{BufRead, BufReader, Write},
    path::{Path, PathBuf},
    process::{Command, Stdio},
};

use cargo_metadata::MetadataCommand;

use ark_ff::UniformRand as _;
use mavros::{
    Project, abi_helpers, compiler::Field, compiler::r1cs_gen::R1CS, driver::Driver,
    vm::interpreter,
};
use mavros_wasm_layout::{
    AD_COEFFS_BASE_PTR_OFFSET, AD_COEFFS_PTR_OFFSET, AD_CURRENT_CNST_TABLES_OFF_OFFSET,
    AD_CURRENT_LOOKUP_WIT_OFF_OFFSET, AD_CURRENT_WIT_MULTIPLICITIES_OFF_OFFSET,
    AD_CURRENT_WIT_OFF_OFFSET, AD_CURRENT_WIT_TABLES_OFF_OFFSET, AD_OUT_DA_PTR_OFFSET,
    AD_OUT_DB_PTR_OFFSET, AD_OUT_DC_PTR_OFFSET, AD_VM_STRUCT_SIZE, MAX_TABLE_KINDS,
    TABLE_INFO_INV_CNST_OFF_OFFSET, TABLE_INFO_INV_WIT_OFF_OFFSET, TABLE_INFO_LENGTH_OFFSET,
    TABLE_INFO_MULTS_BASE_PTR_OFFSET, TABLE_INFO_NUM_INDICES_OFFSET,
    TABLE_INFO_NUM_VALUES_OFFSET, TABLE_INFO_OCCUPANCY_OFFSET, TABLE_INFO_SLOT_SIZE,
    TABLE_INFO_TABLE_IDX_OFFSET, WITGEN_A_PTR_OFFSET, WITGEN_B_PTR_OFFSET, WITGEN_C_PTR_OFFSET,
    WITGEN_CURRENT_CNST_TABLES_OFF_OFFSET, WITGEN_CURRENT_WIT_TABLES_OFF_OFFSET,
    WITGEN_INPUTS_PTR_OFFSET, WITGEN_LOOKUPS_A_PTR_OFFSET, WITGEN_LOOKUPS_B_PTR_OFFSET,
    WITGEN_LOOKUPS_C_PTR_OFFSET, WITGEN_MULTS_CURSOR_PTR_OFFSET, WITGEN_TABLES_LEN_OFFSET,
    WITGEN_TABLES_REGISTRY_OFFSET, WITGEN_VM_STRUCT_SIZE, WITGEN_WITNESS_PTR_OFFSET,
};
use noirc_abi::input_parser::Format;
use rand::SeedableRng;
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
        Some(d) => {
            emit("END:COMPILED:ok");
            d
        }
        None => {
            emit("END:COMPILED:fail");
            return;
        }
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
        Err(_) => {
            emit("END:R1CS:fail");
            None
        }
    };

    // 3. Compile witgen  (depends on R1CS)
    let witgen_binary = r1cs.as_ref().and_then(|_| {
        emit("START:WITGEN_COMPILE");
        match driver.compile_witgen() {
            Ok(b) => {
                let bytes = b.len() * 8;
                emit(&format!("END:WITGEN_COMPILE:ok:{bytes}"));
                Some(b)
            }
            Err(_) => {
                emit("END:WITGEN_COMPILE:fail");
                None
            }
        }
    });

    // 4. Compile AD  (depends on R1CS, independent of witgen)
    let ad_binary = r1cs.as_ref().and_then(|_| {
        emit("START:AD_COMPILE");
        match driver.compile_ad() {
            Ok(b) => {
                let bytes = b.len() * 8;
                emit(&format!("END:AD_COMPILE:ok:{bytes}"));
                Some(b)
            }
            Err(_) => {
                emit("END:AD_COMPILE:fail");
                None
            }
        }
    });

    // Load inputs (needed for witgen run)
    let ordered_params = load_inputs(&root.join("Prover.toml"), &driver);

    // 5. Run witgen  (depends on WITGEN_COMPILE)
    let had_witgen_binary = witgen_binary.is_some();
    let witgen_result = witgen_binary.and_then(|mut binary| {
        emit("START:WITGEN_RUN");
        let r1cs = r1cs.as_ref().unwrap();
        let params = ordered_params.as_ref().map(|v| v.as_slice()).unwrap_or(&[]);
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
        emit(if correct {
            "END:WITGEN_CORRECT:ok"
        } else {
            "END:WITGEN_CORRECT:fail"
        });
    }

    // 7. Witgen leak check  (depends on WITGEN_RUN)
    if let Some(result) = &witgen_result {
        emit("START:WITGEN_NOLEAK");
        let leftover = result.instrumenter.final_memory_usage();
        emit(if leftover == 0 {
            "END:WITGEN_NOLEAK:ok"
        } else {
            "END:WITGEN_NOLEAK:fail"
        });
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
        emit(if correct {
            "END:AD_CORRECT:ok"
        } else {
            "END:AD_CORRECT:fail"
        });
    }

    // 10. AD leak check  (depends on AD_RUN)
    if let Some((_, _, _, _, instrumenter)) = &ad_result {
        emit("START:AD_NOLEAK");
        let leftover = instrumenter.final_memory_usage();
        emit(if leftover == 0 {
            "END:AD_NOLEAK:ok"
        } else {
            "END:AD_NOLEAK:fail"
        });
    }

    // 11. Compile WASM  (depends on R1CS)
    let wasm_path = r1cs.as_ref().and_then(|r1cs| {
        emit("START:WITGEN_WASM_COMPILE");
        let tmpdir = tempfile::tempdir().ok()?;
        let wasm_path = tmpdir.into_path().join("witgen.wasm");
        match driver.compile_llvm_targets(false, r1cs, Some(wasm_path.clone())) {
            Ok(_) if wasm_path.exists() => {
                emit("END:WITGEN_WASM_COMPILE:ok");
                Some(wasm_path)
            }
            Ok(_) => {
                eprintln!(
                    "WASM compile succeeded but output file not found at {:?}",
                    wasm_path
                );
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
        let params = ordered_params.as_ref().map(|v| v.as_slice()).unwrap_or(&[]);
        match run_wasm(wasm_path, r1cs, params) {
            Ok(result) => {
                emit("END:WITGEN_WASM_RUN:ok");
                Some(result)
            }
            Err(e) => {
                eprintln!("WASM run error: {:?}", e);
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
        emit(if correct {
            "END:WITGEN_WASM_CORRECT:ok"
        } else {
            "END:WITGEN_WASM_CORRECT:fail"
        });
    }

    // 13b. Witgen WASM leak check  (depends on WITGEN_WASM_RUN)
    if let Some(result) = &wasm_result {
        emit("START:WITGEN_WASM_NOLEAK");
        emit(if result.live_bytes == 0 {
            "END:WITGEN_WASM_NOLEAK:ok"
        } else {
            "END:WITGEN_WASM_NOLEAK:fail"
        });
    }

    // 14. AD WASM Compile  (depends on R1CS)
    let ad_wasm_path: Option<std::path::PathBuf> = r1cs.as_ref().and_then(|r1cs| {
        emit("START:AD_WASM_COMPILE");
        let tmpdir = tempfile::tempdir().ok()?;
        let wasm_path = tmpdir.into_path().join("ad.wasm");
        match driver.compile_ad_llvm_targets(wasm_path.clone(), r1cs) {
            Ok(_) if wasm_path.exists() => {
                emit("END:AD_WASM_COMPILE:ok");
                Some(wasm_path)
            }
            Ok(_) => {
                eprintln!(
                    "AD WASM compile succeeded but output file not found at {:?}",
                    wasm_path
                );
                emit("END:AD_WASM_COMPILE:fail");
                None
            }
            Err(e) => {
                eprintln!("AD WASM compile error: {:?}", e);
                emit("END:AD_WASM_COMPILE:fail");
                None
            }
        }
    });

    // 15. AD WASM Run  (depends on AD_WASM_COMPILE)
    let ad_wasm_result = ad_wasm_path.as_ref().and_then(|wasm_path| {
        emit("START:AD_WASM_RUN");
        let r1cs = r1cs.as_ref().unwrap();
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let ad_coeffs: Vec<Field> = (0..r1cs.constraints.len())
            .map(|_| ark_bn254::Fr::rand(&mut rng))
            .collect();
        match run_ad_wasm(wasm_path, r1cs, &ad_coeffs) {
            Ok(result) => {
                emit("END:AD_WASM_RUN:ok");
                Some((ad_coeffs, result))
            }
            Err(e) => {
                eprintln!("AD WASM run error: {:?}", e);
                emit("END:AD_WASM_RUN:fail");
                None
            }
        }
    });

    // 16. AD WASM Correct  (depends on AD_WASM_RUN)
    if let (Some((coeffs, result)), Some(r1cs)) = (&ad_wasm_result, &r1cs) {
        emit("START:AD_WASM_CORRECT");
        let correct = r1cs.check_ad_output(coeffs, &result.out_da, &result.out_db, &result.out_dc);
        emit(if correct {
            "END:AD_WASM_CORRECT:ok"
        } else {
            "END:AD_WASM_CORRECT:fail"
        });
    }

    // 16b. AD WASM leak check  (depends on AD_WASM_RUN)
    if let Some((_, result)) = &ad_wasm_result {
        emit("START:AD_WASM_NOLEAK");
        emit(if result.live_bytes == 0 {
            "END:AD_WASM_NOLEAK:ok"
        } else {
            "END:AD_WASM_NOLEAK:fail"
        });
    }
}

fn load_inputs(file_path: &Path, driver: &Driver) -> Option<Vec<interpreter::InputValueOrdered>> {
    let ext = file_path.extension().and_then(|e| e.to_str())?;
    let format = Format::from_ext(ext)?;
    let contents = fs::read_to_string(file_path).ok()?;
    let params = format.parse(&contents, driver.abi()).ok()?;
    Some(abi_helpers::ordered_params_from_btreemap(
        driver.abi(),
        &params,
    ))
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
    live_bytes: usize,
}

/// Read a `TableInfo` record back from one slot of the witgen VM struct's
/// table registry in WASM linear memory. Returns `None` if the slot is
/// unclaimed (occupancy == 0). The host has no kind-specific knowledge —
/// it walks all slots up to `tables_len` and reconstructs whatever was
/// claimed at runtime.
fn read_table_info_slot(
    memory: &Memory,
    store: impl wasmtime::AsContext,
    vm_struct_ptr: u32,
    witness_ptr: u32,
    slot: u32,
) -> Option<RuntimeTableInfo> {
    let slot_base = vm_struct_ptr + WITGEN_TABLES_REGISTRY_OFFSET + slot * TABLE_INFO_SLOT_SIZE;
    let occupancy = read_u32_from_memory(memory, &store, slot_base + TABLE_INFO_OCCUPANCY_OFFSET);
    if occupancy == 0 {
        return None;
    }
    let table_idx = read_u32_from_memory(memory, &store, slot_base + TABLE_INFO_TABLE_IDX_OFFSET);
    let mults_base =
        read_u32_from_memory(memory, &store, slot_base + TABLE_INFO_MULTS_BASE_PTR_OFFSET);
    let inv_cnst_off =
        read_u32_from_memory(memory, &store, slot_base + TABLE_INFO_INV_CNST_OFF_OFFSET);
    let inv_wit_off =
        read_u32_from_memory(memory, &store, slot_base + TABLE_INFO_INV_WIT_OFF_OFFSET);
    let num_indices =
        read_u32_from_memory(memory, &store, slot_base + TABLE_INFO_NUM_INDICES_OFFSET);
    let num_values = read_u32_from_memory(memory, &store, slot_base + TABLE_INFO_NUM_VALUES_OFFSET);
    let length = read_u32_from_memory(memory, &store, slot_base + TABLE_INFO_LENGTH_OFFSET);
    let mults_off = mults_base
        .checked_sub(witness_ptr)
        .expect("table multiplicities pointer is before witness base")
        / FIELD_SIZE as u32;
    Some(RuntimeTableInfo {
        table_idx: table_idx as usize,
        multiplicities_wit_off: mults_off as usize,
        elem_inverses_constraint_section_offset: inv_cnst_off as usize,
        elem_inverses_witness_section_offset: inv_wit_off as usize,
        num_indices: num_indices as usize,
        num_values: num_values as usize,
        length: length as usize,
    })
}

/// Decoded contents of one occupied registry slot, in host-friendly types.
/// Mirrors `vm::bytecode::TableInfo` but with `multiplicities_wit_off`
/// (an index into `out_wit_pre_comm`) instead of a raw pointer, so it
/// survives the wasmtime → host buffer copy unchanged.
#[derive(Clone, Copy)]
struct RuntimeTableInfo {
    table_idx: usize,
    multiplicities_wit_off: usize,
    elem_inverses_constraint_section_offset: usize,
    elem_inverses_witness_section_offset: usize,
    num_indices: usize,
    num_values: usize,
    length: usize,
}

fn read_u32_from_memory(memory: &Memory, store: impl wasmtime::AsContext, ptr: u32) -> u32 {
    let data = memory.data(&store);
    let offset = ptr as usize;
    u32::from_le_bytes(data[offset..offset + 4].try_into().unwrap())
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
    let input_fields: Vec<Field> = params.iter().flat_map(flatten_input_value).collect();

    let vm_struct_size: u32 = WITGEN_VM_STRUCT_SIZE;
    let witness_bytes = (witness_count * FIELD_SIZE) as u32;
    let constraint_bytes = (constraint_count * FIELD_SIZE) as u32;
    let input_bytes = (input_fields.len() * FIELD_SIZE) as u32;
    let our_data_size = vm_struct_size + witness_bytes + 3 * constraint_bytes + input_bytes;

    // Create wasmtime engine and store
    let engine = Engine::default();
    let mut store = Store::new(&engine, ());

    // Load the WASM module
    let wasm_bytes = fs::read(wasm_path)?;
    let module = Module::new(&engine, &wasm_bytes)?;

    // Estimate initial memory: stack (64KB) + module data + our buffers + heap headroom
    let initial_estimate = 65536 + 4096 + our_data_size;
    let pages = ((initial_estimate as usize + 65535) / 65536) as u32;
    let memory_type = wasmtime::MemoryType::new(pages.max(4), None);
    let memory = Memory::new(&mut store, memory_type)?;

    // Create linker, register imported memory, and instantiate
    let mut linker = Linker::new(&engine);
    linker.define(&store, "env", "memory", memory)?;
    let instance = linker.instantiate(&mut store, &module)?;

    // Read __data_end from the WASM module to find where the module's static data ends.
    // Our VM struct and buffers must be placed AFTER this to avoid colliding with the
    // module's data segment (which contains allocator metadata, etc.).
    let data_end_global = instance
        .get_global(&mut store, "__data_end")
        .ok_or("__data_end global not found in WASM module")?;
    let data_end = data_end_global
        .get(&mut store)
        .i32()
        .ok_or("__data_end is not i32")? as u32;
    let data_offset = (data_end + 15) & !15; // align to 16 bytes

    // Calculate memory layout after the module's data
    let vm_struct_ptr = data_offset;
    let witness_ptr = vm_struct_ptr + vm_struct_size;
    let a_ptr = witness_ptr + witness_bytes;
    let b_ptr = a_ptr + constraint_bytes;
    let c_ptr = b_ptr + constraint_bytes;
    let inputs_ptr = c_ptr + constraint_bytes;
    let total_bytes = inputs_ptr + input_bytes;

    // Grow memory if needed
    let needed_pages = ((total_bytes as usize + 65535) / 65536) as u32;
    let current_pages = memory.size(&store) as u32;
    if needed_pages > current_pages {
        memory.grow(&mut store, (needed_pages - current_pages) as u64)?;
    }

    // Cursors the host needs to seed. The per-slot table registry is left
    // at its zero-initialized value (occupancy = 0 means "unclaimed");
    // each lookup helper handles its own first-use claim at runtime, so
    // the host has no per-kind knowledge. The two table-region cursors
    // (cnst/wit) are seeded at the structural starts of those regions —
    // first-use lookups snapshot them and bump by their footprint.
    //
    // `current_wit_tables_off` is stored *relative to challenges_start*
    // (matches how Phase 2 uses it: `out_wit_post_comm[wit_base + i]`,
    // where `out_wit_post_comm` starts at challenges_start).
    let mults_cursor_ptr =
        witness_ptr + (r1cs.witness_layout.multiplicities_start() * FIELD_SIZE) as u32;
    let lookups_a_cursor =
        a_ptr + (r1cs.constraints_layout.lookups_data_start() * FIELD_SIZE) as u32;
    let lookups_b_cursor =
        b_ptr + (r1cs.constraints_layout.lookups_data_start() * FIELD_SIZE) as u32;
    let lookups_c_cursor =
        c_ptr + (r1cs.constraints_layout.lookups_data_start() * FIELD_SIZE) as u32;
    let current_cnst_tables_off = r1cs.constraints_layout.tables_data_start() as u32;
    let current_wit_tables_off =
        (r1cs.witness_layout.tables_data_start() - r1cs.witness_layout.challenges_start()) as u32;

    // Initialize VM struct with buffer pointers
    {
        let data = memory.data_mut(&mut store);
        let off = vm_struct_ptr as usize;
        let w = WITGEN_WITNESS_PTR_OFFSET as usize;
        let a = WITGEN_A_PTR_OFFSET as usize;
        let b = WITGEN_B_PTR_OFFSET as usize;
        let c = WITGEN_C_PTR_OFFSET as usize;
        let mc = WITGEN_MULTS_CURSOR_PTR_OFFSET as usize;
        let la = WITGEN_LOOKUPS_A_PTR_OFFSET as usize;
        let lb = WITGEN_LOOKUPS_B_PTR_OFFSET as usize;
        let lc = WITGEN_LOOKUPS_C_PTR_OFFSET as usize;
        let inputs = WITGEN_INPUTS_PTR_OFFSET as usize;
        let cct = WITGEN_CURRENT_CNST_TABLES_OFF_OFFSET as usize;
        let cwt = WITGEN_CURRENT_WIT_TABLES_OFF_OFFSET as usize;
        data[off + w..off + w + 4].copy_from_slice(&witness_ptr.to_le_bytes());
        data[off + a..off + a + 4].copy_from_slice(&a_ptr.to_le_bytes());
        data[off + b..off + b + 4].copy_from_slice(&b_ptr.to_le_bytes());
        data[off + c..off + c + 4].copy_from_slice(&c_ptr.to_le_bytes());
        data[off + mc..off + mc + 4].copy_from_slice(&mults_cursor_ptr.to_le_bytes());
        data[off + la..off + la + 4].copy_from_slice(&lookups_a_cursor.to_le_bytes());
        data[off + lb..off + lb + 4].copy_from_slice(&lookups_b_cursor.to_le_bytes());
        data[off + lc..off + lc + 4].copy_from_slice(&lookups_c_cursor.to_le_bytes());
        data[off + inputs..off + inputs + 4].copy_from_slice(&inputs_ptr.to_le_bytes());
        data[off + cct..off + cct + 4].copy_from_slice(&current_cnst_tables_off.to_le_bytes());
        data[off + cwt..off + cwt + 4].copy_from_slice(&current_wit_tables_off.to_le_bytes());
    }

    for (i, field) in input_fields.iter().enumerate() {
        write_field_to_memory(
            &memory,
            &mut store,
            inputs_ptr + (i * FIELD_SIZE) as u32,
            field,
        );
    }

    let func = instance
        .get_func(&mut store, "mavros_main")
        .ok_or("mavros_main not found")?;

    let args = vec![wasmtime::Val::I32(vm_struct_ptr as i32)];

    // Call the function
    let mut results = vec![];
    func.call(&mut store, &args, &mut results)?;

    // Walk the registry: read every slot up to `MAX_TABLE_KINDS`, keep
    // only the occupied ones. The host does not name any specific lookup
    // kind; each runtime-claimed slot describes itself fully via its
    // `TableInfoSlot` fields.
    let tables_len =
        read_u32_from_memory(&memory, &store, vm_struct_ptr + WITGEN_TABLES_LEN_OFFSET) as usize;
    let mut runtime_tables: Vec<RuntimeTableInfo> = Vec::new();
    for slot in 0..MAX_TABLE_KINDS {
        if let Some(info) =
            read_table_info_slot(&memory, &store, vm_struct_ptr, witness_ptr, slot)
        {
            runtime_tables.push(info);
        }
    }
    assert_eq!(
        runtime_tables.len(),
        tables_len,
        "WASM `tables_len` ({}) doesn't match the number of occupied registry slots ({})",
        tables_len,
        runtime_tables.len()
    );
    // Phase 2 (witgen_phase2) wants `Vec<TableInfo>` indexed by table id;
    // sort so each table lands at its assigned id.
    runtime_tables.sort_by_key(|t| t.table_idx);

    // Read heap residual from the __live_bytes counter in wasm-runtime
    let live_bytes_fn = instance
        .get_func(&mut store, "__live_bytes")
        .ok_or("live_bytes not found")?;
    let live_bytes_args = vec![];
    let mut live_bytes_out = vec![wasmtime::Val::I32(0)];
    live_bytes_fn.call(&mut store, &live_bytes_args, &mut live_bytes_out)?;
    let live_bytes = live_bytes_out[0]
        .i32()
        .ok_or("__live_bytes did not return i32")? as usize;

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
        out_a.push(read_field_from_memory(
            &memory,
            &store,
            a_ptr + (i * FIELD_SIZE) as u32,
        ));
        out_b.push(read_field_from_memory(
            &memory,
            &store,
            b_ptr + (i * FIELD_SIZE) as u32,
        ));
        out_c.push(read_field_from_memory(
            &memory,
            &store,
            c_ptr + (i * FIELD_SIZE) as u32,
        ));
    }

    // Split witness into pre-commit and post-commit sections
    let pre_comm_count = r1cs.witness_layout.pre_commitment_size();
    let out_wit_pre_comm = out_witness[..pre_comm_count].to_vec();
    let out_wit_post_comm = out_witness[pre_comm_count..].to_vec();

    let (out_wit_pre_comm, out_wit_post_comm, out_a, out_b, out_c) =
        if r1cs.witness_layout.challenges_size > 0 {
            let result = witgen_phase2(
                r1cs,
                out_wit_pre_comm,
                out_wit_post_comm,
                out_a,
                out_b,
                out_c,
                runtime_tables,
            );
            (
                result.out_wit_pre_comm,
                result.out_wit_post_comm,
                result.out_a,
                result.out_b,
                result.out_c,
            )
        } else {
            (out_wit_pre_comm, out_wit_post_comm, out_a, out_b, out_c)
        };

    Ok(WasmResult {
        out_wit_pre_comm,
        out_wit_post_comm,
        out_a,
        out_b,
        out_c,
        live_bytes,
    })
}

fn witgen_phase2(
    r1cs: &R1CS,
    mut out_wit_pre_comm: Vec<Field>,
    out_wit_post_comm: Vec<Field>,
    out_a: Vec<Field>,
    out_b: Vec<Field>,
    out_c: Vec<Field>,
    runtime_tables: Vec<RuntimeTableInfo>,
) -> interpreter::WitgenResult {
    use mavros::vm::bytecode::{AllocationInstrumenter, TableInfo};

    let witness_layout = &r1cs.witness_layout;

    // Re-encode raw-u64 multiplicity slots as Montgomery field elements.
    // Walk the runtime-claimed table list rather than scanning the entire
    // multiplicities region: the host doesn't know each table's length
    // statically, only that the runtime registry recorded `length` for
    // every claimed slot.
    for tbl in &runtime_tables {
        let lo = tbl.multiplicities_wit_off;
        let hi = lo + tbl.length;
        for i in lo..hi {
            out_wit_pre_comm[i] = Field::from(out_wit_pre_comm[i].0.0[0]);
        }
    }

    let witness_base = out_wit_pre_comm.as_mut_ptr();
    let tables = if r1cs.constraints_layout.lookups_data_size == 0 {
        vec![]
    } else {
        assert!(
            !runtime_tables.is_empty(),
            "WASM emitted lookup constraints but no tables were claimed at runtime"
        );

        // The runtime claims tables in dynamic-execution order, assigning
        // ids `0..tables_len`. After sorting by `table_idx` the indices
        // are dense, so we can drop them straight into a Vec.
        let mut tables = Vec::with_capacity(runtime_tables.len());
        for (expected_idx, t) in runtime_tables.iter().enumerate() {
            assert_eq!(
                t.table_idx, expected_idx,
                "table ids are not 0..tables_len (got {} at position {})",
                t.table_idx, expected_idx
            );
            tables.push(TableInfo {
                multiplicities_wit: witness_base.wrapping_add(t.multiplicities_wit_off),
                num_indices: t.num_indices,
                num_values: t.num_values,
                length: t.length,
                elem_inverses_witness_section_offset: t.elem_inverses_witness_section_offset,
                elem_inverses_constraint_section_offset: t.elem_inverses_constraint_section_offset,
            });
        }
        tables
    };

    let phase1 = interpreter::Phase1Result {
        out_wit_pre_comm,
        out_wit_post_comm,
        out_a,
        out_b,
        out_c,
        tables,
        instrumenter: AllocationInstrumenter::new(),
    };

    interpreter::run_phase2_with_fake_challenges(
        phase1,
        r1cs.witness_layout,
        r1cs.constraints_layout,
    )
}

// ── AD WASM Runner ───────────────────────────────────────────────────

/// Output from running AD WASM
struct AdWasmResult {
    out_da: Vec<Field>,
    out_db: Vec<Field>,
    out_dc: Vec<Field>,
    live_bytes: usize,
}

/// Write a field element to WASM memory at ptr
fn write_field_to_memory(
    memory: &Memory,
    mut store: impl wasmtime::AsContextMut,
    ptr: u32,
    field: &Field,
) {
    let limbs = field.0.0;
    let offset = ptr as usize;
    let data = memory.data_mut(&mut store);
    data[offset..offset + 8].copy_from_slice(&limbs[0].to_le_bytes());
    data[offset + 8..offset + 16].copy_from_slice(&limbs[1].to_le_bytes());
    data[offset + 16..offset + 24].copy_from_slice(&limbs[2].to_le_bytes());
    data[offset + 24..offset + 32].copy_from_slice(&limbs[3].to_le_bytes());
}

fn run_ad_wasm(
    wasm_path: &Path,
    r1cs: &R1CS,
    coeffs: &[Field],
) -> Result<AdWasmResult, Box<dyn std::error::Error>> {
    let witness_count = r1cs.witness_layout.size();
    let constraint_count = r1cs.constraints.len();

    let vm_struct_size: u32 = AD_VM_STRUCT_SIZE;
    let da_bytes = (witness_count * FIELD_SIZE) as u32;
    let db_bytes = da_bytes;
    let dc_bytes = da_bytes;
    let coeffs_bytes = (constraint_count * FIELD_SIZE) as u32;
    let our_data_size = vm_struct_size + da_bytes + db_bytes + dc_bytes + coeffs_bytes;

    let engine = Engine::default();
    let mut store = Store::new(&engine, ());

    let wasm_bytes = fs::read(wasm_path)?;
    let module = Module::new(&engine, &wasm_bytes)?;

    let initial_estimate = 65536 + 4096 + our_data_size;
    let pages = ((initial_estimate as usize + 65535) / 65536) as u32;
    let memory_type = wasmtime::MemoryType::new(pages.max(4), None);
    let memory = Memory::new(&mut store, memory_type)?;

    let mut linker = Linker::new(&engine);
    linker.define(&store, "env", "memory", memory)?;
    let instance = linker.instantiate(&mut store, &module)?;

    let data_end_global = instance
        .get_global(&mut store, "__data_end")
        .ok_or("__data_end global not found in WASM module")?;
    let data_end = data_end_global
        .get(&mut store)
        .i32()
        .ok_or("__data_end is not i32")? as u32;
    let data_offset = (data_end + 15) & !15;

    // Layout buffers
    let vm_struct_ptr = data_offset;
    let da_ptr = vm_struct_ptr + vm_struct_size;
    let db_ptr = da_ptr + da_bytes;
    let dc_ptr = db_ptr + db_bytes;
    let coeffs_ptr = dc_ptr + dc_bytes;
    let total_bytes = coeffs_ptr + coeffs_bytes;

    // Grow memory if needed
    let needed_pages = ((total_bytes as usize + 65535) / 65536) as u32;
    let current_pages = memory.size(&store) as u32;
    if needed_pages > current_pages {
        memory.grow(&mut store, (needed_pages - current_pages) as u64)?;
    }

    // Zero out dA, dB, dC buffers
    {
        let data = memory.data_mut(&mut store);
        let start = da_ptr as usize;
        let end = (dc_ptr + dc_bytes) as usize;
        for b in &mut data[start..end] {
            *b = 0;
        }
    }

    // Write coefficients into WASM memory
    for (i, coeff) in coeffs.iter().enumerate() {
        write_field_to_memory(
            &memory,
            &mut store,
            coeffs_ptr + (i * FIELD_SIZE) as u32,
            coeff,
        );
    }

    // AD lookups need an absolute base for random-access coefficient reads
    // and a fresh-witness counter seeded at the lookups-section start. The
    // three table-allocation cursors (`current_*_tables_off`,
    // `current_wit_multiplicities_off`) are seeded at structural layout
    // starts; first-use lookup helpers snapshot them and then bump by their
    // own table footprint — mirrors `vm.data.as_ad.current_*_off`. The
    // per-slot table registry (occupancy + inv_cnst_off arrays) is left at
    // its zero-initialized value, since wasmtime zeros memory and zero
    // means "unclaimed."
    let lookups_wit_start = r1cs.witness_layout.lookups_data_start() as u32;
    let cnst_tables_start = r1cs.constraints_layout.tables_data_start() as u32;
    let wit_tables_start = r1cs.witness_layout.tables_data_start() as u32;
    let wit_mults_start = r1cs.witness_layout.multiplicities_start() as u32;

    // Initialize AD VM struct
    {
        let data = memory.data_mut(&mut store);
        let off = vm_struct_ptr as usize;
        let da = AD_OUT_DA_PTR_OFFSET as usize;
        let db = AD_OUT_DB_PTR_OFFSET as usize;
        let dc = AD_OUT_DC_PTR_OFFSET as usize;
        let coeffs = AD_COEFFS_PTR_OFFSET as usize;
        let wit = AD_CURRENT_WIT_OFF_OFFSET as usize;
        let cbase = AD_COEFFS_BASE_PTR_OFFSET as usize;
        let lwit = AD_CURRENT_LOOKUP_WIT_OFF_OFFSET as usize;
        let cct = AD_CURRENT_CNST_TABLES_OFF_OFFSET as usize;
        let cwt = AD_CURRENT_WIT_TABLES_OFF_OFFSET as usize;
        let cwm = AD_CURRENT_WIT_MULTIPLICITIES_OFF_OFFSET as usize;
        data[off + da..off + da + 4].copy_from_slice(&da_ptr.to_le_bytes());
        data[off + db..off + db + 4].copy_from_slice(&db_ptr.to_le_bytes());
        data[off + dc..off + dc + 4].copy_from_slice(&dc_ptr.to_le_bytes());
        data[off + coeffs..off + coeffs + 4].copy_from_slice(&coeffs_ptr.to_le_bytes());
        data[off + wit..off + wit + 4].copy_from_slice(&0u32.to_le_bytes());
        data[off + cbase..off + cbase + 4].copy_from_slice(&coeffs_ptr.to_le_bytes());
        data[off + lwit..off + lwit + 4].copy_from_slice(&lookups_wit_start.to_le_bytes());
        data[off + cct..off + cct + 4].copy_from_slice(&cnst_tables_start.to_le_bytes());
        data[off + cwt..off + cwt + 4].copy_from_slice(&wit_tables_start.to_le_bytes());
        data[off + cwm..off + cwm + 4].copy_from_slice(&wit_mults_start.to_le_bytes());
    }

    let func: wasmtime::Func = instance
        .get_func(&mut store, "mavros_main")
        .ok_or("mavros_main not found")?;

    // AD main takes only vm_ptr (no input parameters)
    let args = vec![wasmtime::Val::I32(vm_struct_ptr as i32)];
    let mut results = vec![];
    func.call(&mut store, &args, &mut results)?;

    let live_bytes_fn = instance
        .get_func(&mut store, "__live_bytes")
        .ok_or("live_bytes not found")?;
    let live_bytes_args = vec![];
    let mut live_bytes_out = vec![wasmtime::Val::I32(0)];
    live_bytes_fn.call(&mut store, &live_bytes_args, &mut live_bytes_out)?;
    let live_bytes = live_bytes_out[0]
        .i32()
        .ok_or("__live_bytes did not return i32")? as usize;

    // Read dA, dB, dC from memory
    let mut out_da = Vec::with_capacity(witness_count);
    let mut out_db = Vec::with_capacity(witness_count);
    let mut out_dc = Vec::with_capacity(witness_count);

    for i in 0..witness_count {
        out_da.push(read_field_from_memory(
            &memory,
            &store,
            da_ptr + (i * FIELD_SIZE) as u32,
        ));
        out_db.push(read_field_from_memory(
            &memory,
            &store,
            db_ptr + (i * FIELD_SIZE) as u32,
        ));
        out_dc.push(read_field_from_memory(
            &memory,
            &store,
            dc_ptr + (i * FIELD_SIZE) as u32,
        ));
    }

    Ok(AdWasmResult {
        out_da,
        out_db,
        out_dc,
        live_bytes,
    })
}

// ── Parent: discover & run all tests ──────────────────────────────────

/// The step keys in display order.
const STEP_KEYS: &[&str] = &[
    "COMPILED",
    "R1CS",
    "WITGEN_COMPILE",
    "AD_COMPILE",
    "WITGEN_RUN",
    "WITGEN_CORRECT",
    "WITGEN_NOLEAK",
    "AD_RUN",
    "AD_CORRECT",
    "AD_NOLEAK",
    "WITGEN_WASM_COMPILE",
    "WITGEN_WASM_RUN",
    "WITGEN_WASM_CORRECT",
    "WITGEN_WASM_NOLEAK",
    "AD_WASM_COMPILE",
    "AD_WASM_RUN",
    "AD_WASM_CORRECT",
    "AD_WASM_NOLEAK",
];

struct TestResult {
    name: String,
    steps: HashMap<String, Status>,
    rows: Option<usize>,
    cols: Option<usize>,
    witgen_bytes: Option<usize>,
    ad_bytes: Option<usize>,
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
            .is_some_and(|s| s.repr.contains("noir-lang/noir") || s.repr.contains("reilabs/noir"))
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
        eprintln!(
            "Warning: could not locate noir test_programs/execution_success via cargo-metadata"
        );
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
    let mut witgen_bytes = None;
    let mut ad_bytes = None;

    for line in lines {
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
                if *key == "WITGEN_COMPILE" && parts.len() >= 4 {
                    witgen_bytes = parts[3].parse().ok();
                }
                if *key == "AD_COMPILE" && parts.len() >= 4 {
                    ad_bytes = parts[3].parse().ok();
                }
            }
            ["END", key, "fail"] => {
                ended.insert(key.to_string(), false);
            }
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

    TestResult {
        name: name.to_string(),
        steps,
        rows,
        cols,
        witgen_bytes,
        ad_bytes,
    }
}

// ── Regression & growth checks ───────────────────────────────────────

struct ParsedRow {
    name: String,
    cells: Vec<String>,
    rows: Option<usize>,
    cols: Option<usize>,
    witgen_bytes: Option<usize>,
    ad_bytes: Option<usize>,
}

fn parse_status_rows(path: &Path) -> Vec<ParsedRow> {
    let content =
        fs::read_to_string(path).unwrap_or_else(|_| panic!("Cannot read {}", path.display()));
    let mut result = Vec::new();
    for line in content.lines().skip(2) {
        let cells: Vec<String> = line
            .split('|')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect();
        if cells.len() < 23 {
            continue;
        }
        let rows = cells[3].parse().ok();
        let cols = cells[4].parse().ok();
        let witgen_bytes = cells[5].parse().ok();
        let ad_bytes = cells[6].parse().ok();
        result.push(ParsedRow {
            name: cells[0].clone(),
            cells,
            rows,
            cols,
            witgen_bytes,
            ad_bytes,
        });
    }
    result
}

const REGRESSION_COLS: &[(usize, &str)] = &[
    (1, "Compiled"),
    (2, "R1CS"),
    (7, "Witgen Compile"),
    (8, "Witgen Run VM"),
    (9, "Witgen Correct"),
    (10, "Witgen No Leak"),
    (11, "AD Compile"),
    (12, "AD Run VM"),
    (13, "AD Correct"),
    (14, "AD No Leak"),
    (15, "Witgen WASM Compile"),
    (16, "Witgen WASM Run"),
    (17, "Witgen WASM Correct"),
    (18, "Witgen WASM No Leak"),
    (19, "AD WASM Compile"),
    (20, "AD WASM Run"),
    (21, "AD WASM Correct"),
    (22, "AD WASM No Leak"),
];

fn check_regression(baseline_path: &Path, current_path: &Path) -> i32 {
    let baseline = parse_status_rows(baseline_path);
    let current = parse_status_rows(current_path);

    let base_map: HashMap<&str, &ParsedRow> =
        baseline.iter().map(|r| (r.name.as_str(), r)).collect();

    let mut regressions = Vec::new();
    for cur in &current {
        let Some(base) = base_map.get(cur.name.as_str()) else {
            continue;
        };
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

    let base_map: HashMap<&str, &ParsedRow> =
        baseline.iter().map(|r| (r.name.as_str(), r)).collect();

    // Track stats for existing tests (tests in both baseline and current)
    let mut new_checkmarks: Vec<(String, &str)> = Vec::new(); // (test_name, col_name)
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

        let Some(base) = base_map.get(cur.name.as_str()) else {
            continue;
        };

        // For existing tests: count baseline/current checkmarks and new checkmarks
        for &(col, col_name) in REGRESSION_COLS {
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
                new_checkmarks.push((cur.name.clone(), col_name));
            }
        }

        // Check constraint changes
        if let (Some(br), Some(cr)) = (base.rows, cur.rows) {
            if cr > br {
                warnings.push(format!(
                    "| {} | Constraints | {} | {} | +{} ({:+.1}%) |",
                    cur.name,
                    br,
                    cr,
                    cr - br,
                    (cr as f64 - br as f64) / br as f64 * 100.0
                ));
            } else if cr < br {
                improvements.push(format!(
                    "| {} | Constraints | {} | {} | {} ({:.1}%) |",
                    cur.name,
                    br,
                    cr,
                    cr as i64 - br as i64,
                    (cr as f64 - br as f64) / br as f64 * 100.0
                ));
            }
        }

        // Check witness changes
        if let (Some(bc), Some(cc)) = (base.cols, cur.cols) {
            if cc > bc {
                warnings.push(format!(
                    "| {} | Witnesses | {} | {} | +{} ({:+.1}%) |",
                    cur.name,
                    bc,
                    cc,
                    cc - bc,
                    (cc as f64 - bc as f64) / bc as f64 * 100.0
                ));
            } else if cc < bc {
                improvements.push(format!(
                    "| {} | Witnesses | {} | {} | {} ({:.1}%) |",
                    cur.name,
                    bc,
                    cc,
                    cc as i64 - bc as i64,
                    (cc as f64 - bc as f64) / bc as f64 * 100.0
                ));
            }
        }

        // Check witgen bytecode size changes
        if let (Some(bw), Some(cw)) = (base.witgen_bytes, cur.witgen_bytes) {
            if cw > bw {
                warnings.push(format!(
                    "| {} | Witgen Size (bytes) | {} | {} | +{} ({:+.1}%) |",
                    cur.name,
                    bw,
                    cw,
                    cw - bw,
                    (cw as f64 - bw as f64) / bw as f64 * 100.0
                ));
            } else if cw < bw {
                improvements.push(format!(
                    "| {} | Witgen Size (bytes) | {} | {} | {} ({:.1}%) |",
                    cur.name,
                    bw,
                    cw,
                    cw as i64 - bw as i64,
                    (cw as f64 - bw as f64) / bw as f64 * 100.0
                ));
            }
        }

        // Check AD bytecode size changes
        if let (Some(ba), Some(ca)) = (base.ad_bytes, cur.ad_bytes) {
            if ca > ba {
                warnings.push(format!(
                    "| {} | AD Size (bytes) | {} | {} | +{} ({:+.1}%) |",
                    cur.name,
                    ba,
                    ca,
                    ca - ba,
                    (ca as f64 - ba as f64) / ba as f64 * 100.0
                ));
            } else if ca < ba {
                improvements.push(format!(
                    "| {} | AD Size (bytes) | {} | {} | {} ({:.1}%) |",
                    cur.name,
                    ba,
                    ca,
                    ca as i64 - ba as i64,
                    (ca as f64 - ba as f64) / ba as f64 * 100.0
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
    println!(
        "**Overall success rate on test cases: {:.1}%**\n",
        total_current_pct
    );

    // Print positive news section
    let has_positive_news =
        !new_checkmarks.is_empty() || existing_pct_change > 0.0 || !improvements.is_empty();
    if has_positive_news {
        println!("### Positive Changes\n");

        if !new_checkmarks.is_empty() || existing_pct_change.abs() > 0.001 {
            if !new_checkmarks.is_empty() {
                println!(
                    "<details>\n<summary>{} cell(s) turned into checkmarks ✅</summary>\n",
                    new_checkmarks.len()
                );
                for (test, col) in &new_checkmarks {
                    println!("- **{}** / {}", test, col);
                }
                println!("\n</details>\n");
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
            println!("<details>");
            println!("<summary><b>R1CS/bytecode size decreased</b></summary>\n");
            println!("| Test | Metric | Before | After | Change |");
            println!("|------|--------|--------|-------|--------|");
            for imp in &improvements {
                println!("{imp}");
            }
            println!("\n</details>\n");
        }
    }

    if warnings.is_empty() {
        if !has_positive_news {
            println!("No test improvements or R1CS/bytecode size changes detected.");
        } else {
            println!("No R1CS/bytecode size growth detected.");
        }
        return;
    }

    // Print warnings section
    println!("### Warnings\n");
    println!("<details>");
    println!("<summary><b>R1CS/bytecode size growth detected</b></summary>\n");
    println!("| Test | Metric | Before | After | Change |");
    println!("|------|--------|--------|-------|--------|");
    for w in &warnings {
        println!("{w}");
    }
    println!("\n</details>");
}

fn render_markdown(results: &[TestResult]) -> String {
    let mut md = String::new();
    md.push_str("| Test | Compiled | R1CS | Rows | Cols | Witgen Size | AD Size | Witgen Compile | Witgen Run VM | Witgen Correct | Witgen No Leak | AD Compile | AD Run VM | AD Correct | AD No Leak | Witgen WASM Compile | Witgen WASM Run | Witgen WASM Correct | Witgen WASM No Leak | AD WASM Compile | AD WASM Run | AD WASM Correct | AD WASM No Leak |\n");
    md.push_str("|------|----------|------|------|------|-------------|---------|----------------|---------------|----------------|----------------|------------|-----------|------------|------------|---------------------|-----------------|---------------------|---------------------|-----------------|-------------|---------------------|---------------------|\n");

    for r in results {
        let s = |key: &str| r.steps.get(key).copied().unwrap_or(Status::Skip).emoji();
        let rows = r.rows.map_or("-".to_string(), |v| v.to_string());
        let cols = r.cols.map_or("-".to_string(), |v| v.to_string());
        let witgen_sz = r.witgen_bytes.map_or("-".to_string(), |v| v.to_string());
        let ad_sz = r.ad_bytes.map_or("-".to_string(), |v| v.to_string());
        md.push_str(&format!(
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |\n",
            r.name,
            s("COMPILED"),
            s("R1CS"),
            rows,
            cols,
            witgen_sz,
            ad_sz,
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
            s("WITGEN_WASM_NOLEAK"),
            s("AD_WASM_COMPILE"),
            s("AD_WASM_RUN"),
            s("AD_WASM_CORRECT"),
            s("AD_WASM_NOLEAK"),
        ));
    }

    md
}
