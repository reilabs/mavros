//! Running a compiled WASM artifact on the host, and reading its results back out of linear memory.
//!
//! The witgen and AD entry points of a compiled module take pointers into the module's own linear
//! memory, so calling one means laying out the buffers it writes into, placing them past the
//! module's static data, and reading the results back field element by field element.

use std::path::Path;

use mavros_artifacts::Field as RawField;
use mavros_wasm_layout::{
    AD_COEFFS_BASE_PTR_OFFSET, AD_COEFFS_PTR_OFFSET, AD_CURRENT_CNST_TABLES_OFF_OFFSET,
    AD_CURRENT_LOOKUP_WIT_OFF_OFFSET, AD_CURRENT_WIT_MULTIPLICITIES_OFF_OFFSET,
    AD_CURRENT_WIT_OFF_OFFSET, AD_CURRENT_WIT_TABLES_OFF_OFFSET, AD_OUT_DA_PTR_OFFSET,
    AD_OUT_DB_PTR_OFFSET, AD_OUT_DC_PTR_OFFSET, AD_VM_STRUCT_SIZE, TABLE_INFO_INV_CNST_OFF_OFFSET,
    TABLE_INFO_INV_WIT_OFF_OFFSET, TABLE_INFO_KIND_OFFSET, TABLE_INFO_LENGTH_OFFSET,
    TABLE_INFO_MULTS_BASE_PTR_OFFSET, TABLE_INFO_NUM_INDICES_OFFSET, TABLE_INFO_SLOT_SIZE,
    WITGEN_A_BASE_PTR_OFFSET, WITGEN_A_PTR_OFFSET, WITGEN_B_PTR_OFFSET, WITGEN_C_PTR_OFFSET,
    WITGEN_CURRENT_CNST_TABLES_OFF_OFFSET, WITGEN_CURRENT_WIT_TABLES_OFF_OFFSET,
    WITGEN_INPUTS_PTR_OFFSET, WITGEN_LOOKUPS_A_PTR_OFFSET, WITGEN_LOOKUPS_B_PTR_OFFSET,
    WITGEN_LOOKUPS_C_PTR_OFFSET, WITGEN_MULTS_CURSOR_PTR_OFFSET, WITGEN_TABLES_CAP_OFFSET,
    WITGEN_TABLES_LEN_OFFSET, WITGEN_TABLES_PTR_OFFSET, WITGEN_VM_STRUCT_SIZE,
    WITGEN_WITNESS_PTR_OFFSET,
};
use wasmtime::{Config, Engine, Linker, Memory, Store, WasmBacktraceDetails};

use crate::{
    compiler::codegen::hlssa_to_r1cs::R1CS,
    vm::{TableKind, bytecode::TableInfo, interpreter},
    wasm_runtime,
};

/// A wasmtime engine configured the way every lane here runs: with backtrace details, so a trap
/// names the guest frame that caused it.
pub fn wasm_engine() -> wasmtime::Result<Engine> {
    let mut config = Config::new();
    config.wasm_backtrace_details(WasmBacktraceDetails::Enable);
    Engine::new(&config)
}

// FIELD-ASSUMPTION: L3-field-size
const FIELD_SIZE: usize = 32; // 4 x i64 = 32 bytes

/// The `env::memory` type the module imports. Its minimum covers the module's stack and data.
fn imported_memory_type(
    module: &wasmtime::Module,
) -> Result<wasmtime::MemoryType, Box<dyn std::error::Error>> {
    module
        .imports()
        .find(|import| import.module() == "env" && import.name() == "memory")
        .and_then(|import| match import.ty() {
            wasmtime::ExternType::Memory(memory_type) => Some(memory_type),
            _ => None,
        })
        .ok_or_else(|| "WASM module does not import env::memory".into())
}

/// What one witgen run of a compiled module produced.
pub struct WasmResult {
    pub out_wit_pre_comm: Vec<RawField>,
    pub out_wit_post_comm: Vec<RawField>,
    pub out_a: Vec<RawField>,
    pub out_b: Vec<RawField>,
    pub out_c: Vec<RawField>,
    pub live_bytes: usize,
}

/// Read a `TableInfo` record back from one slot of the witgen VM struct's table registry in WASM
/// linear memory. Slot `i` is runtime table id `i`, matching the VM's `Vec<TableInfo>` order.
fn read_table_info_slot(
    memory: &Memory,
    store: impl wasmtime::AsContext,
    tables_ptr: u32,
    wasm_witness_ptr: u32,
    host_witness_base: *mut RawField,
    table_idx: u32,
) -> (usize, TableInfo) {
    let slot_base = tables_ptr + table_idx * TABLE_INFO_SLOT_SIZE;
    let mults_base =
        read_u32_from_memory(memory, &store, slot_base + TABLE_INFO_MULTS_BASE_PTR_OFFSET);
    let inv_cnst_off =
        read_u32_from_memory(memory, &store, slot_base + TABLE_INFO_INV_CNST_OFF_OFFSET);
    let inv_wit_off =
        read_u32_from_memory(memory, &store, slot_base + TABLE_INFO_INV_WIT_OFF_OFFSET);
    let num_indices =
        read_u32_from_memory(memory, &store, slot_base + TABLE_INFO_NUM_INDICES_OFFSET);
    let kind_code = read_u32_from_memory(memory, &store, slot_base + TABLE_INFO_KIND_OFFSET);
    let length = read_u32_from_memory(memory, &store, slot_base + TABLE_INFO_LENGTH_OFFSET);

    let mults_off = mults_base
        .checked_sub(wasm_witness_ptr)
        .expect("table multiplicities pointer is before witness base")
        / FIELD_SIZE as u32; // FIELD-ASSUMPTION: L3-field-size
    (
        mults_off as usize,
        TableInfo {
            multiplicities_wit: host_witness_base.wrapping_add(mults_off as usize),
            num_indices: num_indices as usize,
            kind: TableKind::from_code(kind_code),
            length: length as usize,
            elem_inverses_witness_section_offset: inv_wit_off as usize,
            elem_inverses_constraint_section_offset: inv_cnst_off as usize,
        },
    )
}

fn read_u32_from_memory(memory: &Memory, store: impl wasmtime::AsContext, ptr: u32) -> u32 {
    let data = memory.data(&store);
    let offset = ptr as usize;
    u32::from_le_bytes(data[offset..offset + 4].try_into().unwrap())
}

/// Read a field element from WASM memory
fn read_field_from_memory(memory: &Memory, store: impl wasmtime::AsContext, ptr: u32) -> RawField {
    // FIELD-ASSUMPTION: L3-frame
    use ark_ff::BigInt;
    let data = memory.data(&store);
    let offset = ptr as usize;
    let l0 = u64::from_le_bytes(data[offset..offset + 8].try_into().unwrap());
    let l1 = u64::from_le_bytes(data[offset + 8..offset + 16].try_into().unwrap());
    let l2 = u64::from_le_bytes(data[offset + 16..offset + 24].try_into().unwrap());
    let l3 = u64::from_le_bytes(data[offset + 24..offset + 32].try_into().unwrap());
    ark_bn254::Fr::new_unchecked(BigInt::new([l0, l1, l2, l3]))
}

/// Run a compiled module's witgen entry point and read its outputs back out of linear memory.
pub fn run_witgen(
    wasm_path: &Path,
    r1cs: &R1CS,
    params: &[interpreter::InputValueOrdered],
) -> Result<WasmResult, Box<dyn std::error::Error>> {
    let witness_count = r1cs.witness_layout.size();
    let constraint_count = r1cs.constraints.len();
    let input_fields: Vec<RawField> = interpreter::flatten_param_vec(params);

    let vm_struct_size: u32 = WITGEN_VM_STRUCT_SIZE;
    // FIELD-ASSUMPTION: L3-field-size
    let witness_bytes = (witness_count * FIELD_SIZE) as u32;
    let constraint_bytes = (constraint_count * FIELD_SIZE) as u32;
    let input_bytes = (input_fields.len() * FIELD_SIZE) as u32;
    let tables_cap = r1cs.constraints_layout.tables_data_size as u32;
    let table_info_bytes = tables_cap * TABLE_INFO_SLOT_SIZE;

    // Create wasmtime engine and store
    let engine = wasm_engine()?;
    let mut store = Store::new(&engine, ());

    // Load the WASM module
    let module = wasm_runtime::load_wasmtime_module(&engine, wasm_path)?;

    let memory = Memory::new(&mut store, imported_memory_type(&module)?)?;

    // Create linker, register imported memory, and instantiate
    let mut linker = Linker::new(&engine);
    linker.define(&store, "env", "memory", memory)?;
    let instance = linker.instantiate(&mut store, &module)?;

    // Read __data_end from the WASM module to find where the module's static data ends. Our VM
    // struct and buffers must be placed AFTER this to avoid colliding with the module's data
    // segment (which contains allocator metadata, etc).
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
    let tables_ptr = inputs_ptr + input_bytes;
    let total_bytes = tables_ptr + table_info_bytes;

    // Grow memory if needed
    let needed_pages = ((total_bytes as usize + 65535) / 65536) as u32;
    let current_pages = memory.size(&store) as u32;
    if needed_pages > current_pages {
        memory.grow(&mut store, (needed_pages - current_pages) as u64)?;
    }

    // Cursors the host needs to seed. The table registry is written by first-use lookup helpers.
    // The two table-region cursors (cnst/wit) are seeded at the structural starts of those regions.
    //
    // `current_wit_tables_off` is stored _relative to_ `challenges_start` (matches how Phase 2 uses
    // it: `out_wit_post_comm[wit_base + i]`, where `out_wit_post_comm` starts at challenges_start).
    let mults_cursor_ptr =
        witness_ptr + (r1cs.witness_layout.multiplicities_start() * FIELD_SIZE) as u32; // FIELD-ASSUMPTION: L3-field-size
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
        let a_base = WITGEN_A_BASE_PTR_OFFSET as usize;
        let b = WITGEN_B_PTR_OFFSET as usize;
        let c = WITGEN_C_PTR_OFFSET as usize;
        let mc = WITGEN_MULTS_CURSOR_PTR_OFFSET as usize;
        let la = WITGEN_LOOKUPS_A_PTR_OFFSET as usize;
        let lb = WITGEN_LOOKUPS_B_PTR_OFFSET as usize;
        let lc = WITGEN_LOOKUPS_C_PTR_OFFSET as usize;
        let inputs = WITGEN_INPUTS_PTR_OFFSET as usize;
        let tcap = WITGEN_TABLES_CAP_OFFSET as usize;
        let tptr = WITGEN_TABLES_PTR_OFFSET as usize;
        let cct = WITGEN_CURRENT_CNST_TABLES_OFF_OFFSET as usize;
        let cwt = WITGEN_CURRENT_WIT_TABLES_OFF_OFFSET as usize;
        data[off + w..off + w + 4].copy_from_slice(&witness_ptr.to_le_bytes());
        data[off + a..off + a + 4].copy_from_slice(&a_ptr.to_le_bytes());
        data[off + a_base..off + a_base + 4].copy_from_slice(&a_ptr.to_le_bytes());
        data[off + b..off + b + 4].copy_from_slice(&b_ptr.to_le_bytes());
        data[off + c..off + c + 4].copy_from_slice(&c_ptr.to_le_bytes());
        data[off + mc..off + mc + 4].copy_from_slice(&mults_cursor_ptr.to_le_bytes());
        data[off + la..off + la + 4].copy_from_slice(&lookups_a_cursor.to_le_bytes());
        data[off + lb..off + lb + 4].copy_from_slice(&lookups_b_cursor.to_le_bytes());
        data[off + lc..off + lc + 4].copy_from_slice(&lookups_c_cursor.to_le_bytes());
        data[off + inputs..off + inputs + 4].copy_from_slice(&inputs_ptr.to_le_bytes());
        data[off + tcap..off + tcap + 4].copy_from_slice(&tables_cap.to_le_bytes());
        data[off + tptr..off + tptr + 4].copy_from_slice(&tables_ptr.to_le_bytes());
        data[off + cct..off + cct + 4].copy_from_slice(&current_cnst_tables_off.to_le_bytes());
        data[off + cwt..off + cwt + 4].copy_from_slice(&current_wit_tables_off.to_le_bytes());
    }

    for (i, field) in input_fields.iter().enumerate() {
        write_field_to_memory(
            &memory,
            &mut store,
            inputs_ptr + (i * FIELD_SIZE) as u32, // FIELD-ASSUMPTION: L3-field-size
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
        // FIELD-ASSUMPTION: L3-field-size
        let ptr = witness_ptr + (i * FIELD_SIZE) as u32;
        out_witness.push(read_field_from_memory(&memory, &store, ptr));
    }
    for i in 0..constraint_count {
        out_a.push(read_field_from_memory(
            &memory,
            &store,
            a_ptr + (i * FIELD_SIZE) as u32, // FIELD-ASSUMPTION: L3-field-size
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
    let mut out_wit_pre_comm = out_witness[..pre_comm_count].to_vec();
    let out_wit_post_comm = out_witness[pre_comm_count..].to_vec();

    // Walk the registry in table-id order. The host does not name any specific lookup kind; each
    // runtime-claimed slot describes itself fully via its `TableInfoSlot` fields. `TableInfo`
    // stores a host pointer, so build it only after `out_wit_pre_comm` exists.
    let tables_len =
        read_u32_from_memory(&memory, &store, vm_struct_ptr + WITGEN_TABLES_LEN_OFFSET) as usize;
    assert!(
        tables_len <= tables_cap as usize,
        "WASM `tables_len` ({}) exceeds registry capacity ({})",
        tables_len,
        tables_cap
    );
    let host_witness_base = out_wit_pre_comm.as_mut_ptr();
    let mut runtime_tables: Vec<(usize, TableInfo)> = Vec::with_capacity(tables_len);
    for table_idx in 0..tables_len as u32 {
        runtime_tables.push(read_table_info_slot(
            &memory,
            &store,
            tables_ptr,
            witness_ptr,
            host_witness_base,
            table_idx,
        ));
    }

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
    mut out_wit_pre_comm: Vec<RawField>,
    out_wit_post_comm: Vec<RawField>,
    out_a: Vec<RawField>,
    out_b: Vec<RawField>,
    out_c: Vec<RawField>,
    runtime_tables: Vec<(usize, TableInfo)>,
) -> interpreter::WitgenResult {
    use crate::vm::bytecode::AllocationInstrumenter;

    // Re-encode raw-u64 multiplicity slots as Montgomery field elements. Walk the runtime-claimed
    // table list rather than scanning the entire multiplicities region: the host doesn't know each
    // table's length statically, only that the runtime registry recorded `length` for every claimed
    // slot.
    for (multiplicities_wit_off, tbl) in &runtime_tables {
        let lo = *multiplicities_wit_off;
        let hi = lo + tbl.length;
        for i in lo..hi {
            // FIELD-ASSUMPTION: L4-low-limb
            out_wit_pre_comm[i] = RawField::from(out_wit_pre_comm[i].0.0[0]);
        }
    }

    let tables = if r1cs.constraints_layout.lookups_data_size == 0 {
        vec![]
    } else {
        assert!(
            !runtime_tables.is_empty(),
            "WASM emitted lookup constraints but no tables were claimed at runtime"
        );

        runtime_tables.into_iter().map(|(_, table)| table).collect()
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
pub struct AdWasmResult {
    pub out_da: Vec<RawField>,
    pub out_db: Vec<RawField>,
    pub out_dc: Vec<RawField>,
    pub live_bytes: usize,
}

/// Write a field element to WASM memory at ptr
fn write_field_to_memory(
    memory: &Memory,
    mut store: impl wasmtime::AsContextMut,
    ptr: u32,
    field: &RawField,
) {
    let limbs = field.0.0;
    let offset = ptr as usize;
    let data = memory.data_mut(&mut store);
    data[offset..offset + 8].copy_from_slice(&limbs[0].to_le_bytes());
    data[offset + 8..offset + 16].copy_from_slice(&limbs[1].to_le_bytes());
    data[offset + 16..offset + 24].copy_from_slice(&limbs[2].to_le_bytes());
    data[offset + 24..offset + 32].copy_from_slice(&limbs[3].to_le_bytes());
}

/// Run a compiled module's AD entry point against `ad_coeffs`.
pub fn run_ad(
    wasm_path: &Path,
    r1cs: &R1CS,
    coeffs: &[RawField],
) -> Result<AdWasmResult, Box<dyn std::error::Error>> {
    let witness_count = r1cs.witness_layout.size();
    let constraint_count = r1cs.constraints.len();

    let vm_struct_size: u32 = AD_VM_STRUCT_SIZE;
    // FIELD-ASSUMPTION: L3-field-size
    let da_bytes = (witness_count * FIELD_SIZE) as u32;
    let db_bytes = da_bytes;
    let dc_bytes = da_bytes;
    // FIELD-ASSUMPTION: L3-field-size
    let coeffs_bytes = (constraint_count * FIELD_SIZE) as u32;

    let engine = wasm_engine()?;
    let mut store = Store::new(&engine, ());

    let module = wasm_runtime::load_wasmtime_module(&engine, wasm_path)?;

    let memory = Memory::new(&mut store, imported_memory_type(&module)?)?;

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
            coeffs_ptr + (i * FIELD_SIZE) as u32, // FIELD-ASSUMPTION: L3-field-size
            coeff,
        );
    }

    // AD lookups need an absolute base for random-access coefficient reads and a fresh-witness
    // counter seeded at the lookups-section start. The three table-allocation cursors
    // (`current_*_tables_off`, `current_wit_multiplicities_off`) are seeded at structural layout
    // starts; first-use lookup helpers snapshot them and then bump by their own table footprint.
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
        .get_func(&mut store, "mavros_ad_main")
        .ok_or("mavros_ad_main not found")?;

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
            da_ptr + (i * FIELD_SIZE) as u32, // FIELD-ASSUMPTION: L3-field-size
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
