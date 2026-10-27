use std::{
    alloc::{self, Layout},
    mem::{self, size_of},
};

use tracing::instrument;

pub use crate::InputValueOrdered;

use crate::{
    ConstraintsLayout, TableKind, WitnessLayout,
    array::BoxedValue,
    bytecode::{
        self, AllocationInstrumenter, AllocationType, Arrays, ENTRY_AD, ENTRY_VERIFIER,
        ENTRY_WITGEN, Int128, OpCode, TableInfo, VM, parse_program_header, pow2_rows,
    },
};

/// The part of `MavrosField`'s safety contract a frame pun depends on, checked when each field's
/// accessors are instantiated: `F` covers exactly its storage cells at an alignment a cell meets.
const fn assert_field_layout<F: mavros_artifacts::MavrosField>() {
    assert!(size_of::<F>() == F::STORAGE_CELLS * size_of::<u64>() && F::STORAGE_CELLS >= 1);
    assert!(align_of::<F>() <= align_of::<u64>());
}

/// An opcode handler. Returns the `(pc, frame)` to feed into the next
/// dispatch step. A null `pc` signals that execution should halt (the program
/// has fallen off the base frame in `ret`).
pub type Handler = fn(*const u64, Frame, &mut VM) -> (*const u64, Frame);

/// Tail-call-recursive dispatch. Each step reads the opcode at `pc`,
/// transmutes it into a `Handler`, and tail-calls into the next dispatch with
/// whatever pc/frame the handler produced. Relies on LLVM's tail-call
/// optimization — in debug builds (or anywhere TCO does not fire) this will
/// blow the stack on long programs; enable the `branching-interpreter` feature
/// to switch to a loop-based dispatch for those cases.
#[cfg(not(feature = "branching-interpreter"))]
pub unsafe fn dispatch(pc: *const u64, frame: Frame, vm: &mut VM) {
    if pc.is_null() {
        return;
    }
    let opcode: Handler = unsafe { mem::transmute(*pc) };
    let (next_pc, next_frame) = opcode(pc, frame, vm);
    unsafe { dispatch(next_pc, next_frame, vm) }
}

/// Loop-based dispatch. Does not rely on tail-call optimization, so it works
/// in debug builds.
#[cfg(feature = "branching-interpreter")]
pub unsafe fn dispatch(mut pc: *const u64, mut frame: Frame, vm: &mut VM) {
    while !pc.is_null() {
        let opcode: Handler = unsafe { mem::transmute(*pc) };
        let (next_pc, next_frame) = opcode(pc, frame, vm);
        pc = next_pc;
        frame = next_frame;
    }
}

// We don't want this file to compile if we can't safely pun u64 and pointer, so we add a
// compile-time assertion of the invariant we need.
const _: () = assert!(
    size_of::<*mut u64>() == size_of::<u64>(),
    "Cannot compile for platforms with non-64-bit pointers."
);

#[derive(Debug, Copy, Clone)]
pub struct Frame {
    /// Stores the data in the frame.
    ///
    /// A valid frame has `data` pointing to the start of the frame's actual data section, not the
    /// metadata. The metadata for the frame must be located at addresses as follows:
    ///
    /// - The **size** of the frame's data allocation is at `data.offset(-2)`.
    /// - The **parent frame pointer** is at `data.offset(-1)` and must be cast to a pointer.
    pub data: *mut u64,
}

impl Frame {
    /// Allocates the base frame of the program.
    ///
    /// The base frame is the frame that contains a null data pointer.
    pub fn base_frame(size: u64, vm: &mut VM) -> Self {
        Self::push(
            size,
            Frame {
                data: std::ptr::null_mut(),
            },
            vm,
        )
    }

    /// Pushes a new frame onto the stack with the provided `size` and the given `parent`.
    pub fn push(size: u64, parent: Frame, vm: &mut VM) -> Self {
        unsafe {
            let layout = Layout::array::<u64>(size as usize + 2).unwrap();
            let data = alloc::alloc(layout) as *mut u64;
            *data = size;

            // This punning is safe by the assertion above.
            *data.offset(1) = parent.data as u64;
            let data = data.offset(2);
            vm.allocation_instrumenter
                .alloc(AllocationType::Stack, size as usize + 2);
            Frame { data }
        }
    }

    /// Pops the top frame from the stack, returning it.
    ///
    /// The returned frame may have a data pointer that is nullptr, indicating that it is the top
    /// frame of execution.
    #[inline(always)]
    pub fn pop(self, vm: &mut VM) -> Frame {
        unsafe {
            let real_data = self.data.offset(-2);
            let parent_data = *real_data.offset(1) as *mut u64;
            let size = *real_data;
            alloc::dealloc(
                real_data as *mut u8,
                Layout::array::<u64>(size as usize + 2).unwrap(),
            );
            vm.allocation_instrumenter
                .free(AllocationType::Stack, size as usize + 2);
            Frame { data: parent_data }
        }
    }

    /// The field element whose storage cells start at `offset`.
    ///
    /// Reading the cells as `F` is sound by the safety contract of
    /// [`MavrosField`](mavros_artifacts::MavrosField): `F` is laid out as exactly its storage
    /// cells, with an alignment the cell pointer meets.
    #[inline(always)]
    pub fn read_field<F: mavros_artifacts::MavrosField>(&self, offset: isize) -> F {
        const { assert_field_layout::<F>() };
        unsafe { *(self.data.offset(offset) as *const F) }
    }

    /// The field element slot at `offset`, under the same contract as [`Frame::read_field`].
    #[inline(always)]
    pub fn read_field_mut<F: mavros_artifacts::MavrosField>(&self, offset: isize) -> *mut F {
        const { assert_field_layout::<F>() };
        unsafe { self.data.offset(offset) as *mut F }
    }

    #[inline(always)]
    pub fn read_u64_mut(&self, offset: isize) -> *mut u64 {
        unsafe { self.data.offset(offset) }
    }

    #[inline(always)]
    pub fn read_int128_mut(&self, offset: isize) -> *mut Int128 {
        unsafe { self.data.offset(offset) as *mut Int128 }
    }

    #[inline(always)]
    pub fn read_u64(&self, offset: isize) -> u64 {
        unsafe { *self.data.offset(offset) }
    }

    #[inline(always)]
    pub fn read_int128(&self, offset: isize) -> Int128 {
        unsafe { *(self.data.offset(offset) as *const Int128) }
    }

    /// The number of frame cells a `bits`-wide integer occupies.
    #[inline(always)]
    pub fn int_cells(bits: u64) -> usize {
        (bits as usize).div_ceil(64)
    }

    /// The number of frame cells a guest value of type `T` occupies.
    ///
    /// Every `read_*` below casts the cell pointer to `T`, so `T`'s size is what it occupies. The
    /// generated dispatch uses this as an `#[out]`'s extent when it checks that no frame slice
    /// overlaps it.
    #[inline(always)]
    pub const fn cells_of<T>() -> usize {
        size_of::<T>().div_ceil(size_of::<u64>())
    }

    /// The `len` consecutive cells at `offset`.
    ///
    /// # Safety
    ///
    /// The caller guarantees that the cells are within the frame, and that nothing writes them
    /// while the returned slice lives: a write while a shared reference covers the same cells is
    /// undefined behavior, however the pointer was obtained.
    #[inline(always)]
    pub unsafe fn read_limbs(&self, offset: isize, len: usize) -> &[u64] {
        unsafe { std::slice::from_raw_parts(self.data.offset(offset), len) }
    }

    /// The mutable form of [`Frame::read_limbs`], for a wide integer result.
    ///
    /// # Safety
    ///
    /// As [`Frame::read_limbs`], and nothing may _read_ these cells either while the returned slice
    /// lives.
    #[expect(
        clippy::mut_from_ref,
        reason = "the frame is a raw pointer to cells the VM owns, and a handler's result slice \
                  borrows the frame no more than its operands do"
    )]
    #[inline(always)]
    pub unsafe fn read_limbs_mut(&self, offset: isize, len: usize) -> &mut [u64] {
        unsafe { std::slice::from_raw_parts_mut(self.data.offset(offset), len) }
    }

    #[inline(always)]
    pub fn read_bool(&self, offset: isize) -> bool {
        let a0 = unsafe { *self.data.offset(offset) };
        a0 != 0
    }

    #[inline(always)]
    pub fn read_ptr<A>(&self, offset: isize) -> *mut A {
        unsafe { *self.data.offset(offset) as *mut A }
    }

    #[inline(always)]
    pub fn read_array(&self, offset: isize) -> BoxedValue {
        unsafe { *self.read_array_mut(offset) }
    }

    #[inline(always)]
    pub fn read_array_mut(&self, offset: isize) -> *mut BoxedValue {
        unsafe { self.data.offset(offset) as *mut BoxedValue }
    }

    #[inline(always)]
    pub fn write_u64(&self, offset: isize, value: u64) {
        unsafe {
            *self.data.offset(offset) = value;
        };
    }

    /// Writes `field`'s storage cells at `offset`, under the contract of [`Frame::read_field`].
    #[inline(always)]
    pub fn write_field<F: mavros_artifacts::MavrosField>(&self, offset: isize, field: F) {
        const { assert_field_layout::<F>() };
        unsafe { *(self.data.offset(offset) as *mut F) = field }
    }

    #[inline(always)]
    pub fn memcpy(&self, dest: isize, src: isize, size: usize) {
        unsafe {
            std::ptr::copy_nonoverlapping(self.data.offset(src), self.data.offset(dest), size);
        }
    }

    #[inline(always)]
    pub unsafe fn write_to(&self, dst: *mut u64, src: isize, size: usize) {
        unsafe {
            std::ptr::copy_nonoverlapping(self.data.offset(src), dst, size);
        }
    }
}

pub(crate) fn prepare_dispatch(program: &mut [u64], code_start: usize) {
    // `code_start` is the index of the first function marker, where the opcode
    // stream begins.
    let mut current_offset = code_start;
    while current_offset < program.len() {
        let opcode = program[current_offset];
        if opcode == u64::MAX {
            current_offset += 2;
            continue;
        }
        let next = OpCode::next_opcode(program, current_offset);
        program[current_offset] = bytecode::DISPATCH[opcode as usize] as u64;
        current_offset = next;
    }
}

pub struct WitgenResult<F> {
    pub out_wit_pre_comm: Vec<F>,
    pub out_wit_post_comm: Vec<F>,
    pub out_a: Vec<F>,
    pub out_b: Vec<F>,
    pub out_c: Vec<F>,
    pub instrumenter: AllocationInstrumenter,
}

/// The program executed a trap: a failed assertion, rangecheck, or bounds check.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TrapError {
    message: Option<String>,
    stack_trace: Vec<bytecode::StackFrame>,
}

impl TrapError {
    pub fn message(&self) -> Option<&str> {
        self.message.as_deref()
    }

    pub fn stack_trace(&self) -> &[bytecode::StackFrame] {
        &self.stack_trace
    }

    pub fn relativize_source_paths(&mut self, root: &std::path::Path) {
        for frame in &mut self.stack_trace {
            bytecode::relativize_source_path(&mut frame.location.file, root);
        }
    }
}

impl std::fmt::Display for TrapError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "VM trapped during execution")?;
        if let Some(message) = &self.message {
            write!(f, ": {message}")?;
        }
        let has_explanatory_frame = self
            .stack_trace
            .iter()
            .any(|frame| frame.location.file != "<wrapper_main>");
        for frame in &self.stack_trace {
            if has_explanatory_frame && frame.location.file == "<wrapper_main>" {
                continue;
            }
            write!(f, "\n  at {frame}")?;
        }
        Ok(())
    }
}

impl std::error::Error for TrapError {}

/// Intermediate result from phase 1 of witness generation.
///
/// Contains the pre-commitment witness and all intermediate state needed to
/// complete witness generation in phase 2 (after real Fiat-Shamir challenges
/// are available).
#[derive(Clone)]
pub struct Phase1Result<F> {
    pub out_wit_pre_comm: Vec<F>,
    pub out_wit_post_comm: Vec<F>,
    pub out_a: Vec<F>,
    pub out_b: Vec<F>,
    pub out_c: Vec<F>,
    pub tables: Vec<Phase2Table>,
    pub instrumenter: AllocationInstrumenter,
}

/// A lookup table as phase 2 completes it, located by offsets into the vectors of the
/// [`Phase1Result`] it belongs to.
#[derive(Clone, Debug)]
pub struct Phase2Table {
    pub kind: TableKind,
    pub length: usize,
    /// Where the table's multiplicities start in the pre-commitment witness.
    pub multiplicities_offset: usize,
    pub elem_inverses_witness_section_offset: usize,
    pub elem_inverses_constraint_section_offset: usize,
}

impl Phase2Table {
    /// Locates `table`, whose multiplicities tape points into `witness`.
    fn locate<F: mavros_artifacts::MavrosField>(table: &TableInfo, witness: &[F]) -> Phase2Table {
        let bytes = (table.multiplicities_wit.cell0() as usize)
            .checked_sub(witness.as_ptr() as usize)
            .expect("a table's multiplicities lie in the pre-commitment witness");
        Phase2Table {
            kind: table.kind,
            length: table.length,
            multiplicities_offset: bytes / size_of::<F>(),
            elem_inverses_witness_section_offset: table.elem_inverses_witness_section_offset,
            elem_inverses_constraint_section_offset: table.elem_inverses_constraint_section_offset,
        }
    }
}

fn fix_multiplicities_section<F: mavros_artifacts::MavrosField>(
    wit: &mut [F],
    witness_layout: WitnessLayout,
) {
    #[allow(clippy::needless_range_loop)]
    for i in witness_layout.multiplicities_start()..witness_layout.multiplicities_end() {
        // Phase 1 counts a multiplicity as a raw `u64` in its slot's first storage cell.
        wit[i] = F::from(wit[i].storage_cell0());
    }
}

/// Phase 1 of witness generation: executes the VM to produce the
/// pre-commitment witness and captures all intermediate state needed for
/// phase 2.
///
/// `F` is the field whose opcodes the program was compiled to, and `inputs` are the entry blob's
/// elements in order (for bn254, as [`flatten_param_vec`] flattens them).
#[instrument(skip_all, name = "Interpreter::run_phase1")]
pub fn run_phase1<F: mavros_artifacts::MavrosField>(
    program: &[u64],
    witness_layout: WitnessLayout,
    constraints_layout: ConstraintsLayout,
    inputs: &[F],
    debug_info: Option<bytecode::DebugInfo>,
) -> Result<Phase1Result<F>, TrapError> {
    run_phase1_impl(
        program,
        witness_layout,
        constraints_layout,
        inputs,
        debug_info,
        false,
    )
    .map(|(result, _)| result)
}

/// Profiled phase 1 execution. The returned weights are simulated VM
/// instructions, which are deterministic across machines and runs.
pub fn run_phase1_profiled<F: mavros_artifacts::MavrosField>(
    program: &[u64],
    witness_layout: WitnessLayout,
    constraints_layout: ConstraintsLayout,
    inputs: &[F],
    debug_info: Option<bytecode::DebugInfo>,
) -> Result<(Phase1Result<F>, crate::FlamegraphProfile), TrapError> {
    run_phase1_impl(
        program,
        witness_layout,
        constraints_layout,
        inputs,
        debug_info,
        true,
    )
}

/// Refuses to run a program over a field other than the one its header names.
///
/// The handlers a field opcode dispatches to read and write their field's element width, so running
/// a program over another field's tapes writes past their ends. The check is once per run, in every
/// build.
#[track_caller]
fn check_program_field<F: mavros_artifacts::MavrosField>(header: &bytecode::ProgramHeader) {
    assert!(
        header.field == F::ID,
        "a VM program over {:?} cannot run over {:?}",
        header.field,
        F::ID
    );
}

fn run_phase1_impl<F: mavros_artifacts::MavrosField>(
    program: &[u64],
    witness_layout: WitnessLayout,
    constraints_layout: ConstraintsLayout,
    inputs: &[F],
    debug_info: Option<bytecode::DebugInfo>,
    profile: bool,
) -> Result<(Phase1Result<F>, crate::FlamegraphProfile), TrapError> {
    let header = parse_program_header(program);
    check_program_field::<F>(&header);
    let debug_info = debug_info.unwrap_or_default();
    let entry = *header
        .entry_points
        .get(ENTRY_WITGEN)
        .filter(|entry| **entry != bytecode::NO_ENTRY_POINT)
        .ok_or_else(|| TrapError {
            message: Some("Program has no witgen entry point; use the prover artifact".to_string()),
            stack_trace: Vec::new(),
        })?;
    let global_frame_size = header.global_frame_size;
    let mut out_a = vec![F::ZERO; constraints_layout.size()];
    let mut out_b = vec![F::ZERO; constraints_layout.size()];
    let mut out_c = vec![F::ZERO; constraints_layout.size()];
    let mut out_wit_pre_comm = vec![F::ZERO; witness_layout.pre_commitment_size()];
    // Main takes its inputs as a single `Blob<Field; N>` parameter, one `F` per element.
    assert_eq!(
        inputs.len(),
        header.entry_blob_field_count,
        "{} flattened input fields supplied but the entry has {}",
        inputs.len(),
        header.entry_blob_field_count,
    );
    // The program writes its public parameters to the witness tape with pinned WriteWitness
    // instructions, then its private ones unpinned, so an unused private parameter may never
    // reach the tape.
    let out_wit_post_comm = vec![F::ZERO; witness_layout.post_commitment_size()];
    let mut global_frame = vec![0u64; global_frame_size];
    let mut vm = VM::new_witgen(
        out_a.as_mut_ptr(),
        out_b.as_mut_ptr(),
        out_c.as_mut_ptr(),
        out_wit_pre_comm.as_mut_ptr(),
        unsafe {
            out_wit_pre_comm
                .as_mut_ptr()
                .add(witness_layout.multiplicities_start())
        },
        unsafe {
            out_a
                .as_mut_ptr()
                .add(constraints_layout.lookups_data_start())
        },
        unsafe {
            out_b
                .as_mut_ptr()
                .add(constraints_layout.lookups_data_start())
        },
        unsafe {
            out_c
                .as_mut_ptr()
                .add(constraints_layout.lookups_data_start())
        },
        constraints_layout.tables_data_start(),
        witness_layout.tables_data_start() - witness_layout.challenges_start(),
        global_frame.as_mut_ptr(),
        header.constant_pool,
    );

    let frame_size = program[entry + 1];
    let frame = Frame::base_frame(frame_size, &mut vm);

    // The input blob is stored by value in the frame, starting right after
    // the two return slots, one element per `F::STORAGE_CELLS` cells, as codegen lays out the
    // entry's parameter.
    assert!(
        2 + F::STORAGE_CELLS * inputs.len() <= frame_size as usize,
        "the entry's {frame_size}-cell frame cannot hold {} inputs",
        inputs.len()
    );
    for (input_index, el) in inputs.iter().enumerate() {
        frame.write_field((2 + F::STORAGE_CELLS * input_index) as isize, *el);
    }

    let mut program = program.to_vec();
    prepare_dispatch(&mut program, header.code_start);
    vm.set_debug_context(program.as_ptr(), program.len(), debug_info);
    if profile {
        vm.enable_instruction_profile();
    }

    let pc = unsafe { program.as_mut_ptr().add(entry + 2) };

    unsafe { dispatch(pc, frame, &mut vm) };

    #[cfg(feature = "vm-profile")]
    vm.report_opcode_profile("witgen phase 1");

    if vm.trapped {
        return Err(TrapError {
            message: vm.trap_message.take(),
            stack_trace: std::mem::take(&mut vm.stack_trace),
        });
    }

    fix_multiplicities_section(&mut out_wit_pre_comm, witness_layout);
    let instruction_profile = vm.take_instruction_profile();
    let tables = vm
        .tables
        .iter()
        .map(|table| Phase2Table::locate(table, &out_wit_pre_comm))
        .collect();

    Ok((
        Phase1Result {
            out_wit_pre_comm,
            out_wit_post_comm,
            out_a,
            out_b,
            out_c,
            tables,
            instrumenter: vm.allocation_instrumenter,
        },
        instruction_profile,
    ))
}

/// Phase 2 of witness generation: uses real Fiat-Shamir challenges to
/// complete the post-commitment witness and constraint vectors.
#[instrument(skip_all, name = "Interpreter::run_phase2")]
pub fn run_phase2<F: mavros_artifacts::MavrosField>(
    mut phase1: Phase1Result<F>,
    challenges: &[F],
    witness_layout: WitnessLayout,
    constraints_layout: ConstraintsLayout,
) -> WitgenResult<F> {
    assert_eq!(
        challenges.len(),
        witness_layout.challenges_size,
        "phase 2 takes one challenge per challenge witness"
    );
    // Write real challenges into the post-commitment witness vector.
    for (i, challenge) in challenges.iter().enumerate() {
        phase1.out_wit_post_comm[i] = *challenge;
    }

    // FIELD-ASSUMPTION: L4-logup-challenges (6 sites)
    // Phase 2 reads the single LogUp challenge at the hardcoded post-commitment indices
    // [0] (alpha) and [1] (beta) throughout this function. A small field needs K challenges;
    // these reads become [2k]/[2k+1] inside a `for k in 0..K` loop (see docs/field-agnosticism.md).
    let mut running_prod = F::ONE;
    for tbl in phase1.tables.iter() {
        let alpha = phase1.out_wit_post_comm[0];
        let base = tbl.elem_inverses_constraint_section_offset;

        match tbl.kind {
            TableKind::RangeCheck => {
                // One constraint per entry: denom_i = α - i
                for i in 0..tbl.length {
                    let multiplicity = phase1.out_wit_pre_comm[tbl.multiplicities_offset + i];
                    let denom = alpha - F::from(i as u64);
                    phase1.out_b[base + i] = denom;
                    phase1.out_c[base + i] = multiplicity;
                    if multiplicity != F::ZERO {
                        phase1.out_a[base + i] = running_prod;
                        running_prod *= denom;
                    }
                }
            }
            TableKind::Spread => {
                // One folded constraint per entry. Both operands (key=i,
                // value=spread(i)) are constants, so β·spread(i) folds into the
                // denominator: denom_i = α - i + β·spread(i). Phase 2 recomputes spread(i)
                // from the row index, so the VM dumps nothing for it.
                let beta = phase1.out_wit_post_comm[1];
                for i in 0..tbl.length {
                    let multiplicity = phase1.out_wit_pre_comm[tbl.multiplicities_offset + i];
                    let spread_i = F::from(mavros_limb_arith::spread_u32_to_u64(i as u32));
                    let denom = alpha - F::from(i as u64) + beta * spread_i;
                    phase1.out_b[base + i] = denom;
                    phase1.out_c[base + i] = multiplicity;
                    if multiplicity != F::ZERO {
                        phase1.out_a[base + i] = running_prod;
                        running_prod *= denom;
                    }
                }
            }
            TableKind::Pow2 => {
                // One folded constraint per entry, exactly like the spread table above: both
                // operands (key=n, value=2^n) are constants, so β·2^n folds into the denominator.
                // Phase 2 recomputes 2^n from the row index, so the VM dumps nothing for it.
                let beta = phase1.out_wit_post_comm[1];
                for (n, pow) in pow2_rows::<F>(tbl.length).enumerate() {
                    let multiplicity = phase1.out_wit_pre_comm[tbl.multiplicities_offset + n];
                    let denom = alpha - F::from(n as u64) + beta * pow;
                    phase1.out_b[base + n] = denom;
                    phase1.out_c[base + n] = multiplicity;
                    if multiplicity != F::ZERO {
                        phase1.out_a[base + n] = running_prod;
                        running_prod *= denom;
                    }
                }
            }
            TableKind::Array => {
                // Two constraints per entry: x_i = -β*v_i, denom_i = α - i - x_i
                let beta = phase1.out_wit_post_comm[1];
                for i in 0..tbl.length {
                    let multiplicity = phase1.out_wit_pre_comm[tbl.multiplicities_offset + i];

                    // Read v_i from the x-slot where the VM dumped it
                    let v_i = phase1.out_a[base + 2 * i];
                    let x_i = -beta * v_i;

                    // Fill x-constraint: β * v_i = -x_i
                    phase1.out_a[base + 2 * i] = beta;
                    phase1.out_b[base + 2 * i] = v_i;
                    phase1.out_c[base + 2 * i] = -x_i;

                    // Fill y-constraint slots (will be overwritten by batch inversion)
                    let denom = alpha - F::from(i as u64) - x_i;
                    phase1.out_b[base + 2 * i + 1] = denom;
                    phase1.out_c[base + 2 * i + 1] = multiplicity;
                    if multiplicity != F::ZERO {
                        phase1.out_a[base + 2 * i + 1] = running_prod;
                        running_prod *= denom;
                    }
                }
            }
        }
    }

    let mut running_inv = running_prod.inverse().unwrap();

    for tbl in phase1.tables.iter().rev() {
        let base = tbl.elem_inverses_constraint_section_offset;

        if matches!(
            tbl.kind,
            TableKind::RangeCheck | TableKind::Spread | TableKind::Pow2
        ) {
            // One constraint per entry: y-values live at consecutive offsets.
            for i in (0..tbl.length).rev() {
                let multiplicity = phase1.out_c[base + i];
                let denom = phase1.out_b[base + i];
                let running_prod = phase1.out_a[base + i];
                if multiplicity != F::ZERO {
                    let elem = running_prod * running_inv;
                    phase1.out_a[base + i] = elem;
                    running_inv *= denom;
                }
            }
        } else {
            // Array: y-values at odd offsets.
            for i in (0..tbl.length).rev() {
                let multiplicity = phase1.out_c[base + 2 * i + 1];
                let denom = phase1.out_b[base + 2 * i + 1];
                let running_prod = phase1.out_a[base + 2 * i + 1];
                if multiplicity != F::ZERO {
                    let elem = running_prod * running_inv;
                    phase1.out_a[base + 2 * i + 1] = elem;
                    running_inv *= denom;
                }
            }
        }
    }

    let mut current_lookup_off = 0;

    while current_lookup_off < constraints_layout.lookups_data_size {
        let cnst_off = constraints_layout.lookups_data_start() + current_lookup_off;
        let wit_off = witness_layout.lookups_data_start() - witness_layout.challenges_start()
            + current_lookup_off;

        // Peek at which table this lookup belongs to, to determine width
        let table_ix = phase1.out_a[cnst_off].storage_cell0();
        let table = &phase1.tables[table_ix as usize];

        let alpha = phase1.out_wit_post_comm[0];

        if table.kind == TableKind::RangeCheck {
            // Key-only lookup (rangecheck): 1 constraint per lookup
            let flag_u64 = phase1.out_c[cnst_off].storage_cell0();

            if flag_u64 == 0 {
                let key = phase1.out_b[cnst_off];
                let b_val = alpha - key;
                phase1.out_a[cnst_off] = F::ZERO;
                phase1.out_b[cnst_off] = b_val;
                phase1.out_c[cnst_off] = F::ZERO;
                phase1.out_wit_post_comm[wit_off] = F::ZERO;
            } else {
                let ix_in_table = phase1.out_b[cnst_off].storage_cell0();
                phase1.out_a[cnst_off] = phase1.out_a
                    [table.elem_inverses_constraint_section_offset + ix_in_table as usize];
                phase1.out_b[cnst_off] = phase1.out_b
                    [table.elem_inverses_constraint_section_offset + ix_in_table as usize];
                phase1.out_c[cnst_off] = F::from(flag_u64);
                phase1.out_wit_post_comm[wit_off] = phase1.out_a[cnst_off];
                phase1.out_c[table.elem_inverses_constraint_section_offset + table.length] +=
                    phase1.out_a[cnst_off];
            }

            current_lookup_off += 1;
        } else {
            // Key-value lookup (array or spread): 2 constraints per lookup. Only the table's y-slot
            // stride differs by kind: an array stores x,y per entry (stride 2, y at the odd slot),
            // a folded spread or powers-of-two table only y (stride 1).
            //
            // Entry 1 (x-constraint): out_a=table_id, out_b=result_value, out_c=0
            // Entry 2 (y-constraint): out_a=table_id, out_b=index, out_c=flag
            let entry_stride = match table.kind {
                TableKind::Spread | TableKind::Pow2 => 1,
                _ => 2,
            };
            let beta = phase1.out_wit_post_comm[1];
            let result_value = phase1.out_b[cnst_off];
            let flag_u64 = phase1.out_c[cnst_off + 1].storage_cell0();

            // x-constraint: β * value = -x → x = -β * value
            let x = -beta * result_value;
            phase1.out_a[cnst_off] = beta;
            phase1.out_b[cnst_off] = result_value;
            phase1.out_c[cnst_off] = -x;
            phase1.out_wit_post_comm[wit_off] = x;

            // y-constraint: y * (α - key - x) = flag
            let y_cnst_off = cnst_off + 1;
            let y_wit_off = wit_off + 1;

            if flag_u64 == 0 {
                let key = phase1.out_b[y_cnst_off];
                let b_val = alpha - key - x;
                phase1.out_a[y_cnst_off] = F::ZERO;
                phase1.out_b[y_cnst_off] = b_val;
                phase1.out_c[y_cnst_off] = F::ZERO;
                phase1.out_wit_post_comm[y_wit_off] = F::ZERO;
            } else {
                let ix_in_table = phase1.out_b[y_cnst_off].storage_cell0();
                let tbl_base = table.elem_inverses_constraint_section_offset;
                // Copy precomputed inverse from the table entry's y-slot
                // (array: 2*ix+1; spread: ix).
                let y_slot = tbl_base + entry_stride * ix_in_table as usize + (entry_stride - 1);
                phase1.out_a[y_cnst_off] = phase1.out_a[y_slot];
                phase1.out_b[y_cnst_off] = phase1.out_b[y_slot];
                phase1.out_c[y_cnst_off] = F::from(flag_u64);
                phase1.out_wit_post_comm[y_wit_off] = phase1.out_a[y_cnst_off];
                // Add to sum constraint (array: 2*n; spread: n).
                let sum_off = tbl_base + entry_stride * table.length;
                phase1.out_c[sum_off] += phase1.out_a[y_cnst_off];
            }

            current_lookup_off += 2;
        }
    }

    for tbl in phase1.tables.iter() {
        let base = tbl.elem_inverses_constraint_section_offset;
        let wit_base = tbl.elem_inverses_witness_section_offset;

        if matches!(
            tbl.kind,
            TableKind::RangeCheck | TableKind::Spread | TableKind::Pow2
        ) {
            // One constraint per entry: y-values at consecutive offsets, sum
            // constraint at offset length, one witness per entry.
            for i in 0..tbl.length {
                let multiplicity = phase1.out_c[base + i];
                if multiplicity != F::ZERO {
                    let elem = phase1.out_a[base + i] * multiplicity;
                    phase1.out_a[base + i] = elem;
                    phase1.out_wit_post_comm[wit_base + i] = elem;
                    phase1.out_a[base + tbl.length] += elem;
                }
            }
            phase1.out_b[base + tbl.length] = F::ONE;
        } else {
            // Array: y-values at odd offsets, sum constraint at offset 2*length
            for i in 0..tbl.length {
                let multiplicity = phase1.out_c[base + 2 * i + 1];
                if multiplicity != F::ZERO {
                    let elem = phase1.out_a[base + 2 * i + 1] * multiplicity;
                    phase1.out_a[base + 2 * i + 1] = elem;
                    // x witness at even offset, y witness at odd offset
                    phase1.out_wit_post_comm[wit_base + 2 * i + 1] = elem;
                    phase1.out_a[base + 2 * tbl.length] += elem;
                }
                // x witness: x_i = -β * v_i; out_c stores -x_i = β*v_i, so negate
                phase1.out_wit_post_comm[wit_base + 2 * i] = -phase1.out_c[base + 2 * i];
            }
            phase1.out_b[base + 2 * tbl.length] = F::ONE;
        }
    }

    WitgenResult {
        out_wit_pre_comm: phase1.out_wit_pre_comm,
        out_wit_post_comm: phase1.out_wit_post_comm,
        out_a: phase1.out_a,
        out_b: phase1.out_b,
        out_c: phase1.out_c,
        instrumenter: phase1.instrumenter,
    }
}

fn fake_challenges<F: mavros_artifacts::MavrosField>(count: usize) -> Vec<F> {
    let mut fake_challenges = vec![F::ZERO; count];
    let mut random = F::from(18765435241434657586764563434227903u128);
    for challenge in fake_challenges.iter_mut() {
        *challenge = random;
        random = (random + F::from(17u64)) * random;
    }
    fake_challenges
}

pub fn run_phase2_with_fake_challenges<F: mavros_artifacts::MavrosField>(
    phase1: Phase1Result<F>,
    witness_layout: WitnessLayout,
    constraints_layout: ConstraintsLayout,
) -> WitgenResult<F> {
    run_phase2(
        phase1,
        &fake_challenges(witness_layout.challenges_size),
        witness_layout,
        constraints_layout,
    )
}

/// Both phases of witness generation, with fake challenges. The witness satisfies the constraints;
/// a proof draws its challenges from the committed pre-commitment witness, through [`run_phase1`]
/// and [`run_phase2`].
#[instrument(skip_all, name = "Interpreter::run")]
pub fn run<F: mavros_artifacts::MavrosField>(
    program: &[u64],
    witness_layout: WitnessLayout,
    constraints_layout: ConstraintsLayout,
    inputs: &[F],
    debug_info: Option<bytecode::DebugInfo>,
) -> Result<WitgenResult<F>, TrapError> {
    let phase1 = run_phase1(
        program,
        witness_layout,
        constraints_layout,
        inputs,
        debug_info,
    )?;

    Ok(run_phase2_with_fake_challenges(
        phase1,
        witness_layout,
        constraints_layout,
    ))
}

/// Run witness generation with deterministic per-call-stack instruction
/// counting suitable for a FlameGraph.
pub fn run_profiled<F: mavros_artifacts::MavrosField>(
    program: &[u64],
    witness_layout: WitnessLayout,
    constraints_layout: ConstraintsLayout,
    inputs: &[F],
    debug_info: Option<bytecode::DebugInfo>,
) -> Result<(WitgenResult<F>, crate::FlamegraphProfile), TrapError> {
    let (phase1, profile) = run_phase1_profiled(
        program,
        witness_layout,
        constraints_layout,
        inputs,
        debug_info,
    )?;

    Ok((
        run_phase2_with_fake_challenges(phase1, witness_layout, constraints_layout),
        profile,
    ))
}

/// The three matrix contractions used by a Spartan verifier. These values do not
/// indicate whether a witness satisfies the circuit.
// FIELD-ASSUMPTION: L1-alias
// The default result type is the compiler's BN254 field.
pub struct VerifierResult<F = crate::Field> {
    pub r_aw: F,
    pub r_bw: F,
    pub r_cw: F,
    pub instrumenter: AllocationInstrumenter,
}

/// Compute `(r A w, r B w, r C w)` from the compiled constraint traversal.
///
/// `r` has one entry per constraint and `w` covers the entire witness layout,
/// including multiplicities, challenges, and lookup witnesses. Every entry is
/// arbitrary: in particular, `w[0]` need not be one. No prover inputs or satisfying
/// witness are required. The dedicated verifier entry point reads witness values
/// from `w` and evaluates linear combinations forward, without allocating AD
/// nodes, AD output vectors, or materializing the R1CS matrices.
/// Pass the separate verifier artifact, not the witgen/AD prover binary.
#[instrument(skip_all, name = "Interpreter::run_verifier")]
pub fn run_verifier<F: mavros_artifacts::MavrosField>(
    program: &[u64],
    r: &[F],
    w: &[F],
    witness_layout: WitnessLayout,
    constraints_layout: ConstraintsLayout,
    debug_info: Option<bytecode::DebugInfo>,
) -> Result<VerifierResult<F>, TrapError> {
    let mut a = F::ZERO;
    let mut b = F::ZERO;
    let mut c = F::ZERO;
    let (instrumenter, _) = run_constraint_program(
        program,
        r,
        witness_layout,
        constraints_layout,
        debug_info,
        false,
        ConstraintOutputs::Verifier {
            outputs: [&mut a, &mut b, &mut c],
            witness: w,
        },
    )?;
    Ok(VerifierResult {
        r_aw: a,
        r_bw: b,
        r_cw: c,
        instrumenter,
    })
}

#[instrument(skip_all, name = "Interpreter::run_ad")]
pub fn run_ad<F: mavros_artifacts::MavrosField>(
    program: &[u64],
    coeffs: &[F],
    witness_layout: WitnessLayout,
    constraints_layout: ConstraintsLayout,
    debug_info: Option<bytecode::DebugInfo>,
) -> Result<(Vec<F>, Vec<F>, Vec<F>, AllocationInstrumenter), TrapError> {
    run_ad_impl(
        program,
        coeffs,
        witness_layout,
        constraints_layout,
        debug_info,
        false,
    )
    .map(|(a, b, c, instrumenter, _)| (a, b, c, instrumenter))
}

/// Run automatic differentiation with deterministic per-call-stack
/// instruction counting suitable for a FlameGraph.
pub fn run_ad_profiled<F: mavros_artifacts::MavrosField>(
    program: &[u64],
    coeffs: &[F],
    witness_layout: WitnessLayout,
    constraints_layout: ConstraintsLayout,
    debug_info: Option<bytecode::DebugInfo>,
) -> Result<
    (
        Vec<F>,
        Vec<F>,
        Vec<F>,
        AllocationInstrumenter,
        crate::FlamegraphProfile,
    ),
    TrapError,
> {
    run_ad_impl(
        program,
        coeffs,
        witness_layout,
        constraints_layout,
        debug_info,
        true,
    )
}

fn run_ad_impl<F: mavros_artifacts::MavrosField>(
    program: &[u64],
    coeffs: &[F],
    witness_layout: WitnessLayout,
    constraints_layout: ConstraintsLayout,
    debug_info: Option<bytecode::DebugInfo>,
    profile: bool,
) -> Result<
    (
        Vec<F>,
        Vec<F>,
        Vec<F>,
        AllocationInstrumenter,
        crate::FlamegraphProfile,
    ),
    TrapError,
> {
    let mut out_da = vec![F::ZERO; witness_layout.size()];
    let mut out_db = vec![F::ZERO; witness_layout.size()];
    let mut out_dc = vec![F::ZERO; witness_layout.size()];
    let (instrumenter, profile) = run_constraint_program(
        program,
        coeffs,
        witness_layout,
        constraints_layout,
        debug_info,
        profile,
        ConstraintOutputs::Ad([&mut out_da, &mut out_db, &mut out_dc]),
    )?;
    Ok((out_da, out_db, out_dc, instrumenter, profile))
}

/// Borrowed buffers encode which output capacities each execution mode needs.
enum ConstraintOutputs<'a, F> {
    /// Allocated by `run_ad_impl` with exactly `witness_layout.size()` elements each.
    Ad([&'a mut [F]; 3]),
    Verifier {
        outputs: [&'a mut F; 3],
        witness: &'a [F],
    },
}

impl<F> ConstraintOutputs<'_, F> {
    fn entry(&self) -> usize {
        match self {
            Self::Ad(_) => ENTRY_AD,
            Self::Verifier { .. } => ENTRY_VERIFIER,
        }
    }

    #[cfg(feature = "vm-profile")]
    fn name(&self) -> &'static str {
        match self {
            Self::Ad(_) => "AD",
            Self::Verifier { .. } => "Verifier",
        }
    }
}

/// Shared dispatch for AD and verifier execution. The output storage is owned by
/// the callers above and remains alive until dispatch and reference-counted
/// sensitivity propagation finish.
fn run_constraint_program<F: mavros_artifacts::MavrosField>(
    program: &[u64],
    coeffs: &[F],
    witness_layout: WitnessLayout,
    constraints_layout: ConstraintsLayout,
    debug_info: Option<bytecode::DebugInfo>,
    profile: bool,
    outputs: ConstraintOutputs<'_, F>,
) -> Result<(AllocationInstrumenter, crate::FlamegraphProfile), TrapError> {
    if coeffs.len() != constraints_layout.size() {
        return Err(TrapError {
            message: Some(format!(
                "constraint coefficient length: expected {}, got {}",
                constraints_layout.size(),
                coeffs.len()
            )),
            stack_trace: Vec::new(),
        });
    }
    if let ConstraintOutputs::Verifier { witness, .. } = &outputs
        && witness.len() != witness_layout.size()
    {
        return Err(TrapError {
            message: Some(format!(
                "verifier w length: expected {}, got {}",
                witness_layout.size(),
                witness.len()
            )),
            stack_trace: Vec::new(),
        });
    }
    let header = parse_program_header(program);
    check_program_field::<F>(&header);
    let debug_info = debug_info.unwrap_or_default();
    let entry = *header.entry_points.get(outputs.entry())
        .filter(|entry| **entry != bytecode::NO_ENTRY_POINT)
        .ok_or_else(|| TrapError {
            message: Some(match &outputs {
                ConstraintOutputs::Ad(_) => "Program has no AD entry point",
                ConstraintOutputs::Verifier { .. } => "Program has no forward verifier entry point; compile a separate verifier artifact",
            }.to_string()),
            stack_trace: Vec::new(),
        })?;
    if !header.has_layouts(witness_layout, constraints_layout) {
        return Err(TrapError {
            message: Some("constraint layouts do not match bytecode".to_string()),
            stack_trace: Vec::new(),
        });
    }
    #[cfg(feature = "vm-profile")]
    let profile_name = outputs.name();
    let mut global_frame = vec![0u64; header.global_frame_size];
    // Exclusive borrows keep these buffers alive throughout dispatch and AD propagation.
    let mut vm = match outputs {
        ConstraintOutputs::Ad(buffers) => {
            let [a, b, c] = buffers.map(|buffer| buffer.as_mut_ptr());
            VM::new_ad(
                a,
                b,
                c,
                coeffs.as_ptr(),
                witness_layout,
                constraints_layout,
                global_frame.as_mut_ptr(),
                header.constant_pool,
            )
        }
        ConstraintOutputs::Verifier { outputs, witness } => VM::new_constraint::<F>(
            Arrays {
                as_verifier: crate::verifier::VerifierArrays::new(
                    outputs.map(|output| output as *mut F),
                    witness.as_ptr(),
                    coeffs.as_ptr(),
                    witness_layout,
                    constraints_layout,
                ),
            },
            global_frame.as_mut_ptr(),
            header.constant_pool,
        ),
    };

    let frame = Frame::push(
        program[entry + 1],
        Frame {
            data: std::ptr::null_mut(),
        },
        &mut vm,
    );

    let mut program = program.to_vec();
    prepare_dispatch(&mut program, header.code_start);
    vm.set_debug_context(program.as_ptr(), program.len(), debug_info);
    if profile {
        vm.enable_instruction_profile();
    }

    let pc = unsafe { program.as_mut_ptr().add(entry + 2) };

    unsafe { dispatch(pc, frame, &mut vm) };

    #[cfg(feature = "vm-profile")]
    vm.report_opcode_profile(profile_name);

    if vm.trapped {
        return Err(TrapError {
            message: vm.trap_message.take(),
            stack_trace: std::mem::take(&mut vm.stack_trace),
        });
    }

    let instruction_profile = vm.take_instruction_profile();
    Ok((vm.allocation_instrumenter, instruction_profile))
}

/// Flatten ordered input values into the entry blob's field sequence.
pub fn flatten_param_vec<F: Copy>(vec: &[InputValueOrdered<F>]) -> Vec<F> {
    let mut encoded_value = Vec::new();
    for elem in vec {
        encoded_value.extend(flatten_params(elem));
    }
    encoded_value
}

pub fn flattened_param_count<F>(vec: &[InputValueOrdered<F>]) -> usize {
    vec.iter().map(param_field_count).sum()
}

fn param_field_count<F>(value: &InputValueOrdered<F>) -> usize {
    match value {
        InputValueOrdered::Field(_) => 1,
        InputValueOrdered::Vec(vec_elements) => vec_elements.iter().map(param_field_count).sum(),
        InputValueOrdered::Struct(fields) => fields
            .iter()
            .map(|(_field_name, field_value)| param_field_count(field_value))
            .sum(),
        _ => panic!("Unsupported input value type."),
    }
}

fn flatten_params<F: Copy>(value: &InputValueOrdered<F>) -> Vec<F> {
    let mut encoded_value = Vec::new();
    match value {
        InputValueOrdered::Field(elem) => encoded_value.push(*elem),

        InputValueOrdered::Vec(vec_elements) => {
            for elem in vec_elements {
                encoded_value.extend(flatten_params(elem));
            }
        }
        InputValueOrdered::Struct(fields) => {
            for (_field_name, field_value) in fields {
                encoded_value.extend(flatten_params(field_value));
            }
        }
        _ => panic!(
            "Unsupported input value type. We only support Field, Vecs, and Structs for now."
        ),
    }
    encoded_value
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Field;
    use mavros_int_semantics::IntBits;

    #[test]
    fn verifier_dispatch_uses_each_fields_storage_width() {
        use bytecode::{FieldOpCode as Op, FramePosition as P, Function, Program};
        use mavros_artifacts::MavrosField;

        fn check<F: MavrosField>(emit: impl Fn(Op) -> OpCode) {
            let wl = WitnessLayout {
                algebraic_size: 2,
                multiplicities_size: 2,
                challenges_size: 1,
                tables_data_size: 2,
                lookups_data_size: 1,
            };
            let cl = ConstraintsLayout {
                algebraic_size: 1,
                tables_data_size: 3,
                lookups_data_size: 1,
            };
            // Adjacent field slots catch handlers that assume bn254's four-cell width.
            let [one, value, constant, coeff] =
                std::array::from_fn(|i| P(2 + i * F::STORAGE_CELLS));
            let mut code: Vec<_> = [
                Op::VerifierReadWitness { res: one },
                Op::VerifierReadWitness { res: value },
                Op::VerifierConstant {
                    res: constant,
                    v: value,
                },
                Op::VerifierNextCoeff { v: coeff },
                Op::VerifierAccumulate {
                    v: constant,
                    coeff,
                    matrix: 0,
                },
                Op::VerifierAccumulate {
                    v: value,
                    coeff,
                    matrix: 1,
                },
                Op::VerifierAccumulate {
                    v: one,
                    coeff,
                    matrix: 2,
                },
                Op::VerifierRngchkField {
                    val: value,
                    flag: one,
                    bits: 1,
                },
            ]
            .into_iter()
            .map(emit)
            .collect();
            code.push(OpCode::Ret {});
            let program = Program {
                field: F::ID,
                constraint_layouts: Some(bytecode::ConstraintLayouts::new(wl, cl)),
                functions: vec![Function {
                    name: "verifier".into(),
                    frame_size: 2 + 4 * F::STORAGE_CELLS,
                    code,
                    source_locations: vec![],
                }],
                entry_points: vec![bytecode::NO_ENTRY_POINT, bytecode::NO_ENTRY_POINT, 0],
                entry_blob_field_count: 0,
                global_frame_size: 0,
                constant_pool: vec![],
            }
            .to_binary();
            let w = [3u64, 5, 7, 11, 13, 17, 19, 23].map(F::from);
            let r = [29u64, 31, 37, 41, 43].map(F::from);
            let [w0, value, m0, m1, alpha, y0, y1, lookup] = w;
            // Explicit matrix rows, including table sum and lookup rows, at arbitrary w.
            let rows = [
                [w0 * value, value, w0],
                [y0, alpha, m0],
                [y1, alpha - w0, m1],
                [y0 + y1, w0, lookup],
                [lookup, alpha - value, w0],
            ];
            let expected = std::array::from_fn::<_, 3, _>(|column| {
                rows.iter()
                    .zip(r)
                    .fold(F::ZERO, |sum, (row, r)| sum + row[column] * r)
            });
            let result = run_verifier(&program, &r, &w, wl, cl, None).unwrap();
            assert_eq!([result.r_aw, result.r_bw, result.r_cw], expected);
            assert_eq!(result.instrumenter.final_memory_usage(), 0);
        }
        check::<Field>(Op::bn254);
        check::<mavros_artifacts::Goldilocks>(Op::goldilocks);
    }

    /// The frame's cell count and the compiler's are the same.
    #[test]
    fn the_frames_cell_count_is_the_compilers() {
        for bits in [1usize, 63, 64, 65, 127, 128, 129, 192, 1000, 16383, 16384] {
            assert_eq!(
                Frame::int_cells(bits as u64),
                IntBits::limbs_for_bits(bits),
                "the frame and the model disagree on the cells an int{bits} occupies"
            );
        }
    }

    #[test]
    fn a_limb_slice_reads_and_writes_the_cells_the_layout_gives_it() {
        // A bare cell array rather than a pushed frame, testing the addressing of `data`.
        let mut cells = vec![0u64; 16];
        let frame = Frame {
            data: cells.as_mut_ptr(),
        };

        // An int192 at cell 4 with a sentinel on each side: a slice one cell too long or too short
        // would read a sentinel or miss the value's third cell.
        frame.write_u64(3, 0xdead);
        for (i, word) in [1u64, 2, 3].into_iter().enumerate() {
            frame.write_u64(4 + i as isize, word);
        }
        frame.write_u64(7, 0xbeef);

        let cells = Frame::int_cells(192);
        assert_eq!(cells, 3);
        assert_eq!(unsafe { frame.read_limbs(4, cells) }, &[1, 2, 3]);

        unsafe { frame.read_limbs_mut(4, cells) }.copy_from_slice(&[7, 8, 9]);
        assert_eq!(unsafe { frame.read_limbs(4, cells) }, &[7, 8, 9]);
        assert_eq!(frame.read_u64(3), 0xdead, "the write ran off the low end");
        assert_eq!(frame.read_u64(7), 0xbeef, "the write ran off the high end");
    }

    /// A program over `F` (its field opcodes chosen by `pick`). Its witgen entry reads six inputs
    /// from its blob, writes one algebraic row and looks up into a table of every kind, the
    /// rangecheck table three times (once with a zero flag); its AD entry bumps two fresh
    /// witnesses by the first two coefficients.
    fn every_table_program<F: mavros_artifacts::MavrosField>(
        pick: fn(bytecode::FieldOpCode) -> OpCode,
    ) -> Vec<u64> {
        use bytecode::{ELEM_WORD, FieldOpCode, FramePosition as P, Function, Program};

        // The blob's elements, one per `F::STORAGE_CELLS` cells from cell 2: 3, 5, 1, 2, 30 and 0.
        let input = |index: usize| P(2 + F::STORAGE_CELLS * index);
        let (x, y, one, two, thirty, zero) =
            (input(0), input(1), input(2), input(3), input(4), input(5));
        let (z, words, array) = (P(28), P(32), P(35));
        let witgen = vec![
            pick(FieldOpCode::MulField { res: z, a: x, b: y }),
            pick(FieldOpCode::R1C { a: x, b: y, c: z }),
            pick(FieldOpCode::WriteWitness { val: x }),
            pick(FieldOpCode::WriteWitness { val: y }),
            pick(FieldOpCode::WriteWitness { val: z }),
            pick(FieldOpCode::RngchkField {
                val: two,
                flag: one,
                bits: 2,
            }),
            pick(FieldOpCode::RngchkField {
                val: two,
                flag: one,
                bits: 2,
            }),
            // A zero flag leaves the key unchecked, so it may lie outside the table.
            pick(FieldOpCode::RngchkField {
                val: thirty,
                flag: zero,
                bits: 2,
            }),
            pick(FieldOpCode::SpreadLookupField {
                val: one,
                result: one,
                flag: one,
                bits: 1,
            }),
            pick(FieldOpCode::Pow2LookupField {
                amount: one,
                factor: two,
                flag: one,
                size: 1,
            }),
            OpCode::MovConst {
                res: P(32),
                val: 10,
            },
            OpCode::MovConst {
                res: P(33),
                val: 20,
            },
            OpCode::MovConst {
                res: P(34),
                val: 30,
            },
            OpCode::ArrayAllocFromFrame {
                res: array,
                stride: 1,
                meta: crate::array::BoxedLayout::array(3, false),
                count: 3,
                source: words,
            },
            pick(FieldOpCode::ArrayLookupField {
                array,
                index: two,
                result: thirty,
                flag: one,
                stride: 1,
                elem_kind: ELEM_WORD,
            }),
            OpCode::DecRc { array },
            OpCode::Ret {},
        ];
        let (first, second, coeff) = (P(2), P(3), P(4));
        let ad = vec![
            OpCode::FreshWitness { res: first },
            OpCode::FreshWitness { res: second },
            pick(FieldOpCode::NextDCoeff { v: coeff }),
            pick(FieldOpCode::BumpDa { v: first, coeff }),
            pick(FieldOpCode::NextDCoeff { v: coeff }),
            pick(FieldOpCode::BumpDa { v: second, coeff }),
            OpCode::DecRc { array: first },
            OpCode::DecRc { array: second },
            OpCode::Ret {},
        ];
        let function = |name: &str, frame_size, code: Vec<OpCode>| Function {
            name: name.to_string(),
            frame_size,
            source_locations: vec![bytecode::SourceLocation::new("<test>", 1, 1); code.len()],
            code,
        };
        let (wl, cl) = every_table_layouts();
        Program {
            field: F::ID,
            constraint_layouts: Some(bytecode::ConstraintLayouts::new(wl, cl)),
            functions: vec![function("main", 36, witgen), function("ad", 8, ad)],
            entry_points: vec![0, 1],
            entry_blob_field_count: 6,
            global_frame_size: 0,
            constant_pool: Vec::new(),
        }
        .to_binary()
    }

    /// The layouts [`every_table_program`] fills exactly.
    fn every_table_layouts() -> (WitnessLayout, ConstraintsLayout) {
        let witness_layout = WitnessLayout {
            algebraic_size: 3,
            multiplicities_size: 11,
            challenges_size: 2,
            tables_data_size: 14,
            lookups_data_size: 9,
        };
        let constraints_layout = ConstraintsLayout {
            algebraic_size: 1,
            tables_data_size: 18,
            lookups_data_size: 9,
        };
        (witness_layout, constraints_layout)
    }

    /// The blob [`every_table_program`] reads.
    fn every_table_inputs<F: mavros_artifacts::MavrosField>() -> [F; 6] {
        [3u64, 5, 1, 2, 30, 0].map(F::from)
    }

    /// What [`run_every_table_program`] reports, field-independently.
    #[derive(Debug, PartialEq)]
    struct EveryTableRun {
        /// The algebraic constraint row's `a`, `b` and `c`, as signed integers.
        algebraic_row: [i128; 3],
        /// The pre-commitment witness, as signed integers.
        pre_commitment: Vec<i128>,
        /// Which post-commitment witnesses are nonzero.
        post_commitment_written: Vec<bool>,
        /// The AD run's derivatives at the two fresh witnesses, as signed integers.
        da: [i128; 2],
    }

    /// Runs [`every_table_program`] through the generic entry points and checks, in `F`, that
    /// `a·b = c` on every constraint row and that the post-commitment witness solves the
    /// rangecheck table's rows and the first lookup.
    fn run_every_table_program<F: mavros_artifacts::MavrosField>(
        pick: fn(bytecode::FieldOpCode) -> OpCode,
    ) -> EveryTableRun {
        use crate::test_field::signed;

        let binary = every_table_program::<F>(pick);
        let (witness_layout, constraints_layout) = every_table_layouts();
        let inputs = every_table_inputs::<F>();
        let result = run(&binary, witness_layout, constraints_layout, &inputs, None)
            .expect("every flagged lookup is in range");
        for (row, ((a, b), c)) in result
            .out_a
            .iter()
            .zip(&result.out_b)
            .zip(&result.out_c)
            .enumerate()
        {
            assert_eq!(*a * *b, *c, "constraint row {row}");
        }

        // The post-commitment witness starts with the challenges, then the tables' witnesses (the
        // rangecheck table's first), then the lookups'.
        let post = &result.out_wit_post_comm;
        assert_eq!(post[..2], fake_challenges::<F>(2));
        let alpha = post[0];
        for (row, multiplicity) in [0u64, 0, 2, 0].into_iter().enumerate() {
            let denominator = alpha - F::from(row as u64);
            assert_eq!(
                post[2 + row] * denominator,
                F::from(multiplicity),
                "table row {row}"
            );
        }
        assert_eq!(
            post[16] * (alpha - F::from(2u64)),
            F::ONE,
            "the first lookup"
        );

        let coeffs = (1..=constraints_layout.size() as u64)
            .map(F::from)
            .collect::<Vec<_>>();
        let (da, ..) = run_ad(&binary, &coeffs, witness_layout, constraints_layout, None)
            .expect("the AD entry does not trap");
        EveryTableRun {
            algebraic_row: [result.out_a[0], result.out_b[0], result.out_c[0]].map(signed),
            pre_commitment: result.out_wit_pre_comm.into_iter().map(signed).collect(),
            post_commitment_written: post.iter().map(|value| *value != F::ZERO).collect(),
            da: [signed(da[0]), signed(da[1])],
        }
    }

    /// The generic entry points end to end, comparing a run over Goldilocks' one-cell elements
    /// with one over bn254: the entry blob's stride, phase 2's reads of the `u64`s kept in field
    /// slots, its batch inversion and its challenges, and the AD entry's dA tape.
    #[test]
    fn the_entry_points_run_a_goldilocks_program() {
        use mavros_artifacts::Goldilocks;
        use mavros_artifacts::MavrosField;

        let bn254 = run_every_table_program::<Field>(bytecode::FieldOpCode::bn254);
        let goldilocks = run_every_table_program::<Goldilocks>(bytecode::FieldOpCode::goldilocks);
        // The algebraic witnesses, then the multiplicities of the rangecheck, spread, powers-of-two
        // and array tables, in that order.
        let pre_commitment = [3, 5, 15, 0, 0, 2, 0, 0, 1, 0, 1, 0, 0, 1];
        assert_eq!(bn254.algebraic_row, [3, 5, 15]);
        assert_eq!(bn254.pre_commitment, pre_commitment);
        assert_eq!(bn254.da, [1, 2]);
        // The challenges; the tables' witnesses, nonzero exactly where the row's multiplicity is
        // nonzero (an array row's x witness, `-β·v`, is nonzero whatever its multiplicity); and the
        // lookups' witnesses, zero only for the zero-flag lookup.
        let written = [
            "11", "0010", "01", "01", "101011", "1", "1", "0", "11", "11", "11",
        ]
        .concat()
        .chars()
        .map(|bit| bit == '1')
        .collect::<Vec<_>>();
        assert_eq!(bn254.post_commitment_written, written);
        assert_eq!(goldilocks, bn254);
        // The challenges are the seed's, reduced into Goldilocks.
        let challenges = fake_challenges::<Goldilocks>(2)
            .into_iter()
            .map(|challenge| challenge.to_canonical_limbs()[0])
            .collect::<Vec<_>>();
        assert_eq!(challenges, [3690730912593434115, 2291418963929149985]);
    }

    /// Phase 1's tables locate their multiplicities by offset, so a clone completes after the
    /// original is gone.
    #[test]
    fn a_cloned_phase_one_result_completes_on_its_own() {
        let binary = every_table_program::<Field>(bytecode::FieldOpCode::bn254);
        let (witness_layout, constraints_layout) = every_table_layouts();
        let phase1 = run_phase1(
            &binary,
            witness_layout,
            constraints_layout,
            &every_table_inputs::<Field>(),
            None,
        )
        .expect("every flagged lookup is in range");
        let copy = phase1.clone();
        let original = run_phase2_with_fake_challenges(phase1, witness_layout, constraints_layout);
        let completed = run_phase2_with_fake_challenges(copy, witness_layout, constraints_layout);
        assert_eq!(completed.out_wit_post_comm, original.out_wit_post_comm);
        assert_eq!(completed.out_a, original.out_a);
        assert_eq!(completed.out_b, original.out_b);
        assert_eq!(completed.out_c, original.out_c);
    }

    #[test]
    #[should_panic(expected = "phase 2 takes one challenge per challenge witness")]
    fn phase_two_refuses_a_challenge_count_off_the_layout() {
        let binary = every_table_program::<Field>(bytecode::FieldOpCode::bn254);
        let (witness_layout, constraints_layout) = every_table_layouts();
        let phase1 = run_phase1(
            &binary,
            witness_layout,
            constraints_layout,
            &every_table_inputs::<Field>(),
            None,
        )
        .expect("every flagged lookup is in range");
        run_phase2(
            phase1,
            &[Field::from(7u64)],
            witness_layout,
            constraints_layout,
        );
    }

    /// A program whose witgen entry has a `frame_size`-cell frame and an `inputs`-element blob, and
    /// whose two entries return at once.
    fn empty_program(frame_size: usize, inputs: usize, constraints: usize) -> Vec<u64> {
        use bytecode::{Function, Program};

        let function = |name: &str, frame_size| Function {
            name: name.to_string(),
            frame_size,
            code: vec![OpCode::Ret {}],
            source_locations: vec![bytecode::SourceLocation::new("<test>", 1, 1)],
        };
        let (wl, cl) = empty_layouts(constraints);
        Program {
            field: mavros_artifacts::FieldId::Bn254,
            constraint_layouts: Some(bytecode::ConstraintLayouts::new(wl, cl)),
            functions: vec![function("main", frame_size), function("ad", 2)],
            entry_points: vec![0, 1],
            entry_blob_field_count: inputs,
            global_frame_size: 0,
            constant_pool: Vec::new(),
        }
        .to_binary()
    }

    /// Layouts with no witnesses, and `constraints` algebraic constraints.
    fn empty_layouts(constraints: usize) -> (WitnessLayout, ConstraintsLayout) {
        let witness_layout = WitnessLayout {
            algebraic_size: 0,
            multiplicities_size: 0,
            challenges_size: 0,
            tables_data_size: 0,
            lookups_data_size: 0,
        };
        let constraints_layout = ConstraintsLayout {
            algebraic_size: constraints,
            tables_data_size: 0,
            lookups_data_size: 0,
        };
        (witness_layout, constraints_layout)
    }

    #[test]
    fn an_entry_frame_may_be_exactly_its_inputs() {
        // Two bn254 inputs fill cells 2 to 9.
        let (witness_layout, constraints_layout) = empty_layouts(0);
        let inputs = [1u64, 2].map(Field::from);
        let result = run(
            &empty_program(10, 2, 0),
            witness_layout,
            constraints_layout,
            &inputs,
            None,
        );
        assert!(result.is_ok());
    }

    #[test]
    #[should_panic(expected = "the entry's 9-cell frame cannot hold 2 inputs")]
    fn an_entry_frame_too_small_for_its_inputs_is_refused() {
        let (witness_layout, constraints_layout) = empty_layouts(0);
        let inputs = [1u64, 2].map(Field::from);
        let _ = run(
            &empty_program(9, 2, 0),
            witness_layout,
            constraints_layout,
            &inputs,
            None,
        );
    }

    /// A program runs only over the field its header names: its field opcodes' handlers read and
    /// write that field's element width.
    #[test]
    #[should_panic(expected = "a VM program over Bn254 cannot run over Goldilocks")]
    fn a_witgen_run_over_another_field_is_refused() {
        let (witness_layout, constraints_layout) = empty_layouts(0);
        let _ = run::<mavros_artifacts::Goldilocks>(
            &empty_program(2, 0, 0),
            witness_layout,
            constraints_layout,
            &[],
            None,
        );
    }

    #[test]
    #[should_panic(expected = "a VM program over Bn254 cannot run over Goldilocks")]
    fn an_ad_run_over_another_field_is_refused() {
        let (witness_layout, constraints_layout) = empty_layouts(0);
        let _ = run_ad::<mavros_artifacts::Goldilocks>(
            &empty_program(2, 0, 0),
            &[],
            witness_layout,
            constraints_layout,
            None,
        );
    }

    #[test]
    #[should_panic(expected = "a VM program over Bn254 cannot run over Goldilocks")]
    fn a_verifier_run_over_another_field_is_refused() {
        let (witness_layout, constraints_layout) = empty_layouts(0);
        let _ = run_verifier::<mavros_artifacts::Goldilocks>(
            &empty_program(2, 0, 0),
            &[],
            &[],
            witness_layout,
            constraints_layout,
            None,
        );
    }

    #[test]
    fn an_ad_run_takes_a_coefficient_per_constraint() {
        let (witness_layout, constraints_layout) = empty_layouts(2);
        let coeffs = [1u64, 2].map(Field::from);
        let result = run_ad(
            &empty_program(2, 0, 2),
            &coeffs,
            witness_layout,
            constraints_layout,
            None,
        );
        assert!(result.is_ok());
    }

    #[test]
    fn an_ad_run_with_a_coefficient_past_its_constraints_is_refused() {
        let (witness_layout, constraints_layout) = empty_layouts(2);
        let coeffs = [1u64, 2, 3].map(Field::from);
        let error = run_ad(
            &empty_program(2, 0, 2),
            &coeffs,
            witness_layout,
            constraints_layout,
            None,
        )
        .err()
        .expect("invalid coefficient length must fail");
        assert_eq!(
            error.message(),
            Some("constraint coefficient length: expected 2, got 3")
        );
    }

    #[test]
    fn an_ad_run_short_of_coefficients_is_refused() {
        let (witness_layout, constraints_layout) = empty_layouts(2);
        let error = run_ad(
            &empty_program(2, 0, 2),
            &[Field::from(1u64)],
            witness_layout,
            constraints_layout,
            None,
        )
        .err()
        .expect("invalid coefficient length must fail");
        assert_eq!(
            error.message(),
            Some("constraint coefficient length: expected 2, got 1")
        );
    }

    #[test]
    fn trap_error_source_paths_can_be_made_relative() {
        let mut error = TrapError {
            message: Some("array_get: index 2 out of bounds for array of length 2".into()),
            stack_trace: vec![
                bytecode::StackFrame {
                    function: "helper".to_string(),
                    location: bytecode::SourceLocation::new("/project/src/helper.nr", 24, 7),
                },
                bytecode::StackFrame {
                    function: "wrapper_main".to_string(),
                    location: bytecode::SourceLocation::new("<wrapper_main>", 1, 1),
                },
            ],
        };

        error.relativize_source_paths(std::path::Path::new("/project"));

        assert_eq!(
            error.to_string(),
            "VM trapped during execution: array_get: index 2 out of bounds for array of length 2\n  at helper (src/helper.nr:24:7)"
        );
    }

    #[test]
    fn explanatory_wrapper_frame_is_not_elided() {
        let error = TrapError {
            message: None,
            stack_trace: vec![bytecode::StackFrame {
                function: "wrapper_main".to_string(),
                location: bytecode::SourceLocation::new("<entry point: return>", 1, 1),
            }],
        };

        assert_eq!(
            error.to_string(),
            "VM trapped during execution\n  at wrapper_main (<entry point: return>)"
        );
    }
}
