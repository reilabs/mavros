use std::{
    alloc::{self, Layout},
    marker::PhantomData,
    mem::{self, size_of},
    str::FromStr,
};

use ark_ff::{AdditiveGroup, BigInt, Field as _, Fp, PrimeField as _};
use tracing::instrument;

pub use crate::InputValueOrdered;

use crate::bytecode::{ENTRY_AD, ENTRY_WITGEN, parse_program_header, spread_bits};
use crate::{
    ConstraintsLayout, Field, TableKind, WitnessLayout,
    array::BoxedValue,
    bytecode::{self, AllocationInstrumenter, AllocationType, OpCode, TableInfo, U128, VM},
};

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

    #[inline(always)]
    pub fn read_field(&self, offset: isize) -> Field {
        let a0 = unsafe { *self.data.offset(offset) };
        let a1 = unsafe { *self.data.offset(offset + 1) };
        let a2 = unsafe { *self.data.offset(offset + 2) };
        let a3 = unsafe { *self.data.offset(offset + 3) };
        Fp(BigInt([a0, a1, a2, a3]), PhantomData)
    }

    #[inline(always)]
    pub fn read_field_mut(&self, offset: isize) -> *mut Field {
        unsafe { self.data.offset(offset) as *mut Field }
    }

    #[inline(always)]
    pub fn read_u64_mut(&self, offset: isize) -> *mut u64 {
        unsafe { self.data.offset(offset) }
    }

    #[inline(always)]
    pub fn read_u128_mut(&self, offset: isize) -> *mut U128 {
        unsafe { self.data.offset(offset) as *mut U128 }
    }

    #[inline(always)]
    pub fn read_u64(&self, offset: isize) -> u64 {
        unsafe { *self.data.offset(offset) }
    }

    #[inline(always)]
    pub fn read_u128(&self, offset: isize) -> U128 {
        unsafe { *(self.data.offset(offset) as *const U128) }
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

    #[inline(always)]
    pub fn write_field(&self, offset: isize, field: Field) {
        let a0 = field.0.0[0];
        let a1 = field.0.0[1];
        let a2 = field.0.0[2];
        let a3 = field.0.0[3];
        unsafe {
            *self.data.offset(offset) = a0;
            *self.data.offset(offset + 1) = a1;
            *self.data.offset(offset + 2) = a2;
            *self.data.offset(offset + 3) = a3;
        }
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

fn prepare_dispatch(program: &mut [u64], code_start: usize) {
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

pub struct WitgenResult {
    pub out_wit_pre_comm: Vec<Field>,
    pub out_wit_post_comm: Vec<Field>,
    pub out_a: Vec<Field>,
    pub out_b: Vec<Field>,
    pub out_c: Vec<Field>,
    pub instrumenter: AllocationInstrumenter,
}

/// The program executed a trap: a failed assertion or rangecheck.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TrapError {
    stack_trace: Vec<bytecode::StackFrame>,
}

impl TrapError {
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
pub struct Phase1Result {
    pub out_wit_pre_comm: Vec<Field>,
    pub out_wit_post_comm: Vec<Field>,
    pub out_a: Vec<Field>,
    pub out_b: Vec<Field>,
    pub out_c: Vec<Field>,
    pub tables: Vec<TableInfo>,
    pub instrumenter: AllocationInstrumenter,
}

fn fix_multiplicities_section(wit: &mut [Field], witness_layout: WitnessLayout) {
    #[allow(clippy::needless_range_loop)]
    for i in witness_layout.multiplicities_start()..witness_layout.multiplicities_end() {
        // We used this as a *mut u64 when writing multiplicities, so we need to convert to an actual field element
        wit[i] = Field::from(wit[i].0.0[0]);
    }
}

/// Phase 1 of witness generation: executes the VM to produce the
/// pre-commitment witness and captures all intermediate state needed for
/// phase 2.
#[instrument(skip_all, name = "Interpreter::run_phase1")]
pub fn run_phase1(
    program: &[u64],
    witness_layout: WitnessLayout,
    constraints_layout: ConstraintsLayout,
    ordered_inputs: &[InputValueOrdered],
    debug_info: Option<bytecode::DebugInfo>,
) -> Result<Phase1Result, TrapError> {
    run_phase1_impl(
        program,
        witness_layout,
        constraints_layout,
        ordered_inputs,
        debug_info,
        false,
    )
    .map(|(result, _)| result)
}

/// Profiled phase 1 execution. The returned weights are simulated VM
/// instructions, which are deterministic across machines and runs.
pub fn run_phase1_profiled(
    program: &[u64],
    witness_layout: WitnessLayout,
    constraints_layout: ConstraintsLayout,
    ordered_inputs: &[InputValueOrdered],
    debug_info: Option<bytecode::DebugInfo>,
) -> Result<(Phase1Result, crate::FlamegraphProfile), TrapError> {
    run_phase1_impl(
        program,
        witness_layout,
        constraints_layout,
        ordered_inputs,
        debug_info,
        true,
    )
}

fn run_phase1_impl(
    program: &[u64],
    witness_layout: WitnessLayout,
    constraints_layout: ConstraintsLayout,
    ordered_inputs: &[InputValueOrdered],
    debug_info: Option<bytecode::DebugInfo>,
    profile: bool,
) -> Result<(Phase1Result, crate::FlamegraphProfile), TrapError> {
    let header = parse_program_header(program);
    let debug_info = debug_info.unwrap_or_default();
    let entry = *header
        .entry_points
        .get(ENTRY_WITGEN)
        .expect("Program has no witgen entry point");
    let global_frame_size = header.global_frame_size;
    let mut out_a = vec![Field::ZERO; constraints_layout.size()];
    let mut out_b = vec![Field::ZERO; constraints_layout.size()];
    let mut out_c = vec![Field::ZERO; constraints_layout.size()];
    let mut out_wit_pre_comm = vec![Field::ZERO; witness_layout.pre_commitment_size()];
    let flat_inputs = flatten_param_vec(ordered_inputs);
    // The program itself writes inputs to the witness tape via pinned WriteWitness instructions.
    let out_wit_post_comm = vec![Field::ZERO; witness_layout.post_commitment_size()];
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
        header.struct_layouts,
        header.constant_pool,
    );

    let frame = Frame::base_frame(program[entry + 1], &mut vm);

    // Main takes its inputs as a single Blob<Field; N> parameter stored by
    // value in the frame, starting right after the two return slots.
    for (input_index, el) in flat_inputs.iter().enumerate() {
        unsafe {
            *(frame.data.add(2 + (4 * input_index)) as *mut Field) = *el;
        }
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
            stack_trace: std::mem::take(&mut vm.stack_trace),
        });
    }

    fix_multiplicities_section(&mut out_wit_pre_comm, witness_layout);
    let instruction_profile = vm.take_instruction_profile();

    Ok((
        Phase1Result {
            out_wit_pre_comm,
            out_wit_post_comm,
            out_a,
            out_b,
            out_c,
            tables: vm.tables,
            instrumenter: vm.allocation_instrumenter,
        },
        instruction_profile,
    ))
}

/// Phase 2 of witness generation: uses real Fiat-Shamir challenges to
/// complete the post-commitment witness and constraint vectors.
#[instrument(skip_all, name = "Interpreter::run_phase2")]
pub fn run_phase2(
    mut phase1: Phase1Result,
    challenges: &[Field],
    witness_layout: WitnessLayout,
    constraints_layout: ConstraintsLayout,
) -> WitgenResult {
    // Write real challenges into the post-commitment witness vector.
    for (i, challenge) in challenges.iter().enumerate() {
        phase1.out_wit_post_comm[i] = *challenge;
    }

    let mut running_prod = Field::from(1);
    for tbl in phase1.tables.iter() {
        let alpha = phase1.out_wit_post_comm[0];
        let base = tbl.elem_inverses_constraint_section_offset;

        match tbl.kind {
            TableKind::RangeCheck => {
                // One constraint per entry: denom_i = α - i
                for i in 0..tbl.length {
                    let multiplicity = unsafe { *tbl.multiplicities_wit.add(i) };
                    let denom = alpha - Field::from(i as u64);
                    phase1.out_b[base + i] = denom;
                    phase1.out_c[base + i] = multiplicity;
                    if multiplicity != Field::ZERO {
                        phase1.out_a[base + i] = running_prod;
                        running_prod *= denom;
                    }
                }
            }
            TableKind::Spread => {
                // One folded constraint per entry. Both operands (key=i,
                // value=spread(i)) are constants, so β·spread(i) folds into the
                // denominator: denom_i = α - i + β·spread(i). spread(i) is
                // recomputed here rather than dumped by the VM.
                let beta = phase1.out_wit_post_comm[1];
                for i in 0..tbl.length {
                    let multiplicity = unsafe { *tbl.multiplicities_wit.add(i) };
                    let spread_i = Field::from(spread_bits(i as u32));
                    let denom = alpha - Field::from(i as u64) + beta * spread_i;
                    phase1.out_b[base + i] = denom;
                    phase1.out_c[base + i] = multiplicity;
                    if multiplicity != Field::ZERO {
                        phase1.out_a[base + i] = running_prod;
                        running_prod *= denom;
                    }
                }
            }
            TableKind::Array => {
                // Two constraints per entry: x_i = -β*v_i, denom_i = α - i - x_i
                let beta = phase1.out_wit_post_comm[1];
                for i in 0..tbl.length {
                    let multiplicity = unsafe { *tbl.multiplicities_wit.add(i) };

                    // Read v_i from the x-slot where the VM dumped it
                    let v_i = phase1.out_a[base + 2 * i];
                    let x_i = -beta * v_i;

                    // Fill x-constraint: β * v_i = -x_i
                    phase1.out_a[base + 2 * i] = beta;
                    phase1.out_b[base + 2 * i] = v_i;
                    phase1.out_c[base + 2 * i] = -x_i;

                    // Fill y-constraint slots (will be overwritten by batch inversion)
                    let denom = alpha - Field::from(i as u64) - x_i;
                    phase1.out_b[base + 2 * i + 1] = denom;
                    phase1.out_c[base + 2 * i + 1] = multiplicity;
                    if multiplicity != Field::ZERO {
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

        if matches!(tbl.kind, TableKind::RangeCheck | TableKind::Spread) {
            // One constraint per entry: y-values live at consecutive offsets.
            for i in (0..tbl.length).rev() {
                let multiplicity = phase1.out_c[base + i];
                let denom = phase1.out_b[base + i];
                let running_prod = phase1.out_a[base + i];
                if multiplicity != Field::ZERO {
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
                if multiplicity != Field::ZERO {
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
        let table_ix = phase1.out_a[cnst_off].0.0[0];
        let table = &phase1.tables[table_ix as usize];

        let alpha = phase1.out_wit_post_comm[0];

        if table.kind == TableKind::RangeCheck {
            // Key-only lookup (rangecheck): 1 constraint per lookup
            let flag_u64 = phase1.out_c[cnst_off].0.0[0];

            if flag_u64 == 0 {
                let key = phase1.out_b[cnst_off];
                let b_val = alpha - key;
                phase1.out_a[cnst_off] = Field::ZERO;
                phase1.out_b[cnst_off] = b_val;
                phase1.out_c[cnst_off] = Field::ZERO;
                phase1.out_wit_post_comm[wit_off] = Field::ZERO;
            } else {
                let ix_in_table = phase1.out_b[cnst_off].0.0[0];
                phase1.out_a[cnst_off] = phase1.out_a
                    [table.elem_inverses_constraint_section_offset + ix_in_table as usize];
                phase1.out_b[cnst_off] = phase1.out_b
                    [table.elem_inverses_constraint_section_offset + ix_in_table as usize];
                phase1.out_c[cnst_off] = Field::from(flag_u64);
                phase1.out_wit_post_comm[wit_off] = phase1.out_a[cnst_off];
                phase1.out_c[table.elem_inverses_constraint_section_offset + table.length] +=
                    phase1.out_a[cnst_off];
            }

            current_lookup_off += 1;
        } else {
            // Key-value lookup (array or spread): 2 constraints per lookup. The
            // looked-up key & value are witnesses regardless of how the table
            // is allocated; only the *table's* internal y-slot stride differs —
            // array stores x,y per entry (stride 2, y at the odd slot) while a
            // folded spread table stores just y per entry (stride 1).
            // Entry 1 (x-constraint): out_a=table_id, out_b=result_value, out_c=0
            // Entry 2 (y-constraint): out_a=table_id, out_b=index, out_c=flag
            let entry_stride = match table.kind {
                TableKind::Spread => 1,
                _ => 2,
            };
            let beta = phase1.out_wit_post_comm[1];
            let result_value = phase1.out_b[cnst_off];
            let flag_u64 = phase1.out_c[cnst_off + 1].0.0[0];

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
                phase1.out_a[y_cnst_off] = Field::ZERO;
                phase1.out_b[y_cnst_off] = b_val;
                phase1.out_c[y_cnst_off] = Field::ZERO;
                phase1.out_wit_post_comm[y_wit_off] = Field::ZERO;
            } else {
                let ix_in_table = phase1.out_b[y_cnst_off].0.0[0];
                let tbl_base = table.elem_inverses_constraint_section_offset;
                // Copy precomputed inverse from the table entry's y-slot
                // (array: 2*ix+1; spread: ix).
                let y_slot = tbl_base + entry_stride * ix_in_table as usize + (entry_stride - 1);
                phase1.out_a[y_cnst_off] = phase1.out_a[y_slot];
                phase1.out_b[y_cnst_off] = phase1.out_b[y_slot];
                phase1.out_c[y_cnst_off] = Field::from(flag_u64);
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

        if matches!(tbl.kind, TableKind::RangeCheck | TableKind::Spread) {
            // One constraint per entry: y-values at consecutive offsets, sum
            // constraint at offset length, one witness per entry.
            for i in 0..tbl.length {
                let multiplicity = phase1.out_c[base + i];
                if multiplicity != Field::ZERO {
                    let elem = phase1.out_a[base + i] * multiplicity;
                    phase1.out_a[base + i] = elem;
                    phase1.out_wit_post_comm[wit_base + i] = elem;
                    phase1.out_a[base + tbl.length] += elem;
                }
            }
            phase1.out_b[base + tbl.length] = Field::ONE;
        } else {
            // Array: y-values at odd offsets, sum constraint at offset 2*length
            for i in 0..tbl.length {
                let multiplicity = phase1.out_c[base + 2 * i + 1];
                if multiplicity != Field::ZERO {
                    let elem = phase1.out_a[base + 2 * i + 1] * multiplicity;
                    phase1.out_a[base + 2 * i + 1] = elem;
                    // x witness at even offset, y witness at odd offset
                    phase1.out_wit_post_comm[wit_base + 2 * i + 1] = elem;
                    phase1.out_a[base + 2 * tbl.length] += elem;
                }
                // x witness: x_i = -β * v_i; out_c stores -x_i = β*v_i, so negate
                phase1.out_wit_post_comm[wit_base + 2 * i] = -phase1.out_c[base + 2 * i];
            }
            phase1.out_b[base + 2 * tbl.length] = Field::ONE;
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

fn fake_challenges(count: usize) -> Vec<Field> {
    let mut fake_challenges = vec![Field::ZERO; count];
    let mut random =
        Field::from_bigint(BigInt::from_str("18765435241434657586764563434227903").unwrap())
            .unwrap();
    for challenge in fake_challenges.iter_mut() {
        *challenge = random;
        random = (random + Field::from(17)) * random;
    }
    fake_challenges
}

pub fn run_phase2_with_fake_challenges(
    phase1: Phase1Result,
    witness_layout: WitnessLayout,
    constraints_layout: ConstraintsLayout,
) -> WitgenResult {
    run_phase2(
        phase1,
        &fake_challenges(witness_layout.challenges_size),
        witness_layout,
        constraints_layout,
    )
}

// Old method implementation, for posterity
// Do not use in production
#[instrument(skip_all, name = "Interpreter::run")]
pub fn run(
    program: &[u64],
    witness_layout: WitnessLayout,
    constraints_layout: ConstraintsLayout,
    ordered_inputs: &[InputValueOrdered],
    debug_info: Option<bytecode::DebugInfo>,
) -> Result<WitgenResult, TrapError> {
    let phase1 = run_phase1(
        program,
        witness_layout,
        constraints_layout,
        ordered_inputs,
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
pub fn run_profiled(
    program: &[u64],
    witness_layout: WitnessLayout,
    constraints_layout: ConstraintsLayout,
    ordered_inputs: &[InputValueOrdered],
    debug_info: Option<bytecode::DebugInfo>,
) -> Result<(WitgenResult, crate::FlamegraphProfile), TrapError> {
    let (phase1, profile) = run_phase1_profiled(
        program,
        witness_layout,
        constraints_layout,
        ordered_inputs,
        debug_info,
    )?;

    Ok((
        run_phase2_with_fake_challenges(phase1, witness_layout, constraints_layout),
        profile,
    ))
}

#[instrument(skip_all, name = "Interpreter::run_ad")]
pub fn run_ad(
    program: &[u64],
    coeffs: &[Field],
    witness_layout: WitnessLayout,
    constraints_layout: ConstraintsLayout,
    debug_info: Option<bytecode::DebugInfo>,
) -> Result<(Vec<Field>, Vec<Field>, Vec<Field>, AllocationInstrumenter), TrapError> {
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
pub fn run_ad_profiled(
    program: &[u64],
    coeffs: &[Field],
    witness_layout: WitnessLayout,
    constraints_layout: ConstraintsLayout,
    debug_info: Option<bytecode::DebugInfo>,
) -> Result<
    (
        Vec<Field>,
        Vec<Field>,
        Vec<Field>,
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

fn run_ad_impl(
    program: &[u64],
    coeffs: &[Field],
    witness_layout: WitnessLayout,
    constraints_layout: ConstraintsLayout,
    debug_info: Option<bytecode::DebugInfo>,
    profile: bool,
) -> Result<
    (
        Vec<Field>,
        Vec<Field>,
        Vec<Field>,
        AllocationInstrumenter,
        crate::FlamegraphProfile,
    ),
    TrapError,
> {
    let header = parse_program_header(program);
    let debug_info = debug_info.unwrap_or_default();
    let entry = *header
        .entry_points
        .get(ENTRY_AD)
        .expect("Program has no AD entry point");
    let global_frame_size = header.global_frame_size;
    let mut out_da = vec![Field::ZERO; witness_layout.size()];
    let mut out_db = vec![Field::ZERO; witness_layout.size()];
    let mut out_dc = vec![Field::ZERO; witness_layout.size()];
    let mut global_frame = vec![0u64; global_frame_size];
    let mut vm = VM::new_ad(
        out_da.as_mut_ptr(),
        out_db.as_mut_ptr(),
        out_dc.as_mut_ptr(),
        coeffs.as_ptr(),
        witness_layout,
        constraints_layout,
        global_frame.as_mut_ptr(),
        header.struct_layouts,
        header.constant_pool,
    );

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
    vm.report_opcode_profile("AD");

    if vm.trapped {
        return Err(TrapError {
            stack_trace: std::mem::take(&mut vm.stack_trace),
        });
    }

    let instruction_profile = vm.take_instruction_profile();
    Ok((
        out_da,
        out_db,
        out_dc,
        vm.allocation_instrumenter,
        instruction_profile,
    ))
}

fn flatten_param_vec(vec: &[InputValueOrdered]) -> Vec<Field> {
    let mut encoded_value = Vec::new();
    for elem in vec {
        encoded_value.extend(flatten_params(elem));
    }
    encoded_value
}

fn flatten_params(value: &InputValueOrdered) -> Vec<Field> {
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

    #[test]
    fn trap_error_source_paths_can_be_made_relative() {
        let mut error = TrapError {
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
            "VM trapped during execution\n  at helper (src/helper.nr:24:7)"
        );
    }

    #[test]
    fn explanatory_wrapper_frame_is_not_elided() {
        let error = TrapError {
            stack_trace: vec![bytecode::StackFrame {
                function: "wrapper_main".to_string(),
                location: bytecode::SourceLocation::new("<public return value check>", 1, 1),
            }],
        };

        assert_eq!(
            error.to_string(),
            "VM trapped during execution\n  at wrapper_main (<public return value check>)"
        );
    }
}
