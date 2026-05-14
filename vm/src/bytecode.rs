#![allow(unused_variables)]

use crate::interpreter::dispatch;
use crate::{ConstraintsLayout, Field, WitnessLayout};
use ark_ff::{AdditiveGroup as _, BigInteger as _};
use mavros_opcode_gen::interpreter;

use crate::array::{BoxedLayout, BoxedValue, DataType, StructDescriptor};
use crate::interpreter::{Frame, Handler};

use std::fmt::Display;
use std::ptr;

pub const LIMBS: usize = 4;

/// Element storage kind for array lookup opcodes.
/// Encoded as usize for compatibility with the opcode proc macro.
pub const ELEM_WORD: usize = 0;
pub const ELEM_FIELD: usize = 1;
pub const ELEM_WITNESS: usize = 2;

/// Read an array element as a Field and bump out_db accordingly.
#[inline(always)]
unsafe fn lookup_elem_bump_db(ptr: *mut u64, elem_kind: usize, coeff: Field, vm: &mut VM) {
    match elem_kind {
        ELEM_WORD => unsafe {
            let v = Field::from(*(ptr as *const u64));
            *vm.data.as_ad.out_db += coeff * v;
        },
        ELEM_FIELD => unsafe {
            let v = *(ptr as *const Field);
            *vm.data.as_ad.out_db += coeff * v;
        },
        ELEM_WITNESS => {
            let elem = BoxedValue(unsafe { *(ptr as *const *mut u64) });
            elem.bump_db(coeff, vm);
        }
        _ => unreachable!(),
    }
}

/// Read a pure (non-WitnessOf) array element as a Field value.
#[inline(always)]
unsafe fn read_pure_elem_as_field(ptr: *mut u64, elem_kind: usize) -> Field {
    match elem_kind {
        ELEM_WORD => Field::from(unsafe { *(ptr as *const u64) }),
        ELEM_FIELD => unsafe { *(ptr as *const Field) },
        _ => unreachable!(),
    }
}

#[derive(Clone, Copy)]
pub struct FramePosition(pub usize);

impl FramePosition {
    pub fn offset(&self, offset: isize) -> FramePosition {
        FramePosition(self.0.checked_add_signed(offset).unwrap())
    }
    pub fn return_data_ptr() -> FramePosition {
        FramePosition(0)
    }
    pub fn return_address_ptr() -> FramePosition {
        FramePosition(1)
    }
}

pub struct JumpTarget(pub isize);

#[derive(Clone)]
pub enum AllocationType {
    Stack,
    Heap,
}

#[derive(Clone)]
pub enum AlocationEvent {
    Alloc(AllocationType, usize),
    Free(AllocationType, usize),
}

#[derive(Clone)]
pub struct AllocationInstrumenter {
    pub events: Vec<AlocationEvent>,
}

impl AllocationInstrumenter {
    pub fn new() -> Self {
        Self { events: vec![] }
    }

    pub fn alloc(&mut self, ty: AllocationType, size: usize) {
        self.events.push(AlocationEvent::Alloc(ty, size));
    }

    pub fn free(&mut self, ty: AllocationType, size: usize) {
        self.events.push(AlocationEvent::Free(ty, size));
    }

    /// Returns the final memory usage in bytes (0 means no leak).
    pub fn final_memory_usage(&self) -> usize {
        let mut current_stack = 0usize;
        let mut current_heap = 0usize;

        for event in &self.events {
            match event {
                AlocationEvent::Alloc(AllocationType::Stack, size) => {
                    current_stack += size * 8;
                }
                AlocationEvent::Alloc(AllocationType::Heap, size) => {
                    current_heap += size * 8;
                }
                AlocationEvent::Free(AllocationType::Stack, size) => {
                    current_stack = current_stack.saturating_sub(*size * 8);
                }
                AlocationEvent::Free(AllocationType::Heap, size) => {
                    current_heap = current_heap.saturating_sub(*size * 8);
                }
            }
        }

        current_stack + current_heap
    }
}

#[derive(Clone)]
pub struct TableInfo {
    pub multiplicities_wit: *mut Field,
    pub num_indices: usize,
    pub num_values: usize,
    pub length: usize,
    pub elem_inverses_witness_section_offset: usize,
    pub elem_inverses_constraint_section_offset: usize,
    /// Per-dimension sizes (row-major). For 1-D tables `dims = [length]`;
    /// for 2-D `dims = [d_outer, d_inner]`, etc. Used by Phase 2 to decompose
    /// slot index `s ∈ 0..length` into coordinates `i_j(s)` for the β-power
    /// LogUp denominator `α − v_s − Σ β^j · i_j(s)`.
    pub dims: Vec<usize>,
}

#[derive(Copy, Clone)]
pub struct FwdArrays {
    pub out_a: *mut Field,
    pub out_b: *mut Field,
    pub out_c: *mut Field,
    pub out_a_base: *mut Field,
    pub algebraic_witness: *mut Field,
    pub multiplicities_witness: *mut Field,
    pub lookups_a: *mut Field,
    pub lookups_b: *mut Field,
    pub lookups_c: *mut Field,
    pub elem_inverses_constraint_section_offset: usize,
    pub elem_inverses_witness_section_offset: usize,
}

#[derive(Copy, Clone)]
pub struct AdArrays {
    pub out_da: *mut Field,
    pub out_db: *mut Field,
    pub out_dc: *mut Field,
    pub ad_coeffs: *const Field,
    pub current_wit_off: usize,
    pub logup_wit_challenge_off: usize,
    pub current_wit_multiplicities_off: usize,
    pub current_wit_tables_off: usize,
    pub current_wit_lookups_off: usize,
    pub current_cnst_off: usize,
    pub current_cnst_tables_off: usize,
    pub current_cnst_lookups_off: usize,
    /// Constraint-section base offset of the β-power chain R1Cs (which precede
    /// per-table content in the tables_data section).
    pub beta_chain_cnst_base: usize,
    /// Witness-section base offset of the β-power chain witnesses, relative
    /// to `logup_wit_challenge_off` (i.e. β² is at offset `+1+1`, etc.).
    pub beta_chain_wit_base: usize,
}

pub union Arrays {
    pub as_forward: FwdArrays,
    pub as_ad: AdArrays,
}

pub struct VM {
    pub data: Arrays,
    pub allocation_instrumenter: AllocationInstrumenter,
    pub tables: Vec<TableInfo>,
    pub rgchk_8: Option<usize>,
    pub spread_tables: [Option<usize>; 17],
    pub globals: *mut u64,
    pub struct_layouts: Vec<StructDescriptor>,
    /// Whether AD bumps for the β-power chain R1Cs have been emitted yet
    /// (only relevant on the AD path; emitted once per program by the first
    /// `dnd_array_lookup_field` invocation).
    pub beta_chain_ad_emitted: bool,
}

impl VM {
    pub fn new_witgen(
        out_a: *mut Field,
        out_b: *mut Field,
        out_c: *mut Field,
        algebraic_witness: *mut Field,
        multiplicities_witness: *mut Field,
        lookups_a: *mut Field,
        lookups_b: *mut Field,
        lookups_c: *mut Field,
        elem_inverses_constraint_section_offset: usize,
        elem_inverses_witness_section_offset: usize,
        globals: *mut u64,
        struct_layouts: Vec<StructDescriptor>,
    ) -> Self {
        Self {
            data: Arrays {
                as_forward: FwdArrays {
                    out_a,
                    out_b,
                    out_c,
                    out_a_base: out_a,
                    algebraic_witness,
                    multiplicities_witness,
                    lookups_b,
                    lookups_a,
                    lookups_c,
                    elem_inverses_constraint_section_offset,
                    elem_inverses_witness_section_offset,
                },
            },
            allocation_instrumenter: AllocationInstrumenter::new(),
            tables: vec![],
            rgchk_8: None,
            spread_tables: [None; 17],
            globals,
            struct_layouts,
            beta_chain_ad_emitted: false,
        }
    }

    pub fn new_ad(
        out_da: *mut Field,
        out_db: *mut Field,
        out_dc: *mut Field,
        ad_coeffs: *const Field,

        witness_layout: WitnessLayout,
        constraints_layout: ConstraintsLayout,
        globals: *mut u64,
        struct_layouts: Vec<StructDescriptor>,
    ) -> Self {
        Self {
            data: Arrays {
                as_ad: AdArrays {
                    out_da,
                    out_db,
                    out_dc,
                    ad_coeffs,
                    current_wit_off: 0,
                    logup_wit_challenge_off: witness_layout.challenges_start(),
                    current_wit_multiplicities_off: witness_layout.multiplicities_start(),
                    current_wit_tables_off: witness_layout.per_table_data_start(),
                    current_wit_lookups_off: witness_layout.lookups_data_start(),
                    current_cnst_off: 0,
                    current_cnst_tables_off: constraints_layout.per_table_data_start(),
                    current_cnst_lookups_off: constraints_layout.lookups_data_start(),
                    beta_chain_cnst_base: constraints_layout.tables_data_start(),
                    beta_chain_wit_base: witness_layout.tables_data_start()
                        - witness_layout.challenges_start(),
                },
            },
            allocation_instrumenter: AllocationInstrumenter::new(),
            tables: vec![],
            rgchk_8: None,
            spread_tables: [None; 17],
            globals,
            struct_layouts,
            beta_chain_ad_emitted: false,
        }
    }

    // pub fn new_
}

/// Compute spread of a u32: interleave zero bits between each bit.
fn spread_bits(v: u32) -> u64 {
    let mut x = v as u64;
    x = (x | (x << 16)) & 0x0000_FFFF_0000_FFFF;
    x = (x | (x << 8)) & 0x00FF_00FF_00FF_00FF;
    x = (x | (x << 4)) & 0x0F0F_0F0F_0F0F_0F0F;
    x = (x | (x << 2)) & 0x3333_3333_3333_3333;
    x = (x | (x << 1)) & 0x5555_5555_5555_5555;
    x
}

/// Compact even-positioned bits into contiguous low bits.
fn compact_bits(mut x: u64) -> u32 {
    x &= 0x5555_5555_5555_5555;
    x = (x | (x >> 1)) & 0x3333_3333_3333_3333;
    x = (x | (x >> 2)) & 0x0F0F_0F0F_0F0F_0F0F;
    x = (x | (x >> 4)) & 0x00FF_00FF_00FF_00FF;
    x = (x | (x >> 8)) & 0x0000_FFFF_0000_FFFF;
    x = (x | (x >> 16)) & 0x0000_0000_FFFF_FFFF;
    x as u32
}

/// Extract even bits and odd bits from a spread sum. Returns (odd_bits, even_bits).
fn unspread_bits(v: u64) -> (u32, u32) {
    let even = compact_bits(v);
    let odd = compact_bits(v >> 1);
    (odd, even)
}

/// Emit a forward N-D key-value lookup. Writes N+1 tape entries:
///   Entry 1 (R1C 1, β·k_1=x_1):   table_id, k_1,   0
///   Entry j (R1C j, β^j·k_j=x_j): table_id, k_j,   0
///   Entry N+1 (y · denom = flag):  table_id, value, flag_u64
unsafe fn forward_nd_lookup_emit(
    table_idx: usize,
    keys: &[Field],
    value: Field,
    flag_u64: u64,
    flat_idx: u64,
    vm: &mut VM,
) {
    let table_info = &vm.tables[table_idx];

    if flag_u64 != 0 {
        unsafe {
            let ptr = table_info.multiplicities_wit.offset(flat_idx as isize);
            *(ptr as *mut u64) += flag_u64;
        }
    }

    // Key R1C entries: one per key, all with c=0.
    for key in keys.iter() {
        unsafe {
            *(vm.data.as_forward.lookups_a as *mut u64) = table_idx as u64;
            *vm.data.as_forward.lookups_b = *key;
            *(vm.data.as_forward.lookups_c as *mut u64) = 0;
            vm.data.as_forward.lookups_a = vm.data.as_forward.lookups_a.offset(1);
            vm.data.as_forward.lookups_b = vm.data.as_forward.lookups_b.offset(1);
            vm.data.as_forward.lookups_c = vm.data.as_forward.lookups_c.offset(1);
        }
    }

    // y-constraint entry: (table_id, value, flag).
    unsafe {
        *(vm.data.as_forward.lookups_a as *mut u64) = table_idx as u64;
        *vm.data.as_forward.lookups_b = value;
        *(vm.data.as_forward.lookups_c as *mut u64) = flag_u64;
        vm.data.as_forward.lookups_a = vm.data.as_forward.lookups_a.offset(1);
        vm.data.as_forward.lookups_b = vm.data.as_forward.lookups_b.offset(1);
        vm.data.as_forward.lookups_c = vm.data.as_forward.lookups_c.offset(1);
    }
}

/// Walk an N-D nested pure array along `keys` to fetch the leaf Field. Also
/// records each level's dimension size into `dims_out`. The last level is
/// addressed with `leaf_stride` (LIMBS for Field, 1 for u_N). Intermediate
/// levels are addressed as BoxedArray-of-pointers (stride 1).
unsafe fn nd_array_walk(
    root: BoxedValue,
    keys: &[u64],
    leaf_stride: usize,
    elem_kind: usize,
    dims_out: &mut Vec<usize>,
) -> Field {
    let mut cur = root;
    let n = keys.len();
    for (level, &k) in keys.iter().enumerate() {
        if level == n - 1 {
            // Leaf level: cur is a flat array of leaf elements.
            let leaf_len = cur.layout().array_size() / leaf_stride;
            dims_out.push(leaf_len);
            let leaf_ptr = cur.array_idx(k as usize, leaf_stride);
            return unsafe { read_pure_elem_as_field(leaf_ptr, elem_kind) };
        }
        // Intermediate level: cur is BoxedArray of pointers.
        let level_len = cur.layout().array_size();
        dims_out.push(level_len);
        let inner_ptr = cur.array_idx(k as usize, 1) as *mut BoxedValue;
        cur = unsafe { *inner_ptr };
    }
    unreachable!("nd_array_walk requires at least one key");
}

/// Emit a forward key-value lookup under the β-power LogUp encoding:
/// bumps multiplicity[key] and writes 2 lookup tape entries.
///
/// R1Cs (matched by `run_phase2`):
///   Entry 1 (key R1C): β · key = x₁    A=β  B=key   C=x₁
///   Entry 2 (y R1C):   y · (α − value − x₁) = flag    A=y  B=denom  C=flag
///
/// Tape data laid down by Phase 1 (Phase 2 reads + rewrites these in place):
///   Entry 1: out_a=table_id, out_b=key, out_c=0
///   Entry 2: out_a=table_id, out_b=value, out_c=flag_u64 (low-limb-as-u64)
unsafe fn forward_kv_lookup_emit(
    table_idx: usize,
    key: Field,
    value: Field,
    flag_u64: u64,
    vm: &mut VM,
) {
    let table_info = &vm.tables[table_idx];

    // Multiplicity bump (flat slot index == key for 1-D table; rejected by Phase 2 otherwise).
    if flag_u64 != 0 {
        let key_u64 = ark_ff::PrimeField::into_bigint(key).0[0];
        unsafe {
            let ptr = table_info.multiplicities_wit.offset(key_u64 as isize);
            *(ptr as *mut u64) += flag_u64;
        }
    }

    // Entry 1: key R1C
    unsafe {
        *(vm.data.as_forward.lookups_a as *mut u64) = table_idx as u64;
        *vm.data.as_forward.lookups_b = key;
        *(vm.data.as_forward.lookups_c as *mut u64) = 0;
        vm.data.as_forward.lookups_a = vm.data.as_forward.lookups_a.offset(1);
        vm.data.as_forward.lookups_b = vm.data.as_forward.lookups_b.offset(1);
        vm.data.as_forward.lookups_c = vm.data.as_forward.lookups_c.offset(1);
    }

    // Entry 2: y R1C
    unsafe {
        *(vm.data.as_forward.lookups_a as *mut u64) = table_idx as u64;
        *vm.data.as_forward.lookups_b = value;
        *(vm.data.as_forward.lookups_c as *mut u64) = flag_u64;
        vm.data.as_forward.lookups_a = vm.data.as_forward.lookups_a.offset(1);
        vm.data.as_forward.lookups_b = vm.data.as_forward.lookups_b.offset(1);
        vm.data.as_forward.lookups_c = vm.data.as_forward.lookups_c.offset(1);
    }
}

/// Emit AD bumps for a β-power LogUp N-D key-value lookup:
///   R1C j (1..=N): β^j · k_j = x_j           A=(β^j,1)  B=k_j    C=(x_j,1)
///   R1C N+1:       y · (α − value − Σ x_j) = flag
///                                            A=(y,1)  B=α−value−Σx_j   C=flag
/// Plus each query's `y` contributes (y_wit, 1) to the sum constraint's C.
unsafe fn ad_nd_kv_lookup_emit(
    table_idx: usize,
    keys: &[BoxedValue],
    value: BoxedValue,
    flag: BoxedValue,
    vm: &mut VM,
) {
    let table_info = &vm.tables[table_idx];
    let cnst_off = table_info.elem_inverses_constraint_section_offset;
    let length = table_info.length;
    let n_keys = keys.len();

    // Sum constraint at offset `length`.
    let inv_sum_coeff = unsafe { *vm.data.as_ad.ad_coeffs.add(cnst_off + length) };

    // Allocate one (x_j_coeff, x_j_wit_off) pair per key R1C, plus (y_coeff, y_wit_off).
    let mut x_coeffs = Vec::with_capacity(n_keys);
    let mut x_wit_offs = Vec::with_capacity(n_keys);
    for _ in 0..n_keys {
        let x_coeff = unsafe {
            let r = *vm
                .data
                .as_ad
                .ad_coeffs
                .add(vm.data.as_ad.current_cnst_lookups_off);
            vm.data.as_ad.current_cnst_lookups_off += 1;
            r
        };
        let x_wit_off = unsafe {
            let r = vm.data.as_ad.current_wit_lookups_off;
            vm.data.as_ad.current_wit_lookups_off += 1;
            r
        };
        x_coeffs.push(x_coeff);
        x_wit_offs.push(x_wit_off);
    }
    let y_coeff = unsafe {
        let r = *vm
            .data
            .as_ad
            .ad_coeffs
            .add(vm.data.as_ad.current_cnst_lookups_off);
        vm.data.as_ad.current_cnst_lookups_off += 1;
        r
    };
    let y_wit_off = unsafe {
        let r = vm.data.as_ad.current_wit_lookups_off;
        vm.data.as_ad.current_wit_lookups_off += 1;
        r
    };

    // Per key R1C j: A=(β^j,1), B=k_j, C=(x_j,1)
    // β^1 is at logup_wit_challenge_off + 1, β^j (j>=2) is at
    // table_data_start + (j-2) which translates to logup_wit_challenge_off + 2 + (j-2)
    // because the post-comm witness is laid out as [α, β, β², β³, …].
    for (j, key) in keys.iter().enumerate() {
        let beta_pow_off = if j == 0 {
            // β¹ = β challenge at offset +1.
            vm.data.as_ad.logup_wit_challenge_off + 1
        } else {
            // β^(j+1) at logup_wit_challenge_off + 2 + (j-1) = + 1 + j.
            // (j=1 → β² at offset +2, j=2 → β³ at +3, etc.)
            vm.data.as_ad.logup_wit_challenge_off + 1 + j
        };
        unsafe {
            *vm.data.as_ad.out_da.add(beta_pow_off) += x_coeffs[j];
        }
        key.bump_db(x_coeffs[j], vm);
        unsafe {
            *vm.data.as_ad.out_dc.add(x_wit_offs[j]) += x_coeffs[j];
        }
    }

    // y R1C: A=(y,1), B=(α,1) + (−value) + (−x_1) + … + (−x_N), C=flag
    unsafe {
        *vm.data.as_ad.out_da.add(y_wit_off) += y_coeff;
        *vm.data
            .as_ad
            .out_db
            .add(vm.data.as_ad.logup_wit_challenge_off) += y_coeff;
        for x_wit_off in &x_wit_offs {
            *vm.data.as_ad.out_db.add(*x_wit_off) -= y_coeff;
        }
    }
    value.bump_db(-y_coeff, vm);
    flag.bump_dc(y_coeff, vm);

    // Sum constraint: this query's y_wit contributes (y_wit, 1) to C.
    unsafe {
        *vm.data.as_ad.out_dc.add(y_wit_off) += inv_sum_coeff;
    }
}

/// Emit AD bumps for a β-power LogUp key-value lookup (1-D):
///   R1C 1: β · key = x_1     A=(β,1)  B=key    C=(x_1,1)
///   R1C 2: y · (α − value − x_1) = flag
///                            A=(y,1)  B=α−value−x_1   C=flag
/// Plus each query's `y` contributes (y_wit, 1) to the sum constraint's C.
unsafe fn ad_kv_lookup_emit(
    table_idx: usize,
    key: BoxedValue,
    value: BoxedValue,
    flag: BoxedValue,
    vm: &mut VM,
) {
    let table_info = &vm.tables[table_idx];
    let cnst_off = table_info.elem_inverses_constraint_section_offset;
    let length = table_info.length;

    // R1C 1 coeffs (β · key = x_1)
    let x_coeff = unsafe {
        let r = *vm
            .data
            .as_ad
            .ad_coeffs
            .add(vm.data.as_ad.current_cnst_lookups_off);
        vm.data.as_ad.current_cnst_lookups_off += 1;
        r
    };
    let x_wit_off = unsafe {
        let r = vm.data.as_ad.current_wit_lookups_off;
        vm.data.as_ad.current_wit_lookups_off += 1;
        r
    };
    // R1C 2 coeffs (y · (α − value − x_1) = flag)
    let y_coeff = unsafe {
        let r = *vm
            .data
            .as_ad
            .ad_coeffs
            .add(vm.data.as_ad.current_cnst_lookups_off);
        vm.data.as_ad.current_cnst_lookups_off += 1;
        r
    };
    let y_wit_off = unsafe {
        let r = vm.data.as_ad.current_wit_lookups_off;
        vm.data.as_ad.current_wit_lookups_off += 1;
        r
    };
    // Sum constraint sits at offset `length` (one R1C per table slot + 1 sum).
    let inv_sum_coeff = unsafe { *vm.data.as_ad.ad_coeffs.add(cnst_off + length) };

    // R1C 1: A=(β,1), B=key, C=(x_1,1)
    unsafe {
        *vm.data
            .as_ad
            .out_da
            .add(vm.data.as_ad.logup_wit_challenge_off + 1) += x_coeff;
    }
    key.bump_db(x_coeff, vm);
    unsafe {
        *vm.data.as_ad.out_dc.add(x_wit_off) += x_coeff;
    }

    // R1C 2: A=(y,1), B=(α,1)+(−value)+(−x_1,1), C=flag
    unsafe {
        *vm.data.as_ad.out_da.add(y_wit_off) += y_coeff;
        *vm.data
            .as_ad
            .out_db
            .add(vm.data.as_ad.logup_wit_challenge_off) += y_coeff;
        *vm.data.as_ad.out_db.add(x_wit_off) -= y_coeff;
    }
    value.bump_db(-y_coeff, vm);
    flag.bump_dc(y_coeff, vm);

    // Sum constraint: this query's y_wit contributes (y_wit, 1) to C.
    unsafe {
        *vm.data.as_ad.out_dc.add(y_wit_off) += inv_sum_coeff;
    }
}

#[interpreter]
mod def {
    #[raw_opcode]
    fn jmp(pc: *const u64, frame: Frame, vm: &mut VM, target: JumpTarget) {
        let pc = unsafe { pc.offset(target.0) };
        unsafe { dispatch(pc, frame, vm) };
    }

    #[raw_opcode]
    fn jmp_if(
        pc: *const u64,
        frame: Frame,
        vm: &mut VM,
        #[frame] cond: u64,
        if_t: JumpTarget,
        if_f: JumpTarget,
    ) {
        let target = if cond != 0 { if_t } else { if_f };
        let pc = unsafe { pc.offset(target.0) };
        // println!("jmp_if: cond={} target={:?}", cond, pc);
        unsafe { dispatch(pc, frame, vm) };
    }

    #[raw_opcode]
    fn call(
        pc: *const u64,
        frame: Frame,
        vm: &mut VM,
        func: JumpTarget,
        args: &[(usize, FramePosition)],
        ret: FramePosition,
    ) {
        let func_pc = unsafe { pc.offset(func.0) };
        let func_frame_size = unsafe { *func_pc.offset(-1) };
        let new_frame = Frame::push(func_frame_size, frame, vm);
        let ret_data_ptr = unsafe { frame.data.add(ret.0) };
        let ret_pc = unsafe { pc.offset(4 + 2 * args.len() as isize) };

        unsafe {
            *new_frame.data = ret_data_ptr as u64;
            *new_frame.data.offset(1) = ret_pc as u64;
        };

        let mut current_child = unsafe { new_frame.data.offset(2) };

        for (i, (arg_size, arg_pos)) in args.iter().enumerate() {
            unsafe { frame.write_to(current_child, arg_pos.0 as isize, *arg_size) };
            current_child = unsafe { current_child.add(*arg_size) };
        }

        unsafe { dispatch(func_pc, new_frame, vm) };
    }

    #[raw_opcode]
    fn ret(_pc: *const u64, frame: Frame, vm: &mut VM) {
        let ret_address = unsafe { *frame.data.offset(1) } as *mut u64;
        let new_frame = frame.pop(vm);
        if new_frame.data.is_null() {
            return;
        }
        unsafe { dispatch(ret_address, new_frame, vm) };
    }

    #[raw_opcode]
    fn r1c(
        pc: *const u64,
        frame: Frame,
        vm: &mut VM,
        #[frame] a: Field,
        #[frame] b: Field,
        #[frame] c: Field,
    ) {
        unsafe {
            *vm.data.as_forward.out_a = a;
            *vm.data.as_forward.out_b = b;
            *vm.data.as_forward.out_c = c;
        }

        unsafe {
            vm.data.as_forward.out_a = vm.data.as_forward.out_a.offset(1);
            vm.data.as_forward.out_b = vm.data.as_forward.out_b.offset(1);
            vm.data.as_forward.out_c = vm.data.as_forward.out_c.offset(1);
        };
        let pc = unsafe { pc.offset(4) };
        unsafe { dispatch(pc, frame, vm) };
    }

    #[raw_opcode]
    fn write_witness(pc: *const u64, frame: Frame, vm: &mut VM, #[frame] val: Field) {
        unsafe {
            *vm.data.as_forward.algebraic_witness = val;
            vm.data.as_forward.algebraic_witness = vm.data.as_forward.algebraic_witness.offset(1);
        };
        let pc = unsafe { pc.offset(2) };
        unsafe { dispatch(pc, frame, vm) };
    }

    #[opcode]
    fn nop() {}

    #[opcode]
    fn mov_const(#[out] res: *mut u64, val: u64) {
        unsafe {
            *res = val;
        }
    }

    #[opcode]
    fn mov_frame(frame: Frame, target: FramePosition, source: FramePosition, size: usize) {
        frame.memcpy(target.0 as isize, source.0 as isize, size);
    }

    #[opcode]
    fn write_ptr(
        frame: Frame,
        #[frame] ptr: *mut u64,
        offset: isize,
        src: FramePosition,
        size: usize,
    ) {
        let ptr = unsafe { ptr.offset(offset) };
        unsafe { frame.write_to(ptr, src.0 as isize, size) };
    }

    #[opcode]
    fn add_int(#[out] res: *mut u64, #[frame] a: u64, #[frame] b: u64, bits: u64) {
        unsafe {
            let sum = a.wrapping_add(b);
            *res = if bits >= 64 {
                sum
            } else {
                sum & ((1u64 << bits) - 1)
            };
        }
    }

    #[opcode]
    fn sub_int(#[out] res: *mut u64, #[frame] a: u64, #[frame] b: u64, bits: u64) {
        unsafe {
            let diff = a.wrapping_sub(b);
            *res = if bits >= 64 {
                diff
            } else {
                diff & ((1u64 << bits) - 1)
            };
        }
    }

    #[opcode]
    fn mul_int(#[out] res: *mut u64, #[frame] a: u64, #[frame] b: u64, bits: u64) {
        unsafe {
            let prod = a.wrapping_mul(b);
            *res = if bits >= 64 {
                prod
            } else {
                prod & ((1u64 << bits) - 1)
            };
        }
    }

    #[opcode]
    fn div_u64(#[out] res: *mut u64, #[frame] a: u64, #[frame] b: u64) {
        unsafe {
            *res = a / b;
        }
    }

    #[opcode]
    fn mod_u64(#[out] res: *mut u64, #[frame] a: u64, #[frame] b: u64) {
        unsafe {
            *res = a % b;
        }
    }

    #[opcode]
    fn and_u64(#[out] res: *mut u64, #[frame] a: u64, #[frame] b: u64) {
        unsafe {
            *res = a & b;
        }
    }

    #[opcode]
    fn or_u64(#[out] res: *mut u64, #[frame] a: u64, #[frame] b: u64) {
        unsafe {
            *res = a | b;
        }
    }

    #[opcode]
    fn xor_u64(#[out] res: *mut u64, #[frame] a: u64, #[frame] b: u64) {
        unsafe {
            *res = a ^ b;
        }
    }

    #[opcode]
    fn shl_u64(#[out] res: *mut u64, #[frame] a: u64, #[frame] b: u64, bits: u64) {
        unsafe {
            let shifted = a << b;
            *res = if bits >= 64 {
                shifted
            } else {
                shifted & ((1u64 << bits) - 1)
            };
        }
    }

    #[opcode]
    fn ushr_u64(#[out] res: *mut u64, #[frame] a: u64, #[frame] b: u64) {
        unsafe {
            *res = a >> b;
        }
    }

    #[opcode]
    fn not_u64(#[out] res: *mut u64, #[frame] a: u64) {
        unsafe {
            *res = !a;
        }
    }

    #[opcode]
    fn eq_u64(#[out] res: *mut u64, #[frame] a: u64, #[frame] b: u64) {
        unsafe {
            *res = (a == b) as u64;
        }
    }

    #[opcode]
    fn lt_u64(#[out] res: *mut u64, #[frame] a: u64, #[frame] b: u64) {
        unsafe {
            *res = (a < b) as u64;
        }
    }

    #[opcode]
    fn lt_s64(#[out] res: *mut u64, #[frame] a: u64, #[frame] b: u64, bits: u64) {
        unsafe {
            // Sign-extend from `bits` to 64 bits, then compare as signed
            let shift = 64 - bits;
            let sa = ((a << shift) as i64) >> shift;
            let sb = ((b << shift) as i64) >> shift;
            *res = (sa < sb) as u64;
        }
    }

    #[opcode]
    fn truncate_u64(#[out] res: *mut u64, #[frame] a: u64, to_bits: u64) {
        unsafe {
            let mask = if to_bits >= 64 {
                u64::MAX
            } else {
                (1u64 << to_bits) - 1
            };
            *res = a & mask;
        }
    }

    #[opcode]
    fn truncate_f_to_u(#[out] res: *mut Field, #[frame] a: Field, to_bits: u64) {
        unsafe {
            let limb0 = ark_ff::PrimeField::into_bigint(a).0[0];
            let mask = if to_bits >= 64 {
                u64::MAX
            } else {
                (1u64 << to_bits) - 1
            };
            *res = From::from(limb0 & mask);
        }
    }

    #[opcode]
    fn add_field(#[out] res: *mut Field, #[frame] a: Field, #[frame] b: Field) {
        unsafe {
            *res = a + b;
        }
    }

    #[opcode]
    fn sub_field(#[out] res: *mut Field, #[frame] a: Field, #[frame] b: Field) {
        unsafe {
            *res = a - b;
        }
    }

    #[opcode]
    #[inline(never)]
    fn div_field(#[out] res: *mut Field, #[frame] a: Field, #[frame] b: Field) {
        unsafe {
            *res = if b == Field::ZERO { Field::ZERO } else { a / b };
        }
    }

    #[opcode]
    fn mul_field(#[out] res: *mut Field, #[frame] a: Field, #[frame] b: Field) {
        unsafe {
            *res = a * b;
        }
    }

    #[opcode]
    fn cast_field_to_u64(#[out] res: *mut u64, #[frame] a: Field) {
        unsafe {
            *res = ark_ff::PrimeField::into_bigint(a).0[0];
        }
    }

    #[opcode]
    fn cast_u64_to_field(#[out] res: *mut Field, #[frame] a: u64) {
        unsafe {
            *res = From::from(a);
        }
    }

    #[opcode]
    fn array_alloc(
        #[out] res: *mut BoxedValue,
        stride: usize,
        meta: BoxedLayout,
        items: &[FramePosition],
        frame: Frame,
        vm: &mut VM,
    ) {
        let array = BoxedValue::alloc(meta, vm);
        // println!(
        //     "array_alloc: size={} stride={} has_ptr_elems={} @ {:?}",
        //     meta.size(),
        //     stride,
        //     meta.ptr_elems(),
        //     array.0
        // );
        for (i, item) in items.iter().enumerate() {
            let tgt = array.array_idx(i, stride);
            unsafe {
                frame.write_to(tgt, item.0 as isize, stride);
            }
        }
        // println!(
        //     "array_alloc: array={:?} stride={} size={} storage_size={}",
        //     array.0,
        //     stride,
        //     array.layout().array_size(),
        //     array.layout().underlying_array_size()
        // );
        unsafe {
            *res = array;
        }
    }

    #[opcode]
    #[inline(never)]
    fn tuple_alloc(
        #[out] res: *mut BoxedValue,
        meta: BoxedLayout,
        fields: &[FramePosition],
        frame: Frame,
        vm: &mut VM,
    ) {
        let tuple = BoxedValue::alloc(meta, vm);
        let view = meta.as_struct(&vm.struct_layouts);
        let mut field_offset = 0;
        for (i, field) in fields.iter().enumerate() {
            let size = view.field_size(i);
            let tgt = unsafe { tuple.data().add(field_offset) };
            unsafe {
                frame.write_to(tgt, field.0 as isize, size);
            }
            field_offset += size;
        }
        unsafe {
            *res = tuple;
        }
    }

    #[opcode]
    fn ref_alloc(#[out] res: *mut BoxedValue, meta: BoxedLayout, vm: &mut VM) {
        let cell = BoxedValue::alloc(meta, vm);
        unsafe {
            ptr::write_bytes(cell.data(), 0, meta.ref_cell_elem_size());
            *res = cell;
        }
    }

    #[opcode]
    #[inline(never)]
    fn ref_store(
        #[frame] cell: BoxedValue,
        source: FramePosition,
        stride: usize,
        elem_rc: usize,
        frame: Frame,
        vm: &mut VM,
    ) {
        if elem_rc != 0 {
            let old = unsafe { *(cell.data() as *mut BoxedValue) };
            if !old.0.is_null() {
                old.dec_rc(vm);
            }
        }
        unsafe {
            frame.write_to(cell.data(), source.0 as isize, stride);
        }
    }

    #[opcode]
    fn ref_load(#[out] res: *mut u64, #[frame] cell: BoxedValue, stride: usize) {
        unsafe {
            ptr::copy_nonoverlapping(cell.data(), res, stride);
        }
    }

    #[opcode]
    fn array_get(
        #[out] res: *mut u64,
        #[frame] array: BoxedValue,
        #[frame] index: u64,
        stride: usize,
        vm: &mut VM,
    ) {
        assert!(
            (index as usize) * stride < array.layout().array_size(),
            "array_get: index {} out of bounds for array of length {}",
            index,
            array.layout().array_size() / stride
        );
        let src = array.array_idx(index as usize, stride);
        unsafe {
            ptr::copy_nonoverlapping(src, res, stride);
        }
    }

    #[opcode]
    fn tuple_proj(
        #[out] res: *mut u64,
        #[frame] tuple: BoxedValue,
        field_offset: usize,
        field_size: usize,
        vm: &mut VM,
    ) {
        let src = unsafe { tuple.data().add(field_offset) };
        unsafe {
            ptr::copy_nonoverlapping(src, res, field_size);
        }
    }

    #[opcode]
    #[inline(never)]
    fn array_set(
        #[out] res: *mut BoxedValue,
        #[frame] array: BoxedValue,
        #[frame] index: u64,
        source: FramePosition,
        stride: usize,
        frame: Frame,
        vm: &mut VM,
    ) {
        assert!(
            (index as usize) * stride < array.layout().array_size(),
            "array_set: index {} out of bounds for array of length {}",
            index,
            array.layout().array_size() / stride
        );
        let new_array = array.copy_if_reused(vm);
        let target = new_array.array_idx(index as usize, stride);
        if new_array.layout().data_type() == DataType::BoxedArray {
            if new_array.0 == array.0 {
                // if we're reusing the array, the old element needs to be garbage collected
                let old_elem = unsafe { *(target as *mut BoxedValue) };
                old_elem.dec_rc(vm);
            } else {
                // if we're not reusing the array, we need to bump RC of all _other_ elements,
                // because they're now aliased in the new array.
                for i in 0..new_array.layout().array_size() {
                    if i != index as usize {
                        let elem = unsafe { *(new_array.array_idx(i, stride) as *mut BoxedValue) };
                        elem.inc_rc(1);
                    }
                }
            }
        }
        unsafe {
            frame.write_to(target, source.0 as isize, stride);
            *res = new_array;
        }
    }

    #[opcode]
    fn slice_len(#[out] res: *mut u64, #[frame] array: BoxedValue, stride: usize) {
        let len = array.layout().array_size() / stride;
        unsafe {
            *res = len as u64;
        }
    }

    #[opcode]
    fn inc_rc(#[frame] array: BoxedValue, amount: u64) {
        // println!("inc_array_rc_intro");
        array.inc_rc(amount);
        // println!("inc_array_rc_outro");
    }

    #[opcode]
    #[inline(never)]
    fn dec_rc(#[frame] array: BoxedValue, vm: &mut VM) {
        // println!("dec_array_rc_intro");
        array.dec_rc(vm);
        // println!("dec_array_rc_outro");
    }

    #[opcode]
    fn witness_ref_alloc(#[out] res: *mut BoxedValue, data: Field, vm: &mut VM) {
        let val = BoxedValue::alloc(BoxedLayout::ad_const(), vm);
        let d = val.as_ad_const();
        unsafe {
            (*d).value = data;
            *res = val;
        };
    }

    #[opcode]
    fn bump_da(#[frame] v: BoxedValue, #[frame] coeff: Field, vm: &mut VM) {
        v.bump_da(coeff, vm);
    }

    #[opcode]
    fn bump_db(#[frame] v: BoxedValue, #[frame] coeff: Field, vm: &mut VM) {
        v.bump_db(coeff, vm);
    }

    #[opcode]
    fn bump_dc(#[frame] v: BoxedValue, #[frame] coeff: Field, vm: &mut VM) {
        v.bump_dc(coeff, vm);
    }

    #[opcode]
    fn next_d_coeff(#[out] v: *mut Field, vm: &mut VM) {
        unsafe {
            *v = *vm.data.as_ad.ad_coeffs.add(vm.data.as_ad.current_cnst_off);
            vm.data.as_ad.current_cnst_off += 1;
        };
    }

    #[opcode]
    fn fresh_witness(#[out] res: *mut BoxedValue, vm: &mut VM) {
        let index = unsafe { vm.data.as_ad.current_wit_off as u64 };
        unsafe { vm.data.as_ad.current_wit_off += 1 };
        let val = BoxedValue::alloc(BoxedLayout::ad_witness(), vm);
        let d = val.as_ad_witness();
        unsafe {
            (*d).index = index;
            *res = val;
        }
    }

    #[opcode]
    fn pure_to_witness_ref(#[out] res: *mut BoxedValue, #[frame] v: Field, vm: &mut VM) {
        let val = BoxedValue::alloc(BoxedLayout::ad_const(), vm);
        let d = val.as_ad_const();
        unsafe {
            (*d).value = v;
            *res = val;
        }
    }

    #[opcode]
    fn unbox_field(#[out] res: *mut Field, #[frame] v: BoxedValue) {
        let d = v.as_ad_const();
        unsafe {
            *res = (*d).value;
        }
    }

    #[opcode]
    fn mul_const(
        #[out] res: *mut BoxedValue,
        #[frame] coeff: Field,
        #[frame] v: BoxedValue,
        vm: &mut VM,
    ) {
        let val = BoxedValue::alloc(BoxedLayout::mul_const(), vm);
        let d = val.as_mul_const();
        unsafe {
            (*d).coeff = coeff;
            (*d).value = v;
            (*d).da = Field::ZERO;
            (*d).db = Field::ZERO;
            (*d).dc = Field::ZERO;
            *res = val;
        }
    }

    #[opcode]
    fn add_boxed(
        #[out] res: *mut BoxedValue,
        #[frame] a: BoxedValue,
        #[frame] b: BoxedValue,
        vm: &mut VM,
    ) {
        let val = BoxedValue::alloc(BoxedLayout::ad_sum(), vm);
        let d = val.as_ad_sum();
        unsafe {
            (*d).a = a;
            (*d).b = b;
            (*d).da = Field::ZERO;
            (*d).db = Field::ZERO;
            (*d).dc = Field::ZERO;
            *res = val;
        }
    }

    #[opcode]
    fn assert_eq_u64(#[frame] a: u64, #[frame] b: u64) {
        assert_eq!(a, b);
    }

    #[opcode]
    fn assert_eq_field(#[frame] a: Field, #[frame] b: Field) {
        assert_eq!(a, b);
    }

    #[opcode]
    fn assert_r1c(#[frame] a: Field, #[frame] b: Field, #[frame] c: Field) {
        assert_eq!(a * b, c);
    }

    #[opcode]
    #[inline(never)] // TODO better impl
    fn rangecheck(#[frame] val: Field, max_bits: usize) {
        // Convert field to bigint and check if it fits in max_bits
        let bigint = ark_ff::PrimeField::into_bigint(val);
        let check = bigint.to_bits_le().iter().skip(max_bits).all(|b| !b);
        assert!(check);
    }

    #[opcode]
    fn to_bytes_be(#[frame] val: Field, count: u64, #[out] res: *mut BoxedValue, vm: &mut VM) {
        let val = ark_ff::PrimeField::into_bigint(val);
        let r = BoxedValue::alloc(BoxedLayout::array(count as usize, false), vm);
        unsafe {
            for i in 0..count {
                // Each limb in val.0 is a u64 (8 bytes), little-endian limb order
                let byte_idx = i as usize; // byte index from LSB
                let limb_idx = byte_idx / 8;
                let byte_in_limb = byte_idx % 8;
                let byte_val = if limb_idx < val.0.len() {
                    (val.0[limb_idx] >> (byte_in_limb * 8)) & 0xFF
                } else {
                    0
                };
                *r.array_idx((count - i - 1) as usize, 1) = byte_val;
            }
            *res = r;
        }
    }

    #[opcode]
    fn to_bits_le(#[out] res: *mut BoxedValue, #[frame] val: Field, count: u64, vm: &mut VM) {
        panic!("to_bits_be_lt_8 not implemented");
    }

    #[opcode]
    fn spread_u32(#[out] res: *mut u64, #[frame] val: u64) {
        let result = spread_bits(val as u32);
        unsafe {
            *res = result;
        }
    }

    #[opcode]
    fn unspread_u64(#[out] res_and: *mut u64, #[out] res_xor: *mut u64, #[frame] val: u64) {
        let (and_val, xor_val) = unspread_bits(val);
        unsafe {
            *res_and = and_val as u64;
            *res_xor = xor_val as u64;
        }
    }

    #[opcode]
    fn spread_lookup_field(
        #[frame] val: Field,
        #[frame] result: Field,
        #[frame] flag: Field,
        bits: usize,
        vm: &mut VM,
    ) {
        // Initialize spread table for this bit-width on first call
        if vm.spread_tables[bits].is_none() {
            let length = 1usize << bits;
            let table_info = TableInfo {
                multiplicities_wit: unsafe { vm.data.as_forward.multiplicities_witness },
                num_indices: 1,
                num_values: 1,
                length,
                elem_inverses_constraint_section_offset: unsafe {
                    vm.data.as_forward.elem_inverses_constraint_section_offset
                },
                elem_inverses_witness_section_offset: unsafe {
                    vm.data.as_forward.elem_inverses_witness_section_offset
                },
                dims: vec![length],
            };
            vm.spread_tables[bits] = Some(vm.tables.len());
            vm.tables.push(table_info);

            // β-power LogUp: 1 R1C per slot. Stash spread(i) (value column at β⁰)
            // in the A-base slot; Phase 2 reads it to compute denom_i = α − v_i − β·i.
            unsafe {
                let cnst_off = vm.data.as_forward.elem_inverses_constraint_section_offset;
                for i in 0..length {
                    *vm.data.as_forward.out_a_base.add(cnst_off + i) =
                        Field::from(spread_bits(i as u32));
                }

                vm.data.as_forward.multiplicities_witness =
                    vm.data.as_forward.multiplicities_witness.add(length);
                vm.data.as_forward.elem_inverses_constraint_section_offset += length + 1;
                vm.data.as_forward.elem_inverses_witness_section_offset += length;
            }
        }

        let table_idx = vm.spread_tables[bits].unwrap();
        let flag_u64 = ark_ff::PrimeField::into_bigint(flag).0[0];
        unsafe { forward_kv_lookup_emit(table_idx, val, result, flag_u64, vm) };
    }

    #[opcode]
    fn dspread_lookup_field(
        #[frame] val: BoxedValue,
        #[frame] result: BoxedValue,
        #[frame] flag: BoxedValue,
        bits: usize,
        vm: &mut VM,
    ) {
        if vm.spread_tables[bits].is_none() {
            let length = 1usize << bits;
            let inverses_constraint_section_offset =
                unsafe { vm.data.as_ad.current_cnst_tables_off };
            let inverses_witness_section_offset = unsafe { vm.data.as_ad.current_wit_tables_off };
            let multiplicities_wit_offset = unsafe { vm.data.as_ad.current_wit_multiplicities_off };
            let table_info = TableInfo {
                multiplicities_wit: ptr::null_mut(),
                num_indices: 1,
                num_values: 1,
                length,
                elem_inverses_witness_section_offset: inverses_witness_section_offset,
                elem_inverses_constraint_section_offset: inverses_constraint_section_offset,
                dims: vec![length],
            };
            vm.spread_tables[bits] = Some(vm.tables.len());
            vm.tables.push(table_info);
            unsafe {
                vm.data.as_ad.current_wit_multiplicities_off += length;
                vm.data.as_ad.current_wit_tables_off += length;
                vm.data.as_ad.current_cnst_tables_off += length + 1;
            }

            // Sum constraint at offset `length`.
            let inv_sum_coeff = unsafe {
                *vm.data
                    .as_ad
                    .ad_coeffs
                    .offset(inverses_constraint_section_offset as isize + length as isize)
            };

            for i in 0..length {
                // β-power LogUp per slot:
                //   y_i · (α − spread(i) − β·i) = m_i
                let y_coeff = unsafe {
                    *vm.data
                        .as_ad
                        .ad_coeffs
                        .offset(inverses_constraint_section_offset as isize + i as isize)
                };
                unsafe {
                    // A=(y_i,1)
                    *vm.data
                        .as_ad
                        .out_da
                        .offset(inverses_witness_section_offset as isize + i as isize) +=
                        y_coeff;
                    // B=(α,1)
                    *vm.data
                        .as_ad
                        .out_db
                        .add(vm.data.as_ad.logup_wit_challenge_off) += y_coeff;
                    // B=(slot 0, -spread(i))
                    *vm.data.as_ad.out_db -= y_coeff * Field::from(spread_bits(i as u32));
                    // B=(β, -i)
                    *vm.data
                        .as_ad
                        .out_db
                        .add(vm.data.as_ad.logup_wit_challenge_off + 1) -=
                        y_coeff * Field::from(i as u64);
                    // C=(m_i, 1)
                    *vm.data
                        .as_ad
                        .out_dc
                        .offset(multiplicities_wit_offset as isize + i as isize) += y_coeff;
                }

                // Sum constraint A: (y_wit_i, 1)
                unsafe {
                    *vm.data
                        .as_ad
                        .out_da
                        .offset(inverses_witness_section_offset as isize + i as isize) +=
                        inv_sum_coeff;
                }
            }

            // Sum constraint B = (slot 0, 1).
            unsafe {
                *vm.data.as_ad.out_db += inv_sum_coeff;
            }
        }

        let table_idx = vm.spread_tables[bits].unwrap();
        unsafe { ad_kv_lookup_emit(table_idx, val, result, flag, vm) };
    }

    #[opcode]
    fn rngchk_8_field(#[frame] val: Field, #[frame] flag: Field, vm: &mut VM) {
        if vm.rgchk_8.is_none() {
            let table_info = TableInfo {
                multiplicities_wit: unsafe { vm.data.as_forward.multiplicities_witness },
                num_indices: 1,
                num_values: 0,
                length: 256,
                elem_inverses_constraint_section_offset: unsafe {
                    vm.data.as_forward.elem_inverses_constraint_section_offset
                },
                elem_inverses_witness_section_offset: unsafe {
                    vm.data.as_forward.elem_inverses_witness_section_offset
                },
                dims: vec![256],
            };
            vm.rgchk_8 = Some(vm.tables.len());
            vm.tables.push(table_info);
            unsafe {
                vm.data.as_forward.multiplicities_witness =
                    vm.data.as_forward.multiplicities_witness.offset(256);
                vm.data.as_forward.elem_inverses_constraint_section_offset += 257;
                vm.data.as_forward.elem_inverses_witness_section_offset += 256;
            }
        }
        let flag_u64 = ark_ff::PrimeField::into_bigint(flag).0[0];
        let table_idx = *vm.rgchk_8.as_ref().unwrap();
        let table_info = &vm.tables[table_idx];
        unsafe {
            if flag_u64 != 0 {
                let val_u64 = ark_ff::PrimeField::into_bigint(val).0[0];
                let ptr = table_info.multiplicities_wit.offset(val_u64 as isize);
                *(ptr as *mut u64) += flag_u64;
                *(vm.data.as_forward.lookups_a as *mut u64) = table_idx as u64;
                vm.data.as_forward.lookups_a = vm.data.as_forward.lookups_a.offset(1);
                *(vm.data.as_forward.lookups_b as *mut u64) = val_u64;
                vm.data.as_forward.lookups_b = vm.data.as_forward.lookups_b.offset(1);
            } else {
                *(vm.data.as_forward.lookups_a as *mut u64) = table_idx as u64;
                vm.data.as_forward.lookups_a = vm.data.as_forward.lookups_a.offset(1);
                *(vm.data.as_forward.lookups_b as *mut Field) = val;
                vm.data.as_forward.lookups_b = vm.data.as_forward.lookups_b.offset(1);
            }
            *(vm.data.as_forward.lookups_c as *mut u64) = flag_u64;
            vm.data.as_forward.lookups_c = vm.data.as_forward.lookups_c.offset(1);
        }
    }

    #[opcode]
    fn array_lookup_field(
        #[frame] array: BoxedValue,
        #[frame] index: Field,
        #[frame] result: Field,
        #[frame] flag: Field,
        stride: usize,
        elem_kind: usize,
        vm: &mut VM,
    ) {
        let table_id_ptr = array.table_id();
        let table_idx = unsafe { *table_id_ptr };

        let table_idx = if table_idx == u64::MAX {
            // First lookup on this array: create a new table
            let length = array.layout().array_size() / stride;
            let table_info = TableInfo {
                multiplicities_wit: unsafe { vm.data.as_forward.multiplicities_witness },
                num_indices: 1,
                num_values: 1,
                length,
                elem_inverses_constraint_section_offset: unsafe {
                    vm.data.as_forward.elem_inverses_constraint_section_offset
                },
                elem_inverses_witness_section_offset: unsafe {
                    vm.data.as_forward.elem_inverses_witness_section_offset
                },
                dims: vec![length],
            };
            let new_table_idx = vm.tables.len();
            vm.tables.push(table_info);

            // β-power LogUp: 1 R1C per slot. Stash v_i in the A-base slot for
            // Phase 2 to consume (out_a is a cursor; out_a_base is fixed).
            unsafe {
                let cnst_off = vm.data.as_forward.elem_inverses_constraint_section_offset;
                for i in 0..length {
                    let elem_ptr = array.array_idx(i, stride);
                    let elem_field = read_pure_elem_as_field(elem_ptr, elem_kind);
                    *vm.data.as_forward.out_a_base.add(cnst_off + i) = elem_field;
                }

                vm.data.as_forward.multiplicities_witness =
                    vm.data.as_forward.multiplicities_witness.add(length);
                // 1 constraint per element + 1 sum constraint
                vm.data.as_forward.elem_inverses_constraint_section_offset += length + 1;
                // 1 witness slot per element (y_i)
                vm.data.as_forward.elem_inverses_witness_section_offset += length;
            }

            // Store table index on the array
            unsafe { *table_id_ptr = new_table_idx as u64 };

            new_table_idx
        } else {
            table_idx as usize
        };

        let flag_u64 = ark_ff::PrimeField::into_bigint(flag).0[0];
        unsafe { forward_kv_lookup_emit(table_idx, index, result, flag_u64, vm) };
    }

    /// Allocate a fresh 1-key `WitnessArrayDescriptor` retaining `root` and
    /// holding the single key `key`. Bumps `root`'s RC.
    #[opcode]
    fn make_witness_descriptor(
        #[frame] root: BoxedValue,
        #[frame] key: Field,
        #[out] result: *mut BoxedValue,
        vm: &mut VM,
    ) {
        let layout = BoxedLayout::witness_descriptor(1);
        let desc = BoxedValue::alloc(layout, vm);
        unsafe {
            // root_ptr field (refcounted) — retain.
            root.inc_rc(1);
            *desc.desc_root_ptr() = root;
            // First key.
            *desc.desc_key_ptr(0) = key;
            *result = desc;
        }
    }

    /// Allocate a fresh `(n_keys+1)`-key descriptor from a prior descriptor
    /// plus the new key. Retains the same `root` (bumps RC).
    #[opcode]
    fn extend_witness_descriptor(
        #[frame] desc_in: BoxedValue,
        #[frame] new_key: Field,
        #[out] result: *mut BoxedValue,
        vm: &mut VM,
    ) {
        debug_assert!(desc_in.layout().data_type() == DataType::WitnessArrayDescriptor);
        let prior_n = desc_in.layout().descriptor_n_keys();
        let new_n = prior_n + 1;
        let new_desc = BoxedValue::alloc(BoxedLayout::witness_descriptor(new_n), vm);
        unsafe {
            // Copy root (bump RC since both descriptors retain it).
            let root = *desc_in.desc_root_ptr();
            root.inc_rc(1);
            *new_desc.desc_root_ptr() = root;
            // Copy prior keys.
            for i in 0..prior_n {
                *new_desc.desc_key_ptr(i) = *desc_in.desc_key_ptr(i);
            }
            // Append new key.
            *new_desc.desc_key_ptr(prior_n) = new_key;
            *result = new_desc;
        }
    }

    /// Saturating N-D lookup. The descriptor carries the root and `prior_n`
    /// keys; the new key is `new_key`. Walks the root nesting along all
    /// `prior_n + 1` keys to read the leaf, emits N+1 tape entries (one per
    /// key R1C plus the y-constraint), and bumps the flat-indexed multiplicity.
    ///
    /// On the first call against a given root, registers a new flat-length
    /// table whose `dims` records each dimension size for Phase 2's denominator
    /// reconstruction.
    #[opcode]
    fn witness_array_lookup_field(
        #[frame] desc: BoxedValue,
        #[frame] new_key: Field,
        #[out] result: *mut Field,
        #[frame] flag: Field,
        leaf_stride: usize,
        elem_kind: usize,
        vm: &mut VM,
    ) {
        debug_assert!(desc.layout().data_type() == DataType::WitnessArrayDescriptor);
        let prior_n = desc.layout().descriptor_n_keys();
        let total_n = prior_n + 1;

        // Read root + keys from descriptor.
        let root = unsafe { *desc.desc_root_ptr() };
        let mut keys: Vec<Field> = Vec::with_capacity(total_n);
        let mut key_u64s: Vec<u64> = Vec::with_capacity(total_n);
        for i in 0..prior_n {
            let k = unsafe { *desc.desc_key_ptr(i) };
            keys.push(k);
            key_u64s.push(ark_ff::PrimeField::into_bigint(k).0[0]);
        }
        keys.push(new_key);
        key_u64s.push(ark_ff::PrimeField::into_bigint(new_key).0[0]);

        // First lookup against this root: register table.
        let table_id_ptr = root.table_id();
        let table_idx = unsafe { *table_id_ptr };
        let table_idx = if table_idx == u64::MAX {
            // Walk the root once with the current keys to discover dims; then
            // walk every flat slot to stash leaf values.
            let mut dims: Vec<usize> = Vec::with_capacity(total_n);
            let _probe = unsafe {
                nd_array_walk(root, &key_u64s, leaf_stride, elem_kind, &mut dims)
            };
            let length: usize = dims.iter().product();
            let table_info = TableInfo {
                multiplicities_wit: unsafe { vm.data.as_forward.multiplicities_witness },
                num_indices: total_n,
                num_values: 1,
                length,
                elem_inverses_constraint_section_offset: unsafe {
                    vm.data.as_forward.elem_inverses_constraint_section_offset
                },
                elem_inverses_witness_section_offset: unsafe {
                    vm.data.as_forward.elem_inverses_witness_section_offset
                },
                dims: dims.clone(),
            };
            let new_table_idx = vm.tables.len();
            vm.tables.push(table_info);

            // Stash each leaf at out_a_base[cnst_off + flat_idx]. Walk all
            // flat slots by decomposing s into per-dim coords.
            unsafe {
                let cnst_off = vm.data.as_forward.elem_inverses_constraint_section_offset;
                let mut suffix = vec![1usize; total_n + 1];
                for j in (0..total_n).rev() {
                    suffix[j] = suffix[j + 1] * dims[j];
                }
                for s in 0..length {
                    let coords: Vec<u64> = (0..total_n)
                        .map(|j| ((s / suffix[j + 1]) % dims[j]) as u64)
                        .collect();
                    let mut tmp_dims = Vec::new();
                    let leaf =
                        nd_array_walk(root, &coords, leaf_stride, elem_kind, &mut tmp_dims);
                    *vm.data.as_forward.out_a_base.add(cnst_off + s) = leaf;
                }

                vm.data.as_forward.multiplicities_witness =
                    vm.data.as_forward.multiplicities_witness.add(length);
                vm.data.as_forward.elem_inverses_constraint_section_offset += length + 1;
                vm.data.as_forward.elem_inverses_witness_section_offset += length;
            }

            unsafe { *table_id_ptr = new_table_idx as u64 };
            new_table_idx
        } else {
            table_idx as usize
        };

        let flag_u64 = ark_ff::PrimeField::into_bigint(flag).0[0];

        // Compute flat index from the keys, using table.dims.
        let dims = vm.tables[table_idx].dims.clone();
        let mut suffix = vec![1usize; total_n + 1];
        for j in (0..total_n).rev() {
            suffix[j] = suffix[j + 1] * dims[j];
        }
        let flat_idx: u64 = (0..total_n)
            .map(|j| key_u64s[j] * suffix[j + 1] as u64)
            .sum();

        // Compute the leaf value by walking root with the integer keys.
        let mut tmp_dims = Vec::new();
        let leaf_value = unsafe {
            nd_array_walk(root, &key_u64s, leaf_stride, elem_kind, &mut tmp_dims)
        };
        unsafe { *result = leaf_value };

        // Pin the value into the algebraic witness vector. R1CGen allocated a
        // fresh witness slot for the lookup result; the Lookup constraint will
        // reference that slot. The prover (us) must populate it.
        unsafe {
            *vm.data.as_forward.algebraic_witness = leaf_value;
            vm.data.as_forward.algebraic_witness =
                vm.data.as_forward.algebraic_witness.offset(1);
        }

        unsafe {
            forward_nd_lookup_emit(table_idx, &keys, leaf_value, flag_u64, flat_idx, vm)
        };
    }

    #[opcode]
    fn drngchk_8_field(#[frame] val: BoxedValue, #[frame] flag: BoxedValue, vm: &mut VM) {
        if vm.rgchk_8.is_none() {
            let inverses_constraint_section_offset =
                unsafe { vm.data.as_ad.current_cnst_tables_off };
            let inverses_witness_section_offset = unsafe { vm.data.as_ad.current_wit_tables_off };
            let multiplicities_wit_offset = unsafe { vm.data.as_ad.current_wit_multiplicities_off };
            let table_info = TableInfo {
                multiplicities_wit: ptr::null_mut(),
                num_indices: 1,
                num_values: 0,
                length: 256,
                elem_inverses_witness_section_offset: inverses_witness_section_offset,
                elem_inverses_constraint_section_offset: inverses_constraint_section_offset,
                dims: vec![256],
            };
            vm.rgchk_8 = Some(vm.tables.len());
            vm.tables.push(table_info);
            unsafe {
                vm.data.as_ad.current_wit_multiplicities_off += 256;
                vm.data.as_ad.current_wit_tables_off += 256;
                vm.data.as_ad.current_cnst_tables_off += 257;
            }
            let inv_sum_coeff = unsafe {
                *vm.data
                    .as_ad
                    .ad_coeffs
                    .offset(inverses_constraint_section_offset as isize + 256)
            };

            for i in 0..256 {
                // For each element in the table, we have constraint `elem_inv_witness * (alpha - i) - multiplicity_witness = 0`
                let coeff = unsafe {
                    *vm.data
                        .as_ad
                        .ad_coeffs
                        .offset(inverses_constraint_section_offset as isize + i)
                };
                unsafe {
                    *vm.data
                        .as_ad
                        .out_da
                        .offset(inverses_witness_section_offset as isize + i) += coeff;
                    // if i == 0 {
                    //     println!("bump da at {} from inv by {coeff}", inverses_witness_section_offset as isize + i);
                    // }

                    *vm.data
                        .as_ad
                        .out_db
                        .add(vm.data.as_ad.logup_wit_challenge_off) += coeff;
                    *vm.data.as_ad.out_db -= coeff * Field::from(i as u64);

                    *vm.data
                        .as_ad
                        .out_dc
                        .offset(multiplicities_wit_offset as isize + i) += coeff;
                }

                // Also each inv goes into the A position of the total sum
                unsafe {
                    *vm.data
                        .as_ad
                        .out_da
                        .offset(inverses_witness_section_offset as isize + i) += inv_sum_coeff;
                }
            }

            // The coeff at B on the sum constraint is just `1` so we bump it.
            unsafe {
                *vm.data.as_ad.out_db += inv_sum_coeff;
            }
        }
        let table_idx = *vm.rgchk_8.as_ref().unwrap();
        let table_info = &vm.tables[table_idx];

        let inv_coeff = unsafe {
            let r = *vm
                .data
                .as_ad
                .ad_coeffs
                .add(vm.data.as_ad.current_cnst_lookups_off);
            vm.data.as_ad.current_cnst_lookups_off += 1;
            r
        };

        let inv_sum_coeff = unsafe {
            *vm.data
                .as_ad
                .ad_coeffs
                .add(table_info.elem_inverses_constraint_section_offset + 256)
        };

        let current_inv_wit_offset = unsafe {
            let r = vm.data.as_ad.current_wit_lookups_off;
            vm.data.as_ad.current_wit_lookups_off += 1;
            r
        };

        unsafe {
            // bump for the RHS of the sum
            *vm.data.as_ad.out_dc.add(current_inv_wit_offset) += inv_sum_coeff;

            // bumps for the inversion assert: y*(α-key) = flag
            // da[y] += inv_coeff
            *vm.data.as_ad.out_da.add(current_inv_wit_offset) += inv_coeff;

            // db[α] += inv_coeff
            *vm.data
                .as_ad
                .out_db
                .add(vm.data.as_ad.logup_wit_challenge_off) += inv_coeff;
            // db[key] -= inv_coeff
            val.bump_db(-inv_coeff, vm);

            // dc[flag] += inv_coeff  (RHS is flag, not constant 1)
            flag.bump_dc(inv_coeff, vm);
        }
    }

    #[opcode]
    fn darray_lookup_field(
        #[frame] array: BoxedValue,
        #[frame] index: BoxedValue,
        #[frame] result: BoxedValue,
        #[frame] flag: BoxedValue,
        stride: usize,
        elem_kind: usize,
        vm: &mut VM,
    ) {
        let table_id_ptr = array.table_id();
        let table_idx = unsafe { *table_id_ptr };

        let table_idx = if table_idx == u64::MAX {
            // First AD call on this array: register table and process table-side R1Cs.
            //
            // β-power LogUp layout per slot (1-D):
            //   y_i · (α − v_i − β·i) = m_i           A=(y_i,1)
            //                                         B=(α,1)+(−v_i)+(−β,i)
            //                                         C=(m_i,1)
            // Sum constraint (at offset `length`):
            //   (Σ y_i) · 1 = Σ y_query              (lookups append to C)
            let length = array.layout().array_size() / stride;
            let inverses_constraint_section_offset =
                unsafe { vm.data.as_ad.current_cnst_tables_off };
            let inverses_witness_section_offset = unsafe { vm.data.as_ad.current_wit_tables_off };
            let multiplicities_wit_offset = unsafe { vm.data.as_ad.current_wit_multiplicities_off };
            let table_info = TableInfo {
                multiplicities_wit: ptr::null_mut(),
                num_indices: 1,
                num_values: 1,
                length,
                elem_inverses_witness_section_offset: inverses_witness_section_offset,
                elem_inverses_constraint_section_offset: inverses_constraint_section_offset,
                dims: vec![length],
            };
            let new_table_idx = vm.tables.len();
            vm.tables.push(table_info);
            unsafe {
                vm.data.as_ad.current_wit_multiplicities_off += length;
                vm.data.as_ad.current_wit_tables_off += length;
                vm.data.as_ad.current_cnst_tables_off += length + 1;
            }

            let sum_coeff = unsafe {
                *vm.data
                    .as_ad
                    .ad_coeffs
                    .offset(inverses_constraint_section_offset as isize + length as isize)
            };

            for i in 0..length {
                let elem_ptr = array.array_idx(i, stride);

                let y_coeff = unsafe {
                    *vm.data
                        .as_ad
                        .ad_coeffs
                        .offset(inverses_constraint_section_offset as isize + i as isize)
                };
                unsafe {
                    // A=(y_i, 1): dA[y_wit_i] += y_coeff
                    *vm.data
                        .as_ad
                        .out_da
                        .offset(inverses_witness_section_offset as isize + i as isize) +=
                        y_coeff;
                    // B=(α,1): dB[α] += y_coeff
                    *vm.data
                        .as_ad
                        .out_db
                        .add(vm.data.as_ad.logup_wit_challenge_off) += y_coeff;
                    // B=(−v_i): dB[v_i contributors] -= y_coeff
                    lookup_elem_bump_db(elem_ptr, elem_kind, -y_coeff, vm);
                    // B=(−β, i): dB[β] -= y_coeff · i
                    *vm.data
                        .as_ad
                        .out_db
                        .add(vm.data.as_ad.logup_wit_challenge_off + 1) -=
                        y_coeff * Field::from(i as u64);
                    // C=(m_i, 1): dC[mult_wit_i] += y_coeff
                    *vm.data.as_ad.out_dc.add(multiplicities_wit_offset + i) += y_coeff;
                }

                // Sum constraint A: contains (y_wit_i, 1) for each table slot.
                unsafe {
                    *vm.data
                        .as_ad
                        .out_da
                        .add(inverses_witness_section_offset + i) += sum_coeff;
                }
            }

            // Sum constraint B = (slot 0, 1): bump out_db at slot 0.
            unsafe {
                *vm.data.as_ad.out_db += sum_coeff;
            }

            unsafe { *table_id_ptr = new_table_idx as u64 };
            new_table_idx
        } else {
            table_idx as usize
        };

        unsafe { ad_kv_lookup_emit(table_idx, index, result, flag, vm) };
    }

    /// AD-path N-D array lookup. `keys_arr` is a `BoxedArray` of `n_keys`
    /// `BoxedValue` witness refs (one per dim). On first use against a given
    /// root, registers an N-D table and emits per-slot R1C bumps. Per-call,
    /// emits the N+1 R1C bumps for the query via `ad_nd_kv_lookup_emit`.
    ///
    /// Also emits AD bumps for the β-power chain R1Cs (`β · β^(j-1) = β^j`)
    /// on first ever invocation. These are global R1Cs in the tables_data
    /// constraint section, emitted once per program.
    #[opcode]
    fn dnd_array_lookup_field(
        #[frame] array: BoxedValue,
        #[frame] keys_arr: BoxedValue,
        #[frame] result: BoxedValue,
        #[frame] flag: BoxedValue,
        n_keys: usize,
        leaf_stride: usize,
        elem_kind: usize,
        vm: &mut VM,
    ) {
        // Emit β-power chain R1C bumps once, before any tables are registered
        // (so the chain's constraint offsets are unconditionally tables_data_start..
        // +chain_len).
        if !vm.beta_chain_ad_emitted {
            let chain_base = unsafe { vm.data.as_ad.beta_chain_cnst_base };
            for j in 2..=n_keys {
                let coeff = unsafe { *vm.data.as_ad.ad_coeffs.add(chain_base + j - 2) };
                // β at challenges_start+1; β^(j-1) at challenges_start+1+(j-2)
                //  (j=2 → β at +1, β^(j-1)=β at +1); β^j at +1+(j-1).
                let beta_off = unsafe { vm.data.as_ad.logup_wit_challenge_off } + 1;
                let prev_pow_off =
                    unsafe { vm.data.as_ad.logup_wit_challenge_off } + 1 + (j - 2);
                let new_pow_off =
                    unsafe { vm.data.as_ad.logup_wit_challenge_off } + 1 + (j - 1);
                unsafe {
                    *vm.data.as_ad.out_da.add(beta_off) += coeff;
                    *vm.data.as_ad.out_db.add(prev_pow_off) += coeff;
                    *vm.data.as_ad.out_dc.add(new_pow_off) += coeff;
                }
            }
            vm.beta_chain_ad_emitted = true;
        }

        let table_id_ptr = array.table_id();
        let table_idx = unsafe { *table_id_ptr };

        let table_idx = if table_idx == u64::MAX {
            // First AD call on this array: probe dims by walking root with
            // (0, …, 0) keys, register table, emit per-slot R1C bumps.
            let mut dims: Vec<usize> = Vec::with_capacity(n_keys);
            let probe_keys = vec![0u64; n_keys];
            let _probe = unsafe {
                nd_array_walk(array, &probe_keys, leaf_stride, elem_kind, &mut dims)
            };
            let length: usize = dims.iter().product();

            let inverses_constraint_section_offset =
                unsafe { vm.data.as_ad.current_cnst_tables_off };
            let inverses_witness_section_offset = unsafe { vm.data.as_ad.current_wit_tables_off };
            let multiplicities_wit_offset = unsafe { vm.data.as_ad.current_wit_multiplicities_off };
            let table_info = TableInfo {
                multiplicities_wit: ptr::null_mut(),
                num_indices: n_keys,
                num_values: 1,
                length,
                elem_inverses_witness_section_offset: inverses_witness_section_offset,
                elem_inverses_constraint_section_offset: inverses_constraint_section_offset,
                dims: dims.clone(),
            };
            let new_table_idx = vm.tables.len();
            vm.tables.push(table_info);
            unsafe {
                vm.data.as_ad.current_wit_multiplicities_off += length;
                vm.data.as_ad.current_wit_tables_off += length;
                vm.data.as_ad.current_cnst_tables_off += length + 1;
            }

            let sum_coeff = unsafe {
                *vm.data
                    .as_ad
                    .ad_coeffs
                    .offset(inverses_constraint_section_offset as isize + length as isize)
            };

            // Pre-compute suffix products for slot-coord decomposition.
            let mut suffix = vec![1usize; n_keys + 1];
            for j in (0..n_keys).rev() {
                suffix[j] = suffix[j + 1] * dims[j];
            }

            for s in 0..length {
                // Walk the root to fetch the leaf at (i_1(s), …, i_N(s)).
                let coords: Vec<u64> = (0..n_keys)
                    .map(|j| ((s / suffix[j + 1]) % dims[j]) as u64)
                    .collect();
                let mut tmp_dims = Vec::new();
                let leaf =
                    unsafe { nd_array_walk(array, &coords, leaf_stride, elem_kind, &mut tmp_dims) };

                let y_coeff = unsafe {
                    *vm.data
                        .as_ad
                        .ad_coeffs
                        .offset(inverses_constraint_section_offset as isize + s as isize)
                };
                unsafe {
                    // A=(y_s,1): dA[y_wit_s] += y_coeff
                    *vm.data
                        .as_ad
                        .out_da
                        .offset(inverses_witness_section_offset as isize + s as isize) +=
                        y_coeff;
                    // B=(α,1): dB[α] += y_coeff
                    *vm.data
                        .as_ad
                        .out_db
                        .add(vm.data.as_ad.logup_wit_challenge_off) += y_coeff;
                    // B=(slot 0, -leaf_field_value): dB at constant slot 0
                    *vm.data.as_ad.out_db -= y_coeff * leaf;
                    // B=(β^j, -i_j(s)) for j=1..=n_keys
                    for j in 0..n_keys {
                        let beta_pow_off = if j == 0 {
                            vm.data.as_ad.logup_wit_challenge_off + 1
                        } else {
                            vm.data.as_ad.logup_wit_challenge_off + 1 + j
                        };
                        *vm.data.as_ad.out_db.add(beta_pow_off) -=
                            y_coeff * Field::from(coords[j]);
                    }
                    // C=(m_s,1): dC[mult_wit_s] += y_coeff
                    *vm.data.as_ad.out_dc.add(multiplicities_wit_offset + s) += y_coeff;
                }

                // Sum constraint A: (y_wit_s, 1)
                unsafe {
                    *vm.data
                        .as_ad
                        .out_da
                        .add(inverses_witness_section_offset + s) += sum_coeff;
                }
            }

            // Sum constraint B = (slot 0, 1).
            unsafe {
                *vm.data.as_ad.out_db += sum_coeff;
            }

            unsafe { *table_id_ptr = new_table_idx as u64 };
            new_table_idx
        } else {
            table_idx as usize
        };

        // Read keys from keys_arr (a scratch PrimArray of `BoxedValue` words).
        let keys: Vec<BoxedValue> = (0..n_keys)
            .map(|i| unsafe { *(keys_arr.array_idx(i, 1) as *mut BoxedValue) })
            .collect();
        unsafe { ad_nd_kv_lookup_emit(table_idx, &keys, result, flag, vm) };
        // keys_arr was allocated by codegen as a scratch BoxedArray (rc=1)
        // for this call; nothing else references it, so drop now. Inner
        // BoxedValue keys retain their own RCs via their original SSA chain.
        keys_arr.dec_rc(vm);
    }

    #[opcode]
    fn init_global(
        vm: &mut VM,
        frame: Frame,
        src: FramePosition,
        global_offset: usize,
        size: usize,
    ) {
        unsafe {
            std::ptr::copy_nonoverlapping(
                frame.data.add(src.0),
                vm.globals.add(global_offset),
                size,
            );
        }
    }

    #[opcode]
    fn read_global(#[out] res: *mut u64, vm: &mut VM, global_offset: usize, size: usize) {
        unsafe {
            std::ptr::copy_nonoverlapping(vm.globals.add(global_offset), res, size);
        }
    }

    #[opcode]
    #[inline(never)]
    fn drop_global(vm: &mut VM, global_offset: usize) {
        unsafe {
            let boxed = *(vm.globals.add(global_offset) as *mut BoxedValue);
            boxed.dec_rc(vm);
        }
    }
}

pub struct Function {
    pub name: String,
    pub frame_size: usize,
    pub code: Vec<OpCode>,
}

impl Display for Function {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "fn {} (frame_size = {}):", self.name, self.frame_size)?;
        for op in &self.code {
            writeln!(f, "  {}", op)?;
        }
        Ok(())
    }
}

pub struct Program {
    pub functions: Vec<Function>,
    pub global_frame_size: usize,
    pub struct_layouts: Vec<StructDescriptor>,
}

impl Display for Program {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let max_line_number: usize = self.functions.iter().map(|f| f.code.len()).sum::<usize>() - 1;
        let max_line_number_digits = max_line_number.to_string().len();
        let max_function_idx = self.functions.len().to_string().len() - 1;
        let max_function_idx_digits = max_function_idx.to_string().len();
        let mut line = 0;
        for (i, function) in self.functions.iter().enumerate() {
            writeln!(
                f,
                "{: >max_function_idx_digits$}: fn {} (frame_size = {})",
                i, function.name, function.frame_size
            )?;
            for op in &function.code {
                writeln!(f, "  {: >max_line_number_digits$}: {}", line, op)?;
                line += 1;
            }
        }
        Ok(())
    }
}

/// Encode a single field of a `StructDescriptor` as one `u64`:
/// high bit = refcounted flag, low 32 bits = field size in u64 words.
#[inline(always)]
fn encode_struct_field(size: u32, refcounted: bool) -> u64 {
    (refcounted as u64) << 63 | (size as u64)
}

#[inline(always)]
fn decode_struct_field(word: u64) -> (u32, bool) {
    let refcounted = (word >> 63) != 0;
    let size = (word & 0xFFFF_FFFF) as u32;
    (size, refcounted)
}

impl Program {
    pub fn to_binary(&self) -> Vec<u64> {
        let mut binary = Vec::new();
        // Layout-table header: [num_descriptors, ...descriptors...].
        // Each descriptor: [num_fields, field_0_packed, field_1_packed, ...].
        binary.push(self.struct_layouts.len() as u64);
        for desc in &self.struct_layouts {
            let fields = desc.fields();
            binary.push(fields.len() as u64);
            for &(size, refcounted) in fields {
                binary.push(encode_struct_field(size, refcounted));
            }
        }

        binary.push(self.global_frame_size as u64);
        let mut positions = vec![];
        let mut jumps_to_fix: Vec<(usize, isize)> = vec![];

        for function in &self.functions {
            // Function marker
            binary.push(u64::MAX);
            binary.push(function.frame_size as u64);

            for op in &function.code {
                positions.push(binary.len());
                op.to_binary(&mut binary, &mut jumps_to_fix);
            }
        }
        for (jump_position, add_offset) in jumps_to_fix {
            let target = binary[jump_position];
            let target_pos = positions[target as usize];
            binary[jump_position] =
                (target_pos as isize - (jump_position as isize + add_offset)) as u64;
        }
        binary
    }
}

/// Read the struct-layout table from the binary header and return both the
/// descriptors and the offset (in u64 words) at which the rest of the program
/// (starting with `global_frame_size`) begins.
pub fn parse_struct_layouts(program: &[u64]) -> (Vec<StructDescriptor>, usize) {
    let num_descriptors = program[0] as usize;
    let mut layouts = Vec::with_capacity(num_descriptors);
    let mut off = 1usize;
    for _ in 0..num_descriptors {
        let n = program[off] as usize;
        off += 1;
        let mut fields = Vec::with_capacity(n);
        for _ in 0..n {
            fields.push(decode_struct_field(program[off]));
            off += 1;
        }
        layouts.push(StructDescriptor::new(fields));
    }
    (layouts, off)
}
