#![allow(unused_variables)]

use crate::interpreter::dispatch;
use crate::{ConstraintsLayout, Field, WitnessLayout};
use ark_ff::{AdditiveGroup as _, BigInteger as _};
use opcode_gen::interpreter;

use crate::array::{BoxedLayout, BoxedValue};
use crate::interpreter::{Frame, Handler};

use crate::array::DataType;
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
        ELEM_WORD => {
            let v = Field::from(*(ptr as *const u64));
            *vm.data.as_ad.out_db += coeff * v;
        }
        ELEM_FIELD => {
            let v = *(ptr as *const Field);
            *vm.data.as_ad.out_db += coeff * v;
        }
        ELEM_WITNESS => {
            let elem = BoxedValue(*(ptr as *const *mut u64));
            elem.bump_db(coeff, vm);
        }
        _ => unreachable!(),
    }
}

/// Read a pure (non-WitnessOf) array element as a Field value.
#[inline(always)]
unsafe fn read_pure_elem_as_field(ptr: *mut u64, elem_kind: usize) -> Field {
    match elem_kind {
        ELEM_WORD => Field::from(*(ptr as *const u64)),
        ELEM_FIELD => *(ptr as *const Field),
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
}

pub union Arrays {
    pub as_forward: FwdArrays,
    pub as_ad:      AdArrays,
}

pub struct VM {
    pub data:                    Arrays,
    pub allocation_instrumenter: AllocationInstrumenter,
    pub tables:                  Vec<TableInfo>,
    pub rgchk_8:                 Option<usize>,
    pub spread_tables:           [Option<usize>; 17],
    pub globals:                 *mut u64,
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
                    current_wit_tables_off: witness_layout.tables_data_start(),
                    current_wit_lookups_off: witness_layout.lookups_data_start(),
                    current_cnst_off: 0,
                    current_cnst_tables_off: constraints_layout.tables_data_start(),
                    current_cnst_lookups_off: constraints_layout.lookups_data_start(),
                },
            },
            allocation_instrumenter: AllocationInstrumenter::new(),
            tables: vec![],
            rgchk_8: None,
            spread_tables: [None; 17],
            globals,
        }
    }

    // pub fn new_
}

/// Compute spread of a u32: interleave zero bits between each bit.
fn spread_bits(v: u32) -> u64 {
    let mut x = v as u64;
    x = (x | (x << 16)) & 0x0000_ffff_0000_ffff;
    x = (x | (x << 8)) & 0x00ff_00ff_00ff_00ff;
    x = (x | (x << 4)) & 0x0f0f_0f0f_0f0f_0f0f;
    x = (x | (x << 2)) & 0x3333_3333_3333_3333;
    x = (x | (x << 1)) & 0x5555_5555_5555_5555;
    x
}

/// Compact even-positioned bits into contiguous low bits.
fn compact_bits(mut x: u64) -> u32 {
    x &= 0x5555_5555_5555_5555;
    x = (x | (x >> 1)) & 0x3333_3333_3333_3333;
    x = (x | (x >> 2)) & 0x0f0f_0f0f_0f0f_0f0f;
    x = (x | (x >> 4)) & 0x00ff_00ff_00ff_00ff;
    x = (x | (x >> 8)) & 0x0000_ffff_0000_ffff;
    x = (x | (x >> 16)) & 0x0000_0000_ffff_ffff;
    x as u32
}

/// Extract even bits and odd bits from a spread sum. Returns (odd_bits,
/// even_bits).
fn unspread_bits(v: u64) -> (u32, u32) {
    let even = compact_bits(v);
    let odd = compact_bits(v >> 1);
    (odd, even)
}

/// Emit a forward key-value lookup: bump multiplicity and write 2 lookup tape
/// entries.
unsafe fn forward_kv_lookup_emit(
    table_idx: usize,
    key: Field,
    result: Field,
    flag_u64: u64,
    vm: &mut VM,
) {
    let table_info = &vm.tables[table_idx];

    // Entry 1 (x-constraint): table_id, result_value, 0
    *(vm.data.as_forward.lookups_a as *mut u64) = table_idx as u64;
    *vm.data.as_forward.lookups_b = result;
    *(vm.data.as_forward.lookups_c as *mut u64) = 0;
    vm.data.as_forward.lookups_a = vm.data.as_forward.lookups_a.offset(1);
    vm.data.as_forward.lookups_b = vm.data.as_forward.lookups_b.offset(1);
    vm.data.as_forward.lookups_c = vm.data.as_forward.lookups_c.offset(1);

    // Entry 2 (y-constraint): table_id, key, flag
    *(vm.data.as_forward.lookups_a as *mut u64) = table_idx as u64;
    if flag_u64 != 0 {
        let key_u64 = ark_ff::PrimeField::into_bigint(key).0[0];
        let ptr = table_info.multiplicities_wit.offset(key_u64 as isize);
        *(ptr as *mut u64) += flag_u64;
        *(vm.data.as_forward.lookups_b as *mut u64) = key_u64;
    } else {
        *(vm.data.as_forward.lookups_b as *mut Field) = key;
    }
    *(vm.data.as_forward.lookups_c as *mut u64) = flag_u64;
    vm.data.as_forward.lookups_a = vm.data.as_forward.lookups_a.offset(1);
    vm.data.as_forward.lookups_b = vm.data.as_forward.lookups_b.offset(1);
    vm.data.as_forward.lookups_c = vm.data.as_forward.lookups_c.offset(1);
}

/// Emit AD bumps for a key-value lookup (x-constraint + y-constraint + sum).
unsafe fn ad_kv_lookup_emit(
    table_idx: usize,
    key: BoxedValue,
    result: BoxedValue,
    flag: BoxedValue,
    vm: &mut VM,
) {
    let table_info = &vm.tables[table_idx];
    let cnst_off = table_info.elem_inverses_constraint_section_offset;
    let length = table_info.length;

    let x_coeff = {
        let r = *vm
            .data
            .as_ad
            .ad_coeffs
            .offset(vm.data.as_ad.current_cnst_lookups_off as isize);
        vm.data.as_ad.current_cnst_lookups_off += 1;
        r
    };
    let x_wit_off = {
        let r = vm.data.as_ad.current_wit_lookups_off;
        vm.data.as_ad.current_wit_lookups_off += 1;
        r
    };
    let y_coeff = {
        let r = *vm
            .data
            .as_ad
            .ad_coeffs
            .offset(vm.data.as_ad.current_cnst_lookups_off as isize);
        vm.data.as_ad.current_cnst_lookups_off += 1;
        r
    };
    let y_wit_off = {
        let r = vm.data.as_ad.current_wit_lookups_off;
        vm.data.as_ad.current_wit_lookups_off += 1;
        r
    };
    let inv_sum_coeff = *vm
        .data
        .as_ad
        .ad_coeffs
        .offset(cnst_off as isize + 2 * length as isize);

    // x-constraint: beta * result - x_lookup = 0
    *vm.data
        .as_ad
        .out_da
        .offset(vm.data.as_ad.logup_wit_challenge_off as isize + 1) += x_coeff;
    result.bump_db(x_coeff, vm);
    *vm.data.as_ad.out_dc.offset(x_wit_off as isize) -= x_coeff;

    // y-constraint: y * (alpha - x_lookup - key) = flag
    *vm.data.as_ad.out_da.offset(y_wit_off as isize) += y_coeff;
    *vm.data
        .as_ad
        .out_db
        .offset(vm.data.as_ad.logup_wit_challenge_off as isize) += y_coeff;
    *vm.data.as_ad.out_db.offset(x_wit_off as isize) -= y_coeff;
    key.bump_db(-y_coeff, vm);
    flag.bump_dc(y_coeff, vm);

    // Sum constraint
    *vm.data.as_ad.out_dc.offset(y_wit_off as isize) += inv_sum_coeff;
}

#[interpreter]
mod def {
    #[raw_opcode]
    fn jmp(pc: *const u64, frame: Frame, vm: &mut VM, target: JumpTarget) {
        let pc = unsafe { pc.offset(target.0) };
        // println!("jmp: target={:?}", pc);
        dispatch(pc, frame, vm);
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
        dispatch(pc, frame, vm);
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
        let ret_data_ptr = unsafe { frame.data.offset(ret.0 as isize) };
        let ret_pc = unsafe { pc.offset(4 + 2 * args.len() as isize) };

        unsafe {
            *new_frame.data = ret_data_ptr as u64;
            *new_frame.data.offset(1) = ret_pc as u64;
        };

        let mut current_child = unsafe { new_frame.data.offset(2) };

        for (i, (arg_size, arg_pos)) in args.iter().enumerate() {
            frame.write_to(current_child, arg_pos.0 as isize, *arg_size);
            current_child = unsafe { current_child.offset(*arg_size as isize) };
        }

        dispatch(func_pc, new_frame, vm);
    }

    #[raw_opcode]
    fn ret(_pc: *const u64, frame: Frame, vm: &mut VM) {
        let ret_address = unsafe { *frame.data.offset(1) } as *mut u64;
        let new_frame = frame.pop(vm);
        if new_frame.data.is_null() {
            return;
        }
        dispatch(ret_address, new_frame, vm);
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
        dispatch(pc, frame, vm);
    }

    #[raw_opcode]
    fn write_witness(pc: *const u64, frame: Frame, vm: &mut VM, #[frame] val: Field) {
        unsafe {
            *vm.data.as_forward.algebraic_witness = val;
            vm.data.as_forward.algebraic_witness = vm.data.as_forward.algebraic_witness.offset(1);
        };
        let pc = unsafe { pc.offset(2) };
        dispatch(pc, frame, vm);
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
        frame.write_to(ptr, src.0 as isize, size);
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
            frame.write_to(tgt, item.0 as isize, stride);
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
        for (i, field) in fields.iter().enumerate() {
            let tgt = tuple.tuple_idx(i, &meta.child_sizes());
            frame.write_to(tgt, field.0 as isize, meta.child_sizes()[i]);
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
        frame.write_to(cell.data(), source.0 as isize, stride);
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
        index: u64,
        child_sizes: &[usize],
        vm: &mut VM,
    ) {
        let src = tuple.tuple_idx(index as usize, child_sizes);
        unsafe {
            ptr::copy_nonoverlapping(src, res, child_sizes[index as usize]);
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
        frame.write_to(target, source.0 as isize, stride);
        unsafe {
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
            *v = *vm
                .data
                .as_ad
                .ad_coeffs
                .offset(vm.data.as_ad.current_cnst_off as isize);
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
                    (val.0[limb_idx] >> (byte_in_limb * 8)) & 0xff
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
            };
            vm.spread_tables[bits] = Some(vm.tables.len());
            vm.tables.push(table_info);

            // Fill table x-slots with spread values
            unsafe {
                let cnst_off = vm.data.as_forward.elem_inverses_constraint_section_offset;
                for i in 0..length {
                    *vm.data
                        .as_forward
                        .out_a_base
                        .offset((cnst_off + 2 * i) as isize) = Field::from(spread_bits(i as u32));
                }

                vm.data.as_forward.multiplicities_witness = vm
                    .data
                    .as_forward
                    .multiplicities_witness
                    .offset(length as isize);
                vm.data.as_forward.elem_inverses_constraint_section_offset += 2 * length + 1;
                vm.data.as_forward.elem_inverses_witness_section_offset += 2 * length;
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
            };
            vm.spread_tables[bits] = Some(vm.tables.len());
            vm.tables.push(table_info);
            unsafe {
                vm.data.as_ad.current_wit_multiplicities_off += length;
                vm.data.as_ad.current_wit_tables_off += 2 * length;
                vm.data.as_ad.current_cnst_tables_off += 2 * length + 1;
            }

            let inv_sum_coeff = unsafe {
                *vm.data
                    .as_ad
                    .ad_coeffs
                    .offset(inverses_constraint_section_offset as isize + 2 * length as isize)
            };

            for i in 0..length {
                // x-constraint: β * spread(i) - x = 0
                let x_coeff = unsafe {
                    *vm.data
                        .as_ad
                        .ad_coeffs
                        .offset(inverses_constraint_section_offset as isize + 2 * i as isize)
                };
                unsafe {
                    // x-constraint: β * spread(i) = -x
                    // A=(β,1), B=(w0, spread(i)), C=(x,-1)
                    // da[β] += x_coeff
                    *vm.data
                        .as_ad
                        .out_da
                        .offset(vm.data.as_ad.logup_wit_challenge_off as isize + 1) += x_coeff;
                    // db[w0] += x_coeff * spread(i)
                    *vm.data.as_ad.out_db += x_coeff * Field::from(spread_bits(i as u32));
                    // dc[x_wit] -= x_coeff
                    *vm.data
                        .as_ad
                        .out_dc
                        .offset(inverses_witness_section_offset as isize + 2 * i as isize) -=
                        x_coeff;
                }

                // y-constraint: y * (α - i - x) - m = 0
                let y_coeff = unsafe {
                    *vm.data
                        .as_ad
                        .ad_coeffs
                        .offset(inverses_constraint_section_offset as isize + 2 * i as isize + 1)
                };
                unsafe {
                    *vm.data
                        .as_ad
                        .out_da
                        .offset(inverses_witness_section_offset as isize + 2 * i as isize + 1) +=
                        y_coeff;

                    *vm.data
                        .as_ad
                        .out_db
                        .offset(vm.data.as_ad.logup_wit_challenge_off as isize) += y_coeff;
                    *vm.data.as_ad.out_db -= y_coeff * Field::from(i as u64);
                    *vm.data
                        .as_ad
                        .out_db
                        .offset(inverses_witness_section_offset as isize + 2 * i as isize) -=
                        y_coeff;

                    *vm.data
                        .as_ad
                        .out_dc
                        .offset(multiplicities_wit_offset as isize + i as isize) += y_coeff;
                }

                // Sum: inv goes into A position
                unsafe {
                    *vm.data
                        .as_ad
                        .out_da
                        .offset(inverses_witness_section_offset as isize + 2 * i as isize + 1) +=
                        inv_sum_coeff;
                }
            }

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
            };
            let new_table_idx = vm.tables.len();
            vm.tables.push(table_info);

            // Dump array element values into the x-slots (even offsets) of the table
            // section
            unsafe {
                let cnst_off = vm.data.as_forward.elem_inverses_constraint_section_offset;
                for i in 0..length {
                    let elem_ptr = array.array_idx(i, stride);
                    let elem_field = read_pure_elem_as_field(elem_ptr, elem_kind);
                    // Write it into the x-slot (even offset: 2*i) of the constraint section
                    *vm.data
                        .as_forward
                        .out_a_base
                        .offset((cnst_off + 2 * i) as isize) = elem_field;
                }

                vm.data.as_forward.multiplicities_witness = vm
                    .data
                    .as_forward
                    .multiplicities_witness
                    .offset(length as isize);
                // 2 constraints per element + 1 sum constraint
                vm.data.as_forward.elem_inverses_constraint_section_offset += 2 * length + 1;
                // 2 witness slots per element
                vm.data.as_forward.elem_inverses_witness_section_offset += 2 * length;
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
                // For each element in the table, we have constraint `elem_inv_witness * (alpha
                // - i) - multiplicity_witness = 0`
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
                    //     println!("bump da at {} from inv by {coeff}",
                    // inverses_witness_section_offset as isize + i); }

                    *vm.data
                        .as_ad
                        .out_db
                        .offset(vm.data.as_ad.logup_wit_challenge_off as isize) += coeff;
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
                .offset(vm.data.as_ad.current_cnst_lookups_off as isize);
            vm.data.as_ad.current_cnst_lookups_off += 1;
            r
        };

        let inv_sum_coeff = unsafe {
            *vm.data
                .as_ad
                .ad_coeffs
                .offset(table_info.elem_inverses_constraint_section_offset as isize + 256)
        };

        let current_inv_wit_offset = unsafe {
            let r = vm.data.as_ad.current_wit_lookups_off;
            vm.data.as_ad.current_wit_lookups_off += 1;
            r
        };

        unsafe {
            // bump for the RHS of the sum
            *vm.data.as_ad.out_dc.offset(current_inv_wit_offset as isize) += inv_sum_coeff;

            // bumps for the inversion assert: y*(α-key) = flag
            // da[y] += inv_coeff
            *vm.data.as_ad.out_da.offset(current_inv_wit_offset as isize) += inv_coeff;

            // db[α] += inv_coeff
            *vm.data
                .as_ad
                .out_db
                .offset(vm.data.as_ad.logup_wit_challenge_off as isize) += inv_coeff;
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

        let table_idx =
            if table_idx == u64::MAX {
                // First AD call on this array: create table and process table constraints
                let length = array.layout().array_size() / stride;
                let inverses_constraint_section_offset =
                    unsafe { vm.data.as_ad.current_cnst_tables_off };
                let inverses_witness_section_offset =
                    unsafe { vm.data.as_ad.current_wit_tables_off };
                let multiplicities_wit_offset =
                    unsafe { vm.data.as_ad.current_wit_multiplicities_off };
                let table_info = TableInfo {
                    multiplicities_wit: ptr::null_mut(),
                    num_indices: 1,
                    num_values: 1,
                    length,
                    elem_inverses_witness_section_offset: inverses_witness_section_offset,
                    elem_inverses_constraint_section_offset: inverses_constraint_section_offset,
                };
                let new_table_idx = vm.tables.len();
                vm.tables.push(table_info);
                unsafe {
                    vm.data.as_ad.current_wit_multiplicities_off += length;
                    // 2 witness slots per element (x and y)
                    vm.data.as_ad.current_wit_tables_off += 2 * length;
                    // 2 constraints per element + 1 sum constraint
                    vm.data.as_ad.current_cnst_tables_off += 2 * length + 1;
                }

                let sum_coeff = unsafe {
                    *vm.data
                        .as_ad
                        .ad_coeffs
                        .offset(inverses_constraint_section_offset as isize + 2 * length as isize)
                };

                for i in 0..length {
                    let elem_ptr = array.array_idx(i, stride);

                    // x-constraint at base + 2*i: A=[(beta,1)], B=v_i, C=[(x,-1)]
                    let x_coeff = unsafe {
                        *vm.data
                            .as_ad
                            .ad_coeffs
                            .offset(inverses_constraint_section_offset as isize + 2 * i as isize)
                    };
                    unsafe {
                        // da[beta] += x_coeff (A entry: (beta, 1))
                        *vm.data
                            .as_ad
                            .out_da
                            .offset(vm.data.as_ad.logup_wit_challenge_off as isize + 1) += x_coeff;
                        // db[v_i] += x_coeff (B entry: element value)
                        lookup_elem_bump_db(elem_ptr, elem_kind, x_coeff, vm);
                        // dc[x_wit] -= x_coeff (C entry: (x, -1))
                        *vm.data
                            .as_ad
                            .out_dc
                            .offset(inverses_witness_section_offset as isize + 2 * i as isize) -=
                            x_coeff;
                    }

                    // y-constraint at base + 2*i + 1: A=y_i, B=(alpha - i - x_i), C=mult_i
                    let y_coeff = unsafe {
                        *vm.data.as_ad.ad_coeffs.offset(
                            inverses_constraint_section_offset as isize + 2 * i as isize + 1,
                        )
                    };
                    unsafe {
                        // dA[y_witness] += y_coeff
                        *vm.data.as_ad.out_da.offset(
                            inverses_witness_section_offset as isize + 2 * i as isize + 1,
                        ) += y_coeff;
                        // dB[alpha] += y_coeff
                        *vm.data
                            .as_ad
                            .out_db
                            .offset(vm.data.as_ad.logup_wit_challenge_off as isize) += y_coeff;
                        // dB -= y_coeff * i (constant part)
                        *vm.data.as_ad.out_db -= y_coeff * Field::from(i as u64);
                        // dB[x_witness] -= y_coeff (x_i appears negated in B)
                        *vm.data
                            .as_ad
                            .out_db
                            .offset(inverses_witness_section_offset as isize + 2 * i as isize) -=
                            y_coeff;
                        // dC[mult_witness] += y_coeff
                        *vm.data
                            .as_ad
                            .out_dc
                            .offset(multiplicities_wit_offset as isize + i as isize) += y_coeff;
                    }

                    // Sum constraint: y_i goes into A position
                    unsafe {
                        *vm.data.as_ad.out_da.offset(
                            inverses_witness_section_offset as isize + 2 * i as isize + 1,
                        ) += sum_coeff;
                    }
                }

                // Sum constraint B=1: bump out_db by sum_coeff
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
                frame.data.offset(src.0 as isize),
                vm.globals.offset(global_offset as isize),
                size,
            );
        }
    }

    #[opcode]
    fn read_global(#[out] res: *mut u64, vm: &mut VM, global_offset: usize, size: usize) {
        unsafe {
            std::ptr::copy_nonoverlapping(vm.globals.offset(global_offset as isize), res, size);
        }
    }

    #[opcode]
    #[inline(never)]
    fn drop_global(vm: &mut VM, global_offset: usize) {
        unsafe {
            let boxed = *(vm.globals.offset(global_offset as isize) as *mut BoxedValue);
            boxed.dec_rc(vm);
        }
    }
}

pub struct Function {
    pub name:       String,
    pub frame_size: usize,
    pub code:       Vec<OpCode>,
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
    pub functions:         Vec<Function>,
    pub global_frame_size: usize,
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

impl Program {
    pub fn to_binary(&self) -> Vec<u64> {
        let mut binary = Vec::new();
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
