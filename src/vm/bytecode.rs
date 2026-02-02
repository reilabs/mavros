#![allow(unused_variables)]

use crate::compiler::r1cs_gen::{ConstraintsLayout, WitnessLayout};
use crate::vm::interpreter::dispatch;
use ark_ff::{AdditiveGroup as _, BigInteger as _};
use opcode_gen::interpreter;

use crate::vm::array::{BoxedLayout, BoxedValue};
use crate::{
    compiler::Field,
    vm::interpreter::{Frame, Handler},
};

use crate::vm::array::DataType;
use plotters::prelude::*;
use std::fmt::Display;
use std::path::Path;
use std::ptr;

pub const LIMBS: usize = 4;

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

pub enum AllocationType {
    Stack,
    Heap,
}

pub enum AlocationEvent {
    Alloc(AllocationType, usize),
    Free(AllocationType, usize),
}

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

    pub fn plot(&self, path: &Path) -> usize {
        // Calculate memory usage over time
        let mut stack_usage = Vec::new();
        let mut heap_usage = Vec::new();
        let mut current_stack = 0usize;
        let mut current_heap = 0usize;

        // Process allocation events to build memory usage timeline
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

            stack_usage.push(current_stack);
            heap_usage.push(current_heap);
        }

        if stack_usage.is_empty() {
            return 0; // No events to plot
        }

        // Calculate total memory usage
        let total_usage: Vec<usize> = stack_usage
            .iter()
            .zip(heap_usage.iter())
            .map(|(s, h)| s + h)
            .collect();

        // Find maximum values for each plot
        let max_stack = *stack_usage.iter().max().unwrap_or(&1);
        let max_heap = *heap_usage.iter().max().unwrap_or(&1);
        let max_total = *total_usage.iter().max().unwrap_or(&1);

        // Create the chart with three subplots side by side
        let root = BitMapBackend::new(path, (2400, 800)).into_drawing_area();
        root.fill(&WHITE).unwrap();

        // Split the drawing area into three equal horizontal sections
        let (left, rest) = root.split_horizontally(800);
        let (middle, right) = rest.split_horizontally(800);

        // Common Y-axis scale for all plots
        let common_max = max_total.max(max_stack).max(max_heap);

        // Determine the best unit and conversion factor
        let (unit, divisor, y_label) = if common_max >= 2 * 1024 * 1024 {
            ("MB", 1024 * 1024, "Memory Size (MB)".to_string())
        } else if common_max >= 2 * 1024 {
            ("KB", 1024, "Memory Size (KB)".to_string())
        } else {
            ("B", 1, "Memory Size (bytes)".to_string())
        };

        // Convert data to the appropriate unit
        let total_data: Vec<(usize, f64)> = total_usage
            .iter()
            .enumerate()
            .map(|(i, &size)| (i, size as f64 / divisor as f64))
            .collect();

        let stack_data: Vec<(usize, f64)> = stack_usage
            .iter()
            .enumerate()
            .map(|(i, &size)| (i, size as f64 / divisor as f64))
            .collect();

        let heap_data: Vec<(usize, f64)> = heap_usage
            .iter()
            .enumerate()
            .map(|(i, &size)| (i, size as f64 / divisor as f64))
            .collect();

        let y_max = common_max as f64 / divisor as f64;

        // Plot 1: Total Memory Usage
        let mut chart1 = ChartBuilder::on(&left)
            .caption("Total Memory Usage", ("sans-serif", 20))
            .margin(5)
            .x_label_area_size(30)
            .y_label_area_size(50)
            .build_cartesian_2d(0..total_usage.len(), 0.0..y_max)
            .unwrap();

        chart1
            .configure_mesh()
            .x_labels(5)
            .y_labels(5)
            .x_desc("Event Number")
            .y_desc(y_label.clone())
            .draw()
            .unwrap();

        chart1
            .draw_series(
                total_data
                    .iter()
                    .map(|&(x, y)| Rectangle::new([(x, 0.0), (x + 1, y)], GREEN.filled())),
            )
            .unwrap()
            .label("Total Memory")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], GREEN));

        chart1
            .configure_series_labels()
            .background_style(&WHITE.mix(0.8))
            .border_style(&BLACK)
            .draw()
            .unwrap();

        // Plot 2: Stack Memory Usage
        let mut chart2 = ChartBuilder::on(&middle)
            .caption("Stack Memory Usage", ("sans-serif", 20))
            .margin(5)
            .x_label_area_size(30)
            .y_label_area_size(50)
            .build_cartesian_2d(0..stack_usage.len(), 0.0..y_max)
            .unwrap();

        chart2
            .configure_mesh()
            .x_labels(5)
            .y_labels(5)
            .x_desc("Event Number")
            .y_desc(y_label.clone())
            .draw()
            .unwrap();

        chart2
            .draw_series(
                stack_data
                    .iter()
                    .map(|&(x, y)| Rectangle::new([(x, 0.0), (x + 1, y)], BLUE.filled())),
            )
            .unwrap()
            .label("Stack Memory")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], BLUE));

        chart2
            .configure_series_labels()
            .background_style(&WHITE.mix(0.8))
            .border_style(&BLACK)
            .draw()
            .unwrap();

        // Plot 3: Heap Memory Usage
        let mut chart3 = ChartBuilder::on(&right)
            .caption("Heap Memory Usage", ("sans-serif", 20))
            .margin(5)
            .x_label_area_size(30)
            .y_label_area_size(50)
            .build_cartesian_2d(0..heap_usage.len(), 0.0..y_max)
            .unwrap();

        chart3
            .configure_mesh()
            .x_labels(5)
            .y_labels(5)
            .x_desc("Event Number")
            .y_desc(y_label.clone())
            .draw()
            .unwrap();

        chart3
            .draw_series(
                heap_data
                    .iter()
                    .map(|&(x, y)| Rectangle::new([(x, 0.0), (x + 1, y)], RED.filled())),
            )
            .unwrap()
            .label("Heap Memory")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], RED));

        chart3
            .configure_series_labels()
            .background_style(&WHITE.mix(0.8))
            .border_style(&BLACK)
            .draw()
            .unwrap();

        root.present().unwrap();

        *stack_usage.last().unwrap() + *heap_usage.last().unwrap()
    }
}

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
    pub algebraic_witness: *mut Field,
    pub multiplicities_witness: *mut Field,
    pub lookups_a: *mut Field,
    pub lookups_b: *mut Field,
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
    pub as_ad: AdArrays,
}

pub struct VM {
    pub data: Arrays,
    pub allocation_instrumenter: AllocationInstrumenter,
    pub tables: Vec<TableInfo>,
    pub rgchk_8: Option<usize>,
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
        elem_inverses_constraint_section_offset: usize,
        elem_inverses_witness_section_offset: usize,
    ) -> Self {
        Self {
            data: Arrays {
                as_forward: FwdArrays {
                    out_a,
                    out_b,
                    out_c,
                    algebraic_witness,
                    multiplicities_witness,
                    lookups_b,
                    lookups_a,
                    elem_inverses_constraint_section_offset,
                    elem_inverses_witness_section_offset,
                },
            },
            allocation_instrumenter: AllocationInstrumenter::new(),
            tables: vec![],
            rgchk_8: None,
        }
    }

    pub fn new_ad(
        out_da: *mut Field,
        out_db: *mut Field,
        out_dc: *mut Field,
        ad_coeffs: *const Field,

        witness_layout: WitnessLayout,
        constraints_layout: ConstraintsLayout,
    ) -> Self {
        Self {
            data: Arrays {
                as_ad: AdArrays {
                    out_da,
                    out_db,
                    out_dc,
                    ad_coeffs,
                    current_wit_off: 1,
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
        }
    }

    // pub fn new_
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

        // println!("call: func={:?} (size={})", func_pc, unsafe {*func_pc.offset(-1)});

        dispatch(func_pc, new_frame, vm);
    }

    #[raw_opcode]
    fn ret(_pc: *const u64, frame: Frame, vm: &mut VM) {
        let ret_address = unsafe { *frame.data.offset(1) } as *mut u64;
        let new_frame = frame.pop(vm);
        if new_frame.data.is_null() {
            // println!("finish return");
            return;
        }
        // println!("ret");
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
        // println!("r1cs");

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
    fn add_u64(#[out] res: *mut u64, #[frame] a: u64, #[frame] b: u64) {
        unsafe {
            *res = a + b;
        }
    }

    #[opcode]
    fn sub_u64(#[out] res: *mut u64, #[frame] a: u64, #[frame] b: u64) {
        unsafe {
            *res = a - b;
        }
    }

    #[opcode]
    fn div_u64(#[out] res: *mut u64, #[frame] a: u64, #[frame] b: u64) {
        unsafe {
            *res = a / b;
        }
    }

    #[opcode]
    fn mul_u64(#[out] res: *mut u64, #[frame] a: u64, #[frame] b: u64) {
        unsafe {
            *res = a * b;
        }
    }

    #[opcode]
    fn and_u64(#[out] res: *mut u64, #[frame] a: u64, #[frame] b: u64) {
        unsafe {
            *res = a & b;
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
    fn truncate_u64(#[out] res: *mut u64, #[frame] a: u64, to_bits: u64) {
        unsafe {
            *res = a & ((1 << to_bits) - 1);
        }
    }

    #[opcode]
    fn truncate_f_to_u(#[out] res: *mut Field, #[frame] a: Field, to_bits: u64) {
        unsafe {
            *res = From::from(ark_ff::PrimeField::into_bigint(a).0[0] & ((1 << to_bits) - 1));
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
    fn div_field(#[out] res: *mut Field, #[frame] a: Field, #[frame] b: Field) {
        unsafe {
            *res = a / b;
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
    fn array_get(
        #[out] res: *mut u64,
        #[frame] array: BoxedValue,
        #[frame] index: u64,
        stride: usize,
        vm: &mut VM,
    ) {
        // println!(
        //     "array_get: array={:?} index={} stride={}",
        //     array.0, index, stride
        // );
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
    #[inline(never)] // TODO better impl
    fn rangecheck(#[frame] val: Field, max_bits: usize) {
        // Convert field to bigint and check if it fits in max_bits
        let bigint = ark_ff::PrimeField::into_bigint(val);
        let check = bigint.to_bits_le().iter().skip(max_bits).all(|b| !b);
        assert!(check);
    }

    #[opcode]
    fn to_bytes_be_lt_8(#[frame] val: Field, count: u64, #[out] res: *mut BoxedValue, vm: &mut VM) {
        let val = ark_ff::PrimeField::into_bigint(val);
        let low = val.0[0];
        let r = BoxedValue::alloc(BoxedLayout::array(count as usize, false), vm);
        unsafe {
            for i in 0..count {
                *r.array_idx((count - i - 1) as usize, 1) = (low >> (i * 8)) & 0xFF;
            }
            *res = r;
        }
    }

    #[opcode]
    fn to_bits_le(#[out] res: *mut BoxedValue, #[frame] val: Field, count: u64, vm: &mut VM) {
        panic!("to_bits_be_lt_8 not implemented");
    }

    #[opcode]
    fn rngchk_8_field(#[frame] val: Field, vm: &mut VM) {
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
        let table_idx = *vm.rgchk_8.as_ref().unwrap();
        let table_info = &vm.tables[table_idx];
        let val_u64 = ark_ff::PrimeField::into_bigint(val).0[0];
        unsafe {
            let ptr = table_info.multiplicities_wit.offset(val_u64 as isize);
            *(ptr as *mut u64) += 1; // Use u64 for counting, convert to field later
            *(vm.data.as_forward.lookups_a as *mut u64) = table_idx as u64;
            vm.data.as_forward.lookups_a = vm.data.as_forward.lookups_a.offset(1);
            *(vm.data.as_forward.lookups_b as *mut u64) = val_u64;
            vm.data.as_forward.lookups_b = vm.data.as_forward.lookups_b.offset(1);
        }
    }

    #[opcode]
    fn drngchk_8_field(#[frame] val: BoxedValue, vm: &mut VM) {
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
            let r = *vm.data.as_ad.ad_coeffs.offset(vm.data.as_ad.current_cnst_lookups_off as isize);
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

            // bumps for the inversion assert
            *vm.data.as_ad.out_da.offset(current_inv_wit_offset as isize) += inv_coeff;

            *vm.data.as_ad.out_db.offset(vm.data.as_ad.logup_wit_challenge_off as isize) += inv_coeff;
            val.bump_db(-inv_coeff, vm);

            *vm.data.as_ad.out_dc += inv_coeff;
        }



        // unsafe {}
        // panic!("TODO: implement drngchk_8_field");
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
