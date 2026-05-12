use std::{
    alloc::{self, Layout},
    collections::VecDeque,
    ptr,
};

use crate::{
    Field,
    bytecode::{AllocationType, VM},
};

#[derive(Debug, Clone, Copy)]
pub struct BoxedLayout(pub u64);

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum DataType {
    PrimArray = 0,
    BoxedArray = 1,
    ADConst = 2,
    ADWitness = 3,
    ADSum = 4,
    ADMulConst = 5,
    Struct = 6,
    RefCell = 7,
}

/// Per-shape struct layout, interned by the compiler and shared across all
/// instances of a struct type. Indexed by the 56-bit payload of a
/// `BoxedLayout` whose `DataType` is `Struct`.
#[derive(Debug, Clone)]
pub struct StructDescriptor {
    /// Per-field `(size_in_u64_words, is_refcounted)`.
    fields: Vec<(u32, bool)>,
    /// Total size in u64 words of the struct payload (sum of field sizes).
    total_size: u32,
}

impl StructDescriptor {
    pub fn new(fields: Vec<(u32, bool)>) -> Self {
        let mut total_size: u32 = 0;
        for &(sz, _) in &fields {
            assert!(sz > 0, "struct field size must be > 0");
            total_size = total_size.checked_add(sz).expect("struct payload overflow");
        }
        Self { fields, total_size }
    }

    #[inline(always)]
    pub fn field_count(&self) -> usize {
        self.fields.len()
    }

    #[inline(always)]
    pub fn fields(&self) -> &[(u32, bool)] {
        &self.fields
    }

    #[inline(always)]
    pub fn field_size(&self, i: usize) -> u32 {
        self.fields[i].0
    }

    #[inline(always)]
    pub fn is_refcounted(&self, i: usize) -> bool {
        self.fields[i].1
    }

    #[inline(always)]
    pub fn total_size(&self) -> u32 {
        self.total_size
    }
}

/// A view into a struct's descriptor, hiding the `BoxedLayout`-to-table lookup.
#[derive(Clone, Copy)]
pub struct StructView<'a> {
    desc: &'a StructDescriptor,
}

impl<'a> StructView<'a> {
    #[inline(always)]
    pub fn field_count(&self) -> usize {
        self.desc.field_count()
    }

    #[inline(always)]
    pub fn field_size(&self, i: usize) -> usize {
        self.desc.field_size(i) as usize
    }

    #[inline(always)]
    pub fn is_refcounted(&self, i: usize) -> bool {
        self.desc.is_refcounted(i)
    }

    #[inline(always)]
    pub fn total_size(&self) -> usize {
        self.desc.total_size() as usize
    }
}

// BoxedLayout packing scheme:
// highest byte is type
// rest is length for arrays, descriptor table index for structs, or unused otherwise.

impl BoxedLayout {
    fn new_sized(data_type: DataType, size: usize) -> Self {
        assert!(size < (1 << 56));
        Self(data_type as u64 | ((size as u64) << 8))
    }

    fn new(data_type: DataType) -> Self {
        Self(data_type as u64)
    }

    pub fn array(size: usize, ptr_elems: bool) -> Self {
        assert!(size < (1 << 56));
        if ptr_elems {
            Self::new_sized(DataType::BoxedArray, size)
        } else {
            Self::new_sized(DataType::PrimArray, size)
        }
    }

    /// Build a struct layout pointing at descriptor `idx` in the program's struct-layout table.
    pub fn new_struct(idx: usize) -> Self {
        Self::new_sized(DataType::Struct, idx)
    }

    #[inline(always)]
    pub fn struct_layout_idx(&self) -> usize {
        (self.0 >> 8) as usize
    }

    pub fn ref_cell(elem_size: usize, elem_is_refcounted: bool) -> Self {
        assert!(elem_size < 8);
        Self::new_sized(
            DataType::RefCell,
            (elem_is_refcounted as usize) << 3 | elem_size,
        )
    }

    pub fn ref_cell_elem_size(&self) -> usize {
        (self.0 as usize >> 8) & 0x7
    }

    pub fn ref_cell_elem_is_refcounted(&self) -> bool {
        (self.0 as usize >> 8) & 0x8 != 0
    }

    pub fn ad_const() -> Self {
        Self::new(DataType::ADConst)
    }

    pub fn ad_witness() -> Self {
        Self::new(DataType::ADWitness)
    }

    pub fn mul_const() -> Self {
        Self::new(DataType::ADMulConst)
    }

    pub fn ad_sum() -> Self {
        Self::new(DataType::ADSum)
    }

    pub fn data_type(&self) -> DataType {
        let disciminator = self.0 as u8;
        unsafe { std::mem::transmute(disciminator) }
    }

    pub fn array_size(&self) -> usize {
        self.0 as usize >> 8
    }

    #[inline(always)]
    pub fn as_struct<'a>(&self, table: &'a [StructDescriptor]) -> StructView<'a> {
        StructView {
            desc: &table[self.struct_layout_idx()],
        }
    }

    pub fn is_boxed_array(&self) -> bool {
        self.data_type() == DataType::BoxedArray
    }

    pub fn is_prim_array(&self) -> bool {
        self.data_type() == DataType::PrimArray
    }

    pub fn underlying_array_size(&self, struct_layouts: &[StructDescriptor]) -> usize {
        let base_byte_size = match self.data_type() {
            DataType::ADConst => size_of::<ADConst>(),
            DataType::ADWitness => size_of::<ADWitness>(),
            DataType::ADMulConst => size_of::<ADMulConst>(),
            DataType::ADSum => size_of::<ADSum>(),
            DataType::BoxedArray | DataType::PrimArray => 8 * self.array_size(),
            DataType::Struct => 8 * self.as_struct(struct_layouts).total_size(),
            DataType::RefCell => 8 * self.ref_cell_elem_size(),
        };
        base_byte_size.div_ceil(8) + 3
    }
}

#[derive(Clone, Copy)]
pub struct ADConst {
    pub value: Field,
}

#[derive(Clone, Copy)]
pub struct ADWitness {
    pub index: u64,
}

#[derive(Clone, Copy)]
pub struct ADMulConst {
    pub coeff: Field,
    pub value: BoxedValue,
    pub da: Field,
    pub db: Field,
    pub dc: Field,
}

#[derive(Clone, Copy)]
pub struct ADSum {
    pub a: BoxedValue,
    pub b: BoxedValue,
    pub da: Field,
    pub db: Field,
    pub dc: Field,
}

#[derive(Clone, Copy)]
pub struct BoxedValue(pub *mut u64);

impl BoxedValue {
    pub fn alloc(layout: BoxedLayout, vm: &mut VM) -> Self {
        let arr_size = layout.underlying_array_size(&vm.struct_layouts);
        let ptr = unsafe { alloc::alloc(Layout::array::<u64>(arr_size).unwrap()) } as *mut u64;
        vm.allocation_instrumenter
            .alloc(AllocationType::Heap, arr_size);
        unsafe {
            *ptr = layout.0;
            *ptr.offset(1) = 1;
            *ptr.offset(2) = u64::MAX; // table_id sentinel: no table assigned
        }
        // println!("allocing {:?} of size {} ({:?})", ptr, arr_size, layout.data_type());
        Self(ptr)
    }

    fn rc(&self) -> *mut u64 {
        unsafe { self.0.offset(1) }
    }

    pub fn layout(&self) -> BoxedLayout {
        unsafe { *(self.0 as *mut BoxedLayout) }
    }

    pub fn table_id(&self) -> *mut u64 {
        unsafe { self.0.offset(2) }
    }

    pub fn data(&self) -> *mut u64 {
        unsafe { self.0.offset(3) }
    }

    pub fn as_ad_const(&self) -> *mut ADConst {
        self.data() as *mut ADConst
    }

    pub fn as_ad_witness(&self) -> *mut ADWitness {
        self.data() as *mut ADWitness
    }

    pub fn as_mul_const(&self) -> *mut ADMulConst {
        self.data() as *mut ADMulConst
    }

    pub fn as_ad_sum(&self) -> *mut ADSum {
        self.data() as *mut ADSum
    }

    #[inline(always)]
    pub fn bump_da(&self, amount: Field, vm: &mut VM) {
        match self.layout().data_type() {
            DataType::ADConst => unsafe {
                *vm.data.as_ad.out_da += amount * (*self.as_ad_const()).value
            },
            DataType::ADWitness => unsafe {
                *vm.data
                    .as_ad
                    .out_da
                    .offset((*self.as_ad_witness()).index as isize) += amount
            },
            DataType::ADSum => unsafe {
                let ad_sum = self.as_ad_sum();
                (*ad_sum).da += amount;
            },
            DataType::ADMulConst => unsafe {
                let ad_mul_const = self.as_mul_const();
                (*ad_mul_const).da += amount;
            },
            DataType::PrimArray => {
                panic!("bump_da for PrimArray")
            }
            DataType::BoxedArray => {
                panic!("bump_da for BoxedArray")
            }
            DataType::Struct => {
                panic!("bump_da for Struct")
            }
            DataType::RefCell => {
                panic!("bump_da for RefCell")
            }
        }
    }

    #[inline(always)]
    pub fn bump_db(&self, amount: Field, vm: &mut VM) {
        match self.layout().data_type() {
            DataType::ADConst => unsafe {
                *vm.data.as_ad.out_db += amount * (*self.as_ad_const()).value
            },
            DataType::ADWitness => unsafe {
                *vm.data
                    .as_ad
                    .out_db
                    .offset((*self.as_ad_witness()).index as isize) += amount
            },
            DataType::ADSum => unsafe {
                let ad_sum = self.as_ad_sum();
                (*ad_sum).db += amount;
            },
            DataType::ADMulConst => unsafe {
                let ad_mul_const = self.as_mul_const();
                (*ad_mul_const).db += amount;
            },
            DataType::PrimArray => {
                panic!("bump_db for PrimArray")
            }
            DataType::BoxedArray => {
                panic!("bump_da for BoxedArray")
            }
            DataType::Struct => {
                panic!("bump_db for Struct")
            }
            DataType::RefCell => {
                panic!("bump_db for RefCell")
            }
        }
    }

    #[inline(always)]
    pub fn bump_dc(&self, amount: Field, vm: &mut VM) {
        match self.layout().data_type() {
            DataType::ADConst => unsafe {
                *vm.data.as_ad.out_dc += amount * (*self.as_ad_const()).value
            },
            DataType::ADWitness => unsafe {
                *vm.data
                    .as_ad
                    .out_dc
                    .offset((*self.as_ad_witness()).index as isize) += amount
            },
            DataType::ADSum => unsafe {
                let ad_sum = self.as_ad_sum();
                (*ad_sum).dc += amount;
            },
            DataType::ADMulConst => unsafe {
                let ad_mul_const = self.as_mul_const();
                (*ad_mul_const).dc += amount;
            },
            DataType::PrimArray => {
                panic!("bump_dc for PrimArray")
            }
            DataType::BoxedArray => {
                panic!("bump_dc for BoxedArray")
            }
            DataType::Struct => {
                panic!("bump_dc for Struct")
            }
            DataType::RefCell => {
                panic!("bump_dc for RefCell")
            }
        }
    }

    // fn size(&self) -> usize {
    //     unsafe { *self.meta() }.size()
    // }

    pub fn array_idx(&self, idx: usize, stride: usize) -> *mut u64 {
        unsafe { self.data().offset(idx as isize * stride as isize) }
    }

    pub fn inc_rc(&self, by: u64) {
        let rc = self.rc();
        unsafe {
            *rc += by;
        }
    }

    fn free(&self, vm: &mut VM) {
        let arr_size = self.layout().underlying_array_size(&vm.struct_layouts);
        unsafe {
            alloc::dealloc(self.0 as *mut u8, Layout::array::<u64>(arr_size).unwrap());
            vm.allocation_instrumenter
                .free(AllocationType::Heap, arr_size);
        }
    }

    // #[inline(always)]
    pub fn dec_rc(&self, vm: &mut VM) {
        let mut queue = VecDeque::<BoxedValue>::new();
        queue.push_back(*self);
        while let Some(item) = queue.pop_front() {
            let rc = item.rc();
            let rc_val = unsafe { *rc };
            if rc_val == 1 {
                let layout = item.layout();
                match layout.data_type() {
                    DataType::PrimArray => {
                        item.free(vm);
                    }
                    DataType::BoxedArray => {
                        for i in 0..layout.array_size() {
                            let elem = unsafe { *(item.array_idx(i, 1) as *mut BoxedValue) };
                            queue.push_back(elem);
                        }
                        item.free(vm);
                    }
                    DataType::Struct => {
                        let view = layout.as_struct(&vm.struct_layouts);
                        let mut field_offset = 0;
                        for i in 0..view.field_count() {
                            if view.is_refcounted(i) {
                                let elem =
                                    unsafe { *(item.data().add(field_offset) as *mut BoxedValue) };
                                queue.push_back(elem);
                            }
                            field_offset += view.field_size(i);
                        }
                        item.free(vm);
                    }
                    DataType::RefCell => {
                        if layout.ref_cell_elem_is_refcounted() {
                            let elem = unsafe { *(item.data() as *mut BoxedValue) };
                            if !elem.0.is_null() {
                                queue.push_back(elem);
                            }
                        }
                        item.free(vm);
                    }
                    DataType::ADConst => {
                        item.free(vm);
                    }
                    DataType::ADWitness => {
                        item.free(vm);
                    }
                    DataType::ADSum => {
                        let ad_sum = unsafe { *item.as_ad_sum() };
                        ad_sum.a.bump_da(ad_sum.da, vm);
                        ad_sum.a.bump_db(ad_sum.db, vm);
                        ad_sum.a.bump_dc(ad_sum.dc, vm);
                        ad_sum.b.bump_da(ad_sum.da, vm);
                        ad_sum.b.bump_db(ad_sum.db, vm);
                        ad_sum.b.bump_dc(ad_sum.dc, vm);
                        queue.push_back(ad_sum.a);
                        queue.push_back(ad_sum.b);
                        item.free(vm);
                    }
                    DataType::ADMulConst => {
                        let ad_mul_const = unsafe { *item.as_mul_const() };
                        ad_mul_const
                            .value
                            .bump_da(ad_mul_const.da * ad_mul_const.coeff, vm);
                        ad_mul_const
                            .value
                            .bump_db(ad_mul_const.db * ad_mul_const.coeff, vm);
                        ad_mul_const
                            .value
                            .bump_dc(ad_mul_const.dc * ad_mul_const.coeff, vm);
                        queue.push_back(ad_mul_const.value);
                        item.free(vm);
                    }
                }
            } else {
                unsafe {
                    *rc -= 1;
                }
            }
        }
        // let rc = self.rc();
        // let rc_val = unsafe { *rc };
        // // println!("dec_array_rc from {} at {:?}", unsafe { *rc }, self.0);
        // if rc_val - 1 == 0 {
        //     // println!("Array @{:?} needs to be freed", self.0);
        //     let meta = unsafe { *self.meta() };
        //     if meta.ptr_elems() {
        //         // println!("Array has ptr elements, dropping");
        //         for i in 0..meta.size() {
        //             let elem = unsafe { *(self.idx(i, 1) as *mut BoxedValue) };
        //             elem.dec_rc(vm);
        //         }
        //     }
        //     unsafe {
        //         alloc::dealloc(
        //             self.0 as *mut u8,
        //             Layout::array::<u64>(meta.size() + 2).unwrap(),
        //         );
        //         vm.allocation_instrumenter
        //             .free(AllocationType::Heap, meta.size() + 2);
        //     }
        // } else {
        //     unsafe {
        //         *rc -= 1;
        //     }
        // }
    }

    pub fn copy_if_reused(&self, vm: &mut VM) -> Self {
        let rc = self.rc();
        let rc_val = unsafe { *rc };

        if rc_val == 1 {
            // println!("Array @{:?} is dying, move instead of copy", self.0);
            *self
        } else {
            // println!("Array @{:?} is not dying, copy", self.0);
            let layout = self.layout();
            let new_array = BoxedValue::alloc(layout, vm);

            unsafe {
                ptr::copy_nonoverlapping(self.data(), new_array.data(), layout.array_size());
                // Decrement RC of the old array since we've cloned it.
                *rc -= 1;
            }
            new_array
        }
    }
}
