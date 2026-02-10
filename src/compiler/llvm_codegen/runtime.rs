//! Runtime function declarations for LLVM codegen
//!
//! Declares external functions that will be provided by the WASM runtime:
//! - Field arithmetic operations
//!
//! VM write operations are defined as internal functions for LLVM to optimize.
//!
//! Field elements are represented as [4 x i64] arrays in Montgomery form.
//! Uses WASM-friendly calling convention: field elements passed by value.

use inkwell::builder::Builder;
use inkwell::context::Context;
use inkwell::module::{Linkage, Module};
use inkwell::values::{BasicValueEnum, FunctionValue, PointerValue};
use inkwell::AddressSpace;

use ark_bn254::Fr;

use super::types::FIELD_LIMBS;

const FIELD_SIZE: u32 = 32; // 4 x i64 = 32 bytes

/// VM struct layout (offsets in bytes):
/// - 0: witness write pointer (i32)
/// - 4: a write pointer (i32)
/// - 8: b write pointer (i32)
/// - 12: c write pointer (i32)
const VM_WITNESS_PTR_OFFSET: u32 = 0;
const VM_A_PTR_OFFSET: u32 = 4;
const VM_B_PTR_OFFSET: u32 = 8;
const VM_C_PTR_OFFSET: u32 = 12;

/// Runtime function declarations
pub struct Runtime<'ctx> {
    context: &'ctx Context,
    field_mul_fn: Option<FunctionValue<'ctx>>,
    write_witness_fn: Option<FunctionValue<'ctx>>,
    write_a_fn: Option<FunctionValue<'ctx>>,
    write_b_fn: Option<FunctionValue<'ctx>>,
    write_c_fn: Option<FunctionValue<'ctx>>,
}

impl<'ctx> Runtime<'ctx> {
    pub fn new(context: &'ctx Context, module: &Module<'ctx>) -> Self {
        let mut rt = Self {
            context,
            field_mul_fn: None,
            write_witness_fn: None,
            write_a_fn: None,
            write_b_fn: None,
            write_c_fn: None,
        };

        rt.declare_functions(module);
        rt.define_write_functions(module);
        rt
    }

    fn field_type(&self) -> inkwell::types::ArrayType<'ctx> {
        self.context.i64_type().array_type(FIELD_LIMBS)
    }

    fn declare_functions(&mut self, module: &Module<'ctx>) {
        let field_type = self.field_type();

        // Only field_mul remains as an external function
        let field_mul_type = field_type.fn_type(
            &[field_type.into(), field_type.into()],
            false
        );
        self.field_mul_fn = Some(module.add_function("__field_mul", field_mul_type, None));
    }

    /// Define write functions as internal LLVM functions
    fn define_write_functions(&mut self, module: &Module<'ctx>) {
        self.write_witness_fn = Some(self.define_write_fn(module, "__write_witness", VM_WITNESS_PTR_OFFSET));
        self.write_a_fn = Some(self.define_write_fn(module, "__write_a", VM_A_PTR_OFFSET));
        self.write_b_fn = Some(self.define_write_fn(module, "__write_b", VM_B_PTR_OFFSET));
        self.write_c_fn = Some(self.define_write_fn(module, "__write_c", VM_C_PTR_OFFSET));
    }

    /// Define a single write function
    fn define_write_fn(
        &self,
        module: &Module<'ctx>,
        name: &str,
        ptr_offset: u32,
    ) -> FunctionValue<'ctx> {
        let void_type = self.context.void_type();
        let ptr_type = self.context.ptr_type(AddressSpace::default());
        let field_type = self.field_type();
        let i32_type = self.context.i32_type();

        // Function signature: void(ptr vm_ptr, [4 x i64] value)
        let fn_type = void_type.fn_type(
            &[ptr_type.into(), field_type.into()],
            false
        );

        let function = module.add_function(name, fn_type, Some(Linkage::Internal));

        // Create entry block
        let entry = self.context.append_basic_block(function, "entry");
        let builder = self.context.create_builder();
        builder.position_at_end(entry);

        // Get parameters
        let vm_ptr = function.get_nth_param(0).unwrap().into_pointer_value();
        let value = function.get_nth_param(1).unwrap().into_array_value();

        // Get pointer to the write position slot in VM struct (ptr to ptr)
        let write_pos_ptr = unsafe {
            builder.build_gep(
                ptr_type,
                vm_ptr,
                &[i32_type.const_int((ptr_offset / 4) as u64, false)],
                "pos_ptr",
            ).unwrap()
        };

        // Load current write position directly as pointer
        let write_ptr = builder
            .build_load(ptr_type, write_pos_ptr, "ptr")
            .unwrap()
            .into_pointer_value();

        // Store entire [4 x i64] array at once
        builder.build_store(write_ptr, value).unwrap();

        // Advance write pointer by FIELD_SIZE bytes using GEP on i8
        let i8_type = self.context.i8_type();
        let new_ptr = unsafe {
            builder.build_gep(
                i8_type,
                write_ptr,
                &[i32_type.const_int(FIELD_SIZE as u64, false)],
                "new_ptr",
            ).unwrap()
        };

        // Store updated pointer
        builder.build_store(write_pos_ptr, new_ptr).unwrap();

        // Return void
        builder.build_return(None).unwrap();

        function
    }

    pub fn const_field(&self, value: &Fr) -> BasicValueEnum<'ctx> {
        let i64_type = self.context.i64_type();
        let limbs = value.0.0;

        let mut values = Vec::with_capacity(FIELD_LIMBS as usize);
        for i in 0..FIELD_LIMBS as usize {
            values.push(i64_type.const_int(limbs[i], false));
        }

        let array = i64_type.const_array(&values);
        array.into()
    }

    pub fn field_mul(
        &self,
        builder: &Builder<'ctx>,
        lhs: BasicValueEnum<'ctx>,
        rhs: BasicValueEnum<'ctx>,
    ) -> BasicValueEnum<'ctx> {
        let mul_fn = self.field_mul_fn.expect("__field_mul not declared");

        let call_site = builder
            .build_call(mul_fn, &[lhs.into(), rhs.into()], "field_mul")
            .unwrap();

        call_site
            .try_as_basic_value()
            .left()
            .expect("field_mul should return a value")
    }

    pub fn write_witness(
        &self,
        builder: &Builder<'ctx>,
        vm_ptr: PointerValue<'ctx>,
        value: BasicValueEnum<'ctx>,
    ) {
        let write_fn = self.write_witness_fn.expect("__write_witness not defined");
        builder
            .build_call(write_fn, &[vm_ptr.into(), value.into()], "")
            .unwrap();
    }

    pub fn write_a(
        &self,
        builder: &Builder<'ctx>,
        vm_ptr: PointerValue<'ctx>,
        value: BasicValueEnum<'ctx>,
    ) {
        let write_fn = self.write_a_fn.expect("__write_a not defined");
        builder
            .build_call(write_fn, &[vm_ptr.into(), value.into()], "")
            .unwrap();
    }

    pub fn write_b(
        &self,
        builder: &Builder<'ctx>,
        vm_ptr: PointerValue<'ctx>,
        value: BasicValueEnum<'ctx>,
    ) {
        let write_fn = self.write_b_fn.expect("__write_b not defined");
        builder
            .build_call(write_fn, &[vm_ptr.into(), value.into()], "")
            .unwrap();
    }

    pub fn write_c(
        &self,
        builder: &Builder<'ctx>,
        vm_ptr: PointerValue<'ctx>,
        value: BasicValueEnum<'ctx>,
    ) {
        let write_fn = self.write_c_fn.expect("__write_c not defined");
        builder
            .build_call(write_fn, &[vm_ptr.into(), value.into()], "")
            .unwrap();
    }
}
