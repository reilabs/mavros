//! LLSSA → LLVM Code Generation
//!
//! Translates LLSSA into LLVM IR, which can then be compiled to WebAssembly.
//! Operates on LLSSA + LLType — types are explicit in the LLSSA ops, no TypeInfo needed.

use std::collections::HashMap;
use std::num::NonZeroU32;
use std::path::Path;

use inkwell::AddressSpace;
use inkwell::IntPredicate;
use inkwell::OptimizationLevel;
use inkwell::builder::Builder;
use inkwell::context::Context;
use inkwell::module::{Linkage, Module};
use inkwell::targets::{
    CodeModel, FileType, InitializationConfig, RelocMode, Target, TargetTriple,
};
use inkwell::types::{BasicMetadataTypeEnum, BasicType, BasicTypeEnum};
use inkwell::values::{BasicMetadataValueEnum, BasicValueEnum, FunctionValue, PointerValue};

use crate::compiler::flow_analysis::FlowAnalysis;
use crate::compiler::llssa::{
    FieldArithOp, IntArithOp, IntCmpOp, LLFieldType, LLFunction, LLOp, LLSSA, LLStruct, LLType,
};
use crate::compiler::ssa::{BlockId, DMatrix, FunctionId, Terminator, ValueId};

use mavros_wasm_layout::{
    AD_COEFFS_PTR_OFFSET, AD_CURRENT_WIT_OFF_OFFSET, AD_OUT_DA_PTR_OFFSET, AD_OUT_DB_PTR_OFFSET,
    AD_OUT_DC_PTR_OFFSET, WASM_PTR_SIZE, WITGEN_A_PTR_OFFSET as VM_A_PTR_OFFSET,
    WITGEN_B_PTR_OFFSET as VM_B_PTR_OFFSET, WITGEN_C_PTR_OFFSET as VM_C_PTR_OFFSET,
    WITGEN_WITNESS_PTR_OFFSET as VM_WITNESS_PTR_OFFSET,
};

/// LLSSA → LLVM Code Generator
pub struct LLVMCodeGen<'ctx> {
    context: &'ctx Context,
    module: Module<'ctx>,
    builder: Builder<'ctx>,
    value_map: HashMap<ValueId, BasicValueEnum<'ctx>>,
    block_map: HashMap<BlockId, inkwell::basic_block::BasicBlock<'ctx>>,
    function_map: HashMap<FunctionId, FunctionValue<'ctx>>,
    vm_ptr: Option<PointerValue<'ctx>>,
    // Runtime function declarations
    field_mul_fn: Option<FunctionValue<'ctx>>,
    field_add_fn: Option<FunctionValue<'ctx>>,
    field_sub_fn: Option<FunctionValue<'ctx>>,
    field_div_fn: Option<FunctionValue<'ctx>>,
    malloc_fn: Option<FunctionValue<'ctx>>,
    free_fn: Option<FunctionValue<'ctx>>,
    write_witness_fn: Option<FunctionValue<'ctx>>,
    write_a_fn: Option<FunctionValue<'ctx>>,
    write_b_fn: Option<FunctionValue<'ctx>>,
    write_c_fn: Option<FunctionValue<'ctx>>,
    // AD runtime functions
    ad_next_d_coeff_fn: Option<FunctionValue<'ctx>>,
    ad_fresh_witness_index_fn: Option<FunctionValue<'ctx>>,
    ad_accum_da_fn: Option<FunctionValue<'ctx>>,
    ad_accum_db_fn: Option<FunctionValue<'ctx>>,
    ad_accum_dc_fn: Option<FunctionValue<'ctx>>,
    ad_accum_at_da_fn: Option<FunctionValue<'ctx>>,
    ad_accum_at_db_fn: Option<FunctionValue<'ctx>>,
    ad_accum_at_dc_fn: Option<FunctionValue<'ctx>>,
    field_from_limbs_fn: Option<FunctionValue<'ctx>>,
    field_to_limbs_fn: Option<FunctionValue<'ctx>>,
    // Globals
    globals: Vec<inkwell::values::GlobalValue<'ctx>>,
}

impl<'ctx> LLVMCodeGen<'ctx> {
    pub fn new(context: &'ctx Context, module_name: &str) -> Self {
        let module = context.create_module(module_name);
        let builder = context.create_builder();

        let mut codegen = Self {
            context,
            module,
            builder,
            value_map: HashMap::new(),
            block_map: HashMap::new(),
            function_map: HashMap::new(),
            vm_ptr: None,
            field_mul_fn: None,
            field_add_fn: None,
            field_sub_fn: None,
            field_div_fn: None,
            malloc_fn: None,
            free_fn: None,
            write_witness_fn: None,
            write_a_fn: None,
            write_b_fn: None,
            write_c_fn: None,
            ad_next_d_coeff_fn: None,
            ad_fresh_witness_index_fn: None,
            ad_accum_da_fn: None,
            ad_accum_db_fn: None,
            ad_accum_dc_fn: None,
            ad_accum_at_da_fn: None,
            ad_accum_at_db_fn: None,
            ad_accum_at_dc_fn: None,
            field_from_limbs_fn: None,
            field_to_limbs_fn: None,
            globals: Vec::new(),
        };

        codegen.declare_runtime_functions();
        codegen.define_write_functions();
        codegen
    }

    // ── Type conversion ─────────────────────────────────────────────────

    /// Convert an LLType to the corresponding LLVM type.
    fn convert_type(&self, ty: &LLType) -> BasicTypeEnum<'ctx> {
        match ty {
            LLType::Int(bits) => self
                .context
                .custom_width_int_type(
                    NonZeroU32::new(*bits).expect("Cannot have zero-width integer"),
                )
                .expect("A basic integer type can be created")
                .into(),
            LLType::Ptr => self.context.ptr_type(AddressSpace::default()).into(),
            LLType::Struct(s) => self.convert_struct_type(s),
        }
    }

    /// Convert an LLStruct to an LLVM struct type.
    fn convert_struct_type(&self, s: &LLStruct) -> BasicTypeEnum<'ctx> {
        let fields: Vec<BasicTypeEnum<'ctx>> = s
            .fields
            .iter()
            .map(|f| self.convert_field_type(f))
            .collect();
        self.context.struct_type(&fields, false).into()
    }

    /// Convert an LLFieldType to the corresponding LLVM type.
    fn convert_field_type(&self, ft: &LLFieldType) -> BasicTypeEnum<'ctx> {
        match ft {
            LLFieldType::Int(bits) => self
                .context
                .custom_width_int_type(
                    NonZeroU32::new(*bits).expect("Cannot have zero-width integer"),
                )
                .expect("A basic integer type can be created")
                .into(),
            LLFieldType::Ptr => self.context.ptr_type(AddressSpace::default()).into(),
            LLFieldType::Inline(s) => self.convert_struct_type(s),
            LLFieldType::InlineArray(s, n) => {
                let elem = self.convert_struct_type(s);
                elem.array_type(*n as u32).into()
            }
            LLFieldType::FlexArray(_) => {
                panic!("FlexArray is not supported in LLVM codegen")
            }
        }
    }

    /// The LLVM type for a field element, derived from `LLStruct::field_elem()`.
    fn field_llvm_type(&self) -> BasicTypeEnum<'ctx> {
        self.convert_struct_type(&LLStruct::field_elem())
    }

    /// The LLVM type for raw (non-Montgomery) limbs, derived from `LLStruct::field_elem()`.
    fn limbs_llvm_type(&self) -> BasicTypeEnum<'ctx> {
        self.convert_struct_type(&LLStruct::limbs())
    }

    // ── Runtime functions ───────────────────────────────────────────────

    fn declare_runtime_functions(&mut self) {
        let field_type = self.field_llvm_type();
        let ptr_type = self.context.ptr_type(AddressSpace::default());
        let i32_type = self.context.i32_type();
        let void_type = self.context.void_type();
        let limbs_type = self.limbs_llvm_type();

        // __field_mul(FieldElem, FieldElem) -> FieldElem
        let field_mul_type = field_type.fn_type(&[field_type.into(), field_type.into()], false);
        self.field_mul_fn = Some(
            self.module
                .add_function("__field_mul", field_mul_type, None),
        );

        // malloc(i32) -> ptr  (i32 size for wasm32)
        let malloc_type = ptr_type.fn_type(&[i32_type.into()], false);
        self.malloc_fn = Some(self.module.add_function(
            "malloc",
            malloc_type,
            Some(Linkage::External),
        ));

        // free(ptr) -> void
        let free_type = void_type.fn_type(&[ptr_type.into()], false);
        self.free_fn = Some(
            self.module
                .add_function("free", free_type, Some(Linkage::External)),
        );

        // __field_add(FieldElem, FieldElem) -> FieldElem
        let field_add_type = field_type.fn_type(&[field_type.into(), field_type.into()], false);
        self.field_add_fn = Some(
            self.module
                .add_function("__field_add", field_add_type, None),
        );

        // __field_sub(FieldElem, FieldElem) -> FieldElem
        let field_sub_type = field_type.fn_type(&[field_type.into(), field_type.into()], false);
        self.field_sub_fn = Some(
            self.module
                .add_function("__field_sub", field_sub_type, None),
        );

        // __field_div(FieldElem, FieldElem) -> FieldElem
        let field_div_type = field_type.fn_type(&[field_type.into(), field_type.into()], false);
        self.field_div_fn = Some(
            self.module
                .add_function("__field_div", field_div_type, None),
        );

        // __field_from_limbs([4 x i64]) -> FieldElem  (raw limbs → Montgomery)
        let field_from_limbs_type = field_type.fn_type(&[limbs_type.into()], false);
        self.field_from_limbs_fn = Some(self.module.add_function(
            "__field_from_limbs",
            field_from_limbs_type,
            Some(Linkage::External),
        ));

        // __field_to_limbs(FieldElem) -> [4 x i64]  (Montgomery → raw limbs)
        let field_to_limbs_type = limbs_type.fn_type(&[field_type.into()], false);
        self.field_to_limbs_fn = Some(self.module.add_function(
            "__field_to_limbs",
            field_to_limbs_type,
            Some(Linkage::External),
        ));
    }

    fn define_write_functions(&mut self) {
        self.write_witness_fn =
            Some(self.define_write_fn("__write_witness", VM_WITNESS_PTR_OFFSET));
        self.write_a_fn = Some(self.define_write_fn("__write_a", VM_A_PTR_OFFSET));
        self.write_b_fn = Some(self.define_write_fn("__write_b", VM_B_PTR_OFFSET));
        self.write_c_fn = Some(self.define_write_fn("__write_c", VM_C_PTR_OFFSET));
        self.ad_next_d_coeff_fn = Some(self.define_ad_next_d_coeff_fn());
        self.ad_fresh_witness_index_fn = Some(self.define_ad_fresh_witness_index_fn());
        self.ad_accum_da_fn = Some(self.define_ad_accum_fn("__ad_accum_da", AD_OUT_DA_PTR_OFFSET));
        self.ad_accum_db_fn = Some(self.define_ad_accum_fn("__ad_accum_db", AD_OUT_DB_PTR_OFFSET));
        self.ad_accum_dc_fn = Some(self.define_ad_accum_fn("__ad_accum_dc", AD_OUT_DC_PTR_OFFSET));
        self.ad_accum_at_da_fn =
            Some(self.define_ad_accum_at_fn("__ad_accum_at_da", AD_OUT_DA_PTR_OFFSET));
        self.ad_accum_at_db_fn =
            Some(self.define_ad_accum_at_fn("__ad_accum_at_db", AD_OUT_DB_PTR_OFFSET));
        self.ad_accum_at_dc_fn =
            Some(self.define_ad_accum_at_fn("__ad_accum_at_dc", AD_OUT_DC_PTR_OFFSET));
    }

    fn define_write_fn(&self, name: &str, ptr_offset: u32) -> FunctionValue<'ctx> {
        let void_type = self.context.void_type();
        let ptr_type = self.context.ptr_type(AddressSpace::default());
        let field_type = self.field_llvm_type();
        let i32_type = self.context.i32_type();

        let fn_type = void_type.fn_type(&[ptr_type.into(), field_type.into()], false);
        let function = self
            .module
            .add_function(name, fn_type, Some(Linkage::Internal));

        let entry = self.context.append_basic_block(function, "entry");
        let builder = self.context.create_builder();
        builder.position_at_end(entry);

        let vm_ptr = function.get_nth_param(0).unwrap().into_pointer_value();
        let value = function.get_nth_param(1).unwrap().into_struct_value();

        let write_pos_ptr = unsafe {
            builder
                .build_gep(
                    ptr_type,
                    vm_ptr,
                    &[i32_type.const_int((ptr_offset / 4) as u64, false)],
                    "pos_ptr",
                )
                .unwrap()
        };

        let write_ptr = builder
            .build_load(ptr_type, write_pos_ptr, "ptr")
            .unwrap()
            .into_pointer_value();

        builder.build_store(write_ptr, value).unwrap();

        let new_ptr = unsafe {
            builder
                .build_gep(
                    field_type,
                    write_ptr,
                    &[i32_type.const_int(1, false)],
                    "new_ptr",
                )
                .unwrap()
        };

        builder.build_store(write_pos_ptr, new_ptr).unwrap();
        builder.build_return(None).unwrap();

        function
    }

    /// Load the Field element that `ad_coeffs` currently points at, advance
    /// `ad_coeffs` by one Field, return the loaded value.
    fn define_ad_next_d_coeff_fn(&self) -> FunctionValue<'ctx> {
        let ptr_type = self.context.ptr_type(AddressSpace::default());
        let i32_type = self.context.i32_type();
        let field_type = self.field_llvm_type();

        let fn_type = field_type.fn_type(&[ptr_type.into()], false);
        let function =
            self.module
                .add_function("__ad_next_d_coeff", fn_type, Some(Linkage::Internal));

        let entry = self.context.append_basic_block(function, "entry");
        let builder = self.context.create_builder();
        builder.position_at_end(entry);

        let vm_ptr = function.get_nth_param(0).unwrap().into_pointer_value();

        let coeffs_slot_ptr = unsafe {
            builder
                .build_gep(
                    ptr_type,
                    vm_ptr,
                    &[i32_type.const_int((AD_COEFFS_PTR_OFFSET / WASM_PTR_SIZE) as u64, false)],
                    "coeffs_slot",
                )
                .unwrap()
        };

        let coeffs_ptr = builder
            .build_load(ptr_type, coeffs_slot_ptr, "coeffs_ptr")
            .unwrap()
            .into_pointer_value();

        let value = builder
            .build_load(field_type, coeffs_ptr, "d_coeff")
            .unwrap();

        let next_coeffs_ptr = unsafe {
            builder
                .build_gep(
                    field_type,
                    coeffs_ptr,
                    &[i32_type.const_int(1, false)],
                    "next_coeffs_ptr",
                )
                .unwrap()
        };
        builder
            .build_store(coeffs_slot_ptr, next_coeffs_ptr)
            .unwrap();

        builder.build_return(Some(&value)).unwrap();

        function
    }

    /// Return the current witness counter, post-increment it.
    fn define_ad_fresh_witness_index_fn(&self) -> FunctionValue<'ctx> {
        let ptr_type = self.context.ptr_type(AddressSpace::default());
        let i32_type = self.context.i32_type();

        let fn_type = i32_type.fn_type(&[ptr_type.into()], false);
        let function =
            self.module
                .add_function("__ad_fresh_witness_index", fn_type, Some(Linkage::Internal));

        let entry = self.context.append_basic_block(function, "entry");
        let builder = self.context.create_builder();
        builder.position_at_end(entry);

        let vm_ptr = function.get_nth_param(0).unwrap().into_pointer_value();

        let wit_off_ptr = unsafe {
            builder
                .build_gep(
                    i32_type,
                    vm_ptr,
                    &[i32_type
                        .const_int((AD_CURRENT_WIT_OFF_OFFSET / WASM_PTR_SIZE) as u64, false)],
                    "wit_off_slot",
                )
                .unwrap()
        };

        let index = builder
            .build_load(i32_type, wit_off_ptr, "wit_idx")
            .unwrap()
            .into_int_value();
        let next_index = builder
            .build_int_add(index, i32_type.const_int(1, false), "next_wit_idx")
            .unwrap();
        builder.build_store(wit_off_ptr, next_index).unwrap();

        builder.build_return(Some(&index)).unwrap();

        function
    }

    /// Load `out_d{matrix}` from the VM struct, read the field at position 0,
    /// add `sensitivity * const_value`, write it back. Pointer is NOT advanced.
    fn define_ad_accum_fn(&self, name: &str, out_d_offset: u32) -> FunctionValue<'ctx> {
        let void_type = self.context.void_type();
        let ptr_type = self.context.ptr_type(AddressSpace::default());
        let i32_type = self.context.i32_type();
        let field_type = self.field_llvm_type();

        let fn_type = void_type.fn_type(
            &[ptr_type.into(), field_type.into(), field_type.into()],
            false,
        );
        let function = self
            .module
            .add_function(name, fn_type, Some(Linkage::Internal));

        let entry = self.context.append_basic_block(function, "entry");
        let builder = self.context.create_builder();
        builder.position_at_end(entry);

        let vm_ptr = function.get_nth_param(0).unwrap().into_pointer_value();
        let const_value = function.get_nth_param(1).unwrap();
        let sensitivity = function.get_nth_param(2).unwrap();

        let out_d_slot = unsafe {
            builder
                .build_gep(
                    ptr_type,
                    vm_ptr,
                    &[i32_type.const_int((out_d_offset / WASM_PTR_SIZE) as u64, false)],
                    "out_d_slot",
                )
                .unwrap()
        };
        let out_d_ptr = builder
            .build_load(ptr_type, out_d_slot, "out_d_ptr")
            .unwrap()
            .into_pointer_value();

        let product = {
            let mul_fn = self.field_mul_fn.expect("__field_mul not declared");
            let call = builder
                .build_call(
                    mul_fn,
                    &[const_value.into(), sensitivity.into()],
                    "accum_product",
                )
                .unwrap();
            call.try_as_basic_value()
                .expect_basic("__field_mul should return a value")
        };

        let old = builder
            .build_load(field_type, out_d_ptr, "accum_old")
            .unwrap();
        let new_val = {
            let add_fn = self.field_add_fn.expect("__field_add not declared");
            let call = builder
                .build_call(add_fn, &[old.into(), product.into()], "accum_new")
                .unwrap();
            call.try_as_basic_value()
                .expect_basic("__field_add should return a value")
        };
        builder.build_store(out_d_ptr, new_val).unwrap();
        builder.build_return(None).unwrap();

        function
    }

    /// Load `out_d{matrix}` from the VM struct, read the field at position
    /// `witness_index`, add `sensitivity`, write it back.
    fn define_ad_accum_at_fn(&self, name: &str, out_d_offset: u32) -> FunctionValue<'ctx> {
        let void_type = self.context.void_type();
        let ptr_type = self.context.ptr_type(AddressSpace::default());
        let i32_type = self.context.i32_type();
        let field_type = self.field_llvm_type();

        let fn_type = void_type.fn_type(
            &[ptr_type.into(), i32_type.into(), field_type.into()],
            false,
        );
        let function = self
            .module
            .add_function(name, fn_type, Some(Linkage::Internal));

        let entry = self.context.append_basic_block(function, "entry");
        let builder = self.context.create_builder();
        builder.position_at_end(entry);

        let vm_ptr = function.get_nth_param(0).unwrap().into_pointer_value();
        let index = function.get_nth_param(1).unwrap().into_int_value();
        let sensitivity = function.get_nth_param(2).unwrap();

        let out_d_slot = unsafe {
            builder
                .build_gep(
                    ptr_type,
                    vm_ptr,
                    &[i32_type.const_int((out_d_offset / WASM_PTR_SIZE) as u64, false)],
                    "out_d_slot",
                )
                .unwrap()
        };
        let out_d_ptr = builder
            .build_load(ptr_type, out_d_slot, "out_d_ptr")
            .unwrap()
            .into_pointer_value();

        let target_ptr = unsafe {
            builder
                .build_gep(field_type, out_d_ptr, &[index], "target_ptr")
                .unwrap()
        };
        let old = builder
            .build_load(field_type, target_ptr, "accum_old")
            .unwrap();
        let new_val = {
            let add_fn = self.field_add_fn.expect("__field_add not declared");
            let call = builder
                .build_call(add_fn, &[old.into(), sensitivity.into()], "accum_new")
                .unwrap();
            call.try_as_basic_value()
                .expect_basic("__field_add should return a value")
        };
        builder.build_store(target_ptr, new_val).unwrap();
        builder.build_return(None).unwrap();

        function
    }

    // ── Compilation entry point ─────────────────────────────────────────

    /// Compile LLSSA to LLVM IR.
    pub fn compile(&mut self, llssa: &LLSSA, flow_analysis: &FlowAnalysis) {
        let main_id = llssa.get_main_id();

        // Declare globals
        for (i, ty) in llssa.get_global_types().iter().enumerate() {
            let llvm_ty = self.convert_type(ty);
            let global = self.module.add_global(
                llvm_ty,
                Some(AddressSpace::default()),
                &format!("__mavros_global_{}", i),
            );
            global.set_initializer(&llvm_ty.const_zero());
            self.globals.push(global);
        }

        // First pass: declare all functions
        for (fn_id, function) in llssa.iter_functions() {
            self.declare_function(*fn_id, function, main_id);
        }

        // Second pass: generate function bodies
        for (fn_id, function) in llssa.iter_functions() {
            let cfg = flow_analysis.get_function_cfg(*fn_id);
            self.compile_function(*fn_id, function, cfg);
        }
    }

    fn declare_function(&mut self, fn_id: FunctionId, function: &LLFunction, main_id: FunctionId) {
        let entry = function.get_entry();
        let ptr_type = self.context.ptr_type(AddressSpace::default());

        // First parameter is always VM*
        let mut param_types: Vec<BasicMetadataTypeEnum> = vec![ptr_type.into()];

        for (_, tp) in entry.get_parameters() {
            param_types.push(self.convert_type(tp).into());
        }

        let return_types: Vec<BasicTypeEnum> = function
            .get_returns()
            .iter()
            .map(|tp| self.convert_type(tp))
            .collect();

        let fn_type = if return_types.is_empty() {
            self.context.void_type().fn_type(&param_types, false)
        } else if return_types.len() == 1 {
            return_types[0].fn_type(&param_types, false)
        } else {
            let return_struct = self.context.struct_type(&return_types, false);
            return_struct.fn_type(&param_types, false)
        };

        let name = if fn_id == main_id {
            "mavros_main"
        } else {
            function.get_name()
        };

        let fn_value = self.module.add_function(name, fn_type, None);
        self.function_map.insert(fn_id, fn_value);
    }

    fn compile_function(
        &mut self,
        fn_id: FunctionId,
        function: &LLFunction,
        cfg: &crate::compiler::flow_analysis::CFG,
    ) {
        self.value_map.clear();
        self.block_map.clear();

        let fn_value = self.function_map[&fn_id];
        let entry_block_id = function.get_entry_id();

        // Create entry block
        let entry_bb = self
            .context
            .append_basic_block(fn_value, &format!("block_{}", entry_block_id.0));
        self.block_map.insert(entry_block_id, entry_bb);

        // Create remaining blocks
        for (block_id, _) in function.get_blocks() {
            if *block_id != entry_block_id {
                let bb = self
                    .context
                    .append_basic_block(fn_value, &format!("block_{}", block_id.0));
                self.block_map.insert(*block_id, bb);
            }
        }

        // Map entry block parameters to LLVM function arguments
        self.builder.position_at_end(entry_bb);
        self.vm_ptr = Some(fn_value.get_nth_param(0).unwrap().into_pointer_value());

        let entry = function.get_entry();
        for (i, (param_id, _)) in entry.get_parameters().enumerate() {
            let param_value = fn_value.get_nth_param((i + 1) as u32).unwrap();
            self.value_map.insert(*param_id, param_value);
        }

        // Track phi nodes
        let mut phi_nodes: HashMap<(BlockId, usize), inkwell::values::PhiValue<'ctx>> =
            HashMap::new();

        // Generate code in dominator order
        for block_id in cfg.get_domination_pre_order() {
            self.compile_block(function, block_id, &mut phi_nodes);
        }

        // Wire phi incoming values
        for (block_id, block) in function.get_blocks() {
            if let Some(terminator) = block.get_terminator() {
                let current_bb = self.block_map[block_id];
                match terminator {
                    Terminator::Jmp(target_id, args) => {
                        for (i, arg_id) in args.iter().enumerate() {
                            if let Some(phi) = phi_nodes.get(&(*target_id, i)) {
                                if let Some(arg_val) = self.value_map.get(arg_id) {
                                    phi.add_incoming(&[(arg_val, current_bb)]);
                                }
                            }
                        }
                    }
                    Terminator::JmpIf(..) | Terminator::Return(_) => {}
                }
            }
        }
    }

    fn compile_block(
        &mut self,
        function: &LLFunction,
        block_id: BlockId,
        phi_nodes: &mut HashMap<(BlockId, usize), inkwell::values::PhiValue<'ctx>>,
    ) {
        let block = function.get_block(block_id);
        let bb = self.block_map[&block_id];

        // Non-entry block parameters → phi nodes
        if block_id != function.get_entry_id() {
            self.builder.position_at_end(bb);

            for (i, (param_id, param_type)) in block.get_parameters().enumerate() {
                let llvm_type = self.convert_type(param_type);
                let phi = self
                    .builder
                    .build_phi(llvm_type, &format!("v{}", param_id.0))
                    .unwrap();
                self.value_map.insert(*param_id, phi.as_basic_value());
                phi_nodes.insert((block_id, i), phi);
            }
        }

        self.builder.position_at_end(bb);

        for instruction in block.get_instructions() {
            self.compile_instruction(instruction);
        }

        if let Some(terminator) = block.get_terminator() {
            self.compile_terminator(terminator);
        }
    }

    // ── Instruction compilation ─────────────────────────────────────────

    fn compile_instruction(&mut self, op: &LLOp) {
        match op {
            LLOp::IntConst {
                result,
                bits,
                value,
            } => {
                let int_type = self
                    .context
                    .custom_width_int_type(
                        NonZeroU32::new(*bits).expect("Cannot have zero-width integer"),
                    )
                    .expect("A basic integer type can be created");
                let val = int_type.const_int(*value, false);
                self.value_map.insert(*result, val.into());
            }

            LLOp::IntArith { kind, result, a, b } => {
                let lhs = self.value_map[a].into_int_value();
                let rhs = self.value_map[b].into_int_value();
                let name = &format!("v{}", result.0);

                let val = match kind {
                    IntArithOp::Add => self.builder.build_int_add(lhs, rhs, name).unwrap(),
                    IntArithOp::Sub => self.builder.build_int_sub(lhs, rhs, name).unwrap(),
                    IntArithOp::Mul => self.builder.build_int_mul(lhs, rhs, name).unwrap(),
                    IntArithOp::UDiv => {
                        self.builder.build_int_unsigned_div(lhs, rhs, name).unwrap()
                    }
                    IntArithOp::URem => {
                        self.builder.build_int_unsigned_rem(lhs, rhs, name).unwrap()
                    }
                    IntArithOp::And => self.builder.build_and(lhs, rhs, name).unwrap(),
                    IntArithOp::Or => self.builder.build_or(lhs, rhs, name).unwrap(),
                    IntArithOp::Xor => self.builder.build_xor(lhs, rhs, name).unwrap(),
                    IntArithOp::Shl => self.builder.build_left_shift(lhs, rhs, name).unwrap(),
                    IntArithOp::UShr => self
                        .builder
                        .build_right_shift(lhs, rhs, false, name)
                        .unwrap(),
                    _ => panic!("Unsupported IntArithOp in LLSSA codegen: {:?}", kind),
                };
                self.value_map.insert(*result, val.into());
            }

            LLOp::IntCmp { kind, result, a, b } => {
                let lhs = self.value_map[a].into_int_value();
                let rhs = self.value_map[b].into_int_value();
                let predicate = match kind {
                    IntCmpOp::Eq => IntPredicate::EQ,
                    IntCmpOp::ULt => IntPredicate::ULT,
                    IntCmpOp::SLt => IntPredicate::SLT,
                };
                let val = self
                    .builder
                    .build_int_compare(predicate, lhs, rhs, &format!("v{}", result.0))
                    .unwrap();
                self.value_map.insert(*result, val.into());
            }

            LLOp::Not { result, value } => {
                let val = self.value_map[value].into_int_value();
                let not_val = self
                    .builder
                    .build_not(val, &format!("v{}", result.0))
                    .unwrap();
                self.value_map.insert(*result, not_val.into());
            }

            LLOp::FieldArith { kind, result, a, b } => {
                let lhs = self.value_map[a];
                let rhs = self.value_map[b];

                let val = match kind {
                    FieldArithOp::Mul => {
                        let mul_fn = self.field_mul_fn.expect("__field_mul not declared");
                        let call_site = self
                            .builder
                            .build_call(mul_fn, &[lhs.into(), rhs.into()], "field_mul")
                            .unwrap();
                        call_site
                            .try_as_basic_value()
                            .expect_basic("field_mul should return a value")
                    }
                    FieldArithOp::Add => {
                        let add_fn = self.field_add_fn.expect("__field_add not declared");
                        let call_site = self
                            .builder
                            .build_call(add_fn, &[lhs.into(), rhs.into()], "field_add")
                            .unwrap();
                        call_site
                            .try_as_basic_value()
                            .expect_basic("field_add should return a value")
                    }
                    FieldArithOp::Sub => {
                        let sub_fn = self.field_sub_fn.expect("__field_sub not declared");
                        let call_site = self
                            .builder
                            .build_call(sub_fn, &[lhs.into(), rhs.into()], "field_sub")
                            .unwrap();
                        call_site
                            .try_as_basic_value()
                            .expect_basic("field_sub should return a value")
                    }
                    FieldArithOp::Div => {
                        let div_fn = self.field_div_fn.expect("__field_div not declared");
                        let call_site = self
                            .builder
                            .build_call(div_fn, &[lhs.into(), rhs.into()], "field_div")
                            .unwrap();
                        call_site
                            .try_as_basic_value()
                            .expect_basic("field_div should return a value")
                    }
                    _ => panic!("Unsupported FieldArithOp in LLSSA codegen: {:?}", kind),
                };
                self.value_map.insert(*result, val);
            }

            LLOp::MkStruct {
                result,
                struct_type,
                fields,
            } => {
                let llvm_type = self.convert_struct_type(struct_type).into_struct_type();
                let mut agg = llvm_type.get_undef();
                for (i, field_id) in fields.iter().enumerate() {
                    let field_val = self.value_map[field_id];
                    agg = self
                        .builder
                        .build_insert_value(agg, field_val, i as u32, "mk")
                        .unwrap()
                        .into_struct_value();
                }
                self.value_map.insert(*result, agg.into());
            }

            LLOp::ExtractField {
                result,
                value,
                struct_type: _,
                field,
            } => {
                let agg = self.value_map[value].into_struct_value();
                let val = self
                    .builder
                    .build_extract_value(agg, *field as u32, &format!("v{}", result.0))
                    .unwrap();
                self.value_map.insert(*result, val);
            }

            LLOp::Select {
                result,
                cond,
                if_t,
                if_f,
            } => {
                let c = self.value_map[cond].into_int_value();
                let t = self.value_map[if_t];
                let f = self.value_map[if_f];
                let val = self
                    .builder
                    .build_select(c, t, f, &format!("v{}", result.0))
                    .unwrap();
                self.value_map.insert(*result, val);
            }

            LLOp::Constrain { a, b, c } => {
                let a_val = self.value_map[a];
                let b_val = self.value_map[b];
                let c_val = self.value_map[c];
                let vm_ptr = self.vm_ptr.unwrap();

                self.call_write_fn(self.write_a_fn.unwrap(), vm_ptr, a_val);
                self.call_write_fn(self.write_b_fn.unwrap(), vm_ptr, b_val);
                self.call_write_fn(self.write_c_fn.unwrap(), vm_ptr, c_val);
            }

            LLOp::WriteWitness { value } => {
                let val = self.value_map[value];
                let vm_ptr = self.vm_ptr.unwrap();
                self.call_write_fn(self.write_witness_fn.unwrap(), vm_ptr, val);
            }

            LLOp::Truncate {
                result,
                value,
                to_bits,
            } => {
                let val = self.value_map[value].into_int_value();
                let target_type = self
                    .context
                    .custom_width_int_type(
                        NonZeroU32::new(*to_bits).expect("Cannot have zero-width integer"),
                    )
                    .expect("The target type for truncation is valid");
                let truncated = self
                    .builder
                    .build_int_truncate(val, target_type, &format!("v{}", result.0))
                    .unwrap();
                self.value_map.insert(*result, truncated.into());
            }

            LLOp::ZExt {
                result,
                value,
                to_bits,
            } => {
                let val = self.value_map[value].into_int_value();
                let target_type = self
                    .context
                    .custom_width_int_type(
                        NonZeroU32::new(*to_bits).expect("Cannot have zero-width integer"),
                    )
                    .expect("The target type for zero-extension is valid");
                let extended = self
                    .builder
                    .build_int_z_extend(val, target_type, &format!("v{}", result.0))
                    .unwrap();
                self.value_map.insert(*result, extended.into());
            }

            LLOp::FieldEq { result, a, b } => {
                // Field equality: compare all 4 limbs
                let a_val = self.value_map[a].into_struct_value();
                let b_val = self.value_map[b].into_struct_value();
                let mut eq_acc = self.context.bool_type().const_int(1, false);
                for i in 0..4u32 {
                    let a_limb = self
                        .builder
                        .build_extract_value(a_val, i, "a_l")
                        .unwrap()
                        .into_int_value();
                    let b_limb = self
                        .builder
                        .build_extract_value(b_val, i, "b_l")
                        .unwrap()
                        .into_int_value();
                    let limb_eq = self
                        .builder
                        .build_int_compare(IntPredicate::EQ, a_limb, b_limb, "leq")
                        .unwrap();
                    eq_acc = self.builder.build_and(eq_acc, limb_eq, "eq").unwrap();
                }
                self.value_map.insert(*result, eq_acc.into());
            }

            LLOp::FieldFromLimbs { result, limbs } => {
                // Convert raw limbs (non-Montgomery) to Montgomery form via __field_from_limbs.
                let limb_vals = self.value_map[limbs];
                let from_fn = self
                    .field_from_limbs_fn
                    .expect("__field_from_limbs not declared");
                let call = self
                    .builder
                    .build_call(from_fn, &[limb_vals.into()], "from_limbs")
                    .unwrap();
                let field = call
                    .try_as_basic_value()
                    .expect_basic("__field_from_limbs should return a value");
                self.value_map.insert(*result, field);
            }

            LLOp::FieldToLimbs { result, src } => {
                // Convert Montgomery form to raw limbs via __field_to_limbs.
                let field = self.value_map[src];
                let to_fn = self
                    .field_to_limbs_fn
                    .expect("__field_to_limbs not declared");
                let call = self
                    .builder
                    .build_call(to_fn, &[field.into()], "to_limbs")
                    .unwrap();
                let limb_vals = call
                    .try_as_basic_value()
                    .expect_basic("__field_to_limbs should return a value");
                self.value_map.insert(*result, limb_vals);
            }

            // ── Memory operations ───────────────────────────────────────
            LLOp::NullPtr { result } => {
                let ptr_type = self.context.ptr_type(AddressSpace::default());
                let null = ptr_type.const_null();
                self.value_map.insert(*result, null.into());
            }

            LLOp::HeapAlloc {
                result,
                struct_type,
                flex_count: _,
            } => {
                let struct_ty = self.convert_struct_type(struct_type);
                let size = struct_ty.size_of().unwrap();
                let i32_type = self.context.i32_type();
                let size_i32 = self
                    .builder
                    .build_int_truncate_or_bit_cast(size, i32_type, "size")
                    .unwrap();
                let malloc_fn = self.malloc_fn.expect("malloc not declared");
                let call_site = self
                    .builder
                    .build_call(malloc_fn, &[size_i32.into()], "alloc")
                    .unwrap();
                let ptr_val = call_site
                    .try_as_basic_value()
                    .expect_basic("malloc should return a value");
                self.value_map.insert(*result, ptr_val);
            }

            LLOp::Free { ptr } => {
                let p = self.value_map[ptr].into_pointer_value();
                let free_fn = self.free_fn.expect("free not declared");
                self.builder.build_call(free_fn, &[p.into()], "").unwrap();
            }

            LLOp::Load { result, ptr, ty } => {
                let p = self.value_map[ptr].into_pointer_value();
                let llvm_ty = self.convert_type(ty);
                let val = self
                    .builder
                    .build_load(llvm_ty, p, &format!("v{}", result.0))
                    .unwrap();
                self.value_map.insert(*result, val);
            }

            LLOp::Store { ptr, value } => {
                let p = self.value_map[ptr].into_pointer_value();
                let v = self.value_map[value];
                self.builder.build_store(p, v).unwrap();
            }

            LLOp::StructFieldPtr {
                result,
                ptr,
                struct_type,
                field,
            } => {
                let p = self.value_map[ptr].into_pointer_value();
                let llvm_struct_ty = self.convert_struct_type(struct_type).into_struct_type();
                let gep = unsafe {
                    self.builder
                        .build_struct_gep(llvm_struct_ty, p, *field as u32, "sfp")
                        .unwrap()
                };
                self.value_map.insert(*result, gep.into());
            }

            LLOp::ArrayElemPtr {
                result,
                ptr,
                elem_type,
                index,
            } => {
                let p = self.value_map[ptr].into_pointer_value();
                let idx = self.value_map[index].into_int_value();
                let llvm_elem_ty = self.convert_struct_type(elem_type);
                let gep = unsafe {
                    self.builder
                        .build_gep(llvm_elem_ty, p, &[idx], "aep")
                        .unwrap()
                };
                self.value_map.insert(*result, gep.into());
            }

            LLOp::Memcpy {
                dst,
                src,
                struct_type,
                count,
            } => {
                let dst_ptr = self.value_map[dst].into_pointer_value();
                let src_ptr = self.value_map[src].into_pointer_value();
                let elem_ty = self.convert_struct_type(struct_type);
                let elem_size = elem_ty.size_of().unwrap();
                let total_size = if let Some(count_val) = count {
                    let cnt = self.value_map[count_val].into_int_value();
                    let cnt_ext = self
                        .builder
                        .build_int_z_extend_or_bit_cast(cnt, elem_size.get_type(), "cnt_ext")
                        .unwrap();
                    self.builder
                        .build_int_mul(elem_size, cnt_ext, "total_size")
                        .unwrap()
                } else {
                    elem_size
                };
                let i32_type = self.context.i32_type();
                let total_i32 = self
                    .builder
                    .build_int_truncate_or_bit_cast(total_size, i32_type, "memcpy_size")
                    .unwrap();
                self.builder
                    .build_memcpy(dst_ptr, 1, src_ptr, 1, total_i32)
                    .unwrap();
            }

            LLOp::Trap => {
                let trap_fn = self.module.get_function("llvm.trap").unwrap_or_else(|| {
                    let void_type = self.context.void_type();
                    let trap_type = void_type.fn_type(&[], false);
                    self.module.add_function("llvm.trap", trap_type, None)
                });
                self.builder.build_call(trap_fn, &[], "").unwrap();
                self.builder.build_unreachable().unwrap();
            }

            // ── Calls ───────────────────────────────────────────────────
            LLOp::Call {
                results,
                func,
                args,
            } => {
                let callee = self.function_map[func];
                let vm_ptr = self.vm_ptr.unwrap();

                let mut call_args: Vec<BasicMetadataValueEnum> = vec![vm_ptr.into()];
                for arg in args {
                    call_args.push(self.value_map[arg].into());
                }

                let call_result = self
                    .builder
                    .build_call(callee, &call_args, &format!("call_f{}", func.0))
                    .unwrap();

                if results.len() == 1 {
                    if let Some(val) = call_result.try_as_basic_value().basic() {
                        self.value_map.insert(results[0], val);
                    }
                } else if results.len() > 1 {
                    let ret_struct = call_result
                        .try_as_basic_value()
                        .expect_basic("Expected return value from multi-return call");
                    for (i, result_id) in results.iter().enumerate() {
                        let val = self
                            .builder
                            .build_extract_value(
                                ret_struct.into_struct_value(),
                                i as u32,
                                &format!("v{}", result_id.0),
                            )
                            .unwrap();
                        self.value_map.insert(*result_id, val.into());
                    }
                }
            }

            // ── AD operations ──────────────────────────────────────────
            LLOp::NextDCoeff { result } => {
                let vm_ptr = self.vm_ptr.unwrap();
                let ad_fn = self
                    .ad_next_d_coeff_fn
                    .expect("__ad_next_d_coeff not declared");
                let call_result = self
                    .builder
                    .build_call(ad_fn, &[vm_ptr.into()], "d_coeff")
                    .unwrap();
                let val = call_result
                    .try_as_basic_value()
                    .expect_basic("__ad_next_d_coeff should return a value");
                self.value_map.insert(*result, val);
            }

            LLOp::ADFreshWitness { result } => {
                let vm_ptr = self.vm_ptr.unwrap();
                let ad_fn = self
                    .ad_fresh_witness_index_fn
                    .expect("__ad_fresh_witness_index not declared");
                let call_result = self
                    .builder
                    .build_call(ad_fn, &[vm_ptr.into()], "wit_idx")
                    .unwrap();
                let val = call_result
                    .try_as_basic_value()
                    .expect_basic("__ad_fresh_witness_index should return a value");
                self.value_map.insert(*result, val);
            }

            LLOp::ADWriteConst {
                matrix,
                const_value,
                sensitivity,
            } => {
                let vm_ptr = self.vm_ptr.unwrap();
                let cv = self.value_map[const_value];
                let s = self.value_map[sensitivity];
                let accum_fn = match matrix {
                    DMatrix::A => self.ad_accum_da_fn,
                    DMatrix::B => self.ad_accum_db_fn,
                    DMatrix::C => self.ad_accum_dc_fn,
                }
                .expect("AD accum function not declared");
                self.builder
                    .build_call(accum_fn, &[vm_ptr.into(), cv.into(), s.into()], "")
                    .unwrap();
            }

            LLOp::ADWriteWitness {
                matrix,
                witness_index,
                sensitivity,
            } => {
                let vm_ptr = self.vm_ptr.unwrap();
                let idx = self.value_map[witness_index];
                let s = self.value_map[sensitivity];
                let accum_fn = match matrix {
                    DMatrix::A => self.ad_accum_at_da_fn,
                    DMatrix::B => self.ad_accum_at_db_fn,
                    DMatrix::C => self.ad_accum_at_dc_fn,
                }
                .expect("AD accum_at function not declared");
                self.builder
                    .build_call(accum_fn, &[vm_ptr.into(), idx.into(), s.into()], "")
                    .unwrap();
            }

            LLOp::GlobalAddr { result, global_id } => {
                let global = self.globals[*global_id];
                let ptr = global.as_pointer_value();
                self.value_map.insert(*result, ptr.into());
            }

            _ => panic!("Unsupported LLOp in LLSSA codegen: {:?}", op),
        }
    }

    fn call_write_fn(
        &self,
        write_fn: FunctionValue<'ctx>,
        vm_ptr: PointerValue<'ctx>,
        value: BasicValueEnum<'ctx>,
    ) {
        self.builder
            .build_call(write_fn, &[vm_ptr.into(), value.into()], "")
            .unwrap();
    }

    // ── Terminator compilation ──────────────────────────────────────────

    fn compile_terminator(&mut self, terminator: &Terminator) {
        match terminator {
            Terminator::Jmp(target_id, _args) => {
                let target_bb = self.block_map[target_id];
                self.builder.build_unconditional_branch(target_bb).unwrap();
            }
            Terminator::JmpIf(cond, true_target, false_target) => {
                let cond_val = self.value_map[cond].into_int_value();
                let true_bb = self.block_map[true_target];
                let false_bb = self.block_map[false_target];
                self.builder
                    .build_conditional_branch(cond_val, true_bb, false_bb)
                    .unwrap();
            }
            Terminator::Return(values) => {
                if values.is_empty() {
                    self.builder.build_return(None).unwrap();
                } else if values.len() == 1 {
                    let ret_val = self.value_map[&values[0]];
                    self.builder.build_return(Some(&ret_val)).unwrap();
                } else {
                    let ret_values: Vec<BasicValueEnum> =
                        values.iter().map(|v| self.value_map[v]).collect();
                    let ret_types: Vec<BasicTypeEnum> =
                        ret_values.iter().map(|v| v.get_type()).collect();
                    let struct_type = self.context.struct_type(&ret_types, false);
                    let mut struct_val = struct_type.get_undef();
                    for (i, val) in ret_values.iter().enumerate() {
                        struct_val = self
                            .builder
                            .build_insert_value(struct_val, *val, i as u32, "ret_pack")
                            .unwrap()
                            .into_struct_value();
                    }
                    self.builder.build_return(Some(&struct_val)).unwrap();
                }
            }
        }
    }

    // ── Output ──────────────────────────────────────────────────────────

    pub fn get_ir(&self) -> String {
        self.module.print_to_string().to_string()
    }

    pub fn write_ir(&self, path: &Path) {
        self.module.print_to_file(path).unwrap();
    }

    pub fn compile_to_wasm(&self, path: &Path, optimization: OptimizationLevel) {
        use std::process::Command;

        Target::initialize_webassembly(&InitializationConfig::default());

        let target_triple = TargetTriple::create("wasm32-unknown-unknown");
        let target = Target::from_triple(&target_triple).unwrap();

        let target_machine = target
            .create_target_machine(
                &target_triple,
                "generic",
                "",
                optimization,
                RelocMode::Default,
                CodeModel::Default,
            )
            .unwrap();

        let obj_path = path.with_extension("o");
        target_machine
            .write_to_file(&self.module, FileType::Object, &obj_path)
            .unwrap();

        let runtime_lib = Self::build_wasm_runtime();

        let wasm_ld = std::env::var("LLVM_SYS_180_PREFIX")
            .map(|prefix| {
                let path = std::path::PathBuf::from(&prefix)
                    .join("bin")
                    .join("wasm-ld");
                if path.exists() {
                    path.to_string_lossy().to_string()
                } else {
                    "wasm-ld".to_string()
                }
            })
            .unwrap_or_else(|_| "wasm-ld".to_string());

        let output = Command::new(&wasm_ld)
            .args([
                "--no-entry",
                "--export=mavros_main",
                "--import-memory",
                "--allow-undefined",
                "--stack-first",
                "-z",
                "stack-size=65536",
                "--export=__data_end",
                "--export=__live_bytes",
                "-o",
            ])
            .arg(path)
            .arg(&obj_path)
            .arg(&runtime_lib)
            .output()
            .unwrap_or_else(|_| {
                panic!(
                    "Failed to run wasm-ld (tried: {}). Make sure LLVM with wasm-ld is installed.",
                    wasm_ld
                )
            });

        if !output.status.success() {
            eprintln!(
                "wasm-ld stdout: {}",
                String::from_utf8_lossy(&output.stdout)
            );
            eprintln!(
                "wasm-ld stderr: {}",
                String::from_utf8_lossy(&output.stderr)
            );
            panic!("wasm-ld failed with status: {}", output.status);
        }

        std::fs::remove_file(&obj_path).ok();
    }

    fn build_wasm_runtime() -> std::path::PathBuf {
        use std::process::Command;

        let metadata = cargo_metadata::MetadataCommand::new()
            .exec()
            .expect("Failed to get cargo metadata");

        let workspace_root = metadata.workspace_root.as_std_path();
        let wasm_runtime_dir = workspace_root.join("wasm-runtime");

        let output = Command::new("cargo")
            .current_dir(&wasm_runtime_dir)
            .args(["build", "--target", "wasm32-unknown-unknown", "--release"])
            .output()
            .expect("Failed to run cargo build for wasm-runtime");

        if !output.status.success() {
            eprintln!(
                "wasm-runtime build stderr: {}",
                String::from_utf8_lossy(&output.stderr)
            );
            panic!("Failed to build wasm-runtime for wasm32");
        }

        let lib_path = workspace_root
            .join("target")
            .join("wasm32-unknown-unknown")
            .join("release")
            .join("libmavros_wasm_runtime.a");

        if !lib_path.exists() {
            panic!("wasm-runtime library not found at {:?}", lib_path);
        }

        lib_path
    }
}
