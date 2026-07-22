//! LLSSA → LLVM Code Generation
//!
//! Translates LLSSA into LLVM IR, which can then be compiled to WebAssembly.
//! Operates on LLSSA + Type — types are explicit in the LLSSA ops, no TypeInfo needed.

use std::{num::NonZeroU32, path::Path};

use inkwell::{
    AddressSpace, IntPredicate, OptimizationLevel,
    builder::Builder,
    context::Context,
    debug_info::{
        AsDIScope, DICompileUnit, DIFile, DIFlags, DIFlagsConstants, DILexicalBlock, DISubprogram,
        DWARFEmissionKind, DWARFSourceLanguage, DebugInfoBuilder,
    },
    module::{FlagBehavior, Linkage, Module},
    targets::{CodeModel, FileType, InitializationConfig, RelocMode, Target, TargetTriple},
    types::{BasicMetadataTypeEnum, BasicType, BasicTypeEnum},
    values::{
        BasicMetadataValueEnum, BasicValueEnum, FunctionValue, IntValue, PointerValue, StructValue,
    },
};

use crate::{
    collections::HashMap,
    compiler::{
        analysis::flow_analysis::{self, FlowAnalysis},
        ssa::{
            BlockId, FunctionId, SSAConstantsSnapshot, SourceLocation, Terminator, ValueId,
            llssa::{
                Blob as LLBlob, Constant, FieldArithOp, IntArithOp, IntCmpOp, LLFieldType,
                LLFunction, LLOp, LLSSA, LLStruct, Type,
            },
        },
    },
};

const WASM_STACK_SIZE_BYTES: u32 = 256 * 1024;

/// How to compile a module to WASM.
#[derive(Clone, Debug)]
pub struct WasmCompileOpts {
    /// LLVM mid-end pass pipeline to run before codegen (e.g. `"default<O1>"`).
    pub midend_pipeline: Option<&'static str>,
    /// Codegen (instruction selection) optimization level.
    pub codegen_level: OptimizationLevel,
    /// Pre-built wasm-runtime static library to link against. Callers are
    /// responsible for building it (see [`crate::wasm_runtime`]); codegen
    /// never invokes cargo.
    pub runtime_lib: std::path::PathBuf,
    /// Strip this prefix from source paths embedded in DWARF.
    pub debug_path_root: Option<std::path::PathBuf>,
    /// Emit DWARF sections into a standalone debug WASM beside the stripped executable.
    pub include_debug_info: bool,
}

impl WasmCompileOpts {
    /// Fast compilation at the cost of output quality. A cheap mid-end
    /// pipeline keeps the module small, then codegen runs at `None`
    /// (FastISel). On large programs this compiles several times faster
    /// than `release()` while producing correct output — the right choice
    /// for tests and CI.
    pub fn fast(runtime_lib: std::path::PathBuf) -> Self {
        Self {
            midend_pipeline: Some("default<O1>"),
            codegen_level: OptimizationLevel::None,
            runtime_lib,
            debug_path_root: None,
            include_debug_info: false,
        }
    }

    /// Optimized output for production artifacts.
    pub fn release(runtime_lib: std::path::PathBuf) -> Self {
        Self {
            midend_pipeline: None,
            codegen_level: OptimizationLevel::Aggressive,
            runtime_lib,
            debug_path_root: None,
            include_debug_info: false,
        }
    }

    pub fn with_debug_path_root(mut self, root: impl Into<std::path::PathBuf>) -> Self {
        self.debug_path_root = Some(root.into());
        self
    }

    pub fn with_debug_info(mut self) -> Self {
        self.include_debug_info = true;
        self
    }
}

fn ll_struct_flex_elem(s: &LLStruct) -> Option<&LLStruct> {
    s.fields.iter().find_map(|field| match field {
        LLFieldType::FlexArray(elem) => Some(elem),
        _ => None,
    })
}

/// LLSSA → LLVM Code Generator
pub struct LLVMCodeGen<'ctx> {
    context: &'ctx Context,
    module: Module<'ctx>,
    builder: Builder<'ctx>,
    value_map: HashMap<ValueId, BasicValueEnum<'ctx>>,
    constants: SSAConstantsSnapshot<Constant>,
    block_map: HashMap<BlockId, inkwell::basic_block::BasicBlock<'ctx>>,
    function_map: HashMap<FunctionId, FunctionValue<'ctx>>,
    vm_ptr: Option<PointerValue<'ctx>>,
    // Runtime function declarations
    field_mul_fn: Option<FunctionValue<'ctx>>,
    field_add_fn: Option<FunctionValue<'ctx>>,
    field_sub_fn: Option<FunctionValue<'ctx>>,
    field_div_fn: Option<FunctionValue<'ctx>>,
    field_lt_fn: Option<FunctionValue<'ctx>>,
    malloc_fn: Option<FunctionValue<'ctx>>,
    free_fn: Option<FunctionValue<'ctx>>,
    field_from_limbs_fn: Option<FunctionValue<'ctx>>,
    field_to_limbs_fn: Option<FunctionValue<'ctx>>,
    // Globals
    globals: Vec<inkwell::values::GlobalValue<'ctx>>,
    const_data_counter: usize,
    /// Exported symbol names of the program's entry points, in entry order.
    entry_symbols: Vec<String>,
    debug_builder: DebugInfoBuilder<'ctx>,
    debug_compile_unit: DICompileUnit<'ctx>,
    debug_path_root: Option<std::path::PathBuf>,
    debug_files: HashMap<String, DIFile<'ctx>>,
    debug_subprograms: HashMap<FunctionId, DISubprogram<'ctx>>,
    debug_scopes: HashMap<(FunctionId, String), DILexicalBlock<'ctx>>,
}

/// The exported symbol name of the entry point at `index` in the SSA's entry-point list.
pub fn entry_export_symbol(index: usize) -> String {
    match index {
        0 => "mavros_main".to_string(),
        1 => "mavros_ad_main".to_string(),
        i => format!("mavros_entry_{i}"),
    }
}

impl<'ctx> LLVMCodeGen<'ctx> {
    pub fn new(context: &'ctx Context, module_name: &str) -> Self {
        let module = context.create_module(module_name);
        module.add_basic_value_flag(
            "Debug Info Version",
            FlagBehavior::Warning,
            context.i32_type().const_int(3, false),
        );
        let (debug_builder, debug_compile_unit) = module.create_debug_info_builder(
            true,
            // DWARF has no Noir language code; C is the conventional generic frontend choice.
            DWARFSourceLanguage::C,
            module_name,
            ".",
            "mavros",
            false,
            "",
            0,
            "",
            DWARFEmissionKind::Full,
            0,
            false,
            false,
            "",
            "",
        );
        let builder = context.create_builder();

        let mut codegen = Self {
            context,
            module,
            builder,
            value_map: HashMap::default(),
            constants: HashMap::default(),
            block_map: HashMap::default(),
            function_map: HashMap::default(),
            vm_ptr: None,
            field_mul_fn: None,
            field_add_fn: None,
            field_sub_fn: None,
            field_div_fn: None,
            field_lt_fn: None,
            malloc_fn: None,
            free_fn: None,
            field_from_limbs_fn: None,
            field_to_limbs_fn: None,
            globals: Vec::new(),
            const_data_counter: 0,
            entry_symbols: Vec::new(),
            debug_builder,
            debug_compile_unit,
            debug_path_root: None,
            debug_files: HashMap::default(),
            debug_subprograms: HashMap::default(),
            debug_scopes: HashMap::default(),
        };

        codegen.declare_runtime_functions();
        codegen
    }

    pub fn set_debug_path_root(&mut self, root: Option<std::path::PathBuf>) {
        self.debug_path_root = root;
    }

    fn function_source_location(function: &LLFunction) -> SourceLocation {
        function
            .get_entry()
            .first_location()
            .cloned()
            .or_else(|| {
                function
                    .get_blocks()
                    .find_map(|(_, block)| block.first_location().cloned())
            })
            .unwrap_or_else(|| SourceLocation::synthetic(function.get_name()))
    }

    fn dwarf_coordinate(value: u64) -> u32 {
        u32::try_from(value).unwrap_or(u32::MAX)
    }

    fn debug_file(&mut self, location: &SourceLocation) -> DIFile<'ctx> {
        if let Some(file) = self.debug_files.get(location.file.as_ref()) {
            return *file;
        }

        let original_path = Path::new(location.file.as_ref());
        let path = self
            .debug_path_root
            .as_deref()
            .and_then(|root| original_path.strip_prefix(root).ok())
            .unwrap_or(original_path);
        let filename = path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or(location.file.as_ref());
        let directory = path
            .parent()
            .and_then(|parent| parent.to_str())
            .filter(|parent| !parent.is_empty())
            .unwrap_or(".");
        let file = self.debug_builder.create_file(filename, directory);
        self.debug_files.insert(location.file.to_string(), file);
        file
    }

    fn attach_debug_subprogram(
        &mut self,
        fn_id: FunctionId,
        function: &LLFunction,
        fn_value: FunctionValue<'ctx>,
    ) {
        let location = Self::function_source_location(function);
        let file = self.debug_file(&location);
        // Stack traces only need line-level symbolication, so the debug signature omits types.
        let subroutine_type =
            self.debug_builder
                .create_subroutine_type(file, None, &[], DIFlags::PUBLIC);
        let line = Self::dwarf_coordinate(location.start.line);
        let subprogram = self.debug_builder.create_function(
            self.debug_compile_unit.as_debug_info_scope(),
            function.get_name(),
            fn_value.get_name().to_str().ok(),
            file,
            line,
            subroutine_type,
            false,
            true,
            line,
            DIFlags::PUBLIC,
            false,
        );
        fn_value.set_subprogram(subprogram);
        self.debug_subprograms.insert(fn_id, subprogram);
    }

    fn set_debug_location(&mut self, fn_id: FunctionId, location: &SourceLocation) {
        let file = self.debug_file(location);
        let line = Self::dwarf_coordinate(location.start.line);
        let column = Self::dwarf_coordinate(location.start.column);
        let key = (fn_id, location.file.to_string());
        let scope = if let Some(scope) = self.debug_scopes.get(&key) {
            *scope
        } else {
            let parent = self.debug_subprograms[&fn_id];
            let scope = self.debug_builder.create_lexical_block(
                parent.as_debug_info_scope(),
                file,
                line,
                column,
            );
            self.debug_scopes.insert(key, scope);
            scope
        };
        let debug_location = self.debug_builder.create_debug_location(
            self.context,
            line,
            column,
            scope.as_debug_info_scope(),
            None,
        );
        self.builder.set_current_debug_location(debug_location);
    }

    // ── Type conversion ─────────────────────────────────────────────────

    /// Convert an Type to the corresponding LLVM type.
    fn convert_type(&self, ty: &Type) -> BasicTypeEnum<'ctx> {
        match ty {
            Type::Int(bits) => self
                .context
                .custom_width_int_type(
                    NonZeroU32::new(*bits).expect("Cannot have zero-width integer"),
                )
                .expect("A basic integer type can be created")
                .into(),
            Type::Ptr => self.context.ptr_type(AddressSpace::default()).into(),
            Type::Struct(s) => self.convert_struct_type(s),
        }
    }

    fn int_mask(&self, bits: u32, value: u128) -> IntValue<'ctx> {
        let words = [value as u64, (value >> 64) as u64];
        self.context
            .custom_width_int_type(NonZeroU32::new(bits).expect("Cannot have zero-width integer"))
            .expect("A basic integer type can be created")
            .const_int_arbitrary_precision(&words)
    }

    fn low_bits_mask(bits: u32) -> u128 {
        if bits == 128 {
            u128::MAX
        } else {
            (1u128 << bits) - 1
        }
    }

    fn widen_or_trunc_int(
        &self,
        value: IntValue<'ctx>,
        to_bits: u32,
        name: &str,
    ) -> IntValue<'ctx> {
        let from_bits = value.get_type().get_bit_width();
        if from_bits == to_bits {
            value
        } else if from_bits < to_bits {
            let ty = self
                .context
                .custom_width_int_type(
                    NonZeroU32::new(to_bits).expect("Cannot have zero-width integer"),
                )
                .expect("A basic integer type can be created");
            self.builder.build_int_z_extend(value, ty, name).unwrap()
        } else {
            let ty = self
                .context
                .custom_width_int_type(
                    NonZeroU32::new(to_bits).expect("Cannot have zero-width integer"),
                )
                .expect("A basic integer type can be created");
            self.builder.build_int_truncate(value, ty, name).unwrap()
        }
    }

    fn compile_spread_bits(
        &self,
        value: IntValue<'ctx>,
        active_bits: u8,
        result_bits: u32,
        name: &str,
    ) -> IntValue<'ctx> {
        assert!(
            active_bits <= 64,
            "Spread only supports widths up to 64, got {}",
            active_bits
        );
        let mut x = self.widen_or_trunc_int(value, result_bits, "spread_wide");
        x = self
            .builder
            .build_and(
                x,
                self.int_mask(result_bits, Self::low_bits_mask(active_bits as u32)),
                "spread_active",
            )
            .unwrap();
        for (shift, mask) in [
            (32, 0x0000_0000_FFFF_FFFF_0000_0000_FFFF_FFFFu128),
            (16, 0x0000_FFFF_0000_FFFF_0000_FFFF_0000_FFFFu128),
            (8, 0x00FF_00FF_00FF_00FF_00FF_00FF_00FF_00FFu128),
            (4, 0x0F0F_0F0F_0F0F_0F0F_0F0F_0F0F_0F0F_0F0Fu128),
            (2, 0x3333_3333_3333_3333_3333_3333_3333_3333u128),
            (1, 0x5555_5555_5555_5555_5555_5555_5555_5555u128),
        ] {
            if result_bits > shift {
                let shamt = x.get_type().const_int(shift as u64, false);
                let shifted = self
                    .builder
                    .build_left_shift(x, shamt, "spread_shl")
                    .unwrap();
                let or = self.builder.build_or(x, shifted, "spread_or").unwrap();
                x = self
                    .builder
                    .build_and(or, self.int_mask(result_bits, mask), "spread_mask")
                    .unwrap();
            }
        }
        self.widen_or_trunc_int(x, result_bits, name)
    }

    fn compact_spread_bits(
        &self,
        value: IntValue<'ctx>,
        active_bits: u8,
        result_bits: u32,
        name: &str,
    ) -> IntValue<'ctx> {
        assert!(
            active_bits <= 64,
            "Unspread only supports active widths up to 64, got {}",
            active_bits
        );
        let work_bits = value.get_type().get_bit_width();
        let mut x = self
            .builder
            .build_and(
                value,
                self.int_mask(work_bits, 0x5555_5555_5555_5555_5555_5555_5555_5555u128),
                "unspread_mask0",
            )
            .unwrap();
        for (shift, mask) in [
            (1, 0x3333_3333_3333_3333_3333_3333_3333_3333u128),
            (2, 0x0F0F_0F0F_0F0F_0F0F_0F0F_0F0F_0F0F_0F0Fu128),
            (4, 0x00FF_00FF_00FF_00FF_00FF_00FF_00FF_00FFu128),
            (8, 0x0000_FFFF_0000_FFFF_0000_FFFF_0000_FFFFu128),
            (16, 0x0000_0000_FFFF_FFFF_0000_0000_FFFF_FFFFu128),
            (32, 0x0000_0000_0000_0000_FFFF_FFFF_FFFF_FFFFu128),
        ] {
            if work_bits > shift {
                let shamt = x.get_type().const_int(shift as u64, false);
                let shifted = self
                    .builder
                    .build_right_shift(x, shamt, false, "unspread_shr")
                    .unwrap();
                let or = self.builder.build_or(x, shifted, "unspread_or").unwrap();
                x = self
                    .builder
                    .build_and(or, self.int_mask(work_bits, mask), "unspread_mask")
                    .unwrap();
            }
        }
        self.widen_or_trunc_int(x, result_bits, name)
    }

    /// Materialise an LLSSA constant as an LLVM constant value, recursively.
    fn materialize_const(&self, c: &Constant) -> BasicValueEnum<'ctx> {
        match c {
            Constant::Int { bits, value } => self.int_mask(*bits, *value).into(),
            Constant::NullPtr => self
                .context
                .ptr_type(AddressSpace::default())
                .const_null()
                .into(),
            Constant::Struct { layout, values } => {
                let fields: Vec<BasicValueEnum<'ctx>> =
                    values.iter().map(|v| self.materialize_const(v)).collect();
                self.convert_struct_type(layout)
                    .into_struct_type()
                    .const_named_struct(&fields)
                    .into()
            }
            Constant::Blob(_) => {
                panic!("Blob constants cannot be materialized as normal LLVM values")
            }
        }
    }

    fn materialize_const_data_element(
        &self,
        elem_type: &LLStruct,
        value: &Constant,
    ) -> StructValue<'ctx> {
        if let Constant::Struct { layout, .. } = value {
            if layout == elem_type {
                return self.materialize_const(value).into_struct_value();
            }
        }

        assert_eq!(
            elem_type.fields.len(),
            1,
            "scalar const data must target a single-field element struct"
        );
        assert!(
            value.matches_field(&elem_type.fields[0]),
            "const data element {:?} does not match {}",
            value,
            elem_type
        );
        let field = self.materialize_const(value);
        self.convert_struct_type(elem_type)
            .into_struct_type()
            .const_named_struct(&[field])
    }

    fn materialize_const_data(
        &mut self,
        elem_type: &LLStruct,
        blob: &LLBlob,
    ) -> PointerValue<'ctx> {
        assert!(
            !blob.is_empty(),
            "ConstDataPtr should not be emitted for empty data"
        );
        let elem_ty = self.convert_struct_type(elem_type).into_struct_type();
        let values: Vec<StructValue<'ctx>> = blob
            .elements
            .iter()
            .map(|value| self.materialize_const_data_element(elem_type, value))
            .collect();
        let array_ty = elem_ty.array_type(values.len() as u32);
        let array_value = elem_ty.const_array(&values);
        let name = format!("__mavros_const_data_{}", self.const_data_counter);
        self.const_data_counter += 1;

        let global = self
            .module
            .add_global(array_ty, Some(AddressSpace::default()), &name);
        global.set_initializer(&array_value);
        global.set_constant(true);
        global.set_linkage(Linkage::Private);
        global.set_unnamed_addr(true);

        let zero = self.context.i32_type().const_zero();
        unsafe {
            self.builder
                .build_gep(
                    array_ty,
                    global.as_pointer_value(),
                    &[zero, zero],
                    "const_data",
                )
                .unwrap()
        }
    }

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
            LLFieldType::FlexArray(s) => {
                let elem = self.convert_struct_type(s);
                elem.array_type(0).into()
            }
        }
    }

    /// The LLVM type for a field element, derived from `LLStruct::field_elem()`.
    // FIELD-ASSUMPTION: L3-llstruct
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
        let bool_type = self.context.bool_type();
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

        // __field_lt(FieldElem, FieldElem) -> bool
        let field_lt_type = bool_type.fn_type(&[field_type.into(), field_type.into()], false);
        self.field_lt_fn = Some(self.module.add_function("__field_lt", field_lt_type, None));

        // FIELD-ASSUMPTION: L3-limb-op
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

    // ── Compilation entry point ─────────────────────────────────────────

    /// Compile LLSSA to LLVM IR.
    pub fn compile(&mut self, llssa: &LLSSA, flow_analysis: &FlowAnalysis) {
        let entry_points: Vec<FunctionId> = llssa.get_entry_points().to_vec();

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

        self.constants = llssa.const_snapshot();

        self.entry_symbols = (0..entry_points.len()).map(entry_export_symbol).collect();

        // First pass: declare all functions
        for (fn_id, function) in llssa.iter_functions() {
            self.declare_function(*fn_id, function, &entry_points);
        }

        // Second pass: generate function bodies
        for (fn_id, function) in llssa.iter_functions() {
            let cfg = flow_analysis.get_function_cfg(*fn_id);
            self.compile_function(*fn_id, function, cfg, &entry_points);
        }
        self.builder.unset_current_debug_location();
        self.debug_builder.finalize();
    }

    fn declare_function(
        &mut self,
        fn_id: FunctionId,
        function: &LLFunction,
        entry_points: &[FunctionId],
    ) {
        let entry = function.get_entry();
        let ptr_type = self.context.ptr_type(AddressSpace::default());
        let entry_index = entry_points.iter().position(|e| *e == fn_id);

        // First parameter is always VM*
        let mut param_types: Vec<BasicMetadataTypeEnum> = vec![ptr_type.into()];

        if entry_index.is_none() {
            for (_, tp) in entry.get_parameters().skip(1) {
                param_types.push(self.convert_type(tp).into());
            }
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

        let name = match entry_index {
            Some(i) => self.entry_symbols[i].clone(),
            None => function.get_name().to_string(),
        };

        let fn_value = self.module.add_function(&name, fn_type, None);
        self.attach_debug_subprogram(fn_id, function, fn_value);
        self.function_map.insert(fn_id, fn_value);
    }

    fn compile_function(
        &mut self,
        fn_id: FunctionId,
        function: &LLFunction,
        cfg: &flow_analysis::CFG,
        entry_points: &[FunctionId],
    ) {
        self.value_map.clear();
        for (vid, constant) in &self.constants {
            if !matches!(constant.as_ref(), Constant::Blob(_)) {
                self.value_map
                    .insert(*vid, self.materialize_const(constant.as_ref()));
            }
        }
        self.block_map.clear();

        let fn_value = self.function_map[&fn_id];
        let function_location = Self::function_source_location(function);
        self.set_debug_location(fn_id, &function_location);
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
        if entry_points.contains(&fn_id) {
            self.load_main_params_from_memory(entry.get_parameters());
        } else {
            for (i, (param_id, _)) in entry.get_parameters().enumerate() {
                let param_value = fn_value.get_nth_param(i as u32).unwrap();
                self.value_map.insert(*param_id, param_value);
            }
        }

        // Track phi nodes
        let mut phi_nodes: HashMap<(BlockId, usize), inkwell::values::PhiValue<'ctx>> =
            HashMap::default();

        // Generate code in dominator order
        for block_id in cfg.get_domination_pre_order() {
            self.compile_block(fn_id, function, block_id, &mut phi_nodes);
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

    fn load_main_params_from_memory<'a>(
        &mut self,
        parameters: impl Iterator<Item = &'a (ValueId, Type)>,
    ) {
        let vm_ptr = self
            .vm_ptr
            .expect("main parameters are loaded relative to the VM pointer");
        let ptr_type = self.context.ptr_type(AddressSpace::default());
        let mut parameters = parameters;
        if let Some((vm_param, _)) = parameters.next() {
            self.value_map.insert(*vm_param, vm_ptr.into());
        }

        // Main's only remaining parameter is the input blob, which is bound
        // directly to the host-provided inputs buffer; its elements are loaded
        // lazily at each ArrayGet site.
        let Some((blob_param, blob_type)) = parameters.next() else {
            return;
        };
        assert!(
            matches!(blob_type, Type::Ptr),
            "main parameter must be the input blob pointer, got {:?}",
            blob_type
        );
        assert!(
            parameters.next().is_none(),
            "main must have at most one parameter besides the VM pointer"
        );

        let vm_type = self
            .convert_struct_type(&LLStruct::witgen_vm())
            .into_struct_type();
        let input_slot = self
            .builder
            .build_struct_gep(
                vm_type,
                vm_ptr,
                LLStruct::WITGEN_VM_INPUTS as u32,
                "inputs_slot",
            )
            .unwrap();
        let input_ptr = self
            .builder
            .build_load(ptr_type, input_slot, "inputs_ptr")
            .unwrap()
            .into_pointer_value();
        self.value_map.insert(*blob_param, input_ptr.into());
    }

    fn compile_block(
        &mut self,
        fn_id: FunctionId,
        function: &LLFunction,
        block_id: BlockId,
        phi_nodes: &mut HashMap<(BlockId, usize), inkwell::values::PhiValue<'ctx>>,
    ) {
        let block = function.get_block(block_id);
        let bb = self.block_map[&block_id];
        let block_location = block
            .first_location()
            .or_else(|| function.get_entry().first_location())
            .cloned()
            .unwrap_or_else(|| SourceLocation::synthetic(function.get_name()));
        self.set_debug_location(fn_id, &block_location);

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

        for (instruction, location) in block.get_instructions_with_source_locations() {
            self.set_debug_location(fn_id, location);
            self.compile_instruction(instruction);
        }

        if let Some(terminator) = block.get_terminator() {
            self.compile_terminator(terminator);
        }
    }

    // ── Instruction compilation ─────────────────────────────────────────

    fn compile_instruction(&mut self, op: &LLOp) {
        match op {
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
                    IntArithOp::SDiv => self.builder.build_int_signed_div(lhs, rhs, name).unwrap(),
                    IntArithOp::SRem => self.builder.build_int_signed_rem(lhs, rhs, name).unwrap(),
                    IntArithOp::And => self.builder.build_and(lhs, rhs, name).unwrap(),
                    IntArithOp::Or => self.builder.build_or(lhs, rhs, name).unwrap(),
                    IntArithOp::Xor => self.builder.build_xor(lhs, rhs, name).unwrap(),
                    IntArithOp::Shl => {
                        // Mask shift count modulo bit width — LLVM treats shifts
                        // by >= bit_width as poison, but the VM's Rust `<<` on
                        // x86 masks to bit_width-1. Match the VM.
                        let bw = lhs.get_type().get_bit_width();
                        let mask = lhs.get_type().const_int((bw - 1) as u64, false);
                        let masked_rhs = self.builder.build_and(rhs, mask, "shamt").unwrap();
                        self.builder
                            .build_left_shift(lhs, masked_rhs, name)
                            .unwrap()
                    }
                    IntArithOp::UShr => {
                        let bw = lhs.get_type().get_bit_width();
                        let mask = lhs.get_type().const_int((bw - 1) as u64, false);
                        let masked_rhs = self.builder.build_and(rhs, mask, "shamt").unwrap();
                        self.builder
                            .build_right_shift(lhs, masked_rhs, false, name)
                            .unwrap()
                    }
                };
                self.value_map.insert(*result, val.into());
            }

            LLOp::Spread {
                result,
                value,
                bits,
                result_bits,
            } => {
                let input = self.value_map[value].into_int_value();
                let val =
                    self.compile_spread_bits(input, *bits, *result_bits, &format!("v{}", result.0));
                self.value_map.insert(*result, val.into());
            }

            LLOp::Unspread {
                result_odd,
                result_even,
                value,
                bits,
                odd_bits,
                even_bits,
            } => {
                let input = self.value_map[value].into_int_value();
                let active_input_bits = (*bits as u32) * 2;
                let input = self
                    .builder
                    .build_and(
                        input,
                        self.int_mask(
                            input.get_type().get_bit_width(),
                            Self::low_bits_mask(active_input_bits),
                        ),
                        "unspread_active",
                    )
                    .unwrap();
                let odd_source = self
                    .builder
                    .build_right_shift(
                        input,
                        input.get_type().const_int(1, false),
                        false,
                        "unspread_odd_src",
                    )
                    .unwrap();
                let odd = self.compact_spread_bits(
                    odd_source,
                    *bits,
                    *odd_bits,
                    &format!("v{}", result_odd.0),
                );
                let even = self.compact_spread_bits(
                    input,
                    *bits,
                    *even_bits,
                    &format!("v{}", result_even.0),
                );
                self.value_map.insert(*result_odd, odd.into());
                self.value_map.insert(*result_even, even.into());
            }

            LLOp::IntCmp { kind, result, a, b } => {
                let predicate = match kind {
                    IntCmpOp::Eq => IntPredicate::EQ,
                    IntCmpOp::ULt => IntPredicate::ULT,
                    IntCmpOp::SLt => IntPredicate::SLT,
                };
                // icmp accepts pointer operands directly; this happens for
                // null checks on RC'd cell slots.
                let val = match (self.value_map[a], self.value_map[b]) {
                    (BasicValueEnum::PointerValue(lhs), BasicValueEnum::PointerValue(rhs)) => self
                        .builder
                        .build_int_compare(predicate, lhs, rhs, &format!("v{}", result.0))
                        .unwrap(),
                    (lhs, rhs) => self
                        .builder
                        .build_int_compare(
                            predicate,
                            lhs.into_int_value(),
                            rhs.into_int_value(),
                            &format!("v{}", result.0),
                        )
                        .unwrap(),
                };
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

            LLOp::FieldLt { result, a, b } => {
                let lhs = self.value_map[a];
                let rhs = self.value_map[b];
                let lt_fn = self.field_lt_fn.expect("__field_lt not declared");
                let call_site = self
                    .builder
                    .build_call(lt_fn, &[lhs.into(), rhs.into()], "field_lt")
                    .unwrap();
                let val = call_site
                    .try_as_basic_value()
                    .expect_basic("field_lt should return a value");
                self.value_map.insert(*result, val);
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
            LLOp::HeapAlloc {
                result,
                struct_type,
                flex_count,
            } => {
                let struct_ty = self.convert_struct_type(struct_type);
                let size = struct_ty.size_of().unwrap();
                let i32_type = self.context.i32_type();
                let mut size_i32 = self
                    .builder
                    .build_int_truncate_or_bit_cast(size, i32_type, "size")
                    .unwrap();
                if let Some(count) = flex_count {
                    let flex_elem = ll_struct_flex_elem(struct_type)
                        .expect("flex_count provided for struct with no FlexArray field");
                    let elem_ty = self.convert_struct_type(flex_elem);
                    let elem_size = elem_ty.size_of().unwrap();
                    let elem_size_i32 = self
                        .builder
                        .build_int_truncate_or_bit_cast(elem_size, i32_type, "flex_elem_size")
                        .unwrap();
                    let count_i32 = self
                        .builder
                        .build_int_truncate_or_bit_cast(
                            self.value_map[count].into_int_value(),
                            i32_type,
                            "flex_count",
                        )
                        .unwrap();
                    let flex_size = self
                        .builder
                        .build_int_mul(elem_size_i32, count_i32, "flex_size")
                        .unwrap();
                    size_i32 = self
                        .builder
                        .build_int_add(size_i32, flex_size, "alloc_size")
                        .unwrap();
                }
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
                let gep = self
                    .builder
                    .build_struct_gep(llvm_struct_ty, p, *field as u32, "sfp")
                    .unwrap();
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

            LLOp::ConstDataPtr {
                result,
                elem_type,
                blob,
            } => {
                let blob_data = match self.constants.get(blob).map(|constant| constant.as_ref()) {
                    Some(Constant::Blob(blob)) => blob.clone(),
                    _ => panic!("ConstDataPtr input v{} is not a blob", blob.0),
                };
                let ptr = self.materialize_const_data(elem_type, &blob_data);
                self.value_map.insert(*result, ptr.into());
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
                        self.value_map.insert(*result_id, val);
                    }
                }
            }

            LLOp::GlobalAddr { result, global_id } => {
                let global = self.globals[*global_id];
                let ptr = global.as_pointer_value();
                self.value_map.insert(*result, ptr.into());
            }

            _ => panic!("Unsupported LLOp in LLSSA codegen: {:?}", op),
        }
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

    pub fn compile_to_wasm(&self, path: &Path, opts: WasmCompileOpts) {
        use std::process::Command;

        Target::initialize_webassembly(&InitializationConfig::default());

        let target_triple = TargetTriple::create("wasm32-unknown-unknown");
        let target = Target::from_triple(&target_triple).unwrap();

        // The module must carry the wasm32 triple + datalayout before any
        // mid-end passes run: without a datalayout the optimizer folds
        // struct GEPs using host (8-byte-pointer) field offsets, which
        // miscompiles every VM-struct access on wasm32.
        self.module.set_triple(&target_triple);

        let target_machine = target
            .create_target_machine(
                &target_triple,
                "generic",
                "",
                opts.codegen_level,
                RelocMode::Default,
                CodeModel::Default,
            )
            .unwrap();

        self.module
            .set_data_layout(&target_machine.get_target_data().get_data_layout());

        if let Some(pipeline) = opts.midend_pipeline {
            self.module
                .run_passes(
                    pipeline,
                    &target_machine,
                    inkwell::passes::PassBuilderOptions::create(),
                )
                .unwrap();
        }

        let obj_path = path.with_extension("o");
        target_machine
            .write_to_file(&self.module, FileType::Object, &obj_path)
            .unwrap();

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
            .arg("--no-entry")
            .args((!opts.include_debug_info).then_some("--strip-debug"))
            .args(
                self.entry_symbols
                    .iter()
                    .map(|symbol| format!("--export={symbol}")),
            )
            .args([
                "--import-memory",
                "--allow-undefined",
                "--stack-first",
                "-z",
                &format!("stack-size={WASM_STACK_SIZE_BYTES}"),
                "--export=__data_end",
                "--export=__live_bytes",
                "-o",
            ])
            .arg(path)
            .arg(&obj_path)
            .arg(&opts.runtime_lib)
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

        let debug_path = wasm_debug_info_path(path);
        if opts.include_debug_info {
            let linked = std::fs::read(path).unwrap_or_else(|error| {
                panic!("failed to read linked WASM {}: {error}", path.display())
            });
            let external_url = debug_path
                .file_name()
                .and_then(|name| name.to_str())
                .expect("WASM debug sidecar path must have a UTF-8 filename");
            let (stripped, debug) = crate::wasm_debug::split_debug_info(&linked, external_url)
                .unwrap_or_else(|error| {
                    panic!(
                        "failed to split WASM debug info from {}: {error}",
                        path.display()
                    )
                });
            std::fs::write(path, stripped).unwrap_or_else(|error| {
                panic!("failed to write stripped WASM {}: {error}", path.display())
            });
            std::fs::write(&debug_path, debug).unwrap_or_else(|error| {
                panic!(
                    "failed to write WASM debug sidecar {}: {error}",
                    debug_path.display()
                )
            });
        } else {
            std::fs::remove_file(debug_path).ok();
        }

        std::fs::remove_file(&obj_path).ok();
    }
}

/// Path used for a WASM module's standalone DWARF sidecar.
pub fn wasm_debug_info_path(wasm_path: &Path) -> std::path::PathBuf {
    wasm_path.with_extension("debug.wasm")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::{
        analysis::flow_analysis::FlowAnalysis,
        ssa::{
            SourcePosition,
            llssa::{
                LLSSA,
                builder::{LLEmitter, LLSSABuilder},
            },
        },
    };

    #[test]
    fn wasm_debug_sidecar_has_a_wasm_extension() {
        assert_eq!(
            wasm_debug_info_path(Path::new("target/program.wasm")),
            Path::new("target/program.debug.wasm")
        );
    }

    #[test]
    fn emits_instruction_source_locations_as_llvm_debug_info() {
        let mut ssa = LLSSA::with_main("located_main".to_string());
        let main_id = ssa.get_unique_entrypoint_id();
        let location = SourceLocation::new(
            "/tmp/mavros-project/src/main.nr",
            SourcePosition::new(12, 7),
            SourcePosition::new(12, 16),
        );
        let mut ssa_builder = LLSSABuilder::new(&mut ssa);
        ssa_builder.modify_function(main_id, |function| {
            let entry = function.function.get_entry_id();
            let mut block = function.block(entry).with_source_location(location);
            let field_type = LLStruct::field_elem();
            let one = block.emit_struct_const(
                field_type.clone(),
                vec![
                    Constant::Int { bits: 64, value: 1 },
                    Constant::Int { bits: 64, value: 0 },
                    Constant::Int { bits: 64, value: 0 },
                    Constant::Int { bits: 64, value: 0 },
                ],
            );
            let two = block.emit_struct_const(
                field_type,
                vec![
                    Constant::Int { bits: 64, value: 2 },
                    Constant::Int { bits: 64, value: 0 },
                    Constant::Int { bits: 64, value: 0 },
                    Constant::Int { bits: 64, value: 0 },
                ],
            );
            block.field_arith(FieldArithOp::Add, one, two);
            block.terminate_return(Vec::new());
        });

        let flow = FlowAnalysis::run(&ssa);
        let context = Context::create();
        let mut codegen = LLVMCodeGen::new(&context, "debug_test");
        codegen.set_debug_path_root(Some("/tmp/mavros-project".into()));
        codegen.compile(&ssa, &flow);
        codegen.module.verify().unwrap();
        let ir = codegen.get_ir();

        assert!(ir.contains("!DIFile(filename: \"main.nr\", directory: \"src\")"));
        assert!(ir.contains("!DISubprogram(name: \"located_main\""));
        assert!(ir.contains("!DILocation(line: 12, column: 7"));
        assert!(
            ir.lines().any(|line| {
                line.contains("call") && line.contains("@__field_add") && line.contains("!dbg")
            }),
            "field add must carry a debug location:\n{ir}"
        );
    }
}
