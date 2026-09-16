//! LLSSA → LLVM Code Generation
//!
//! Translates LLSSA into LLVM IR, which can then be compiled to WebAssembly.
//! Operates on LLSSA + Type — types are explicit in the LLSSA ops, no TypeInfo needed.

use std::{num::NonZeroU32, path::Path};

use inkwell::{
    AddressSpace, IntPredicate, OptimizationLevel,
    attributes::{Attribute, AttributeLoc},
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

use mavros_int_semantics::{IntBits, int_bits::HOST_LIMB_BITS};

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

// CONSTANTS
// ================================================================================================

const WASM_STACK_SIZE_BYTES: u32 = 256 * 1024;

/// The widest integer whose multiply LLVM's wasm32 lowering keeps to a single libcall.
///
/// Between one host word and this one an `iN` multiply becomes `__multi3`, which
/// `compiler_builtins` already defines weakly in the runtime archive; narrower than that it is a
/// machine instruction. Above it LLVM expands the multiply inline and quadratically: 7.2 MB of code
/// at `i16384`, and from about `i5700` upwards a module that every wasm engine we run refuses to
/// _instantiate_, having compiled and linked it cleanly.
///
/// The routing threshold is therefore where the libcall stops to ensure simplicity and uniform
/// behavior. Technically LLVM would continue to work between i129 and i5700, but we deemed this not
/// worthwhile.
const INLINE_MUL_MAX_BITS: u32 = 2 * HOST_LIMB_BITS as u32;

// UTILITIES
// ================================================================================================

/// Whether `kind` at `bits` is emitted around a call to the runtime's `__int_mul`.
///
/// The name is not a link: `mavros-wasm-runtime` is a link-time dependency of the emitted module
/// and not a crate dependency of this one, so there is nothing here for rustdoc to resolve.
///
/// The three operations are one problem rather than three. LLVM builds `urem` and `srem` as
/// `n - (n / d) * d`, so their expansion _contains_ a multiply of the same width and inherits its
/// whole cost -- which is why `__multi3` appears in exactly these three above 128 bits and in
/// nothing else.
///
/// `udiv` and `sdiv` stay on LLVM's own lowering, which expands them to a bit-serial loop with no
/// multiply in it: 261-397 KB at `i16384`, large but linear and accepted everywhere.
fn needs_the_wide_multiply(kind: &IntArithOp, bits: u32) -> bool {
    bits > INLINE_MUL_MAX_BITS
        && matches!(kind, IntArithOp::Mul | IntArithOp::URem | IntArithOp::SRem)
}

/// Whether this particular operation is emitted around the helper call.
///
/// The cost [`needs_the_wide_multiply`] describes is the cost of an **expansion**, and there is no
/// expansion where there is no instruction: `IRBuilder` folds two constant operands as it is
/// handed them, before any pass runs, so a constant wide product is an `APInt` multiply and an
/// answer. Routing one anyway would replace a value the rest of the module can fold through with
/// a call it cannot -- and would cost the conformance sweep every point it has at these widths,
/// since a call is not something LLVM folds.
fn routes_through_the_wide_multiply(
    kind: &IntArithOp,
    lhs: IntValue<'_>,
    rhs: IntValue<'_>,
) -> bool {
    needs_the_wide_multiply(kind, lhs.get_type().get_bit_width())
        && !(lhs.is_const() && rhs.is_const())
}

// COMPILATION
// ================================================================================================

/// How to compile a module to WASM.
#[derive(Clone, Debug)]
pub struct WasmCompileOpts {
    /// LLVM mid-end pass pipeline to run before codegen (e.g. `"default<O1>"`).
    pub midend_pipeline: Option<&'static str>,

    /// Codegen (instruction selection) optimization level.
    pub codegen_level: OptimizationLevel,

    /// Pre-built wasm-runtime static library to link against. Callers are responsible for building
    /// it (see [`crate::wasm_runtime`]); codegen never invokes cargo.
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
    int_mul_fn: Option<FunctionValue<'ctx>>,
    /// Scratch buffers for the wide multiply helper, keyed by `(width, slot)` and living in the
    /// current function's entry block. Cleared per function.
    wide_scratch: HashMap<(u32, usize), PointerValue<'ctx>>,
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
            int_mul_fn: None,
            wide_scratch: HashMap::default(),
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

    /// A `bits`-wide LLVM constant carrying `value`, little-endian.
    fn int_mask(&self, bits: u32, value: u128) -> IntValue<'ctx> {
        self.int_pattern(&IntBits::from_u128(bits as usize, value))
    }

    /// An LLVM constant carrying `pattern`, at the pattern's own width.
    ///
    /// The limbs go straight through: a normalised pattern is already exactly the little-endian
    /// word sequence `const_int_arbitrary_precision` wants, at exactly the length it wants.
    fn int_pattern(&self, pattern: &IntBits) -> IntValue<'ctx> {
        let bits = u32::try_from(pattern.bits()).expect("An integer width fits a u32");
        self.context
            .custom_width_int_type(NonZeroU32::new(bits).expect("Cannot have zero-width integer"))
            .expect("A basic integer type can be created")
            .const_int_arbitrary_precision(pattern.limbs())
    }

    /// The low `bits` bits set, as a host `u128`.
    fn low_bits_mask(bits: u32) -> u128 {
        assert!(bits <= 128, "a {bits}-bit mask does not fit in a u128");
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
            Constant::Int(pattern) => self.int_pattern(pattern).into(),
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

        // __int_mul(result_ptr, a_ptr, b_ptr, limbs) -> void
        //
        // Emitted using pointers to ensure that one body can evaluate at any width.
        let int_mul_type = void_type.fn_type(
            &[
                ptr_type.into(),
                ptr_type.into(),
                ptr_type.into(),
                i32_type.into(),
            ],
            false,
        );
        self.int_mul_fn = Some(self.module.add_function(
            "__int_mul",
            int_mul_type,
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

        // The LLSSA type checker of last resort, and the only one that reads the IR that actually
        // gets emitted. It costs ~0.04% of a compile which is cheap enough to run under every
        // build.
        if let Err(message) = self.module.verify() {
            panic!("LLVM rejected the generated module:\n{message}");
        }
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
        self.wide_scratch.clear();
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

    /// Emit one integer arithmetic operation on an exact-width `iN`.
    ///
    /// Split out of [`Self::compile_instruction`] so that the conformance test can call **this**
    /// with constant operands and read LLVM's own constant fold back.
    ///
    /// `convert_type` maps `Type::Int(bits)` to a `custom_width_int_type(bits)`, so the operands
    /// are the operation's declared width rather than a host-sized register. This ensures that
    /// `sdiv`/`srem`/`ashr` read the sign bit in the right place with no preamble, where the VM's
    /// `_int` lane needs `signed_cell` to recover it from a wider cell.
    fn build_int_arith(
        &mut self,
        kind: &IntArithOp,
        lhs: IntValue<'ctx>,
        rhs: IntValue<'ctx>,
        name: &str,
    ) -> IntValue<'ctx> {
        if routes_through_the_wide_multiply(kind, lhs, rhs) {
            return self.build_around_the_wide_multiply(kind, lhs, rhs, name);
        }

        match kind {
            IntArithOp::Add => self.builder.build_int_add(lhs, rhs, name).unwrap(),
            IntArithOp::Sub => self.builder.build_int_sub(lhs, rhs, name).unwrap(),
            IntArithOp::Mul => self.builder.build_int_mul(lhs, rhs, name).unwrap(),
            IntArithOp::UDiv => self.builder.build_int_unsigned_div(lhs, rhs, name).unwrap(),
            IntArithOp::URem => self.builder.build_int_unsigned_rem(lhs, rhs, name).unwrap(),
            IntArithOp::SDiv => self.builder.build_int_signed_div(lhs, rhs, name).unwrap(),
            IntArithOp::SRem => self.builder.build_int_signed_rem(lhs, rhs, name).unwrap(),
            IntArithOp::And => self.builder.build_and(lhs, rhs, name).unwrap(),
            IntArithOp::Or => self.builder.build_or(lhs, rhs, name).unwrap(),
            IntArithOp::Xor => self.builder.build_xor(lhs, rhs, name).unwrap(),
            IntArithOp::Shl => {
                let reduced_rhs = self.reduce_shift_count(lhs, rhs);
                self.builder
                    .build_left_shift(lhs, reduced_rhs, name)
                    .unwrap()
            }
            IntArithOp::UShr | IntArithOp::AShr => {
                let reduced_rhs = self.reduce_shift_count(lhs, rhs);
                self.builder
                    .build_right_shift(lhs, reduced_rhs, matches!(kind, IntArithOp::AShr), name)
                    .unwrap()
            }
        }
    }

    /// The three helper-routed operations, each built around one call to `__int_mul`.
    ///
    /// The remainders are synthesised as `n - (n / d) * d`, which is the identity LLVM's own
    /// expansion uses; only the multiply in it is custom.
    ///
    /// A zero divisor is poison in LLVM's `udiv`/`sdiv` and unspecified in the model, so this
    /// inherits LLVM's behavior.
    fn build_around_the_wide_multiply(
        &mut self,
        kind: &IntArithOp,
        lhs: IntValue<'ctx>,
        rhs: IntValue<'ctx>,
        name: &str,
    ) -> IntValue<'ctx> {
        let quotient = match kind {
            IntArithOp::Mul => return self.build_wide_multiply(lhs, rhs, name),
            IntArithOp::URem => self
                .builder
                .build_int_unsigned_div(lhs, rhs, "wide_rem_quot")
                .unwrap(),
            IntArithOp::SRem => self
                .builder
                .build_int_signed_div(lhs, rhs, "wide_rem_quot")
                .unwrap(),
            other => unreachable!("{other:?} is not routed through the wide multiply"),
        };
        let product = self.build_wide_multiply(quotient, rhs, "wide_rem_prod");
        self.builder.build_int_sub(lhs, product, name).unwrap()
    }

    /// `lhs * rhs` at the operands' own width, computed by the runtime helper.
    ///
    /// The value is widened to a whole number of limbs before it is stored, and narrowed again
    /// after. LLVM's _store_ size for an `iN` is `ceil(N / 8)` bytes, so storing an `i1000`
    /// directly would leave the top limb's high three bytes unwritten. Those bits sit above the
    /// width, and a garbage bit at position `p >= N` contributes to the product only at positions
    /// `>= N`, which the truncation discards.
    ///
    /// What the widening buys is therefore **definedness** rather than a value as reading
    /// uninitialized memory is undefined behaviour on both sides of the call, in a body that has
    /// no other.
    ///
    /// Widening is by zero-extension for every one of the three callers, because a product's low
    /// bits do not depend on how its operands are read -- the sign, like the padding, lives
    /// entirely in the bits above the width.
    fn build_wide_multiply(
        &mut self,
        lhs: IntValue<'ctx>,
        rhs: IntValue<'ctx>,
        name: &str,
    ) -> IntValue<'ctx> {
        let bits = lhs.get_type().get_bit_width();
        let limbs = bits.div_ceil(HOST_LIMB_BITS as u32);
        let padded_bits = limbs * HOST_LIMB_BITS as u32;

        let padded_type = self
            .context
            .custom_width_int_type(NonZeroU32::new(padded_bits).expect("a padded width is nonzero"))
            .expect("A basic integer type can be created");
        let a_slot = self.wide_scratch_slot(padded_type, 0);
        let b_slot = self.wide_scratch_slot(padded_type, 1);
        let out_slot = self.wide_scratch_slot(padded_type, 2);

        let a = self.widen_or_trunc_int(lhs, padded_bits, "wide_mul_a");
        let b = self.widen_or_trunc_int(rhs, padded_bits, "wide_mul_b");
        self.builder.build_store(a_slot, a).unwrap();
        self.builder.build_store(b_slot, b).unwrap();

        let int_mul = self.int_mul_fn.expect("__int_mul not declared");
        self.builder
            .build_call(
                int_mul,
                &[
                    out_slot.into(),
                    a_slot.into(),
                    b_slot.into(),
                    self.context
                        .i32_type()
                        .const_int(u64::from(limbs), false)
                        .into(),
                ],
                "",
            )
            .unwrap();

        let product = self
            .builder
            .build_load(padded_type, out_slot, "wide_mul_out")
            .unwrap()
            .into_int_value();
        self.widen_or_trunc_int(product, bits, name)
    }

    /// A scratch buffer holding one `slot_type` value, in the current function's entry block.
    ///
    /// Memoised per `(width, slot)`, so a function pays the cost only for the widths it uses rather
    /// than for its call sites: helper calls within one function never overlap in time, so they
    /// share their buffers. The alloca is placed at the **top of the entry block** because one
    /// emitted where the call is would allocate afresh on every iteration of an enclosing loop,
    /// against a 256 KB wasm stack.
    ///
    /// It is typed as the value that goes into it rather than as the `[k x i64]` the helper reads,
    /// so that both are described by one type and the alloca's size is exactly the store's size.
    /// This ensures the helper remains inside the object.
    ///
    /// The alignment is then raised to a limb's, which the helper needs because it reads the buffer
    /// as `u64`s. LLVM has no alignment entry for an integer this wide in either the default layout
    /// or wasm32's, so a wide `iN` inherits `i64`'s (eight bytes preferred, four ABI).
    fn wide_scratch_slot(
        &mut self,
        slot_type: inkwell::types::IntType<'ctx>,
        slot: usize,
    ) -> PointerValue<'ctx> {
        let bits = slot_type.get_bit_width();
        if let Some(ptr) = self.wide_scratch.get(&(bits, slot)) {
            return *ptr;
        }

        let here = self
            .builder
            .get_insert_block()
            .expect("a helper call is emitted into a block");
        let entry = here
            .get_parent()
            .expect("a block belongs to a function")
            .get_first_basic_block()
            .expect("a function being compiled has an entry block");

        // Positioning before an instruction takes that instruction's debug location with it, so the
        // caller's has to be carried across by hand: everything the operation goes on to emit after
        // this would otherwise be attributed to the entry block.
        let caller_location = self.builder.get_current_debug_location();
        match entry.get_first_instruction() {
            Some(first) => self.builder.position_before(&first),
            None => self.builder.position_at_end(entry),
        }

        let ptr = self
            .builder
            .build_alloca(slot_type, &format!("wide_scratch_{bits}_{slot}"))
            .unwrap();

        let allocation = ptr.as_instruction().expect("an alloca is an instruction");
        let natural = allocation.get_alignment().unwrap_or_default();
        allocation
            .set_alignment(natural.max(HOST_LIMB_BITS as u32 / 8))
            .expect("a maximum of two powers of two is a power of two");

        self.builder.position_at_end(here);
        match caller_location {
            Some(location) => self.builder.set_current_debug_location(location),
            None => self.builder.unset_current_debug_location(),
        }
        self.wide_scratch.insert((bits, slot), ptr);
        ptr
    }

    /// Hold a shift count below the operand width, as `count % bit_width`.
    ///
    /// LLVM makes a shift by at or past the width into a **poison value**, so a total backend
    /// cannot simply pass the count through. Reducing is what the VM's `shift_amount` does with the
    /// same operands, so the two backends answer the same thing on an amount neither should have
    /// been handed.
    ///
    /// A mask by `bit_width - 1` is that modulo only where the width is a power of two. At any
    /// other width it is a submask: it stays below the width, but it corrupts counts that were
    /// already _in range_, which `hlssa_to_r1cs` applies literally. So the power-of-two case keeps
    /// the `and` and every other width gets a real `urem`.
    ///
    /// That `urem` goes through [`Self::build_int_arith`] rather than straight to the builder,
    /// because at a wide non-2pow width it is exactly the remainder [`needs_the_wide_multiply`]
    /// routes: a shift by a runtime amount at `int1000` would otherwise carry the whole multiply
    /// blowup that the operand's own operation was routed around. The amount is a runtime value, so
    /// no fold can remove it.
    fn reduce_shift_count(&mut self, lhs: IntValue<'ctx>, rhs: IntValue<'ctx>) -> IntValue<'ctx> {
        let ty = lhs.get_type();
        let bw = ty.get_bit_width();
        if bw.is_power_of_two() {
            let mask = ty.const_int(u64::from(bw - 1), false);
            return self.builder.build_and(rhs, mask, "shamt").unwrap();
        }
        let width = ty.const_int(u64::from(bw), false);
        self.build_int_arith(&IntArithOp::URem, rhs, width, "shamt")
    }

    /// Lower one LLSSA instruction.
    ///
    /// The match is **exhaustive on purpose**. Every `LLOp` has a lowering here, so a new variant
    /// should be a compile error naming this function rather than a panic reached by whichever
    /// program happens to emit one first.
    fn compile_instruction(&mut self, op: &LLOp) {
        match op {
            LLOp::IntArith { kind, result, a, b } => {
                let lhs = self.value_map[a].into_int_value();
                let rhs = self.value_map[b].into_int_value();
                let val = self.build_int_arith(kind, lhs, rhs, &format!("v{}", result.0));
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
                    let f = self.module.add_function("llvm.trap", trap_type, None);
                    // LLVM recognizes the intrinsic by name and supplies its attribute set itself —
                    // the emitted IR reads `cold noreturn nounwind memory(inaccessiblemem: write)`,
                    // none of which is set here. `noreturn` is restated anyway because this
                    // lowering _depends_ on it: it is what tells LLVM that everything after the
                    // call is dead.
                    let noreturn = Attribute::get_named_enum_kind_id("noreturn");
                    f.add_attribute(
                        AttributeLoc::Function,
                        self.context.create_enum_attribute(noreturn, 0),
                    );
                    f
                });
                // No `unreachable` here. `Trap` is an LLSSA _instruction_, so the block it sits in
                // carries on and ends with a terminator of its own; `unreachable` is an LLVM
                // _terminator_, and two terminators in one basic block is invalid IR.
                //
                // Closing the block properly instead would mean opening a fresh one for the
                // remainder, and the phi wiring in `compile_function` names the block that ends an
                // LLSSA block by its `block_map` entry — so the successors' incoming edges would
                // all be wrong. `noreturn` says the same thing to the optimizer at no structural
                // cost.
                self.builder.build_call(trap_fn, &[], "").unwrap();
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

        // `WASM_LD` is set by the nix devshell to the lld that ships with the LLVM we build
        // against, so the linker tracks the toolchain. Outside nix we fall back to whatever
        // `wasm-ld` is on `$PATH`.
        let wasm_ld = std::env::var("WASM_LD")
            .ok()
            .filter(|path| std::path::Path::new(path).exists())
            .unwrap_or_else(|| "wasm-ld".to_string());

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
    use mavros_int_semantics::{IntOp, residue};

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
    fn a_low_bits_mask_is_exact_at_the_host_boundary() {
        // 127 and 128 are the pair that a `1u128 << bits` gets wrong: the shift is defined only
        // below 128, so the top of the range has to be answered without one.
        assert_eq!(LLVMCodeGen::low_bits_mask(0), 0);
        assert_eq!(LLVMCodeGen::low_bits_mask(1), 1);
        assert_eq!(LLVMCodeGen::low_bits_mask(64), u64::MAX as u128);
        assert_eq!(LLVMCodeGen::low_bits_mask(127), u128::MAX >> 1);
        assert_eq!(LLVMCodeGen::low_bits_mask(128), u128::MAX);
    }

    #[test]
    #[should_panic(expected = "does not fit in a u128")]
    fn a_mask_wider_than_the_host_is_rejected_rather_than_wrapped() {
        // Without the assert this is `(1u128 << 129) - 1`, which panics in a debug build but
        // _masks its shift amount_ in a release one and answers `1` -- a plausible number, and
        // wrong. The rejection has to be explicit to be present at both optimisation levels.
        let _ = LLVMCodeGen::low_bits_mask(129);
    }

    /// The IR of one `op` at `bits`, built over two function parameters.
    ///
    /// Parameters rather than constants because a constant pair never becomes an instruction:
    /// `IRBuilder` folds it, which is what the conformance sweep reads and what the routing
    /// deliberately leaves alone. Only a runtime operand shows the lowering.
    fn arith_ir(op: &IntArithOp, bits: u32) -> String {
        let context = Context::create();
        let mut codegen = LLVMCodeGen::new(&context, "arith_shape");
        let ty = context
            .custom_width_int_type(NonZeroU32::new(bits).unwrap())
            .unwrap();
        let function = codegen.module.add_function(
            "subject",
            ty.fn_type(&[ty.into(), ty.into()], false),
            None,
        );
        codegen
            .builder
            .position_at_end(context.append_basic_block(function, "entry"));

        let a = function.get_nth_param(0).unwrap().into_int_value();
        let b = function.get_nth_param(1).unwrap().into_int_value();
        let result = codegen.build_int_arith(op, a, b, "r");
        codegen.builder.build_return(Some(&result)).unwrap();

        codegen
            .module
            .verify()
            .unwrap_or_else(|error| panic!("the emitted module does not verify: {error}"));
        codegen.get_ir()
    }

    #[test]
    fn a_wide_multiply_is_a_helper_call_and_a_narrow_one_is_not() {
        let wide = arith_ir(&IntArithOp::Mul, 256);
        assert!(
            wide.contains("call void @__int_mul"),
            "a 256-bit multiply must reach the runtime helper:\n{wide}"
        );
        assert!(
            !wide.contains("mul i256"),
            "and must not also expand inline:\n{wide}"
        );

        // 128 is the last width LLVM lowers to a single `__multi3`, which the runtime archive
        // already defines. Routing it would buy nothing and cost a call.
        let narrow = arith_ir(&IntArithOp::Mul, INLINE_MUL_MAX_BITS);
        assert!(
            narrow.contains("mul i128"),
            "a 128-bit multiply stays LLVM's:\n{narrow}"
        );
        assert!(
            !narrow.contains("call void @__int_mul"),
            "and calls nothing -- the declaration is always in the module:\n{narrow}"
        );
    }

    #[test]
    fn a_wide_remainder_is_a_division_a_helper_call_and_a_subtraction() {
        for (op, division) in [
            (IntArithOp::URem, "udiv i256"),
            (IntArithOp::SRem, "sdiv i256"),
        ] {
            let ir = arith_ir(&op, 256);
            // The identity LLVM's own expansion uses, with only the multiply in it replaced --
            // which is what holds the remainders to a size that does not grow with the width.
            assert!(ir.contains(division), "{op:?} keeps LLVM's division:\n{ir}");
            assert!(
                ir.contains("call void @__int_mul"),
                "{op:?} multiplies through the helper:\n{ir}"
            );
            assert!(ir.contains("sub i256"), "{op:?} subtracts:\n{ir}");
            assert!(
                !ir.contains("rem i256"),
                "{op:?} must not leave a remainder for LLVM to expand:\n{ir}"
            );
        }
    }

    #[test]
    fn a_wide_shift_at_a_non_power_of_two_width_reduces_through_the_helper() {
        // `reduce_shift_count` emits a real `urem` off the power-of-two path, and at a wide width
        // that remainder carries the same expansion the shift's own operand was routed around.
        // The amount is a runtime value, so nothing folds it away.
        let odd = arith_ir(&IntArithOp::Shl, 192);
        assert!(
            odd.contains("call void @__int_mul"),
            "a 192-bit shift reduces its count through the helper:\n{odd}"
        );
        assert!(
            !odd.contains("rem i192"),
            "and leaves no wide remainder behind:\n{odd}"
        );

        // At a power of two the reduction is an `and`, so there is no remainder to route.
        let even = arith_ir(&IntArithOp::Shl, 256);
        assert!(
            !even.contains("call void @__int_mul"),
            "a 256-bit shift needs no helper:\n{even}"
        );
    }

    #[test]
    fn the_scratch_buffers_are_shared_and_sit_in_the_entry_block() {
        // Two blocks, with every multiply in the second one, because that is the only shape the
        // placement is visible in: in a single-block function the entry block _is_ where the call
        // is, and an alloca emitted beside the call would sit in the right place by accident.
        // A `body` block stands for a loop body, where allocating per visit is the actual hazard
        // against a 256 KB wasm stack.
        let context = Context::create();
        let mut codegen = LLVMCodeGen::new(&context, "scratch_placement");
        let ty = context
            .custom_width_int_type(NonZeroU32::new(256).unwrap())
            .unwrap();
        let function = codegen.module.add_function(
            "subject",
            ty.fn_type(&[ty.into(), ty.into()], false),
            None,
        );
        let entry = context.append_basic_block(function, "entry");
        let body = context.append_basic_block(function, "body");

        codegen.builder.position_at_end(entry);
        codegen.builder.build_unconditional_branch(body).unwrap();

        codegen.builder.position_at_end(body);
        let a = function.get_nth_param(0).unwrap().into_int_value();
        let b = function.get_nth_param(1).unwrap().into_int_value();
        let mut last = a;
        for _ in 0..3 {
            last = codegen.build_int_arith(&IntArithOp::Mul, a, b, "r");
        }
        codegen.builder.build_return(Some(&last)).unwrap();
        codegen.module.verify().expect("the module verifies");

        let ir = codegen.get_ir();
        let (entry_text, body_text) = ir
            .split_once("body:")
            .expect("the subject has a body block");

        // Three slots for the helper's three pointers, and three call sites at one width share
        // them: an alloca per site would multiply the stack cost by the site count.
        assert_eq!(
            ir.matches("alloca").count(),
            3,
            "three call sites at one width want three buffers:\n{ir}"
        );
        assert_eq!(
            ir.matches("@__int_mul").count(),
            4,
            "three calls and one declaration:\n{ir}"
        );

        // And all three are in the entry block, where they are executed once per activation.
        assert_eq!(
            entry_text.matches("alloca").count(),
            3,
            "every buffer belongs to the entry block:\n{ir}"
        );
        assert_eq!(
            body_text.matches("alloca").count(),
            0,
            "and none to the block holding the calls:\n{ir}"
        );
    }

    #[test]
    fn a_routed_operation_keeps_its_own_source_location() {
        // `wide_scratch_slot` positions the builder into the entry block to place its allocas, and
        // positioning _before an instruction_ carries that instruction's debug location with it.
        //
        // Left alone, the store, the call and the load that follow are all attributed to the entry
        // block rather than to the operation, and only for the first multiply of each width in a
        // function
        let context = Context::create();
        let mut codegen = LLVMCodeGen::new(&context, "routed_location");
        let ty = context
            .custom_width_int_type(NonZeroU32::new(256).unwrap())
            .unwrap();
        let function = codegen.module.add_function(
            "subject",
            ty.fn_type(&[ty.into(), ty.into()], false),
            None,
        );

        let file = codegen.debug_builder.create_file("subject.nr", "src");
        let signature = codegen
            .debug_builder
            .create_subroutine_type(file, None, &[], 0);
        let subprogram = codegen.debug_builder.create_function(
            file.as_debug_info_scope(),
            "subject",
            None,
            file,
            1,
            signature,
            false,
            true,
            1,
            0,
            false,
        );
        function.set_subprogram(subprogram);
        let scope = subprogram.as_debug_info_scope();
        let entry_line = 10;
        let body_line = 20;

        let entry = context.append_basic_block(function, "entry");
        let body = context.append_basic_block(function, "body");

        codegen.builder.position_at_end(entry);
        codegen.builder.set_current_debug_location(
            codegen
                .debug_builder
                .create_debug_location(&context, entry_line, 1, scope, None),
        );
        codegen.builder.build_unconditional_branch(body).unwrap();

        codegen.builder.position_at_end(body);
        codegen.builder.set_current_debug_location(
            codegen
                .debug_builder
                .create_debug_location(&context, body_line, 1, scope, None),
        );
        let a = function.get_nth_param(0).unwrap().into_int_value();
        let b = function.get_nth_param(1).unwrap().into_int_value();
        let product = codegen.build_int_arith(&IntArithOp::Mul, a, b, "r");
        codegen.builder.build_return(Some(&product)).unwrap();
        codegen.module.verify().expect("the module verifies");

        let ir = codegen.get_ir();

        // The metadata node each `!DILocation` line declares, so a `!dbg` can be read back.
        let line_of = |node: &str| -> Option<u32> {
            let declaration = ir
                .lines()
                .find(|line| line.starts_with(&format!("{node} = !DILocation")))?;
            let after = declaration.split("line: ").nth(1)?;
            after.split(',').next()?.parse().ok()
        };
        let located = |needle: &str| -> u32 {
            let instruction = ir
                .lines()
                .find(|line| line.contains(needle))
                .unwrap_or_else(|| panic!("no {needle} in:\n{ir}"));
            let node = instruction
                .rsplit("!dbg ")
                .next()
                .unwrap_or_else(|| panic!("{needle} carries no !dbg:\n{ir}"))
                .trim();
            line_of(node).unwrap_or_else(|| panic!("{node} is not a DILocation:\n{ir}"))
        };

        assert_eq!(
            located("call void @__int_mul"),
            body_line,
            "the helper call belongs to the multiply, not to the entry block:\n{ir}"
        );
        assert_eq!(
            located("store i256"),
            body_line,
            "and so does the store that feeds it:\n{ir}"
        );
        assert_eq!(
            codegen
                .builder
                .get_current_debug_location()
                .map(|location| location.get_line()),
            Some(body_line),
            "the builder is handed back where it was, with the location it had"
        );
    }

    /// Compiles `subject(operands: ptr) -> i64` around one `bits`-wide `kind`, takes it all the way
    /// to a wasm modulethen instantiates it, writes two operands into its memory and calls it.
    ///
    /// This is the **only** check the routed lowering has as routing is suppressed for a constant
    /// pair, so the conformance sweep beside it folds the direct lowering every time and never sees
    /// the one built around a call.
    ///
    /// Importantly the operands are **loaded from memory** instead of widened from `i64` params. A
    /// parameter widened into a `bits`-wide operand leaves most of that operand's bits known-zero,
    /// and the mid-end narrows the multiply back to something it can do inline. That makes this
    /// test agree with itself under _any_ routing, but the routing is what we want to check. A load
    /// is opaque, which is also how a wide value reaches a multiply in a real program.
    ///
    /// The digest carries the **top** word and not only the low one, because the low limb is the
    /// thing computed correctly by every limb miscount. The low word is in it because a remainder
    /// is smaller than its divisor and its top word can be a genuine zero.
    fn routed_op_through_wasm(
        kind: &IntArithOp,
        bits: u32,
        opts_for: fn(std::path::PathBuf) -> WasmCompileOpts,
    ) -> u64 {
        let limbs = bits.div_ceil(HOST_LIMB_BITS as u32);
        let operand_stride = limbs * (HOST_LIMB_BITS as u32 / 8);

        let context = Context::create();
        let mut codegen = LLVMCodeGen::new(&context, "wide_engine_acceptance");
        let i64_type = context.i64_type();
        let ptr_type = context.ptr_type(AddressSpace::default());
        let subject = codegen.module.add_function(
            "subject",
            i64_type.fn_type(&[ptr_type.into()], false),
            None,
        );
        codegen
            .builder
            .position_at_end(context.append_basic_block(subject, "entry"));

        let wide = context
            .custom_width_int_type(NonZeroU32::new(bits).unwrap())
            .unwrap();
        let base = subject.get_nth_param(0).unwrap().into_pointer_value();
        let operand = |index: u32, name: &str| {
            let slot = if index == 0 {
                base
            } else {
                unsafe {
                    codegen.builder.build_in_bounds_gep(
                        context.i8_type(),
                        base,
                        &[context
                            .i32_type()
                            .const_int(u64::from(index * operand_stride), false)],
                        name,
                    )
                }
                .unwrap()
            };
            let load = codegen.builder.build_load(wide, slot, name).unwrap();
            inkwell::values::BasicValue::as_instruction_value(&load)
                .expect("a load is an instruction")
                .set_alignment(HOST_LIMB_BITS as u32 / 8)
                .expect("eight is a power of two");
            load.into_int_value()
        };
        let a = operand(0, "a");
        let b = operand(1, "b");

        let result = codegen.build_int_arith(kind, a, b, "result");
        // A constant amount, so this reduction folds and adds no routing of its own.
        let top_of = wide.const_int(u64::from(bits - 64), false);
        let shifted = codegen.build_int_arith(&IntArithOp::UShr, result, top_of, "top");
        let high = codegen.widen_or_trunc_int(shifted, 64, "high");
        let low = codegen.widen_or_trunc_int(result, 64, "low");
        let answer = codegen.builder.build_xor(low, high, "answer").unwrap();
        codegen.builder.build_return(Some(&answer)).unwrap();
        codegen.module.verify().expect("the module verifies");

        // `compile_to_wasm` exports whatever is in here, and wasm-ld garbage-collects the rest.
        codegen.entry_symbols = vec!["subject".to_string()];

        let dir = tempfile::tempdir().expect("a temporary directory");
        let wasm_path = dir.path().join("subject.wasm");
        codegen.compile_to_wasm(&wasm_path, opts_for(crate::wasm_runtime::locate_or_build()));

        let engine = wasmtime::Engine::default();
        let module = wasmtime::Module::from_file(&engine, &wasm_path)
            .expect("the wasm engine accepts the module");
        let mut store = wasmtime::Store::new(&engine, ());

        // The module is linked with `--import-memory`, so the host supplies one; its declared
        // minimum is read back rather than guessed, the stack and static data being the module's
        // business and not this test's. Anything else it imports is a runtime symbol the archive
        // failed to define, which `--allow-undefined` would otherwise have hidden until here.
        let mut memory = None;
        let mut imports: Vec<wasmtime::Extern> = Vec::new();
        for import in module.imports() {
            let wasmtime::ExternType::Memory(memory_type) = import.ty() else {
                panic!(
                    "the module imports {}::{}, which the runtime archive should have defined",
                    import.module(),
                    import.name()
                );
            };
            let created = wasmtime::Memory::new(&mut store, memory_type)
                .expect("a memory for the module's import");
            memory = Some(created);
            imports.push(created.into());
        }
        let memory = memory.expect("the module imports its memory");

        let instance = wasmtime::Instance::new(&mut store, &module, &imports)
            .expect("the wasm engine instantiates the module");

        // The operands go after the module's own static data, whose end the linker exports.
        let wasmtime::Val::I32(data_end) = instance
            .get_global(&mut store, "__data_end")
            .expect("wasm-ld exports __data_end")
            .get(&mut store)
        else {
            panic!("__data_end is not an i32");
        };
        let operands_at = (data_end as usize).next_multiple_of(16);
        let needed = operands_at + 2 * operand_stride as usize;
        if memory.data_size(&store) < needed {
            let pages = (needed - memory.data_size(&store)).div_ceil(64 * 1024) as u64;
            memory
                .grow(&mut store, pages)
                .expect("room for the operands");
        }
        for (index, word) in [SUBJECT_A, SUBJECT_B].into_iter().enumerate() {
            let bytes = operand_bytes(bits as usize, word);
            memory
                .write(&mut store, operands_at + index * bytes.len(), &bytes)
                .expect("the operands are written into the module's memory");
        }

        instance
            .get_typed_func::<i32, u64>(&mut store, "subject")
            .expect("the subject is exported")
            .call(&mut store, operands_at as i32)
            .expect("the subject runs")
    }

    /// The two seeds the subject's operands are built from.
    const SUBJECT_A: u64 = 0xDEAD_BEEF_1234_5678;
    const SUBJECT_B: u64 = 0x0FED_CBA9_8765_4321;

    /// One operand: `seed` in every limb, so the product's every column is a real one.
    ///
    /// A sparse operand is what lets the mid-end narrow the multiply, and a sparse one is also
    /// what an off-by-one-limb bug can get right by accident.
    fn operand_pattern(bits: usize, seed: u64) -> IntBits {
        let limbs = vec![seed; IntBits::limbs_for_bits(bits)];
        IntBits::from_limbs(bits, &limbs)
    }

    /// The same operand as the little-endian bytes the subject loads it from.
    fn operand_bytes(bits: usize, seed: u64) -> Vec<u8> {
        operand_pattern(bits, seed)
            .limbs()
            .iter()
            .flat_map(|limb| limb.to_le_bytes())
            .collect()
    }

    /// The model operation an emitted `kind` is held to.
    ///
    /// A left shift is one map on the bit pattern whichever way the operands are read, which is why
    /// the model has a single `Shl` and `int_arith_op` ignores the sign for it.
    fn model_op(kind: &IntArithOp) -> IntOp {
        match kind {
            IntArithOp::Mul => IntOp::UMul,
            IntArithOp::URem => IntOp::URem,
            IntArithOp::SRem => IntOp::SRem,
            IntArithOp::Shl => IntOp::Shl,
            other => unreachable!("{other:?} has no engine test"),
        }
    }

    /// What the model says the subject answers at `bits`, digested the way the subject digests it.
    fn expected_digest(kind: &IntArithOp, bits: usize) -> u64 {
        let result = residue(
            model_op(kind),
            &operand_pattern(bits, SUBJECT_A),
            &operand_pattern(bits, SUBJECT_B),
        )
        .expect("the model answers for every operation with an engine test");
        result.limbs()[0] ^ result.shifted_right(bits - 64).limbs()[0]
    }

    #[test]
    fn a_wide_multiply_survives_the_engine_and_computes_the_model_s_answer() {
        // 16384 is the width at which LLVM's own expansion is refused by both wasmtime and V8,
        // so a green run there is the whole objection to a wide multiply answered rather than
        // avoided. 1000 is neither a limb multiple nor a power of two, which is where the limb
        // count and the shift-count reduction are both at their least forgiving.
        for bits in [16384, 1000] {
            assert_eq!(
                routed_op_through_wasm(&IntArithOp::Mul, bits, WasmCompileOpts::fast),
                expected_digest(&IntArithOp::Mul, bits as usize),
                "the wasm lane disagrees with the model at {bits} bits"
            );
        }
    }

    #[test]
    fn the_routed_lowering_agrees_with_the_model_through_wasm() {
        for kind in [IntArithOp::URem, IntArithOp::SRem, IntArithOp::Shl] {
            assert_eq!(
                routed_op_through_wasm(&kind, 1000, WasmCompileOpts::fast),
                expected_digest(&kind, 1000),
                "the wasm lane disagrees with the model for {kind:?} at 1000 bits"
            );
        }
    }

    #[test]
    #[ignore = "LLVM's own `udiv i16383` expansion takes about nineteen seconds to compile"]
    fn a_wide_remainder_reaches_the_engine_at_the_cap() {
        assert_eq!(
            routed_op_through_wasm(&IntArithOp::URem, 16383, WasmCompileOpts::fast),
            expected_digest(&IntArithOp::URem, 16383)
        );
    }

    #[test]
    fn the_release_configuration_builds_a_wide_multiply_too() {
        assert_eq!(
            routed_op_through_wasm(&IntArithOp::Mul, 16384, WasmCompileOpts::release),
            expected_digest(&IntArithOp::Mul, 16384)
        );
    }

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
                    Constant::int(64, 1),
                    Constant::int(64, 0),
                    Constant::int(64, 0),
                    Constant::int(64, 0),
                ],
            );
            let two = block.emit_struct_const(
                field_type,
                vec![
                    Constant::int(64, 2),
                    Constant::int(64, 0),
                    Constant::int(64, 0),
                    Constant::int(64, 0),
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

/// The LLVM backend's conformance relation to the normative model in `mavros-int-semantics`.
///
/// The backend emits IR rather than computing values, so it cannot be checked the way an
/// interpreter can. What it can be checked against is **LLVM's own constant folder**: build the
/// real lowering — [`LLVMCodeGen::build_int_arith`], not a mirror of it — with two constant
/// operands, and LLVM folds the instruction as it emits it, handing back the value that lowering
/// means. That composes the two decisions the backend actually makes, which instruction each
/// `IntArithOp` picks and how the shift count is masked, with LLVM's definition of the instruction
/// chosen.
///
/// The relation is the VM's: equal to [`residue`](mavros_int_semantics::residue) wherever the model
/// has an opinion, and total. It is vacuous on a zero divisor and a signed `INT_MIN / -1`.
///
/// Which `IntArithOp` an operation lowers to is not transcribed here either: the sweep calls
/// `hlssa_to_llssa::int_arith_op`, the same table the real lowering applies, reached through
/// the model's own
/// `IntOp -> ArithGroup` renaming. A copy of that table would only have checked itself.
///
/// Two constant operands are also the one shape [`routes_through_the_wide_multiply`] answers
/// `false` for, so what the folder reads is always the direct lowering. That is deliberate as a
/// folded wide product is worth more to the module than a call, and a call something the folder
/// can't answer for.
///
/// What this does **not** prove is that the emitted IR reaches a backend unchanged.
#[cfg(test)]
mod int_semantics_conformance {
    use inkwell::{context::Context, values::AnyValue};
    use mavros_int_semantics::{IntBits, IntOp, corners, mask, residue};

    use super::*;
    use crate::compiler::ssa::{hlssa::BinaryArithOpKind, hlssa_to_llssa::int_arith_op};

    /// The `IntArithOp` a given opcode lowers to.
    ///
    /// Delegates to the compiler's own table rather than restating it, so that this sweep covers
    /// the lowering's instruction choice as well as LLVM's definition of the instruction chosen.
    /// Note there is no `sshl`: a left shift is one map on the bit pattern, which is why
    /// `int_arith_op` ignores the sign for it just as the VM has no signed `cell_shl`.
    fn lowering(kind: BinaryArithOpKind) -> IntArithOp {
        int_arith_op(kind.group(), kind.is_signed())
    }

    /// The raw pattern a folded constant holds, or [`None`] if LLVM did not fold it to one.
    ///
    /// LLVM prints an `iN` constant as a **signed** decimal, so `i8 -1` is the pattern `0xFF`;
    /// parsing as `i128` and re-masking is what recovers the pattern. `i1` is the exception,
    /// printed as `true`/`false` — which is the same corner the model calls out, `bool` being the
    /// one width whose only negative value is `1`. A `poison` matches none of these and yields
    /// [`None`], which is the intended reading of it here.
    fn folded_pattern(value: IntValue<'_>, bits: usize) -> Option<u128> {
        let printed = value.print_to_string().to_string();
        let literal = printed.rsplit(' ').next()?;
        let signed: i128 = match literal {
            "true" => 1,
            "false" => 0,
            other => other.parse().ok()?,
        };
        Some((signed as u128) & mask(bits))
    }

    #[test]
    fn the_emitted_instructions_agree_with_the_model() {
        let context = Context::create();
        let mut codegen = LLVMCodeGen::new(&context, "int_semantics_conformance");

        // The builder has to be positioned somewhere before it will emit, even for operands it is
        // about to fold away. Nothing is ever read back out of this function.
        let scratch =
            codegen
                .module
                .add_function("scratch", context.void_type().fn_type(&[], false), None);
        codegen
            .builder
            .position_at_end(context.append_basic_block(scratch, "entry"));

        let mut checked = 0usize;
        let mut unfolded = Vec::new();

        // Driven from `BinaryArithOpKind`, the vocabulary being lowered _from_: the two are no
        // longer in bijection, so sweeping the model's sixteen would cover only one of
        // `UShl`/`SShl`. Shifts share the one width set, because `reduce_shift_count` emits a real
        // `urem` off the power-of-two path rather than an `and` that is a modulo only there.
        for kind in BinaryArithOpKind::ALL {
            let op = IntOp::from(kind);
            let sign = kind.sign();
            for &bits in corners::widths_for(sign.is_signed()) {
                let ty = context
                    .custom_width_int_type(NonZeroU32::new(bits as u32).unwrap())
                    .unwrap();
                let rhs_values = if op.is_shift() {
                    corners::shift_amounts(bits, bits)
                } else {
                    corners::values(bits)
                };

                for a in corners::values(bits) {
                    for b in &rhs_values {
                        let lhs = ty.const_int_arbitrary_precision(&[a as u64, (a >> 64) as u64]);
                        let rhs = ty.const_int_arbitrary_precision(&[*b as u64, (*b >> 64) as u64]);
                        let val = codegen.build_int_arith(&lowering(kind), lhs, rhs, "");

                        let Some(want) = residue(
                            op,
                            &IntBits::from_u128(bits, a),
                            &IntBits::from_u128(bits, *b),
                        )
                        .map(|v| u128::try_from(&v).expect("a narrow answer fits a host word")) else {
                            // The model declines; LLVM folds to `poison`. Nothing to compare,
                            // and nothing may crash getting here, which is the whole claim.
                            continue;
                        };

                        match folded_pattern(val, bits) {
                            Some(got) => {
                                assert_eq!(
                                    got, want,
                                    "{op:?}/{sign:?} at {bits} bits: {a:#x} {b:#x} folded to \
                                     {got:#x}, model says {want:#x}"
                                );
                                checked += 1;
                            }
                            // Recorded rather than asserted on the spot so that a folder that
                            // stopped folding shows up as one summary rather than one failure.
                            None => unfolded.push(format!("{op:?}/{sign:?} {bits} {a:#x} {b:#x}")),
                        }
                    }
                }
            }
        }

        assert!(
            unfolded.is_empty(),
            "LLVM did not fold {} constant operations, e.g. {:?}",
            unfolded.len(),
            &unfolded[..unfolded.len().min(5)]
        );

        // Without this an implementation that folded nothing would pass every assertion above.
        assert!(
            checked > 25_000,
            "the sweep only reached {checked} specified points"
        );
    }

    /// Whether LLVM folded `value` to exactly `want`, or [`None`] if it folded to no constant.
    ///
    /// Read as a comparison rather than as a value, which is what makes the wide half affordable.
    /// Printing a wide constant is a decimal conversion quadratic in the width, and pulling one
    /// out limb by limb leaves a uniqued constant per limb in the context for the whole test.
    /// Folding an `icmp eq` against the model's own answer costs one constant and one bit at any
    /// width. The price is that a failure has to fetch the operands separately to say what it saw,
    /// which is the right way round: that path runs once.
    fn folds_to(codegen: &LLVMCodeGen<'_>, value: IntValue<'_>, want: &IntBits) -> Option<bool> {
        // A `poison` is not a `ConstantInt`, and comparing against one folds to poison in turn,
        // so this is the same reading of an unfolded result the narrow half takes.
        if !value.is_const() {
            return None;
        }
        let expected = codegen.int_pattern(want);
        let equal = codegen
            .builder
            .build_int_compare(IntPredicate::EQ, value, expected, "")
            .unwrap();
        equal.get_zero_extended_constant().map(|bit| bit == 1)
    }

    #[test]
    fn the_emitted_instructions_agree_with_the_model_at_wide_widths() {
        let context = Context::create();
        let mut codegen = LLVMCodeGen::new(&context, "int_semantics_conformance_wide");
        let scratch =
            codegen
                .module
                .add_function("scratch", context.void_type().fn_type(&[], false), None);
        codegen
            .builder
            .position_at_end(context.append_basic_block(scratch, "entry"));

        let mut checked = 0usize;
        let mut suppressed = 0usize;
        let mut unfolded = Vec::new();

        for kind in BinaryArithOpKind::ALL {
            let op = IntOp::from(kind);
            let sign = kind.sign();

            // Every operation at every wide width, the signed ones included, rather than off
            // `wide_widths_for(sign)` -- which is empty for a signed sweep, so driving from it
            // would run zero cases and pass. `MAX_LOWERED_SIGNED_BITS` says which widths a
            // _lowering_ may emit, not which widths an instruction must be right at.
            for bits in corners::WIDE_WIDTHS {
                let routable = needs_the_wide_multiply(&lowering(kind), bits as u32);
                let mut folded_here = 0usize;

                // Both sides come from the shared generator so that "what a shift's right operand
                // is" stays a fact about the operation rather than one restated per sweep.
                let (lhs_values, rhs_values) = corners::wide_operands(op, bits);

                for a in &lhs_values {
                    for b in &rhs_values {
                        let lhs = codegen.int_pattern(a);
                        let rhs = codegen.int_pattern(b);
                        let val = codegen.build_int_arith(&lowering(kind), lhs, rhs, "");

                        let Some(want) = residue(op, a, b) else {
                            // The model declines; LLVM folds to `poison`. Nothing to compare, and
                            // nothing may crash getting here, which is the whole claim.
                            continue;
                        };

                        match folds_to(&codegen, val, &want) {
                            Some(true) => folded_here += 1,
                            Some(false) => panic!(
                                "{op:?}/{sign:?} at {bits} bits: {a:?} {b:?} did not fold to the \
                                 model's {want:?}"
                            ),
                            None => unfolded.push(format!("{op:?}/{sign:?} {bits} {a:?} {b:?}")),
                        }
                    }
                }

                checked += folded_here;

                // Read off the sweep rather than off `needs_the_wide_multiply` again, so that this
                // says something the body could falsify. A shape the routing would take, that the
                // sweep nevertheless reached and folded, is the suppression in
                // `routes_through_the_wide_multiply` working: a routed operation is a call, and a
                // call loads its result rather than folding, which `unfolded` would catch.
                if routable && folded_here > 0 {
                    suppressed += 1;
                }
            }
        }

        assert!(
            unfolded.is_empty(),
            "LLVM did not fold {} constant operations, e.g. {:?}",
            unfolded.len(),
            &unfolded[..unfolded.len().min(3)]
        );

        // `BinaryArithOpKind::ALL` lowers to `Mul` from both `UMul` and `SMul`, so the four kinds
        // reaching the three routed opcodes are counted once per wide width. What the routing does
        // emit at these shapes is not this relation's business at all -- see
        // `tests::the_routed_lowering_agrees_with_the_model_through_wasm`.
        assert_eq!(
            suppressed,
            4 * corners::WIDE_WIDTHS.len(),
            "a routable shape was not reached, or did not fold, at every wide width"
        );

        // Without this an implementation that folded nothing would pass every assertion above.
        assert!(
            checked > 25_000,
            "the sweep only reached {checked} specified points"
        );
    }
}
