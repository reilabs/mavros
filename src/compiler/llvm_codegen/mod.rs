//! LLVM Code Generation for Mavros SSA
//!
//! This module translates Mavros SSA representation into LLVM IR,
//! which can then be compiled to native code or WebAssembly.

mod runtime;
mod types;

use std::collections::HashMap;
use std::path::Path;

use inkwell::builder::Builder;
use inkwell::context::Context;
use inkwell::module::Module;
use inkwell::targets::{
    CodeModel, FileType, InitializationConfig, RelocMode, Target, TargetTriple,
};
use inkwell::types::{BasicMetadataTypeEnum, BasicType, BasicTypeEnum, FunctionType};
use inkwell::values::{
    BasicMetadataValueEnum, BasicValue, BasicValueEnum, FunctionValue, IntValue, PointerValue,
};
use inkwell::AddressSpace;
use inkwell::IntPredicate;
use inkwell::OptimizationLevel;

use crate::compiler::analysis::types::{FunctionTypeInfo, TypeInfo};
use crate::compiler::flow_analysis::{FlowAnalysis, CFG};
use crate::compiler::ir::r#type::{Type, TypeExpr};
use crate::compiler::ssa::{
    BinaryArithOpKind, Block, BlockId, CmpKind, Const, Function, FunctionId, Terminator, ValueId,
    SSA,
};
use crate::compiler::taint_analysis::ConstantTaint;

use self::runtime::Runtime;
use self::types::TypeConverter;

/// LLVM Code Generator for Mavros SSA
pub struct LLVMCodeGen<'ctx> {
    context: &'ctx Context,
    module: Module<'ctx>,
    builder: Builder<'ctx>,
    type_converter: TypeConverter<'ctx>,
    runtime: Runtime<'ctx>,
    value_map: HashMap<ValueId, BasicValueEnum<'ctx>>,
    block_map: HashMap<BlockId, inkwell::basic_block::BasicBlock<'ctx>>,
    function_map: HashMap<FunctionId, FunctionValue<'ctx>>,
    vm_ptr: Option<PointerValue<'ctx>>,
}

impl<'ctx> LLVMCodeGen<'ctx> {
    pub fn new(context: &'ctx Context, module_name: &str) -> Self {
        let module = context.create_module(module_name);
        let builder = context.create_builder();
        let type_converter = TypeConverter::new(context);
        let runtime = Runtime::new(context, &module);

        Self {
            context,
            module,
            builder,
            type_converter,
            runtime,
            value_map: HashMap::new(),
            block_map: HashMap::new(),
            function_map: HashMap::new(),
            vm_ptr: None,
        }
    }

    /// Compile SSA to LLVM IR
    pub fn compile(
        &mut self,
        ssa: &SSA<ConstantTaint>,
        flow_analysis: &FlowAnalysis,
        type_info: &TypeInfo<ConstantTaint>,
    ) {
        // First pass: declare all functions
        let main_id = ssa.get_main_id();
        for (fn_id, function) in ssa.iter_functions() {
            self.declare_function(*fn_id, function, type_info, main_id);
        }

        // Second pass: generate function bodies
        for (fn_id, function) in ssa.iter_functions() {
            let cfg = flow_analysis.get_function_cfg(*fn_id);
            let fn_type_info = type_info.get_function(*fn_id);
            self.compile_function(*fn_id, function, cfg, fn_type_info);
        }
    }

    /// Declare a function (create its signature without body)
    fn declare_function(
        &mut self,
        fn_id: FunctionId,
        function: &Function<ConstantTaint>,
        type_info: &TypeInfo<ConstantTaint>,
        main_id: FunctionId,
    ) {
        let fn_type_info = type_info.get_function(fn_id);
        let entry = function.get_entry();

        // Build parameter types - first parameter is always VM*
        let mut param_types: Vec<BasicMetadataTypeEnum> = Vec::new();
        let ptr_type = self.context.ptr_type(AddressSpace::default());
        param_types.push(ptr_type.into()); // VM* as first param

        // Add the regular function parameters - field elements passed directly as [4 x i64]
        for (_, tp) in entry.get_parameters() {
            param_types.push(self.type_converter.convert_type(tp).into());
        }

        // Build return type - for now, we'll return a struct if multiple values
        let return_types: Vec<BasicTypeEnum> = function
            .get_returns()
            .iter()
            .map(|tp| self.type_converter.convert_type(tp))
            .collect();

        let fn_type = if return_types.is_empty() {
            self.context.void_type().fn_type(&param_types, false)
        } else if return_types.len() == 1 {
            return_types[0].fn_type(&param_types, false)
        } else {
            // Multiple returns - use a struct
            let return_struct = self.context.struct_type(&return_types, false);
            return_struct.fn_type(&param_types, false)
        };

        // Export the entry point function as "mavros_main"
        let name = if fn_id == main_id {
            "mavros_main"
        } else {
            function.get_name()
        };
        let fn_value = self.module.add_function(name, fn_type, None);
        self.function_map.insert(fn_id, fn_value);
    }

    /// Compile a single function
    fn compile_function(
        &mut self,
        fn_id: FunctionId,
        function: &Function<ConstantTaint>,
        cfg: &CFG,
        type_info: &FunctionTypeInfo<ConstantTaint>,
    ) {
        // Clear per-function state
        self.value_map.clear();
        self.block_map.clear();

        let fn_value = self.function_map[&fn_id];
        let entry = function.get_entry();
        let entry_block_id = function.get_entry_id();

        // Create entry block first (LLVM requires entry block to be first)
        let entry_bb = self.context.append_basic_block(fn_value, &format!("block_{}", entry_block_id.0));
        self.block_map.insert(entry_block_id, entry_bb);

        // Create remaining basic blocks
        for (block_id, _) in function.get_blocks() {
            if *block_id != entry_block_id {
                let block_name = format!("block_{}", block_id.0);
                let bb = self.context.append_basic_block(fn_value, &block_name);
                self.block_map.insert(*block_id, bb);
            }
        }

        // Map entry block parameters to function arguments
        let entry_bb = self.block_map[&function.get_entry_id()];
        self.builder.position_at_end(entry_bb);

        // First parameter is always VM*
        self.vm_ptr = Some(fn_value.get_nth_param(0).unwrap().into_pointer_value());

        // Map SSA parameters to LLVM function arguments (starting from index 1)
        // Field elements are passed directly as [4 x i64] values
        for (i, (param_id, _param_type)) in entry.get_parameters().enumerate() {
            let param_value = fn_value.get_nth_param((i + 1) as u32).unwrap();
            self.value_map.insert(*param_id, param_value);
        }

        // Generate constants
        for (val_id, constant) in function.iter_consts() {
            let value = self.compile_const(constant);
            self.value_map.insert(*val_id, value);
        }

        // Track phi nodes that need incoming values: (target_block, param_index) -> PhiValue
        let mut phi_nodes: HashMap<(BlockId, usize), inkwell::values::PhiValue<'ctx>> = HashMap::new();

        // Generate code for each block in dominator order
        for block_id in cfg.get_domination_pre_order() {
            self.compile_block_with_phis(function, block_id, type_info, &mut phi_nodes);
        }

        // Now add incoming values to phi nodes
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
                    Terminator::JmpIf(_, true_target, false_target) => {
                        // JmpIf doesn't pass arguments to targets in this SSA
                    }
                    Terminator::Return(_) => {}
                }
            }
        }
    }

    /// Compile a block, tracking phi nodes for later
    fn compile_block_with_phis(
        &mut self,
        function: &Function<ConstantTaint>,
        block_id: BlockId,
        type_info: &FunctionTypeInfo<ConstantTaint>,
        phi_nodes: &mut HashMap<(BlockId, usize), inkwell::values::PhiValue<'ctx>>,
    ) {
        let block = function.get_block(block_id);
        let bb = self.block_map[&block_id];

        // If this isn't the entry block, we need to handle block parameters (phi nodes)
        if block_id != function.get_entry_id() {
            self.builder.position_at_end(bb);

            // Block parameters become phi nodes
            for (i, (param_id, param_type)) in block.get_parameters().enumerate() {
                let llvm_type = self.type_converter.convert_type(param_type);
                let phi = self.builder.build_phi(llvm_type, &format!("v{}", param_id.0)).unwrap();
                self.value_map.insert(*param_id, phi.as_basic_value());
                phi_nodes.insert((block_id, i), phi);
            }
        }

        self.builder.position_at_end(bb);

        // Generate instructions
        for instruction in block.get_instructions() {
            self.compile_instruction(instruction, type_info);
        }

        // Generate terminator
        if let Some(terminator) = block.get_terminator() {
            self.compile_terminator(function, block_id, terminator);
        }
    }

    /// Compile a constant value
    fn compile_const(&self, constant: &Const) -> BasicValueEnum<'ctx> {
        match constant {
            Const::U(bits, value) => {
                let int_type = self.context.custom_width_int_type(*bits as u32);
                int_type.const_int(*value as u64, false).into()
            }
            Const::Field(field_val) => {
                // Field is represented as [4 x i64] in Montgomery form
                self.runtime.const_field(field_val)
            }
            Const::WitnessRef(_) => {
                todo!("WitnessRef constants not yet supported in LLVM codegen")
            }
        }
    }

    /// Compile a single instruction
    ///
    /// Currently only supports operations needed for the `power` example:
    /// - BinaryArithOp (field mul, int add)
    /// - Cmp (for loop conditions)
    /// - Constrain (write constraints)
    /// - WriteWitness
    fn compile_instruction(
        &mut self,
        instruction: &crate::compiler::ssa::OpCode<ConstantTaint>,
        type_info: &FunctionTypeInfo<ConstantTaint>,
    ) {
        use crate::compiler::ssa::OpCode;

        match instruction {
            OpCode::BinaryArithOp {
                kind,
                result,
                lhs,
                rhs,
            } => {
                let lhs_val = self.value_map[lhs];
                let rhs_val = self.value_map[rhs];
                let result_type = type_info.get_value_type(*result);

                let result_val = match &result_type.expr {
                    TypeExpr::Field => {
                        self.compile_field_binop(*kind, lhs_val, rhs_val)
                    }
                    TypeExpr::U(_) => {
                        self.compile_int_binop(*kind, lhs_val, rhs_val)
                    }
                    _ => panic!("Unsupported type for binary op: {:?}", result_type),
                };

                self.value_map.insert(*result, result_val);
            }

            OpCode::Cmp { kind, result, lhs, rhs } => {
                let lhs_type = type_info.get_value_type(*lhs);
                assert!(
                    matches!(lhs_type.expr, TypeExpr::U(_)),
                    "Cmp only supports U types, got {:?}",
                    lhs_type
                );

                let lhs_val = self.value_map[lhs].into_int_value();
                let rhs_val = self.value_map[rhs].into_int_value();

                let predicate = match kind {
                    CmpKind::Lt => IntPredicate::ULT,
                    CmpKind::Eq => IntPredicate::EQ,
                };

                let cmp_result = self
                    .builder
                    .build_int_compare(predicate, lhs_val, rhs_val, &format!("v{}", result.0))
                    .unwrap();

                self.value_map.insert(*result, cmp_result.into());
            }

            OpCode::Constrain { a, b, c } => {
                // R1CS constraint: a * b = c
                // Write the three constraint values to VM output
                let a_val = self.value_map[a];
                let b_val = self.value_map[b];
                let c_val = self.value_map[c];
                let vm_ptr = self.vm_ptr.unwrap();

                self.runtime.write_a(&self.builder, vm_ptr, a_val);
                self.runtime.write_b(&self.builder, vm_ptr, b_val);
                self.runtime.write_c(&self.builder, vm_ptr, c_val);
            }

            OpCode::WriteWitness { result, value, .. } => {
                let val = self.value_map[value];
                let vm_ptr = self.vm_ptr.unwrap();

                self.runtime.write_witness(&self.builder, vm_ptr, val);

                if let Some(result_id) = result {
                    self.value_map.insert(*result_id, val);
                }
            }

            OpCode::Call { results, function, args } => {
                let callee = self.function_map[function];
                let vm_ptr = self.vm_ptr.unwrap();

                // Build call args: VM* first, then mapped args
                let mut call_args: Vec<BasicMetadataValueEnum> = Vec::new();
                call_args.push(vm_ptr.into());
                for arg in args {
                    call_args.push(self.value_map[arg].into());
                }

                let call_result = self.builder
                    .build_call(callee, &call_args, &format!("call_f{}", function.0))
                    .unwrap();

                // Map return values
                if results.len() == 1 {
                    if let Some(val) = call_result.try_as_basic_value().left() {
                        self.value_map.insert(results[0], val);
                    }
                } else if results.len() > 1 {
                    let ret_struct = call_result.try_as_basic_value().left()
                        .expect("Expected return value from multi-return call");
                    for (i, result_id) in results.iter().enumerate() {
                        let val = self.builder
                            .build_extract_value(ret_struct.into_struct_value(), i as u32, &format!("v{}", result_id.0))
                            .unwrap();
                        self.value_map.insert(*result_id, val.into());
                    }
                }
            }

            // All other opcodes are not yet supported
            _ => {
                panic!("Unsupported opcode in LLVM codegen: {:?}", instruction);
            }
        }
    }

    /// Compile integer binary operation (only Add supported for loop counters)
    fn compile_int_binop(
        &self,
        kind: BinaryArithOpKind,
        lhs: BasicValueEnum<'ctx>,
        rhs: BasicValueEnum<'ctx>,
    ) -> BasicValueEnum<'ctx> {
        let lhs_int = lhs.into_int_value();
        let rhs_int = rhs.into_int_value();

        let result = match kind {
            BinaryArithOpKind::Add => self.builder.build_int_add(lhs_int, rhs_int, "add").unwrap(),
            _ => panic!("Unsupported integer binary op: {:?}", kind),
        };

        result.into()
    }

    /// Compile field binary operation (only Mul supported)
    fn compile_field_binop(
        &self,
        kind: BinaryArithOpKind,
        lhs: BasicValueEnum<'ctx>,
        rhs: BasicValueEnum<'ctx>,
    ) -> BasicValueEnum<'ctx> {
        match kind {
            BinaryArithOpKind::Mul => self.runtime.field_mul(&self.builder, lhs, rhs),
            _ => panic!("Unsupported field binary op: {:?}", kind),
        }
    }

    /// Compile a terminator instruction
    fn compile_terminator(
        &mut self,
        function: &Function<ConstantTaint>,
        current_block_id: BlockId,
        terminator: &Terminator,
    ) {
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
                    // Multiple returns - pack into struct
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

    /// Get the compiled LLVM IR as a string
    pub fn get_ir(&self) -> String {
        self.module.print_to_string().to_string()
    }

    /// Write LLVM IR to a file
    pub fn write_ir(&self, path: &Path) {
        self.module.print_to_file(path).unwrap();
    }

    /// Compile to WebAssembly
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

        // First write to an object file
        let obj_path = path.with_extension("o");
        target_machine
            .write_to_file(&self.module, FileType::Object, &obj_path)
            .unwrap();

        // Build the wasm-runtime and get the path to the static library
        let runtime_lib = Self::build_wasm_runtime();

        // Link with wasm-ld to produce final WASM module with exports
        // Find wasm-ld: try LLVM prefix from env, then fall back to PATH
        let wasm_ld = std::env::var("LLVM_SYS_180_PREFIX")
            .map(|prefix| {
                let path = std::path::PathBuf::from(&prefix).join("bin").join("wasm-ld");
                if path.exists() {
                    path.to_string_lossy().to_string()
                } else {
                    "wasm-ld".to_string()
                }
            })
            .unwrap_or_else(|_| "wasm-ld".to_string());


        let output = Command::new(&wasm_ld)
            .args([
                "--no-entry",               // No entry point (we call main explicitly)
                "--export=mavros_main",     // Export main function
                "--import-memory",          // Import memory from host
                "--allow-undefined",        // Allow undefined symbols (will be imported)
                "--stack-first",            // Place stack at start of memory (before data)
                "-z", "stack-size=65536",   // Explicit stack size (64KB)
                "--export=__data_end",      // Export data end marker for host to place buffers after
                "-o",
            ])
            .arg(path)
            .arg(&obj_path)
            .arg(&runtime_lib)
            .output()
            .expect(&format!("Failed to run wasm-ld (tried: {}). Make sure LLVM with wasm-ld is installed and either in PATH or LLVM_SYS_180_PREFIX is set.", wasm_ld));

        if !output.status.success() {
            eprintln!("wasm-ld stdout: {}", String::from_utf8_lossy(&output.stdout));
            eprintln!("wasm-ld stderr: {}", String::from_utf8_lossy(&output.stderr));
            panic!("wasm-ld failed with status: {}", output.status);
        }

        // Clean up object file
        std::fs::remove_file(&obj_path).ok();
    }

    /// Build the wasm-runtime crate for wasm32 and return path to the static library
    fn build_wasm_runtime() -> std::path::PathBuf {
        use std::process::Command;

        // Find workspace root using cargo metadata
        let metadata = cargo_metadata::MetadataCommand::new()
            .exec()
            .expect("Failed to get cargo metadata");

        let workspace_root = metadata.workspace_root.as_std_path();
        let wasm_runtime_dir = workspace_root.join("wasm-runtime");

        // Build the wasm-runtime for wasm32
        let output = Command::new("cargo")
            .current_dir(&wasm_runtime_dir)
            .args([
                "build",
                "--target", "wasm32-unknown-unknown",
                "--release",
            ])
            .output()
            .expect("Failed to run cargo build for wasm-runtime");

        if !output.status.success() {
            eprintln!("wasm-runtime cargo build stdout: {}", String::from_utf8_lossy(&output.stdout));
            eprintln!("wasm-runtime cargo build stderr: {}", String::from_utf8_lossy(&output.stderr));
            panic!("Failed to build wasm-runtime for wasm32");
        }

        // Return path to the static library
        let lib_path = workspace_root
            .join("target")
            .join("wasm32-unknown-unknown")
            .join("release")
            .join("libmavros_wasm_runtime.a");

        if !lib_path.exists() {
            eprintln!("wasm-runtime library not found at {:?}", lib_path);
            // List what's actually in the directory
            let dir = lib_path.parent().unwrap();
            if dir.exists() {
                eprintln!("Contents of {:?}:", dir);
                if let Ok(entries) = std::fs::read_dir(dir) {
                    for entry in entries.flatten() {
                        eprintln!("  {:?}", entry.file_name());
                    }
                }
            } else {
                eprintln!("Directory {:?} does not exist", dir);
            }
            panic!("wasm-runtime library not found");
        }

        lib_path
    }

    /// Get the LLVM module
    pub fn get_module(&self) -> &Module<'ctx> {
        &self.module
    }
}
