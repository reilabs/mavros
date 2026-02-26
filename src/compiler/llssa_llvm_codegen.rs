//! LLSSA → LLVM Code Generation
//!
//! Translates LLSSA into LLVM IR, which can then be compiled to WebAssembly.
//! Parallel to `llvm_codegen/` but operates on LLSSA + LLType instead of
//! HLSSA + Type. No TypeInfo is needed — types are explicit in the LLSSA ops.

use std::collections::HashMap;
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
    FieldArithOp, IntArithOp, IntCmpOp, LLFunction, LLOp, LLSSA, LLStruct, LLType,
};
use crate::compiler::ssa::{BlockId, FunctionId, Terminator, ValueId};

use crate::compiler::hlssa_to_llssa::field_elem_struct;

/// Number of 64-bit limbs in a BN254 field element.
const FIELD_LIMBS: u32 = 4;
const FIELD_SIZE: u32 = 32; // 4 × i64 = 32 bytes

/// VM struct layout (offsets in bytes):
const VM_WITNESS_PTR_OFFSET: u32 = 0;
const VM_A_PTR_OFFSET: u32 = 4;
const VM_B_PTR_OFFSET: u32 = 8;
const VM_C_PTR_OFFSET: u32 = 12;

/// LLSSA → LLVM Code Generator
pub struct LLSSACodeGen<'ctx> {
    context: &'ctx Context,
    module: Module<'ctx>,
    builder: Builder<'ctx>,
    value_map: HashMap<ValueId, BasicValueEnum<'ctx>>,
    block_map: HashMap<BlockId, inkwell::basic_block::BasicBlock<'ctx>>,
    function_map: HashMap<FunctionId, FunctionValue<'ctx>>,
    vm_ptr: Option<PointerValue<'ctx>>,
    // Runtime function declarations
    field_mul_fn: Option<FunctionValue<'ctx>>,
    write_witness_fn: Option<FunctionValue<'ctx>>,
    write_a_fn: Option<FunctionValue<'ctx>>,
    write_b_fn: Option<FunctionValue<'ctx>>,
    write_c_fn: Option<FunctionValue<'ctx>>,
}

impl<'ctx> LLSSACodeGen<'ctx> {
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
            write_witness_fn: None,
            write_a_fn: None,
            write_b_fn: None,
            write_c_fn: None,
        };

        codegen.declare_runtime_functions();
        codegen.define_write_functions();
        codegen
    }

    // ── Type conversion ─────────────────────────────────────────────────

    /// Convert an LLType to the corresponding LLVM type.
    fn convert_type(&self, ty: &LLType) -> BasicTypeEnum<'ctx> {
        match ty {
            LLType::Int(bits) => self.context.custom_width_int_type(*bits).into(),
            LLType::Ptr => self.context.ptr_type(AddressSpace::default()).into(),
            LLType::Struct(s) => self.convert_struct_type(s),
        }
    }

    /// Convert an LLStruct to an LLVM type.
    /// FIELD_ELEM (4×i64) is mapped to `[4 x i64]` for runtime compatibility.
    fn convert_struct_type(&self, s: &LLStruct) -> BasicTypeEnum<'ctx> {
        if *s == field_elem_struct() {
            self.field_type().into()
        } else {
            panic!("Unsupported struct type in LLSSA codegen: {}", s);
        }
    }

    /// The LLVM type for a field element: `[4 x i64]`.
    fn field_type(&self) -> inkwell::types::ArrayType<'ctx> {
        self.context.i64_type().array_type(FIELD_LIMBS)
    }

    // ── Runtime functions ───────────────────────────────────────────────

    fn declare_runtime_functions(&mut self) {
        let field_type = self.field_type();
        let field_mul_type = field_type.fn_type(&[field_type.into(), field_type.into()], false);
        self.field_mul_fn = Some(
            self.module
                .add_function("__field_mul", field_mul_type, None),
        );
    }

    fn define_write_functions(&mut self) {
        self.write_witness_fn =
            Some(self.define_write_fn("__write_witness", VM_WITNESS_PTR_OFFSET));
        self.write_a_fn = Some(self.define_write_fn("__write_a", VM_A_PTR_OFFSET));
        self.write_b_fn = Some(self.define_write_fn("__write_b", VM_B_PTR_OFFSET));
        self.write_c_fn = Some(self.define_write_fn("__write_c", VM_C_PTR_OFFSET));
    }

    fn define_write_fn(&self, name: &str, ptr_offset: u32) -> FunctionValue<'ctx> {
        let void_type = self.context.void_type();
        let ptr_type = self.context.ptr_type(AddressSpace::default());
        let field_type = self.field_type();
        let i32_type = self.context.i32_type();

        let fn_type = void_type.fn_type(&[ptr_type.into(), field_type.into()], false);
        let function = self
            .module
            .add_function(name, fn_type, Some(Linkage::Internal));

        let entry = self.context.append_basic_block(function, "entry");
        let builder = self.context.create_builder();
        builder.position_at_end(entry);

        let vm_ptr = function.get_nth_param(0).unwrap().into_pointer_value();
        let value = function.get_nth_param(1).unwrap().into_array_value();

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

        let i8_type = self.context.i8_type();
        let new_ptr = unsafe {
            builder
                .build_gep(
                    i8_type,
                    write_ptr,
                    &[i32_type.const_int(FIELD_SIZE as u64, false)],
                    "new_ptr",
                )
                .unwrap()
        };

        builder.build_store(write_pos_ptr, new_ptr).unwrap();
        builder.build_return(None).unwrap();

        function
    }

    // ── Compilation entry point ─────────────────────────────────────────

    /// Compile LLSSA to LLVM IR.
    pub fn compile(&mut self, llssa: &LLSSA, flow_analysis: &FlowAnalysis) {
        let main_id = llssa.get_main_id();

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
                let int_type = self.context.custom_width_int_type(*bits);
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
                    IntArithOp::And => self.builder.build_and(lhs, rhs, name).unwrap(),
                    IntArithOp::Or => self.builder.build_or(lhs, rhs, name).unwrap(),
                    IntArithOp::Xor => self.builder.build_xor(lhs, rhs, name).unwrap(),
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
                            .left()
                            .expect("field_mul should return a value")
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
                // For FIELD_ELEM, build a [4 x i64] via insertvalue chain
                assert_eq!(
                    *struct_type,
                    field_elem_struct(),
                    "Only FIELD_ELEM struct supported"
                );
                let arr_type = self.field_type();
                let mut agg = arr_type.get_undef();
                for (i, field_id) in fields.iter().enumerate() {
                    let field_val = self.value_map[field_id];
                    agg = self
                        .builder
                        .build_insert_value(agg, field_val, i as u32, "mk")
                        .unwrap()
                        .into_array_value();
                }
                self.value_map.insert(*result, agg.into());
            }

            LLOp::ExtractField {
                result,
                value,
                struct_type,
                field,
            } => {
                assert_eq!(
                    *struct_type,
                    field_elem_struct(),
                    "Only FIELD_ELEM struct supported"
                );
                let agg = self.value_map[value].into_array_value();
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
                    if let Some(val) = call_result.try_as_basic_value().left() {
                        self.value_map.insert(results[0], val);
                    }
                } else if results.len() > 1 {
                    let ret_struct = call_result
                        .try_as_basic_value()
                        .left()
                        .expect("Expected return value from multi-return call");
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

            // FieldFromLimbs: identity at LLVM level — already [4 x i64]
            LLOp::FieldFromLimbs { result, limbs } => {
                let val = self.value_map[limbs];
                self.value_map.insert(*result, val);
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
