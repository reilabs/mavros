//! The driver API for the compilation pipeline.

use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
};

use ark_ff::AdditiveGroup as _;
use noirc_frontend::{
    debug::DebugInstrumenter,
    monomorphization::{Monomorphizer, debug_types::DebugTypeTracker},
};
use tracing::info;

use crate::{
    Project,
    compiler::{
        Field,
        analysis::{
            flow_analysis::FlowAnalysis, types::Types,
            witness_taint_inference::WitnessTaintInference,
        },
        codegen::{
            CodeGenOptions,
            bytecode::CodeGen,
            hlssa_to_r1cs::{R1CGen, R1CS},
            llssa_to_llvm::WasmCompileOpts,
        },
        pass_manager::PassManager,
        passes::{
            common_subexpression_elimination::CSE,
            dead_code_elimination::{self, DCE},
            deduplicate_phis::DeduplicatePhis,
            defunctionalize::Defunctionalize,
            elide_tuples::ElideTuples,
            fix_double_jumps::FixDoubleJumps,
            instruction_lowering::InstructionLowering,
            lower_guards::LowerGuards,
            lower_map_casts::LowerMapCasts,
            mem2reg::Mem2Reg,
            prepare_entry_point::PrepareEntryPoint,
            rc_insertion::RCInsertion,
            remove_unreachable_blocks::RemoveUnreachableBlocks,
            remove_unreachable_functions::RemoveUnreachableFunctions,
            simplifier::Simplifier,
            simplify_asserts::SimplifyAsserts,
            sparse_conditional_constant_propagation::SCCP,
            specializer::Specializer,
            strip_witness_of::StripWitnessOf,
            trivial_phi_elimination::TrivialPhiElimination,
            witness_lowering::WitnessLowering,
            witness_write_to_fresh::WitnessWriteToFresh,
            witness_write_to_void::WitnessWriteToVoid,
        },
        ssa::{DefaultSSAAnnotator, hlssa::HLSSA},
        untaint_control_flow::UntaintControlFlow,
    },
};

pub struct Driver {
    project: Project,
    initial_ssa: Option<HLSSA>,
    static_struct_access_ssa: Option<HLSSA>,
    monomorphized_ssa: Option<HLSSA>,
    witness_spilled_ssa: Option<HLSSA>,
    r1cs_ssa: Option<HLSSA>,
    base_witgen_ssa: Option<HLSSA>,
    abi: Option<noirc_abi::Abi>,
    draw_cfg: bool,
    main_is_unconstrained: bool,
}

#[derive(Debug)]
pub enum Error {
    NoirCompilerError(Vec<noirc_errors::reporter::CustomDiagnostic>),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::NoirCompilerError(diagnostics) => {
                write!(f, "Noir compiler error ({} diagnostics)", diagnostics.len())
            }
        }
    }
}

impl std::error::Error for Error {}

impl Driver {
    pub fn new(project: Project, draw_cfg: bool) -> Self {
        let dir = project.get_only_crate().root_dir.join("mavros_debug");
        if dir.exists() {
            fs::remove_dir_all(&dir).unwrap();
        }
        fs::create_dir(&dir).unwrap();
        Self {
            project,
            initial_ssa: None,
            static_struct_access_ssa: None,
            monomorphized_ssa: None,
            witness_spilled_ssa: None,
            r1cs_ssa: None,
            base_witgen_ssa: None,
            abi: None,
            draw_cfg,
            main_is_unconstrained: false,
        }
    }

    pub fn get_debug_output_dir(&self) -> PathBuf {
        self.project.package_root().join("mavros_debug")
    }

    /// Root directory of the package being compiled (the workspace member's
    /// directory, not the workspace root). `Prover.toml` is read from here.
    pub fn package_root(&self) -> &std::path::Path {
        self.project.package_root()
    }

    #[tracing::instrument(skip_all)]
    pub fn run_noir_compiler(&mut self) -> Result<(), Error> {
        let (mut context, crate_id) = nargo::prepare_package(
            self.project.file_manager(),
            self.project.parsed_files(),
            self.project.get_only_crate(),
        );
        noirc_driver::check_crate(
            &mut context,
            crate_id,
            &noirc_driver::CompileOptions::default(),
        )
        .map_err(Error::NoirCompilerError)?;

        let main = context.get_main_function(context.root_crate_id()).unwrap();
        let debug_type_tracker =
            DebugTypeTracker::build_from_debug_instrumenter(&DebugInstrumenter::default());
        let mut monomorphizer =
            Monomorphizer::new(&mut context.def_interner, debug_type_tracker, false);
        monomorphizer.compile_main(main).unwrap();
        monomorphizer.process_queue().unwrap();
        let program = monomorphizer.into_program();

        self.abi = Some(noirc_driver::gen_abi(
            &context,
            &main,
            program.return_visibility(),
            BTreeMap::default(),
        ));

        // Convert monomorphized AST directly to SSA, bypassing Noir's SSA generation
        let (ssa, main_is_unconstrained) = HLSSA::from_program(&program);
        self.initial_ssa = Some(ssa);
        self.main_is_unconstrained = main_is_unconstrained;

        fs::write(
            self.get_debug_output_dir().join("initial_ssa.txt"),
            self.initial_ssa
                .as_ref()
                .unwrap()
                .to_string(&DefaultSSAAnnotator),
        )
        .unwrap();

        Ok(())
    }

    #[tracing::instrument(skip_all)]
    pub fn make_struct_access_static(&mut self) -> Result<(), Error> {
        let mut pass_manager = PassManager::new(
            "make_struct_access_static".to_string(),
            self.draw_cfg,
            vec![
                Box::new(Defunctionalize::new()),
                Box::new(PrepareEntryPoint::new(self.main_is_unconstrained)),
                // Eliminate all tuple types immediately after the entry point is prepared, so every
                // subsequent pass operates on tuple-free IR.
                Box::new(ElideTuples::new()),
                // Fold constants and prune statically-decided branches (e.g. monomorphized
                // generic dispatch) BEFORE pruning functions: calls in never-taken branches must
                // not keep their callees alive into witness type inference and untaint CF, which
                // can ICE on semantically-dead code they would otherwise have to type.
                Box::new(SCCP::new()),
                Box::new(RemoveUnreachableFunctions::new()),
                Box::new(RemoveUnreachableBlocks::new()),
                // Use preserve_blocks() to keep empty intermediate blocks intact.
                // TODO: Remove once untaint_control_flow handles multiple jumps into merge blocks.
                Box::new(DCE::new(dead_code_elimination::Config::preserve_blocks())),
            ],
        );

        pass_manager.set_debug_output_dir(self.get_debug_output_dir().clone());
        let mut ssa = self.initial_ssa.clone().unwrap();
        pass_manager.run(&mut ssa);
        self.static_struct_access_ssa = Some(ssa);
        Ok(())
    }

    #[tracing::instrument(skip_all)]
    pub fn monomorphize(&mut self) -> Result<(), Error> {
        let mut ssa = self.static_struct_access_ssa.clone().unwrap();

        // Mem2Reg + cleanup before WTI so witness inference sees clean SSA
        // and Guard never wraps promotable Store/Load.
        PassManager::new(
            "pre_wti".to_string(),
            self.draw_cfg,
            vec![
                Box::new(Mem2Reg::new()),
                // Mem2Reg promotes each scalarized leaf cell into its own block-parameter phi. For
                // an aggregate threaded through control flow that is mostly trivial phis (the same
                // value from every predecessor); collapse them before they reach WTI and codegen.
                Box::new(TrivialPhiElimination::new()),
                Box::new(RemoveUnreachableFunctions::new()),
            ],
        )
        .run(&mut ssa);

        let flow_analysis = FlowAnalysis::run(&ssa);

        if self.draw_cfg {
            flow_analysis.generate_images(
                self.get_debug_output_dir().join("initial_state"),
                &ssa,
                "initial state".to_string(),
            );
        }

        let mut witness_inference = WitnessTaintInference::new();
        witness_inference.run(&mut ssa, &flow_analysis);

        fs::write(
            self.get_debug_output_dir().join("monomorphized_ssa.txt"),
            ssa.to_string(&witness_inference),
        )
        .unwrap();

        let mut untaint_cf = UntaintControlFlow::new();
        self.monomorphized_ssa = Some(untaint_cf.run(ssa, &witness_inference));

        fs::write(
            self.get_debug_output_dir().join("untainted_ssa.txt"),
            self.monomorphized_ssa
                .as_ref()
                .unwrap()
                .to_string(&DefaultSSAAnnotator),
        )
        .unwrap();

        Ok(())
    }

    #[tracing::instrument(skip_all)]
    pub fn spill_witness(&mut self) -> Result<(), Error> {
        let mut pass_manager = PassManager::new(
            "witness_spilling".to_string(),
            self.draw_cfg,
            vec![
                Box::new(InstructionLowering::pure_guards()),
                Box::new(InstructionLowering::witness_memory_ops()),
                Box::new(FixDoubleJumps::new()),
                // Fold pure constants and prune constant-condition branches before the cleanup
                // rounds, so Simplifier/CSE/DCE work on the reduced CFG.
                Box::new(SCCP::new()),
                // Simplify → CSE → DCE, twice. The doubled rounds let
                // CSE-dedup expose new fold operands and folds expose new CSE
                // matches. Each Simplifier internally iterates to fixed point
                // for purely-algebraic folds.
                Box::new(Simplifier::new()),
                Box::new(CSE::pre_r1c()),
                Box::new(DCE::new(dead_code_elimination::Config::pre_r1c())),
                Box::new(Simplifier::new()),
                Box::new(CSE::pre_r1c()),
                Box::new(DCE::new(dead_code_elimination::Config::pre_r1c())),
                Box::new(CSE::pre_r1c()),
                Box::new(DeduplicatePhis::new()),
                Box::new(DCE::new(dead_code_elimination::Config::pre_r1c())),
                Box::new(FixDoubleJumps::new()),
                Box::new(SimplifyAsserts::new()),
                Box::new(DCE::new(dead_code_elimination::Config::pre_r1c())),
                Box::new(InstructionLowering::witness_array_access()),
                Box::new(InstructionLowering::witness_integer_ops()),
                // After the last pre-spilling lowering, run cleanup twice
                // back-to-back. The first round exposes folds/dedup opportunities
                // that the second round can then consume.
                Box::new(Simplifier::new()),
                Box::new(CSE::pre_r1c()),
                Box::new(DCE::new(dead_code_elimination::Config::pre_r1c())),
                Box::new(Simplifier::new()),
                Box::new(CSE::pre_r1c()),
                Box::new(DCE::new(dead_code_elimination::Config::pre_r1c())),
                Box::new(Specializer::new(5.0)),
                // Specialization exposes fresh constants (folded call arguments and branch
                // conditions); propagate them before the post-specialization cleanup.
                Box::new(SCCP::new()),
                Box::new(Simplifier::new()),
                Box::new(CSE::pre_r1c()),
                Box::new(DCE::new(dead_code_elimination::Config::pre_r1c())),
                Box::new(InstructionLowering::lookup_spilling()),
                Box::new(InstructionLowering::degree_spilling()),
                Box::new(Simplifier::new()),
                Box::new(CSE::pre_r1c()),
                Box::new(DCE::new(dead_code_elimination::Config::pre_r1c())),
                Box::new(Simplifier::new()),
                Box::new(CSE::pre_r1c()),
                Box::new(DCE::new(dead_code_elimination::Config::pre_r1c())),
                Box::new(RemoveUnreachableFunctions::new()),
                Box::new(FixDoubleJumps::new()),
            ],
        );

        pass_manager.set_debug_output_dir(self.get_debug_output_dir().clone());
        let mut ssa = self.monomorphized_ssa.clone().unwrap();
        pass_manager.run(&mut ssa);
        self.witness_spilled_ssa = Some(ssa);
        Ok(())
    }

    #[tracing::instrument(skip_all)]
    pub fn generate_r1cs(&mut self) -> Result<R1CS, Error> {
        let mut r1cs_ssa = self.witness_spilled_ssa.clone().unwrap();

        let mut r1cs_phase_1 = PassManager::new(
            "r1cs_phase_1".to_string(),
            self.draw_cfg,
            vec![
                Box::new(WitnessWriteToFresh::new()),
                Box::new(DCE::new(dead_code_elimination::Config::post_r1c())),
                Box::new(FixDoubleJumps::new()),
            ],
        );
        r1cs_phase_1.set_debug_output_dir(self.get_debug_output_dir().clone());
        r1cs_phase_1.run(&mut r1cs_ssa);

        let flow_analysis = FlowAnalysis::run(&r1cs_ssa);
        let type_info = Types::new().run(&r1cs_ssa, &flow_analysis);

        let mut r1cs_gen = R1CGen::new();
        r1cs_gen.run(&r1cs_ssa, &type_info);
        let r1cs = r1cs_gen.seal();
        let mut num_non_zero_terms = 0;
        for r1c in r1cs.constraints.iter() {
            for (_, coeff) in r1c.a.iter() {
                if *coeff != Field::ZERO {
                    num_non_zero_terms += 1;
                }
            }
            for (_, coeff) in r1c.b.iter() {
                if *coeff != Field::ZERO {
                    num_non_zero_terms += 1;
                }
            }
            for (_, coeff) in r1c.c.iter() {
                if *coeff != Field::ZERO {
                    num_non_zero_terms += 1;
                }
            }
        }
        self.r1cs_ssa = Some(r1cs_ssa);
        info!(
            message = %"R1CS generated",
            num_constraints = r1cs.constraints.len(),
            num_terms = num_non_zero_terms,
            algebraic_constraints = r1cs.constraints_layout.algebraic_size,
            tables_constraints = r1cs.constraints_layout.tables_data_size,
            lookups_constraints = r1cs.constraints_layout.lookups_data_size,
            total_witness = r1cs.witness_layout.size()

        );

        Ok(r1cs)
    }

    pub fn compile_witgen(&mut self, options: CodeGenOptions) -> Result<Vec<u64>, Error> {
        self.prepare_base_witgen_ssa();
        let ssa = self.base_witgen_ssa.as_ref().unwrap();

        let flow_analysis = FlowAnalysis::run(ssa);
        let type_info = Types::new().run(ssa, &flow_analysis);

        let codegen = CodeGen::new(options);
        let program = codegen.run(ssa, &flow_analysis, &type_info);
        fs::write(
            self.get_debug_output_dir().join("witgen_bytecode.txt"),
            format!("{}", program),
        )
        .unwrap();

        let binary = program.to_binary();

        info!(message = %"Witgen binary generated", binary_size = binary.len() * 8);

        Ok(binary)
    }

    pub fn compile_ad(&self) -> Result<Vec<u64>, Error> {
        let mut ssa = self.r1cs_ssa.clone().unwrap();
        let mut ad_pm = PassManager::new(
            "ad".to_string(),
            self.draw_cfg,
            vec![
                Box::new(WitnessLowering::new()),
                // Cleanup before lowering Map casts: dead hint computations
                // (e.g. strip-maps feeding unconstrained calls) are easy to
                // delete while still single instructions, but not once they
                // are expanded into loops with block-parameter cycles.
                Box::new(CSE::post_r1c()),
                Box::new(DCE::new(dead_code_elimination::Config::post_r1c())),
                Box::new(LowerMapCasts::new()),
                Box::new(CSE::post_r1c()),
                Box::new(DCE::new(dead_code_elimination::Config::post_r1c())),
                Box::new(RCInsertion::new()),
                Box::new(FixDoubleJumps::new()),
            ],
        );
        ad_pm.set_debug_output_dir(self.get_debug_output_dir().clone());
        ad_pm.run(&mut ssa);
        let flow_analysis = FlowAnalysis::run(&ssa);
        let type_info = Types::new().run(&ssa, &flow_analysis);

        let codegen = CodeGen::new(CodeGenOptions::default());
        let program = codegen.run(&ssa, &flow_analysis, &type_info);
        fs::write(
            self.get_debug_output_dir().join("ad_bytecode.txt"),
            format!("{}", program),
        )
        .unwrap();
        let binary = program.to_binary();

        info!(message = %"AD binary generated", binary_size = binary.len() * 8);

        Ok(binary)
    }

    pub fn abi(&self) -> &noirc_abi::Abi {
        self.abi.as_ref().unwrap()
    }

    #[tracing::instrument(skip_all)]
    fn prepare_base_witgen_ssa(&mut self) {
        if self.base_witgen_ssa.is_some() {
            return;
        }

        let mut ssa = self.witness_spilled_ssa.clone().unwrap();

        let mut pass_manager = PassManager::new(
            "base_witgen".to_string(),
            self.draw_cfg,
            vec![
                Box::new(LowerGuards::new()),
                Box::new(WitnessWriteToVoid::new()),
                Box::new(StripWitnessOf::new()),
                Box::new(DCE::new(dead_code_elimination::Config::post_r1c())),
                Box::new(RCInsertion::new()),
                Box::new(FixDoubleJumps::new()),
            ],
        );
        pass_manager.set_debug_output_dir(self.get_debug_output_dir().clone());
        pass_manager.run(&mut ssa);

        self.base_witgen_ssa = Some(ssa);
    }

    #[tracing::instrument(skip_all)]
    pub fn compile_llvm_targets(
        &mut self,
        emit_llvm: bool,
        r1cs: &R1CS,
        wasm_config: Option<(std::path::PathBuf, WasmCompileOpts)>,
        options: CodeGenOptions,
    ) -> Result<Option<String>, Error> {
        use crate::compiler::codegen::llssa_to_llvm::LLVMCodeGen;
        use crate::compiler::ssa::hlssa_to_llssa;
        use inkwell::context::Context;

        self.prepare_base_witgen_ssa();
        let ssa = self.base_witgen_ssa.as_ref().unwrap();

        let flow_analysis = FlowAnalysis::run(ssa);
        let type_info = Types::new().run(ssa, &flow_analysis);

        // Dump HLSSA just before lowering
        fs::write(
            self.get_debug_output_dir()
                .join("hlssa_before_lowering.txt"),
            ssa.to_string(&DefaultSSAAnnotator),
        )
        .unwrap();

        // Lower HLSSA → LLSSA
        let llssa = hlssa_to_llssa::lower_with_layout(
            ssa,
            &flow_analysis,
            &type_info,
            r1cs.witness_layout,
            r1cs.constraints_layout,
            options,
        );

        // Dump LLSSA after lowering
        fs::write(
            self.get_debug_output_dir().join("llssa_after_lowering.txt"),
            llssa.to_string(&DefaultSSAAnnotator),
        )
        .unwrap();

        // Compile LLSSA → LLVM
        let ll_flow_analysis = FlowAnalysis::run(&llssa);
        let context = Context::create();
        let mut codegen = LLVMCodeGen::new(&context, "mavros_module");
        codegen.compile(&llssa, &ll_flow_analysis);

        let llvm_ir = if emit_llvm {
            let ir = codegen.get_ir();
            fs::write(self.get_debug_output_dir().join("witgen.ll"), &ir).unwrap();
            info!(message = %"LLVM IR generated", ir_size = ir.len());
            Some(ir)
        } else {
            None
        };

        if let Some((wasm_path, wasm_opts)) = wasm_config {
            codegen.write_ir(&wasm_path.with_extension("ll"));
            codegen.compile_to_wasm(&wasm_path, wasm_opts);
            info!(message = %"WASM object generated", path = %wasm_path.display());
            self.write_wasm_metadata(&wasm_path, r1cs)?;
        }

        Ok(llvm_ir)
    }

    #[tracing::instrument(skip_all)]
    pub fn compile_ad_llvm_targets(
        &mut self,
        wasm_path: std::path::PathBuf,
        r1cs: &R1CS,
        wasm_opts: WasmCompileOpts,
    ) -> Result<(), Error> {
        use crate::compiler::codegen::llssa_to_llvm::LLVMCodeGen;
        use crate::compiler::ssa::hlssa_to_llssa;
        use inkwell::context::Context;

        // Prepare AD SSA: same pass pipeline as compile_ad()
        let mut ssa = self.r1cs_ssa.clone().unwrap();
        let mut ad_pm = PassManager::new(
            "ad_wasm".to_string(),
            self.draw_cfg,
            vec![
                Box::new(WitnessLowering::new()),
                // Cleanup before lowering Map casts: dead hint computations
                // (e.g. strip-maps feeding unconstrained calls) are easy to
                // delete while still single instructions, but not once they
                // are expanded into loops with block-parameter cycles.
                Box::new(CSE::post_r1c()),
                Box::new(DCE::new(dead_code_elimination::Config::post_r1c())),
                Box::new(LowerMapCasts::new()),
                Box::new(CSE::post_r1c()),
                Box::new(DCE::new(dead_code_elimination::Config::post_r1c())),
                Box::new(RCInsertion::new()),
                Box::new(FixDoubleJumps::new()),
            ],
        );
        ad_pm.set_debug_output_dir(self.get_debug_output_dir().clone());
        ad_pm.run(&mut ssa);

        let flow_analysis = FlowAnalysis::run(&ssa);
        let type_info = Types::new().run(&ssa, &flow_analysis);

        // Dump HLSSA just before lowering
        fs::write(
            self.get_debug_output_dir()
                .join("ad_hlssa_before_lowering.txt"),
            ssa.to_string(&DefaultSSAAnnotator),
        )
        .unwrap();

        // Lower HLSSA → LLSSA
        let llssa = hlssa_to_llssa::lower_with_layout(
            &ssa,
            &flow_analysis,
            &type_info,
            r1cs.witness_layout,
            r1cs.constraints_layout,
            CodeGenOptions::default(),
        );

        // Dump LLSSA after lowering
        fs::write(
            self.get_debug_output_dir()
                .join("ad_llssa_after_lowering.txt"),
            llssa.to_string(&DefaultSSAAnnotator),
        )
        .unwrap();

        // Compile LLSSA → LLVM
        let ll_flow_analysis = FlowAnalysis::run(&llssa);
        let context = Context::create();
        let mut codegen = LLVMCodeGen::new(&context, "mavros_ad_module");
        codegen.compile(&llssa, &ll_flow_analysis);

        codegen.write_ir(&wasm_path.with_extension("ll"));
        codegen.compile_to_wasm(&wasm_path, wasm_opts);
        info!(message = %"AD WASM object generated", path = %wasm_path.display());
        self.write_ad_wasm_metadata(&wasm_path, r1cs)?;

        Ok(())
    }

    /// Write AD WASM metadata JSON file
    fn write_ad_wasm_metadata(&self, wasm_path: &Path, r1cs: &R1CS) -> Result<(), Error> {
        let metadata = serde_json::json!({
            "witnessCount": r1cs.witness_layout.size(),
            "constraintCount": r1cs.constraints.len(),
        });

        let metadata_path = format!("{}.meta.json", wasm_path.display());
        fs::write(
            &metadata_path,
            serde_json::to_string_pretty(&metadata).unwrap(),
        )
        .unwrap();

        info!(message = %"AD WASM metadata generated", path = %metadata_path);

        Ok(())
    }

    /// Write WASM metadata JSON file
    fn write_wasm_metadata(&self, wasm_path: &Path, r1cs: &R1CS) -> Result<(), Error> {
        let abi = self.abi.as_ref().unwrap();

        // Build parameter info
        let mut parameters = Vec::new();
        for param in &abi.parameters {
            let element_count = count_abi_type_elements(&param.typ);
            parameters.push(serde_json::json!({
                "name": param.name,
                "elementCount": element_count
            }));
        }

        let metadata = serde_json::json!({
            "witnessCount": r1cs.witness_layout.size(),
            "constraintCount": r1cs.constraints.len(),
            "parameters": parameters
        });

        let metadata_path = format!("{}.meta.json", wasm_path.display());
        fs::write(
            &metadata_path,
            serde_json::to_string_pretty(&metadata).unwrap(),
        )
        .unwrap();

        info!(message = %"WASM metadata generated", path = %metadata_path);

        Ok(())
    }
}

/// Count the number of field elements in an ABI type
fn count_abi_type_elements(typ: &noirc_abi::AbiType) -> usize {
    use noirc_abi::AbiType;
    match typ {
        AbiType::Field => 1,
        AbiType::Integer { .. } => 1,
        AbiType::Boolean => 1,
        AbiType::String { length } => *length as usize,
        AbiType::Array { length, typ } => (*length as usize) * count_abi_type_elements(typ),
        AbiType::Struct { fields, .. } => {
            fields.iter().map(|(_, t)| count_abi_type_elements(t)).sum()
        }
        AbiType::Tuple { fields } => fields.iter().map(count_abi_type_elements).sum(),
    }
}
