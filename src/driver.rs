use std::{collections::BTreeMap, fs, path::PathBuf};

use ark_ff::AdditiveGroup as _;
use tracing::info;

use crate::{
    Project,
    compiler::{
        Field,
        analysis::types::Types,
        codegen::CodeGen,
        flow_analysis::FlowAnalysis,
        ir::r#type::Empty,
        monomorphization::Monomorphization,
        pass_manager::PassManager,
        passes::{
            arithmetic_simplifier::ArithmeticSimplifier, witness_to_ref::WitnessToRef, common_subexpression_elimination::CSE, condition_propagation::ConditionPropagation, dead_code_elimination::{self, DCE}, deduplicate_phis::DeduplicatePhis, explicit_witness::ExplicitWitness, fix_double_jumps::FixDoubleJumps, mem2reg::Mem2Reg, prepare_entry_point::PrepareEntryPoint, pull_into_assert::PullIntoAssert, rc_insertion::RCInsertion, remove_unreachable_blocks::RemoveUnreachableBlocks, specializer::Specializer, struct_access_simplifier::MakeStructAccessStatic, witness_write_to_fresh::WitnessWriteToFresh, witness_write_to_void::WitnessWriteToVoid
        },
        r1cs_gen::{R1CGen, R1CS},
        ssa::{DefaultSsaAnnotator, SSA},
        taint_analysis::{ConstantTaint, TaintAnalysis},
        untaint_control_flow::UntaintControlFlow,
    },
};

pub struct Driver {
    project: Project,
    initial_ssa: Option<SSA<Empty>>,
    static_struct_access_ssa: Option<SSA<Empty>>,
    monomorphized_ssa: Option<SSA<ConstantTaint>>,
    explicit_witness_ssa: Option<SSA<ConstantTaint>>,
    r1cs_ssa: Option<SSA<ConstantTaint>>,
    base_witgen_ssa: Option<SSA<ConstantTaint>>,
    abi: Option<noirc_abi::Abi>,
    draw_cfg: bool,
}

#[derive(Debug)]
pub enum Error {
    NoirCompilerError(Vec<noirc_errors::reporter::CustomDiagnostic>),
}

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
            explicit_witness_ssa: None,
            r1cs_ssa: None,
            base_witgen_ssa: None,
            abi: None,
            draw_cfg,
        }
    }

    pub fn get_debug_output_dir(&self) -> PathBuf {
        let dir = self
            .project
            .get_only_crate()
            .root_dir
            .join("mavros_debug");
        dir
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
            &noirc_driver::CompileOptions {
                deny_warnings: false,
                debug_comptime_in_file: None,
                ..Default::default()
            },
        )
        .map_err(Error::NoirCompilerError)?;

        let main = context.get_main_function(context.root_crate_id()).unwrap();
        let program =
            noirc_frontend::monomorphization::monomorphize(main, &mut context.def_interner, false)
                .unwrap();

        self.abi = Some(noirc_driver::gen_abi(&context, &main, program.return_visibility(), BTreeMap::default()));

        let ssa = noirc_evaluator::ssa::SsaBuilder::from_program(
            program,
            noirc_evaluator::ssa::SsaLogging::None,
            true,
            &None,
            None,
        )
        .unwrap();

        let passes = vec![
            noirc_evaluator::ssa::SsaPass::new(
                noirc_evaluator::ssa::ssa_gen::Ssa::defunctionalize,
                "Defunctionalization",
            ),
            noirc_evaluator::ssa::SsaPass::new(
                noirc_evaluator::ssa::ssa_gen::Ssa::remove_unreachable_functions,
                "Removing Unreachable Functions",
            ),
        ];

        let ssa = ssa.run_passes(&&passes).unwrap();

        self.initial_ssa = Some(SSA::from_noir(&ssa.ssa));

        fs::write(
            self.get_debug_output_dir().join("initial_ssa.txt"),
            self.initial_ssa
                .as_ref()
                .unwrap()
                .to_string(&DefaultSsaAnnotator),
        )
        .unwrap();

        Ok(())
    }

    #[tracing::instrument(skip_all)]
    pub fn make_struct_access_static(&mut self) -> Result<(), Error> {
        let mut pass_manager = PassManager::<Empty>::new(
            "make_struct_access_static".to_string(),
            self.draw_cfg,
            vec![
                Box::new(PrepareEntryPoint::new()),
                Box::new(RemoveUnreachableBlocks::new()),
                Box::new(MakeStructAccessStatic::new()),
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
        let flow_analysis = FlowAnalysis::run(&ssa);
        // let type_info = Types::new().run(ssa, &flow_analysis);
        let call_loops = flow_analysis.get_call_graph().detect_loops();
        if !call_loops.is_empty() {
            todo!(
                "Call loops detected: {:?}. We don't support recursion yet.",
                call_loops
            );
        }

        if self.draw_cfg {
            flow_analysis.generate_images(
                self.get_debug_output_dir().join("initial_state"),
                &ssa,
                "initial state".to_string(),
            );
        }

        let mut taint_analysis = TaintAnalysis::new();
        taint_analysis.run(&ssa, &flow_analysis).unwrap();

        fs::write(
            self.get_debug_output_dir().join("taint_analysed_ssa.txt"),
            ssa.to_string(&taint_analysis),
        )
        .unwrap();

        let mut monomorphization = Monomorphization::new();
        monomorphization.run(&mut ssa, &mut taint_analysis).unwrap();

        fs::write(
            self.get_debug_output_dir().join("monomorphized_ssa.txt"),
            ssa.to_string(&taint_analysis),
        )
        .unwrap();

        let flow_analysis = FlowAnalysis::run(&ssa);

        if self.draw_cfg {
            flow_analysis.generate_images(
                self.get_debug_output_dir().join("before_untaint_cf"),
                &ssa,
                "before untaint control flow".to_string(),
            );
        }

        let mut untaint_cf = UntaintControlFlow::new();
        self.monomorphized_ssa = Some(untaint_cf.run(ssa, &taint_analysis, &flow_analysis));

        fs::write(
            self.get_debug_output_dir().join("untainted_ssa.txt"),
            self.monomorphized_ssa
                .as_ref()
                .unwrap()
                .to_string(&DefaultSsaAnnotator),
        )
        .unwrap();

        Ok(())
    }

    #[tracing::instrument(skip_all)]
    pub fn explictize_witness(&mut self) -> Result<(), Error> {
        let mut pass_manager = PassManager::<ConstantTaint>::new(
            "explictize_witness".to_string(),
            self.draw_cfg,
            vec![
                Box::new(FixDoubleJumps::new()),
                Box::new(Mem2Reg::new()),
                Box::new(ArithmeticSimplifier::new()),
                Box::new(CSE::new()),
                Box::new(ConditionPropagation::new()),
                Box::new(CSE::new()),
                Box::new(DeduplicatePhis::new()),
                Box::new(DCE::new(dead_code_elimination::Config::pre_r1c())),
                Box::new(FixDoubleJumps::new()),
                Box::new(PullIntoAssert::new()),
                Box::new(DCE::new(dead_code_elimination::Config::pre_r1c())),
                Box::new(Specializer::new(5.0)),
                Box::new(DCE::new(dead_code_elimination::Config::pre_r1c())),
                Box::new(ExplicitWitness::new()),
                Box::new(FixDoubleJumps::new()),
            ],
        );

        pass_manager.set_debug_output_dir(self.get_debug_output_dir().clone());
        let mut ssa = self.monomorphized_ssa.clone().unwrap();
        pass_manager.run(&mut ssa);
        self.explicit_witness_ssa = Some(ssa);
        Ok(())
    }

    #[tracing::instrument(skip_all)]
    pub fn generate_r1cs(&mut self) -> Result<R1CS, Error> {
        let mut r1cs_ssa = self.explicit_witness_ssa.clone().unwrap();

        let mut r1cs_phase_1 = PassManager::<ConstantTaint>::new(
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

    pub fn compile_witgen(&mut self) -> Result<Vec<u64>, Error> {
        self.prepare_base_witgen_ssa();
        let ssa = self.base_witgen_ssa.as_ref().unwrap();

        let flow_analysis = FlowAnalysis::run(ssa);
        let type_info = Types::new().run(ssa, &flow_analysis);

        let codegen = CodeGen::new();
        let program = codegen.run(&ssa, &flow_analysis, &type_info);
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
        let mut ad_pm = PassManager::<ConstantTaint>::new(
            "ad".to_string(),
            self.draw_cfg,
            vec![
                Box::new(WitnessToRef::new()),
                Box::new(RCInsertion::new()),
                Box::new(FixDoubleJumps::new()),
            ],
        );
        ad_pm.set_debug_output_dir(self.get_debug_output_dir().clone());
        ad_pm.run(&mut ssa);
        let flow_analysis = FlowAnalysis::run(&ssa);
        let type_info = Types::new().run(&ssa, &flow_analysis);

        let codegen = CodeGen::new();
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

        let mut ssa = self.explicit_witness_ssa.clone().unwrap();

        let mut pass_manager = PassManager::<ConstantTaint>::new(
            "base_witgen".to_string(),
            self.draw_cfg,
            vec![
                Box::new(WitnessWriteToVoid::new()),
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
        wasm_config: Option<(std::path::PathBuf, &R1CS)>,
    ) -> Result<Option<String>, Error> {
        use crate::compiler::llvm_codegen::LLVMCodeGen;
        use inkwell::context::Context;
        use inkwell::OptimizationLevel;

        self.prepare_base_witgen_ssa();
        let ssa = self.base_witgen_ssa.as_ref().unwrap();

        let flow_analysis = FlowAnalysis::run(ssa);
        let type_info = Types::new().run(ssa, &flow_analysis);

        let context = Context::create();
        let mut codegen = LLVMCodeGen::new(&context, "mavros_module");
        codegen.compile(ssa, &flow_analysis, &type_info);

        let llvm_ir = if emit_llvm {
            let ir = codegen.get_ir();
            fs::write(self.get_debug_output_dir().join("witgen.ll"), &ir).unwrap();
            info!(message = %"LLVM IR generated", ir_size = ir.len());
            Some(ir)
        } else {
            None
        };

        if let Some((wasm_path, r1cs)) = wasm_config {
            codegen.write_ir(&wasm_path.with_extension("ll"));
            codegen.compile_to_wasm(&wasm_path, OptimizationLevel::Aggressive);
            info!(message = %"WASM object generated", path = %wasm_path.display());
            self.write_wasm_metadata(&wasm_path, r1cs)?;
        }

        Ok(llvm_ir)
    }

    /// Write WASM metadata JSON file
    fn write_wasm_metadata(&self, wasm_path: &std::path::PathBuf, r1cs: &R1CS) -> Result<(), Error> {
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
        fs::write(&metadata_path, serde_json::to_string_pretty(&metadata).unwrap()).unwrap();

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
