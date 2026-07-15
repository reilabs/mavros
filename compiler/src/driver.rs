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
            arg_promotion::ArgPromotion,
            array_boundary_expansion::ArrayBoundaryExpansion,
            array_sroa::ArraySroa,
            common_subexpression_elimination::CSE,
            dead_code_elimination::{self, DCE},
            deduplicate_phis::DeduplicatePhis,
            defunctionalize::Defunctionalize,
            elide_tuples::ElideTuples,
            fix_double_jumps::FixDoubleJumps,
            instruction_lowering::InstructionLowering,
            lookup_spilling::LookupSpilling,
            lower_guards::LowerGuards,
            lower_map_casts::LowerMapCasts,
            mem2reg::Mem2Reg,
            merge_identical_functions::MergeIdenticalFunctions,
            normalize_asserts::NormalizeAsserts,
            partial_redundancy_elimination::PRE,
            prepare_entry_point::PrepareEntryPoint,
            rc_insertion::RCInsertion,
            remove_unreachable_blocks::RemoveUnreachableBlocks,
            remove_unreachable_functions::RemoveUnreachableFunctions,
            simplifier::Simplifier,
            simplify_asserts::SimplifyAsserts,
            sparse_conditional_simplification::SCS,
            specializer::Specializer,
            strip_witness_of::StripWitnessOf,
            trivial_phi_elimination::TrivialPhiElimination,
            witness_lowering::WitnessLowering,
            witness_write_to_fresh::WitnessWriteToFresh,
            witness_write_to_void::WitnessWriteToVoid,
        },
        purify_witness_slices::PurifyWitnessSlices,
        ssa::{
            DefaultSSAAnnotator,
            hlssa::{Constant, HLSSA},
        },
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
    /// The final multi-entry-point SSA: the witgen program with the AD program merged in as a
    /// second entry point. All executable artifacts (bytecode, LLVM, WASM) are compiled from
    /// this single SSA.
    program_ssa: Option<HLSSA>,
    abi: Option<noirc_abi::Abi>,
    draw_cfg: bool,
    main_is_unconstrained: bool,
}

#[derive(Debug)]
pub enum Error {
    NoirCompilerError(Vec<noirc_errors::reporter::CustomDiagnostic>),
    /// The program contains an assertion (or range/equality constraint) that can never be
    /// satisfied, discovered while symbolically executing the program to generate R1CS. Such a
    /// program will never execute, so it is rejected rather than compiled into constraints.
    UnsatisfiableProgram(String),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::NoirCompilerError(diagnostics) => {
                write!(f, "Noir compiler error ({} diagnostics)", diagnostics.len())
            }
            Error::UnsatisfiableProgram(message) => {
                write!(f, "program will never execute: {message}")
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
            program_ssa: None,
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

    fn write_debug_text(&self, path: impl AsRef<Path>, contents: impl Into<String>) {
        fs::write(path, self.normalize_debug_text(contents.into())).unwrap();
    }

    fn normalize_debug_text(&self, mut contents: String) -> String {
        let root = self.project.package_root();
        let root = root.to_string_lossy();
        contents = contents.replace(root.as_ref(), "$PROJECT_ROOT");
        if let Ok(canonical_root) = fs::canonicalize(self.project.package_root()) {
            let canonical_root = canonical_root.to_string_lossy();
            contents = contents.replace(canonical_root.as_ref(), "$PROJECT_ROOT");
        }
        contents
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
            Monomorphizer::new(&mut context.def_interner, debug_type_tracker, None, false);
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
        let (ssa, main_is_unconstrained) =
            HLSSA::from_program_with_file_manager(&program, Some(self.project.file_manager()));
        self.initial_ssa = Some(ssa);
        self.main_is_unconstrained = main_is_unconstrained;

        self.write_debug_text(
            self.get_debug_output_dir().join("initial_ssa.txt"),
            self.initial_ssa
                .as_ref()
                .unwrap()
                .to_string(&DefaultSSAAnnotator),
        );

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
                // Normalize `assert(a == b)` / `assert(a < b)` into `AssertCmp` (witness-agnostic,
                // no `AssertR1C`) so the SCS conditional layer below can see the asserted-equal and
                // asserted-constant facts. Run pre-WTI: operands are still scalar, so asserted
                // constants fold away here and never become witnesses/constraints downstream.
                Box::new(NormalizeAsserts::new()),
                // Fold constants and prune statically-decided branches (e.g. monomorphized
                // generic dispatch) BEFORE pruning functions: calls in never-taken branches must
                // not keep their callees alive into witness type inference and untaint CF, which
                // can ICE on semantically-dead code they would otherwise have to type.
                Box::new(SCS::new(dead_code_elimination::Config::preserve_blocks())),
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

    /// Performs comprehensive monomorphization on the SSA.
    ///
    /// Must have had [`Self::make_struct_access_static`] run prior to being run.
    #[tracing::instrument(skip_all)]
    pub fn monomorphize(&mut self) -> Result<(), Error> {
        let mut ssa = self.static_struct_access_ssa.clone().unwrap();

        // Mem2Reg + cleanup before WTI so witness inference sees clean SSA
        // and Guard never wraps promotable Store/Load.
        PassManager::new(
            "pre_wti".to_string(),
            self.draw_cfg,
            vec![
                // The initial mem2reg handles the obvious conversions of heap traffic to SSA
                // variables using the points-to analysis.
                Box::new(Mem2Reg::new()),
                // Fold the constants the initial mem2reg just promoted out of memory (and prune any
                // now-decidable branches) so the downstream ArraySroa / ArgPromotion /
                // ArrayBoundaryExpansion cluster sees constant indices and a reduced CFG — more
                // arrays prove `Split`-able, more refs prove promotable. `preserve_blocks()`: this
                // is still pre-untaint, which cannot yet handle multiple jumps into merge blocks.
                Box::new(SCS::new(dead_code_elimination::Config::preserve_blocks())),
                // Promote `Ref<Array>` locals to array values, then peel every proven-`Split` array
                // into per-cell values so the cells become individually `Pure`-able / promotable.
                Box::new(ArraySroa::new()),
                // A second Mem2Reg promotes the underlying allocs now reached only through the
                // peeled per-cell refs (the array-of-ref case); it self-skips functions with
                // nothing newly promotable, so it is cheap when no ref-arrays were peeled.
                Box::new(Mem2Reg::new()),
                // Promote `Ref<T>` parameters across call boundaries into by-value in/out values,
                // turning callee/caller pointee memory into clean locals.
                Box::new(ArgPromotion::new()),
                // A third Mem2Reg promotes the materialized callee allocs and the now-local caller
                // allocs that arg promotion exposed.
                Box::new(Mem2Reg::new()),
                // Sever array *call boundaries*: expand an array parameter/return whose only
                // obstacle to being `Split` is crossing the boundary into per-cell scalars,
                // reconstructing the array locally on each side.
                Box::new(ArrayBoundaryExpansion::new()),
                // A fresh ArraySroa peels the now-local reconstructed arrays (they only become
                // `Split` on this re-analysis), and a Mem2Reg promotes any per-cell ref allocs the
                // peel exposes.
                Box::new(ArraySroa::new()),
                Box::new(Mem2Reg::new()),
                // Reclaim cells a side never used: dead by-value formals, their matching arguments
                // at every static call site, dead return slots, and the now-dead caller-side
                // `ArrayGet`s — interprocedurally. (`pre_wti` otherwise runs no DCE, and
                // TrivialPhiElimination does not touch entry-block formals).
                //
                // Uses `preserve_blocks()` so it does not collapse empty intermediate blocks into
                // multiple-predecessor merges that the later untaint_control_flow cannot yet
                // handle.
                Box::new(DCE::new(dead_code_elimination::Config::preserve_blocks())),
                // Re-normalize any `Cmp`-fed asserts the structural passes above introduced, then
                // fold the post-mem2reg/SROA constants and prune now-decidable branches BEFORE
                // WTI/untaint. This is the last optimization before witness typing: pruning here
                // keeps dead branches from tainting witnesses (bloating constraints) and from
                // reaching untaint, which can ICE on semantically-dead code. `preserve_blocks()`
                // for the same pre-untaint reason as the DCE above. SCS's `JmpIf->Jmp` folding can
                // strand trivial phis, so it runs before TrivialPhiElimination; its branch-pruning
                // can orphan callees, so RemoveUnreachableFunctions stays after it (cf. line 207).
                Box::new(NormalizeAsserts::new()),
                Box::new(SCS::new(dead_code_elimination::Config::preserve_blocks())),
                // The only value-numbering sweep before witness typing: dedup here shrinks
                // untaint's input (fewer values to taint, guard, and select) and hands the first
                // post-untaint sites already-unified value pairs. Runs after SCS (which
                // copy-propagates asserted-equal leaders into operands, making them structurally
                // congruent to the recomputed analysis) and before TrivialPhiElimination (a PRE
                // redirect can make both preds' jump args identical, and such a phi must collapse
                // before WTI taints it into a needless `Select`).
                //
                // `pre_untaint()` speculates under `preserve_structure`: hoists are pure
                // instruction insertion above `Jmp` terminators — no edge splits or merge params
                // for untaint to absorb — and the speculation gate reads witness-ness from the
                // read-only joined WTI approximation, since types do not yet carry `WitnessOf`.
                Box::new(PRE::pre_untaint()),
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

        self.write_debug_text(
            self.get_debug_output_dir().join("monomorphized_ssa.txt"),
            ssa.to_string(&witness_inference),
        );

        let (mut ssa, purified) = PurifyWitnessSlices::new().run(ssa, &witness_inference);

        let mut untaint_cf = UntaintControlFlow::new();
        self.monomorphized_ssa = Some(if purified {
            PassManager::new(
                "elide_tuples_from_slice_substitution".to_string(),
                self.draw_cfg,
                vec![
                    Box::new(ElideTuples::new()),
                    Box::new(RemoveUnreachableFunctions::new()),
                ],
            )
            .run(&mut ssa);

            let flow_analysis = FlowAnalysis::run(&ssa);
            let mut witness_inference = WitnessTaintInference::new();
            witness_inference.run(&mut ssa, &flow_analysis);
            untaint_cf.run(ssa, &witness_inference)
        } else {
            untaint_cf.run(ssa, &witness_inference)
        });

        self.write_debug_text(
            self.get_debug_output_dir().join("untainted_ssa.txt"),
            self.monomorphized_ssa
                .as_ref()
                .unwrap()
                .to_string(&DefaultSSAAnnotator),
        );

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
                // Re-normalize asserts to `AssertCmp` so the conditional facts are visible to the
                // SCS runs in this phase too (the `AssertR1C` lowering stays in `SimplifyAsserts`,
                // below). New `Cmp`-fed asserts can appear after the pre-WTI passes and lowering.
                Box::new(NormalizeAsserts::new()),
                // Fold pure constants and prune constant-condition branches before the cleanup
                // rounds, so Simplifier/PRE work on the reduced CFG. (SCS reclaims the blocks
                // its own `JmpIf -> Jmp` folds orphan, so no trailing RemoveUnreachableBlocks is
                // needed here — the InstructionLowering runs below type every block's instructions
                // against reachable-only `Types` info and would ICE on a leftover orphan.)
                Box::new(SCS::new(dead_code_elimination::Config::pre_r1c())),
                // Simplify → PRE, twice (PRE runs its own integrated DCE). The
                // doubled rounds let PRE-dedup expose new fold operands and
                // folds expose new dedup matches. Each Simplifier internally
                // iterates to fixed point for purely-algebraic folds.
                Box::new(Simplifier::new()),
                Box::new(PRE::pre_r1c()),
                Box::new(Simplifier::new()),
                Box::new(PRE::pre_r1c()),
                // Re-run SCS after cleanup exposes new constants and branch predicate facts.
                Box::new(SCS::new(dead_code_elimination::Config::pre_r1c())),
                Box::new(PRE::pre_r1c()),
                Box::new(DeduplicatePhis::new()),
                Box::new(DCE::new(dead_code_elimination::Config::pre_r1c())),
                Box::new(FixDoubleJumps::new()),
                Box::new(SimplifyAsserts::new()),
                Box::new(DCE::new(dead_code_elimination::Config::pre_r1c())),
                Box::new(InstructionLowering::witness_array_access()),
                Box::new(InstructionLowering::slice_select()),
                Box::new(InstructionLowering::witness_integer_ops()),
                // After the last pre-spilling lowering, run cleanup twice
                // back-to-back. The first round exposes folds/dedup opportunities
                // that the second round can then consume.
                Box::new(Simplifier::new()),
                Box::new(PRE::pre_r1c()),
                Box::new(Simplifier::new()),
                Box::new(PRE::pre_r1c()),
                Box::new(Specializer::new(5.0)),
                // Specialization exposes fresh constants (folded call arguments and branch
                // conditions); propagate them before the post-specialization cleanup.
                Box::new(SCS::new(dead_code_elimination::Config::pre_r1c())),
                Box::new(Simplifier::new()),
                Box::new(PRE::pre_r1c()),
                Box::new(LookupSpilling::new()),
                Box::new(InstructionLowering::degree_spilling()),
                Box::new(Simplifier::new()),
                Box::new(PRE::pre_r1c()),
                Box::new(Simplifier::new()),
                Box::new(PRE::pre_r1c()),
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
        r1cs_gen
            .run(&r1cs_ssa, &type_info)
            .map_err(|e| Error::UnsatisfiableProgram(e.message))?;
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

    /// Compiles the program (both the witgen and AD entry points) into a single VM binary.
    pub fn compile_bytecode(&mut self, options: CodeGenOptions) -> Result<Vec<u64>, Error> {
        self.prepare_program_ssa();
        let ssa = self.program_ssa.as_ref().unwrap();

        let flow_analysis = FlowAnalysis::run(ssa);
        let type_info = Types::new().run(ssa, &flow_analysis);

        let codegen = CodeGen::new(options);
        let program = codegen.run(ssa, &flow_analysis, &type_info);
        self.write_debug_text(
            self.get_debug_output_dir().join("program_bytecode.txt"),
            format!("{}", program),
        );

        let binary = program.to_binary();

        info!(message = %"Program binary generated", binary_size = binary.len() * 8);

        Ok(binary)
    }

    pub fn abi(&self) -> &noirc_abi::Abi {
        self.abi.as_ref().unwrap()
    }

    /// Builds the final multi-entry-point SSA.
    ///
    /// The witgen and AD halves need different lowerings of the same witness-spilled program
    /// (witnesses as plain values vs. witness-tape references), so each half is lowered on its
    /// own clone. The AD half is then merged into the witgen program as a second entry point and
    /// everything downstream (refcounting, codegen, every artifact) compiles the whole program
    /// together.
    #[tracing::instrument(skip_all)]
    fn prepare_program_ssa(&mut self) {
        if self.program_ssa.is_some() {
            return;
        }

        let mut ssa = self.witness_spilled_ssa.clone().unwrap();

        let mut witgen_pm = PassManager::new(
            "witgen_lowering".to_string(),
            self.draw_cfg,
            vec![
                Box::new(LowerGuards::new()),
                Box::new(WitnessWriteToVoid::new()),
                Box::new(StripWitnessOf::new()),
                Box::new(DCE::new(dead_code_elimination::Config::post_r1c())),
            ],
        );
        witgen_pm.set_debug_output_dir(self.get_debug_output_dir().clone());
        witgen_pm.run(&mut ssa);

        let mut ad_ssa = self.r1cs_ssa.clone().unwrap();
        let mut ad_pm = PassManager::new(
            "ad_lowering".to_string(),
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
            ],
        );
        ad_pm.set_debug_output_dir(self.get_debug_output_dir().clone());
        ad_pm.run(&mut ad_ssa);

        // `SSA::merge` re-interns constants by value, which would silently conflate function
        // pointers across the two halves; by this point defunctionalization must have removed
        // them all.
        ad_ssa.for_each_const(|_, cv| {
            assert!(
                !const_contains_fn_ptr(cv),
                "ICE: FnPtr constant survived until program merge"
            );
        });

        let ad_main_id = ad_ssa.get_unique_entrypoint_id();
        let fn_remap = ssa.merge(ad_ssa);
        let ad_entry = fn_remap[&ad_main_id];
        ssa.get_function_mut(ad_entry)
            .set_name("ad_main".to_string());
        ssa.add_entry_point(ad_entry);

        let mut tail_pm = PassManager::new(
            "program_tail".to_string(),
            self.draw_cfg,
            vec![
                // The merge above put two near-copies of the whole program into one SSA;
                // fold the byte-identical functions (notably `globals_init`) before anything
                // downstream pays for them. Both codegen backends emit every function in the
                // SSA, so this is where program size is won.
                Box::new(MergeIdenticalFunctions::new()),
                Box::new(RCInsertion::new()),
                Box::new(FixDoubleJumps::new()),
            ],
        );
        tail_pm.set_debug_output_dir(self.get_debug_output_dir().clone());
        tail_pm.run(&mut ssa);

        self.write_debug_text(
            self.get_debug_output_dir().join("program_ssa.txt"),
            ssa.to_string(&DefaultSSAAnnotator),
        );

        self.program_ssa = Some(ssa);
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

        self.prepare_program_ssa();
        let ssa = self.program_ssa.as_ref().unwrap();

        let flow_analysis = FlowAnalysis::run(ssa);
        let type_info = Types::new().run(ssa, &flow_analysis);

        // Dump HLSSA just before lowering
        self.write_debug_text(
            self.get_debug_output_dir()
                .join("hlssa_before_lowering.txt"),
            ssa.to_string(&DefaultSSAAnnotator),
        );

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
        self.write_debug_text(
            self.get_debug_output_dir().join("llssa_after_lowering.txt"),
            llssa.to_string(&DefaultSSAAnnotator),
        );

        // Compile LLSSA → LLVM
        let ll_flow_analysis = FlowAnalysis::run(&llssa);
        let context = Context::create();
        let mut codegen = LLVMCodeGen::new(&context, "mavros_module");
        codegen.compile(&llssa, &ll_flow_analysis);

        let llvm_ir = if emit_llvm {
            let ir = codegen.get_ir();
            fs::write(self.get_debug_output_dir().join("program.ll"), &ir).unwrap();
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

/// Whether a constant contains a function pointer anywhere within it.
fn const_contains_fn_ptr(constant: &Constant) -> bool {
    match constant {
        Constant::FnPtr(_) => true,
        Constant::Blob(blob) => blob.elements.iter().any(const_contains_fn_ptr),
        Constant::U(..) | Constant::I(..) | Constant::Field(_) => false,
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
