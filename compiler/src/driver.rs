//! The driver API for the compilation pipeline.

use std::{
    cell::RefCell,
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
    rc::Rc,
    sync::Arc,
};

use mavros_vm::bytecode::DebugInfo;
use noirc_frontend::{
    debug::DebugInstrumenter,
    monomorphization::{Monomorphizer, debug_types::DebugTypeTracker},
};
use tracing::info;

use mavros_artifacts::{
    FieldConfig, FieldId, ManifestParam, MavrosField, ParamVisibility, R1CSArtifact, ReturnSlot,
    WitnessManifest,
};

use crate::{
    Project,
    compiler::{
        analysis::{
            flow_analysis::FlowAnalysis,
            types::{Types, check_casts_into_field},
            witness_taint_inference::WitnessTaintInference,
        },
        codegen::{
            CodeGenOptions,
            bytecode::{CodeGen, WitnessMode},
            hlssa_to_r1cs::{R1CGen, R1CS, R1CSProfile, SealedR1CS, logup_soundness_report},
            llssa_to_llvm::WasmCompileOpts,
        },
        diagnostic::{self, Diagnostic},
        located::synthetic_file,
        pass_manager::PassManager,
        passes::{
            arg_promotion::ArgPromotion,
            array_boundary_expansion::ArrayBoundaryExpansion,
            array_sroa::ArraySroa,
            assert_constant::AssertConstantValidation,
            common_subexpression_elimination::CSE,
            dead_code_elimination::{self, DCE},
            deduplicate_phis::DeduplicatePhis,
            defunctionalize::Defunctionalize,
            elide_tuples::ElideTuples,
            fix_double_jumps::FixDoubleJumps,
            instruction_lowering::InstructionLowering,
            lookup_spilling::LookupSpilling,
            lower_degenerate_array::LowerDegenerateArray,
            lower_map_casts::LowerMapCasts,
            lower_zst_slices::LowerZstSlices,
            mem2reg::Mem2Reg,
            merge_identical_functions::MergeIdenticalFunctions,
            normalize_asserts::NormalizeAsserts,
            partial_redundancy_elimination::PRE,
            prepare_entry_point::{
                ENTRY_RETURN_ORIGIN, EntryBlobLayout, PrepareEntryPoint, entry_param_origin,
            },
            purify_witness_slices::PurifyWitnessSlices,
            rc_insertion::RCInsertion,
            remove_unreachable_blocks::RemoveUnreachableBlocks,
            remove_unreachable_functions::RemoveUnreachableFunctions,
            return_reachability::ReturnReachabilityValidation,
            simplifier::Simplifier,
            simplify_asserts::SimplifyAsserts,
            sparse_conditional_simplification::SCS,
            specializer::Specializer,
            strip_witness_of::StripWitnessOf,
            trivial_phi_elimination::TrivialPhiElimination,
            wide_witness_ints::WideWitnessInts,
            witness_lowering::WitnessLowering,
            witness_write_to_fresh::WitnessWriteToFresh,
            witness_write_to_void::WitnessWriteToVoid,
        },
        ssa::{DefaultSSAAnnotator, SourceLocation, hlssa::HLSSA},
        untaint_control_flow::UntaintControlFlow,
    },
    error::ToolchainError,
};

/// Executable VM bytecode plus optional standalone source metadata.
pub struct BytecodeArtifact {
    pub binary: Vec<u64>,
    pub debug_info: Option<DebugInfo>,
}

/// First witness column of each entry parameter and of the computed return, as the R1CS
/// generator allocated them after dead-code elimination; `None` where nothing was written.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EntryColumns {
    pub params: Vec<Option<usize>>,
    pub returns: Option<usize>,
}

/// Default LogUp bits-of-security target when `--logup-soundness` is not given: one repetition on
/// bn254; on Goldilocks three up to `D = 2^20` table entries and lookups, and four up to `2^31`.
pub const DEFAULT_LOGUP_SOUNDNESS_BITS: u32 = 128;

pub struct Driver {
    /// The Noir package being compiled, or `None` for a driver seeded with a hand-built
    /// [`HLSSA`] (see [`Driver::from_ssa`]). Only the frontend stage and the source-path
    /// normalization of the debug dumps need it.
    project: Option<Project>,
    /// Where the per-stage debug dumps go. Derived from the package root for a project-backed
    /// driver; supplied by the caller for a seeded one.
    debug_output_dir: PathBuf,
    initial_ssa: Option<HLSSA>,
    static_struct_access_ssa: Option<HLSSA>,
    monomorphized_ssa: Option<HLSSA>,
    witness_spilled_ssa: Option<HLSSA>,
    r1cs_ssa: Option<HLSSA>,
    constraint_layouts: Option<crate::vm::bytecode::ConstraintLayouts>,
    r1cs_profile: Option<R1CSProfile>,
    r1cs_profiling: bool,
    /// The final multi-entry-point SSA: the witgen program with the AD program merged in as a
    /// second entry point. All executable artifacts (bytecode, LLVM, WASM) are compiled from
    /// this single SSA.
    program_ssa: Option<HLSSA>,
    abi: Option<noirc_abi::Abi>,
    entry_blob_layout: Option<EntryBlobLayout>,
    entry_columns: Option<EntryColumns>,
    draw_cfg: bool,
    main_is_unconstrained: bool,
    /// Requested LogUp bits-of-security (from `--logup-soundness`); sizes the LogUp repetitions.
    logup_soundness: u32,
}

#[derive(Debug)]
pub enum Error {
    /// The Noir compiler driver returned errors.
    NoirCompilerError {
        diagnostics: Vec<noirc_errors::reporter::CustomDiagnostic>,
        /// Rendered while Noir's file manager and parsed files are available.
        rendered: String,
    },

    /// The program has no returning entry path (checked before optimization), or contains an
    /// impossible assertion/range/equality constraint found during R1CS generation. Static return
    /// validation carries source locations; R1CS failures use a synthetic location because symbolic
    /// assertion failures do not retain their spans.
    UnsatisfiableProgram(Vec<Diagnostic>),

    /// The `--logup-soundness` target is out of reach: the field is too small for any number of
    /// LogUp repetitions (one distinguishes less than a bit at the argument's degree), or the
    /// target needs more than
    /// [`MAX_LOGUP_REPETITIONS`](crate::compiler::codegen::hlssa_to_r1cs::MAX_LOGUP_REPETITIONS).
    LogupSoundnessUnsupported(String),

    /// One or more Noir `assert_constant` operands are dynamic in a reachable calling context.
    ///
    /// Carries every failing assertion so a single compile reports them all.
    AssertConstantFailed(Vec<SourceLocation>),

    /// A program to compile to WASM over a field the runtime archive has no instance for.
    WasmFieldUnsupported(FieldId),

    /// A program to compile to LLVM or WASM whose LogUp argument repeats: the LLSSA AD emitters
    /// bind a single repetition's challenges.
    WasmLogupRepetitionsUnsupported(usize),

    /// R1CS coefficients requested in a field other than the program's.
    R1csFieldMismatch {
        requested: FieldId,
        program: FieldId,
    },
    /// Missing ABI or inconsistent entry-point/R1CS layouts.
    WitnessManifest(String),

    Toolchain(ToolchainError),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::NoirCompilerError { rendered, .. } => f.write_str(rendered),
            Error::UnsatisfiableProgram(diagnostics) => {
                write!(f, "{}", diagnostic::render_all(diagnostics))
            }
            Error::LogupSoundnessUnsupported(message) => write!(f, "{message}"),
            Error::WasmFieldUnsupported(field) => {
                write!(f, "the WASM backend does not support the {field:?} field")
            }
            Error::WasmLogupRepetitionsUnsupported(repetitions) => write!(
                f,
                "the LLVM and WASM backends run one LogUp repetition, but --logup-soundness \
                 needs {repetitions} for this program; lower it or use the VM"
            ),
            Error::R1csFieldMismatch { requested, program } => write!(
                f,
                "R1CS generation in {requested:?} was requested for a program over {program:?}"
            ),
            Error::AssertConstantFailed(locations) => {
                for (i, location) in locations.iter().enumerate() {
                    if i > 0 {
                        writeln!(f)?;
                    }
                    write!(
                        f,
                        "assert_constant failed at {location}: value may depend on witness data"
                    )?;
                }
                Ok(())
            }
            Error::WitnessManifest(message) => write!(f, "witness manifest: {message}"),
            Error::Toolchain(error) => write!(f, "{error}"),
        }
    }
}

impl std::error::Error for Error {}

impl From<ToolchainError> for Error {
    fn from(error: ToolchainError) -> Self {
        Error::Toolchain(error)
    }
}

impl Driver {
    pub fn new(project: Project, draw_cfg: bool) -> Self {
        let dir = project.get_only_crate().root_dir.join("mavros_debug");
        Self::fresh_debug_dir(&dir);
        Self {
            project: Some(project),
            debug_output_dir: dir,
            initial_ssa: None,
            static_struct_access_ssa: None,
            monomorphized_ssa: None,
            witness_spilled_ssa: None,
            r1cs_ssa: None,
            constraint_layouts: None,
            r1cs_profile: None,
            r1cs_profiling: false,
            program_ssa: None,
            abi: None,
            entry_blob_layout: None,
            entry_columns: None,
            draw_cfg,
            main_is_unconstrained: false,
            logup_soundness: DEFAULT_LOGUP_SOUNDNESS_BITS,
        }
    }

    /// Builds a driver around an already-constructed [`HLSSA`], bypassing the Noir frontend, so a
    /// feature can be prototyped before the frontend supports it.
    ///
    /// Such a driver has no [`Self::abi`], so [`Self::entry_blob_field_count`],
    /// [`Self::witness_manifest`], [`Self::r1cs_artifact`] and the WASM metadata sidecar are
    /// unavailable, and no [`Self::package_root`]: all belong to the Noir package, which an SSA
    /// does not carry.
    ///
    /// `debug_output_dir` is created, or emptied if it exists, as [`Self::new`] does with the
    /// package's `mavros_debug`.
    pub fn from_ssa(ssa: HLSSA, debug_output_dir: PathBuf, draw_cfg: bool) -> Self {
        Self::fresh_debug_dir(&debug_output_dir);
        Self {
            project: None,
            debug_output_dir,
            initial_ssa: Some(ssa),
            static_struct_access_ssa: None,
            monomorphized_ssa: None,
            witness_spilled_ssa: None,
            r1cs_ssa: None,
            constraint_layouts: None,
            r1cs_profile: None,
            r1cs_profiling: false,
            program_ssa: None,
            abi: None,
            entry_blob_layout: None,
            entry_columns: None,
            draw_cfg,
            main_is_unconstrained: false,
            logup_soundness: DEFAULT_LOGUP_SOUNDNESS_BITS,
        }
    }

    /// Create the debug dump directory, discarding anything a previous run left in it.
    fn fresh_debug_dir(dir: &Path) {
        if dir.exists() {
            fs::remove_dir_all(dir).unwrap();
        }
        fs::create_dir_all(dir).unwrap();
    }

    /// Override the LogUp bits-of-security target (defaults to
    /// [`DEFAULT_LOGUP_SOUNDNESS_BITS`]).
    pub fn set_logup_soundness(&mut self, bits: u32) {
        self.logup_soundness = bits;
    }

    pub fn enable_r1cs_profile(&mut self) {
        self.r1cs_profiling = true;
    }

    pub fn get_debug_output_dir(&self) -> PathBuf {
        self.debug_output_dir.clone()
    }

    /// Root directory of the package being compiled (the workspace member's directory, below the
    /// workspace root). `Prover.toml` is read from here.
    ///
    /// Panics on a driver seeded with [`Self::from_ssa`], which has no package.
    pub fn package_root(&self) -> &std::path::Path {
        self.project().package_root()
    }

    /// The Noir package being compiled. Panics on a driver seeded with [`Self::from_ssa`].
    fn project(&self) -> &Project {
        self.project
            .as_ref()
            .expect("this driver was seeded with an HLSSA and has no Noir package")
    }

    pub fn r1cs_profile(&self) -> Option<&R1CSProfile> {
        self.r1cs_profile.as_ref()
    }

    fn write_debug_text(&self, path: impl AsRef<Path>, contents: impl Into<String>) {
        fs::write(path, self.normalize_debug_text(contents.into())).unwrap();
    }

    fn normalize_debug_text(&self, mut contents: String) -> String {
        // A seeded driver has no package root, and a hand-built SSA carries no source paths to
        // normalize away in the first place.
        let Some(project) = self.project.as_ref() else {
            return contents;
        };
        let root = project.package_root();
        let root = root.to_string_lossy();
        contents = contents.replace(root.as_ref(), "$PROJECT_ROOT");
        if let Ok(canonical_root) = fs::canonicalize(project.package_root()) {
            let canonical_root = canonical_root.to_string_lossy();
            contents = contents.replace(canonical_root.as_ref(), "$PROJECT_ROOT");
        }
        contents
    }

    fn noir_error(&self, diagnostics: Vec<noirc_errors::reporter::CustomDiagnostic>) -> Error {
        let functions = noirc_frontend::error_reporting::function_locations_for_diagnostics(
            &diagnostics,
            self.project().parsed_files(),
        );
        let rendered =
            diagnostic::render_noir(&diagnostics, self.project().file_manager(), &functions);
        Error::NoirCompilerError {
            diagnostics,
            rendered,
        }
    }

    #[tracing::instrument(skip_all)]
    pub fn run_noir_compiler(&mut self) -> Result<(), Error> {
        let (mut context, crate_id) = nargo::prepare_package(
            self.project().file_manager(),
            self.project().parsed_files(),
            self.project().get_only_crate(),
        );
        noirc_driver::check_crate(
            &mut context,
            crate_id,
            &noirc_driver::CompileOptions::default(),
        )
        .map_err(|diagnostics| self.noir_error(diagnostics))?;

        let main = context.get_main_function(context.root_crate_id()).unwrap();
        let debug_type_tracker =
            DebugTypeTracker::build_from_debug_instrumenter(&DebugInstrumenter::default());
        let mut monomorphizer = Monomorphizer::new(
            &mut context.def_interner,
            self.project().file_manager().as_file_map(),
            debug_type_tracker,
            None,
            false,
        );
        monomorphizer
            .compile_main(main)
            .map_err(|error| self.noir_error(vec![error.into()]))?;
        monomorphizer
            .process_queue()
            .map_err(|error| self.noir_error(vec![error.into()]))?;
        let program = monomorphizer.into_program();

        self.abi = Some(noirc_driver::gen_abi(
            &context,
            &main,
            program.return_visibility(),
            BTreeMap::default(),
        ));

        // Convert monomorphized AST directly to SSA, bypassing Noir's SSA generation
        // FIELD-ASSUMPTION: L1-noir-frontend
        // The driver has no field selector, so a Noir package compiles over bn254.
        let (ssa, main_is_unconstrained) = HLSSA::from_program_with_file_manager(
            &program,
            Some(self.project().file_manager()),
            FieldConfig::bn254(),
        );
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
        let mut ssa = self.initial_ssa.clone().unwrap();

        // The program as written, before `WideWitnessInts` emits casts of a wide integer's limbs
        // into Field.
        let flow = FlowAnalysis::run(&ssa);
        check_casts_into_field(&ssa, &Types::new().run(&ssa, &flow));

        let failures = Rc::new(RefCell::new(None));
        let mut validation = PassManager::new(
            "validate_return_reachability".to_string(),
            self.draw_cfg,
            vec![
                Box::new(Defunctionalize::new()),
                Box::new(ReturnReachabilityValidation::new(Rc::clone(&failures))),
            ],
        );
        validation.set_debug_output_dir(self.get_debug_output_dir().clone());
        validation.run(&mut ssa);
        let failures = failures
            .borrow_mut()
            .take()
            .expect("return reachability validation did not record a result");
        if !failures.is_empty() {
            return Err(Error::UnsatisfiableProgram(
                failures
                    .into_iter()
                    .map(|diagnostic| self.attach_compiled_source(diagnostic))
                    .collect(),
            ));
        }

        let entry_blob_layout = Rc::new(RefCell::new(None));
        let mut prepare_entry_point = PrepareEntryPoint::new(self.main_is_unconstrained)
            .with_layout_sink(Rc::clone(&entry_blob_layout));
        if let Some(abi) = &self.abi {
            prepare_entry_point = prepare_entry_point
                .with_abi_return(abi.return_type.is_some())
                .with_param_visibility(Self::abi_param_is_public(abi));
        }
        let mut pass_manager = PassManager::new(
            "make_struct_access_static".to_string(),
            self.draw_cfg,
            vec![
                // First, so that no later pass, the guard IR included, meets a bit window wider than
                // the field carries, a bit or byte decomposition of an integer of any width (each
                // becomes decompositions of field elements), or a range check wider than the field.
                Box::new(InstructionLowering::operand_shapes()),
                Box::new(prepare_entry_point),
                // Lower leaf-less slices to their `u32` length. Must precede the elision, which
                // breaks on leaf-less slices.
                Box::new(LowerZstSlices::new()),
                // Eliminate all tuple types after the entry point is prepared, so every
                // subsequent pass operates on tuple-free IR.
                Box::new(ElideTuples::new()),
                // Normalize `assert(a == b)` / `assert(a < b)` into `AssertCmp` (witness-agnostic,
                // no `AssertR1C`) so the SCS conditional layer below can see the asserted-equal and
                // asserted-constant facts. Run pre-WTI: operands are scalar there, so asserted
                // constants fold away here and never become witnesses/constraints downstream.
                Box::new(NormalizeAsserts::new()),
                // Algebraic identities before the fold, because some of them _produce_ the
                // constants the fold then propagates — `x - x -> 0` above all. This is the only
                // pre-untaint Simplifier run, and it has to be here and no later: the DCE below is
                // the first that rewrites a dead `Div`/`Mod` into its check, and once a division
                // has become a bare assert the operand chain is pinned live for good. Nothing in
                // this pass touches the CFG (it rewrites instructions and aliases results), so it
                // is safe on the pre-untaint IR the phase holds.
                Box::new(Simplifier::new()),
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
        pass_manager.run(&mut ssa);
        self.entry_blob_layout = Some(
            entry_blob_layout
                .borrow_mut()
                .take()
                .expect("prepare_entry_point records the entry blob layout it used"),
        );
        self.static_struct_access_ssa = Some(ssa);
        Ok(())
    }

    /// Attach the source text the compiler read for `diagnostic`'s file, where there is one, so the
    /// snippet shows the code that was compiled, even if the file has changed by the time the error
    /// prints.
    fn attach_compiled_source(&self, diagnostic: Diagnostic) -> Diagnostic {
        let Some(project) = self.project.as_ref() else {
            return diagnostic;
        };

        match compiled_source(project.file_manager(), &diagnostic.location().file) {
            Some(source) => diagnostic.with_source(source),
            None => diagnostic,
        }
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
                // Make constant the index of a dynamic read out of a zero- or one-cell array of
                // references, which is the only thing standing between such an array and the
                // ArraySroa / Mem2Reg cluster below: a `Split` array must be constant-indexed, and
                // nothing beneath HLSSA can represent a dynamically selected handle. Must precede
                // them to be of any use.
                Box::new(LowerDegenerateArray::new()),
                // The initial mem2reg handles the obvious conversions of heap traffic to SSA
                // variables using the points-to analysis.
                Box::new(Mem2Reg::new()),
                // Fold the constants the initial mem2reg just promoted out of memory (and prune any
                // branches they decide) so the downstream ArraySroa / ArgPromotion /
                // ArrayBoundaryExpansion cluster sees constant indices and a reduced CFG — more
                // arrays prove `Split`-able, more refs prove promotable. `preserve_blocks()`: this
                // runs pre-untaint, and untaint handles no merge block with multiple jumps into it.
                Box::new(SCS::new(dead_code_elimination::Config::preserve_blocks())),
                // Promote `Ref<Array>` locals to array values, then peel every proven-`Split` array
                // into per-cell values so the cells become individually `Pure`-able / promotable.
                Box::new(ArraySroa::new()),
                // A second Mem2Reg promotes the underlying allocs the peel leaves reachable only
                // through the per-cell refs (the array-of-ref case); it self-skips functions with
                // nothing newly promotable, so it is cheap when no ref-arrays were peeled.
                Box::new(Mem2Reg::new()),
                // Promote `Ref<T>` parameters across call boundaries into by-value in/out values,
                // turning callee/caller pointee memory into clean locals.
                Box::new(ArgPromotion::new()),
                // A third Mem2Reg promotes the materialized callee allocs and the caller allocs
                // that arg promotion made local.
                Box::new(Mem2Reg::new()),
                // Sever array _call boundaries_: expand an array parameter/return whose only
                // obstacle to being `Split` is crossing the boundary into per-cell scalars,
                // reconstructing the array locally on each side.
                Box::new(ArrayBoundaryExpansion::new()),
                // A fresh ArraySroa peels the reconstructed arrays, local after the expansion (they
                // only become `Split` on this re-analysis), and a Mem2Reg promotes any per-cell ref
                // allocs the peel exposes.
                Box::new(ArraySroa::new()),
                Box::new(Mem2Reg::new()),
                // Reclaim cells a side never used: dead by-value formals, their matching arguments
                // at every static call site, dead return slots, and the caller-side `ArrayGet`s
                // those leave dead — interprocedurally. (`pre_wti` otherwise runs no DCE, and
                // TrivialPhiElimination does not touch entry-block formals).
                //
                // Uses `preserve_blocks()` so it does not collapse empty intermediate blocks into
                // multiple-predecessor merges, which the later untaint_control_flow does not
                // handle.
                Box::new(DCE::new(dead_code_elimination::Config::preserve_blocks())),
                // Re-normalize any `Cmp`-fed asserts the structural passes above introduced, then
                // fold the post-mem2reg/SROA constants and prune the branches they decide BEFORE
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
                // read-only joined WTI approximation, since types carry no `WitnessOf` there.
                Box::new(PRE::pre_untaint()),
                // Mem2Reg promotes each scalarized leaf cell into a separate block-parameter phi.
                // For an aggregate threaded through control flow that is mostly trivial phis (the
                // same value from every predecessor); collapse them before they reach WTI and
                // codegen.
                Box::new(TrivialPhiElimination::new()),
                Box::new(RemoveUnreachableFunctions::new()),
                // Purify witness-length slices into `(physical, log_len, start)` tuples. Reads
                // the read-only joined WTI approximation (no specialization — context splitting
                // happens exactly once, below); the elision cleans up the tuples it introduces.
                Box::new(PurifyWitnessSlices::new()),
                Box::new(ElideTuples::new()),
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
        let witness_inference = Rc::new(witness_inference);
        let failures = Rc::new(RefCell::new(None));
        PassManager::new(
            "assert_constant_validation".to_string(),
            self.draw_cfg,
            vec![Box::new(AssertConstantValidation::new(
                Rc::clone(&witness_inference),
                Rc::clone(&failures),
            ))],
        )
        .run(&mut ssa);
        let failures = failures
            .borrow_mut()
            .take()
            .expect("assert_constant validation pass did not record a result");
        if !failures.is_empty() {
            return Err(Error::AssertConstantFailed(failures));
        }

        self.write_debug_text(
            self.get_debug_output_dir().join("monomorphized_ssa.txt"),
            ssa.to_string(witness_inference.as_ref()),
        );

        let mut untaint_cf = UntaintControlFlow::new();
        self.monomorphized_ssa = Some(untaint_cf.run(ssa, witness_inference.as_ref()));

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
        let mut ssa = self.monomorphized_ssa.clone().unwrap();

        let mut pass_manager = PassManager::new(
            "witness_spilling".to_string(),
            self.draw_cfg,
            vec![
                Box::new(InstructionLowering::pure_bit_decompositions()),
                // Lower the remaining (pure-length) slice pops/inserts/removes. The
                // witness-length ones were already rewritten by `PurifyWitnessSlices`.
                Box::new(InstructionLowering::slice_ops()),
                Box::new(InstructionLowering::pure_guards()),
                Box::new(InstructionLowering::witness_memory_ops()),
                Box::new(FixDoubleJumps::new()),
                // Re-normalize asserts to `AssertCmp` so the conditional facts are visible to the
                // SCS runs in this phase too (the `AssertR1C` lowering stays in `SimplifyAsserts`,
                // below). New `Cmp`-fed asserts can appear after the pre-WTI passes and lowering.
                Box::new(NormalizeAsserts::new()),
                // Fold pure constants and prune constant-condition branches before the cleanup
                // rounds, so Simplifier/PRE work on the reduced CFG. (SCS reclaims the blocks that
                // its `JmpIf -> Jmp` folds orphan, so no trailing RemoveUnreachableBlocks is needed
                // here — the InstructionLowering runs below type every block's instructions against
                // reachable-only `Types` info and would ICE on a leftover orphan.)
                Box::new(SCS::new(dead_code_elimination::Config::pre_r1c())),
                // Simplify → PRE, twice (PRE runs an integrated DCE). The doubled rounds let
                // PRE-dedup expose new fold operands and folds expose new dedup matches. Each
                // Simplifier internally iterates to fixed point for purely-algebraic folds.
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
                // Immediately before the lowerings that read an integer's width: a witnessed value
                // too wide for one field element becomes its limbs here, and every lowering below
                // this sees limbs it can work with.
                Box::new(WideWitnessInts::new()),
                // After the representation, so a value it carries as limbs has already had its
                // narrowing rewritten into a recombination.
                Box::new(InstructionLowering::witness_narrowing_casts()),
                Box::new(InstructionLowering::witness_integer_ops()),
                // After the last pre-spilling lowering, run cleanup twice
                // back-to-back. The first round exposes folds/dedup opportunities
                // that the second round can then consume.
                Box::new(Simplifier::new()),
                Box::new(PRE::pre_r1c()),
                Box::new(Simplifier::new()),
                Box::new(PRE::pre_r1c()),
                Box::new(Specializer::new(5.0)),
                // Specialization exposes fresh constants (folded call arguments and branch conds);
                // propagate them before the post-specialization cleanup.
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
        pass_manager.run(&mut ssa);
        self.witness_spilled_ssa = Some(ssa);
        Ok(())
    }

    /// Generates the program's R1CS with coefficients in `F`, which must be the raw type of the
    /// program's field.
    #[tracing::instrument(skip_all)]
    pub fn generate_r1cs<F: MavrosField>(&mut self) -> Result<SealedR1CS<F>, Error> {
        let mut r1cs_ssa = self.witness_spilled_ssa.clone().unwrap();
        let program_field = r1cs_ssa.field().id();
        if F::ID != program_field {
            return Err(Error::R1csFieldMismatch {
                requested: F::ID,
                program: program_field,
            });
        }

        let mut r1cs_phase_1 = PassManager::new(
            "r1cs_phase_1".to_string(),
            self.draw_cfg,
            vec![
                Box::new(WitnessWriteToFresh::new()),
                // Prunes the entry's input blob, whose reads died with the witness writes.
                Box::new(DCE::new(dead_code_elimination::Config::r1cs_gen())),
                Box::new(FixDoubleJumps::new()),
            ],
        );
        r1cs_phase_1.set_debug_output_dir(self.get_debug_output_dir().clone());
        r1cs_phase_1.run(&mut r1cs_ssa);

        let flow_analysis = FlowAnalysis::run(&r1cs_ssa);
        let type_info = Types::new().run(&r1cs_ssa, &flow_analysis);

        let mut r1cs_gen = R1CGen::<F>::new(r1cs_ssa.field());
        if self.r1cs_profiling {
            r1cs_gen.enable_profile();
        }

        r1cs_gen.run(&r1cs_ssa, &type_info).map_err(|e| {
            Error::UnsatisfiableProgram(vec![
                self.attach_compiled_source(Diagnostic::error(
                    e.message,
                    e.location
                        .unwrap_or_else(|| SourceLocation::synthetic("r1cs_generation")),
                )),
            ])
        })?;

        let logup_repetitions = if r1cs_gen.has_lookup_tables() {
            self.logup_repetitions(&r1cs_gen, r1cs_ssa.field())?
        } else {
            1
        };

        // Captured before `seal` consumes `r1cs_gen`: where the wrapper's writes landed.
        self.entry_columns = Some(self.resolve_entry_columns(r1cs_gen.entry_columns()));
        let (r1cs, profile) = match r1cs_gen.seal_with_profile(logup_repetitions) {
            Ok((r1cs, profile)) => (r1cs, Some(profile)),
            Err(error) => (error.into_r1cs(), None),
        };

        let mut num_non_zero_terms = 0;
        for r1c in r1cs.constraints.iter() {
            for (_, coeff) in r1c.a.iter() {
                if *coeff != F::ZERO {
                    num_non_zero_terms += 1;
                }
            }
            for (_, coeff) in r1c.b.iter() {
                if *coeff != F::ZERO {
                    num_non_zero_terms += 1;
                }
            }
            for (_, coeff) in r1c.c.iter() {
                if *coeff != F::ZERO {
                    num_non_zero_terms += 1;
                }
            }
        }

        self.r1cs_ssa = Some(r1cs_ssa);
        self.constraint_layouts = Some(crate::vm::bytecode::ConstraintLayouts::new(
            r1cs.witness_layout,
            r1cs.constraints_layout,
        ));
        self.r1cs_profile = profile;

        info!(
            message = %"R1CS generated",
            num_constraints = r1cs.constraints.len(),
            num_terms = num_non_zero_terms,
            algebraic_constraints = r1cs.constraints_layout.algebraic_size,
            tables_constraints = r1cs.constraints_layout.tables_data_size,
            lookups_constraints = r1cs.constraints_layout.lookups_data_size,
            total_witness = r1cs.witness_layout.size()

        );

        Ok(SealedR1CS::new(r1cs))
    }

    /// The LogUp repetitions `--logup-soundness` asks for, sized by the argument's degree `D`.
    fn logup_repetitions<F: MavrosField>(
        &self,
        r1cs_gen: &R1CGen<F>,
        field: FieldConfig,
    ) -> Result<usize, Error> {
        let table_entries = r1cs_gen.num_table_entries();
        let lookups = r1cs_gen.num_lookups();
        let report = logup_soundness_report(field, self.logup_soundness, table_entries + lookups)
            .map_err(Error::LogupSoundnessUnsupported)?;
        info!(
            message = %"LogUp soundness",
            requested_bits = report.requested_bits,
            field_bits = report.field_bits,
            table_entries,
            lookups,
            soundness_degree_d = report.soundness_degree,
            per_repetition_bits = report.per_repetition_bits,
            repetitions = report.repetitions,
            achieved_bits = report.achieved_bits,
        );
        if let Some((repetitions, bits)) = report.near_optimal_alternative {
            info!(
                message = %"LogUp soundness: a lower target saves a repetition",
                target_bits = bits,
                repetitions,
            );
        }
        Ok(report.repetitions as usize)
    }

    /// Only `pub` pins; a `call_data` parameter is private.
    fn abi_param_is_public(abi: &noirc_abi::Abi) -> Vec<bool> {
        abi.parameters
            .iter()
            .map(|param| param.visibility == noirc_abi::AbiVisibility::Public)
            .collect()
    }

    /// A parameter without recorded columns was elided, which only a private one may be.
    fn resolve_entry_columns(&self, recorded: &BTreeMap<Arc<str>, Vec<usize>>) -> EntryColumns {
        let blob = self
            .entry_blob_layout
            .as_ref()
            .expect("prepare_entry_point records the entry blob layout before R1CS generation");
        let run_of = |origin: String, width: usize, what: &str| -> Option<usize> {
            let columns = recorded.get(synthetic_file(origin).as_str())?;
            if columns.len() != width {
                ice!(
                    "{what} occupies {} witness columns but flattens to {width}",
                    columns.len()
                );
            }
            if !columns.windows(2).all(|pair| pair[1] == pair[0] + 1) {
                ice!("{what} occupies non-contiguous witness columns {columns:?}");
            }
            columns.first().copied()
        };
        let params = blob
            .params
            .iter()
            .enumerate()
            .map(|(index, slot)| {
                let column = run_of(
                    entry_param_origin(index),
                    slot.width,
                    &format!("entry parameter {index}"),
                );
                if column.is_none() && slot.width != 0 && slot.public {
                    ice!("public entry parameter {index} has no witness columns");
                }
                column
            })
            .collect();
        let returns = blob.return_width.and_then(|width| {
            let column = run_of(ENTRY_RETURN_ORIGIN.to_string(), width, "the return");
            if column.is_none() && width != 0 {
                ice!("the computed return has no witness columns");
            }
            column
        });
        EntryColumns { params, returns }
    }

    pub fn entry_columns(&self) -> Option<&EntryColumns> {
        self.entry_columns.as_ref()
    }

    /// Compiles the program (both the witgen and AD entry points) into a single VM binary.
    pub fn compile_bytecode(&mut self, options: CodeGenOptions) -> Result<Vec<u64>, Error> {
        Ok(self.compile_bytecode_artifact(options)?.binary)
    }

    /// Compiles VM bytecode and, when requested, returns debug information as a separate value.
    /// The executable `binary` is identical regardless of `include_debug_info`.
    pub fn compile_bytecode_artifact(
        &mut self,
        options: CodeGenOptions,
    ) -> Result<BytecodeArtifact, Error> {
        self.compile_vm_artifact(options, WitnessMode::Ad)
    }

    /// Compile a separate forward verifier binary without witgen or AD functions.
    pub fn compile_verifier_bytecode(
        &mut self,
        options: CodeGenOptions,
    ) -> Result<Vec<u64>, Error> {
        Ok(self.compile_verifier_bytecode_artifact(options)?.binary)
    }

    /// Compile the verifier-only binary and optional debug metadata.
    pub fn compile_verifier_bytecode_artifact(
        &mut self,
        options: CodeGenOptions,
    ) -> Result<BytecodeArtifact, Error> {
        self.compile_vm_artifact(options, WitnessMode::Verifier)
    }

    fn compile_vm_artifact(
        &mut self,
        options: CodeGenOptions,
        mode: WitnessMode,
    ) -> Result<BytecodeArtifact, Error> {
        self.prepare_program_ssa();
        let ssa = self.program_ssa.as_ref().unwrap();

        let flow_analysis = FlowAnalysis::run(ssa);
        let type_info = Types::new().run(ssa, &flow_analysis);

        let codegen = CodeGen::new(options, mode);
        let mut program = codegen.run(ssa, &flow_analysis, &type_info);
        program.constraint_layouts = self.constraint_layouts;
        // Only a project-backed driver has an ABI to check the header against; one from
        // [`Self::from_ssa`] states the entry's shape only in its HLSSA.
        if mode == WitnessMode::Ad && self.abi.is_some() {
            assert_eq!(
                program.entry_blob_field_count,
                self.entry_blob_field_count(),
                "ICE: entry blob field count in program header disagrees with the Noir ABI"
            );
        }
        self.write_debug_text(
            self.get_debug_output_dir()
                .join(if mode == WitnessMode::Verifier {
                    "verifier_bytecode.txt"
                } else {
                    "program_bytecode.txt"
                }),
            format!("{}", program),
        );

        let (binary, debug_info) = if options.include_debug_info {
            let (binary, debug_info) = program.to_binary_and_debug_info();
            (binary, Some(debug_info))
        } else {
            (program.to_binary(), None)
        };

        info!(message = %"Program binary generated", binary_size = binary.len() * 8);

        Ok(BytecodeArtifact { binary, debug_info })
    }

    pub fn abi(&self) -> &noirc_abi::Abi {
        self.abi.as_ref().unwrap()
    }

    /// Requires a project-backed driver after [`Self::generate_r1cs`]; validated against `r1cs`.
    pub fn witness_manifest<F: MavrosField>(
        &self,
        r1cs: &R1CS<F>,
    ) -> Result<WitnessManifest, Error> {
        let abi = self.abi.as_ref().ok_or_else(|| {
            Error::WitnessManifest("driver has no ABI (seeded from HLSSA)".to_string())
        })?;
        let blob = self.entry_blob_layout.as_ref().ok_or_else(|| {
            Error::WitnessManifest(
                "entry blob layout not captured; run make_struct_access_static first".to_string(),
            )
        })?;
        let columns = self.entry_columns.as_ref().ok_or_else(|| {
            Error::WitnessManifest(
                "entry columns not recorded; run generate_r1cs first".to_string(),
            )
        })?;
        if abi.parameters.len() != blob.params.len() || blob.total != abi.field_count() as usize {
            return Err(Error::WitnessManifest(format!(
                "ABI flattens to {} parameters and {} fields but the entry point has {} and {}",
                abi.parameters.len(),
                abi.field_count(),
                blob.params.len(),
                blob.total
            )));
        }
        let returns = match (blob.return_width, abi.return_type.as_ref()) {
            (Some(len), Some(ret)) => Some(ReturnSlot {
                column_start: columns.returns,
                len,
                visibility: abi_visibility(ret.visibility),
            }),
            (None, None) => None,
            _ => {
                return Err(Error::WitnessManifest(
                    "ABI and entry point disagree on the presence of a return".to_string(),
                ));
            }
        };
        let manifest = WitnessManifest {
            parameters: abi
                .parameters
                .iter()
                .zip(&blob.params)
                .zip(&columns.params)
                .map(|((param, slot), column_start)| ManifestParam {
                    name: param.name.clone(),
                    visibility: abi_visibility(param.visibility),
                    width: slot.width,
                    column_start: *column_start,
                })
                .collect(),
            returns,
        };
        manifest.validate(r1cs).map_err(Error::WitnessManifest)?;
        Ok(manifest)
    }

    pub fn r1cs_artifact<F: MavrosField>(
        &self,
        r1cs: SealedR1CS<F>,
    ) -> Result<R1CSArtifact<F>, Error> {
        let manifest = self.witness_manifest(&r1cs)?;
        Ok(R1CSArtifact::new(manifest, r1cs))
    }

    /// The parameters only; the return is computed, not an input.
    pub fn entry_blob_field_count(&self) -> usize {
        self.abi().field_count() as usize
    }

    /// Column 0, the surviving parameter columns and the return columns.
    pub fn protected_r1cs_cols(&self) -> Vec<usize> {
        let blob = self
            .entry_blob_layout
            .as_ref()
            .expect("prepare_entry_point records the entry blob layout");
        let entry = self
            .entry_columns
            .as_ref()
            .expect("generate_r1cs records the entry columns");
        let mut columns = vec![0];
        for (start, param) in entry.params.iter().zip(&blob.params) {
            if let Some(start) = start {
                columns.extend(*start..*start + param.width);
            }
        }
        if let Some(start) = entry.returns {
            let width = blob
                .return_width
                .expect("return columns have a return width");
            columns.extend(start..start + width);
        }
        columns
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
                Box::new(InstructionLowering::guards()),
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
                // Cleanup before lowering Map casts: dead hint computations (e.g. strip-maps
                // feeding unconstrained calls) are easy to delete while they are single
                // instructions, and hard to delete once expanded into loops with block-parameter
                // cycles.
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
                !cv.contains_fn_ptr(),
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
                // The merge above put two near-copies of the whole program into one SSA; fold the
                // byte-identical functions (notably `globals_init`) before anything downstream pays
                // for them. Both codegen backends emit every function in the SSA, so this is where
                // program size is won.
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
        use crate::compiler::codegen::llssa_to_llvm::{FieldRuntimeSymbols, LLVMCodeGen};
        use crate::compiler::ssa::hlssa_to_llssa;
        use inkwell::context::Context;

        // Every later stage's SSA takes the initial one's field.
        let field = self
            .initial_ssa
            .as_ref()
            .expect("a driver holds its initial SSA")
            .field();
        if FieldRuntimeSymbols::of(field).is_none() {
            return Err(Error::WasmFieldUnsupported(field.id()));
        }
        if r1cs.witness_layout.logup_repetitions > 1 {
            return Err(Error::WasmLogupRepetitionsUnsupported(
                r1cs.witness_layout.logup_repetitions,
            ));
        }

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
        let mut codegen = LLVMCodeGen::new(&context, "mavros_module", llssa.field());
        codegen.set_debug_path_root(
            wasm_config
                .as_ref()
                .and_then(|(_, opts)| opts.debug_path_root.clone()),
        );
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
            let debug_path = wasm_opts
                .include_debug_info
                .then(|| crate::compiler::codegen::llssa_to_llvm::wasm_debug_info_path(&wasm_path));
            codegen.compile_to_wasm(&wasm_path, wasm_opts)?;
            info!(message = %"WASM object generated", path = %wasm_path.display());
            if let Some(debug_path) = debug_path {
                info!(message = %"WASM debug info generated", path = %debug_path.display());
            }
            self.write_wasm_metadata(&wasm_path, r1cs)?;
        }

        Ok(llvm_ir)
    }

    /// Write WASM metadata JSON file.
    ///
    /// The sidecar names the entry point's parameters, so it exists only for a project-backed
    /// driver. A driver seeded with [`Self::from_ssa`] has no ABI and writes none.
    fn write_wasm_metadata(&self, wasm_path: &Path, r1cs: &R1CS) -> Result<(), Error> {
        let Some(abi) = self.abi.as_ref() else {
            info!(message = %"WASM metadata skipped: driver has no ABI");
            return Ok(());
        };

        // Build parameter info
        let mut parameters = Vec::new();
        for param in &abi.parameters {
            let element_count = param.typ.field_count();
            parameters.push(serde_json::json!({
                "name": param.name,
                "elementCount": element_count
            }));
        }

        let return_element_count = abi
            .return_type
            .as_ref()
            .map_or(0, |ret| ret.abi_type.field_count());

        let manifest = serde_json::to_value(self.witness_manifest(r1cs)?).unwrap();
        let metadata = serde_json::json!({
            "witnessCount": r1cs.witness_layout.size(),
            "constraintCount": r1cs.constraints.len(),
            "parameters": parameters,
            "hasReturn": abi.return_type.is_some(),
            "returnElementCount": return_element_count,
            "entryBlobFieldCount": self.entry_blob_field_count(),
            "witnessManifest": manifest
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

fn abi_visibility(visibility: noirc_abi::AbiVisibility) -> ParamVisibility {
    match visibility {
        noirc_abi::AbiVisibility::Public => ParamVisibility::Public,
        noirc_abi::AbiVisibility::Private => ParamVisibility::Private,
        noirc_abi::AbiVisibility::DataBus => ParamVisibility::DataBus,
    }
}

/// The text `file_manager` holds for `path`, which is how a [`SourceLocation`] names its file.
fn compiled_source(file_manager: &fm::FileManager, path: &str) -> Option<Arc<str>> {
    let id = file_manager.name_to_id(PathBuf::from(path))?;
    file_manager.fetch_file(id).map(Arc::from)
}

// TESTS
// ================================================================================================

#[cfg(test)]
mod tests {
    use noirc_abi::{AbiType, Sign};

    use super::{Diagnostic, Error, SourceLocation};
    use crate::compiler::{passes::prepare_entry_point::PrepareEntryPoint, ssa::hlssa::Type};

    /// ABI and HLSSA flattening must agree on the VM entry blob's width.
    #[test]
    fn abi_and_hlssa_field_counts_agree() {
        let mut cases: Vec<(AbiType, Type)> = vec![
            (AbiType::Field, Type::field()),
            (AbiType::Boolean, Type::int(1)),
            // A Noir `str<12>` lowers to an array of 12 bytes.
            (AbiType::String { length: 12 }, Type::int(8).array_of(12)),
            (
                AbiType::Array {
                    length: 4,
                    typ: Box::new(AbiType::Field),
                },
                Type::field().array_of(4),
            ),
            // Structs lower to tuples.
            (
                AbiType::Struct {
                    path: "Pair".to_string(),
                    fields: vec![
                        ("a".to_string(), AbiType::Field),
                        (
                            "b".to_string(),
                            AbiType::Array {
                                length: 2,
                                typ: Box::new(AbiType::Boolean),
                            },
                        ),
                    ],
                },
                Type::tuple_of(vec![Type::field(), Type::int(1).array_of(2)]),
            ),
            // Nesting: array of tuples.
            (
                AbiType::Array {
                    length: 3,
                    typ: Box::new(AbiType::Tuple {
                        fields: vec![
                            AbiType::Field,
                            AbiType::Integer {
                                sign: Sign::Unsigned,
                                width: 8,
                            },
                        ],
                    }),
                },
                Type::tuple_of(vec![Type::field(), Type::int(8)]).array_of(3),
            ),
            // WitnessOf is transparent on the HLSSA side; the ABI never sees it.
            (AbiType::Field, Type::witness_of(Type::field())),
        ];
        for width in [1, 8, 32, 64] {
            for sign in [Sign::Unsigned, Sign::Signed] {
                cases.push((AbiType::Integer { sign, width }, Type::int(width as usize)));
            }
        }
        for (abi, hlssa) in &cases {
            assert_eq!(
                abi.field_count() as usize,
                PrepareEntryPoint::flattened_field_count(hlssa),
                "flattening mismatch for ABI type {abi:?} vs HLSSA type {hlssa:?}",
            );
        }
    }

    /// The stdlib is the case the source hand-off exists for: `Project` registers those files under
    /// relative `std/...` paths that name nothing on disk, so a renderer that read the path would
    /// have no snippet to draw.
    #[test]
    fn the_compiled_source_of_an_embedded_file_is_found_by_its_path() {
        let mut file_manager = fm::FileManager::new(std::path::Path::new("/no/such/root"));
        let id = file_manager
            .add_file_with_source_canonical_path(
                std::path::Path::new("std/lib.nr"),
                "fn embedded() {}\n".to_string(),
            )
            .expect("the embedded file is registered");
        let path = file_manager.path(id).unwrap().to_string_lossy().to_string();

        assert_eq!(
            super::compiled_source(&file_manager, &path).as_deref(),
            Some("fn embedded() {}\n")
        );
        assert!(super::compiled_source(&file_manager, "std/absent.nr").is_none());
    }

    /// The error's `Display` has to be the rendered diagnostics in full: every caller here prints
    /// an `Error` and nothing asks it for a renderer.
    #[test]
    fn an_unsatisfiable_program_displays_as_its_rendered_diagnostics() {
        let refusal = Error::UnsatisfiableProgram(vec![
            Diagnostic::error("first refusal", SourceLocation::synthetic("a_pass"))
                .with_note("a standing fact"),
            Diagnostic::error("second refusal", SourceLocation::synthetic("a_pass")),
        ]);

        assert_eq!(
            refusal.to_string(),
            "error: first refusal\n  --> <a_pass>:1:1\n  = note: a standing fact\n\n\
             error: second refusal\n  --> <a_pass>:1:1\n"
        );
    }

    /// The runtime archive has no Goldilocks instance, so the WASM path refuses a Goldilocks
    /// program with an error before lowering it.
    #[test]
    fn the_wasm_path_refuses_a_goldilocks_program() {
        use mavros_artifacts::{FieldConfig, FieldId, Goldilocks};

        use super::Driver;
        use crate::compiler::{
            codegen::CodeGenOptions,
            ssa::hlssa::{HLSSA, builder::HLSSABuilder},
        };

        let mut ssa = HLSSA::with_main_in(FieldConfig::goldilocks(), "main".to_string());
        let main_id = ssa.get_unique_entrypoint_id();
        HLSSABuilder::new(&mut ssa).modify_function(main_id, |b| {
            let entry = b.function.get_entry_id();
            b.block(entry)
                .with_source_location(SourceLocation::synthetic("main"))
                .terminate_return(vec![]);
        });
        let scratch = tempfile::TempDir::new().unwrap();
        let mut driver = Driver::from_ssa(ssa, scratch.path().to_path_buf(), false);
        driver.make_struct_access_static().unwrap();
        driver.monomorphize().unwrap();
        driver.spill_witness().unwrap();
        let r1cs = driver.generate_r1cs::<Goldilocks>().unwrap();
        let options = CodeGenOptions {
            check_constraints: false,
            include_debug_info: false,
        };
        // The refusal precedes any read of the R1CS, so a bn254 one stands in for this program's.
        let bn254_r1cs = mavros_artifacts::R1CS {
            witness_layout: r1cs.witness_layout,
            constraints_layout: r1cs.constraints_layout,
            constraints: vec![],
        };
        match driver.compile_llvm_targets(false, &bn254_r1cs, None, options) {
            Err(Error::WasmFieldUnsupported(FieldId::Goldilocks)) => {}
            other => panic!("expected the WASM path to refuse Goldilocks, got {other:?}"),
        }
    }

    /// R1CS generation refuses coefficients in a field other than the program's.
    #[test]
    fn r1cs_generation_refuses_a_field_other_than_the_programs() {
        use mavros_artifacts::{Field, FieldConfig, FieldId};

        use super::Driver;
        use crate::compiler::ssa::hlssa::{HLSSA, builder::HLSSABuilder};

        let mut ssa = HLSSA::with_main_in(FieldConfig::goldilocks(), "main".to_string());
        let main_id = ssa.get_unique_entrypoint_id();
        HLSSABuilder::new(&mut ssa).modify_function(main_id, |b| {
            let entry = b.function.get_entry_id();
            b.block(entry)
                .with_source_location(SourceLocation::synthetic("main"))
                .terminate_return(vec![]);
        });
        let scratch = tempfile::TempDir::new().unwrap();
        let mut driver = Driver::from_ssa(ssa, scratch.path().to_path_buf(), false);
        driver.make_struct_access_static().unwrap();
        driver.monomorphize().unwrap();
        driver.spill_witness().unwrap();
        match driver.generate_r1cs::<Field>() {
            Err(Error::R1csFieldMismatch {
                requested: FieldId::Bn254,
                program: FieldId::Goldilocks,
            }) => {}
            other => panic!("expected a field mismatch, got {:?}", other.map(|_| ())),
        }
    }

    /// A serialized coefficient is its field's canonical limbs as a fixed-size array: four for
    /// bn254, with no length, and one for Goldilocks.
    #[test]
    fn r1cs_coefficients_serialize_as_their_canonical_limbs() {
        use ark_ff::PrimeField;
        use mavros_artifacts::{Field, Goldilocks, MavrosField, R1C};

        let minus_one = -<Field as MavrosField>::ONE;
        let bn254 = R1C {
            a: vec![(3, minus_one)],
            b: vec![],
            c: vec![(0, <Field as MavrosField>::ONE)],
        };
        let bytes = bincode::serialize(&bn254).unwrap();
        let expected = bincode::serialize(&(
            vec![(3usize, minus_one.into_bigint().0)],
            Vec::<(usize, [u64; 4])>::new(),
            vec![(0usize, [1u64, 0, 0, 0])],
        ))
        .unwrap();
        assert_eq!(bytes, expected);
        let back: R1C<Field> = bincode::deserialize(&bytes).unwrap();
        assert_eq!((back.a, back.c), (bn254.a, bn254.c));

        let top = -Goldilocks::ONE;
        let goldilocks = R1C {
            a: vec![(7, top)],
            b: vec![(1, Goldilocks::from(5u64))],
            c: vec![],
        };
        let bytes = bincode::serialize(&goldilocks).unwrap();
        let expected = bincode::serialize(&(
            vec![(7usize, [top.value()])],
            vec![(1usize, [5u64])],
            Vec::<(usize, [u64; 1])>::new(),
        ))
        .unwrap();
        assert_eq!(bytes, expected);
        let back: R1C<Goldilocks> = bincode::deserialize(&bytes).unwrap();
        assert_eq!((back.a, back.b), (goldilocks.a, goldilocks.b));
    }
}
