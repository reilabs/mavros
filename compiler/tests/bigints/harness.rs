//! An end-to-end test harness for programs built directly as SSA.
//!
//! It tests arbitrary-width integers without a Noir front-end that can write them. For **given
//! operands** at a **given width**, it checks that the model, the constraint system and the VM
//! agree.
//!
//! [`BinaryOpOracle::check`]:
//!
//! 1. asks the model what the operation means (a value or a rejection);
//! 2. compiles `main(a, b) -> c { a op b }`, declaring the model's answer as the return value, so
//!    the entry point's return assertion is the equality test;
//! 3. runs witness generation on the VM and checks every constraint against the witness.
//!
//! An accepted run means the VM produced a witness that satisfies the R1CS and carries the model's
//! value. A rejected run means the guard IR refused the operands.
//!
//! Each compiled program is also checked against its **AD** entry point on both lanes, since that
//! answer depends on the circuit, not on the operands. See [`Compiled::assert_the_ad_lane_agrees`].
//!
//! [`Compiled::run_wasm`] makes the same judgement with the compiled WASM against the **same**
//! R1CS, so a disagreement between the lanes is one between two implementations of one program. It
//! answers [`None`] only when the linker produced no module; a lane that fails to compile fails the
//! test.

use std::{
    path::{Path, PathBuf},
    sync::OnceLock,
};

use ark_ff::UniformRand as _;
use mavros_artifacts::{Field, InputValueOrdered};
use mavros_int_semantics::{IntBits, IntOp, Outcome, residue};
use num_bigint::BigUint;
use rand::{SeedableRng as _, rngs::StdRng};
use tempfile::TempDir;

use mavros_compiler::{
    api,
    compiler::{
        analysis::{flow_analysis::FlowAnalysis, types::Types},
        codegen::{
            CodeGenOptions, bytecode::CodeGen, hlssa_to_r1cs::R1CS, llssa_to_llvm::WasmCompileOpts,
        },
        located::synthetic_file,
        passes::prepare_entry_point::{self, PrepareEntryPoint},
        ssa::{
            SourceLocation, ValueId,
            hlssa::{
                BinaryArithOpKind, HLSSA, Type,
                builder::{HLBlockEmitter, HLEmitter as _, HLSSABuilder},
            },
        },
    },
    driver::{Driver, Error as DriverError},
    vm::bytecode::{DebugInfo, parse_program_header},
    wasm_host::{self, LoadedModule},
    wasm_runtime,
};

// ---------------------------------------------------------------------------
// Building programs
// ---------------------------------------------------------------------------

/// Builds `main(params...) -> returns...` with `body` as its single-block body.
///
/// Each parameter becomes a range-checked witness column, so the body runs on the witnessed lane;
/// constants folded in it take the pure lane.
pub fn main_program(
    params: &[Type],
    returns: &[Type],
    body: impl FnOnce(&mut HLBlockEmitter<'_>, &[ValueId]) -> Vec<ValueId>,
) -> HLSSA {
    let mut ssa = HLSSA::with_main("main".to_string());
    let main_id = ssa.get_unique_entrypoint_id();
    let mut sb = HLSSABuilder::new(&mut ssa);
    sb.modify_function(main_id, |b| {
        for typ in returns {
            b.function.add_return_type(typ.clone());
        }
        let entry = b.function.get_entry_id();
        let mut e = b
            .block(entry)
            .with_source_location(SourceLocation::synthetic("oracle_main"));
        let param_values: Vec<ValueId> =
            params.iter().map(|t| e.add_parameter(t.clone())).collect();
        let results = body(&mut e, &param_values);
        e.terminate_return(results);
    });
    ssa
}

/// Builds `main(lhs: int(bits), rhs: int(bits)) -> int(bits) { lhs op rhs }`.
pub fn binary_op_program(kind: BinaryArithOpKind, bits: usize) -> HLSSA {
    main_program(
        &[Type::int(bits), Type::int(bits)],
        &[Type::int(bits)],
        |e, params| vec![e.bin(kind, params[0], params[1])],
    )
}

// INPUTS
// ================================================================================================

/// The field element the entry point expects for one integer input: its unsigned magnitude, however
/// the operation reads it.
#[must_use]
pub fn input_field(value: &IntBits) -> Field {
    Field::from(BigUint::from(value))
}

/// The positional input block: the entry point's parameters followed by its declared return
/// values, in declaration order, each already flattened by [`input_field`].
#[must_use]
pub fn input_block(values: &[&IntBits]) -> Vec<InputValueOrdered> {
    values
        .iter()
        .map(|v| InputValueOrdered::Field(input_field(v)))
        .collect()
}

// EXECUTION VERDICT
// ================================================================================================

/// The results of one run of a compiled program.
pub enum Verdict {
    /// Witness generation ran, every constraint holds against the witness it produced, and the
    /// `a`/`b`/`c` evaluations the VM emitted agree with the constraint system.
    Accepted { witness: Vec<Field> },

    /// The VM refused to produce a witness at all — an assertion in the guard IR, a division check,
    /// a range check. This is what the model's `Rejected` predicts.
    Trapped { message: String, in_return_check: bool },

    /// A witness was produced but does not satisfy constraint `index`. An under-constrained circuit
    /// never lands here on an honest witness; a _wrongly_-constrained one does.
    Unsatisfied { index: usize },

    /// Every constraint holds, but the `a`/`b`/`c` vectors the VM emitted do not match the ones the
    /// constraint system computes from the witness: the witness is right and the VM's bookkeeping
    /// is wrong.
    AbcMismatch,
}

impl std::fmt::Debug for Verdict {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Verdict::Accepted { witness } => {
                write!(f, "Accepted({} witness columns)", witness.len())
            }
            Verdict::Trapped {
                message,
                in_return_check,
            } => {
                let site = if *in_return_check {
                    "at the return check"
                } else {
                    "in the program"
                };
                write!(f, "Trapped {site}: {message}")
            }
            Verdict::Unsatisfied { index } => write!(f, "Unsatisfied at constraint {index}"),
            Verdict::AbcMismatch => write!(f, "AbcMismatch"),
        }
    }
}

impl Verdict {
    #[must_use]
    pub fn is_accepted(&self) -> bool {
        matches!(self, Verdict::Accepted { .. })
    }

    /// Whether the program refused its operands, as opposed to computing a value the declared
    /// return disagreed with. This is the verdict a model rejection predicts.
    #[must_use]
    pub fn is_refusal(&self) -> bool {
        matches!(
            self,
            Verdict::Trapped {
                in_return_check: false,
                ..
            }
        )
    }

    /// Whether the run refused its operands: by trapping, or by producing a witness that a
    /// constraint refuses.
    #[must_use]
    pub fn rejects(&self) -> bool {
        matches!(self, Verdict::Trapped { .. } | Verdict::Unsatisfied { .. })
    }

    /// The witness of an accepted run, for a caller that wants to perturb it.
    #[must_use]
    pub fn witness(&self) -> Option<&[Field]> {
        match self {
            Verdict::Accepted { witness } => Some(witness),
            _ => None,
        }
    }
}

/// A hand-built program compiled all the way to executable artifacts.
///
/// The R1CS and the VM binary are positionally coupled, so they are kept together.
pub struct Compiled {
    /// The compiled constraint system: the two layouts witness generation runs against, and the
    /// constraints a witness is judged by.
    r1cs: R1CS,

    /// The VM bytecode, holding both the witgen and the AD entry point, cloned per run.
    binary: Vec<u64>,

    /// Source metadata for the VM's stack frames.
    debug_info: Option<DebugInfo>,

    /// Where the pipeline wrote its per-stage SSA dumps.
    debug_output_dir: PathBuf,

    /// The temporary directory holding `debug_output_dir`; dropping it deletes the dumps.
    ///
    /// [`None`] when they were directed at a caller-named directory instead.
    scratch: Option<TempDir>,

    /// Where the entry blob's return guard sits, when `main` returns anything.
    ///
    /// [`Self::with_guard`] splices it in, so a caller states only parameters and declared returns.
    guard_slot: Option<usize>,

    /// The same program compiled to WASM, when the linker produced a module.
    ///
    /// [`None`] only when it did not; a compilation that _fails_ panics. See [`compile_wasm`].
    wasm: Option<WasmArtifact>,
}

/// A linked WASM module and the directory holding it.
pub struct WasmArtifact {
    path: PathBuf,
    _scratch: TempDir,

    /// The compiled module.
    loaded: OnceLock<Result<LoadedModule, String>>,
}

impl WasmArtifact {
    fn loaded(&self) -> Result<&LoadedModule, &str> {
        self.loaded
            .get_or_init(|| {
                LoadedModule::load(&self.path).map_err(|error| with_causes(error.as_ref()))
            })
            .as_ref()
            .map_err(String::as_str)
    }
}

/// `error` and every cause beneath it, where a wasm engine puts its reason.
fn with_causes(error: &(dyn std::error::Error + 'static)) -> String {
    let mut message = error.to_string();
    let mut cause = error.source();
    while let Some(inner) = cause {
        message.push_str(&format!(": {inner}"));
        cause = inner.source();
    }
    message
}

/// Environment variable naming a directory to keep the pipeline's per-stage dumps in.
pub const KEEP_DUMPS_ENV: &str = "MAVROS_ORACLE_DUMPS";

/// Environment variable fixing the seed the AD check draws its coefficients from, to replay a
/// failure it reported.
pub const AD_SEED_ENV: &str = "MAVROS_ORACLE_AD_SEED";

impl Compiled {
    /// Runs the production pipeline over `ssa`, from `make_struct_access_static` to the VM
    /// binary. The Noir frontend is the only stage skipped, because the program did not come
    /// from Noir source.
    pub fn new(ssa: HLSSA) -> Result<Self, DriverError> {
        let (debug_output_dir, scratch) = match std::env::var_os(KEEP_DUMPS_ENV) {
            Some(root) => {
                static NEXT: std::sync::atomic::AtomicUsize =
                    std::sync::atomic::AtomicUsize::new(0);
                let n = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                (PathBuf::from(root).join(format!("program-{n}")), None)
            }
            None => {
                let dir = TempDir::new().expect("a temporary directory for the pipeline's dumps");
                (PathBuf::from(dir.path()), Some(dir))
            }
        };

        // `wrap_main` lays the entry blob out as the parameters, a one-field return guard when
        // `main` returns anything, and the declared returns. The guard is set, so the declared
        // return is checked against the computed one.
        let entry = ssa.get_unique_entrypoint();
        let guard_slot = (!entry.get_returns().is_empty()).then(|| {
            entry
                .get_param_types()
                .iter()
                .map(PrepareEntryPoint::flattened_field_count)
                .sum::<usize>()
        });

        let mut driver = Driver::from_ssa(ssa, debug_output_dir.clone(), false);
        driver.make_struct_access_static()?;
        driver.monomorphize()?;
        driver.spill_witness()?;

        let r1cs = driver.generate_r1cs()?;
        let artifact = driver.compile_bytecode_artifact(CodeGenOptions {
            // The VM checks constraints as it runs, so a disagreement shows up first as a trap
            // rather than later as a wrong witness.
            check_constraints: true,
            include_debug_info: true,
        })?;

        let wasm = compile_wasm(&mut driver, &r1cs);

        let compiled = Self {
            r1cs,
            binary: artifact.binary,
            debug_info: artifact.debug_info,
            debug_output_dir,
            scratch,
            wasm,
            guard_slot,
        };
        compiled.assert_the_ad_lane_agrees();
        Ok(compiled)
    }

    /// Runs the AD entry point on each lane and holds it to the constraint system, panicking on a
    /// disagreement, which is a compiler bug.
    ///
    /// AD weights each constraint matrix by random coefficients, which depends on the circuit
    /// alone, so it is checked once per program. The coefficients come from a seed the panic names;
    /// setting [`AD_SEED_ENV`] to it replays the failure.
    fn assert_the_ad_lane_agrees(&self) {
        let seed = match std::env::var(AD_SEED_ENV) {
            Ok(seed) => seed
                .parse()
                .unwrap_or_else(|_| panic!("{AD_SEED_ENV} is not a u64: {seed:?}")),
            Err(_) => rand::random(),
        };
        let mut rng = StdRng::seed_from_u64(seed);
        let coeffs: Vec<Field> = (0..self.r1cs.constraints.len())
            .map(|_| Field::rand(&mut rng))
            .collect();
        let replay = format!("replay with {AD_SEED_ENV}={seed}");

        let mut binary = self.binary.clone();
        let (a, b, c, _) =
            api::run_ad_from_binary(&mut binary, &self.r1cs, &coeffs, self.debug_info.clone())
                .unwrap_or_else(|error| {
                    panic!("the AD entry point trapped on the VM: {error}; {replay}")
                });
        assert!(
            api::check_ad(&self.r1cs, &coeffs, &a, &b, &c),
            "the VM's AD entry point disagrees with the constraint system; {replay}"
        );

        if let Some(wasm) = &self.wasm {
            let loaded = wasm.loaded().unwrap_or_else(|error| {
                panic!("the WASM engine refused the module: {error}; {replay}")
            });
            let result =
                wasm_host::run_ad_loaded(loaded, &self.r1cs, &coeffs).unwrap_or_else(|error| {
                    panic!("the AD entry point trapped on WASM: {error}; {replay}")
                });
            assert!(
                api::check_ad(
                    &self.r1cs,
                    &coeffs,
                    &result.out_da,
                    &result.out_db,
                    &result.out_dc
                ),
                "the WASM AD entry point disagrees with the constraint system; {replay}"
            );
        }
    }

    /// Whether the WASM lane was built for this program.
    #[must_use]
    pub fn wasm_is_available(&self) -> bool {
        self.wasm.is_some()
    }

    /// Generates a witness by running the compiled WASM, and judges it as [`Self::run`] does,
    /// against the same R1CS. Returns [`None`] when the module could not be built.
    ///
    /// # Panics
    ///
    /// If the module was built but no engine will load it: a broken lane, which scored as a trap
    /// would pass a test that expects a rejection.
    pub fn run_wasm(&self, inputs: &[InputValueOrdered]) -> Option<Verdict> {
        let wasm = self.wasm.as_ref()?;
        let inputs = self.with_guard(inputs);
        let loaded = wasm
            .loaded()
            .unwrap_or_else(|error| panic!("the WASM module did not load: {error}"));
        let entry_blob_field_count = parse_program_header(&self.binary).entry_blob_field_count;
        let run = wasm_host::run_witgen_loaded(loaded, &self.r1cs, &inputs, entry_blob_field_count);
        let result = match run {
            Ok(result) => result,
            Err(error) => {
                return Some(Verdict::Trapped {
                    message: with_causes(error.as_ref()),
                    // A WASM trap carries no mavros stack, so the return check cannot be told from
                    // any other refusal here. Callers that need the distinction use `run`.
                    in_return_check: false,
                });
            }
        };

        let witness = [
            result.out_wit_pre_comm.as_slice(),
            result.out_wit_post_comm.as_slice(),
        ]
        .concat();
        if let Some(index) = self.first_unsatisfied_constraint(&witness) {
            return Some(Verdict::Unsatisfied { index });
        }
        if !self.r1cs.check_witgen_output(
            &result.out_wit_pre_comm,
            &result.out_wit_post_comm,
            &result.out_a,
            &result.out_b,
            &result.out_c,
        ) {
            return Some(Verdict::AbcMismatch);
        }
        Some(Verdict::Accepted { witness })
    }

    /// Where the pipeline wrote its per-stage SSA dumps, if they will outlive this value.
    ///
    /// [`None`] in the normal case, where they are in a temporary directory that is deleted with
    /// it. Set [`KEEP_DUMPS_ENV`] to get a non-temporary directory.
    #[must_use]
    pub fn kept_debug_output_dir(&self) -> Option<&Path> {
        self.scratch
            .is_none()
            .then_some(self.debug_output_dir.as_path())
    }

    /// The witness column the return guard is written to, where the entry blob has one.
    ///
    /// The witness opens with the constant one and then the input blob in order, so this is one
    /// past the guard's place in the blob.
    #[must_use]
    pub fn guard_column(&self) -> Option<usize> {
        self.guard_slot.map(|slot| slot + 1)
    }

    /// The witness column input `index` of a test's input block is written to: its place in the
    /// blob [`Self::run`] builds, past the constant one and the return guard where that sits
    /// before it.
    #[must_use]
    pub fn input_column(&self, index: usize) -> usize {
        match self.guard_slot {
            Some(slot) if slot <= index => index + 2,
            _ => index + 1,
        }
    }

    /// `inputs` with the entry blob's return guard spliced in, set to check the declared return.
    ///
    /// The guard is a slot of the wrapper, not a value of the program, so a test gives only the
    /// parameters and the declared return, and this assembles the blob.
    fn with_guard(&self, inputs: &[InputValueOrdered]) -> Vec<InputValueOrdered> {
        let Some(slot) = self.guard_slot else {
            return inputs.to_vec();
        };
        let mut blob = inputs.to_vec();
        blob.insert(slot, InputValueOrdered::Field(Field::from(1u64)));
        blob
    }

    /// Generates a witness for `inputs` by running the VM, and judges it.
    ///
    /// The counterpart of [`Self::run_wasm`], against the same R1CS.
    pub fn run(&self, inputs: &[InputValueOrdered]) -> Verdict {
        let inputs = self.with_guard(inputs);
        let inputs = inputs.as_slice();
        let mut binary = self.binary.clone();
        let result = match api::run_witgen_from_binary(
            &mut binary,
            &self.r1cs,
            inputs,
            self.debug_info.clone(),
        ) {
            Ok(result) => result,
            Err(error) => {
                let return_check = synthetic_file(prepare_entry_point::RETURN_CHECK_ORIGIN);
                let in_return_check = error
                    .stack_trace()
                    .iter()
                    .any(|frame| frame.location.file == return_check);
                return Verdict::Trapped {
                    message: error.to_string(),
                    in_return_check,
                };
            }
        };

        let witness = [
            result.out_wit_pre_comm.as_slice(),
            result.out_wit_post_comm.as_slice(),
        ]
        .concat();
        if let Some(index) = self.first_unsatisfied_constraint(&witness) {
            return Verdict::Unsatisfied { index };
        }
        if !api::check_witgen(&self.r1cs, &result) {
            return Verdict::AbcMismatch;
        }
        Verdict::Accepted { witness }
    }

    /// The index of the first constraint `witness` fails, or [`None`] if it satisfies them all.
    ///
    /// Delegates to [`R1CS::unsatisfied_constraint`].
    #[must_use]
    pub fn first_unsatisfied_constraint(&self, witness: &[Field]) -> Option<usize> {
        self.r1cs.unsatisfied_constraint(witness)
    }
}

// ENTRY POINT
// ================================================================================================

/// One compiled binary-operation program, held across operand pairs.
pub struct BinaryOpOracle {
    /// The SSA operation the program computes.
    kind: BinaryArithOpKind,

    /// The model's counterpart of `kind`, derived once here: the mapping is many-to-one, so it
    /// cannot be undone. `UShl` and `SShl` both answer `IntOp::Shl`.
    op: IntOp,

    /// The width of both operands and the result, which [`Self::check`] requires its operands to
    /// have.
    bits: usize,

    /// The program itself, compiled once here and run once per operand pair.
    compiled: Compiled,
}

impl BinaryOpOracle {
    /// Compiles `main(lhs, rhs) -> result { lhs op rhs }` at the given width.
    ///
    /// # Errors
    ///
    /// Propagates a pipeline refusal. A pipeline _panic_ is not caught.
    pub fn new(kind: BinaryArithOpKind, bits: usize) -> Result<Self, DriverError> {
        Ok(Self {
            kind,
            op: IntOp::from(kind),
            bits,
            compiled: Compiled::new(binary_op_program(kind, bits))?,
        })
    }

    /// The compiled artifacts, for a caller that wants to inspect or mutate a witness.
    #[must_use]
    pub fn compiled(&self) -> &Compiled {
        &self.compiled
    }

    /// Runs one operand pair and reports whether the pipeline agreed with [`mavros_int_semantics`].
    ///
    /// The model's answer is the declared return, which the entry point checks against the
    /// computed one. For a pair the model rejects, the residue is declared instead, so a missing
    /// guard shows as an acceptance, and [`Verdict::is_refusal`] keeps a trap at the return check
    /// from counting as the guard firing.
    ///
    /// # Errors
    ///
    /// Returns a description of the disagreement, suitable for an assertion message.
    ///
    /// # Panics
    ///
    /// If either operand's width is not the one this oracle was compiled for: a wider operand would
    /// be refused by the entry point's range check and wrongly scored as a refusal.
    pub fn check(&self, lhs: &IntBits, rhs: &IntBits) -> Result<(), String> {
        assert_eq!(
            (lhs.bits(), rhs.bits()),
            (self.bits, self.bits),
            "operands do not match the widths this oracle was compiled for"
        );
        let expectation = eval_expectation(self.op, lhs, rhs);
        let verdict = self
            .compiled
            .run(&input_block(&[lhs, rhs, &expectation.declared_return]));

        if expectation.accepted {
            if verdict.is_accepted() {
                return Ok(());
            }
            return Err(format!(
                "{:?}({lhs:?}, {rhs:?}) is {:?} in the model, but the pipeline answered \
                 {verdict:?}{}",
                self.kind,
                expectation.declared_return,
                self.where_to_look()
            ));
        }

        if verdict.is_refusal() {
            return Ok(());
        }

        Err(format!(
            "{:?}({lhs:?}, {rhs:?}) is rejected by the model, but the pipeline answered \
             {verdict:?}{}",
            self.kind,
            self.where_to_look()
        ))
    }

    /// The tail of a failure message: where the per-stage dumps are, when there are any to point
    /// at, and otherwise how to get them.
    fn where_to_look(&self) -> String {
        match self.compiled.kept_debug_output_dir() {
            Some(dir) => format!(" — SSA dumps in {}", dir.display()),
            None => format!(" — set {KEEP_DUMPS_ENV} to keep the SSA dumps"),
        }
    }

    /// Confirms that this program's declared return really is constrained, by running the same
    /// operands with an answer the model did not give and requiring the return check to refuse it.
    ///
    /// # Errors
    ///
    /// Returns a description if the wrong answer was accepted, or refused somewhere other than the
    /// return check, which means the operands were refused and says nothing about the return.
    ///
    /// # Panics
    ///
    /// If the model rejects these operands, since then there is no right answer to perturb.
    pub fn check_a_wrong_answer_is_refused(
        &self,
        lhs: &IntBits,
        rhs: &IntBits,
    ) -> Result<(), String> {
        let expected = mavros_int_semantics::eval(self.op, lhs, rhs)
            .value()
            .expect("operands the model rejects have no right answer to perturb");

        // Flipping the low bit keeps the value in range, so only the return check can refuse it.
        let wrong = expected.xor(&IntBits::from_u128(expected.bits(), 1));

        let verdict = self.compiled.run(&input_block(&[lhs, rhs, &wrong]));
        match verdict {
            Verdict::Trapped {
                in_return_check: true,
                ..
            } => Ok(()),
            other => Err(format!(
                "{:?}({lhs:?}, {rhs:?}): the return is not constrained — declaring {wrong:?} \
                 instead of {expected:?} gave {other:?}{}",
                self.kind,
                self.where_to_look()
            )),
        }
    }
}

/// What the model says about one operand pair, in the form the oracle needs it.
struct Expectation {
    /// Whether the model gives the operation a value, as opposed to rejecting it.
    accepted: bool,

    /// The value to declare as the program's return: the model's value, or for a rejected
    /// operation the residue, so a guard that fails to fire shows as an acceptance.
    declared_return: IntBits,
}

fn eval_expectation(op: IntOp, lhs: &IntBits, rhs: &IntBits) -> Expectation {
    match mavros_int_semantics::eval(op, lhs, rhs) {
        Outcome::Value(value) => Expectation {
            accepted: true,
            declared_return: value,
        },
        Outcome::Rejected(_) => Expectation {
            accepted: false,
            // `None` means the model leaves the pattern unspecified (division, where the backends
            // disagree). Zero stands in: the run should refuse before the return is compared.
            declared_return: residue(op, lhs, rhs).unwrap_or_else(|| IntBits::zero(lhs.bits())),
        },
    }
}

// BYTECODE DISPATCH
// ================================================================================================

/// The VM bytecode `ssa` lowers to, as the disassembly the driver writes to `program_bytecode.txt`.
///
/// Code generation only, with none of the passes between, useful for debugging.
#[must_use]
pub fn bytecode_listing(ssa: &HLSSA) -> String {
    let flow_analysis = FlowAnalysis::run(ssa);
    let type_info = Types::new().run(ssa, &flow_analysis);
    let program = CodeGen::new(
        CodeGenOptions {
            check_constraints: false,
            include_debug_info: false,
        },
        mavros_compiler::compiler::codegen::bytecode::WitnessMode::Ad,
    )
    .run(ssa, &flow_analysis, &type_info);
    format!("{program}")
}

/// Compile `driver`'s program to a linked WASM module, or [`None`] if no module came out.
///
/// A failed compilation panics, as does a missing `wasm-ld` or runtime archive, so a broken lane
/// cannot pass by skipping. [`None`] means only that no scratch directory could be made, or that
/// the linker succeeded without writing a module; [`Compiled::wasm_is_available`] tells those
/// apart from a lane that ran.
fn compile_wasm(driver: &mut Driver, r1cs: &R1CS) -> Option<WasmArtifact> {
    let scratch = TempDir::new().ok()?;
    let path = scratch.path().join("program.wasm");
    let options = WasmCompileOpts::fast(wasm_runtime::locate_or_build());

    driver
        .compile_llvm_targets(
            false,
            r1cs,
            Some((path.clone(), options)),
            CodeGenOptions::default(),
        )
        .unwrap_or_else(|error| panic!("the WASM lane failed to compile: {error}"));

    path.exists().then_some(WasmArtifact {
        path,
        _scratch: scratch,
        loaded: OnceLock::new(),
    })
}
