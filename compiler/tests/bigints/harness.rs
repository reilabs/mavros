//! An end-to-end test harness for programs consisting of hand-built SSA.
//!
//! This exists to allow testing of arbitrary-width integer support without the need for a Noir
//! front-end with syntactic support for the same. It is designed to check whether, for **given
//! operands** at a **given width**, the model, the constraint system, and the VM all agree.
//!
//! Concretely, [`BinaryOpOracle::check`]:
//!
//! 1. asks the model what the operation means (a value, or a rejection);
//! 2. compiles `main(a, b) -> c { a op b }` from hand-built SSA, declaring the model's answer as
//!    the return value, so that the entry point's own return assertion is the equality test;
//! 3. runs witness generation on the VM and checks every emitted constraint against the witness.
//!
//! An accepted run means the VM produced a witness, the witness satisfies the R1CS, and the value
//! it carries agrees with the model. A rejected run means the guard IR refused the operands.
//!
//! Every program is also held to its **AD** entry point on both lanes when it is compiled, since
//! that answer depends on the circuit rather than on the operands. See
//! [`Compiled::assert_the_ad_lane_agrees`].
//!
//! [`Compiled::run_wasm`] is the same judgement against the compiled WASM, run against the **same**
//! R1CS, so a disagreement between the two lanes is a disagreement between two implementations of
//! one program. It answers [`None`] only where no module came out of the linker; a lane that fails
//! to compile fails the test rather than reporting itself absent.

use std::path::{Path, PathBuf};

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
    vm::bytecode::DebugInfo,
    wasm_host, wasm_runtime,
};

// ---------------------------------------------------------------------------
// Building programs
// ---------------------------------------------------------------------------

/// Builds `main(params...) -> returns...` with `body` as its single-block body.
///
/// The parameters are ordinary typed entry-point parameters, so `prepare_entry_point` turns each
/// into a witness column with the range check its type calls for, and witness taint inference then
/// reads the body as the _witnessed_ lane. Constants folded in the body take the pure lane instead,
/// which is how the same builder reaches the compile-time evaluators.
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

/// Builds `main(lhs: int(lhs_bits), rhs: int(rhs_bits)) -> int(lhs_bits) { lhs op rhs }`.
pub fn binary_op_program(kind: BinaryArithOpKind, lhs_bits: usize, rhs_bits: usize) -> HLSSA {
    main_program(
        &[Type::int(lhs_bits), Type::int(rhs_bits)],
        &[Type::int(lhs_bits)],
        |e, params| vec![e.bin(kind, params[0], params[1])],
    )
}

// INPUTS
// ================================================================================================

/// The field element the entry point expects for one integer input.
///
/// `prepare_entry_point` writes each flattened input to a witness column as a single field element
/// and range-checks it there, so an integer crosses the boundary as its unsigned magnitude whatever
/// the operation will read it as.
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
    /// Witness generation ran, every constraint holds against the witness it produced, and the VM's
    /// own `a`/`b`/`c` evaluations agree with the constraint system.
    Accepted { witness: Vec<Field> },

    /// The VM refused to produce a witness at all — an assertion in the guard IR, a division check,
    /// a range check. This is what the model's `Rejected` predicts.
    Trapped { message: String, in_return_check: bool },

    /// A witness was produced but does not satisfy constraint `index`. An under-constrained circuit
    /// never lands here on an honest witness; a _wrongly_-constrained one does.
    Unsatisfied { index: usize },

    /// Every constraint holds, but the `a`/`b`/`c` vectors the VM emitted alongside the witness do
    /// not match the ones the constraint system computes from it. The witness is fine and the VM's
    /// own bookkeeping about it is not.
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

    /// The witness of an accepted run, for a caller that wants to perturb it.
    #[must_use]
    pub fn witness(&self) -> Option<&[Field]> {
        match self {
            Verdict::Accepted { witness } => Some(witness),
            _ => None,
        }
    }
}

/// A hand-built program taken all the way to executable artifacts.
///
/// Holds the R1CS and the VM binary together because the two are positionally coupled, and hence
/// meaningful only as a pair.
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
    /// [`Self::with_guard`] splices it in, so a caller states parameters and declared returns and
    /// nothing else.
    guard_slot: Option<usize>,

    /// The same program compiled to WASM, when a module came out of the linker.
    ///
    /// [`None`] only where none did; a compilation that *fails* fails loudly. See [`compile_wasm`]
    /// for which is which.
    wasm: Option<WasmArtifact>,
}

/// A linked WASM module and the directory holding it.
pub struct WasmArtifact {
    path: PathBuf,
    _scratch: TempDir,
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

        // `wrap_main` writes the entry blob as parameters, then a one-field return guard when
        // `main` returns anything, then the declared return values. The guard says whether to
        // check the declared return against the computed one, which is exactly what this oracle
        // is for, so it is set and the caller supplies only the values.
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
            // The VM's own constraint checking, which is what makes a trap the _first_ sign of
            // disagreement rather than a wrong witness discovered later.
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

    /// Runs the AD entry point on each lane and holds it to the constraint system.
    ///
    /// AD is the prover's other half: for random coefficients it computes each constraint matrix
    /// weighted by them, and that answer depends on the circuit alone, not on any input. It is
    /// checked once per program, here, rather than once per run, ensuring every test in this suite
    /// is run on the AD lane as well.
    ///
    /// A disagreement is a compiler bug rather than a verdict about operands, so it panics.
    ///
    /// The coefficients are random, so that no one choice of them can hide a wrong row, but drawn
    /// from a **seed** the panic names: a disagreement that only some coefficients expose is then
    /// replayed by setting [`AD_SEED_ENV`] to it, rather than lost with the run.
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
            let result =
                wasm_host::run_ad(&wasm.path, &self.r1cs, &coeffs).unwrap_or_else(|error| {
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

    /// Generates a witness by running the compiled WASM, and judges it the way [`Self::run`] does.
    ///
    /// The two lanes are judged against the **same** R1CS, which is the whole point: a disagreement
    /// between them is a disagreement between two implementations of one program, not between two
    /// programs. Returns [`None`] when the module could not be built.
    pub fn run_wasm(&self, inputs: &[InputValueOrdered]) -> Option<Verdict> {
        let wasm = self.wasm.as_ref()?;
        let inputs = self.with_guard(inputs);
        let result = match wasm_host::run_witgen(&wasm.path, &self.r1cs, &inputs) {
            Ok(result) => result,
            Err(error) => {
                return Some(Verdict::Trapped {
                    message: error.to_string(),
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

    /// `inputs` with the entry blob's return guard spliced in, set to check the declared return.
    ///
    /// The guard is the wrapper's own slot rather than a value of the program, so a test states
    /// the parameters and the declared return and this puts the blob together.
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

    /// The left operand's width, which is also the result's.
    ///
    /// Kept, with `rhs_bits`, so [`Self::check`] can turn away operands of another width.
    lhs_bits: usize,

    /// The right operand's width, which differs from `lhs_bits` only for a shift, which reads its
    /// amount at a width of its own.
    rhs_bits: usize,

    /// The program itself, compiled once here and run once per operand pair.
    compiled: Compiled,
}

impl BinaryOpOracle {
    /// Compiles `main(lhs, rhs) -> result { lhs op rhs }` at the given widths.
    ///
    /// # Errors
    ///
    /// Propagates a pipeline refusal. Note that a pipeline _panic_ is not caught.
    pub fn new(
        kind: BinaryArithOpKind,
        lhs_bits: usize,
        rhs_bits: usize,
    ) -> Result<Self, DriverError> {
        Ok(Self {
            kind,
            op: IntOp::from(kind),
            lhs_bits,
            rhs_bits,
            compiled: Compiled::new(binary_op_program(kind, lhs_bits, rhs_bits))?,
        })
    }

    /// The compiled artifacts, for a caller that wants to inspect or mutate a witness.
    #[must_use]
    pub fn compiled(&self) -> &Compiled {
        &self.compiled
    }

    /// Runs one operand pair and reports whether the pipeline agreed with [`mavros_int_semantics`].
    ///
    /// The model's answer enters the program as the _declared return value_, which the entry point
    /// wrapper constrains against what the body computed. So a value disagreement is not something
    /// this function inspects afterwards.
    ///
    /// A model rejection is expected to surface as a refusal: the guard IR is what turns "Noir
    /// would reject this" into a runtime trap. The residue is declared as the return so that a
    /// _missing_ guard shows up as an acceptance, and [`Verdict::is_refusal`] keeps a trap at the
    /// return check from being read as the guard having fired.
    ///
    /// # Errors
    ///
    /// Returns a description of the disagreement, suitable for an assertion message.
    ///
    /// # Panics
    ///
    /// If either operand's width is not the one this oracle was compiled for. An operand wider than
    /// its parameter would be caught only by the entry point's input range check, and would then be
    /// scored as a refusal, which is wrong.
    pub fn check(&self, lhs: &IntBits, rhs: &IntBits) -> Result<(), String> {
        assert_eq!(
            (lhs.bits(), rhs.bits()),
            (self.lhs_bits, self.rhs_bits),
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
    /// Returns a description if the wrong answer was accepted, or refused somewhere other than
    /// the return check. The latter means that the program refused the _operands_, so the run tells
    /// us nothing about the return.
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

        // Flipping the low bit lands on a different pattern of the same width, so it stays a
        // legal input — the entry point range-checks every declared value before the return
        // check ever sees it.
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

    /// The value to declare as the program's return.
    ///
    /// For an accepted operation it is the model's value. For a rejected one it is the residue (the
    /// pattern a total evaluator produces anyway) so that a guard that fails to fire is caught as
    /// an _acceptance_ of the residue rather than as a second, unrelated failure of the return
    /// assertion.
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
            // `None` is the model declining to specify the pattern (the division cases, where the
            // backends genuinely disagree). Zero stands in: the run is expected to refuse before
            // the return is ever compared, and if it does not, any declared value makes that
            // visible.
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
    let program = CodeGen::new(CodeGenOptions {
        check_constraints: false,
        include_debug_info: false,
    })
    .run(ssa, &flow_analysis, &type_info);
    format!("{program}")
}

/// Compile `driver`'s program to a linked WASM module, or [`None`] if no module came out.
///
/// A compilation **failure is a failure**, and says so: `compile_llvm_targets` reports a broken
/// WASM lane by returning an error, and swallowing one would leave every assertion below to pass
/// by skipping. The absent-toolchain case is not this one — `wasm_runtime::locate_or_build` and
/// `compile_to_wasm` both panic where `wasm-ld` or the runtime archive is missing, so a bare
/// `cargo test` outside the development shell never reaches the `None` here at all.
///
/// [`None`] is therefore narrow: a scratch directory that could not be made, or a linker that
/// reported success and wrote nothing. [`Compiled::wasm_is_available`] is how a test that needs
/// this lane says so, and it is the one thing that tells those two apart from a lane that ran.
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
    })
}
