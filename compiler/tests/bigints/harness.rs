//! An end-to-end test harness for programs built directly as SSA.
//!
//! It tests arbitrary-width integers without a Noir front-end that can write them. For **given
//! operands** at a **given width**, it checks that the model, the constraint system and the VM
//! agree.
//!
//! [`BinaryOpOracle::check`]:
//!
//! 1. asks the model what the operation means (a value or a rejection);
//! 2. compiles `main(a, b) -> c { a op b }` and declares the model's answer as the expected
//!    return value, which the harness compares with the return columns the program wrote;
//! 3. runs witness generation on the VM and checks every constraint against the witness.
//!
//! An accepted run means the VM produced a witness that satisfies the R1CS and carries the model's
//! value. A rejected run means the guard IR refused the operands.
//!
//! Each compiled program is also checked against its **AD** entry point on each lane it has, since
//! that answer depends on the circuit alone. See [`Compiled::assert_the_ad_lane_agrees`].
//!
//! [`Compiled::run_wasm`] makes the same judgement with the compiled WASM against the **same**
//! R1CS, so a disagreement between the lanes is one between two implementations of one program.
//! Only bn254 has a WASM runtime, so only a bn254 program has that lane. It answers [`None`] only
//! when the linker produced no module; a lane that fails to compile fails the test.

use std::{
    any::Any,
    path::{Path, PathBuf},
    sync::OnceLock,
};

use ark_ff::UniformRand as _;
use mavros_artifacts::{Field, FieldConfig, Goldilocks, InputValueOrdered, MavrosField};
use mavros_int_semantics::{IntBits, IntOp, Outcome, residue};
use num_bigint::BigUint;
use rand::{RngCore as _, SeedableRng as _, rngs::StdRng};
use tempfile::TempDir;

use mavros_compiler::{
    api,
    compiler::{
        analysis::{flow_analysis::FlowAnalysis, types::Types},
        codegen::{
            CodeGenOptions,
            bytecode::CodeGen,
            hlssa_to_r1cs::{R1CS, SealedR1CS},
            llssa_to_llvm::WasmCompileOpts,
        },
        passes::prepare_entry_point::PrepareEntryPoint,
        ssa::{
            SourceLocation, ValueId,
            hlssa::{
                BinaryArithOpKind, HLSSA, Type,
                builder::{HLBlockEmitter, HLEmitter as _, HLSSABuilder},
            },
        },
    },
    driver::{Driver, EntryColumns, Error as DriverError},
    vm::bytecode::{DebugInfo, parse_program_header},
    wasm_host::{self, LoadedModule},
    wasm_runtime,
};

// ---------------------------------------------------------------------------
// Fields
// ---------------------------------------------------------------------------

/// A field the harness compiles and runs programs over.
pub trait HarnessField: MavrosField {
    /// The field's configuration, which a program built for it carries.
    const CONFIG: FieldConfig;

    /// The LogUp soundness target the harness compiles at, or [`None`] for the driver's default.
    const LOGUP_SOUNDNESS_BITS: Option<u32>;

    /// A random element, for the coefficients of the AD check.
    fn random(rng: &mut StdRng) -> Self;

    /// `value` as an element. bn254 reduces it modulo the modulus; Goldilocks refuses a value at or
    /// above its modulus, since no entry point takes one.
    fn of_biguint(value: &BigUint) -> Self;
}

impl HarnessField for Field {
    const CONFIG: FieldConfig = FieldConfig::bn254();
    const LOGUP_SOUNDNESS_BITS: Option<u32> = None;

    fn random(rng: &mut StdRng) -> Self {
        Field::rand(rng)
    }

    fn of_biguint(value: &BigUint) -> Self {
        Field::from(value.clone())
    }
}

impl HarnessField for Goldilocks {
    const CONFIG: FieldConfig = FieldConfig::goldilocks();

    // A program with an integer parameter range-checks it, so it has a lookup argument, and the
    // driver's default target needs more LogUp challenges on a 64-bit field than R1CS generation
    // implements. 40 bits takes one challenge for any argument of degree up to `2^23`. This
    // under-provisions the argument by design and is for testing the arithmetic only.
    const LOGUP_SOUNDNESS_BITS: Option<u32> = Some(40);

    // A uniform 128-bit integer reduced modulo `p < 2^64` is within `2^-64` of uniform.
    fn random(rng: &mut StdRng) -> Self {
        Goldilocks::from(u128::from(rng.next_u64()) << 64 | u128::from(rng.next_u64()))
    }

    fn of_biguint(value: &BigUint) -> Self {
        u64::try_from(value)
            .ok()
            .and_then(Goldilocks::from_canonical)
            .unwrap_or_else(|| {
                panic!(
                    "the input {value} is not below the Goldilocks modulus, so no entry point \
                     takes it"
                )
            })
    }
}

// ---------------------------------------------------------------------------
// Building programs
// ---------------------------------------------------------------------------

/// Builds `main(params...) -> returns...` over bn254 with `body` as its single-block body.
///
/// Each parameter becomes a witness column, range-checked for an integer, so the body runs on the
/// witnessed lane; constants folded in it take the pure lane.
pub fn main_program(
    params: &[Type],
    returns: &[Type],
    body: impl FnOnce(&mut HLBlockEmitter<'_>, &[ValueId]) -> Vec<ValueId>,
) -> HLSSA {
    main_program_in(FieldConfig::bn254(), params, returns, body)
}

/// [`main_program`] over `field`.
pub fn main_program_in(
    field: FieldConfig,
    params: &[Type],
    returns: &[Type],
    body: impl FnOnce(&mut HLBlockEmitter<'_>, &[ValueId]) -> Vec<ValueId>,
) -> HLSSA {
    let mut ssa = HLSSA::with_main_in(field, "main".to_string());
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

/// Builds `main(lhs: int(bits), rhs: int(bits)) -> int(bits) { lhs op rhs }` over `field`.
pub fn binary_op_program_in(field: FieldConfig, kind: BinaryArithOpKind, bits: usize) -> HLSSA {
    main_program_in(
        field,
        &[Type::int(bits), Type::int(bits)],
        &[Type::int(bits)],
        |e, params| vec![e.bin(kind, params[0], params[1])],
    )
}

// INPUTS
// ================================================================================================

/// The bn254 element the entry point expects for one integer input: its unsigned magnitude, however
/// the operation reads it.
#[must_use]
pub fn input_field(value: &IntBits) -> Field {
    input_element(value)
}

/// [`input_field`] in `F`.
#[must_use]
pub fn input_element<F: HarnessField>(value: &IntBits) -> F {
    F::of_biguint(&BigUint::from(value))
}

/// The positional input block over bn254: the entry point's parameters followed by its declared
/// return values, in declaration order, each flattened by [`input_field`].
#[must_use]
pub fn input_block(values: &[&IntBits]) -> Vec<InputValueOrdered> {
    input_block_in(values)
}

/// [`input_block`] in `F`.
#[must_use]
pub fn input_block_in<F: HarnessField>(values: &[&IntBits]) -> Vec<InputValueOrdered<F>> {
    values
        .iter()
        .map(|v| InputValueOrdered::Field(input_element(v)))
        .collect()
}

// EXECUTION VERDICT
// ================================================================================================

/// The results of one run of a compiled program.
pub enum Verdict<F: MavrosField = Field> {
    /// Witness generation ran, every constraint holds against the witness it produced, and the
    /// `a`/`b`/`c` evaluations the VM emitted agree with the constraint system.
    Accepted { witness: Vec<F> },

    /// The VM refused to produce a witness at all — an assertion in the guard IR, a division check,
    /// a range check. This is what the model's `Rejected` predicts.
    Trapped { message: String },

    /// A witness was produced and satisfies every constraint, but the return columns the program
    /// wrote hold `actual` where the test declared `expected`: the program computed a different
    /// answer.
    ReturnMismatch { expected: Vec<F>, actual: Vec<F> },

    /// A witness was produced but does not satisfy constraint `index`. An under-constrained circuit
    /// never lands here on an honest witness; a _wrongly_-constrained one does.
    Unsatisfied { index: usize },

    /// Every constraint holds, but the `a`/`b`/`c` vectors the VM emitted do not match the ones the
    /// constraint system computes from the witness: the witness is right and the VM's bookkeeping
    /// is wrong.
    AbcMismatch,
}

impl<F: MavrosField> std::fmt::Debug for Verdict<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Verdict::Accepted { witness } => {
                write!(f, "Accepted({} witness columns)", witness.len())
            }
            Verdict::Trapped { message } => write!(f, "Trapped: {message}"),
            Verdict::ReturnMismatch { expected, actual } => {
                write!(
                    f,
                    "ReturnMismatch: expected {expected:?}, computed {actual:?}"
                )
            }
            Verdict::Unsatisfied { index } => write!(f, "Unsatisfied at constraint {index}"),
            Verdict::AbcMismatch => write!(f, "AbcMismatch"),
        }
    }
}

impl<F: MavrosField> Verdict<F> {
    #[must_use]
    pub fn is_accepted(&self) -> bool {
        matches!(self, Verdict::Accepted { .. })
    }

    /// Whether the program refused its operands, as opposed to computing a value the declared
    /// return disagreed with. This is the verdict a model rejection predicts.
    #[must_use]
    pub fn is_refusal(&self) -> bool {
        matches!(self, Verdict::Trapped { .. })
    }

    /// Whether the run refused its operands: by trapping, or by producing a witness that a
    /// constraint refuses.
    #[must_use]
    pub fn rejects(&self) -> bool {
        matches!(self, Verdict::Trapped { .. } | Verdict::Unsatisfied { .. })
    }

    /// The witness of an accepted run, for a caller that wants to perturb it.
    #[must_use]
    pub fn witness(&self) -> Option<&[F]> {
        match self {
            Verdict::Accepted { witness } => Some(witness),
            _ => None,
        }
    }
}

/// A hand-built program compiled all the way to executable artifacts.
///
/// The R1CS and the VM binary are positionally coupled, so they are kept together.
pub struct Compiled<F: HarnessField = Field> {
    /// The compiled constraint system: the two layouts witness generation runs against, and the
    /// constraints a witness is judged by.
    r1cs: SealedR1CS<F>,

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

    /// Flattened width of each parameter, in signature order.
    param_widths: Vec<usize>,

    /// Flattened width of the parameters: the entry blob, and the prefix of a test's input block.
    param_width: usize,

    /// Flattened width of the return, which follows the parameters in a test's input block.
    return_width: usize,

    /// Where the parameters and the return landed in the witness.
    entry_columns: EntryColumns,

    /// The same program compiled to WASM, when the field has a WASM runtime and the linker produced
    /// a module.
    ///
    /// [`None`] for a field without a WASM runtime, or when no module came out; a compilation that
    /// _fails_ panics. See [`compile_wasm`].
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
    /// [`Self::compile`] for a program over bn254, the field with a WASM lane.
    pub fn new(ssa: HLSSA) -> Result<Self, DriverError> {
        Self::compile(ssa)
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
        let (params, declared) = self.split_inputs(inputs);
        let loaded = wasm
            .loaded()
            .unwrap_or_else(|error| panic!("the WASM module did not load: {error}"));
        let entry_blob_field_count = parse_program_header(&self.binary).entry_blob_field_count;
        let run = wasm_host::run_witgen_loaded(loaded, &self.r1cs, params, entry_blob_field_count);
        let result = match run {
            Ok(result) => result,
            Err(error) => {
                return Some(Verdict::Trapped {
                    message: with_causes(error.as_ref()),
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
        Some(self.judge_return(witness, declared))
    }
}

impl<F: HarnessField> Compiled<F> {
    /// This program as a bn254 one, the field with a WASM lane, or [`None`] over any other field.
    #[must_use]
    pub fn as_bn254(&self) -> Option<&Compiled> {
        (self as &dyn Any).downcast_ref::<Compiled>()
    }

    /// Runs the production pipeline over `ssa`, from `make_struct_access_static` to the VM
    /// binary. The Noir frontend is the only stage skipped, because the program did not come
    /// from Noir source.
    ///
    /// `ssa` must be a program over `F`. Only bn254 has a WASM runtime, so only a bn254 program
    /// gets a WASM lane.
    pub fn compile(ssa: HLSSA) -> Result<Self, DriverError> {
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

        // The entry blob is the parameters; the return is computed and written after them. A
        // test's input block states both, so it is split at the parameter width.
        let entry = ssa.get_unique_entrypoint();
        let param_widths: Vec<usize> = entry
            .get_param_types()
            .iter()
            .map(PrepareEntryPoint::flattened_field_count)
            .collect();
        let param_width = param_widths.iter().sum::<usize>();
        let return_width = entry
            .get_returns()
            .iter()
            .map(PrepareEntryPoint::flattened_field_count)
            .sum::<usize>();

        let mut driver = Driver::from_ssa(ssa, debug_output_dir.clone(), false);
        if let Some(bits) = F::LOGUP_SOUNDNESS_BITS {
            driver.set_logup_soundness(bits);
        }
        driver.make_struct_access_static()?;
        driver.monomorphize()?;
        driver.spill_witness()?;

        let r1cs = driver.generate_r1cs::<F>()?;
        let entry_columns = driver
            .entry_columns()
            .cloned()
            .expect("generate_r1cs records the entry columns");
        let artifact = driver.compile_bytecode_artifact(CodeGenOptions {
            // The VM checks constraints as it runs, so a disagreement shows up first as a trap
            // rather than later as a wrong witness.
            check_constraints: true,
            include_debug_info: true,
        })?;

        let wasm = (r1cs.system() as &dyn Any)
            .downcast_ref::<R1CS>()
            .and_then(|r1cs| compile_wasm(&mut driver, r1cs));

        let compiled = Self {
            r1cs,
            binary: artifact.binary,
            debug_info: artifact.debug_info,
            debug_output_dir,
            scratch,
            wasm,
            param_widths,
            param_width,
            return_width,
            entry_columns,
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
        let coeffs: Vec<F> = (0..self.r1cs.constraints.len())
            .map(|_| F::random(&mut rng))
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
            let (r1cs, coeffs) = (self.r1cs.system() as &dyn Any)
                .downcast_ref::<R1CS>()
                .zip((&coeffs as &dyn Any).downcast_ref::<Vec<Field>>())
                .expect("only a bn254 program has a WASM lane");
            let loaded = wasm.loaded().unwrap_or_else(|error| {
                panic!("the WASM engine refused the module: {error}; {replay}")
            });
            let result = wasm_host::run_ad_loaded(loaded, r1cs, coeffs).unwrap_or_else(|error| {
                panic!("the AD entry point trapped on WASM: {error}; {replay}")
            });
            assert!(
                api::check_ad(r1cs, coeffs, &result.out_da, &result.out_db, &result.out_dc),
                "the WASM AD entry point disagrees with the constraint system; {replay}"
            );
        }
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

    /// The witness column input `index` of a test's input block is written to. Every parameter
    /// of a hand-built program is treated as public, so each has columns; the declared returns
    /// past the parameters name the columns the program's computed return was written to.
    #[must_use]
    pub fn input_column(&self, index: usize) -> usize {
        if index < self.param_width {
            self.param_columns()[index]
        } else {
            self.entry_columns
                .returns
                .expect("a program with a return writes its return columns")
                + (index - self.param_width)
        }
    }

    /// The witness columns the program's computed return was written to; empty for a program
    /// without one.
    #[must_use]
    pub fn return_columns(&self) -> std::ops::Range<usize> {
        if self.return_width == 0 {
            return 0..0;
        }
        let start = self
            .entry_columns
            .returns
            .expect("a program with a return writes its return columns");
        start..start + self.return_width
    }

    /// The witness column of each parameter field, in blob order.
    fn param_columns(&self) -> Vec<usize> {
        self.entry_columns
            .params
            .iter()
            .zip(&self.param_widths)
            .flat_map(|(start, width)| {
                // A zero-width parameter has no columns and no `column_start`.
                if *width == 0 {
                    return 0..0;
                }
                let start = start.expect("every parameter of a hand-built program is pinned");
                start..start + width
            })
            .collect()
    }

    /// A test's input block split into the entry blob and the declared return.
    fn split_inputs<'a>(
        &self,
        inputs: &'a [InputValueOrdered<F>],
    ) -> (&'a [InputValueOrdered<F>], &'a [InputValueOrdered<F>]) {
        assert_eq!(
            inputs.len(),
            self.param_width + self.return_width,
            "an input block states every parameter and every declared return field"
        );
        inputs.split_at(self.param_width)
    }

    /// Accepts `witness` unless the return columns disagree with `declared`.
    fn judge_return(&self, witness: Vec<F>, declared: &[InputValueOrdered<F>]) -> Verdict<F> {
        if self.return_width == 0 {
            return Verdict::Accepted { witness };
        }
        let actual = witness[self.return_columns()].to_vec();
        let expected: Vec<F> = declared
            .iter()
            .map(|value| match value {
                InputValueOrdered::Field(field) => *field,
                other => panic!("a declared return field is a field element: {other:?}"),
            })
            .collect();
        if actual == expected {
            Verdict::Accepted { witness }
        } else {
            Verdict::ReturnMismatch { expected, actual }
        }
    }

    /// Generates a witness for `inputs` by running the VM, and judges it.
    ///
    /// The counterpart of [`Self::run_wasm`], against the same R1CS.
    pub fn run(&self, inputs: &[InputValueOrdered<F>]) -> Verdict<F> {
        let (params, declared) = self.split_inputs(inputs);
        let mut binary = self.binary.clone();
        let result = match api::run_witgen_from_binary(
            &mut binary,
            &self.r1cs,
            params,
            self.debug_info.clone(),
        ) {
            Ok(result) => result,
            Err(error) => {
                return Verdict::Trapped {
                    message: error.to_string(),
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
        self.judge_return(witness, declared)
    }

    /// The index of the first constraint `witness` fails, or [`None`] if it satisfies them all.
    ///
    /// Delegates to [`R1CS::unsatisfied_constraint`].
    #[must_use]
    pub fn first_unsatisfied_constraint(&self, witness: &[F]) -> Option<usize> {
        self.r1cs.unsatisfied_constraint(witness)
    }
}

// ENTRY POINT
// ================================================================================================

/// One compiled binary-operation program, held across operand pairs.
pub struct BinaryOpOracle<F: HarnessField = Field> {
    /// The SSA operation the program computes.
    kind: BinaryArithOpKind,

    /// The model's counterpart of `kind`, derived once here: the mapping is many-to-one, so it
    /// cannot be undone. `UShl` and `SShl` both answer `IntOp::Shl`.
    op: IntOp,

    /// The width of both operands and the result, which [`Self::check`] requires its operands to
    /// have.
    bits: usize,

    /// The program itself, compiled once here and run once per operand pair.
    compiled: Compiled<F>,
}

impl BinaryOpOracle {
    /// [`Self::compile`] over bn254.
    ///
    /// # Errors
    ///
    /// Propagates a pipeline refusal. A pipeline _panic_ is not caught.
    pub fn new(kind: BinaryArithOpKind, bits: usize) -> Result<Self, DriverError> {
        Self::compile(kind, bits)
    }
}

impl<F: HarnessField> BinaryOpOracle<F> {
    /// Compiles `main(lhs, rhs) -> result { lhs op rhs }` over `F` at the given width.
    ///
    /// # Errors
    ///
    /// Propagates a pipeline refusal. A pipeline _panic_ is not caught.
    pub fn compile(kind: BinaryArithOpKind, bits: usize) -> Result<Self, DriverError> {
        Ok(Self {
            kind,
            op: IntOp::from(kind),
            bits,
            compiled: Compiled::compile(binary_op_program_in(F::CONFIG, kind, bits))?,
        })
    }

    /// The compiled artifacts, for a caller that wants to inspect or mutate a witness.
    #[must_use]
    pub fn compiled(&self) -> &Compiled<F> {
        &self.compiled
    }

    /// Runs one operand pair and reports whether the pipeline agreed with [`mavros_int_semantics`].
    ///
    /// The model's answer is the declared return, which the harness checks against the computed
    /// one. For a pair the model rejects, the residue is declared instead: a program that fails
    /// to refuse the pair computes exactly that residue, so a missing check shows as an
    /// acceptance rather than as a return mismatch.
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
            .run(&input_block_in(&[lhs, rhs, &expectation.declared_return]));

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

    /// Confirms that this program's return columns are constrained, by forging an answer the
    /// model did not give into an honest witness and requiring a constraint to refuse it.
    ///
    /// # Errors
    ///
    /// Returns a description if the forged witness satisfies every constraint, or if the honest
    /// run was not accepted, which leaves nothing to forge and says nothing about the return.
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
        let honest = self.compiled.run(&input_block_in(&[lhs, rhs, &expected]));
        let Some(witness) = honest.witness() else {
            return Err(format!(
                "{:?}({lhs:?}, {rhs:?}): the honest run was not accepted, so there is no witness \
                 to forge: {honest:?}{}",
                self.kind,
                self.where_to_look()
            ));
        };

        // Flipping the low bit keeps the forged return a valid `bits`-wide integer, so only the
        // binding of the return can refuse it.
        let wrong = expected.xor(&IntBits::from_u128(expected.bits(), 1));
        let columns = self.compiled.return_columns();
        assert_eq!(columns.len(), 1, "a binary operation returns one field");
        let mut forged = witness.to_vec();
        forged[columns.start] = input_element(&wrong);

        if self
            .compiled
            .first_unsatisfied_constraint(&forged)
            .is_some()
        {
            return Ok(());
        }
        Err(format!(
            "{:?}({lhs:?}, {rhs:?}): the return is not constrained: a witness claiming {wrong:?} \
             instead of {expected:?} satisfies every constraint{}",
            self.kind,
            self.where_to_look()
        ))
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
    let options = WasmCompileOpts::fast(
        wasm_runtime::locate_or_build().unwrap_or_else(|error| panic!("{error}")),
    );

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
