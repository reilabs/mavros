use std::{
    fs,
    path::{Path, PathBuf},
};

use crate::{
    CompiledArtifacts, Project,
    abi_helpers::ordered_params_from_btreemap,
    api::interpreter::InputValueOrdered,
    compiled_artifacts::ArtifactError,
    compiler::{Field, r1cs_gen::R1CS},
    driver::{Driver, Error as DriverError},
    vm::interpreter,
};
use noirc_abi::input_parser::{Format, InputValue};

#[derive(Debug)]
pub enum ApiError {
    Project(crate::Error),
    Driver { inner: DriverError },
    Io(std::io::Error),
    UnsupportedInputExt(String),
    InputsParse(String),
    InputsEncode(String),
    Artifact(ArtifactError),
}

impl std::fmt::Display for ApiError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ApiError::Project(e) => write!(f, "{e}"),
            ApiError::Driver { inner } => write!(f, "driver error: {inner:?}"),
            ApiError::Io(e) => write!(f, "{e}"),
            ApiError::UnsupportedInputExt(ext) => {
                write!(f, "unsupported input file extension: {ext}")
            }
            ApiError::InputsParse(e) => write!(f, "failed to parse inputs: {e}"),
            ApiError::InputsEncode(e) => write!(f, "failed to encode inputs: {e}"),
            ApiError::Artifact(e) => write!(f, "artifact error: {e}"),
        }
    }
}

impl From<ArtifactError> for ApiError {
    fn from(e: ArtifactError) -> Self {
        ApiError::Artifact(e)
    }
}

impl std::error::Error for ApiError {}

pub fn compile_to_r1cs(root: PathBuf, draw_graphs: bool) -> Result<(Driver, R1CS), ApiError> {
    let project = Project::new(root).map_err(ApiError::Project)?;
    let mut driver = Driver::new(project, draw_graphs);
    driver.run_noir_compiler().unwrap();
    driver.make_struct_access_static().unwrap();
    driver.monomorphize().unwrap();
    driver.explictize_witness().unwrap();

    let r1cs = driver.generate_r1cs().unwrap();
    Ok((driver, r1cs))
}

pub fn compile_to_artifacts(
    root: PathBuf,
    draw_graphs: bool,
) -> Result<CompiledArtifacts, ApiError> {
    let (mut driver, r1cs) = compile_to_r1cs(root, draw_graphs)?;
    let witgen_binary = compile_witgen(&mut driver)?;
    let ad_binary = compile_ad(&driver)?;
    Ok(CompiledArtifacts::new(r1cs, witgen_binary, ad_binary))
}

pub fn save_artifacts<P: AsRef<Path>>(
    artifacts: &CompiledArtifacts,
    path: P,
) -> Result<(), ApiError> {
    artifacts.save_to_file(path)?;
    Ok(())
}

pub fn load_artifacts<P: AsRef<Path>>(path: P) -> Result<CompiledArtifacts, ApiError> {
    Ok(CompiledArtifacts::load_from_file(path)?)
}

pub fn read_prover_inputs(
    root: &Path,
    abi: &noirc_abi::Abi,
) -> Result<Vec<InputValueOrdered>, ApiError> {
    let file_path = root.join("Prover.toml");
    let ext = file_path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or_default();

    let Some(format) = Format::from_ext(ext) else {
        return Err(ApiError::UnsupportedInputExt(ext.to_string()));
    };

    let inputs_src = fs::read_to_string(&file_path).map_err(ApiError::Io)?;
    let inputs = format.parse(&inputs_src, abi).unwrap();
    let ordered_params = ordered_params_from_btreemap(abi, &inputs);

    Ok(ordered_params)
}

pub fn run_witgen_from_binary(
    binary: &mut [u64],
    r1cs: &R1CS,
    params: &[InputValueOrdered],
) -> interpreter::WitgenResult {
    interpreter::run(binary, r1cs.witness_layout, r1cs.constraints_layout, params)
}

/// Phase 1: execute the VM and produce the pre-commitment witness. Returns
/// intermediate state needed by [`run_witgen_phase2`].
pub fn run_witgen_phase1(
    binary: &mut [u64],
    r1cs: &R1CS,
    params: &[InputValueOrdered],
) -> interpreter::Phase1Result {
    interpreter::run_phase1(binary, r1cs.witness_layout, r1cs.constraints_layout, params)
}

/// Phase 2: given real Fiat-Shamir challenges, complete witness generation.
pub fn run_witgen_phase2(
    phase1: interpreter::Phase1Result,
    challenges: &[Field],
    r1cs: &R1CS,
) -> interpreter::WitgenResult {
    interpreter::run_phase2(
        phase1,
        challenges,
        r1cs.witness_layout,
        r1cs.constraints_layout,
    )
}

pub fn compile_witgen(driver: &mut Driver) -> Result<Vec<u64>, ApiError> {
    Ok(driver
        .compile_witgen()
        .map_err(|e| ApiError::Driver { inner: e })?)
}

pub fn compile_ad(driver: &Driver) -> Result<Vec<u64>, ApiError> {
    Ok(driver
        .compile_ad()
        .map_err(|e| ApiError::Driver { inner: e })?)
}

pub fn run_ad_from_binary(
    binary: &mut [u64],
    r1cs: &R1CS,
    coeffs: &[Field],
) -> (
    Vec<Field>,
    Vec<Field>,
    Vec<Field>,
    crate::vm::bytecode::AllocationInstrumenter,
) {
    interpreter::run_ad(binary, coeffs, r1cs.witness_layout, r1cs.constraints_layout)
}

pub fn random_ad_coeffs(r1cs: &R1CS) -> Vec<Field> {
    use ark_ff::UniformRand as _;
    let mut rng = rand::thread_rng();
    (0..r1cs.constraints.len())
        .map(|_| ark_bn254::Fr::rand(&mut rng))
        .collect()
}

pub fn check_witgen(r1cs: &R1CS, res: &interpreter::WitgenResult) -> bool {
    r1cs.check_witgen_output(
        &res.out_wit_pre_comm,
        &res.out_wit_post_comm,
        &res.out_a,
        &res.out_b,
        &res.out_c,
    )
}

pub fn check_ad(r1cs: &R1CS, coeffs: &[Field], a: &[Field], b: &[Field], c: &[Field]) -> bool {
    r1cs.check_ad_output(coeffs, a, b, c)
}

pub fn debug_output_dir(driver: &Driver) -> PathBuf {
    driver.get_debug_output_dir()
}
