use std::{fs, path::PathBuf, process::ExitCode};

use ark_ff::UniformRand as _;
use clap::Parser;
use mavros::{Error, Project, abi_helpers, compiler::Field, driver::Driver, vm::interpreter};
use tracing::{error, info, warn};
use tracing_forest::ForestLayer;
use tracing_subscriber::{EnvFilter, Registry, layer::SubscriberExt, util::SubscriberInitExt};
use noirc_abi::input_parser::Format;

/// The default Noir project path for the CLI to extract from.
const DEFAULT_NOIR_PROJECT_PATH: &str = "./";

#[derive(Clone, Debug, Parser)]
pub struct ProgramOptions {
    /// The root of the Noir project to extract.
    #[arg(long, value_name = "PATH", default_value = DEFAULT_NOIR_PROJECT_PATH, value_parser = parse_path)]
    pub root: PathBuf,

    /// Enable debugging mode which will generate graphs
    #[arg(long, action = clap::ArgAction::SetTrue)]
    pub draw_graphs: bool,

    #[arg(long, action = clap::ArgAction::SetTrue)]
    pub pprint_r1cs: bool,

    #[arg(long, action = clap::ArgAction::SetTrue)]
    pub emit_llvm: bool,

    #[arg(long, action = clap::ArgAction::SetTrue)]
    pub emit_wasm: bool,

    #[arg(long, action = clap::ArgAction::SetTrue)]
    pub skip_vm: bool,
}

/// The main function for the CLI utility, responsible for parsing program
/// options and handing them off to the actual execution of the tool.
fn main() -> ExitCode {
    // Parse args and hand-off immediately.
    let args = ProgramOptions::parse();

    Registry::default()
        .with(ForestLayer::default())
        .with(EnvFilter::from_default_env())
        .init();

    run(&args).unwrap_or_else(|err| {
        eprintln!("Error Encountered: {err:?}");
        ExitCode::FAILURE
    })
}

/// The main execution of the CLI utility. Should be called directly from the
/// `main` function of the application.
///
/// # Errors
///
/// - [`Error`] if the extraction process fails for any reason.
pub fn run(args: &ProgramOptions) -> Result<ExitCode, Error> {
    let project = Project::new(args.root.clone())?;

    let mut driver = Driver::new(project, args.draw_graphs);

    driver.run_noir_compiler().unwrap();
    driver.make_struct_access_static().unwrap();
    driver.monomorphize().unwrap();
    driver.explictize_witness().unwrap();
    let r1cs = driver.generate_r1cs().unwrap();

    if args.pprint_r1cs {
        use std::io::Write;
        let mut r1cs_file =
            fs::File::create(driver.get_debug_output_dir().join("r1cs.txt")).unwrap();
        for r1c in r1cs.constraints.iter() {
            writeln!(r1cs_file, "{}", r1c).unwrap();
        }
    }

    if args.emit_llvm || args.emit_wasm {
        let wasm_config = if args.emit_wasm {
            let wasm_path = driver.get_debug_output_dir().join("witgen.wasm");
            info!(message = %"Generating WebAssembly", path = %wasm_path.display());
            Some((wasm_path, &r1cs))
        } else {
            None
        };

        if args.emit_llvm {
            info!(message = %"Generating LLVM IR");
        }

        driver.compile_llvm_targets(args.emit_llvm, wasm_config).unwrap();
    }

    // Skip VM execution if requested
    if args.skip_vm {
        info!(message = %"Skipping VM execution as requested");
        return Ok(ExitCode::SUCCESS);
    }

    let file_path = args.root.join("Prover.toml");
    let ext = file_path.extension().and_then(|e| e.to_str()).unwrap();
    let format = Format::from_ext(ext).unwrap();
    let inputs = std::fs::read_to_string(file_path).unwrap();
    let inputs = format.parse(&inputs, driver.abi()).unwrap();
    let ordered_params = abi_helpers::ordered_params_from_btreemap(driver.abi(), &inputs);

    let mut binary = driver.compile_witgen().unwrap();

    let witgen_result = interpreter::run(
        &mut binary,
        r1cs.witness_layout,
        r1cs.constraints_layout,
        &ordered_params,
    );

    let correct = r1cs.check_witgen_output(
        &witgen_result.out_wit_pre_comm,
        &witgen_result.out_wit_post_comm,
        &witgen_result.out_a,
        &witgen_result.out_b,
        &witgen_result.out_c,
    );
    if !correct {
        error!(message = %"Witgen output is incorrect");
    } else {
        info!(message = %"Witgen output is correct");
    }

    let leftover_memory = witgen_result
        .instrumenter
        .plot(&driver.get_debug_output_dir().join("witgen_vm_memory.png"));
    if leftover_memory > 0 {
        warn!(message = %"VM memory leak detected", leftover_memory);
    } else {
        info!(message = %"VM memory leak not detected");
    }

    fs::write(
        driver.get_debug_output_dir().join("witness_pre_comm.txt"),
        witgen_result
            .out_wit_pre_comm
            .iter()
            .map(|w| w.to_string())
            .collect::<Vec<_>>()
            .join("\n"),
    )
    .unwrap();
    fs::write(
        driver.get_debug_output_dir().join("a.txt"),
        witgen_result
            .out_a
            .iter()
            .map(|w| w.to_string())
            .collect::<Vec<_>>()
            .join("\n"),
    )
    .unwrap();
    fs::write(
        driver.get_debug_output_dir().join("b.txt"),
        witgen_result
            .out_b
            .iter()
            .map(|w| w.to_string())
            .collect::<Vec<_>>()
            .join("\n"),
    )
    .unwrap();
    fs::write(
        driver.get_debug_output_dir().join("c.txt"),
        witgen_result
            .out_c
            .iter()
            .map(|w| w.to_string())
            .collect::<Vec<_>>()
            .join("\n"),
    )
    .unwrap();

    let mut ad_binary = driver.compile_ad().unwrap();

    let mut ad_coeffs: Vec<Field> = vec![];
    for _ in 0..r1cs.constraints.len() {
        ad_coeffs.push(ark_bn254::Fr::rand(&mut rand::thread_rng()));
    }

    let (ad_a, ad_b, ad_c, ad_instrumenter) = interpreter::run_ad(
        &mut ad_binary,
        &ad_coeffs,
        r1cs.witness_layout,
        r1cs.constraints_layout,
    );

    let leftover_memory =
        ad_instrumenter.plot(&driver.get_debug_output_dir().join("ad_vm_memory.png"));
    if leftover_memory > 0 {
        warn!(message = %"AD VM memory leak detected", leftover_memory);
    } else {
        info!(message = %"AD VM memory leak not detected");
    }

    let correct = r1cs.check_ad_output(&ad_coeffs, &ad_a, &ad_b, &ad_c);
    if !correct {
        error!(message = %"AD output is incorrect");
    } else {
        info!(message = %"AD output is correct");
    }

    fs::write(
        driver.get_debug_output_dir().join("ad_a.txt"),
        ad_a.iter()
            .map(|w| w.to_string())
            .collect::<Vec<_>>()
            .join("\n"),
    )
    .unwrap();
    fs::write(
        driver.get_debug_output_dir().join("ad_b.txt"),
        ad_b.iter()
            .map(|w| w.to_string())
            .collect::<Vec<_>>()
            .join("\n"),
    )
    .unwrap();
    fs::write(
        driver.get_debug_output_dir().join("ad_c.txt"),
        ad_c.iter()
            .map(|w| w.to_string())
            .collect::<Vec<_>>()
            .join("\n"),
    )
    .unwrap();
    fs::write(
        driver.get_debug_output_dir().join("ad_coeffs.txt"),
        ad_coeffs
            .iter()
            .map(|w| w.to_string())
            .collect::<Vec<_>>()
            .join("\n"),
    )
    .unwrap();

    Ok(ExitCode::SUCCESS)
}

// Copied from: https://github.com/noir-lang/noir/blob/5071093f9b51e111a49a5f78d827774ef8e80c74/tooling/nargo_cli/src/cli/mod.rs#L301
/// Parses a path and turns it into an absolute one by joining to the current
/// directory.
fn parse_path(path: &str) -> Result<PathBuf, String> {
    use fm::NormalizePath;
    let mut path: PathBuf = path
        .parse()
        .map_err(|e| format!("failed to parse path: {e}"))?;
    if !path.is_absolute() {
        path = std::env::current_dir().unwrap().join(path).normalize();
    }
    Ok(path)
}

