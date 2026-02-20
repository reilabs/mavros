use std::{fs, path::PathBuf, process::ExitCode};

use clap::{Parser, Subcommand};
use mavros::api;
use mavros::compiler::Field;

type Error = Box<dyn std::error::Error>;
use tracing::{error, info, warn};
use tracing_forest::ForestLayer;
use tracing_subscriber::{EnvFilter, Registry, layer::SubscriberExt, util::SubscriberInitExt};

/// The default Noir project path for the CLI to extract from.
const DEFAULT_NOIR_PROJECT_PATH: &str = "./";

#[derive(Clone, Debug, Parser)]
#[command(name = "mavros")]
pub struct ProgramOptions {
    #[command(subcommand)]
    pub command: Option<Command>,

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

#[derive(Clone, Debug, Subcommand)]
pub enum Command {
    /// Compile a Noir project and output R1CS and binary artifacts.
    Compile {
        /// Path to the Noir project root.
        #[arg(default_value = DEFAULT_NOIR_PROJECT_PATH, value_parser = parse_path)]
        path: PathBuf,

        /// Output path for R1CS constraints (bincode).
        #[arg(long, default_value = "target/r1cs.bin")]
        r1cs_output: PathBuf,

        /// Output path for binaries and ABI (JSON).
        #[arg(long, default_value = "target/basic.json")]
        binary_output: PathBuf,

        /// Enable debugging mode which will generate graphs.
        #[arg(long, action = clap::ArgAction::SetTrue)]
        draw_graphs: bool,
    },
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

    let result = match &args.command {
        Some(Command::Compile {
            path,
            r1cs_output,
            binary_output,
            draw_graphs,
        }) => run_compile(path, r1cs_output, binary_output, *draw_graphs),
        None => run(&args),
    };

    result.unwrap_or_else(|err| {
        eprintln!("Error Encountered: {err:?}");
        ExitCode::FAILURE
    })
}

/// Compile phase: compile the Noir project and save R1CS and binary artifacts
/// to separate files.
pub fn run_compile(
    path: &PathBuf,
    r1cs_output: &PathBuf,
    binary_output: &PathBuf,
    draw_graphs: bool,
) -> Result<ExitCode, Error> {
    info!(message = %"Compiling Noir project", root = ?path, r1cs_output = ?r1cs_output, binary_output = ?binary_output);

    let (mut driver, r1cs) = api::compile_to_r1cs(path.clone(), draw_graphs)?;
    let witgen_binary = api::compile_witgen(&mut driver)?;
    let ad_binary = api::compile_ad(&driver)?;

    // Ensure output directories exist
    if let Some(parent) = r1cs_output.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }
    if let Some(parent) = binary_output.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }

    // Save R1CS as bincode
    let r1cs_bytes = bincode::serialize(&r1cs)?;
    fs::write(r1cs_output, r1cs_bytes)?;

    let basic = serde_json::json!({
        "witgen_binary": witgen_binary,
        "ad_binary": ad_binary,
        "abi": serde_json::to_value(driver.abi())?,
    });
    let basic_json = serde_json::to_string_pretty(&basic)?;
    fs::write(binary_output, basic_json)?;

    info!(
        message = %"Artifacts saved successfully",
        r1cs_output = ?r1cs_output,
        binary_output = ?binary_output,
        r1cs_constraints = r1cs.constraints.len(),
        witgen_binary_size = witgen_binary.len() * 8,
        ad_binary_size = ad_binary.len() * 8,
    );

    Ok(ExitCode::SUCCESS)
}

/// The main execution of the CLI utility (full pipeline). Should be called directly from the
/// `main` function of the application.
pub fn run(args: &ProgramOptions) -> Result<ExitCode, Error> {
    let (mut driver, r1cs) = api::compile_to_r1cs(args.root.clone(), args.draw_graphs)?;
    if args.pprint_r1cs {
        use std::io::Write;
        let mut r1cs_file =
            fs::File::create(api::debug_output_dir(&driver).join("r1cs.txt")).unwrap();
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

        driver
            .compile_llvm_targets(args.emit_llvm, wasm_config)
            .unwrap();
    }

    // Skip VM execution if requested
    if args.skip_vm {
        info!(message = %"Skipping VM execution as requested");
        return Ok(ExitCode::SUCCESS);
    }

    let params = api::read_prover_inputs(&args.root, driver.abi())?;
    let mut binary = api::compile_witgen(&mut driver)?;

    let witgen_result = api::run_witgen_from_binary(&mut binary, &r1cs, &params);

    let correct = api::check_witgen(&r1cs, &witgen_result);
    if !correct {
        error!(message = %"Witgen output is incorrect");
    } else {
        info!(message = %"Witgen output is correct");
    }

    let leftover_memory = witgen_result
        .instrumenter
        .plot(&api::debug_output_dir(&driver).join("witgen_vm_memory.png"));
    if leftover_memory > 0 {
        warn!(message = %"VM memory leak detected", leftover_memory);
    } else {
        info!(message = %"VM memory leak not detected");
    }

    fs::write(
        api::debug_output_dir(&driver).join("witness_pre_comm.txt"),
        witgen_result
            .out_wit_pre_comm
            .iter()
            .map(|w| w.to_string())
            .collect::<Vec<_>>()
            .join("\n"),
    )
    .unwrap();
    fs::write(
        api::debug_output_dir(&driver).join("a.txt"),
        witgen_result
            .out_a
            .iter()
            .map(|w| w.to_string())
            .collect::<Vec<_>>()
            .join("\n"),
    )
    .unwrap();
    fs::write(
        api::debug_output_dir(&driver).join("b.txt"),
        witgen_result
            .out_b
            .iter()
            .map(|w| w.to_string())
            .collect::<Vec<_>>()
            .join("\n"),
    )
    .unwrap();
    fs::write(
        api::debug_output_dir(&driver).join("c.txt"),
        witgen_result
            .out_c
            .iter()
            .map(|w| w.to_string())
            .collect::<Vec<_>>()
            .join("\n"),
    )
    .unwrap();

    let mut ad_binary = api::compile_ad(&driver)?;

    let ad_coeffs: Vec<Field> = api::random_ad_coeffs(&r1cs);

    let (ad_a, ad_b, ad_c, ad_instrumenter) =
        api::run_ad_from_binary(&mut ad_binary, &r1cs, &ad_coeffs);

    let leftover_memory =
        ad_instrumenter.plot(&api::debug_output_dir(&driver).join("ad_vm_memory.png"));
    if leftover_memory > 0 {
        warn!(message = %"AD VM memory leak detected", leftover_memory);
    } else {
        info!(message = %"AD VM memory leak not detected");
    }

    let correct = api::check_ad(&r1cs, &ad_coeffs, &ad_a, &ad_b, &ad_c);
    if !correct {
        error!(message = %"AD output is incorrect");
    } else {
        info!(message = %"AD output is correct");
    }

    fs::write(
        api::debug_output_dir(&driver).join("ad_a.txt"),
        ad_a.iter()
            .map(|w| w.to_string())
            .collect::<Vec<_>>()
            .join("\n"),
    )
    .unwrap();
    fs::write(
        api::debug_output_dir(&driver).join("ad_b.txt"),
        ad_b.iter()
            .map(|w| w.to_string())
            .collect::<Vec<_>>()
            .join("\n"),
    )
    .unwrap();
    fs::write(
        api::debug_output_dir(&driver).join("ad_c.txt"),
        ad_c.iter()
            .map(|w| w.to_string())
            .collect::<Vec<_>>()
            .join("\n"),
    )
    .unwrap();
    fs::write(
        api::debug_output_dir(&driver).join("ad_coeffs.txt"),
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
