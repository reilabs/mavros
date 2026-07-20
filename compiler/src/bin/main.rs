use std::{
    fs,
    path::{Path, PathBuf},
    process::ExitCode,
};

use clap::{Parser, Subcommand};
use mavros_compiler::Project;
use mavros_compiler::api;
use mavros_compiler::compiler::Field;
use mavros_compiler::compiler::codegen::CodeGenOptions;
use mavros_compiler::compiler::codegen::hlssa_to_r1cs::R1CS;
use mavros_compiler::compiler::codegen::llssa_to_llvm::WasmCompileOpts;
use mavros_compiler::driver::Driver;
use mavros_compiler::plotting;

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

    /// Print absolute paths in VM stack traces and WASM debug info instead of paths relative to the Noir package root.
    #[arg(long, action = clap::ArgAction::SetTrue)]
    pub absolute_paths: bool,

    /// Emit standalone debug-information sidecars for VM and WASM artifacts.
    #[arg(long, global = true, action = clap::ArgAction::SetTrue)]
    pub include_debug_info: bool,
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
        }) => run_compile(
            path,
            r1cs_output,
            binary_output,
            *draw_graphs,
            args.include_debug_info,
            args.absolute_paths,
        ),
        None => run(&args),
    };

    result.unwrap_or_else(|err| {
        eprintln!("Error Encountered: {err}");
        ExitCode::FAILURE
    })
}

pub fn compile_to_r1cs(root: PathBuf, draw_graphs: bool) -> Result<(Driver, R1CS), Error> {
    let project = Project::new(root)?;
    let mut driver = Driver::new(project, draw_graphs);
    driver.run_noir_compiler()?;
    driver.make_struct_access_static()?;
    driver.monomorphize()?;
    driver.spill_witness()?;

    let r1cs = driver.generate_r1cs()?;
    Ok((driver, r1cs))
}

fn debug_path_root(driver: &Driver, absolute_paths: bool) -> Option<&Path> {
    if absolute_paths {
        None
    } else {
        Some(driver.package_root())
    }
}

/// Compile phase: compile the Noir project and save R1CS and binary artifacts
/// to separate files.
pub fn run_compile(
    path: &PathBuf,
    r1cs_output: &PathBuf,
    binary_output: &PathBuf,
    draw_graphs: bool,
    include_debug_info: bool,
    absolute_paths: bool,
) -> Result<ExitCode, Error> {
    info!(message = %"Compiling Noir project", root = ?path, r1cs_output = ?r1cs_output, binary_output = ?binary_output);

    let (mut driver, r1cs) = compile_to_r1cs(path.clone(), draw_graphs)?;
    let artifact = api::compile_bytecode_artifact(
        &mut driver,
        CodeGenOptions {
            include_debug_info,
            ..CodeGenOptions::default()
        },
    )?;
    let binary = artifact.binary;

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
        "binary": binary,
        "abi": serde_json::to_value(driver.abi())?,
    });
    let basic_json = serde_json::to_string_pretty(&basic)?;
    fs::write(binary_output, basic_json)?;

    let debug_output = binary_output.with_extension("debug.json");
    if let Some(mut debug_info) = artifact.debug_info {
        if let Some(root) = debug_path_root(&driver, absolute_paths) {
            debug_info.relativize_source_paths(root);
        }
        fs::write(&debug_output, serde_json::to_string_pretty(&debug_info)?)?;
        info!(message = %"VM debug info generated", path = %debug_output.display());
    } else if debug_output.exists() {
        fs::remove_file(debug_output)?;
    }

    info!(
        message = %"Artifacts saved successfully",
        r1cs_output = ?r1cs_output,
        binary_output = ?binary_output,
        r1cs_constraints = r1cs.constraints.len(),
        binary_size = binary.len() * 8,
    );

    Ok(ExitCode::SUCCESS)
}

/// The main execution of the CLI utility (full pipeline). Should be called directly from the
/// `main` function of the application.
pub fn run(args: &ProgramOptions) -> Result<ExitCode, Error> {
    let (mut driver, r1cs) = compile_to_r1cs(args.root.clone(), args.draw_graphs)?;
    let codegen_options = CodeGenOptions {
        include_debug_info: args.include_debug_info,
        ..CodeGenOptions::default()
    };
    let source_path_root = debug_path_root(&driver, args.absolute_paths).map(Path::to_path_buf);
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
            let wasm_path = driver.get_debug_output_dir().join("program.wasm");
            info!(message = %"Generating WebAssembly", path = %wasm_path.display());
            let runtime_lib = mavros_compiler::wasm_runtime::locate_or_build();
            let mut opts = WasmCompileOpts::release(runtime_lib);
            if let Some(root) = &source_path_root {
                opts = opts.with_debug_path_root(root);
            }
            if args.include_debug_info {
                opts = opts.with_debug_info();
            }
            Some((wasm_path, opts))
        } else {
            None
        };

        if args.emit_llvm {
            info!(message = %"Generating LLVM IR");
        }

        driver
            .compile_llvm_targets(args.emit_llvm, &r1cs, wasm_config, codegen_options)
            .unwrap();
    }

    // Skip VM execution if requested
    if args.skip_vm {
        info!(message = %"Skipping VM execution as requested");
        return Ok(ExitCode::SUCCESS);
    }

    let params = api::read_prover_inputs(driver.package_root(), driver.abi())?;
    let artifact = api::compile_bytecode_artifact(&mut driver, codegen_options)?;
    let mut binary = artifact.binary;
    let vm_debug_info = artifact.debug_info;

    let witgen_result =
        api::run_witgen_from_binary(&mut binary, &r1cs, &params, vm_debug_info.clone()).map_err(
            |mut error| {
                if let Some(root) = &source_path_root {
                    error.relativize_source_paths(root);
                }
                error
            },
        )?;

    let correct = api::check_witgen(&r1cs, &witgen_result);
    if !correct {
        error!(message = %"Witgen output is incorrect");
    } else {
        info!(message = %"Witgen output is correct");
    }

    let leftover_memory = plotting::plot_memory_chart(
        &witgen_result.instrumenter,
        &api::debug_output_dir(&driver).join("witgen_vm_memory.png"),
    );
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

    let ad_coeffs: Vec<Field> = api::random_ad_coeffs(&r1cs);

    let (ad_a, ad_b, ad_c, ad_instrumenter) =
        api::run_ad_from_binary(&mut binary, &r1cs, &ad_coeffs, vm_debug_info)?;

    let leftover_memory = plotting::plot_memory_chart(
        &ad_instrumenter,
        &api::debug_output_dir(&driver).join("ad_vm_memory.png"),
    );
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn debug_info_is_excluded_by_default_with_path_and_metadata_opt_ins() {
        let relative = ProgramOptions::try_parse_from(["mavros"]).unwrap();
        assert!(!relative.absolute_paths);
        assert!(!relative.include_debug_info);

        let absolute =
            ProgramOptions::try_parse_from(["mavros", "--absolute-paths", "--include-debug-info"])
                .unwrap();
        assert!(absolute.absolute_paths);
        assert!(absolute.include_debug_info);

        let compile =
            ProgramOptions::try_parse_from(["mavros", "compile", ".", "--include-debug-info"])
                .unwrap();
        assert!(compile.include_debug_info);
    }
}
