use std::{
    collections::{BTreeMap, btree_map::Entry},
    fs,
    path::{Path, PathBuf},
    process::ExitCode,
};

use clap::{
    Parser, Subcommand,
    builder::{PossibleValuesParser, TypedValueParser},
};
use mavros_artifacts::{Field as Bn254, FieldId, Goldilocks, MavrosField};
use mavros_compiler::compiler::codegen::r1cs_compact;
use mavros_compiler::{
    Project,
    api::{self, R1csOptions},
    compiler::codegen::{CodeGenOptions, llssa_to_llvm::WasmCompileOpts},
    driver::{DEFAULT_LOGUP_SOUNDNESS_BITS, Driver},
    panic_handler, plotting,
};

use tracing::{error, info, warn};
use tracing_forest::{ForestLayer, printer::PrettyPrinter};
use tracing_subscriber::{
    EnvFilter, Registry, fmt::writer::BoxMakeWriter, layer::SubscriberExt, util::SubscriberInitExt,
};

type Error = Box<dyn std::error::Error>;

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

    /// Compute (rAw, rBw, rCw) for supplied vectors, without witness generation.
    #[arg(long, requires = "r", requires = "w", conflicts_with_all = ["skip_vm", "profile"])]
    pub verify: bool,

    /// JSON array of decimal field strings, one per constraint, for --verify.
    #[arg(long, value_name = "PATH", requires = "verify")]
    pub r: Option<PathBuf>,

    /// JSON array of decimal field strings covering the full witness layout, for --verify.
    #[arg(long, value_name = "PATH", requires = "verify")]
    pub w: Option<PathBuf>,

    /// Print absolute paths in VM stack traces and WASM debug info instead of paths relative to the
    /// Noir package root.
    #[arg(long, action = clap::ArgAction::SetTrue)]
    pub absolute_paths: bool,

    /// Emit standalone debug-information sidecars for VM and WASM artifacts.
    #[arg(long, global = true, action = clap::ArgAction::SetTrue)]
    pub include_debug_info: bool,

    /// Generate per-function constraint, witness, witgen, and AD FlameGraphs.
    #[arg(long, conflicts_with = "skip_vm", action = clap::ArgAction::SetTrue)]
    pub profile: bool,

    /// Target LogUp lookup-argument soundness, in bits of security. The compiler repeats the
    /// argument at independent challenges as many times as needed (<= 16) and reports the count.
    #[arg(long, value_name = "BITS", global = true, default_value_t = DEFAULT_LOGUP_SOUNDNESS_BITS)]
    pub logup_soundness: u32,

    /// The prime field the program is compiled and executed over.
    #[arg(long, global = true, default_value_t = FieldId::Bn254, value_parser = field_parser())]
    pub field: FieldId,

    /// Run the report-only post-seal compaction analysis and log how many R1CS rows/columns a
    /// compaction pass could remove.
    #[arg(long, action = clap::ArgAction::SetTrue)]
    pub analyze_r1cs: bool,
}

/// Parses `--field`, listing every supported field's name as a possible value.
fn field_parser() -> impl TypedValueParser<Value = FieldId> {
    PossibleValuesParser::new(FieldId::ALL.map(FieldId::name))
        .map(|name| name.parse().expect("every possible value names a field"))
}

/// The settings `mavros compile` applies to each package.
#[derive(Clone, Copy, Debug)]
pub struct CompileSettings {
    pub r1cs: R1csOptions,
    pub include_debug_info: bool,
    pub absolute_paths: bool,
    pub field: FieldId,
}

impl Default for CompileSettings {
    fn default() -> Self {
        CompileSettings {
            r1cs: R1csOptions::default(),
            include_debug_info: false,
            absolute_paths: false,
            field: FieldId::Bn254,
        }
    }
}

#[derive(Clone, Debug, Subcommand)]
pub enum Command {
    /// Compile a Noir project into one Mavros artifact.
    Compile {
        /// Path to the Noir project root.
        #[arg(default_value = DEFAULT_NOIR_PROJECT_PATH, value_parser = parse_path)]
        path: PathBuf,

        /// Output path for the Mavros artifact.
        #[arg(long, default_value = "target/program.mavros")]
        output: PathBuf,

        /// Optionally write a separate verifier binary and layouts (JSON).
        #[arg(long)]
        verifier_output: Option<PathBuf>,

        /// Enable debugging mode which will generate graphs.
        #[arg(long, action = clap::ArgAction::SetTrue)]
        draw_graphs: bool,
    },
}

/// The main function for the CLI utility, responsible for parsing program
/// options and handing them off to the actual execution of the tool.
fn main() -> ExitCode {
    panic_handler::install();

    // Parse args and hand-off immediately.
    let args = ProgramOptions::parse();

    // Keep verifier stdout machine-readable even when tracing is enabled.
    let log_writer = if args.verify {
        BoxMakeWriter::new(std::io::stderr)
    } else {
        BoxMakeWriter::new(std::io::stdout)
    };
    Registry::default()
        .with(ForestLayer::from(PrettyPrinter::new().writer(log_writer)))
        .with(EnvFilter::from_default_env())
        .init();

    let result = match &args.command {
        Some(Command::Compile { .. }) if args.verify => Err(std::io::Error::other(
            "--verify executes the verifier and cannot be combined with the compile subcommand",
        )
        .into()),
        Some(Command::Compile { .. }) if args.profile => Err(std::io::Error::other(
            "--profile runs the full pipeline and cannot be combined with the compile subcommand",
        )
        .into()),
        Some(Command::Compile {
            path,
            output,
            verifier_output,
            draw_graphs,
        }) => run_compile(
            path,
            output,
            verifier_output.as_ref(),
            CompileSettings {
                r1cs: R1csOptions {
                    draw_graphs: *draw_graphs,
                    logup_soundness: args.logup_soundness,
                    profile: false,
                },
                include_debug_info: args.include_debug_info,
                absolute_paths: args.absolute_paths,
                field: args.field,
            },
        ),
        None => run(&args),
    };

    result.unwrap_or_else(|err| {
        eprintln!("Error Encountered: {err}");
        ExitCode::FAILURE
    })
}

fn debug_path_root(driver: &Driver, absolute_paths: bool) -> Option<&Path> {
    if absolute_paths {
        None
    } else {
        Some(driver.package_root())
    }
}

/// Compile phase: compile the Noir project and write its artifact.
pub fn run_compile(
    path: &PathBuf,
    output: &PathBuf,
    verifier_output: Option<&PathBuf>,
    settings: CompileSettings,
) -> Result<ExitCode, Error> {
    let requested_root = fs::canonicalize(path)?;
    let roots = Project::binary_package_roots(&requested_root)?;
    let multiple = roots.len() > 1;
    if multiple
        && std::iter::once(output)
            .chain(verifier_output)
            .any(|output| {
                output.components().any(|component| {
                    !matches!(
                        component,
                        std::path::Component::Normal(_) | std::path::Component::CurDir
                    )
                })
            })
    {
        return Err(std::io::Error::other(
            "For multi-package compilation, output paths must be relative paths within each member directory",
        ).into());
    }
    run_packages(&roots, |root| {
        let rebase = |path: &PathBuf| {
            if root != &requested_root {
                root.join(path)
            } else {
                path.clone()
            }
        };
        let verifier_output = verifier_output.map(rebase);
        run_compile_package(root, &rebase(output), verifier_output.as_ref(), settings)
    })
}

/// Compile one package over the field `settings` names.
fn run_compile_package(
    path: &PathBuf,
    output: &PathBuf,
    verifier_output: Option<&PathBuf>,
    settings: CompileSettings,
) -> Result<ExitCode, Error> {
    let compile = match settings.field {
        FieldId::Bn254 => compile_package::<Bn254>,
        FieldId::Goldilocks => compile_package::<Goldilocks>,
    };

    compile(path, output, verifier_output, settings)
}

/// Compile one package over `F`, which [`run_compile_package`] picks from `settings.field`.
fn compile_package<F: MavrosField>(
    path: &PathBuf,
    output: &PathBuf,
    verifier_output: Option<&PathBuf>,
    settings: CompileSettings,
) -> Result<ExitCode, Error> {
    let CompileSettings {
        r1cs: r1cs_options,
        include_debug_info,
        absolute_paths,
        field: _,
    } = settings;
    check_compile_outputs(output, verifier_output.map(|path| path.as_path()))?;
    info!(message = %"Compiling Noir project", root = ?path, output = ?output);

    let (mut driver, r1cs) = api::compile_to_r1cs_with::<F>(path.clone(), r1cs_options)?;
    let bytecode = api::compile_bytecode_artifact(
        &mut driver,
        CodeGenOptions {
            include_debug_info,
            ..CodeGenOptions::default()
        },
    )?;

    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent)?;
    }

    let artifact = driver.artifact(r1cs, bytecode.binary)?;
    let artifact_bytes = {
        let bytes = artifact.to_bytes()?;
        fs::write(output, &bytes)?;
        bytes.len()
    };

    write_vm_debug_info(output, bytecode.debug_info, &driver, absolute_paths)?;

    if let Some(output) = verifier_output {
        let verifier = api::compile_verifier_bytecode_artifact(
            &mut driver,
            CodeGenOptions {
                include_debug_info,
                ..CodeGenOptions::default()
            },
        )?;
        if let Some(parent) = output.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(
            output,
            serde_json::to_string_pretty(&serde_json::json!({
                "binary": verifier.binary,
                "witness_layout": artifact.r1cs().witness_layout,
                "constraints_layout": artifact.r1cs().constraints_layout,
                "circuit_digest": artifact.circuit_digest_hex(),
            }))?,
        )?;
        write_vm_debug_info(output, verifier.debug_info, &driver, absolute_paths)?;
    }

    info!(
        message = %"Artifact saved successfully",
        output = ?output,
        artifact_bytes,
        r1cs_constraints = artifact.r1cs().constraints.len(),
        binary_size = artifact.bytecode().len() * 8,
    );

    Ok(ExitCode::SUCCESS)
}

/// Reserve debug sidecars even without debug info: stale sidecars are removed on that path.
fn check_compile_outputs(output: &Path, verifier: Option<&Path>) -> Result<(), Error> {
    let mut outputs = vec![output.to_path_buf(), output.with_extension("debug.json")];
    if let Some(verifier) = verifier {
        outputs.extend([
            verifier.to_path_buf(),
            verifier.with_extension("debug.json"),
        ]);
    }
    let mut seen = std::collections::BTreeSet::new();
    let mut probes = BTreeMap::new();
    for output in outputs {
        if !seen.insert(output_identity(&output, &mut probes)?) {
            return Err(format!("compile output paths overlap: {}", output.display()).into());
        }
    }
    Ok(())
}

fn output_identity(
    path: &Path,
    probes: &mut BTreeMap<PathBuf, tempfile::TempDir>,
) -> std::io::Result<PathBuf> {
    let mut normalized = PathBuf::new();
    for component in std::path::absolute(path)?.components() {
        if component == std::path::Component::ParentDir {
            normalized.pop();
        } else {
            normalized.push(component);
        }
        // Resolve existing parents before interpreting `..`; output files need not exist yet.
        match fs::canonicalize(&normalized) {
            Ok(path) => normalized = path,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => return Err(error),
        }
    }
    if normalized.try_exists()? {
        return Ok(normalized);
    }
    let parent = normalized
        .ancestors()
        .skip(1)
        .find(|parent| parent.is_dir())
        .expect("an absolute output path has an existing root");
    let probe = match probes.entry(parent.to_path_buf()) {
        Entry::Occupied(entry) => entry.into_mut(),
        Entry::Vacant(entry) => entry.insert(tempfile::tempdir_in(parent)?),
    };
    // Resolve missing names on the destination filesystem, preserving its case folding and
    // Unicode normalization rules. Shared probes give aliases the same canonical spelling.
    let shadow = probe.path().join(normalized.strip_prefix(parent).unwrap());
    fs::create_dir_all(&shadow)?;
    let resolved = fs::canonicalize(shadow)?;
    Ok(parent.join(resolved.strip_prefix(probe.path()).unwrap()))
}

fn write_vm_debug_info(
    output: &Path,
    debug_info: Option<mavros_vm::bytecode::DebugInfo>,
    driver: &Driver,
    absolute_paths: bool,
) -> Result<(), Error> {
    let debug_output = output.with_extension("debug.json");
    if let Some(mut debug_info) = debug_info {
        if let Some(root) = debug_path_root(driver, absolute_paths) {
            debug_info.relativize_source_paths(root);
        }
        fs::write(&debug_output, serde_json::to_string_pretty(&debug_info)?)?;
        info!(message = %"VM debug info generated", path = %debug_output.display());
    } else if debug_output.exists() {
        fs::remove_file(debug_output)?;
    }
    Ok(())
}

/// The main execution of the CLI utility (full pipeline). Should be called directly from the
/// `main` function of the application.
pub fn run(args: &ProgramOptions) -> Result<ExitCode, Error> {
    let roots = Project::binary_package_roots(&args.root)?;
    if args.verify && roots.len() > 1 {
        return Err("--verify requires a single binary package; use --root to select a member or set default-member in Nargo.toml".into());
    }
    let runtime_lib = if args.emit_wasm {
        Some(mavros_compiler::wasm_runtime::locate_or_build()?)
    } else {
        None
    };
    run_packages(&roots, |root| {
        let mut options = args.clone();
        options.root = root.clone();
        run_package(&options, runtime_lib.as_deref())
    })
}

/// Run each selected binary with its own inputs and artifacts.
/// Report package errors and return a combined exit status.
fn run_packages(
    roots: &[PathBuf],
    mut run: impl FnMut(&PathBuf) -> Result<ExitCode, Error>,
) -> Result<ExitCode, Error> {
    let mut success = true;
    for root in roots {
        match run(root) {
            Ok(status) => success &= status == ExitCode::SUCCESS,
            Err(error) => {
                eprintln!("Package {}: {error}", root.display());
                success = false;
            }
        }
    }
    Ok(if success {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    })
}

/// Run the package at `args.root` over the field `args.field` names.
fn run_package(args: &ProgramOptions, runtime_lib: Option<&Path>) -> Result<ExitCode, Error> {
    match args.field {
        FieldId::Bn254 => run_package_over::<Bn254>(args, runtime_lib),
        FieldId::Goldilocks => run_package_over::<Goldilocks>(args, runtime_lib),
    }
}

/// Run the package at `args.root` over `F`, which [`run_package`] picks from `args.field`.
fn run_package_over<F: MavrosField>(
    args: &ProgramOptions,
    runtime_lib: Option<&Path>,
) -> Result<ExitCode, Error> {
    let (mut driver, r1cs) = api::compile_to_r1cs_with::<F>(
        args.root.clone(),
        R1csOptions {
            draw_graphs: args.draw_graphs,
            logup_soundness: args.logup_soundness,
            profile: args.profile,
        },
    )?;

    let source_path_root = debug_path_root(&driver, args.absolute_paths).map(Path::to_path_buf);
    let codegen_options = CodeGenOptions {
        include_debug_info: args.include_debug_info || args.profile,
        ..CodeGenOptions::default()
    };

    if args.analyze_r1cs {
        let stats = r1cs_compact::analyze(&r1cs, driver.protected_r1cs_cols());
        info!(
            message = %"R1CS compaction opportunity",
            total_rows = r1cs.constraints.len(),
            algebraic_rows = stats.algebraic_rows,
            removable_rows = stats.removable_rows,
            algebraic_cols = stats.algebraic_cols,
            removable_cols = stats.removable_cols,
            duplicate_rows = stats.duplicate_rows,
            tautology_rows = stats.tautology_rows,
            pinned_cols = stats.pinned_cols,
            merged_cols = stats.merged_cols,
            eliminated_cols = stats.eliminated_cols,
            linear_elim_rows = stats.linear_elim_rows,
            rounds = stats.rounds,
        );
    }
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
            let runtime_lib =
                runtime_lib.expect("run() resolves the runtime when --emit-wasm is set");
            let mut opts = WasmCompileOpts::release(runtime_lib.to_path_buf());
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

        driver.compile_llvm_targets(args.emit_llvm, &r1cs, wasm_config, codegen_options)?;
    }

    // Skip VM execution if requested
    if args.skip_vm {
        info!(message = %"Skipping VM execution as requested");
        return Ok(ExitCode::SUCCESS);
    }

    let artifact = if args.verify {
        api::compile_verifier_bytecode_artifact(&mut driver, codegen_options)?
    } else {
        api::compile_bytecode_artifact(&mut driver, codegen_options)?
    };
    let mut binary = artifact.binary;
    let vm_debug_info = artifact.debug_info;

    if args.verify {
        let r = read_verifier_vector::<F>(args.r.as_deref().expect("--verify requires --r"))?;
        let w = read_verifier_vector::<F>(args.w.as_deref().expect("--verify requires --w"))?;
        let result = api::run_verifier_from_binary(
            &binary,
            &r,
            &w,
            r1cs.witness_layout,
            r1cs.constraints_layout,
            vm_debug_info,
        )
        .map_err(|mut error| {
            if let Some(root) = &source_path_root {
                error.relativize_source_paths(root);
            }
            error
        })?;
        println!(
            "{}",
            serde_json::json!({
                "r_aw": result.r_aw.to_string(),
                "r_bw": result.r_bw.to_string(),
                "r_cw": result.r_cw.to_string(),
            })
        );
        return Ok(ExitCode::SUCCESS);
    }

    let (params, declared_return) = api::read_prover_inputs(driver.package_root(), driver.abi())?;

    let witgen_execution = if args.profile {
        api::run_witgen_profiled_from_binary(&mut binary, &r1cs, &params, vm_debug_info.clone())
    } else {
        api::run_witgen_from_binary(&mut binary, &r1cs, &params, vm_debug_info.clone())
            .map(|result| (result, mavros_artifacts::FlamegraphProfile::default()))
    };
    let (witgen_result, witgen_profile) = witgen_execution.map_err(|mut error| {
        if let Some(root) = &source_path_root {
            error.relativize_source_paths(root);
        }
        error
    })?;

    // Every consistency check below accumulates into this; one failure fails the run.
    let mut checks_passed = true;

    let correct = api::check_witgen(&r1cs, &witgen_result);
    checks_passed &= correct;
    if !correct {
        error!(message = %"Witgen output is incorrect");
    } else {
        info!(message = %"Witgen output is correct");
    }

    match driver.witness_manifest(&r1cs) {
        Err(error) => {
            error!(message = %"Could not build the witness manifest", reason = %error);
            checks_passed = false;
        }
        Ok(manifest) => {
            if let Err(reason) = mavros_compiler::abi_helpers::check_witness_manifest(
                &manifest,
                &params,
                declared_return.as_ref(),
                &witgen_result.out_wit_pre_comm,
            ) {
                error!(message = %"Witness manifest disagrees with the generated witness", reason = %reason);
                checks_passed = false;
            }
        }
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

    let ad_coeffs: Vec<F> = api::random_ad_coeffs(&r1cs);

    let ad_execution = if args.profile {
        api::run_ad_profiled_from_binary(&mut binary, &r1cs, &ad_coeffs, vm_debug_info)
    } else {
        api::run_ad_from_binary(&mut binary, &r1cs, &ad_coeffs, vm_debug_info).map(
            |(a, b, c, instrumenter)| {
                (
                    a,
                    b,
                    c,
                    instrumenter,
                    mavros_artifacts::FlamegraphProfile::default(),
                )
            },
        )
    };
    let (ad_a, ad_b, ad_c, ad_instrumenter, ad_profile) = ad_execution?;

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
    checks_passed &= correct;
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

    if args.profile {
        let r1cs_profile = driver
            .r1cs_profile()
            .expect("--profile enables R1CS size profiling");
        let output_dir = api::debug_output_dir(&driver).join("flamegraphs");
        for (profile, name, title, count_name) in [
            (
                &r1cs_profile.constraints,
                "constraint_size",
                "R1CS constraints per function",
                "constraints",
            ),
            (
                &r1cs_profile.witnesses,
                "witness_size",
                "R1CS witnesses per function",
                "witnesses",
            ),
            (
                &witgen_profile,
                "witgen_time",
                "Witness generation simulated time",
                "instructions",
            ),
            (
                &ad_profile,
                "ad_time",
                "Automatic differentiation simulated time",
                "instructions",
            ),
        ] {
            let (folded_path, svg_path) =
                mavros_compiler::flamegraph::render(profile, &output_dir, name, title, count_name)?;
            if let Some(svg_path) = svg_path {
                info!(message = %"FlameGraph generated", path = %svg_path.display());
            } else {
                warn!(
                    message = %"flamegraph.pl was not found; folded profile generated but SVG rendering skipped",
                    path = %folded_path.display()
                );
            }
        }
        for (profile, name) in [(&witgen_profile, "witgen_time"), (&ad_profile, "ad_time")] {
            let path = mavros_compiler::flamegraph::write_cpuprofile(profile, &output_dir, name)?;
            info!(message = %"Chrome DevTools CPU profile generated", path = %path.display());
        }
    }

    if checks_passed {
        Ok(ExitCode::SUCCESS)
    } else {
        Ok(ExitCode::FAILURE)
    }
}

/// Read verifier inputs over `F` without losing precision to JSON's numeric representation.
fn read_verifier_vector<F: MavrosField>(path: &Path) -> Result<Vec<F>, Error> {
    let values: Vec<String> = serde_json::from_str(&fs::read_to_string(path)?).map_err(|err| {
        format!(
            "{}: expected a JSON array of decimal field strings: {err}",
            path.display()
        )
    })?;
    values
        .into_iter()
        .enumerate()
        .map(|(index, value)| {
            if value.is_empty()
                || !value.bytes().all(|byte| byte.is_ascii_digit())
                || (value.len() > 1 && value.starts_with('0'))
            {
                return Err(format!(
                    "{}: expected a canonical unsigned decimal field element at index {index}",
                    path.display()
                )
                .into());
            }
            let integer = num_bigint::BigUint::parse_bytes(value.as_bytes(), 10)
                .expect("nonempty decimal digits were validated above");
            F::try_from_canonical_limbs(&integer.to_u64_digits()).ok_or_else(|| {
                format!(
                    "{}: field element at index {index} must be less than the modulus",
                    path.display()
                )
                .into()
            })
        })
        .collect()
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
    fn compile_output_collisions_are_rejected_before_writing() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        for (output, verifier) in [
            ("p.mavros", "p.mavros"),
            ("p.mavros", "p.json"),
            ("p.mavros", "p.debug.json"),
            ("p.debug.json", "p.json"),
            ("p.mavros", "./p.mavros"),
            ("p.mavros", "unused/../p.mavros"),
        ] {
            let output = root.join(output);
            let verifier = root.join(verifier);
            for include_debug_info in [false, true] {
                fs::write(&output, "keep artifact").unwrap();
                let error = run_compile_package(
                    &root.to_path_buf(),
                    &output,
                    Some(&verifier),
                    CompileSettings {
                        include_debug_info,
                        ..CompileSettings::default()
                    },
                )
                .unwrap_err();
                assert!(
                    error.to_string().contains("output paths overlap"),
                    "{error}"
                );
                assert_eq!(fs::read_to_string(&output).unwrap(), "keep artifact");
            }
        }
        check_compile_outputs(&root.join("p.mavros"), Some(&root.join("v.json"))).unwrap();
        #[cfg(unix)]
        {
            fs::create_dir_all(root.join("nested/sub")).unwrap();
            std::os::unix::fs::symlink(root.join("nested/sub"), root.join("alias")).unwrap();
            assert!(
                check_compile_outputs(
                    &root.join("nested/p.mavros"),
                    Some(&root.join("alias/../p.json"))
                )
                .is_err()
            );
        }
    }

    #[test]
    fn compile_output_collisions_follow_filesystem_case_sensitivity() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        let sentinel = root.join("sentinel");
        fs::write(&sentinel, "keep").unwrap();
        fs::create_dir(root.join("nested")).unwrap();
        let case_insensitive = root.join("SENTINEL").exists();
        for (output, verifier) in [
            ("sentinel", "SENTINEL"),
            ("p.mavros", "P.mavros"),
            ("p.mavros", "P.json"),
            ("new/p.mavros", "NEW/P.json"),
            ("nested/p.mavros", "NESTED/P.json"),
            ("é.mavros", "É.json"),
        ] {
            let result = check_compile_outputs(&root.join(output), Some(&root.join(verifier)));
            assert_eq!(result.is_err(), case_insensitive, "{output}, {verifier}");
            if let Err(error) = result {
                assert!(
                    error.to_string().contains("output paths overlap"),
                    "{error}"
                );
            }
            assert_eq!(fs::read_to_string(&sentinel).unwrap(), "keep");
            // No output directories or temporary probes survive the check.
            assert_eq!(fs::read_dir(root).unwrap().count(), 2);
            assert_eq!(fs::read_dir(root.join("nested")).unwrap().count(), 0);
        }
    }

    #[test]
    fn verifier_cli_requires_vectors_and_rejects_conflicting_modes() {
        for args in [
            vec!["mavros", "--verify"],
            vec!["mavros", "--verify", "--r", "r.json"],
            vec!["mavros", "--verify", "--w", "w.json"],
            vec!["mavros", "--r", "r.json", "--w", "w.json"],
            vec![
                "mavros",
                "--verify",
                "--r",
                "r.json",
                "--w",
                "w.json",
                "--skip-vm",
            ],
            vec![
                "mavros",
                "--verify",
                "--r",
                "r.json",
                "--w",
                "w.json",
                "--profile",
            ],
        ] {
            assert!(ProgramOptions::try_parse_from(args).is_err());
        }
    }

    #[test]
    fn verifier_vectors_use_decimal_strings() {
        use ark_ff::PrimeField;

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("vector.json");
        fs::write(&path, r#"["0", "1", "12345678901234567890"]"#).unwrap();
        assert_eq!(
            read_verifier_vector::<Bn254>(&path).unwrap(),
            vec![
                Bn254::from(0u64),
                Bn254::from(1u64),
                Bn254::from(12345678901234567890u64),
            ]
        );
        for input in [
            r#"[1]"#,
            r#"["nope"]"#,
            r#"[""]"#,
            r#"["-1"]"#,
            r#"["00"]"#,
            r#"["01"]"#,
            r#"["+1"]"#,
            r#"[" 1"]"#,
            "{}",
        ] {
            fs::write(&path, input).unwrap();
            assert!(read_verifier_vector::<Bn254>(&path).is_err(), "{input}");
        }

        let modulus = num_bigint::BigUint::from(Bn254::MODULUS);
        let max = &modulus - 1u8;
        fs::write(&path, serde_json::to_vec(&[max.to_string()]).unwrap()).unwrap();
        assert_eq!(
            read_verifier_vector::<Bn254>(&path).unwrap(),
            vec![-Bn254::from(1u64)]
        );
        for integer in [
            modulus.clone(),
            modulus + 1u8,
            num_bigint::BigUint::from(1u8) << 512usize,
        ] {
            fs::write(
                &path,
                serde_json::to_vec(&["1".to_string(), integer.to_string()]).unwrap(),
            )
            .unwrap();
            let error = read_verifier_vector::<Bn254>(&path)
                .unwrap_err()
                .to_string();
            assert!(error.contains("index 1"), "{error}");
            assert!(error.contains("less than the modulus"), "{error}");
        }
    }

    fn workspace_fixture() -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        fs::write(
            dir.path().join("Nargo.toml"),
            "[workspace]\nmembers = [\"a\", \"b\"]\n",
        )
        .unwrap();
        for (name, condition, y) in [("a", "x == y", 2), ("b", "x != y", 0)] {
            let root = dir.path().join(name);
            fs::create_dir_all(root.join("src")).unwrap();
            fs::write(
                root.join("Nargo.toml"),
                format!("[package]\nname = \"{name}\"\ntype = \"bin\"\nauthors = []\n"),
            )
            .unwrap();
            fs::write(
                root.join("src/main.nr"),
                format!("fn main(x: Field, y: pub Field) {{ assert({condition}); }}\n"),
            )
            .unwrap();
            fs::write(
                root.join("Prover.toml"),
                format!("x = \"1\"\ny = \"{y}\"\n"),
            )
            .unwrap();
        }
        dir
    }

    #[test]
    fn workspace_cli_runs_later_members_after_failure() {
        let dir = workspace_fixture();
        let options =
            ProgramOptions::try_parse_from(["mavros", "--root", dir.path().to_str().unwrap()])
                .unwrap();
        assert_eq!(run(&options).unwrap(), ExitCode::FAILURE);
        // b is compiled and executed despite a's failed assertion, using b's own inputs.
        assert!(
            dir.path()
                .join("b/mavros_debug/program_bytecode.txt")
                .is_file()
        );
        fs::write(dir.path().join("a/Prover.toml"), "x = \"1\"\ny = \"1\"\n").unwrap();
        assert_eq!(run(&options).unwrap(), ExitCode::SUCCESS);
    }

    #[test]
    fn workspace_compile_keeps_member_artifacts_separate() {
        let dir = workspace_fixture();
        let root = dir.path().to_path_buf();
        let output = PathBuf::from("target/program.mavros");
        assert!(
            run_compile(
                &root,
                &root.join("shared.mavros"),
                None,
                CompileSettings {
                    include_debug_info: true,
                    ..CompileSettings::default()
                }
            )
            .is_err()
        );
        assert!(!root.join("shared.mavros").exists());
        assert_eq!(
            run_compile(
                &root,
                &output,
                Some(&PathBuf::from("target/verifier.json")),
                CompileSettings {
                    include_debug_info: true,
                    ..CompileSettings::default()
                }
            )
            .unwrap(),
            ExitCode::SUCCESS
        );
        for name in ["a", "b"] {
            for file in [
                "program.mavros",
                "program.debug.json",
                "verifier.json",
                "verifier.debug.json",
            ] {
                assert!(root.join(name).join("target").join(file).is_file());
            }
        }
        assert_ne!(
            fs::read(root.join("a/target/program.mavros")).unwrap(),
            fs::read(root.join("b/target/program.mavros")).unwrap()
        );
    }

    #[test]
    fn workspace_compile_rebases_outputs_for_a_single_selected_member() {
        let dir = workspace_fixture();
        for (manifest, member) in [
            (
                "[workspace]\nmembers = ['a', 'b']\ndefault-member = 'b'\n",
                "b",
            ),
            ("[workspace]\nmembers = ['a']\n", "a"),
        ] {
            fs::write(dir.path().join("Nargo.toml"), manifest).unwrap();
            assert_eq!(
                run_compile(
                    &dir.path().to_path_buf(),
                    &PathBuf::from("target/program.mavros"),
                    None,
                    CompileSettings {
                        include_debug_info: true,
                        ..CompileSettings::default()
                    }
                )
                .unwrap(),
                ExitCode::SUCCESS
            );
            for file in ["program.mavros", "program.debug.json"] {
                assert!(dir.path().join(member).join("target").join(file).is_file());
            }
        }
    }

    /// Covers LogUp slots, unused private inputs and partially read compound parameters.
    #[test]
    fn compile_writes_one_artifact() {
        use mavros_artifacts::{Artifact, ArtifactPreamble, ParamVisibility};
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().to_path_buf();
        fs::create_dir_all(root.join("src")).unwrap();
        fs::write(
            root.join("Nargo.toml"),
            "[package]\nname = \"manifest\"\ntype = \"bin\"\nauthors = []\n",
        )
        .unwrap();
        fs::write(
            root.join("src/main.nr"),
            "fn main(unused: Field, unused_u8: u8, x: [Field; 3], y: pub [u32; 3]) -> pub [Field; 2] {
                let arr = [1, x[1], 100, x[1] + 1];
                [arr[y[0]], x[1]]
            }",
        )
        .unwrap();
        let output = root.join("target/program.mavros");
        assert_eq!(
            run_compile(&root, &output, None, CompileSettings::default()).unwrap(),
            ExitCode::SUCCESS
        );
        let bytes = fs::read(&output).unwrap();
        let artifact: Artifact = Artifact::from_bytes(&bytes).unwrap();
        let manifest = artifact.manifest();
        assert_eq!(manifest.entry_blob_field_count(), 8);
        let abi: noirc_abi::Abi = serde_json::from_slice(artifact.abi()).unwrap();
        let names: Vec<_> = abi.parameters.iter().map(|p| p.name.as_str()).collect();
        assert_eq!(names, ["unused", "unused_u8", "x", "y"]);
        assert_eq!(
            artifact.compiler(),
            mavros_compiler::driver::COMPILER_VERSION
        );
        assert_eq!(
            ArtifactPreamble::peek(&bytes).unwrap().compiler.as_deref(),
            Some(mavros_compiler::driver::COMPILER_VERSION)
        );
        let columns: Vec<_> = manifest
            .parameters
            .iter()
            .map(|p| (p.name.as_str(), p.visibility, p.column_start, p.width))
            .collect();
        // The unused Field disappears; the u8 range check and partial array read retain columns.
        assert_eq!(
            columns,
            vec![
                ("unused", ParamVisibility::Private, None, 1),
                ("unused_u8", ParamVisibility::Private, Some(4), 1),
                ("x", ParamVisibility::Private, Some(5), 3),
                ("y", ParamVisibility::Public, Some(1), 3)
            ]
        );
        let ret = manifest.returns.unwrap();
        let layout = artifact.r1cs().witness_layout;
        let start = ret.column_start.unwrap();
        assert_eq!(ret.len, 2);
        assert!(
            start >= 8 && start + ret.len <= layout.algebraic_size,
            "the computed return sits after the parameters: {ret:?} in {layout:?}"
        );
        assert_eq!(ret.visibility, ParamVisibility::Public);
        assert!(
            layout.challenges_size > 0,
            "a witness-indexed array read should need LogUp challenges"
        );

        // Compiling again yields the same bytes: the digest is a stable identity.
        let again = root.join("target/again.mavros");
        assert_eq!(
            run_compile(&root, &again, None, CompileSettings::default()).unwrap(),
            ExitCode::SUCCESS
        );
        assert_eq!(fs::read(&again).unwrap(), bytes);

        // What a consumer does with the file: witgen and AD from its bytecode, R1CS and ABI.
        fs::write(
            root.join("Prover.toml"),
            "unused = \"7\"\nunused_u8 = \"9\"\nx = [\"1\", \"5\", \"2\"]\ny = [\"3\", \"0\", \"0\"]\nreturn = [\"6\", \"5\"]\n",
        )
        .unwrap();
        let (params, declared_return) = api::read_prover_inputs(&root, &abi).unwrap();
        let r1cs = artifact.r1cs();
        let mut bytecode = artifact.bytecode().to_vec();
        let witgen = api::run_witgen_from_binary(&mut bytecode, r1cs, &params, None).unwrap();
        assert!(api::check_witgen(r1cs, &witgen));
        mavros_compiler::abi_helpers::check_witness_manifest(
            manifest,
            &params,
            declared_return.as_ref(),
            &witgen.out_wit_pre_comm,
        )
        .unwrap();
        use ark_ff::UniformRand as _;
        use rand::SeedableRng as _;
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let coeffs: Vec<Bn254> = (0..r1cs.constraints.len())
            .map(|_| Bn254::rand(&mut rng))
            .collect();
        let (a, b, c, _) = api::run_ad_from_binary(&mut bytecode, r1cs, &coeffs, None).unwrap();
        assert!(api::check_ad(r1cs, &coeffs, &a, &b, &c));
    }

    #[test]
    fn direct_package_compile_keeps_explicit_output_paths() {
        let dir = workspace_fixture();
        // Direct package compilation resolves relative output paths from the caller's
        // working directory.
        let output_dir = tempfile::tempdir_in(".").unwrap();
        let output = Path::new(output_dir.path().file_name().unwrap()).join("program.mavros");
        assert_eq!(
            run_compile(
                &dir.path().join("a"),
                &output,
                None,
                CompileSettings::default()
            )
            .unwrap(),
            ExitCode::SUCCESS
        );
        assert!(output.is_file());
    }

    #[test]
    fn package_errors_are_handled_consistently_for_any_member_count() {
        let roots = [PathBuf::from("a"), PathBuf::from("b")];
        for count in [1, 2] {
            let mut visited = Vec::new();
            let result = run_packages(&roots[..count], |root| {
                visited.push(root.clone());
                Err(std::io::Error::other("same package failure").into())
            });
            assert_eq!(result.unwrap(), ExitCode::FAILURE);
            assert_eq!(visited, roots[..count]);
        }
    }

    #[test]
    fn debug_info_is_excluded_by_default_with_path_and_metadata_opt_ins() {
        let relative = ProgramOptions::try_parse_from(["mavros"]).unwrap();
        assert!(!relative.absolute_paths);
        assert!(!relative.include_debug_info);
        assert!(!relative.profile);

        let profiled = ProgramOptions::try_parse_from(["mavros", "--profile"]).unwrap();
        assert!(profiled.profile);
        assert!(ProgramOptions::try_parse_from(["mavros", "--profile", "--skip-vm"]).is_err());

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

    #[test]
    fn logup_soundness_defaults_to_128_and_is_overridable() {
        let default = ProgramOptions::try_parse_from(["mavros"]).unwrap();
        assert_eq!(default.logup_soundness, DEFAULT_LOGUP_SOUNDNESS_BITS);
        assert_eq!(default.logup_soundness, 128);

        let overridden =
            ProgramOptions::try_parse_from(["mavros", "--logup-soundness", "96"]).unwrap();
        assert_eq!(overridden.logup_soundness, 96);

        // `global = true`, so it is also accepted after the `compile` subcommand.
        let on_subcommand =
            ProgramOptions::try_parse_from(["mavros", "compile", ".", "--logup-soundness", "80"])
                .unwrap();
        assert_eq!(on_subcommand.logup_soundness, 80);
    }

    #[test]
    fn profile_run_writes_all_supported_artifacts() {
        let project = tempfile::tempdir().unwrap();
        fs::create_dir(project.path().join("src")).unwrap();
        fs::write(
            project.path().join("Nargo.toml"),
            "[package]\nname = \"profile_smoke_test\"\ntype = \"bin\"\nauthors = []\n\n[dependencies]\n",
        )
        .unwrap();
        fs::write(project.path().join("Prover.toml"), "x = \"1\"\n").unwrap();
        fs::write(
            project.path().join("src/main.nr"),
            "fn main(x: Field) { assert_eq(x, 1); }\n",
        )
        .unwrap();

        let root = project.path().to_str().unwrap();
        let options =
            ProgramOptions::try_parse_from(["mavros", "--root", root, "--profile"]).unwrap();
        assert_eq!(run(&options).unwrap(), ExitCode::SUCCESS);

        let output_dir = project.path().join("mavros_debug/flamegraphs");
        for filename in [
            "constraint_size.folded",
            "witness_size.folded",
            "witgen_time.folded",
            "ad_time.folded",
            "witgen_time.cpuprofile",
            "ad_time.cpuprofile",
        ] {
            assert!(
                output_dir.join(filename).is_file(),
                "expected profiling artifact {filename}"
            );
        }

        let flamegraph_available = std::process::Command::new("flamegraph.pl")
            .arg("--help")
            .output()
            .is_ok();
        if flamegraph_available {
            for filename in [
                "constraint_size.svg",
                "witness_size.svg",
                "witgen_time.svg",
                "ad_time.svg",
            ] {
                assert!(
                    output_dir.join(filename).is_file(),
                    "expected rendered FlameGraph {filename}"
                );
            }
        }
    }

    #[test]
    fn field_defaults_to_bn254_and_is_global() {
        let default = ProgramOptions::try_parse_from(["mavros"]).unwrap();
        assert_eq!(default.field, FieldId::Bn254);

        let selected = ProgramOptions::try_parse_from(["mavros", "--field", "goldilocks"]).unwrap();
        assert_eq!(selected.field, FieldId::Goldilocks);

        let on_subcommand =
            ProgramOptions::try_parse_from(["mavros", "compile", ".", "--field", "goldilocks"])
                .unwrap();
        assert_eq!(on_subcommand.field, FieldId::Goldilocks);

        let before_subcommand =
            ProgramOptions::try_parse_from(["mavros", "--field", "goldilocks", "compile", "."])
                .unwrap();
        assert_eq!(before_subcommand.field, FieldId::Goldilocks);

        assert!(ProgramOptions::try_parse_from(["mavros", "--field", "babybear"]).is_err());
    }

    /// A package whose `main` squares `x` into `y` and reads `x` back from a witness-indexed
    /// array. The read is a lookup, so on Goldilocks the R1CS repeats its LogUp argument.
    fn field_fixture(prover_toml: &str) -> tempfile::TempDir {
        let project = tempfile::tempdir().unwrap();
        fs::create_dir(project.path().join("src")).unwrap();
        fs::write(
            project.path().join("Nargo.toml"),
            "[package]\nname = \"field_selector\"\ntype = \"bin\"\nauthors = []\n\n[dependencies]\n",
        )
        .unwrap();
        fs::write(
            project.path().join("src/main.nr"),
            "fn main(x: Field, y: pub Field, a: [Field; 3], i: u32) {\n\
             assert_eq(x * x, y);\n\
             assert_eq(a[i], x);\n\
             }\n",
        )
        .unwrap();
        fs::write(project.path().join("Prover.toml"), prover_toml).unwrap();
        project
    }

    /// What `mavros --root <project> --field <field>` returns for the package.
    fn run_over(project: &Path, field: &str) -> Result<ExitCode, Error> {
        let root = project.to_str().unwrap();
        let options =
            ProgramOptions::try_parse_from(["mavros", "--root", root, "--field", field]).unwrap();
        run_package(&options, None)
    }

    #[test]
    fn run_executes_over_the_selected_field() {
        use mavros_artifacts::field::goldilocks::MODULUS;

        let x = MODULUS - 1;
        // `(p - 1)^2 = 1` holds only modulo the Goldilocks prime.
        let wraps = field_fixture(&format!(
            "x = \"{x}\"\ny = \"1\"\na = [\"1\", \"2\", \"{x}\"]\ni = \"2\"\n"
        ));
        assert_eq!(
            run_over(wraps.path(), "goldilocks").unwrap(),
            ExitCode::SUCCESS
        );
        // The inputs decode and witgen runs; only the constraint check fails.
        assert_eq!(run_over(wraps.path(), "bn254").unwrap(), ExitCode::FAILURE);

        // The integer square is the bn254 answer and is not a canonical Goldilocks input.
        let square = u128::from(x) * u128::from(x);
        let integer = field_fixture(&format!(
            "x = \"{x}\"\ny = \"{square}\"\na = [\"1\", \"2\", \"{x}\"]\ni = \"2\"\n"
        ));
        assert_eq!(
            run_over(integer.path(), "bn254").unwrap(),
            ExitCode::SUCCESS
        );
        let error = run_over(integer.path(), "goldilocks")
            .unwrap_err()
            .to_string();
        assert!(error.contains(&square.to_string()), "{error}");
        assert!(error.to_lowercase().contains("goldilocks"), "{error}");
    }

    #[test]
    fn compile_writes_artifacts_that_name_the_selected_field() {
        use mavros_artifacts::Artifact;
        use mavros_compiler::vm::bytecode::parse_program_header;

        let project = field_fixture("");
        let root = project.path().to_path_buf();
        let output = root.join("target/program.mavros");
        let verifier = root.join("target/verifier.json");
        assert_eq!(
            run_compile(
                &root,
                &output,
                Some(&verifier),
                CompileSettings {
                    field: FieldId::Goldilocks,
                    ..CompileSettings::default()
                }
            )
            .unwrap(),
            ExitCode::SUCCESS
        );
        let bytes = fs::read(&output).unwrap();
        let artifact = Artifact::<Goldilocks>::from_bytes(&bytes).unwrap();
        assert_eq!(artifact.r1cs().witness_layout.logup_repetitions, 3);
        assert!(Artifact::<Bn254>::from_bytes(&bytes).is_err());
        assert_eq!(
            parse_program_header(artifact.bytecode()).field,
            FieldId::Goldilocks
        );
        let json: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(&verifier).unwrap()).unwrap();
        let words: Vec<u64> = serde_json::from_value(json["binary"].clone()).unwrap();
        assert_eq!(parse_program_header(&words).field, FieldId::Goldilocks);
    }

    #[test]
    fn goldilocks_verifier_vectors_refuse_values_outside_the_field() {
        use mavros_artifacts::field::goldilocks::MODULUS;

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("vector.json");
        fs::write(&path, format!(r#"["0", "{}"]"#, MODULUS - 1)).unwrap();
        assert_eq!(
            read_verifier_vector::<Goldilocks>(&path).unwrap(),
            vec![Goldilocks::from(0u64), Goldilocks::from(MODULUS - 1)]
        );
        // `2^64` has a second limb, which the one-limb field must refuse before decoding.
        for integer in [
            u128::from(MODULUS),
            u128::from(u64::MAX),
            1 << 64,
            u128::MAX,
        ] {
            fs::write(&path, format!(r#"["1", "{integer}"]"#)).unwrap();
            let error = read_verifier_vector::<Goldilocks>(&path)
                .unwrap_err()
                .to_string();
            assert!(error.contains("index 1"), "{error}");
            assert!(error.contains("less than the modulus"), "{error}");
        }
    }
}
