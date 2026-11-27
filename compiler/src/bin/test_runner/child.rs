//! The child process: one test on one field, stage by stage.

use super::*;

pub(crate) struct SingleRun<F: MavrosField> {
    pub(crate) driver: Driver,
    pub(crate) r1cs: Option<SealedR1CS<F>>,
    pub(crate) inputs: Result<ProverInputs<F>, String>,
    pub(crate) manifest: Manifest,
}

pub(crate) fn run_single(
    root: PathBuf,
    expect_failure: bool,
    expected_error: Option<&str>,
    analyze: bool,
    field: FieldId,
) {
    match field {
        FieldId::Bn254 => run_single_in::<Bn254>(root, expect_failure, expected_error, analyze),
        FieldId::Goldilocks => {
            run_single_in::<Goldilocks>(root, expect_failure, expected_error, analyze)
        }
    }
}

pub(crate) fn run_single_in<F: MavrosField>(
    root: PathBuf,
    expect_failure: bool,
    expected_error: Option<&str>,
    analyze: bool,
) {
    if let Some(run) = run_single_vm::<F>(root, expect_failure, expected_error, analyze) {
        run_single_wasm(run);
    }
}

pub(crate) fn emit(line: &str) {
    let stdout = std::io::stdout();
    let mut out = stdout.lock();
    let _ = writeln!(out, "{line}");
    let _ = out.flush();
}

/// Set to make each child also emit `INFO:HASH:<artifact>:<blake3>` per artifact and per witness
/// the VM and WASM runs produce. `STATUS.md` records artifact sizes only, so a same-size change
/// (such as renumbered opcodes) is visible only to a content hash.
pub(crate) const ARTIFACT_HASHES_ENV: &str = "MAVROS_ARTIFACT_HASHES";

pub(crate) fn artifact_hashes_enabled() -> bool {
    env::var_os(ARTIFACT_HASHES_ENV).is_some()
}

/// The witness's storage cells, in order, as bytes to hash.
pub(crate) fn witness_bytes<F: MavrosField>(witness: &[F]) -> Vec<u8> {
    let mut cells = vec![0u64; F::STORAGE_CELLS];
    let mut bytes = Vec::with_capacity(witness.len() * F::STORAGE_CELLS * 8);
    for value in witness {
        value.to_frame_cells(&mut cells);
        bytes.extend(cells.iter().flat_map(|cell| cell.to_le_bytes()));
    }
    bytes
}

pub(crate) fn emit_artifact_hash(artifact: &str, bytes: &[u8]) {
    emit(&format!(
        "INFO:HASH:{artifact}:{}",
        blake3::hash(bytes).to_hex()
    ));
}

/// Runs the native pipeline for one test directory, emitting START/END
/// markers for the parent to parse. With `expect_failure` (execution_failure /
/// execution_panic tests) the native AD pipeline is skipped: AD output only
/// feeds proving, and a program whose witgen is expected to trap is never
/// proven. The caller then runs the WASM compile and runtime checks, on every field.
///
/// With `expected_error`, only a refusal at `COMPILED` that says the text is a `reject` there; any
/// other refusal there is a `fail`, since the fixture is written to provoke that one.
/// The `COMPILED` outcome of compiling a fixture that expects `expected_error`.
///
/// A program the frontend or a Mavros validation pass proves invalid never executes, which an
/// expected-failure fixture accepts as a `reject`, as it does a statically unsatisfiable R1CS.
/// `frontend_rejection_expected` and `mavros_rejection_expected` say which refusals count. Any
/// other compiler error stays a plain `fail`, so a real regression is never masked.
pub(crate) fn compile_outcome(
    field: FieldId,
    result: &Result<(), DriverError>,
    expected_error: Option<&str>,
) -> &'static str {
    let reject_or_fail = |expected: bool| if expected { "reject" } else { "fail" };
    match result {
        Ok(()) => "ok",
        Err(error @ DriverError::NoirCompilerError { .. }) => {
            eprintln!("Noir compiler rejected program: {error}");
            reject_or_fail(frontend_rejection_expected(field, error, expected_error))
        }
        Err(
            error @ (DriverError::AssertConstantFailed(_)
            | DriverError::UnsatisfiableProgram(_)
            | DriverError::LoweringRefused(_)),
        ) => {
            eprintln!("Mavros compiler rejected program: {error}");
            reject_or_fail(mavros_rejection_expected(error, expected_error))
        }
        Err(error) => {
            eprintln!("Compile error: {error}");
            "fail"
        }
    }
}

pub(crate) fn run_single_vm<F: MavrosField>(
    root: PathBuf,
    expect_failure: bool,
    expected_error: Option<&str>,
    analyze: bool,
) -> Option<SingleRun<F>> {
    let checking_codegen = CodeGenOptions {
        check_constraints: true,
        include_debug_info: true,
    };

    // 1. Compile
    emit("START:COMPILED");
    let project = match Project::new(root.clone()) {
        Ok(project) => project,
        Err(error) => {
            eprintln!("Project error: {error}");
            emit("END:COMPILED:fail");
            return None;
        }
    };
    let mut driver = Driver::new(project, false);
    let compile_result = (|| -> Result<(), DriverError> {
        driver.run_noir_compiler(F::CONFIG)?;
        driver.make_struct_access_static()?;
        driver.monomorphize()?;
        driver.spill_witness()?;
        Ok(())
    })();
    let outcome = compile_outcome(F::ID, &compile_result, expected_error);
    emit(&format!("END:COMPILED:{outcome}"));
    if outcome != "ok" {
        return None;
    }

    // 2. R1CS
    emit("START:R1CS");
    let r1cs = match driver.generate_r1cs::<F>() {
        Ok(r) => {
            let rows = r.constraints.len();
            let cols = r.witness_layout.size();
            emit(&format!("END:R1CS:ok:{rows}:{cols}"));
            if artifact_hashes_enabled() {
                emit_artifact_hash(
                    "r1cs",
                    &bincode::serialize(r.system()).expect("R1CS must serialize"),
                );
            }
            // Opt-in, report-only: how many rows/cols a post-seal compaction could remove. The
            // parent only echoes this `INFO:` line under `--analyze-r1cs`; it stays off the
            // `END:R1CS` line so STATUS.md and the determinism fingerprint never depend on it.
            if analyze {
                let stats = r1cs_compact::analyze(&r, driver.protected_r1cs_cols());
                emit(&format!(
                    "INFO:R1CS_ANALYSIS:total_rows={rows} algebraic_rows={} removable_rows={} \
                     removable_cols={} dups={} taut={} pins={} merges={} elim={} lin_rows={} \
                     rounds={}",
                    stats.algebraic_rows,
                    stats.removable_rows,
                    stats.removable_cols,
                    stats.duplicate_rows,
                    stats.tautology_rows,
                    stats.pinned_cols,
                    stats.merged_cols,
                    stats.eliminated_cols,
                    stats.linear_elim_rows,
                    stats.rounds,
                ));
            }
            Some(r)
        }

        // A program whose assertion can never hold is rejected as unsatisfiable during R1CS
        // generation. For an execution_failure test that is the expected outcome, so it gets a
        // separate `reject` marker, distinct from `fail`, which flags a real problem (such as an
        // unsupported construct) that must never be silently accepted.
        Err(error @ DriverError::UnsatisfiableProgram(_)) => {
            eprintln!("R1CS rejected program as unsatisfiable: {error}");
            emit("END:R1CS:reject");
            None
        }
        Err(e) => {
            eprintln!("R1CS generation error: {e}");
            emit("END:R1CS:fail");
            None
        }
    };

    // 3. Compile bytecode: a single binary holding both the witgen and AD entry points, plus its
    //    standalone debug sidecar (depends on R1CS).
    let program_artifact = r1cs.as_ref().and_then(|_| {
        emit("START:COMPILE");
        match driver.compile_bytecode_artifact(checking_codegen) {
            Ok(artifact) => {
                let bytes = artifact.binary.len() * 8;
                let debug_info = serde_json::to_vec_pretty(
                    artifact
                        .debug_info
                        .as_ref()
                        .expect("debug info was requested for the VM test artifact"),
                )
                .expect("VM debug info must serialize");
                let debug_bytes = debug_info.len();
                emit(&format!("END:COMPILE:ok:{bytes}:{debug_bytes}"));
                if artifact_hashes_enabled() {
                    let binary: Vec<u8> = artifact
                        .binary
                        .iter()
                        .flat_map(|word| word.to_le_bytes())
                        .collect();
                    emit_artifact_hash("bytecode", &binary);
                    emit_artifact_hash("vm_debug", &debug_info);
                }
                Some(artifact)
            }
            Err(_) => {
                emit("END:COMPILE:fail");
                None
            }
        }
    });

    let source_path_root = driver.package_root().to_path_buf();

    // Load inputs (needed for witgen run)
    let inputs = api::read_prover_inputs::<F>(
        driver.package_root(),
        driver.abi(),
        driver.limb_bits(),
    )
    .map_err(|e| e.to_string());
    let ordered_params = inputs.as_ref().map(|(params, _)| params.clone());

    let manifest = r1cs.as_ref().map(|r1cs| {
        driver
            .witness_manifest(r1cs)
            .map_err(|error| error.to_string())
    });

    // 4. Run witgen  (depends on COMPILE)
    let witgen_result = program_artifact.as_ref().and_then(|artifact| {
        emit("START:WITGEN_RUN");
        let r1cs = r1cs.as_ref().unwrap();
        let params = match &ordered_params {
            Ok(params) => params.as_slice(),
            Err(reason) => {
                eprintln!("witgen run error: {reason}");
                emit("END:WITGEN_RUN:input-error");
                return None;
            }
        };
        match interpreter::run(
            &artifact.binary,
            r1cs.witness_layout,
            r1cs.constraints_layout,
            &interpreter::flatten_param_vec(params),
            artifact.debug_info.clone(),
        ) {
            Ok(result) => {
                // The parent scores an expected failure by its run steps, so a wrong `return`
                // fails here rather than at the correctness check.
                if let Some(reason) =
                    declared_return_mismatch(&manifest, &inputs, &result.out_wit_pre_comm)
                {
                    eprintln!("witgen run error: {reason}");
                    emit("END:WITGEN_RUN:return-mismatch");
                    return None;
                }
                emit("END:WITGEN_RUN:ok");
                if artifact_hashes_enabled() {
                    emit_artifact_hash("witness_pre", &witness_bytes(&result.out_wit_pre_comm));
                    emit_artifact_hash("witness_post", &witness_bytes(&result.out_wit_post_comm));
                }
                Some(result)
            }
            Err(mut e) => {
                e.relativize_source_paths(&source_path_root);
                eprintln!("witgen run error: {e}");
                emit("END:WITGEN_RUN:fail");
                None
            }
        }
    });

    // 5. Check witgen correctness  (depends on WITGEN_RUN)
    if let (Some(result), Some(r1cs)) = (&witgen_result, &r1cs) {
        emit("START:WITGEN_CORRECT");
        let correct = r1cs.check_witgen_output(
            &result.out_wit_pre_comm,
            &result.out_wit_post_comm,
            &result.out_a,
            &result.out_b,
            &result.out_c,
        ) && check_parameter_columns(&manifest, &inputs, &result.out_wit_pre_comm);
        emit(if correct {
            "END:WITGEN_CORRECT:ok"
        } else {
            "END:WITGEN_CORRECT:fail"
        });
    }

    // 6. Witgen leak check  (depends on WITGEN_RUN)
    if let Some(result) = &witgen_result {
        emit("START:WITGEN_NOLEAK");
        let leftover = result.instrumenter.final_memory_usage();
        emit(if leftover == 0 {
            "END:WITGEN_NOLEAK:ok"
        } else {
            "END:WITGEN_NOLEAK:fail"
        });
    }

    // 7. Run AD  (depends on COMPILE, independent of witgen). Skipped for expected-failure
    //    tests, so the native AD steps report as not-applicable.
    let ad_result = program_artifact
        .as_ref()
        .filter(|_| !expect_failure)
        .and_then(|artifact| {
            emit("START:AD_RUN");
            let r1cs = r1cs.as_ref().unwrap();
            let ad_coeffs: Vec<F> =
                api::random_ad_coeffs_with(r1cs, &mut rand::rngs::StdRng::seed_from_u64(42));
            match interpreter::run_ad(
                &artifact.binary,
                &ad_coeffs,
                r1cs.witness_layout,
                r1cs.constraints_layout,
                artifact.debug_info.clone(),
            ) {
                Ok((ad_a, ad_b, ad_c, ad_instrumenter)) => {
                    emit("END:AD_RUN:ok");
                    Some((ad_coeffs, ad_a, ad_b, ad_c, ad_instrumenter))
                }
                Err(mut e) => {
                    e.relativize_source_paths(&source_path_root);
                    eprintln!("AD run error: {e}");
                    emit("END:AD_RUN:fail");
                    None
                }
            }
        });

    // 8. Check AD correctness  (depends on AD_RUN)
    if let (Some((coeffs, ad_a, ad_b, ad_c, _)), Some(r1cs)) = (&ad_result, &r1cs) {
        emit("START:AD_CORRECT");
        let correct = r1cs.check_ad_output(coeffs, ad_a, ad_b, ad_c);
        emit(if correct {
            "END:AD_CORRECT:ok"
        } else {
            "END:AD_CORRECT:fail"
        });
    }

    // 9. AD leak check  (depends on AD_RUN)
    if let Some((_, _, _, _, instrumenter)) = &ad_result {
        emit("START:AD_NOLEAK");
        let leftover = instrumenter.final_memory_usage();
        emit(if leftover == 0 {
            "END:AD_NOLEAK:ok"
        } else {
            "END:AD_NOLEAK:fail"
        });
    }

    Some(SingleRun {
        driver,
        r1cs,
        inputs,
        manifest,
    })
}

pub(crate) fn run_single_wasm<F: MavrosField>(run: SingleRun<F>) {
    let SingleRun {
        mut driver,
        r1cs,
        inputs,
        manifest,
    } = run;
    let checking_codegen = CodeGenOptions {
        check_constraints: true,
        include_debug_info: true,
    };
    let source_path_root = driver.package_root().to_path_buf();
    let ordered_params = inputs.as_ref().map(|(params, _)| params.clone());

    // 10. Compile WASM: a single module exporting both the witgen and AD entry
    //     points (depends on R1CS).
    let wasm_path = r1cs.as_ref().and_then(|r1cs| {
        emit("START:WASM_COMPILE");
        let tmpdir = tempfile::tempdir().ok()?;
        let wasm_path = tmpdir.keep().join("program.wasm");
        let wasm_opts = WasmCompileOpts::fast(
            wasm_runtime::locate_or_build().unwrap_or_else(|error| panic!("{error}")),
        )
        .with_debug_path_root(&source_path_root)
        .with_debug_info();
        match driver.compile_llvm_targets(
            false,
            r1cs,
            Some((wasm_path.clone(), wasm_opts)),
            checking_codegen,
        ) {
            Ok(_) if wasm_path.exists() => {
                let bytes = fs::metadata(&wasm_path).map(|m| m.len()).unwrap_or(0);
                let debug_path = wasm_debug_info_path(&wasm_path);
                let Ok(debug_metadata) = fs::metadata(&debug_path) else {
                    eprintln!(
                        "WASM compile succeeded but debug sidecar was not found at {:?}",
                        debug_path
                    );
                    emit("END:WASM_COMPILE:fail");
                    return None;
                };
                let debug_bytes = debug_metadata.len();
                emit(&format!("END:WASM_COMPILE:ok:{bytes}:{debug_bytes}"));
                if artifact_hashes_enabled() {
                    let module = fs::read(&wasm_path).expect("WASM module must be readable");
                    let debug = fs::read(&debug_path).expect("WASM debug sidecar must be readable");
                    emit_artifact_hash("wasm", &module);
                    emit_artifact_hash("wasm_debug", &debug);
                }
                Some(wasm_path)
            }
            Ok(_) => {
                eprintln!(
                    "WASM compile succeeded but output file not found at {:?}",
                    wasm_path
                );
                emit("END:WASM_COMPILE:fail");
                None
            }
            Err(e) => {
                eprintln!("WASM compile error: {:?}", e);
                emit("END:WASM_COMPILE:fail");
                None
            }
        }
    });

    // The module is compiled once, by whichever of the two runs below reaches it first.
    let loaded_module = OnceCell::new();
    let load = |wasm_path: &Path| -> Result<&wasm_host::LoadedModule, Box<dyn std::error::Error>> {
        loaded_module
            .get_or_init(|| wasm_host::LoadedModule::load(wasm_path).map_err(|e| format!("{e:?}")))
            .as_ref()
            .map_err(|e| e.clone().into())
    };

    // 11. Run witgen WASM  (depends on WASM_COMPILE)
    let wasm_result = wasm_path.as_ref().and_then(|wasm_path| {
        emit("START:WITGEN_WASM_RUN");
        let r1cs = r1cs.as_ref().unwrap();
        let params = match &ordered_params {
            Ok(params) => params.as_slice(),
            Err(reason) => {
                eprintln!("WASM run error: {reason}");
                emit("END:WITGEN_WASM_RUN:input-error");
                return None;
            }
        };

        let entry_blob_field_count = driver.entry_blob_field_count();
        if interpreter::flattened_param_count(params) != entry_blob_field_count {
            eprintln!("WASM run error: Unexpected number of inputs supplied");
            emit("END:WITGEN_WASM_RUN:error");
            return None;
        }

        match load(wasm_path).and_then(|loaded| {
            wasm_host::run_witgen_loaded(loaded, r1cs, params, entry_blob_field_count)
        }) {
            Ok(result) => {
                if let Some(reason) =
                    declared_return_mismatch(&manifest, &inputs, &result.out_wit_pre_comm)
                {
                    eprintln!("WASM run error: {reason}");
                    emit("END:WITGEN_WASM_RUN:return-mismatch");
                    return None;
                }
                emit("END:WITGEN_WASM_RUN:ok");
                if artifact_hashes_enabled() {
                    emit_artifact_hash(
                        "wasm_witness_pre",
                        &witness_bytes(&result.out_wit_pre_comm),
                    );
                    emit_artifact_hash(
                        "wasm_witness_post",
                        &witness_bytes(&result.out_wit_post_comm),
                    );
                }
                Some(result)
            }
            Err(e) => {
                eprintln!("WASM run error: {:?}", e);
                emit("END:WITGEN_WASM_RUN:fail");
                None
            }
        }
    });

    // 12. Check witgen WASM correctness  (depends on WITGEN_WASM_RUN)
    if let (Some(result), Some(r1cs)) = (&wasm_result, &r1cs) {
        emit("START:WITGEN_WASM_CORRECT");

        let correct = r1cs.check_witgen_output(
            &result.out_wit_pre_comm,
            &result.out_wit_post_comm,
            &result.out_a,
            &result.out_b,
            &result.out_c,
        ) && check_parameter_columns(&manifest, &inputs, &result.out_wit_pre_comm);
        emit(if correct {
            "END:WITGEN_WASM_CORRECT:ok"
        } else {
            "END:WITGEN_WASM_CORRECT:fail"
        });
    }

    // 13. Witgen WASM leak check  (depends on WITGEN_WASM_RUN)
    if let Some(result) = &wasm_result {
        emit("START:WITGEN_WASM_NOLEAK");
        emit(if result.live_bytes == 0 {
            "END:WITGEN_WASM_NOLEAK:ok"
        } else {
            "END:WITGEN_WASM_NOLEAK:fail"
        });
    }

    // 14. AD WASM Run  (depends on WASM_COMPILE; same module, AD entry point)
    let ad_wasm_result = wasm_path.as_ref().and_then(|wasm_path| {
        emit("START:AD_WASM_RUN");
        let r1cs = r1cs.as_ref().unwrap();
        let ad_coeffs: Vec<F> =
            api::random_ad_coeffs_with(r1cs, &mut rand::rngs::StdRng::seed_from_u64(42));
        match load(wasm_path).and_then(|loaded| wasm_host::run_ad_loaded(loaded, r1cs, &ad_coeffs))
        {
            Ok(result) => {
                emit("END:AD_WASM_RUN:ok");
                Some((ad_coeffs, result))
            }
            Err(e) => {
                eprintln!("AD WASM run error: {:?}", e);
                emit("END:AD_WASM_RUN:fail");
                None
            }
        }
    });

    // 15. AD WASM Correct  (depends on AD_WASM_RUN)
    if let (Some((coeffs, result)), Some(r1cs)) = (&ad_wasm_result, &r1cs) {
        emit("START:AD_WASM_CORRECT");
        let correct = r1cs.check_ad_output(coeffs, &result.out_da, &result.out_db, &result.out_dc);
        emit(if correct {
            "END:AD_WASM_CORRECT:ok"
        } else {
            "END:AD_WASM_CORRECT:fail"
        });
    }

    // 16. AD WASM leak check  (depends on AD_WASM_RUN)
    if let Some((_, result)) = &ad_wasm_result {
        emit("START:AD_WASM_NOLEAK");
        emit(if result.live_bytes == 0 {
            "END:AD_WASM_NOLEAK:ok"
        } else {
            "END:AD_WASM_NOLEAK:fail"
        });
    }
}

pub(crate) type ProverInputs<F> = (
    Vec<interpreter::InputValueOrdered<F>>,
    Option<interpreter::InputValueOrdered<F>>,
);

pub(crate) type Manifest = Option<Result<WitnessManifest, String>>;

/// The parameter columns only: the declared return was already scored at the run step.
pub(crate) fn check_parameter_columns<F: MavrosField>(
    manifest: &Manifest,
    inputs: &Result<ProverInputs<F>, String>,
    wit_pre_comm: &[F],
) -> bool {
    let (Some(manifest), Ok((params, _))) = (manifest, inputs) else {
        return false;
    };
    manifest
        .as_ref()
        .map_err(Clone::clone)
        .and_then(|manifest| {
            abi_helpers::check_witness_manifest(manifest, params, None, wit_pre_comm)
        })
        .inspect_err(|reason| eprintln!("witness manifest check failed: {reason}"))
        .is_ok()
}

pub(crate) fn declared_return_mismatch<F: MavrosField>(
    manifest: &Manifest,
    inputs: &Result<ProverInputs<F>, String>,
    wit_pre_comm: &[F],
) -> Option<String> {
    let Ok((_, Some(declared_return))) = inputs else {
        return None;
    };
    match manifest {
        Some(Ok(manifest)) => {
            abi_helpers::check_declared_return(manifest, declared_return, wit_pre_comm).err()
        }
        Some(Err(reason)) => Some(reason.clone()),
        None => Some("no witness manifest: R1CS generation did not succeed".to_string()),
    }
}
