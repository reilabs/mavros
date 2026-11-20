use mavros_artifacts::{FieldConfig, MavrosField, SealedR1CS};
use mavros_compiler::{
    Project,
    compiler::{
        codegen::CodeGenOptions,
        ssa::{
            SourceLocation, ValueId,
            hlssa::{
                HLSSA, Type,
                builder::{HLBlockEmitter, HLSSABuilder},
            },
        },
    },
    driver::{Driver, Error},
};
use tempfile::TempDir;

/// Create an isolated Noir binary package from a complete source program.
pub fn noir_project(source: &str) -> TempDir {
    let dir = tempfile::tempdir().unwrap();
    std::fs::create_dir(dir.path().join("src")).unwrap();
    std::fs::write(
        dir.path().join("Nargo.toml"),
        "[package]\nname = 'test_program'\ntype = 'bin'\nauthors = []\n",
    )
    .unwrap();
    std::fs::write(dir.path().join("src/main.nr"), source).unwrap();
    dir
}

/// `value` as the TOML array of its `len` little-endian bytes, the spelling of a `[u8; len]` input.
#[allow(dead_code)]
pub fn le_bytes(value: &num_bigint::BigUint, len: usize) -> String {
    let mut bytes = value.to_bytes_le();
    assert!(bytes.len() <= len, "{value} takes more than {len} bytes");
    bytes.resize(len, 0);
    let bytes: Vec<String> = bytes.iter().map(u8::to_string).collect();
    format!("[{}]", bytes.join(", "))
}

/// The frontend refuses `source` before Mavros lowering, with a located diagnostic that says
/// `expected` in its message, one of its secondaries or one of its notes.
///
/// Matching the text is what makes a refusal test meaningful: a bare `is_err()` is satisfied by a
/// typo in the program as readily as by the rule under test.
#[allow(dead_code)]
pub fn assert_frontend_refused(source: &str, expected: &str) {
    let dir = noir_project(source);
    let mut driver = Driver::new(Project::new(dir.path().to_path_buf()).unwrap(), false);
    let Err(Error::NoirCompilerError {
        diagnostics,
        rendered,
    }) = driver.run_noir_compiler(FieldConfig::bn254())
    else {
        panic!("expected a frontend diagnostic for {source}");
    };
    assert!(rendered.contains(expected), "{rendered}");
    assert!(rendered.contains("src/main.nr:"), "{rendered}");
    assert!(rendered.contains('|'), "{rendered}");
    assert!(
        diagnostics.iter().any(|diagnostic| {
            let located = !diagnostic.secondaries.is_empty();
            let says = std::iter::once(diagnostic.message.as_str())
                .chain(
                    diagnostic
                        .secondaries
                        .iter()
                        .map(|label| label.message.as_str()),
                )
                .chain(diagnostic.notes.iter().map(String::as_str))
                .any(|text| text.contains(expected));
            located && says
        }),
        "missing located diagnostic {expected:?} for {source}: {diagnostics:?}"
    );
}

/// A hand-built `main` over `field` taking `params` and returning one `ret`, which `body` computes
/// from the parameters.
#[allow(dead_code)]
pub fn hand_built_main(
    field: FieldConfig,
    params: &[Type],
    ret: Type,
    body: impl FnOnce(&mut HLBlockEmitter, &[ValueId]) -> ValueId,
) -> HLSSA {
    let mut ssa = HLSSA::with_main_in(field, "main".to_string());
    let main = ssa.get_unique_entrypoint_id();
    HLSSABuilder::new(&mut ssa).modify_function(main, |b| {
        b.function.add_return_type(ret);
        let entry = b.function.get_entry_id();
        let mut e = b
            .block(entry)
            .with_source_location(SourceLocation::synthetic("main"));
        let params: Vec<ValueId> = params
            .iter()
            .map(|typ| e.add_parameter(typ.clone()))
            .collect();
        let returned = body(&mut e, &params);
        e.terminate_return(vec![returned]);
    });
    ssa
}

/// A hand-built program compiled over `F`, with the driver that compiled it.
#[allow(dead_code)]
pub struct Seeded<F: MavrosField> {
    /// Holds the pipeline's dumps.
    pub dumps: TempDir,
    pub driver: Driver,
    pub r1cs: SealedR1CS<F>,
    /// VM bytecode that checks each constraint as it runs.
    pub binary: Vec<u64>,
}

/// `ssa` through every stage after the Noir frontend, over `F`, with `param_is_public` giving each
/// entry-point parameter's visibility.
#[allow(dead_code)]
pub fn compile_seeded<F: MavrosField>(ssa: HLSSA, param_is_public: Vec<bool>) -> Seeded<F> {
    let dumps = tempfile::tempdir().unwrap();
    let mut driver = Driver::from_ssa(ssa, dumps.path().to_path_buf(), false)
        .with_param_visibility(param_is_public);
    driver.make_struct_access_static().unwrap();
    driver.monomorphize().unwrap();
    driver.spill_witness().unwrap();
    let r1cs = driver.generate_r1cs::<F>().unwrap();
    let binary = driver
        .compile_bytecode(CodeGenOptions {
            check_constraints: true,
            include_debug_info: false,
        })
        .unwrap();
    Seeded {
        dumps,
        driver,
        r1cs,
        binary,
    }
}
