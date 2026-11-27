use mavros_artifacts::{
    Bn254, FieldConfig, Goldilocks, InputValueOrdered, MavrosField, SealedR1CS,
};
use mavros_compiler::{
    Project, abi_helpers, api,
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

/// The frontend refuses `source` over `field` before Mavros lowering, with a located diagnostic
/// that says `expected` in its message, one of its secondaries or one of its notes.
///
/// Matching the text is what makes a refusal test meaningful: a bare `is_err()` is satisfied by a
/// typo in the program as readily as by the rule under test.
#[allow(dead_code)]
pub fn assert_frontend_refused(field: FieldConfig, source: &str, expected: &str) {
    let dir = noir_project(source);
    let mut driver = Driver::new(Project::new(dir.path().to_path_buf()).unwrap(), false);
    let Err(Error::NoirCompilerError {
        diagnostics,
        rendered,
    }) = driver.run_noir_compiler(field)
    else {
        panic!(
            "expected a frontend diagnostic for {source} on {:?}",
            field.id()
        );
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
    let mut ssa = HLSSA::with_main(field, "main".to_string());
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
pub struct HandBuilt<F: MavrosField> {
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
pub fn compile_hand_built<F: MavrosField>(ssa: HLSSA, param_is_public: Vec<bool>) -> HandBuilt<F> {
    let dumps = tempfile::tempdir().unwrap();
    let mut driver = Driver::from_ssa(ssa, dumps.path().to_path_buf(), false)
        .with_param_visibility(param_is_public);
    let r1cs = api::compile_driver_to_r1cs::<F>(&mut driver).unwrap();
    let binary = driver
        .compile_bytecode(CodeGenOptions {
            check_constraints: true,
            include_debug_info: false,
        })
        .unwrap();
    HandBuilt {
        dumps,
        driver,
        r1cs,
        binary,
    }
}

/// A Noir `main` compiled over `F`, with bytecode that leaves the constraints to
/// [`api::check_witgen`].
#[allow(dead_code)]
pub struct Compiled<F: MavrosField> {
    pub dir: TempDir,
    pub driver: Driver,
    pub r1cs: SealedR1CS<F>,
    pub binary: Vec<u64>,
}

/// `source` compiled over `F`.
#[allow(dead_code)]
pub fn compile<F: MavrosField>(source: &str) -> Compiled<F> {
    let dir = noir_project(source);
    let (mut driver, r1cs) = api::compile_to_r1cs::<F>(dir.path().to_path_buf(), false)
        .unwrap_or_else(|error| panic!("{source}: {error}"));
    let binary = api::compile_bytecode(
        &mut driver,
        CodeGenOptions {
            check_constraints: false,
            ..Default::default()
        },
    )
    .unwrap();
    Compiled {
        dir,
        driver,
        r1cs,
        binary,
    }
}

#[allow(dead_code)]
impl<F: MavrosField> Compiled<F> {
    /// The parameters and declared return `inputs`, a `Prover.toml`, gives.
    pub fn inputs(
        &self,
        inputs: &str,
    ) -> (Vec<InputValueOrdered<F>>, Option<InputValueOrdered<F>>) {
        std::fs::write(self.dir.path().join("Prover.toml"), inputs).unwrap();
        api::read_prover_inputs(
            self.dir.path(),
            self.driver.abi(),
            self.driver.limb_bits(),
        )
        .unwrap()
    }

    /// Runs `params` and checks the witness against them and `declared_return`, or returns why the
    /// run or its constraints failed.
    pub fn run(
        &mut self,
        params: &[InputValueOrdered<F>],
        declared_return: Option<&InputValueOrdered<F>>,
    ) -> Result<(), String> {
        let result = api::run_witgen_from_binary(&mut self.binary, &self.r1cs, params, None)
            .map_err(|error| format!("witness generation trapped: {error:?}"))?;
        if !api::check_witgen(&self.r1cs, &result) {
            return Err("the witness breaks a constraint".to_string());
        }
        abi_helpers::check_witness_manifest(
            &self.driver.witness_manifest(&self.r1cs).unwrap(),
            params,
            declared_return,
            &result.out_wit_pre_comm,
        )
    }
}

/// `source` with its `main` constrained and unconstrained.
#[allow(dead_code)]
pub fn in_both_modes(source: &str) -> [String; 2] {
    ["", "unconstrained "].map(|mode| source.replace("fn main", &format!("{mode}fn main")))
}

/// Compiles `source` over `F`, constrained and unconstrained, and runs it on `inputs`, which
/// declare its `return`.
#[allow(dead_code)]
pub fn check<F: MavrosField>(source: &str, inputs: &str) {
    for source in in_both_modes(source) {
        let mut compiled = compile::<F>(&source);
        let (params, declared_return) = compiled.inputs(inputs);
        compiled
            .run(&params, declared_return.as_ref())
            .unwrap_or_else(|error| panic!("{:?} {source}: {error}", F::ID));
    }
}

/// [`check`] on bn254 and on Goldilocks.
#[allow(dead_code)]
pub fn check_both_fields(source: &str, inputs: &str) {
    check::<Bn254>(source, inputs);
    check::<Goldilocks>(source, inputs);
}
