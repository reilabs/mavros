use mavros_compiler::{api, compiler::codegen::CodeGenOptions};

#[test]
fn immutable_reference_boundary_preserves_values_and_constraints() {
    let dir = tempfile::tempdir().unwrap();
    let fixture = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../noir_tests/immutable_ref_boundary");
    std::fs::create_dir(dir.path().join("src")).unwrap();
    for file in ["Nargo.toml", "Prover.toml", "src/main.nr"] {
        std::fs::copy(fixture.join(file), dir.path().join(file)).unwrap();
    }
    let (mut driver, r1cs) = api::compile_to_r1cs(dir.path().to_path_buf(), false).unwrap();
    let mut artifact = driver
        .compile_bytecode_artifact(CodeGenOptions {
            check_constraints: true,
            include_debug_info: true,
        })
        .unwrap();
    let inputs = api::read_prover_inputs(driver.package_root(), driver.abi()).unwrap();
    let result =
        api::run_witgen_from_binary(&mut artifact.binary, &r1cs, &inputs, artifact.debug_info)
            .unwrap();
    assert!(api::check_witgen(&r1cs, &result));
}
