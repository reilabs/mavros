use mavros_compiler::{Project, api, compiler::codegen::CodeGenOptions, driver::Driver};
use mavros_vm::interpreter;

#[test]
fn immutable_reference_boundary_preserves_values_and_constraints() {
    let dir = tempfile::tempdir().unwrap();
    let fixture = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../noir_tests/immutable_ref_boundary");
    std::fs::create_dir(dir.path().join("src")).unwrap();
    for file in ["Nargo.toml", "Prover.toml", "src/main.nr"] {
        std::fs::copy(fixture.join(file), dir.path().join(file)).unwrap();
    }
    let mut driver = Driver::new(Project::new(dir.path().to_path_buf()).unwrap(), false);
    driver.run_noir_compiler().unwrap();
    driver.make_struct_access_static().unwrap();
    driver.monomorphize().unwrap();
    driver.spill_witness().unwrap();
    let r1cs = driver.generate_r1cs().unwrap();
    let artifact = driver
        .compile_bytecode_artifact(CodeGenOptions {
            check_constraints: true,
            include_debug_info: true,
        })
        .unwrap();
    let inputs = api::read_prover_inputs(driver.package_root(), driver.abi()).unwrap();
    let result = interpreter::run(
        &artifact.binary,
        r1cs.witness_layout,
        r1cs.constraints_layout,
        &inputs,
        artifact.debug_info,
    )
    .unwrap();
    assert!(r1cs.check_witgen_output(
        &result.out_wit_pre_comm,
        &result.out_wit_post_comm,
        &result.out_a,
        &result.out_b,
        &result.out_c,
    ));
}
