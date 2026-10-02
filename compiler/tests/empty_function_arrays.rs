mod common;

use mavros_compiler::{api, compiler::codegen::CodeGenOptions, driver::Error};

#[test]
fn calling_an_empty_function_array_is_rejected_without_a_compiler_panic() {
    for container in ["[]", "[].as_vector()"] {
        let typ = if container == "[]" { "; 0" } else { "" };
        let source = format!(
            "fn main(x: u32) {{
                let callbacks: [fn(()) -> (){typ}] = {container};
                callbacks[x - 1](());
            }}"
        );
        let dir = common::noir_project(&source);
        match api::compile_to_r1cs(dir.path().to_path_buf(), false) {
            Err(error) => assert!(
                matches!(
                    error.downcast_ref::<Error>(),
                    Some(Error::UnsatisfiableProgram(_))
                ),
                "expected an unsatisfiable program: {error}",
            ),
            Ok(_) => panic!("an empty function array cannot be indexed"),
        }
    }
}

#[test]
fn empty_function_calls_only_fail_when_their_branch_is_taken() {
    for mode in ["", "unconstrained "] {
        let source = format!(
            "{mode}fn main(take: bool, x: u32) -> pub Field {{
                if take {{
                    let callbacks: [fn(()) -> Field; 0] = [];
                    callbacks[x](())
                }} else {{ 7 }}
            }}"
        );
        let dir = common::noir_project(&source);
        let (mut driver, r1cs) = api::compile_to_r1cs(dir.path().to_path_buf(), false).unwrap();
        let mut binary = api::compile_bytecode(
            &mut driver,
            CodeGenOptions {
                check_constraints: true,
                ..Default::default()
            },
        )
        .unwrap();
        for take in [false, true] {
            std::fs::write(
                dir.path().join("Prover.toml"),
                format!("take = {take}\nx = 0\nreturn = 7"),
            )
            .unwrap();
            let params = api::read_prover_inputs(dir.path(), driver.abi()).unwrap();
            let result = api::run_witgen_from_binary(&mut binary, &r1cs, &params, None);
            assert_eq!(result.is_ok(), !take, "{mode}take = {take}");
            if let Ok(result) = result {
                assert!(api::check_witgen(&r1cs, &result));
            }
        }
    }
}

#[test]
fn unreachable_empty_calls_preserve_aggregate_return_types() {
    for returns in [
        "[Field; 2]",
        "[Field]",
        "&Field",
        "(Field, [Field])",
        "fn(Field) -> Field",
    ] {
        let dir = common::noir_project(&format!(
            "fn main(x: u32) {{
                if false {{
                    let callbacks: [fn(()) -> {returns}; 0] = [];
                    let _ = callbacks[x](());
                }}
                assert(x == 1);
            }}"
        ));
        api::compile_to_r1cs(dir.path().to_path_buf(), false)
            .unwrap_or_else(|error| panic!("{returns}: {error}"));
    }
}
