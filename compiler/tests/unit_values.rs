use mavros_compiler::{api, compiler::codegen::CodeGenOptions, vm::bytecode::parse_program_header};

/// Exercise the production frontend, wrapper, ABI encoder and VM together.
/// Counts are explicit so dropping `with_abi_return` cannot make both sides agree.
#[test]
fn abi_return_guards_match_the_generated_entry_blob() {
    let cases = [
        ("fn main() {}", "", 0, true),
        (
            "fn main(x: Field) -> pub () { assert(x != 0); }",
            "x = '3'",
            1,
            true,
        ),
        (
            "struct Empty {} fn main() -> pub Empty { Empty {} }",
            "return = {}",
            1,
            true,
        ),
        (
            "fn main(x: Field) -> pub Field { x + 1 }",
            "x = '3'\nreturn = '4'",
            3,
            true,
        ),
        (
            "fn main(x: Field) -> pub Field { x + 1 }",
            "x = '3'\nreturn = '5'",
            3,
            false,
        ),
        (
            "fn main(x: Field) -> pub Field { x + 1 }",
            "x = '3'",
            3,
            true,
        ),
    ];
    for (source, inputs, expected_fields, accepted) in cases {
        run(source, inputs, expected_fields, accepted);
    }
}

#[test]
fn nested_statement_tuples_preserve_projection_offsets() {
    run(
        r#"
        fn main(x: Field) {
            let t = ((x, { assert(x != 0); }), 7);
            assert_eq(t.1, 7);
            assert_eq(t.0.0, x);
            let projected = (((x, { assert(x != 0); }), 9).0, 11);
            assert_eq(projected.1, 11);
            assert_eq(projected.0.0, x);
            let blocked = ({ (x, { assert(x != 0); }) }, 13);
            assert_eq(blocked.1, 13);
            assert_eq(blocked.0.0, x);
        }
        "#,
        "x = '3'",
        1,
        true,
    );
}

fn run(source: &str, inputs: &str, expected_fields: usize, accepted: bool) {
    let dir = tempfile::tempdir().unwrap();
    std::fs::create_dir(dir.path().join("src")).unwrap();
    std::fs::write(
        dir.path().join("Nargo.toml"),
        "[package]\nname = 'unit_values'\ntype = 'bin'\nauthors = []\n",
    )
    .unwrap();
    std::fs::write(dir.path().join("src/main.nr"), source).unwrap();
    std::fs::write(dir.path().join("Prover.toml"), inputs).unwrap();
    let (mut driver, r1cs) = api::compile_to_r1cs(dir.path().to_path_buf(), false).unwrap();
    let params = api::read_prover_inputs(dir.path(), driver.abi()).unwrap();
    let mut artifact = driver
        .compile_bytecode_artifact(CodeGenOptions {
            check_constraints: true,
            include_debug_info: true,
        })
        .unwrap();
    assert_eq!(
        parse_program_header(&artifact.binary).entry_blob_field_count,
        expected_fields,
        "{source}"
    );
    let result =
        api::run_witgen_from_binary(&mut artifact.binary, &r1cs, &params, artifact.debug_info);
    assert_eq!(result.is_ok(), accepted, "{source}");
    if let Ok(result) = result {
        assert!(r1cs.check_witgen_output(
            &result.out_wit_pre_comm,
            &result.out_wit_post_comm,
            &result.out_a,
            &result.out_b,
            &result.out_c,
        ));
    }
}
