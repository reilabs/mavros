use mavros_compiler::{api, compiler::codegen::CodeGenOptions, vm::bytecode::parse_program_header};

mod common;

#[test]
fn entry_blob_and_return_checks_follow_the_abi() {
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
            0,
            true,
        ),
        (
            "fn main(x: Field) -> pub Field { x + 1 }",
            "x = '3'\nreturn = '4'",
            1,
            true,
        ),
        (
            "fn main(x: Field) -> pub Field { x + 1 }",
            "x = '3'\nreturn = '5'",
            1,
            false,
        ),
        (
            "fn main(x: Field) -> pub Field { x + 1 }",
            "x = '3'",
            1,
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
        fn check(value: Field) { assert(value != 0); }

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
            let callbacks = ((check, { assert(x != 0); }), 17);
            callbacks.0.0(x);
            assert_eq(callbacks.1, 17);
        }
        "#,
        "x = '3'",
        1,
        true,
    );
}

#[test]
fn unconstrained_unit_returns_preserve_reference_argument_lowering() {
    run(
        r#"
        unconstrained fn check(value: &Field) { assert(*value == 3); }
        fn main(x: Field) {
            // Safety: the input is constrained to the checked value.
            let result = unsafe { check(&x) };
            assert_eq(result, ());
            assert_eq(x, 3);
        }
        "#,
        "x = '3'",
        1,
        true,
    );
}

#[test]
fn statement_tuple_operands_keep_signed_arithmetic() {
    for assertion in [
        "assert(((x, { assert(x != 0); }), 7).0.0 > -1);",
        "assert_eq(((-x, { assert(x != 0); }), 7).0.0 / 2, -4);",
        "assert_eq(((-x, { assert(x != 0); }), 7).0.0 >> 1, -4);",
        "assert_eq(((-x, { assert(x != 0); }), 7).0.0 as i16, -8);",
        "assert_eq(-((x, { assert(x != 0); }), 7).0.0, -8);",
    ] {
        run(
            &format!("fn main(x: i8) {{ {assertion} }}"),
            "x = '8'",
            1,
            true,
        );
    }
}

#[test]
fn statement_tuple_signed_comparison_rejects_one_less_than_negative_one() {
    run(
        "fn main(x: i8) { assert(((x, { assert(x != 0); }), 7).0.0 < -1); }",
        "x = '1'",
        1,
        false,
    );
}

#[test]
fn statement_tuple_operands_keep_array_and_string_types() {
    for body in [
        "assert_eq((([x, x], { assert(x != 0); }), 7).0.0.len(), 2);",
        "assert_eq(((\"hi\", { assert(x != 0); }), 7).0.0.as_bytes(), [104, 105]);",
        "assert_eq((([104 as u8, 105], { assert(x != 0); }), 7).0.0.as_str_unchecked(), \"hi\");",
    ] {
        run(&format!("fn main(x: u8) {{ {body} }}"), "x = '1'", 1, true);
    }
}

#[test]
fn empty_array_unconstrained_returns_are_prepared() {
    for (typ, value, check) in [
        (
            "[Field; 0]",
            "[]",
            "let extended = result.as_vector().push_back(x); assert_eq(extended[0], x);",
        ),
        (
            "([Field; 0], Field)",
            "([], x)",
            "assert_eq(result.0.len(), 0); assert_eq(result.1, x);",
        ),
    ] {
        run(
            &format!(
                "unconstrained fn empty(x: Field) -> {typ} {{ assert(x != 0); {value} }}
                fn main(x: Field) {{ let result = unsafe {{ empty(x) }}; {check} }}"
            ),
            "x = '1'",
            1,
            true,
        );
    }
}

#[test]
fn zero_field_entry_parameters_need_no_witness_slots() {
    for (params, inputs, fields, body) in [
        ("e: Empty", "e = {}", 0, "let _ = e;"),
        (
            "e: Empty, x: Field",
            "e = {}\nx = '3'",
            1,
            "let _ = e; assert_eq(x, 3);",
        ),
        ("e: [Empty; 2]", "e = [{}, {}]", 0, "assert_eq(e.len(), 2);"),
    ] {
        run(
            &format!("struct Empty {{}} fn main({params}) {{ {body} }}"),
            inputs,
            fields,
            true,
        );
    }
}

fn run(source: &str, inputs: &str, expected_fields: usize, accepted: bool) {
    let dir = common::noir_project(source);
    std::fs::write(dir.path().join("Prover.toml"), inputs).unwrap();
    let (mut driver, r1cs) = api::compile_to_r1cs(dir.path().to_path_buf(), false).unwrap();
    let (params, declared_return) = api::read_prover_inputs(dir.path(), driver.abi()).unwrap();
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
    let Ok(result) = result else {
        assert!(!accepted, "{source}");
        return;
    };
    assert!(api::check_witgen(&r1cs, &result));
    let manifest = driver.witness_manifest(&r1cs).unwrap();
    let checked = mavros_compiler::abi_helpers::check_witness_manifest(
        &manifest,
        &params,
        declared_return.as_ref(),
        &result.out_wit_pre_comm,
    );
    assert_eq!(checked.is_ok(), accepted, "{source}: {checked:?}");
}

#[test]
fn an_unread_entry_blob_survives_bytecode_dce() {
    // The inputs are declared but not read yet; the body only checks the hard-coded limbs.
    run(
        "global LIMBS: [Field; 4] = [0x1f, 0x2e, 0x3d, 0x4c];

        fn main(owner: Field, path: [Field; 3]) -> pub Field {
            let mut digest = 0;
            for i in 0..4 {
                digest = digest * 256 + LIMBS[i];
            }
            assert(digest != 0);
            digest
        }",
        "owner = '1'\npath = ['2', '3', '4']\nreturn = '523124044'",
        4,
        true,
    );
}

#[test]
fn data_bus_parameters_are_private() {
    use mavros_artifacts::ParamVisibility;
    use mavros_compiler::abi_helpers;

    let dir = common::noir_project(
        "fn main(x: call_data(0) Field, y: pub Field, _unused: call_data(0) Field) -> return_data Field {
            x * y
        }",
    );
    std::fs::write(
        dir.path().join("Prover.toml"),
        "x = '6'\ny = '7'\n_unused = '1'\nreturn = '42'",
    )
    .unwrap();
    let (mut driver, r1cs) = api::compile_to_r1cs(dir.path().to_path_buf(), false).unwrap();
    let manifest = driver.witness_manifest(&r1cs).unwrap();
    let columns: Vec<_> = manifest
        .parameters
        .iter()
        .map(|p| (p.name.as_str(), p.visibility, p.column_start))
        .collect();
    // The public `y` is pinned first; the read `x` follows unpinned; the unread one is elided.
    assert_eq!(
        columns,
        vec![
            ("x", ParamVisibility::DataBus, Some(2)),
            ("y", ParamVisibility::Public, Some(1)),
            ("_unused", ParamVisibility::DataBus, None),
        ]
    );
    let ret = manifest.returns.unwrap();
    assert_eq!(ret.visibility, ParamVisibility::DataBus);
    assert!(ret.column_start.is_some(), "the return is bound to columns");

    let (params, declared_return) = api::read_prover_inputs(dir.path(), driver.abi()).unwrap();
    let mut artifact = driver
        .compile_bytecode_artifact(CodeGenOptions {
            check_constraints: true,
            include_debug_info: true,
        })
        .unwrap();
    let result =
        api::run_witgen_from_binary(&mut artifact.binary, &r1cs, &params, artifact.debug_info)
            .unwrap();
    assert!(api::check_witgen(&r1cs, &result));
    abi_helpers::check_witness_manifest(
        &manifest,
        &params,
        declared_return.as_ref(),
        &result.out_wit_pre_comm,
    )
    .unwrap();
}
