mod common;

use mavros_compiler::{abi_helpers, api, compiler::codegen::CodeGenOptions};

fn check(body: &str, expected: &[u64]) {
    for mode in ["", "unconstrained "] {
        let source = format!(
            "fn inc(x: Field) -> Field {{ x + 1 }}
            fn dbl(x: Field) -> Field {{ x * 2 }}
            fn square(x: Field) -> Field {{ x * x }}
            type Callback = fn(Field) -> Field;
            fn replace(refs: [&mut Callback; 1]) {{ *refs[0] = dbl; }}
            fn get_ref(refs: [&mut Callback; 1]) -> &mut Callback {{ refs[0] }}
            {mode}fn main(i: u32, x: Field) -> pub Field {{ {body} }}"
        );
        let dir = common::noir_project(&source);
        let (mut driver, r1cs) = api::compile_to_r1cs(dir.path().to_path_buf(), false)
            .unwrap_or_else(|error| panic!("{source}: {error}"));
        let mut binary = api::compile_bytecode(
            &mut driver,
            CodeGenOptions {
                check_constraints: true,
                ..Default::default()
            },
        )
        .unwrap();
        for (index, expected) in expected.iter().enumerate() {
            std::fs::write(
                dir.path().join("Prover.toml"),
                format!("i = {index}\nx = 3\nreturn = {expected}"),
            )
            .unwrap();
            let params = api::read_prover_inputs(dir.path(), driver.abi()).unwrap();
            let result = api::run_witgen_from_binary(&mut binary, &r1cs, &params, None)
                .unwrap_or_else(|error| panic!("{source}, index {index}: {error}"));
            assert!(api::check_witgen(&r1cs, &result));
            abi_helpers::check_return_guard(
                driver.abi(),
                &r1cs.witness_layout,
                &params,
                &result.out_wit_pre_comm,
            )
            .unwrap();
        }
    }
}

#[test]
fn vector_operations_preserve_removed_and_remaining_functions() {
    for (body, expected) in [
        (
            "let (rest, last) = [inc, dbl, square].as_vector().pop_back(); rest[i](x) + last(x)",
            [13, 15],
        ),
        (
            "let (first, rest) = [inc, dbl, square].as_vector().pop_front(); first(x) + rest[i](x)",
            [10, 13],
        ),
        (
            "let (rest, removed) = [inc, dbl, square].as_vector().remove(1); rest[i](x) + removed(x)",
            [10, 15],
        ),
    ] {
        check(body, &expected);
    }
    check(
        "let v = [inc, square].as_vector().insert(1, dbl); v[i](x)",
        &[4, 6, 9],
    );
}

#[test]
fn stores_through_references_in_arrays_and_vectors_reach_the_original_function() {
    for store in [
        "let refs = [&mut f]; *refs[0] = dbl;",
        "let refs = [&mut f].as_vector(); *refs[0] = dbl;",
        "let (refs, r) = [&mut f].as_vector().pop_back(); *r = dbl; let _ = refs;",
        "let (r, refs) = [&mut f].as_vector().pop_front(); *r = dbl; let _ = refs;",
        "let (refs, r) = [&mut f].as_vector().remove(0); *r = dbl; let _ = refs;",
        "let refs = [].as_vector().insert(0, &mut f); *refs[0] = dbl;",
        "replace([&mut f]);",
        "let r = get_ref([&mut f]); *r = dbl;",
    ] {
        check(
            &format!("let mut f: fn(Field) -> Field = inc; {store} f(x)"),
            &[6],
        );
    }
}
