mod common;

use mavros_artifacts::{Field, Goldilocks, MavrosField};
use mavros_compiler::{abi_helpers, api, compiler::codegen::CodeGenOptions};

/// Compiles `main` over `F`, constrained and unconstrained, runs it on `inputs` and checks the
/// witness against `inputs`' declared `return`.
fn check<F: MavrosField>(main: &str, inputs: &str) {
    for mode in ["", "unconstrained "] {
        let source = format!(
            "struct Pair {{ lo: u8, hi: u8 }}
            struct One {{ a: u8 }}
            struct Empty {{}}
            fn single(One {{ a }}: One) -> u8 {{ a + 1 }}
            fn single_tuple((x,): (u8,)) -> u8 {{ x * 2 }}
            fn empties(Empty {{}}: Empty, e: Empty, y: u8) -> u8 {{ let _ = e; y }}
            fn swapped(Pair {{ lo, hi }}: Pair) -> Pair {{ Pair {{ lo: hi, hi: lo }} }}
            fn nested(((a, b), mut c): ((u8, u8), u8)) -> u8 {{ c += a; c * b }}
            fn product((x, y): (u8, u8)) -> u8 {{ x * y }}
            fn difference(r: (u8, u8)) -> u8 {{ r.0 - r.1 }}
            fn written((mut a, b): ([u8; 2], u8)) -> u8 {{ a[0] = b; a[0] + a[1] }}
            global SUMS: [fn((u8, u8)) -> u8; 2] = [|(x, y): (u8, u8)| x + y, |(x, y): (u8, u8)| x * y];
            {mode}fn main(p: (u8, u8), q: (u8, u8), pick: bool) -> pub u8 {{ {main} }}"
        );
        let dir = common::noir_project(&source);
        std::fs::write(dir.path().join("Prover.toml"), inputs).unwrap();
        let (mut driver, r1cs) = api::compile_to_r1cs::<F>(dir.path().to_path_buf(), false)
            .unwrap_or_else(|error| panic!("{source}: {error}"));
        let mut binary = api::compile_bytecode(
            &mut driver,
            CodeGenOptions {
                check_constraints: true,
                ..Default::default()
            },
        )
        .unwrap();
        let (params, declared_return) = api::read_prover_inputs(dir.path(), driver.abi()).unwrap();
        let result = api::run_witgen_from_binary(&mut binary, &r1cs, &params, None)
            .unwrap_or_else(|error| panic!("{source}: {error}"));
        assert!(api::check_witgen(&r1cs, &result));
        abi_helpers::check_witness_manifest(
            &driver.witness_manifest(&r1cs).unwrap(),
            &params,
            declared_return.as_ref(),
            &result.out_wit_pre_comm,
        )
        .unwrap_or_else(|error| panic!("{source}: {error}"));
    }
}

fn check_both_fields(main: &str, inputs: &str) {
    check::<Field>(main, inputs);
    check::<Goldilocks>(main, inputs);
}

// The monomorphizer expands a tuple or struct pattern parameter into one parameter per leaf, while
// a caller passes one argument per declared parameter.

#[test]
fn a_function_takes_a_struct_pattern_and_a_nested_tuple_pattern_with_a_mutable_leaf() {
    check_both_fields(
        "let Pair { lo, hi } = swapped(Pair { lo: p.0, hi: p.1 }); nested(((lo, hi), q.0))",
        "p = [3, 4]\nq = [1, 0]\npick = true\nreturn = 15",
    );
}

#[test]
fn a_closure_with_captures_takes_its_environment_before_its_pattern() {
    check_both_fields(
        "let offset = q.1; let f = |(x, y): (u8, u8)| x + y + offset; f(p)",
        "p = [3, 4]\nq = [0, 5]\npick = true\nreturn = 12",
    );
}

#[test]
fn one_call_site_reaches_a_pattern_closure_and_a_plain_one() {
    let main = "let f = if pick { |(x, y): (u8, u8)| x * y } else { |r: (u8, u8)| r.0 - r.1 }; \
                f(p) + [p, q].map(|(x, y)| x + y)[1]";
    check_both_fields(main, "p = [5, 3]\nq = [1, 2]\npick = true\nreturn = 18");
    check_both_fields(main, "p = [5, 3]\nq = [1, 2]\npick = false\nreturn = 5");
}

#[test]
fn a_named_pattern_function_and_a_plain_one_share_a_call_site() {
    let main = "let f = if pick { product } else { difference }; f(p)";
    check_both_fields(main, "p = [5, 3]\nq = [0, 0]\npick = true\nreturn = 15");
    check_both_fields(main, "p = [5, 3]\nq = [0, 0]\npick = false\nreturn = 2");
}

#[test]
fn a_closure_held_by_a_global_takes_a_pattern() {
    check_both_fields(
        "SUMS[0](p) + SUMS[1](q)",
        "p = [5, 3]\nq = [2, 4]\npick = true\nreturn = 16",
    );
}

#[test]
fn a_write_through_a_mutable_array_leaf_stays_local() {
    check_both_fields(
        "let a = [p.0, p.1]; written((a, q.0)) + a[0]",
        "p = [5, 3]\nq = [7, 0]\npick = true\nreturn = 15",
    );
}

/// One leaf matches the declared parameter count, so only the declared type tells the leaf from
/// the whole parameter.
#[test]
fn a_pattern_of_one_leaf_takes_the_whole_parameter() {
    let main = "let f = if pick { single_tuple } else { |(x,): (u8,)| x }; \
                single(One { a: p.0 }) + f((p.1,))";
    check_both_fields(main, "p = [5, 3]\nq = [0, 0]\npick = true\nreturn = 12");
    check_both_fields(main, "p = [5, 3]\nq = [0, 0]\npick = false\nreturn = 9");
}

/// An empty struct pattern has no leaves, so the next parameter of its type matches it.
#[test]
fn an_empty_struct_pattern_takes_no_parameter() {
    check_both_fields(
        "empties(Empty {}, Empty {}, p.1)",
        "p = [5, 3]\nq = [0, 0]\npick = true\nreturn = 3",
    );
}
