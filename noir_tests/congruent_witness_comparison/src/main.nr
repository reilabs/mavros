// Exercises the witnessed-comparison fold: `pre == post` compares two structurally-identical
// (congruent) booleans derived from a witnessed unconstrained `Option`, so its result is
// `WitnessOf(u1)`. The Click-Cooper writeback proves the operands congruent and folds the comparison
// to `true`, and SCCP keeps the folded constant witness-typed via `cast <const> to WitnessOf`.
//
// This mirrors the structure of `execution_failure/regression_5202` (whose witnessed comparison
// first surfaced this case) but is satisfiable, so it guards witgen/AD correctness of the fold
// end-to-end as an execution_success test.

unconstrained fn get_unconstrained_option() -> Option<u32> {
    Option::some(13)
}

unconstrained fn should_i_assert() -> bool {
    false
}

fn get_magical_boolean() -> bool {
    // Safety: testing context
    let option = unsafe { get_unconstrained_option() };

    let pre = option.is_some() as Field;

    // Safety: testing context. `should_i_assert` is unconstrained and false here, so the assert is
    // not executed; its presence forces `option` through witness spilling so the comparison below is
    // a *witnessed* one.
    if unsafe { should_i_assert() } {
        assert_eq(option, Option::some(42));
    }

    let post = option.is_some() as Field;

    // `pre` and `post` are the same (congruent) witnessed computation, so this is provably true.
    pre == post
}

fn main() {
    assert(get_magical_boolean());
}
