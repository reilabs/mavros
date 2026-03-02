// Tests TupleProj on Ref types in witness type inference.
// The passport's SHA round state is a mutable struct with multiple fields,
// modified in a loop. This exercises TupleProj on Ref(Tuple(...)).
struct State {
    a: Field,
    b: Field,
    c: Field,
    d: Field,
    e: Field,
}

fn compress(state: &mut State, k: Field) {
    let temp = state.a + k;
    state.e = state.d;
    state.d = state.c;
    state.c = state.b;
    state.b = state.a;
    state.a = temp;
}

fn main(x: Field) {
    let mut state = State { a: x, b: 0, c: 0, d: 0, e: 0 };
    compress(&mut state, 42);
    assert_eq(state.a, x + 42);
    assert_eq(state.b, x);
}
