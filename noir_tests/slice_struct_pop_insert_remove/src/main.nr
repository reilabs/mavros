// Pure-length pop, insert, and remove on a slice of multi-leaf struct elements: after tuple
// elision the slice is two leaf slices, so each op must fan out componentwise (the combination
// the wl corpus never reaches, because witness-length ops are rewritten before elision).
struct P {
    a: Field,
    b: Field,
}

fn main(x: Field) {
    let s: [P] = [P { a: 1, b: 2 }, P { a: x, b: 4 }];
    let (rest, last) = s.pop_back();
    assert_eq(last.a, x);
    assert_eq(last.b, 4);
    assert_eq(rest.len(), 1);

    let grown = rest.insert(1, P { a: 5, b: 6 });
    assert_eq(grown.len(), 2);
    assert_eq(grown[0].a, 1);
    assert_eq(grown[1].b, 6);

    let (shrunk, removed) = grown.remove(0);
    assert_eq(removed.b, 2);
    assert_eq(shrunk.len(), 1);
    assert_eq(shrunk[0].a, 5);
}
