// A wildcard arm over the untouched variants. In `is_rect` the duplicate bodies are adjacent,
// so the elaborator merges them into a default and the one remaining case is tested. In
// `length` they are split by `Line`, and `deduplicate_cases` only merges consecutive
// duplicates — so no default is produced and the last case runs untested instead.
enum Shape {
    Empty,
    Line(Field),
    Rect(Field, Field),
}

fn is_rect(s: Shape) -> bool {
    match s {
        Shape::Rect(_, _) => true,
        _ => false,
    }
}

fn length(s: Shape) -> Field {
    match s {
        Shape::Line(l) => l,
        _ => 0,
    }
}

fn main(w: Field, h: Field) {
    assert(is_rect(Shape::Rect(w, h)));
    assert(!is_rect(Shape::Line(w)));
    assert(!is_rect(Shape::Empty));
    assert_eq(length(Shape::Line(w)), w);
    assert_eq(length(Shape::Rect(w, h)), 0);
    assert_eq(length(Shape::Empty), 0);
}
