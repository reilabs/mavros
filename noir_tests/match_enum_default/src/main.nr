// A wildcard arm: the untouched variants share one body, so the elaborator merges them into
// a default and every remaining case is tested.
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
