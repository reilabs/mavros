// Nested patterns switch again on a case argument, which is why `bind_case_arguments` must
// record the argument's type.
enum Shape {
    Empty,
    Line(Field),
    Rect(Field, Field),
}

enum Wrapper {
    Inner(Shape),
    Nothing,
}

fn inner_area(w: Wrapper) -> Field {
    match w {
        Wrapper::Inner(Shape::Rect(a, b)) => a * b,
        Wrapper::Inner(Shape::Line(l)) => l,
        Wrapper::Inner(Shape::Empty) => 0,
        Wrapper::Nothing => 0,
    }
}

// A literal inside a payload.
fn is_unit_line(s: Shape) -> bool {
    match s {
        Shape::Line(1) => true,
        _ => false,
    }
}

fn main(w: Field, h: Field) {
    assert_eq(inner_area(Wrapper::Inner(Shape::Rect(w, h))), w * h);
    assert_eq(inner_area(Wrapper::Inner(Shape::Line(w))), w);
    assert_eq(inner_area(Wrapper::Inner(Shape::Empty)), 0);
    assert_eq(inner_area(Wrapper::Nothing), 0);
    assert(is_unit_line(Shape::Line(w - 5)));
    assert(!is_unit_line(Shape::Line(w)));
    assert(!is_unit_line(Shape::Empty));
}
