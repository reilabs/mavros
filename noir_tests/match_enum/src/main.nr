// Enum variants with zero, one and two fields: the tag is projected from field 0, the
// payload from field idx + 1, and the last arm runs untested because there is no default.
enum Shape {
    Empty,
    Line(Field),
    Rect(Field, Field),
}

// A single-variant enum has one untested case, so no tag is projected at all.
enum Only {
    Just(u32),
}

fn area(s: Shape) -> Field {
    match s {
        Shape::Empty => 0,
        Shape::Line(l) => l,
        Shape::Rect(w, h) => w * h,
    }
}

fn unwrap(o: Only) -> u32 {
    match o {
        Only::Just(v) => v,
    }
}

fn main(w: Field, h: Field, n: u32) {
    assert_eq(area(Shape::Empty), 0);
    assert_eq(area(Shape::Line(w)), w);
    assert_eq(area(Shape::Rect(w, h)), w * h);
    assert_eq(unwrap(Only::Just(n)), n);
}
