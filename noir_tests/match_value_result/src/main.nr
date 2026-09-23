// Matches used as values: scalar, nested and tuple results all flow through a merge
// parameter.
enum Shape {
    Empty,
    Line(Field),
    Rect(Field, Field),
}

fn dims(s: Shape) -> (Field, Field) {
    match s {
        Shape::Empty => (0, 0),
        Shape::Line(l) => (l, 1),
        Shape::Rect(w, h) => (w, h),
    }
}

fn main(x: u8, y: u8, w: Field, h: Field) {
    let scalar = match x {
        2 => y,
        _ => 0,
    };
    assert_eq(scalar, y);

    let nested = match x {
        2 => match y {
            5 => 25,
            _ => 20,
        },
        _ => 0,
    };
    assert_eq(nested, 25);

    let (a, b) = dims(Shape::Rect(w, h));
    assert_eq(a, w);
    assert_eq(b, h);
    let (c, d) = dims(Shape::Line(w));
    assert_eq(c, w);
    assert_eq(d, 1);
    let (e, f) = dims(Shape::Empty);
    assert_eq(e, 0);
    assert_eq(f, 0);
}
