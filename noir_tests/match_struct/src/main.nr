// A struct pattern is a single untested case that emits no condition; literal fields
// become nested switches on the bound fields.
struct Point {
    x: u32,
    y: u32,
}

struct Segment {
    start: Point,
    end: Point,
}

fn sum(p: Point) -> u32 {
    match p {
        Point { x, y } => x + y,
    }
}

fn on_axis(p: Point) -> u32 {
    match p {
        Point { x: 0, y } => y,
        Point { x, y: 0 } => x,
        Point { x, y } => x + y,
    }
}

fn starts_at_origin(s: Segment) -> bool {
    match s {
        Segment { start: Point { x: 0, y: 0 }, end: _ } => true,
        _ => false,
    }
}

fn main(x: u32, y: u32) {
    assert_eq(sum(Point { x, y }), x + y);
    assert_eq(on_axis(Point { x: 0, y }), y);
    assert_eq(on_axis(Point { x, y: 0 }), x);
    assert_eq(on_axis(Point { x, y }), x + y);
    let origin = Point { x: x - 3, y: 0 };
    assert(starts_at_origin(Segment { start: origin, end: Point { x, y } }));
    assert(!starts_at_origin(Segment { start: Point { x, y }, end: origin }));
}
