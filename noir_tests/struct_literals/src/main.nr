pub struct Point {
    pub x: Field,
    pub y: Field,
}

fn sum_points(points: [Point; 3]) -> Point {
    let mut result = Point { x: 0, y: 0 };
    for i in 0..3 {
        result = Point { x: result.x + points[i].x, y: result.y + points[i].y };
    }
    result
}

fn main(x1: Field, y1: Field, x2: Field, y2: Field, x3: Field, y3: Field, sum_x: pub Field, sum_y: pub Field) {
    let points = [
        Point { x: x1, y: y1 },
        Point { x: x2, y: y2 },
        Point { x: x3, y: y3 },
    ];

    let result = sum_points(points);

    assert_eq(result.x, sum_x);
    assert_eq(result.y, sum_y);
}
