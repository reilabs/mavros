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

fn main(x1: u8, y1: u8, x2: u8, y2: u8, x3: u8, y3: u8, sum_x: pub u8, sum_y: pub u8) {
    let x1 = x1 as Field;
    let y1 = y1 as Field;
    let x2 = x2 as Field;
    let y2 = y2 as Field;
    let x3 = x3 as Field;
    let y3 = y3 as Field;
    let sum_x = sum_x as Field;
    let sum_y = sum_y as Field;

    let points = [
        Point { x: x1, y: y1 },
        Point { x: x2, y: y2 },
        Point { x: x3, y: y3 },
    ];

    let result = sum_points(points);

    assert_eq(result.x, sum_x);
    assert_eq(result.y, sum_y);
}
