// Matches in statement position: unit arms with side effects, a unit scrutinee, and arms
// that assert. The merge block carries no parameter.
fn main(x: u8, y: u8) {
    let mut acc: u8 = 0;
    match x {
        1 => {
            acc = acc + 1;
        },
        2 => {
            acc = acc + 2;
        },
        _ => {
            acc = acc + 100;
        },
    }
    assert_eq(acc, 2);

    match () {
        _ => {
            acc = acc + 1;
        },
    }
    assert_eq(acc, 3);

    match y {
        5 => assert(x == 2),
        _ => assert(x != 2),
    }
}
