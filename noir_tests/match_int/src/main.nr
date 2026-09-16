// Integer literals with a fallback on unsigned, signed and Field scrutinees, and scrutinees
// taken from a parameter, an array element, a struct field, a call and a mutable local.
struct Wrap {
    inner: u8,
}

fn classify_u8(v: u8) -> u8 {
    match v {
        0 => 10,
        1 => 20,
        2 => 30,
        _ => 40,
    }
}

fn classify_i8(v: i8) -> i8 {
    match v {
        -3 => -1,
        0 => 0,
        5 => 1,
        _ => 2,
    }
}

fn classify_field(v: Field) -> Field {
    match v {
        1 => 100,
        2 => 200,
        _ => 300,
    }
}

// A lone binding pattern is a match with no cases: only the default runs.
fn identity(v: u32) -> u32 {
    match v {
        other => other + 1,
    }
}

fn make(v: u8) -> u8 {
    v + 1
}

fn main(x: u8, y: u8, z: u8, w: u8, f: Field, n: i8, arr: [u8; 2], s: Wrap) {
    assert_eq(classify_u8(x), 10);
    assert_eq(classify_u8(y), 20);
    assert_eq(classify_u8(z), 30);
    assert_eq(classify_u8(w), 40);
    assert_eq(classify_i8(n), -1);
    assert_eq(classify_i8(n + 3), 0);
    assert_eq(classify_i8(n + 8), 1);
    assert_eq(classify_i8(n - 1), 2);
    assert_eq(classify_field(f), 200);
    assert_eq(classify_field(f + 5), 300);
    assert_eq(identity(x as u32 + 41), 42);

    let from_array = match arr[1] {
        2 => 20,
        _ => 0,
    };
    assert_eq(from_array, 20);
    let from_field = match s.inner {
        3 => 30,
        _ => 0,
    };
    assert_eq(from_field, 30);
    let from_call = match make(y) {
        2 => 200,
        _ => 0,
    };
    assert_eq(from_call, 200);
    let mut m = y;
    m = m + 4;
    let from_mut = match m {
        5 => 50,
        _ => 0,
    };
    assert_eq(from_mut, 50);
}
