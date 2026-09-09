fn check_merge(condition: bool, x: u64) {
    // Both arms, mixed pure/witness leaves, and multiple values at the merge.
    let (mut arr, step) = if condition {
        ([[x; 3]; 4], x)
    } else {
        ([[7; 3]; 4], 2)
    };
    for i in 0..4 {
        if condition {
            arr[i][0] = arr[i][0] + step;
        } else {
            arr[i][2] = arr[i][2] + step;
        }
    }
    let c = condition as u64;
    for i in 0..4 {
        assert_eq(arr[i][0], c * (x + x) + (1 - c) * 7);
        assert_eq(arr[i][1], c * x + (1 - c) * 7);
        assert_eq(arr[i][2], c * x + (1 - c) * 9);
    }
}

fn check_nested(outer: bool, inner: bool, x: u64) {
    let mut arr = [[3; 2]; 2];
    let divisor = outer as u64;
    if outer {
        // When the outer arm is inactive, this division must stay guarded even
        // though the inner array merge introduces its own counted loops.
        arr = if inner { [[x / divisor; 2]; 2] } else { [[x + 1; 2]; 2] };
    }
    let o = outer as u64;
    let i = inner as u64;
    let expected = o * (i * x + (1 - i) * (x + 1)) + (1 - o) * 3;
    for row in 0..2 {
        for col in 0..2 { assert_eq(arr[row][col], expected); }
    }
}

fn check_empty(condition: bool, x: u64) {
    let a: [[u64; 0]; 2] = [[], []];
    let b: [[u64; 0]; 2] = [[], []];
    let (empty, tag) = if condition { (a, x) } else { (b, 7) };
    for row in 0..2 { assert_eq(empty[row].len(), 0); }
    let c = condition as u64;
    assert_eq(tag, c * x + (1 - c) * 7);
}

fn main(flag: bool, other: bool, x: u64) {
    check_merge(flag, x);
    check_merge(!flag, x);
    check_nested(flag, other, x);
    check_nested(!flag, other, x);
    check_nested(flag, !other, x);
    check_nested(!flag, !other, x);
    check_empty(flag, x);
    check_empty(!flag, x);
}
