fn check_writes(enabled: bool, index: u32) {
    let original: [Field; 3] = [2, 5, 9];
    let mut arr = original;
    if enabled {
        arr[index] = arr[index] + 3;
        arr[index] = arr[index] * 2;
        arr[0] = arr[0] + 7;
    }
    let c = enabled as Field;
    for j in 0..3 {
        let hit = (index == j) as Field;
        let delta = c * hit * (original[j] + 6);
        let extra = c * ((j == 0) as Field) * 7;
        assert_eq(arr[j], original[j] + delta + extra);
    }
    assert_eq(original, [2, 5, 9]);

    // The updated arm is the else arm, and it has multiple writes to one cell.
    let reversed = if enabled {
        original
    } else {
        let mut changed = original;
        changed[1] = 8;
        changed[1] = changed[1] + 2;
        changed
    };
    assert_eq(reversed[0], 2);
    assert_eq(reversed[1], c * 5 + (1 - c) * 10);
    assert_eq(reversed[2], 9);

    // Neither arm is an update of the other: retain the general array merge.
    let replaced = if enabled { [11, 12, 13] } else { original };
    for j in 0..3 {
        assert_eq(replaced[j], c * (11 + j as Field) + (1 - c) * original[j]);
    }
}

fn check_nested(enabled: bool) {
    let mut cube: [[[Field; 2]; 2]; 2] = [[[2; 2]; 2]; 2];
    if enabled {
        cube[1][0][1] = cube[1][0][1] + 7;
        cube[1][1][0] = cube[1][1][0] + 9;
        cube[1][0][1] = cube[1][0][1] * 3;
    }
    let c = enabled as Field;
    for i in 0..2 {
        for j in 0..2 {
            for k in 0..2 {
                let first = ((i == 1) & (j == 0) & (k == 1)) as Field;
                let second = ((i == 1) & (j == 1) & (k == 0)) as Field;
                assert_eq(cube[i][j][k], 2 + c * (first * 25 + second * 9));
            }
        }
    }
}

fn check_pure_indices(enabled: bool) {
    let mut arr: [Field; 3] = [2, 5, 9];
    // The final index is out of bounds but its branch is inactive.
    for i in 0..4 {
        if enabled & (i < 3) {
            arr[i] = arr[i] + 3;
            arr[i] = arr[i] * 2;
            arr[0] = arr[0] + 7;
        }
    }
    let c = enabled as Field;
    assert_eq(arr[0], 2 + c * 29);
    assert_eq(arr[1], 5 + c * 11);
    assert_eq(arr[2], 9 + c * 15);
}

fn main(flag: bool, index: u32) {
    // An inactive out-of-bounds access and an active, overlapping write chain.
    check_writes(flag, index);
    check_writes(!flag, index % 3);
    check_pure_indices(flag);
    check_pure_indices(!flag);
    check_nested(flag);
    check_nested(!flag);
}
