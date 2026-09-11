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
    check_sparse_aliases(flag);
    check_sparse_aliases(!flag);
    check_model(flag, flag);
    check_model(flag, !flag);
    check_model(!flag, flag);
    check_model(!flag, !flag);
}

// Compare nested writes against a scalar model, including runtime aliases and
// a whole-row replacement between writes. Loop indices remain pure.
fn check_model(enabled: bool, outer: bool) {
    for i in 0..2 {
        for j in 0..3 {
            for k in 0..3 {
                let original: [[Field; 3]; 2] = [[2, 3, 5], [7, 11, 13]];
                let replacement: [Field; 3] = [17, 19, 23];
                let mut arr = original;
                if outer {
                    if enabled {
                        arr[i][j] = arr[i][j] + 3;
                        arr[i][k] = arr[i][k] * 2;
                        arr[1] = replacement;
                        arr[i][j] = arr[i][j] + 5;
                    }
                }
                for r in 0..2 {
                    for c in 0..3 {
                        let mut expected = original[r][c];
                        if (r == i) & (c == j) { expected = expected + 3; }
                        if (r == i) & (c == k) { expected = expected * 2; }
                        if r == 1 { expected = replacement[c]; }
                        if (r == i) & (c == j) { expected = expected + 5; }
                        let active = (outer & enabled) as Field;
                        assert_eq(arr[r][c], active * expected + (1 - active) * original[r][c]);
                    }
                }
            }
        }
    }
}

// This stays profitable after the cost gate. Distinct loop values can alias at
// runtime, so removing original stores must preserve last-write semantics.
fn check_sparse_aliases(enabled: bool) {
    for i in 0..3 {
        for j in 0..3 {
            let mut arr: [Field; 128] = [2; 128];
            if enabled {
                arr[i] = arr[i] + 3;
                arr[j] = arr[j] * 2;
                arr[i] = arr[i] + 7;
            }
            // Check every possible written cell, one adjacent cell, and the tail.
            assert_eq(arr[127], 2);
            for k in 0..4 {
                let mut expected = 2;
                if k == i { expected = expected + 3; }
                if k == j { expected = expected * 2; }
                if k == i { expected = expected + 7; }
                let active = enabled as Field;
                assert_eq(arr[k], active * expected + (1 - active) * 2);
            }
        }
    }
}
