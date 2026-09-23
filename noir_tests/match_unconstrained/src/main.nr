// Matches lowered for the VM: a plain unconstrained match, and arms that break out of or
// continue a loop.
unconstrained fn classify(v: u8) -> u8 {
    match v {
        1 => 10,
        _ => 0,
    }
}

unconstrained fn find(arr: [u8; 4], target: u8) -> u32 {
    let mut idx: u32 = 99;
    for i in 0..4 {
        match arr[i] == target {
            true => {
                idx = i;
                break;
            },
            false => {},
        }
    }
    idx
}

unconstrained fn sum_odds(arr: [u8; 4]) -> u8 {
    let mut total: u8 = 0;
    for i in 0..4 {
        match arr[i] % 2 {
            0 => {
                continue;
            },
            _ => {
                total = total + arr[i];
            },
        }
    }
    total
}

fn main(arr: [u8; 4], target: u8, v: u8) {
    // Safety: every result is checked below.
    let c = unsafe { classify(v) };
    assert_eq(c, 10);
    let found = unsafe { find(arr, target) };
    assert_eq(found, 2);
    let odds = unsafe { sum_odds(arr) };
    assert_eq(odds, 4);
}
