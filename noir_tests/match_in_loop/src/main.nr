// A match inside a constrained loop, so the loop is unrolled around it.
fn count_target(arr: [u8; 4], target: u8) -> u32 {
    let mut count: u32 = 0;
    for i in 0..4 {
        match arr[i] == target {
            true => {
                count = count + 1;
            },
            false => {},
        }
    }
    count
}

fn sum_odds(arr: [u8; 4]) -> u8 {
    let mut total: u8 = 0;
    for i in 0..4 {
        match arr[i] % 2 {
            0 => {},
            _ => {
                total = total + arr[i];
            },
        }
    }
    total
}

fn main(arr: [u8; 4], target: u8) {
    assert_eq(count_target(arr, target), 1);
    assert_eq(sum_odds(arr), 4);
}
