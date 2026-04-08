// Test witness-indexed array lookups, including from global arrays.

global PRIMES: [Field; 5] = [2, 3, 5, 7, 11];

fn main(x: Field, idx: u32) {
    // 1. Witness-indexed lookup on a local constant array
    let local_arr: [Field; 5] = [10, 20, 30, 40, 50];
    let local_val = local_arr[idx];
    assert_eq(local_val, 30); // idx=2 -> 30

    // 2. Witness-indexed lookup on a global array
    let global_val = PRIMES[idx];
    assert_eq(global_val, 5); // idx=2 -> 5

    // 3. Mixed-mode array: constants and witness values
    let mixed: [Field; 4] = [1, x, 100, x + 1];
    let mixed_val = mixed[idx];
    assert_eq(mixed_val, 100); // idx=2 -> 100

    // 4. Array of u32 elements
    let u32_arr: [u32; 4] = [10, 20, 30, 40];
    let u32_val = u32_arr[idx];
    assert_eq(u32_val, 30); // idx=2 -> 30
}
