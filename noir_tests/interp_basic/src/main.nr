// Self-checking program for the monomorphized-AST interpreter (the field-agnostic oracle).
// Every `assert` encodes a known-correct result, so a clean interpret run means the AST
// computes what the source claims. Exercises the basic-functionality slice: native integer
// arithmetic (including a constant above the Goldilocks modulus), function calls, control
// flow, arrays with mutation, tuples, and a cast. All arithmetic is overflow-free so it holds
// under Noir's checked-arithmetic semantics.

fn double(x: u64) -> u64 {
    x * 2
}

fn main() {
    // Native u64 arithmetic with a constant above the Goldilocks modulus
    // (p = 2^64 - 2^32 + 1 = 18446744069414584321, itself < 2^64).
    let p_minus_one: u64 = 18446744069414584320; // p - 1, a legal u64
    assert(p_minus_one + 1 == 18446744069414584321);

    // Function call.
    assert(double(21) == 42);

    // for-loop accumulation: 0 + 1 + ... + 9 == 45.
    let mut sum: u32 = 0;
    for i in 0..10 {
        sum = sum + i;
    }
    assert(sum == 45);

    // Array indexing and in-place mutation.
    let mut arr: [u32; 3] = [10, 20, 30];
    arr[1] = 99;
    assert(arr[1] == 99);
    assert(arr[2] == 30);

    // Heterogeneous tuple.
    let t: (u32, u64) = (7, 100);
    assert(t.0 == 7);
    assert(t.1 == 100);

    // Cast u64 -> Field (field-dependent value, but the assert holds under any field).
    let f: Field = p_minus_one as Field;
    assert(f == 18446744069414584320);

    // Conditional.
    let c: u32 = if sum > 40 { 1 } else { 0 };
    assert(c == 1);
}
