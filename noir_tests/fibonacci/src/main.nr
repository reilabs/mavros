fn fib(a: Field, b: Field, n: u32) -> Field {
    if n == 0 {
        a
    } else if n == 1 {
        b
    } else {
        fib(a, b, n - 1) + fib(a, b, n - 2)
    }
}

fn main(a: Field, b: Field, result: pub Field) {
    let r = fib(a, b, 10);
    assert_eq(r, result);
}
