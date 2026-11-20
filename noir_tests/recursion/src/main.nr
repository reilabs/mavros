// ===== Direct recursion =====

// Tail-recursive factorial: factorial(n, acc) = n! * acc
// n is constant (Pure), acc carries witness data
fn factorial(n: u32, acc: Field) -> Field {
    if n == 0 {
        acc
    } else {
        factorial(n - 1, acc * (n as Field))
    }
}

// Recursive summation: repeated addition of a witness value
// sum_copies(n, x) = n * x
fn sum_copies(n: u32, x: Field) -> Field {
    if n == 0 {
        0
    } else {
        x + sum_copies(n - 1, x)
    }
}

// ===== Mutual recursion =====

// ping and pong accumulate witness values while bouncing between each other.
// ping adds `a` to the accumulator, pong adds `b`.
// Total after n bounces starting from ping:
//   n=0 -> acc
//   n=1 -> acc + a
//   n=2 -> acc + a + b
//   n=3 -> acc + a + b + a
//   n=4 -> acc + a + b + a + b
//   ...
//   For even n: acc + (n/2)*a + (n/2)*b
//   For odd n:  acc + ((n+1)/2)*a + (n/2)*b
fn ping(n: u32, acc: Field, a: Field, b: Field) -> Field {
    if n == 0 {
        acc
    } else {
        pong(n - 1, acc + a, a, b)
    }
}

fn pong(n: u32, acc: Field, a: Field, b: Field) -> Field {
    if n == 0 {
        acc
    } else {
        ping(n - 1, acc + b, a, b)
    }
}

// ===== Mutual recursion returning tuples =====

// count_even and count_odd: mutual recursion that returns a tuple (even_count, odd_count).
// Walks down from n to 0, counting how many times it passed through each function.
// count_even(n, x) when called with even n: adds x to even accumulator
// count_odd(n, x) when called with odd n: adds x to odd accumulator
fn count_even(n: u32, x: Field) -> (Field, Field) {
    if n == 0 {
        (x, 0) // base case: one even step (n=0), zero odd steps
    } else {
        let (e, o) = count_odd(n - 1, x);
        (e + x, o) // add x to even count
    }
}

fn count_odd(n: u32, x: Field) -> (Field, Field) {
    if n == 0 {
        (0, x) // base case: zero even steps, one odd step (we're in odd)
    } else {
        let (e, o) = count_even(n - 1, x);
        (e, o + x) // add x to odd count
    }
}

fn main(
    x: u8,
    a: u8,
    b: u8,
    expected_fact: pub u16,
    expected_sum: pub u8,
    expected_ping: pub u8,
    expected_even: pub u8,
    expected_odd: pub u8,
) {
    let x = x as Field;
    let a = a as Field;
    let b = b as Field;
    let expected_fact = expected_fact as Field;
    let expected_sum = expected_sum as Field;
    let expected_ping = expected_ping as Field;
    let expected_even = expected_even as Field;
    let expected_odd = expected_odd as Field;

    // --- Direct recursion ---
    // factorial(5, x) = 120 * x
    let f = factorial(5, x);
    assert_eq(f, expected_fact);

    // sum_copies(7, x) = 7 * x
    let s = sum_copies(7, x);
    assert_eq(s, expected_sum);

    // --- Mutual recursion ---
    // ping(6, 0, a, b):
    //   bounces: ping->pong->ping->pong->ping->pong->(done)
    //   acc = 0 + a + b + a + b + a + b = 3a + 3b
    let p = ping(6, 0, a, b);
    assert_eq(p, expected_ping);

    // --- Mutual recursion returning tuples ---
    // count_even(4, x):
    //   count_even(4,x) -> count_odd(3,x) -> count_even(2,x) -> count_odd(1,x) -> count_even(0,x)
    //   count_even(0,x) = (x, 0)
    //   count_odd(1,x): (e,o) = count_even(0,x) = (x,0); return (x, 0+x) = (x, x)
    //   count_even(2,x): (e,o) = count_odd(1,x) = (x,x); return (x+x, x) = (2x, x)
    //   count_odd(3,x): (e,o) = count_even(2,x) = (2x,x); return (2x, x+x) = (2x, 2x)
    //   count_even(4,x): (e,o) = count_odd(3,x) = (2x,2x); return (2x+x, 2x) = (3x, 2x)
    let (ec, oc) = count_even(4, x);
    assert_eq(ec, expected_even);
    assert_eq(oc, expected_odd);
}
