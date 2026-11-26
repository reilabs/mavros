// The monomorphizer expands a tuple or struct pattern parameter into one parameter per leaf, while
// each caller passes one argument per declared parameter.
struct Pair {
    lo: u8,
    hi: u8,
}

fn swapped(Pair { lo, hi }: Pair) -> Pair {
    Pair { lo: hi, hi: lo }
}

fn nested(((a, b), mut c): ((u8, u8), u8)) -> u8 {
    c += a;
    c * b
}

unconstrained fn unconstrained_sum(points: [(u8, u8); 2]) -> [u8; 2] {
    points.map(|(x, y)| x + y)
}

fn main(points: [(u8, u8); 2], pair: Pair, pick: bool) -> pub [u8; 6] {
    let offset = pair.lo;
    // A closure with captures takes its environment ahead of its pattern's leaves.
    let shifted = points.map(|(x, y)| x + y + offset);
    // One call site reaches a pattern closure and a plain one.
    let f = if pick { |(x, y): (u8, u8)| x * y } else { |p: (u8, u8)| p.0 - p.1 };
    // Safety: each sum is recomputed in-circuit below.
    let sums = unsafe { unconstrained_sum(points) };
    assert_eq(sums[0], points[0].0 + points[0].1);
    assert_eq(sums[1], points[1].0 + points[1].1);
    let Pair { lo, hi } = swapped(pair);
    [shifted[0], shifted[1], f(points[1]), nested(((lo, hi), 1)), sums[1], lo]
}
