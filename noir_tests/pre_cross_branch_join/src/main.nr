// PRE join / elimination shapes (stages 1 and 3): mix evaluates x * x in both arms of a
// pure branch and once more below the merge - the cross-branch full-availability join -
// while main's commuted duplicate sums pin the elimination sweep and the overflow
// rangecheck deduplication that rides on it.
fn mix(i: u32, x: Field) -> Field {
    let mut t: Field = 0;
    if i < 2 {
        t = x * x + 1;
    } else {
        t = x * x + 2;
    }
    t + x * x
}

fn main(x: Field, a: u32, b: u32) {
    let mut acc: Field = 0;
    for i in 0..4 {
        acc += mix(i, x);
    }
    let s1 = a + b;
    let s2 = b + a;
    assert(s1 + s2 == 40);
    assert(acc == 78);
}
