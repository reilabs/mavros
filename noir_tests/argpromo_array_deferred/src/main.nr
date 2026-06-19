// argument promotion guard (C1): a `&mut [Field; N]` parameter has an aggregate
// pointee, so promotion is deferred (passing the array by value would trade one
// ref for an N-element copy + return). The ref is preserved; result stays correct.
fn bump_first(arr: &mut [Field; 3]) {
    arr[0] = arr[0] + 1;
}

fn main(a: Field, b: Field, c: Field, e0: pub Field) {
    let mut xs = [a, b, c];
    bump_first(&mut xs);
    assert_eq(xs[0], e0);
}
