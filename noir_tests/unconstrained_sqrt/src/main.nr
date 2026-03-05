unconstrained fn sqrt(x: Field) -> Field {
    let mut i: u64 = 0;
    let mut sq: Field = 0;
    while (sq != x) {
        i = i + 1;
        sq = i as Field * i as Field;
    }
    i as Field
}

fn main(x: Field) {
    let root = unsafe { sqrt(x) };
    assert(root * root == x);
}
