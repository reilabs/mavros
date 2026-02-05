fn main(x: Field, expected: pub Field) {
    let fns: [fn(Field) -> Field; 4] = [
        |a: Field| -> Field { a + 1 },
        |a: Field| -> Field { a * 2 },
        |a: Field| -> Field { a + 10 },
        |a: Field| -> Field { a * 17 },
    ];

    let mut sum: Field = 0;
    for i in 0..4 {
        sum += fns[i](x);
    }

    // x=1: (1+1) + (1*2) + (1+10) + (1*17) = 2 + 2 + 11 + 17 = 32
    assert_eq(sum, expected);
}
