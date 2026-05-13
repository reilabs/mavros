// Exercises a tuple with more than the legacy 14-field cap.
pub struct Big {
    a0: Field,
    a1: Field,
    a2: Field,
    a3: Field,
    a4: Field,
    a5: Field,
    a6: Field,
    a7: Field,
    a8: Field,
    a9: Field,
    a10: Field,
    a11: Field,
    a12: Field,
    a13: Field,
    a14: Field,
    a15: Field,
    a16: Field,
    a17: Field,
    a18: Field,
    a19: Field,
}

fn main(x: Field, expected_sum: pub Field) {
    let big = Big {
        a0: x,
        a1: x + 1,
        a2: x + 2,
        a3: x + 3,
        a4: x + 4,
        a5: x + 5,
        a6: x + 6,
        a7: x + 7,
        a8: x + 8,
        a9: x + 9,
        a10: x + 10,
        a11: x + 11,
        a12: x + 12,
        a13: x + 13,
        a14: x + 14,
        a15: x + 15,
        a16: x + 16,
        a17: x + 17,
        a18: x + 18,
        a19: x + 19,
    };

    let sum =
        big.a0
        + big.a1
        + big.a2
        + big.a3
        + big.a4
        + big.a5
        + big.a6
        + big.a7
        + big.a8
        + big.a9
        + big.a10
        + big.a11
        + big.a12
        + big.a13
        + big.a14
        + big.a15
        + big.a16
        + big.a17
        + big.a18
        + big.a19;

    assert_eq(sum, expected_sum);
}
