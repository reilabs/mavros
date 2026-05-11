fn main(six_as_u32: u32) {
    assert_eq(loop_incl(3), 6 as u32);
}


fn loop_incl(x: u32) -> u32 {
    let mut sum = 0;
    for i in 0..=x {
        sum = sum + i;
    }
    sum
}
