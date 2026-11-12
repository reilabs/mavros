fn main(x: u8) {
    let x = x as Field;

    for i in 0..1000000 {
        let r = x + i as Field;
        r.assert_max_bit_size::<32>();
    }
}

