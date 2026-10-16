// A bit size past the field fails `assert_max_bit_size`'s `static_assert`, whatever `x` is. Its
// range check holds for every element, so it is dropped rather than lowered.
fn main(x: Field) {
    x.assert_max_bit_size::<300>();
}
