fn main(x: [u8; 2], y: pub u8) {
  let x = x.map(|v| v as Field);
  let y = y as Field;

  assert_eq(x[0] * x[1], y);
}
