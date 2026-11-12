fn main(x: u8, y: pub u8, z: pub u8) {
  let x = x as Field;
  let y = y as Field;
  let z = z as Field;

  let r = x + y;
  assert_eq(r, z);
}

