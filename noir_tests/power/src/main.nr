fn main(x: u8, y: pub [u8; 32]) {
  let x = x as Field;
  // Folded by hand rather than read with `Field::from_le_bytes_checked`, which refuses 32 bytes
  // on a field with a narrower modulus: `y` is the bn254 power, and the program still compiles
  // and runs its AD on every field.
  let mut expected = 0;
  for i in 0..32 {
    expected = expected * 256 + y[31 - i] as Field;
  }

  let mut r = 1;
  for i in 0..1000000 {
    r *= x;
  }
  assert(r == expected);
}
