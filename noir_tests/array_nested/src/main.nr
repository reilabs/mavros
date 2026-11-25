fn main(x: [[[u8; 2]; 2]; 2], y: [[u8; 2]; 2]) {
  let x = x.map(|v| v.map(|v| v.map(|v| v as Field)));
  let y = y.map(|v| v.map(|v| v as Field));

  assert_eq(x[0][0][0] * x[0][0][1], y[0][0]);
  assert_eq(x[0][1][0] * x[0][1][1], y[0][1]);
  assert_eq(x[1][0][0] * x[1][0][1], y[1][0]);
  assert_eq(x[1][1][0] * x[1][1][1], y[1][1]);
}
