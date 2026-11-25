fn f1(x: Field, y: Field) -> (Field, Field) {
  (x + y, x - y)
}

fn fn2(x: Field, y: Field) -> Field {
  let (a, b) = f1(x, y);
  a + b
}

fn main(x: pub u8, y: pub u8) {
  let x = x as Field;
  let y = y as Field;

  assert_eq(fn2(x,y), 2*x);
}

