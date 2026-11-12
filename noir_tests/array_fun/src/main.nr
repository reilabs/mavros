
fn mk_arrs(x: Field) -> [[Field; 3]; 4] {
  let a1 = [1,1,1];
  let a2 = [x,x,x];
  [a1, a2, a1, a1]
}

fn main(x: u8, y: pub u8) {
  let x = x as Field;
  let y = y as Field;

  let a = mk_arrs(x);
  assert_eq(a[1][1], x);
}
