fn main(i: u32, x: u8) {
    let x = x as Field;
    let a = [x, x].as_vector().push_back(x);
    let _ = a[i];
    assert(x == 7);
}
