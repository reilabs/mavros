// Unit-returning main: no return columns, the blob is exactly [x].
fn main(x: u8) -> pub () {
    let x = x as Field;

    assert(x != 0);
}
