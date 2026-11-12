// Wrong `return` value: the host compares it with the computed return after
// witness generation and must reject the run.
fn main(x: u8) -> pub u8 {
    x * 2
}
