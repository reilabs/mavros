// Wrong `return` value: the host compares it with the computed return after
// witness generation and must reject the run.
fn main(x: Field) -> pub Field {
    x * 2
}
