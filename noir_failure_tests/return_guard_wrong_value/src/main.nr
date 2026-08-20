// Wrong `return` value with guard=1: the return check must reject the witness.
fn main(x: Field) -> pub Field {
    x * 2
}
