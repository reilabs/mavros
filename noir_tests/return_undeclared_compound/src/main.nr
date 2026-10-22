// No `return` key with a compound return (struct, array, string, tuple): every
// leaf is written to the witness and nothing is checked against it.
struct Out {
    tag: str<3>,
    vals: [u8; 2],
    flag: bool,
}

fn main(x: u8) -> pub (Out, Field) {
    let out = Out {
        tag: "abc",
        vals: [x, x + 1],
        flag: true,
    };
    (out, x as Field + 100)
}
