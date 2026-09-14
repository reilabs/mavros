// Guard=0 with a compound return: zero-fills every declared-return leaf
// (struct, array, string, tuple), covering each `zero_param` arm.
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
