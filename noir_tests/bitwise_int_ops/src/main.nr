// Tests bitwise AND, OR, XOR on multi-bit integer witnesses.
// The passport's SHA-256/SHA-512 round functions use all three:
//   Ch(e,f,g) = (e & f) ^ (!e & g)       - AND + XOR
//   rotation = (x >> n) | (x << (32-n))   - OR
fn main(a: u32, b: u32) {
    // AND: masking
    let and_result = a & b;
    assert_eq(and_result, 0x00F0); // 0x0FF0 & 0x00FF = 0x00F0

    // OR: combining
    let c: u32 = 0xFF00;
    let d: u32 = 0x00FF;
    let or_result = c | d;
    assert_eq(or_result, 0xFFFF);

    // XOR: self-inverse
    let x: u32 = 0xDEADBEEF;
    let y: u32 = 0x12345678;
    let xor_result = x ^ y;
    let round_trip = xor_result ^ y;
    assert_eq(round_trip, x);
}
