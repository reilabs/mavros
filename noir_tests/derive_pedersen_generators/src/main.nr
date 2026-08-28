use std::embedded_curve_ops::EmbeddedCurvePoint;
use std::hash::derive_generators;

fn derive_after_offset(offset: u32) -> [EmbeddedCurvePoint; 1] {
    derive_generators("DEFAULT_DOMAIN_SEPARATOR".as_bytes(), offset + 1)
}

fn main() {
    let generators = derive_after_offset(0);

    assert_eq(
        generators[0].x,
        0x054aa86a73cb8a34525e5bbed6e43ba1198e860f5f3950268f71df4591bde402,
    );
    assert_eq(
        generators[0].y,
        0x209dcfbf2cfb57f9f6046f44d71ac6faf87254afc7407c04eb621a6287cac126,
    );

    let default_generators: [EmbeddedCurvePoint; 8] =
        derive_generators("DEFAULT_DOMAIN_SEPARATOR".as_bytes(), 0);
    let expected = [
        EmbeddedCurvePoint {
            x: 0x083e7911d835097629f0067531fc15cafd79a89beecb39903f69572c636f4a5a,
            y: 0x1a7f5efaad7f315c25a918f30cc8d7333fccab7ad7c90f14de81bcc528f9935d,
        },
        EmbeddedCurvePoint {
            x: 0x054aa86a73cb8a34525e5bbed6e43ba1198e860f5f3950268f71df4591bde402,
            y: 0x209dcfbf2cfb57f9f6046f44d71ac6faf87254afc7407c04eb621a6287cac126,
        },
        EmbeddedCurvePoint {
            x: 0x1c44f2a5207c81c28a8321a5815ce8b1311024bbed131819bbdaf5a2ada84748,
            y: 0x03aaee36e6422a1d0191632ac6599ae9eba5ac2c17a8c920aa3caf8b89c5f8a8,
        },
        EmbeddedCurvePoint {
            x: 0x26d8b1160c6821a30c65f6cb47124afe01c29f4338f44d4a12c9fccf22fb6fb2,
            y: 0x05c70c3b9c0d25a4c100e3a27bf3cc375f8af8cdd9498ec4089a823d7464caff,
        },
        EmbeddedCurvePoint {
            x: 0x20ed9c6a1d27271c4498bfce0578d59db1adbeaa8734f7facc097b9b994fcf6e,
            y: 0x29cd7d370938b358c62c4a00f73a0d10aba7e5aaa04704a0713f891ebeb92371,
        },
        EmbeddedCurvePoint {
            x: 0x0224a8abc6c8b8d50373d64cd2a1ab1567bf372b3b1f7b861d7f01257052d383,
            y: 0x2358629b90eafb299d6650a311e79914b0215eb0a790810b26da5a826726d711,
        },
        EmbeddedCurvePoint {
            x: 0x0f106f6d46bc904a5290542490b2f238775ff3c445b2f8f704c466655f460a2a,
            y: 0x29ab84d472f1d33f42fe09c47b8f7710f01920d6155250126731e486877bcf27,
        },
        EmbeddedCurvePoint {
            x: 0x0298f2e42249f0519c8a8abd91567ebe016e480f219b8c19461d6a595cc33696,
            y: 0x035bec4b8520a4ece27bd5aafabee3dfe1390d7439c419a8c55aceb207aac83b,
        },
    ];
    for i in 0..8 {
        assert_eq(default_generators[i].x, expected[i].x);
        assert_eq(default_generators[i].y, expected[i].y);
    }

    let length_generator: [EmbeddedCurvePoint; 1] =
        derive_generators("pedersen_hash_length".as_bytes(), 0);
    assert_eq(
        length_generator[0].x,
        0x2df8b940e5890e4e1377e05373fae69a1d754f6935e6a780b666947431f2cdcd,
    );
    assert_eq(
        length_generator[0].y,
        0x2ecd88d15967bc53b885912e0d16866154acb6aac2d3f85e27ca7eefb2c19083,
    );
}
