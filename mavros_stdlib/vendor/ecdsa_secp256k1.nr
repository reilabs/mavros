//! secp256k1 field and curve definitions for the ECDSA replacement. The generic bignum and
//! bigcurve machinery is shared with the secp256r1 vendor module.

pub mod bignum {
    pub mod fields {
        pub mod secp256k1Fq {
            use crate::mavros::vendor::ecdsa_secp256r1::bignum::bignum::{
                BigNum, derive_bignum,
            };
            use crate::mavros::vendor::ecdsa_secp256r1::bignum::params::BigNumParams;

            pub global Secp256k1_Fq_PARAMS: BigNumParams<3, 256> = BigNumParams {
                has_multiplicative_inverse: true,
                modulus: [
                    0xfffffffffffffffffffffefffffc2f,
                    0xffffffffffffffffffffffffffffff,
                    0xffff,
                ],
                double_modulus: [
                    0x01fffffffffffffffffffffdfffff85e,
                    0x01fffffffffffffffffffffffffffffe,
                    0x01fffe,
                ],
                redc_param: [0x400000f440, 0x00, 0x400000],
            };

            #[derive_bignum(3, 256, quote { Secp256k1_Fq_PARAMS })]
            pub struct Secp256k1_Fq {
                limbs: [u128; 3],
            }

            pub fn secp256k1_fq_from_limbs(limbs: [u128; 3]) -> Secp256k1_Fq {
                Secp256k1_Fq { limbs }
            }
        }

        pub mod secp256k1Fr {
            use crate::mavros::vendor::ecdsa_secp256r1::bignum::bignum::{
                BigNum, derive_bignum,
            };
            use crate::mavros::vendor::ecdsa_secp256r1::bignum::params::BigNumParams;

            pub global Secp256k1_Fr_PARAMS: BigNumParams<3, 256> = BigNumParams {
                has_multiplicative_inverse: true,
                modulus: [
                    0xaedce6af48a03bbfd25e8cd0364141,
                    0xfffffffffffffffffffffffffffeba,
                    0xffff,
                ],
                double_modulus: [
                    0x015db9cd5e9140777fa4bd19a06c8282,
                    0x01fffffffffffffffffffffffffffd74,
                    0x01fffe,
                ],
                redc_param: [0x48c6542dd7f1100b685ccbf26fb027, 0x5154, 0x400000],
            };

            #[derive_bignum(3, 256, quote { Secp256k1_Fr_PARAMS })]
            pub struct Secp256k1_Fr {
                limbs: [u128; 3],
            }
        }
    }

    pub use fields::secp256k1Fq::Secp256k1_Fq;
    pub use fields::secp256k1Fr::Secp256k1_Fr;
}

pub mod bigcurve {
    pub mod curves {
        pub mod secp256k1 {
            use crate::mavros::vendor::ecdsa_secp256k1::bignum::fields::secp256k1Fq::secp256k1_fq_from_limbs;
            pub use crate::mavros::vendor::ecdsa_secp256k1::bignum::{
                Secp256k1_Fq, Secp256k1_Fr,
            };
            use crate::mavros::vendor::ecdsa_secp256r1::bigcurve::bigcurve::{
                BigCurve, BigCurveParams, derive_curve_impl,
            };
            use crate::mavros::vendor::ecdsa_secp256r1::bigcurve::scalar_field::ScalarField;
            use crate::mavros::vendor::ecdsa_secp256r1::bignum::BigNum;

            pub global SECP256k1_PARAM: BigCurveParams<Secp256k1_Fq> = BigCurveParams {
                a: secp256k1_fq_from_limbs([0x00, 0x00, 0x00]),
                b: secp256k1_fq_from_limbs([0x07, 0x00, 0x00]),
                one: [
                    secp256k1_fq_from_limbs([
                        0x9bfcdb2dce28d959f2815b16f81798,
                        0x667ef9dcbbac55a06295ce870b0702,
                        0x79be,
                    ]),
                    secp256k1_fq_from_limbs([
                        0x17b448a68554199c47d08ffb10d4b8,
                        0xda7726a3c4655da4fbfc0e1108a8fd,
                        0x483a,
                    ]),
                ],
                offset_generator: [
                    secp256k1_fq_from_limbs([
                        0xad9b0399372a80a40485fb4765c830,
                        0x592524e36cf8f8b10532990e0c318b,
                        0x4130,
                    ]),
                    secp256k1_fq_from_limbs([
                        0x0f86d7ffdfd71295914f8c96c8e696,
                        0xb2a472eb30ec190dc82c3f416b8c6d,
                        0xfedf,
                    ]),
                ],
                offset_generator_final: [
                    secp256k1_fq_from_limbs([
                        0xf69e3511e0f28578e8516402091e2d,
                        0xa0afbd65c86c4bb7593dace897e155,
                        0x4f04,
                    ]),
                    secp256k1_fq_from_limbs([
                        0x414e4311b6eaed7bdf0a669162ba0b,
                        0x5cccfc054797f3e2b9133b36f2d7a7,
                        0x765c,
                    ]),
                ],
            };

            #[derive_curve_impl(quote { crate::mavros::vendor::ecdsa_secp256k1::bignum::Secp256k1_Fq }, quote { SECP256k1_PARAM })]
            pub struct Secp256k1 {
                pub x: Secp256k1_Fq,
                pub y: Secp256k1_Fq,
                pub is_infinity: bool,
            }
        }
    }
}
