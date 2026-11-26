mod common;

use mavros_artifacts::{Bn254, FieldConfig, FieldId, Goldilocks, MavrosField};
use mavros_compiler::{
    Project, abi_helpers, api,
    compiler::codegen::CodeGenOptions,
    driver::{Driver, Error},
};

fn frontend(source: &str, field: FieldConfig) -> Result<(), Error> {
    let dir = common::noir_project(source);
    let mut driver = Driver::new(Project::new(dir.path().to_path_buf()).unwrap(), false);
    driver.run_noir_compiler(field)
}

#[test]
fn portable_stdlib_replacements_compile_on_both_fields() {
    for field in [FieldConfig::bn254(), FieldConfig::goldilocks()] {
        for source in [
            "fn main(x: u8) -> pub u8 { x + 1 }",
            "fn main(x: [u8; 3]) -> pub [u8; 32] { std::hash::blake2s(x) }",
            "fn main(x: [u8; 3]) -> pub [u8; 32] { std::hash::blake3(x) }",
            "fn main(x: [u8; 16], iv: [u8; 16], key: [u8; 16]) -> pub [u8; 32] {
                std::aes128::aes128_encrypt(x, iv, key)
            }",
        ] {
            frontend(source, field).unwrap_or_else(|error| panic!("{source}: {error}"));
        }
    }
}

#[test]
fn bn254_crypto_stays_available_and_is_unresolved_on_goldilocks() {
    for (source, unavailable_item) in [
        (
            "fn main(x: [u8; 32], y: [u8; 32], sig: [u8; 64], hash: [u8; 32]) -> pub bool {
                std::ecdsa_secp256k1::verify_signature(x, y, sig, hash)
            }",
            "ecdsa_secp256k1",
        ),
        (
            "fn main(x: [u8; 32], y: [u8; 32], sig: [u8; 64], hash: [u8; 32]) -> pub bool {
                std::ecdsa_secp256r1::verify_signature(x, y, sig, hash)
            }",
            "ecdsa_secp256r1",
        ),
        (
            "fn main(x: [u8; 4]) -> pub bool {
                std::hash::poseidon2_permutation(x.map(|v| v as Field))[0] != 0
            }",
            "poseidon2_permutation",
        ),
    ] {
        frontend(source, FieldConfig::bn254()).unwrap();
        for source in [
            source.to_owned(),
            source.replace("fn main", "unconstrained fn main"),
        ] {
            let error = frontend(&source, FieldConfig::goldilocks()).unwrap_err();
            let Error::NoirCompilerError {
                diagnostics,
                rendered,
            } = &error
            else {
                panic!("expected a Noir resolution error, got {error}");
            };
            let expected = format!("Could not resolve '{unavailable_item}' in path");
            assert!(
                diagnostics
                    .iter()
                    .any(|diagnostic| diagnostic.message == expected),
                "expected {expected:?}, got {diagnostics:?}"
            );
            assert!(rendered.contains("src/main.nr:"), "{error}");
        }
    }
}

fn check_program<F: MavrosField>(source: &str, inputs: &[String], unconstrained: bool) {
    let dir = common::noir_project(source);
    let (mut driver, r1cs) = api::compile_to_r1cs::<F>(dir.path().to_path_buf(), false).unwrap();
    if F::ID == FieldId::Goldilocks && !unconstrained {
        assert!(r1cs.witness_layout.logup_repetitions > 1);
    }
    let mut binary = driver
        .compile_bytecode(CodeGenOptions {
            check_constraints: true,
            ..Default::default()
        })
        .unwrap();
    for input in inputs {
        std::fs::write(dir.path().join("Prover.toml"), input).unwrap();
        let (params, expected) = api::read_prover_inputs::<F>(dir.path(), driver.abi()).unwrap();
        let expected = expected.expect("each vector declares its expected digest");
        let result = api::run_witgen_from_binary(&mut binary, &r1cs, &params, None).unwrap();
        assert!(api::check_witgen(&r1cs, &result));
        abi_helpers::check_witness_manifest(
            &driver.witness_manifest(&r1cs).unwrap(),
            &params,
            Some(&expected),
            &result.out_wit_pre_comm,
        )
        .unwrap();
    }
}

fn check_sha256<F: MavrosField>(unconstrained: bool) {
    let mode = if unconstrained { "unconstrained " } else { "" };
    let source = format!(
        "{mode}fn main(block: [u32; 16], state: [u32; 8]) -> pub [u32; 8] {{
            std::hash::sha256_compression(block, state)
        }}"
    );
    let mut inputs = Vec::new();
    let state: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];
    for (message, expected) in [
        (
            b"".as_slice(),
            [
                0xe3b0c442u32,
                0x98fc1c14,
                0x9afbf4c8,
                0x996fb924,
                0x27ae41e4,
                0x649b934c,
                0xa495991b,
                0x7852b855,
            ],
        ),
        (
            b"abc".as_slice(),
            [
                0xba7816bf, 0x8f01cfea, 0x414140de, 0x5dae2223, 0xb00361a3, 0x96177a9c, 0xb410ff61,
                0xf20015ad,
            ],
        ),
        (
            // High bits and long carry chains; digest generated independently with hashlib.
            &[0xff; 55],
            [
                0xaadaed00, 0xa3c5fbb8, 0x072ae7f1, 0x984ba819, 0x9fbe5272, 0xde427d11, 0xeaf31583,
                0xaf37db51,
            ],
        ),
    ] {
        let mut padded = [0u8; 64];
        padded[..message.len()].copy_from_slice(message);
        padded[message.len()] = 0x80;
        padded[56..].copy_from_slice(&(message.len() as u64 * 8).to_be_bytes());
        let block: Vec<_> = padded
            .chunks_exact(4)
            .map(|word| u32::from_be_bytes(word.try_into().unwrap()))
            .collect();
        inputs.push(format!(
            "block = {block:?}\nstate = {state:?}\nreturn = {expected:?}\n"
        ));
    }
    check_program::<F>(&source, &inputs, unconstrained);
}

#[test]
fn sha256_goldilocks_witnesses_match_known_digests() {
    check_sha256::<Goldilocks>(false);
}

#[test]
fn sha256_goldilocks_unconstrained_matches_known_digests() {
    check_sha256::<Goldilocks>(true);
}

#[test]
fn sha256_bn254_witnesses_match_known_digests() {
    check_sha256::<Bn254>(false);
}

#[test]
fn blake_goldilocks_witnesses_match_reference_digests() {
    // BLAKE2s reference from Python hashlib; BLAKE3 from the independent Rust implementation.
    let blake2s = [
        80u8, 140, 94, 140, 50, 124, 20, 226, 225, 167, 43, 163, 78, 235, 69, 47, 55, 69, 139, 32,
        158, 214, 58, 41, 77, 153, 155, 76, 134, 103, 89, 130,
    ];
    let blake3 = *blake3::hash(b"abc").as_bytes();
    let inputs = [format!(
        "input = [97, 98, 99]\nreturn = [{blake2s:?}, {blake3:?}]\n"
    )];
    for mode in ["", "unconstrained "] {
        let source = format!(
            "{mode}fn main(input: [u8; 3]) -> pub [[u8; 32]; 2] {{
            [std::hash::blake2s(input), std::hash::blake3(input)]
        }}"
        );
        check_program::<Goldilocks>(&source, &inputs, !mode.is_empty());
    }
}

/// `unsafe_cast` leaves its bound to the caller, so only the standard library may call it.
#[test]
fn unsafe_cast_is_internal_to_the_standard_library() {
    common::assert_frontend_refused(
        FieldConfig::bn254(),
        "use std::mavros::unsafe_cast;\n\nfn main(x: u8) -> pub u8 {\n    unsafe_cast(x as Field)\n}",
        "unsafe_cast is private and not visible from the current module",
    );
}

/// Noir's `x as uN` and `x as iN` on a `Field` keep the low bits, witnessed or pure, on both fields.
/// The frontend puts no bit window in front of the cast: the witnessed one gets
/// `LowerWitnessNarrowingCast`'s, and both backends truncate the pure one. The low byte has its top
/// bit set, so `x as i8` reads negative and sign-extends.
#[test]
fn field_narrowing_casts_truncate_on_both_fields() {
    // 2^40 + 456, which both fields carry: the low byte is 200, or -56 as an `i8`.
    let x = common::le_bytes(&1099511628232u64.into(), 8);
    let inputs = [format!("x = {x}\nreturn = [200, 456, 4294967240]\n")];
    for mode in ["", "unconstrained "] {
        let source = format!(
            "{mode}fn main(x: [u8; 8]) -> pub [u32; 3] {{ \
             let x = Field::from_le_bytes_checked(x); \
             [(x as u8) as u32, x as u32, (x as i8) as u32] }}"
        );
        check_program::<Bn254>(&source, &inputs, !mode.is_empty());
        check_program::<Goldilocks>(&source, &inputs, !mode.is_empty());
    }
}

/// A bn254 `Field` above `2^128` cast to `u128` keeps its low 128 bits, witnessed or pure. Bit 130
/// is set, so an untruncated value would shift down to 9.
#[test]
fn field_to_u128_truncates_above_the_host_word() {
    // 2^130 + 2^127 + 2^32 + 200.
    let x: num_bigint::BigUint = "1531270651144223085585185733447251919048".parse().unwrap();
    let inputs = [format!(
        "x = {}\nreturn = [1, 200]\n",
        common::le_bytes(&x, 32)
    )];
    for mode in ["", "unconstrained "] {
        let source = format!(
            "{mode}fn main(x: [u8; 32]) -> pub [u32; 2] {{ \
             let w = Field::from_le_bytes_checked(x) as u128; [(w >> 127) as u32, w as u32] }}"
        );
        check_program::<Bn254>(&source, &inputs, !mode.is_empty());
    }
}

/// A witnessed element's bytes, little- and big-endian, as a counted loop decomposes them, on
/// both fields.
#[test]
fn a_witnessed_element_decomposes_into_its_bytes_in_either_order() {
    let main = "fn main(x: u32, y: u32) -> pub ([u8; 8], [u8; 8]) {
            let f = (x as Field) + (y as Field) * 16777216;
            (f.to_le_bytes(), f.to_be_bytes())
        }";
    for (x, y) in [
        (0u64, 0u64),
        (0x0102_0304, 0x0506_0708),
        (u32::MAX.into(), u32::MAX.into()),
    ] {
        let value = x + (y << 24);
        let le: Vec<String> = value.to_le_bytes().iter().map(u8::to_string).collect();
        let be: Vec<String> = value.to_be_bytes().iter().map(u8::to_string).collect();
        common::check_both_fields(
            main,
            &format!(
                "x = {x}\ny = {y}\nreturn = [[{}], [{}]]",
                le.join(", "),
                be.join(", ")
            ),
        );
    }
}

/// An unconstrained byte decomposition into more bytes than a field element holds compiles on
/// every field, and fails at the standard library's `static_assert` on the byte count.
#[test]
fn to_le_bytes_past_the_element_compiles_and_fails_at_its_bound() {
    fn check<F: MavrosField>() {
        let source = "unconstrained fn decompose(x: Field) -> [u8; 40] { x.to_le_bytes() }
            fn main(x: u8) -> pub u8 {
                // Safety: the decomposition fails before its bytes are used.
                let bytes = unsafe { decompose(x as Field) };
                bytes[0]
            }";
        let dir = common::noir_project(source);
        std::fs::write(dir.path().join("Prover.toml"), "x = 7\n").unwrap();
        let (mut driver, r1cs) =
            api::compile_to_r1cs::<F>(dir.path().to_path_buf(), false).unwrap();
        let artifact = driver
            .compile_bytecode_artifact(CodeGenOptions {
                include_debug_info: true,
                ..CodeGenOptions::default()
            })
            .unwrap();
        let mut binary = artifact.binary;
        let (params, _) = api::read_prover_inputs::<F>(dir.path(), driver.abi()).unwrap();
        let Err(error) =
            api::run_witgen_from_binary(&mut binary, &r1cs, &params, artifact.debug_info)
        else {
            panic!("{:?}: 40 bytes are past every field's element", F::ID);
        };
        // The trap carries no message, so its innermost frame tells the byte-count assertion from
        // any other failure: the `static_assert` opening `to_le_bytes`, at line 96 of the pinned
        // standard library's `field/mod.nr`.
        let frame = error.stack_trace().first();
        assert!(
            frame.is_some_and(|frame| frame.function.starts_with("to_le_bytes")
                && frame.location.file.ends_with("field/mod.nr")
                && frame.location.line == 96),
            "{:?}: {error:?}",
            F::ID
        );
    }
    check::<Bn254>();
    check::<Goldilocks>();
}
