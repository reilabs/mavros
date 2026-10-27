mod common;

use mavros_artifacts::{Field, FieldConfig, Goldilocks, MavrosField};
use mavros_compiler::{
    Project, abi_helpers, api,
    compiler::codegen::CodeGenOptions,
    driver::{Driver, Error},
};

fn frontend(source: &str, field: FieldConfig) -> Result<(), Error> {
    let dir = common::noir_project(source);
    let mut driver = Driver::new(Project::new(dir.path().to_path_buf()).unwrap(), false);
    driver.run_noir_compiler_for_field(field)
}

#[test]
fn portable_stdlib_replacements_compile_on_both_fields() {
    for field in [FieldConfig::bn254(), FieldConfig::goldilocks()] {
        for source in [
            "fn main(x: Field) -> pub Field { x + 1 }",
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
fn bn254_crypto_stays_available_and_reports_targeted_goldilocks_errors() {
    for (source, message) in [
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
            "fn main(x: [Field; 4]) -> pub [Field; 4] {
                std::hash::poseidon2_permutation(x)
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
            assert!(matches!(error, Error::NoirCompilerError { .. }), "{error}");
            let rendered = error.to_string();
            assert!(rendered.contains(message), "{error}");
            assert!(rendered.contains("src/main.nr:"), "{error}");
        }
    }
}

fn check_program<F: MavrosField>(source: &str, field: FieldConfig, inputs: &[String]) {
    let dir = common::noir_project(source);
    let mut driver = Driver::new(Project::new(dir.path().to_path_buf()).unwrap(), false);
    driver.run_noir_compiler_for_field(field).unwrap();
    driver.make_struct_access_static().unwrap();
    driver.monomorphize().unwrap();
    driver.spill_witness().unwrap();
    // Match the Goldilocks integer test harness: test arithmetic with one LogUp challenge.
    // Production's 128-bit target remains unchanged and still rejects insufficient challenges.
    if field == FieldConfig::goldilocks() {
        driver.set_logup_soundness(40);
    }
    let r1cs = driver.generate_r1cs::<F>().unwrap();
    let mut binary = driver
        .compile_bytecode(CodeGenOptions {
            check_constraints: true,
            ..Default::default()
        })
        .unwrap();
    for input in inputs {
        std::fs::write(dir.path().join("Prover.toml"), input).unwrap();
        let (params, expected) =
            api::read_prover_inputs_for_field::<F>(dir.path(), driver.abi()).unwrap();
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

fn check_sha256<F: MavrosField>(field: FieldConfig, unconstrained: bool) {
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
    check_program::<F>(&source, field, &inputs);
}

#[test]
fn sha256_goldilocks_witnesses_match_known_digests() {
    check_sha256::<Goldilocks>(FieldConfig::goldilocks(), false);
}

#[test]
fn sha256_goldilocks_unconstrained_matches_known_digests() {
    check_sha256::<Goldilocks>(FieldConfig::goldilocks(), true);
}

#[test]
fn sha256_bn254_witnesses_match_known_digests() {
    check_sha256::<Field>(FieldConfig::bn254(), false);
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
        check_program::<Goldilocks>(&source, FieldConfig::goldilocks(), &inputs);
    }
}
