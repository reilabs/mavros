use noirc_abi::{Abi, AbiParameter, AbiReturnType, AbiType, AbiVisibility, Sign};

use super::*;

fn string_abi(param_names: &[&str], has_return: bool) -> Abi {
    let string = AbiType::String { length: 2 };
    Abi {
        parameters: param_names
            .iter()
            .map(|name| AbiParameter {
                name: (*name).to_string(),
                typ: string.clone(),
                visibility: if name.starts_with("pub_") {
                    AbiVisibility::Public
                } else {
                    AbiVisibility::Private
                },
            })
            .collect(),
        return_type: has_return.then(|| AbiReturnType {
            abi_type: string,
            visibility: AbiVisibility::Public,
        }),
        error_types: Default::default(),
    }
}

fn str_value(text: &str) -> Value {
    Value::Str(text.as_bytes().to_vec())
}

fn int_abi(sign: Sign, width: u32) -> Abi {
    let mut abi = string_abi(&["x"], false);
    abi.parameters[0].typ = AbiType::Integer { sign, width };
    abi
}

#[test]
fn missing_parameter_is_an_error() {
    let dir = tempfile::tempdir().unwrap();
    let abi = string_abi(&["x"], false);
    let error = crate::api::read_prover_inputs::<ark_bn254::Fr>(dir.path(), &abi).unwrap_err();
    assert!(error.to_string().contains("Prover.toml"), "{error}");
    std::fs::write(dir.path().join("Prover.toml"), "y = 'ab'\n").unwrap();
    let error = crate::api::read_prover_inputs::<ark_bn254::Fr>(dir.path(), &abi).unwrap_err();
    assert!(
        error.to_string().contains("`x`"),
        "error does not name the parameter: {error}"
    );
}

/// The elements of `value` split at `widths`, little-endian.
fn limb_elements<F: MavrosField>(value: u128, widths: &[usize]) -> InputValueOrdered<F> {
    let mut low = 0;
    InputValueOrdered::Vec(
        widths
            .iter()
            .map(|width| {
                let limb = (value >> low) & ((1u128 << width) - 1);
                low += width;
                InputValueOrdered::Field(F::from(limb))
            })
            .collect(),
    )
}

#[test]
fn integers_cross_as_their_bit_pattern_and_never_reduce() {
    use mavros_artifacts::{Goldilocks, field::goldilocks::MODULUS};

    let elements = |sign, width, value: BigInt| {
        ordered_params::<Goldilocks>(&int_abi(sign, width), &[Value::Int(value)])
    };
    let u63_max = (1u64 << 63) - 1;
    for value in [0, 1, u63_max] {
        assert_eq!(
            elements(Sign::Unsigned, 63, value.into()).unwrap(),
            vec![InputValueOrdered::Field(Goldilocks::from(value))],
        );
    }
    assert_eq!(
        elements(Sign::Signed, 8, BigInt::from(-1)).unwrap(),
        vec![InputValueOrdered::Field(Goldilocks::from(255u64))],
    );
    assert_eq!(
        elements(Sign::Signed, 8, BigInt::from(-128)).unwrap(),
        vec![InputValueOrdered::Field(Goldilocks::from(128u64))],
    );
    // Wider than one element carries, the pattern crosses as its limbs, so the modulus, and `-1` as
    // an `i64` (pattern `2^64 - 1`), keep their patterns.
    let widths = |bits| int_limb_widths(bits, Goldilocks::CONFIG);
    for (sign, width, value, pattern) in [
        (Sign::Unsigned, 64, BigInt::from(MODULUS), MODULUS as u128),
        (Sign::Signed, 64, BigInt::from(-1), u64::MAX as u128),
        (
            Sign::Unsigned,
            128,
            BigInt::from(1u128 << 100),
            1u128 << 100,
        ),
    ] {
        assert_eq!(
            elements(sign, width, value).unwrap(),
            vec![limb_elements(pattern, &widths(width as usize))],
        );
    }
    for (sign, value) in [
        (Sign::Unsigned, 256),
        (Sign::Unsigned, -1),
        (Sign::Signed, 128),
        (Sign::Signed, -129),
    ] {
        let error = elements(sign, 8, BigInt::from(value)).unwrap_err();
        assert!(error.contains("out of range"), "{error}");
    }
}

#[test]
fn structs_and_tuples_keep_declaration_order_and_name_nested_refusals() {
    use mavros_artifacts::Goldilocks;

    let u32_type = AbiType::Integer {
        sign: Sign::Unsigned,
        width: 32,
    };
    let mut abi = string_abi(&["x"], false);
    abi.parameters[0].typ = AbiType::Array {
        length: 2,
        typ: Box::new(AbiType::Struct {
            path: "S".to_string(),
            fields: vec![
                ("b".to_string(), AbiType::Boolean),
                (
                    "a".to_string(),
                    AbiType::Tuple {
                        fields: vec![u32_type.clone(), u32_type],
                    },
                ),
            ],
        }),
    };
    let element = |b: bool, first: u64, second: u64| {
        Value::Seq(vec![
            Value::Bool(b),
            Value::Seq(vec![Value::Int(first.into()), Value::Int(second.into())]),
        ])
    };
    let scalar = |value: u64| InputValueOrdered::Field(Goldilocks::from(value));
    let elements = |first, last| {
        ordered_params::<Goldilocks>(
            &abi,
            &[Value::Seq(vec![
                element(true, 2, 3),
                element(false, first, last),
            ])],
        )
    };
    let tuple = |first, second| {
        InputValueOrdered::Struct(vec![
            ("0".to_string(), scalar(first)),
            ("1".to_string(), scalar(second)),
        ])
    };
    assert_eq!(
        elements(4, 5).unwrap(),
        vec![InputValueOrdered::Vec(vec![
            InputValueOrdered::Struct(vec![
                ("b".to_string(), scalar(1)),
                ("a".to_string(), tuple(2, 3)),
            ]),
            InputValueOrdered::Struct(vec![
                ("b".to_string(), scalar(0)),
                ("a".to_string(), tuple(4, 5)),
            ]),
        ])],
    );
    let error = elements(4, 1 << 32).unwrap_err();
    assert!(error.contains("`x[1].a.1`"), "{error}");
}

#[test]
fn booleans_and_strings_cross_one_element_per_scalar() {
    use mavros_artifacts::Goldilocks;

    let mut abi = string_abi(&["x", "pub_y"], false);
    abi.parameters[1].typ = AbiType::Boolean;
    let elements = ordered_params::<Goldilocks>(&abi, &[str_value("ab"), Value::Bool(true)]);
    assert_eq!(
        elements.unwrap(),
        vec![
            InputValueOrdered::Vec(vec![
                InputValueOrdered::Field(Goldilocks::from(b'a' as u64)),
                InputValueOrdered::Field(Goldilocks::from(b'b' as u64)),
            ]),
            InputValueOrdered::Field(Goldilocks::from(1u64)),
        ],
    );
    assert!(ordered_params::<Goldilocks>(&abi, &[str_value("ab")]).is_err());
    let mut field_abi = abi.clone();
    field_abi.parameters[1].typ = AbiType::Field;
    let error = ordered_params::<Goldilocks>(&field_abi, &[str_value("ab"), Value::Int(1.into())])
        .unwrap_err();
    assert!(
        error.contains("`pub_y`") && error.contains("Field"),
        "{error}"
    );
}

/// An integer type one element holds injectively crosses whole; a wider one crosses as its limbs,
/// whatever the value, from `Prover.toml` and from a byte blob alike.
#[test]
fn a_u64_crosses_whole_on_bn254_and_as_limbs_on_goldilocks() {
    use mavros_artifacts::{Field, Goldilocks, field::goldilocks::MODULUS};

    let dir = tempfile::tempdir().unwrap();
    let abi = int_abi(Sign::Unsigned, 64);
    std::fs::write(dir.path().join("Prover.toml"), format!("x = '{MODULUS}'\n")).unwrap();
    let (params, _) = crate::api::read_prover_inputs::<Field>(dir.path(), &abi).unwrap();
    assert_eq!(params, vec![InputValueOrdered::Field(Field::from(MODULUS))]);
    let limbs = vec![limb_elements::<Goldilocks>(
        MODULUS as u128,
        &int_limb_widths(64, Goldilocks::CONFIG),
    )];
    let (params, _) = crate::api::read_prover_inputs::<Goldilocks>(dir.path(), &abi).unwrap();
    assert_eq!(params, limbs);
    let blob = crate::api::decode_param_bytes::<Goldilocks>(&abi, &MODULUS.to_le_bytes()).unwrap();
    assert_eq!(blob, limbs);
}

#[test]
fn byte_blobs_and_prover_toml_give_the_same_elements() {
    use mavros_artifacts::Field;

    let mut abi = string_abi(&["x", "pub_y"], true);
    abi.parameters[1].typ = AbiType::Array {
        length: 2,
        typ: Box::new(AbiType::Integer {
            sign: Sign::Signed,
            width: 12,
        }),
    };
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(
        dir.path().join("Prover.toml"),
        "x = 'ab'\npub_y = [-2, 2047]\nreturn = 'cd'\n",
    )
    .unwrap();
    let (params, declared) = crate::api::read_prover_inputs::<Field>(dir.path(), &abi).unwrap();
    // `i12` takes two little-endian bytes: -2 is 0x0ffe, 2047 is 0x07ff.
    let blob = [b'a', b'b', 0xfe, 0x0f, 0xff, 0x07];
    assert_eq!(
        crate::api::decode_param_bytes::<Field>(&abi, &blob).unwrap(),
        params
    );
    assert_eq!(
        Some(crate::api::decode_return_bytes::<Field>(&abi, b"cd").unwrap()),
        declared
    );
    // A padding bit above the width is refused.
    let error =
        crate::api::decode_param_bytes::<Field>(&abi, &[b'a', b'b', 0xfe, 0x1f, 0xff, 0x07])
            .unwrap_err();
    assert!(error.to_string().contains("padding"), "{error}");
    assert!(crate::api::decode_param_bytes::<Field>(&abi, &blob[..5]).is_err());
    let no_return = string_abi(&["x"], false);
    assert!(crate::api::decode_return_bytes::<Field>(&no_return, b"cd").is_err());
}

#[test]
fn a_parameter_of_no_elements_takes_no_bytes() {
    use mavros_artifacts::Field;

    let mut abi = string_abi(&["x"], false);
    abi.parameters[0].typ = AbiType::Array {
        length: 2,
        typ: Box::new(AbiType::Struct {
            path: "E".to_string(),
            fields: vec![],
        }),
    };
    let empty = InputValueOrdered::Struct(vec![]);
    assert_eq!(
        crate::api::decode_param_bytes::<Field>(&abi, &[]).unwrap(),
        vec![InputValueOrdered::Vec(vec![empty.clone(), empty])]
    );
}

#[test]
fn the_return_is_not_an_input() {
    let abi = string_abi(&[], true);
    assert_eq!(ordered_params::<ark_bn254::Fr>(&abi, &[]).unwrap(), vec![]);
    assert_eq!(declared_return::<ark_bn254::Fr>(&abi, None).unwrap(), None);
    assert_eq!(
        declared_return(&abi, Some(&str_value("ab"))).unwrap(),
        Some(InputValueOrdered::Vec(vec![
            InputValueOrdered::Field(ark_bn254::Fr::from(b'a' as u64)),
            InputValueOrdered::Field(ark_bn254::Fr::from(b'b' as u64)),
        ]))
    );
    let no_return = string_abi(&[], false);
    assert_eq!(
        declared_return::<ark_bn254::Fr>(&no_return, Some(&str_value("ab"))).unwrap(),
        None
    );
}

/// `fn main(x: str<2>, pub_y: pub str<2>) -> pub str<2>`: columns `[1, y, x, ..., ret]`.
fn string_manifest(abi: &Abi) -> WitnessManifest {
    use mavros_artifacts::{ManifestParam, ParamVisibility, ReturnSlot};
    let parameters = abi
        .parameters
        .iter()
        .map(|param| ManifestParam {
            name: param.name.clone(),
            visibility: if param.visibility == AbiVisibility::Public {
                ParamVisibility::Public
            } else {
                ParamVisibility::Private
            },
            width: 2,
            column_start: Some(if param.visibility == AbiVisibility::Public {
                1
            } else {
                3
            }),
        })
        .collect::<Vec<_>>();
    WitnessManifest {
        parameters,
        returns: Some(ReturnSlot {
            column_start: Some(6),
            len: 2,
            visibility: ParamVisibility::Public,
        }),
    }
}

#[test]
fn witness_manifest_check_compares_every_named_column() {
    let abi = string_abi(&["x", "pub_y"], true);
    let params = ordered_params(&abi, &[str_value("ab"), str_value("cd")]).unwrap();
    let declared = declared_return(&abi, Some(&str_value("ef"))).unwrap();
    let manifest = string_manifest(&abi);
    let field = |byte: u8| ark_bn254::Fr::from(byte as u64);
    // witness[0] = 1, then the public y, the private x, an internal column, the return.
    let mut witness = vec![ark_bn254::Fr::from(1u64)];
    witness.extend(b"cdab".iter().map(|b| field(*b)));
    witness.push(ark_bn254::Fr::from(99u64));
    witness.extend(b"ef".iter().map(|b| field(*b)));
    let check = |declared: Option<&InputValueOrdered>, witness: &[ark_bn254::Fr]| {
        check_witness_manifest(&manifest, &params, declared, witness)
    };
    assert_eq!(check(declared.as_ref(), &witness), Ok(()));
    let mut wrong = witness.clone();
    wrong[1] = field(b'z');
    let error = check(declared.as_ref(), &wrong).unwrap_err();
    assert!(
        error.contains("parameter `pub_y`") && error.contains("witness[1]"),
        "{error}"
    );

    let mut wrong = witness.clone();
    wrong.swap(6, 7);
    let error = check(declared.as_ref(), &wrong).unwrap_err();
    assert!(error.contains("declared return"), "{error}");
    assert_eq!(check(None, &wrong), Ok(()));

    let error = check(declared.as_ref(), &witness[..4]).unwrap_err();
    assert!(error.contains("no column"), "{error}");

    let error = check_witness_manifest(&manifest, &params[..1], None, &witness).unwrap_err();
    assert!(error.contains("entry blob"), "{error}");
}
