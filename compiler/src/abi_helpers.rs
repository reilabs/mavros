use noirc_abi::{AbiType, MAIN_RETURN_NAME, input_parser::InputValue};
use std::collections::BTreeMap;

use mavros_artifacts::InputValueOrdered;

/// Converts a BTreeMap of input values (keyed by parameter name) into a Vec of
/// InputValueOrdered, ordered according to the ABI parameter order.
/// Return values (keyed by "return") are appended after the regular parameters.
pub fn ordered_params_from_btreemap(
    abi: &noirc_abi::Abi,
    unordered_params: &BTreeMap<String, InputValue>,
) -> Vec<InputValueOrdered> {
    let mut ordered_params = Vec::new();
    for param in &abi.parameters {
        let param_value = unordered_params
            .get(&param.name)
            .expect("Parameter not found in unordered params");

        ordered_params.push(ordered_param(&param.typ, param_value));
    }

    if let Some(return_type) = &abi.return_type {
        if let Some(return_value) = unordered_params.get(MAIN_RETURN_NAME) {
            ordered_params.push(ordered_param(&return_type.abi_type, return_value));
        }
    }

    ordered_params
}

fn ordered_param(abi_type: &AbiType, value: &InputValue) -> InputValueOrdered {
    match (value, abi_type) {
        (InputValue::Field(elem), _) => InputValueOrdered::Field(elem.into_repr()),

        (InputValue::Vec(vec_elements), AbiType::Array { typ, .. }) => InputValueOrdered::Vec(
            vec_elements
                .iter()
                .map(|elem| ordered_param(typ, elem))
                .collect(),
        ),
        (InputValue::Struct(object), AbiType::Struct { fields, .. }) => InputValueOrdered::Struct(
            fields
                .iter()
                .map(|(field_name, field_type)| {
                    let field_value = object.get(field_name).expect("Field not found in struct");
                    (field_name.clone(), ordered_param(field_type, field_value))
                })
                .collect::<Vec<_>>(),
        ),
        (InputValue::String(string), AbiType::String { length }) => {
            let bytes = string.as_bytes();
            assert_eq!(
                bytes.len(),
                *length as usize,
                "String value length does not match ABI string length"
            );
            InputValueOrdered::Vec(
                bytes
                    .iter()
                    .map(|byte| InputValueOrdered::Field(ark_bn254::Fr::from(*byte as u64)))
                    .collect(),
            )
        }
        (InputValue::String(_string), _) => {
            panic!("String input did not match ABI string type");
        }

        (InputValue::Vec(vec_elements), AbiType::Tuple { fields }) => {
            assert_eq!(
                vec_elements.len(),
                fields.len(),
                "Tuple value length does not match ABI tuple field count"
            );
            InputValueOrdered::Struct(
                fields
                    .iter()
                    .zip(vec_elements.iter())
                    .enumerate()
                    .map(|(idx, (field_type, field_value))| {
                        (idx.to_string(), ordered_param(field_type, field_value))
                    })
                    .collect(),
            )
        }
        _ => unreachable!("value should have already been checked to match abi type"),
    }
}
