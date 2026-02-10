use noirc_abi::{AbiType, MAIN_RETURN_NAME, input_parser::InputValue};
use std::collections::BTreeMap;

use crate::vm::interpreter::InputValueOrdered;

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

        (InputValue::Vec(vec_elements), AbiType::Array { typ, .. }) => {
            InputValueOrdered::Vec(
                vec_elements
                    .iter()
                    .map(|elem| ordered_param(typ, elem))
                    .collect(),
            )
        }
        (InputValue::Struct(object), AbiType::Struct { fields, .. }) => {
            InputValueOrdered::Struct(
                fields
                    .iter()
                    .map(|(field_name, field_type)| {
                        let field_value = object.get(field_name).expect("Field not found in struct");
                        (field_name.clone(), ordered_param(field_type, field_value))
                    })
                    .collect::<Vec<_>>(),
            )
        }
        (InputValue::String(_string), _) => {
            panic!("Strings are not supported in ordered params");
        }

        (InputValue::Vec(_vec_elements), AbiType::Tuple { fields: _fields }) => {
            panic!("Tuples are not supported in ordered params");
        }
        _ => unreachable!("value should have already been checked to match abi type"),
    }
}
