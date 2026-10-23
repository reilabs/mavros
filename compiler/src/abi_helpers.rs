use noirc_abi::{AbiType, MAIN_RETURN_NAME, input_parser::InputValue};
use std::collections::BTreeMap;
use tracing::warn;

use mavros_artifacts::{InputValueOrdered, WitnessManifest};

pub fn ordered_params_from_btreemap(
    abi: &noirc_abi::Abi,
    unordered_params: &BTreeMap<String, InputValue>,
) -> Result<Vec<InputValueOrdered>, String> {
    let mut ordered_params = Vec::new();
    for param in &abi.parameters {
        let param_value = unordered_params
            .get(&param.name)
            .ok_or_else(|| format!("inputs do not supply parameter `{}`", param.name))?;

        ordered_params.push(ordered_param(&param.typ, param_value)?);
    }

    let flattened = crate::vm::interpreter::flattened_param_count(&ordered_params);
    if flattened != abi.field_count() as usize {
        return Err("ordered params do not flatten to the ABI's field count".to_string());
    }
    Ok(ordered_params)
}

pub fn declared_return_from_btreemap(
    abi: &noirc_abi::Abi,
    unordered_params: &BTreeMap<String, InputValue>,
) -> Result<Option<InputValueOrdered>, String> {
    match (&abi.return_type, unordered_params.get(MAIN_RETURN_NAME)) {
        (Some(return_type), Some(value)) => Ok(Some(ordered_param(&return_type.abi_type, value)?)),
        (Some(_), None) => {
            warn!(message = %format!(
                "the ABI declares a return value but the inputs do not supply \
                 `{MAIN_RETURN_NAME}`; the computed return is not checked"
            ));
            Ok(None)
        }
        // Noir's parser drops `return` for an ABI without one; only a hand-built map gets here.
        (None, Some(_)) => {
            warn!(message = %format!(
                "the inputs supply `{MAIN_RETURN_NAME}` but the program has no return; it is \
                 ignored"
            ));
            Ok(None)
        }
        (None, None) => Ok(None),
    }
}

/// Checks that the named witness columns hold the corresponding input fields, and that the
/// return columns hold `declared_return` when one was given.
/// `manifest` must already be validated against the R1CS.
pub fn check_witness_manifest(
    manifest: &WitnessManifest,
    ordered_params: &[InputValueOrdered],
    declared_return: Option<&InputValueOrdered>,
    wit_pre_comm: &[mavros_artifacts::Field],
) -> Result<(), String> {
    let blob = crate::vm::interpreter::flatten_param_vec(ordered_params);
    let expected = manifest.entry_blob_field_count();
    if blob.len() != expected {
        return Err(format!(
            "inputs flatten to {} fields but the manifest expects an entry blob of {expected}",
            blob.len()
        ));
    }
    for (param, slot) in manifest.blob_slots() {
        if let Some(start) = param.column_start {
            let fields = blob.get(slot.clone()).ok_or_else(|| {
                format!(
                    "parameter `{}`: blob slots {}.. are past the end of the inputs",
                    param.name, slot.start
                )
            })?;
            check_columns(
                &format!("parameter `{}`", param.name),
                "the entry blob",
                fields,
                start,
                wit_pre_comm,
            )?;
        }
    }
    if let Some(declared) = declared_return {
        check_declared_return(manifest, declared, wit_pre_comm)?;
    }
    Ok(())
}

/// `manifest` must already be validated against the R1CS.
pub fn check_declared_return(
    manifest: &WitnessManifest,
    declared_return: &InputValueOrdered,
    wit_pre_comm: &[mavros_artifacts::Field],
) -> Result<(), String> {
    let Some(ret) = manifest.returns else {
        return Err(format!(
            "declared return: the inputs declare `{MAIN_RETURN_NAME}` but the program has no return"
        ));
    };
    let declared = crate::vm::interpreter::flatten_param_vec(std::slice::from_ref(declared_return));
    if declared.len() != ret.len {
        return Err(format!(
            "declared return: `{MAIN_RETURN_NAME}` flattens to {} fields but the return has {}",
            declared.len(),
            ret.len
        ));
    }
    match ret.column_start {
        Some(start) => check_columns(
            "declared return",
            &format!("`{MAIN_RETURN_NAME}`"),
            &declared,
            start,
            wit_pre_comm,
        ),
        None => Ok(()),
    }
}

fn check_columns(
    what: &str,
    source: &str,
    expected: &[mavros_artifacts::Field],
    column_start: usize,
    wit_pre_comm: &[mavros_artifacts::Field],
) -> Result<(), String> {
    for (j, expected) in expected.iter().enumerate() {
        let column = column_start + j;
        match wit_pre_comm.get(column) {
            Some(actual) if actual == expected => {}
            Some(actual) => {
                return Err(format!(
                    "{what}: witness[{column}] holds {actual} but {source} is {expected}"
                ));
            }
            None => {
                return Err(format!(
                    "{what}: witness has no column {column} (pre-commitment witness has {} \
                     values)",
                    wit_pre_comm.len()
                ));
            }
        }
    }
    Ok(())
}

fn ordered_param(abi_type: &AbiType, value: &InputValue) -> Result<InputValueOrdered, String> {
    Ok(match (value, abi_type) {
        (InputValue::Field(elem), _) => InputValueOrdered::Field(elem.into_repr()),
        (InputValue::Vec(vec_elements), AbiType::Array { typ, length }) => {
            if vec_elements.len() != *length as usize {
                return Err("Array value length does not match ABI array length".to_string());
            }
            InputValueOrdered::Vec(
                vec_elements
                    .iter()
                    .map(|elem| ordered_param(typ, elem))
                    .collect::<Result<_, _>>()?,
            )
        }
        (InputValue::Struct(object), AbiType::Struct { fields, .. }) => InputValueOrdered::Struct(
            fields
                .iter()
                .map(|(field_name, field_type)| {
                    let field_value = object
                        .get(field_name)
                        .ok_or_else(|| "Field not found in struct".to_string())?;
                    Ok((field_name.clone(), ordered_param(field_type, field_value)?))
                })
                .collect::<Result<Vec<_>, String>>()?,
        ),
        (InputValue::String(string), AbiType::String { length }) => {
            let bytes = string.as_bytes();
            if bytes.len() != *length as usize {
                return Err("String value length does not match ABI string length".to_string());
            }
            InputValueOrdered::Vec(
                bytes
                    .iter()
                    // FIELD-ASSUMPTION: L1-direct-ref (7 sites)
                    .map(|byte| InputValueOrdered::Field(ark_bn254::Fr::from(*byte as u64)))
                    .collect(),
            )
        }
        (InputValue::String(_string), _) => {
            return Err("String input did not match ABI string type".to_string());
        }
        (InputValue::Vec(vec_elements), AbiType::Tuple { fields }) => {
            if vec_elements.len() != fields.len() {
                return Err("Tuple value length does not match ABI tuple field count".to_string());
            }
            InputValueOrdered::Struct(
                fields
                    .iter()
                    .zip(vec_elements.iter())
                    .enumerate()
                    .map(|(idx, (field_type, field_value))| {
                        Ok((idx.to_string(), ordered_param(field_type, field_value)?))
                    })
                    .collect::<Result<Vec<_>, String>>()?,
            )
        }
        _ => {
            return Err("value should have already been checked to match abi type".to_string());
        }
    })
}

// TESTS
// ================================================================================================

#[cfg(test)]
mod tests {
    use noirc_abi::{Abi, AbiParameter, AbiReturnType, AbiType, AbiVisibility};

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

    #[test]
    fn missing_parameter_is_an_error() {
        let abi = string_abi(&["x"], false);
        let error = ordered_params_from_btreemap(&abi, &BTreeMap::new()).unwrap_err();
        assert!(
            error.contains("`x`"),
            "error does not name the parameter: {error}"
        );
    }

    #[test]
    fn the_return_is_not_an_input() {
        let abi = string_abi(&[], true);
        let inputs = BTreeMap::from([(
            MAIN_RETURN_NAME.to_string(),
            InputValue::String("ab".to_string()),
        )]);
        assert_eq!(ordered_params_from_btreemap(&abi, &inputs).unwrap(), vec![]);
        assert_eq!(
            declared_return_from_btreemap(&abi, &BTreeMap::new()).unwrap(),
            None
        );
        assert_eq!(
            declared_return_from_btreemap(&abi, &inputs).unwrap(),
            Some(InputValueOrdered::Vec(vec![
                InputValueOrdered::Field(ark_bn254::Fr::from(b'a' as u64)),
                InputValueOrdered::Field(ark_bn254::Fr::from(b'b' as u64)),
            ]))
        );
        let no_return = string_abi(&[], false);
        assert_eq!(
            declared_return_from_btreemap(&no_return, &inputs).unwrap(),
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
        let inputs = BTreeMap::from([
            ("x".to_string(), InputValue::String("ab".to_string())),
            ("pub_y".to_string(), InputValue::String("cd".to_string())),
            (
                MAIN_RETURN_NAME.to_string(),
                InputValue::String("ef".to_string()),
            ),
        ]);
        let params = ordered_params_from_btreemap(&abi, &inputs).unwrap();
        let declared = declared_return_from_btreemap(&abi, &inputs).unwrap();
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
}
