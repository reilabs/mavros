use noir_bigint::{BigInt, BigUint};
use noirc_abi::{
    AbiType, MAIN_RETURN_NAME, Sign,
    bytes::{self, Value},
};
use tracing::warn;

use mavros_artifacts::{FieldConfig, InputValueOrdered, MavrosField, WitnessManifest};

use crate::compiler::passes::wide_witness_ints::{int_limb_widths, multi_cell_int_bits};

/// How many entry-blob elements a value of `typ` takes on `field`: one per scalar and per string
/// byte, and one per witness limb for an integer that one element cannot carry.
///
/// # Panics
///
/// If `typ` holds a `Field`, which crosses no entry point: [`bytes::check_abi`] refuses it, and the
/// driver checks every ABI it generates.
pub(crate) fn abi_type_elements(typ: &AbiType, field: FieldConfig) -> usize {
    match typ {
        AbiType::Field => panic!("a `Field` crosses no entry point"),
        AbiType::Integer { width, .. } if *width as usize > multi_cell_int_bits(field) => {
            int_limb_widths(*width as usize, field).len()
        }
        AbiType::Integer { .. } | AbiType::Boolean => 1,
        AbiType::Array { length, typ } => *length as usize * abi_type_elements(typ, field),
        AbiType::Struct { fields, .. } => fields
            .iter()
            .map(|(_, typ)| abi_type_elements(typ, field))
            .sum(),
        AbiType::Tuple { fields } => fields.iter().map(|typ| abi_type_elements(typ, field)).sum(),
        AbiType::String { length } => *length as usize,
    }
}

/// How many entry-blob elements the parameters of `abi` take on `field`.
pub(crate) fn abi_param_elements(abi: &noirc_abi::Abi, field: FieldConfig) -> usize {
    abi.parameters
        .iter()
        .map(|param| abi_type_elements(&param.typ, field))
        .sum()
}

/// The parameters `params`, one byte-codec value per parameter of `abi` in declaration order, as
/// elements of `F`.
pub fn ordered_params<F: MavrosField>(
    abi: &noirc_abi::Abi,
    params: &[Value],
) -> Result<Vec<InputValueOrdered<F>>, String> {
    bytes::encode_params(abi, params).map_err(|error| format!("parameters: {error}"))?;
    let ordered_params = abi
        .parameters
        .iter()
        .zip(params)
        .map(|(param, value)| ordered_value(&param.typ, value, &param.name))
        .collect::<Result<Vec<_>, _>>()?;

    let flattened = crate::vm::interpreter::flattened_param_count(&ordered_params);
    if flattened != abi_param_elements(abi, F::CONFIG) {
        return Err("ordered params do not flatten to the ABI's element count".to_string());
    }
    Ok(ordered_params)
}

/// The expected return, as elements of `F`, from the byte-codec value the inputs give, if any.
pub fn declared_return<F: MavrosField>(
    abi: &noirc_abi::Abi,
    declared: Option<&Value>,
) -> Result<Option<InputValueOrdered<F>>, String> {
    match (&abi.return_type, declared) {
        (Some(return_type), Some(value)) => {
            bytes::encode_return(abi, value).map_err(|error| error.to_string())?;
            ordered_value(&return_type.abi_type, value, MAIN_RETURN_NAME).map(Some)
        }
        (Some(_), None) => {
            warn!(message = %format!(
                "the ABI declares a return value but the inputs do not supply \
                 `{MAIN_RETURN_NAME}`; the computed return is not checked"
            ));
            Ok(None)
        }
        // The byte codec drops `return` for an ABI without one; only a hand-built value gets here.
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

/// Checks that the named witness columns hold the corresponding input fields, and that the return
/// columns hold `declared_return` when one was given.
///
/// `manifest` must already be validated against the R1CS.
pub fn check_witness_manifest<F: MavrosField>(
    manifest: &WitnessManifest,
    ordered_params: &[InputValueOrdered<F>],
    declared_return: Option<&InputValueOrdered<F>>,
    wit_pre_comm: &[F],
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
pub fn check_declared_return<F: MavrosField>(
    manifest: &WitnessManifest,
    declared_return: &InputValueOrdered<F>,
    wit_pre_comm: &[F],
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

fn check_columns<F: MavrosField>(
    what: &str,
    source: &str,
    expected: &[F],
    column_start: usize,
    wit_pre_comm: &[F],
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

/// The elements of `value`, a value of `abi_type` that the byte codec accepts, at `path` (named as
/// the byte codec names it): an integer as its two's complement bit pattern at its width, a boolean
/// as `0` or `1`, a string as one element per byte. An integer wider than one element carries is
/// its pattern's limbs, little-endian, as [`abi_type_elements`] counts them, so no pattern is
/// reduced.
fn ordered_value<F: MavrosField>(
    abi_type: &AbiType,
    value: &Value,
    path: &str,
) -> Result<InputValueOrdered<F>, String> {
    Ok(match (abi_type, value) {
        (AbiType::Integer { sign, width }, Value::Int(value)) => {
            let pattern = match (sign, value.sign()) {
                (Sign::Signed, noir_bigint::Sign::Minus) => (BigInt::from(1u8) << *width) + value,
                _ => value.clone(),
            };
            let pattern = pattern
                .to_biguint()
                .expect("the byte codec holds an integer to its width");
            // No element reaches `2^multi_cell_int_bits`, so every one is below the modulus.
            let element = |pattern: &BigUint| {
                InputValueOrdered::Field(
                    F::try_from_canonical_limbs(&pattern.to_u64_digits())
                        .expect("an injective width is below the modulus"),
                )
            };
            let width = *width as usize;
            if width > multi_cell_int_bits(F::CONFIG) {
                let mut low = 0;
                let limbs = int_limb_widths(width, F::CONFIG)
                    .into_iter()
                    .map(|limb_width| {
                        let limb = (&pattern >> low) & ((BigUint::from(1u8) << limb_width) - 1u8);
                        low += limb_width;
                        element(&limb)
                    })
                    .collect();
                InputValueOrdered::Vec(limbs)
            } else {
                element(&pattern)
            }
        }
        (AbiType::Boolean, Value::Bool(value)) => InputValueOrdered::Field(F::from(*value as u64)),
        (AbiType::String { .. }, Value::Str(string)) => InputValueOrdered::Vec(
            string
                .iter()
                .map(|byte| InputValueOrdered::Field(F::from(*byte as u64)))
                .collect(),
        ),
        (AbiType::Array { typ, .. }, Value::Seq(elements)) => InputValueOrdered::Vec(
            elements
                .iter()
                .enumerate()
                .map(|(index, element)| ordered_value(typ, element, &format!("{path}[{index}]")))
                .collect::<Result<_, _>>()?,
        ),
        (AbiType::Struct { fields, .. }, Value::Seq(values)) => InputValueOrdered::Struct(
            fields
                .iter()
                .zip(values)
                .map(|((name, typ), value)| {
                    Ok((
                        name.clone(),
                        ordered_value(typ, value, &format!("{path}.{name}"))?,
                    ))
                })
                .collect::<Result<_, String>>()?,
        ),
        (AbiType::Tuple { fields }, Value::Seq(values)) => InputValueOrdered::Struct(
            fields
                .iter()
                .zip(values)
                .enumerate()
                .map(|(index, (typ, value))| {
                    let ordered = ordered_value(typ, value, &format!("{path}.{index}"))?;
                    Ok((index.to_string(), ordered))
                })
                .collect::<Result<_, String>>()?,
        ),
        _ => return Err(format!("`{path}`: the value does not match its ABI type")),
    })
}

#[cfg(test)]
mod tests;
