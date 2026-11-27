use noir_bigint::{BigInt, BigUint};
use noirc_abi::{
    AbiType, MAIN_RETURN_NAME, Sign,
    bytes::{self, Value},
};
use tracing::warn;

use mavros_artifacts::{FieldConfig, InputValueOrdered, LimbBits, MavrosField, WitnessManifest};

use crate::compiler::passes::shared::limbs::{int_elements, int_limbs};

/// How many entry-blob elements a value of `typ` takes on `field` cut at `limb_bits`: one per
/// scalar and per string byte, and one per witness limb for an integer that one element cannot
/// carry.
///
/// # Panics
///
/// If `typ` holds a `Field`, which crosses no entry point: [`bytes::check_abi`] refuses it, and the
/// driver checks every ABI it generates.
pub(crate) fn abi_type_elements(typ: &AbiType, field: FieldConfig, limb_bits: LimbBits) -> usize {
    let elements = |typ| abi_type_elements(typ, field, limb_bits);
    match typ {
        AbiType::Field => panic!("a `Field` crosses no entry point"),
        AbiType::Integer { width, .. } => int_elements(field, limb_bits, *width as usize),
        AbiType::Boolean => 1,
        AbiType::Array { length, typ } => *length as usize * elements(typ),
        AbiType::Struct { fields, .. } => fields.iter().map(|(_, typ)| elements(typ)).sum(),
        AbiType::Tuple { fields } => fields.iter().map(elements).sum(),
        AbiType::String { length } => *length as usize,
    }
}

/// How many entry-blob elements the parameters of `abi` take on `field` cut at `limb_bits`.
pub(crate) fn abi_param_elements(
    abi: &noirc_abi::Abi,
    field: FieldConfig,
    limb_bits: LimbBits,
) -> usize {
    abi.parameters
        .iter()
        .map(|param| abi_type_elements(&param.typ, field, limb_bits))
        .sum()
}

/// `limb_bits`, checked against `F`.
fn checked_limb_bits<F: MavrosField>(limb_bits: LimbBits) -> Result<LimbBits, String> {
    LimbBits::new(F::CONFIG, limb_bits.bits())
}

/// The parameters `params`, one byte-codec value per parameter of `abi` in declaration order, as
/// elements of `F`, with each wide integer cut at `limb_bits`, the program's
/// [`WitnessManifest::limb_bits`].
pub fn ordered_params<F: MavrosField>(
    abi: &noirc_abi::Abi,
    limb_bits: LimbBits,
    params: &[Value],
) -> Result<Vec<InputValueOrdered<F>>, String> {
    let limb_bits = checked_limb_bits::<F>(limb_bits)?;
    bytes::encode_params(abi, params).map_err(|error| format!("parameters: {error}"))?;
    let ordered_params: Vec<_> = abi
        .parameters
        .iter()
        .zip(params)
        .map(|(param, value)| ordered_value(&param.typ, value, limb_bits))
        .collect();
    debug_assert_eq!(
        crate::vm::interpreter::flattened_param_count(&ordered_params),
        abi_param_elements(abi, F::CONFIG, limb_bits),
        "ordered params flatten to the ABI's element count"
    );
    Ok(ordered_params)
}

/// The expected return, as elements of `F` with each wide integer cut at `limb_bits`, from the
/// byte-codec value the inputs give, if any.
pub fn declared_return<F: MavrosField>(
    abi: &noirc_abi::Abi,
    limb_bits: LimbBits,
    declared: Option<&Value>,
) -> Result<Option<InputValueOrdered<F>>, String> {
    let limb_bits = checked_limb_bits::<F>(limb_bits)?;
    match (&abi.return_type, declared) {
        (Some(return_type), Some(value)) => {
            bytes::encode_return(abi, value).map_err(|error| error.to_string())?;
            Ok(Some(ordered_value(&return_type.abi_type, value, limb_bits)))
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

/// Checks that the named witness columns hold the corresponding input elements, and that the return
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

/// The elements of `value`, a value of `abi_type` that the byte codec accepts: an integer as its
/// two's complement bit pattern at its width, a boolean as `0` or `1`, a string as one element per
/// byte. An integer wider than one element carries is its pattern's limbs of `limb_bits`,
/// little-endian, as [`abi_type_elements`] counts them, so no pattern is reduced.
fn ordered_value<F: MavrosField>(
    abi_type: &AbiType,
    value: &Value,
    limb_bits: LimbBits,
) -> InputValueOrdered<F> {
    match (abi_type, value) {
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
            if let Some(widths) = int_limbs(F::CONFIG, limb_bits, *width as usize) {
                let mut low = 0;
                let limbs = widths
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
                .map(|element| ordered_value(typ, element, limb_bits))
                .collect(),
        ),
        (AbiType::Struct { fields, .. }, Value::Seq(values)) => InputValueOrdered::Struct(
            fields
                .iter()
                .zip(values)
                .map(|((name, typ), value)| (name.clone(), ordered_value(typ, value, limb_bits)))
                .collect(),
        ),
        (AbiType::Tuple { fields }, Value::Seq(values)) => InputValueOrdered::Struct(
            fields
                .iter()
                .zip(values)
                .enumerate()
                .map(|(index, (typ, value))| {
                    (index.to_string(), ordered_value(typ, value, limb_bits))
                })
                .collect(),
        ),
        _ => ice!("the byte codec accepted a value that does not match its ABI type"),
    }
}

#[cfg(test)]
mod tests;
