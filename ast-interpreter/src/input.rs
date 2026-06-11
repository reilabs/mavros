//! Bridge `Prover.toml` inputs into interpreter [`Value`]s.
//!
//! Noir's ABI parser yields an [`InputValue`] tree keyed by parameter name; we map each value
//! onto the corresponding monomorphized parameter [`Type`] (which is precise — concrete widths,
//! structs already lowered to tuples). Integer inputs are decoded from their field encoding via
//! [`field_to_bigint`] (the two's-complement bit pattern in `[0, 2^bits)` for signed types) and
//! then range-checked against the declared width, since the ABI parser only bounds them by the
//! field modulus.

use acvm::{AcirField, FieldElement};

use noirc_abi::{
    Abi, MAIN_RETURN_NAME,
    input_parser::{Format, InputValue},
};
use noirc_frontend::monomorphization::ast::{Program, Type};

use super::error::InterpretError;
use super::value::{IntValue, Value, field_to_bigint};

/// Parse `toml_src` against `abi` and bind each value to `main`'s parameters in order.
pub fn inputs_from_prover_toml(
    program: &Program,
    abi: &Abi,
    toml_src: &str,
) -> Result<Vec<Value>, InterpretError> {
    let map = Format::Toml
        .parse(toml_src, abi)
        .map_err(|e| InterpretError::Internal(format!("failed to parse Prover.toml: {e}")))?;

    let main = program
        .functions
        .first()
        .ok_or_else(|| InterpretError::Internal("program has no functions".to_string()))?;

    let mut inputs = Vec::with_capacity(main.parameters.len());
    for (_, _, name, typ, _) in &main.parameters {
        let input = map
            .get(name)
            .ok_or_else(|| InterpretError::Internal(format!("missing input '{name}'")))?;
        inputs.push(value_from_input(input, typ)?);
    }
    Ok(inputs)
}

/// Decode the expected `main` return value recorded in `Prover.toml` (the `return = ...` field),
/// if present. Noir's test corpus records this as the program's known-correct output, so it is a
/// ground-truth reference the interpreter's result can be checked against — a real differential,
/// not just "did it error".
pub fn expected_return_from_prover_toml(
    program: &Program,
    abi: &Abi,
    toml_src: &str,
) -> Result<Option<Value>, InterpretError> {
    let map = Format::Toml
        .parse(toml_src, abi)
        .map_err(|e| InterpretError::Internal(format!("failed to parse Prover.toml: {e}")))?;
    let Some(input) = map.get(MAIN_RETURN_NAME) else {
        return Ok(None);
    };
    let main = program
        .functions
        .first()
        .ok_or_else(|| InterpretError::Internal("program has no functions".to_string()))?;
    Ok(Some(value_from_input(input, &main.return_type)?))
}

fn value_from_input(input: &InputValue, typ: &Type) -> Result<Value, InterpretError> {
    match (input, typ) {
        (InputValue::Field(field), Type::Field) => Ok(Value::Field(*field)),
        (InputValue::Field(field), Type::Integer(signedness, bits)) => {
            // The ABI TOML parser only checks `value < field_modulus`, not `value < 2^bits`, so a
            // valid-but-oversized input (e.g. "9999" for a u8) reaches here. Reject it rather than
            // silently truncate. A valid integer witness holds the value's two's-complement bit
            // pattern in `[0, 2^bits)`.
            let width = bits.bit_size();
            let raw = field_to_bigint(field);
            if raw.bits() > width as u64 {
                return Err(InterpretError::Type(format!(
                    "integer input does not fit a {width}-bit type"
                )));
            }
            Ok(Value::Int(IntValue::canonical(
                signedness.is_signed(),
                width,
                raw,
            )))
        }
        (InputValue::Field(field), Type::Bool) => Ok(Value::Bool(*field != FieldElement::zero())),
        (InputValue::Vec(elements), Type::Array(length, element_type)) => {
            if elements.len() != *length as usize {
                return Err(InterpretError::Type(format!(
                    "array input has {} elements, type expects {length}",
                    elements.len()
                )));
            }
            let values = elements
                .iter()
                .map(|element| value_from_input(element, element_type))
                .collect::<Result<_, _>>()?;
            Ok(Value::Array(values))
        }
        (InputValue::Vec(elements), Type::Tuple(types)) => {
            if elements.len() != types.len() {
                return Err(InterpretError::Type(format!(
                    "tuple input has {} elements, type expects {}",
                    elements.len(),
                    types.len()
                )));
            }
            let values = types
                .iter()
                .zip(elements)
                .map(|(typ, element)| value_from_input(element, typ))
                .collect::<Result<_, _>>()?;
            Ok(Value::Tuple(values))
        }
        (InputValue::String(s), Type::String(_)) => Ok(Value::Str(s.clone())),
        // Struct inputs need the ABI's field ordering to map onto the lowered tuple; not handled.
        (input, typ) => Err(InterpretError::Unsupported(format!(
            "input value {input:?} for parameter type {typ:?}"
        ))),
    }
}
