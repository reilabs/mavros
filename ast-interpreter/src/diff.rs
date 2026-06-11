//! Cross-field differential: compare the interpreter's result for the *same* program compiled
//! under two different fields (bn254 vs Goldilocks).
//!
//! The field is selected at compile time, so the two runs are separate processes; each dumps its
//! per-program outcomes to JSON (see the `dump_corpus_outcomes` test) and a third step diffs the
//! two files. The rule: **integer / bool / struct values are field-independent and must match
//! exactly; `Field` values are field-specific and may differ.** A program that computes the same
//! integers under both fields confirms the Goldilocks frontend did not corrupt the monomorphized
//! AST's integer semantics. An integer that differs is exactly the corruption we are hunting.
//!
//! NOTE: the Goldilocks arm cannot run end-to-end yet — the auto-injected bn254 stdlib blocks
//! Goldilocks compilation (CRY-9). This module is the comparison machinery, validated today by a
//! bn254-vs-bn254 self-diff and unit tests; it runs against a real Goldilocks dump once the stdlib
//! port lands. The comparison logic itself needs no Goldilocks compile and is tested now.

use num_bigint::BigInt;
use serde::{Deserialize, Serialize};

use super::value::Value;

/// A field-independent encoding of an interpreter [`Value`] for cross-field comparison.
///
/// `Field` is intentionally opaque (carries no value): field elements are field-specific, so a
/// difference there is expected, not a divergence. Integers carry their exact value as a `BigInt`
/// so the comparison is precise regardless of width.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum DiffValue {
    Field,
    Int { signed: bool, bits: u8, value: BigInt },
    Bool(bool),
    Unit,
    Str(String),
    Array(Vec<DiffValue>),
    Tuple(Vec<DiffValue>),
    Function,
}

impl DiffValue {
    pub fn from_value(value: &Value) -> DiffValue {
        match value {
            Value::Field(_) => DiffValue::Field,
            Value::Int(int) => DiffValue::Int {
                signed: int.signed,
                bits: int.bits,
                value: int.value.clone(),
            },
            Value::Bool(b) => DiffValue::Bool(*b),
            Value::Unit => DiffValue::Unit,
            Value::Str(s) => DiffValue::Str(s.clone()),
            Value::Array(elements) => {
                DiffValue::Array(elements.iter().map(DiffValue::from_value).collect())
            }
            Value::Tuple(elements) => {
                DiffValue::Tuple(elements.iter().map(DiffValue::from_value).collect())
            }
            Value::Function(_) => DiffValue::Function,
        }
    }
}

/// The outcome of interpreting one program under one field, ready to serialize and diff.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum DiffOutcome {
    /// Interpreted to a value (the program's return).
    Returned(DiffValue),
    /// Did not reach interpretation, or interpretation errored (compile limit, unsupported
    /// construct, assertion failure...). The string is the reason, for human triage.
    Errored(String),
}

/// Whether two cross-field outcomes for the same program are equivalent.
///
/// `Ok(())` means equivalent. `Err(reason)` describes the first divergence found — which, for two
/// `Returned` outcomes, is an integer/structure mismatch (the corruption signal). Two `Errored`
/// outcomes are treated as equivalent (both blocked, e.g. an unsupported construct on both sides);
/// a `Returned` vs `Errored` split is a divergence worth surfacing.
pub fn outcomes_equivalent(a: &DiffOutcome, b: &DiffOutcome) -> Result<(), String> {
    match (a, b) {
        (DiffOutcome::Returned(x), DiffOutcome::Returned(y)) => values_equivalent(x, y),
        (DiffOutcome::Errored(_), DiffOutcome::Errored(_)) => Ok(()),
        (DiffOutcome::Returned(_), DiffOutcome::Errored(e)) => Err(format!(
            "one field returned a value, the other errored: {e}"
        )),
        (DiffOutcome::Errored(e), DiffOutcome::Returned(_)) => Err(format!(
            "one field errored ({e}), the other returned a value"
        )),
    }
}

/// Whether two values are cross-field equivalent: integers/bools/structure must match exactly,
/// `Field` values may differ.
pub fn values_equivalent(a: &DiffValue, b: &DiffValue) -> Result<(), String> {
    match (a, b) {
        // Field values are field-specific; any difference there is expected.
        (DiffValue::Field, DiffValue::Field) => Ok(()),
        (DiffValue::Function, DiffValue::Function) => Ok(()),
        (DiffValue::Unit, DiffValue::Unit) => Ok(()),
        (DiffValue::Bool(x), DiffValue::Bool(y)) => {
            if x == y {
                Ok(())
            } else {
                Err(format!("bool differs: {x} vs {y}"))
            }
        }
        (DiffValue::Str(x), DiffValue::Str(y)) => {
            if x == y {
                Ok(())
            } else {
                Err(format!("string differs: {x:?} vs {y:?}"))
            }
        }
        (
            DiffValue::Int {
                signed: s1,
                bits: b1,
                value: v1,
            },
            DiffValue::Int {
                signed: s2,
                bits: b2,
                value: v2,
            },
        ) => {
            if s1 == s2 && b1 == b2 && v1 == v2 {
                Ok(())
            } else {
                Err(format!(
                    "integer differs: {v1} (s={s1},{b1}b) vs {v2} (s={s2},{b2}b)"
                ))
            }
        }
        (DiffValue::Array(xs), DiffValue::Array(ys)) => elementwise(xs, ys, "array"),
        (DiffValue::Tuple(xs), DiffValue::Tuple(ys)) => elementwise(xs, ys, "tuple"),
        (x, y) => Err(format!("shape differs: {x:?} vs {y:?}")),
    }
}

fn elementwise(xs: &[DiffValue], ys: &[DiffValue], kind: &str) -> Result<(), String> {
    if xs.len() != ys.len() {
        return Err(format!(
            "{kind} length differs: {} vs {}",
            xs.len(),
            ys.len()
        ));
    }
    for (i, (x, y)) in xs.iter().zip(ys).enumerate() {
        values_equivalent(x, y).map_err(|e| format!("{kind}[{i}]: {e}"))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn int(value: &str) -> DiffValue {
        DiffValue::Int {
            signed: false,
            bits: 64,
            value: value.parse().unwrap(),
        }
    }

    #[test]
    fn equal_integers_are_equivalent() {
        assert!(values_equivalent(&int("42"), &int("42")).is_ok());
    }

    #[test]
    fn differing_integers_diverge() {
        // This is the corruption signal: the same program produced different integers per field.
        assert!(values_equivalent(&int("42"), &int("43")).is_err());
    }

    #[test]
    fn field_values_may_differ() {
        // Field elements are field-specific; a difference there is not a divergence.
        assert!(values_equivalent(&DiffValue::Field, &DiffValue::Field).is_ok());
    }

    #[test]
    fn nested_integer_difference_is_found() {
        let a = DiffValue::Tuple(vec![
            DiffValue::Field,
            DiffValue::Array(vec![int("1"), int("2")]),
        ]);
        let b = DiffValue::Tuple(vec![
            DiffValue::Field,
            DiffValue::Array(vec![int("1"), int("9")]),
        ]);
        let err = values_equivalent(&a, &b).unwrap_err();
        assert!(
            err.contains("tuple[1]"),
            "path should point at the mismatch: {err}"
        );
    }

    #[test]
    fn shape_mismatch_diverges() {
        assert!(values_equivalent(&int("0"), &DiffValue::Bool(false)).is_err());
    }

    #[test]
    fn returned_vs_errored_diverges() {
        let a = DiffOutcome::Returned(int("1"));
        let b = DiffOutcome::Errored("unsupported".to_string());
        assert!(outcomes_equivalent(&a, &b).is_err());
        // Two errors are equivalent (both blocked the same way).
        assert!(outcomes_equivalent(&b.clone(), &b).is_ok());
    }
}
