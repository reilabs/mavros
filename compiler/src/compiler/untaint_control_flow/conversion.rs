//! Witness representation conversions shared by branch and merge lowering.

use crate::compiler::ssa::{
    ValueId,
    hlssa::{
        CastTarget, Type, TypeExpr,
        builder::{HLEmitter, HLInstrBuilder},
    },
};

/// Convert a value from source_type to target_type. Scalar witness injections
/// become a single `WitnessOf` cast; arrays and slices become one composite
/// `Map` cast, lowered to a loop late by `LowerMapCasts` (and erased entirely
/// in the witgen pipeline by `StripWitnessOf`). Conversions are pure — the
/// result is a fresh value — so they are safe to execute unconditionally,
/// including in guarded (tainted) regions.
pub(super) fn emit_value_conversion(
    value: ValueId,
    source_type: &Type,
    target_type: &Type,
    builder: &mut impl HLEmitter,
) -> ValueId {
    match CastTarget::conversion(source_type, target_type) {
        None => value,
        Some(target) => builder.cast_to(target, value),
    }
}

/// Recursively strip WitnessOf from a value (for unconstrained call args).
pub(super) fn emit_strip_witness(
    value: ValueId,
    source_type: &Type,
    target_type: &Type,
    builder: &mut HLInstrBuilder<'_>,
) -> ValueId {
    if source_type == target_type {
        return value;
    }
    // Toplevel WitnessOf(X) → X: emit ValueOf, then keep stripping inside.
    if let TypeExpr::WitnessOf(inner) = &source_type.expr {
        let unwrapped = builder.value_of(value);
        return emit_strip_witness(unwrapped, inner, target_type, builder);
    }
    match CastTarget::strip_conversion(source_type, target_type) {
        None => value,
        Some(target) => builder.cast_to(target, value),
    }
}
