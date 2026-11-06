use crate::{
    collections::HashMap,
    compiler::{
        passes::{shared::limbs::witness_limb_bits, wide_witness_ints::utils::multi_cell_int_bits},
        ssa::{
            ValueId,
            hlssa::{Type, TypeExpr},
        },
    },
};
use mavros_artifacts::FieldConfig;

/// The widths a `bits`-wide integer splits into, little-endian, with the top limb carrying the
/// remainder rather than a full limb's width.
pub(super) fn limb_widths(bits: usize, limb_bits: usize) -> Vec<usize> {
    let count = bits.div_ceil(limb_bits);
    (0..count)
        .map(|index| {
            let low = index * limb_bits;
            (bits - low).min(limb_bits)
        })
        .collect()
}

/// The types a type expands into: one per limb for a witnessed wide integer, itself otherwise.
pub(super) fn limb_types(ty: &Type, field: FieldConfig) -> Vec<Type> {
    match &ty.expr {
        TypeExpr::WitnessOf(inner) => match &inner.expr {
            TypeExpr::Int(bits) if *bits > multi_cell_int_bits(field) => {
                limb_widths(*bits, witness_limb_bits(field))
                    .into_iter()
                    .map(|width| Type::witness_of(Type::int(width)))
                    .collect()
            }
            _ => vec![ty.clone()],
        },
        // A sequence of wide elements is transposed: one sequence per limb position, holding that
        // limb of every element. Each array's element type is a single limb, at most one host cell
        // wide, so the lookup tape reads it with its existing cell tag and needs no notion of a
        // wide element.
        //
        // The alternative, one sequence of `n * k` limbs, would index it at `i * k + j`, putting
        // arithmetic between the caller's index and the tape's. Here every limb sequence is
        // indexed by the caller's index.
        TypeExpr::Array(inner, count) => element_limb_types(inner, field)
            .into_iter()
            .map(|leaf| leaf.array_of(*count))
            .collect(),
        TypeExpr::Slice(inner) => element_limb_types(inner, field)
            .into_iter()
            .map(Type::slice_of)
            .collect(),
        TypeExpr::Ref(inner) => limb_types(inner, field)
            .into_iter()
            .map(|leaf| leaf.ref_of())
            .collect(),
        _ => vec![ty.clone()],
    }
}

/// A sequence element splits whether or not it is witnessed.
///
/// A sequence is stored as one sequence per limb, so a **pure** wide element splits too: kept
/// whole, a constant wide sequence would hold each entry as one field element, capping its width at
/// the modulus. Reassembling an element costs interpreter work, not constraints.
fn element_limb_types(ty: &Type, field: FieldConfig) -> Vec<Type> {
    match &ty.expr {
        TypeExpr::Int(bits) if *bits > multi_cell_int_bits(field) => {
            limb_widths(*bits, witness_limb_bits(field))
                .into_iter()
                .map(Type::int)
                .collect()
        }
        _ => limb_types(ty, field),
    }
}

/// The width of a witnessed integer that this pass represents as limbs, or [`None`] otherwise.
pub(super) fn wide_witness_width(ty: &Type, field: FieldConfig) -> Option<usize> {
    match &ty.expr {
        TypeExpr::WitnessOf(inner) => match &inner.expr {
            TypeExpr::Int(bits) if *bits > multi_cell_int_bits(field) => Some(*bits),
            _ => None,
        },
        _ => None,
    }
}

/// The limbs of a value; an unmapped value is one limb, the value itself.
pub(super) fn limbs_of(value_map: &HashMap<ValueId, Vec<ValueId>>, value: ValueId) -> Vec<ValueId> {
    value_map
        .get(&value)
        .cloned()
        .unwrap_or_else(|| vec![value])
}

/// The single limb of a value this pass does not represent as limbs.
pub(super) fn single(value_map: &HashMap<ValueId, Vec<ValueId>>, value: ValueId) -> ValueId {
    let limbs = limbs_of(value_map, value);
    assert_eq!(
        limbs.len(),
        1,
        "ICE: wide_witness_ints expected a single-limb value, got {} limbs for v{}",
        limbs.len(),
        value.0
    );
    limbs[0]
}

pub(super) fn flat_limbs(
    value_map: &HashMap<ValueId, Vec<ValueId>>,
    values: &[ValueId],
) -> Vec<ValueId> {
    values
        .iter()
        .flat_map(|value| limbs_of(value_map, *value))
        .collect()
}

/// The limb types of a sequence element, refused if they do not line up with the sequences.
///
/// A constructor's element type and result type are computed separately. Where they disagree, a
/// `zip` would silently build fewer limb sequences than the plan minted values for.
#[track_caller]
pub(super) fn element_types(elem_type: &Type, field: FieldConfig, expected: usize) -> Vec<Type> {
    let types = element_limb_types(elem_type, field);
    assert_eq!(
        types.len(),
        expected,
        "ICE: a sequence of {expected} limbs was built from an element of {} limbs",
        types.len()
    );
    types
}

/// Pair two limb lists, refusing a pair that does not line up.
#[track_caller]
pub(super) fn paired(
    left: Vec<ValueId>,
    right: Vec<ValueId>,
) -> impl Iterator<Item = (ValueId, ValueId)> {
    assert_eq!(
        left.len(),
        right.len(),
        "ICE: a limb-moving opcode met {} limbs on one side and {} on the other",
        left.len(),
        right.len()
    );
    left.into_iter().zip(right)
}
