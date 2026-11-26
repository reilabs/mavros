//! The witness limb width a program is cut at.
use std::iter::successors;

use mavros_artifacts::FieldConfig;
use num_bigint::BigInt;

use crate::compiler::{
    analysis::{flow_analysis::FlowAnalysis, types::TypeInfo, value_range_analysis::field_modulus},
    passes::{
        shared::{
            limbs::{limb_bits_for_modulus, widest_injective_int_bits_for_modulus},
            unsupported::unsupported_on_this_field,
        },
        wide_witness_ints::{
            rewriter::{IdentityShape, lowered_identity},
            types::limb_widths,
        },
    },
    ssa::hlssa::HLSSA,
};

#[cfg(test)]
pub(super) use tests::default_limb_config;

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct LimbConfig {
    field: FieldConfig,
    limb_bits: usize,
    injective_bits: usize,
}

impl LimbConfig {
    /// The limb `ssa` is cut at.
    ///
    /// The widest size up to `max_limb_bits` at which every lowered-to identity is sound.
    pub(super) fn of(
        ssa: &HLSSA,
        type_info: &TypeInfo,
        cfg: &FlowAnalysis,
        max_limb_bits: usize,
    ) -> Self {
        Self::admitting(
            ssa.field(),
            &required_identities(ssa, type_info, cfg),
            max_limb_bits,
        )
    }

    pub(super) fn field(&self) -> FieldConfig {
        self.field
    }

    pub(super) fn limb_bits(&self) -> usize {
        self.limb_bits
    }

    pub(super) fn injective_bits(&self) -> usize {
        self.injective_bits
    }

    /// The widest limb of `field`, up to `max_limb_bits`, at which every identity is sound.
    fn admitting(field: FieldConfig, required: &[Identity], max_limb_bits: usize) -> Self {
        let modulus = field_modulus(field);
        let limb_bits = choose_limb_bits_for_modulus(&modulus, required, max_limb_bits)
            .unwrap_or_else(|(narrowest, (bits, shape))| {
                unsupported_on_this_field(
                    format_args!(
                        "a {bits}-bit {shape:?} identity over witness limbs needs each of its \
                         columns and the carries it is grouped with to fit a field element, at \
                         some limb down to {narrowest} bits"
                    ),
                    field,
                )
            });
        Self {
            field,
            limb_bits,
            injective_bits: widest_injective_int_bits_for_modulus(&modulus),
        }
    }
}

/// An identity the program lowers to, with the width of its operands.
type Identity = (usize, &'static dyn IdentityShape);

/// Every identity the program's products and divisions lower to.
fn required_identities(ssa: &HLSSA, type_info: &TypeInfo, cfg: &FlowAnalysis) -> Vec<Identity> {
    let field = ssa.field();
    let mut required = Vec::new();
    for fid in ssa.get_function_ids() {
        if !type_info.has_function(fid) {
            continue;
        }
        let fti = type_info.get_function(fid);
        let function = ssa.get_function(fid);
        for bid in cfg.get_function_cfg(fid).get_domination_pre_order() {
            for op in function.get_block(bid).get_instructions() {
                required.extend(lowered_identity(op, fti, field));
            }
        }
    }
    required
}

/// The widest candidate limb up to `max_limb_bits` at which every one of the identities holds, or
/// the narrowest candidate with an identity it refuses.
fn choose_limb_bits_for_modulus(
    modulus: &BigInt,
    required: &[Identity],
    max_limb_bits: usize,
) -> Result<usize, (usize, Identity)> {
    let injective = widest_injective_int_bits_for_modulus(modulus);
    let mut refused = None;
    for limb_bits in limb_candidates(modulus, max_limb_bits) {
        let is_unsound =
            |(bits, shape): &&Identity| !shape.admits(&limb_widths(*bits, limb_bits), injective);
        match required.iter().find(is_unsound) {
            None => return Ok(limb_bits),
            Some(identity) => refused = Some((limb_bits, *identity)),
        }
    }
    Err(refused.expect("a field has a limb"))
}

/// Candidate limb widths up to `max_limb_bits`; widest to narrowest.
fn limb_candidates(modulus: &BigInt, max_limb_bits: usize) -> impl Iterator<Item = usize> {
    successors(Some(limb_bits_for_modulus(modulus)), |bits| {
        (bits / 2 >= 2).then_some(bits / 2)
    })
    .filter(move |bits| *bits <= max_limb_bits)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::{
        analysis::types::Types,
        passes::shared::limbs::witness_limb_bits,
        passes::wide_witness_ints::{
            rewriter::EuclideanDivision,
            test_utils::{fields, limbed_widths, operation_with_in, run_pass},
        },
        ssa::hlssa::{BinaryArithOpKind, Type},
    };

    #[test]
    fn chooses_expected_limb_widths() {
        let get_limb_width = |modulus_bits: usize, widest: Option<usize>| {
            let required: Vec<Identity> = widest
                .into_iter()
                .map(|bits| (bits, &EuclideanDivision as &'static dyn IdentityShape))
                .collect();
            let modulus = BigInt::from(1) << (modulus_bits - 1);
            choose_limb_bits_for_modulus(&modulus, &required, limb_bits_for_modulus(&modulus)).ok()
        };

        assert_eq!(get_limb_width(131, None), Some(64));
        assert_eq!(get_limb_width(131, Some(257)), Some(64));
        assert_eq!(get_limb_width(131, Some(320)), Some(32));
        assert_eq!(get_limb_width(131, Some(16384)), Some(32));

        assert_eq!(get_limb_width(130, Some(191)), Some(64));
        assert_eq!(get_limb_width(130, Some(192)), Some(32));

        assert_eq!(get_limb_width(129, Some(128)), Some(32));

        assert_eq!(get_limb_width(3, Some(8)), None);
    }

    /// Every product and division on a configured field keeps the default limb, and lowers at it.
    #[test]
    fn configured_fields_stay_default() {
        for field in fields() {
            let default = default_limb_config(field);

            for bits in limbed_widths(field) {
                for kind in [
                    BinaryArithOpKind::UMul,
                    BinaryArithOpKind::SMul,
                    BinaryArithOpKind::UDiv,
                    BinaryArithOpKind::SDiv,
                ] {
                    let mut ssa =
                        operation_with_in(field, kind, bits, Type::witness_of(Type::int(bits)));
                    let flow = FlowAnalysis::run(&ssa);
                    let types = Types::new().run(&ssa, &flow);
                    assert_eq!(
                        LimbConfig::of(&ssa, &types, &flow, default.limb_bits()),
                        default,
                        "{field:?} int{bits} {kind:?}"
                    );
                    run_pass(&mut ssa);
                }
            }
        }
    }

    pub(in super::super) fn default_limb_config(field: FieldConfig) -> LimbConfig {
        LimbConfig::admitting(field, &[], witness_limb_bits(field))
    }
}
