//! The witness limb width a program is cut at.
use mavros_artifacts::{FieldConfig, LimbBits};

use crate::compiler::{
    analysis::{
        flow_analysis::FlowAnalysis,
        types::{TypeInfo, Types},
        value_range_analysis::field_modulus,
    },
    passes::{
        shared::{
            limbs::{widest_injective_int_bits_for_modulus, witness_limb_bits},
            unreachable::unreachable_on_this_field,
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
    pub(super) fn of(ssa: &HLSSA) -> Self {
        Self::cut_at(ssa.field(), ssa.limb_bits())
    }

    fn cut_at(field: FieldConfig, limb_bits: LimbBits) -> Self {
        Self {
            field,
            limb_bits: limb_bits.bits(),
            injective_bits: widest_injective_int_bits_for_modulus(&field_modulus(field)),
        }
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
}

/// The widest limb of `field`, up to `max_limb_bits`, at which every identity is sound.
fn admitted_limb_bits(field: FieldConfig, required: &[Identity], max_limb_bits: usize) -> LimbBits {
    let candidates = LimbBits::candidates(field).filter(|limb| limb.bits() <= max_limb_bits);
    let injective = widest_injective_int_bits_for_modulus(&field_modulus(field));
    choose_limb_bits(candidates, injective, required).unwrap_or_else(
        |(narrowest, (bits, shape))| {
            unreachable_on_this_field(
                format_args!(
                    "a {bits}-bit {shape:?} identity over witness limbs needs each of its columns \
                     and the carries it is grouped with to fit a field element, at some limb \
                     down to {} bits",
                    narrowest.bits()
                ),
                field,
            )
        },
    )
}

/// The limb `ssa` is cut at: the widest, up to `max_limb_bits` or else the field's default, at
/// which every product and division in it could lower to a sound identity.
pub(crate) fn scanned_limb_bits(ssa: &HLSSA, max_limb_bits: Option<usize>) -> LimbBits {
    let field = ssa.field();
    let flow = FlowAnalysis::run(ssa);
    let types = Types::new().run(ssa, &flow);
    admitted_limb_bits(
        field,
        &required_identities(ssa, &types, &flow),
        max_limb_bits.unwrap_or_else(|| witness_limb_bits(field)),
    )
}

/// An identity the program lowers to, with the width of its operands.
type Identity = (usize, &'static dyn IdentityShape);

/// Every identity the program's products and divisions could lower to.
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

/// The first of `candidates`, widest first, at which every one of `required` is sound for an
/// element `injective` bits wide; or the narrowest candidate with an identity it refuses.

fn choose_limb_bits<L: Copy + Into<usize>>(
    candidates: impl IntoIterator<Item = L>,
    injective: usize,
    required: &[Identity],
) -> Result<L, (L, Identity)> {
    let mut refused = None;
    for limb in candidates {
        let limb_bits = limb.into();
        let is_unsound =
            |(bits, shape): &&Identity| !shape.admits(&limb_widths(*bits, limb_bits), injective);
        match required.iter().find(is_unsound) {
            None => return Ok(limb),
            Some(identity) => refused = Some((limb, *identity)),
        }
    }
    Err(refused.expect("a field has a limb"))
}

#[cfg(test)]
mod tests {
    use mavros_artifacts::field::limb_bits::{candidates_for_widest, widest_for_modulus_bits};
    use num_bigint::BigInt;

    use super::*;
    use crate::compiler::{
        passes::wide_witness_ints::{
            rewriter::EuclideanDivision,
            test_utils::{fields, limbed_widths, operation_with, run_pass},
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
            let candidates = candidates_for_widest(widest_for_modulus_bits(modulus_bits)?);
            let injective = widest_injective_int_bits_for_modulus(&modulus);
            choose_limb_bits(candidates, injective, &required).ok()
        };

        assert_eq!(get_limb_width(131, None), Some(64));
        assert_eq!(get_limb_width(131, Some(257)), Some(64));
        assert_eq!(get_limb_width(131, Some(320)), Some(32));
        assert_eq!(get_limb_width(131, Some(16384)), Some(32));

        assert_eq!(get_limb_width(130, Some(191)), Some(64));
        assert_eq!(get_limb_width(130, Some(192)), Some(32));

        assert_eq!(get_limb_width(129, Some(128)), Some(32));

        assert_eq!(get_limb_width(5, Some(8)), None);
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
                        operation_with(field, kind, bits, Type::witness_of(Type::int(bits)));
                    let limb_bits = scanned_limb_bits(&ssa, None);
                    assert_eq!(
                        limb_bits.bits(),
                        default.limb_bits(),
                        "{field:?} int{bits} {kind:?}"
                    );
                    ssa.set_limb_bits(limb_bits);
                    run_pass(&mut ssa);
                }
            }
        }
    }

    pub(in super::super) fn default_limb_config(field: FieldConfig) -> LimbConfig {
        LimbConfig::cut_at(
            field,
            admitted_limb_bits(field, &[], witness_limb_bits(field)),
        )
    }
}
