//! The witness limb width a program is cut at.
use std::collections::BTreeSet;

use mavros_artifacts::FieldConfig;
use num_bigint::{BigInt, BigUint};
use num_traits::One;

use crate::compiler::{
    analysis::{flow_analysis::FlowAnalysis, types::TypeInfo, value_range_analysis::field_modulus},
    pass_manager::{Analysis, AnalysisId, AnalysisStore},
    passes::{
        shared::limbs::{limb_bits_for_modulus, widest_injective_int_bits_for_modulus},
        wide_witness_ints::{
            rewriter::{divided, multiplied},
            types::limb_widths,
        },
    },
    ssa::hlssa::HLSSA,
};

#[cfg(test)]
pub(super) use tests::default_limb_config;

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct LimbConfig {
    pub(super) field: FieldConfig,
    pub(super) limb_bits: usize,
}

impl Analysis for LimbConfig {
    fn dependencies() -> Vec<AnalysisId> {
        vec![TypeInfo::id(), FlowAnalysis::id()]
    }

    fn compute(ssa: &HLSSA, store: &AnalysisStore) -> Self {
        let field = ssa.field();
        let required_widths = schoolbook_widths(ssa, store);
        Self {
            field,
            limb_bits: choose_limb_bits_for_modulus(&field_modulus(field), &required_widths),
        }
    }
}

/// Every width the program's products and divisions take the schoolbook at.
fn schoolbook_widths(ssa: &HLSSA, store: &AnalysisStore) -> BTreeSet<usize> {
    let field = ssa.field();
    let type_info = store.get::<TypeInfo>();
    let cfg = store.get::<FlowAnalysis>();

    let mut required = BTreeSet::new();
    for fid in ssa.get_function_ids() {
        if !type_info.has_function(fid) {
            continue;
        }
        let fti = type_info.get_function(fid);
        let function = ssa.get_function(fid);
        for bid in cfg.get_function_cfg(fid).get_domination_pre_order() {
            for op in function.get_block(bid).get_instructions() {
                if let Some(operands) =
                    multiplied(op, fti, field).or_else(|| divided(op, fti, field))
                {
                    required.insert(operands.bits);
                }
            }
        }
    }
    required
}

/// The widest admissible limb.
fn choose_limb_bits_for_modulus(modulus: &BigInt, required: &BTreeSet<usize>) -> usize {
    let default = limb_bits_for_modulus(modulus);
    let injective = widest_injective_int_bits_for_modulus(modulus);
    let mut limb_bits = default;
    while limb_bits >= 2 {
        let fits = |bits: &usize| holds_a_schoolbook(&limb_widths(*bits, limb_bits), injective);
        if required.iter().all(fits) {
            return limb_bits;
        }
        limb_bits /= 2;
    }
    default
}

/// Whether a a product with an addend, fits at limb `widths``.
fn holds_a_schoolbook(widths: &[usize], injective: usize) -> bool {
    let k = widths.len();
    let full = |width: usize| (BigUint::one() << width) - 1u8;
    let (h, m, top) = (widths[0], full(widths[0]), full(widths[k - 1]));
    let product_carry = (&m * (k - 1)).max(&m * (k - 2) + &top * 2u8);
    let c = (product_carry + 1u8).bits() as usize;
    h + c <= injective
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::{
        analysis::types::Types,
        passes::{
            shared::limbs::witness_limb_bits,
            wide_witness_ints::test_utils::{fields, limbed_widths, operation_with_in},
        },
        ssa::hlssa::{BinaryArithOpKind, Type},
    };

    #[test]
    fn chooses_expected_limb_widths() {
        let get_limb_width = |modulus_bits: usize, widest: Option<usize>| {
            let required = widest.into_iter().collect();
            choose_limb_bits_for_modulus(&(BigInt::from(1) << (modulus_bits - 1)), &required)
        };

        assert_eq!(get_limb_width(131, None), 64);
        assert_eq!(get_limb_width(131, Some(257)), 64);
        assert_eq!(get_limb_width(131, Some(320)), 32);
        assert_eq!(get_limb_width(131, Some(16384)), 32);

        assert_eq!(get_limb_width(130, Some(191)), 64);
        assert_eq!(get_limb_width(130, Some(192)), 32);

        assert_eq!(get_limb_width(129, Some(128)), 32);

        assert_eq!(get_limb_width(3, Some(8)), 1);
    }

    #[test]
    fn configured_fields_stay_default() {
        for field in fields() {
            let default = default_limb_config(field);

            for bits in limbed_widths(field) {
                for kind in [BinaryArithOpKind::UMul, BinaryArithOpKind::UDiv] {
                    let ssa =
                        operation_with_in(field, kind, bits, Type::witness_of(Type::int(bits)));
                    let flow = FlowAnalysis::run(&ssa);
                    let types = Types::new().run(&ssa, &flow);
                    let mut store = AnalysisStore::new();
                    store.insert_with_deps(flow, vec![]);
                    store.insert_with_deps(types, vec![]);
                    let config = LimbConfig::compute(&ssa, &store);
                    assert_eq!(
                        (config.field, config.limb_bits),
                        (default.field, default.limb_bits),
                        "{field:?} int{bits} {kind:?}"
                    );
                    assert!(
                        holds_a_schoolbook(
                            &limb_widths(bits, config.limb_bits),
                            widest_injective_int_bits_for_modulus(&field_modulus(field)),
                        ),
                        "{field:?} int{bits} {kind:?} has an identity at its own limb"
                    );
                }
            }
        }
    }
    pub(in super::super) fn default_limb_config(field: FieldConfig) -> LimbConfig {
        LimbConfig {
            field,
            limb_bits: witness_limb_bits(field),
        }
    }
}
