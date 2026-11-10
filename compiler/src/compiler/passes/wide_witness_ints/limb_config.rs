//! The witness limb width a program is cut at.
use mavros_artifacts::FieldConfig;
use num_bigint::{BigInt, BigUint};
use num_traits::One;

use crate::compiler::{
    analysis::{flow_analysis::FlowAnalysis, types::TypeInfo, value_range_analysis::field_modulus},
    pass_manager::{Analysis, AnalysisId, AnalysisStore},
    passes::{
        shared::limbs::{limb_bits_for_modulus, widest_injective_int_bits_for_modulus},
        wide_witness_ints::rewriter::{divided, multiplied},
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
        let type_info = store.get::<TypeInfo>();
        let cfg = store.get::<FlowAnalysis>();

        let mut widest = None;
        for fid in ssa.get_function_ids() {
            if !type_info.has_function(fid) {
                continue;
            }
            let fti = type_info.get_function(fid);
            let function = ssa.get_function(fid);
            for bid in cfg.get_function_cfg(fid).get_domination_pre_order() {
                for op in function.get_block(bid).get_instructions() {
                    if let Some(operands) = multiplied(op, fti, field) {
                        widest = widest.max(Some(operands.bits));
                    } else if let Some(operands) = divided(op, fti, field) {
                        widest = widest.max(Some(operands.bits));
                    }
                }
            }
        }

        Self {
            field,
            limb_bits: choose_limb_bits(field, widest),
        }
    }
}

pub(super) fn choose_limb_bits(field: FieldConfig, widest: Option<usize>) -> usize {
    choose_limb_bits_for_modulus(&field_modulus(field), widest)
}

fn choose_limb_bits_for_modulus(modulus: &BigInt, widest: Option<usize>) -> usize {
    let default = limb_bits_for_modulus(modulus);
    let injective = widest_injective_int_bits_for_modulus(modulus);
    let mut limb_bits = default;
    while limb_bits >= 2 {
        if widest.is_none_or(|bits| is_evaluable(bits, limb_bits, injective)) {
            return limb_bits;
        }
        limb_bits /= 2;
    }
    default
}

/// Whether a `bits`-wide product of two full-range witnessed operands cut at `limb_bits`-bit limbs
/// is evaluable on a field whose widest injective width is `injective`.
pub(super) fn is_evaluable(bits: usize, limb_bits: usize, injective: usize) -> bool {
    let k = bits.div_ceil(limb_bits); // k is number of limbs
    let limb_max = (BigUint::one() << limb_bits) - 1u8; // highest possible value of a limb
    let msl_bits = bits - (k - 1) * limb_bits;
    let msl_max = (BigUint::one() << msl_bits) - 1u8; // highest possible value of most significant limb

    let largest_column_val = if k == 1 {
        &msl_max * &msl_max
    } else {
        let full_limb_inner_product_max = &limb_max * &limb_max;
        // column k-2 has k-1 inner product pairs, none of which involve the top limb,
        // so largest value is (k - 1) * maximum inner product value
        let column_k_minus_2_max = &full_limb_inner_product_max * BigUint::from(k - 1);
        // column k-1 has k inner product pairs, but two involve the top limb, which may be narrower
        let column_k_minus_1_max =
            &full_limb_inner_product_max * BigUint::from(k - 2) + ((&limb_max * &msl_max) << 1);
        column_k_minus_2_max.max(column_k_minus_1_max)
    };
    largest_column_val.bits() <= injective as u64
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
        let get_limb_width = |modulus_bits: usize, widest| {
            choose_limb_bits_for_modulus(&(BigInt::from(1) << (modulus_bits - 1)), widest)
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
                let ssa = operation_with_in(
                    field,
                    BinaryArithOpKind::UMul,
                    bits,
                    Type::witness_of(Type::int(bits)),
                );
                let flow = FlowAnalysis::run(&ssa);
                let types = Types::new().run(&ssa, &flow);
                let mut store = AnalysisStore::new();
                store.insert_with_deps(flow, vec![]);
                store.insert_with_deps(types, vec![]);
                assert_eq!(
                    LimbConfig::compute(&ssa, &store),
                    default,
                    "{field:?} int{bits}"
                );
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
