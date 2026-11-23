//! Products too wide for one field element, lowered to a schoolbook multiplication over limbs and checked by a
//! [`PolynomialIdentity`].

use super::identity::{ChainLayout, Factor, IdentityTemplate, LimbLayout, Polarity, Target, Term};
use super::lower::exceeds_single_cell_product;
use super::{Operands, Rewriter};
use crate::compiler::{
    analysis::types::FunctionTypeInfo,
    passes::wide_witness_ints::types::limb_widths,
    ssa::hlssa::{BinaryArithOpKind, OpCode},
};
use mavros_artifacts::FieldConfig;

#[cfg(test)]
pub(crate) use tests::division_carries;
#[cfg(test)]
mod tests;

impl Rewriter<'_> {
    /// Lowers a product the single cell cannot hold. Returns whether `op` was one and was lowered.
    ///
    /// An unsigned product is a schoolbook at the witness limb, checked by the polynomial
    /// identity. A signed one is handed to [`Self::lower_signed_product`].
    pub(super) fn lower_product(&mut self, op: &OpCode) -> bool {
        let Some(Operands { bits, lhs, rhs }) = multiplied(op, self.types, self.field()) else {
            return false;
        };
        let (guard, inner) = self.split_guard(op);
        let OpCode::BinaryArithOp { kind, result, .. } = inner else {
            ice_unreachable!("`multiplied` matches only a binary operation");
        };
        if *kind == BinaryArithOpKind::SMul {
            self.lower_signed_product(*result, lhs, rhs, bits, guard);
            return true;
        }

        let widths = limb_widths(bits, self.limb_bits());
        let lhs = self.factor(lhs, bits);
        let rhs = self.factor(rhs, bits);
        let answer = LimbLayout::new(bits, self.limb_bits());
        let identity = self.identity(UnsignedProduct, (&lhs, &rhs, Target::Witnessed(answer)));
        let answer = self.lower_identity(&identity, guard);
        self.deliver(*result, answer, &widths, bits, guard);
        true
    }
}

/// `a·b = c`, solving for `c`.
#[derive(Clone, Copy, Debug)]
pub(super) struct UnsignedProduct;

impl IdentityTemplate for UnsignedProduct {
    /// `a`, `b`, and `c`.
    type Factors<'a> = (&'a Factor, &'a Factor, Target<'a>);

    fn terms<'a>(self, (a, b, c): Self::Factors<'a>) -> (Vec<Term<'a>>, Option<Target<'a>>) {
        let product = Term {
            polarity: Polarity::Positive,
            lhs: a,
            rhs: Some(b),
            gated: true,
        };
        (vec![product], Some(c))
    }

    fn largest_layout(
        self,
        bits: usize,
        limb_bits: usize,
        injective: usize,
    ) -> Option<ChainLayout> {
        let layout = LimbLayout::new(bits, limb_bits);
        let full = Factor::full(layout);
        let identity = self.instantiate(injective, (&full, &full, Target::Witnessed(layout)))?;
        Some(identity.into_layout())
    }
}

/// The product the schoolbook lowers, if `op` is one: a multiplication under either reading,
/// guarded or not, with a witnessed operand, at a width the single cell does not take.
pub(in super::super) fn multiplied(
    op: &OpCode,
    types: &FunctionTypeInfo,
    field: FieldConfig,
) -> Option<Operands> {
    exceeds_single_cell_product(
        op,
        types,
        field,
        &[BinaryArithOpKind::UMul, BinaryArithOpKind::SMul],
    )
}
