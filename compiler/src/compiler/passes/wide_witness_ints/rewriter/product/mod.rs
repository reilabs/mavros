//! Products too wide for one field element, lowered to limbs and checked by a
//! [`PolynomialIdentity`](super::identity::PolynomialIdentity).

use super::identity::{Factor, IdentityShape, IdentityTemplate, Polarity, Target, Term};
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
    /// An unsigned product is witnessed at the configured limb and checked by the polynomial
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
        let lhs = self.factor(lhs, bits, &widths);
        let rhs = self.factor(rhs, bits, &widths);
        let identity = self.identity(UnsignedProduct, &widths, (&lhs, &rhs, &widths));
        let answer = self.lower_identity(&identity, guard);
        self.deliver(*result, answer, &widths, bits, guard);
        true
    }
}

/// `a·b = c`, solving for `c`.
#[derive(Clone, Copy, Debug)]
pub(super) struct UnsignedProduct;

impl IdentityShape for UnsignedProduct {
    fn num_products(&self) -> usize {
        1
    }

    fn num_linear_terms(&self) -> usize {
        0
    }

    fn is_checked(&self) -> bool {
        true
    }
}

impl<'a> IdentityTemplate<'a> for UnsignedProduct {
    /// `a`, `b`, and the widths `c` is solved at.
    type Factors = (&'a Factor, &'a Factor, &'a [usize]);

    fn terms(self, (a, b, widths): Self::Factors) -> (Vec<Term<'a>>, Option<Target<'a>>) {
        let product = Term {
            polarity: Polarity::Positive,
            lhs: a,
            rhs: Some(b),
            gated: true,
        };
        (vec![product], Some(Target::Witnessed(widths)))
    }
}

/// The product the representation lowers, if `op` is one: a multiplication under either reading,
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
