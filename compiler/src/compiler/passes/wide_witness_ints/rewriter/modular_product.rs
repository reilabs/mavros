use super::{
    Rewriter,
    identity::{ChainLayout, Factor, IdentityTemplate, LimbLayout, Polarity, Target, Term},
    lower::unguarded,
};
use crate::compiler::{
    analysis::types::FunctionTypeInfo,
    passes::{
        shared::{divmod_guard::nonzero_hint_divisor, limbs::witness_limb_bits},
        wide_witness_ints::types::limb_widths,
    },
    ssa::{
        ValueId,
        hlssa::{BinaryArithOpKind, CastTarget, OpCode, builder::HLEmitter},
    },
};
use mavros_artifacts::FieldConfig;
use mavros_int_semantics::IntBits;
use num_bigint::BigUint;
use num_traits::{One, Zero};

/// `a·b − q·m − r = 0`: a product reduced by a modulus in one identity.
#[derive(Clone, Copy, Debug)]
pub(in super::super) struct ModularProduct;

impl IdentityTemplate for ModularProduct {
    /// `a`, `b`, `q`, `m` and `r`.
    type Factors<'a> = (&'a Factor, &'a Factor, &'a Factor, &'a Factor, &'a Factor);

    /// The product is gated, so it is zero where the guard is off, as `q` and `r` are hinted to be.
    fn terms<'a>((a, b, q, m, r): Self::Factors<'a>) -> (Vec<Term<'a>>, Option<Target<'a>>) {
        let terms = vec![
            Term {
                polarity: Polarity::Positive,
                lhs: a,
                rhs: Some(b),
                gated: true,
            },
            Term {
                polarity: Polarity::Negative,
                lhs: q,
                rhs: Some(m),
                gated: false,
            },
            Term {
                polarity: Polarity::Negative,
                lhs: r,
                rhs: None,
                gated: false,
            },
        ];
        (terms, None)
    }

    /// `q` is as wide as `a·b`, twice the operands, which it must be for a modulus as small as one.
    fn largest_layout(
        self,
        bits: usize,
        limb_bits: usize,
        injective: usize,
    ) -> Option<ChainLayout> {
        let full = Factor::full(LimbLayout::new(bits, limb_bits));
        let quotient = Factor::full(LimbLayout::new(2 * bits, limb_bits));
        let identity = Self::instantiate(injective, (&full, &full, &quotient, &full, &full))?;
        Some(identity.into_layout())
    }
}

impl Rewriter<'_> {
    /// A modular product with a witnessed operand; returns `true` if `op` was one and was lowered.
    ///
    /// The quotient `q` of `a·b` by `m`, twice the operands' width, and the remainder `r` are
    /// witnessed as limbs from a pure-side division at that width, and pinned by
    /// `a·b − q·m − r == 0` and `r < m`, the latter refusing a zero modulus. Together they admit
    /// exactly the Euclidean pair, and the answer is `r`.
    ///
    /// Under a guard, the pure side divides zero by one where the guard is off, so `q`, `r` and
    /// the identity's columns are zero there, and its checks hold whatever the untaken branch left
    /// in the operands once the product is gated. Only the carries' range checks and `r < m` are
    /// guarded. The answer is `r`, zero there already, so it needs no select.
    pub(super) fn lower_modular_product(&mut self, op: &OpCode) -> bool {
        let Some(bits) = modular_multiplied(op, self.types, self.field()) else {
            return false;
        };
        let (guard, inner) = self.split_guard(op);
        let OpCode::MulMod {
            result,
            lhs,
            rhs,
            modulus,
        } = *inner
        else {
            ice_unreachable!("`modular_multiplied` matches only a modular product");
        };

        let wide = 2 * bits;
        let widths = limb_widths(bits, self.limb_bits());
        let quotient_widths = limb_widths(wide, self.limb_bits());

        let a = self.factor(lhs, bits);
        let b = self.factor(rhs, bits);
        let m = self.factor(modulus, bits);
        let (quotient_hint, remainder_hint) =
            self.modular_product_hints(lhs, rhs, modulus, bits, guard);

        let (quotient_max, remainder_max) = quotient_and_remainder_max(&a, &b, &m);
        let q = self.witness_hinted(quotient_hint, wide, &quotient_widths, &quotient_max);
        let r = self.witness_hinted(remainder_hint, bits, &widths, &remainder_max);

        let identity = self.identity(ModularProduct, (&a, &b, &q, &m, &r));
        self.lower_identity(&identity, guard);
        self.constrain_below(&r, &m, &widths, guard);

        let answer = r.limbs.iter().map(|limb| self.limb_field(limb)).collect();
        self.deliver(result, answer, &widths, bits, None);
        true
    }

    /// The pure-side quotient and remainder of `lhs·rhs` by `modulus`, at twice and once `bits`.
    ///
    /// Under a guard it divides zero by one where the guard is off, so both are zero there. A zero
    /// modulus is hinted as one, which the constraints refuse anyway.
    fn modular_product_hints(
        &mut self,
        lhs: ValueId,
        rhs: ValueId,
        modulus: ValueId,
        bits: usize,
        guard: Option<ValueId>,
    ) -> (ValueId, ValueId) {
        let wide = 2 * bits;
        let pure_lhs = self.pure_whole(lhs, bits);
        let pure_lhs = self.cast(pure_lhs, CastTarget::Int(wide));
        let pure_rhs = self.pure_whole(rhs, bits);
        let pure_rhs = self.cast(pure_rhs, CastTarget::Int(wide));
        let mut pure_product = self.bin(BinaryArithOpKind::UMul, pure_lhs, pure_rhs);
        let pure_modulus = self.pure_whole(modulus, bits);
        let mut pure_modulus = self.cast(pure_modulus, CastTarget::Int(wide));
        if let Some(condition) = guard {
            let condition = self.pure_of(condition, self.is_witness(condition));
            let zero = self.int_const(IntBits::zero(wide));
            let one = self.int_const(IntBits::one(wide));
            pure_product = self.select(condition, pure_product, zero);
            pure_modulus = self.select(condition, pure_modulus, one);
        }
        let pure_modulus = nonzero_hint_divisor(self, pure_modulus, wide);
        let quotient_hint = self.bin(BinaryArithOpKind::UDiv, pure_product, pure_modulus);
        let remainder_hint = self.bin(BinaryArithOpKind::URem, pure_product, pure_modulus);
        let remainder_hint = self.cast(remainder_hint, CastTarget::Int(bits));
        (quotient_hint, remainder_hint)
    }
}

/// The largest quotient and remainder of `a·b` by `m`, from the factors' bounds.
fn quotient_and_remainder_max(a: &Factor, b: &Factor, m: &Factor) -> (BigUint, BigUint) {
    let product_max = a.largest() * b.largest();
    let modulus_max = m.largest();
    let quotient_max = &product_max / m.smallest().max(BigUint::one());
    let remainder_max = if modulus_max.is_zero() {
        BigUint::zero()
    } else {
        (modulus_max - 1u8).min(product_max)
    };
    (quotient_max, remainder_max)
}

/// The width of the modular product the representation lowers, if `op` is one: guarded or not,
/// with a witnessed operand, at a width each of its factors spans more than one limb of.
pub(in super::super) fn modular_multiplied(
    op: &OpCode,
    types: &FunctionTypeInfo,
    field: FieldConfig,
) -> Option<usize> {
    let OpCode::MulMod {
        lhs, rhs, modulus, ..
    } = unguarded(op)
    else {
        return None;
    };
    let bits = types.get_value_type(*lhs).int_width()?;
    let witnessed = [lhs, rhs, modulus]
        .into_iter()
        .any(|value| types.get_value_type(*value).is_witness_of());
    (witnessed && bits > witness_limb_bits(field)).then_some(bits)
}
