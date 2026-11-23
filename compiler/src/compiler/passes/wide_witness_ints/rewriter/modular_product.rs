use super::identity::{ChainLayout, Factor, IdentityTemplate, LimbLayout, Polarity, Target, Term};

/// `a·b − q·m − r = 0`: a product reduced by a modulus in one identity.
#[derive(Clone, Copy, Debug)]
#[allow(dead_code)]
pub(in super::super) struct ModularProduct;

impl IdentityTemplate for ModularProduct {
    /// `a`, `b`, `q`, `m` and `r`.
    type Factors<'a> = (&'a Factor, &'a Factor, &'a Factor, &'a Factor, &'a Factor);

    /// The product is gated, so it is zero where the guard is off, as `q` and `r` are hinted to be.
    fn terms<'a>(self, (a, b, q, m, r): Self::Factors<'a>) -> (Vec<Term<'a>>, Option<Target<'a>>) {
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
        let identity = self.instantiate(injective, (&full, &full, &quotient, &full, &full))?;
        Some(identity.into_layout())
    }
}
