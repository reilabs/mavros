use super::{
    super::{types::limb_widths, utils::multi_cell_int_bits},
    Rewriter,
};
use crate::compiler::ssa::{
    ValueId,
    hlssa::{BinaryArithOpKind, CastTarget, OpCode, builder::HLEmitter},
};
use mavros_int_semantics::IntBits;

impl Rewriter<'_> {
    /// A single witnessed `bits`-wide value, split into range-checked limbs constrained to
    /// recombine to it.
    ///
    /// As `bits <= multi_cell_int_bits`, the source has an element and the limb sum reaches it
    /// without wrapping. A value is decomposed once per block per width; later requests reuse the
    /// limbs.
    pub(super) fn decompose(&mut self, value: ValueId, bits: usize) -> Vec<ValueId> {
        if let Some(limbs) = self.decomposed.get(&(value, bits)) {
            return limbs.clone();
        }
        let pure = self.cast(value, CastTarget::ValueOf);
        let (limbs, sum) = self.witness_limbs_of(pure, bits);
        let value_field = self.cast(value, CastTarget::Field);
        self.constrain_equal(value_field, sum, None);

        self.decomposed.insert((value, bits), limbs.clone());
        limbs
    }

    /// The limbs of a pure `bits`-wide `value` at the limb widths, each written to a witness
    /// column and range-checked at its width, with their place-weighted field sum. The caller ties
    /// that sum to the witnessed source.
    pub(super) fn witness_limbs_of(
        &mut self,
        pure: ValueId,
        bits: usize,
    ) -> (Vec<ValueId>, ValueId) {
        let widths = limb_widths(bits, self.limb_bits());
        let mut limbs = Vec::with_capacity(widths.len());
        let mut fields = Vec::with_capacity(widths.len());
        for (index, width) in widths.iter().enumerate() {
            let (limb, limb_field) = self.witness_limb(pure, bits, index, *width, *width);
            limbs.push(limb);
            fields.push(limb_field);
        }

        let mut sum = None;
        for (index, limb_field) in fields.iter().enumerate() {
            sum = Some(self.add_at_place(sum, *limb_field, index));
        }
        (limbs, sum.expect("an integer has at least one limb"))
    }

    /// A **pure** wide value injected into the representation as witnessed, range-checked limbs,
    /// written into `results`.
    ///
    /// [`Self::decompose`] cannot serve past the field's width: it ties the limbs to the source's
    /// field element, which a value wider than the modulus lacks. The source is a hint, so the
    /// limbs **are** the value, each bounded by a range check. A constant enters as one witnessed
    /// constant per limb, pinned by being one.
    pub(super) fn inject(&mut self, value: ValueId, bits: usize, results: &[ValueId]) {
        let widths = limb_widths(bits, self.limb_bits());
        assert_eq!(
            widths.len(),
            results.len(),
            "ICE: an int{bits} is {} limbs, not {}",
            widths.len(),
            results.len()
        );
        if let Some(pattern) = self.known(value) {
            for (index, (result, width)) in results.iter().zip(&widths).enumerate() {
                let limb = self.int_const(pattern.bit_range(index * self.limb_bits(), *width));
                self.push(OpCode::Cast {
                    result: *result,
                    value: limb,
                    target: CastTarget::WitnessOf,
                });
            }
            return;
        }
        for (index, (result, width)) in results.iter().zip(&widths).enumerate() {
            let low = index * self.limb_bits();
            let hint = self.shifted_down(value, bits, low);
            let narrowed = self.cast(hint, CastTarget::Int(*width));
            self.push(OpCode::Cast {
                result: *result,
                value: narrowed,
                target: CastTarget::WitnessOf,
            });
            let limb_field = self.cast(*result, CastTarget::Field);
            self.push(OpCode::Rangecheck {
                value: limb_field,
                max_bits: *width,
            });
        }
    }

    /// The low `bits` of a limb list as a single witnessed value of that width; the inverse of
    /// [`Self::decompose`].
    ///
    /// It needs no separate constraint: a linear combination of pinned values is pinned.
    ///
    /// `bits` must be at most [`multi_cell_int_bits`], and the assertion is what makes this sound:
    /// the sum is formed in **one field element**, so a target the modulus cannot carry would wrap
    /// silently. A wider target uses [`Self::pure_limbs_at`] instead.
    pub(super) fn recombine(
        &mut self,
        limbs: &[ValueId],
        widths: &[usize],
        bits: usize,
    ) -> ValueId {
        assert!(
            bits <= multi_cell_int_bits(self.field),
            "ICE: an int{bits} recombination is summed in one field element, which cannot carry it"
        );
        let target = limb_widths(bits, self.limb_bits());
        let mut sum = None;
        for (index, target_width) in target.iter().enumerate() {
            let mut limb = limbs[index];
            if *target_width < widths[index] {
                limb = self.truncate_limb(limb, *target_width);
            }

            let limb_field = self.cast(limb, CastTarget::Field);
            sum = Some(self.add_at_place(sum, limb_field, index));
        }

        let sum = sum.expect("an integer has at least one limb");
        self.cast(sum, CastTarget::Int(bits))
    }

    /// Pure limbs shifted into place and or-ed together, as one pure value of `bits`.
    ///
    /// The **pure** counterpart of [`Self::recombine`], which sums field elements and so wraps once
    /// the value passes the modulus. This is integer arithmetic at the value's width, which both
    /// compiled backends have at every width, and costs interpreter work rather than constraints.
    /// It lets a sequence hold elements the field cannot carry.
    pub(super) fn recombine_pure(&mut self, limbs: &[ValueId], bits: usize) -> ValueId {
        let mut sum = None;
        for (index, limb) in limbs.iter().enumerate() {
            let widened = self.cast(*limb, CastTarget::Int(bits));
            let placed = if index == 0 {
                widened
            } else {
                let amount =
                    self.int_const(IntBits::from_u128(bits, (index * self.limb_bits()) as u128));
                self.bin(BinaryArithOpKind::UShl, widened, amount)
            };
            sum = Some(match sum {
                None => placed,
                Some(acc) => self.bin(BinaryArithOpKind::Or, acc, placed),
            });
        }
        sum.expect("an integer has at least one limb")
    }
}
