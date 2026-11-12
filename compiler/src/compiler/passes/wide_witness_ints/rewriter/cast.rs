use crate::compiler::ssa::{
    ValueId,
    hlssa::{BinaryArithOpKind, CastTarget, CmpKind, OpCode, builder::HLEmitter},
};
use mavros_int_semantics::IntBits;

use super::super::{
    types::{limb_widths, paired},
    utils::multi_cell_int_bits,
};
use super::{Rewriter, sign::Sign};

impl Rewriter<'_> {
    /// A cast, which is where a value enters and leaves the representation.
    pub(super) fn lower_cast(&mut self, result: ValueId, value: ValueId, target: &CastTarget) {
        let source_type = self.types.get_value_type(value);
        let source_bits = source_type.int_width();

        match target {
            CastTarget::Int(to_bits) | CastTarget::IntUnchecked(to_bits) => {
                // The other side of the field boundary: a wide value read back out of the field
                // elements its limbs were pinned as. Each limb returns at its width, so there
                // is nothing to recombine and no place value to mint.
                if source_bits.is_none() {
                    let limbs = self.limbs(value);
                    let results = self.limbs(result);
                    if limbs.len() == 1 && results.len() > 1 {
                        self.lower_field_into_limbs(value, *to_bits, &results);
                        return;
                    }
                    let widths = limb_widths(*to_bits, self.limb_bits());
                    assert_eq!(
                        limbs.len(),
                        results.len(),
                        "ICE: a non-integer of {} limbs was read back as an int{to_bits} of {}",
                        limbs.len(),
                        results.len()
                    );
                    assert_eq!(
                        widths.len(),
                        results.len(),
                        "ICE: an int{to_bits} is {} limbs, not {}",
                        widths.len(),
                        results.len()
                    );
                    for ((result, limb), width) in results.into_iter().zip(limbs).zip(widths) {
                        self.push(OpCode::Cast {
                            result,
                            value: limb,
                            target: CastTarget::IntUnchecked(width),
                        });
                    }
                    return;
                }
                let from_bits = source_bits.expect("a non-integer source returned above");
                match self.wide_width(result) {
                    // Into the representation, or between two widths inside it.
                    Some(_) => {
                        let limbs = self.relimb(value, from_bits, *to_bits);
                        for (result, limb) in paired(self.limbs(result), limbs) {
                            self.push(OpCode::Cast {
                                result,
                                value: limb,
                                target: CastTarget::Nop,
                            });
                        }
                    }
                    // Out of it. A target the field carries is one element again, so the low limbs
                    // recombine into it and the linear combination keeps them pinned. A target it
                    // cannot carry has no element to sum into, so the limbs recombine as integer
                    // arithmetic at the target's width, as the witness strip does.
                    None => {
                        let combined = if *to_bits > multi_cell_int_bits(self.field()) {
                            let limbs = self.pure_limbs_at(value, from_bits, *to_bits);
                            self.recombine_pure(&limbs, *to_bits)
                        } else {
                            let widths = limb_widths(from_bits, self.limb_bits());
                            let limbs = self.limbs(value);
                            self.recombine(&limbs, &widths, *to_bits)
                        };
                        self.push(OpCode::Cast {
                            result,
                            value: combined,
                            target: CastTarget::Nop,
                        });
                    }
                }
            }

            CastTarget::WitnessOf => {
                let bits = self.wide_width(result).expect(
                    "ICE: a witness injection reached the multi-cell representation without a wide result",
                );
                let results = self.limbs(result);
                self.inject(value, bits, &results);
            }

            CastTarget::ValueOf => {
                let bits = self.wide_width(value).expect(
                    "ICE: a witness strip reached the multi-cell representation without a wide source",
                );
                let stripped: Vec<ValueId> = self
                    .limbs(value)
                    .into_iter()
                    .map(|limb| self.cast(limb, CastTarget::ValueOf))
                    .collect();
                let sum = self.recombine_pure(&stripped, bits);
                self.push(OpCode::Cast {
                    result,
                    value: sum,
                    target: CastTarget::Nop,
                });
            }

            CastTarget::Nop => {
                for (result, limb) in paired(self.limbs(result), self.limbs(value)) {
                    self.push(OpCode::Cast {
                        result,
                        value: limb,
                        target: CastTarget::Nop,
                    });
                }
            }

            // A wide value's injective field image is one element per limb; the array lowering's
            // lookup chain moves it through a field. Each limb is already held to its width, so
            // each is an element.
            CastTarget::FieldLimbs => {
                let results = self.limbs(result);
                let limbs = self.operand_limbs(value, results.len());
                for (result, limb) in results.into_iter().zip(limbs) {
                    self.push(OpCode::Cast {
                        result,
                        value: limb,
                        target: CastTarget::Field,
                    });
                }
            }

            // A wide witnessed value's residue: its limbs at their place values, summed in one
            // element. Each limb is held to its width, so the sum is the integer reduced modulo
            // the field's modulus, and as a linear combination of pinned limbs it is pinned.
            CastTarget::Field => {
                let limbs = self.limbs(value);
                let sum = self.limb_sum(&limbs);
                self.push(OpCode::Cast {
                    result,
                    value: sum,
                    target: CastTarget::Nop,
                });
            }

            // A whole sequence entering the representation: `k` sequences out, each holding one
            // limb position of every element.
            //
            // Both sides are transposed already, so this is a mapped cast per limb sequence and
            // nothing here reads an element. Every limb sequence is narrow, so `LowerMapCasts`
            // expands each into the ordinary per-element witness injection later in the pipeline.
            CastTarget::Map(inner) => {
                for (result, value) in paired(self.limbs(result), self.limbs(value)) {
                    self.push(OpCode::Cast {
                        result,
                        value,
                        target: CastTarget::Map(inner.clone()),
                    });
                }
            }

            CastTarget::ArrayToSlice => {
                for (result, value) in paired(self.limbs(result), self.limbs(value)) {
                    self.push(OpCode::Cast {
                        result,
                        value,
                        target: CastTarget::ArrayToSlice,
                    });
                }
            }
        }
    }

    /// A sign extension whose target is held as limbs: the source's limbs, with its sign bit `s`
    /// filling every bit above them.
    ///
    /// Every limb of the answer is linear in `s`. Below the source's top limb they are the source's
    /// limbs; the limb holding the source's top limb (`w_s` bits of a `w`-bit limb) is that limb
    /// plus `s·(2^w - 2^w_s)`; every limb above it is `s·(2^w - 1)`. Each is below `2^w`, as its
    /// pieces are and cover disjoint bits, so the cut that pins `s` is the whole cost.
    ///
    /// A sign known at compile time (read off a known top limb) fills at compile time: that limb
    /// with its fill and every limb above it are constants.
    ///
    /// The source is witnessed: the type rule keeps the witness wrapper across the extension, and
    /// only a witnessed result is held as limbs.
    pub(super) fn lower_sext(
        &mut self,
        result: ValueId,
        value: ValueId,
        from_bits: usize,
        to_bits: usize,
    ) {
        assert!(
            self.is_witness(value),
            "ICE: a sign extension into the representation from a pure int{from_bits}"
        );
        let limb_bits = self.limb_bits();
        let source_widths = limb_widths(from_bits, limb_bits);
        let source = match self.wide_width(value) {
            Some(_) => self.limbs(value),
            None => self.decompose(value, from_bits),
        };
        let top = source.len() - 1;
        let top_width = source_widths[top];
        let sign = self.top_bit((source[top], true), top_width);

        let widths = limb_widths(to_bits, limb_bits);
        let mut answer = Vec::with_capacity(widths.len());
        for (index, width) in widths.iter().enumerate() {
            let limb = match index.cmp(&top) {
                std::cmp::Ordering::Less => self.cast(source[index], CastTarget::Field),
                std::cmp::Ordering::Equal
                    if *width == top_width || matches!(sign, Sign::Known(false)) =>
                {
                    self.cast(source[top], CastTarget::Field)
                }
                std::cmp::Ordering::Equal => {
                    let fill = IntBits::all_ones(width - top_width)
                        .cast(*width)
                        .shifted_left(top_width);
                    match sign {
                        Sign::Known(_) => {
                            let known = self.known(source[top]).unwrap_or_else(|| {
                                ice!("a known sign was read from a limb that is not known")
                            });
                            self.limb_const(&known.cast(*width).or(&fill))
                        }
                        Sign::Bit(bit, _) => {
                            let limb = self.cast(source[top], CastTarget::Field);
                            let fill = self.limb_const(&fill);
                            let fill = self.bin(BinaryArithOpKind::UMul, bit, fill);
                            self.bin(BinaryArithOpKind::UAdd, limb, fill)
                        }
                    }
                }
                std::cmp::Ordering::Greater => match sign {
                    Sign::Known(negative) => {
                        let fill = if negative {
                            IntBits::all_ones(*width)
                        } else {
                            IntBits::zero(*width)
                        };
                        self.limb_const(&fill)
                    }
                    Sign::Bit(bit, _) => {
                        let ones = self.limb_const(&IntBits::all_ones(*width));
                        self.bin(BinaryArithOpKind::UMul, bit, ones)
                    }
                },
            };
            answer.push(limb);
        }
        self.deliver(result, answer, &widths, to_bits, None);
    }

    /// The assertion of a comparison, which is one assertion per limb.
    ///
    /// A pair of limbs known equal at compile time holds already and asserts nothing. One known to
    /// differ is kept, so that the program is refused as any assertion of two unequal constants is.
    pub(super) fn lower_assert_compare(
        &mut self,
        kind: CmpKind,
        lhs: ValueId,
        rhs: ValueId,
        guard: Option<ValueId>,
    ) {
        assert!(
            matches!(kind, CmpKind::Eq),
            "ICE: a {kind:?} assertion of a wide witnessed integer reached the multi-cell representation; the carry chain takes every ordering"
        );

        let pairs = self.operand_pair(lhs, rhs);
        for (lhs, rhs) in pairs {
            if self.known_equal(lhs, rhs) == Some(true) {
                continue;
            }
            self.push_guarded(
                guard,
                OpCode::AssertCmp {
                    kind: CmpKind::Eq,
                    lhs,
                    rhs,
                },
            );
        }
    }
}
