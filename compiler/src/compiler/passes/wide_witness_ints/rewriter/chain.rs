use super::{
    super::types::{limb_widths, paired},
    Operands, Rewriter,
    lower::{unguarded, witnessed_width},
};

use crate::compiler::{
    analysis::types::FunctionTypeInfo,
    passes::shared::limbs::widest_cell_sum_bits,
    ssa::{
        ValueId,
        hlssa::{
            BinaryArithOpKind, CastTarget, OpCode,
            builder::HLEmitter,
            {CmpKind, Constant},
        },
    },
};

use mavros_artifacts::FieldConfig;
use mavros_int_semantics::IntBits;
use num_bigint::BigUint;
/// Who bounds a chain's top answer limb at its width.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum TopLimb {
    /// The chain, as it does every other limb.
    Checked,

    /// The caller, which cuts the limb's top bit out with [`Rewriter::sign_bit`] and so bounds it
    /// with the two range checks of the cut.
    CutByCaller,
}

/// Where the carry out of a chain's top limb goes.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum CarryOut {
    /// Nowhere: the operation is checked, and a carry past the top limb is its overflow.
    Zero,

    /// Out, always: an asserted ordering, whose subtraction has to borrow.
    One,

    /// Witnessed and handed back: an ordering, which **is** that borrow.
    Witnessed,
}

/// What a carry chain leaves behind.
pub(super) struct Chain {
    /// Each limb of the answer as a field element, range-checked at its width, the top one
    /// unless [`TopLimb::CutByCaller`] says the caller bounds it.
    pub(super) answer: Vec<ValueId>,

    /// The width of each limb, little-endian.
    pub(super) widths: Vec<usize>,

    /// The carry out of the top limb, where the chain witnesses one.
    pub(super) carry_out: Option<ValueId>,
}

impl Rewriter<'_> {
    /// The unsigned sum, difference and ordering at a width whose sum one element cannot carry,
    /// through the carry [`Chain`] whatever form the operands are in, as the chain handles
    /// overflow. Returns whether `op` was lowered.
    pub(super) fn lower_through_chain(&mut self, op: &OpCode) -> bool {
        let Some(Operands { bits, lhs, rhs }) = chained(op, self.types, self.field()) else {
            return false;
        };
        let (guard, inner) = self.split_guard(op);

        match inner {
            OpCode::BinaryArithOp { kind, result, .. } => match kind {
                BinaryArithOpKind::UAdd | BinaryArithOpKind::USub => {
                    self.lower_add_sub(*kind, *result, lhs, rhs, bits, guard);
                }
                BinaryArithOpKind::SAdd | BinaryArithOpKind::SSub => {
                    self.lower_signed_add_sub(*kind, *result, lhs, rhs, bits, guard);
                }
                _ => ice_unreachable!("`chained` matches only a sum or a difference"),
            },
            // An ordering cannot fail, so a guard has nothing to withhold from it: the chain holds
            // for any pair of operands, and every limb of either is bounded whatever the guard.
            OpCode::Cmp { kind, result, .. } => {
                let signed = kind.is_signed();
                let chain = self.ordering_chain(signed, lhs, rhs, bits, CarryOut::Witnessed, None);
                self.push(OpCode::Cast {
                    result: *result,
                    value: chain.carry_out.expect("an ordering witnesses its borrow"),
                    target: CastTarget::Int(1),
                });
            }
            OpCode::AssertCmp { kind, .. } => {
                self.ordering_chain(kind.is_signed(), lhs, rhs, bits, CarryOut::One, guard);
            }
            _ => ice_unreachable!("`chained` matches only these"),
        }
        true
    }

    /// The chain whose top borrow is `lhs < rhs`, under the signed reading where `signed`.
    ///
    /// A signed ordering is the unsigned ordering of the operands with their top bits flipped
    /// (offset binary), which maps the signed range onto `[0, 2^N)` in order. Only the top limb
    /// changes ([`Self::flip_top_bit`]).
    fn ordering_chain(
        &mut self,
        signed: bool,
        lhs: ValueId,
        rhs: ValueId,
        bits: usize,
        out: CarryOut,
        guard: Option<ValueId>,
    ) -> Chain {
        if !signed {
            return self.carry_chain(true, lhs, rhs, bits, out, guard);
        }
        let widths = limb_widths(bits, self.limb_bits());
        let mut lhs = self.chain_limbs(lhs, bits, widths.len());
        let mut rhs = self.chain_limbs(rhs, bits, widths.len());
        let top = widths.len() - 1;
        lhs[top] = self.flip_top_bit(lhs[top], widths[top]);
        rhs[top] = self.flip_top_bit(rhs[top], widths[top]);
        self.carry_chain_over(true, &lhs, &rhs, widths, out, guard)
    }

    /// A signed sum or difference: the unsigned one's limbs, with overflow read off the sign bits
    /// rather than the carry out of the top limb.
    ///
    /// The chain says `a + b = r + 2^N·c` for a sum and `a - b = r - 2^N·c` for a difference, with
    /// `c` the witnessed carry or borrow out of the top limb. Reading each pattern `x` as
    /// `x - 2^N·s_x` for its sign bit `s_x`, the signed sum is `r + 2^N·(c - s_a - s_b)` and the
    /// signed difference `r - 2^N·(c + s_a - s_b)`. Either fits exactly when it equals `r`'s signed
    /// reading, `r - 2^N·s_r`, so the one check is `c + s_r == s_a + s_b` for a sum and
    /// `c + s_a == s_r + s_b` for a difference: linear in four pinned bits.
    ///
    /// The answer's sign is cut out of its top limb without the guard, as the operands' are: an
    /// honest carry keeps every answer limb in range wherever the operands are, and they are on
    /// every path. Only the check is guarded. The cut bounds the top limb at its width, so the
    /// chain does not check it again ([`TopLimb::CutByCaller`]), except where that limb is one bit
    /// wide and so is the sign bit itself, with no cut.
    fn lower_signed_add_sub(
        &mut self,
        kind: BinaryArithOpKind,
        result: ValueId,
        lhs: ValueId,
        rhs: ValueId,
        bits: usize,
        guard: Option<ValueId>,
    ) {
        let subtract = kind == BinaryArithOpKind::SSub;
        let widths = limb_widths(bits, self.limb_bits());
        let top = widths.len() - 1;
        let lhs = self.chain_limbs(lhs, bits, widths.len());
        let rhs = self.chain_limbs(rhs, bits, widths.len());
        let lhs_sign = self.sign_bit(lhs[top], widths[top]);
        let rhs_sign = self.sign_bit(rhs[top], widths[top]);

        let top_limb = if widths[top] > 1 {
            TopLimb::CutByCaller
        } else {
            TopLimb::Checked
        };
        let chain = self.carry_chain_bounded(
            subtract,
            &lhs,
            &rhs,
            widths,
            CarryOut::Witnessed,
            guard,
            top_limb,
        );

        let carry = chain
            .carry_out
            .expect("a signed sum or difference witnesses its top carry");
        let answer_top = self.cast(chain.answer[top], CastTarget::Int(chain.widths[top]));
        let answer_sign = self.sign_bit((answer_top, true), chain.widths[top]);

        let (left, right) = if subtract {
            (
                self.bin(BinaryArithOpKind::UAdd, carry, lhs_sign),
                self.bin(BinaryArithOpKind::UAdd, answer_sign, rhs_sign),
            )
        } else {
            (
                self.bin(BinaryArithOpKind::UAdd, carry, answer_sign),
                self.bin(BinaryArithOpKind::UAdd, lhs_sign, rhs_sign),
            )
        };
        self.constrain_equal(left, right, guard);
        self.deliver(result, chain.answer, &chain.widths, bits, guard);
    }

    /// The top bit of a `width`-bit limb, as a field element that is zero or one.
    ///
    /// The sign-bit primitive. A known limb answers at compile time and a pure one on the pure
    /// side, so neither costs a constraint. A witnessed one is cut at `width - 1` by
    /// [`Self::cut_segment`], which witnesses the bit, range-checks it at one bit, and range-checks
    /// the rest of the limb at `width - 1`. Together these pin the bit and bound the limb below
    /// `2^width`: the other bit value puts the rest out of range. A limb is cut once per block, and
    /// later requests reuse the bit ([`Self::signs`]). A one-bit limb is its top bit and is not
    /// bounded here.
    pub(super) fn sign_bit(&mut self, (limb, witnessed): (ValueId, bool), width: usize) -> ValueId {
        if let Some(pattern) = self.known(limb) {
            return self.field_const(match pattern.cast(width).bit(width - 1) {
                Some(true) => self.field().one(),
                _ => self.field().zero(),
            });
        }
        if !witnessed {
            let top = self.shifted_down(limb, width, width - 1);
            let bit = self.cast(top, CastTarget::Int(1));
            return self.cast(bit, CastTarget::Field);
        }
        let whole = self.cast(limb, CastTarget::Field);
        if width == 1 {
            return whole;
        }
        if let Some(bit) = self.signs.get(&(limb, width)) {
            return *bit;
        }
        let pieces = self.cut_segment(limb, whole, width, 0, &[0, width - 1, width]);
        let &[_, (_, 1, bit)] = pieces.as_slice() else {
            ice!("a limb's top bit was cut as {pieces:?}");
        };
        self.signs.insert((limb, width), bit);
        bit
    }

    /// A `width`-bit limb with its top bit flipped, as [`Self::carry_chain_over`] reads it.
    ///
    /// A known or pure limb flips on the pure side. A witnessed one is `x + 2^(w - 1) - s·2^w` for
    /// its top bit `s`: linear in the [`Self::sign_bit`] cut, and below `2^w`.
    fn flip_top_bit(
        &mut self,
        (limb, witnessed): (ValueId, bool),
        width: usize,
    ) -> (ValueId, bool) {
        let top = IntBits::one(width).shifted_left(width - 1);
        if let Some(pattern) = self.known(limb) {
            return (self.int_const(pattern.cast(width).xor(&top)), false);
        }
        if !witnessed {
            let top = self.int_const(top);
            return (self.bin(BinaryArithOpKind::Xor, limb, top), false);
        }
        let sign = self.sign_bit((limb, true), width);
        let whole = self.cast(limb, CastTarget::Field);
        let half = self.field_const(self.field().two_pow(width - 1));
        let raised = self.bin(BinaryArithOpKind::UAdd, whole, half);
        let place = self.field_const(self.field().two_pow(width));
        let cleared = self.bin(BinaryArithOpKind::UMul, sign, place);
        let flipped = self.bin(BinaryArithOpKind::USub, raised, cleared);
        (self.cast(flipped, CastTarget::Int(width)), true)
    }

    /// A checked sum or difference, whose carry out of the top limb is its overflow.
    ///
    /// Under a guard the answer is zero where the guard is off: the chain's range checks are off
    /// there, so the limbs it would leave are unbounded.
    fn lower_add_sub(
        &mut self,
        kind: BinaryArithOpKind,
        result: ValueId,
        lhs: ValueId,
        rhs: ValueId,
        bits: usize,
        guard: Option<ValueId>,
    ) {
        let subtract = kind == BinaryArithOpKind::USub;
        let chain = self.carry_chain(subtract, lhs, rhs, bits, CarryOut::Zero, guard);
        self.deliver(result, chain.answer, &chain.widths, bits, guard);
    }

    /// Hand the limbs of an answer to `result`, each a field element range-checked at its width,
    /// and zero where a guard is off.
    ///
    /// A result held as one element fits one, as the operands do: only their sum or product needed
    /// limbs. A limb known at compile time, as every limb of `0 / d` is, is delivered as a
    /// **witnessed** integer constant.
    pub(super) fn deliver(
        &mut self,
        result: ValueId,
        answer: Vec<ValueId>,
        widths: &[usize],
        bits: usize,
        guard: Option<ValueId>,
    ) {
        assert!(
            self.is_witness(result),
            "ICE: a gadget delivered a pure result, but every operation it lowers has a witnessed operand"
        );
        let kept: Vec<ValueId> = answer
            .into_iter()
            .zip(widths)
            .map(|(limb, width)| match guard {
                Some(condition) => {
                    let zero = self.field_const(self.field().zero());
                    self.select(condition, limb, zero)
                }
                None => match self.constant_limb(limb, *width) {
                    Some(pattern) => {
                        let constant = self.int_const(pattern);
                        self.cast(constant, CastTarget::WitnessOf)
                    }
                    None => limb,
                },
            })
            .collect();

        let results = self.limbs(result);
        if results.len() == widths.len() {
            for ((result, limb), width) in paired(results, kept).zip(widths) {
                self.push(OpCode::Cast {
                    result,
                    value: limb,
                    target: CastTarget::Int(*width),
                });
            }
        } else {
            let limbs: Vec<ValueId> = kept
                .into_iter()
                .zip(widths)
                .map(|(limb, width)| self.cast(limb, CastTarget::Int(*width)))
                .collect();
            let whole = self.recombine(&limbs, widths, bits);
            self.push(OpCode::Cast {
                result,
                value: whole,
                target: CastTarget::Nop,
            });
        }
    }

    /// The pattern of an answer limb known at compile time, read at its limb's `width`.
    pub(super) fn constant_limb(&self, limb: ValueId, width: usize) -> Option<IntBits> {
        match self.ssa.get_const(limb).as_deref() {
            Some(Constant::Int(pattern)) => {
                assert!(
                    BigUint::from(pattern).bits() <= width as u64,
                    "ICE: a known answer limb does not fit its width of {width} bits"
                );
                Some(pattern.cast(width))
            }
            Some(Constant::Field(element)) => {
                let limbs = element.into_bigint().0;
                assert!(
                    IntBits::field_limbs_fit(&limbs, width),
                    "ICE: a known answer limb does not fit its width of {width} bits"
                );
                Some(IntBits::from_field_limbs(&limbs, width))
            }
            _ => None,
        }
    }

    /// `lhs + rhs` or `lhs - rhs` limb by limb, each limb's carry a witnessed bit.
    ///
    /// For a limb of width `w`, the answer is `a + b + c_in - 2^w * c_out` for a sum and
    /// `a - b - c_in + 2^w * c_out` for a difference, range-checked at `w`. This pins `c_out`: the
    /// other value moves the answer by `2^w`, to `2^w` or past it, or below zero, where it wraps to
    /// an element no range check at `w` admits. Either answer is within `2^(w + 1)` of zero, far
    /// inside the modulus, so the identity holds as one field element whichever carry the prover
    /// picks.
    ///
    /// Every check the chain makes is under `guard`. The decomposition of an operand that is still
    /// one element is not: its checks restate that operand's bounds, which hold whether the guard
    /// is on or off.
    fn carry_chain(
        &mut self,
        subtract: bool,
        lhs: ValueId,
        rhs: ValueId,
        bits: usize,
        out: CarryOut,
        guard: Option<ValueId>,
    ) -> Chain {
        let widths = limb_widths(bits, self.limb_bits());
        let lhs = self.chain_limbs(lhs, bits, widths.len());
        let rhs = self.chain_limbs(rhs, bits, widths.len());
        self.carry_chain_over(subtract, &lhs, &rhs, widths, out, guard)
    }

    /// [`Self::carry_chain`] over operands already cut into limbs at `widths`, each limb paired
    /// with whether it is witnessed, which decides how its hint is read. One list may mix both: a
    /// known constant limb is a pure value beside witnessed ones.
    pub(super) fn carry_chain_over(
        &mut self,
        subtract: bool,
        lhs: &[(ValueId, bool)],
        rhs: &[(ValueId, bool)],
        widths: Vec<usize>,
        out: CarryOut,
        guard: Option<ValueId>,
    ) -> Chain {
        self.carry_chain_bounded(subtract, lhs, rhs, widths, out, guard, TopLimb::Checked)
    }

    /// [`Self::carry_chain_over`], with `top_limb` saying who bounds the answer's top limb.
    #[allow(clippy::too_many_arguments)]
    fn carry_chain_bounded(
        &mut self,
        subtract: bool,
        lhs: &[(ValueId, bool)],
        rhs: &[(ValueId, bool)],
        widths: Vec<usize>,
        out: CarryOut,
        guard: Option<ValueId>,
        top_limb: TopLimb,
    ) -> Chain {
        assert!(
            subtract || out != CarryOut::One,
            "ICE: a sum whose top carry is forced out has no meaning"
        );
        assert!(
            lhs.len() == widths.len() && rhs.len() == widths.len(),
            "ICE: a chain of {} limbs met operands of {} and {}",
            widths.len(),
            lhs.len(),
            rhs.len()
        );

        let (fold, unfold) = if subtract {
            (BinaryArithOpKind::USub, BinaryArithOpKind::UAdd)
        } else {
            (BinaryArithOpKind::UAdd, BinaryArithOpKind::USub)
        };

        // The carry into the next limb: its hint, which the next hint is computed from, and its
        // column, which the next identity reads.
        let mut carry: Option<(ValueId, ValueId)> = None;
        let mut answer = Vec::with_capacity(widths.len());
        for (index, width) in widths.iter().enumerate() {
            let top = index + 1 == widths.len();
            let a = self.cast(lhs[index].0, CastTarget::Field);
            let b = self.cast(rhs[index].0, CastTarget::Field);
            let mut limb = self.bin(fold, a, b);
            if let Some((_, column)) = carry {
                limb = self.bin(fold, limb, column);
            }

            let place = self.field().two_pow(*width);
            let next = if !top || out == CarryOut::Witnessed {
                let hint = self.carry_hint(
                    subtract,
                    lhs[index],
                    rhs[index],
                    carry.map(|(hint, _)| hint),
                    *width,
                );
                let hint_field = self.cast(hint, CastTarget::Field);
                let column = self.write_witness(hint_field);
                self.push_guarded(
                    guard,
                    OpCode::Rangecheck {
                        value: column,
                        max_bits: 1,
                    },
                );
                let place = self.field_const(place);
                let scaled = self.bin(BinaryArithOpKind::UMul, column, place);
                limb = self.bin(unfold, limb, scaled);
                Some((hint, column))
            } else {
                if out == CarryOut::One {
                    let place = self.field_const(place);
                    limb = self.bin(unfold, limb, place);
                }
                None
            };

            if !top || top_limb == TopLimb::Checked {
                self.push_guarded(
                    guard,
                    OpCode::Rangecheck {
                        value: limb,
                        max_bits: *width,
                    },
                );
            }
            answer.push(limb);
            carry = next;
        }

        // The top limb mints a carry only where `out` witnesses one, so that is the only carry
        // that survives the loop.
        Chain {
            answer,
            widths,
            carry_out: carry.map(|(_, column)| column),
        }
    }

    /// The honest carry out of one limb, computed on the pure side at one bit past the limb.
    ///
    /// A sum's carry is its bit `w`; a difference borrows exactly when the subtrahend exceeds the
    /// minuend. Neither reaches `2^(w + 1)`, so neither wraps.
    fn carry_hint(
        &mut self,
        subtract: bool,
        (a, a_witnessed): (ValueId, bool),
        (b, b_witnessed): (ValueId, bool),
        carry_in: Option<ValueId>,
        width: usize,
    ) -> ValueId {
        let wider = CastTarget::Int(width + 1);
        let a = self.pure_of(a, a_witnessed);
        let a = self.cast(a, wider.clone());
        let b = self.pure_of(b, b_witnessed);
        let mut b = self.cast(b, wider.clone());
        if let Some(carry_in) = carry_in {
            let carry_in = self.cast(carry_in, wider);
            b = self.bin(BinaryArithOpKind::UAdd, b, carry_in);
        }

        if subtract {
            let borrow = self.fresh();
            self.push(OpCode::Cmp {
                kind: CmpKind::ULt,
                result: borrow,
                lhs: a,
                rhs: b,
            });
            borrow
        } else {
            let sum = self.bin(BinaryArithOpKind::UAdd, a, b);
            let high = self.shifted_down(sum, width + 1, width);
            self.cast(high, CastTarget::Int(1))
        }
    }

    /// A limb's value on the pure side, where a hint is computed.
    pub(super) fn pure_of(&mut self, limb: ValueId, witnessed: bool) -> ValueId {
        if witnessed {
            self.cast(limb, CastTarget::ValueOf)
        } else {
            limb
        }
    }

    /// An operand of the chain as its `count` limbs at `bits`.
    ///
    /// A witnessed operand still held as one value is below the representation threshold and is
    /// split by [`Self::decompose`], which ties its limbs to it. A pure one is cut on the pure
    /// side, at no constraint cost.
    pub(super) fn chain_operand(
        &mut self,
        value: ValueId,
        bits: usize,
        count: usize,
    ) -> Vec<ValueId> {
        if self.limbs(value).len() == 1 && self.is_witness(value) {
            return self.decompose(value, bits);
        }
        self.operand_limbs(value, count)
    }

    /// [`Self::chain_operand`], each limb paired with whether it is witnessed.
    pub(super) fn chain_limbs(
        &mut self,
        value: ValueId,
        bits: usize,
        count: usize,
    ) -> Vec<(ValueId, bool)> {
        let witnessed = self.is_witness(value);
        self.chain_operand(value, bits, count)
            .into_iter()
            .map(|limb| (limb, witnessed))
            .collect()
    }
}

/// The operation the carry chain lowers, if `op` is one: a sum, difference or ordering under
/// either reading, guarded or not, with a witnessed operand, past [`widest_cell_sum_bits`].
pub(super) fn chained(
    op: &OpCode,
    types: &FunctionTypeInfo,
    field: FieldConfig,
) -> Option<Operands> {
    let (lhs, rhs) = match unguarded(op) {
        OpCode::BinaryArithOp {
            kind:
                BinaryArithOpKind::UAdd
                | BinaryArithOpKind::USub
                | BinaryArithOpKind::SAdd
                | BinaryArithOpKind::SSub,
            lhs,
            rhs,
            ..
        }
        | OpCode::Cmp {
            kind: CmpKind::ULt | CmpKind::SLt,
            lhs,
            rhs,
            ..
        }
        | OpCode::AssertCmp {
            kind: CmpKind::ULt | CmpKind::SLt,
            lhs,
            rhs,
        } => (*lhs, *rhs),
        _ => return None,
    };
    let bits = witnessed_width(types, lhs, rhs)?;
    (bits > widest_cell_sum_bits(field)).then_some(Operands { bits, lhs, rhs })
}
