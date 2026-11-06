use super::{
    super::{types::limb_widths, utils::multi_cell_int_bits},
    Operands, Rewriter,
    lower::{unguarded, witnessed_width},
    sign::Sign,
};

use crate::compiler::{
    analysis::types::FunctionTypeInfo,
    passes::shared::{
        limbs::{ceil_log2, max_pow2_table_size, single_cell_shift_fits},
        shift_guard::emit_pure_shift_amount_check,
    },
    ssa::{
        ValueId,
        hlssa::{BinaryArithOpKind, CastTarget, CmpKind, LookupTarget, OpCode, builder::HLEmitter},
    },
};
use mavros_artifacts::FieldConfig;
use mavros_int_semantics::IntBits;
use num_bigint::BigUint;
use num_traits::{ToPrimitive, Zero};
use std::collections::BTreeSet;
impl Rewriter<'_> {
    /// A shift the single cell cannot hold, returning `true` if `op` was one and was lowered.
    ///
    /// An amount known at compile time is a relabelling of bits, which [`Self::shift_by_constant`]
    /// performs by cutting the operand where its bits land on a limb boundary. Any other amount is
    /// [`Self::shift_by_amount`]'s: a split of every limb at the amount within a limb, and a barrel
    /// over the bits of how many whole limbs it moves. A signed right shift is either, with the
    /// vacated bits filled by [`Self::fill_with_sign`] or by complements around the shift.
    pub(super) fn lower_shift(&mut self, op: &OpCode) -> bool {
        let Some(Operands { bits, lhs, rhs }) = shifted(op, self.types, self.field) else {
            return false;
        };
        let (guard, inner) = self.split_guard(op);
        let OpCode::BinaryArithOp { kind, result, .. } = inner else {
            ice_unreachable!("`shifted` matches only a binary operation");
        };
        // A left shift is one map on the pattern under either reading, so a signed one is this.
        let left = matches!(kind, BinaryArithOpKind::UShl | BinaryArithOpKind::SShl);
        let signed = *kind == BinaryArithOpKind::SShr;
        // A pure operand shifted by a known witness amount has no witness to cut, so it takes the
        // general route, whose every limb is witnessed by the split.
        match self.known_amount(rhs).filter(|_| self.is_witness(lhs)) {
            Some(amount) => match amount.to_usize().filter(|amount| *amount < bits) {
                Some(amount) => {
                    let widths = self.shift_widths(*result, bits);
                    let (mut answer, sign) =
                        self.shift_by_constant(lhs, bits, &widths, left, amount, signed);
                    if let Some(sign) = sign {
                        self.fill_with_sign(&mut answer, &widths, sign, bits, amount);
                    }
                    self.deliver(*result, answer, &widths, bits, None);
                }
                None => self.shift_out_of_range(*result, bits, guard),
            },
            None => {
                let widths = limb_widths(bits, self.limb_bits());
                let answer = self.shift_by_amount(lhs, rhs, bits, left, signed, guard);
                self.deliver(*result, answer, &widths, bits, None);
            }
        }
        true
    }

    /// The whole amount, where every limb of it is known at compile time.
    fn known_amount(&self, amount: ValueId) -> Option<BigUint> {
        let mut value = BigUint::zero();
        for (index, limb) in self.limbs(amount).into_iter().enumerate() {
            value += BigUint::from(&self.known(limb)?) << (index * self.limb_bits());
        }
        Some(value)
    }

    /// A shift by a known amount at or past the width, which refuses the program wherever the guard
    /// is on. The answer is zero, so it is in range where the guard is off.
    fn shift_out_of_range(&mut self, result: ValueId, bits: usize, guard: Option<ValueId>) {
        let (zero, one) = (
            self.field_const(self.field.zero()),
            self.field_const(self.field.one()),
        );
        self.push_guarded(
            guard,
            OpCode::AssertCmp {
                kind: CmpKind::Eq,
                lhs: zero,
                rhs: one,
            },
        );
        let widths = self.shift_widths(result, bits);
        let answer = vec![zero; widths.len()];
        self.deliver(result, answer, &widths, bits, None);
    }

    /// The limb widths a value of `bits` is held at here: its limbs' widths, or `bits` where it is
    /// still one element.
    fn shift_widths(&self, value: ValueId, bits: usize) -> Vec<usize> {
        if self.limbs(value).len() > 1 {
            limb_widths(bits, self.limb_bits())
        } else {
            vec![bits]
        }
    }

    /// `lhs` shifted by an `amount` below `bits` that is known at compile time.
    ///
    /// Every answer bit is an operand bit, so the operand is cut wherever a run of its bits stops
    /// landing inside one answer limb: at its limb boundaries, where its bits meet the answer's
    /// limb boundaries, and where the discarded bits begin. Each piece of a cut limb is witnessed
    /// and range-checked at its width, except the lowest, which is the limb minus the others and
    /// is range-checked too: the limb and the recombined pieces are below `2^w`, so the difference
    /// is either the lowest piece or an element near `p`, which that check rejects. Each answer
    /// limb is then a sum of pieces.
    ///
    /// A limb nothing cuts costs nothing, so an amount that is a whole number of limbs is free.
    ///
    /// The answer is one field element per limb of `answer_widths` (the result's). Where `signed`
    /// asks for it and the shift moves anything, the operand's sign is also returned, for
    /// [`Self::fill_with_sign`]: the piece left by a cut at its top bit or, where the top limb is
    /// known, its top bit.
    fn shift_by_constant(
        &mut self,
        lhs: ValueId,
        bits: usize,
        answer_widths: &[usize],
        left: bool,
        amount: usize,
        signed: bool,
    ) -> (Vec<ValueId>, Option<Sign>) {
        let segments = self.limbs(lhs);
        let segment_widths = self.shift_widths(lhs, bits);
        let signed = signed && amount > 0;
        let top_width = segment_widths[segment_widths.len() - 1];
        let known_top = self.known(segments[segments.len() - 1]);

        // A known top limb's sign is read off its pattern. Cutting it would leave a known answer
        // limb as arithmetic on constants, which `deliver` hands on as a pure value.
        let mut cuts = BTreeSet::from([0, bits, if left { bits - amount } else { amount }]);
        if signed && known_top.is_none() {
            cuts.insert(bits - 1);
        }
        let mut start = 0usize;
        for width in &segment_widths {
            cuts.insert(start);
            start += width;
        }
        let mut start = 0usize;
        for width in answer_widths {
            let source = if left {
                start.checked_sub(amount)
            } else {
                Some(start + amount)
            };
            if let Some(source) = source.filter(|source| *source < bits) {
                cuts.insert(source);
            }
            start += width;
        }

        // Each piece as where it starts in the operand, how wide it is, and its field element.
        let mut pieces: Vec<(usize, usize, ValueId)> = Vec::new();
        let mut start = 0;
        for (segment, width) in segments.into_iter().zip(&segment_widths) {
            let end = start + width;
            let bounds: Vec<usize> = cuts.range(start..=end).copied().collect();
            let whole = self.cast(segment, CastTarget::Field);
            if bounds.len() == 2 {
                pieces.push((start, *width, whole));
            } else {
                pieces.extend(self.cut_segment(segment, whole, *width, start, &bounds));
            }
            start = end;
        }

        let mut answer = Vec::with_capacity(answer_widths.len());
        let mut start = 0;
        for width in answer_widths {
            let end = start + width;
            let mut sum = None;
            for (source, piece_width, piece) in &pieces {
                let landed = if left {
                    source + amount
                } else if let Some(landed) = source.checked_sub(amount) {
                    landed
                } else {
                    continue;
                };
                if landed < start || landed >= end {
                    continue;
                }
                assert!(
                    landed + piece_width <= end,
                    "ICE: a piece of a shifted operand straddles a limb of the answer"
                );
                let scaled = if landed == start {
                    *piece
                } else {
                    let place = self.field_const(self.field.two_pow(landed - start));
                    self.bin(BinaryArithOpKind::UMul, *piece, place)
                };
                sum = Some(match sum {
                    None => scaled,
                    Some(acc) => self.bin(BinaryArithOpKind::UAdd, acc, scaled),
                });
            }
            answer.push(sum.unwrap_or_else(|| self.field_const(self.field.zero())));
            start = end;
        }

        let sign = signed.then(|| match known_top {
            Some(pattern) => Sign::Known(pattern.cast(top_width).bit(top_width - 1) == Some(true)),
            None => {
                let (_, _, bit) = pieces
                    .iter()
                    .find(|(source, width, _)| *source == bits - 1 && *width == 1)
                    .unwrap_or_else(|| ice!("the operand's top bit was not cut out"));
                Sign::Bit(*bit, true)
            }
        });
        (answer, sign)
    }

    /// One witnessed `width`-bit `segment` starting at `start`, cut at `bounds` (which run from
    /// `start` to its end). Returns each piece's start, width and field element.
    ///
    /// A known segment is cut at compile time and costs nothing.
    pub(super) fn cut_segment(
        &mut self,
        segment: ValueId,
        whole: ValueId,
        width: usize,
        start: usize,
        bounds: &[usize],
    ) -> Vec<(usize, usize, ValueId)> {
        let known = self.known(segment);
        let pure = if known.is_none() {
            Some(self.cast(segment, CastTarget::ValueOf))
        } else {
            None
        };

        let mut pieces = Vec::with_capacity(bounds.len() - 1);
        let mut lowest = whole;
        for window in bounds.windows(2).skip(1) {
            let (low, piece_width) = (window[0] - start, window[1] - window[0]);
            let piece = match (&known, pure) {
                (Some(pattern), _) => self.limb_const(&pattern.bit_range(low, piece_width)),
                (None, Some(pure)) => self.witness_window(pure, width, low, piece_width),
                (None, None) => ice_unreachable!("a segment is known or has a pure shadow"),
            };
            let place = self.field_const(self.field.two_pow(low));
            let scaled = self.bin(BinaryArithOpKind::UMul, piece, place);
            lowest = self.bin(BinaryArithOpKind::USub, lowest, scaled);
            pieces.push((window[0], piece_width, piece));
        }

        let lowest_width = bounds[1] - start;
        if known.is_none() {
            self.push(OpCode::Rangecheck {
                value: lowest,
                max_bits: lowest_width,
            });
        }
        pieces.insert(0, (start, lowest_width, lowest));
        pieces
    }

    /// Bits `offset..offset + width` of a pure `value` of `bits`, written to a witness column and
    /// range-checked at `width`, as a field element.
    fn witness_window(
        &mut self,
        value: ValueId,
        bits: usize,
        offset: usize,
        width: usize,
    ) -> ValueId {
        let hint = self.shifted_down(value, bits, offset);
        let narrowed = self.cast(hint, CastTarget::Int(width));
        let hint_field = self.cast(narrowed, CastTarget::Field);
        let column = self.write_witness(hint_field);
        self.push(OpCode::Rangecheck {
            value: column,
            max_bits: width,
        });
        column
    }

    /// `lhs` shifted by an amount not known at compile time.
    ///
    /// **The amount** is `n = q·h + r`, with `r` below the limb width `h` and `q` the number of
    /// whole limbs it moves. Its low limb is witnessed as `h·Σ e_t·2^t + r` with every `e_t` a bit
    /// and `r` read out of the powers-of-two table beside its power `f = 2^r`, which bounds `r`
    /// below `h`; its other limbs are held to zero. With `s` bits of `q`, that is `n < h·2^s`
    /// already, and `n < bits` exactly where `h·2^s` is the width; elsewhere `bits - 1 - n` is
    /// range-checked as well.
    ///
    /// **Each limb** `x_i` is split as `x_i·m = hi_i·2^h + lo_i` with both halves range-checked,
    /// where `m` is `f` for `<<` and `2^(h - r)` for `>>`, pinned by `m·f == 2^h`. As `x_i·m` is
    /// below `2^2h`, which [`witness_limb_bits`] keeps injective, the split is unique. Answer
    /// limb `j` before the whole limbs move is `lo_j + hi_(j-1)` for `<<` or `hi_j + lo_(j+1)` for
    /// `>>`; the two halves cover disjoint bits, so neither sum carries.
    ///
    /// **The whole limbs** move by `q` through a barrel: stage `t` moves every limb `2^t` places
    /// where `e_t` is set, each of its limbs being `a + e_t·(b - a)`. For a `<<` whose top limb is
    /// narrower than `h`, that limb is not split: its `x·m` enters the barrel whole, and a final
    /// cut keeps the answer's low `w` bits and range-checks the `h` above them.
    ///
    /// Under a guard a witnessed amount's lowest limb is zero where the guard is off, so every
    /// check holds there, and the operand's limbs are in range on every path. Holding the amount's
    /// other limbs to zero is the one check that needs the guard. A pure amount is decomposed on
    /// the pure side, by [`Self::pure_shift_amount`], and every product above is by a constant.
    ///
    /// **A sign-filling right shift**, where `signed`, is the operand [`Self::complement`]ed by its
    /// sign, shifted as above, and the answer complemented back. For a negative operand that is
    /// `!(!lhs >> n)`, and `!lhs` is not negative, so complementing back fills the vacated bits
    /// with ones.
    ///
    /// The answer is one field element per limb.
    fn shift_by_amount(
        &mut self,
        lhs: ValueId,
        rhs: ValueId,
        bits: usize,
        left: bool,
        signed: bool,
        guard: Option<ValueId>,
    ) -> Vec<ValueId> {
        let limb_bits = self.limb_bits();
        let widths = limb_widths(bits, limb_bits);
        let count = widths.len();
        let stages = ceil_log2(count);
        let (e, factor, factor_hint) = self.shift_amount(rhs, bits, stages, left, guard);

        // The split of every limb, as its two halves' field elements.
        let two_pow_limb = self.field_const(self.field.two_pow(limb_bits));
        let hint_width = CastTarget::Int(2 * limb_bits);
        let mut operand = self.chain_limbs(lhs, bits, count);
        let sign = signed.then(|| self.operand_sign(&operand, &widths));
        if let Some(sign) = sign {
            operand = operand
                .into_iter()
                .zip(&widths)
                .map(|(limb, width)| self.complement(limb, *width, sign))
                .collect();
        }
        let mut halves = Vec::with_capacity(count);
        let top_width = widths[count - 1];
        let narrow_top = left && top_width < limb_bits;
        for (index, (&(limb, witnessed), width)) in operand.iter().zip(&widths).enumerate() {
            if self.known(limb).is_some_and(|pattern| pattern.is_zero()) {
                let zero = self.field_const(self.field.zero());
                halves.push((zero, zero));
                continue;
            }
            let limb_field = self.cast(limb, CastTarget::Field);
            let product = self.bin(BinaryArithOpKind::UMul, limb_field, factor);

            // `<<` moves at most `h - 1` places, so it pushes one bit fewer out of the limb. A
            // narrow top limb's high half is discarded, and the cut below splits it anyway.
            let high_bits = if left { width - 1 } else { *width };
            if high_bits == 0 || (narrow_top && index + 1 == count) {
                halves.push((self.field_const(self.field.zero()), product));
                continue;
            }
            let pure = self.pure_of(limb, witnessed);
            let pure = self.cast(pure, hint_width.clone());
            let pure = self.bin(BinaryArithOpKind::UMul, pure, factor_hint);
            let high = self.witness_window(pure, 2 * limb_bits, limb_bits, high_bits);
            let scaled = self.bin(BinaryArithOpKind::UMul, high, two_pow_limb);
            let low = self.bin(BinaryArithOpKind::USub, product, scaled);
            self.push(OpCode::Rangecheck {
                value: low,
                max_bits: limb_bits,
            });
            halves.push((high, low));
        }

        // The answer's limbs before the whole limbs move.
        let zero = self.field_const(self.field.zero());
        let mut limbs: Vec<ValueId> = (0..count)
            .map(|index| {
                let (own, neighbour) = if left {
                    (
                        halves[index].1,
                        index.checked_sub(1).map(|below| halves[below].0),
                    )
                } else {
                    (halves[index].0, halves.get(index + 1).map(|above| above.1))
                };
                match neighbour {
                    Some(neighbour) => self.bin(BinaryArithOpKind::UAdd, own, neighbour),
                    None => own,
                }
            })
            .collect();

        for (stage, bit) in e.iter().enumerate() {
            let step = 1 << stage;
            limbs = (0..count)
                .map(|index| {
                    let stay = limbs[index];
                    let from = if left {
                        index.checked_sub(step).map(|from| limbs[from])
                    } else {
                        limbs.get(index + step).copied()
                    };
                    let from = from.unwrap_or(zero);
                    let moved = self.bin(BinaryArithOpKind::USub, from, stay);
                    let moved = self.bin(BinaryArithOpKind::UMul, *bit, moved);
                    self.bin(BinaryArithOpKind::UAdd, stay, moved)
                })
                .collect();
        }

        // The top limb of a `<<` has to be below its width. It was never split, so it can be
        // `x_top·f + hi`, below `2^(w + h)`, and everything above `w` is cut away here.
        if narrow_top {
            let top = limbs[count - 1];
            let reach = top_width + limb_bits;
            let pure = self.cast(top, CastTarget::ValueOf);
            let pure = self.cast(pure, CastTarget::Int(reach));
            let high = self.witness_window(pure, reach, top_width, limb_bits);
            let place = self.field_const(self.field.two_pow(top_width));
            let scaled = self.bin(BinaryArithOpKind::UMul, high, place);
            let low = self.bin(BinaryArithOpKind::USub, top, scaled);
            self.push(OpCode::Rangecheck {
                value: low,
                max_bits: top_width,
            });
            limbs[count - 1] = low;
        }

        match sign {
            Some(sign) => {
                let answer = self.answer_limbs(limbs, &widths);
                let restored = answer
                    .into_iter()
                    .zip(&widths)
                    .map(|(limb, width)| self.complement(limb, *width, sign))
                    .collect();
                self.limb_fields(restored)
            }
            None => limbs,
        }
    }

    /// The amount of a [`Self::shift_by_amount`]: the `stages` bits of how many whole limbs it
    /// moves, and the multiplier that splits a limb, as a field element and as a pure hint at twice
    /// the limb width.
    fn shift_amount(
        &mut self,
        rhs: ValueId,
        bits: usize,
        stages: usize,
        left: bool,
        guard: Option<ValueId>,
    ) -> (Vec<ValueId>, ValueId, ValueId) {
        let limb_bits = self.limb_bits();
        let log_limb = limb_bits.trailing_zeros() as usize;
        assert!(
            limb_bits.is_power_of_two() && log_limb <= max_pow2_table_size(self.field),
            "ICE: a {limb_bits}-bit limb has no powers-of-two table to split it at an amount"
        );

        if !self.is_witness(rhs) {
            return self.pure_shift_amount(rhs, bits, stages, left, guard);
        }
        // The amount is as wide as the value, so it is held in limbs exactly where the value is.
        let (low, rest, low_bits) = if bits > multi_cell_int_bits(self.field) {
            let widths = limb_widths(bits, limb_bits);
            let limbs = self.operand_limbs(rhs, widths.len());
            (limbs[0], limbs[1..].to_vec(), widths[0])
        } else {
            (rhs, Vec::new(), bits)
        };

        // Every limb above the lowest is zero, or the amount is past any width: every width is
        // below `2^h`, the most the lowest limb holds.
        let zero = self.field_const(self.field.zero());
        for limb in rest {
            let limb = self.cast(limb, CastTarget::Field);
            self.constrain_equal(limb, zero, guard);
        }

        // Where the guard is off the amount is zero, which every check below admits.
        let mut low_field = self.cast(low, CastTarget::Field);
        let mut low_pure = self.cast(low, CastTarget::ValueOf);
        if let Some(condition) = guard {
            let flag = self.cast(condition, CastTarget::Field);
            low_field = self.bin(BinaryArithOpKind::UMul, flag, low_field);
            let pure_condition = self.pure_of(condition, self.is_witness(condition));
            let pure_zero = self.int_const(IntBits::zero(low_bits));
            low_pure = self.select(pure_condition, low_pure, pure_zero);
        }
        let hints = self.amount_hints(low_pure, low_bits, stages);

        // `n = h·Σ e_t·2^t + r`, each `e_t` a bit and `r` a key of the table.
        let within_field = self.cast(hints.within, CastTarget::Field);
        let within_column = self.write_witness(within_field);
        let mut sum = within_column;
        let mut e = Vec::with_capacity(stages);
        for (stage, hint) in hints.stages.into_iter().enumerate() {
            let hint = self.cast(hint, CastTarget::Field);
            let bit = self.write_witness(hint);
            self.push(OpCode::Rangecheck {
                value: bit,
                max_bits: 1,
            });
            let place = self.field_const(self.field.two_pow(log_limb + stage));
            let scaled = self.bin(BinaryArithOpKind::UMul, bit, place);
            sum = self.bin(BinaryArithOpKind::UAdd, sum, scaled);
            e.push(bit);
        }
        self.constrain_equal(low_field, sum, None);

        // Where `h·2^stages` is not the width, the decomposition alone does not bound the amount
        // below it, so it is checked explicitly.
        if limb_bits << stages != bits {
            let largest = self.field_const(self.field.constant((bits - 1) as u64));
            let headroom = self.bin(BinaryArithOpKind::USub, largest, low_field);
            self.push(OpCode::Rangecheck {
                value: headroom,
                max_bits: ceil_log2(bits),
            });
        }

        // `f = 2^r`, which the table pins against `r` and so bounds `r` below `h`. Unguarded, as
        // `r` is already zero where the guard is off.
        let power_field = self.cast(hints.power, CastTarget::Field);
        let power = self.write_witness(power_field);
        let one_flag = self.field_const(self.field.one());
        self.push(OpCode::Lookup {
            target: LookupTarget::Pow2(log_limb as u8),
            args: vec![within_column, power],
            flag: one_flag,
        });
        if left {
            return (e, power, hints.power);
        }

        // `2^(h - r)`, pinned by its product with `f`: `2^h` is not zero, so neither is `f`, and
        // the cofactor is its unique quotient.
        let cofactor_hint = self.cofactor_hint(hints.power);
        let cofactor_field = self.cast(cofactor_hint, CastTarget::Field);
        let cofactor = self.write_witness(cofactor_field);
        let two_pow_limb = self.field_const(self.field.two_pow(limb_bits));
        self.push(OpCode::Constrain {
            a: power,
            b: cofactor,
            c: two_pow_limb,
        });
        (e, cofactor, cofactor_hint)
    }

    /// The pure side of an amount `n = q·h + r` of `amount_bits`, decomposed the same way by both
    /// routes: each of the `stages` bits of `q` as a one-bit integer, `r`, and `2^r` at twice the
    /// limb width.
    ///
    /// Only the low `log2(h) + stages` bits are read, so a larger amount is taken modulo
    /// `h·2^stages`; the caller's check rules that out.
    fn amount_hints(&mut self, amount: ValueId, amount_bits: usize, stages: usize) -> AmountHints {
        let limb_bits = self.limb_bits();
        let log_limb = limb_bits.trailing_zeros() as usize;
        let stages = (0..stages)
            .map(|stage| {
                let bit = self.shifted_down(amount, amount_bits, log_limb + stage);
                self.cast(bit, CastTarget::Int(1))
            })
            .collect();
        let within = self.cast(amount, CastTarget::Int(log_limb));
        let hint_bits = 2 * limb_bits;
        let one = self.int_const(IntBits::one(hint_bits));
        let within_wide = self.cast(within, CastTarget::Int(hint_bits));
        let power = self.bin(BinaryArithOpKind::UShl, one, within_wide);
        AmountHints {
            stages,
            within,
            power,
        }
    }

    /// `2^(h - r)` on the pure side, from `2^r` at twice the limb width.
    fn cofactor_hint(&mut self, power: ValueId) -> ValueId {
        let limb_bits = self.limb_bits();
        let two_pow_limb = self.two_pow_const(2 * limb_bits, limb_bits);
        self.bin(BinaryArithOpKind::UDiv, two_pow_limb, power)
    }

    /// [`Self::shift_amount`] for a pure amount, which is known wherever the constraints are built,
    /// so nothing is witnessed: its check is [`emit_pure_shift_amount_check`]'s comparison, and its
    /// bits and multiplier are pure, so each split and barrel stage is linear.
    ///
    /// Under a guard only the comparison is guarded. The amount is not zeroed where the guard is
    /// off; the split and barrel still hold, and the answer there is the operand shifted by that
    /// amount modulo `h·2^stages`.
    fn pure_shift_amount(
        &mut self,
        rhs: ValueId,
        bits: usize,
        stages: usize,
        left: bool,
        guard: Option<ValueId>,
    ) -> (Vec<ValueId>, ValueId, ValueId) {
        emit_pure_shift_amount_check(self, guard, rhs, bits);

        let hints = self.amount_hints(rhs, bits, stages);
        let e = hints
            .stages
            .into_iter()
            .map(|bit| self.cast(bit, CastTarget::Field))
            .collect();
        let multiplier = if left {
            hints.power
        } else {
            self.cofactor_hint(hints.power)
        };
        let multiplier_field = self.cast(multiplier, CastTarget::Field);
        (e, multiplier_field, multiplier)
    }
}

/// The pure side of a shift amount, from [`Rewriter::amount_hints`].
struct AmountHints {
    /// Each bit of how many whole limbs the amount moves, as a one-bit integer.
    stages: Vec<ValueId>,

    /// The amount within a limb, `r`, at `log2(h)` bits.
    within: ValueId,

    /// `2^r`, at twice the limb width.
    power: ValueId,
}

/// The shift the representation lowers, if `op` is one, under either reading, guarded or not, with
/// a witnessed operand, at a width the single cell does not take.
pub(super) fn shifted(
    op: &OpCode,
    types: &FunctionTypeInfo,
    field: FieldConfig,
) -> Option<Operands> {
    let OpCode::BinaryArithOp { kind, lhs, rhs, .. } = unguarded(op) else {
        return None;
    };
    let left = match kind {
        BinaryArithOpKind::UShl | BinaryArithOpKind::SShl => true,
        BinaryArithOpKind::UShr | BinaryArithOpKind::SShr => false,
        _ => return None,
    };
    let bits = witnessed_width(types, *lhs, *rhs)?;
    (!single_cell_shift_fits(field, bits, left)).then_some(Operands {
        bits,
        lhs: *lhs,
        rhs: *rhs,
    })
}
