use crate::compiler::{
    passes::shared::limbs::half_limb_bits,
    ssa::{
        ValueId,
        hlssa::{BinaryArithOpKind, CastTarget, OpCode, builder::HLEmitter},
    },
};

use super::{
    super::types::{limb_types, limb_widths},
    Rewriter,
};

use mavros_int_semantics::IntBits;

impl Rewriter<'_> {
    /// A witnessed spread whose answer is held as limbs, under `guard` where there is one.
    ///
    /// A spread is limb-local: each half-limb of the operand spreads into one answer limb, by the
    /// narrow spread the witness lowering builds, its lookup keyed by the half-limb. The lookups
    /// also bound the half-limbs [`Self::cut_unbounded`] leaves unbounded, even for an answer used
    /// only in part, as dead-code elimination keeps every witnessed spread until its lookup is
    /// built. Only the half-limbs the read reaches are cut, the one it straddles spreads only the
    /// bits the read takes, and anything above the read inside a cut is refused. A limb known to
    /// be zero spreads to zero and costs nothing.
    pub(super) fn lower_spread(
        &mut self,
        guard: Option<ValueId>,
        result: ValueId,
        value: ValueId,
        value_bits: usize,
    ) {
        let bits = self
            .types
            .get_value_type(value)
            .int_width()
            .unwrap_or_else(|| ice!("a spread of a non-integer reached the representation"));
        let containers = limb_widths(bits, half_limb_bits(self.limb_bits()));
        let results = self.limbs(result);
        assert_eq!(
            results.len(),
            containers.len(),
            "ICE: an int{bits} spread is a limb per half-limb, not {} for {}",
            results.len(),
            containers.len()
        );

        // `None` is a piece of a limb known to be zero, which `cut_unbounded` would cut into as
        // many pieces as the read reaches half-limbs.
        let mut pieces: Vec<Option<(ValueId, usize)>> = Vec::with_capacity(containers.len());
        if self.wide_width(value).is_some() {
            let half = half_limb_bits(self.limb_bits());
            for (limb, width, read) in self.limbs_and_reads(value, bits, value_bits) {
                let zero = self.known(limb).is_some_and(|pattern| pattern.is_zero());
                match read {
                    0 if zero => {}
                    0 => self.constrain_zero(limb, guard),
                    read if zero => pieces.extend(std::iter::repeat_n(None, read.div_ceil(half))),
                    read => pieces.extend(
                        self.cut_unbounded(limb, width, read, guard)
                            .into_iter()
                            .map(Some),
                    ),
                }
            }
        } else {
            pieces = self
                .cut_unbounded(value, bits, value_bits, guard)
                .into_iter()
                .map(Some)
                .collect();
        }

        for (index, (result, container)) in results.into_iter().zip(containers).enumerate() {
            match pieces.get(index) {
                Some(&Some((piece, read))) => self.push_guarded(
                    guard,
                    OpCode::Spread {
                        result,
                        value: piece,
                        value_bits: read,
                    },
                ),
                Some(None) | None => self.define_zero(result, 2 * container),
            }
        }
    }

    /// A witnessed unspread of an operand held as limbs, under `guard` where there is one.
    ///
    /// Limb-local in the other direction: each operand limb unspreads into a half-limb of each
    /// stream, by the narrow unspread the witness lowering already builds, whose lookups bound
    /// both. The half-limbs are then reassembled into the streams, which may be limbs or one
    /// element each.
    ///
    /// A limb the read straddles unspreads only the bits the read takes. A limb the read takes a
    /// single bit of has no odd bit to unspread, so it is range-checked to one bit and is itself
    /// the even half-limb. A limb wholly above the read is held to zero. A limb known to be zero
    /// unspreads to zero and costs nothing.
    pub(super) fn lower_unspread(
        &mut self,
        guard: Option<ValueId>,
        result_odd: ValueId,
        result_even: ValueId,
        value: ValueId,
        value_bits: usize,
    ) {
        let bits = self.wide_width(value).unwrap_or_else(|| {
            ice!(
                "an unspread whose operand is one element answers streams that are one element too"
            )
        });
        let mut odd_parts = Vec::new();
        let mut even_parts = Vec::new();

        for (limb, width, read) in self.limbs_and_reads(value, bits, value_bits) {
            let (odd_width, even_width) = (width / 2, width.div_ceil(2));
            let zero = self.known(limb).is_some_and(|pattern| pattern.is_zero());
            let (odd, even) = match read {
                _ if zero => (None, self.zero_limb(even_width)),
                0 => {
                    self.constrain_zero(limb, guard);
                    (None, self.zero_limb(even_width))
                }
                1 => {
                    let limb_field = self.cast(limb, CastTarget::Field);
                    // Under a guard the bit is witnessed and tied to the limb under it: where the
                    // guard is off the limb may hold more than a bit, which an `Int` of the even
                    // half-limb's width would truncate.
                    let bit = match guard {
                        None => limb_field,
                        Some(_) => {
                            let pure = self.cast(limb, CastTarget::ValueOf);
                            let low = self.cast(pure, CastTarget::Int(1));
                            let low_field = self.cast(low, CastTarget::Field);
                            let bit = self.write_witness(low_field);
                            self.constrain_equal(limb_field, bit, guard);
                            bit
                        }
                    };
                    self.push_guarded(
                        guard,
                        OpCode::Rangecheck {
                            value: bit,
                            max_bits: 1,
                        },
                    );
                    (None, self.relabel(bit, even_width))
                }
                _ => {
                    let (odd, even) = (self.fresh(), self.fresh());
                    self.push_guarded(
                        guard,
                        OpCode::Unspread {
                            result_odd: odd,
                            result_even: even,
                            value: limb,
                            value_bits: read,
                        },
                    );
                    (Some(odd), even)
                }
            };
            // A one-bit top limb has an even bit and no odd one, so it adds no odd half-limb.
            if odd_width > 0 {
                let odd = odd.unwrap_or_else(|| self.zero_limb(odd_width));
                odd_parts.push((odd, odd_width));
            }
            even_parts.push((even, even_width));
        }

        self.assemble_half_limbs(result_odd, &odd_parts);
        self.assemble_half_limbs(result_even, &even_parts);
    }

    /// The limbs of `value`, an `int(bits)` held as limbs, low first, each with its width and the
    /// number of its bits that a read of the low `read` bits takes, which is zero for a limb wholly
    /// above the read.
    fn limbs_and_reads(
        &self,
        value: ValueId,
        bits: usize,
        read: usize,
    ) -> Vec<(ValueId, usize, usize)> {
        let limb_bits = self.limb_bits();
        self.limbs(value)
            .into_iter()
            .zip(limb_widths(bits, limb_bits))
            .enumerate()
            .map(|(index, (limb, width))| {
                (
                    limb,
                    width,
                    read.saturating_sub(index * limb_bits).min(width),
                )
            })
            .collect()
    }

    /// The half-limbs of a `bits`-wide witnessed `value` that the low `read` bits reach, low first,
    /// each typed at its width and paired with how many of its bits the read takes. **None is
    /// bounded**: the caller bounds each.
    ///
    /// The high pieces are witnessed from the pure shadow and the lowest is the source minus them,
    /// so the pieces recombine to the source; bits above the read land in the lowest. Once each is
    /// bounded at its read, their sum is below `2^read`.
    ///
    /// Without a guard the lowest piece is derived, and its narrow lookup bounds it. Under a guard
    /// it is witnessed too, and tied to the source under the guard: where the guard is off nothing
    /// bounds a derived piece, so one carrying the bits above the read would sit in an `Int` it
    /// does not fit, and the VM's truncating cast would disagree with the constraint system on the
    /// witness.
    fn cut_unbounded(
        &mut self,
        value: ValueId,
        bits: usize,
        read: usize,
        guard: Option<ValueId>,
    ) -> Vec<(ValueId, usize)> {
        let half = half_limb_bits(self.limb_bits());
        let containers = limb_widths(bits, half);
        let reads = limb_widths(read, half);
        if containers.len() == 1 {
            return vec![(value, read)];
        }

        let pure = self.cast(value, CastTarget::ValueOf);
        let source = self.cast(value, CastTarget::Field);
        let first = if guard.is_some() { 0 } else { 1 };
        let mut rest = source;
        let mut sum = None;
        let mut pieces = Vec::with_capacity(reads.len());
        for (index, piece_read) in reads.iter().enumerate().skip(first) {
            let hint = self.shifted_down(pure, bits, index * half);
            let narrowed = self.cast(hint, CastTarget::Int(*piece_read));
            let hint_field = self.cast(narrowed, CastTarget::Field);
            let written = self.write_witness(hint_field);
            let placed = if index == 0 {
                written
            } else {
                let place = self.field_const(self.field().two_pow(index * half));
                self.bin(BinaryArithOpKind::UMul, written, place)
            };
            if guard.is_some() {
                sum = Some(match sum {
                    None => placed,
                    Some(acc) => self.bin(BinaryArithOpKind::UAdd, acc, placed),
                });
            } else {
                rest = self.bin(BinaryArithOpKind::USub, rest, placed);
            }
            pieces.push((self.relabel(written, containers[index]), *piece_read));
        }
        match sum {
            Some(sum) => self.constrain_equal(source, sum, guard),
            None => {
                let lowest = self.relabel(rest, containers[0]);
                pieces.insert(0, (lowest, reads[0]));
            }
        }
        pieces
    }

    /// Half-limbs, low first, recombined into `result`, which spans their total width and is held
    /// as limbs or as one element.
    ///
    /// Every piece but the last is a full half-limb, so each lands at a multiple of one; each is
    /// already bounded, so the sum is below `2^width` and needs no separate constraint.
    fn assemble_half_limbs(&mut self, result: ValueId, parts: &[(ValueId, usize)]) {
        let half = half_limb_bits(self.limb_bits());
        let targets = self.limbs(result);
        let target_widths: Vec<usize> =
            limb_types(self.types.get_value_type(result), self.limb_config())
                .iter()
                .map(|ty| ty.int_width().expect("a stream is an integer"))
                .collect();
        let groups: Vec<&[(ValueId, usize)]> = if targets.len() == 1 {
            vec![parts]
        } else {
            parts.chunks(self.limb_bits() / half).collect()
        };
        assert_eq!(
            groups.len(),
            targets.len(),
            "ICE: {} groups of half-limbs were assembled into a stream of {} limbs",
            groups.len(),
            targets.len()
        );

        for ((target, target_width), group) in targets.into_iter().zip(target_widths).zip(groups) {
            let width: usize = group.iter().map(|(_, width)| width).sum();
            assert_eq!(
                width, target_width,
                "ICE: half-limbs of {width} bits were assembled into an int{target_width}"
            );
            let mut sum = None;
            for (index, (part, _)) in group.iter().enumerate() {
                let part_field = self.cast(*part, CastTarget::Field);
                let placed = if index == 0 {
                    part_field
                } else {
                    let place = self.field_const(self.field().two_pow(index * half));
                    self.bin(BinaryArithOpKind::UMul, part_field, place)
                };
                sum = Some(match sum {
                    None => placed,
                    Some(acc) => self.bin(BinaryArithOpKind::UAdd, acc, placed),
                });
            }
            self.push(OpCode::Cast {
                result: target,
                value: sum.expect("a limb is at least one half-limb"),
                target: CastTarget::IntUnchecked(width),
            });
        }
    }

    /// Hold a witnessed integer to zero, where `guard` holds.
    fn constrain_zero(&mut self, value: ValueId, guard: Option<ValueId>) {
        let value_field = self.cast(value, CastTarget::Field);
        let zero = self.field_const(self.field().zero());
        self.constrain_equal(value_field, zero, guard);
    }

    /// Define `result` as a witnessed zero of `width` bits, which is a constant and pinned by being
    /// one: [`Self::zero_limb`] into a value the caller names.
    pub(super) fn define_zero(&mut self, result: ValueId, width: usize) {
        let zero = self.int_const(IntBits::zero(width));
        self.push(OpCode::Cast {
            result,
            value: zero,
            target: CastTarget::WitnessOf,
        });
    }
}
