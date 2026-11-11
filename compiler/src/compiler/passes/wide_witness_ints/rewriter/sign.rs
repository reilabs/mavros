use super::{
    super::types::limb_widths,
    CarryOut, Rewriter,
    division::Answer,
    product::{Factor, Limb, Schoolbook, full_bounds},
};

use crate::compiler::{
    passes::shared::overflow_guard::abs_as_u,
    ssa::{
        ValueId,
        hlssa::{BinaryArithOpKind, CastTarget, builder::HLEmitter},
    },
};
use mavros_int_semantics::IntBits;
use num_bigint::BigUint;

/// A sign bit as the signed gadgets read it.
#[derive(Clone, Copy, Debug)]
pub(super) enum Sign {
    /// Known at compile time.
    Known(bool),

    /// A field element that is zero or one, with whether it is witnessed.
    Bit(ValueId, bool),
}

impl Rewriter<'_> {
    /// A signed product the single cell cannot hold.
    ///
    /// The unsigned schoolbook over the operands' magnitudes, whose overflow check keeps the
    /// product below `2^N`; then [`Self::check_fits`] under the product's sign; then the product
    /// negated by that sign. Every check past the operands is guarded, except the cut of the
    /// product's top bit, which reads a product that is zero where the guard is off; the answer is
    /// zero there too, as the unsigned product's is.
    pub(super) fn lower_signed_product(
        &mut self,
        result: ValueId,
        lhs: ValueId,
        rhs: ValueId,
        bits: usize,
        guard: Option<ValueId>,
    ) {
        let widths = limb_widths(bits, self.limb_bits());
        let (lhs_magnitude, lhs_sign, _) = self.magnitude(lhs, bits, &widths, guard);
        let (rhs_magnitude, rhs_sign, _) = if lhs == rhs {
            (lhs_magnitude.clone(), lhs_sign, None)
        } else {
            self.magnitude(rhs, bits, &widths, guard)
        };
        let lhs_factor = self.factor_of(&lhs_magnitude, &widths);
        let rhs_factor = self.factor_of(&rhs_magnitude, &widths);
        let product = self.schoolbook(Schoolbook {
            lhs: &lhs_factor,
            rhs: &rhs_factor,
            addend: None,
            pinned: None,
            widths: &widths,
            bits,
            operation: "multiplication",
            guard,
        });

        // Off the guard the schoolbook's checks are off, so a top column the untaken branch
        // overflowed is past its width. The negation reads that column through a cast to a limb's
        // width, which truncates on the pure side while the constraints carry it whole, so the
        // negation's products would disagree with their columns. The product is zero there
        // instead.
        let product = match guard {
            Some(condition) => {
                let zero = self.field_const(self.field().zero());
                product
                    .into_iter()
                    .map(|limb| self.select(condition, limb, zero))
                    .collect()
            }
            None => product,
        };
        let product = self.answer_limbs(product, &widths);

        // A square is never negative.
        let sign = if lhs == rhs {
            Sign::Known(false)
        } else {
            self.xor_signs(lhs_sign, rhs_sign)
        };
        self.check_fits(&product, &widths, sign, guard);
        let answer = self.negate_if(product, &widths, sign, guard);
        let answer = self.limb_fields(answer);
        self.deliver(result, answer, &widths, bits, guard);
    }

    /// The signed quotient and remainder of `dividend / divisor`, each as its magnitude and the
    /// sign [`Self::answer`] negates it by.
    ///
    /// The unsigned division of the magnitudes, which refuses a zero divisor, then Noir's
    /// truncation toward zero: the quotient's sign is the XOR of the operands' signs, and the
    /// remainder takes the dividend's. The quotient's magnitude is held to fit under its sign for
    /// both answers, as the model refuses `INT_MIN % -1` along with `INT_MIN / -1`. The remainder's
    /// always fits, being below the divisor's magnitude and so below `2^(N - 1)`.
    pub(super) fn divide_signed(
        &mut self,
        dividend: ValueId,
        divisor: ValueId,
        bits: usize,
        widths: &[usize],
        guard: Option<ValueId>,
    ) -> (Answer, Answer) {
        let (n, dividend_sign, pure_dividend) = self.magnitude(dividend, bits, widths, guard);
        let (d, divisor_sign, pure_divisor) = self.magnitude(divisor, bits, widths, guard);
        let n = self.factor_of(&n, widths);
        let d = self.factor_of(&d, widths);
        let mut pure_side = |operand: ValueId, magnitude: Option<ValueId>| {
            magnitude.unwrap_or_else(|| {
                let whole = self.pure_whole(operand, bits);
                self.pure_magnitude(whole, bits).0
            })
        };

        let pure_dividend = pure_side(dividend, pure_dividend);
        let pure_divisor = pure_side(divisor, pure_divisor);
        let (q, r) = self.divide_factors(&n, &d, pure_dividend, pure_divisor, bits, widths, guard);
        let q = self.chain_limbs_of(&q, widths.len());
        let r = self.chain_limbs_of(&r, widths.len());

        let quotient_sign = self.xor_signs(dividend_sign, divisor_sign);
        self.check_fits(&q, widths, quotient_sign, guard);
        (
            Answer::Signed(q, quotient_sign),
            Answer::Signed(r, dividend_sign),
        )
    }

    /// Fill the bits vacated by a right shift by a known nonzero `amount` with the operand's
    /// `sign`, which [`Self::shift_by_constant`] cut out, in an `answer` with limbs at `widths`.
    ///
    /// The unsigned answer has those bits clear, so filling adds `s` times each limb's share of
    /// `2^N - 2^(N - amount)`: linear in the sign, and each limb stays below its width. A sign
    /// known to be set fills a known limb at compile time, so it stays a constant, which
    /// [`Self::deliver`] hands on as a witnessed constant.
    pub(super) fn fill_with_sign(
        &mut self,
        answer: &mut [ValueId],
        widths: &[usize],
        sign: Sign,
        bits: usize,
        amount: usize,
    ) {
        if let Sign::Known(false) = sign {
            return;
        }

        let vacated = bits - amount;
        let mut start = 0;
        for (limb, width) in answer.iter_mut().zip(widths) {
            let end = start + width;
            let low = vacated.max(start);
            if low < end {
                let pattern = IntBits::all_ones(end - low)
                    .cast(*width)
                    .shifted_left(low - start);
                *limb = match (sign, self.constant_limb(*limb, *width)) {
                    (Sign::Known(true), Some(known)) => self.limb_const(&known.or(&pattern)),
                    (Sign::Known(true), None) => {
                        let fill = self.limb_const(&pattern);
                        self.bin(BinaryArithOpKind::UAdd, *limb, fill)
                    }
                    (Sign::Bit(bit, _), _) => {
                        let fill = self.limb_const(&pattern);
                        let scaled = self.bin(BinaryArithOpKind::UMul, bit, fill);
                        self.bin(BinaryArithOpKind::UAdd, *limb, scaled)
                    }
                    (Sign::Known(false), _) => ice_unreachable!("a clear sign fills nothing"),
                };
            }
            start = end;
        }
    }

    /// The magnitude of a `bits`-wide operand read as signed: its limbs at `widths`, each with
    /// whether it is witnessed; its sign; and, for a pure operand, the whole magnitude the limbs
    /// were cut from, which a division's hint reads.
    ///
    /// `|INT_MIN|` is `2^(N - 1)`, which `N` unsigned bits hold, so every magnitude fits its
    /// width. A constant's is taken at compile time, so limbs it does not reach stay known zeros;
    /// a pure operand's is taken on the pure side, at no cost; and a witnessed one is its limbs
    /// [`Self::negate_if`] by its sign.
    fn magnitude(
        &mut self,
        value: ValueId,
        bits: usize,
        widths: &[usize],
        guard: Option<ValueId>,
    ) -> (Vec<(ValueId, bool)>, Sign, Option<ValueId>) {
        let limb_bits = self.limb_bits();
        if let Some(pattern) = self.known(value) {
            let signed = pattern.cast(bits).to_signed();
            let magnitude = IntBits::from_biguint(bits, signed.magnitude());
            let limbs = widths
                .iter()
                .enumerate()
                .map(|(index, width)| {
                    let limb = magnitude.bit_range(index * limb_bits, *width);
                    (self.int_const(limb), false)
                })
                .collect();
            let negative = signed.sign() == num_bigint::Sign::Minus;
            return (limbs, Sign::Known(negative), None);
        }

        if !self.is_witness(value) {
            let (whole, sign) = self.pure_magnitude(value, bits);
            let limbs = widths
                .iter()
                .enumerate()
                .map(|(index, width)| {
                    let shifted = self.shifted_down(whole, bits, index * limb_bits);
                    (self.cast(shifted, CastTarget::Int(*width)), false)
                })
                .collect();
            let sign = self.cast(sign, CastTarget::Field);
            return (limbs, Sign::Bit(sign, false), Some(whole));
        }

        let limbs = self.chain_limbs(value, bits, widths.len());
        let sign = self.operand_sign(&limbs, widths);
        (self.negate_if(limbs, widths, sign, guard), sign, None)
    }

    /// `|value|` for a pure `value` of `bits` read as signed, as a pure value of `bits`, and its
    /// sign as a pure `int1`.
    ///
    /// [`abs_as_u`], with the sign read by a division rather than a shift for the reason
    /// [`Self::shifted_down`] gives.
    fn pure_magnitude(&mut self, value: ValueId, bits: usize) -> (ValueId, ValueId) {
        let top = self.shifted_down(value, bits, bits - 1);
        let sign = self.cast(top, CastTarget::Int(1));
        (abs_as_u(self, value, sign, bits), sign)
    }

    /// The sign of a value held as `limbs` at `widths`, which is its top limb's top bit.
    pub(super) fn operand_sign(&mut self, limbs: &[(ValueId, bool)], widths: &[usize]) -> Sign {
        let top = widths.len() - 1;
        self.top_bit(limbs[top], widths[top])
    }

    /// The top bit of a limb `width` bits wide, known where the limb is, and otherwise the
    /// [`Self::sign_bit`] cut.
    pub(super) fn top_bit(&mut self, (limb, witnessed): (ValueId, bool), width: usize) -> Sign {
        match self.known(limb) {
            Some(pattern) => Sign::Known(pattern.cast(width).bit(width - 1) == Some(true)),
            None => Sign::Bit(self.sign_bit((limb, witnessed), width), witnessed),
        }
    }

    /// A sign as a field element that is zero or one.
    fn sign_field(&self, sign: Sign) -> ValueId {
        match sign {
            Sign::Known(true) => self.field_const(self.field().one()),
            Sign::Known(false) => self.field_const(self.field().zero()),
            Sign::Bit(bit, _) => bit,
        }
    }

    /// `a ⊕ b`, which on two bits is `a + b - 2ab`: one product where both are witnessed, and
    /// linear wherever either is known.
    fn xor_signs(&mut self, a: Sign, b: Sign) -> Sign {
        match (a, b) {
            (Sign::Known(a), Sign::Known(b)) => Sign::Known(a != b),
            (Sign::Known(false), other) | (other, Sign::Known(false)) => other,
            (Sign::Known(true), Sign::Bit(bit, witnessed))
            | (Sign::Bit(bit, witnessed), Sign::Known(true)) => {
                let one = self.field_const(self.field().one());
                Sign::Bit(self.bin(BinaryArithOpKind::USub, one, bit), witnessed)
            }
            (Sign::Bit(a, a_witnessed), Sign::Bit(b, b_witnessed)) => {
                let sum = self.bin(BinaryArithOpKind::UAdd, a, b);
                let product = self.bin(BinaryArithOpKind::UMul, a, b);
                let twice = self.bin(BinaryArithOpKind::UAdd, product, product);
                let xor = self.bin(BinaryArithOpKind::USub, sum, twice);
                Sign::Bit(xor, a_witnessed || b_witnessed)
            }
        }
    }

    /// `limbs` at `widths` negated modulo `2^N` where `sign` is one, and kept where it is zero.
    ///
    /// Each limb is [`Self::complement`]ed by the sign, then the carry chain adds the sign to the
    /// low limb, its top carry witnessed and dropped: `!x + 1` is `2^N - x` for every nonzero `x`,
    /// and for zero it carries out of the top limb and leaves zero, as it should. The chain's
    /// checks are under `guard`.
    pub(super) fn negate_if(
        &mut self,
        limbs: Vec<(ValueId, bool)>,
        widths: &[usize],
        sign: Sign,
        guard: Option<ValueId>,
    ) -> Vec<(ValueId, bool)> {
        if let Sign::Known(false) = sign {
            return limbs;
        }
        let flipped: Vec<(ValueId, bool)> = limbs
            .iter()
            .zip(widths)
            .map(|(limb, width)| self.complement(*limb, *width, sign))
            .collect();
        let addend: Vec<(ValueId, bool)> = widths
            .iter()
            .enumerate()
            .map(|(index, width)| match (index, sign) {
                (0, Sign::Known(_)) => (self.int_const(IntBits::one(*width)), false),
                (0, Sign::Bit(bit, witnessed)) => {
                    (self.cast(bit, CastTarget::Int(*width)), witnessed)
                }
                _ => (self.int_const(IntBits::zero(*width)), false),
            })
            .collect();
        let chain = self.carry_chain_over(
            false,
            &flipped,
            &addend,
            widths.to_vec(),
            CarryOut::Witnessed,
            guard,
        );
        self.answer_limbs(chain.answer, widths)
    }

    /// Hold a magnitude, as `limbs` at `widths`, to at most `2^(N - 1) - 1 + s` for the sign `s` it
    /// is about to take: below `2^(N - 1)`, or exactly `2^(N - 1)` (`INT_MIN`) where `s` is one.
    ///
    /// The magnitude's top bit `t` is cut out of its top limb, leaving `rest`, and one constraint
    /// asks `t·(rest + Σ low limbs + (1 - s)) == 0`. The sum is of non-negative integers far below
    /// the modulus, so it is zero exactly when each term is: either `t` is zero, or every other bit
    /// is and `s` is one. The constraint is under `guard`; the cut is not, as every caller passes
    /// limbs in range on every path.
    ///
    /// A product's schoolbook still checks its top column: the cut reads it through a cast that
    /// truncates on the pure side, so without that check an overflowing product would be refused
    /// only by the cut's constraint, after witness generation has run on.
    fn check_fits(
        &mut self,
        limbs: &[(ValueId, bool)],
        widths: &[usize],
        sign: Sign,
        guard: Option<ValueId>,
    ) {
        let top = widths.len() - 1;
        let width = widths[top];
        let t = match self.top_bit(limbs[top], width) {
            Sign::Known(false) => return,
            t => self.sign_field(t),
        };
        let whole = self.cast(limbs[top].0, CastTarget::Field);
        let half = self.field_const(self.field().two_pow(width - 1));
        let high = self.bin(BinaryArithOpKind::UMul, t, half);
        let mut sum = self.bin(BinaryArithOpKind::USub, whole, high);
        for &(limb, _) in &limbs[..top] {
            let limb = self.cast(limb, CastTarget::Field);
            sum = self.bin(BinaryArithOpKind::UAdd, sum, limb);
        }
        let one = self.field_const(self.field().one());
        let sign = self.sign_field(sign);
        let positive = self.bin(BinaryArithOpKind::USub, one, sign);
        sum = self.bin(BinaryArithOpKind::UAdd, sum, positive);
        let product = self.bin(BinaryArithOpKind::UMul, t, sum);
        let zero = self.field_const(self.field().zero());
        self.constrain_equal(product, zero, guard);
    }

    /// Limbs at `widths` as a schoolbook [`Factor`]: a known limb is a constant with its value as
    /// its bound, and any other is bounded by its width.
    fn factor_of(&mut self, limbs: &[(ValueId, bool)], widths: &[usize]) -> Factor {
        let mut witnessed = false;
        let (limbs, bounds) = limbs
            .iter()
            .zip(full_bounds(widths))
            .zip(widths)
            .map(
                |((&(limb, limb_witnessed), full), width)| match self.known(limb) {
                    Some(pattern) => {
                        let pattern = pattern.cast(*width);
                        let bound = BigUint::from(&pattern);
                        (Limb::Constant(pattern), bound)
                    }
                    None => {
                        witnessed |= limb_witnessed;
                        (Limb::Value(limb), full)
                    }
                },
            )
            .unzip();
        Factor {
            limbs,
            witnessed,
            bounds,
        }
    }

    /// A gadget's answer, one field element per limb of `widths`, as limbs the chain and the
    /// schoolbook read: a known limb as its constant, and any other as witnessed, which an answer
    /// with a witnessed operand is. A pure one would read back unchanged through `ValueOf`.
    pub(super) fn answer_limbs(
        &mut self,
        answer: Vec<ValueId>,
        widths: &[usize],
    ) -> Vec<(ValueId, bool)> {
        answer
            .into_iter()
            .zip(widths)
            .map(|(limb, width)| match self.constant_limb(limb, *width) {
                Some(pattern) => (self.int_const(pattern), false),
                None => (self.cast(limb, CastTarget::Int(*width)), true),
            })
            .collect()
    }

    /// Limbs as the field elements [`Self::deliver`] takes, a known limb as a field constant.
    ///
    /// `deliver` hands a known limb on as a witnessed constant, but recognizes one only as a
    /// constant: a cast of one would reach a witnessed result as a pure value, as every limb of
    /// `0 % d` would.
    pub(super) fn limb_fields(&mut self, limbs: Vec<(ValueId, bool)>) -> Vec<ValueId> {
        limbs
            .into_iter()
            .map(|(limb, _)| match self.known(limb) {
                Some(pattern) => self.limb_const(&pattern),
                None => self.cast(limb, CastTarget::Field),
            })
            .collect()
    }

    /// A limb `width` bits wide complemented where `sign` is one and kept where it is zero.
    ///
    /// `x + s·(2^w - 1 - 2x)`, which is `x` or `2^w - 1 - x`, and so below `2^w` either way because
    /// `x` is and `s` is a bit: it needs no separate check. It costs one product where both the
    /// limb and the sign are witnessed, and is linear otherwise.
    pub(super) fn complement(
        &mut self,
        (limb, witnessed): (ValueId, bool),
        width: usize,
        sign: Sign,
    ) -> (ValueId, bool) {
        let known = self.known(limb);
        let ones = self.limb_const(&IntBits::all_ones(width));
        match sign {
            Sign::Known(false) => (limb, witnessed),
            Sign::Known(true) => match known {
                Some(pattern) => (self.int_const(pattern.cast(width).complement()), false),
                None => {
                    let x = self.cast(limb, CastTarget::Field);
                    let flipped = self.bin(BinaryArithOpKind::USub, ones, x);
                    (self.cast(flipped, CastTarget::Int(width)), witnessed)
                }
            },
            Sign::Bit(sign, sign_witnessed) => {
                let x = self.cast(limb, CastTarget::Field);
                let twice = self.bin(BinaryArithOpKind::UAdd, x, x);
                let span = self.bin(BinaryArithOpKind::USub, ones, twice);
                let moved = self.bin(BinaryArithOpKind::UMul, sign, span);
                let flipped = self.bin(BinaryArithOpKind::UAdd, x, moved);
                let witnessed = (witnessed && known.is_none()) || sign_witnessed;
                (self.cast(flipped, CastTarget::Int(width)), witnessed)
            }
        }
    }
}
