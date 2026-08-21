//! The single definition of "this division is undefined", shared by every pass that has to make a
//! failable `Div`/`Mod` explicit.
//!
//! Three consumers, which must not drift:
//!
//! - `LowerPureGuards::lower_divmod_guard` — a *guarded* division turns the condition into "this
//!   branch must be inactive", so an inactive bad division is not an error.
//! - `LowerPureGuards::lower_unguarded_divmod` — an unguarded division whose quotient is used
//!   asserts the condition is false, then performs the division unchanged.
//! - `DCE` — an unguarded division whose quotient is *not* used is replaced by the assertion
//!   alone. Mavros builds HLSSA straight from Noir's monomorphized AST and never runs Noir's SSA
//!   pipeline, so nothing upstream has already attached a failure to the division; if the division
//!   is simply deleted, the failure Noir promises disappears with it.
//!
//! [`divmod_provably_defined`] is the fourth member of that set and must not drift either: it is
//! the *discharge* of the same condition, so a consumer that can prove it need emit nothing at all.
//! Both `LowerPureGuards` (at each of its two sites) and `DCE` consult it — the latter in its mark
//! phase as well as its sweep, since a check it will not emit must not hold the operand chain that
//! feeds it live either.

use num_bigint::BigInt;
use num_traits::{One, Zero};

use crate::compiler::{
    analysis::value_range_analysis::ValueRange,
    ssa::{
        ValueId,
        hlssa::{CmpKind, OpCode, Type, TypeExpr, builder::HLEmitter},
    },
    util::bit_mask,
};

/// Whether a `Div`/`Mod` on this operand type can fail, and so needs a check.
///
/// Every scalar numeric type can: integers by a zero divisor (and signed additionally by
/// `INT_MIN / -1`), and `Field` by a zero divisor, which has no multiplicative inverse. Anything
/// else is not a division operand type at all.
pub fn divmod_can_fail(ty: &Type) -> bool {
    matches!(
        ty.strip_witness().expr,
        TypeExpr::U(_) | TypeExpr::I(_) | TypeExpr::Field
    )
}

/// Whether the range domain **proves** this division is defined, so the check
/// [`emit_divmod_failure_cond`] would build is dead and need not be emitted at all.
///
/// This is the exact complement of that function: it discharges the same disjunction, pattern by
/// pattern, rather than constraining it.
///
/// - A zero divisor is the failure every operand type shares.
/// - A signed operand additionally fails at `INT_MIN / −1`, whose mathematical quotient is one past
///   the top of the type. Either half is enough to rule that out: a divisor that is never `−1`, or
///   a dividend that is never `INT_MIN`.
///
/// The queries are over *bit patterns*, not mathematical values, which is what lets one predicate
/// serve `U`, `I` and `Field` alike — and it is the reason the domain carries both readings, since
/// `−1` is the pattern `2^bits − 1` read one way and `−1` read the other.
///
/// `lhs_type` must already be stripped of any `WitnessOf` wrapper and satisfy [`divmod_can_fail`];
/// anything else answers `false`, since a check that is not understood must not be dropped.
///
/// Note that ⊥ answers `false` throughout, by [`ValueRange::proves_excludes_pattern`]. That matters
/// here more than anywhere: a `Div` whose operand range is ⊥ is exactly the division the check is
/// there to reject.
///
/// This is the domain's first *constraint-eliding* consumer, so it also inherits the one soundness
/// assumption the domain cannot discharge itself: `WriteWitness` gives a minted witness the range
/// of its **hint**, which binds the prover only through whatever constraints accompany it. A
/// divisor that is a hint written without them would be proved nonzero on the honest execution
/// alone. Every hint in the tree today is pinned; see the `WriteWitness` transfer in
/// `value_range_analysis`.
pub fn divmod_provably_defined(lhs: &ValueRange, rhs: &ValueRange, lhs_type: &Type) -> bool {
    let zero = BigInt::zero();
    match &lhs_type.expr {
        TypeExpr::U(_) | TypeExpr::Field => rhs.proves_excludes_pattern(&zero),
        TypeExpr::I(bits) if *bits > 0 => {
            if !rhs.proves_excludes_pattern(&zero) {
                return false;
            }
            let minus_one = (BigInt::one() << bits) - BigInt::one();
            let int_min = BigInt::one() << (bits - 1);
            rhs.proves_excludes_pattern(&minus_one) || lhs.proves_excludes_pattern(&int_min)
        }
        _ => false,
    }
}

/// A `u1` that is true exactly when `lhs / rhs` (or `lhs % rhs`) is undefined: a zero divisor, or
/// — for signed operands — `INT_MIN / -1`, whose mathematical quotient is one past the top of the
/// type.
///
/// `lhs_type` must already be stripped of any `WitnessOf` wrapper and satisfy [`divmod_can_fail`].
pub fn emit_divmod_failure_cond(
    emitter: &mut impl HLEmitter,
    lhs: ValueId,
    rhs: ValueId,
    lhs_type: &Type,
) -> ValueId {
    let zero_val = match &lhs_type.expr {
        TypeExpr::U(b) => emitter.u_const(*b, 0),
        TypeExpr::I(b) => emitter.i_const(*b, 0),
        TypeExpr::Field => emitter.field_const(emitter.field().constant(0u64)),
        other => unreachable!("divmod failure condition on a non-numeric operand type: {other:?}"),
    };
    let is_zero = emitter.eq(rhs, zero_val);
    match &lhs_type.expr {
        TypeExpr::I(bits) => {
            let min_val = emitter.i_const(*bits, 1u128 << (*bits - 1));
            let minus_one = emitter.i_const(*bits, bit_mask(*bits));
            let lhs_is_min = emitter.eq(lhs, min_val);
            let rhs_is_minus_one = emitter.eq(rhs, minus_one);
            let signed_overflow = emitter.and(lhs_is_min, rhs_is_minus_one);
            emitter.or(is_zero, signed_overflow)
        }
        _ => is_zero,
    }
}

/// Emit `assert(!undefined(lhs, rhs))` — the unguarded form of the check.
///
/// The caller decides whether the division itself follows.
pub fn emit_divmod_is_defined_assert(
    emitter: &mut impl HLEmitter,
    lhs: ValueId,
    rhs: ValueId,
    lhs_type: &Type,
) {
    let failure = emit_divmod_failure_cond(emitter, lhs, rhs, lhs_type);
    let zero_u1 = emitter.u_const(1, 0);
    emitter.emit(OpCode::AssertCmp {
        kind: CmpKind::Eq,
        lhs: failure,
        rhs: zero_u1,
    });
}

#[cfg(test)]
mod tests {
    use mavros_artifacts::FieldConfig;

    use super::*;
    use crate::compiler::analysis::value_range_analysis::{Interval, Width};

    fn u8_range(lo: i64, hi: i64) -> ValueRange {
        ValueRange::from_unsigned(Width::Bits(8), Interval::closed(lo, hi))
    }

    fn i8_range(lo: i64, hi: i64) -> ValueRange {
        ValueRange::from_signed(Width::Bits(8), Interval::closed(lo, hi))
    }

    fn field_range(lo: i64, hi: i64) -> ValueRange {
        ValueRange::from_unsigned(Width::Field(FieldConfig::bn254()), Interval::closed(lo, hi))
    }

    #[test]
    fn unsigned_needs_only_a_nonzero_divisor() {
        let anything = ValueRange::full(Width::Bits(8));
        assert!(divmod_provably_defined(
            &anything,
            &u8_range(1, 255),
            &Type::u(8)
        ));
        assert!(!divmod_provably_defined(
            &anything,
            &u8_range(0, 1),
            &Type::u(8)
        ));
    }

    #[test]
    fn field_needs_only_a_nonzero_divisor() {
        let anything = ValueRange::full(Width::Field(FieldConfig::bn254()));
        assert!(divmod_provably_defined(
            &anything,
            &field_range(1, 9),
            &Type::field()
        ));
        assert!(!divmod_provably_defined(
            &anything,
            &field_range(0, 9),
            &Type::field()
        ));
    }

    #[test]
    fn signed_also_needs_int_min_over_minus_one_ruled_out() {
        let anything = ValueRange::full(Width::Bits(8));
        // Nonzero, but `-1` is still in reach and the dividend could be `INT_MIN`.
        assert!(!divmod_provably_defined(
            &anything,
            &i8_range(-128, -1),
            &Type::i(8)
        ));
        // Either half of the overflow case is enough on its own: a divisor that is never `-1`...
        assert!(divmod_provably_defined(
            &anything,
            &i8_range(-128, -2),
            &Type::i(8)
        ));
        // ...or a dividend that is never `INT_MIN`.
        assert!(divmod_provably_defined(
            &i8_range(-127, 127),
            &i8_range(-128, -1),
            &Type::i(8)
        ));
        // A zero divisor still sinks it, whatever the dividend.
        assert!(!divmod_provably_defined(
            &i8_range(0, 0),
            &i8_range(0, 5),
            &Type::i(8)
        ));
    }

    #[test]
    fn bottom_never_discharges_the_check() {
        // The division whose operands are provably unreachable is exactly the one the check exists
        // to reject, so ⊥ must not be read as "cannot fail". See the `proves_*` predicates.
        let bottom = ValueRange::empty(Width::Bits(8));
        let anything = ValueRange::full(Width::Bits(8));
        assert!(!divmod_provably_defined(&anything, &bottom, &Type::u(8)));
        assert!(!divmod_provably_defined(&bottom, &bottom, &Type::i(8)));
        assert!(!divmod_provably_defined(
            &bottom,
            &u8_range(1, 255),
            &Type::i(8)
        ));
    }

    #[test]
    fn a_type_that_cannot_divide_is_never_discharged() {
        // `divmod_can_fail` refuses these, so the question should not arise -- but answering
        // "provably defined" for an operand type this does not understand would silently delete a
        // check if the two ever drifted apart. Note the divisor range here is one that *would*
        // discharge a `U`/`Field` division, so it is the type arm alone that refuses.
        assert!(!divmod_provably_defined(
            &ValueRange::full(Width::NonScalar),
            &u8_range(1, 255),
            &Type::function()
        ));
        assert!(!divmod_can_fail(&Type::function()));
    }
}
