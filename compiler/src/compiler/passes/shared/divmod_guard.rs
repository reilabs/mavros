//! The single definition of "this division is undefined", shared by every pass that makes a
//! failable `Div`/`Mod` explicit.
//!
//! Three consumers, which must not drift:
//!
//! - `LowerPureGuards::lower_divmod_guard`: a _guarded_ division turns the condition into "this
//!   branch must be inactive", so a bad division in an inactive branch is not an error. This
//!   branches on the operands, so it needs them pure. A guarded witness field division instead
//!   asserts the condition false under the guard (`lower_witness_field_div_guard`), and a guarded
//!   witness integer division is left to its lowering, which rejects the same executions.
//! - `LowerPureGuards::lower_unguarded_divmod`: an unguarded division whose quotient is used
//!   asserts the condition is false, then divides unchanged. An integer division by a witness is
//!   the exception: its lowering refuses a zero divisor, and when signed `INT_MIN / -1`
//!   ([`divisor_checked_by_its_lowering`]).
//! - `DCE`: an unguarded division whose quotient is _not_ used is replaced by the assertion alone.
//!   Mavros builds HLSSA from Noir's monomorphized AST and never runs Noir's SSA pipeline, so
//!   nothing upstream attaches a failure to the division; deleting it would drop the failure Noir
//!   promises. After taint inference, the exception above is kept whole instead, as the division
//!   itself is the check.
//!
//! [`divmod_provably_defined`] is a fourth member that must not drift either: it _discharges_ the
//! same condition, so a consumer that can prove it emits nothing. `LowerPureGuards` (at both its
//! sites) and `DCE` consult it; DCE in its mark phase as well as its sweep, since a check it will
//! not emit must not keep the operand chain that feeds it alive.
//!
//! None of them reads signedness off the operand's type, which has none. Every entry point takes a
//! `signed` flag that its caller took from the `Div`/`Mod` opcode, so the check and its division
//! are decided by one source.

use num_bigint::BigInt;
use num_traits::{One, Zero};

use mavros_int_semantics::IntBits;

use crate::compiler::{
    analysis::value_range_analysis::ValueRange,
    ssa::{
        ValueId,
        hlssa::{
            ArithGroup, BinaryArithOpKind, CmpKind, OpCode, Type, TypeExpr, builder::HLEmitter,
        },
    },
};

/// Whether a `Div`/`Mod` on this operand type can fail, and so needs a check.
///
/// Every scalar numeric type can: integers by a zero divisor (and signed additionally by
/// `INT_MIN / -1`), and `Field` by a zero divisor, which has no multiplicative inverse. Anything
/// else is not a division operand type at all.
pub fn divmod_can_fail(ty: &Type) -> bool {
    matches!(ty.strip_witness().expr, TypeExpr::Int(_) | TypeExpr::Field)
}

/// Whether the witness lowering a `kind` division by a `rhs_type` divisor reaches rejects every
/// execution [`emit_divmod_failure_cond`] would, so no separate check is needed.
///
/// It does for an integer division by a witness: a zero divisor fails the unsigned `r < d` check
/// (which a signed division runs on magnitudes), and `INT_MIN / -1` fails the quotient's range
/// check. The remainder's lowering has both checks too.
///
/// The check is made only when the lowering is reached, so a dead division answered `true` here
/// must be kept until then, which `DCE`'s `owes_witness_check` ensures.
pub fn divisor_checked_by_its_lowering(kind: BinaryArithOpKind, rhs_type: &Type) -> bool {
    matches!(kind.group(), ArithGroup::Div | ArithGroup::Rem)
        && rhs_type.is_witness_of()
        && matches!(rhs_type.strip_witness().expr, TypeExpr::Int(_))
}

/// Whether the range domain **proves** this division defined, so the check
/// [`emit_divmod_failure_cond`] would build is dead.
///
/// It discharges that function's disjunction pattern by pattern:
///
/// - a zero divisor, for every operand type;
/// - for a signed operand, `INT_MIN / −1`, ruled out by a divisor never `−1` or a dividend never
///   `INT_MIN`.
///
/// The queries are over bit patterns, so one predicate serves `U`, `I` and `Field`: `−1` is the
/// pattern `2^bits − 1`.
///
/// `lhs_type` must be stripped of `WitnessOf` and satisfy [`divmod_can_fail`]; anything else
/// answers `false`, as a check not understood must not be dropped.
///
/// `signed` comes from the caller's opcode, not `lhs_type`, which carries no sign. The failure
/// condition and the re-planted division read the same opcode, so this cannot discharge a check
/// their lowering still needs. The width comes from the type.
///
/// ⊥ answers `false` ([`ValueRange::proves_excludes_pattern`]): a division whose operand range is ⊥
/// is exactly the one the check exists to reject.
///
/// As a constraint-eliding consumer, this assumes every hint is pinned by its constraints, since
/// the domain gives a `WriteWitness` its hint's range; see the `WriteWitness` transfer in
/// `value_range_analysis`.
pub fn divmod_provably_defined(
    lhs: &ValueRange,
    rhs: &ValueRange,
    lhs_type: &Type,
    signed: bool,
) -> bool {
    let zero = BigInt::zero();
    let bits = match &lhs_type.expr {
        TypeExpr::Int(bits) => Some(*bits),
        TypeExpr::Field => None,
        // A check that is not understood must not be dropped.
        _ => return false,
    };

    if !rhs.proves_excludes_pattern(&zero) {
        return false;
    }
    if !signed {
        return true;
    }
    match bits {
        // A field has no `INT_MIN`, so a nonzero divisor is the whole condition however the
        // operation reads its operands.
        None => true,
        Some(bits) if bits > 0 => {
            let minus_one = (BigInt::one() << bits) - BigInt::one();
            let int_min = BigInt::one() << (bits - 1);
            rhs.proves_excludes_pattern(&minus_one) || lhs.proves_excludes_pattern(&int_min)
        }
        // A zero-width signed integer has no sign bit to reason about; refuse rather than guess.
        Some(_) => false,
    }
}

/// A `u1` that is true exactly when `lhs / rhs` (or `lhs % rhs`) is undefined: a zero divisor, or
/// — for signed operands — `INT_MIN / -1`, whose mathematical quotient is one past the top of the
/// type.
///
/// `lhs_type` must already be stripped of any `WitnessOf` wrapper and satisfy [`divmod_can_fail`].
///
/// `signed` decides whether the overflow disjunct is emitted, and comes from the opcode via the
/// caller, for the reason spelled out on [`divmod_provably_defined`]. The type supplies only the
/// width; it has no sign left to supply.
pub fn emit_divmod_failure_cond(
    emitter: &mut impl HLEmitter,
    lhs: ValueId,
    rhs: ValueId,
    lhs_type: &Type,
    signed: bool,
) -> ValueId {
    let zero_val = match &lhs_type.expr {
        TypeExpr::Int(b) => emitter.int_const(IntBits::zero(*b)),
        TypeExpr::Field => emitter.field_const(emitter.field().constant(0u64)),
        other => {
            ice_unreachable!("divmod failure condition on a non-numeric operand type: {other:?}")
        }
    };
    let is_zero = emitter.eq(rhs, zero_val);
    match &lhs_type.expr {
        TypeExpr::Int(bits) if signed => {
            let bits = *bits;

            let min_val = emitter.int_const(IntBits::from_signed(bits, &IntBits::signed_min(bits)));
            let minus_one = emitter.int_const(IntBits::all_ones(bits));
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
/// The caller decides whether the division itself follows, and supplies `signed` as for
/// [`emit_divmod_failure_cond`].
pub fn emit_divmod_is_defined_assert(
    emitter: &mut impl HLEmitter,
    lhs: ValueId,
    rhs: ValueId,
    lhs_type: &Type,
    signed: bool,
) {
    let failure = emit_divmod_failure_cond(emitter, lhs, rhs, lhs_type, signed);
    let zero_u1 = emitter.int_const(IntBits::zero(1));
    emitter.emit(OpCode::AssertCmp {
        kind: CmpKind::Eq,
        lhs: failure,
        rhs: zero_u1,
    });
}

/// The divisor a division's hint divides by: `divisor`, or one where it is zero.
///
/// The constraints refuse a zero divisor, but the hint runs first, and an integer division by a
/// witness has no assertion ahead of it (see [`divisor_checked_by_its_lowering`]). Without this
/// the hint would compile to a `udiv` by zero, which LLVM leaves undefined. Dividing by one changes
/// nothing an honest run computes. The single-cell and limb-wise divisions both hint through this.
pub fn nonzero_hint_divisor(b: &mut impl HLEmitter, divisor: ValueId, bits: usize) -> ValueId {
    let zero = b.int_const(IntBits::zero(bits));
    let one = b.int_const(IntBits::one(bits));
    let is_zero = b.eq(divisor, zero);
    b.select(is_zero, one, divisor)
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
            &Type::int(8),
            false
        ));
        assert!(!divmod_provably_defined(
            &anything,
            &u8_range(0, 1),
            &Type::int(8),
            false
        ));
    }

    #[test]
    fn field_needs_only_a_nonzero_divisor() {
        let anything = ValueRange::full(Width::Field(FieldConfig::bn254()));
        assert!(divmod_provably_defined(
            &anything,
            &field_range(1, 9),
            &Type::field(),
            false
        ));
        assert!(!divmod_provably_defined(
            &anything,
            &field_range(0, 9),
            &Type::field(),
            false
        ));
    }

    #[test]
    fn signed_also_needs_int_min_over_minus_one_ruled_out() {
        let anything = ValueRange::full(Width::Bits(8));
        // Nonzero, but `-1` is still in reach and the dividend could be `INT_MIN`.
        assert!(!divmod_provably_defined(
            &anything,
            &i8_range(-128, -1),
            &Type::int(8),
            true
        ));
        // Either half of the overflow case is enough on its own: a divisor that is never `-1`...
        assert!(divmod_provably_defined(
            &anything,
            &i8_range(-128, -2),
            &Type::int(8),
            true
        ));
        // ...or a dividend that is never `INT_MIN`.
        assert!(divmod_provably_defined(
            &i8_range(-127, 127),
            &i8_range(-128, -1),
            &Type::int(8),
            true
        ));
        // A zero divisor still sinks it, whatever the dividend.
        assert!(!divmod_provably_defined(
            &i8_range(0, 0),
            &i8_range(0, 5),
            &Type::int(8),
            true
        ));
    }

    #[test]
    fn bottom_never_discharges_the_check() {
        // The division whose operands are provably unreachable is exactly the one the check exists
        // to reject, so ⊥ must not be read as "cannot fail". See the `proves_*` predicates.
        let bottom = ValueRange::empty(Width::Bits(8));
        let anything = ValueRange::full(Width::Bits(8));
        assert!(!divmod_provably_defined(
            &anything,
            &bottom,
            &Type::int(8),
            false
        ));
        assert!(!divmod_provably_defined(
            &bottom,
            &bottom,
            &Type::int(8),
            true
        ));
        assert!(!divmod_provably_defined(
            &bottom,
            &u8_range(1, 255),
            &Type::int(8),
            true
        ));
    }

    #[test]
    fn a_type_that_cannot_divide_is_never_discharged() {
        // `divmod_can_fail` refuses these, so the question should not arise -- but answering
        // "provably defined" for an operand type this does not understand would silently delete a
        // check if the two ever drifted apart. Note the divisor range here is one that _would_
        // discharge a `U`/`Field` division, so it is the type arm alone that refuses.
        assert!(!divmod_provably_defined(
            &ValueRange::full(Width::NonScalar),
            &u8_range(1, 255),
            &Type::function_returning(vec![Type::field()]),
            false
        ));
        assert!(!divmod_can_fail(&Type::function_returning(vec![
            Type::field()
        ])));
    }

    #[test]
    fn the_sign_parameter_decides_the_overflow_half_not_the_type() {
        let anything = ValueRange::full(Width::Bits(8));
        let nonzero_but_reaches_minus_one = i8_range(-128, -1);

        // Read as signed, `INT_MIN / -1` is still in reach, so a nonzero divisor is not enough...
        assert!(!divmod_provably_defined(
            &anything,
            &nonzero_but_reaches_minus_one,
            &Type::int(8),
            true
        ));
        // ...and read as unsigned, there is no such case and the same operands discharge.
        assert!(divmod_provably_defined(
            &anything,
            &nonzero_but_reaches_minus_one,
            &Type::int(8),
            false
        ));
    }
}
