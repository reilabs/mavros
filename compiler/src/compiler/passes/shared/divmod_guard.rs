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

use crate::compiler::{
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
