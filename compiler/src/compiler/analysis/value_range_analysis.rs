//! Determines the smallest closed interval over which any numeric value in the SSA can range.
//!
//! # Two Readings of One Bit Pattern
//!
//! An integer in the SSA is a bit pattern of some width. This pattern has _two_ readings as an
//! integer, as both unsigned and 2's-complement signed, with which one is relevant being the
//! property of an _operation_ rather than of the value. As a result [`ValueRange`] carries both,
//! with the consumer making the selection:
//!
//! ```text
//! γ(ValueRange{Bits(n), u, s}) = { x ∈ [0, 2ⁿ) : x ∈ u ∧ dec_n(x) ∈ s }
//!       where dec_n(x) = x − 2ⁿ·[x ≥ 2ⁿ⁻¹]
//! ```
//!
//! The pair constrains one value from two directions, which is what ensures that it is as
//! expressive as a wrapped range while every component stays a plain, non-wrapping interval. An
//! interval that would straddle the wrap point saturates to the full width _on that reading only_,
//! and the other reading retains all of its information.
//!
//! [`Interval`] itself is the calculator, implementing ordinary interval arithmetic over ℤ, with no
//! opinion about widths or signs. `ValueRange` is the domain element.
//!
//! # Width::Field
//!
//! A field element's canonical integer *is* its value; there is no second reading. So
//! [`Width::Field`] keeps `signed == unsigned`, both ⊆ `[0, p−1]`. Keeping the struct total this
//! way avoids an `Option<Interval>` that every call site would have to unwrap.

use mavros_artifacts::FieldConfig;
use num_bigint::{BigInt, Sign};
use num_traits::{One, Signed, ToPrimitive, Zero};
use tracing::{Level, instrument};

use crate::{
    collections::HashMap,
    compiler::{
        Field,
        analysis::{
            flow_analysis::{CFG, FlowAnalysis},
            types::{FunctionTypeInfo, TypeInfo},
        },
        pass_manager::{Analysis, AnalysisId, AnalysisStore},
        ssa::{
            BlockId, FunctionId, Instruction, Terminator, ValueId,
            hlssa::{
                BinaryArithOpKind, CastTarget, Constant, HLFunction, HLSSA, OpCode, Type, TypeExpr,
            },
        },
    },
};

// VALUE RANGE ANALYSIS
// ================================================================================================

pub struct ValueRangeAnalysis;

const ITER_LIMIT: usize = 8;

impl ValueRangeAnalysis {
    pub fn new() -> Self {
        Self
    }

    #[instrument(skip_all, name = "ValueRangeAnalysis::run")]
    pub fn run(&self, ssa: &HLSSA, cfg: &FlowAnalysis, types: &TypeInfo) -> ValueRanges {
        let mut result = ValueRanges {
            functions: HashMap::default(),
        };

        // The configured field, threaded through the interval algebra so that no
        // static field is named.
        let field = ssa.field();

        // Constants are module-level singletons; pre-compute their bounds once.
        let constant_bounds = compute_constant_bounds(ssa);

        for (function_id, function) in ssa.iter_functions() {
            let func_cfg = cfg.get_function_cfg(*function_id);
            let func_types = types.get_function(*function_id);
            let function_ranges =
                self.run_function(function, func_cfg, func_types, &constant_bounds, field);
            result.functions.insert(*function_id, function_ranges);
        }
        result
    }

    #[instrument(skip_all, level = Level::TRACE, fields(function = function.get_name()))]
    fn run_function(
        &self,
        function: &HLFunction,
        cfg: &CFG,
        types: &FunctionTypeInfo,
        constant_bounds: &HashMap<ValueId, ValueRange>,
        field: FieldConfig,
    ) -> FunctionValueRanges {
        let mut bounds: HashMap<ValueId, ValueRange> = constant_bounds.clone();

        // Initial state: every value's bound is its declared type's full range.
        // Iteration only narrows from there.
        for (_block_id, block) in function.get_blocks() {
            for (vid, ty) in block.get_parameters() {
                bounds.insert(*vid, ValueRange::for_type(ty, field));
            }
            for instr in block.get_instructions() {
                for vid in instr.get_results() {
                    bounds.insert(
                        *vid,
                        ValueRange::for_type(types.get_value_type(*vid), field),
                    );
                }
            }
        }

        let entry_block_id = function.get_entry_id();
        let order: Vec<BlockId> = cfg.get_domination_pre_order().collect();

        for _iter in 0..ITER_LIMIT {
            let mut changed = false;

            for &block_id in &order {
                let block = function.get_block(block_id);

                if block_id != entry_block_id {
                    // Only `Jmp` carries block arguments, so a `JmpIf` predecessor contributes
                    // nothing — which is right precisely because a block it can reach has no
                    // parameters to contribute to. Were that invariant ever broken, the join below
                    // would run over a strict subset of the predecessors and *narrow* rather than
                    // give up, so we assert for safety.
                    debug_assert!(
                        block.get_parameters().next().is_none()
                            || cfg.get_predecessors(block_id).all(|p| matches!(
                                function.get_block(p).get_terminator().unwrap(),
                                Terminator::Jmp(..)
                            )),
                        "a parameterised block is reachable by a terminator that passes no arguments"
                    );

                    let pred_args: Vec<Vec<ValueId>> = cfg
                        .get_predecessors(block_id)
                        .filter_map(|p| {
                            let term = function.get_block(p).get_terminator().unwrap();
                            match term {
                                Terminator::Jmp(t, args) if *t == block_id => Some(args.clone()),
                                _ => None,
                            }
                        })
                        .collect();

                    for (idx, (param_id, param_type)) in block.get_parameters().enumerate() {
                        let mut joined: Option<ValueRange> = None;
                        for args in &pred_args {
                            if let Some(arg_id) = args.get(idx) {
                                let arg_range = bounds
                                    .get(arg_id)
                                    .cloned()
                                    .unwrap_or_else(|| ValueRange::for_type(param_type, field));
                                joined = Some(match joined {
                                    None => arg_range,
                                    Some(j) => j.join(&arg_range),
                                });
                            }
                        }
                        let new_range =
                            joined.unwrap_or_else(|| ValueRange::for_type(param_type, field));
                        Self::overwrite(&mut bounds, *param_id, new_range, &mut changed);
                    }
                }

                for instr in block.get_instructions() {
                    self.transfer(instr, types, &mut bounds, &mut changed, field);
                }
            }

            if !changed {
                break;
            }
        }

        FunctionValueRanges { values: bounds }
    }

    fn overwrite(
        bounds: &mut HashMap<ValueId, ValueRange>,
        v: ValueId,
        new: ValueRange,
        changed: &mut bool,
    ) {
        if bounds.get(&v) != Some(&new) {
            bounds.insert(v, new);
            *changed = true;
        }
    }

    fn transfer(
        &self,
        instr: &OpCode,
        types: &FunctionTypeInfo,
        bounds: &mut HashMap<ValueId, ValueRange>,
        changed: &mut bool,
        field: FieldConfig,
    ) {
        // Both readings of an operand's bit pattern. Every rule below picks its reading from the
        // *operation*, not from the operand's declared signedness — that is the point of the pair.
        let range = |bounds: &HashMap<ValueId, ValueRange>, v: ValueId| -> ValueRange {
            match bounds.get(&v) {
                Some(r) => r.clone(),
                None => ValueRange::for_type(types.get_value_type(v), field),
            }
        };
        let width_of = |v: ValueId| Width::of_type(types.get_value_type(v), field);

        match instr {
            OpCode::Cast {
                result,
                value,
                target,
            } => {
                let in_r = range(bounds, *value);
                let r = match target {
                    // Every numeric cast is a raw-bit operation: the LLSSA lowering zero-extends or
                    // truncates, and the field target assembles the raw bits into limbs. None of
                    // them sign-extends — `SExt` is the opcode that does that — so a negative `i8`
                    // casts to `Field` as 255 rather than `p − 1`.
                    CastTarget::Field => in_r.reinterpret_to(Width::Field(field)),
                    CastTarget::U(n) | CastTarget::I(n) => in_r.reinterpret_to(Width::Bits(*n)),
                    // ValueOf strips the WitnessOf wrapper: payload unchanged.
                    CastTarget::Nop | CastTarget::WitnessOf | CastTarget::ValueOf => in_r,
                    // Sequence-level casts carry no scalar range.
                    CastTarget::ArrayToSlice | CastTarget::Map(_) => {
                        ValueRange::full(Width::NonScalar)
                    }
                };
                Self::set(bounds, *result, types, field, r, changed);
            }

            OpCode::SExt {
                result,
                value,
                from_bits,
                ..
            } => {
                let in_r = range(bounds, *value);
                // The low `from_bits` bits, re-read as two's complement at that width — which is
                // exactly the signed reading the source already carries when it is that wide.
                let mut signed = in_r.unsigned().wrap_to_signed_bits(*from_bits);
                if in_r.width() == Width::Bits(*from_bits) {
                    signed = signed.intersect(in_r.signed());
                }
                let width = width_of(*result);
                Self::set(
                    bounds,
                    *result,
                    types,
                    field,
                    ValueRange::from_signed(width, signed),
                    changed,
                );
            }

            OpCode::BitRange {
                result,
                value,
                offset,
                width,
            } => {
                let in_r = range(bounds, *value);
                // `(v >> offset) & mask(width)`, with the result keeping the *source's* type. The
                // mask is exact whenever the shifted value already fits, which subsumes the
                // identity case a signed source used to need a rule of its own for.
                let shifted = in_r.unsigned().div_const_pos(&(BigInt::one() << *offset));
                let masked = if shifted.fits_in_unsigned_bits(*width) {
                    shifted
                } else {
                    Interval::unsigned_full(*width)
                };
                let out_width = width_of(*result);
                Self::set(
                    bounds,
                    *result,
                    types,
                    field,
                    ValueRange::from_unsigned(out_width, masked),
                    changed,
                );
            }

            // The witness takes the range of its *hint*, which is a claim about witness generation
            // rather than about the circuit: `WriteWitness` mints an unconstrained R1CS variable,
            // and only the constraints emitted around it pin the value a prover may choose. So this
            // transfer is sound exactly under the standing invariant that **every hint is pinned by
            // accompanying constraints** — `bit_range.rs::lower_witness_bit_range` and
            // `witness_bitwise.rs::wrap_shifted_product` are the pattern to follow. A hint written
            // without them leaves the range describing the honest prover only, which is no longer
            // merely imprecise now that consumers can ELIDE constraints on the strength of what
            // this domain says.
            OpCode::WriteWitness {
                result: Some(r),
                value,
                ..
            } => {
                let in_r = range(bounds, *value);
                Self::set(bounds, *r, types, field, in_r, changed);
            }
            OpCode::WriteWitness { result: None, .. } => {}

            OpCode::FreshWitness {
                result,
                result_type,
            } => {
                Self::overwrite(
                    bounds,
                    *result,
                    ValueRange::for_type(result_type, field),
                    changed,
                );
            }

            OpCode::Cmp { result, .. } => {
                // Both Eq and Lt yield a u1 boolean regardless of operand types.
                Self::overwrite(
                    bounds,
                    *result,
                    ValueRange::from_unsigned(Width::Bits(1), Interval::unsigned_full(1)),
                    changed,
                );
            }

            OpCode::Not { result, value } => {
                let in_r = range(bounds, *value);
                let width = width_of(*result);
                let r = match width {
                    // `!x` complements every bit, which reads as `mask − x` unsigned and as
                    // `−x − 1` signed. These are one map, not two: `dec(mask − enc(x)) = −x − 1`.
                    Width::Bits(n) => {
                        let mask = (BigInt::one() << n) - BigInt::one();
                        ValueRange::new(
                            width,
                            Interval::singleton(mask).sub(in_r.unsigned()),
                            in_r.signed().neg().sub(&Interval::singleton(1)),
                        )
                    }
                    Width::Field(_) | Width::NonScalar => ValueRange::full(width),
                };
                Self::set(bounds, *result, types, field, r, changed);
            }

            OpCode::BinaryArithOp {
                kind,
                result,
                lhs,
                rhs,
            } => {
                let result_ty = types.get_value_type(*result);
                let width = Width::of_type(result_ty, field);
                let signed = type_is_signed(result_ty);
                let l = range(bounds, *lhs);
                let r_in = range(bounds, *rhs);
                let r = Self::binary_arith(
                    *kind,
                    width,
                    signed,
                    &l,
                    &r_in,
                    width_of(*lhs) == width,
                    width_of(*rhs) == width,
                    types.get_value_type(*rhs),
                );
                Self::set(bounds, *result, types, field, r, changed);
            }

            OpCode::MulConst {
                result,
                const_val,
                var,
            } => {
                let result_ty = types.get_value_type(*result);
                let width = Width::of_type(result_ty, field);
                let c = range(bounds, *const_val);
                let v = range(bounds, *var);
                // The result takes the *variable's* type, and the constant is a plain multiplier
                // rather than a bit pattern, so it is read at its own declared signedness and then
                // scales both of the variable's readings.
                let factor = c.by_type(types.get_value_type(*const_val));
                let r = if c.is_empty() || v.is_empty() || width_of(*var) != width {
                    Self::unknown_or_empty(width, c.is_empty() || v.is_empty())
                } else {
                    Self::wrap_or_trap(
                        width,
                        type_is_signed(result_ty),
                        factor.mul(v.unsigned()),
                        factor.mul(v.signed()),
                    )
                };
                Self::set(bounds, *result, types, field, r, changed);
            }

            OpCode::Select {
                result, if_t, if_f, ..
            } => {
                let width = width_of(*result);
                let t = range(bounds, *if_t).constrain_to(width);
                let f = range(bounds, *if_f).constrain_to(width);
                Self::set(bounds, *result, types, field, t.join(&f), changed);
            }

            OpCode::Guard { inner, .. } => {
                self.transfer(inner, types, bounds, changed, field);
                // A guarded *failable* operation is not simply its inner operation. `LowerPureGuards`
                // branches on the failure condition: the failing side asserts the guard's condition
                // is false and yields the result type's zero, so an inactive guard around an
                // operation that would have overflowed, divided by zero or shifted out of range
                // produces `0` — a value the computed range can easily exclude.
                if guard_may_produce_zero(inner, types) {
                    for vid in inner.get_results() {
                        let width = Width::of_type(types.get_value_type(*vid), field);
                        let zero = ValueRange::from_unsigned(width, Interval::singleton(0));
                        let joined = match bounds.get(vid) {
                            Some(computed) => computed.join(&zero),
                            None => zero,
                        };
                        Self::overwrite(bounds, *vid, joined, changed);
                    }
                }
            }

            // Other opcodes: keep the type-based default bound.
            //
            // TODO(stage-1): `Rangecheck(v, k)` proves `v.unsigned ⊆ [0, 2^k)`, but harvesting it
            // needs an assumption pre-pass that skips guard-nested checks, so it stays unmodelled.
            _ => {
                for vid in instr.get_results() {
                    let r = ValueRange::for_type(types.get_value_type(*vid), field);
                    Self::overwrite(bounds, *vid, r, changed);
                }
            }
        }
    }

    /// The transfer for `BinaryArithOp`, split out for room to breathe.
    ///
    /// `lhs_matches` / `rhs_matches` say whether each operand is the same width as the result.
    /// Only shifts legitimately mix widths — the type analysis widens the result to the wider of
    /// the two, and a shift amount is typically a narrow integer — and a shift reads its amount as
    /// a count rather than as a bit pattern, so it needs no width agreement. Every other rule here
    /// reasons about the result's bit pattern and gives up if an operand is not a reading of it.
    #[allow(clippy::too_many_arguments)]
    fn binary_arith(
        kind: BinaryArithOpKind,
        width: Width,
        signed: bool,
        l: &ValueRange,
        r: &ValueRange,
        lhs_matches: bool,
        rhs_matches: bool,
        rhs_type: &Type,
    ) -> ValueRange {
        use BinaryArithOpKind::*;

        // An unreachable operand makes the result unreachable. Propagating it matters: the bitwise
        // rules below read `hi` directly, and ⊥ spells `[1, 0]`, whose `hi` is a perfectly
        // plausible-looking zero.
        if l.is_empty() || r.is_empty() {
            return ValueRange::empty(width);
        }

        match kind {
            Add | Sub | Mul => {
                if !(lhs_matches && rhs_matches) {
                    return ValueRange::full(width);
                }
                // The same formula in both readings, because the operation is a ring homomorphism
                // modulo the width: `dec` commutes with `+`, `−` and `×`.
                let (raw_u, raw_s) = match kind {
                    Add => (l.unsigned().add(r.unsigned()), l.signed().add(r.signed())),
                    Sub => (l.unsigned().sub(r.unsigned()), l.signed().sub(r.signed())),
                    _ => (l.unsigned().mul(r.unsigned()), l.signed().mul(r.signed())),
                };
                Self::wrap_or_trap(width, signed, raw_u, raw_s)
            }

            Div | Mod => {
                let Width::Bits(_) = width else {
                    // A field `Div` is multiplication by a modular inverse, which no interval
                    // bounds at all, and a field `Mod` does not exist.
                    return ValueRange::full(width);
                };
                if !(lhs_matches && rhs_matches) {
                    return ValueRange::full(width);
                }
                // The reading the instruction is performed in. Unsigned division is performed on
                // the raw pattern; signed division on its two's-complement value.
                let (dividend, divisor) = if signed {
                    (l.signed(), r.signed())
                } else {
                    (l.unsigned(), r.unsigned())
                };
                // Signed division truncates toward zero while `div_const_pos` floors, and the two
                // agree only for a non-negative dividend. An unsigned reading satisfies that by
                // construction, so this gate is now vacuous on the unsigned side.
                if !dividend.is_non_negative() {
                    return ValueRange::full(width);
                }
                let math = match (kind, divisor.lo(), divisor.hi()) {
                    (Div, Some(lo), Some(hi)) if lo == hi && lo.is_positive() => {
                        dividend.div_const_pos(lo)
                    }
                    // `x % d < d`, and a non-negative `x % d` is also no larger than `x` itself.
                    (Mod, Some(lo), Some(hi)) if lo.is_positive() => {
                        let mut cap = hi - BigInt::one();
                        if let Some(d) = dividend.hi()
                            && d < &cap
                        {
                            cap = d.clone();
                        }
                        Interval::closed(BigInt::zero(), cap)
                    }
                    _ => return ValueRange::full(width),
                };
                Self::from_reading(width, signed, math)
            }

            And | Or | Xor => {
                if !matches!(width, Width::Bits(_)) || !(lhs_matches && rhs_matches) {
                    return ValueRange::full(width);
                }
                // Bitwise operations act on the raw pattern, so the unsigned readings bound them
                // unconditionally. The old rule needed both operands to be non-negative because it
                // had only the mathematical reading to work with; here the gate is vacuous.
                let (Some(lh), Some(rh)) = (l.unsigned().hi(), r.unsigned().hi()) else {
                    return ValueRange::full(width);
                };
                let cap = match kind {
                    // `x & y <= min(x, y)`.
                    And => lh.min(rh).clone(),
                    // `x | y` and `x ^ y` fit in as many bits as the wider operand needs.
                    _ => next_pow2_minus_one(lh.max(rh)),
                };
                ValueRange::from_unsigned(width, Interval::closed(BigInt::zero(), cap))
            }

            Shl | Shr => {
                let Width::Bits(n) = width else {
                    return ValueRange::full(width);
                };
                if !lhs_matches {
                    return ValueRange::full(width);
                }
                // The amount is a count, read at its own declared signedness, exactly as the
                // constant folder decodes it.
                let amount = r.by_type(rhs_type);
                let constant = match (amount.lo(), amount.hi()) {
                    (Some(a), Some(b)) if a == b && !a.is_negative() => {
                        a.to_usize().filter(|k| *k < n)
                    }
                    _ => None,
                };
                match (kind, constant) {
                    // A left shift is a multiply by `2^k`, modelled as *wrapping* rather than
                    // trapping: the backends mask an out-of-range shift while the constant folder
                    // refuses one, and the wrapping reading is sound under either reading of an
                    // overflowing shift.
                    (Shl, Some(k)) => {
                        let factor = Interval::singleton(BigInt::one() << k);
                        ValueRange::new(
                            width,
                            l.unsigned().mul(&factor).wrap_to_unsigned_bits(n),
                            l.signed().mul(&factor).wrap_to_signed_bits(n),
                        )
                    }
                    // A right shift is a floored divide in whichever reading it is performed:
                    // logical on an unsigned type, arithmetic on a signed one. Both floor, which
                    // is what `div_const_pos` does.
                    (Shr, Some(k)) => {
                        let divisor = BigInt::one() << k;
                        Self::from_reading(
                            width,
                            signed,
                            if signed {
                                l.signed().div_const_pos(&divisor)
                            } else {
                                l.unsigned().div_const_pos(&divisor)
                            },
                        )
                    }
                    // An unknown amount is still known to be a non-negative count, and a right
                    // shift is monotone in it: it can only move a value toward zero, or toward −1
                    // if the shift is arithmetic and the value is negative.
                    (Shr, None) if signed => match (l.signed().lo(), l.signed().hi()) {
                        (Some(a), Some(b)) => ValueRange::from_signed(
                            width,
                            Interval::closed(
                                if a.is_negative() {
                                    a.clone()
                                } else {
                                    BigInt::zero()
                                },
                                if b.is_negative() {
                                    -BigInt::one()
                                } else {
                                    b.clone()
                                },
                            ),
                        ),
                        _ => ValueRange::full(width),
                    },
                    (Shr, None) => match l.unsigned().hi() {
                        Some(h) => ValueRange::from_unsigned(
                            width,
                            Interval::closed(BigInt::zero(), h.clone()),
                        ),
                        None => ValueRange::full(width),
                    },
                    (Shl, None) => ValueRange::full(width),
                    _ => unreachable!("kind is Shl or Shr"),
                }
            }
        }
    }

    /// Combine the two readings of an `Add`, `Sub` or `Mul` result.
    ///
    /// Both were computed by the same formula, but only one of them is *exact*: the declared type
    /// traps on overflow in its own reading, so an out-of-range result there is a program failure
    /// and may be intersected away. The other reading is merely congruent modulo the width, and is
    /// wrapped into its window instead.
    ///
    /// Intersecting both would be unsound. A `u32` add of `2³¹ − 1` and `1` is a perfectly legal
    /// operation whose signed reading leaves `[−2³¹, 2³¹)`; capping that reading too would report
    /// the whole thing unreachable.
    fn wrap_or_trap(width: Width, signed: bool, raw_u: Interval, raw_s: Interval) -> ValueRange {
        match width {
            Width::Bits(n) if signed => ValueRange::new(
                width,
                raw_u.wrap_to_unsigned_bits(n),
                raw_s.intersect(&Interval::signed_full(n)),
            ),
            Width::Bits(n) => ValueRange::new(
                width,
                raw_u.intersect(&Interval::unsigned_full(n)),
                raw_s.wrap_to_signed_bits(n),
            ),
            // Field arithmetic wraps rather than trapping — which is exactly why
            // `LowerSideEffectFreeGuards` may drop the guard on it — so nothing may be intersected
            // away here.
            Width::Field(f) => ValueRange::from_unsigned(width, raw_u.wrap_to_field(f)),
            Width::NonScalar => ValueRange::full(width),
        }
    }

    /// A range built from whichever reading an instruction was performed in.
    fn from_reading(width: Width, signed: bool, math: Interval) -> ValueRange {
        if signed {
            ValueRange::from_signed(width, math)
        } else {
            ValueRange::from_unsigned(width, math)
        }
    }

    /// ⊥ when an input was unreachable, and "no information" otherwise.
    fn unknown_or_empty(width: Width, empty: bool) -> ValueRange {
        if empty {
            ValueRange::empty(width)
        } else {
            ValueRange::full(width)
        }
    }

    /// Store a computed range as the range of `result`, constraining it to the declared width.
    fn set(
        bounds: &mut HashMap<ValueId, ValueRange>,
        result: ValueId,
        types: &FunctionTypeInfo,
        field: FieldConfig,
        range: ValueRange,
        changed: &mut bool,
    ) {
        let width = Width::of_type(types.get_value_type(result), field);
        Self::overwrite(bounds, result, range.constrain_to(width), changed);
    }
}

impl Analysis for ValueRanges {
    fn dependencies() -> Vec<AnalysisId> {
        vec![FlowAnalysis::id(), TypeInfo::id()]
    }

    fn compute(ssa: &HLSSA, store: &AnalysisStore) -> Self {
        let cfg = store.get::<FlowAnalysis>();
        let types = store.get::<TypeInfo>();
        ValueRangeAnalysis::new().run(ssa, cfg, types)
    }
}

// INTERVAL
// ================================================================================================

/// A closed integer interval `[lo, hi]` over the integers ℤ, with `None` endpoints representing
/// −∞ / +∞.
///
/// Top is `(None, None)`, and the empty interval is any pair with `lo > hi` (we normalize such
/// pairs back to `EMPTY`).
///
/// This is a _calculator_, not a domain element. It carries no width and no signedness, so the same
/// type expresses an unsigned reading, a signed reading, a magnitude bound and a quotient bound.
/// The domain element that pairs two of these is [`ValueRange`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Interval {
    lo: Option<BigInt>,
    hi: Option<BigInt>,
}

/// Construction and the lattice operations.
///
/// Every constructor normalizes an inverted `lo > hi` pair to the single [`Interval::empty`]
/// representation, so ⊥ is unique and the analysis' equality-driven fixpoint cannot oscillate
/// between two spellings of the same interval.
impl Interval {
    /// Lower bound; `None` means −∞.
    pub fn lo(&self) -> Option<&BigInt> {
        self.lo.as_ref()
    }

    /// Upper bound; `None` means +∞.
    pub fn hi(&self) -> Option<&BigInt> {
        self.hi.as_ref()
    }

    /// `(-∞, +∞)`.
    pub fn top() -> Self {
        Self { lo: None, hi: None }
    }

    /// The unique empty interval (used as the bottom of the lattice). Any
    /// "lo > hi" interval is normalized to this representation.
    pub fn empty() -> Self {
        Self {
            lo: Some(BigInt::one()),
            hi: Some(BigInt::zero()),
        }
    }

    pub fn is_empty(&self) -> bool {
        match (&self.lo, &self.hi) {
            (Some(l), Some(h)) => l > h,
            _ => false,
        }
    }

    pub fn singleton<I: Into<BigInt>>(v: I) -> Self {
        let v: BigInt = v.into();
        Self {
            lo: Some(v.clone()),
            hi: Some(v),
        }
    }

    /// Closed interval `[a, b]`. Returns `EMPTY` if `a > b`.
    pub fn closed<A: Into<BigInt>, B: Into<BigInt>>(a: A, b: B) -> Self {
        let lo: BigInt = a.into();
        let hi: BigInt = b.into();
        if lo > hi {
            Self::empty()
        } else {
            Self {
                lo: Some(lo),
                hi: Some(hi),
            }
        }
    }

    /// `[0, 2^bits − 1]`.
    pub fn unsigned_full(bits: usize) -> Self {
        Self::closed(BigInt::zero(), (BigInt::one() << bits) - BigInt::one())
    }

    /// `[−2^(bits−1), 2^(bits−1) − 1]`.
    pub fn signed_full(bits: usize) -> Self {
        if bits == 0 {
            return Self::singleton(0);
        }
        let half = BigInt::one() << (bits - 1);
        Self::closed(-half.clone(), half - BigInt::one())
    }

    /// `[0, p − 1]` — the integer range of a Field element.
    pub fn field_top(field: FieldConfig) -> Self {
        // FIELD-ASSUMPTION: L4-modulus-query
        Self::closed(BigInt::zero(), field_modulus(field) - BigInt::one())
    }

    /// Lattice join — smallest interval containing both inputs.
    pub fn join(&self, other: &Self) -> Self {
        if self.is_empty() {
            return other.clone();
        }
        if other.is_empty() {
            return self.clone();
        }
        Self {
            lo: min_lo(self.lo.as_ref(), other.lo.as_ref()),
            hi: max_hi(self.hi.as_ref(), other.hi.as_ref()),
        }
    }

    /// Lattice meet — the intersection of the two intervals.
    pub fn intersect(&self, other: &Self) -> Self {
        if self.is_empty() || other.is_empty() {
            return Self::empty();
        }
        let lo = max_lo(self.lo.as_ref(), other.lo.as_ref());
        let hi = min_hi(self.hi.as_ref(), other.hi.as_ref());
        match (&lo, &hi) {
            (Some(l), Some(h)) if l > h => Self::empty(),
            _ => Self { lo, hi },
        }
    }
}

/// Geometric queries regarding where the interval sits relative to zero, and whether it is
/// contained in the range of a fixed-width representation.
///
/// These are plain _set_ predicates: they ask whether every member of the interval has the
/// property, so all of them hold vacuously for ⊥. That is the mathematically correct answer, and
/// consumers reasoning about set containment depend on it. Anything about to **elide a
/// constraint** wants the proof-strength predicates below instead.
impl Interval {
    pub fn is_non_negative(&self) -> bool {
        self.lo.as_ref().is_some_and(|l| !l.is_negative())
    }

    /// True iff `v` is a member of the interval. ⊥ contains nothing.
    pub fn contains(&self, v: &BigInt) -> bool {
        !self.is_empty()
            && self.lo.as_ref().is_none_or(|l| l <= v)
            && self.hi.as_ref().is_none_or(|h| v <= h)
    }

    pub fn is_non_positive(&self) -> bool {
        self.hi.as_ref().is_some_and(|h| !h.is_positive())
    }

    /// True iff every value in the interval fits in `bits`-bit unsigned representation (i.e. is in
    /// `[0, 2^bits)`).
    pub fn fits_in_unsigned_bits(&self, bits: usize) -> bool {
        let cap = BigInt::one() << bits;
        self.lo.as_ref().is_some_and(|l| !l.is_negative())
            && self.hi.as_ref().is_some_and(|h| h < &cap)
    }

    /// True iff every value in the interval fits in `bits`-bit two's-complement signed
    /// representation (i.e. is in `[−2^(bits−1), 2^(bits−1))`).
    pub fn fits_in_signed_bits(&self, bits: usize) -> bool {
        if bits == 0 {
            return matches!((&self.lo, &self.hi), (Some(l), Some(h)) if l.is_zero() && h.is_zero());
        }
        let half = BigInt::one() << (bits - 1);
        self.lo.as_ref().is_some_and(|l| l >= &(-half.clone()))
            && self.hi.as_ref().is_some_and(|h| h < &half)
    }

    /// True iff every value, viewed as `bits`-bit two's-complement, has its sign bit set to
    /// . Equivalent to `[0, 2^(bits-1))` containment.
    pub fn is_non_negative_in_signed(&self, bits: usize) -> bool {
        if bits == 0 {
            return false;
        }
        let half = BigInt::one() << (bits - 1);
        self.lo.as_ref().is_some_and(|l| !l.is_negative())
            && self.hi.as_ref().is_some_and(|h| h < &half)
    }
}

/// Proof-strength counterparts of the containment queries above, answering `false` on ⊥.
///
/// ⊥ is not a free pass. The analysis derives it _from_ the constraints the circuit is about to
/// emit, so dropping a constraint on the strength of ⊥ is circular: `u32 x − y` with `x = [0,0]`
/// and `y = [1,1]` gives a raw `[−1,−1]` that the declared type caps to ⊥. The only reason that
/// value cannot occur is the very rangecheck the caller would then skip — leaving a prover-chosen
/// value satisfying a circuit that should be unsatisfiable.
///
/// These are deliberately separate functions rather than a change to the set predicates, so that
/// each family remains clear about what knowledge it actually provides.
impl Interval {
    /// [`Interval::fits_in_unsigned_bits`], but `false` on ⊥, for callers eliding a constraint.
    pub fn proves_fits_in_unsigned_bits(&self, bits: usize) -> bool {
        !self.is_empty() && self.fits_in_unsigned_bits(bits)
    }

    /// [`Interval::fits_in_signed_bits`], but `false` on ⊥, for callers eliding a constraint.
    pub fn proves_fits_in_signed_bits(&self, bits: usize) -> bool {
        !self.is_empty() && self.fits_in_signed_bits(bits)
    }

    /// [`Interval::is_non_negative_in_signed`], but `false` on ⊥ , for callers eliding a
    /// constraint, or replacing a computed sign bit with a constant.
    pub fn proves_non_negative_in_signed(&self, bits: usize) -> bool {
        !self.is_empty() && self.is_non_negative_in_signed(bits)
    }
}

/// Ordinary interval arithmetic over ℤ.
///
/// These are exact hulls of the pointwise result, with no width or signedness applied: the caller
/// is responsible for reinterpreting or constraining the result to whatever representation it is
/// destined for. ⊥ is _absorbing_ throughout, and an unbounded endpoint stays unbounded.
impl Interval {
    pub fn add(&self, other: &Self) -> Self {
        if self.is_empty() || other.is_empty() {
            return Self::empty();
        }
        Self {
            lo: opt_add(self.lo.as_ref(), other.lo.as_ref()),
            hi: opt_add(self.hi.as_ref(), other.hi.as_ref()),
        }
    }

    pub fn sub(&self, other: &Self) -> Self {
        if self.is_empty() || other.is_empty() {
            return Self::empty();
        }
        // [a, b] - [c, d] = [a - d, b - c]
        Self {
            lo: opt_sub(self.lo.as_ref(), other.hi.as_ref()),
            hi: opt_sub(self.hi.as_ref(), other.lo.as_ref()),
        }
    }

    pub fn mul(&self, other: &Self) -> Self {
        if self.is_empty() || other.is_empty() {
            return Self::empty();
        }

        // The four "extreme products" between endpoint pairs determine the hull. `opt_mul` returns
        // `None` when an endpoint is ±∞ (and the other factor is non-zero); below, any `None` in
        // the candidates forces the corresponding side to ±∞ too.
        let products = [
            opt_mul(self.lo.as_ref(), other.lo.as_ref()),
            opt_mul(self.lo.as_ref(), other.hi.as_ref()),
            opt_mul(self.hi.as_ref(), other.lo.as_ref()),
            opt_mul(self.hi.as_ref(), other.hi.as_ref()),
        ];
        let mut lo = products[0].clone();
        let mut hi = products[0].clone();

        for p in &products[1..] {
            lo = match (&lo, p) {
                (None, _) | (_, None) => None,
                (Some(a), Some(b)) if b < a => Some(b.clone()),
                _ => lo,
            };
            hi = match (&hi, p) {
                (None, _) | (_, None) => None,
                (Some(a), Some(b)) if b > a => Some(b.clone()),
                _ => hi,
            };
        }

        Self { lo, hi }
    }

    pub fn neg(&self) -> Self {
        if self.is_empty() {
            return Self::empty();
        }
        Self {
            lo: self.hi.as_ref().map(|h| -h),
            hi: self.lo.as_ref().map(|l| -l),
        }
    }

    /// `[a, b] / d` for a constant divisor `d > 0`.
    pub fn div_const_pos(&self, d: &BigInt) -> Self {
        debug_assert!(d.is_positive());
        if self.is_empty() {
            return Self::empty();
        }
        Self {
            lo: self.lo.as_ref().map(|l| floor_div(l, d)),
            hi: self.hi.as_ref().map(|h| floor_div(h, d)),
        }
    }
}

/// Modular reduction into a fixed window — the *wrapping* half of a change of representation.
///
/// A run of consecutive integers stays consecutive under reduction, so the reduction is exact
/// unless the run is wider than the window or straddles its far edge; in those cases it saturates
/// to the whole window, which is the widest that reading could ever have been. Saturating rather
/// than adopting a wrapped `[lo > hi]` range is what keeps every component a plain interval, and is
/// what makes the [`ValueRange::join`] of two ranges unique.
impl Interval {
    /// Reduce modulo `2^bits` into `[0, 2^bits)`.
    pub fn wrap_to_unsigned_bits(&self, bits: usize) -> Self {
        self.wrap_into(&BigInt::zero(), &(BigInt::one() << bits))
    }

    /// Reduce modulo `2^bits` into `[−2^(bits−1), 2^(bits−1))`.
    pub fn wrap_to_signed_bits(&self, bits: usize) -> Self {
        let size = BigInt::one() << bits;
        let start = if bits == 0 {
            BigInt::zero()
        } else {
            -(BigInt::one() << (bits - 1))
        };
        self.wrap_into(&start, &size)
    }

    /// Reduce modulo `p` into `[0, p)`.
    pub fn wrap_to_field(&self, field: FieldConfig) -> Self {
        // FIELD-ASSUMPTION: L4-modulus-query
        self.wrap_into(&BigInt::zero(), &field_modulus(field))
    }

    /// Reduce modulo `size` into `[start, start + size)`.
    fn wrap_into(&self, start: &BigInt, size: &BigInt) -> Self {
        debug_assert!(size.is_positive());
        let whole_window = || Self::closed(start.clone(), start + size - BigInt::one());

        if self.is_empty() {
            return Self::empty();
        }
        let (Some(lo), Some(hi)) = (self.lo(), self.hi()) else {
            return whole_window();
        };
        let span = hi - lo;
        if &span >= size {
            return whole_window();
        }

        let offset = {
            let rem = (lo - start) % size;
            if rem.is_negative() { rem + size } else { rem }
        };
        let wrapped_lo = start + offset;
        let wrapped_hi = &wrapped_lo + span;
        if &wrapped_hi < &(start + size) {
            Self::closed(wrapped_lo, wrapped_hi)
        } else {
            whole_window()
        }
    }
}

// WIDTH
// ================================================================================================

/// The bit pattern a [`ValueRange`] is two readings of.
///
/// `NonScalar` is the "this value is not a number" case (function pointers, blobs, aggregates);
/// both readings are top.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Width {
    Bits(usize),
    Field(FieldConfig),
    NonScalar,
}

impl Width {
    /// The width implied by a declared SSA type, looking through `WitnessOf`.
    pub fn of_type(ty: &Type, field: FieldConfig) -> Self {
        match &ty.strip_witness().expr {
            TypeExpr::U(n) | TypeExpr::I(n) => Width::Bits(*n),
            TypeExpr::Field => Width::Field(field),
            _ => Width::NonScalar,
        }
    }

    /// The widest interval the unsigned reading of this width can take.
    fn unsigned_full(self) -> Interval {
        match self {
            Width::Bits(n) => Interval::unsigned_full(n),
            Width::Field(f) => Interval::field_top(f),
            Width::NonScalar => Interval::top(),
        }
    }

    /// The widest interval the signed reading of this width can take.
    ///
    /// A field element has no second reading, so its "signed" reading is its unsigned one.
    fn signed_full(self) -> Interval {
        match self {
            Width::Bits(n) => Interval::signed_full(n),
            Width::Field(f) => Interval::field_top(f),
            Width::NonScalar => Interval::top(),
        }
    }
}

// VALUE RANGE
// ================================================================================================

/// The domain element, providing the unsigned and signed readings of one bit pattern, plus the
/// width they are readings _of_.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValueRange {
    width: Width,
    unsigned: Interval,
    signed: Interval,
}

impl ValueRange {
    /// Everything the width admits: the initial state of every value, and the answer for anything
    /// the analysis does not model.
    pub fn full(width: Width) -> Self {
        Self {
            width,
            unsigned: width.unsigned_full(),
            signed: width.signed_full(),
        }
    }

    /// The unreachable range. No bit pattern satisfies both readings.
    ///
    /// A [`Width::NonScalar`] has no bit pattern to be unreachable, and its invariant forces both
    /// readings to top, so this returns [`ValueRange::full`] for that width.
    pub fn empty(width: Width) -> Self {
        Self::new(width, Interval::empty(), Interval::empty())
    }

    /// Initial bound for a value of the given declared type, looking through `WitnessOf`.
    ///
    /// Non-numeric types get TOP.
    pub fn for_type(ty: &Type, field: FieldConfig) -> Self {
        Self::full(Width::of_type(ty, field))
    }

    /// Both readings at once, reduced. This is the only route into the struct.
    fn new(width: Width, unsigned: Interval, signed: Interval) -> Self {
        let reduced = Self {
            width,
            unsigned,
            signed,
        }
        .normalized();
        debug_assert_eq!(
            reduced.clone().normalized(),
            reduced,
            "the reduction must be idempotent, or the fixed point can oscillate"
        );
        reduced
    }

    /// A range known through its **unsigned** reading; whatever the signed reading follows from is
    /// recovered by the reduction.
    pub fn from_unsigned(width: Width, unsigned: Interval) -> Self {
        Self::new(width, unsigned, width.signed_full())
    }

    /// A range known through its **signed** reading; the unsigned reading is recovered by the
    /// reduction.
    pub fn from_signed(width: Width, signed: Interval) -> Self {
        Self::new(width, width.unsigned_full(), signed)
    }

    /// The reduced-product step: `α ∘ γ`, the tightest pair of intervals denoting the same set of
    /// bit patterns as the pair it is given.
    ///
    /// γ is a union of at most two runs — the patterns whose sign bit is clear, on which the two
    /// readings agree, and those whose sign bit is set, on which they differ by `2ⁿ`. Recovering
    /// those two runs and re-hulling them is therefore closed-form, and being a function of γ alone
    /// it is **canonical**: two pairs denoting the same set reduce to the same bytes.
    ///
    /// That canonicity is load-bearing rather than cosmetic. `ValueRangeAnalysis::overwrite`
    /// replaces rather than meets, and drives its `changed` flag off structural equality, so two
    /// spellings of one set would leave the fixed point oscillating until `ITER_LIMIT` cut it off.
    /// The reduction is also idempotent — proved by the same argument, since the recovered runs are
    /// unchanged by a second pass — which the tests pin.
    fn normalized(self) -> Self {
        match self.width {
            // A non-scalar has no bit pattern, so it must have exactly one representation: any
            // other would let `opt_mul`'s `∞·0 = 0` mint a second spelling of "no information".
            Width::NonScalar => Self::full(Width::NonScalar),

            // A field element's canonical integer *is* its value, so the two readings are one
            // reading and the reduction is a plain intersection.
            Width::Field(f) => {
                let both = self
                    .unsigned
                    .intersect(&self.signed)
                    .intersect(&Interval::field_top(f));
                Self {
                    width: self.width,
                    unsigned: both.clone(),
                    signed: both,
                }
            }

            Width::Bits(n) => {
                let two_n = BigInt::one() << n;
                // A zero-width pattern has no sign bit, which the empty negative half encodes.
                let half = if n == 0 {
                    two_n.clone()
                } else {
                    BigInt::one() << (n - 1)
                };
                let shift = Interval::singleton(two_n.clone());

                // The sign-bit-clear patterns, where `dec(x) = x` and both readings constrain the
                // same integer.
                let positive = self
                    .unsigned
                    .intersect(&self.signed)
                    .intersect(&Interval::closed(BigInt::zero(), &half - BigInt::one()));
                // The sign-bit-set patterns, where `dec(x) = x − 2ⁿ`, so the signed reading
                // constrains `x` only after being shifted back up.
                let negative = self
                    .unsigned
                    .intersect(&self.signed.add(&shift))
                    .intersect(&Interval::closed(half, two_n - BigInt::one()));

                if positive.is_empty() && negative.is_empty() {
                    return Self {
                        width: self.width,
                        unsigned: Interval::empty(),
                        signed: Interval::empty(),
                    };
                }
                Self {
                    width: self.width,
                    unsigned: positive.join(&negative),
                    signed: positive.join(&negative.sub(&shift)),
                }
            }
        }
    }

    /// A **wrapping** change of width: the same bit pattern, zero-extended or truncated to `width`
    /// and re-read there.
    ///
    /// This is what `Cast` does at runtime — the LLSSA lowering emits a `zext` or a `truncate` for
    /// every integer target, and builds the low limbs directly for a field one. It never turns a
    /// reachable value into an unreachable one. Either direction discards the old signed reading
    /// (every arm rebuilds from the unsigned one) since it was the sign at the *old* width and says
    /// nothing at the new one; nothing is lost by that, because a reduced range has already folded
    /// it into the unsigned reading.
    pub fn reinterpret_to(&self, width: Width) -> Self {
        if self.width == width {
            return self.clone();
        }
        match width {
            Width::Bits(n) => Self::from_unsigned(width, self.unsigned.wrap_to_unsigned_bits(n)),
            Width::Field(f) => Self::from_unsigned(width, self.unsigned.wrap_to_field(f)),
            Width::NonScalar => Self::full(Width::NonScalar),
        }
    }

    /// A **trapping** change of width: the value is asserted to be representable at `width`, and
    /// anything that is not is discarded.
    ///
    /// ⊥ is a legitimate result — it means the assertion cannot hold — which is precisely why the
    /// consumers that elide constraints must ask the `proves_*` predicates rather than the plain
    /// set ones.
    ///
    /// Only the **unsigned** reading crosses the width boundary. It is the raw integer, so it means
    /// the same thing at any width; the signed reading is two's complement at the *old* width and
    /// is a statement about a sign bit that has moved, so it is dropped and recovered by the
    /// reduction rather than carried over. Carrying it is how a widening used to manufacture ⊥: a
    /// `Bits(8)` `[200, 255]` has the signed reading `[−56, −1]`, and re-reading that at `Bits(16)`
    /// asks for a pattern that is both `≥ 200` and `≥ 2^15`, which nothing satisfies.
    pub fn constrain_to(&self, width: Width) -> Self {
        if self.width == width {
            return self.clone();
        }
        Self::new(
            width,
            self.unsigned.intersect(&width.unsigned_full()),
            width.signed_full(),
        )
    }

    pub fn width(&self) -> Width {
        self.width
    }

    /// The unsigned reading — the raw bit pattern read as a non-negative integer. This is what a
    /// lowering rule wants whenever it is about to `cast_to_field` the value.
    pub fn unsigned(&self) -> &Interval {
        &self.unsigned
    }

    /// The two's-complement signed reading — the *mathematical* value of a signed integer.
    pub fn signed(&self) -> &Interval {
        &self.signed
    }

    /// True when the value is provably unreachable.
    pub fn is_empty(&self) -> bool {
        self.unsigned.is_empty() || self.signed.is_empty()
    }

    /// True when, read as two's complement at this width, the sign bit is provably 0.
    ///
    /// Answered from the **unsigned** reading, capped below `2^(n−1)`, by way of
    /// [`ValueRange::is_non_negative_at_width`] at this range's own width — where the reduction
    /// makes that equivalent to asking whether the signed reading goes negative, so nothing is
    /// given up by not consulting it. Consumers that only want "is this value non-negative" should
    /// use this rather than picking a reading by hand.
    pub fn is_non_negative_as_signed(&self) -> bool {
        match self.width {
            Width::Bits(n) => self.is_non_negative_at_width(n),
            // A field element's canonical integer is always non-negative.
            Width::Field(_) => !self.is_empty(),
            Width::NonScalar => false,
        }
    }

    /// As [`ValueRange::is_non_negative_as_signed`], but for a caller that already knows the width
    /// it means to read the pattern at — e.g. `SExt`, which carries its own `from_bits`.
    ///
    /// Only the **unsigned** reading answers this, because it is the one that means the same thing
    /// at every width: capped below `2^(bits−1)`, bit `bits − 1` of the pattern is clear whatever
    /// width the pattern is held at. The signed reading is the two's-complement value at *this
    /// range's own* width and says nothing about a narrower one — a `U(32)` pinned to `200` has a
    /// non-negative signed reading and a set bit 7, so consulting it here would report the pattern
    /// `0xC8` as sign-bit-clear and let a caller hardcode the wrong sign.
    ///
    /// Nothing is lost by leaving it out. At the range's own width the reduction already makes the
    /// two tests equivalent: a reduced `Bits(n)` range has a sign-bit-set part iff its unsigned
    /// hull reaches `2^(n−1)` iff its signed hull goes negative.
    ///
    /// ⊥ answers `false`: this is a proof query, and its callers use it to drop a computed sign bit
    /// in favor of a constant. See the proof-strength predicates on [`Interval`].
    pub fn is_non_negative_at_width(&self, bits: usize) -> bool {
        !self.is_empty() && self.unsigned.is_non_negative_in_signed(bits)
    }

    /// True when the value provably is **not** the bit pattern `raw`, named by its unsigned (raw)
    /// reading.
    ///
    /// γ admits a pattern only when *both* readings admit it, so excluding it from either one is
    /// enough. Both are consulted rather than just the unsigned reading because the reduction
    /// leaves each component a *hull*, and γ may have a hole in the middle of that hull: at `n = 8`
    /// with `u = [0, 255]` and `s = [−1, 1]`, γ is `{255, 0, 1}`, and it is the signed reading that
    /// rules out the pattern `5`.
    ///
    /// ⊥ answers `false`, like every other `proves_*` query. An unreachable value cannot be `raw`,
    /// but the analysis derives ⊥ *from* the constraints the caller is about to drop, so answering
    /// `true` would be circular. See the proof-strength predicates on [`Interval`].
    pub fn proves_excludes_pattern(&self, raw: &BigInt) -> bool {
        if self.is_empty() {
            return false;
        }
        match self.width {
            // No bit pattern to speak of, and both readings are top by invariant.
            Width::NonScalar => false,
            // One reading, and the pattern *is* the canonical integer.
            Width::Field(_) => !self.unsigned.contains(raw),
            Width::Bits(n) => {
                !self.unsigned.contains(raw) || !self.signed.contains(&decode_signed(n, raw))
            }
        }
    }

    /// Lattice join, componentwise and then reduced.
    pub fn join(&self, other: &Self) -> Self {
        debug_assert_eq!(
            self.width, other.width,
            "joining ValueRanges of different widths"
        );
        Self::new(
            self.width,
            self.unsigned.join(&other.unsigned),
            self.signed.join(&other.signed),
        )
    }

    /// Lattice meet, componentwise and then reduced.
    pub fn intersect(&self, other: &Self) -> Self {
        debug_assert_eq!(
            self.width, other.width,
            "intersecting ValueRanges of different widths"
        );
        Self::new(
            self.width,
            self.unsigned.intersect(&other.unsigned),
            self.signed.intersect(&other.signed),
        )
    }

    /// The interval the *declared type* says is this value's mathematical reading: the signed one
    /// for a signed integer type, the unsigned one otherwise.
    ///
    /// FIXME: This exists only while signedness still lives in the type.
    pub fn by_type(&self, ty: &Type) -> &Interval {
        match &ty.strip_witness().expr {
            TypeExpr::I(_) => &self.signed,
            _ => &self.unsigned,
        }
    }
}

/// `dec_n` from the module doc: the two's-complement reading of the `bits`-wide pattern `raw`.
///
/// A `raw` outside `[0, 2^bits)` is not a pattern of this width at all; the only caller pairs this
/// with a containment test against the unsigned reading, which rejects such a value first.
fn decode_signed(bits: usize, raw: &BigInt) -> BigInt {
    if bits == 0 {
        return BigInt::zero();
    }
    let half = BigInt::one() << (bits - 1);
    if raw >= &half {
        raw - (BigInt::one() << bits)
    } else {
        raw.clone()
    }
}

/// Whether a declared type reads its bit pattern as two's-complement signed.
///
/// The domain itself is sign-agnostic; this is consulted only where the *operation* is not —
/// which reading an arithmetic opcode traps on, and whether `Div`, `Mod` and `Shr` are the signed
/// or the unsigned instruction.
///
/// FIXME: This exists only while signedness still lives in the type. From Stage 2 the opcode
/// carries it and this query disappears.
fn type_is_signed(ty: &Type) -> bool {
    matches!(&ty.strip_witness().expr, TypeExpr::I(_))
}

/// Whether a guarded operation can yield a zero it would not have computed.
///
/// This is the complement, restricted to scalars, of what
/// `LowerSideEffectFreeGuards::can_drop_guard` accepts. Everything that pass accepts is computed
/// unconditionally, so its guard says nothing about its value. Everything it refuses is rewritten
/// by `LowerPureGuards` into a branch on the operation's failure condition whose failing side
/// asserts the condition false and produces the result type's zero
/// (`pure_guards.rs::emit_guard_failure_default`, and the `TypeExpr::Field` arm of
/// `lower_divmod_guard`).
///
/// Integer `Add`/`Sub`/`Mul` are in the set and their field counterparts are not, because field
/// arithmetic cannot fail. `ArrayGet` and `ArraySet` also have failure branches, and their results
/// *can* be scalars — they are left out because the transfer does not model them either way: they
/// fall to the `_` arm, which answers with the full range of the declared type, and that already
/// contains the zero the failure branch would produce.
fn guard_may_produce_zero(inner: &OpCode, types: &FunctionTypeInfo) -> bool {
    use BinaryArithOpKind::*;
    match inner {
        OpCode::BinaryArithOp {
            kind: Add | Sub | Mul,
            lhs,
            ..
        } => !matches!(
            types.get_value_type(*lhs).strip_witness().expr,
            TypeExpr::Field
        ),
        OpCode::BinaryArithOp {
            kind: Div | Mod | Shl | Shr,
            ..
        } => true,
        _ => false,
    }
}

// FUNCTION VALUE RANGES
// ================================================================================================

pub struct FunctionValueRanges {
    values: HashMap<ValueId, ValueRange>,
}

impl FunctionValueRanges {
    /// Get the range for a value, returning an unconstrained non-scalar range if the value isn't
    /// in our map (e.g. fresh values created downstream of this analysis).
    pub fn get(&self, v: ValueId) -> ValueRange {
        self.values
            .get(&v)
            .cloned()
            .unwrap_or_else(|| ValueRange::full(Width::NonScalar))
    }

    pub fn try_get(&self, v: ValueId) -> Option<&ValueRange> {
        self.values.get(&v)
    }
}

// VALUE RANGES
// ================================================================================================

pub struct ValueRanges {
    functions: HashMap<FunctionId, FunctionValueRanges>,
}

impl ValueRanges {
    pub fn get_function(&self, id: FunctionId) -> &FunctionValueRanges {
        self.functions
            .get(&id)
            .expect("ValueRanges: function not found")
    }
}

// UTILITIES
// ================================================================================================

/// Floor division for `BigInt`, matching Rust integer semantics for positive divisors (rounds
/// toward −∞).
fn floor_div(a: &BigInt, d: &BigInt) -> BigInt {
    let (q, r) = (a / d, a % d);
    if !r.is_zero() && r.is_negative() != d.is_negative() {
        q - BigInt::one()
    } else {
        q
    }
}

/// `lo` operands: `None` = −∞ (the smallest possible). So the min is the one that's `None` if any,
/// otherwise the smaller `Some`.
fn min_lo(a: Option<&BigInt>, b: Option<&BigInt>) -> Option<BigInt> {
    match (a, b) {
        (None, _) | (_, None) => None,
        (Some(x), Some(y)) => Some(if x <= y { x.clone() } else { y.clone() }),
    }
}

/// `hi` operands: `None` = +∞.
fn max_hi(a: Option<&BigInt>, b: Option<&BigInt>) -> Option<BigInt> {
    match (a, b) {
        (None, _) | (_, None) => None,
        (Some(x), Some(y)) => Some(if x >= y { x.clone() } else { y.clone() }),
    }
}

/// `lo` operands meeting (intersection): we take the larger.
fn max_lo(a: Option<&BigInt>, b: Option<&BigInt>) -> Option<BigInt> {
    match (a, b) {
        (None, b) => b.cloned(),
        (a, None) => a.cloned(),
        (Some(x), Some(y)) => Some(if x >= y { x.clone() } else { y.clone() }),
    }
}

/// `hi` operands meeting: we take the smaller.
fn min_hi(a: Option<&BigInt>, b: Option<&BigInt>) -> Option<BigInt> {
    match (a, b) {
        (None, b) => b.cloned(),
        (a, None) => a.cloned(),
        (Some(x), Some(y)) => Some(if x <= y { x.clone() } else { y.clone() }),
    }
}

fn opt_add(a: Option<&BigInt>, b: Option<&BigInt>) -> Option<BigInt> {
    match (a, b) {
        (Some(x), Some(y)) => Some(x + y),
        _ => None,
    }
}

fn opt_sub(a: Option<&BigInt>, b: Option<&BigInt>) -> Option<BigInt> {
    match (a, b) {
        (Some(x), Some(y)) => Some(x - y),
        _ => None,
    }
}

fn opt_mul(a: Option<&BigInt>, b: Option<&BigInt>) -> Option<BigInt> {
    match (a, b) {
        (Some(x), Some(y)) => Some(x * y),
        // ±∞ * 0 is treated as 0 (consistent with interval arithmetic).
        (None, Some(z)) | (Some(z), None) if z.is_zero() => Some(BigInt::zero()),
        _ => None,
    }
}

/// Convert a `Constant::I(bits, encoded)` u128 bit pattern back to a signed `BigInt`.
///
/// Two's-complement decode for any `bits ∈ [1, 128]`.
fn signed_const_to_bigint(bits: usize, encoded: u128) -> BigInt {
    if bits == 0 {
        return BigInt::zero();
    }

    let value = BigInt::from(encoded);
    if bits > 128 {
        // No upper bits to interpret; the encoding carries at most 128 bits.
        return value;
    }

    // bits ∈ [1, 128]; build 2^bits and 2^(bits-1) without overflowing u128 (which would happen for
    // bits == 128 if we shifted u128 directly).
    let two_n = BigInt::one() << bits;
    let half = &two_n >> 1;

    if value < half { value } else { value - two_n }
}

// FIELD-ASSUMPTION: L4-modulus-query
/// The modulus of the configured field as a `BigInt`, read through the [`FieldConfig`] instance API
/// so that no concrete prime is named. Shared with the integer-lowering width gates.
pub fn field_modulus(field: FieldConfig) -> BigInt {
    let limbs = field.modulus_limbs();
    let bytes_le: Vec<u8> = limbs.iter().flat_map(|l| l.to_le_bytes()).collect();
    BigInt::from_bytes_le(Sign::Plus, &bytes_le)
}

/// Convert a Field element to a `BigInt` (always non-negative, in `[0, p)`).
fn field_to_bigint(f: &Field) -> BigInt {
    let limbs = f.into_bigint().0; // [u64; 4]
    let bytes_le: Vec<u8> = limbs.iter().flat_map(|l| l.to_le_bytes()).collect();
    BigInt::from_bytes_le(Sign::Plus, &bytes_le)
}

/// Pre-compute the singleton interval for every constant in the SSA's constant storage.
///
/// Constants are module-level singletons shared across functions, so this runs once before the
/// per-function fixed-point.
fn compute_constant_bounds(ssa: &HLSSA) -> HashMap<ValueId, ValueRange> {
    let field = ssa.field();
    ssa.const_snapshot()
        .iter()
        .map(|(vid, cv)| {
            let r = match cv.as_ref() {
                Constant::U(bits, v) => {
                    ValueRange::from_unsigned(Width::Bits(*bits), Interval::singleton(*v))
                }
                Constant::I(bits, encoded) => ValueRange::from_signed(
                    Width::Bits(*bits),
                    Interval::singleton(signed_const_to_bigint(*bits, *encoded)),
                ),
                Constant::Field(f) => ValueRange::from_unsigned(
                    Width::Field(field),
                    Interval::singleton(field_to_bigint(f)),
                ),
                Constant::FnPtr(_) | Constant::Blob(_) => ValueRange::full(Width::NonScalar),
            };
            (*vid, r)
        })
        .collect()
}

/// `next_power_of_two(m + 1) - 1` — the smallest `2^k - 1` that's `>= m`.
fn next_pow2_minus_one(m: &BigInt) -> BigInt {
    if m.is_zero() {
        return BigInt::zero();
    }
    let bits = m.bits(); // number of bits to represent m (BigInt::bits())
    // If m is already 2^k - 1, that's our answer; else 2^bits - 1.
    let candidate = (BigInt::one() << bits) - BigInt::one();
    if &candidate >= m {
        candidate
    } else {
        (BigInt::one() << (bits + 1)) - BigInt::one()
    }
}

// TESTS
// ================================================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::ssa::hlssa::builder::{HLEmitter, HLSSABuilder};

    fn f() -> FieldConfig {
        FieldConfig::bn254()
    }

    /// `[a, b]`, for readability in the tables below.
    fn iv(a: i64, b: i64) -> Interval {
        Interval::closed(a, b)
    }

    /// The concrete set a `Bits(n)` range denotes: every bit pattern admitted by *both* readings.
    /// This is γ, written out directly, and it is what the invariant tests compare against.
    fn gamma(r: &ValueRange, n: usize) -> Vec<i64> {
        gamma_of(n, r.unsigned(), r.signed())
    }

    /// γ of a raw, unreduced pair of readings — the specification the reduction has to preserve.
    fn gamma_of(n: usize, unsigned: &Interval, signed: &Interval) -> Vec<i64> {
        let two_n: i64 = 1 << n;
        let half = two_n / 2;
        (0..two_n)
            .filter(|x| {
                let dec = if *x >= half { x - two_n } else { *x };
                contains(unsigned, *x) && contains(signed, dec)
            })
            .collect()
    }

    fn contains(i: &Interval, v: i64) -> bool {
        i.contains(&BigInt::from(v))
    }

    #[test]
    fn excluding_a_pattern_is_exactly_non_membership_of_gamma() {
        // `proves_excludes_pattern` is not an approximation: γ admits a pattern iff *both* readings
        // do, so testing both is precisely the membership test. Brute-forced over every pattern at
        // `n = 8`, against γ computed independently.
        //
        // The pairs deliberately include ones the reduction rewrites (a wrap-straddling set, a
        // hole in the middle of the unsigned hull) — those are where consulting only the unsigned
        // reading would answer wrongly.
        let cases = [
            (iv(0, 255), iv(-1, 1)),    // wrap-straddling: γ = {255, 0, 1}
            (iv(0, 255), iv(-128, -1)), // the whole negative half
            (iv(16, 200), iv(-128, 127)),
            (iv(0, 0), iv(0, 0)),
            (iv(0, 255), iv(-128, 127)), // top
        ];
        for (u, s) in cases {
            let r = ValueRange::new(Width::Bits(8), u, s);
            let g = gamma(&r, 8);
            for x in 0..256i64 {
                assert_eq!(
                    r.proves_excludes_pattern(&BigInt::from(x)),
                    !g.contains(&x),
                    "pattern {x} against {r:?}"
                );
            }
        }
    }

    #[test]
    fn excluding_a_pattern_declines_on_bottom_and_non_scalars() {
        // ⊥ excludes every pattern set-theoretically, and answering so would be circular: the
        // analysis derives ⊥ from the very constraint the caller is asking permission to drop.
        let bottom = ValueRange::empty(Width::Bits(8));
        assert!(bottom.is_empty());
        for x in 0..256i64 {
            assert!(!bottom.proves_excludes_pattern(&BigInt::from(x)));
        }
        // A non-scalar has no bit pattern to reason about at all.
        for x in 0..4i64 {
            assert!(!ValueRange::full(Width::NonScalar).proves_excludes_pattern(&BigInt::from(x)));
        }
    }

    #[test]
    fn excluding_a_pattern_reads_a_field_as_its_canonical_integer() {
        // One reading, and no sign bit: the pattern *is* the value, so `p − 1` is `p − 1` and not
        // `−1`. The divmod discharge only ever asks a field range about zero, which is what the
        // first two assertions pin.
        let nonzero = ValueRange::from_unsigned(Width::Field(f()), iv(1, 9));
        assert!(nonzero.proves_excludes_pattern(&BigInt::zero()));
        let maybe_zero = ValueRange::from_unsigned(Width::Field(f()), iv(0, 9));
        assert!(!maybe_zero.proves_excludes_pattern(&BigInt::zero()));
        // Above the range but still a legal field element.
        assert!(nonzero.proves_excludes_pattern(&BigInt::from(10)));
        // Outside `[0, p)` entirely: not a field element, so trivially excluded.
        assert!(nonzero.proves_excludes_pattern(&field_modulus(f())));
    }

    #[test]
    fn excluding_a_pattern_finds_minus_one_and_int_min_through_either_reading() {
        // The three patterns the divmod discharge asks about, at `i8`: `0`, `−1` (raw `255`) and
        // `INT_MIN` (raw `128`). Each is nameable from either side, which is the point of asking
        // about patterns rather than about mathematical values.
        let minus_one = BigInt::from(255);
        let int_min = BigInt::from(128);

        // A tight *signed* reading rules out `−1` while the unsigned hull still contains `255`.
        let non_negative = ValueRange::from_signed(Width::Bits(8), iv(0, 127));
        assert!(non_negative.proves_excludes_pattern(&minus_one));
        assert!(non_negative.proves_excludes_pattern(&int_min));
        assert!(!non_negative.proves_excludes_pattern(&BigInt::zero()));

        // A tight *unsigned* reading rules out `INT_MIN` the other way round.
        let small = ValueRange::from_unsigned(Width::Bits(8), iv(1, 100));
        assert!(small.proves_excludes_pattern(&BigInt::zero()));
        assert!(small.proves_excludes_pattern(&int_min));
        assert!(small.proves_excludes_pattern(&minus_one));

        // Everything: nothing is excluded.
        let top = ValueRange::full(Width::Bits(8));
        assert!(!top.proves_excludes_pattern(&BigInt::zero()));
        assert!(!top.proves_excludes_pattern(&minus_one));
        assert!(!top.proves_excludes_pattern(&int_min));
    }

    #[test]
    fn a_field_width_keeps_its_two_readings_equal() {
        // A field element's canonical integer *is* its value; there is no second reading, so the
        // struct must not be able to represent one. `opt_mul`'s `inf * 0 = 0` rule and the
        // equality-driven `overwrite` together mean a second representation of the same value
        // would make the fixpoint oscillate.
        let r = ValueRange::from_unsigned(Width::Field(f()), iv(3, 9));
        assert_eq!(r.unsigned(), r.signed());
        let r = ValueRange::from_signed(Width::Field(f()), iv(3, 9));
        assert_eq!(r.unsigned(), r.signed());
        assert_eq!(
            ValueRange::full(Width::Field(f())).unsigned(),
            &Interval::field_top(f())
        );
    }

    #[test]
    fn non_scalar_is_unconstrained_in_both_readings() {
        // The old domain answered `IntInterval::top()` for every non-numeric value, and the `_`
        // transfer arm and the `ArrayToSlice`/`Map` cast arm both have to agree on one
        // representation of "no information" or `changed` never settles.
        let r = ValueRange::full(Width::NonScalar);
        assert_eq!(r.unsigned(), &Interval::top());
        assert_eq!(r.signed(), &Interval::top());
    }

    #[test]
    fn constructors_cap_each_reading_to_its_width() {
        let r = ValueRange::from_unsigned(Width::Bits(8), iv(-5, 1000));
        assert_eq!(r.unsigned(), &iv(0, 255));
        let r = ValueRange::from_signed(Width::Bits(8), iv(-1000, 1000));
        assert_eq!(r.signed(), &iv(-128, 127));
    }

    #[test]
    fn one_known_reading_determines_the_other() {
        // Naming a single reading is still how most transfers build a range, but the reduction now
        // derives the other one instead of leaving it full — which is where the extra precision
        // over the single-interval domain comes from.
        let n = 8;
        let r = ValueRange::from_unsigned(Width::Bits(n), iv(200, 200));
        // 200 reads as -56 in 8-bit two's complement.
        assert_eq!(r.signed(), &iv(-56, -56));
        assert_eq!(gamma(&r, n), vec![200]);

        let r = ValueRange::from_signed(Width::Bits(n), iv(-56, -56));
        assert_eq!(r.unsigned(), &iv(200, 200));
        assert_eq!(gamma(&r, n), vec![200]);

        // A range spanning the sign boundary keeps information in both readings and neither is
        // able to express it alone.
        let r = ValueRange::from_signed(Width::Bits(n), iv(-1, 1));
        assert_eq!(r.unsigned(), &iv(0, 255));
        assert_eq!(r.signed(), &iv(-1, 1));
        assert_eq!(gamma(&r, n), vec![0, 1, 255]);
    }

    #[test]
    fn by_type_selects_the_declared_reading() {
        // The one place the domain still consults type-level signedness: an instruction reading its
        // operand as a mathematical count rather than as a bit pattern.
        let r = ValueRange::from_unsigned(Width::Bits(8), iv(200, 200));
        assert_eq!(r.by_type(&Type::u(8)), &iv(200, 200));
        assert_eq!(r.by_type(&Type::i(8)), &iv(-56, -56));
        assert_eq!(r.by_type(&Type::witness_of(Type::i(8))), &iv(-56, -56));

        let r = ValueRange::from_unsigned(Width::Field(f()), iv(0, 7));
        assert_eq!(r.by_type(&Type::field()), &iv(0, 7));
    }

    #[test]
    fn the_reduction_preserves_gamma_is_idempotent_and_is_canonical() {
        // The three properties the fixed point depends on, brute-forced over every pair of
        // interesting endpoints at eight bits. Canonicity is the load-bearing one: `overwrite`
        // replaces rather than meets and drives `changed` off structural equality, so two spellings
        // of one concrete set would leave the solver oscillating until `ITER_LIMIT` cut it off.
        let n = 8;
        let endpoints = [-129i64, -128, -1, 0, 1, 127, 128, 255];
        let mut canonical: HashMap<Vec<i64>, ValueRange> = HashMap::default();

        for &ul in &endpoints {
            for &uh in &endpoints {
                for &sl in &endpoints {
                    for &sh in &endpoints {
                        let (u, s) = (iv(ul, uh), iv(sl, sh));
                        let expected = gamma_of(n, &u, &s);
                        let r = ValueRange::new(Width::Bits(n), u, s);

                        assert_eq!(gamma(&r, n), expected, "gamma changed for {r:?}");
                        assert_eq!(r.clone().normalized(), r, "not idempotent for {r:?}");
                        // The two components must never disagree about emptiness, or ⊥ has two
                        // spellings of its own.
                        assert_eq!(r.unsigned().is_empty(), r.signed().is_empty());
                        assert_eq!(r.is_empty(), expected.is_empty());

                        match canonical.get(&expected) {
                            Some(seen) => assert_eq!(seen, &r, "two spellings of {expected:?}"),
                            None => {
                                canonical.insert(expected, r);
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn a_wrapped_reading_saturates_rather_than_inverting() {
        // Adopting a wrapped `[lo > hi]` range would make `join` non-unique, so a straddling
        // interval gives up that reading entirely. The other one keeps its information.
        assert_eq!(iv(256, 300).wrap_to_unsigned_bits(8), iv(0, 44));
        assert_eq!(iv(-1, -1).wrap_to_unsigned_bits(8), iv(255, 255));
        assert_eq!(iv(255, 257).wrap_to_unsigned_bits(8), iv(0, 255));
        assert_eq!(iv(0, 1000).wrap_to_unsigned_bits(8), iv(0, 255));
        assert_eq!(Interval::top().wrap_to_unsigned_bits(8), iv(0, 255));
        assert!(Interval::empty().wrap_to_unsigned_bits(8).is_empty());

        assert_eq!(iv(200, 200).wrap_to_signed_bits(8), iv(-56, -56));
        assert_eq!(iv(-1, 1).wrap_to_signed_bits(8), iv(-1, 1));
        assert_eq!(iv(127, 128).wrap_to_signed_bits(8), iv(-128, 127));
    }

    #[test]
    fn a_boolean_has_a_signed_reading_of_minus_one() {
        // In one-bit two's complement the pattern `1` *is* `-1`. No consumer reads the signed
        // component of a boolean, and this pins what it would see if one started to.
        let r = ValueRange::full(Width::Bits(1));
        assert_eq!(r.unsigned(), &iv(0, 1));
        assert_eq!(r.signed(), &iv(-1, 0));
        assert_eq!(gamma(&r, 1), vec![0, 1]);
    }

    #[test]
    fn a_non_scalar_has_exactly_one_representation() {
        // `opt_mul`'s `inf * 0 = 0` rule can hand the reduction a bounded reading for a value that
        // is not a number at all. Forcing top unconditionally is what stops that becoming a second
        // spelling of "no information" that `changed` then flips between forever.
        let r = ValueRange::new(Width::NonScalar, iv(0, 5), iv(0, 5));
        assert_eq!(r, ValueRange::full(Width::NonScalar));
        assert_eq!(
            ValueRange::empty(Width::NonScalar),
            ValueRange::full(Width::NonScalar)
        );
    }

    #[test]
    fn reinterpret_is_a_raw_bit_operation_and_never_empties() {
        // What `Cast` does at runtime: zero-extend or truncate the pattern. Widening a negative
        // signed value is *not* sign extension — `i8 -1` is 255 at any wider width.
        let minus_one = ValueRange::from_signed(Width::Bits(8), iv(-1, -1));
        assert_eq!(
            minus_one.reinterpret_to(Width::Bits(16)).unsigned(),
            &iv(255, 255)
        );
        assert_eq!(
            minus_one.reinterpret_to(Width::Field(f())).unsigned(),
            &iv(255, 255)
        );

        // Truncation wraps, and a source that no longer fits gives up the reading.
        let wide = ValueRange::from_unsigned(Width::Bits(16), iv(300, 300));
        assert_eq!(wide.reinterpret_to(Width::Bits(8)).unsigned(), &iv(44, 44));
        let wide = ValueRange::from_unsigned(Width::Bits(16), iv(250, 300));
        assert_eq!(wide.reinterpret_to(Width::Bits(8)).unsigned(), &iv(0, 255));

        // Reinterpreting at the width it already has must not discard the second reading.
        let both = ValueRange::from_signed(Width::Bits(8), iv(-1, 1));
        assert_eq!(both.reinterpret_to(Width::Bits(8)), both);
    }

    #[test]
    fn constrain_is_the_trapping_counterpart_and_may_empty() {
        let wide = ValueRange::from_unsigned(Width::Bits(16), iv(300, 300));
        assert!(wide.constrain_to(Width::Bits(8)).is_empty());
        let fits = ValueRange::from_unsigned(Width::Bits(16), iv(3, 9));
        assert_eq!(fits.constrain_to(Width::Bits(8)).unsigned(), &iv(3, 9));
    }

    #[test]
    fn constraining_down_keeps_the_patterns_that_still_fit() {
        // The narrowing half of `constraining_to_a_wider_width_does_not_manufacture_bottom`, and
        // the same mistake: `[128, 255]` is representable at `Bits(8)` -- every one of those
        // patterns is a `u8` -- but read as two's complement *there* it is negative, so carrying
        // the old width's signed reading across and intersecting it with `signed_full(8)` used to
        // discard the whole top half of the range. The assertion `constrain_to` models is about
        // representability, which only the unsigned reading decides.
        let wide = ValueRange::from_unsigned(Width::Bits(16), iv(0, 200));
        let narrowed = wide.constrain_to(Width::Bits(8));
        assert_eq!(narrowed.unsigned(), &iv(0, 200));
        // Both readings survive, and at the new width 128..=200 is what makes the signed one
        // straddle zero.
        assert_eq!(narrowed.signed(), &iv(-128, 127));

        // The trap is only sprung when the range reaches the new sign bit, so a range below it
        // must be unaffected either way.
        let low = ValueRange::from_unsigned(Width::Bits(16), iv(0, 127));
        assert_eq!(low.constrain_to(Width::Bits(8)).signed(), &iv(0, 127));
    }

    #[test]
    fn constraining_to_a_wider_width_does_not_manufacture_bottom() {
        // Every `Bits(8)` pattern is representable at `Bits(16)`, so the assertion holds for all of
        // them and nothing may be discarded. The signed reading has to be dropped for that: at
        // `Bits(8)` it reads `[-56, -1]`, which at `Bits(16)` describes patterns above `2^15` and
        // so intersects the unsigned reading to nothing.
        let negative = ValueRange::from_unsigned(Width::Bits(8), iv(200, 255));
        assert_eq!(negative.signed(), &iv(-56, -1), "the trap this rules out");

        let widened = negative.constrain_to(Width::Bits(16));
        assert!(!widened.is_empty());
        assert_eq!(widened.unsigned(), &iv(200, 255));
        // And at the new width those patterns are simply positive.
        assert_eq!(widened.signed(), &iv(200, 255));
    }

    #[test]
    fn field_arithmetic_wraps_instead_of_emptying() {
        // `MulConst` with a multiplier of `[2, 3]` and a value of `p - 1` produces a raw
        // `[2p - 2, 3p - 3]`. Intersecting that with `[0, p)` — which is what capping to the
        // declared type used to do — reports a value that definitely exists as unreachable.
        let p = field_modulus(f());
        let raw = Interval::closed(&p - BigInt::from(1), &p - BigInt::from(1))
            .mul(&Interval::closed(2, 3));
        let capped = raw.intersect(&Interval::field_top(f()));
        assert!(
            capped.is_empty(),
            "the old capping really did produce bottom"
        );

        let wrapped = ValueRangeAnalysis::wrap_or_trap(Width::Field(f()), false, raw.clone(), raw);
        assert!(!wrapped.is_empty());
    }

    #[test]
    fn the_sign_bit_test_reads_the_pattern_at_the_width_it_was_asked_about() {
        // `SExt`'s source may be signed or unsigned, and the answer must be the same either way,
        // since the question is about one bit of the encoding rather than about a reading.
        let unsigned_small = ValueRange::from_unsigned(Width::Bits(8), iv(0, 100));
        assert!(unsigned_small.is_non_negative_at_width(8));

        let unsigned_big = ValueRange::from_unsigned(Width::Bits(8), iv(0, 200));
        assert!(!unsigned_big.is_non_negative_at_width(8));

        let signed_pos = ValueRange::from_signed(Width::Bits(8), iv(0, 100));
        assert!(signed_pos.is_non_negative_at_width(8));

        let signed_neg = ValueRange::from_signed(Width::Bits(8), iv(-100, 100));
        assert!(!signed_neg.is_non_negative_at_width(8));
    }

    #[test]
    fn the_sign_bit_test_declines_at_a_width_narrower_than_the_range() {
        // The signed reading is two's complement at the range's *own* width, so it says nothing
        // about a narrower one. `200` held in a `U(32)` reads as a non-negative `200` there while
        // bit 7 of the pattern `0xC8` is set, so a predicate that consulted it would license
        // `lower_integer_sext` to hardcode `sign = 0` and sign-extend `200` to `200` instead of
        // to `-56`.
        let wide = ValueRange::from_unsigned(Width::Bits(32), iv(200, 200));
        assert!(wide.signed().is_non_negative(), "the trap this rules out");
        assert!(!wide.is_non_negative_at_width(8));
        // It still answers at its own width, and still answers at a narrower one when the
        // unsigned reading actually settles it.
        assert!(wide.is_non_negative_at_width(32));
        assert!(wide.is_non_negative_at_width(9));
        let small = ValueRange::from_unsigned(Width::Bits(32), iv(0, 100));
        assert!(small.is_non_negative_at_width(8));
    }

    #[test]
    fn a_field_element_is_always_non_negative_as_signed() {
        assert!(ValueRange::full(Width::Field(f())).is_non_negative_as_signed());
        assert!(!ValueRange::full(Width::NonScalar).is_non_negative_as_signed());
    }

    #[test]
    fn signed_constants_decode_to_their_mathematical_value() {
        // Guards `compute_constant_bounds`: `Constant::I` stores the raw two's-complement bits, so
        // the signed reading has to decode them rather than take them at face value.
        assert_eq!(signed_const_to_bigint(8, 0xFF), BigInt::from(-1));
        assert_eq!(signed_const_to_bigint(8, 0x80), BigInt::from(-128));
        assert_eq!(signed_const_to_bigint(8, 0x7F), BigInt::from(127));
        assert_eq!(signed_const_to_bigint(1, 1), BigInt::from(-1));
        // 128 bits must not overflow the u128 shift used to build 2^bits.
        assert_eq!(signed_const_to_bigint(128, u128::MAX), BigInt::from(-1));
    }

    #[test]
    fn join_and_intersect_act_on_both_readings() {
        let a = ValueRange::from_unsigned(Width::Bits(8), iv(0, 10));
        let b = ValueRange::from_unsigned(Width::Bits(8), iv(20, 30));
        assert_eq!(a.join(&b).unsigned(), &iv(0, 30));
        assert!(a.intersect(&b).is_empty());
    }

    #[test]
    fn width_of_type_ignores_signedness_but_not_the_bit_count() {
        // The whole point of the rewrite: `u32` and `i32` are the same bit pattern, so they must
        // produce the same width. When `TypeExpr::U` disappears this becomes the only rule left.
        assert_eq!(
            Width::of_type(&Type::u(32), f()),
            Width::of_type(&Type::i(32), f())
        );
        assert_eq!(Width::of_type(&Type::u(32), f()), Width::Bits(32));
        assert_eq!(Width::of_type(&Type::field(), f()), Width::Field(f()));
        assert_eq!(
            Width::of_type(&Type::u(8).array_of(3), f()),
            Width::NonScalar
        );
    }

    #[test]
    fn set_containment_and_proof_containment_differ_exactly_at_bottom() {
        let bottom = Interval::empty();

        // The set predicates stay mathematically honest: the empty set is contained in everything,
        // and `[1, 0]` also passes the raw lo/hi tests by accident. Both readings say "true".
        assert!(bottom.fits_in_unsigned_bits(8));
        assert!(bottom.fits_in_signed_bits(8));
        assert!(bottom.is_non_negative_in_signed(8));

        // The proof predicates — the ones a constraint-eliding site is allowed to consult — do not.
        assert!(!bottom.proves_fits_in_unsigned_bits(8));
        assert!(!bottom.proves_fits_in_signed_bits(8));
        assert!(!bottom.proves_non_negative_in_signed(8));

        // On anything else the two agree, so this is the only behavioral difference.
        for range in [
            Interval::closed(0, 7),
            Interval::closed(-4, 4),
            Interval::closed(0, 255),
            Interval::closed(200, 300),
            Interval::top(),
        ] {
            assert_eq!(
                range.fits_in_unsigned_bits(8),
                range.proves_fits_in_unsigned_bits(8)
            );
            assert_eq!(
                range.fits_in_signed_bits(8),
                range.proves_fits_in_signed_bits(8)
            );
            assert_eq!(
                range.is_non_negative_in_signed(8),
                range.proves_non_negative_in_signed(8)
            );
        }
    }

    #[test]
    fn bottom_value_range_never_proves_a_clear_sign_bit() {
        // `SExt` replaces its computed sign bit with a constant zero on the strength of this, so a
        // ⊥ operand must not be able to license it.
        let bottom = ValueRange::from_unsigned(Width::Bits(8), Interval::empty());
        assert!(bottom.is_empty());
        assert!(!bottom.is_non_negative_at_width(8));
        assert!(!bottom.is_non_negative_as_signed());

        // A ⊥ *field* element likewise proves nothing, even though every reachable field element
        // is non-negative.
        let bottom_field = ValueRange::from_unsigned(Width::Field(f()), Interval::empty());
        assert!(!bottom_field.is_non_negative_as_signed());
    }

    /// `binary_arith` with both operands the result's width, which is every case that occurs
    /// outside a shift by a narrower amount.
    fn arith(
        kind: BinaryArithOpKind,
        width: Width,
        signed: bool,
        l: &ValueRange,
        r: &ValueRange,
        rhs_type: &Type,
    ) -> ValueRange {
        ValueRangeAnalysis::binary_arith(kind, width, signed, l, r, true, true, rhs_type)
    }

    fn u8r(lo: i64, hi: i64) -> ValueRange {
        ValueRange::from_unsigned(Width::Bits(8), iv(lo, hi))
    }

    fn i8r(lo: i64, hi: i64) -> ValueRange {
        ValueRange::from_signed(Width::Bits(8), iv(lo, hi))
    }

    #[test]
    fn integer_arithmetic_traps_in_one_reading_and_wraps_in_the_other() {
        use BinaryArithOpKind::*;
        let w = Width::Bits(8);

        // Unsigned: the trap prunes the overflowing tail of the sum, and the signed reading
        // follows from what is left.
        let sum = arith(Add, w, false, &u8r(200, 255), &u8r(1, 1), &Type::u(8));
        assert_eq!(sum.unsigned(), &iv(201, 255));
        assert_eq!(sum.signed(), &iv(-55, -1));

        // Signed: the trap is on the signed reading instead, and the unsigned one wraps.
        let sum = arith(Add, w, true, &i8r(-2, -1), &i8r(1, 1), &Type::i(8));
        assert_eq!(sum.signed(), &iv(-1, 0));
        assert_eq!(sum.unsigned(), &iv(0, 255));

        // The legal `u8` add whose *signed* reading leaves the type. Capping both readings here —
        // rather than only the one the type traps on — would report it unreachable.
        let sum = arith(Add, w, false, &u8r(127, 127), &u8r(1, 1), &Type::u(8));
        assert!(!sum.is_empty());
        assert_eq!(sum.unsigned(), &iv(128, 128));
        assert_eq!(sum.signed(), &iv(-128, -128));

        // Underflow of an unsigned subtraction is the ⊥ the proof predicates exist for.
        let diff = arith(Sub, w, false, &u8r(0, 0), &u8r(1, 1), &Type::u(8));
        assert!(diff.is_empty());

        let product = arith(Mul, w, false, &u8r(2, 3), &u8r(4, 5), &Type::u(8));
        assert_eq!(product.unsigned(), &iv(8, 15));

        // An unreachable operand makes the result unreachable rather than a plausible zero.
        let bottom = ValueRange::empty(w);
        assert!(arith(Add, w, false, &bottom, &u8r(1, 1), &Type::u(8)).is_empty());
        assert!(arith(And, w, false, &bottom, &u8r(1, 1), &Type::u(8)).is_empty());
    }

    #[test]
    fn bitwise_operations_no_longer_need_non_negative_operands() {
        use BinaryArithOpKind::*;
        let w = Width::Bits(8);

        // Both operands negative as signed, which the old rule refused to bound at all. The raw
        // patterns are 200..=255 and 254..=255, so the AND is capped by the smaller.
        let and = arith(And, w, true, &i8r(-56, -1), &i8r(-2, -1), &Type::i(8));
        assert_eq!(and.unsigned(), &iv(0, 255));

        let and = arith(And, w, true, &i8r(-1, -1), &i8r(0, 3), &Type::i(8));
        assert_eq!(and.unsigned(), &iv(0, 3));

        let or = arith(Or, w, false, &u8r(0, 5), &u8r(0, 9), &Type::u(8));
        assert_eq!(or.unsigned(), &iv(0, 15));
        let xor = arith(Xor, w, false, &u8r(0, 5), &u8r(0, 9), &Type::u(8));
        assert_eq!(xor.unsigned(), &iv(0, 15));
    }

    #[test]
    fn division_and_remainder_pick_the_reading_they_are_performed_in() {
        use BinaryArithOpKind::*;
        let w = Width::Bits(8);

        let quot = arith(Div, w, false, &u8r(200, 255), &u8r(4, 4), &Type::u(8));
        assert_eq!(quot.unsigned(), &iv(50, 63));

        // A negative signed dividend still gives up: signed division truncates toward zero while
        // interval division floors, and the two only agree above zero.
        let quot = arith(Div, w, true, &i8r(-8, -1), &i8r(2, 2), &Type::i(8));
        assert_eq!(quot.signed(), &Interval::signed_full(8));

        // `x % d < d`, and `x % d <= x`: the second term is what the old rule was missing.
        let rem = arith(Mod, w, false, &u8r(0, 3), &u8r(10, 10), &Type::u(8));
        assert_eq!(rem.unsigned(), &iv(0, 3));
        let rem = arith(Mod, w, false, &u8r(0, 200), &u8r(10, 10), &Type::u(8));
        assert_eq!(rem.unsigned(), &iv(0, 9));

        // Field division is multiplication by a modular inverse, which no interval bounds. The old
        // rule reached for `div_const_pos` here and got an answer that is simply wrong.
        let fw = Width::Field(f());
        let quot = arith(
            Div,
            fw,
            false,
            &ValueRange::from_unsigned(fw, iv(0, 100)),
            &ValueRange::from_unsigned(fw, iv(2, 2)),
            &Type::field(),
        );
        assert_eq!(quot, ValueRange::full(fw));
    }

    #[test]
    fn shifts_by_a_constant_amount_are_exact() {
        use BinaryArithOpKind::*;
        let w = Width::Bits(8);

        let shl = arith(Shl, w, false, &u8r(1, 3), &u8r(2, 2), &Type::u(8));
        assert_eq!(shl.unsigned(), &iv(4, 12));
        // An overflowing left shift is modelled as wrapping, which is sound whichever way the
        // backends and the constant folder settle their disagreement about it.
        let shl = arith(Shl, w, false, &u8r(200, 200), &u8r(1, 1), &Type::u(8));
        assert!(!shl.is_empty());

        // Logical on an unsigned type, arithmetic on a signed one — both floor.
        let shr = arith(Shr, w, false, &u8r(200, 255), &u8r(1, 1), &Type::u(8));
        assert_eq!(shr.unsigned(), &iv(100, 127));
        let shr = arith(Shr, w, true, &i8r(-8, -5), &i8r(1, 1), &Type::i(8));
        assert_eq!(shr.signed(), &iv(-4, -3));

        // An out-of-range amount is a runtime error the backends disagree about, so it is not
        // modelled at all.
        let shr = arith(Shr, w, false, &u8r(200, 255), &u8r(9, 9), &Type::u(8));
        assert_eq!(shr, ValueRange::full(w));
    }

    #[test]
    fn a_right_shift_by_an_unknown_amount_is_still_monotone() {
        use BinaryArithOpKind::*;
        let w = Width::Bits(8);

        // Shifting right can only move a value toward zero...
        let shr = arith(Shr, w, false, &u8r(0, 60), &u8r(0, 7), &Type::u(8));
        assert_eq!(shr.unsigned(), &iv(0, 60));

        // ...or toward -1, when the shift is arithmetic and the value is negative.
        let shr = arith(Shr, w, true, &i8r(-40, -10), &i8r(0, 7), &Type::i(8));
        assert_eq!(shr.signed(), &iv(-40, -1));
        let shr = arith(Shr, w, true, &i8r(-40, 10), &i8r(0, 7), &Type::i(8));
        assert_eq!(shr.signed(), &iv(-40, 10));

        // A left shift by an unknown amount has no bound at all.
        let shl = arith(Shl, w, false, &u8r(0, 1), &u8r(0, 7), &Type::u(8));
        assert_eq!(shl, ValueRange::full(w));
    }

    #[test]
    fn an_operand_of_a_different_width_is_not_a_reading_of_the_result() {
        use BinaryArithOpKind::*;
        let w = Width::Bits(8);
        let narrow = ValueRange::from_unsigned(Width::Bits(4), iv(1, 1));

        assert_eq!(
            ValueRangeAnalysis::binary_arith(
                Add,
                w,
                false,
                &u8r(1, 1),
                &narrow,
                true,
                false,
                &Type::u(4)
            ),
            ValueRange::full(w)
        );
        // A shift is the exception: its amount is a count, not a reading of the result, so a
        // narrower one is fine.
        let shl = ValueRangeAnalysis::binary_arith(
            Shl,
            w,
            false,
            &u8r(1, 3),
            &narrow,
            true,
            false,
            &Type::u(4),
        );
        assert_eq!(shl.unsigned(), &iv(2, 6));
    }

    /// Run the whole analysis over a hand-built function and return the entry's ranges.
    fn run_analysis(ssa: &mut HLSSA) -> FunctionValueRanges {
        let flow = FlowAnalysis::run(ssa);
        let types = crate::compiler::analysis::types::Types::new().run(ssa, &flow);
        let ranges = ValueRangeAnalysis::new().run(ssa, &flow, &types);
        let entry = ssa.get_unique_entrypoint_id();
        FunctionValueRanges {
            values: ranges.get_function(entry).values.clone(),
        }
    }

    #[test]
    fn a_select_across_widths_keeps_the_narrow_branchs_values() {
        // `Types` unifies a `Select`'s alternatives with `get_arithmetic_result_type`, so a `u8`
        // and a `u16` branch give a `u16` result and the `u8` operand's range has to be re-read at
        // the wider width. Carrying its *signed* reading across is what made that go wrong: `200`
        // at `u8` reads as `-56`, which at `u16` describes a pattern above `2^15`, so
        // `constrain_to` reduced the whole branch to ⊥ — and `join` treats ⊥ as the identity, so
        // the merged range silently became the other branch's alone. That is an unsound narrowing,
        // not merely an imprecise one: `200` really can come out of this select.
        let mut ssa = HLSSA::with_main("main".to_string());
        let main_id = ssa.get_unique_entrypoint_id();
        let selected;
        {
            let mut sb = HLSSABuilder::new(&mut ssa);
            selected = sb.modify_function(main_id, |b| {
                b.function.add_return_type(Type::u(16));
                let entry = b.function.get_entry_id();
                let mut e = b.test_block(entry);
                // Opaque rather than `u_const(1, 1)`: the `Select` transfer ignores its condition
                // today, so a constant would still join both branches -- but it would stop doing
                // so the moment anyone teaches it to fold a known condition, and the test would
                // keep passing while testing nothing.
                let cond = e.add_parameter(Type::u(1));
                let narrow = e.u_const(8, 200);
                let wide = e.u_const(16, 1000);
                let r = e.select(cond, narrow, wide);
                e.terminate_return(vec![r]);
                r
            });
        }

        let ranges = run_analysis(&mut ssa);
        let r = ranges.get(selected);
        assert_eq!(r.width(), Width::Bits(16));
        assert_eq!(r.unsigned(), &iv(200, 1000));
    }

    #[test]
    fn a_block_parameter_joins_both_readings_across_its_predecessors() {
        // The block-param join and the `overwrite`/`ITER_LIMIT` sweep are the parts of the solver
        // that canonicity protects, and nothing else here reaches them. Merging 3 with 200 leaves
        // information in *both* readings — neither interval alone denotes `{3, 200}` — which is the
        // reduced product doing the only thing a single interval could not.
        let mut ssa = HLSSA::with_main("main".to_string());
        let main_id = ssa.get_unique_entrypoint_id();
        let merged;
        {
            let mut sb = HLSSABuilder::new(&mut ssa);
            merged = sb.modify_function(main_id, |b| {
                b.function.add_return_type(Type::u(8));

                let merge = b.add_block(|_| {});
                let then_block = b.add_block(|_| {});
                let else_block = b.add_block(|_| {});

                let param = {
                    let mut e = b.test_block(merge);
                    let p = e.add_parameter(Type::u(8));
                    e.terminate_return(vec![p]);
                    p
                };

                let entry = b.function.get_entry_id();
                {
                    let mut e = b.test_block(entry);
                    let cond = e.u_const(1, 1);
                    e.terminate_jmp_if(cond, then_block, else_block);
                }
                {
                    let mut e = b.test_block(then_block);
                    let small = e.u_const(8, 3);
                    e.terminate_jmp(merge, vec![small]);
                }
                {
                    let mut e = b.test_block(else_block);
                    let large = e.u_const(8, 200);
                    e.terminate_jmp(merge, vec![large]);
                }
                param
            });
        }

        let ranges = run_analysis(&mut ssa);
        let r = ranges.get(merged);
        assert_eq!(r.width(), Width::Bits(8));
        assert_eq!(r.unsigned(), &iv(3, 200));
        // 200 reads as -56, so the signed component pins the top of the range where the unsigned
        // one cannot.
        assert_eq!(r.signed(), &iv(-56, 3));
        assert_eq!(gamma(&r, 8), vec![3, 200]);
    }

    /// Build `guard(cond) { result = lhs <kind> rhs }` over two `u8` constants and return the
    /// analysed range of `result`.
    fn guarded_u8_op(kind: BinaryArithOpKind, lhs: u128, rhs: u128) -> ValueRange {
        let mut ssa = HLSSA::with_main("main".to_string());
        let main_id = ssa.get_unique_entrypoint_id();
        let guarded;
        {
            let mut sb = HLSSABuilder::new(&mut ssa);
            guarded = sb.modify_function(main_id, |b| {
                b.function.add_return_type(Type::u(8));
                let entry = b.function.get_entry_id();
                let mut e = b.test_block(entry);
                let x = e.add_parameter(Type::field());
                // A witness condition, which is what leaves a `Guard` standing.
                let w = e.write_witness(x);
                let cond = e.eq(w, x);
                let a = e.u_const(8, lhs);
                let b_ = e.u_const(8, rhs);
                let result = e.fresh_value();
                e.emit(OpCode::Guard {
                    condition: cond,
                    inner: Box::new(OpCode::BinaryArithOp {
                        kind,
                        result,
                        lhs: a,
                        rhs: b_,
                    }),
                });
                e.terminate_return(vec![result]);
                result
            });
        }
        run_analysis(&mut ssa).get(guarded)
    }

    #[test]
    fn a_guarded_failable_operation_can_still_produce_zero() {
        use BinaryArithOpKind::*;

        // `0 - 1` on a `u8` underflows, so the range of the operation *itself* is ⊥. An inactive
        // guard around it does not underflow — it produces zero — and a transfer that recursed
        // into the inner operation alone would report the whole thing unreachable.
        let r = guarded_u8_op(Sub, 0, 1);
        assert!(!r.is_empty());
        assert_eq!(r.unsigned(), &iv(0, 0));

        // Division by zero and an out-of-range shift also produce zero when inactive, but the
        // transfer has no bound on either operation to begin with, so the join is invisible.
        assert_eq!(guarded_u8_op(Div, 7, 0).unsigned(), &iv(0, 255));
        assert_eq!(guarded_u8_op(Shl, 3, 9).unsigned(), &iv(0, 255));

        // An operation that does not fail keeps its own value, joined with the zero the failure
        // branch would have produced.
        assert_eq!(guarded_u8_op(Add, 3, 4).unsigned(), &iv(0, 7));

        // Without the fix, `known_sign` could read a range that excludes zero and hardcode a sign
        // bit the inactive branch contradicts.
        assert!(!guarded_u8_op(Sub, 3, 4).unsigned().is_empty());
    }
}
