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
//! # Flow Sensitivity
//!
//! One range per value, valid wherever it is live, with a single exception. A conditional branch
//! decides a comparison, and inside the region it uniquely enters that comparison is a fact about
//! its operands. [`FunctionValueRanges::get_at`] serves the narrowed reading and
//! [`FunctionValueRanges::get`] the flow-insensitive one; see [`BranchFact`] for why a consumer may
//! elide a check on the strength of the former.
//!
//! # Width::Field
//!
//! A field element's canonical integer _is_ its value; there is no second reading. So
//! [`Width::Field`] keeps `signed == unsigned`, both ⊆ `[0, p−1]`. Keeping the struct total this
//! way avoids an `Option<Interval>` that every call site would have to unwrap.

use mavros_artifacts::FieldConfig;
use num_bigint::{BigInt, Sign};
use num_traits::{One, Signed, ToPrimitive, Zero};
use tracing::{Level, instrument, warn};

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
                ArithGroup, BinaryArithOpKind, CastTarget, CmpKind, Constant, HLFunction, HLSSA,
                OpCode, Type, TypeExpr,
            },
        },
    },
};

// VALUE RANGE ANALYSIS
// ================================================================================================

pub struct ValueRangeAnalysis;

/// How many times a single value's range may be refined before the solver gives up on the descent
/// terminating and widens it instead.
///
/// It's generous because most programs terminate in a measured 3-5 rounds. The loop-heavy ones need
/// some real depth, so the budget is sized to allow those to converge too (e.g. `passport_08`
/// converges at 86 rounds).
const WIDEN_AFTER: usize = 96;

/// Backstop on the rounds, not the termination argument.
///
/// [`Interval::widen`] is what guarantees the solver stops: every endpoint is given up on at most
/// once, so the iteration is bounded by [`WIDEN_AFTER`] plus a couple of rounds to settle. This
/// limit exists so that a bug in that argument costs a wide answer rather than a hang, and so
/// reaching it is reported.
const ITER_LIMIT: usize = 256;

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
        // How many times each value has been refined, which is what `overwrite` widens against.
        let mut refinements: HashMap<ValueId, usize> = HashMap::default();

        // Initial state: every value's bound is its declared type's full range, and iteration only
        // narrows from there.
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
        // Structural, so it is computed once and reused by every round of the fixed point.
        let branch_facts = collect_branch_facts(function, cfg);

        let mut converged = false;
        for _iter in 0..ITER_LIMIT {
            let mut changed = false;

            for &block_id in &order {
                let block = function.get_block(block_id);

                if block_id != entry_block_id {
                    // Only `Jmp` carries block arguments, so a `JmpIf` predecessor contributes
                    // nothing — which is right precisely because a block it can reach has no
                    // parameters to contribute to. Were that invariant ever broken, the join below
                    // would run over a strict subset of the predecessors and _narrow_ rather than
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
                        Self::overwrite(
                            &mut bounds,
                            &mut refinements,
                            *param_id,
                            new_range,
                            &mut changed,
                        );
                    }
                }

                let facts = branch_facts
                    .get(&block_id)
                    .map(Vec::as_slice)
                    .unwrap_or_default();
                for instr in block.get_instructions() {
                    self.transfer(
                        instr,
                        types,
                        &mut bounds,
                        &mut refinements,
                        &mut changed,
                        field,
                        facts,
                    );
                }
            }

            if !changed {
                converged = true;
                break;
            }
        }

        // Not a failure, but not nothing either: this function's ranges are wider than the domain
        // could have made them, so a consumer that discharges checks off them will discharge fewer.
        // Silent until now, which is why nobody knew how often it happens. See [`ITER_LIMIT`].
        if !converged {
            warn!(
                function = function.get_name(),
                rounds = ITER_LIMIT,
                "value-range analysis stopped before reaching a fixed point"
            );
        }

        FunctionValueRanges {
            values: bounds,
            facts: branch_facts,
        }
    }

    /// Store `new` as `v`'s range, widening once `v` has been refined [`WIDEN_AFTER`] times.
    ///
    /// The widening is per _value_ rather than per round. A value that settles quickly is never
    /// widened however long some other value in the same function keeps the solver going.
    fn overwrite(
        bounds: &mut HashMap<ValueId, ValueRange>,
        refinements: &mut HashMap<ValueId, usize>,
        v: ValueId,
        new: ValueRange,
        changed: &mut bool,
    ) {
        let Some(old) = bounds.get(&v).cloned() else {
            bounds.insert(v, new);
            *changed = true;
            return;
        };
        if old == new {
            return;
        }

        let seen = refinements.entry(v).or_insert(0);
        *seen += 1;
        let next = if *seen > WIDEN_AFTER {
            old.widen(&new)
        } else {
            new
        };

        if old != next {
            bounds.insert(v, next);
            *changed = true;
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn transfer(
        &self,
        instr: &OpCode,
        types: &FunctionTypeInfo,
        bounds: &mut HashMap<ValueId, ValueRange>,
        refinements: &mut HashMap<ValueId, usize>,
        changed: &mut bool,
        field: FieldConfig,
        facts: &[BranchFact],
    ) {
        // The flow-insensitive bound: what is true of a value everywhere it is live.
        let flat = |bounds: &HashMap<ValueId, ValueRange>, v: ValueId| -> ValueRange {
            match bounds.get(&v) {
                Some(r) => r.clone(),
                None => ValueRange::for_type(types.get_value_type(v), field),
            }
        };

        // Both readings of an operand's bit pattern.
        //
        // Reading an operand _here_ additionally narrows it by whatever the branches dominating
        // this block have already decided about it. Only reads are narrowed; what the rules below
        // store is the range of a value **defined in this block**, and SSA dominance puts every use
        // of such a value inside the same region, so the narrower bound holds at all of them.
        let range = |bounds: &HashMap<ValueId, ValueRange>, v: ValueId| -> ValueRange {
            narrow(v, flat(bounds, v), facts, |other| flat(bounds, other))
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
                    CastTarget::Int(n) => in_r.reinterpret_to(Width::Bits(*n)),
                    // ValueOf strips the WitnessOf wrapper: payload unchanged.
                    CastTarget::Nop | CastTarget::WitnessOf | CastTarget::ValueOf => in_r,
                    // Sequence-level casts carry no scalar range.
                    CastTarget::ArrayToSlice | CastTarget::Map(_) => {
                        ValueRange::full(Width::NonScalar)
                    }
                };
                Self::set(bounds, refinements, *result, types, field, r, changed);
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
                    refinements,
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
                // `(v >> offset) & mask(width)`, with the result keeping the _source's_ type. The
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
                    refinements,
                    *result,
                    types,
                    field,
                    ValueRange::from_unsigned(out_width, masked),
                    changed,
                );
            }

            // The witness takes the range of its _hint_, which is a claim about witness generation
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
                Self::set(bounds, refinements, *r, types, field, in_r, changed);
            }
            OpCode::WriteWitness { result: None, .. } => {}

            OpCode::FreshWitness {
                result,
                result_type,
            } => {
                Self::overwrite(
                    bounds,
                    refinements,
                    *result,
                    ValueRange::for_type(result_type, field),
                    changed,
                );
            }

            OpCode::Cmp { result, .. } => {
                // Both Eq and Lt yield a u1 boolean regardless of operand types.
                Self::overwrite(
                    bounds,
                    refinements,
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
                Self::set(bounds, refinements, *result, types, field, r, changed);
            }

            OpCode::BinaryArithOp {
                kind,
                result,
                lhs,
                rhs,
            } => {
                let result_ty = types.get_value_type(*result);
                let width = Width::of_type(result_ty, field);
                let l = range(bounds, *lhs);
                let r_in = range(bounds, *rhs);
                let r = Self::binary_arith(
                    *kind,
                    width,
                    &l,
                    &r_in,
                    width_of(*lhs) == width,
                    width_of(*rhs) == width,
                );
                Self::set(bounds, refinements, *result, types, field, r, changed);
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

                // `MulConst` is a _field_ multiply of raw patterns, so the multiplier is read as
                // the pattern it is. There is no opcode here to carry a sign and nothing to read
                // one from.
                //
                // That is only the whole story while the result really is a field element, which is
                // what makes the `false` below a fact rather than a default: `types.rs` gives
                // `MulConst` its `var`'s type, and `witness_lowering` — the only thing that builds
                // one — builds it from `WitnessOf(Field)` operands. Were an integer ever to reach
                // here, `wrap_or_trap` would intersect the reading the operation does *not* trap
                // on, so the invariant is asserted rather than left to the comment.
                debug_assert!(
                    matches!(width, Width::Field(_)),
                    "MulConst is a field multiply; a {width:?} result would need a sign to read \
                     its operands with"
                );
                let factor = c.unsigned();
                let r = if c.is_empty() || v.is_empty() || width_of(*var) != width {
                    Self::unknown_or_empty(width, c.is_empty() || v.is_empty())
                } else {
                    Self::wrap_or_trap(
                        width,
                        false,
                        factor.mul(v.unsigned()),
                        factor.mul(v.signed()),
                    )
                };
                Self::set(bounds, refinements, *result, types, field, r, changed);
            }

            OpCode::Select {
                result, if_t, if_f, ..
            } => {
                let width = width_of(*result);
                let t = range(bounds, *if_t).constrain_to(width);
                let f = range(bounds, *if_f).constrain_to(width);
                Self::set(
                    bounds,
                    refinements,
                    *result,
                    types,
                    field,
                    t.join(&f),
                    changed,
                );
            }

            OpCode::Guard { inner, .. } => {
                // This arm writes each result **twice**: once for the inner operation, and once
                // again below with zero joined in. The inner call reports into a scratch flag and
                // `changed` is decided once, by comparing what the results hold now against what
                // they held before the guard ran.
                let before: Vec<(ValueId, Option<ValueRange>, usize)> = inner
                    .get_results()
                    .map(|vid| {
                        (
                            *vid,
                            bounds.get(vid).cloned(),
                            refinements.get(vid).copied().unwrap_or(0),
                        )
                    })
                    .collect();
                let mut transient = false;
                self.transfer(
                    inner,
                    types,
                    bounds,
                    refinements,
                    &mut transient,
                    field,
                    facts,
                );

                // A guarded _failable_ operation is not simply its inner operation.
                // `LowerPureGuards` branches on the failure condition: the failing side asserts the
                // guard's condition is false and yields the result type's zero, so an inactive
                // guard around an operation that would have overflowed, divided by zero or shifted
                // out of range produces `0` — a value the computed range can easily exclude.
                if guard_may_produce_zero(inner, types) {
                    for vid in inner.get_results() {
                        let width = Width::of_type(types.get_value_type(*vid), field);
                        let zero = ValueRange::from_unsigned(width, Interval::singleton(0));
                        let joined = match bounds.get(vid) {
                            Some(computed) => computed.join(&zero),
                            None => zero,
                        };
                        Self::overwrite(bounds, refinements, *vid, joined, &mut transient);
                    }
                }

                for (vid, prior, prior_refinements) in before {
                    let moved = bounds.get(&vid) != prior.as_ref();
                    *changed |= moved;
                    refinements.insert(vid, prior_refinements + usize::from(moved));
                }
            }

            // Other opcodes: keep the type-based default bound.
            //
            // While `Rangecheck(v, k)` proves `v.unsigned ⊆ [0, 2^k)` we do not use this fact. Its
            // use has been measured to provide no movement across the corpus, making it not worth
            // the added complexity. Unlike the interprocedural range analysis, this may be worth
            // revisiting in the future if we find consumers that could benefit.
            _ => {
                for vid in instr.get_results() {
                    let r = ValueRange::for_type(types.get_value_type(*vid), field);
                    Self::overwrite(bounds, refinements, *vid, r, changed);
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
    ///
    /// `kind` carries the sign as well as the operation, and both are read off it. It used to
    /// arrive alongside a separate `signed` flag, which meant a caller could spell an `SDiv` with
    /// `signed: false` — an operation the pipeline cannot produce — and have it silently mean
    /// something.
    #[allow(clippy::too_many_arguments)]
    fn binary_arith(
        kind: BinaryArithOpKind,
        width: Width,
        l: &ValueRange,
        r: &ValueRange,
        lhs_matches: bool,
        rhs_matches: bool,
    ) -> ValueRange {
        use ArithGroup::*;
        let group = kind.group();
        let signed = kind.is_signed();

        // An unreachable operand makes the result unreachable. Propagating it matters because the
        // bitwise rules below read `hi` directly, and ⊥ spells `[1, 0]`, whose `hi` is a perfectly
        // plausible-looking zero.
        if l.is_empty() || r.is_empty() {
            return ValueRange::empty(width);
        }

        match group {
            Add | Sub | Mul => {
                if !(lhs_matches && rhs_matches) {
                    return ValueRange::full(width);
                }

                // The same formula in both readings, because the operation is a ring homomorphism
                // modulo the width: `dec` commutes with `+`, `−` and `×`.
                let (raw_u, raw_s) = match group {
                    Add => (l.unsigned().add(r.unsigned()), l.signed().add(r.signed())),
                    Sub => (l.unsigned().sub(r.unsigned()), l.signed().sub(r.signed())),
                    _ => (l.unsigned().mul(r.unsigned()), l.signed().mul(r.signed())),
                };
                Self::wrap_or_trap(width, signed, raw_u, raw_s)
            }

            Div | Rem => {
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

                let math = match (group, divisor.lo(), divisor.hi()) {
                    (Div, Some(lo), Some(hi)) if lo == hi && lo.is_positive() => {
                        dividend.div_const_pos(lo)
                    }
                    // `x % d < d`, and a non-negative `x % d` is also no larger than `x` itself.
                    (Rem, Some(lo), Some(hi)) if lo.is_positive() => {
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
                let cap = match group {
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

                // The amount is read as a magnitude, which makes the arms below sound regardless of
                // the originally-declared sign of the operand.
                //
                // This is not because a shift amount cannot be negative — it can. Noir types it as
                // the shifted value's own type (`bind_type_variables_for_infix` unifies both
                // operands of every infix operator, shifts included), so `i32 s<< i32` is
                // well-typed and `noir_tests/signed_shift` contains one. But a negative amount is a
                // program failure, rejected by `pure_guards::emit_invalid_shift_cond`, and reading
                // it here as the large magnitude its raw bits spell is exactly what keeps it out of
                // the constant arm: `2^n - 1` never satisfies `k < n`, so it falls to the
                // conservative unknown-amount arms below rather than folding to a plausible small
                // shift.
                let amount = r.unsigned();
                let constant = match (amount.lo(), amount.hi()) {
                    (Some(a), Some(b)) if a == b && !a.is_negative() => {
                        a.to_usize().filter(|k| *k < n)
                    }
                    _ => None,
                };
                match (group, constant) {
                    // A left shift is a multiply by `2^k`, modelled as _wrapping_ rather than
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
    /// Both were computed by the same formula, but only one of them is _exact_: the declared type
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
    #[allow(clippy::too_many_arguments)]
    fn set(
        bounds: &mut HashMap<ValueId, ValueRange>,
        refinements: &mut HashMap<ValueId, usize>,
        result: ValueId,
        types: &FunctionTypeInfo,
        field: FieldConfig,
        range: ValueRange,
        changed: &mut bool,
    ) {
        let width = Width::of_type(types.get_value_type(result), field);
        Self::overwrite(
            bounds,
            refinements,
            result,
            range.constrain_to(width),
            changed,
        );
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

// BRANCH REFINEMENT
// ================================================================================================

/// A comparison that a conditional branch has already decided, on the edge that decided it.
///
/// The analysis is otherwise flow-insensitive. It is also what lets a loop counter be bounded at
/// all. A loop header's parameter joins its own back-edge argument, so without the guard it is ⊤
/// from the first round and stays there; with it, the body sees `i < n`, the increment is bounded
/// by `n`, and the join closes on `[0, n]`. That is the difference between proving and not proving
/// that `n - 1 - i` cannot underflow, which is one R1CS row per iteration of every loop in the
/// standard library that counts down.
///
/// A consumer that discharges a runtime check on the strength of one of these is asserting that the
/// check cannot fail *on any execution that reaches the instruction*. Both regimes this analysis
/// runs in support that, for different reasons:
///
/// - **After `UntaintControlFlow`**, every surviving `JmpIf` is on a *pure* condition: witness ones
///   have been linearized into `Select`s with their blocks' instructions wrapped in `Guard`. A pure
///   branch is real control flow in both backends, and `hlssa_to_r1cs` interprets it rather than
///   flattening it, so the untaken arm is never evaluated at all.
/// - **Before it**, a witness `JmpIf` is still real control flow in the IR, so the fact is a true
///   statement about executions that take the edge. Executions that do not take it never evaluate
///   the instruction, and Noir owes no rejection for an operation inside an `if` that did not run.
///   Linearization preserves that: it wraps the arm's instructions in `Guard`, and an inactive
///   guard around a failable operation yields zero rather than rejecting.
#[derive(Debug, Clone)]
struct BranchFact {
    kind: CmpKind,
    lhs: ValueId,
    rhs: ValueId,
    /// Whether this is the edge on which the comparison held.
    holds: bool,
}

impl BranchFact {
    /// The operand facing `value` across the comparison, or `None` if this fact is not about
    /// `value` at all.
    ///
    /// A comparison of a value with _itself_ says nothing usable: if it decided either way it did
    /// so without constraining anything.
    fn opposite(&self, value: ValueId) -> Option<ValueId> {
        if self.lhs == self.rhs {
            return None;
        }
        if value == self.lhs {
            Some(self.rhs)
        } else if value == self.rhs {
            Some(self.lhs)
        } else {
            None
        }
    }

    /// The bound this fact puts on `value`, given `other` (the range of the operand facing it).
    ///
    /// `None` where the fact implies no bound, either because the comparison is one this does not
    /// model or because `other` is unbounded on the side that would have supplied the endpoint.
    fn constraint(&self, value: ValueId, width: Width, other: &ValueRange) -> Option<ValueRange> {
        let is_lhs = value == self.lhs;
        match (self.kind, self.holds) {
            // An equality that held pins both sides to the same set of patterns. A width mismatch
            // would make that intersection meaningless, and `ValueRange::intersect` rejects one.
            (CmpKind::Eq, true) => (other.width() == width).then(|| other.clone()),
            // Inequality leaves a hole rather than an interval, which this domain cannot express.
            (CmpKind::Eq, false) => None,
            // The unsigned reading is the raw pattern read as a non-negative integer, which is the
            // same number whatever width the pattern is held at, so a `<` on it needs no width
            // agreement to mean something.
            (CmpKind::ULt, holds) => Some(ValueRange::from_unsigned(
                width,
                Self::half_line(other.unsigned(), is_lhs, holds)?,
            )),
            // The signed reading is not: it is the two's-complement value at _this range's own_
            // width, so an endpoint taken from a differently-wide operand bounds a different number
            // than the one being narrowed. `Eq` above declines a width mismatch for the same reason
            // and this arm must too -- more so, because narrowing only ever tightens, and these
            // bounds are what `overflow_provably_impossible` and friends discharge runtime
            // rejections on. HLSSA does not enforce equal widths on a `Cmp` (`analysis::types`
            // gives the result `int(1)` without looking at the operands), so this is a real gate
            // rather than a restatement of an invariant.
            (CmpKind::SLt, holds) if other.width() == width => Some(ValueRange::from_signed(
                width,
                Self::half_line(other.signed(), is_lhs, holds)?,
            )),
            (CmpKind::SLt, _) => None,
        }
    }

    /// The half-line a decided `<` confines one of its operands to, on whichever reading the
    /// comparison uses, while `other` is the range of the operand on the far side.
    fn half_line(other: &Interval, is_lhs: bool, holds: bool) -> Option<Interval> {
        Some(match (is_lhs, holds) {
            // `value < other`, so `value` is below the largest `other` can be.
            (true, true) => Interval::at_most(other.hi()?.clone() - BigInt::one()),
            // `value >= other`, so `value` is at least the smallest `other` can be.
            (true, false) => Interval::at_least(other.lo()?.clone()),
            // `other < value`.
            (false, true) => Interval::at_least(other.lo()?.clone() + BigInt::one()),
            // `other >= value`.
            (false, false) => Interval::at_most(other.hi()?.clone()),
        })
    }
}

/// Narrow `base`, the flow-insensitive range of `value`, by every fact in force.
///
/// `flat` reads the *unnarrowed* range of the operand on the far side of each comparison, so that
/// narrowing `a` in `a < b` can never consult a bound on `b` that was itself derived from `a`.
fn narrow(
    value: ValueId,
    base: ValueRange,
    facts: &[BranchFact],
    flat: impl Fn(ValueId) -> ValueRange,
) -> ValueRange {
    let mut narrowed = base;
    for fact in facts {
        let Some(other) = fact.opposite(value) else {
            continue;
        };
        if let Some(c) = fact.constraint(value, narrowed.width(), &flat(other)) {
            narrowed = narrowed.intersect(&c);
        }
    }
    narrowed
}

/// The branch facts in force in each block of `function`.
///
/// A fact is attached to a branch target only when that target's **sole** predecessor is the
/// branching block. Arriving at a block with two predecessors does not say which edge was taken,
/// and the fact would be false on the other one. From the target it propagates down the dominator
/// tree, because every path to a dominated block runs through the target and so through the edge
/// that established it.
fn collect_branch_facts(function: &HLFunction, cfg: &CFG) -> HashMap<BlockId, Vec<BranchFact>> {
    let mut comparisons: HashMap<ValueId, (CmpKind, ValueId, ValueId)> = HashMap::default();
    let mut branches: HashMap<BlockId, (ValueId, BlockId, BlockId)> = HashMap::default();
    let mut children: HashMap<BlockId, Vec<BlockId>> = HashMap::default();

    for (block_id, block) in function.get_blocks() {
        for instr in block.get_instructions() {
            if let OpCode::Cmp {
                kind,
                result,
                lhs,
                rhs,
            } = instr
            {
                comparisons.insert(*result, (*kind, *lhs, *rhs));
            }
        }
        // A branch whose two targets are the same block decides nothing.
        if let Some(Terminator::JmpIf(cond, t, f)) = block.get_terminator()
            && t != f
        {
            branches.insert(*block_id, (*cond, *t, *f));
        }
        if let Some(idom) = cfg.get_immediate_dominator(*block_id) {
            children.entry(idom).or_default().push(*block_id);
        }
    }

    // The walk below must not depend on the map's iteration order.
    for kids in children.values_mut() {
        kids.sort();
    }

    let mut facts: HashMap<BlockId, Vec<BranchFact>> = HashMap::default();
    let mut stack = vec![(function.get_entry_id(), Vec::<BranchFact>::new())];
    while let Some((block_id, active)) = stack.pop() {
        let branch = branches
            .get(&block_id)
            .and_then(|(cond, t, f)| comparisons.get(cond).map(|cmp| (*cmp, *t, *f)));

        for child in children.get(&block_id).into_iter().flatten().copied() {
            let mut inherited = active.clone();
            if let Some(((kind, lhs, rhs), t, f)) = branch {
                let holds = if child == t {
                    Some(true)
                } else if child == f {
                    Some(false)
                } else {
                    None
                };

                // `child == t` already implies `t` is dominated by this block, but not that it is
                // entered only from here.
                if let Some(holds) = holds
                    && cfg.get_predecessors(child).count() == 1
                {
                    inherited.push(BranchFact {
                        kind,
                        lhs,
                        rhs,
                        holds,
                    });
                }
            }
            stack.push((child, inherited));
        }
        facts.insert(block_id, active);
    }
    facts
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

    /// The widening step: keep every endpoint `refined` left alone, and **drop** every one it moved
    /// inward.
    ///
    /// This is what ensures that the solver terminates. The iteration descends from ⊤ and each round is
    /// free to shave a bound by as little as one. A chain like `[-31, 3] ⊐ [-30, 3] ⊐ [-29, 3] ⊐ …`
    /// is a fixed point that exists but is `2^63` rounds away. Widening replaces that descent with
    /// a single jump: an endpoint that has kept moving is given up on and released to infinity,
    /// which [`ValueRange::new`]'s reduction then clamps back to the operand width.
    ///
    /// Sound because it only ever _loosens_: the result contains `refined`, and `refined` is what
    /// a sound transfer computed, so the result over-approximates whatever `refined` did.
    ///
    /// This terminates because a dropped endpoint cannot be dropped twice: it is already at the
    /// width's extreme, so the next round widens it to the same place, `changed` stays false, and
    /// the value is stable. Each endpoint is therefore given up on at most once.
    ///
    /// ⊥ is returned untouched. It is already stable as nothing refines an empty interval, so it
    /// needs no help terminating, and widening it would throw away a proof of unreachability for
    /// nothing.
    #[must_use]
    pub fn widen(&self, refined: &Self) -> Self {
        if self.is_empty() || refined.is_empty() {
            return refined.clone();
        }

        // An endpoint already at ±∞ **stays** there. That is the half that makes this terminate:
        // releasing an endpoint has to be final, or the next round's refinement pulls it back in,
        // the round after releases it again, and the operator that was supposed to stop the
        // oscillation has become the oscillation.
        let lo = match (&self.lo, &refined.lo) {
            (None, _) | (_, None) => None,
            (Some(was), Some(now)) if now > was => None,
            (Some(_), Some(now)) => Some(now.clone()),
        };
        let hi = match (&self.hi, &refined.hi) {
            (None, _) | (_, None) => None,
            (Some(was), Some(now)) if now < was => None,
            (Some(_), Some(now)) => Some(now.clone()),
        };
        Self { lo, hi }
    }

    /// `(−∞, hi]`. Cannot be inverted, so it needs no normalization.
    pub fn at_most(hi: BigInt) -> Self {
        Self {
            lo: None,
            hi: Some(hi),
        }
    }

    /// `[lo, +∞)`. Cannot be inverted, so it needs no normalization.
    pub fn at_least(lo: BigInt) -> Self {
        Self {
            lo: Some(lo),
            hi: None,
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

/// Modular reduction into a fixed window — the _wrapping_ half of a change of representation.
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
            TypeExpr::Int(n) => Width::Bits(*n),
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

            // A field element's canonical integer _is_ its value, so the two readings are one
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
    /// (every arm rebuilds from the unsigned one) since it was the sign at the _old_ width and says
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
    /// the same thing at any width; the signed reading is two's complement at the _old_ width and
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

    /// The two's-complement signed reading — the _mathematical_ value of a signed integer.
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
    /// width the pattern is held at. The signed reading is the two's-complement value at _this
    /// range's own_ width and says nothing about a narrower one — a `U(32)` pinned to `200` has a
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
    /// γ admits a pattern only when _both_ readings admit it, so excluding it from either one is
    /// enough. Both are consulted rather than just the unsigned reading because the reduction
    /// leaves each component a _hull_, and γ may have a hole in the middle of that hull: at `n = 8`
    /// with `u = [0, 255]` and `s = [−1, 1]`, γ is `{255, 0, 1}`, and it is the signed reading that
    /// rules out the pattern `5`.
    ///
    /// ⊥ answers `false`, like every other `proves_*` query. An unreachable value cannot be `raw`,
    /// but the analysis derives ⊥ _from_ the constraints the caller is about to drop, so answering
    /// `true` would be circular. See the proof-strength predicates on [`Interval`].
    pub fn proves_excludes_pattern(&self, raw: &BigInt) -> bool {
        if self.is_empty() {
            return false;
        }
        match self.width {
            // No bit pattern to speak of, and both readings are top by invariant.
            Width::NonScalar => false,
            // One reading, and the pattern _is_ the canonical integer.
            Width::Field(_) => !self.unsigned.contains(raw),
            Width::Bits(n) => {
                !self.unsigned.contains(raw) || !self.signed.contains(&decode_signed(n, raw))
            }
        }
    }

    /// Whether the raw pattern is provably a valid shift amount for a `bound`-bit operand, i.e.
    /// provably in `[0, bound)`.
    ///
    /// Asked of the pattern rather than of a chosen reading, because that allows the answer to
    /// serve both shift kinds. `pure_guards::emit_invalid_shift_cond` zero-extends the amount to
    /// `max(bound, 64)` bits before comparing, so a pattern below `bound` is non-negative and in
    /// range under the signed reading too, and a signed shift needs no separate query.
    ///
    /// ⊥ answers `false`, like every other `proves_*` query: the caller is about to drop the very
    /// check the analysis would be reasoning from.
    pub fn proves_shift_amount_below(&self, bound: usize) -> bool {
        if self.is_empty() || bound == 0 {
            return false;
        }
        // Only the upper end is asked about. The lower one carries nothing: the unsigned reading is
        // a raw bit pattern, so it is non-negative at every `Width` by construction, and a
        // negative amount shows up here as the *large* magnitude its bits spell — which is exactly
        // what the upper test rejects.
        match self.unsigned.hi() {
            Some(hi) => hi < &BigInt::from(bound),
            // An unbounded end proves nothing.
            None => false,
        }
    }

    /// The single raw bit pattern this range admits, or `None` if it admits more than one.
    ///
    /// Asked of the unsigned reading for the same reason as [`Self::proves_shift_amount_below`]:
    /// that reading _is_ the bit pattern so a caller who gets an answer can mint the literal
    /// straight from it rather than having to undo a width-dependent interpretation first.
    ///
    /// ⊥ answers `None`. An unreachable value is not a constant as it has no value at all, and a
    /// caller reaching for this is about to _replace_ a computation with the literal, which on ⊥
    /// would mean replacing it with an arbitrary one.
    pub fn proves_constant(&self) -> Option<&BigInt> {
        if self.is_empty() {
            return None;
        }
        match (self.unsigned.lo(), self.unsigned.hi()) {
            (Some(lo), Some(hi)) if lo == hi => Some(lo),
            _ => None,
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

    /// [`Interval::widen`] on both readings at once, then reduced.
    ///
    /// Both are widened because either one can be the endpoint that keeps moving: a value can have
    /// a settled unsigned hull and a signed one still creeping, which is exactly what
    /// `signed_for_range` does. The reduction afterwards is what ensures that a dropped endpoint
    /// turns back into the width's own extreme, so the result is still a range _of the correct
    /// width_ rather than an unbounded one.
    #[must_use]
    pub fn widen(&self, refined: &Self) -> Self {
        debug_assert_eq!(
            self.width, refined.width,
            "widening ValueRanges of different widths"
        );
        Self::new(
            self.width,
            self.unsigned.widen(&refined.unsigned),
            self.signed.widen(&refined.signed),
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
/// _can_ be scalars — they are left out because the transfer does not model them either way: they
/// fall to the `_` arm, which answers with the full range of the declared type, and that already
/// contains the zero the failure branch would produce.
fn guard_may_produce_zero(inner: &OpCode, types: &FunctionTypeInfo) -> bool {
    use ArithGroup::*;
    match inner {
        OpCode::BinaryArithOp { kind, lhs, .. } => match kind.group() {
            Add | Sub | Mul => !matches!(
                types.get_value_type(*lhs).strip_witness().expr,
                TypeExpr::Field
            ),
            Div | Rem | Shl | Shr => true,
            And | Or | Xor => false,
        },
        _ => false,
    }
}

// FUNCTION VALUE RANGES
// ================================================================================================

pub struct FunctionValueRanges {
    values: HashMap<ValueId, ValueRange>,
    facts: HashMap<BlockId, Vec<BranchFact>>,
}

impl FunctionValueRanges {
    /// Get the range for a value, returning an unconstrained non-scalar range if the value isn't
    /// in our map (e.g. fresh values created downstream of this analysis).
    ///
    /// This is the **flow-insensitive** answer: true wherever the value is live. A consumer
    /// deciding something about one particular instruction wants [`Self::get_at`] instead, which
    /// is never wider and is frequently much narrower for a loop counter.
    pub fn get(&self, v: ValueId) -> ValueRange {
        self.values
            .get(&v)
            .cloned()
            .unwrap_or_else(|| ValueRange::full(Width::NonScalar))
    }

    /// The range of `v` **as seen from `block`**: [`Self::get`] narrowed by facts the conditional
    /// branches dominating `block` have already decided about `v`.
    ///
    /// Sound for a decision about an instruction that stays in `block`. It is _not_ sound to carry
    /// the answer to another block, so a pass that hoists or sinks the instruction it asked about
    /// must re-ask at the destination.
    pub fn get_at(&self, block: BlockId, v: ValueId) -> ValueRange {
        let Some(facts) = self.facts.get(&block) else {
            return self.get(v);
        };
        narrow(v, self.get(v), facts, |other| self.get(other))
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

/// Convert a `Constant::Int(bits, encoded)` u128 bit pattern back to a signed `BigInt`.
///
/// Two's-complement decode for any `bits ∈ [1, 128]`.
///
/// Test-only. `compute_constant_bounds` used to call this on the signed half of the constant tag;
/// with one `Constant::Int` it names a constant by its raw pattern instead and lets the domain's
/// reduction recover the signed reading. What survives here is the _statement_ of that equivalence
/// — `the_two_routes_to_one_pattern_agree` is what stops the reduction quietly drifting from the
/// decode it is supposed to reproduce.
#[cfg(test)]
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
                // One known bit pattern. Which reading it is _given_ through does not matter: the
                // domain carries both, and the reduction is canonical, so naming the pattern as an
                // unsigned singleton recovers exactly the signed reading `from_signed` would have
                // been handed. `the_two_routes_to_one_pattern_agree` pins that.
                Constant::Int(bits, v) => {
                    ValueRange::from_unsigned(Width::Bits(*bits), Interval::singleton(*v))
                }
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

    /// The concrete set a `Bits(n)` range denotes: every bit pattern admitted by _both_ readings.
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
        // `proves_excludes_pattern` is not an approximation: γ admits a pattern iff _both_ readings
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
        // One reading, and no sign bit: the pattern _is_ the value, so `p − 1` is `p − 1` and not
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

        // A tight _signed_ reading rules out `−1` while the unsigned hull still contains `255`.
        let non_negative = ValueRange::from_signed(Width::Bits(8), iv(0, 127));
        assert!(non_negative.proves_excludes_pattern(&minus_one));
        assert!(non_negative.proves_excludes_pattern(&int_min));
        assert!(!non_negative.proves_excludes_pattern(&BigInt::zero()));

        // A tight _unsigned_ reading rules out `INT_MIN` the other way round.
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
        // A field element's canonical integer _is_ its value; there is no second reading, so the
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
    fn both_readings_of_one_pattern_stay_available() {
        // The domain keeps both readings of every value and never picks between them by looking at
        // a type -- there is no longer a type-level sign to look at. Consumers that need one ask
        // for it by name, and which name they ask for follows their opcode.
        let r = ValueRange::from_unsigned(Width::Bits(8), iv(200, 200));
        assert_eq!(r.unsigned(), &iv(200, 200));
        assert_eq!(r.signed(), &iv(-56, -56));

        // A field element has one reading, so the two agree.
        let r = ValueRange::from_unsigned(Width::Field(f()), iv(0, 7));
        assert_eq!(r.unsigned(), &iv(0, 7));
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
        // In one-bit two's complement the pattern `1` _is_ `-1`. No consumer reads the signed
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
        // signed value is _not_ sign extension — `i8 -1` is 255 at any wider width.
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
        // patterns is a `u8` -- but read as two's complement _there_ it is negative, so carrying
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
        // The signed reading is two's complement at the range's _own_ width, so it says nothing
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
    fn the_two_routes_to_one_pattern_agree() {
        // What lets `compute_constant_bounds` name a constant by its raw pattern alone. A
        // `Constant::Int` says nothing about how to read itself, so the range built from it is the
        // unsigned singleton -- and that has to be _the same domain element_ the signed route
        // would have produced, or collapsing the two constant tags would have quietly widened or
        // narrowed every signed constant's range.
        //
        // It holds because the domain carries both readings and its reduction is canonical: each
        // route pins one reading and lets the reduction recover the other, and both denote the
        // same single bit pattern. Checked at the boundaries, where the two readings differ most.
        for (bits, raw) in [
            (8usize, 0xFFu128),
            (8, 0x80),
            (8, 0x7F),
            (8, 0),
            (1, 1),
            (32, 0xDEAD),
        ] {
            let from_pattern =
                ValueRange::from_unsigned(Width::Bits(bits), Interval::singleton(raw));
            let from_signed_reading = ValueRange::from_signed(
                Width::Bits(bits),
                Interval::singleton(signed_const_to_bigint(bits, raw)),
            );
            assert_eq!(
                from_pattern, from_signed_reading,
                "int{bits} pattern {raw:#x} disagrees between the two routes"
            );
        }
    }

    #[test]
    fn signed_constants_decode_to_their_mathematical_value() {
        // The decode `the_two_routes_to_one_pattern_agree` is stated against: raw two's-complement
        // bits have to be decoded rather than taken at face value.
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
            Width::of_type(&Type::int(32), f()),
            Width::of_type(&Type::int(32), f())
        );
        assert_eq!(Width::of_type(&Type::int(32), f()), Width::Bits(32));
        assert_eq!(Width::of_type(&Type::field(), f()), Width::Field(f()));
        assert_eq!(
            Width::of_type(&Type::int(8).array_of(3), f()),
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
    fn a_shift_amount_is_proved_in_range_from_the_raw_pattern() {
        // In range under both readings, so the check `LowerPureGuards` would emit is redundant.
        let small = ValueRange::from_unsigned(Width::Bits(8), Interval::closed(0, 7));
        assert!(small.proves_shift_amount_below(8));
        assert!(small.proves_shift_amount_below(32));

        // At the bound, not below it. A shift *by* the width is the failure being checked for.
        let at_bound = ValueRange::from_unsigned(Width::Bits(8), Interval::closed(8, 8));
        assert!(!at_bound.proves_shift_amount_below(8));
        assert!(at_bound.proves_shift_amount_below(9));

        // A width's worth of unknown proves nothing, which is the common case at this pass: the
        // amount often only becomes a literal after inlining.
        assert!(!ValueRange::full(Width::Bits(8)).proves_shift_amount_below(8));
        assert!(!ValueRange::full(Width::Bits(32)).proves_shift_amount_below(32));

        // A *negative* amount is a failure too, and it is caught without asking the signed
        // reading: `-1` at eight bits is the pattern `255`, which is not below any width.
        let negative = ValueRange::from_signed(Width::Bits(8), Interval::closed(-1, -1));
        assert!(!negative.proves_shift_amount_below(8));
        let maybe_negative = ValueRange::from_signed(Width::Bits(8), Interval::closed(-1, 3));
        assert!(!maybe_negative.proves_shift_amount_below(8));
        // Non-negative and small under the signed reading is in range under both.
        let signed_small = ValueRange::from_signed(Width::Bits(8), Interval::closed(0, 3));
        assert!(signed_small.proves_shift_amount_below(8));

        // Nothing is ever proved of a non-scalar, whose readings are top by invariant.
        assert!(!ValueRange::full(Width::NonScalar).proves_shift_amount_below(8));

        // ⊥ answers `false`, like every other `proves_*` query: the caller is about to drop the
        // check the analysis would have been reasoning from.
        let bottom = ValueRange::from_unsigned(Width::Bits(8), Interval::empty());
        assert!(bottom.is_empty());
        assert!(!bottom.proves_shift_amount_below(8));

        // A zero bound admits no amount at all, so it discharges nothing however tight the range
        // is. The narrowest operand that does have a legal shift is one bit, by zero.
        assert!(!small.proves_shift_amount_below(0));
        let zero = ValueRange::from_unsigned(Width::Bits(1), Interval::closed(0, 0));
        assert!(zero.proves_shift_amount_below(1));
    }

    #[test]
    fn a_range_reports_a_constant_only_when_it_admits_exactly_one_pattern() {
        let pinned = ValueRange::from_unsigned(Width::Bits(8), Interval::closed(3, 3));
        assert_eq!(pinned.proves_constant(), Some(&BigInt::from(3)));

        // Two patterns is not a constant, however narrow the range is.
        let pair = ValueRange::from_unsigned(Width::Bits(8), Interval::closed(3, 4));
        assert_eq!(pair.proves_constant(), None);
        assert_eq!(ValueRange::full(Width::Bits(8)).proves_constant(), None);

        // The answer is the raw pattern, not the signed reading of it: a caller mints the literal
        // from this, and the literal is bits.
        let negative = ValueRange::from_signed(Width::Bits(8), Interval::closed(-1, -1));
        assert_eq!(negative.proves_constant(), Some(&BigInt::from(255)));

        // An unbounded end is not a constant even when the other one is pinned.
        let half_open =
            ValueRange::from_unsigned(Width::NonScalar, Interval::at_least(BigInt::from(7)));
        assert_eq!(half_open.proves_constant(), None);

        // ⊥ is not a constant: it has no value at all, and the caller is about to substitute one.
        let bottom = ValueRange::from_unsigned(Width::Bits(8), Interval::empty());
        assert_eq!(bottom.proves_constant(), None);
    }

    #[test]
    fn bottom_value_range_never_proves_a_clear_sign_bit() {
        // `SExt` replaces its computed sign bit with a constant zero on the strength of this, so a
        // ⊥ operand must not be able to license it.
        let bottom = ValueRange::from_unsigned(Width::Bits(8), Interval::empty());
        assert!(bottom.is_empty());
        assert!(!bottom.is_non_negative_at_width(8));
        assert!(!bottom.is_non_negative_as_signed());

        // A ⊥ _field_ element likewise proves nothing, even though every reachable field element
        // is non-negative.
        let bottom_field = ValueRange::from_unsigned(Width::Field(f()), Interval::empty());
        assert!(!bottom_field.is_non_negative_as_signed());
    }

    /// `binary_arith` with both operands the result's width, which is every case that occurs
    /// outside a shift by a narrower amount.
    fn arith(kind: BinaryArithOpKind, width: Width, l: &ValueRange, r: &ValueRange) -> ValueRange {
        ValueRangeAnalysis::binary_arith(kind, width, l, r, true, true)
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
        let sum = arith(UAdd, w, &u8r(200, 255), &u8r(1, 1));
        assert_eq!(sum.unsigned(), &iv(201, 255));
        assert_eq!(sum.signed(), &iv(-55, -1));

        // Signed: the trap is on the signed reading instead, and the unsigned one wraps.
        let sum = arith(SAdd, w, &i8r(-2, -1), &i8r(1, 1));
        assert_eq!(sum.signed(), &iv(-1, 0));
        assert_eq!(sum.unsigned(), &iv(0, 255));

        // The legal `u8` add whose _signed_ reading leaves the type. Capping both readings here —
        // rather than only the one the type traps on — would report it unreachable.
        let sum = arith(UAdd, w, &u8r(127, 127), &u8r(1, 1));
        assert!(!sum.is_empty());
        assert_eq!(sum.unsigned(), &iv(128, 128));
        assert_eq!(sum.signed(), &iv(-128, -128));

        // Underflow of an unsigned subtraction is the ⊥ the proof predicates exist for.
        let diff = arith(USub, w, &u8r(0, 0), &u8r(1, 1));
        assert!(diff.is_empty());

        let product = arith(UMul, w, &u8r(2, 3), &u8r(4, 5));
        assert_eq!(product.unsigned(), &iv(8, 15));

        // An unreachable operand makes the result unreachable rather than a plausible zero.
        let bottom = ValueRange::empty(w);
        assert!(arith(UAdd, w, &bottom, &u8r(1, 1)).is_empty());
        assert!(arith(And, w, &bottom, &u8r(1, 1)).is_empty());
    }

    #[test]
    fn bitwise_operations_no_longer_need_non_negative_operands() {
        use BinaryArithOpKind::*;
        let w = Width::Bits(8);

        // Both operands negative as signed, which the old rule refused to bound at all. The raw
        // patterns are 200..=255 and 254..=255, so the AND is capped by the smaller.
        let and = arith(And, w, &i8r(-56, -1), &i8r(-2, -1));
        assert_eq!(and.unsigned(), &iv(0, 255));

        let and = arith(And, w, &i8r(-1, -1), &i8r(0, 3));
        assert_eq!(and.unsigned(), &iv(0, 3));

        let or = arith(Or, w, &u8r(0, 5), &u8r(0, 9));
        assert_eq!(or.unsigned(), &iv(0, 15));
        let xor = arith(Xor, w, &u8r(0, 5), &u8r(0, 9));
        assert_eq!(xor.unsigned(), &iv(0, 15));
    }

    #[test]
    fn division_and_remainder_pick_the_reading_they_are_performed_in() {
        use BinaryArithOpKind::*;
        let w = Width::Bits(8);

        let quot = arith(UDiv, w, &u8r(200, 255), &u8r(4, 4));
        assert_eq!(quot.unsigned(), &iv(50, 63));

        // A negative signed dividend still gives up: signed division truncates toward zero while
        // interval division floors, and the two only agree above zero.
        let quot = arith(SDiv, w, &i8r(-8, -1), &i8r(2, 2));
        assert_eq!(quot.signed(), &Interval::signed_full(8));

        // `x % d < d`, and `x % d <= x`: the second term is what the old rule was missing.
        let rem = arith(URem, w, &u8r(0, 3), &u8r(10, 10));
        assert_eq!(rem.unsigned(), &iv(0, 3));
        let rem = arith(URem, w, &u8r(0, 200), &u8r(10, 10));
        assert_eq!(rem.unsigned(), &iv(0, 9));

        // Field division is multiplication by a modular inverse, which no interval bounds. The old
        // rule reached for `div_const_pos` here and got an answer that is simply wrong.
        let fw = Width::Field(f());
        let quot = arith(
            UDiv,
            fw,
            &ValueRange::from_unsigned(fw, iv(0, 100)),
            &ValueRange::from_unsigned(fw, iv(2, 2)),
        );
        assert_eq!(quot, ValueRange::full(fw));
    }

    #[test]
    fn shifts_by_a_constant_amount_are_exact() {
        use BinaryArithOpKind::*;
        let w = Width::Bits(8);

        let shl = arith(UShl, w, &u8r(1, 3), &u8r(2, 2));
        assert_eq!(shl.unsigned(), &iv(4, 12));
        // An overflowing left shift is modelled as wrapping, which is sound whichever way the
        // backends and the constant folder settle their disagreement about it.
        let shl = arith(UShl, w, &u8r(200, 200), &u8r(1, 1));
        assert!(!shl.is_empty());

        // Logical on an unsigned type, arithmetic on a signed one — both floor.
        let shr = arith(UShr, w, &u8r(200, 255), &u8r(1, 1));
        assert_eq!(shr.unsigned(), &iv(100, 127));
        let shr = arith(SShr, w, &i8r(-8, -5), &i8r(1, 1));
        assert_eq!(shr.signed(), &iv(-4, -3));

        // An out-of-range amount is a runtime error the backends disagree about, so it is not
        // modelled at all.
        let shr = arith(UShr, w, &u8r(200, 255), &u8r(9, 9));
        assert_eq!(shr, ValueRange::full(w));
    }

    #[test]
    fn a_right_shift_by_an_unknown_amount_is_still_monotone() {
        use BinaryArithOpKind::*;
        let w = Width::Bits(8);

        // Shifting right can only move a value toward zero...
        let shr = arith(UShr, w, &u8r(0, 60), &u8r(0, 7));
        assert_eq!(shr.unsigned(), &iv(0, 60));

        // ...or toward -1, when the shift is arithmetic and the value is negative.
        let shr = arith(SShr, w, &i8r(-40, -10), &i8r(0, 7));
        assert_eq!(shr.signed(), &iv(-40, -1));
        let shr = arith(SShr, w, &i8r(-40, 10), &i8r(0, 7));
        assert_eq!(shr.signed(), &iv(-40, 10));

        // A left shift by an unknown amount has no bound at all.
        let shl = arith(UShl, w, &u8r(0, 1), &u8r(0, 7));
        assert_eq!(shl, ValueRange::full(w));
    }

    #[test]
    fn an_operand_of_a_different_width_is_not_a_reading_of_the_result() {
        use BinaryArithOpKind::*;
        let w = Width::Bits(8);
        let narrow = ValueRange::from_unsigned(Width::Bits(4), iv(1, 1));

        assert_eq!(
            ValueRangeAnalysis::binary_arith(UAdd, w, &u8r(1, 1), &narrow, true, false),
            ValueRange::full(w)
        );
        // A shift is the exception: its amount is a count, not a reading of the result, so a
        // narrower one is fine.
        let shl = ValueRangeAnalysis::binary_arith(UShl, w, &u8r(1, 3), &narrow, true, false);
        assert_eq!(shl.unsigned(), &iv(2, 6));
    }

    /// Run the whole analysis over a hand-built function and return the entry's ranges.
    fn run_analysis(ssa: &mut HLSSA) -> FunctionValueRanges {
        let flow = FlowAnalysis::run(ssa);
        let types = crate::compiler::analysis::types::Types::new().run(ssa, &flow);
        let ranges = ValueRangeAnalysis::new().run(ssa, &flow, &types);
        let entry = ssa.get_unique_entrypoint_id();
        let function = ranges.get_function(entry);
        FunctionValueRanges {
            values: function.values.clone(),
            facts: function.facts.clone(),
        }
    }

    /// The canonical counted loop, as `for i in 0..16 { .. 15 - i .. }` lowers to it:
    ///
    /// ```text
    /// entry:                          jmp header(0)
    /// header(i):  c = i < 16;         jmp_if c to body, else to exit
    /// body:       d = 15 - i; e = i + 1;  jmp header(e)
    /// exit:                           return i
    /// ```
    ///
    /// Returns `(header, body, i, d, e)`.
    fn counted_loop(ssa: &mut HLSSA) -> (BlockId, BlockId, ValueId, ValueId, ValueId) {
        let main_id = ssa.get_unique_entrypoint_id();
        let mut sb = HLSSABuilder::new(ssa);
        sb.modify_function(main_id, |b| {
            b.function.add_return_type(Type::int(32));
            let header = b.add_block(|_| {});
            let body = b.add_block(|_| {});
            let exit = b.add_block(|_| {});

            let entry = b.function.get_entry_id();
            {
                let mut e = b.test_block(entry);
                let zero = e.int_const(32, 0);
                e.terminate_jmp(header, vec![zero]);
            }
            let counter = {
                let mut e = b.test_block(header);
                let i = e.add_parameter(Type::int(32));
                let limit = e.int_const(32, 16);
                let c = e.ult(i, limit);
                e.terminate_jmp_if(c, body, exit);
                i
            };
            let (down, next) = {
                let mut e = b.test_block(body);
                let top = e.int_const(32, 15);
                let down = e.usub(top, counter);
                let one = e.int_const(32, 1);
                let next = e.uadd(counter, one);
                e.terminate_jmp(header, vec![next]);
                (down, next)
            };
            {
                let mut e = b.test_block(exit);
                e.terminate_return(vec![counter]);
            }
            (header, body, counter, down, next)
        })
    }

    #[test]
    fn a_loop_counter_is_bounded_by_its_own_guard() {
        // Without the guard this is ⊤ from the first round and stays there: the header parameter
        // joins its own back-edge argument, which is derived from the parameter. The guard breaks
        // that circle — the body sees `i < 16`, so the increment is bounded by 16, so the join
        // closes on `[0, 16]` instead of the full width.
        let mut ssa = HLSSA::with_main("main".to_string());
        let (_, _, counter, _, _) = counted_loop(&mut ssa);
        let ranges = run_analysis(&mut ssa);
        assert_eq!(ranges.get(counter).unsigned(), &iv(0, 16));
    }

    #[test]
    fn the_counter_is_one_tighter_inside_the_body_than_at_the_header() {
        // 16 is a value the counter really takes — on the round that ends the loop — so the
        // flow-insensitive answer must keep it. The body is reached only when the guard held, so
        // there, and only there, it is at most 15. Getting this backwards in either direction is
        // the whole risk of the feature: `get` too tight is unsound, `get_at` too loose costs the
        // rows this was written to recover.
        let mut ssa = HLSSA::with_main("main".to_string());
        let (header, body, counter, _, _) = counted_loop(&mut ssa);
        let ranges = run_analysis(&mut ssa);

        assert_eq!(ranges.get_at(header, counter).unsigned(), &iv(0, 16));
        assert_eq!(ranges.get_at(body, counter).unsigned(), &iv(0, 15));
    }

    #[test]
    fn a_value_computed_under_the_guard_keeps_the_narrower_bound_everywhere() {
        // `15 - i` is what the overflow check is about, and it is computed in the body. Its range
        // is stored flow-insensitively because SSA dominance puts every _use_ of it inside the
        // same region, so the bound that held where it was computed holds wherever it is read.
        // `[0, 15]` rather than `[-1, 15]` is exactly the difference between discharging that
        // check and emitting it.
        let mut ssa = HLSSA::with_main("main".to_string());
        let (_, _, _, down, next) = counted_loop(&mut ssa);
        let ranges = run_analysis(&mut ssa);

        assert_eq!(ranges.get(down).unsigned(), &iv(0, 15));
        assert_eq!(ranges.get(next).unsigned(), &iv(1, 16));
    }

    #[test]
    fn widening_keeps_a_settled_endpoint_and_drops_a_moving_one() {
        // The whole operator in one case: `hi` did not move, so it survives; `lo` was shaved by
        // one, so it is given up on and released to −∞.
        let was = Interval::closed(-30, 3);
        let now = Interval::closed(-29, 3);
        let widened = was.widen(&now);
        assert_eq!(widened.lo(), None, "a moving endpoint is released");
        assert_eq!(
            widened.hi(),
            Some(&BigInt::from(3)),
            "a settled one is kept"
        );
    }

    #[test]
    fn widening_only_ever_loosens() {
        // The soundness property, and the only one that matters: whatever the transfer computed is
        // still admitted afterwards. A widening that excluded a value the refinement allowed would
        // hand a consumer a bound too tight and delete a rejection.
        let cases = [
            (Interval::closed(-30, 3), Interval::closed(-29, 3)),
            (Interval::closed(0, 100), Interval::closed(10, 90)),
            (Interval::closed(0, 100), Interval::closed(0, 100)),
            (Interval::at_least(BigInt::from(5)), Interval::closed(7, 9)),
            (Interval::at_most(BigInt::from(5)), Interval::closed(1, 4)),
        ];
        for (was, now) in cases {
            let widened = was.widen(&now);
            assert_eq!(
                widened.intersect(&now),
                now,
                "widening {was:?} toward {now:?} lost part of the refinement"
            );
        }
    }

    #[test]
    fn widening_settles_after_one_step() {
        // Termination: a released endpoint is already at ±∞, so the next round releases it to the
        // same place and the solver sees no change. Each endpoint is given up on at most once,
        // which is what bounds the iteration.
        let was = Interval::closed(-30, 3);
        let once = was.widen(&Interval::closed(-29, 3));
        let twice = once.widen(&Interval::closed(-28, 3));
        assert_eq!(once, twice, "a released endpoint must not move again");
    }

    #[test]
    fn widening_leaves_bottom_alone() {
        // ⊥ is already stable, and it is a proof of unreachability worth keeping.
        let empty = Interval::empty();
        assert!(Interval::closed(0, 10).widen(&empty).is_empty());
        assert!(empty.widen(&empty).is_empty());
    }

    #[test]
    fn a_widened_range_is_clamped_back_to_its_width() {
        // `Interval::widen` releases to ±∞, but a `ValueRange` is a range *of a width*, so the
        // reduction has to bring it back. Otherwise the solver would start handing out bounds
        // outside the operand's own domain.
        let was = ValueRange::from_unsigned(Width::Bits(8), Interval::closed(3, 200));
        let now = ValueRange::from_unsigned(Width::Bits(8), Interval::closed(4, 199));
        let widened = was.widen(&now);
        assert_eq!(
            widened.unsigned(),
            &iv(0, 255),
            "both endpoints moved, so both are released — to the width's own extremes"
        );
        assert_eq!(widened.width(), Width::Bits(8));
    }

    #[test]
    fn the_solver_terminates_on_a_bound_that_creeps() {
        // The shape `signed_for_range` produces: a loop whose counter bound tightens by exactly one
        // per round. The fixed point exists but is 2^63 rounds away, and before widening the solver
        // was still moving after 2000 rounds. What is asserted here is only that the answer is
        // *sound* — the counter really is inside its width — because which bound the widening
        // settles on is the operator's business, not this test's.
        let mut ssa = HLSSA::with_main("main".to_string());
        let (_, _, counter, _, _) = counted_loop(&mut ssa);
        let ranges = run_analysis(&mut ssa);
        let range = ranges.get(counter);
        assert!(
            !range.is_empty(),
            "the counter must not come back unreachable"
        );
        assert!(
            range.unsigned().lo().is_some() && range.unsigned().hi().is_some(),
            "a `Bits` range stays bounded by its width however it was reached"
        );
    }

    #[test]
    fn a_width_mismatched_signed_fact_narrows_nothing() {
        // A `Cmp`'s operands are not required to be the same width: `analysis::types` gives the
        // result `int(1)` without looking at them, exactly as it did for `BinaryArithOp` before
        // `assert_int_arith_widths`. So the narrowing has to survive one, and for the signed
        // reading surviving means declining.
        //
        // `other` here is `-1` at 32 bits. Read at 32 bits that is the number -1; taken as an
        // endpoint for an 8-bit value it would say `value <= -2`, which is a claim about a
        // different number entirely. And narrowing only ever tightens, so a bogus bound cannot
        // make a consumer emit a check it would otherwise skip -- only delete one it needs. The
        // unsigned reading is width-independent and so is left alone; the sibling assertion is
        // what keeps this from passing on a `constraint` that had simply stopped answering.
        let (value, other) = (ValueId(1), ValueId(2));
        let wide_minus_one = ValueRange::from_signed(Width::Bits(32), Interval::closed(-1, -1));

        let signed = BranchFact {
            kind: CmpKind::SLt,
            lhs: value,
            rhs: other,
            holds: true,
        };
        let base = ValueRange::full(Width::Bits(8));
        let narrowed = narrow(value, base.clone(), &[signed.clone()], |_| {
            wide_minus_one.clone()
        });
        assert_eq!(
            narrowed, base,
            "a mismatched-width `SLt` must not bound anything"
        );

        // At the value's own width the same fact does bound it, so the guard above is not simply
        // switching the feature off.
        let matched = ValueRange::from_signed(Width::Bits(8), Interval::closed(-1, -1));
        let narrowed = narrow(value, base.clone(), &[signed], |_| matched.clone());
        assert_eq!(narrowed.signed(), &iv(-128, -2));

        // The unsigned reading is the raw pattern as a non-negative integer, which means the same
        // thing at every width, so a mismatch there is not an obstacle and must not be treated as
        // one.
        let unsigned = BranchFact {
            kind: CmpKind::ULt,
            lhs: value,
            rhs: other,
            holds: true,
        };
        let wide_ten = ValueRange::from_unsigned(Width::Bits(32), Interval::closed(10, 10));
        let narrowed = narrow(value, base, &[unsigned], |_| wide_ten.clone());
        assert_eq!(narrowed.unsigned(), &iv(0, 9));
    }

    #[test]
    fn a_nested_loop_bounds_both_counters() {
        // Each nesting level needs its own trip round through the fixed point before its counter
        // stops being ⊤, so this is really a test that `ITER_LIMIT` is generous enough for the
        // shapes the standard library actually contains. Both counters bounded means the solver
        // still converged; a `[0, 255]` here would mean it ran out of rounds.
        let mut ssa = HLSSA::with_main("main".to_string());
        let main_id = ssa.get_unique_entrypoint_id();
        let (outer_counter, inner_counter, down) = {
            let mut sb = HLSSABuilder::new(&mut ssa);
            sb.modify_function(main_id, |b| {
                b.function.add_return_type(Type::int(8));
                let outer = b.add_block(|_| {});
                let inner_pre = b.add_block(|_| {});
                let inner = b.add_block(|_| {});
                let body = b.add_block(|_| {});
                let outer_latch = b.add_block(|_| {});
                let exit = b.add_block(|_| {});

                let entry = b.function.get_entry_id();
                {
                    let mut e = b.test_block(entry);
                    let zero = e.int_const(8, 0);
                    e.terminate_jmp(outer, vec![zero]);
                }
                let i = {
                    let mut e = b.test_block(outer);
                    let i = e.add_parameter(Type::int(8));
                    let four = e.int_const(8, 4);
                    let c = e.ult(i, four);
                    e.terminate_jmp_if(c, inner_pre, exit);
                    i
                };
                {
                    let mut e = b.test_block(inner_pre);
                    let zero = e.int_const(8, 0);
                    e.terminate_jmp(inner, vec![zero]);
                }
                let j = {
                    let mut e = b.test_block(inner);
                    let j = e.add_parameter(Type::int(8));
                    let four = e.int_const(8, 4);
                    let c = e.ult(j, four);
                    e.terminate_jmp_if(c, body, outer_latch);
                    j
                };
                let down = {
                    let mut e = b.test_block(body);
                    let three = e.int_const(8, 3);
                    let down = e.usub(three, j);
                    let one = e.int_const(8, 1);
                    let next = e.uadd(j, one);
                    e.terminate_jmp(inner, vec![next]);
                    down
                };
                {
                    let mut e = b.test_block(outer_latch);
                    let one = e.int_const(8, 1);
                    let next = e.uadd(i, one);
                    e.terminate_jmp(outer, vec![next]);
                }
                {
                    let mut e = b.test_block(exit);
                    e.terminate_return(vec![i]);
                }
                (i, j, down)
            })
        };

        let ranges = run_analysis(&mut ssa);
        assert_eq!(ranges.get(outer_counter).unsigned(), &iv(0, 4));
        assert_eq!(ranges.get(inner_counter).unsigned(), &iv(0, 4));
        assert_eq!(ranges.get(down).unsigned(), &iv(0, 3));
    }

    #[test]
    fn the_untaken_side_of_a_guard_bounds_from_below() {
        // `i >= 16` on the exit edge. Nothing in the loop lowering reads this, but a `if x < c`
        // in user code has two sides and the negative one is a bound just as much.
        let mut ssa = HLSSA::with_main("main".to_string());
        let main_id = ssa.get_unique_entrypoint_id();
        let (value, then_block, else_block) = {
            let mut sb = HLSSABuilder::new(&mut ssa);
            sb.modify_function(main_id, |b| {
                b.function.add_return_type(Type::int(8));
                let then_block = b.add_block(|_| {});
                let else_block = b.add_block(|_| {});
                let entry = b.function.get_entry_id();
                let v = {
                    let mut e = b.test_block(entry);
                    let v = e.add_parameter(Type::int(8));
                    let limit = e.int_const(8, 10);
                    let c = e.ult(v, limit);
                    e.terminate_jmp_if(c, then_block, else_block);
                    v
                };
                for block in [then_block, else_block] {
                    let mut e = b.test_block(block);
                    e.terminate_return(vec![v]);
                }
                (v, then_block, else_block)
            })
        };

        let ranges = run_analysis(&mut ssa);
        assert_eq!(ranges.get(value).unsigned(), &iv(0, 255));
        assert_eq!(ranges.get_at(then_block, value).unsigned(), &iv(0, 9));
        assert_eq!(ranges.get_at(else_block, value).unsigned(), &iv(10, 255));
    }

    #[test]
    fn a_target_with_two_predecessors_learns_nothing() {
        // Arriving at a block reachable from elsewhere does not say which edge was taken, so the
        // fact would be false on the other one. Here the entry's `else` and a third block both
        // jump to the same place.
        let mut ssa = HLSSA::with_main("main".to_string());
        let main_id = ssa.get_unique_entrypoint_id();
        let (value, shared) = {
            let mut sb = HLSSABuilder::new(&mut ssa);
            sb.modify_function(main_id, |b| {
                b.function.add_return_type(Type::int(8));
                let then_block = b.add_block(|_| {});
                let shared = b.add_block(|_| {});
                let entry = b.function.get_entry_id();
                let v = {
                    let mut e = b.test_block(entry);
                    let v = e.add_parameter(Type::int(8));
                    let limit = e.int_const(8, 10);
                    let c = e.ult(v, limit);
                    e.terminate_jmp_if(c, then_block, shared);
                    v
                };
                {
                    // The taken side also falls into `shared`, giving it a second predecessor.
                    let mut e = b.test_block(then_block);
                    e.terminate_jmp(shared, vec![]);
                }
                {
                    let mut e = b.test_block(shared);
                    e.terminate_return(vec![v]);
                }
                (v, shared)
            })
        };

        let ranges = run_analysis(&mut ssa);
        assert_eq!(ranges.get_at(shared, value).unsigned(), &iv(0, 255));
    }

    #[test]
    fn a_signed_guard_bounds_the_signed_reading() {
        // `SLt` decides the two's-complement reading, so the bound belongs on that component. Read
        // as unsigned the surviving patterns are two runs, which is precisely what the reduced
        // product exists to express.
        let mut ssa = HLSSA::with_main("main".to_string());
        let main_id = ssa.get_unique_entrypoint_id();
        let (value, then_block) = {
            let mut sb = HLSSABuilder::new(&mut ssa);
            sb.modify_function(main_id, |b| {
                b.function.add_return_type(Type::int(8));
                let then_block = b.add_block(|_| {});
                let else_block = b.add_block(|_| {});
                let entry = b.function.get_entry_id();
                let v = {
                    let mut e = b.test_block(entry);
                    let v = e.add_parameter(Type::int(8));
                    let limit = e.int_const(8, 3);
                    let c = e.slt(v, limit);
                    e.terminate_jmp_if(c, then_block, else_block);
                    v
                };
                for block in [then_block, else_block] {
                    let mut e = b.test_block(block);
                    e.terminate_return(vec![v]);
                }
                (v, then_block)
            })
        };

        let ranges = run_analysis(&mut ssa);
        assert_eq!(ranges.get_at(then_block, value).signed(), &iv(-128, 2));
    }

    #[test]
    fn comparing_a_value_with_itself_bounds_nothing() {
        // `x < x` cannot hold, so the branch is unreachable and there is nothing to learn. The two
        // rules would otherwise each narrow `x` using `x`, which is the one shape where the
        // "read the far operand flat" discipline has no far operand to read.
        let mut ssa = HLSSA::with_main("main".to_string());
        let main_id = ssa.get_unique_entrypoint_id();
        let (value, then_block) = {
            let mut sb = HLSSABuilder::new(&mut ssa);
            sb.modify_function(main_id, |b| {
                b.function.add_return_type(Type::int(8));
                let then_block = b.add_block(|_| {});
                let else_block = b.add_block(|_| {});
                let entry = b.function.get_entry_id();
                let v = {
                    let mut e = b.test_block(entry);
                    let v = e.add_parameter(Type::int(8));
                    let c = e.ult(v, v);
                    e.terminate_jmp_if(c, then_block, else_block);
                    v
                };
                for block in [then_block, else_block] {
                    let mut e = b.test_block(block);
                    e.terminate_return(vec![v]);
                }
                (v, then_block)
            })
        };

        let ranges = run_analysis(&mut ssa);
        assert_eq!(ranges.get_at(then_block, value).unsigned(), &iv(0, 255));
    }

    #[test]
    fn a_select_across_widths_keeps_the_narrow_branchs_values() {
        // `Types` unifies a `Select`'s alternatives with `get_arithmetic_result_type`, so a `u8`
        // and a `u16` branch give a `u16` result and the `u8` operand's range has to be re-read at
        // the wider width. Carrying its _signed_ reading across is what made that go wrong: `200`
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
                b.function.add_return_type(Type::int(16));
                let entry = b.function.get_entry_id();
                let mut e = b.test_block(entry);
                // Opaque rather than `int_const(1, 1)`: the `Select` transfer ignores its condition
                // today, so a constant would still join both branches -- but it would stop doing
                // so the moment anyone teaches it to fold a known condition, and the test would
                // keep passing while testing nothing.
                let cond = e.add_parameter(Type::int(1));
                let narrow = e.int_const(8, 200);
                let wide = e.int_const(16, 1000);
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
        // information in _both_ readings — neither interval alone denotes `{3, 200}` — which is the
        // reduced product doing the only thing a single interval could not.
        let mut ssa = HLSSA::with_main("main".to_string());
        let main_id = ssa.get_unique_entrypoint_id();
        let merged;
        {
            let mut sb = HLSSABuilder::new(&mut ssa);
            merged = sb.modify_function(main_id, |b| {
                b.function.add_return_type(Type::int(8));

                let merge = b.add_block(|_| {});
                let then_block = b.add_block(|_| {});
                let else_block = b.add_block(|_| {});

                let param = {
                    let mut e = b.test_block(merge);
                    let p = e.add_parameter(Type::int(8));
                    e.terminate_return(vec![p]);
                    p
                };

                let entry = b.function.get_entry_id();
                {
                    let mut e = b.test_block(entry);
                    let cond = e.int_const(1, 1);
                    e.terminate_jmp_if(cond, then_block, else_block);
                }
                {
                    let mut e = b.test_block(then_block);
                    let small = e.int_const(8, 3);
                    e.terminate_jmp(merge, vec![small]);
                }
                {
                    let mut e = b.test_block(else_block);
                    let large = e.int_const(8, 200);
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
                b.function.add_return_type(Type::int(8));
                let entry = b.function.get_entry_id();
                let mut e = b.test_block(entry);
                let x = e.add_parameter(Type::field());
                // A witness condition, which is what leaves a `Guard` standing.
                let w = e.write_witness(x);
                let cond = e.eq(w, x);
                let a = e.int_const(8, lhs);
                let b_ = e.int_const(8, rhs);
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

        // `0 - 1` on a `u8` underflows, so the range of the operation _itself_ is ⊥. An inactive
        // guard around it does not underflow — it produces zero — and a transfer that recursed
        // into the inner operation alone would report the whole thing unreachable.
        let r = guarded_u8_op(USub, 0, 1);
        assert!(!r.is_empty());
        assert_eq!(r.unsigned(), &iv(0, 0));

        // Division by zero and an out-of-range shift also produce zero when inactive, but the
        // transfer has no bound on either operation to begin with, so the join is invisible.
        assert_eq!(guarded_u8_op(UDiv, 7, 0).unsigned(), &iv(0, 255));
        assert_eq!(guarded_u8_op(UShl, 3, 9).unsigned(), &iv(0, 255));

        // An operation that does not fail keeps its own value, joined with the zero the failure
        // branch would have produced.
        assert_eq!(guarded_u8_op(UAdd, 3, 4).unsigned(), &iv(0, 7));

        // Without the fix, `known_sign` could read a range that excludes zero and hardcode a sign
        // bit the inactive branch contradicts.
        assert!(!guarded_u8_op(USub, 3, 4).unsigned().is_empty());
    }

    #[test]
    fn a_guard_counts_one_refinement_per_round_not_two() {
        // The `Guard` arm writes each result **twice** per round, and the two writes disagree by
        // construction: the second is the first joined with zero. So `overwrite` sees a move on
        // both of them in every round, including every round after the result has settled.
        //
        // Left uncorrected that is two refinements per round for a value that is not moving, and
        // [`WIDEN_AFTER`] would be reached at roughly half the intended budget — releasing the
        // endpoints of a range that had been stable for eighty rounds. What the counter is meant
        // to measure is how often the pair moved, which after settling is never.
        //
        // The rounds are driven directly rather than through a program that takes enough of them
        // to widen: the solver reaches its fixed point in three rounds on anything writable here,
        // so a whole-program test could only assert this by accident.
        let mut ssa = HLSSA::with_main("main".to_string());
        let main_id = ssa.get_unique_entrypoint_id();
        let (guard, result, lhs, rhs) = {
            let mut sb = HLSSABuilder::new(&mut ssa);
            sb.modify_function(main_id, |b| {
                b.function.add_return_type(Type::int(8));
                let entry = b.function.get_entry_id();
                let mut e = b.test_block(entry);
                let x = e.add_parameter(Type::field());
                // A witness condition, which is what leaves a `Guard` standing.
                let w = e.write_witness(x);
                let cond = e.eq(w, x);
                let a = e.int_const(8, 3);
                let c = e.int_const(8, 4);
                let result = e.fresh_value();
                let guard = OpCode::Guard {
                    condition: cond,
                    inner: Box::new(OpCode::BinaryArithOp {
                        kind: BinaryArithOpKind::UAdd,
                        result,
                        lhs: a,
                        rhs: c,
                    }),
                };
                e.emit(guard.clone());
                e.terminate_return(vec![result]);
                (guard, result, a, c)
            })
        };

        let flow = FlowAnalysis::run(&ssa);
        let types = crate::compiler::analysis::types::Types::new().run(&ssa, &flow);
        let function_types = types.get_function(main_id);

        let analysis = ValueRangeAnalysis::new();
        // The operand bounds the solver would have seeded from the constant pool. They matter:
        // a computed range that already contains zero makes the second write a no-op, and the
        // double count this is about would not arise at all.
        let mut bounds: HashMap<ValueId, ValueRange> = HashMap::default();
        for (value, n) in [(lhs, 3), (rhs, 4)] {
            bounds.insert(
                value,
                ValueRange::from_unsigned(Width::Bits(8), Interval::closed(n, n)),
            );
        }
        let mut refinements: HashMap<ValueId, usize> = HashMap::default();
        let round = |bounds: &mut HashMap<ValueId, ValueRange>,
                     refinements: &mut HashMap<ValueId, usize>| {
            let mut changed = false;
            analysis.transfer(
                &guard,
                function_types,
                bounds,
                refinements,
                &mut changed,
                f(),
                &[],
            );
            changed
        };

        // Two rounds to settle: the first stores the pair, the second reproduces it.
        round(&mut bounds, &mut refinements);
        assert!(
            !round(&mut bounds, &mut refinements),
            "the guard should reach its fixed point on the second round"
        );
        // `3 + 4` is 7, joined with the zero an inactive guard produces.
        assert_eq!(bounds[&result].unsigned(), &iv(0, 7));

        let settled = refinements.get(&result).copied().unwrap_or(0);
        for _ in 0..8 {
            assert!(
                !round(&mut bounds, &mut refinements),
                "a settled guard must not report a change"
            );
        }
        assert_eq!(
            refinements.get(&result).copied().unwrap_or(0),
            settled,
            "a settled guarded result kept accruing refinements, so it would be widened for \
             standing still"
        );
    }
}

/// The range domain's conformance relation to the normative model in `mavros-int-semantics`.
///
/// This analysis provides something different from every other evaluator in the batch. It does not
/// compute a value, so it can neither equal [`eval`](semantics::eval) nor refine it. Instead it has
/// to be sound, that is that the set of values an execution can actually produce is contained in
/// the set its answer denotes:
///
/// ```text
/// { v : eval(op, sign, bits, a, bits, b) == Value(v),  a ∈ γ(L),  b ∈ γ(R) }  ⊆  γ(binary_arith(..))
/// ```
///
/// Two things about that statement are important to note.
///
/// It quantifies over [`Outcome::Value`] only. A rejected execution produces no value at all,
/// instead becoming a runtime constraint failure, so the analysis owes nothing on those inputs.
/// That is exactly the licence `wrap_or_trap` relies on when it returns the *non*-wrapping interval
/// for an operation that would overflow, and stating the relation this way turns that licence into
/// something the sweep checks rather than something the comment asserts.
///
/// And γ is over **both** readings, because a `ValueRange` denotes the patterns its unsigned and
/// signed intervals both admit. Checking one reading would pass a range that is unsound in the
/// other.
#[cfg(test)]
mod int_semantics_conformance {
    use mavros_int_semantics::{self as semantics, IntOp, Outcome, Raw, corners};

    use super::*;

    /// Whether a bit pattern is in γ of a range: admitted by both readings.
    fn in_gamma(range: &ValueRange, bits: usize, v: Raw) -> bool {
        range.unsigned().contains(&BigInt::from(v))
            && range.signed().contains(&signed_const_to_bigint(bits, v))
    }

    /// Every bit pattern a range denotes, at a width small enough to enumerate.
    fn gamma(range: &ValueRange, bits: usize) -> Vec<Raw> {
        assert!(bits <= 8, "γ is enumerated, so the width has to stay small");
        (0..=semantics::mask(bits))
            .filter(|v| in_gamma(range, bits, *v))
            .collect()
    }

    /// The input ranges each operand is swept over at `bits`.
    ///
    /// Chosen for what they straddle rather than for coverage: the whole width, the two singletons
    /// whose readings disagree most (`0` and all-ones, which is `-1`), a run of small
    /// non-negatives, and a run sitting astride the sign boundary.
    fn input_ranges(bits: usize) -> Vec<ValueRange> {
        let width = Width::Bits(bits);
        let top = semantics::mask(bits);
        let half = top / 2;
        let closed = |lo: Raw, hi: Raw| {
            ValueRange::from_unsigned(width, Interval::closed(BigInt::from(lo), BigInt::from(hi)))
        };

        let mut out = vec![
            ValueRange::full(width),
            closed(0, 0),
            closed(top, top),
            closed(0, top.min(3)),
            closed(half, top.min(half + 3)),
        ];
        // The signed reading as the _known_ one, so the reduction is entered from both sides.
        out.push(ValueRange::from_signed(
            width,
            Interval::closed(BigInt::from(-1i64), BigInt::from(1i64)),
        ));
        out
    }

    /// Every operation the transfer has a rule for, paired with the model operation it means.
    ///
    /// Spelled out so that adding a `BinaryArithOpKind` is a compile error in a test that would
    /// otherwise silently stop covering it.
    fn operations() -> Vec<(BinaryArithOpKind, IntOp, semantics::Sign)> {
        use BinaryArithOpKind as K;
        use semantics::Sign::{Signed, Unsigned};

        vec![
            (K::UAdd, IntOp::Add, Unsigned),
            (K::SAdd, IntOp::Add, Signed),
            (K::USub, IntOp::Sub, Unsigned),
            (K::SSub, IntOp::Sub, Signed),
            (K::UMul, IntOp::Mul, Unsigned),
            (K::SMul, IntOp::Mul, Signed),
            (K::UDiv, IntOp::Div, Unsigned),
            (K::SDiv, IntOp::Div, Signed),
            (K::URem, IntOp::Rem, Unsigned),
            (K::SRem, IntOp::Rem, Signed),
            (K::UShl, IntOp::Shl, Unsigned),
            (K::SShl, IntOp::Shl, Signed),
            (K::UShr, IntOp::Shr, Unsigned),
            (K::SShr, IntOp::Shr, Signed),
            (K::And, IntOp::And, Unsigned),
            (K::Or, IntOp::Or, Unsigned),
            (K::Xor, IntOp::Xor, Unsigned),
        ]
    }

    #[test]
    fn the_transfer_is_sound_for_every_accepted_execution() {
        let mut checked = 0usize;

        for bits in corners::EXHAUSTIVE_WIDTHS {
            let width = Width::Bits(bits);
            let ranges = input_ranges(bits);

            for (kind, op, sign) in operations() {
                for l in &ranges {
                    for r in &ranges {
                        // Both operands at the result's width, which is the case every rule here
                        // is written for; a mismatched width makes the transfer answer TOP, and a
                        // TOP answer is sound by construction.
                        let out = ValueRangeAnalysis::binary_arith(kind, width, l, r, true, true);

                        for a in gamma(l, bits) {
                            for b in gamma(r, bits) {
                                let Outcome::Value(v) = semantics::eval(op, sign, bits, a, bits, b)
                                else {
                                    // A rejected execution produces no value, so there is nothing
                                    // for the range to have contained.
                                    continue;
                                };

                                assert!(
                                    in_gamma(&out, bits, v),
                                    "{kind:?} at {bits} bits: {a:#x} ∈ γ({l:?}) and {b:#x} ∈ \
                                     γ({r:?}) give {v:#x}, which escapes {out:?}"
                                );
                                checked += 1;
                            }
                        }
                    }
                }
            }
        }

        // An analysis that answered TOP everywhere would be sound and useless, and so would a sweep
        // whose ranges all turned out empty.
        assert!(
            checked > 50_000,
            "the sweep only reached {checked} accepted executions"
        );
    }
}
