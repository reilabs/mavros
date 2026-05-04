use std::collections::HashMap;

use ark_ff::PrimeField;
use num_bigint::{BigInt, Sign};
use num_traits::{One, Signed, ToPrimitive, Zero};
use tracing::{Level, instrument};

use crate::compiler::{
    Field,
    analysis::types::{FunctionTypeInfo, TypeInfo},
    flow_analysis::{CFG, FlowAnalysis},
    ir::r#type::{Type, TypeExpr},
    pass_manager::{Analysis, AnalysisId, AnalysisStore},
    ssa::{
        BinaryArithOpKind, BlockId, CastTarget, ConstValue, FunctionId, HLFunction, HLSSA,
        Instruction, OpCode, Terminator, ValueId,
    },
};

/// A closed integer interval `[lo, hi]` over the integers ℤ, with `None`
/// endpoints representing −∞ / +∞. Top is `(None, None)`; the empty interval
/// is any pair with `lo > hi` (we normalise such pairs back to `EMPTY`).
///
/// This is the abstract domain of the value-range analysis: every numeric SSA
/// value is summarised by the smallest interval the analysis can prove
/// contains its **integer interpretation**, regardless of whether the SSA type
/// is `u_n`, `i_n`, or `Field`. So the same domain handles the full `u128`
/// range, large `Field` constants, negative `i_n` values — anything an SSA
/// integer might denote.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IntInterval {
    pub lo: Option<BigInt>,
    pub hi: Option<BigInt>,
}

impl IntInterval {
    /// `(-∞, +∞)`.
    pub fn top() -> Self {
        Self {
            lo: None,
            hi: None,
        }
    }

    /// The unique empty interval (used as the bottom of the lattice). Any
    /// "lo > hi" interval is normalised to this representation.
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

    /// Initial bound for a value of the given declared type, looking through
    /// `WitnessOf`. Non-numeric types get TOP.
    pub fn for_type(ty: &Type) -> Self {
        match &ty.strip_witness().expr {
            TypeExpr::U(n) => Self::unsigned_full(*n),
            TypeExpr::I(n) => Self::signed_full(*n),
            _ => Self::top(),
        }
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

    // ---- Geometric queries ----

    pub fn is_non_negative(&self) -> bool {
        self.lo.as_ref().is_some_and(|l| !l.is_negative())
    }

    pub fn is_non_positive(&self) -> bool {
        self.hi.as_ref().is_some_and(|h| !h.is_positive())
    }

    /// True iff every value in the interval fits in `bits`-bit unsigned
    /// representation (i.e. is in `[0, 2^bits)`).
    pub fn fits_in_unsigned_bits(&self, bits: usize) -> bool {
        let cap = BigInt::one() << bits;
        self.lo.as_ref().is_some_and(|l| !l.is_negative())
            && self.hi.as_ref().is_some_and(|h| h < &cap)
    }

    /// True iff every value in the interval fits in `bits`-bit two's-complement
    /// signed representation (i.e. is in `[−2^(bits−1), 2^(bits−1))`).
    pub fn fits_in_signed_bits(&self, bits: usize) -> bool {
        if bits == 0 {
            return matches!((&self.lo, &self.hi), (Some(l), Some(h)) if l.is_zero() && h.is_zero());
        }
        let half = BigInt::one() << (bits - 1);
        self.lo.as_ref().is_some_and(|l| l >= &(-half.clone()))
            && self.hi.as_ref().is_some_and(|h| h < &half)
    }

    /// True iff every value, viewed as `bits`-bit two's-complement, has its
    /// sign bit set to 0. Equivalent to `[0, 2^(bits-1))` containment.
    pub fn is_non_negative_in_signed(&self, bits: usize) -> bool {
        if bits == 0 {
            return false;
        }
        let half = BigInt::one() << (bits - 1);
        self.lo.as_ref().is_some_and(|l| !l.is_negative())
            && self.hi.as_ref().is_some_and(|h| h < &half)
    }

    // ---- Arithmetic transfers (interval arithmetic) ----

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
        // The four "extreme products" between endpoint pairs determine the
        // hull: when an endpoint is ±∞, the corresponding products are ±∞ too,
        // unless the other factor is 0.
        let products = [
            opt_mul(self.lo.as_ref(), other.lo.as_ref()),
            opt_mul(self.lo.as_ref(), other.hi.as_ref()),
            opt_mul(self.hi.as_ref(), other.lo.as_ref()),
            opt_mul(self.hi.as_ref(), other.hi.as_ref()),
        ];
        let lo = products.iter().fold(None, |acc, p| match (acc, p) {
            (None, _) => p.clone(),
            (Some(_), None) => None,
            (Some(a), Some(b)) => Some(if &a < b { a } else { b.clone() }),
        });
        // For hi, None means +∞ (a top endpoint); we want the maximum.
        let hi = products.iter().fold(None, |acc, p| match (acc, p) {
            (None, _) => p.clone(),
            (Some(_), None) => None,
            (Some(a), Some(b)) => Some(if &a > b { a } else { b.clone() }),
        });
        // The fold above conflates "I haven't seen anything yet" (acc = None)
        // with "+∞ / -∞" (entry = None). Recompute cleanly:
        Self::mul_hull(&products)
    }

    fn mul_hull(products: &[Option<BigInt>; 4]) -> Self {
        // lo = min of all products, where None among the inputs means -∞ for
        // the lo side and +∞ for the hi side. To distinguish, we build them
        // separately: an input None contributes -∞ to the lo aggregation and
        // +∞ to the hi aggregation.
        let mut lo: Option<BigInt> = products[0].clone();
        for p in &products[1..] {
            lo = match (&lo, p) {
                (None, _) => None,                 // -∞ stays -∞
                (_, None) => None,                 // saw a -∞ candidate
                (Some(a), Some(b)) if b < a => Some(b.clone()),
                _ => lo,
            };
        }
        let mut hi: Option<BigInt> = products[0].clone();
        for p in &products[1..] {
            hi = match (&hi, p) {
                (None, _) => None,
                (_, None) => None,
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

/// Floor division for `BigInt`, matching Rust integer semantics for positive
/// divisors (rounds toward −∞).
fn floor_div(a: &BigInt, d: &BigInt) -> BigInt {
    let (q, r) = (a / d, a % d);
    if !r.is_zero() && r.is_negative() != d.is_negative() {
        q - BigInt::one()
    } else {
        q
    }
}

/// `lo` operands: `None` = −∞ (the smallest possible). So the min is the one
/// that's `None` if any, otherwise the smaller `Some`.
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

/// Convert a `ConstValue::I(bits, encoded)` u128 bit pattern back to a
/// signed `BigInt`. Two's-complement decode.
fn signed_const_to_bigint(bits: usize, encoded: u128) -> BigInt {
    if bits == 0 || bits >= 128 {
        return BigInt::from(encoded);
    }
    let sign_bit = 1u128 << (bits - 1);
    if encoded & sign_bit == 0 {
        BigInt::from(encoded)
    } else {
        // Negative: subtract 2^bits.
        let two_n = BigInt::one() << bits;
        BigInt::from(encoded) - two_n
    }
}

/// Convert a Field element to a `BigInt` (always non-negative, in `[0, p)`).
fn field_to_bigint(f: &Field) -> BigInt {
    let limbs = f.into_bigint().0; // [u64; 4]
    let bytes_le: Vec<u8> = limbs.iter().flat_map(|l| l.to_le_bytes()).collect();
    BigInt::from_bytes_le(Sign::Plus, &bytes_le)
}

pub struct FunctionValueRanges {
    values: HashMap<ValueId, IntInterval>,
}

impl FunctionValueRanges {
    /// Get the interval for a value, returning `top()` (i.e. "unknown") if
    /// the value isn't in our map (e.g. fresh values created downstream of
    /// this analysis).
    pub fn get(&self, v: ValueId) -> IntInterval {
        self.values
            .get(&v)
            .cloned()
            .unwrap_or_else(IntInterval::top)
    }

    pub fn try_get(&self, v: ValueId) -> Option<&IntInterval> {
        self.values.get(&v)
    }
}

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

pub struct ValueRangeAnalysis;

const ITER_LIMIT: usize = 8;

impl ValueRangeAnalysis {
    pub fn new() -> Self {
        Self
    }

    #[instrument(skip_all, name = "ValueRangeAnalysis::run")]
    pub fn run(&self, ssa: &HLSSA, cfg: &FlowAnalysis, types: &TypeInfo) -> ValueRanges {
        let mut result = ValueRanges {
            functions: HashMap::new(),
        };
        for (function_id, function) in ssa.iter_functions() {
            let func_cfg = cfg.get_function_cfg(*function_id);
            let func_types = types.get_function(*function_id);
            let function_ranges = self.run_function(function, func_cfg, func_types);
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
    ) -> FunctionValueRanges {
        let mut bounds: HashMap<ValueId, IntInterval> = HashMap::new();

        // Initial state: every value's bound is its declared type's full range.
        // Iteration only narrows from there.
        for (_block_id, block) in function.get_blocks() {
            for (vid, ty) in block.get_parameters() {
                bounds.insert(*vid, IntInterval::for_type(ty));
            }
            for instr in block.get_instructions() {
                for vid in instr.get_results() {
                    bounds.insert(*vid, IntInterval::for_type(types.get_value_type(*vid)));
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
                    let pred_args: Vec<Vec<ValueId>> = cfg
                        .get_predecessors(block_id)
                        .into_iter()
                        .filter_map(|p| {
                            let term = function.get_block(p).get_terminator().unwrap();
                            match term {
                                Terminator::Jmp(t, args) if *t == block_id => Some(args.clone()),
                                _ => None,
                            }
                        })
                        .collect();

                    for (idx, (param_id, param_type)) in block.get_parameters().enumerate() {
                        let mut joined: Option<IntInterval> = None;
                        for args in &pred_args {
                            if let Some(arg_id) = args.get(idx) {
                                let arg_range = bounds
                                    .get(arg_id)
                                    .cloned()
                                    .unwrap_or_else(IntInterval::top);
                                joined = Some(match joined {
                                    None => arg_range,
                                    Some(j) => j.join(&arg_range),
                                });
                            }
                        }
                        let new_range = joined.unwrap_or_else(|| IntInterval::for_type(param_type));
                        Self::overwrite(&mut bounds, *param_id, new_range, &mut changed);
                    }
                }

                for instr in block.get_instructions() {
                    self.transfer(instr, types, &mut bounds, &mut changed);
                }
            }

            if !changed {
                break;
            }
        }

        FunctionValueRanges { values: bounds }
    }

    fn overwrite(
        bounds: &mut HashMap<ValueId, IntInterval>,
        v: ValueId,
        new: IntInterval,
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
        bounds: &mut HashMap<ValueId, IntInterval>,
        changed: &mut bool,
    ) {
        use BinaryArithOpKind::*;
        let get = |bounds: &HashMap<ValueId, IntInterval>, v: ValueId| -> IntInterval {
            bounds
                .get(&v)
                .cloned()
                .unwrap_or_else(IntInterval::top)
        };
        let cap_to_type = |result: ValueId, r: IntInterval| {
            r.intersect(&IntInterval::for_type(types.get_value_type(result)))
        };

        match instr {
            OpCode::Const { result, value } => {
                let r = match value {
                    ConstValue::U(_, v) => IntInterval::singleton(*v),
                    ConstValue::I(bits, encoded) => {
                        IntInterval::singleton(signed_const_to_bigint(*bits, *encoded))
                    }
                    ConstValue::Field(f) => IntInterval::singleton(field_to_bigint(f)),
                    ConstValue::FnPtr(_) => IntInterval::top(),
                };
                Self::overwrite(bounds, *result, cap_to_type(*result, r), changed);
            }

            OpCode::Cast {
                result,
                value,
                target,
            } => {
                let in_r = get(bounds, *value);
                let r = match target {
                    CastTarget::Field => in_r,
                    CastTarget::U(n) => {
                        // Source bits stay if they fit in [0, 2^n); otherwise
                        // we don't know what the wrap looks like.
                        if in_r.fits_in_unsigned_bits(*n) {
                            in_r
                        } else {
                            IntInterval::unsigned_full(*n)
                        }
                    }
                    CastTarget::I(n) => {
                        if in_r.fits_in_signed_bits(*n) {
                            in_r
                        } else {
                            IntInterval::signed_full(*n)
                        }
                    }
                    CastTarget::Nop | CastTarget::WitnessOf => in_r,
                    CastTarget::ArrayToSlice => IntInterval::top(),
                };
                Self::overwrite(bounds, *result, cap_to_type(*result, r), changed);
            }

            OpCode::Truncate {
                result,
                value,
                to_bits,
                ..
            } => {
                let in_r = get(bounds, *value);
                // Truncate semantics: result = value mod 2^to_bits, in [0, 2^to_bits).
                // If the input is already in that range, preserve. Otherwise
                // we lose precision and fall back to the full unsigned range.
                let cap = IntInterval::unsigned_full(*to_bits);
                let r = if cap.intersect(&in_r) == in_r && in_r.is_non_negative() {
                    in_r
                } else {
                    cap
                };
                Self::overwrite(bounds, *result, cap_to_type(*result, r), changed);
            }

            OpCode::SExt {
                result,
                value,
                from_bits,
                ..
            } => {
                let in_r = get(bounds, *value);
                let r = if in_r.fits_in_signed_bits(*from_bits) {
                    in_r
                } else {
                    IntInterval::signed_full(*from_bits)
                };
                Self::overwrite(bounds, *result, cap_to_type(*result, r), changed);
            }

            OpCode::WriteWitness {
                result: Some(r),
                value,
                ..
            } => {
                let in_r = get(bounds, *value);
                Self::overwrite(bounds, *r, cap_to_type(*r, in_r), changed);
            }
            OpCode::WriteWitness { result: None, .. } => {}

            OpCode::FreshWitness {
                result,
                result_type,
            } => {
                Self::overwrite(
                    bounds,
                    *result,
                    IntInterval::for_type(result_type),
                    changed,
                );
            }

            OpCode::ValueOf { result, value } => {
                let in_r = get(bounds, *value);
                Self::overwrite(bounds, *result, cap_to_type(*result, in_r), changed);
            }

            OpCode::Cmp { result, .. } => {
                Self::overwrite(bounds, *result, IntInterval::unsigned_full(1), changed);
            }

            OpCode::Not { result, .. } => {
                Self::overwrite(
                    bounds,
                    *result,
                    IntInterval::for_type(types.get_value_type(*result)),
                    changed,
                );
            }

            OpCode::BinaryArithOp {
                kind,
                result,
                lhs,
                rhs,
            } => {
                let l = get(bounds, *lhs);
                let r_in = get(bounds, *rhs);
                let result_ty = types.get_value_type(*result);
                let raw = match kind {
                    Add => l.add(&r_in),
                    Sub => l.sub(&r_in),
                    Mul => l.mul(&r_in),
                    Div => {
                        // Only useful if rhs is a positive constant.
                        if let (Some(d_lo), Some(d_hi)) = (&r_in.lo, &r_in.hi) {
                            if d_lo == d_hi && d_lo.is_positive() {
                                l.div_const_pos(d_lo)
                            } else {
                                IntInterval::for_type(result_ty)
                            }
                        } else {
                            IntInterval::for_type(result_ty)
                        }
                    }
                    Mod => {
                        // result is in [0, |rhs| - 1], assuming Noir semantics.
                        match (&r_in.lo, &r_in.hi) {
                            (Some(lo), Some(hi)) if lo.is_positive() => {
                                IntInterval::closed(BigInt::zero(), hi - BigInt::one())
                            }
                            _ => IntInterval::for_type(result_ty),
                        }
                    }
                    And => {
                        // Result fits in the narrower of the two operand widths.
                        let l_hi = l.hi.clone().unwrap_or_else(|| {
                            type_unsigned_max(result_ty).unwrap_or_else(|| BigInt::from(0))
                        });
                        let r_hi = r_in.hi.clone().unwrap_or_else(|| {
                            type_unsigned_max(result_ty).unwrap_or_else(|| BigInt::from(0))
                        });
                        let cap = if l_hi <= r_hi { l_hi } else { r_hi };
                        IntInterval::closed(BigInt::zero(), cap)
                    }
                    Or | Xor => {
                        // result <= next_pow2(max(l.hi, r.hi)) - 1.
                        match (&l.hi, &r_in.hi) {
                            (Some(lh), Some(rh)) => {
                                let m = if lh >= rh { lh.clone() } else { rh.clone() };
                                if m.is_negative() {
                                    IntInterval::for_type(result_ty)
                                } else {
                                    let cap = next_pow2_minus_one(&m);
                                    IntInterval::closed(BigInt::zero(), cap)
                                }
                            }
                            _ => IntInterval::for_type(result_ty),
                        }
                    }
                    Shl | Shr => IntInterval::for_type(result_ty),
                };
                Self::overwrite(bounds, *result, cap_to_type(*result, raw), changed);
            }

            OpCode::MulConst {
                result,
                const_val,
                var,
            } => {
                let c = get(bounds, *const_val);
                let v = get(bounds, *var);
                Self::overwrite(bounds, *result, cap_to_type(*result, c.mul(&v)), changed);
            }

            OpCode::Select {
                result,
                if_t,
                if_f,
                ..
            } => {
                let t = get(bounds, *if_t);
                let f = get(bounds, *if_f);
                Self::overwrite(bounds, *result, cap_to_type(*result, t.join(&f)), changed);
            }

            OpCode::Guard { inner, .. } => {
                self.transfer(inner, types, bounds, changed);
            }

            // Other opcodes: keep the type-based default bound.
            _ => {
                for vid in instr.get_results() {
                    let r = IntInterval::for_type(types.get_value_type(*vid));
                    Self::overwrite(bounds, *vid, r, changed);
                }
            }
        }
    }
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

fn type_unsigned_max(ty: &Type) -> Option<BigInt> {
    match &ty.strip_witness().expr {
        TypeExpr::U(n) => Some((BigInt::one() << *n) - BigInt::one()),
        _ => None,
    }
}

// `ToPrimitive` / `Sign` are imported but only used inside helpers; keep them
// for completeness even if some are currently unreferenced.
#[allow(dead_code)]
fn _suppress_unused_imports(_: &dyn ToPrimitive, _: Sign) {}

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
