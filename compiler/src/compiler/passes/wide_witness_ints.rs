//! The multi-cell representation: a witnessed integer too wide for one field element becomes a
//! group of limbs.
//!
//! A `WitnessOf(Int(N))` with `N > multi_cell_int_bits` becomes `k = ceil(N/h)` values, stored in
//! little-endian order, where `h` is [`witness_limb_bits`]. Limbs `0..k-1` are `WitnessOf(Int(h))`
//! and the top one is `WitnessOf(Int(N - (k-1)h))`, so a limb's declared width remains correct.
//!
//! A **pure** `Int(N)` is untouched as it exists in the hint domain where both backends can compute
//! on it at arbitrary width.
//!
//! This representation is sound because **every limb is range-checked** to its declared width when
//! it is created or modified. `2^h` is invertible mod `p`, so an unbounded limb lets a prover solve
//! for any value, so it is crucial to avoid that.
//!
//! Limbs are created in two shapes:
//!
//! - **A single value becoming limbs** via [`Rewriter::decompose`] says that each limb is witnessed
//!   from the pure shadow, range-checked to its width, and the limbs are then constrained to
//!   recombine to the value from which they came. Without the reconstruction constraint the limbs
//!   would be range-checked garbage unrelated to the source.
//! - **A value that is limbs from the start** (a widening cast, a pure-to-witness injection) has no
//!   source value to agree with, so they are simply range-checked.
//!
//! Everything else simply moves limbs around, and so they remain constrained.
//!
//! # Strategy
//!
//! Types come from a single [`TypeInfo`] snapshot taken on the original, type-consistent IR; they
//! are never recomputed mid-pass. Each function is handled in two phases, as `elide_tuples` does
//! for tuples:
//!
//! 1. **Plan:** Build a `value_map: ValueId -> Vec<ValueId>` mapping every value to its limbs. A
//!    value with no wide witness integer in its type maps to itself, so nothing narrow undergoes
//!    needless churn.
//! 2. **Rewrite:** Flatten returns, block parameters, instructions and terminators through that
//!    map, emitting the gadgets above where an instruction does more than move limbs.

use mavros_artifacts::FieldConfig;
use mavros_int_semantics::IntBits;

use crate::collections::HashMap;
use crate::compiler::{
    Field,
    analysis::{
        flow_analysis::FlowAnalysis,
        types::{FunctionTypeInfo, TypeInfo},
    },
    pass_manager::{Analysis, AnalysisId, AnalysisStore, Pass},
    passes::shared::limbs::{widest_injective_int_bits, witness_limb_bits},
    ssa::{
        BlockId, FunctionId, Instruction, Located, Terminator, ValueId,
        hlssa::{
            BinaryArithOpKind, CastTarget, CmpKind, Constant, HLSSA, OpCode, Type, TypeExpr,
            builder::two_pow_pattern,
        },
    },
};

// WIDE WITNESSED INTEGER SPILLING PASS
// ================================================================================================

pub struct WideWitnessInts {}

impl WideWitnessInts {
    pub fn new() -> Self {
        WideWitnessInts {}
    }
}

impl Default for WideWitnessInts {
    fn default() -> Self {
        Self::new()
    }
}

impl Pass for WideWitnessInts {
    fn name(&self) -> &'static str {
        "wide_witness_ints"
    }

    fn needs(&self) -> Vec<AnalysisId> {
        vec![TypeInfo::id(), FlowAnalysis::id()]
    }

    fn run(&self, ssa: &mut HLSSA, store: &AnalysisStore) {
        let field = ssa.field();
        for global in ssa.get_global_types() {
            assert!(
                limb_types(global, field).len() == 1,
                "ICE: a global carries a wide witnessed integer, which has no slot layout here"
            );
        }

        let type_info = store.get::<TypeInfo>();
        let cfg = store.get::<FlowAnalysis>();

        let function_ids: Vec<FunctionId> = ssa.get_function_ids().collect();
        for fid in function_ids {
            let reachable: Vec<BlockId> = cfg
                .get_function_cfg(fid)
                .get_domination_pre_order()
                .collect();
            let value_map = plan_function(ssa, fid, &reachable, type_info);
            if value_map.values().all(|limbs| limbs.len() == 1) {
                continue;
            }
            rewrite_function(ssa, fid, &reachable, &value_map, type_info, field);
        }
    }

    fn preserves(&self) -> Vec<AnalysisId> {
        // Signatures, values and block parameters all change.
        vec![]
    }
}

// TYPES
// ================================================================================================

/// The widths a `bits`-wide integer splits into, little-endian, with the top limb carrying the
/// remainder rather than a full limb's width.
fn limb_widths(bits: usize, limb_bits: usize) -> Vec<usize> {
    let count = bits.div_ceil(limb_bits);
    (0..count)
        .map(|index| {
            let low = index * limb_bits;
            (bits - low).min(limb_bits)
        })
        .collect()
}

/// The parallel types a type expands into, which is itself unless it's a witnessed wide integer.
fn limb_types(ty: &Type, field: FieldConfig) -> Vec<Type> {
    match &ty.expr {
        TypeExpr::WitnessOf(inner) => match &inner.expr {
            TypeExpr::Int(bits) if *bits > multi_cell_int_bits(field) => {
                limb_widths(*bits, witness_limb_bits(field))
                    .into_iter()
                    .map(|width| Type::witness_of(Type::int(width)))
                    .collect()
            }
            _ => vec![ty.clone()],
        },
        // A sequence carrying one is refused by `width_validation` before here: an element is read
        // off the lookup tape as a single field element for the moment.
        TypeExpr::Array(inner, _) | TypeExpr::Slice(inner) => {
            assert!(
                wide_witness_width(inner, field).is_none(),
                "ICE: a sequence of wide witnessed integers reached the multi-cell representation"
            );
            vec![ty.clone()]
        }
        TypeExpr::Ref(inner) => limb_types(inner, field)
            .into_iter()
            .map(|leaf| leaf.ref_of())
            .collect(),
        _ => vec![ty.clone()],
    }
}

/// The width of a witnessed integer that this pass represents as limbs, or [`None`] otherwise.
fn wide_witness_width(ty: &Type, field: FieldConfig) -> Option<usize> {
    match &ty.expr {
        TypeExpr::WitnessOf(inner) => match &inner.expr {
            TypeExpr::Int(bits) if *bits > multi_cell_int_bits(field) => Some(*bits),
            _ => None,
        },
        _ => None,
    }
}

// PLAN
// ================================================================================================

fn plan_function(
    ssa: &HLSSA,
    fid: FunctionId,
    reachable: &[BlockId],
    type_info: &TypeInfo,
) -> HashMap<ValueId, Vec<ValueId>> {
    let field = ssa.field();
    let func = ssa.get_function(fid);
    let fti = type_info.get_function(fid);
    let mut value_map: HashMap<ValueId, Vec<ValueId>> = HashMap::default();

    let plan_value = |value: ValueId, ty: &Type, map: &mut HashMap<ValueId, Vec<ValueId>>| {
        let count = limb_types(ty, field).len();
        let limbs = if count == 1 {
            vec![value]
        } else {
            (0..count).map(|_| ssa.fresh_value()).collect()
        };
        map.insert(value, limbs);
    };

    for bid in reachable {
        for (pid, ty) in func.get_block(*bid).get_parameters() {
            plan_value(*pid, ty, &mut value_map);
        }
    }

    for bid in reachable {
        for instr in func.get_block(*bid).get_instructions() {
            for result in instr.get_results() {
                let ty = fti.get_value_type(*result);
                plan_value(*result, ty, &mut value_map);
            }
        }
    }

    value_map
}

/// The limbs of a value; an unmapped value is its own single limb.
fn limbs_of(value_map: &HashMap<ValueId, Vec<ValueId>>, value: ValueId) -> Vec<ValueId> {
    value_map
        .get(&value)
        .cloned()
        .unwrap_or_else(|| vec![value])
}

/// The single limb of a value this pass does not represent as limbs.
fn single(value_map: &HashMap<ValueId, Vec<ValueId>>, value: ValueId) -> ValueId {
    let limbs = limbs_of(value_map, value);
    assert_eq!(
        limbs.len(),
        1,
        "ICE: wide_witness_ints expected a single-limb value, got {} limbs for v{}",
        limbs.len(),
        value.0
    );
    limbs[0]
}

fn flat_limbs(value_map: &HashMap<ValueId, Vec<ValueId>>, values: &[ValueId]) -> Vec<ValueId> {
    values
        .iter()
        .flat_map(|value| limbs_of(value_map, *value))
        .collect()
}

/// Pair two limb lists, refusing a pair that does not line up.
#[track_caller]
fn paired(left: Vec<ValueId>, right: Vec<ValueId>) -> impl Iterator<Item = (ValueId, ValueId)> {
    assert_eq!(
        left.len(),
        right.len(),
        "ICE: a limb-moving opcode met {} limbs on one side and {} on the other",
        left.len(),
        right.len()
    );
    left.into_iter().zip(right)
}

// REWRITE
// ================================================================================================

fn rewrite_function(
    ssa: &mut HLSSA,
    fid: FunctionId,
    reachable: &[BlockId],
    value_map: &HashMap<ValueId, Vec<ValueId>>,
    type_info: &TypeInfo,
    field: FieldConfig,
) {
    let mut function = ssa.take_function(fid);
    let fti = type_info.get_function(fid);

    let old_returns = function.take_returns();
    for ty in old_returns {
        for limb in limb_types(&ty, field) {
            function.add_return_type(limb);
        }
    }

    for bid in reachable {
        let block = function.get_block_mut(*bid);

        let old_params = block.take_parameters();
        let mut new_params = Vec::new();
        for (pid, ty) in old_params {
            let ids = limbs_of(value_map, pid);
            let types = limb_types(&ty, field);
            debug_assert_eq!(ids.len(), types.len());
            for (id, ty) in ids.into_iter().zip(types) {
                new_params.push((id, ty));
            }
        }
        block.put_parameters(new_params);

        let old_instructions = block.take_instructions();
        let mut new_instructions = Vec::with_capacity(old_instructions.len());
        for instr in &old_instructions {
            let location = instr.location().clone();
            let mut rewriter = Rewriter {
                ssa,
                value_map,
                types: fti,
                field,
                out: Vec::new(),
            };
            rewriter.lower(instr.as_ref());
            new_instructions.extend(
                rewriter
                    .out
                    .into_iter()
                    .map(|op| Located::new(op, location.clone())),
            );
        }
        block.put_instructions(new_instructions);

        let terminator = match block.take_terminator().unwrap() {
            Terminator::Jmp(dest, args) => Terminator::Jmp(dest, flat_limbs(value_map, &args)),
            Terminator::JmpIf(cond, t, f) => Terminator::JmpIf(single(value_map, cond), t, f),
            Terminator::Return(values) => Terminator::Return(flat_limbs(value_map, &values)),
        };
        block.set_terminator(terminator);
    }

    ssa.put_function(fid, function);
}

/// One instruction's worth of rewriting, plus the minting the gadgets need.
///
/// `ssa` is borrowed immutably because minting a value or interning a constant does not need more
/// than that: the function being rewritten is out of the SSA for the duration.
struct Rewriter<'a> {
    ssa: &'a HLSSA,
    value_map: &'a HashMap<ValueId, Vec<ValueId>>,
    types: &'a FunctionTypeInfo,
    field: FieldConfig,
    out: Vec<OpCode>,
}

impl Rewriter<'_> {
    fn limb_bits(&self) -> usize {
        witness_limb_bits(self.field)
    }

    fn limbs(&self, value: ValueId) -> Vec<ValueId> {
        limbs_of(self.value_map, value)
    }

    /// The limbs of an operand, splitting a **pure** one that the plan never saw.
    ///
    /// `check_widths` makes both operands of a non-shift the same width, and a mixed pure/witness
    /// pair is legal, so one witnessed operand is sufficient to require a witness lowering.
    fn operand_limbs(&mut self, value: ValueId, expected: usize) -> Vec<ValueId> {
        let mapped = self.limbs(value);
        if mapped.len() == expected {
            return mapped;
        }
        assert_eq!(
            mapped.len(),
            1,
            "ICE: an operand of {} limbs met one of {expected}",
            mapped.len()
        );

        let bits = int_width(self.types.get_value_type(value))
            .unwrap_or_else(|| panic!("ICE: a non-integer operand met a wide witnessed integer"));
        let widths = limb_widths(bits, self.limb_bits());
        assert_eq!(
            widths.len(),
            expected,
            "ICE: an int{bits} operand met a value of {expected} limbs"
        );

        widths
            .into_iter()
            .enumerate()
            .map(|(index, width)| {
                let shifted = self.shifted_down(value, bits, index * self.limb_bits());
                self.cast(shifted, CastTarget::Int(width))
            })
            .collect()
    }

    /// The limbs of one operation's two operands, which are the same length by construction.
    fn operand_pair(&mut self, lhs: ValueId, rhs: ValueId) -> Vec<(ValueId, ValueId)> {
        let expected = self.limbs(lhs).len().max(self.limbs(rhs).len());
        let lhs = self.operand_limbs(lhs, expected);
        let rhs = self.operand_limbs(rhs, expected);
        lhs.into_iter().zip(rhs).collect()
    }

    fn one(&self, value: ValueId) -> ValueId {
        single(self.value_map, value)
    }

    /// The declared width of a witnessed integer this pass represents as limbs.
    fn wide_width(&self, value: ValueId) -> Option<usize> {
        wide_witness_width(self.types.get_value_type(value), self.field)
    }

    fn fresh(&self) -> ValueId {
        self.ssa.fresh_value()
    }

    fn push(&mut self, op: OpCode) {
        self.out.push(op);
    }

    fn cast(&mut self, value: ValueId, target: CastTarget) -> ValueId {
        let result = self.fresh();
        self.push(OpCode::Cast {
            result,
            value,
            target,
        });
        result
    }

    fn bin(&mut self, kind: BinaryArithOpKind, lhs: ValueId, rhs: ValueId) -> ValueId {
        let result = self.fresh();
        self.push(OpCode::BinaryArithOp {
            kind,
            result,
            lhs,
            rhs,
        });
        result
    }

    /// `value >> offset` on the pure side, as a division rather than a shift.
    ///
    /// A shift by a constant is rewritten into a `BitRange` by `Simplifier`, which runs **after**
    /// this pass and therefore after the rule that lowers one — so the window would survive into
    /// codegen and be refused there. A division by the same power of two says the same thing and is
    /// the shape `lookup_spilling`'s own chunk extraction already takes.
    fn shifted_down(&mut self, value: ValueId, bits: usize, offset: usize) -> ValueId {
        if offset == 0 {
            return value;
        }
        let divisor = self.two_pow_const(bits, offset);
        self.bin(BinaryArithOpKind::UDiv, value, divisor)
    }

    /// A constant integer of `bits` raw bits carrying `2^exponent`, at any width.
    ///
    /// The composition and its bound are [`two_pow_pattern`]'s; this rewriter cannot reach
    /// `HLEmitter::two_pow_const` because it interns constants through the SSA rather than through
    /// an emitter.
    fn two_pow_const(&self, bits: usize, exponent: usize) -> ValueId {
        self.int_const(two_pow_pattern(bits, exponent))
    }

    fn write_witness(&mut self, value: ValueId) -> ValueId {
        let result = self.fresh();
        self.push(OpCode::WriteWitness {
            result: Some(result),
            value,
            pinned: false,
        });
        result
    }

    fn field_const(&self, value: Field) -> ValueId {
        self.ssa.add_const(Constant::Field(value))
    }

    fn int_const(&self, pattern: IntBits) -> ValueId {
        self.ssa.add_const(Constant::Int(pattern))
    }

    /// `2^(index * h)` as a field element, the place value of limb `index`.
    fn place_value(&self, index: usize) -> ValueId {
        self.field_const(self.field.two_pow(index * self.limb_bits()))
    }
}

// GADGETS
// ================================================================================================

impl Rewriter<'_> {
    /// A single witnessed value of `bits` wide, split into range-checked limbs that are constrained
    /// to recombine to it.
    ///
    /// The hints come from the pure shadow, which both compiled backends can compute at any width;
    /// the constraints are what ensures that they are a representation. As
    /// `bits <= multi_cell_int_bits`, the source has an element and the limb sum below the modulus
    /// reaches it without wrapping.
    fn decompose(&mut self, value: ValueId, bits: usize) -> Vec<ValueId> {
        let widths = limb_widths(bits, self.limb_bits());
        let pure = self.cast(value, CastTarget::ValueOf);

        let mut limbs = Vec::with_capacity(widths.len());
        let mut fields = Vec::with_capacity(widths.len());
        for (index, width) in widths.iter().enumerate() {
            let low = index * self.limb_bits();
            let hint = self.shifted_down(pure, bits, low);
            let narrowed = self.cast(hint, CastTarget::Int(*width));

            // A witness column written from the hint: the injection would keep the pure shadow as a
            // live operand all the way into R1CS, where a witness strip is an ICE because a hint is
            // not a constraint.
            let hint_field = self.cast(narrowed, CastTarget::Field);
            let written = self.write_witness(hint_field);
            let limb = self.cast(written, CastTarget::Int(*width));
            let limb_field = self.cast(limb, CastTarget::Field);
            self.push(OpCode::Rangecheck {
                value: limb_field,
                max_bits: *width,
            });
            limbs.push(limb);
            fields.push(limb_field);
        }

        let mut sum = fields[0];
        for (index, limb_field) in fields.iter().enumerate().skip(1) {
            let place = self.place_value(index);
            let scaled = self.bin(BinaryArithOpKind::UMul, *limb_field, place);
            sum = self.bin(BinaryArithOpKind::UAdd, sum, scaled);
        }

        let value_field = self.cast(value, CastTarget::Field);
        let difference = self.bin(BinaryArithOpKind::USub, value_field, sum);
        let zero = self.field_const(self.field.zero());
        let one = self.field_const(self.field.one());
        self.push(OpCode::Constrain {
            a: one,
            b: difference,
            c: zero,
        });

        limbs
    }

    /// The low `bits` of a limb list, as a single witnessed value of that width, the inverse of
    /// [`Self::decompose`].
    ///
    /// It needs no constraint of its own: a linear combination of values that are already pinned is
    /// itself pinned. `bits` is at most [`multi_cell_int_bits`], so the sum reaches the target's
    /// element without wrapping.
    fn recombine(&mut self, limbs: &[ValueId], widths: &[usize], bits: usize) -> ValueId {
        let target = limb_widths(bits, self.limb_bits());
        let mut sum = None;
        for (index, target_width) in target.iter().enumerate() {
            let mut limb = limbs[index];
            if *target_width < widths[index] {
                limb = self.truncate_limb(limb, *target_width);
            }

            let limb_field = self.cast(limb, CastTarget::Field);
            let scaled = if index == 0 {
                limb_field
            } else {
                let place = self.place_value(index);
                self.bin(BinaryArithOpKind::UMul, limb_field, place)
            };

            sum = Some(match sum {
                None => scaled,
                Some(acc) => self.bin(BinaryArithOpKind::UAdd, acc, scaled),
            });
        }

        let sum = sum.expect("an integer has at least one limb");
        self.cast(sum, CastTarget::Int(bits))
    }

    /// The low `width` bits of a limb, as a witnessed value of that width.
    fn truncate_limb(&mut self, limb: ValueId, width: usize) -> ValueId {
        let result = self.fresh();
        self.push(OpCode::BitRange {
            result,
            value: limb,
            offset: 0,
            width,
        });
        result
    }

    /// A witnessed zero of `width` bits, which is a constant and therefore pinned by being one.
    fn zero_limb(&mut self, width: usize) -> ValueId {
        let zero = self.int_const(IntBits::zero(width));
        self.cast(zero, CastTarget::WitnessOf)
    }

    /// The limbs of `value` read at `to_bits`, whatever the two widths are.
    fn relimb(&mut self, value: ValueId, from_bits: usize, to_bits: usize) -> Vec<ValueId> {
        let limb_bits = self.limb_bits();
        let source_widths = limb_widths(from_bits, limb_bits);
        let source = match self.wide_width(value) {
            Some(_) => self.limbs(value),
            None => self.decompose(value, from_bits),
        };

        limb_widths(to_bits, limb_bits)
            .into_iter()
            .enumerate()
            .map(|(index, width)| match source.get(index) {
                None => self.zero_limb(width),
                Some(limb) => match width.cmp(&source_widths[index]) {
                    std::cmp::Ordering::Equal => *limb,
                    std::cmp::Ordering::Greater => self.cast(*limb, CastTarget::Int(width)),
                    std::cmp::Ordering::Less => self.truncate_limb(*limb, width),
                },
            })
            .collect()
    }
}

// PER-INSTRUCTION REWRITING
// ================================================================================================

impl Rewriter<'_> {
    /// Rewrite one instruction into the limb-wise instructions that replace it.
    fn lower(&mut self, op: &OpCode) {
        let touches_wide = op
            .get_inputs()
            .chain(op.get_results())
            .any(|value| self.limbs(*value).len() > 1);
        if !touches_wide {
            self.push(op.clone());
            return;
        }

        match op {
            OpCode::Cast {
                result,
                value,
                target,
            } => self.lower_cast(*result, *value, target),

            OpCode::Cmp {
                kind,
                result,
                lhs,
                rhs,
            } => self.lower_compare(*kind, *result, *lhs, *rhs),

            OpCode::AssertCmp { kind, lhs, rhs } => self.lower_assert_compare(*kind, *lhs, *rhs),

            // Everything below moves limbs without reading them, so each is its narrow self once
            // per limb.
            OpCode::Select {
                result,
                cond,
                if_t,
                if_f,
            } => {
                let cond = self.one(*cond);
                let results = self.limbs(*result);
                let then = self.operand_limbs(*if_t, results.len());
                let otherwise = self.operand_limbs(*if_f, results.len());
                for ((result, if_t), if_f) in results.into_iter().zip(then).zip(otherwise) {
                    self.push(OpCode::Select {
                        result,
                        cond,
                        if_t,
                        if_f,
                    });
                }
            }

            OpCode::Alloc { result, value } => {
                for (result, value) in paired(self.limbs(*result), self.limbs(*value)) {
                    self.push(OpCode::Alloc { result, value });
                }
            }

            OpCode::Load { result, ptr } => {
                for (result, ptr) in paired(self.limbs(*result), self.limbs(*ptr)) {
                    self.push(OpCode::Load { result, ptr });
                }
            }

            OpCode::Store { ptr, value } => {
                for (ptr, value) in paired(self.limbs(*ptr), self.limbs(*value)) {
                    self.push(OpCode::Store { ptr, value });
                }
            }

            OpCode::WriteWitness {
                result,
                value,
                pinned,
            } => {
                let values = self.limbs(*value);
                let results = match result {
                    Some(result) => self.limbs(*result).into_iter().map(Some).collect(),
                    None => vec![None; values.len()],
                };
                for (result, value) in results.into_iter().zip(values) {
                    self.push(OpCode::WriteWitness {
                        result,
                        value,
                        pinned: *pinned,
                    });
                }
            }

            OpCode::FreshWitness {
                result,
                result_type,
            } => {
                let types = limb_types(result_type, self.field);
                for (result, result_type) in self.limbs(*result).into_iter().zip(types) {
                    self.push(OpCode::FreshWitness {
                        result,
                        result_type,
                    });
                }
            }

            OpCode::Call {
                results,
                function,
                args,
                unconstrained,
            } => self.push(OpCode::Call {
                results: flat_limbs(self.value_map, results),
                function: function.clone(),
                args: flat_limbs(self.value_map, args),
                unconstrained: *unconstrained,
            }),

            // A wide value reaching anything else is a shape this pass does not represent, which
            // should have been refused by width validation.
            other => panic!(
                "ICE: {other:?} reached the multi-cell representation with a wide witnessed operand; width validation should have refused the program"
            ),
        }
    }

    /// A cast, which is where a value enters and leaves the representation.
    fn lower_cast(&mut self, result: ValueId, value: ValueId, target: &CastTarget) {
        let source_type = self.types.get_value_type(value);
        let source_bits = int_width(source_type);

        match target {
            CastTarget::Int(to_bits) => {
                let from_bits = source_bits.unwrap_or_else(|| {
                    panic!(
                        "ICE: a width cast of a non-integer reached the multi-cell representation"
                    )
                });
                match self.wide_width(result) {
                    // Into the representation, or between two widths inside it.
                    Some(_) => {
                        let limbs = self.relimb(value, from_bits, *to_bits);
                        for (result, limb) in paired(self.limbs(result), limbs) {
                            self.push(OpCode::Cast {
                                result,
                                value: limb,
                                target: CastTarget::Nop,
                            });
                        }
                    }
                    // Out of it: the target is one element again, so the low limbs recombine.
                    None => {
                        let widths = limb_widths(from_bits, self.limb_bits());
                        let limbs = self.limbs(value);
                        let combined = self.recombine(&limbs, &widths, *to_bits);
                        self.push(OpCode::Cast {
                            result,
                            value: combined,
                            target: CastTarget::Nop,
                        });
                    }
                }
            }

            CastTarget::WitnessOf => {
                let bits = self.wide_width(result).expect(
                    "ICE: a witness injection reached the multi-cell representation without a wide result",
                );
                let widths = limb_widths(bits, self.limb_bits());
                let results = self.limbs(result);
                for (index, (result, width)) in results.into_iter().zip(&widths).enumerate() {
                    let low = index * self.limb_bits();
                    let hint = self.shifted_down(value, bits, low);
                    let narrowed = self.cast(hint, CastTarget::Int(*width));
                    self.push(OpCode::Cast {
                        result,
                        value: narrowed,
                        target: CastTarget::WitnessOf,
                    });
                    let limb_field = self.cast(result, CastTarget::Field);
                    self.push(OpCode::Rangecheck {
                        value: limb_field,
                        max_bits: *width,
                    });
                }
            }

            CastTarget::ValueOf => {
                let bits = self.wide_width(value).expect(
                    "ICE: a witness strip reached the multi-cell representation without a wide source",
                );
                let limbs = self.limbs(value);
                let mut sum = None;
                for (index, limb) in limbs.into_iter().enumerate() {
                    let pure = self.cast(limb, CastTarget::ValueOf);
                    let widened = self.cast(pure, CastTarget::Int(bits));
                    let placed = if index == 0 {
                        widened
                    } else {
                        let amount = self.int_const(IntBits::from_u128(
                            bits,
                            (index * self.limb_bits()) as u128,
                        ));
                        self.bin(BinaryArithOpKind::UShl, widened, amount)
                    };
                    sum = Some(match sum {
                        None => placed,
                        Some(acc) => self.bin(BinaryArithOpKind::Or, acc, placed),
                    });
                }
                self.push(OpCode::Cast {
                    result,
                    value: sum.expect("an integer has at least one limb"),
                    target: CastTarget::Nop,
                });
            }

            CastTarget::Nop => {
                for (result, limb) in paired(self.limbs(result), self.limbs(value)) {
                    self.push(OpCode::Cast {
                        result,
                        value: limb,
                        target: CastTarget::Nop,
                    });
                }
            }

            // A value at a width the field cannot carry has no element, so this cast is refused by
            // the language rule long before here.
            CastTarget::Field => panic!(
                "ICE: a value wider than the field carries injectively reached a cast to Field"
            ),

            CastTarget::Map(_) | CastTarget::ArrayToSlice => panic!(
                "ICE: a wide witnessed integer inside a sequence reached the multi-cell representation; wide array elements are not supported"
            ),
        }
    }

    /// Equality, which is the conjunction of the limbs' own.
    fn lower_compare(&mut self, kind: CmpKind, result: ValueId, lhs: ValueId, rhs: ValueId) {
        assert!(
            matches!(kind, CmpKind::Eq),
            "ICE: a {kind:?} of a wide witnessed integer reached the multi-cell representation; width validation should have refused the program"
        );

        let pairs = self.operand_pair(lhs, rhs);
        let mut all = None;
        for (lhs, rhs) in pairs {
            let equal = self.fresh();
            self.push(OpCode::Cmp {
                kind: CmpKind::Eq,
                result: equal,
                lhs,
                rhs,
            });
            all = Some(match all {
                None => equal,
                Some(acc) => self.bin(BinaryArithOpKind::And, acc, equal),
            });
        }
        self.push(OpCode::Cast {
            result,
            value: all.expect("an integer has at least one limb"),
            target: CastTarget::Nop,
        });
    }

    /// The assertion of one, which is one assertion per limb rather than a conjunction: the two
    /// mean the same thing and this costs no conjunction to build.
    fn lower_assert_compare(&mut self, kind: CmpKind, lhs: ValueId, rhs: ValueId) {
        assert!(
            matches!(kind, CmpKind::Eq),
            "ICE: a {kind:?} assertion of a wide witnessed integer reached the multi-cell representation; width validation should have refused the program"
        );

        let pairs = self.operand_pair(lhs, rhs);
        for (lhs, rhs) in pairs {
            self.push(OpCode::AssertCmp {
                kind: CmpKind::Eq,
                lhs,
                rhs,
            });
        }
    }
}

/// The declared width of an integer type, looking through a witness wrapper.
fn int_width(ty: &Type) -> Option<usize> {
    match &ty.strip_witness().expr {
        TypeExpr::Int(bits) => Some(*bits),
        _ => None,
    }
}

// UTILITIES
// ================================================================================================

/// The widest witnessed integer that is carried as a single field element.
///
/// Above it a value has no element to be carried in, which is what forces the limbs. It is
/// [`widest_injective_int_bits`] and therefore field-derived: 253 on bn254, 63 on goldilocks.
///
/// Units 9 to 13 lower this to the width their gadgets are built at. It is one constant, and
/// `the_representation_threshold_is_where_an_element_stops_being_injective` is what fails when it
/// moves without its reason moving with it.
pub fn multi_cell_int_bits(field: FieldConfig) -> usize {
    widest_injective_int_bits(field)
}

// TESTS
// ================================================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::{
        analysis::types::Types,
        passes::shared::limbs::narrow_int_bits,
        ssa::hlssa::builder::{HLEmitter, HLSSABuilder},
    };

    fn bn254() -> FieldConfig {
        FieldConfig::bn254()
    }

    /// The threshold is where a value stops having a field element, which is what forces limbs.
    ///
    /// It is deliberately **not** the narrower width the single-cell lowerings stop at: between the
    /// two a value is one element that no lowering will touch, which `width_validation` refuses.
    #[test]
    fn the_representation_threshold_is_where_an_element_stops_being_injective() {
        let field = bn254();

        assert_eq!(multi_cell_int_bits(field), widest_injective_int_bits(field));
        assert!(multi_cell_int_bits(field) > narrow_int_bits(field));
    }

    /// A limb's declared width is the width it is held to, and the top one carries the remainder.
    #[test]
    fn the_top_limb_carries_the_remainder() {
        assert_eq!(limb_widths(320, 64), vec![64, 64, 64, 64, 64]);
        assert_eq!(limb_widths(300, 64), vec![64, 64, 64, 64, 44]);
        assert_eq!(limb_widths(255, 64), vec![64, 64, 64, 63]);
        assert_eq!(limb_widths(64, 64), vec![64]);
    }

    /// Only a witnessed integer past the threshold becomes limbs.
    #[test]
    fn a_narrow_or_pure_integer_is_left_alone() {
        let field = bn254();
        let wide = multi_cell_int_bits(field) + 1;

        assert_eq!(limb_types(&Type::int(wide), field).len(), 1);
        assert_eq!(
            limb_types(
                &Type::witness_of(Type::int(multi_cell_int_bits(field))),
                field
            )
            .len(),
            1
        );
        assert_eq!(
            limb_types(&Type::witness_of(Type::int(wide)), field).len(),
            wide.div_ceil(witness_limb_bits(field))
        );
    }

    /// A reference to a wide value expands, because a reference is a place and every limb needs
    /// one; a sequence of them does not, because a lookup element is read as one field element.
    #[test]
    fn a_reference_expands_and_a_sequence_does_not() {
        let field = bn254();
        let element = Type::witness_of(Type::int(320));
        let expected = 320usize.div_ceil(witness_limb_bits(field));

        assert_eq!(limb_types(&element.ref_of(), field).len(), expected);
        assert_eq!(
            limb_types(&Type::witness_of(Type::int(64)).array_of(4), field).len(),
            1
        );
    }

    /// `main(wide) { jmp block_1(wide) }` — a wide value crossing a block boundary.
    fn program_passing_a_wide_value_across_a_block(bits: usize) -> HLSSA {
        let mut ssa = HLSSA::with_main("main".to_string());
        let main = ssa.get_unique_entrypoint_id();
        let mut builder = HLSSABuilder::new(&mut ssa);
        let carried = builder.fresh_value();
        let start = builder.fresh_value();
        builder.modify_function(main, |fb| {
            let entry = fb.function.get_entry_id();
            let next = {
                let mut editor = fb.block(entry);
                let (next, block) = editor.add_block();
                block.push_parameter(carried, Type::witness_of(Type::int(bits)));
                block.set_terminator(Terminator::Return(vec![]));
                next
            };
            let entry_type = Type::witness_of(Type::int(bits));
            fb.function
                .get_block_mut(entry)
                .push_parameter(start, entry_type);
            fb.block(entry)
                .set_terminator(Terminator::Jmp(next, vec![start]));
        });
        ssa
    }

    /// Every block parameter carrying a wide value becomes one parameter per limb, and the jump
    /// that feeds it carries one argument per limb.
    ///
    /// This is the case a memoised table could get silently wrong: the limbs of a value crossing a
    /// block boundary would be re-derived at the consumer from the pure shadow, which is a hint
    /// rather than a constraint, so producer and consumer would be unrelated witnesses.
    #[test]
    fn a_block_parameter_becomes_one_parameter_per_limb() {
        let bits = 320usize;
        let mut ssa = program_passing_a_wide_value_across_a_block(bits);

        let expected = limb_widths(bits, witness_limb_bits(bn254()));
        run_pass(&mut ssa);

        let main = ssa.get_unique_entrypoint_id();
        let function = ssa.get_function(main);
        let widths: Vec<usize> = function
            .get_block(BlockId(1))
            .get_parameters()
            .map(|(_, ty)| match &ty.strip_witness().expr {
                TypeExpr::Int(bits) => *bits,
                other => panic!("a limb is an integer, got {other:?}"),
            })
            .collect();

        assert_eq!(widths, expected);

        // And the jump that feeds it carries one argument per limb: a parameter list and an
        // argument list that disagree is the failure this flattening has to avoid, and it is one an
        // arity check downstream would report a long way from its cause.
        let Some(Terminator::Jmp(_, args)) = function.get_block(BlockId(0)).get_terminator() else {
            panic!("the entry block jumps to the block carrying the value");
        };
        assert_eq!(args.len(), expected.len());
    }

    /// `main(a: WitnessOf(int(from))) -> WitnessOf(int(to)) { a as int(to) }`.
    fn program_widening(from: usize, to: usize) -> HLSSA {
        let mut ssa = HLSSA::with_main("main".to_string());
        let main = ssa.get_unique_entrypoint_id();
        let value = ssa.fresh_value();
        let result = ssa.fresh_value();
        let mut builder = HLSSABuilder::new(&mut ssa);
        builder.modify_function(main, |fb| {
            fb.function.add_return_type(Type::witness_of(Type::int(to)));
            let entry = fb.function.get_entry_id();
            fb.function
                .get_block_mut(entry)
                .push_parameter(value, Type::witness_of(Type::int(from)));
            let mut block = fb.test_block(entry);
            block.emit(OpCode::Cast {
                result,
                value,
                target: CastTarget::Int(to),
            });
            block.terminate_return(vec![result]);
        });
        ssa
    }

    /// Every emitted opcode of one function, in order.
    fn emitted(ssa: &HLSSA) -> Vec<OpCode> {
        let main = ssa.get_unique_entrypoint_id();
        let function = ssa.get_function(main);
        function
            .get_block(function.get_entry_id())
            .get_instructions()
            .cloned()
            .collect()
    }

    /// A value entering the representation is witnessed, bounded and tied back to where it came
    /// from: one of each, per limb, and one tie for the lot.
    ///
    /// A structural audit rather than a behavioral one, because neither half is visible to an
    /// honest witness. Drop the range check and a prover can move value between limbs while the
    /// sum still holds; drop the reconstruction and the limbs stop being this value's at all. In
    /// that second case the limb loses its only consumer, so it is eliminated along with its
    /// witness column and a column-by-column perturbation has nothing left to find.
    #[test]
    fn a_value_entering_the_representation_is_witnessed_bounded_and_tied_back() {
        // A source that is not a whole number of limbs, so the top limb's own width is the thing it
        // is bounded at rather than a full limb's.
        let (from, to) = (200usize, 320usize);
        let mut ssa = program_widening(from, to);
        run_pass(&mut ssa);

        let ops = emitted(&ssa);
        let limbs = from.div_ceil(witness_limb_bits(bn254()));
        assert!(
            limbs > 1,
            "the source has to span more than one limb to be a test"
        );

        let written = ops
            .iter()
            .filter(|op| matches!(op, OpCode::WriteWitness { .. }))
            .count();
        let bounded: Vec<usize> = ops
            .iter()
            .filter_map(|op| match op {
                OpCode::Rangecheck { max_bits, .. } => Some(*max_bits),
                _ => None,
            })
            .collect();
        let tied = ops
            .iter()
            .filter(|op| matches!(op, OpCode::Constrain { .. }))
            .count();

        assert_eq!(written, limbs, "one witness column per limb of the source");
        assert_eq!(
            bounded,
            limb_widths(from, witness_limb_bits(bn254())),
            "each limb is bounded at its own declared width"
        );
        assert_eq!(
            tied, 1,
            "one reconstruction, tying the limbs to their source"
        );
    }

    /// The limbs a widening does not reach are the zero constant, which is pinned by being one.
    ///
    /// So a widening mints witnesses for the **source's** limbs and no more: padding a 320-bit
    /// target from a 128-bit source costs three constants, not three columns.
    #[test]
    fn the_limbs_above_the_source_cost_no_witness() {
        let mut narrow_source = program_widening(128, 320);
        let mut wider_source = program_widening(192, 320);
        run_pass(&mut narrow_source);
        run_pass(&mut wider_source);

        let count = |ssa: &HLSSA| {
            emitted(ssa)
                .iter()
                .filter(|op| matches!(op, OpCode::WriteWitness { .. }))
                .count()
        };
        assert_eq!(count(&narrow_source), 2);
        assert_eq!(count(&wider_source), 3);
    }

    fn run_pass(ssa: &mut HLSSA) {
        let flow = FlowAnalysis::run(ssa);
        let types = Types::new().run(ssa, &flow);
        let mut store = AnalysisStore::new();
        store.insert_with_deps(flow, vec![]);
        store.insert_with_deps(types, vec![]);
        WideWitnessInts::new().run(ssa, &store);
    }
}
