//! The multi-cell representation: a witnessed integer too wide for one field element becomes a
//! group of limbs.
//!
//! A `WitnessOf(Int(N))` with `N > multi_cell_int_bits` becomes `k = ceil(N/h)` values, stored in
//! little-endian order, where `h` is [`witness_limb_bits`]. Limbs `0..k-1` are `WitnessOf(Int(h))`
//! and the top one is `WitnessOf(Int(N - (k-1)h))`, so a limb's declared width remains correct.
//!
//! A **pure** `Int(N)` **scalar** is untouched as it exists in the hint domain where both backends
//! can compute on it at arbitrary width.
//!
//! A **sequence** of wide elements is transposed instead: `Array(E, n)` becomes `k` parallel
//! `Array(E_j, n)`, each holding one limb position of every element, and a read or a write is that
//! operation once per limb sequence at the caller's own index. This is [`element_limb_types`], and
//! it splits a **pure** wide element too. A sequence has to make one field element per entry for
//! the lookup tape to address it; transposing makes every entry a limb the tape already knows how
//! to read. Putting such an element back together is [`Rewriter::recombine_pure`], which is integer
//! arithmetic at the value's own width and so costs interpreter work rather than constraints.
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
            BinaryArithOpKind, Blob, CastTarget, CmpKind, Constant, HLSSA, LookupTarget, OpCode,
            Type, TypeExpr, builder::two_pow_pattern,
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
                "ICE: a global carries a wide integer, as a value or as a sequence element, and \
                 neither has a slot layout here"
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
        // A sequence of wide elements is transposed: one sequence per limb position, each holding
        // the corresponding limb of every element. The element type of each array is a single limb,
        // which is at most one host cell wide, so the lookup tape reads it with the tag it already
        // has for a cell and needs no notion of a wide element at all.
        //
        // The alternative (one sequence of `n * k` limbs) would make the index `i * k + j` and so
        // put arithmetic between the caller's index and the tape's. Here every limb sequence is
        // indexed by the caller's own index.
        TypeExpr::Array(inner, count) => element_limb_types(inner, field)
            .into_iter()
            .map(|leaf| leaf.array_of(*count))
            .collect(),
        TypeExpr::Slice(inner) => element_limb_types(inner, field)
            .into_iter()
            .map(Type::slice_of)
            .collect(),
        TypeExpr::Ref(inner) => limb_types(inner, field)
            .into_iter()
            .map(|leaf| leaf.ref_of())
            .collect(),
        _ => vec![ty.clone()],
    }
}

/// A sequence element splits whether or not it is witnessed.
///
/// A sequence is stored one sequence per limb, so a **pure** wide element splits too — otherwise a
/// purely-constant wide sequence stays whole, its entries have to become one field element each,
/// and the widest it can hold is what the modulus carries. Splitting it costs pure arithmetic to
/// put an element back together, which is interpreter work rather than constraints.
fn element_limb_types(ty: &Type, field: FieldConfig) -> Vec<Type> {
    match &ty.expr {
        TypeExpr::Int(bits) if *bits > multi_cell_int_bits(field) => {
            limb_widths(*bits, witness_limb_bits(field))
                .into_iter()
                .map(Type::int)
                .collect()
        }
        _ => limb_types(ty, field),
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

    // A wide value's expansion has to reach the values that merely **carry** it, which are not
    // identified by their types.
    //
    // A witness-indexed array read moves its element through a field element: the lowering reads
    // the hint, strips it to the pure domain, casts that to `Field`, writes it as a witness, pins
    // it with `constrain_lookup`, and casts back to the element's own width. Only the first and
    // last of those are typed as the wide integer. The middle ones are a pure `int`, a `Field` and
    // a `WitnessOf(Field)` — each of which `limb_types` calls a single value, because each of them
    // **is** a single value everywhere else in the program.
    //
    // So the expansion is propagated along the instructions that carry a value without reading it,
    // to a fixed point. Two facts make that sound rather than a heuristic: a value the modulus
    // cannot carry has no single field element **in either domain**, so its field image is one
    // element per limb whether it was stripped or not; and every step below preserves the value.
    let expansion = |map: &HashMap<ValueId, Vec<ValueId>>, value: ValueId| -> usize {
        map.get(&value).map_or(1, Vec::len)
    };
    loop {
        let mut grew = false;
        for bid in reachable {
            for instr in func.get_block(*bid).get_instructions() {
                let (result, count) = match instr {
                    // The field image of a wide integer, pure or witnessed.
                    OpCode::Cast {
                        result,
                        value,
                        target: CastTarget::Field,
                    } => {
                        let Some(bits) = int_width(fti.get_value_type(*value))
                            .filter(|bits| *bits > multi_cell_int_bits(field))
                        else {
                            continue;
                        };
                        (*result, limb_widths(bits, witness_limb_bits(field)).len())
                    }
                    // Not converted at all, so whatever the source stands for the result stands for
                    // too.
                    //
                    // `WitnessOf` and `ValueOf` are deliberately **not** here: they are where the
                    // representation is entered and left, and the rewriter decomposes and
                    // recombines at them explicitly. A strip in particular yields one pure wide
                    // value, because the pure lane holds such a value whole.
                    OpCode::Cast {
                        result,
                        value,
                        target: CastTarget::Nop,
                    } => (*result, expansion(&value_map, *value)),
                    // Written to a witness column, one per limb.
                    OpCode::WriteWitness {
                        result: Some(result),
                        value,
                        ..
                    } => (*result, expansion(&value_map, *value)),
                    // Read out of a transposed sequence, which yields its limbs.
                    //
                    // We keep the limbs to avoid doing the same work twice, so reassembly only
                    // happens where the whole value is required.
                    OpCode::ArrayGet { result, array, .. } => {
                        (*result, limb_types(fti.get_value_type(*array), field).len())
                    }
                    _ => continue,
                };
                if count <= 1 || expansion(&value_map, result) == count {
                    continue;
                }
                let limbs = (0..count).map(|_| ssa.fresh_value()).collect();
                value_map.insert(result, limbs);
                grew = true;
            }
        }
        if !grew {
            break;
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

/// The limb types of a sequence element, refused if they do not line up with the sequences.
///
/// A constructor names its element type separately from its result, so the two are computed from
/// different places and a `zip` between them builds fewer limb sequences than the plan minted
/// values for whenever they disagree. That failure is silent (the missing sequences are simply
/// never emitted) so the two are checked against each other here instead.
#[track_caller]
fn element_types(elem_type: &Type, field: FieldConfig, expected: usize) -> Vec<Type> {
    let types = element_limb_types(elem_type, field);
    assert_eq!(
        types.len(),
        expected,
        "ICE: a sequence of {expected} limbs was built from an element of {} limbs",
        types.len()
    );
    types
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
            .unwrap_or_else(|| ice!("a non-integer operand met a wide witnessed integer"));
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

    /// A **pure** wide value injected into the representation as witnessed, range-checked limbs.
    ///
    /// The counterpart of [`Self::decompose`] above the width the field carries. There is no
    /// reconstruction constraint here as the source is a hint rather than a constrained value, so
    /// there is no prior value to tie the limbs back to. The limbs **are** the value's definition,
    /// and each one is bounded by an individual range check.
    ///
    /// [`Self::decompose`] cannot be used here: it constrains the limbs against the source's field
    /// element, which a value wider than the modulus does not have. The limbs are written into
    /// `results`, so that entering the representation costs only the injection.
    fn inject(&mut self, value: ValueId, bits: usize, results: &[ValueId]) {
        let widths = limb_widths(bits, self.limb_bits());
        assert_eq!(
            widths.len(),
            results.len(),
            "ICE: an int{bits} is {} limbs, not {}",
            widths.len(),
            results.len()
        );
        for (index, (result, width)) in results.iter().zip(&widths).enumerate() {
            let low = index * self.limb_bits();
            let hint = self.shifted_down(value, bits, low);
            let narrowed = self.cast(hint, CastTarget::Int(*width));
            self.push(OpCode::Cast {
                result: *result,
                value: narrowed,
                target: CastTarget::WitnessOf,
            });
            let limb_field = self.cast(*result, CastTarget::Field);
            self.push(OpCode::Rangecheck {
                value: limb_field,
                max_bits: *width,
            });
        }
    }

    /// The low `bits` of a limb list, as a single witnessed value of that width, the inverse of
    /// [`Self::decompose`].
    ///
    /// It needs no constraint of its own: a linear combination of values that are already pinned is
    /// itself pinned.
    ///
    /// `bits` has to be at most [`multi_cell_int_bits`], and the assertion is the whole reason this
    /// is sound: the sum is formed in **one field element**, so a target the modulus cannot carry
    /// would wrap silently. [`Self::pure_limbs_at`] is what a wider target takes instead.
    fn recombine(&mut self, limbs: &[ValueId], widths: &[usize], bits: usize) -> ValueId {
        assert!(
            bits <= multi_cell_int_bits(self.field),
            "ICE: an int{bits} recombination is summed in one field element, which cannot carry it"
        );
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

    /// Pure limbs shifted into place and or-ed together, as one pure value of `bits`.
    ///
    /// The **pure** counterpart of [`Self::recombine`], and the distinction is not cosmetic: that
    /// one sums field elements, so it wraps once the value passes the modulus. This one is integer
    /// arithmetic at the value's own width, which both compiled backends have at every width, and
    /// which costs interpreter work rather than constraints. It is what lets a sequence hold
    /// elements the field cannot carry.
    fn recombine_pure(&mut self, limbs: &[ValueId], bits: usize) -> ValueId {
        let mut sum = None;
        for (index, limb) in limbs.iter().enumerate() {
            let widened = self.cast(*limb, CastTarget::Int(bits));
            let placed = if index == 0 {
                widened
            } else {
                let amount =
                    self.int_const(IntBits::from_u128(bits, (index * self.limb_bits()) as u128));
                self.bin(BinaryArithOpKind::UShl, widened, amount)
            };
            sum = Some(match sum {
                None => placed,
                Some(acc) => self.bin(BinaryArithOpKind::Or, acc, placed),
            });
        }
        sum.expect("an integer has at least one limb")
    }

    /// The low `width` bits of a limb, as a value **of that width**.
    ///
    /// A narrowing cast rather than a `BitRange`: `BitRange` keeps its source's type, so a window
    /// cut out of an `int64` limb is still an `int64` carrying `width` bits. That is right for a
    /// window but wrong for a limb as the caller is building a limb list whose widths are the
    /// target's, and a limb whose declared width disagrees with its position meets an operand split
    /// at the true width and the two cannot be paired. The cast lowers to the same mask, witness
    /// and range check.
    fn truncate_limb(&mut self, limb: ValueId, width: usize) -> ValueId {
        self.cast(limb, CastTarget::Int(width))
    }

    /// A witnessed zero of `width` bits, which is a constant and therefore pinned by being one.
    fn zero_limb(&mut self, width: usize) -> ValueId {
        let zero = self.int_const(IntBits::zero(width));
        self.cast(zero, CastTarget::WitnessOf)
    }

    /// The limbs of a wide value as **pure** values, cut to the shape a `to_bits` target needs.
    ///
    /// What a narrowing out of the representation takes when the target is wider than the field
    /// carries, where [`Self::recombine`] cannot serve: that one sums field elements, and a sum
    /// reaching `2^to_bits` wraps once the target passes the modulus. Reading these back is
    /// [`Self::recombine_pure`], which is integer arithmetic at the target's own width.
    fn pure_limbs_at(&mut self, value: ValueId, from_bits: usize, to_bits: usize) -> Vec<ValueId> {
        assert!(
            to_bits <= from_bits,
            "ICE: an int{from_bits} held as limbs was widened to an int{to_bits} on the way out of \
             the representation"
        );
        let witnessed = self.wide_width(value).is_some();
        let source_widths = limb_widths(from_bits, self.limb_bits());
        let limbs = self.limbs(value);

        limb_widths(to_bits, self.limb_bits())
            .into_iter()
            .enumerate()
            .map(|(index, width)| {
                let limb = limbs[index];
                let pure = if witnessed {
                    self.cast(limb, CastTarget::ValueOf)
                } else {
                    limb
                };
                if width < source_widths[index] {
                    self.cast(pure, CastTarget::Int(width))
                } else {
                    pure
                }
            })
            .collect()
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

            // The transpose. An element-major sequence of `k`-limb values becomes `k` limb-major
            // sequences, each holding one limb position of every element.
            //
            // A pure element beside a witnessed one arrives as a single value which `operand_limbs`
            // splits: the same mixed-domain case a wide comparison meets.
            OpCode::MkSeq {
                result,
                elems,
                seq_type,
                elem_type,
            } => {
                let results = self.limbs(*result);
                let count = results.len();
                let elem_types = element_types(elem_type, self.field, count);
                let per_element: Vec<Vec<ValueId>> = elems
                    .iter()
                    .map(|elem| self.operand_limbs(*elem, count))
                    .collect();
                for (index, (result, elem_type)) in results.into_iter().zip(elem_types).enumerate()
                {
                    let elems = per_element.iter().map(|limbs| limbs[index]).collect();
                    self.push(OpCode::MkSeq {
                        result,
                        elems,
                        seq_type: *seq_type,
                        elem_type,
                    });
                }
            }

            // The witness-indexed read, which is the lookup the array lowering built at
            // `driver.rs:619` — before this pass, so the index and the flag are already narrow and
            // already pinned. One lookup per limb sequence against that same index and flag, so
            // each limb is selected by the same `sum(hit) == 1` argument rather than a new one.
            OpCode::Lookup {
                target: LookupTarget::Array(array),
                args,
                flag,
            }
            | OpCode::DLookup {
                target: LookupTarget::Array(array),
                args,
                flag,
            } => {
                assert_eq!(
                    args.len(),
                    2,
                    "ICE: an array lookup takes an index and a result"
                );
                let dynamic = matches!(op, OpCode::DLookup { .. });
                let index = self.one(args[0]);
                let flag = self.one(*flag);
                let results = self.limbs(args[1]);
                let arrays = self.limbs(*array);
                // `paired` rather than a `zip`: a limb short here is a limb the tape never pins,
                // which an honest witness still satisfies. It is the one mismatch in this pass
                // that would be unsound rather than merely wrong.
                for (result, array) in paired(results, arrays) {
                    let target = LookupTarget::Array(array);
                    let args = vec![index, result];
                    self.push(if dynamic {
                        OpCode::DLookup { target, args, flag }
                    } else {
                        OpCode::Lookup { target, args, flag }
                    });
                }
            }

            // A blob-backed sequence: one blob per limb, holding that limb of every element.
            //
            // The elements are constants, so the split is arithmetic on the patterns rather than
            // emitted code, which is what lets a constant sequence carry elements the field
            // cannot hold where reading one whole entry as a field element could not.
            OpCode::MkSeqOfBlob {
                result,
                element_type,
                blob,
            } => {
                let results = self.limbs(*result);
                let Some(Constant::Blob(blob)) = self.ssa.get_const(*blob).map(|c| (*c).clone())
                else {
                    ice!("a blob-backed sequence without a blob constant")
                };
                let bits = int_width(element_type)
                    .unwrap_or_else(|| ice!("a wide blob sequence of {element_type}"));
                let widths = limb_widths(bits, self.limb_bits());
                let elem_types = element_types(element_type, self.field, results.len());
                for (index, ((result, width), element_type)) in
                    results.into_iter().zip(&widths).zip(elem_types).enumerate()
                {
                    let low = index * self.limb_bits();
                    let elements = blob
                        .elements
                        .iter()
                        .map(|element| {
                            let Constant::Int(pattern) = element else {
                                ice!("a wide blob element that is not an integer")
                            };
                            Constant::Int(pattern.bit_range(low, *width))
                        })
                        .collect();
                    let limb_blob = self
                        .ssa
                        .add_const(Constant::Blob(Blob::new(element_type.clone(), elements)));
                    self.push(OpCode::MkSeqOfBlob {
                        result,
                        element_type,
                        blob: limb_blob,
                    });
                }
            }

            OpCode::MkRepeated {
                result,
                element,
                seq_type,
                count,
                elem_type,
            } => {
                let results = self.limbs(*result);
                let elem_types = element_types(elem_type, self.field, results.len());
                let elements = self.operand_limbs(*element, results.len());
                for ((result, element), elem_type) in
                    results.into_iter().zip(elements).zip(elem_types)
                {
                    self.push(OpCode::MkRepeated {
                        result,
                        element,
                        seq_type: *seq_type,
                        count: *count,
                        elem_type,
                    });
                }
            }

            // One read per limb sequence, at the caller's own index. The index is narrow and is
            // shared rather than re-derived, so a witnessed one is looked up once per limb against
            // the same value the array lowering pinned.
            OpCode::ArrayGet {
                result,
                array,
                index,
            } => {
                let index = self.one(*index);
                for (result, array) in paired(self.limbs(*result), self.limbs(*array)) {
                    self.push(OpCode::ArrayGet {
                        result,
                        array,
                        index,
                    });
                }
            }

            OpCode::ArraySet {
                result,
                array,
                index,
                value,
            } => {
                let index = self.one(*index);
                let results = self.limbs(*result);
                let arrays = self.limbs(*array);
                let values = self.operand_limbs(*value, results.len());
                for ((result, array), value) in paired(results, arrays).zip(values) {
                    self.push(OpCode::ArraySet {
                        result,
                        array,
                        index,
                        value,
                    });
                }
            }

            // The slice family that survives this far. Every limb sequence is the same length,
            // because every operation that changes one is applied identically to all `k` of them —
            // which is what lets the length be read off any single limb, and is the same reason the
            // index is shared rather than re-derived.
            //
            // `SlicePop`, `SliceInsert` and `SliceRemove` have no arm because
            // `InstructionLowering::slice_ops` replaces each of them with an assert, a get and a
            // copy loop before this pass runs. A wide one therefore arrives here as the `ArrayGet`,
            // `ArraySet` and `SliceLen` this pass already handles.
            OpCode::SliceLen { result, slice } => {
                let slice = self.limbs(*slice);
                self.push(OpCode::SliceLen {
                    result: self.one(*result),
                    slice: slice[0],
                });
            }

            OpCode::SlicePush {
                dir,
                result,
                slice,
                values,
            } => {
                let results = self.limbs(*result);
                let slices = self.limbs(*slice);
                let per_value: Vec<Vec<ValueId>> = values
                    .iter()
                    .map(|value| self.operand_limbs(*value, results.len()))
                    .collect();
                for (index, (result, slice)) in paired(results, slices).enumerate() {
                    let values = per_value.iter().map(|limbs| limbs[index]).collect();
                    self.push(OpCode::SlicePush {
                        dir: *dir,
                        result,
                        slice,
                        values,
                    });
                }
            }

            // The bitwise three, which are the one arithmetic family with **no cross-limb
            // interaction**: bit `i` of the answer depends on bit `i` of each operand and nothing
            // else, so a limb of the answer is the same operation on the matching pair of operand
            // limbs. There is no carry to thread and no reconstruction to re-establish — each limb
            // is already held to its own width, and an operation that cannot set a bit the operands
            // did not have between them cannot break that.
            OpCode::BinaryArithOp {
                kind:
                    kind @ (BinaryArithOpKind::And | BinaryArithOpKind::Or | BinaryArithOpKind::Xor),
                result,
                lhs,
                rhs,
            } => {
                let results = self.limbs(*result);
                let pairs = self.operand_pair(*lhs, *rhs);
                assert_eq!(
                    results.len(),
                    pairs.len(),
                    "ICE: a bitwise result of {} limbs met operands of {}",
                    results.len(),
                    pairs.len()
                );
                for (result, (lhs, rhs)) in results.into_iter().zip(pairs) {
                    self.push(OpCode::BinaryArithOp {
                        kind: *kind,
                        result,
                        lhs,
                        rhs,
                    });
                }
            }

            // The complement, which is the unary member of the same family and limb-wise for the
            // same reason: bit `i` of the answer depends on bit `i` of the operand alone. Each limb
            // is complemented at **its own** width, so the top one — which may be narrower than a
            // full limb — does not acquire bits the value's width does not have.
            OpCode::Not { result, value } => {
                let results = self.limbs(*result);
                let values = self.operand_limbs(*value, results.len());
                for (result, value) in paired(results, values) {
                    self.push(OpCode::Not { result, value });
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
                let results: Vec<Option<ValueId>> = match result {
                    Some(result) => self.limbs(*result).into_iter().map(Some).collect(),
                    None => vec![None; values.len()],
                };
                assert_eq!(
                    results.len(),
                    values.len(),
                    "ICE: {} witness columns were written from {} limbs",
                    results.len(),
                    values.len()
                );
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
                let results = self.limbs(*result);
                assert_eq!(
                    results.len(),
                    types.len(),
                    "ICE: {} fresh witnesses were minted for a type of {} limbs",
                    results.len(),
                    types.len()
                );
                for (result, result_type) in results.into_iter().zip(types) {
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

            // A wide value reaching anything else is a shape this pass does not represent. For a
            // witnessed operand `width_validation` is what refuses it; a **pure** one has no width
            // rule to refuse it and reaches here only by being read out of a transposed sequence,
            // which no arm above hands to anything but another limb-mover.
            other => ice!(
                "{other:?} reached the multi-cell representation with a wide operand, which is a shape it does not represent"
            ),
        }
    }

    /// A cast, which is where a value enters and leaves the representation.
    fn lower_cast(&mut self, result: ValueId, value: ValueId, target: &CastTarget) {
        let source_type = self.types.get_value_type(value);
        let source_bits = int_width(source_type);

        match target {
            CastTarget::Int(to_bits) => {
                // The other side of the field boundary: a wide value read back out of the field
                // elements its limbs were pinned as. Each limb returns at its own width, so there
                // is nothing to recombine and no place value to mint.
                if source_bits.is_none() {
                    let limbs = self.limbs(value);
                    let results = self.limbs(result);
                    let widths = limb_widths(*to_bits, self.limb_bits());
                    assert_eq!(
                        limbs.len(),
                        results.len(),
                        "ICE: a non-integer of {} limbs was read back as an int{to_bits} of {}",
                        limbs.len(),
                        results.len()
                    );
                    assert_eq!(
                        widths.len(),
                        results.len(),
                        "ICE: an int{to_bits} is {} limbs, not {}",
                        widths.len(),
                        results.len()
                    );
                    for ((result, limb), width) in results.into_iter().zip(limbs).zip(widths) {
                        self.push(OpCode::Cast {
                            result,
                            value: limb,
                            target: CastTarget::Int(width),
                        });
                    }
                    return;
                }
                let from_bits = source_bits.expect("a non-integer source returned above");
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
                    // Out of it. A target the field carries is one element again, so the low
                    // limbs recombine there and the linear combination carries the pinning with
                    // them. A target it cannot carry has no element to be summed into, so the
                    // limbs go back together as integer arithmetic at the target's own width —
                    // which is what the witness strip does, and for the same reason.
                    None => {
                        let combined = if *to_bits > multi_cell_int_bits(self.field) {
                            let limbs = self.pure_limbs_at(value, from_bits, *to_bits);
                            self.recombine_pure(&limbs, *to_bits)
                        } else {
                            let widths = limb_widths(from_bits, self.limb_bits());
                            let limbs = self.limbs(value);
                            self.recombine(&limbs, &widths, *to_bits)
                        };
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
                let results = self.limbs(result);
                self.inject(value, bits, &results);
            }

            CastTarget::ValueOf => {
                let bits = self.wide_width(value).expect(
                    "ICE: a witness strip reached the multi-cell representation without a wide source",
                );
                let stripped: Vec<ValueId> = self
                    .limbs(value)
                    .into_iter()
                    .map(|limb| self.cast(limb, CastTarget::ValueOf))
                    .collect();
                let sum = self.recombine_pure(&stripped, bits);
                self.push(OpCode::Cast {
                    result,
                    value: sum,
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

            // A wide value's field image is one element per limb, which the array lowering's lookup
            // chain moves through a field. The limbs are already held to their own widths, so each
            // is an element.
            CastTarget::Field => {
                let results = self.limbs(result);
                let limbs = self.operand_limbs(value, results.len());
                for (result, limb) in results.into_iter().zip(limbs) {
                    self.push(OpCode::Cast {
                        result,
                        value: limb,
                        target: CastTarget::Field,
                    });
                }
            }

            // A whole sequence entering the representation: `k` sequences out, each holding one
            // limb position of every element.
            //
            // Both sides are transposed already, so this is a mapped cast per limb sequence and
            // nothing here reads an element. Every limb sequence is narrow, so `LowerMapCasts`
            // expands each into the ordinary per-element witness injection later in the pipeline.
            CastTarget::Map(inner) => {
                for (result, value) in paired(self.limbs(result), self.limbs(value)) {
                    self.push(OpCode::Cast {
                        result,
                        value,
                        target: CastTarget::Map(inner.clone()),
                    });
                }
            }

            CastTarget::ArrayToSlice => {
                for (result, value) in paired(self.limbs(result), self.limbs(value)) {
                    self.push(OpCode::Cast {
                        result,
                        value,
                        target: CastTarget::ArrayToSlice,
                    });
                }
            }
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
    use mavros_int_semantics::int_bits::HOST_LIMB_BITS;

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

    /// A reference and a sequence both expand, and a sequence expands whether or not its element
    /// is witnessed.
    ///
    /// A reference is a place and every limb needs one. A sequence is transposed, so the count is
    /// the element's limbs rather than the sequence's length. A **pure** wide element counts too,
    /// which is the one place [`element_limb_types`] parts company with [`limb_types`]. A narrow
    /// element does not expand at any of it, which is what keeps a `[u128; n]` on bn254 off this
    /// path entirely.
    #[test]
    fn a_reference_and_a_sequence_both_expand_by_the_element() {
        let field = bn254();
        let wide = multi_cell_int_bits(field) + 1;
        let expected = wide.div_ceil(witness_limb_bits(field));
        assert!(expected > 1, "the width has to span limbs to be a test");

        assert_eq!(
            limb_types(&Type::witness_of(Type::int(wide)).ref_of(), field).len(),
            expected
        );
        assert_eq!(
            limb_types(&Type::witness_of(Type::int(wide)).array_of(4), field).len(),
            expected
        );
        assert_eq!(
            limb_types(&Type::int(wide).array_of(4), field).len(),
            expected,
            "a pure wide element is transposed too, or a constant sequence could not be read"
        );
        assert_eq!(
            limb_types(&Type::slice_of(Type::int(wide)), field).len(),
            expected
        );

        // A narrow element, at the widest width the double lane holds, stays one sequence.
        assert_eq!(
            limb_types(
                &Type::witness_of(Type::int(2 * HOST_LIMB_BITS)).array_of(4),
                field
            )
            .len(),
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

    /// `main(value: int(bits)) -> WitnessOf<int(bits)> { value as witness }`, a **pure** wide value
    /// injected into the representation.
    fn program_injecting(bits: usize) -> HLSSA {
        let mut ssa = HLSSA::with_main("main".to_string());
        let main = ssa.get_unique_entrypoint_id();
        let value = ssa.fresh_value();
        let result = ssa.fresh_value();
        let mut builder = HLSSABuilder::new(&mut ssa);
        builder.modify_function(main, |fb| {
            fb.function
                .add_return_type(Type::witness_of(Type::int(bits)));
            let entry = fb.function.get_entry_id();
            fb.function
                .get_block_mut(entry)
                .push_parameter(value, Type::int(bits));
            let mut block = fb.test_block(entry);
            block.emit(OpCode::Cast {
                result,
                value,
                target: CastTarget::WitnessOf,
            });
            block.terminate_return(vec![result]);
        });
        ssa
    }

    /// A **pure** value injected into the representation is bounded, and that bound is all there is.
    ///
    /// The counterpart of the test above. [`Rewriter::decompose`] ties its limbs back to a value
    /// that already existed, so a limb out of range breaks the reconstruction. [`Rewriter::inject`]
    /// has no prior value to tie back to so the per-limb range check is the **only** thing that
    /// bounds them. `2^h` being invertible mod `p` means an unbounded limb lets a prover solve for
    /// any value at all.
    ///
    /// **No honest witness can see this**, so we check it here.
    #[test]
    fn a_pure_value_injected_into_the_representation_is_bounded_per_limb() {
        // Above the representation threshold, so the value is limbs at all, and not a whole number
        // of them, so the top limb is bounded at its own width rather than a full limb's.
        let bits = 328usize;
        let mut ssa = program_injecting(bits);
        run_pass(&mut ssa);

        let ops = emitted(&ssa);
        let bounded: Vec<usize> = ops
            .iter()
            .filter_map(|op| match op {
                OpCode::Rangecheck { max_bits, .. } => Some(*max_bits),
                _ => None,
            })
            .collect();

        assert_eq!(
            bounded,
            limb_widths(bits, witness_limb_bits(bn254())),
            "each injected limb is bounded at its own declared width"
        );
        assert!(
            bounded.len() > 1,
            "the value has to span more than one limb to be a test"
        );
        // And there is deliberately no reconstruction: a hint has nothing to be tied back to.
        assert!(
            !ops.iter().any(|op| matches!(op, OpCode::Constrain { .. })),
            "an injection ties nothing back; if it does, this test is checking the wrong gadget"
        );
    }

    /// `main(shadow, table, index, flag)`, the part of `witness_array_access`'s output that this
    /// arm reads: a hint out of the pure shadow, across the field boundary, into a column, and a
    /// lookup tying that column to the table.
    fn program_reading_a_sequence_at_a_witness_index(bits: usize, slots: usize) -> HLSSA {
        let mut ssa = HLSSA::with_main("main".to_string());
        let main = ssa.get_unique_entrypoint_id();
        let (shadow, table) = (ssa.fresh_value(), ssa.fresh_value());
        let (hint_index, index, flag) = (ssa.fresh_value(), ssa.fresh_value(), ssa.fresh_value());
        let (hint, image, column) = (ssa.fresh_value(), ssa.fresh_value(), ssa.fresh_value());
        let mut builder = HLSSABuilder::new(&mut ssa);
        builder.modify_function(main, |fb| {
            let entry = fb.function.get_entry_id();
            {
                let block = fb.function.get_block_mut(entry);
                // The hint comes off the pure shadow and the lookup pins it against the witnessed
                // table, which is the pair `witness_array_access` leaves behind.
                block.push_parameter(shadow, Type::int(bits).array_of(slots));
                block.push_parameter(hint_index, Type::int(32));
                block.push_parameter(table, Type::witness_of(Type::int(bits)).array_of(slots));
                block.push_parameter(index, Type::witness_of(Type::int(32)));
                block.push_parameter(flag, Type::witness_of(Type::field()));
            }
            let mut block = fb.test_block(entry);
            block.emit(OpCode::ArrayGet {
                result: hint,
                array: shadow,
                index: hint_index,
            });
            block.emit(OpCode::Cast {
                result: image,
                value: hint,
                target: CastTarget::Field,
            });
            block.emit(OpCode::WriteWitness {
                result: Some(column),
                value: image,
                pinned: false,
            });
            block.emit(OpCode::Lookup {
                target: LookupTarget::Array(table),
                args: vec![index, column],
                flag,
            });
            block.terminate_return(vec![]);
        });
        ssa
    }

    /// A transposed read is **one lookup per limb**, every one of them at the same index.
    ///
    /// The count is what makes the read sound, and no run can see it: a limb the tape never pins
    /// is a limb nothing else reads either, so it is eliminated rather than left as a free column
    /// and `every_column_of_a_wide_sequence_is_pinned` stays green without it. So the lookups are
    /// counted here, where they are emitted.
    ///
    /// The shared index is the other half: `k` lookups against `k` tables at `k` **different**
    /// indices would pin `k` limbs of no single element.
    #[test]
    fn a_transposed_read_is_one_lookup_per_limb_at_one_index() {
        let bits = 320usize;
        let mut ssa = program_reading_a_sequence_at_a_witness_index(bits, 2);
        run_pass(&mut ssa);

        let lookups: Vec<(ValueId, ValueId, ValueId)> = emitted(&ssa)
            .into_iter()
            .filter_map(|op| match op {
                OpCode::Lookup {
                    target: LookupTarget::Array(array),
                    args,
                    ..
                } => Some((array, args[0], args[1])),
                _ => None,
            })
            .collect();

        let expected = limb_widths(bits, witness_limb_bits(bn254())).len();
        assert!(expected > 1, "the element has to span limbs to be a test");
        assert_eq!(
            lookups.len(),
            expected,
            "an int{bits} element is {expected} limbs and each one needs its own lookup"
        );

        let indices: Vec<ValueId> = lookups.iter().map(|(_, index, _)| *index).collect();
        assert!(
            indices.windows(2).all(|pair| pair[0] == pair[1]),
            "every limb is read at the same index, or the limbs are not one element's"
        );

        let mut tables: Vec<ValueId> = lookups.iter().map(|(array, ..)| *array).collect();
        let mut results: Vec<ValueId> = lookups.iter().map(|(.., result)| *result).collect();
        tables.sort();
        tables.dedup();
        results.sort();
        results.dedup();
        assert_eq!(tables.len(), expected, "one table per limb sequence");
        assert_eq!(results.len(), expected, "one column per limb");
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
