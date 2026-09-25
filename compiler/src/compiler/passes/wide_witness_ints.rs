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
//! Everything else either moves limbs around or, as the bitwise operations do, acts on each limb
//! within its own width, and so they remain constrained. The exceptions are the carry chain and
//! the schoolbook product.
//!
//! # The Carry Chain
//!
//! The unsigned sum, difference and ordering are one gadget: `a < b` **is** the borrow out of the
//! top limb of `a - b`. Per limb of width `w`, the answer is `a + b + c_in - 2^w c_out` for a sum
//! and `a - b - c_in + 2^w c_out` for a difference, with `c_out` a witnessed carry checked to be a
//! bit and the answer range-checked at `w`, which is what pins the carry. A checked sum or
//! difference lets no carry out of the top limb, an ordering witnesses that carry as its answer,
//! and an asserted ordering fixes it at one.
//!
//! The chain runs past [`widest_cell_sum_bits`], one bit short of the threshold above: at that one
//! width the operands still have an element each but their sum does not, so the operands are
//! decomposed into limbs first. A sum or difference is then recombined into its element.
//!
//! # The Schoolbook Product
//!
//! An unsigned product runs here wherever the single cell cannot hold it, which is past
//! [`single_cell_product_fits`]: from the width whose product passes the modulus, apart from the
//! double lane's own two-limb product. Each column of the answer sums its partial products and the
//! carries into it, and is reduced into its limb, range-checked at the limb width, and a witnessed
//! carry range-checked at the width its bound needs. The top column carries nothing out, so it is
//! range-checked at the top limb's width instead, and every partial product landing past it is
//! held to zero, one constraint per left limb. [`plan_product`] decides where the reductions fall
//! and states why each identity holds as integers rather than modulo `p`.
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
use num_bigint::BigUint;
use num_traits::{One, Zero};

use crate::collections::HashMap;
use crate::compiler::{
    Field,
    analysis::{
        flow_analysis::FlowAnalysis,
        types::{FunctionTypeInfo, TypeInfo},
    },
    pass_manager::{Analysis, AnalysisId, AnalysisStore, Pass},
    passes::shared::{
        limbs::{
            single_cell_product_fits, widest_cell_sum_bits, widest_injective_int_bits,
            witness_limb_bits,
        },
        unsupported::unsupported_on_this_field,
    },
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
            // A function with no limbs can still hold an operation whose operands have an element
            // each and whose sum or product does not, which the chain and the schoolbook lower all
            // the same.
            let fti = type_info.get_function(fid);
            let lowers_here = || {
                reachable.iter().any(|bid| {
                    ssa.get_function(fid)
                        .get_block(*bid)
                        .get_instructions()
                        .any(|op| {
                            chained(op, fti, field).is_some()
                                || multiplied(op, fti, field).is_some()
                        })
                })
            };
            if value_map.values().all(|limbs| limbs.len() == 1) && !lowers_here() {
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
        let mut decomposed = HashMap::default();
        for instr in &old_instructions {
            let location = instr.location().clone();
            let mut rewriter = Rewriter {
                ssa,
                value_map,
                types: fti,
                field,
                decomposed: &mut decomposed,
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

    /// The decompositions already emitted in this block, by value and width.
    ///
    /// Kept per block as limbs minted in one block are in scope only in the blocks that block
    /// dominates, and this pass does not track dominance. Within a block the instructions are
    /// rewritten in order, so an earlier decomposition is always in scope.
    decomposed: &'a mut HashMap<(ValueId, usize), Vec<ValueId>>,

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

    fn select(&mut self, cond: ValueId, if_t: ValueId, if_f: ValueId) -> ValueId {
        let result = self.fresh();
        self.push(OpCode::Select {
            result,
            cond,
            if_t,
            if_f,
        });
        result
    }

    /// `op`, under `guard` where there is one.
    fn push_guarded(&mut self, guard: Option<ValueId>, op: OpCode) {
        match guard {
            Some(condition) => self.push(OpCode::Guard {
                condition,
                inner: Box::new(op),
            }),
            None => self.push(op),
        }
    }

    fn is_witness(&self, value: ValueId) -> bool {
        self.types.get_value_type(value).is_witness_of()
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
    ///
    /// A value is decomposed once per block at a given width. A later request reuses the limbs,
    /// which are already tied to the value, instead of paying the columns and checks again.
    fn decompose(&mut self, value: ValueId, bits: usize) -> Vec<ValueId> {
        if let Some(limbs) = self.decomposed.get(&(value, bits)) {
            return limbs.clone();
        }
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

        self.decomposed.insert((value, bits), limbs.clone());
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

// THE CARRY CHAIN
// ================================================================================================

/// Where the carry out of a chain's top limb goes.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum CarryOut {
    /// Nowhere: the operation is checked, and a carry past the top limb is its overflow.
    Zero,

    /// Out, always: an asserted ordering, whose subtraction has to borrow.
    One,

    /// Witnessed and handed back: an ordering, which **is** that borrow.
    Witnessed,
}

/// An operation the carry chain lowers, with the width it is lowered at.
struct Chained {
    /// The width of both operands, which `check_widths` makes the same, and so of every limb list
    /// the chain splits them into.
    bits: usize,

    /// The left operand as the program names it: the minuend of a difference, and the side an
    /// ordering asks is the smaller. One element or limbs, witnessed or pure.
    lhs: ValueId,

    /// The right operand as the program names it, in the same forms as `lhs`.
    rhs: ValueId,
}

/// What a carry chain leaves behind.
struct Chain {
    /// Each limb of the answer as a field element, range-checked at its own width.
    answer: Vec<ValueId>,

    /// The width of each limb, little-endian.
    widths: Vec<usize>,

    /// The carry out of the top limb, where the chain witnesses one.
    carry_out: Option<ValueId>,
}

impl Rewriter<'_> {
    /// The unsigned sum, difference and ordering at a width whose sum one element cannot carry,
    /// returning `true` if the provided `op` was lowered or `false` otherwise.
    ///
    /// All three are lowered using the carry [`Chain`] regardless of the operand format, as it is
    /// necessary to handle overflow properly.
    fn lower_through_chain(&mut self, op: &OpCode) -> bool {
        let Some(Chained { bits, lhs, rhs }) = chained(op, self.types, self.field) else {
            return false;
        };
        let (guard, inner) = match op {
            OpCode::Guard { condition, inner } => (Some(self.one(*condition)), inner.as_ref()),
            other => (None, other),
        };

        match inner {
            OpCode::BinaryArithOp { kind, result, .. } => {
                self.lower_add_sub(*kind, *result, lhs, rhs, bits, guard);
            }
            // An ordering cannot fail, so a guard has nothing to withhold from it: the chain holds
            // for any pair of operands, and every limb of either is bounded whatever the guard.
            OpCode::Cmp { result, .. } => {
                let chain = self.carry_chain(true, lhs, rhs, bits, CarryOut::Witnessed, None);
                self.push(OpCode::Cast {
                    result: *result,
                    value: chain.carry_out.expect("an ordering witnesses its borrow"),
                    target: CastTarget::Int(1),
                });
            }
            OpCode::AssertCmp { .. } => {
                self.carry_chain(true, lhs, rhs, bits, CarryOut::One, guard);
            }
            _ => ice_unreachable!("`chained` matches only these"),
        }
        true
    }

    /// A checked sum or difference, whose carry out of the top limb is its overflow.
    ///
    /// Under a guard the answer is zero where the guard is off, as the single-cell lowering's is:
    /// there the operands are whatever the not-taken branch left, the chain's range checks are off
    /// with the guard, and the limbs it would otherwise leave are unbounded.
    fn lower_add_sub(
        &mut self,
        kind: BinaryArithOpKind,
        result: ValueId,
        lhs: ValueId,
        rhs: ValueId,
        bits: usize,
        guard: Option<ValueId>,
    ) {
        let subtract = kind == BinaryArithOpKind::USub;
        let chain = self.carry_chain(subtract, lhs, rhs, bits, CarryOut::Zero, guard);
        self.deliver(result, chain.answer, &chain.widths, bits, guard);
    }

    /// Hand the limbs of an answer to `result`, each a field element range-checked at its width.
    ///
    /// Under a guard each limb is zero where the guard is off. A result held as limbs takes them as
    /// they are; one still held as an element has one, as the operands do, and it is only their
    /// sum or product that lacks one, which the limbs have already carried.
    fn deliver(
        &mut self,
        result: ValueId,
        answer: Vec<ValueId>,
        widths: &[usize],
        bits: usize,
        guard: Option<ValueId>,
    ) {
        let kept: Vec<ValueId> = answer
            .into_iter()
            .map(|limb| match guard {
                Some(condition) => {
                    let zero = self.field_const(self.field.zero());
                    self.select(condition, limb, zero)
                }
                None => limb,
            })
            .collect();

        let results = self.limbs(result);
        if results.len() == widths.len() {
            for ((result, limb), width) in paired(results, kept).zip(widths) {
                self.push(OpCode::Cast {
                    result,
                    value: limb,
                    target: CastTarget::Int(*width),
                });
            }
        } else {
            let limbs: Vec<ValueId> = kept
                .into_iter()
                .zip(widths)
                .map(|(limb, width)| self.cast(limb, CastTarget::Int(*width)))
                .collect();
            let whole = self.recombine(&limbs, widths, bits);
            self.push(OpCode::Cast {
                result,
                value: whole,
                target: CastTarget::Nop,
            });
        }
    }

    /// `lhs + rhs` or `lhs - rhs` limb by limb, each limb's carry a witnessed bit.
    ///
    /// Per limb of width `w`, the answer is `a + b + c_in - 2^w * c_out` for a sum and
    /// `a - b - c_in + 2^w * c_out` for a difference, range-checked at `w`. This pins `c_out`: the
    /// other choice moves the answer by `2^w`, either to `2^w` or past it, or below zero, where it
    /// wraps to an element no range check at `w` admits. Either answer is within `2^(w + 1)` of
    /// zero, far inside the modulus, so the identity is one field element whichever carry the
    /// prover picks.
    ///
    /// Every check the chain itself makes is under `guard`. The decomposition of an operand that
    /// is still one element is not: its checks restate that operand's own bounds, which hold
    /// whether the guard is on or off.
    fn carry_chain(
        &mut self,
        subtract: bool,
        lhs: ValueId,
        rhs: ValueId,
        bits: usize,
        out: CarryOut,
        guard: Option<ValueId>,
    ) -> Chain {
        assert!(
            subtract || out != CarryOut::One,
            "ICE: a sum whose top carry is forced out has no meaning"
        );
        let (fold, unfold) = if subtract {
            (BinaryArithOpKind::USub, BinaryArithOpKind::UAdd)
        } else {
            (BinaryArithOpKind::UAdd, BinaryArithOpKind::USub)
        };

        let widths = limb_widths(bits, self.limb_bits());
        let (lhs_witnessed, rhs_witnessed) = (self.is_witness(lhs), self.is_witness(rhs));
        let lhs = self.chain_operand(lhs, bits, widths.len());
        let rhs = self.chain_operand(rhs, bits, widths.len());

        // The carry into the next limb: its hint, which the next hint is computed from, and its
        // column, which the next identity reads.
        let mut carry: Option<(ValueId, ValueId)> = None;
        let mut answer = Vec::with_capacity(widths.len());
        for (index, width) in widths.iter().enumerate() {
            let top = index + 1 == widths.len();
            let a = self.cast(lhs[index], CastTarget::Field);
            let b = self.cast(rhs[index], CastTarget::Field);
            let mut limb = self.bin(fold, a, b);
            if let Some((_, column)) = carry {
                limb = self.bin(fold, limb, column);
            }

            let place = self.field.two_pow(*width);
            let next = if !top || out == CarryOut::Witnessed {
                let hint = self.carry_hint(
                    subtract,
                    (lhs[index], lhs_witnessed),
                    (rhs[index], rhs_witnessed),
                    carry.map(|(hint, _)| hint),
                    *width,
                );
                let hint_field = self.cast(hint, CastTarget::Field);
                let column = self.write_witness(hint_field);
                self.push_guarded(
                    guard,
                    OpCode::Rangecheck {
                        value: column,
                        max_bits: 1,
                    },
                );
                let place = self.field_const(place);
                let scaled = self.bin(BinaryArithOpKind::UMul, column, place);
                limb = self.bin(unfold, limb, scaled);
                Some((hint, column))
            } else {
                if out == CarryOut::One {
                    let place = self.field_const(place);
                    limb = self.bin(unfold, limb, place);
                }
                None
            };

            self.push_guarded(
                guard,
                OpCode::Rangecheck {
                    value: limb,
                    max_bits: *width,
                },
            );
            answer.push(limb);
            carry = next;
        }

        // The top limb mints a carry only where `out` witnesses one, so that is the only carry
        // that survives the loop.
        Chain {
            answer,
            widths,
            carry_out: carry.map(|(_, column)| column),
        }
    }

    /// The honest carry out of one limb, computed on the pure side at one bit past the limb.
    ///
    /// A sum's carry is its bit `w`; a difference borrows exactly when what it takes away exceeds
    /// what it takes it from. Neither reaches `2^(w + 1)`, so neither wraps.
    fn carry_hint(
        &mut self,
        subtract: bool,
        (a, a_witnessed): (ValueId, bool),
        (b, b_witnessed): (ValueId, bool),
        carry_in: Option<ValueId>,
        width: usize,
    ) -> ValueId {
        let wider = CastTarget::Int(width + 1);
        let a = self.pure_of(a, a_witnessed);
        let a = self.cast(a, wider.clone());
        let b = self.pure_of(b, b_witnessed);
        let mut b = self.cast(b, wider.clone());
        if let Some(carry_in) = carry_in {
            let carry_in = self.cast(carry_in, wider);
            b = self.bin(BinaryArithOpKind::UAdd, b, carry_in);
        }

        if subtract {
            let borrow = self.fresh();
            self.push(OpCode::Cmp {
                kind: CmpKind::ULt,
                result: borrow,
                lhs: a,
                rhs: b,
            });
            borrow
        } else {
            let sum = self.bin(BinaryArithOpKind::UAdd, a, b);
            let high = self.shifted_down(sum, width + 1, width);
            self.cast(high, CastTarget::Int(1))
        }
    }

    /// A limb's value on the pure side, where a hint is computed.
    fn pure_of(&mut self, limb: ValueId, witnessed: bool) -> ValueId {
        if witnessed {
            self.cast(limb, CastTarget::ValueOf)
        } else {
            limb
        }
    }

    /// An operand of the chain as its `count` limbs at `bits`.
    ///
    /// A witnessed operand that is still one value is below the representation threshold, and is
    /// split with [`Self::decompose`] so its limbs are tied to it; a pure one is cut on the pure side,
    /// which costs no constraint because it is not witnessed.
    fn chain_operand(&mut self, value: ValueId, bits: usize, count: usize) -> Vec<ValueId> {
        if self.limbs(value).len() == 1 && self.is_witness(value) {
            return self.decompose(value, bits);
        }
        self.operand_limbs(value, count)
    }
}

/// The operation the carry chain lowers, if `op` is one: an unsigned sum, difference or ordering,
/// guarded or not, with a witnessed operand, past [`widest_cell_sum_bits`].
fn chained(op: &OpCode, types: &FunctionTypeInfo, field: FieldConfig) -> Option<Chained> {
    let inner = match op {
        OpCode::Guard { inner, .. } => inner.as_ref(),
        other => other,
    };
    let (lhs, rhs) = match inner {
        OpCode::BinaryArithOp {
            kind: BinaryArithOpKind::UAdd | BinaryArithOpKind::USub,
            lhs,
            rhs,
            ..
        }
        | OpCode::Cmp {
            kind: CmpKind::ULt,
            lhs,
            rhs,
            ..
        }
        | OpCode::AssertCmp {
            kind: CmpKind::ULt,
            lhs,
            rhs,
        } => (*lhs, *rhs),
        _ => return None,
    };
    let bits = int_width(types.get_value_type(lhs))?;
    let witnessed =
        types.get_value_type(lhs).is_witness_of() || types.get_value_type(rhs).is_witness_of();
    (witnessed && bits > widest_cell_sum_bits(field)).then_some(Chained { bits, lhs, rhs })
}

// THE SCHOOLBOOK PRODUCT
// ================================================================================================

/// An unsigned product the schoolbook lowers, with the width it is lowered at.
///
/// Read as [`Chained`] is: `lhs` and `rhs` are the operands as the program names them, one element
/// or limbs, witnessed or pure, and `check_widths` makes them the same width.
struct Multiplied {
    bits: usize,
    lhs: ValueId,
    rhs: ValueId,
}

/// One operand of the schoolbook, limb by limb.
struct Factor {
    /// Each limb at its own width.
    limbs: Vec<Limb>,

    /// Whether `limbs` are witnessed, which decides how their hints are read.
    witnessed: bool,

    /// The largest value each limb can take: its width's, or a constant limb's own.
    bounds: Vec<BigUint>,
}

/// One limb of a [`Factor`].
enum Limb {
    /// A value, witnessed or pure as the operand is.
    Value(ValueId),

    /// A limb of a constant operand, interned only where the plan reads it.
    Constant(IntBits),
}

/// One step in accumulating a column of the product.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Step {
    /// Add the partial product of limb `lhs` of the left operand and limb `rhs` of the right.
    Product { lhs: usize, rhs: usize },

    /// Add column `m` of the product whole: every partial product landing in it, as one witnessed
    /// value the evaluation identity pins.
    Column(usize),

    /// Add the next carry out of the column below, in the order that column made them.
    Carry,

    /// Bound the accumulator.
    ///
    /// Below the top column it splits into its low limb, which stays in the column, and a witnessed
    /// carry of `carry_bits` into the next; the top column may carry nothing out, so there it is
    /// range-checked at the top limb's width instead.
    Reduce { carry_bits: Option<usize> },
}

/// How a product's columns are accumulated and where they are reduced, decided on the operands'
/// bounds alone.
#[derive(Debug, PartialEq, Eq)]
struct ProductPlan {
    /// The steps of each column of the answer, little-endian.
    ///
    /// A column whose bound passes its limb's width ends in a reduction, and the accumulator left
    /// afterwards is that limb.
    columns: Vec<Vec<Step>>,

    /// The overflow check.
    ///
    /// Each left limb with the right limbs whose partial products with it land at or past the top
    /// of the answer, grouped so that one group's sum times the limb stays inside an element. Empty
    /// where the product is evaluated, whose identity is its own overflow check.
    overflow: Vec<(usize, Vec<usize>)>,

    /// Whether the columns are witnessed whole and pinned by evaluating the product, instead of
    /// summed from partial products.
    evaluated: bool,

    /// The width the pure side computes its hints at, which no accumulator's bound reaches.
    hint_bits: usize,
}

/// The schoolbook plan for operands whose limbs are bounded by `lhs` and `rhs`, at the answer's
/// limb `widths`, on a field whose widest injective width is `injective`.
///
/// [`None`] where one partial product and the headroom it is reduced in do not fit an element.
///
/// **Every bound here is an integer bound, kept under the field.** A column adds partial products
/// and the carries into it, all non-negative, while the sum stays below `2^injective`, so the field
/// element is the integer sum. Where the next term would pass that, the column is reduced first:
/// its low limb is range-checked at the limb width and the rest leaves as a carry range-checked at
/// `carry_bits`, whose identity is `low + 2^w·carry` below `2^injective` and so holds as integers
/// too. The carry's bound is its range check's, `2^carry_bits - 1`, not the honest maximum.
///
/// **The overflow check is on the operands.** Every partial product is non-negative, so the product
/// stays below `2^N` exactly when every partial product landing at or past the top column's place
/// is zero and the top column itself fits its width. The first half is `a_i · Σ b_j == 0` over the
/// `b_j` that land there with `a_i`, which holds as integers while the sum times the limb is below
/// the modulus, and a zero sum of non-negative terms is zero term by term. It costs one constraint
/// per left limb rather than one per partial product.
///
/// **Where both operands are witnessed, the product is evaluated instead** (`evaluate`). Each
/// partial product of two witnessed limbs is a row of its own, so the schoolbook pays about
/// `k^2 / 2`; evaluating pays `2k - 1`. The columns `c_0 .. c_(k-1)` are witnessed whole and the
/// identity `(Σ a_i x^i)(Σ b_j x^j) = Σ c_m x^m` is constrained at `2k - 1` distinct points, which
/// makes it an identity of polynomials of degree `2k - 2` over the field, so every coefficient
/// agrees modulo `p`.
///
/// Every column's integer sum is non-negative, and the plan requires each to be below
/// `2^injective`, so a coefficient equal modulo `p` is equal as an integer: `c_m` is column `m`,
/// and the columns past the answer are zero which serves as the overflow check. Where some column
/// could pass the modulus, the plan falls back to the schoolbook, which reduces as often as it
/// needs to.
fn plan_product(
    lhs: &[BigUint],
    rhs: &[BigUint],
    widths: &[usize],
    injective: usize,
    evaluate: bool,
) -> Option<ProductPlan> {
    let count = widths.len();
    let fits = |bound: &BigUint| bound.bits() <= injective as u64;
    let mut hint_bits = widths.iter().copied().max().unwrap_or(1);

    let column_bound = |column: usize| -> BigUint {
        (0..count)
            .filter(|left| *left <= column && column - left < count)
            .map(|left| &lhs[left] * &rhs[column - left])
            .sum()
    };
    let evaluated = evaluate && (0..2 * count - 1).all(|column| fits(&column_bound(column)));

    let mut columns = Vec::with_capacity(count);
    let mut incoming: Vec<BigUint> = Vec::new();
    for (column, &width) in widths.iter().enumerate() {
        let top = column + 1 == count;
        let limit = BigUint::one() << width;

        let products: Vec<(Step, BigUint)> = if evaluated {
            let bound = column_bound(column);
            (!bound.is_zero())
                .then_some((Step::Column(column), bound))
                .into_iter()
                .collect()
        } else {
            (0..=column)
                .filter_map(|left| {
                    let right = column - left;
                    let bound = &lhs[left] * &rhs[right];
                    (!bound.is_zero()).then_some((
                        Step::Product {
                            lhs: left,
                            rhs: right,
                        },
                        bound,
                    ))
                })
                .collect()
        };
        let carries = std::mem::take(&mut incoming)
            .into_iter()
            .map(|bound| (Step::Carry, bound));
        let terms: Vec<(Step, BigUint)> = products.into_iter().chain(carries).collect();

        let mut steps = Vec::new();
        let mut accumulated = BigUint::zero();
        let mut reduce = |accumulated: &mut BigUint, steps: &mut Vec<Step>| {
            if top {
                steps.push(Step::Reduce { carry_bits: None });
            } else {
                let carry_bits = (&*accumulated >> width).bits() as usize;
                steps.push(Step::Reduce {
                    carry_bits: Some(carry_bits),
                });
                incoming.push((BigUint::one() << carry_bits) - 1u8);
            }
            *accumulated = &limit - 1u8;
        };

        for (step, bound) in terms {
            if !fits(&(&accumulated + &bound)) {
                if accumulated < limit {
                    return None;
                }
                reduce(&mut accumulated, &mut steps);
                if !fits(&(&accumulated + &bound)) {
                    return None;
                }
            }
            accumulated += bound;
            hint_bits = hint_bits.max(accumulated.bits() as usize);
            steps.push(step);
        }
        if accumulated >= limit {
            reduce(&mut accumulated, &mut steps);
        }
        columns.push(steps);
    }

    let mut overflow = Vec::new();
    for (left, bound) in lhs.iter().enumerate() {
        if evaluated || bound.is_zero() {
            continue;
        }
        let mut group = Vec::new();
        let mut sum = BigUint::zero();
        for right in count.saturating_sub(left)..count {
            if rhs[right].is_zero() {
                continue;
            }
            if !fits(&(bound * (&sum + &rhs[right]))) {
                if group.is_empty() {
                    return None;
                }
                overflow.push((left, std::mem::take(&mut group)));
                sum = BigUint::zero();
            }
            sum += &rhs[right];
            group.push(right);
        }
        if !group.is_empty() {
            overflow.push((left, group));
        }
    }

    Some(ProductPlan {
        columns,
        overflow,
        hint_bits,
        evaluated,
    })
}

/// Whether the schoolbook takes an unsigned product of two `bits`-wide witnessed operands on
/// `field`, evaluated or summed.
///
/// The bound is the operands' full range; a constant operand only tightens it. An evaluation the
/// field cannot hold falls back to summing, so it is the summing plan that decides.
pub fn schoolbook_product_fits(field: FieldConfig, bits: usize) -> bool {
    let widths = limb_widths(bits, witness_limb_bits(field));
    let bounds: Vec<BigUint> = widths
        .iter()
        .map(|width| (BigUint::one() << *width) - 1u8)
        .collect();
    plan_product(
        &bounds,
        &bounds,
        &widths,
        widest_injective_int_bits(field),
        true,
    )
    .is_some()
}

/// A column being accumulated: its field element and its mirror on the pure side, or nothing while
/// it is still zero.
#[derive(Clone, Copy)]
struct Accumulator {
    sum: Option<(ValueId, ValueId)>,
}

impl Rewriter<'_> {
    /// An unsigned product the single cell cannot hold, returning `true` if `op` was one and was
    /// lowered.
    ///
    /// Schoolbook at the witness limb, column by column, each column reduced into its answer limb
    /// and a witnessed carry into the next. [`plan_product`] decides where the reductions fall and
    /// states the soundness argument; this only follows it.
    ///
    /// Under a guard every range check and the overflow check are off where the guard is, and the
    /// answer is zero there, for the reason [`Self::lower_add_sub`] gives. The carries are written
    /// either way.
    fn lower_product(&mut self, op: &OpCode) -> bool {
        let Some(Multiplied { bits, lhs, rhs }) = multiplied(op, self.types, self.field) else {
            return false;
        };
        let (guard, inner) = match op {
            OpCode::Guard { condition, inner } => (Some(self.one(*condition)), inner.as_ref()),
            other => (None, other),
        };
        let OpCode::BinaryArithOp { result, .. } = inner else {
            ice_unreachable!("`multiplied` matches only a binary operation");
        };

        let widths = limb_widths(bits, self.limb_bits());
        let lhs = self.factor(lhs, bits, &widths);
        let rhs = self.factor(rhs, bits, &widths);

        let evaluate = lhs.witnessed && rhs.witnessed;
        let plan = plan_product(
            &lhs.bounds,
            &rhs.bounds,
            &widths,
            widest_injective_int_bits(self.field),
            evaluate,
        )
        .unwrap_or_else(|| {
            unsupported_on_this_field(
                format_args!(
                    "a {bits}-bit unsigned multiplication is a schoolbook product of witness limbs, which needs one partial product and the carry it is reduced into to fit a field element"
                ),
                self.field,
            )
        });

        let mut fields = FactorForms::new(widths.len());
        let columns = if plan.evaluated {
            self.evaluate_product(&plan, &lhs, &rhs, &mut fields, guard)
        } else {
            Vec::new()
        };
        let answer =
            self.accumulate_columns(&plan, &widths, &lhs, &rhs, &columns, &mut fields, guard);
        self.check_no_overflow(&plan, &lhs, &rhs, &mut fields, guard);
        self.deliver(*result, answer, &widths, bits, guard);
        true
    }

    /// An operand of the schoolbook as its limbs at `bits`, and what each of them can be.
    ///
    /// A constant is cut at compile time, so a limb it does not reach is a known zero and every
    /// partial product and overflow term against it drops out of the plan. This ensures that a
    /// product by a small constant is as cheap as the constant is narrow.
    fn factor(&mut self, value: ValueId, bits: usize, widths: &[usize]) -> Factor {
        if let Some(constant) = self.ssa.get_const(value)
            && let Constant::Int(pattern) = constant.as_ref()
        {
            let patterns: Vec<IntBits> = widths
                .iter()
                .enumerate()
                .map(|(index, width)| pattern.bit_range(index * self.limb_bits(), *width))
                .collect();
            return Factor {
                bounds: patterns.iter().map(BigUint::from).collect(),
                limbs: patterns.into_iter().map(Limb::Constant).collect(),
                witnessed: false,
            };
        }

        Factor {
            limbs: self
                .chain_operand(value, bits, widths.len())
                .into_iter()
                .map(Limb::Value)
                .collect(),
            witnessed: self.is_witness(value),
            bounds: widths
                .iter()
                .map(|width| (BigUint::one() << *width) - 1u8)
                .collect(),
        }
    }

    /// The columns of an evaluated product, witnessed and pinned by the identity [`plan_product`]
    /// states, each as its column and its hint.
    ///
    /// Under a guard the left limbs are scaled by it, so where it is off the product is zero and so
    /// is every column: the identity holds whatever the operands the branch not taken left behind.
    fn evaluate_product(
        &mut self,
        plan: &ProductPlan,
        lhs: &Factor,
        rhs: &Factor,
        forms: &mut FactorForms,
        guard: Option<ValueId>,
    ) -> Vec<(ValueId, ValueId)> {
        let count = lhs.limbs.len();
        let hint = CastTarget::Int(plan.hint_bits);
        let flag = guard.map(|condition| {
            let field = self.cast(condition, CastTarget::Field);
            let pure = self.pure_of(condition, self.is_witness(condition));
            (field, self.cast(pure, hint.clone()))
        });

        let lhs_fields: Vec<ValueId> = (0..count)
            .map(|index| {
                let limb = forms.field(self, Side::Lhs, index, lhs);
                match flag {
                    Some((flag, _)) => self.bin(BinaryArithOpKind::UMul, flag, limb),
                    None => limb,
                }
            })
            .collect();
        let rhs_fields: Vec<ValueId> = (0..count)
            .map(|index| forms.field(self, Side::Rhs, index, rhs))
            .collect();

        let mut columns = Vec::with_capacity(count);
        for column in 0..count {
            let mut hinted = None;
            for left in 0..=column {
                let a = forms.pure(self, Side::Lhs, left, lhs, &hint);
                let b = forms.pure(self, Side::Rhs, column - left, rhs, &hint);
                let product = self.bin(BinaryArithOpKind::UMul, a, b);
                hinted = Some(match hinted {
                    None => product,
                    Some(sum) => self.bin(BinaryArithOpKind::UAdd, sum, product),
                });
            }
            let mut hinted = hinted.expect("a column has at least one partial product");
            if let Some((_, flag)) = flag {
                hinted = self.bin(BinaryArithOpKind::UMul, flag, hinted);
            }
            let hint_field = self.cast(hinted, CastTarget::Field);
            columns.push((self.write_witness(hint_field), hinted));
        }

        let column_fields: Vec<ValueId> = columns.iter().map(|(column, _)| *column).collect();
        for point in 0..(2 * count - 1) as u64 {
            let a = self.evaluate_at(&lhs_fields, point);
            let b = self.evaluate_at(&rhs_fields, point);
            let c = self.evaluate_at(&column_fields, point);
            self.push(OpCode::Constrain { a, b, c });
        }
        columns
    }

    /// `Σ coefficients[i] · point^i` as a field element, which is linear in the coefficients.
    fn evaluate_at(&mut self, coefficients: &[ValueId], point: u64) -> ValueId {
        if point == 0 {
            return coefficients[0];
        }
        let base = self.field.constant(point);
        let mut power = self.field.one();
        let mut sum = None;
        for coefficient in coefficients {
            let term = if power == self.field.one() {
                *coefficient
            } else {
                let place = self.field_const(power);
                self.bin(BinaryArithOpKind::UMul, *coefficient, place)
            };
            sum = Some(match sum {
                None => term,
                Some(sum) => self.bin(BinaryArithOpKind::UAdd, sum, term),
            });
            power = power * base;
        }
        sum.expect("a polynomial has at least one coefficient")
    }

    /// Every column of the answer, as field elements, following `plan`.
    ///
    /// `columns` are an evaluated product's witnessed columns, which a [`Step::Column`] adds.
    #[allow(clippy::too_many_arguments)]
    fn accumulate_columns(
        &mut self,
        plan: &ProductPlan,
        widths: &[usize],
        lhs: &Factor,
        rhs: &Factor,
        columns: &[(ValueId, ValueId)],
        forms: &mut FactorForms,
        guard: Option<ValueId>,
    ) -> Vec<ValueId> {
        let hint = CastTarget::Int(plan.hint_bits);
        let mut answer = Vec::with_capacity(widths.len());
        let mut incoming: std::collections::VecDeque<(ValueId, ValueId)> = Default::default();

        for (steps, &width) in plan.columns.iter().zip(widths) {
            let mut outgoing = Vec::new();
            let mut column = Accumulator { sum: None };
            for step in steps {
                match *step {
                    Step::Product {
                        lhs: left,
                        rhs: right,
                    } => {
                        let a = forms.field(self, Side::Lhs, left, lhs);
                        let b = forms.field(self, Side::Rhs, right, rhs);
                        let product = self.bin(BinaryArithOpKind::UMul, a, b);
                        let a = forms.pure(self, Side::Lhs, left, lhs, &hint);
                        let b = forms.pure(self, Side::Rhs, right, rhs, &hint);
                        let hinted = self.bin(BinaryArithOpKind::UMul, a, b);
                        self.accumulate(&mut column, product, hinted);
                    }
                    Step::Column(index) => {
                        let (value, hinted) = columns[index];
                        self.accumulate(&mut column, value, hinted);
                    }
                    Step::Carry => {
                        let (carry, hinted) = incoming
                            .pop_front()
                            .expect("the plan adds each carry the column below made");
                        self.accumulate(&mut column, carry, hinted);
                    }
                    Step::Reduce { carry_bits } => {
                        let (sum, hinted) = column
                            .sum
                            .expect("the plan reduces only a column that has a bound");
                        match carry_bits {
                            None => self.push_guarded(
                                guard,
                                OpCode::Rangecheck {
                                    value: sum,
                                    max_bits: width,
                                },
                            ),
                            Some(carry_bits) => {
                                let carry = self.shifted_down(hinted, plan.hint_bits, width);
                                let carry = self.cast(carry, CastTarget::Int(carry_bits));
                                let carry_field = self.cast(carry, CastTarget::Field);
                                let written = self.write_witness(carry_field);
                                self.push_guarded(
                                    guard,
                                    OpCode::Rangecheck {
                                        value: written,
                                        max_bits: carry_bits,
                                    },
                                );
                                let place = self.field_const(self.field.two_pow(width));
                                let scaled = self.bin(BinaryArithOpKind::UMul, written, place);
                                let low = self.bin(BinaryArithOpKind::USub, sum, scaled);
                                self.push_guarded(
                                    guard,
                                    OpCode::Rangecheck {
                                        value: low,
                                        max_bits: width,
                                    },
                                );

                                let low_hint = self.cast(hinted, CastTarget::Int(width));
                                let low_hint = self.cast(low_hint, hint.clone());
                                column.sum = Some((low, low_hint));
                                outgoing.push((written, self.cast(carry, hint.clone())));
                            }
                        }
                    }
                }
            }
            answer.push(match column.sum {
                Some((sum, _)) => sum,
                None => self.field_const(self.field.zero()),
            });
            incoming.extend(outgoing);
        }
        assert!(
            incoming.is_empty(),
            "ICE: the top column of a product carried out of the answer"
        );
        answer
    }

    fn accumulate(&mut self, column: &mut Accumulator, term: ValueId, hinted: ValueId) {
        column.sum = Some(match column.sum {
            None => (term, hinted),
            Some((sum, sum_hint)) => (
                self.bin(BinaryArithOpKind::UAdd, sum, term),
                self.bin(BinaryArithOpKind::UAdd, sum_hint, hinted),
            ),
        });
    }

    /// The partial products at or past the top of the answer, held to zero one left limb at a time.
    fn check_no_overflow(
        &mut self,
        plan: &ProductPlan,
        lhs: &Factor,
        rhs: &Factor,
        forms: &mut FactorForms,
        guard: Option<ValueId>,
    ) {
        let zero = self.field_const(self.field.zero());
        for (left, group) in &plan.overflow {
            let a = forms.field(self, Side::Lhs, *left, lhs);
            let mut sum = None;
            for right in group {
                let b = forms.field(self, Side::Rhs, *right, rhs);
                sum = Some(match sum {
                    None => b,
                    Some(sum) => self.bin(BinaryArithOpKind::UAdd, sum, b),
                });
            }
            let sum = sum.expect("an overflow group names at least one limb");
            let (a, b) = match guard {
                None => (a, sum),
                Some(condition) => {
                    let flag = self.cast(condition, CastTarget::Field);
                    (flag, self.bin(BinaryArithOpKind::UMul, a, sum))
                }
            };
            self.push(OpCode::Constrain { a, b, c: zero });
        }
    }
}

/// Which operand of a product a limb belongs to.
#[derive(Clone, Copy)]
enum Side {
    Lhs,
    Rhs,
}

/// The field element and the widened pure hint of each operand limb.
struct FactorForms {
    field: [Vec<Option<ValueId>>; 2],
    pure: [Vec<Option<ValueId>>; 2],
}

impl FactorForms {
    fn new(count: usize) -> Self {
        Self {
            field: [vec![None; count], vec![None; count]],
            pure: [vec![None; count], vec![None; count]],
        }
    }

    fn field(
        &mut self,
        rewriter: &mut Rewriter<'_>,
        side: Side,
        index: usize,
        factor: &Factor,
    ) -> ValueId {
        *self.field[side as usize][index].get_or_insert_with(|| match &factor.limbs[index] {
            Limb::Value(limb) => rewriter.cast(*limb, CastTarget::Field),
            // A witness limb is never wider than the host word, so its pattern is one host limb.
            Limb::Constant(pattern) => {
                let element = rewriter.field.constant(pattern.limbs()[0]);
                rewriter.field_const(element)
            }
        })
    }

    fn pure(
        &mut self,
        rewriter: &mut Rewriter<'_>,
        side: Side,
        index: usize,
        factor: &Factor,
        hint: &CastTarget,
    ) -> ValueId {
        *self.pure[side as usize][index].get_or_insert_with(|| match &factor.limbs[index] {
            Limb::Value(limb) => {
                let pure = rewriter.pure_of(*limb, factor.witnessed);
                rewriter.cast(pure, hint.clone())
            }
            Limb::Constant(pattern) => {
                let CastTarget::Int(bits) = hint else {
                    ice_unreachable!("a hint is an integer width");
                };
                rewriter.int_const(pattern.cast(*bits))
            }
        })
    }
}

/// The product the schoolbook lowers, if `op` is one: an unsigned multiplication, guarded or not,
/// with a witnessed operand, at a width the single cell does not take.
fn multiplied(op: &OpCode, types: &FunctionTypeInfo, field: FieldConfig) -> Option<Multiplied> {
    let inner = match op {
        OpCode::Guard { inner, .. } => inner.as_ref(),
        other => other,
    };
    let OpCode::BinaryArithOp {
        kind: BinaryArithOpKind::UMul,
        lhs,
        rhs,
        ..
    } = inner
    else {
        return None;
    };
    let bits = int_width(types.get_value_type(*lhs))?;
    let witnessed =
        types.get_value_type(*lhs).is_witness_of() || types.get_value_type(*rhs).is_witness_of();
    (witnessed && !single_cell_product_fits(field, bits)).then_some(Multiplied {
        bits,
        lhs: *lhs,
        rhs: *rhs,
    })
}

// PER-INSTRUCTION REWRITING
// ================================================================================================

impl Rewriter<'_> {
    /// Rewrite one instruction into the limb-wise instructions that replace it.
    fn lower(&mut self, op: &OpCode) {
        if self.lower_through_chain(op) || self.lower_product(op) {
            return;
        }

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

            OpCode::AssertCmp { kind, lhs, rhs } => {
                self.lower_assert_compare(*kind, *lhs, *rhs, None);
            }

            // An assertion in a branch on a witness, which holds only where the branch is taken.
            OpCode::Guard { condition, inner } if matches!(**inner, OpCode::AssertCmp { .. }) => {
                let OpCode::AssertCmp { kind, lhs, rhs } = **inner else {
                    ice_unreachable!("matched above");
                };
                let guard = self.one(*condition);
                self.lower_assert_compare(kind, lhs, rhs, Some(guard));
            }

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

    /// The assertion of a comparison, which is one assertion per limb.
    fn lower_assert_compare(
        &mut self,
        kind: CmpKind,
        lhs: ValueId,
        rhs: ValueId,
        guard: Option<ValueId>,
    ) {
        assert!(
            matches!(kind, CmpKind::Eq),
            "ICE: a {kind:?} assertion of a wide witnessed integer reached the multi-cell representation; width validation should have refused the program"
        );

        let pairs = self.operand_pair(lhs, rhs);
        for (lhs, rhs) in pairs {
            self.push_guarded(
                guard,
                OpCode::AssertCmp {
                    kind: CmpKind::Eq,
                    lhs,
                    rhs,
                },
            );
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
/// An operation whose gadget needs more headroom than one element leaves does not move this: it
/// asks its own bound locally, as the carry chain asks [`widest_cell_sum_bits`], one bit narrower.
/// It is one constant, and `the_representation_threshold_is_where_an_element_stops_being_injective`
/// is what fails when it moves without its reason moving with it.
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

    /// `main(lhs, rhs: WitnessOf<int(bits)>) { op }`, returning the operation's result where it
    /// has a type.
    fn program_chaining(
        bits: usize,
        returns: Option<Type>,
        op: impl FnOnce(ValueId, ValueId, ValueId) -> OpCode,
    ) -> HLSSA {
        let mut ssa = HLSSA::with_main("main".to_string());
        let main = ssa.get_unique_entrypoint_id();
        let (lhs, rhs, result) = (ssa.fresh_value(), ssa.fresh_value(), ssa.fresh_value());
        let mut builder = HLSSABuilder::new(&mut ssa);
        builder.modify_function(main, |fb| {
            let entry = fb.function.get_entry_id();
            for value in [lhs, rhs] {
                fb.function
                    .get_block_mut(entry)
                    .push_parameter(value, Type::witness_of(Type::int(bits)));
            }
            let returned = match returns {
                Some(returned) => {
                    fb.function.add_return_type(returned);
                    vec![result]
                }
                None => vec![],
            };
            let mut block = fb.test_block(entry);
            block.emit(op(result, lhs, rhs));
            block.terminate_return(returned);
        });
        ssa
    }

    fn difference(bits: usize) -> HLSSA {
        program_chaining(
            bits,
            Some(Type::witness_of(Type::int(bits))),
            |result, lhs, rhs| OpCode::BinaryArithOp {
                kind: BinaryArithOpKind::USub,
                result,
                lhs,
                rhs,
            },
        )
    }

    /// Every carry the chain witnesses is a bit, and every limb of the answer is bounded at its own
    /// width.
    ///
    /// A structural audit, because perturbing a column by one cannot see the first half: a carry of
    /// two moves its limb's answer by `2^w` and breaks that limb's range check anyway. What the bit
    /// check stops is a carry that is not one step away, a field element that shifts value between
    /// two neighbouring limbs' identities while each still passes.
    #[test]
    fn every_carry_is_a_bit_and_every_limb_of_the_chain_is_bounded() {
        let h = witness_limb_bits(bn254());
        for bits in [254usize, 320] {
            let k = limb_widths(bits, h).len();
            let sum = |kind| {
                program_chaining(
                    bits,
                    Some(Type::witness_of(Type::int(bits))),
                    move |result, lhs, rhs| OpCode::BinaryArithOp {
                        kind,
                        result,
                        lhs,
                        rhs,
                    },
                )
            };
            let ordering = program_chaining(
                bits,
                Some(Type::witness_of(Type::int(1))),
                |result, lhs, rhs| OpCode::Cmp {
                    kind: CmpKind::ULt,
                    result,
                    lhs,
                    rhs,
                },
            );
            let assertion = program_chaining(bits, None, |_, lhs, rhs| OpCode::AssertCmp {
                kind: CmpKind::ULt,
                lhs,
                rhs,
            });

            // A checked operation has no carry out of its top limb, and an ordering witnesses the
            // one it has; an asserted ordering fixes it, so witnesses none either.
            for (what, mut ssa, carries) in [
                ("sum", sum(BinaryArithOpKind::UAdd), k - 1),
                ("difference", sum(BinaryArithOpKind::USub), k - 1),
                ("ordering", ordering, k),
                ("assertion", assertion, k - 1),
            ] {
                run_pass(&mut ssa);
                let ops = emitted(&ssa);

                let written: Vec<ValueId> = ops
                    .iter()
                    .filter_map(|op| match op {
                        OpCode::WriteWitness {
                            result: Some(result),
                            ..
                        } => Some(*result),
                        _ => None,
                    })
                    .collect();
                let bits_checked: Vec<ValueId> = ops
                    .iter()
                    .filter_map(|op| match op {
                        OpCode::Rangecheck { value, max_bits: 1 } => Some(*value),
                        _ => None,
                    })
                    .collect();
                let limbs_checked: Vec<usize> = ops
                    .iter()
                    .filter_map(|op| match op {
                        OpCode::Rangecheck { max_bits, .. } if *max_bits > 1 => Some(*max_bits),
                        _ => None,
                    })
                    .collect();

                assert_eq!(
                    written.len(),
                    carries,
                    "int{bits} {what}: one column per carry"
                );
                assert_eq!(
                    written, bits_checked,
                    "int{bits} {what}: each carry is a bit"
                );
                assert_eq!(
                    limbs_checked,
                    limb_widths(bits, h),
                    "int{bits} {what}: each limb of the answer is bounded at its own width"
                );
            }
        }
    }

    /// A guarded chain checks nothing where the guard is off, since the operands are then whatever
    /// the branch not taken left, and its answer is zero there.
    #[test]
    fn a_guarded_chain_is_checked_only_under_its_guard() {
        let bits = 320usize;
        let mut ssa = difference(bits);
        let main = ssa.get_unique_entrypoint_id();
        let condition = ssa.fresh_value();
        {
            let function = ssa.get_function_mut(main);
            let entry = function.get_entry_mut();
            entry.push_parameter(condition, Type::witness_of(Type::int(1)));
            let instructions: Vec<_> = entry
                .take_instructions()
                .into_iter()
                .map(|op| {
                    let guarded = OpCode::Guard {
                        condition,
                        inner: Box::new(op.as_ref().clone()),
                    };
                    Located::new(guarded, op.location().clone())
                })
                .collect();
            entry.put_instructions(instructions);
        }
        run_pass(&mut ssa);
        let ops = emitted(&ssa);

        assert!(
            !ops.iter().any(|op| matches!(op, OpCode::Rangecheck { .. })),
            "no range check is unguarded"
        );
        let guarded = ops
            .iter()
            .filter(|op| {
                matches!(op, OpCode::Guard { condition: c, inner }
                    if *c == condition && matches!(inner.as_ref(), OpCode::Rangecheck { .. }))
            })
            .count();
        let k = limb_widths(bits, witness_limb_bits(bn254())).len();
        assert_eq!(
            guarded,
            2 * k - 1,
            "every carry and every limb, under the guard"
        );
        let selected = ops
            .iter()
            .filter(|op| matches!(op, OpCode::Select { cond, .. } if *cond == condition))
            .count();
        assert_eq!(
            selected, k,
            "each limb of the answer is zero where the guard is off"
        );
    }

    /// The one width whose operands have an element each and whose sum does not goes through the
    /// chain, on a decomposition of each operand; one bit narrower, the pass leaves it alone.
    #[test]
    fn a_sum_the_element_cannot_carry_takes_the_chain_without_limbs() {
        let field = bn254();
        let wide = widest_cell_sum_bits(field) + 1;
        assert_eq!(
            wide,
            multi_cell_int_bits(field),
            "the band is one width wide"
        );

        let mut ssa = difference(wide);
        let main = ssa.get_unique_entrypoint_id();
        let operands: Vec<ValueId> = ssa
            .get_function(main)
            .get_entry()
            .get_parameters()
            .map(|(value, _)| *value)
            .collect();
        run_pass(&mut ssa);
        let ops = emitted(&ssa);
        let tied = ops
            .iter()
            .filter(|op| matches!(op, OpCode::Constrain { .. }))
            .count();
        assert_eq!(
            tied, 2,
            "each operand is decomposed and tied back to its element"
        );
        assert!(
            !ops.iter().any(|op| matches!(
                op,
                OpCode::BinaryArithOp { lhs, rhs, .. }
                    if operands.contains(lhs) && operands.contains(rhs)
            )),
            "the single-cell difference is gone"
        );

        let mut narrower = difference(wide - 1);
        let before = format!("{:?}", emitted(&narrower));
        run_pass(&mut narrower);
        assert_eq!(
            format!("{:?}", emitted(&narrower)),
            before,
            "one bit narrower, the sum fits its element"
        );
    }

    /// An operand in the band is decomposed once per block, however many chained operations read
    /// it and whichever side it is on.
    #[test]
    fn a_band_operand_is_decomposed_once_per_block() {
        let bits = multi_cell_int_bits(bn254());
        let constraints = |ssa: &HLSSA| {
            emitted(ssa)
                .iter()
                .filter(|op| matches!(op, OpCode::Constrain { .. }))
                .count()
        };

        // `x + x`: one value, one decomposition.
        let mut doubled = program_chaining(
            bits,
            Some(Type::witness_of(Type::int(bits))),
            |result, lhs, _| OpCode::BinaryArithOp {
                kind: BinaryArithOpKind::UAdd,
                result,
                lhs,
                rhs: lhs,
            },
        );
        run_pass(&mut doubled);
        assert_eq!(constraints(&doubled), 1, "x + x decomposes x once");

        // `lhs - rhs` and then `lhs < rhs` in the same block: two values, two decompositions.
        let mut ssa = difference(bits);
        let main = ssa.get_unique_entrypoint_id();
        let operands: Vec<ValueId> = ssa
            .get_function(main)
            .get_entry()
            .get_parameters()
            .map(|(value, _)| *value)
            .collect();
        let less = ssa.fresh_value();
        ssa.get_function_mut(main)
            .get_entry_mut()
            .push_test_instruction(OpCode::Cmp {
                kind: CmpKind::ULt,
                result: less,
                lhs: operands[0],
                rhs: operands[1],
            });
        run_pass(&mut ssa);
        // Counting reconstructions alone cannot tell a reused decomposition from an ordering that
        // was never lowered, which would leave the count at two as well.
        assert!(
            !emitted(&ssa).iter().any(|op| matches!(
                op,
                OpCode::Cmp { lhs, rhs, .. } if *lhs == operands[0] && *rhs == operands[1]
            )),
            "the ordering went through the chain"
        );
        assert_eq!(
            constraints(&ssa),
            2,
            "the ordering reuses the decompositions the difference made"
        );
    }

    // THE SCHOOLBOOK
    // --------------------------------------------------------------------------------------------

    /// Every limb at its width's full range, which is what a witnessed operand is held to.
    fn full_range(widths: &[usize]) -> Vec<BigUint> {
        widths
            .iter()
            .map(|width| (BigUint::one() << *width) - 1u8)
            .collect()
    }

    /// Re-derive what `plan` claims from its steps alone and hold it to the soundness argument
    /// [`plan_product`] states without reading how the plan was built.
    fn assert_plan_is_sound(
        plan: &ProductPlan,
        lhs: &[BigUint],
        rhs: &[BigUint],
        widths: &[usize],
        injective: usize,
    ) {
        let limit = BigUint::one() << injective;
        let count = widths.len();
        let mut added = crate::collections::HashSet::default();
        let mut incoming: Vec<BigUint> = Vec::new();

        for (column, (steps, &width)) in plan.columns.iter().zip(widths).enumerate() {
            let top = column + 1 == count;
            let mut accumulated = BigUint::zero();
            let mut carries = incoming.iter();
            let mut outgoing = Vec::new();
            for step in steps {
                match *step {
                    Step::Product {
                        lhs: left,
                        rhs: right,
                    } => {
                        assert_eq!(
                            left + right,
                            column,
                            "a partial product in the wrong column"
                        );
                        assert!(added.insert((left, right)), "a partial product added twice");
                        accumulated += &lhs[left] * &rhs[right];
                    }
                    Step::Column(index) => {
                        assert!(plan.evaluated, "a whole column in a summed plan");
                        assert_eq!(index, column, "a column added to another column");
                        for left in 0..=column {
                            let right = column - left;
                            if right < count && !(&lhs[left] * &rhs[right]).is_zero() {
                                assert!(added.insert((left, right)), "a product added twice");
                                accumulated += &lhs[left] * &rhs[right];
                            }
                        }
                    }
                    Step::Carry => {
                        accumulated += carries.next().expect("a carry the column below made");
                    }
                    Step::Reduce { carry_bits: None } => {
                        assert!(top, "only the top column reduces without a carry");
                        accumulated = (BigUint::one() << width) - 1u8;
                    }
                    Step::Reduce {
                        carry_bits: Some(carry_bits),
                    } => {
                        assert!(!top, "the top column carries nothing out");
                        assert!(
                            accumulated < BigUint::one() << (width + carry_bits),
                            "column {column}'s honest carry does not fit its range check"
                        );
                        assert!(
                            width + carry_bits <= injective,
                            "column {column}'s split does not stay below the modulus"
                        );
                        outgoing.push((BigUint::one() << carry_bits) - 1u8);
                        accumulated = (BigUint::one() << width) - 1u8;
                    }
                }
                assert!(accumulated < limit, "column {column} passes the modulus");
                assert!(
                    accumulated.bits() as usize <= plan.hint_bits,
                    "column {column} passes the hint width"
                );
            }
            assert!(carries.next().is_none(), "column {column} drops a carry");
            assert!(
                accumulated < BigUint::one() << width,
                "column {column} ends outside its limb"
            );
            incoming = outgoing;
        }
        assert!(incoming.is_empty(), "the top column carries out");

        // An evaluated product holds every column, the ones past the answer included, to its
        // integer sum, which has to stay below the modulus for that to mean anything.
        if plan.evaluated {
            assert!(
                plan.overflow.is_empty(),
                "an evaluated product with overflow terms"
            );
            for column in 0..2 * count - 1 {
                let sum: BigUint = (0..count)
                    .filter(|left| *left <= column && column - left < count)
                    .map(|left| &lhs[left] * &rhs[column - left])
                    .sum();
                assert!(
                    sum < limit,
                    "column {column} of an evaluated product passes the modulus"
                );
            }
        }

        let mut covered = crate::collections::HashSet::default();
        for (left, group) in &plan.overflow {
            let sum: BigUint = group.iter().map(|right| &rhs[*right]).sum();
            assert!(
                &lhs[*left] * sum < limit,
                "an overflow term passes the modulus"
            );
            for right in group {
                assert!(left + right >= count, "an overflow term inside the answer");
                assert!(
                    covered.insert((*left, *right)),
                    "an overflow term checked twice"
                );
            }
        }

        for left in 0..count {
            for right in 0..count {
                if (&lhs[left] * &rhs[right]).is_zero() {
                    continue;
                }
                let (set, what) = if left + right < count {
                    (&added, "added to its column")
                } else if plan.evaluated {
                    continue;
                } else {
                    (&covered, "held to zero")
                };
                assert!(
                    set.contains(&(left, right)),
                    "the partial product ({left}, {right}) is never {what}"
                );
            }
        }
    }

    /// The plan is sound on any field that can hold it, and on bn254 it can at every width.
    ///
    /// The injective widths below bn254's are fields this compiler cannot be configured for. They
    /// are what reaches the reductions part-way along a column, which bn254 never does, and a field
    /// too narrow for one product plus its headroom is where the plan refuses.
    #[test]
    fn the_schoolbook_plan_is_sound_on_any_field_that_holds_it() {
        let bn254 = widest_injective_int_bits(bn254());
        let sparse = |widths: &[usize]| -> Vec<BigUint> {
            full_range(widths)
                .into_iter()
                .enumerate()
                .map(|(index, bound)| match index % 3 {
                    0 => bound,
                    1 => BigUint::zero(),
                    _ => BigUint::from(5u8),
                })
                .collect()
        };
        for limb_bits in [16usize, 32, 64] {
            for bits in [127usize, 200, 253, 254, 320, 1000] {
                let widths = limb_widths(bits, limb_bits);
                let full = full_range(&widths);
                for injective in [bn254, 2 * limb_bits + 2, 2 * limb_bits + 1, 2 * limb_bits] {
                    for rhs in [full.clone(), sparse(&widths)] {
                        for evaluate in [false, true] {
                            match plan_product(&full, &rhs, &widths, injective, evaluate) {
                                Some(plan) => {
                                    assert_plan_is_sound(&plan, &full, &rhs, &widths, injective)
                                }
                                None => assert_ne!(injective, bn254, "bn254 refuses int{bits}"),
                            }
                        }
                    }
                }
            }
        }
    }

    /// On bn254 a whole column fits an element, so each reduces once, at its end, and each left
    /// limb's overflow terms are one constraint. Evaluated, every column fits too, so the product
    /// is evaluated and has no overflow terms at all.
    #[test]
    fn bn254_reduces_each_column_once() {
        let widths = limb_widths(320, witness_limb_bits(bn254()));
        let full = full_range(&widths);
        let injective = widest_injective_int_bits(bn254());

        for bits in [127usize, 254, 320, 1000, 16384] {
            let widths = limb_widths(bits, witness_limb_bits(bn254()));
            let full = full_range(&widths);
            let plan = plan_product(&full, &full, &widths, injective, true)
                .unwrap_or_else(|| panic!("bn254 holds an int{bits} product"));
            assert!(plan.evaluated, "int{bits} is evaluated on bn254");
            assert!(plan.overflow.is_empty(), "int{bits} has no overflow terms");
        }

        let plan = plan_product(&full, &full, &widths, injective, false)
            .expect("bn254 holds an int320 product");

        for (column, steps) in plan.columns.iter().enumerate() {
            let reductions = steps
                .iter()
                .filter(|step| matches!(step, Step::Reduce { .. }))
                .count();
            assert_eq!(reductions, 1, "column {column}");
            assert!(matches!(steps.last(), Some(Step::Reduce { .. })));
        }
        assert_eq!(
            plan.overflow.len(),
            widths.len() - 1,
            "one overflow constraint per left limb above the lowest"
        );
    }

    /// A field one partial product fills refuses; a field that holds one with room to spare but not
    /// a whole column reduces part-way along it.
    #[test]
    fn a_narrow_field_reduces_more_often_or_refuses() {
        let widths = limb_widths(320, 64);
        let full = full_range(&widths);

        // Goldilocks' 32-bit limb: `(2^32 - 1)^2` alone has 64 bits against an injective 63.
        let goldilocks = limb_widths(320, 32);
        for evaluate in [false, true] {
            assert!(
                plan_product(
                    &full_range(&goldilocks),
                    &full_range(&goldilocks),
                    &goldilocks,
                    63,
                    evaluate
                )
                .is_none()
            );
        }

        // Five products to a column do not fit 130 bits, so an evaluation falls back to summing.
        let injective = 2 * 64 + 2;
        let fallback = plan_product(&full, &full, &widths, injective, true)
            .expect("two products and a carry fit 130 bits");
        assert!(
            !fallback.evaluated,
            "a column past the modulus cannot be evaluated"
        );

        let plan = plan_product(&full, &full, &widths, injective, false)
            .expect("two products and a carry fit 130 bits");
        assert_plan_is_sound(&plan, &full, &full, &widths, injective);
        assert!(
            plan.columns.iter().any(|steps| {
                steps
                    .iter()
                    .filter(|step| matches!(step, Step::Reduce { .. }))
                    .count()
                    > 1
            }),
            "a column of five products reduces before its end"
        );
    }

    /// The funnel's question, asked of every width class bn254 has.
    #[test]
    fn bn254_holds_a_product_at_every_width() {
        for bits in [1usize, 127, 128, 129, 253, 254, 1000, 16383, 16384] {
            assert!(schoolbook_product_fits(bn254(), bits), "int{bits}");
        }
    }

    /// `main(lhs: WitnessOf<int(bits)>, rhs: rhs_type) -> WitnessOf<int(bits)> { lhs * rhs }`.
    fn product_with(bits: usize, rhs_type: Type) -> HLSSA {
        let mut ssa = HLSSA::with_main("main".to_string());
        let main = ssa.get_unique_entrypoint_id();
        let (lhs, rhs, result) = (ssa.fresh_value(), ssa.fresh_value(), ssa.fresh_value());
        let mut builder = HLSSABuilder::new(&mut ssa);
        builder.modify_function(main, |fb| {
            let entry = fb.function.get_entry_id();
            let block = fb.function.get_block_mut(entry);
            block.push_parameter(lhs, Type::witness_of(Type::int(bits)));
            block.push_parameter(rhs, rhs_type);
            fb.function
                .add_return_type(Type::witness_of(Type::int(bits)));
            let mut block = fb.test_block(entry);
            block.emit(OpCode::BinaryArithOp {
                kind: BinaryArithOpKind::UMul,
                result,
                lhs,
                rhs,
            });
            block.terminate_return(vec![result]);
        });
        ssa
    }

    /// `main(lhs, rhs: WitnessOf<int(bits)>) -> WitnessOf<int(bits)> { lhs * rhs }`.
    fn product(bits: usize) -> HLSSA {
        product_with(bits, Type::witness_of(Type::int(bits)))
    }

    /// Every carry a product witnesses is range-checked at the width its plan gives it and every
    /// limb of the answer at its own width, however the columns are formed.
    ///
    /// Evaluated, with both operands witnessed, each column is a witness of its own that only the
    /// `2k - 1` evaluations pin, and there is nothing else to hold to zero. Summed, with a pure
    /// right operand, the columns are linear and every left limb above the lowest is held to a zero
    /// overflow.
    ///
    /// A structural audit for the reason
    /// [`every_carry_is_a_bit_and_every_limb_of_the_chain_is_bounded`] gives: a carry check is
    /// invisible to perturbing its column by one, which the limb's own check catches first.
    #[test]
    fn every_carry_and_every_limb_of_a_product_is_bounded() {
        let h = witness_limb_bits(bn254());
        for bits in [254usize, 320] {
            let widths = limb_widths(bits, h);
            let k = widths.len();
            let full = full_range(&widths);
            for (what, rhs_type, evaluated) in [
                ("evaluated", Type::witness_of(Type::int(bits)), true),
                ("summed", Type::int(bits), false),
            ] {
                let plan = plan_product(
                    &full,
                    &full,
                    &widths,
                    widest_injective_int_bits(bn254()),
                    evaluated,
                )
                .expect("bn254 holds the product");
                assert_eq!(plan.evaluated, evaluated);
                let carry_bits: Vec<usize> = plan
                    .columns
                    .iter()
                    .flatten()
                    .filter_map(|step| match step {
                        Step::Reduce { carry_bits } => *carry_bits,
                        _ => None,
                    })
                    .collect();

                let mut ssa = product_with(bits, rhs_type);
                run_pass(&mut ssa);
                let ops = emitted(&ssa);

                let checked = |value: ValueId| {
                    ops.iter().find_map(|op| match op {
                        OpCode::Rangecheck {
                            value: checked,
                            max_bits,
                        } if *checked == value => Some(*max_bits),
                        _ => None,
                    })
                };
                let written: Vec<ValueId> = ops
                    .iter()
                    .filter_map(|op| match op {
                        OpCode::WriteWitness {
                            result: Some(result),
                            ..
                        } => Some(*result),
                        _ => None,
                    })
                    .collect();
                let (carries, columns): (Vec<ValueId>, Vec<ValueId>) =
                    written.iter().partition(|value| checked(**value).is_some());
                assert_eq!(
                    carries
                        .iter()
                        .map(|value| checked(*value).unwrap())
                        .collect::<Vec<_>>(),
                    carry_bits,
                    "int{bits} {what}: one column per carry, each checked at its planned width"
                );
                assert_eq!(
                    columns.len(),
                    if evaluated { k } else { 0 },
                    "int{bits} {what}: a witnessed column per column of the answer"
                );

                let limbs: Vec<usize> = ops
                    .iter()
                    .filter_map(|op| match op {
                        OpCode::Rangecheck { value, max_bits } if !written.contains(value) => {
                            Some(*max_bits)
                        }
                        _ => None,
                    })
                    .collect();
                assert_eq!(
                    limbs, widths,
                    "int{bits} {what}: each limb of the answer is bounded at its own width"
                );

                let constraints = ops
                    .iter()
                    .filter(|op| matches!(op, OpCode::Constrain { .. }))
                    .count();
                assert_eq!(
                    constraints,
                    if evaluated { 2 * k - 1 } else { k - 1 },
                    "int{bits} {what}: the evaluations, or one overflow constraint per left limb"
                );
            }
        }
    }

    /// The evaluations are at `2k - 1` distinct points, as many as the degree-`2k - 2` identity
    /// needs to be one of polynomials, and each reads every limb of both operands and every column.
    ///
    /// Structural because an honest witness satisfies the identity at any set of points: one point
    /// short, or two the same, and a prover could move value into the columns past the answer
    /// that nothing else would see.
    #[test]
    fn an_evaluated_product_is_pinned_at_distinct_points() {
        let bits = 320usize;
        let k = limb_widths(bits, witness_limb_bits(bn254())).len();
        let mut ssa = product(bits);
        run_pass(&mut ssa);
        let ops = emitted(&ssa);

        let definitions: HashMap<ValueId, OpCode> = ops
            .iter()
            .flat_map(|op| op.get_results().map(move |result| (*result, op.clone())))
            .collect();
        // The constant each term of a linear combination is scaled by, `1` where it is not.
        let terms = |value: ValueId| -> Vec<(ValueId, Field)> {
            let mut out = Vec::new();
            let mut stack = vec![value];
            while let Some(value) = stack.pop() {
                match definitions.get(&value) {
                    Some(OpCode::BinaryArithOp {
                        kind: BinaryArithOpKind::UAdd,
                        lhs,
                        rhs,
                        ..
                    }) => stack.extend([*lhs, *rhs]),
                    Some(OpCode::BinaryArithOp {
                        kind: BinaryArithOpKind::UMul,
                        lhs,
                        rhs,
                        ..
                    }) if matches!(ssa.get_const(*rhs).as_deref(), Some(Constant::Field(_))) => {
                        let Some(Constant::Field(scale)) = ssa.get_const(*rhs).as_deref().cloned()
                        else {
                            unreachable!()
                        };
                        out.push((*lhs, scale));
                    }
                    _ => out.push((value, bn254().one())),
                }
            }
            out
        };

        let evaluations: Vec<(Vec<(ValueId, Field)>, Vec<(ValueId, Field)>)> = ops
            .iter()
            .filter_map(|op| match op {
                OpCode::Constrain { a, b, c } => {
                    Some((terms(*a), terms(*b).into_iter().chain(terms(*c)).collect()))
                }
                _ => None,
            })
            .collect();
        assert_eq!(evaluations.len(), 2 * k - 1, "one evaluation per point");

        // The point is the scale of the second coefficient, and `0` where only the first is read.
        let mut points: Vec<Field> = evaluations
            .iter()
            .map(|(a, _)| {
                if a.len() == 1 {
                    bn254().zero()
                } else {
                    a.iter()
                        .map(|(_, scale)| *scale)
                        .find(|scale| *scale != bn254().one())
                        .unwrap_or(bn254().one())
                }
            })
            .collect();
        points.sort_by_key(|point| format!("{point:?}"));
        points.dedup();
        assert_eq!(points.len(), 2 * k - 1, "the points are distinct");
        for (index, (a, rest)) in evaluations.iter().enumerate() {
            if index == 0 {
                continue;
            }
            assert_eq!(a.len(), k, "evaluation {index} reads every left limb");
            assert_eq!(
                rest.len(),
                2 * k,
                "evaluation {index} reads every right limb and column"
            );
        }
    }

    /// A guarded product checks nothing where the guard is off, and its answer is zero there.
    ///
    /// Evaluated, the product itself is made zero there instead, by scaling the left limbs: the
    /// identity then holds of the operands the branch not taken left behind, and its range checks
    /// are guarded all the same.
    #[test]
    fn a_guarded_product_is_checked_only_under_its_guard() {
        let bits = 320usize;
        let mut ssa = product(bits);
        let main = ssa.get_unique_entrypoint_id();
        let condition = ssa.fresh_value();
        {
            let function = ssa.get_function_mut(main);
            let entry = function.get_entry_mut();
            entry.push_parameter(condition, Type::witness_of(Type::int(1)));
            let instructions: Vec<_> = entry
                .take_instructions()
                .into_iter()
                .map(|op| {
                    let guarded = OpCode::Guard {
                        condition,
                        inner: Box::new(op.as_ref().clone()),
                    };
                    Located::new(guarded, op.location().clone())
                })
                .collect();
            entry.put_instructions(instructions);
        }
        run_pass(&mut ssa);
        let ops = emitted(&ssa);

        assert!(
            !ops.iter().any(|op| matches!(op, OpCode::Rangecheck { .. })),
            "no range check is unguarded"
        );
        let k = limb_widths(bits, witness_limb_bits(bn254())).len();
        let guarded = ops
            .iter()
            .filter(|op| {
                matches!(op, OpCode::Guard { condition: c, inner }
                    if *c == condition && matches!(inner.as_ref(), OpCode::Rangecheck { .. }))
            })
            .count();
        assert_eq!(
            guarded,
            2 * k - 1,
            "every carry and every limb, under the guard"
        );

        let flag: Vec<ValueId> = ops
            .iter()
            .filter_map(|op| match op {
                OpCode::Cast {
                    result,
                    value,
                    target: CastTarget::Field,
                } if *value == condition => Some(*result),
                _ => None,
            })
            .collect();
        let scaled = ops
            .iter()
            .filter(|op| {
                matches!(op, OpCode::BinaryArithOp { kind: BinaryArithOpKind::UMul, lhs, .. }
                    if flag.contains(lhs))
            })
            .count();
        assert_eq!(
            scaled, k,
            "every left limb is scaled by the guard, so the product is zero where it is off"
        );
        let evaluations = ops
            .iter()
            .filter(|op| matches!(op, OpCode::Constrain { .. }))
            .count();
        assert_eq!(
            evaluations,
            2 * k - 1,
            "the evaluations need no guard of their own"
        );

        let selected = ops
            .iter()
            .filter(|op| matches!(op, OpCode::Select { cond, .. } if *cond == condition))
            .count();
        assert_eq!(
            selected, k,
            "each limb of the answer is zero where the guard is off"
        );
    }

    /// A product the single cell cannot hold goes through the schoolbook, on a decomposition of each
    /// operand while they still have an element; one it can hold, and the double lane's width,
    /// are left alone.
    #[test]
    fn a_product_the_single_cell_cannot_hold_takes_the_schoolbook() {
        let field = bn254();
        let widest = widest_injective_int_bits(field) / 2;
        assert!(!single_cell_product_fits(field, widest + 1));

        let mut ssa = product(widest + 1);
        let main = ssa.get_unique_entrypoint_id();
        let operands: Vec<ValueId> = ssa
            .get_function(main)
            .get_entry()
            .get_parameters()
            .map(|(value, _)| *value)
            .collect();
        run_pass(&mut ssa);
        let ops = emitted(&ssa);
        assert!(
            !ops.iter().any(|op| matches!(
                op,
                OpCode::BinaryArithOp { lhs, rhs, .. }
                    if operands.contains(lhs) && operands.contains(rhs)
            )),
            "the single-cell product is gone"
        );
        let tied = ops
            .iter()
            .filter(|op| matches!(op, OpCode::Constrain { .. }))
            .count();
        assert_eq!(tied, 2 + 3, "two decompositions and three evaluations");

        for bits in [widest, 2 * HOST_LIMB_BITS] {
            let mut ssa = product(bits);
            let before = format!("{:?}", emitted(&ssa));
            run_pass(&mut ssa);
            assert_eq!(format!("{:?}", emitted(&ssa)), before, "int{bits}");
        }
    }

    /// A constant factor is cut at compile time, so the limbs it does not reach cost nothing: a
    /// product by a one-limb constant has no partial product above the diagonal and nothing to hold
    /// to zero.
    #[test]
    fn a_constant_factor_drops_the_limbs_it_does_not_reach() {
        let bits = 320usize;
        let mut ssa = product(bits);
        let main = ssa.get_unique_entrypoint_id();
        let five = ssa.add_const(Constant::Int(IntBits::from_u128(bits, 5)));
        {
            let entry = ssa.get_function_mut(main).get_entry_mut();
            let instructions: Vec<_> = entry
                .take_instructions()
                .into_iter()
                .map(|op| {
                    let OpCode::BinaryArithOp {
                        kind, result, lhs, ..
                    } = op.as_ref().clone()
                    else {
                        panic!("the program is one product")
                    };
                    let rewritten = OpCode::BinaryArithOp {
                        kind,
                        result,
                        lhs,
                        rhs: five,
                    };
                    Located::new(rewritten, op.location().clone())
                })
                .collect();
            entry.put_instructions(instructions);
        }
        run_pass(&mut ssa);
        let ops = emitted(&ssa);

        let k = limb_widths(bits, witness_limb_bits(bn254())).len();
        let products = ops
            .iter()
            .filter(|op| {
                matches!(
                    op,
                    OpCode::BinaryArithOp {
                        kind: BinaryArithOpKind::UMul,
                        ..
                    }
                )
            })
            .count();
        assert_eq!(
            ops.iter()
                .filter(|op| matches!(op, OpCode::Constrain { .. }))
                .count(),
            0,
            "no partial product reaches past the answer"
        );
        // Per column: the partial product and its hint, then the carry's place value.
        assert_eq!(products, 2 * k + (k - 1), "one partial product per column");
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
