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
//! within its own width, and so they remain constrained. The exceptions are the carry chain, the
//! schoolbook product, the division built from the two, the shift, and the sign and the magnitude.
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
//! The signed three are the same chain. A signed sum or difference keeps the unsigned one's limbs
//! and witnesses its top carry, and is checked by one linear relation between that carry and the
//! sign bits of both operands and of the answer. A signed ordering is the unsigned ordering of the
//! operands with their top bits flipped, which is offset binary.
//!
//! # The Schoolbook Product
//!
//! An unsigned product runs here wherever the single cell cannot hold it, which is past
//! [`single_cell_product_fits`]: from the width whose product passes the modulus, apart from the
//! double lane's own two-limb product. A signed one runs here past
//! [`single_cell_signed_product_fits`], which has no such escape, as the product of its operands'
//! magnitudes (see the sign, below). Each column of the answer sums its partial products and the
//! carries into it, and is reduced into its limb, range-checked at the limb width, and a witnessed
//! carry range-checked at the width its bound needs. The top column carries nothing out, so it is
//! range-checked at the top limb's width instead, and every partial product landing past it is
//! held to zero, one constraint per left limb. [`plan_product`] decides where the reductions fall
//! and states why each identity holds as integers rather than modulo `p`.
//!
//! # The Division
//!
//! An unsigned division or remainder runs here wherever its single-cell lowering could not form
//! `q·d` in one element, which is where the product does. The quotient and the remainder are
//! witnessed as limbs from the pure side's own division and pinned by two checks: `q·d + r == n`,
//! which is the schoolbook with `r` as one more term in each column and every column **held to the
//! dividend's limb** rather than range-checked, and `r < d`, which is the carry chain with its top
//! borrow forced out and so also refuses a zero divisor. Together they admit exactly the pair
//! Euclidean division gives. An operand's known limbs bound both answers, and a limb they cannot
//! reach is a known zero rather than a column. A signed division is the unsigned one of the
//! operands' magnitudes, past [`single_cell_signed_product_fits`].
//!
//! # The Shift
//!
//! A shift under either reading runs here wherever the single cell cannot hold it, which is past
//! [`single_cell_shift_fits`]. A left shift is one map on the pattern, and a signed right shift is
//! the unsigned one with the sign filled in. An
//! amount known at compile time only moves bits, so the operand is cut where a run of its bits
//! stops landing inside one limb of the answer, and each piece is range-checked at its own width;
//! moving an operand held as limbs by a whole number of them costs nothing. Any other amount is
//! `q·h + r`, with `r` read out of the powers-of-two table beside `2^r`. Every limb is split at
//! `2^r` into two halves, each range-checked, that land either side of a limb boundary without
//! overlapping, and a barrel over the bits of `q` moves the whole limbs. The amount's own bound is
//! the decomposition where that reaches the width, and an explicit range check where it does not.
//! A pure amount is known wherever the constraints are built, so its decomposition is pure, its
//! bound a comparison, and the split and the barrel linear.
//!
//! # The Sign
//!
//! A signed value's sign is the top bit of its top limb, which [`Rewriter::sign_bit`] cuts out as a
//! witnessed bit beside the rest of the limb, range-checked one bit narrower. The chain's checks
//! above are linear in it, as is a sign extension into the representation, whose limbs past the
//! source's are that bit times all ones, and the fill of a right shift by a known amount.
//!
//! The rest is sign-magnitude. A limb complemented where the sign is one,
//! [`Rewriter::complement`], is `x + s·(2^w - 1 - 2x)`: one product, and below `2^w` with no check
//! of its own. A value negated where the sign is one, [`Rewriter::negate_if`], is each limb
//! complemented and the sign added through the chain, its top carry dropped. A product, quotient
//! and remainder run the unsigned gadgets on the operands' magnitudes, and the answer's magnitude
//! is held to `2^(N - 1) - 1 + s` for the sign `s` it takes, by one constraint on its top bit, and
//! negated by it. A right shift by any other amount is the unsigned one between two complements by
//! the operand's sign.
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

use std::collections::BTreeSet;

use mavros_artifacts::FieldConfig;
use mavros_int_semantics::IntBits;
use num_bigint::{BigInt, BigUint};
use num_traits::{One, ToPrimitive, Zero};

use crate::collections::HashMap;
use crate::compiler::{
    Field,
    analysis::{
        flow_analysis::FlowAnalysis,
        types::{FunctionTypeInfo, TypeInfo},
        value_range_analysis::field_modulus,
    },
    pass_manager::{Analysis, AnalysisId, AnalysisStore, Pass},
    passes::shared::{
        divmod_guard::nonzero_hint_divisor,
        limbs::{
            LimbBudget, ceil_log2, limb_bits_for_modulus, max_pow2_table_size,
            single_cell_product_fits, single_cell_shift_fits, single_cell_signed_product_fits,
            widest_cell_sum_bits, widest_injective_int_bits, widest_injective_int_bits_for_modulus,
            witness_limb_bits,
        },
        overflow_guard::abs_as_u,
        shift_guard::{amount_type_stays_below, emit_pure_shift_amount_check},
        unsupported::unsupported_on_this_field,
    },
    ssa::{
        BlockId, FunctionId, Instruction, Located, Terminator, ValueId,
        hlssa::{
            BinaryArithOpKind, Blob, CastTarget, CmpKind, Constant, HLSSA, LocatedOpCode,
            LookupTarget, OpCode, Type, TypeExpr, builder::HLEmitter,
        },
    },
    util::field_constant,
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
            // each and whose sum or product does not, which the chain, the schoolbook and the
            // division lower all the same.
            let fti = type_info.get_function(fid);
            let lowers_here = || {
                reachable.iter().any(|bid| {
                    ssa.get_function(fid)
                        .get_block(*bid)
                        .get_instructions()
                        .any(|op| {
                            chained(op, fti, field).is_some()
                                || multiplied(op, fti, field).is_some()
                                || divided(op, fti, field).is_some()
                                || shifted(op, fti, field).is_some()
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
                    // Pure integer reads are recombined immediately: their users, including
                    // arithmetic and function calls, expect one whole integer in the hint domain.
                    OpCode::ArrayGet { result, array, .. } => {
                        if matches!(fti.get_value_type(*result).expr, TypeExpr::Int(_)) {
                            continue;
                        }
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

    let mut known = HashMap::default();
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
        let mut signs = HashMap::default();
        let mut divisions = HashMap::default();
        for instr in &old_instructions {
            let location = instr.location().clone();
            let mut rewriter = Rewriter {
                ssa,
                value_map,
                types: fti,
                field,
                decomposed: &mut decomposed,
                signs: &mut signs,
                divisions: &mut divisions,
                known: &mut known,
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

/// A division's operands, guard and reading, which decide both of its answers: dividend, divisor,
/// guard, and whether it is signed.
type Division = (ValueId, ValueId, Option<ValueId>, bool);

/// One answer of a division, as far as it has been built.
#[derive(Clone)]
enum Answer {
    /// Limbs as the field elements [`Rewriter::deliver`] takes.
    Built(Vec<ValueId>),

    /// A signed answer's magnitude as limbs, each with whether it is witnessed, and the sign it
    /// takes, which [`Rewriter::answer`] negates it by the first time it is asked for.
    Signed(Vec<(ValueId, bool)>, Sign),
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
    decomposed: &'a mut HashMap<(ValueId, usize), Vec<ValueId>>,

    /// The top bits already cut out of a witnessed limb in this block, by limb and width.
    signs: &'a mut HashMap<(ValueId, usize), ValueId>,

    /// The divisions already built in this block, as their quotient and remainder, by dividend,
    /// divisor, guard and reading.
    divisions: &'a mut HashMap<Division, (Answer, Answer)>,

    /// The values this function's rewrite has minted whose pattern is known at compile time but
    /// that are not constants themselves, by value.
    known: &'a mut HashMap<ValueId, IntBits>,

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
    /// pair is legal, so one witnessed operand is sufficient to require a witness lowering. A
    /// constant is split at compile time, so its limbs are constants too.
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

        let constant = self.known(value);
        widths
            .into_iter()
            .enumerate()
            .map(|(index, width)| match &constant {
                Some(pattern) => self.int_const(pattern.bit_range(index * self.limb_bits(), width)),
                None => {
                    let shifted = self.shifted_down(value, bits, index * self.limb_bits());
                    self.cast(shifted, CastTarget::Int(width))
                }
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

    /// The guard's condition, as its one limb, and the operation it wraps, or no condition and `op`
    /// itself where there is no guard.
    fn split_guard<'o>(&self, op: &'o OpCode) -> (Option<ValueId>, &'o OpCode) {
        match op {
            OpCode::Guard { condition, inner } => (Some(self.one(*condition)), inner.as_ref()),
            other => (None, other),
        }
    }

    /// The declared width of a witnessed integer this pass represents as limbs.
    fn wide_width(&self, value: ValueId) -> Option<usize> {
        wide_witness_width(self.types.get_value_type(value), self.field)
    }

    fn fresh(&self) -> ValueId {
        self.ssa.fresh_value()
    }

    fn push(&mut self, op: OpCode) {
        if let OpCode::Cast {
            result,
            value,
            target,
        } = &op
            && let Some(pattern) = self.known(*value)
        {
            let pattern = match target {
                CastTarget::WitnessOf | CastTarget::ValueOf | CastTarget::Nop => Some(pattern),
                CastTarget::Int(bits) => Some(pattern.cast(*bits)),
                _ => None,
            };
            if let Some(pattern) = pattern {
                self.known.insert(*result, pattern);
            }
        }
        self.out.push(op);
    }

    /// The pattern of an integer known at compile time: a constant, or a value minted from one
    /// through casts that keep or cut its pattern.
    fn known(&self, value: ValueId) -> Option<IntBits> {
        match self.ssa.get_const(value).as_deref() {
            Some(Constant::Int(pattern)) => Some(pattern.clone()),
            Some(_) => None,
            None => self.known.get(&value).cloned(),
        }
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

    /// `value >> offset` on the pure side, as a division rather than a shift.
    ///
    /// At a width the field carries, a shift by a constant is rewritten into a `BitRange` by
    /// `Simplifier`, which runs **after** this pass and therefore after the rule that lowers one —
    /// so the window would survive into codegen and be refused there. A division by the same power
    /// of two says the same thing at every width and is the shape `lookup_spilling`'s own chunk
    /// extraction already takes.
    fn shifted_down(&mut self, value: ValueId, bits: usize, offset: usize) -> ValueId {
        if offset == 0 {
            return value;
        }
        let divisor = self.two_pow_const(bits, offset);
        self.bin(BinaryArithOpKind::UDiv, value, divisor)
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

    /// A limb's pattern as a field constant.
    fn limb_const(&self, pattern: &IntBits) -> ValueId {
        self.field_const(
            field_constant(self.field, pattern)
                .unwrap_or_else(|| ice!("a limb's pattern {pattern:?} is wider than the field")),
        )
    }

    /// `2^(index * h)` as a field element, the place value of limb `index`.
    fn place_value(&self, index: usize) -> ValueId {
        self.field_const(self.field.two_pow(index * self.limb_bits()))
    }
}

/// The rewriter is an emitter, so that the helpers every lowering shares can build into it. What it
/// emits goes through [`Rewriter::push`], which is what tracks the constants a cast carries.
impl HLEmitter for Rewriter<'_> {
    fn fresh_value(&mut self) -> ValueId {
        self.fresh()
    }

    fn emit(&mut self, instruction: OpCode) {
        self.push(instruction);
    }

    fn emit_located(&mut self, instruction: LocatedOpCode) {
        // Every instruction the rewriter emits takes the location of the one it replaces.
        self.push(instruction.as_ref().clone());
    }

    fn emit_constant(&mut self, value: Constant) -> ValueId {
        self.ssa.add_const(value)
    }

    fn field(&self) -> FieldConfig {
        self.field
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
            let (limb, limb_field) = self.witness_limb(pure, bits, index, *width, *width);
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
        self.constrain_equal(value_field, sum, None);

        self.decomposed.insert((value, bits), limbs.clone());
        limbs
    }

    /// Limb `index` of a pure `value` of `bits`, at `width`, written to a witness column and
    /// range-checked at `checked` bits, as the limb and its field element.
    ///
    /// The column is written from the hint rather than injected as an injection would keep the pure
    /// shadow as a live operand all the way into R1CS, where a witness strip is an ICE because a
    /// hint is not a constraint.
    fn witness_limb(
        &mut self,
        value: ValueId,
        bits: usize,
        index: usize,
        width: usize,
        checked: usize,
    ) -> (ValueId, ValueId) {
        let hint = self.shifted_down(value, bits, index * self.limb_bits());
        let narrowed = self.cast(hint, CastTarget::Int(width));
        let hint_field = self.cast(narrowed, CastTarget::Field);
        let written = self.write_witness(hint_field);
        let limb = self.cast(written, CastTarget::Int(width));
        let limb_field = self.cast(limb, CastTarget::Field);
        self.push(OpCode::Rangecheck {
            value: limb_field,
            max_bits: checked,
        });
        (limb, limb_field)
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
    ///
    /// A constant enters as witnessed constants, one per limb, which are pinned by being constants
    /// and so need no range check.
    fn inject(&mut self, value: ValueId, bits: usize, results: &[ValueId]) {
        let widths = limb_widths(bits, self.limb_bits());
        assert_eq!(
            widths.len(),
            results.len(),
            "ICE: an int{bits} is {} limbs, not {}",
            widths.len(),
            results.len()
        );
        if let Some(pattern) = self.known(value) {
            for (index, (result, width)) in results.iter().zip(&widths).enumerate() {
                let limb = self.int_const(pattern.bit_range(index * self.limb_bits(), *width));
                self.push(OpCode::Cast {
                    result: *result,
                    value: limb,
                    target: CastTarget::WitnessOf,
                });
            }
            return;
        }
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

/// Who bounds a chain's top answer limb at its width.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum TopLimb {
    /// The chain, as it does every other limb.
    Checked,

    /// The caller, which cuts the limb's top bit out with [`Rewriter::sign_bit`] and so bounds it
    /// with the cut's own two range checks.
    CutByCaller,
}

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

/// The operands of an operation one of the gadgets lowers, with the width it is lowered at.
struct Operands {
    /// The width of both operands, which `check_widths` makes the same, and so of every limb list
    /// the gadget splits them into.
    bits: usize,

    /// The left operand as the program names it: the minuend of a difference, the side an ordering
    /// asks is the smaller, and the dividend of a division. One element or limbs, witnessed or
    /// pure.
    lhs: ValueId,

    /// The right operand as the program names it, in the same forms as `lhs`.
    rhs: ValueId,
}

/// What a carry chain leaves behind.
struct Chain {
    /// Each limb of the answer as a field element, range-checked at its own width, the top one
    /// unless [`TopLimb::CutByCaller`] says the caller bounds it.
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
        let Some(Operands { bits, lhs, rhs }) = chained(op, self.types, self.field) else {
            return false;
        };
        let (guard, inner) = self.split_guard(op);

        match inner {
            OpCode::BinaryArithOp { kind, result, .. } => match kind {
                BinaryArithOpKind::UAdd | BinaryArithOpKind::USub => {
                    self.lower_add_sub(*kind, *result, lhs, rhs, bits, guard);
                }
                BinaryArithOpKind::SAdd | BinaryArithOpKind::SSub => {
                    self.lower_signed_add_sub(*kind, *result, lhs, rhs, bits, guard);
                }
                _ => ice_unreachable!("`chained` matches only a sum or a difference"),
            },
            // An ordering cannot fail, so a guard has nothing to withhold from it: the chain holds
            // for any pair of operands, and every limb of either is bounded whatever the guard.
            OpCode::Cmp { kind, result, .. } => {
                let signed = kind.is_signed();
                let chain = self.ordering_chain(signed, lhs, rhs, bits, CarryOut::Witnessed, None);
                self.push(OpCode::Cast {
                    result: *result,
                    value: chain.carry_out.expect("an ordering witnesses its borrow"),
                    target: CastTarget::Int(1),
                });
            }
            OpCode::AssertCmp { kind, .. } => {
                self.ordering_chain(kind.is_signed(), lhs, rhs, bits, CarryOut::One, guard);
            }
            _ => ice_unreachable!("`chained` matches only these"),
        }
        true
    }

    /// The chain whose top borrow is `lhs < rhs`, under the signed reading where `signed`.
    ///
    /// A signed ordering is the unsigned ordering of the operands in offset binary, which is the
    /// top bit of each flipped: that adds `2^(N - 1)` to every signed value, mapping the signed
    /// range onto `[0, 2^N)` in order. Only the top limb holds that bit, so only it changes, and
    /// [`Self::flip_top_bit`] says what the change costs.
    fn ordering_chain(
        &mut self,
        signed: bool,
        lhs: ValueId,
        rhs: ValueId,
        bits: usize,
        out: CarryOut,
        guard: Option<ValueId>,
    ) -> Chain {
        if !signed {
            return self.carry_chain(true, lhs, rhs, bits, out, guard);
        }
        let widths = limb_widths(bits, self.limb_bits());
        let mut lhs = self.chain_limbs(lhs, bits, widths.len());
        let mut rhs = self.chain_limbs(rhs, bits, widths.len());
        let top = widths.len() - 1;
        lhs[top] = self.flip_top_bit(lhs[top], widths[top]);
        rhs[top] = self.flip_top_bit(rhs[top], widths[top]);
        self.carry_chain_over(true, &lhs, &rhs, widths, out, guard)
    }

    /// A signed sum or difference: the unsigned one's limbs, with its overflow read off the sign
    /// bits rather than the carry out of the top limb.
    ///
    /// The chain says `a + b = r + 2^N·c` for a sum and `a - b = r - 2^N·c` for a difference, with
    /// `c` the witnessed carry or borrow out of the top limb. Reading each pattern `x` as
    /// `x - 2^N·s_x` for its sign bit `s_x`, the signed sum is `r + 2^N·(c - s_a - s_b)` and the
    /// signed difference `r - 2^N·(c + s_a - s_b)`. Either fits exactly when it is `r`'s own signed
    /// reading, `r - 2^N·s_r`, so the one check is `c + s_r == s_a + s_b` for a sum and
    /// `c + s_a == s_r + s_b` for a difference: linear in four bits, all of them pinned.
    ///
    /// The answer's sign is cut out of its top limb without the guard, as the operands' are: an
    /// honest carry keeps every limb of the answer in range wherever the operands are, and they
    /// are on every path. Only the check itself is under the guard. The cut bounds that limb at its
    /// width, so the chain does not check it again ([`TopLimb::CutByCaller`]), except where it is
    /// one bit wide and is its own sign with no cut.
    fn lower_signed_add_sub(
        &mut self,
        kind: BinaryArithOpKind,
        result: ValueId,
        lhs: ValueId,
        rhs: ValueId,
        bits: usize,
        guard: Option<ValueId>,
    ) {
        let subtract = kind == BinaryArithOpKind::SSub;
        let widths = limb_widths(bits, self.limb_bits());
        let top = widths.len() - 1;
        let lhs = self.chain_limbs(lhs, bits, widths.len());
        let rhs = self.chain_limbs(rhs, bits, widths.len());
        let lhs_sign = self.sign_bit(lhs[top], widths[top]);
        let rhs_sign = self.sign_bit(rhs[top], widths[top]);

        let top_limb = if widths[top] > 1 {
            TopLimb::CutByCaller
        } else {
            TopLimb::Checked
        };
        let chain = self.carry_chain_bounded(
            subtract,
            &lhs,
            &rhs,
            widths,
            CarryOut::Witnessed,
            guard,
            top_limb,
        );

        let carry = chain
            .carry_out
            .expect("a signed sum or difference witnesses its top carry");
        let answer_top = self.cast(chain.answer[top], CastTarget::Int(chain.widths[top]));
        let answer_sign = self.sign_bit((answer_top, true), chain.widths[top]);

        let (left, right) = if subtract {
            (
                self.bin(BinaryArithOpKind::UAdd, carry, lhs_sign),
                self.bin(BinaryArithOpKind::UAdd, answer_sign, rhs_sign),
            )
        } else {
            (
                self.bin(BinaryArithOpKind::UAdd, carry, answer_sign),
                self.bin(BinaryArithOpKind::UAdd, lhs_sign, rhs_sign),
            )
        };
        self.constrain_equal(left, right, guard);
        self.deliver(result, chain.answer, &chain.widths, bits, guard);
    }

    /// The top bit of a limb `width` bits wide, as a field element that is zero or one.
    ///
    /// The sign-bit primitive. A known limb answers at compile time and a pure one on the pure
    /// side, so neither costs a constraint. A witnessed one is cut at `width - 1` by
    /// [`Self::cut_segment`], which witnesses the bit, range-checks it at one bit, and range-checks
    /// what it leaves of the limb at `width - 1`. Together they pin the bit and bound the limb below
    /// `2^width`: the other value of the bit puts that remainder out of range. A limb is cut once
    /// per block, and a later request reads the same bit ([`Self::signs`]). A one-bit limb is its
    /// own top bit, and is bounded by nothing here.
    fn sign_bit(&mut self, (limb, witnessed): (ValueId, bool), width: usize) -> ValueId {
        if let Some(pattern) = self.known(limb) {
            return self.field_const(match pattern.cast(width).bit(width - 1) {
                Some(true) => self.field.one(),
                _ => self.field.zero(),
            });
        }
        if !witnessed {
            let top = self.shifted_down(limb, width, width - 1);
            let bit = self.cast(top, CastTarget::Int(1));
            return self.cast(bit, CastTarget::Field);
        }
        let whole = self.cast(limb, CastTarget::Field);
        if width == 1 {
            return whole;
        }
        if let Some(bit) = self.signs.get(&(limb, width)) {
            return *bit;
        }
        let pieces = self.cut_segment(limb, whole, width, 0, &[0, width - 1, width]);
        let &[_, (_, 1, bit)] = pieces.as_slice() else {
            ice!("a limb's top bit was cut as {pieces:?}");
        };
        self.signs.insert((limb, width), bit);
        bit
    }

    /// A limb `width` bits wide with its top bit flipped, in the form [`Self::carry_chain_over`]
    /// reads.
    ///
    /// A known limb or a pure one flips on the pure side. A witnessed one is
    /// `x + 2^(w - 1) - s·2^w` for its top bit `s`, which is linear in the [`Self::sign_bit`] cut,
    /// so it costs that cut and nothing more: it is below `2^w` because `x` is and `s` is its top
    /// bit, and its hint is the same combination read back on the pure side.
    fn flip_top_bit(
        &mut self,
        (limb, witnessed): (ValueId, bool),
        width: usize,
    ) -> (ValueId, bool) {
        let top = IntBits::one(width).shifted_left(width - 1);
        if let Some(pattern) = self.known(limb) {
            return (self.int_const(pattern.cast(width).xor(&top)), false);
        }
        if !witnessed {
            let top = self.int_const(top);
            return (self.bin(BinaryArithOpKind::Xor, limb, top), false);
        }
        let sign = self.sign_bit((limb, true), width);
        let whole = self.cast(limb, CastTarget::Field);
        let half = self.field_const(self.field.two_pow(width - 1));
        let raised = self.bin(BinaryArithOpKind::UAdd, whole, half);
        let place = self.field_const(self.field.two_pow(width));
        let cleared = self.bin(BinaryArithOpKind::UMul, sign, place);
        let flipped = self.bin(BinaryArithOpKind::USub, raised, cleared);
        (self.cast(flipped, CastTarget::Int(width)), true)
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
    ///
    /// A limb the gadget knows at compile time is a constant, as every limb of `0 / d` is, and it is
    /// delivered as a **witnessed** integer constant, as [`Self::zero_limb`] makes one.
    fn deliver(
        &mut self,
        result: ValueId,
        answer: Vec<ValueId>,
        widths: &[usize],
        bits: usize,
        guard: Option<ValueId>,
    ) {
        assert!(
            self.is_witness(result),
            "ICE: a gadget delivered a pure result, but every operation it lowers has a witnessed operand"
        );
        let kept: Vec<ValueId> = answer
            .into_iter()
            .zip(widths)
            .map(|(limb, width)| match guard {
                Some(condition) => {
                    let zero = self.field_const(self.field.zero());
                    self.select(condition, limb, zero)
                }
                None => match self.constant_limb(limb, *width) {
                    Some(pattern) => {
                        let constant = self.int_const(pattern);
                        self.cast(constant, CastTarget::WitnessOf)
                    }
                    None => limb,
                },
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

    /// The pattern of an answer limb known at compile time, read at its limb's `width`.
    fn constant_limb(&self, limb: ValueId, width: usize) -> Option<IntBits> {
        match self.ssa.get_const(limb).as_deref() {
            Some(Constant::Int(pattern)) => {
                assert!(
                    BigUint::from(pattern).bits() <= width as u64,
                    "ICE: a known answer limb does not fit its width of {width} bits"
                );
                Some(pattern.cast(width))
            }
            Some(Constant::Field(element)) => {
                let limbs = element.into_bigint().0;
                assert!(
                    IntBits::field_limbs_fit(&limbs, width),
                    "ICE: a known answer limb does not fit its width of {width} bits"
                );
                Some(IntBits::from_field_limbs(&limbs, width))
            }
            _ => None,
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
        let widths = limb_widths(bits, self.limb_bits());
        let lhs = self.chain_limbs(lhs, bits, widths.len());
        let rhs = self.chain_limbs(rhs, bits, widths.len());
        self.carry_chain_over(subtract, &lhs, &rhs, widths, out, guard)
    }

    /// [`Self::carry_chain`] over operands already cut into limbs at `widths`, each limb with
    /// whether it is witnessed, which decides how its hint is read.
    ///
    /// A limb list may mix the two: a limb that is a known constant is a pure value beside the
    /// witnessed ones.
    fn carry_chain_over(
        &mut self,
        subtract: bool,
        lhs: &[(ValueId, bool)],
        rhs: &[(ValueId, bool)],
        widths: Vec<usize>,
        out: CarryOut,
        guard: Option<ValueId>,
    ) -> Chain {
        self.carry_chain_bounded(subtract, lhs, rhs, widths, out, guard, TopLimb::Checked)
    }

    /// [`Self::carry_chain_over`], with `top_limb` saying who bounds the answer's top limb.
    #[allow(clippy::too_many_arguments)]
    fn carry_chain_bounded(
        &mut self,
        subtract: bool,
        lhs: &[(ValueId, bool)],
        rhs: &[(ValueId, bool)],
        widths: Vec<usize>,
        out: CarryOut,
        guard: Option<ValueId>,
        top_limb: TopLimb,
    ) -> Chain {
        assert!(
            subtract || out != CarryOut::One,
            "ICE: a sum whose top carry is forced out has no meaning"
        );
        assert!(
            lhs.len() == widths.len() && rhs.len() == widths.len(),
            "ICE: a chain of {} limbs met operands of {} and {}",
            widths.len(),
            lhs.len(),
            rhs.len()
        );

        let (fold, unfold) = if subtract {
            (BinaryArithOpKind::USub, BinaryArithOpKind::UAdd)
        } else {
            (BinaryArithOpKind::UAdd, BinaryArithOpKind::USub)
        };

        // The carry into the next limb: its hint, which the next hint is computed from, and its
        // column, which the next identity reads.
        let mut carry: Option<(ValueId, ValueId)> = None;
        let mut answer = Vec::with_capacity(widths.len());
        for (index, width) in widths.iter().enumerate() {
            let top = index + 1 == widths.len();
            let a = self.cast(lhs[index].0, CastTarget::Field);
            let b = self.cast(rhs[index].0, CastTarget::Field);
            let mut limb = self.bin(fold, a, b);
            if let Some((_, column)) = carry {
                limb = self.bin(fold, limb, column);
            }

            let place = self.field.two_pow(*width);
            let next = if !top || out == CarryOut::Witnessed {
                let hint = self.carry_hint(
                    subtract,
                    lhs[index],
                    rhs[index],
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

            if !top || top_limb == TopLimb::Checked {
                self.push_guarded(
                    guard,
                    OpCode::Rangecheck {
                        value: limb,
                        max_bits: *width,
                    },
                );
            }
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

    /// [`Self::chain_operand`], each limb paired with whether it is witnessed.
    fn chain_limbs(&mut self, value: ValueId, bits: usize, count: usize) -> Vec<(ValueId, bool)> {
        let witnessed = self.is_witness(value);
        self.chain_operand(value, bits, count)
            .into_iter()
            .map(|limb| (limb, witnessed))
            .collect()
    }
}

/// The operation the carry chain lowers, if `op` is one: a sum, difference or ordering under
/// either reading, guarded or not, with a witnessed operand, past [`widest_cell_sum_bits`].
fn chained(op: &OpCode, types: &FunctionTypeInfo, field: FieldConfig) -> Option<Operands> {
    let (lhs, rhs) = match unguarded(op) {
        OpCode::BinaryArithOp {
            kind:
                BinaryArithOpKind::UAdd
                | BinaryArithOpKind::USub
                | BinaryArithOpKind::SAdd
                | BinaryArithOpKind::SSub,
            lhs,
            rhs,
            ..
        }
        | OpCode::Cmp {
            kind: CmpKind::ULt | CmpKind::SLt,
            lhs,
            rhs,
            ..
        }
        | OpCode::AssertCmp {
            kind: CmpKind::ULt | CmpKind::SLt,
            lhs,
            rhs,
        } => (*lhs, *rhs),
        _ => return None,
    };
    let bits = witnessed_width(types, lhs, rhs)?;
    (bits > widest_cell_sum_bits(field)).then_some(Operands { bits, lhs, rhs })
}

// THE SCHOOLBOOK PRODUCT
// ================================================================================================

/// One operand of the schoolbook, limb by limb.
struct Factor {
    /// Each limb at its own width.
    limbs: Vec<Limb>,

    /// Whether the operand is witnessed, which decides how the hints of its [`Limb::Value`]s are
    /// read. A witnessed operand can still have known limbs, which are [`Limb::Constant`]s.
    witnessed: bool,

    /// The largest value each limb can take: its width's, or a constant limb's own.
    bounds: Vec<BigUint>,
}

impl Factor {
    /// The largest value the whole operand can take, from its limbs' bounds at `limb_bits` apart.
    fn largest(&self, limb_bits: usize) -> BigUint {
        self.bounds
            .iter()
            .enumerate()
            .map(|(index, bound)| bound << (index * limb_bits))
            .sum()
    }

    /// The smallest value the whole operand can take: its known limbs, with every other at zero.
    fn smallest(&self, limb_bits: usize) -> BigUint {
        self.limbs
            .iter()
            .enumerate()
            .filter_map(|(index, limb)| match limb {
                Limb::Constant(pattern) => Some(BigUint::from(pattern) << (index * limb_bits)),
                Limb::Value(_) => None,
            })
            .sum()
    }
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

    /// Add limb `index` of the addend, which lands in the column of the same index.
    Addend(usize),

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

/// The schoolbook plan for operands whose limbs are bounded by `lhs` and `rhs`, plus an `addend`
/// whose limbs are bounded as given (empty for none), at the answer's limb `widths`, on a field
/// whose widest injective width is `injective`.
///
/// The addend is what lets a division state `q·d + r` as one sum: limb `i` of `r` is one more
/// non-negative term in column `i`, so every argument below holds of it unchanged.
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
/// **Where summing would cost more, the product is evaluated instead** (`evaluate`). Each partial
/// product of two witnessed limbs is a row of its own, so the schoolbook pays about `k^2 / 2` of
/// them on two full operands, and evaluating pays `2k - 1` whatever the operands; the caller counts
/// the rows summing would pay with [`witnessed_partial_products`] and evaluates only where both
/// operands are witnessed and that is more. The columns `c_0 .. c_(k-1)` are witnessed whole and the
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
    addend: &[BigUint],
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
        let added = addend
            .get(column)
            .filter(|bound| !bound.is_zero())
            .map(|bound| (Step::Addend(column), bound.clone()));
        let carries = std::mem::take(&mut incoming)
            .into_iter()
            .map(|bound| (Step::Carry, bound));
        let terms: Vec<(Step, BigUint)> =
            products.into_iter().chain(added).chain(carries).collect();

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
    schoolbook_fits(field, bits, false)
}

/// Whether the representation takes an unsigned division or remainder of `bits`-wide operands on
/// `field`: its `q·d + r` is the schoolbook with the remainder added to its columns.
///
/// The bound is every operand's full range, as in [`schoolbook_product_fits`]; a constant only
/// tightens it.
pub fn schoolbook_division_fits(field: FieldConfig, bits: usize) -> bool {
    schoolbook_fits(field, bits, true)
}

/// Whether [`plan_product`] plans a `bits`-wide schoolbook on `field` at full operand ranges, with
/// a full-range addend or without one.
fn schoolbook_fits(field: FieldConfig, bits: usize, with_addend: bool) -> bool {
    let widths = limb_widths(bits, witness_limb_bits(field));
    let bounds = full_bounds(&widths);
    let addend = if with_addend { &bounds[..] } else { &[] };
    plan_product(
        &bounds,
        &bounds,
        addend,
        &widths,
        widest_injective_int_bits(field),
        true,
    )
    .is_some()
}

/// The bound of each limb at `widths` that nothing narrower is known about: its width's own.
fn full_bounds(widths: &[usize]) -> Vec<BigUint> {
    widths
        .iter()
        .map(|width| (BigUint::one() << *width) - 1u8)
        .collect()
}

/// How many multiplications of two witnessed limbs summing `lhs · rhs` costs: one per partial
/// product that lands in the answer, and one per left limb for the overflow check past it.
///
/// Evaluating the product instead costs `2k - 1` of them whatever the operands are, which is fewer
/// wherever every limb is witnessed and `k` is at least two. A known limb multiplies for free, so
/// an operand with enough of them, such as a narrow value widened, is cheaper summed.
///
/// A witnessed guard adds to both sides and is left out of the comparison: evaluating scales each
/// of the `k` left limbs by it, and summing scales each overflow check, which is then one more
/// product. On two dense operands that never makes the choice worse: at two limbs the two tie, and
/// past that evaluating stays ahead.
fn witnessed_partial_products(lhs: &Factor, rhs: &Factor) -> usize {
    let count = lhs.limbs.len();
    let witnessed = |factor: &Factor, index: usize| {
        matches!(factor.limbs[index], Limb::Value(_)) && !factor.bounds[index].is_zero()
    };
    let mut products = 0;
    for left in (0..count).filter(|left| witnessed(lhs, *left)) {
        let right = (0..count).filter(|right| witnessed(rhs, *right));
        let (inside, past): (Vec<usize>, Vec<usize>) =
            right.partition(|right| left + right < count);
        products += inside.len() + usize::from(!past.is_empty());
    }
    products
}

/// A column being accumulated: its field element and its mirror on the pure side, or nothing while
/// it is still zero.
#[derive(Clone, Copy)]
struct Accumulator {
    sum: Option<(ValueId, ValueId)>,
}

/// One schoolbook: `lhs · rhs`, plus `addend` where there is one, either range-checked into limbs
/// or `pinned` to a target's.
#[derive(Clone, Copy)]
struct Schoolbook<'a> {
    lhs: &'a Factor,
    rhs: &'a Factor,
    addend: Option<&'a Factor>,

    /// The target's limbs as field elements, each already bounded at its width.
    pinned: Option<&'a [ValueId]>,

    /// The answer's limb widths, which are the operands' too.
    widths: &'a [usize],

    /// The width the operation is at, for a refusal.
    bits: usize,

    /// What the operation is called, for a refusal.
    operation: &'static str,
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
        let Some(Operands { bits, lhs, rhs }) = multiplied(op, self.types, self.field) else {
            return false;
        };
        let (guard, inner) = self.split_guard(op);
        let OpCode::BinaryArithOp { kind, result, .. } = inner else {
            ice_unreachable!("`multiplied` matches only a binary operation");
        };
        if *kind == BinaryArithOpKind::SMul {
            self.lower_signed_product(*result, lhs, rhs, bits, guard);
            return true;
        }

        let widths = limb_widths(bits, self.limb_bits());
        let lhs = self.factor(lhs, bits, &widths);
        let rhs = self.factor(rhs, bits, &widths);
        let answer = self.schoolbook(
            Schoolbook {
                lhs: &lhs,
                rhs: &rhs,
                addend: None,
                pinned: None,
                widths: &widths,
                bits,
                operation: "multiplication",
            },
            guard,
        );
        self.deliver(*result, answer, &widths, bits, guard);
        true
    }

    /// `lhs · rhs (+ addend)` at `widths`, following the plan [`plan_product`] makes for it, and
    /// returning each limb of the answer as a field element.
    ///
    /// Where the schoolbook is `pinned`, each column's final value is constrained to equal that
    /// limb of the target instead of being range-checked, as a limb of the target is already
    /// bounded. That identity holds as integers for the reason a reduction's does, so it states
    /// `lhs · rhs + addend == target` exactly, and the answer returned is the target's.
    ///
    /// Under a guard a product's checks are all off where it is. A pinned schoolbook is a
    /// division's, whose left operand and addend are witnessed from hints that are zero there, so
    /// every column, carry and overflow term is zero there too and only the pins need the guard.
    fn schoolbook(&mut self, product: Schoolbook<'_>, guard: Option<ValueId>) -> Vec<ValueId> {
        let Schoolbook {
            lhs,
            rhs,
            addend,
            widths,
            bits,
            operation,
            ..
        } = product;
        let evaluate = lhs.witnessed
            && rhs.witnessed
            && witnessed_partial_products(lhs, rhs) > 2 * widths.len() - 1;
        let plan = plan_product(
            &lhs.bounds,
            &rhs.bounds,
            addend.map_or(&[][..], |addend| &addend.bounds),
            widths,
            widest_injective_int_bits(self.field),
            evaluate,
        )
        .unwrap_or_else(|| {
            unsupported_on_this_field(
                format_args!(
                    "a {bits}-bit unsigned {operation} is a schoolbook product of witness limbs, which needs one partial product and the carry it is reduced into to fit a field element"
                ),
                self.field,
            )
        });

        let checks = if product.pinned.is_some() {
            None
        } else {
            guard
        };
        let mut forms = FactorForms::new(widths.len());
        let columns = if plan.evaluated {
            self.evaluate_product(&plan, lhs, rhs, &mut forms, checks)
        } else {
            Vec::new()
        };
        let answer = self.accumulate_columns(&plan, &product, &columns, &mut forms, checks, guard);
        self.check_no_overflow(&plan, lhs, rhs, &mut forms, checks);
        answer
    }

    /// An operand of the schoolbook as its limbs at `bits`, and what each of them can be.
    ///
    /// A constant is cut at compile time, so a limb it does not reach is a known zero and every
    /// partial product and overflow term against it drops out of the plan. This ensures that a
    /// product by a small constant is as cheap as the constant is narrow.
    ///
    /// A witnessed operand can have known limbs too: the limbs above a widened value, a limb a
    /// shift by a constant empties, or an answer limb an earlier gadget knew. Each is taken as the
    /// constant it is, with its value as its bound, which is sound for the reason an injected
    /// constant needs no range check: a constant is pinned by being one.
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

        let (limbs, bounds) = self
            .chain_operand(value, bits, widths.len())
            .into_iter()
            .zip(full_bounds(widths))
            .zip(widths)
            .map(|((limb, full), width)| match self.known(limb) {
                Some(pattern) => {
                    let pattern = pattern.cast(*width);
                    let bound = BigUint::from(&pattern);
                    (Limb::Constant(pattern), bound)
                }
                None => (Limb::Value(limb), full),
            })
            .unzip();
        Factor {
            limbs,
            witnessed: self.is_witness(value),
            bounds,
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
    ///
    /// Where `product` is pinned, the range check that would bound a column's final value is
    /// replaced by its equality with the target's limb, which bounds it as tightly, and a column
    /// that never needed reducing is held to that limb all the same. The range checks are under
    /// `checks` and the equalities under `pins`.
    fn accumulate_columns(
        &mut self,
        plan: &ProductPlan,
        product: &Schoolbook<'_>,
        columns: &[(ValueId, ValueId)],
        forms: &mut FactorForms,
        checks: Option<ValueId>,
        pins: Option<ValueId>,
    ) -> Vec<ValueId> {
        let Schoolbook {
            lhs,
            rhs,
            addend,
            pinned,
            widths,
            ..
        } = *product;
        let hint = CastTarget::Int(plan.hint_bits);
        let mut answer = Vec::with_capacity(widths.len());
        let mut incoming: std::collections::VecDeque<(ValueId, ValueId)> = Default::default();

        for (index, (steps, &width)) in plan.columns.iter().zip(widths).enumerate() {
            let mut outgoing = Vec::new();
            let mut column = Accumulator { sum: None };
            for (position, step) in steps.iter().enumerate() {
                // The one check a pinned column's equality takes the place of.
                let bounds_the_answer = pinned.is_none() || position + 1 < steps.len();
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
                    Step::Column(evaluated) => {
                        let (value, hinted) = columns[evaluated];
                        self.accumulate(&mut column, value, hinted);
                    }
                    Step::Addend(limb) => {
                        let addend = addend.expect("the plan adds limbs of an addend it was given");
                        let value = forms.field(self, Side::Addend, limb, addend);
                        let hinted = forms.pure(self, Side::Addend, limb, addend, &hint);
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
                            None => {
                                if bounds_the_answer {
                                    self.push_guarded(
                                        checks,
                                        OpCode::Rangecheck {
                                            value: sum,
                                            max_bits: width,
                                        },
                                    );
                                }
                            }
                            Some(carry_bits) => {
                                let carry = self.shifted_down(hinted, plan.hint_bits, width);
                                let carry = self.cast(carry, CastTarget::Int(carry_bits));
                                let carry_field = self.cast(carry, CastTarget::Field);
                                let written = self.write_witness(carry_field);
                                self.push_guarded(
                                    checks,
                                    OpCode::Rangecheck {
                                        value: written,
                                        max_bits: carry_bits,
                                    },
                                );
                                let place = self.field_const(self.field.two_pow(width));
                                let scaled = self.bin(BinaryArithOpKind::UMul, written, place);
                                let low = self.bin(BinaryArithOpKind::USub, sum, scaled);
                                if bounds_the_answer {
                                    self.push_guarded(
                                        checks,
                                        OpCode::Rangecheck {
                                            value: low,
                                            max_bits: width,
                                        },
                                    );
                                }

                                let low_hint = self.cast(hinted, CastTarget::Int(width));
                                let low_hint = self.cast(low_hint, hint.clone());
                                column.sum = Some((low, low_hint));
                                outgoing.push((written, self.cast(carry, hint.clone())));
                            }
                        }
                    }
                }
            }
            let value = match column.sum {
                Some((sum, _)) => sum,
                None => self.field_const(self.field.zero()),
            };
            answer.push(match pinned {
                Some(target) => {
                    self.constrain_equal(value, target[index], pins);
                    target[index]
                }
                None => value,
            });
            incoming.extend(outgoing);
        }
        assert!(
            incoming.is_empty(),
            "ICE: the top column of a product carried out of the answer"
        );
        answer
    }

    /// `lhs == rhs` as field elements, where `guard` holds.
    fn constrain_equal(&mut self, lhs: ValueId, rhs: ValueId, guard: Option<ValueId>) {
        let difference = self.bin(BinaryArithOpKind::USub, lhs, rhs);
        let flag = match guard {
            Some(condition) => self.cast(condition, CastTarget::Field),
            None => self.field_const(self.field.one()),
        };
        let zero = self.field_const(self.field.zero());
        self.push(OpCode::Constrain {
            a: flag,
            b: difference,
            c: zero,
        });
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
    Addend,
}

/// The field element and the widened pure hint of each operand limb.
struct FactorForms {
    field: [Vec<Option<ValueId>>; 3],
    pure: [Vec<Option<ValueId>>; 3],
}

impl FactorForms {
    fn new(count: usize) -> Self {
        Self {
            field: std::array::from_fn(|_| vec![None; count]),
            pure: std::array::from_fn(|_| vec![None; count]),
        }
    }

    fn field(
        &mut self,
        rewriter: &mut Rewriter<'_>,
        side: Side,
        index: usize,
        factor: &Factor,
    ) -> ValueId {
        *self.field[side as usize][index]
            .get_or_insert_with(|| rewriter.limb_field(&factor.limbs[index]))
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

/// The product the schoolbook lowers, if `op` is one: a multiplication under either reading,
/// guarded or not, with a witnessed operand, at a width the single cell does not take.
fn multiplied(op: &OpCode, types: &FunctionTypeInfo, field: FieldConfig) -> Option<Operands> {
    past_the_single_cell_product(
        op,
        types,
        field,
        &[BinaryArithOpKind::UMul, BinaryArithOpKind::SMul],
    )
}

/// A binary operation of one of `kinds`, guarded or not, with a witnessed operand, at a width whose
/// product one element cannot hold: past [`single_cell_product_fits`] for an unsigned one, and
/// past [`single_cell_signed_product_fits`] for a signed one.
fn past_the_single_cell_product(
    op: &OpCode,
    types: &FunctionTypeInfo,
    field: FieldConfig,
    kinds: &[BinaryArithOpKind],
) -> Option<Operands> {
    let OpCode::BinaryArithOp { kind, lhs, rhs, .. } = unguarded(op) else {
        return None;
    };
    if !kinds.contains(kind) {
        return None;
    }
    let bits = witnessed_width(types, *lhs, *rhs)?;
    let fits = if kind.is_signed() {
        single_cell_signed_product_fits(field, bits)
    } else {
        single_cell_product_fits(field, bits)
    };
    (!fits).then_some(Operands {
        bits,
        lhs: *lhs,
        rhs: *rhs,
    })
}

// THE DIVISION
// ================================================================================================

impl Rewriter<'_> {
    /// An unsigned division or remainder whose product the single cell cannot hold, returning
    /// `true` if `op` was one and was lowered.
    ///
    /// The quotient `q` and remainder `r` are witnessed as limbs from the pure side's own division,
    /// and pinned by `q·d + r == n` and `r < d`. The first is one schoolbook, the remainder an
    /// addend to its columns and the dividend's limbs the target each column is held to; the
    /// second is the carry chain with its top borrow forced out, which a zero divisor cannot pass.
    /// Together they admit exactly the one pair that Euclidean division gives.
    ///
    /// Under a guard the pure side divides zero by one where it is off, so the quotient and the
    /// remainder are zero there, and so is everything the schoolbook builds from them: its checks
    /// hold whatever the branch not taken left behind, and only `q·d + r == n` and `r < d` are
    /// guarded. The answer is those zeros, range-checked like any other limb, so it needs no
    /// selecting. `d / d` is not witnessed at all, and is selected to zero instead.
    fn lower_division(&mut self, op: &OpCode) -> bool {
        let Some(Operands {
            bits,
            lhs: dividend,
            rhs: divisor,
        }) = divided(op, self.types, self.field)
        else {
            return false;
        };
        let (guard, inner) = self.split_guard(op);
        let OpCode::BinaryArithOp { kind, result, .. } = inner else {
            ice_unreachable!("`divided` matches only a binary operation");
        };

        let widths = limb_widths(bits, self.limb_bits());
        let signed = kind.is_signed();
        let key = (dividend, divisor, guard, signed);
        let (mut quotient, mut remainder) = match self.divisions.remove(&key) {
            Some(answers) => answers,
            // `d / d` is one and `d % d` zero under either reading, wherever `d` is not zero.
            None if dividend == divisor => {
                let (q, r) = self.divide_by_itself(divisor, bits, &widths, guard);
                (Answer::Built(q), Answer::Built(r))
            }
            None if signed => self.divide_signed(dividend, divisor, bits, &widths, guard),
            None => {
                let (q, r) = self.divide(dividend, divisor, bits, &widths, guard);
                (Answer::Built(q), Answer::Built(r))
            }
        };
        let wanted = match kind {
            BinaryArithOpKind::UDiv | BinaryArithOpKind::SDiv => &mut quotient,
            BinaryArithOpKind::URem | BinaryArithOpKind::SRem => &mut remainder,
            _ => ice_unreachable!("`divided` matches only a division or a remainder"),
        };
        let answer = self.answer(wanted, &widths, guard);
        self.divisions.insert(key, (quotient, remainder));
        // A signed answer is the magnitude's negated by a chain checked only under the guard, so
        // off it the answer is selected to zero, as `d / d`'s is.
        let selected = if dividend == divisor || signed {
            guard
        } else {
            None
        };
        self.deliver(*result, answer, &widths, bits, selected);
        true
    }

    /// `answer`'s limbs as the field elements [`Self::deliver`] takes, negating a signed one by
    /// its sign under `guard` and keeping what that builds for the next time it is asked for.
    fn answer(
        &mut self,
        answer: &mut Answer,
        widths: &[usize],
        guard: Option<ValueId>,
    ) -> Vec<ValueId> {
        match answer {
            Answer::Built(limbs) => limbs.clone(),
            Answer::Signed(magnitude, sign) => {
                let negated = self.negate_if(magnitude.clone(), widths, *sign, guard);
                let limbs = self.limb_fields(negated);
                *answer = Answer::Built(limbs.clone());
                limbs
            }
        }
    }

    /// The quotient and the remainder of `dividend / divisor`, each as its limbs' field elements.
    fn divide(
        &mut self,
        dividend: ValueId,
        divisor: ValueId,
        bits: usize,
        widths: &[usize],
        guard: Option<ValueId>,
    ) -> (Vec<ValueId>, Vec<ValueId>) {
        let n = self.factor(dividend, bits, widths);
        let d = self.factor(divisor, bits, widths);
        let pure_dividend = self.pure_whole(dividend, bits);
        let pure_divisor = self.pure_whole(divisor, bits);
        let (q, r) = self.divide_factors(&n, &d, pure_dividend, pure_divisor, bits, widths, guard);
        let quotient = q.limbs.iter().map(|limb| self.limb_field(limb)).collect();
        let remainder = r.limbs.iter().map(|limb| self.limb_field(limb)).collect();
        (quotient, remainder)
    }

    /// The quotient and the remainder of `n / d` as limbs, witnessed from the pure side's division
    /// of `pure_dividend` by `pure_divisor`, which are the two operands' whole values there.
    #[allow(clippy::too_many_arguments)]
    fn divide_factors(
        &mut self,
        n: &Factor,
        d: &Factor,
        mut pure_dividend: ValueId,
        mut pure_divisor: ValueId,
        bits: usize,
        widths: &[usize],
        guard: Option<ValueId>,
    ) -> (Factor, Factor) {
        // What the answers can be, from what the operands can: `q <= n / d` and `r < d`. Only a
        // known limb says more than the width does, and a constant divisor says the most.
        //
        // A divisor of zero has neither answer, and the chain below refuses it, so the bounds only
        // have to hold for a divisor of at least one.
        let limb_bits = self.limb_bits();
        let dividend_max = n.largest(limb_bits);
        let divisor_max = d.largest(limb_bits);
        let divisor_min = d.smallest(limb_bits).max(BigUint::one());
        let quotient_max = &dividend_max / divisor_min;
        let remainder_max = if divisor_max.is_zero() {
            BigUint::zero()
        } else {
            (&divisor_max - 1u8).min(dividend_max)
        };

        if let Some(condition) = guard {
            let condition = self.pure_of(condition, self.is_witness(condition));
            let zero = self.int_const(IntBits::zero(bits));
            let one = self.int_const(IntBits::one(bits));
            pure_dividend = self.select(condition, pure_dividend, zero);
            pure_divisor = self.select(condition, pure_divisor, one);
        }
        let pure_divisor = nonzero_hint_divisor(self, pure_divisor, bits);
        let quotient_hint = self.bin(BinaryArithOpKind::UDiv, pure_dividend, pure_divisor);
        let remainder_hint = self.bin(BinaryArithOpKind::URem, pure_dividend, pure_divisor);
        let q = self.witness_hinted(quotient_hint, bits, widths, &quotient_max);
        let r = self.witness_hinted(remainder_hint, bits, widths, &remainder_max);

        let target: Vec<ValueId> = n.limbs.iter().map(|limb| self.limb_field(limb)).collect();
        self.schoolbook(
            Schoolbook {
                lhs: &q,
                rhs: d,
                addend: Some(&r),
                pinned: Some(&target),
                widths,
                bits,
                operation: "division",
            },
            guard,
        );

        // `r < d` reads only the limbs either side can reach, as above them both are zero.
        let reached = (0..widths.len())
            .rev()
            .find(|index| !r.bounds[*index].is_zero() || !d.bounds[*index].is_zero())
            .map_or(1, |top| top + 1);
        let remainder_limbs = self.chain_limbs_of(&r, reached);
        let divisor_limbs = self.chain_limbs_of(d, reached);
        self.carry_chain_over(
            true,
            &remainder_limbs,
            &divisor_limbs,
            widths[..reached].to_vec(),
            CarryOut::One,
            guard,
        );
        (q, r)
    }

    /// `d / d` and `d % d`, which are one and zero wherever `d` is not zero, so only that is
    /// checked: `0 < d`, the chain with its top borrow forced out.
    fn divide_by_itself(
        &mut self,
        divisor: ValueId,
        bits: usize,
        widths: &[usize],
        guard: Option<ValueId>,
    ) -> (Vec<ValueId>, Vec<ValueId>) {
        let divisor = self.chain_limbs(divisor, bits, widths.len());
        let zero: Vec<(ValueId, bool)> = widths
            .iter()
            .map(|width| (self.int_const(IntBits::zero(*width)), false))
            .collect();
        self.carry_chain_over(true, &zero, &divisor, widths.to_vec(), CarryOut::One, guard);

        let (zero, one) = (
            self.field_const(self.field.zero()),
            self.field_const(self.field.one()),
        );
        let mut quotient = vec![zero; widths.len()];
        quotient[0] = one;
        (quotient, vec![zero; widths.len()])
    }

    /// A pure `hint` of `bits` witnessed as limbs at `widths`, each range-checked no wider than
    /// `max`, the largest value the hint can honestly take, lets that limb be.
    ///
    /// A limb `max` does not reach is not witnessed at all: it is a constant zero, which drops it
    /// and every term against it out of the schoolbook's plan. The limbs are the value's
    /// definition, and the constraints the caller builds on them are what tie them to anything.
    fn witness_hinted(
        &mut self,
        hint: ValueId,
        bits: usize,
        widths: &[usize],
        max: &BigUint,
    ) -> Factor {
        let mut limbs = Vec::with_capacity(widths.len());
        let mut bounds = Vec::with_capacity(widths.len());
        for (index, width) in widths.iter().enumerate() {
            let low = index * self.limb_bits();
            let checked = ((max >> low).bits() as usize).min(*width);
            if checked == 0 {
                limbs.push(Limb::Constant(IntBits::zero(*width)));
                bounds.push(BigUint::zero());
                continue;
            }

            let (limb, _) = self.witness_limb(hint, bits, index, *width, checked);
            limbs.push(Limb::Value(limb));
            bounds.push((BigUint::one() << checked) - 1u8);
        }
        Factor {
            limbs,
            witnessed: true,
            bounds,
        }
    }

    /// A value as one pure integer of `bits`, which is what the pure side divides.
    fn pure_whole(&mut self, value: ValueId, bits: usize) -> ValueId {
        if !self.is_witness(value) {
            return value;
        }
        match self.wide_width(value) {
            Some(_) => {
                let limbs = self.pure_limbs_at(value, bits, bits);
                self.recombine_pure(&limbs, bits)
            }
            None => self.cast(value, CastTarget::ValueOf),
        }
    }

    /// A limb of a [`Factor`] as a field element.
    fn limb_field(&mut self, limb: &Limb) -> ValueId {
        match limb {
            Limb::Value(limb) => self.cast(*limb, CastTarget::Field),
            // A witness limb is never wider than the host word, so its pattern is one host limb.
            Limb::Constant(pattern) => self.field_const(self.field.constant(pattern.limbs()[0])),
        }
    }

    /// The low `count` limbs of a [`Factor`] as the carry chain reads them.
    fn chain_limbs_of(&mut self, factor: &Factor, count: usize) -> Vec<(ValueId, bool)> {
        factor.limbs[..count]
            .iter()
            .map(|limb| match limb {
                Limb::Value(limb) => (*limb, factor.witnessed),
                Limb::Constant(pattern) => (self.int_const(pattern.clone()), false),
            })
            .collect()
    }
}

/// The division the representation lowers, if `op` is one: a division or remainder under either
/// reading, guarded or not, with a witnessed operand, at a width whose product the single cell
/// cannot hold.
///
/// That is where the single-cell lowering's `q·d` could wrap: it forms the product in one element,
/// and a quotient range-checked at the width times a divisor of the width passes the modulus from
/// the width [`single_cell_product_fits`] refuses.
fn divided(op: &OpCode, types: &FunctionTypeInfo, field: FieldConfig) -> Option<Operands> {
    past_the_single_cell_product(
        op,
        types,
        field,
        &[
            BinaryArithOpKind::UDiv,
            BinaryArithOpKind::URem,
            BinaryArithOpKind::SDiv,
            BinaryArithOpKind::SRem,
        ],
    )
}

// THE SHIFT
// ================================================================================================

impl Rewriter<'_> {
    /// A shift the single cell cannot hold, returning `true` if `op` was one and was lowered.
    ///
    /// An amount known at compile time is a relabelling of bits, which [`Self::shift_by_constant`]
    /// performs by cutting the operand where its bits land on a limb boundary. Any other amount is
    /// [`Self::shift_by_amount`]'s: a split of every limb at the amount within a limb, and a barrel
    /// over the bits of how many whole limbs it moves. A signed right shift is either, with the
    /// vacated bits filled by [`Self::fill_with_sign`] or by complements around the shift.
    fn lower_shift(&mut self, op: &OpCode) -> bool {
        let Some(Operands { bits, lhs, rhs }) = shifted(op, self.types, self.field) else {
            return false;
        };
        let (guard, inner) = self.split_guard(op);
        let OpCode::BinaryArithOp { kind, result, .. } = inner else {
            ice_unreachable!("`shifted` matches only a binary operation");
        };
        if !limb_shift_fits(self.field) {
            unsupported_on_this_field(
                format_args!(
                    "a {bits}-bit witness shift splits each witness limb at its amount, which needs a limb times a power of two below the limb to fit a field element"
                ),
                self.field,
            );
        }

        // A left shift is one map on the pattern under either reading, so a signed one is this.
        let left = matches!(kind, BinaryArithOpKind::UShl | BinaryArithOpKind::SShl);
        let signed = *kind == BinaryArithOpKind::SShr;
        // A pure operand shifted by a known witness amount has no witness to cut, so it takes the
        // general route, whose every limb is witnessed by the split.
        match self.known_amount(rhs).filter(|_| self.is_witness(lhs)) {
            Some(amount) => match amount.to_usize().filter(|amount| *amount < bits) {
                Some(amount) => {
                    let widths = self.shift_widths(*result, bits);
                    let (mut answer, sign) =
                        self.shift_by_constant(lhs, bits, &widths, left, amount, signed);
                    if let Some(sign) = sign {
                        self.fill_with_sign(&mut answer, &widths, sign, bits, amount);
                    }
                    self.deliver(*result, answer, &widths, bits, None);
                }
                None => self.shift_out_of_range(*result, bits, guard),
            },
            None => {
                let widths = limb_widths(bits, self.limb_bits());
                let answer = self.shift_by_amount(lhs, rhs, bits, left, signed, guard);
                self.deliver(*result, answer, &widths, bits, None);
            }
        }
        true
    }

    /// The whole amount, where every limb of it is known at compile time.
    fn known_amount(&self, amount: ValueId) -> Option<BigUint> {
        let mut value = BigUint::zero();
        for (index, limb) in self.limbs(amount).into_iter().enumerate() {
            value += BigUint::from(&self.known(limb)?) << (index * self.limb_bits());
        }
        Some(value)
    }

    /// A shift by a known amount at or past the width, which the program is refused for wherever
    /// the guard is on. The answer is zero, so it stays in range where the guard is off.
    fn shift_out_of_range(&mut self, result: ValueId, bits: usize, guard: Option<ValueId>) {
        let (zero, one) = (
            self.field_const(self.field.zero()),
            self.field_const(self.field.one()),
        );
        self.push_guarded(
            guard,
            OpCode::AssertCmp {
                kind: CmpKind::Eq,
                lhs: zero,
                rhs: one,
            },
        );
        let widths = self.shift_widths(result, bits);
        let answer = vec![zero; widths.len()];
        self.deliver(result, answer, &widths, bits, None);
    }

    /// The limb widths a value of `bits` is held at here: its limbs', or its own where it is still
    /// one element.
    fn shift_widths(&self, value: ValueId, bits: usize) -> Vec<usize> {
        if self.limbs(value).len() > 1 {
            limb_widths(bits, self.limb_bits())
        } else {
            vec![bits]
        }
    }

    /// `lhs` shifted by an `amount` below `bits` that is known at compile time.
    ///
    /// Every bit of the answer is a bit of the operand, so the operand is cut wherever a run of
    /// its bits stops landing inside one limb of the answer: at its own limb boundaries, at the
    /// bits that land on the answer's, and where the discarded bits begin. Each piece of a limb
    /// that is cut is witnessed and range-checked at its own width except the lowest, which is what
    /// the others leave of the limb and is range-checked too. The limb is below `2^w` and so are
    /// the pieces recombined, so the difference is either the lowest piece as an integer or an
    /// element near `p`, which that range check rejects. An answer limb is then a sum of pieces.
    ///
    /// A limb nothing cuts costs nothing, so an amount that is a whole number of limbs is free.
    ///
    /// The answer is one field element per limb of `answer_widths`, which are the result's. Where
    /// `signed` asks for it and the shift moves anything, the operand's sign is handed back too,
    /// for [`Self::fill_with_sign`]: the piece a cut at its top bit leaves, or, where the top limb
    /// is known, its top bit.
    fn shift_by_constant(
        &mut self,
        lhs: ValueId,
        bits: usize,
        answer_widths: &[usize],
        left: bool,
        amount: usize,
        signed: bool,
    ) -> (Vec<ValueId>, Option<Sign>) {
        let segments = self.limbs(lhs);
        let segment_widths = self.shift_widths(lhs, bits);
        let signed = signed && amount > 0;
        let top_width = segment_widths[segment_widths.len() - 1];
        let known_top = self.known(segments[segments.len() - 1]);

        // A known top limb's sign is read off its pattern instead. Cut, it would leave a known limb
        // of the answer as arithmetic on constants, which `deliver` hands on as a pure value.
        let mut cuts = BTreeSet::from([0, bits, if left { bits - amount } else { amount }]);
        if signed && known_top.is_none() {
            cuts.insert(bits - 1);
        }
        let mut start = 0usize;
        for width in &segment_widths {
            cuts.insert(start);
            start += width;
        }
        let mut start = 0usize;
        for width in answer_widths {
            let source = if left {
                start.checked_sub(amount)
            } else {
                Some(start + amount)
            };
            if let Some(source) = source.filter(|source| *source < bits) {
                cuts.insert(source);
            }
            start += width;
        }

        // Each piece as where it starts in the operand, how wide it is, and its field element.
        let mut pieces: Vec<(usize, usize, ValueId)> = Vec::new();
        let mut start = 0;
        for (segment, width) in segments.into_iter().zip(&segment_widths) {
            let end = start + width;
            let bounds: Vec<usize> = cuts.range(start..=end).copied().collect();
            let whole = self.cast(segment, CastTarget::Field);
            if bounds.len() == 2 {
                pieces.push((start, *width, whole));
            } else {
                pieces.extend(self.cut_segment(segment, whole, *width, start, &bounds));
            }
            start = end;
        }

        let mut answer = Vec::with_capacity(answer_widths.len());
        let mut start = 0;
        for width in answer_widths {
            let end = start + width;
            let mut sum = None;
            for (source, piece_width, piece) in &pieces {
                let landed = if left {
                    source + amount
                } else if let Some(landed) = source.checked_sub(amount) {
                    landed
                } else {
                    continue;
                };
                if landed < start || landed >= end {
                    continue;
                }
                assert!(
                    landed + piece_width <= end,
                    "ICE: a piece of a shifted operand straddles a limb of the answer"
                );
                let scaled = if landed == start {
                    *piece
                } else {
                    let place = self.field_const(self.field.two_pow(landed - start));
                    self.bin(BinaryArithOpKind::UMul, *piece, place)
                };
                sum = Some(match sum {
                    None => scaled,
                    Some(acc) => self.bin(BinaryArithOpKind::UAdd, acc, scaled),
                });
            }
            answer.push(sum.unwrap_or_else(|| self.field_const(self.field.zero())));
            start = end;
        }

        let sign = signed.then(|| match known_top {
            Some(pattern) => Sign::Known(pattern.cast(top_width).bit(top_width - 1) == Some(true)),
            None => {
                let (_, _, bit) = pieces
                    .iter()
                    .find(|(source, width, _)| *source == bits - 1 && *width == 1)
                    .unwrap_or_else(|| ice!("the operand's top bit was not cut out"));
                Sign::Bit(*bit, true)
            }
        });
        (answer, sign)
    }

    /// One witnessed `segment` of `width` bits starting at `start`, cut at `bounds`, which run from
    /// `start` to its end, as each piece's start, width and field element.
    ///
    /// A known segment is cut at compile time and costs nothing.
    fn cut_segment(
        &mut self,
        segment: ValueId,
        whole: ValueId,
        width: usize,
        start: usize,
        bounds: &[usize],
    ) -> Vec<(usize, usize, ValueId)> {
        let known = self.known(segment);
        let pure = if known.is_none() {
            Some(self.cast(segment, CastTarget::ValueOf))
        } else {
            None
        };

        let mut pieces = Vec::with_capacity(bounds.len() - 1);
        let mut lowest = whole;
        for window in bounds.windows(2).skip(1) {
            let (low, piece_width) = (window[0] - start, window[1] - window[0]);
            let piece = match (&known, pure) {
                (Some(pattern), _) => self.limb_const(&pattern.bit_range(low, piece_width)),
                (None, Some(pure)) => self.witness_window(pure, width, low, piece_width),
                (None, None) => ice_unreachable!("a segment is known or has a pure shadow"),
            };
            let place = self.field_const(self.field.two_pow(low));
            let scaled = self.bin(BinaryArithOpKind::UMul, piece, place);
            lowest = self.bin(BinaryArithOpKind::USub, lowest, scaled);
            pieces.push((window[0], piece_width, piece));
        }

        let lowest_width = bounds[1] - start;
        if known.is_none() {
            self.push(OpCode::Rangecheck {
                value: lowest,
                max_bits: lowest_width,
            });
        }
        pieces.insert(0, (start, lowest_width, lowest));
        pieces
    }

    /// Bits `offset..offset + width` of a pure `value` of `bits`, written to a witness column and
    /// range-checked at `width`, as a field element.
    fn witness_window(
        &mut self,
        value: ValueId,
        bits: usize,
        offset: usize,
        width: usize,
    ) -> ValueId {
        let hint = self.shifted_down(value, bits, offset);
        let narrowed = self.cast(hint, CastTarget::Int(width));
        let hint_field = self.cast(narrowed, CastTarget::Field);
        let column = self.write_witness(hint_field);
        self.push(OpCode::Rangecheck {
            value: column,
            max_bits: width,
        });
        column
    }

    /// `lhs` shifted by an amount not known at compile time.
    ///
    /// **The amount** is `n = q·h + r`, with `r` below the limb width `h` and `q` the number of whole
    /// limbs it moves. Its low limb is witnessed as `h·Σ e_t·2^t + r` with every `e_t` a bit and `r`
    /// read out of the powers-of-two table beside its power `f = 2^r`, which bounds `r` below `h`;
    /// its other limbs are held to zero. With `s` bits of `q`, that is `n < h·2^s` already, and
    /// `n < bits` exactly where `h·2^s` is the width; elsewhere `bits - 1 - n` is range-checked as
    /// well.
    ///
    /// **Each limb** `x_i` is split as `x_i·m = hi_i·2^h + lo_i` with both halves range-checked,
    /// where `m` is `f` for `<<` and `2^(h - r)` for `>>`, which is pinned by `m·f == 2^h`. As `x_i·m`
    /// is below `2^2h`, which [`limb_shift_fits`] keeps inside an element, that split is unique.
    /// For `<<`, `lo_i` is `x_i`'s low `h - r` bits moved up by `r` and `hi_i` its top `r` bits; for
    /// `>>`, `hi_i` is `x_i`'s bits above `r` and `lo_i` its low `r` bits moved to the top. Answer
    /// limb `j` before the whole limbs move is then `lo_j + hi_(j-1)` or `hi_j + lo_(j+1)`: the two
    /// halves cover disjoint bits, so neither sum carries.
    ///
    /// **The whole limbs** move by `q` through a barrel, one stage per bit `e_t` moving every limb
    /// `2^t` places, each limb of a stage `a + e_t·(b - a)`. A `<<` whose top limb is narrower than
    /// `h` then cuts it to its width. Its high half would be cut away there too, so that limb is not
    /// split: its `x·m` goes into the barrel whole, the answer's top limb is then below `2^(w + h)`,
    /// and the cut keeps its low `w` bits and range-checks the `h` above them.
    ///
    /// Under a guard a witnessed amount's lowest limb is zero where the guard is off, so every check
    /// here holds there whatever the branch not taken left behind, and the operand's limbs are in
    /// range on every path. Its other limbs are held to zero only under the guard, which is the one
    /// check here that needs it.
    ///
    /// A pure amount is decomposed on the pure side instead, by [`Self::pure_shift_amount`], and
    /// every product above is by a constant.
    ///
    /// **A sign-filling right shift**, where `signed`, is the operand [`Self::complement`]ed by its
    /// sign, shifted as above, and the answer complemented back. For a negative operand that is
    /// `!(!lhs >> n)`, and `!lhs` is not negative, so its unsigned shift is its arithmetic one and
    /// complementing back fills the vacated bits with ones. It costs the sign's cut and two
    /// products per limb more than the unsigned shift.
    ///
    /// The answer is one field element per limb.
    fn shift_by_amount(
        &mut self,
        lhs: ValueId,
        rhs: ValueId,
        bits: usize,
        left: bool,
        signed: bool,
        guard: Option<ValueId>,
    ) -> Vec<ValueId> {
        let limb_bits = self.limb_bits();
        let widths = limb_widths(bits, limb_bits);
        let count = widths.len();
        let stages = ceil_log2(count);
        let (e, factor, factor_hint) = self.shift_amount(rhs, bits, stages, left, guard);

        // The split of every limb, as its two halves' field elements.
        let two_pow_limb = self.field_const(self.field.two_pow(limb_bits));
        let hint_width = CastTarget::Int(2 * limb_bits);
        let mut operand = self.chain_limbs(lhs, bits, count);
        let sign = signed.then(|| self.operand_sign(&operand, &widths));
        if let Some(sign) = sign {
            operand = operand
                .into_iter()
                .zip(&widths)
                .map(|(limb, width)| self.complement(limb, *width, sign))
                .collect();
        }
        let mut halves = Vec::with_capacity(count);
        let top_width = widths[count - 1];
        let narrow_top = left && top_width < limb_bits;
        for (index, (&(limb, witnessed), width)) in operand.iter().zip(&widths).enumerate() {
            if self.known(limb).is_some_and(|pattern| pattern.is_zero()) {
                let zero = self.field_const(self.field.zero());
                halves.push((zero, zero));
                continue;
            }
            let limb_field = self.cast(limb, CastTarget::Field);
            let product = self.bin(BinaryArithOpKind::UMul, limb_field, factor);

            // `<<` moves at most `h - 1` places, so it pushes one bit fewer out of the limb. A
            // narrow top limb's high half is discarded, and the cut below splits it anyway.
            let high_bits = if left { width - 1 } else { *width };
            if high_bits == 0 || (narrow_top && index + 1 == count) {
                halves.push((self.field_const(self.field.zero()), product));
                continue;
            }
            let pure = self.pure_of(limb, witnessed);
            let pure = self.cast(pure, hint_width.clone());
            let pure = self.bin(BinaryArithOpKind::UMul, pure, factor_hint);
            let high = self.witness_window(pure, 2 * limb_bits, limb_bits, high_bits);
            let scaled = self.bin(BinaryArithOpKind::UMul, high, two_pow_limb);
            let low = self.bin(BinaryArithOpKind::USub, product, scaled);
            self.push(OpCode::Rangecheck {
                value: low,
                max_bits: limb_bits,
            });
            halves.push((high, low));
        }

        // The answer's limbs before the whole limbs move.
        let zero = self.field_const(self.field.zero());
        let mut limbs: Vec<ValueId> = (0..count)
            .map(|index| {
                let (own, neighbour) = if left {
                    (
                        halves[index].1,
                        index.checked_sub(1).map(|below| halves[below].0),
                    )
                } else {
                    (halves[index].0, halves.get(index + 1).map(|above| above.1))
                };
                match neighbour {
                    Some(neighbour) => self.bin(BinaryArithOpKind::UAdd, own, neighbour),
                    None => own,
                }
            })
            .collect();

        for (stage, bit) in e.iter().enumerate() {
            let step = 1 << stage;
            limbs = (0..count)
                .map(|index| {
                    let stay = limbs[index];
                    let from = if left {
                        index.checked_sub(step).map(|from| limbs[from])
                    } else {
                        limbs.get(index + step).copied()
                    };
                    let from = from.unwrap_or(zero);
                    let moved = self.bin(BinaryArithOpKind::USub, from, stay);
                    let moved = self.bin(BinaryArithOpKind::UMul, *bit, moved);
                    self.bin(BinaryArithOpKind::UAdd, stay, moved)
                })
                .collect();
        }

        // The top limb of a `<<` has to be below its own width. It was never split, so it can be
        // `x_top·f + hi`, below `2^(w + h)`, and everything above `w` is cut away here.
        if narrow_top {
            let top = limbs[count - 1];
            let reach = top_width + limb_bits;
            let pure = self.cast(top, CastTarget::ValueOf);
            let pure = self.cast(pure, CastTarget::Int(reach));
            let high = self.witness_window(pure, reach, top_width, limb_bits);
            let place = self.field_const(self.field.two_pow(top_width));
            let scaled = self.bin(BinaryArithOpKind::UMul, high, place);
            let low = self.bin(BinaryArithOpKind::USub, top, scaled);
            self.push(OpCode::Rangecheck {
                value: low,
                max_bits: top_width,
            });
            limbs[count - 1] = low;
        }

        match sign {
            Some(sign) => {
                let answer = self.answer_limbs(limbs, &widths);
                let restored = answer
                    .into_iter()
                    .zip(&widths)
                    .map(|(limb, width)| self.complement(limb, *width, sign))
                    .collect();
                self.limb_fields(restored)
            }
            None => limbs,
        }
    }

    /// The amount of a [`Self::shift_by_amount`], as the `stages` bits of how many whole limbs it
    /// moves and the multiplier that splits a limb, as a field element and as a pure hint at twice
    /// the limb width.
    fn shift_amount(
        &mut self,
        rhs: ValueId,
        bits: usize,
        stages: usize,
        left: bool,
        guard: Option<ValueId>,
    ) -> (Vec<ValueId>, ValueId, ValueId) {
        let limb_bits = self.limb_bits();
        let log_limb = limb_bits.trailing_zeros() as usize;
        assert!(
            limb_bits.is_power_of_two() && log_limb <= max_pow2_table_size(self.field),
            "ICE: a {limb_bits}-bit limb has no powers-of-two table to split it at an amount"
        );

        let amount_bits = int_width(self.types.get_value_type(rhs))
            .unwrap_or_else(|| ice!("a shift by a non-integer amount"));
        if !self.is_witness(rhs) {
            return self.pure_shift_amount(rhs, amount_bits, bits, stages, left, guard);
        }
        let (low, rest, low_bits) = if amount_bits > multi_cell_int_bits(self.field) {
            let widths = limb_widths(amount_bits, limb_bits);
            let limbs = self.operand_limbs(rhs, widths.len());
            (limbs[0], limbs[1..].to_vec(), widths[0])
        } else {
            (rhs, Vec::new(), amount_bits)
        };

        // Every limb above the lowest is zero, or the amount is past any width.
        let zero = self.field_const(self.field.zero());
        for limb in rest {
            let limb = self.cast(limb, CastTarget::Field);
            self.constrain_equal(limb, zero, guard);
        }

        // Where the guard is off the amount is zero, which every check below admits.
        let mut low_field = self.cast(low, CastTarget::Field);
        let mut low_pure = self.cast(low, CastTarget::ValueOf);
        if let Some(condition) = guard {
            let flag = self.cast(condition, CastTarget::Field);
            low_field = self.bin(BinaryArithOpKind::UMul, flag, low_field);
            let pure_condition = self.pure_of(condition, self.is_witness(condition));
            let pure_zero = self.int_const(IntBits::zero(low_bits));
            low_pure = self.select(pure_condition, low_pure, pure_zero);
        }
        let hints = self.amount_hints(low_pure, low_bits, stages);

        // `n = h·Σ e_t·2^t + r`, each `e_t` a bit and `r` a key of the table. A bit the amount is too
        // narrow to hold is a known zero.
        let within_field = self.cast(hints.within, CastTarget::Field);
        let within_column = self.write_witness(within_field);
        let mut sum = within_column;
        let mut e = Vec::with_capacity(stages);
        for (stage, hint) in hints.stages.into_iter().enumerate() {
            let Some(hint) = hint else {
                e.push(zero);
                continue;
            };
            let hint = self.cast(hint, CastTarget::Field);
            let bit = self.write_witness(hint);
            self.push(OpCode::Rangecheck {
                value: bit,
                max_bits: 1,
            });
            let place = self.field_const(self.field.two_pow(log_limb + stage));
            let scaled = self.bin(BinaryArithOpKind::UMul, bit, place);
            sum = self.bin(BinaryArithOpKind::UAdd, sum, scaled);
            e.push(bit);
        }
        self.constrain_equal(low_field, sum, None);

        // Where `h·2^t` passes the width the decomposition does not bound the amount by it, unless
        // the amount's own width already does.
        if limb_bits << stages != bits && !amount_type_stays_below(amount_bits, bits) {
            let largest = self.field_const(self.field.constant((bits - 1) as u64));
            let headroom = self.bin(BinaryArithOpKind::USub, largest, low_field);
            self.push(OpCode::Rangecheck {
                value: headroom,
                max_bits: ceil_log2(bits),
            });
        }

        // `f = 2^r`, which the table pins against `r` and so bounds `r` below `h`. Unguarded, as
        // `r` is already zero where the guard is off.
        let power_field = self.cast(hints.power, CastTarget::Field);
        let power = self.write_witness(power_field);
        let one_flag = self.field_const(self.field.one());
        self.push(OpCode::Lookup {
            target: LookupTarget::Pow2(log_limb as u8),
            args: vec![within_column, power],
            flag: one_flag,
        });
        if left {
            return (e, power, hints.power);
        }

        // `2^(h - r)`, pinned by its product with `f`: `2^h` is not zero, so neither is `f`, and
        // the cofactor is its unique quotient.
        let cofactor_hint = self.cofactor_hint(hints.power);
        let cofactor_field = self.cast(cofactor_hint, CastTarget::Field);
        let cofactor = self.write_witness(cofactor_field);
        let two_pow_limb = self.field_const(self.field.two_pow(limb_bits));
        self.push(OpCode::Constrain {
            a: power,
            b: cofactor,
            c: two_pow_limb,
        });
        (e, cofactor, cofactor_hint)
    }

    /// The pure side of an amount `n = q·h + r` of `amount_bits`, which both routes decompose the
    /// same way: each of the `stages` bits of `q` as a one-bit integer, or [`None`] where the amount
    /// is too narrow to hold it; `r`; and `2^r` at twice the limb width.
    ///
    /// Only the low `log2(h) + stages` bits are read, so an amount past what those spell is taken
    /// modulo `h·2^stages`, which is for the check the caller builds to rule out.
    fn amount_hints(&mut self, amount: ValueId, amount_bits: usize, stages: usize) -> AmountHints {
        let limb_bits = self.limb_bits();
        let log_limb = limb_bits.trailing_zeros() as usize;
        let stages = (0..stages)
            .map(|stage| {
                (log_limb + stage < amount_bits).then(|| {
                    let bit = self.shifted_down(amount, amount_bits, log_limb + stage);
                    self.cast(bit, CastTarget::Int(1))
                })
            })
            .collect();
        let within = self.cast(amount, CastTarget::Int(log_limb));
        let hint_bits = 2 * limb_bits;
        let one = self.int_const(IntBits::one(hint_bits));
        let within_wide = self.cast(within, CastTarget::Int(hint_bits));
        let power = self.bin(BinaryArithOpKind::UShl, one, within_wide);
        AmountHints {
            stages,
            within,
            power,
        }
    }

    /// `2^(h - r)` on the pure side, from `2^r` at twice the limb width.
    fn cofactor_hint(&mut self, power: ValueId) -> ValueId {
        let limb_bits = self.limb_bits();
        let two_pow_limb = self.two_pow_const(2 * limb_bits, limb_bits);
        self.bin(BinaryArithOpKind::UDiv, two_pow_limb, power)
    }

    /// [`Self::shift_amount`] for a pure amount.
    ///
    /// Such an amount is known wherever the constraints are built, so nothing about it is
    /// witnessed: its check is [`emit_pure_shift_amount_check`]'s comparison, and its bits and its
    /// multiplier are pure values, which make each limb's split and each stage of the barrel
    /// linear. It is read whole, as the pure side holds it whole at any width.
    ///
    /// Under a guard only the comparison is guarded, and the amount is not zeroed where the guard
    /// is off. Nothing else here reads the amount except through hints taken from it, so the split
    /// and the barrel hold for any amount the branch not taken left behind, and the answer there is
    /// the operand shifted by that amount modulo `h·2^stages`.
    fn pure_shift_amount(
        &mut self,
        rhs: ValueId,
        amount_bits: usize,
        bits: usize,
        stages: usize,
        left: bool,
        guard: Option<ValueId>,
    ) -> (Vec<ValueId>, ValueId, ValueId) {
        emit_pure_shift_amount_check(self, guard, rhs, amount_bits, bits);

        let hints = self.amount_hints(rhs, amount_bits, stages);
        let zero = self.field_const(self.field.zero());
        let e = hints
            .stages
            .into_iter()
            .map(|bit| match bit {
                Some(bit) => self.cast(bit, CastTarget::Field),
                None => zero,
            })
            .collect();
        let multiplier = if left {
            hints.power
        } else {
            self.cofactor_hint(hints.power)
        };
        let multiplier_field = self.cast(multiplier, CastTarget::Field);
        (e, multiplier_field, multiplier)
    }
}

/// The pure side of a shift amount, from [`Rewriter::amount_hints`].
struct AmountHints {
    /// Each bit of how many whole limbs the amount moves, as a one-bit integer, or [`None`] where
    /// the amount is too narrow to hold it.
    stages: Vec<Option<ValueId>>,

    /// The amount within a limb, `r`, at `log2(h)` bits.
    within: ValueId,

    /// `2^r`, at twice the limb width.
    power: ValueId,
}

/// Whether the limb-wise shift takes a shift on `field`: a limb times a power of two no greater than
/// the limb's own place, `hi·2^h + lo`, has to be one element for the split to be unique.
pub fn limb_shift_fits(field: FieldConfig) -> bool {
    limb_shift_fits_modulus(&field_modulus(field))
}

/// The body of [`limb_shift_fits`], stated against the modulus so that a field this compiler cannot
/// yet be configured for can still be checked.
fn limb_shift_fits_modulus(modulus: &BigInt) -> bool {
    2 * limb_bits_for_modulus(modulus, LimbBudget::DEFAULT)
        <= widest_injective_int_bits_for_modulus(modulus)
}

/// The shift the representation lowers, if `op` is one, under either reading, guarded or not, with a
/// witnessed operand, at a width the single cell does not take.
fn shifted(op: &OpCode, types: &FunctionTypeInfo, field: FieldConfig) -> Option<Operands> {
    let OpCode::BinaryArithOp { kind, lhs, rhs, .. } = unguarded(op) else {
        return None;
    };
    let left = match kind {
        BinaryArithOpKind::UShl | BinaryArithOpKind::SShl => true,
        BinaryArithOpKind::UShr | BinaryArithOpKind::SShr => false,
        _ => return None,
    };
    let bits = witnessed_width(types, *lhs, *rhs)?;
    (!single_cell_shift_fits(field, bits, left)).then_some(Operands {
        bits,
        lhs: *lhs,
        rhs: *rhs,
    })
}

// THE SIGN AND THE MAGNITUDE
// ================================================================================================

/// A sign bit as the signed gadgets read it.
#[derive(Clone, Copy, Debug)]
enum Sign {
    /// Known at compile time.
    Known(bool),

    /// A field element that is zero or one, with whether it is witnessed.
    Bit(ValueId, bool),
}

impl Rewriter<'_> {
    /// A signed product the single cell cannot hold.
    ///
    /// The unsigned schoolbook over the operands' magnitudes, whose own overflow check holds the
    /// product below `2^N`, then [`Self::check_fits`] under the sign the product takes, and the
    /// product negated by that sign. Every check past the operands is under the guard but the cut
    /// of the product's top bit, which reads a product that is zero where the guard is off, and the
    /// answer is zero there too, as the unsigned product's is.
    fn lower_signed_product(
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
        let product = self.schoolbook(
            Schoolbook {
                lhs: &lhs_factor,
                rhs: &rhs_factor,
                addend: None,
                pinned: None,
                widths: &widths,
                bits,
                operation: "multiplication",
            },
            guard,
        );

        // Off the guard the schoolbook's checks are off, and a top column the branch not taken
        // overflowed is past its width. The cast to a limb's width that the negation reads it
        // through truncates on the pure side while the constraints carry the column whole, so the
        // negation's products would disagree with their own columns there. The product is zero
        // there instead.
        let product = match guard {
            Some(condition) => {
                let zero = self.field_const(self.field.zero());
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
    /// The unsigned division of the operands' magnitudes, which refuses a zero divisor, then
    /// Noir's truncation toward zero: the quotient takes the sign of the two operands' signs
    /// combined, and the remainder the dividend's. The quotient's magnitude is held to fit under
    /// its sign for both answers, as the model refuses `INT_MIN % -1` along with `INT_MIN / -1`.
    /// The remainder's always fits, being below the divisor's magnitude and so below `2^(N - 1)`.
    fn divide_signed(
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

    /// Fill the bits a right shift by a known `amount` above zero vacates with the operand's
    /// `sign`, which [`Self::shift_by_constant`] cut out, in an `answer` whose limbs are at
    /// `widths`.
    ///
    /// The unsigned answer has those bits clear, so filling them is adding `s` times each limb's
    /// share of `2^N - 2^(N - amount)`: linear in the sign, and below each limb's width still.
    ///
    /// A sign known to be set fills a known limb at compile time. Added as arithmetic, the fill
    /// would turn a constant limb, which [`Self::deliver`] hands on as a witnessed constant, into
    /// a pure value where the result is witnessed.
    fn fill_with_sign(
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

    /// The magnitude of a `bits`-wide operand read as signed, as limbs at `widths` each with
    /// whether it is witnessed, its sign, and, for a pure operand, the whole magnitude the limbs
    /// were cut from, which a division's hint reads.
    ///
    /// `|INT_MIN|` is `2^(N - 1)`, which `N` unsigned bits hold, so every magnitude fits its
    /// width. A constant's is taken at compile time, so the limbs a constant factor does not reach
    /// stay known zeros, and a pure operand's is taken on the pure side, at no cost. A witnessed
    /// one is its limbs [`Self::negate_if`] by its sign.
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
    fn operand_sign(&mut self, limbs: &[(ValueId, bool)], widths: &[usize]) -> Sign {
        let top = widths.len() - 1;
        self.top_bit(limbs[top], widths[top])
    }

    /// The top bit of a limb `width` bits wide, known where the limb is, and otherwise the
    /// [`Self::sign_bit`] cut.
    fn top_bit(&mut self, (limb, witnessed): (ValueId, bool), width: usize) -> Sign {
        match self.known(limb) {
            Some(pattern) => Sign::Known(pattern.cast(width).bit(width - 1) == Some(true)),
            None => Sign::Bit(self.sign_bit((limb, witnessed), width), witnessed),
        }
    }

    /// A sign as a field element that is zero or one.
    fn sign_field(&self, sign: Sign) -> ValueId {
        match sign {
            Sign::Known(true) => self.field_const(self.field.one()),
            Sign::Known(false) => self.field_const(self.field.zero()),
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
                let one = self.field_const(self.field.one());
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

    /// A limb `width` bits wide complemented where `sign` is one and kept where it is zero.
    ///
    /// `x + s·(2^w - 1 - 2x)`, which is `x` or `2^w - 1 - x`, and so below `2^w` either way because
    /// `x` is and `s` is a bit: it needs no check of its own. One product where both the limb and
    /// the sign are witnessed, and linear otherwise.
    fn complement(
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

    /// `limbs` at `widths` negated modulo `2^N` where `sign` is one, and kept where it is zero.
    ///
    /// Each limb [`Self::complement`]ed by the sign, then the carry chain adds the sign as the
    /// low limb's addend, its top carry witnessed and dropped: `!x + 1` is `2^N - x` for every `x`
    /// but zero, whose negation carries out of the top limb and leaves zero, as it should. The
    /// chain's checks are under `guard`.
    fn negate_if(
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

    /// Hold a magnitude held as `limbs` at `widths` to `2^(N - 1) - 1 + s`, for the sign `s` it is
    /// about to take: below `2^(N - 1)`, or exactly that where `s` is one, which is `INT_MIN`.
    ///
    /// The magnitude's top bit `t` is cut out of its top limb, and one constraint asks that
    /// `t·(rest + Σ low limbs + (1 - s)) == 0`, `rest` being what the cut leaves of the top limb.
    /// Every term of that sum is a non-negative integer and their sum is far below the modulus, so
    /// it is zero exactly when each term is: either `t` is zero, or every other bit is and `s` is
    /// one. The constraint is under `guard`. The cut is not, as every caller hands this limbs that
    /// are in range on every path: a product is zero where the guard is off, and so is the hint a
    /// quotient is witnessed from.
    ///
    /// The cut also bounds the top limb at its width, but a product's schoolbook checks its top
    /// column all the same. The cut reads the column through a cast to the limb's width, which
    /// truncates on the pure side, so a product past the width is refused by the cut's constraint
    /// alone and witness generation runs on; the schoolbook's check is what refuses there.
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
        let half = self.field_const(self.field.two_pow(width - 1));
        let high = self.bin(BinaryArithOpKind::UMul, t, half);
        let mut sum = self.bin(BinaryArithOpKind::USub, whole, high);
        for &(limb, _) in &limbs[..top] {
            let limb = self.cast(limb, CastTarget::Field);
            sum = self.bin(BinaryArithOpKind::UAdd, sum, limb);
        }
        let one = self.field_const(self.field.one());
        let sign = self.sign_field(sign);
        let positive = self.bin(BinaryArithOpKind::USub, one, sign);
        sum = self.bin(BinaryArithOpKind::UAdd, sum, positive);
        let product = self.bin(BinaryArithOpKind::UMul, t, sum);
        let zero = self.field_const(self.field.zero());
        self.constrain_equal(product, zero, guard);
    }

    /// Limbs at `widths` as a schoolbook [`Factor`]: a known limb is a constant with its own value
    /// as its bound, and any other is bounded by its width.
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
    fn answer_limbs(&mut self, answer: Vec<ValueId>, widths: &[usize]) -> Vec<(ValueId, bool)> {
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
    /// `deliver` hands a known limb on as a witnessed constant, and it recognises one only as a
    /// constant: a cast of one would reach the result as a pure value where the result is
    /// witnessed, as every limb of `0 % d` would.
    fn limb_fields(&mut self, limbs: Vec<(ValueId, bool)>) -> Vec<ValueId> {
        limbs
            .into_iter()
            .map(|(limb, _)| match self.known(limb) {
                Some(pattern) => self.limb_const(&pattern),
                None => self.cast(limb, CastTarget::Field),
            })
            .collect()
    }
}

// PER-INSTRUCTION REWRITING
// ================================================================================================

impl Rewriter<'_> {
    /// Rewrite one instruction into the limb-wise instructions that replace it.
    fn lower(&mut self, op: &OpCode) {
        if self.lower_through_chain(op)
            || self.lower_product(op)
            || self.lower_division(op)
            || self.lower_shift(op)
        {
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

            OpCode::SExt {
                result,
                value,
                from_bits,
                to_bits,
            } => self.lower_sext(*result, *value, *from_bits, *to_bits),

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
                let arrays = self.limbs(*array);
                let pure_bits = match self.types.get_value_type(*result).expr {
                    TypeExpr::Int(bits) => Some(bits),
                    _ => None,
                };
                let results = if pure_bits.is_some() {
                    arrays.iter().map(|_| self.fresh()).collect()
                } else {
                    self.limbs(*result)
                };
                for (result, array) in paired(results.clone(), arrays) {
                    self.push(OpCode::ArrayGet {
                        result,
                        array,
                        index,
                    });
                }
                if let Some(bits) = pure_bits {
                    let value = self.recombine_pure(&results, bits);
                    self.push(OpCode::Cast {
                        result: *result,
                        value,
                        target: CastTarget::Nop,
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
            // witnessed operand `width_validation` is what refuses it. Pure integer reads from
            // transposed sequences are recombined above before arithmetic uses them.
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

    /// A sign extension whose target is held as limbs: the source's limbs, with its sign bit
    /// filling every bit above them.
    ///
    /// Every limb of the answer is linear in that bit `s`. Below the source's top limb they are the
    /// source's own; the limb holding the source's top limb, `w_s` bits of a `w`-bit limb, is that
    /// limb plus `s·(2^w - 2^w_s)`; and every limb above it is `s·(2^w - 1)`. Each is below `2^w`,
    /// as its pieces are and cover disjoint bits, so the cut that pins `s` is the whole cost.
    ///
    /// A sign known at compile time, which is one read off a known top limb, fills at compile time:
    /// that limb with its fill and every limb above it are constants.
    ///
    /// The source is witnessed: the type rule keeps the witness wrapper across the extension, and
    /// only a witnessed result is held as limbs.
    fn lower_sext(&mut self, result: ValueId, value: ValueId, from_bits: usize, to_bits: usize) {
        assert!(
            self.is_witness(value),
            "ICE: a sign extension into the representation from a pure int{from_bits}"
        );
        let limb_bits = self.limb_bits();
        let source_widths = limb_widths(from_bits, limb_bits);
        let source = match self.wide_width(value) {
            Some(_) => self.limbs(value),
            None => self.decompose(value, from_bits),
        };
        let top = source.len() - 1;
        let top_width = source_widths[top];
        let sign = self.top_bit((source[top], true), top_width);

        let widths = limb_widths(to_bits, limb_bits);
        let mut answer = Vec::with_capacity(widths.len());
        for (index, width) in widths.iter().enumerate() {
            let limb = match index.cmp(&top) {
                std::cmp::Ordering::Less => self.cast(source[index], CastTarget::Field),
                std::cmp::Ordering::Equal
                    if *width == top_width || matches!(sign, Sign::Known(false)) =>
                {
                    self.cast(source[top], CastTarget::Field)
                }
                std::cmp::Ordering::Equal => {
                    let fill = IntBits::all_ones(width - top_width)
                        .cast(*width)
                        .shifted_left(top_width);
                    match sign {
                        Sign::Known(_) => {
                            let known = self.known(source[top]).unwrap_or_else(|| {
                                ice!("a known sign was read from a limb that is not known")
                            });
                            self.limb_const(&known.cast(*width).or(&fill))
                        }
                        Sign::Bit(bit, _) => {
                            let limb = self.cast(source[top], CastTarget::Field);
                            let fill = self.limb_const(&fill);
                            let fill = self.bin(BinaryArithOpKind::UMul, bit, fill);
                            self.bin(BinaryArithOpKind::UAdd, limb, fill)
                        }
                    }
                }
                std::cmp::Ordering::Greater => match sign {
                    Sign::Known(negative) => {
                        let fill = if negative {
                            IntBits::all_ones(*width)
                        } else {
                            IntBits::zero(*width)
                        };
                        self.limb_const(&fill)
                    }
                    Sign::Bit(bit, _) => {
                        let ones = self.limb_const(&IntBits::all_ones(*width));
                        self.bin(BinaryArithOpKind::UMul, bit, ones)
                    }
                },
            };
            answer.push(limb);
        }
        self.deliver(result, answer, &widths, to_bits, None);
    }

    /// Equality, which is the conjunction of the limbs' own.
    ///
    /// A pair of limbs both known at compile time is decided here rather than compared, because a
    /// witnessed constant is still a witness to the comparison's lowering, which would spend a
    /// gadget on it. A pair known to differ decides the whole equality, and when every pair is
    /// known equal so is the value.
    fn lower_compare(&mut self, kind: CmpKind, result: ValueId, lhs: ValueId, rhs: ValueId) {
        assert!(
            matches!(kind, CmpKind::Eq),
            "ICE: a {kind:?} of a wide witnessed integer reached the multi-cell representation; width validation should have refused the program"
        );

        let mut pairs = Vec::new();
        let mut decided = true;
        for (lhs, rhs) in self.operand_pair(lhs, rhs) {
            match self.known_equal(lhs, rhs) {
                Some(true) => {}
                Some(false) => decided = false,
                None => pairs.push((lhs, rhs)),
            }
        }
        if !decided || pairs.is_empty() {
            let answer = self.int_const(IntBits::from_u128(1, u128::from(decided)));
            let target = if self.is_witness(result) {
                CastTarget::WitnessOf
            } else {
                CastTarget::Nop
            };
            self.push(OpCode::Cast {
                result,
                value: answer,
                target,
            });
            return;
        }

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

    /// Whether two limbs are equal, where both are known at compile time.
    fn known_equal(&self, lhs: ValueId, rhs: ValueId) -> Option<bool> {
        let lhs = self.known(lhs)?;
        let rhs = self.known(rhs)?;
        Some(BigUint::from(&lhs) == BigUint::from(&rhs))
    }

    /// The assertion of a comparison, which is one assertion per limb.
    ///
    /// A pair of limbs known equal at compile time holds already and asserts nothing. One known to
    /// differ is kept, so that the program is refused as any assertion of two unequal constants is.
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
            if self.known_equal(lhs, rhs) == Some(true) {
                continue;
            }
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

/// The operation a `Guard` wraps, or `op` itself where there is none.
fn unguarded(op: &OpCode) -> &OpCode {
    match op {
        OpCode::Guard { inner, .. } => inner.as_ref(),
        other => other,
    }
}

/// The width of an integer operation with a witnessed operand, read off its left one, or [`None`]
/// where both are pure or it is not an integer operation.
fn witnessed_width(types: &FunctionTypeInfo, lhs: ValueId, rhs: ValueId) -> Option<usize> {
    let bits = int_width(types.get_value_type(lhs))?;
    (types.get_value_type(lhs).is_witness_of() || types.get_value_type(rhs).is_witness_of())
        .then_some(bits)
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

    /// The columns a signed operation adds to the chain are its sign bits, and each is cut out of
    /// its top limb: witnessed, checked to be a bit, and pinned by the rest of that limb being
    /// range-checked one bit narrower.
    #[test]
    fn every_sign_bit_is_cut_out_of_its_top_limb() {
        let h = witness_limb_bits(bn254());
        let audit = |what: &str, ssa: &mut HLSSA, columns: usize, narrower: Vec<usize>, ties| {
            run_pass(ssa);
            let ops = emitted(ssa);
            let mut written: Vec<u64> = ops
                .iter()
                .filter_map(|op| match op {
                    OpCode::WriteWitness {
                        result: Some(result),
                        ..
                    } => Some(result.0),
                    _ => None,
                })
                .collect();
            let mut bits_checked: Vec<u64> = ops
                .iter()
                .filter_map(|op| match op {
                    OpCode::Rangecheck { value, max_bits: 1 } => Some(value.0),
                    _ => None,
                })
                .collect();
            let mut checked: Vec<usize> = ops
                .iter()
                .filter_map(|op| match op {
                    OpCode::Rangecheck { max_bits, .. } if *max_bits > 1 => Some(*max_bits),
                    _ => None,
                })
                .collect();
            let relations = ops
                .iter()
                .filter(|op| matches!(op, OpCode::Constrain { .. }))
                .count();

            written.sort_unstable();
            bits_checked.sort_unstable();
            checked.sort_unstable();
            let mut narrower = narrower;
            narrower.sort_unstable();
            assert_eq!(written.len(), columns, "{what}: its columns");
            assert_eq!(written, bits_checked, "{what}: each column is a bit");
            assert_eq!(checked, narrower, "{what}: the range checks");
            assert_eq!(relations, ties, "{what}: the relations between the bits");
        };

        for bits in [254usize, 320] {
            let widths = limb_widths(bits, h);
            let (k, top) = (widths.len(), widths[widths.len() - 1]);
            let with_cuts = |cuts: usize, top_checked: bool| {
                let mut checks = widths.clone();
                if !top_checked {
                    checks.pop();
                }
                checks.extend(std::iter::repeat_n(top - 1, cuts));
                checks
            };
            let arith = |kind| {
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
            let mut ordering = program_chaining(
                bits,
                Some(Type::witness_of(Type::int(1))),
                |result, lhs, rhs| OpCode::Cmp {
                    kind: CmpKind::SLt,
                    result,
                    lhs,
                    rhs,
                },
            );
            let mut assertion = program_chaining(bits, None, |_, lhs, rhs| OpCode::AssertCmp {
                kind: CmpKind::SLt,
                lhs,
                rhs,
            });

            // Both operands' signs and the answer's, beside a carry out of every limb.
            for kind in [BinaryArithOpKind::SAdd, BinaryArithOpKind::SSub] {
                let what = format!("int{bits} {kind:?}");
                audit(&what, &mut arith(kind), k + 3, with_cuts(3, false), 1);
            }
            // Both operands' signs, and the chain an unsigned ordering has.
            audit(
                &format!("int{bits} SLt"),
                &mut ordering,
                k + 2,
                with_cuts(2, true),
                0,
            );
            audit(
                &format!("int{bits} asserted SLt"),
                &mut assertion,
                k + 1,
                with_cuts(2, true),
                0,
            );
        }

        // A sign extension into the representation cuts its source's sign and checks nothing else:
        // every limb past the source is that bit times all ones.
        let source = 320usize;
        let top = limb_widths(source, h).last().copied().expect("a limb");
        let mut extension = program_chaining(
            source,
            Some(Type::witness_of(Type::int(1000))),
            |result, value, _| OpCode::SExt {
                result,
                value,
                from_bits: source,
                to_bits: 1000,
            },
        );
        audit(
            "int320 to int1000 SExt",
            &mut extension,
            1,
            vec![top - 1],
            0,
        );
    }

    /// An operand's sign is cut once per block, however many signed operations read it, and an
    /// operation that reads one operand twice cuts it once.
    #[test]
    fn an_operand_sign_is_cut_once_per_block() {
        let bits = 320usize;
        let top = *limb_widths(bits, witness_limb_bits(bn254()))
            .last()
            .expect("a limb");
        let mut ssa = HLSSA::with_main("main".to_string());
        let main = ssa.get_unique_entrypoint_id();
        let (lhs, rhs) = (ssa.fresh_value(), ssa.fresh_value());
        let (sum, ordering, double) = (ssa.fresh_value(), ssa.fresh_value(), ssa.fresh_value());
        let mut builder = HLSSABuilder::new(&mut ssa);
        builder.modify_function(main, |fb| {
            let entry = fb.function.get_entry_id();
            for value in [lhs, rhs] {
                fb.function
                    .get_block_mut(entry)
                    .push_parameter(value, Type::witness_of(Type::int(bits)));
            }
            fb.function
                .add_return_type(Type::witness_of(Type::int(bits)));
            fb.function.add_return_type(Type::witness_of(Type::int(1)));
            fb.function
                .add_return_type(Type::witness_of(Type::int(bits)));
            let mut block = fb.test_block(entry);
            block.emit(OpCode::BinaryArithOp {
                kind: BinaryArithOpKind::SAdd,
                result: sum,
                lhs,
                rhs,
            });
            block.emit(OpCode::Cmp {
                kind: CmpKind::SLt,
                result: ordering,
                lhs,
                rhs,
            });
            block.emit(OpCode::BinaryArithOp {
                kind: BinaryArithOpKind::SAdd,
                result: double,
                lhs,
                rhs: lhs,
            });
            block.terminate_return(vec![sum, ordering, double]);
        });
        run_pass(&mut ssa);

        let cuts = emitted(&ssa)
            .iter()
            .filter(|op| matches!(op, OpCode::Rangecheck { max_bits, .. } if *max_bits == top - 1))
            .count();
        assert_eq!(cuts, 4, "the sign cuts");
    }

    /// A sign extension whose source's top limb is known fills at compile time.
    #[test]
    fn a_sign_extension_of_a_known_top_limb_fills_at_compile_time() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let main = ssa.get_unique_entrypoint_id();
        let (value, widened, extended) = (ssa.fresh_value(), ssa.fresh_value(), ssa.fresh_value());
        let mut builder = HLSSABuilder::new(&mut ssa);
        builder.modify_function(main, |fb| {
            let entry = fb.function.get_entry_id();
            fb.function
                .get_block_mut(entry)
                .push_parameter(value, Type::witness_of(Type::int(64)));
            fb.function
                .add_return_type(Type::witness_of(Type::int(1000)));
            let mut block = fb.test_block(entry);
            block.emit(OpCode::Cast {
                result: widened,
                value,
                target: CastTarget::Int(320),
            });
            block.emit(OpCode::SExt {
                result: extended,
                value: widened,
                from_bits: 320,
                to_bits: 1000,
            });
            block.terminate_return(vec![extended]);
        });
        run_pass(&mut ssa);

        let ops = emitted(&ssa);
        assert!(
            !ops.iter()
                .any(|op| matches!(op, OpCode::Rangecheck { max_bits: 1, .. })),
            "the known sign was cut"
        );
        let constant = |value: &ValueId| ssa.get_const(*value).is_some();
        assert!(
            !ops.iter().any(|op| matches!(
                op,
                OpCode::BinaryArithOp { lhs, rhs, .. } if constant(lhs) && constant(rhs)
            )),
            "a limb of the fill is arithmetic on two constants"
        );
    }

    /// A guarded chain checks nothing where the guard is off, since the operands are then whatever
    /// the branch not taken left, and its answer is zero there.
    #[test]
    fn a_guarded_chain_is_checked_only_under_its_guard() {
        let bits = 320usize;
        let mut ssa = difference(bits);
        let condition = guard_every_instruction(&mut ssa);
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

    /// Re-derive what `plan` claims from its steps alone and hold it to the soundness argument
    /// [`plan_product`] states without reading how the plan was built.
    fn assert_plan_is_sound(
        plan: &ProductPlan,
        lhs: &[BigUint],
        rhs: &[BigUint],
        addend: &[BigUint],
        widths: &[usize],
        injective: usize,
    ) {
        let limit = BigUint::one() << injective;
        let count = widths.len();
        let mut added = crate::collections::HashSet::default();
        let mut added_addends = crate::collections::HashSet::default();
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
                    Step::Addend(index) => {
                        assert_eq!(index, column, "an addend limb in the wrong column");
                        assert!(added_addends.insert(index), "an addend limb added twice");
                        accumulated += &addend[index];
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

        for (index, bound) in addend.iter().enumerate() {
            assert!(
                bound.is_zero() || added_addends.contains(&index),
                "addend limb {index} is never added"
            );
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
            full_bounds(widths)
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
                let full = full_bounds(&widths);
                for injective in [bn254, 2 * limb_bits + 2, 2 * limb_bits + 1, 2 * limb_bits] {
                    for rhs in [full.clone(), sparse(&widths)] {
                        for addend in [Vec::new(), full.clone(), sparse(&widths)] {
                            for evaluate in [false, true] {
                                let planned = plan_product(
                                    &full, &rhs, &addend, &widths, injective, evaluate,
                                );
                                match planned {
                                    Some(plan) => assert_plan_is_sound(
                                        &plan, &full, &rhs, &addend, &widths, injective,
                                    ),
                                    None => {
                                        assert_ne!(injective, bn254, "bn254 refuses int{bits}")
                                    }
                                }
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
        let full = full_bounds(&widths);
        let injective = widest_injective_int_bits(bn254());

        for bits in [127usize, 254, 320, 1000, 16384] {
            let widths = limb_widths(bits, witness_limb_bits(bn254()));
            let full = full_bounds(&widths);
            let plan = plan_product(&full, &full, &[], &widths, injective, true)
                .unwrap_or_else(|| panic!("bn254 holds an int{bits} product"));
            assert!(plan.evaluated, "int{bits} is evaluated on bn254");
            assert!(plan.overflow.is_empty(), "int{bits} has no overflow terms");
        }

        let plan = plan_product(&full, &full, &[], &widths, injective, false)
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
        let full = full_bounds(&widths);

        // Goldilocks' 32-bit limb: `(2^32 - 1)^2` alone has 64 bits against an injective 63.
        let goldilocks = limb_widths(320, 32);
        for evaluate in [false, true] {
            assert!(
                plan_product(
                    &full_bounds(&goldilocks),
                    &full_bounds(&goldilocks),
                    &[],
                    &goldilocks,
                    63,
                    evaluate
                )
                .is_none()
            );
        }

        // Five products to a column do not fit 130 bits, so an evaluation falls back to summing.
        let injective = 2 * 64 + 2;
        let fallback = plan_product(&full, &full, &[], &widths, injective, true)
            .expect("two products and a carry fit 130 bits");
        assert!(
            !fallback.evaluated,
            "a column past the modulus cannot be evaluated"
        );

        let plan = plan_product(&full, &full, &[], &widths, injective, false)
            .expect("two products and a carry fit 130 bits");
        assert_plan_is_sound(&plan, &full, &full, &[], &widths, injective);
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
        operation_with(BinaryArithOpKind::UMul, bits, rhs_type)
    }

    /// `main(lhs: WitnessOf<int(bits)>, rhs: rhs_type) -> WitnessOf<int(bits)> { lhs op rhs }`.
    fn operation_with(kind: BinaryArithOpKind, bits: usize, rhs_type: Type) -> HLSSA {
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
                kind,
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
            let full = full_bounds(&widths);
            for (what, rhs_type, evaluated) in [
                ("evaluated", Type::witness_of(Type::int(bits)), true),
                ("summed", Type::int(bits), false),
            ] {
                let plan = plan_product(
                    &full,
                    &full,
                    &[],
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
        let condition = guard_every_instruction(&mut ssa);
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

    // THE DIVISION
    // --------------------------------------------------------------------------------------------

    /// What a structural audit of one lowered program reads: every instruction by its results, and
    /// every range check by the value it bounds, looking through casts on both sides.
    struct Audit {
        ops: Vec<OpCode>,
        definitions: HashMap<ValueId, OpCode>,
    }

    impl Audit {
        fn of(ssa: &HLSSA) -> Self {
            let ops = emitted(ssa);
            let definitions = ops
                .iter()
                .flat_map(|op| op.get_results().map(move |result| (*result, op.clone())))
                .collect();
            Self { ops, definitions }
        }

        /// `value` with every cast it went through taken off.
        fn root(&self, mut value: ValueId) -> ValueId {
            while let Some(OpCode::Cast { value: source, .. }) = self.definitions.get(&value) {
                value = *source;
            }
            value
        }

        /// Every witness column written, in order.
        fn written(&self) -> Vec<ValueId> {
            self.ops
                .iter()
                .filter_map(|op| match op {
                    OpCode::WriteWitness {
                        result: Some(result),
                        ..
                    } => Some(*result),
                    _ => None,
                })
                .collect()
        }

        /// Every range check as the root of what it bounds, its width, and its guard.
        fn checks(&self) -> Vec<(ValueId, usize, Option<ValueId>)> {
            self.ops
                .iter()
                .filter_map(|op| match op {
                    OpCode::Rangecheck { value, max_bits } => {
                        Some((self.root(*value), *max_bits, None))
                    }
                    OpCode::Guard { condition, inner } => match inner.as_ref() {
                        OpCode::Rangecheck { value, max_bits } => {
                            Some((self.root(*value), *max_bits, Some(*condition)))
                        }
                        _ => None,
                    },
                    _ => None,
                })
                .collect()
        }

        /// The width each written column is checked at, in the order they are written, [`None`]
        /// for a column nothing checks.
        fn written_checks(&self) -> Vec<Option<usize>> {
            let checks = self.checks();
            self.written()
                .into_iter()
                .map(|column| {
                    checks
                        .iter()
                        .find(|(root, _, _)| *root == column)
                        .map(|(_, bits, _)| *bits)
                })
                .collect()
        }

        fn constraints(&self) -> usize {
            self.ops
                .iter()
                .filter(|op| matches!(op, OpCode::Constrain { .. }))
                .count()
        }
    }

    /// The carries the division's schoolbook plans, at full operand ranges.
    fn division_carries(widths: &[usize], evaluate: bool) -> Vec<usize> {
        let full = full_bounds(widths);
        let plan = plan_product(
            &full,
            &full,
            &full,
            widths,
            widest_injective_int_bits(bn254()),
            evaluate,
        )
        .expect("bn254 holds the division");
        assert_eq!(plan.evaluated, evaluate);
        plan.columns
            .iter()
            .flatten()
            .filter_map(|step| match step {
                Step::Reduce { carry_bits } => *carry_bits,
                _ => None,
            })
            .collect()
    }

    /// Every column a division witnesses is range-checked, and every column of `q·d + r` is held
    /// to the dividend's limb instead of being checked, however the product is formed.
    ///
    /// In order: the quotient's limbs and the remainder's, each at its own width; the schoolbook's
    /// carries at the widths its plan gives them; and the borrows of `r < d`, each a bit. The only
    /// columns left unchecked are an evaluated product's, which its evaluations pin. What is left
    /// over is the chain's own answer, one check per limb, and the constraints are the product's
    /// (its evaluations or its overflow terms) plus one equality per limb of the dividend.
    ///
    /// Structural for the reason [`every_carry_and_every_limb_of_a_product_is_bounded`] gives, and
    /// because a pinned column has no check left to perturb against: the equality is its bound.
    #[test]
    fn every_column_of_a_division_is_bounded_and_every_limb_pinned() {
        let h = witness_limb_bits(bn254());
        for kind in [BinaryArithOpKind::UDiv, BinaryArithOpKind::URem] {
            for bits in [254usize, 320] {
                let widths = limb_widths(bits, h);
                let k = widths.len();
                for (what, rhs_type, evaluated) in [
                    ("evaluated", Type::witness_of(Type::int(bits)), true),
                    ("summed", Type::int(bits), false),
                ] {
                    let mut ssa = operation_with(kind, bits, rhs_type);
                    run_pass(&mut ssa);
                    let audit = Audit::of(&ssa);
                    let what = format!("int{bits} {kind:?}, {what}");

                    let mut expected: Vec<Option<usize>> = Vec::new();
                    expected.extend(widths.iter().map(|width| Some(*width)));
                    expected.extend(widths.iter().map(|width| Some(*width)));
                    if evaluated {
                        expected.extend(std::iter::repeat_n(None, k));
                    }
                    expected.extend(division_carries(&widths, evaluated).into_iter().map(Some));
                    expected.extend(std::iter::repeat_n(Some(1), k - 1));
                    assert_eq!(audit.written_checks(), expected, "{what}");

                    let written = audit.written();
                    let rest: Vec<usize> = audit
                        .checks()
                        .into_iter()
                        .filter(|(root, _, _)| !written.contains(root))
                        .map(|(_, bits, _)| bits)
                        .collect();
                    assert_eq!(rest, widths, "{what}: the chain's limbs, and nothing else");

                    let product = if evaluated { 2 * k - 1 } else { k - 1 };
                    assert_eq!(audit.constraints(), product + k, "{what}");
                }
            }
        }
    }

    /// A constant divisor bounds both answers, so they cost only the limbs they can reach: the
    /// quotient's top limb is checked narrower, the remainder is one limb at the divisor's width,
    /// and `r < d` reads only that limb.
    #[test]
    fn a_constant_divisor_narrows_the_quotient_and_the_remainder() {
        let bits = 320usize;
        let mut ssa = operation_with(BinaryArithOpKind::UDiv, bits, Type::int(bits));
        let main = ssa.get_unique_entrypoint_id();
        let ten = ssa.add_const(Constant::Int(IntBits::from_u128(bits, 10)));
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
                        panic!("the program is one division")
                    };
                    let rewritten = OpCode::BinaryArithOp {
                        kind,
                        result,
                        lhs,
                        rhs: ten,
                    };
                    Located::new(rewritten, op.location().clone())
                })
                .collect();
            entry.put_instructions(instructions);
        }
        run_pass(&mut ssa);
        let audit = Audit::of(&ssa);

        let h = witness_limb_bits(bn254());
        let quotient_top = ((BigUint::one() << bits) - 1u8) / 10u8 >> (4 * h);
        let checks = audit.written_checks();
        assert_eq!(
            checks[..6],
            [
                Some(h),
                Some(h),
                Some(h),
                Some(h),
                Some(quotient_top.bits() as usize),
                Some(4)
            ],
            "the quotient's five limbs, the top one narrower, and a four-bit remainder"
        );
        assert!(
            checks[6..]
                .iter()
                .all(|check| check.is_some_and(|bits| bits > 1)),
            "carries only, and no borrow: `r < 10` is one limb"
        );
        assert_eq!(
            audit.constraints(),
            5,
            "one equality per limb of the dividend, and nothing to hold to zero"
        );
    }

    /// A guarded division guards only what reads its operands, and its answer is zero where the
    /// guard is off without being selected.
    ///
    /// The pure side divides zero by one where the guard is off, so the quotient, the remainder and
    /// every column and carry the schoolbook builds from them are zero there, and their checks hold
    /// unguarded whatever the operands. `q·d + r == n` and `r < d` read the operands, and are the
    /// only checks under the guard.
    #[test]
    fn a_guarded_division_is_checked_only_under_its_guard() {
        let bits = 320usize;
        let k = limb_widths(bits, witness_limb_bits(bn254())).len();
        let mut ssa = operation_with(
            BinaryArithOpKind::URem,
            bits,
            Type::witness_of(Type::int(bits)),
        );
        let condition = guard_every_instruction(&mut ssa);
        run_pass(&mut ssa);
        let audit = Audit::of(&ssa);

        let (unguarded, guarded): (Vec<_>, Vec<_>) = audit
            .checks()
            .into_iter()
            .partition(|(_, _, guard)| guard.is_none());
        assert!(
            unguarded.len() > 2 * k,
            "the quotient's and the remainder's limbs, and the schoolbook's columns and carries"
        );
        assert_eq!(
            guarded.len(),
            2 * k - 1,
            "the chain of `r < d`, each limb and each carry but the forced top one"
        );
        assert!(
            guarded
                .iter()
                .all(|(_, _, guard)| *guard == Some(condition)),
            "under the division's own guard"
        );

        let flag: Vec<ValueId> = audit
            .ops
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
        let flagged = audit
            .ops
            .iter()
            .filter(|op| {
                matches!(op, OpCode::Constrain { a, b, .. } if flag.contains(a) || flag.contains(b))
            })
            .count();
        assert_eq!(
            flagged, k,
            "each equality, and nothing else, holds only where the guard does"
        );

        let selected = audit
            .ops
            .iter()
            .filter(|op| matches!(op, OpCode::Select { .. }))
            .count();
        assert_eq!(
            selected, 3,
            "the dividend and divisor on the pure side, and the divisor again away from zero"
        );
    }

    /// A division the single cell cannot hold is lowered here, on a decomposition of each operand
    /// while they still have an element; one it can hold, and the double lane's width, are left
    /// to the single-cell lowering.
    #[test]
    fn a_division_the_single_cell_cannot_hold_is_lowered_here() {
        let field = bn254();
        let widest = widest_injective_int_bits(field) / 2;
        for kind in [BinaryArithOpKind::UDiv, BinaryArithOpKind::URem] {
            for bits in [widest + 1, 2 * HOST_LIMB_BITS + 1] {
                let mut ssa = operation_with(kind, bits, Type::witness_of(Type::int(bits)));
                let main = ssa.get_unique_entrypoint_id();
                let operands: Vec<ValueId> = ssa
                    .get_function(main)
                    .get_entry()
                    .get_parameters()
                    .map(|(value, _)| *value)
                    .collect();
                run_pass(&mut ssa);
                assert!(
                    !emitted(&ssa).iter().any(|op| matches!(
                        op,
                        OpCode::BinaryArithOp { lhs, rhs, .. }
                            if operands.contains(lhs) && operands.contains(rhs)
                    )),
                    "int{bits} {kind:?}: the single-cell division is gone"
                );
            }
            for bits in [widest, 2 * HOST_LIMB_BITS] {
                let mut ssa = operation_with(kind, bits, Type::witness_of(Type::int(bits)));
                let before = format!("{:?}", emitted(&ssa));
                run_pass(&mut ssa);
                assert_eq!(format!("{:?}", emitted(&ssa)), before, "int{bits} {kind:?}");
            }
        }
    }

    /// `d / d` needs no quotient or remainder column: it is one wherever `d` is not zero, so the
    /// only thing checked is `0 < d`, the chain's borrows and limbs.
    #[test]
    fn a_value_divided_by_itself_checks_only_that_it_is_not_zero() {
        let bits = 320usize;
        let widths = limb_widths(bits, witness_limb_bits(bn254()));
        let mut ssa = operation_with(
            BinaryArithOpKind::UDiv,
            bits,
            Type::witness_of(Type::int(bits)),
        );
        let main = ssa.get_unique_entrypoint_id();
        {
            let entry = ssa.get_function_mut(main).get_entry_mut();
            let lhs = entry
                .get_parameters()
                .next()
                .map(|(value, _)| *value)
                .unwrap();
            let instructions: Vec<_> = entry
                .take_instructions()
                .into_iter()
                .map(|op| {
                    let OpCode::BinaryArithOp { kind, result, .. } = op.as_ref().clone() else {
                        panic!("the program is one division")
                    };
                    let rewritten = OpCode::BinaryArithOp {
                        kind,
                        result,
                        lhs,
                        rhs: lhs,
                    };
                    Located::new(rewritten, op.location().clone())
                })
                .collect();
            entry.put_instructions(instructions);
        }
        run_pass(&mut ssa);
        let audit = Audit::of(&ssa);
        assert_eq!(
            audit.written_checks(),
            vec![Some(1); widths.len() - 1],
            "the borrows only"
        );
        assert_eq!(audit.checks().len(), 2 * widths.len() - 1);
        assert_eq!(audit.constraints(), 0);
    }

    /// `main(a, b: WitnessOf<int64>) { (a as int(bits)) == rhs }`, where `rhs` is `b` widened the
    /// same way or the given constant, compared or asserted.
    fn program_comparing_widened(bits: usize, constant: Option<BigUint>, assert: bool) -> HLSSA {
        let mut ssa = HLSSA::with_main("main".to_string());
        let main = ssa.get_unique_entrypoint_id();
        let [a, b, wide_a, wide_b, result] = std::array::from_fn(|_| ssa.fresh_value());
        let mut builder = HLSSABuilder::new(&mut ssa);
        builder.modify_function(main, |fb| {
            let entry = fb.function.get_entry_id();
            for value in [a, b] {
                fb.function
                    .get_block_mut(entry)
                    .push_parameter(value, Type::witness_of(Type::int(64)));
            }
            if !assert {
                fb.function.add_return_type(Type::witness_of(Type::int(1)));
            }
            let rhs = match &constant {
                Some(pattern) => fb
                    .ssa
                    .add_const(Constant::Int(IntBits::from_biguint(bits, pattern))),
                None => wide_b,
            };
            let mut block = fb.test_block(entry);
            for (result, value) in [(wide_a, a), (wide_b, b)] {
                block.emit(OpCode::Cast {
                    result,
                    value,
                    target: CastTarget::Int(bits),
                });
            }
            if assert {
                block.emit(OpCode::AssertCmp {
                    kind: CmpKind::Eq,
                    lhs: wide_a,
                    rhs,
                });
                block.terminate_return(vec![]);
            } else {
                block.emit(OpCode::Cmp {
                    kind: CmpKind::Eq,
                    result,
                    lhs: wide_a,
                    rhs,
                });
                block.terminate_return(vec![result]);
            }
        });
        ssa
    }

    /// An equality between limbs both known at compile time is decided here, and spends nothing.
    ///
    /// The limbs above a widened value's source are witnessed constants, which the comparison's
    /// lowering would otherwise meet as witnesses and give a gadget each. Two values widened from
    /// 64 bits into 320 therefore compare in their low limb alone, and a constant with a bit above
    /// that limb decides the comparison without comparing anything.
    #[test]
    fn an_equality_of_limbs_known_at_compile_time_is_decided_here() {
        let bits = 320usize;
        let compared = |ssa: &HLSSA| {
            emitted(ssa)
                .iter()
                .filter(|op| matches!(op, OpCode::Cmp { .. } | OpCode::AssertCmp { .. }))
                .count()
        };

        for assert in [false, true] {
            let mut ssa = program_comparing_widened(bits, None, assert);
            run_pass(&mut ssa);
            assert_eq!(
                compared(&ssa),
                1,
                "two widened values compare in the low limb alone, assert={assert}"
            );

            let small = BigUint::from(7u8);
            let mut ssa = program_comparing_widened(bits, Some(small), assert);
            run_pass(&mut ssa);
            assert_eq!(
                compared(&ssa),
                1,
                "a small constant is compared in the low limb alone, assert={assert}"
            );
        }

        let unreachable: BigUint = BigUint::from(1u8) << 300;
        let mut ssa = program_comparing_widened(bits, Some(unreachable.clone()), false);
        run_pass(&mut ssa);
        assert_eq!(compared(&ssa), 0, "a limb known to differ decides it");
        let ops = emitted(&ssa);
        let Some(OpCode::Cast { value, target, .. }) = ops.last() else {
            panic!("the answer is delivered last")
        };
        assert_eq!(*target, CastTarget::WitnessOf, "the answer stays witnessed");
        assert!(
            matches!(ssa.get_const(*value).as_deref(), Some(Constant::Int(pattern)) if pattern.is_zero()),
            "the answer is false"
        );

        // An assertion that cannot hold is kept, to be refused as any other is.
        let mut ssa = program_comparing_widened(bits, Some(unreachable), true);
        run_pass(&mut ssa);
        assert_eq!(
            compared(&ssa),
            2,
            "the low limb and the one known to differ"
        );
    }

    /// A division whose answer the lowering knows limb by limb hands those limbs on as known, so an
    /// equality the answer reaches is decided here rather than compared.
    ///
    /// `0 / d` knows every limb of both answers, and `(0 / d) == 0` then compares nothing; the zero
    /// divisor is still refused by the division's own `0 < d`.
    #[test]
    fn a_known_answer_limb_is_known_to_the_comparison_it_reaches() {
        let bits = 320usize;
        for kind in [BinaryArithOpKind::UDiv, BinaryArithOpKind::URem] {
            let mut ssa = HLSSA::with_main("main".to_string());
            let main = ssa.get_unique_entrypoint_id();
            let [divisor, answer, result] = std::array::from_fn(|_| ssa.fresh_value());
            let mut builder = HLSSABuilder::new(&mut ssa);
            builder.modify_function(main, |fb| {
                let entry = fb.function.get_entry_id();
                fb.function
                    .get_block_mut(entry)
                    .push_parameter(divisor, Type::witness_of(Type::int(bits)));
                fb.function.add_return_type(Type::witness_of(Type::int(1)));
                let zero = fb.ssa.add_const(Constant::Int(IntBits::zero(bits)));
                let mut block = fb.test_block(entry);
                block.emit(OpCode::BinaryArithOp {
                    kind,
                    result: answer,
                    lhs: zero,
                    rhs: divisor,
                });
                block.emit(OpCode::Cmp {
                    kind: CmpKind::Eq,
                    result,
                    lhs: answer,
                    rhs: zero,
                });
                block.terminate_return(vec![result]);
            });
            run_pass(&mut ssa);

            // The one equality left is the pure side's own zero test of the divisor, at the
            // division's width; a limb comparison would be at a limb's.
            let ops = emitted(&ssa);
            let compared: Vec<&OpCode> = ops
                .iter()
                .filter(|op| {
                    matches!(
                        op,
                        OpCode::Cmp {
                            kind: CmpKind::Eq,
                            ..
                        }
                    )
                })
                .collect();
            let the_hints_own = |rhs: &ValueId| {
                matches!(
                    ssa.get_const(*rhs).as_deref(),
                    Some(Constant::Int(pattern)) if pattern.bits() == bits && pattern.is_zero()
                )
            };
            assert!(
                matches!(compared[..], [OpCode::Cmp { rhs, .. }] if the_hints_own(rhs)),
                "{kind:?}: the equality compared limbs it knew: {compared:?}"
            );
            let Some(OpCode::Cast { value, target, .. }) = ops.last() else {
                panic!("{kind:?}: the answer is delivered last")
            };
            assert_eq!(
                *target,
                CastTarget::WitnessOf,
                "{kind:?}: the answer stays witnessed"
            );
            assert!(
                matches!(ssa.get_const(*value).as_deref(), Some(Constant::Int(pattern)) if pattern.is_one()),
                "{kind:?}: the answer is true"
            );
            assert!(
                ops.iter()
                    .any(|op| matches!(op, OpCode::Rangecheck { max_bits: 1, .. })),
                "{kind:?}: the chain for `0 < d` still borrows"
            );
        }
    }

    /// A constant entering the representation as witnessed is its limbs as witnessed constants,
    /// with no column and no range check between them.
    #[test]
    fn a_constant_injected_into_the_representation_costs_nothing() {
        let bits = 328usize;
        let mut ssa = HLSSA::with_main("main".to_string());
        let main = ssa.get_unique_entrypoint_id();
        let result = ssa.fresh_value();
        let pattern: BigUint = (BigUint::from(1u8) << 320) + 5u8;
        let mut builder = HLSSABuilder::new(&mut ssa);
        builder.modify_function(main, |fb| {
            fb.function
                .add_return_type(Type::witness_of(Type::int(bits)));
            let entry = fb.function.get_entry_id();
            let value = fb
                .ssa
                .add_const(Constant::Int(IntBits::from_biguint(bits, &pattern)));
            let mut block = fb.test_block(entry);
            block.emit(OpCode::Cast {
                result,
                value,
                target: CastTarget::WitnessOf,
            });
            block.terminate_return(vec![result]);
        });
        run_pass(&mut ssa);

        let ops = emitted(&ssa);
        assert!(
            !ops.iter()
                .any(|op| matches!(op, OpCode::Rangecheck { .. } | OpCode::WriteWitness { .. })),
            "a constant is pinned by being one: {ops:?}"
        );
        let limbs: Vec<BigUint> = ops
            .iter()
            .map(|op| match op {
                OpCode::Cast {
                    value,
                    target: CastTarget::WitnessOf,
                    ..
                } => match ssa.get_const(*value).as_deref() {
                    Some(Constant::Int(limb)) => BigUint::from(limb),
                    other => panic!("a limb that is not a constant: {other:?}"),
                },
                other => panic!("an injected constant emits {other:?}"),
            })
            .collect();
        let limb_count = limb_widths(bits, witness_limb_bits(bn254())).len();
        let mut expected = vec![BigUint::zero(); limb_count];
        expected[0] = BigUint::from(5u8);
        expected[limb_count - 1] = BigUint::from(1u8);
        assert_eq!(limbs, expected);
    }

    /// A quotient and a remainder of the same operands in one block are one gadget: the second
    /// reads its answer out of the first, so the pair writes no more columns and checks no more
    /// than the quotient alone.
    #[test]
    fn a_quotient_beside_its_remainder_is_one_division() {
        let bits = 320usize;
        let program = |kinds: &[BinaryArithOpKind]| {
            let mut ssa = HLSSA::with_main("main".to_string());
            let main = ssa.get_unique_entrypoint_id();
            let (lhs, rhs) = (ssa.fresh_value(), ssa.fresh_value());
            let results: Vec<ValueId> = kinds.iter().map(|_| ssa.fresh_value()).collect();
            let mut builder = HLSSABuilder::new(&mut ssa);
            builder.modify_function(main, |fb| {
                let entry = fb.function.get_entry_id();
                let block = fb.function.get_block_mut(entry);
                block.push_parameter(lhs, Type::witness_of(Type::int(bits)));
                block.push_parameter(rhs, Type::witness_of(Type::int(bits)));
                for _ in kinds {
                    fb.function
                        .add_return_type(Type::witness_of(Type::int(bits)));
                }
                let mut block = fb.test_block(entry);
                for (kind, result) in kinds.iter().zip(&results) {
                    block.emit(OpCode::BinaryArithOp {
                        kind: *kind,
                        result: *result,
                        lhs,
                        rhs,
                    });
                }
                block.terminate_return(results.clone());
            });
            run_pass(&mut ssa);
            let audit = Audit::of(&ssa);
            (
                audit.written().len(),
                audit.checks().len(),
                audit.constraints(),
            )
        };

        let quotient = program(&[BinaryArithOpKind::UDiv]);
        let both = program(&[BinaryArithOpKind::UDiv, BinaryArithOpKind::URem]);
        assert_eq!(both, quotient);
    }

    // THE SHIFT
    // --------------------------------------------------------------------------------------------

    /// `ssa`'s one operation with its right operand replaced by the constant `pattern`.
    fn with_constant_rhs(ssa: &mut HLSSA, pattern: IntBits) {
        let main = ssa.get_unique_entrypoint_id();
        let constant = ssa.add_const(Constant::Int(pattern));
        let entry = ssa.get_function_mut(main).get_entry_mut();
        let instructions: Vec<_> = entry
            .take_instructions()
            .into_iter()
            .map(|op| {
                let OpCode::BinaryArithOp {
                    kind, result, lhs, ..
                } = op.as_ref().clone()
                else {
                    panic!("the program is one operation")
                };
                let rewritten = OpCode::BinaryArithOp {
                    kind,
                    result,
                    lhs,
                    rhs: constant,
                };
                Located::new(rewritten, op.location().clone())
            })
            .collect();
        entry.put_instructions(instructions);
    }

    /// Every column a shift by a witnessed amount writes is bounded, and every half of a split limb
    /// is range-checked.
    ///
    /// In order: `r`, which the table bounds; the bits of `q`, each a bit; `f = 2^r`, which the table
    /// pins; for a right shift the cofactor, which its product with `f` pins; then each limb's high
    /// half, at one bit under its width for a left shift and at its width for a right one. A left
    /// shift does not split a narrow top limb, and cuts the answer's top limb instead, whose bits
    /// above its width are the last column, at the limb width. What is checked without being written
    /// is each split limb's low half at the limb width, the explicit amount bound where the
    /// decomposition passes the width, and the cut top limb at its own width. The constraints are
    /// one per upper limb of the amount, the decomposition, and the cofactor's.
    ///
    /// Structural for the reason [`every_carry_and_every_limb_of_a_product_is_bounded`] gives: a
    /// low half is a difference, not a column, so perturbing a column is the only way to reach it.
    #[test]
    fn every_column_of_a_shift_is_bounded_and_every_half_checked() {
        let h = witness_limb_bits(bn254());
        for bits in [254usize, 320] {
            let widths = limb_widths(bits, h);
            let k = widths.len();
            let stages = ceil_log2(k);
            let amount_limbs = k;
            for kind in [BinaryArithOpKind::UShl, BinaryArithOpKind::UShr] {
                let left = kind == BinaryArithOpKind::UShl;
                let mut ssa = operation_with(kind, bits, Type::witness_of(Type::int(bits)));
                run_pass(&mut ssa);
                let audit = Audit::of(&ssa);
                let what = format!("int{bits} {kind:?}");

                let mut expected: Vec<Option<usize>> = vec![None];
                expected.extend(std::iter::repeat_n(Some(1), stages));
                expected.push(None);
                if !left {
                    expected.push(None);
                }
                let top = widths[k - 1];
                let narrow_top = left && top < h;
                let split = if narrow_top { k - 1 } else { k };
                expected.extend(
                    widths[..split]
                        .iter()
                        .map(|width| Some(if left { width - 1 } else { *width })),
                );
                if narrow_top {
                    expected.push(Some(h));
                }
                assert_eq!(audit.written_checks(), expected, "{what}");

                let written = audit.written();
                let mut rest: Vec<usize> = audit
                    .checks()
                    .into_iter()
                    .filter(|(root, _, _)| !written.contains(root))
                    .map(|(_, bits, _)| bits)
                    .collect();
                rest.sort_unstable();
                let mut want = vec![h; split];
                if h << stages != bits {
                    want.push(ceil_log2(bits));
                }
                if narrow_top {
                    want.push(top);
                }
                want.sort_unstable();
                assert_eq!(
                    rest, want,
                    "{what}: the low halves and the bounds, and nothing else"
                );

                let lookups = audit
                    .ops
                    .iter()
                    .filter(|op| {
                        matches!(
                            op,
                            OpCode::Lookup {
                                target: LookupTarget::Pow2(_),
                                ..
                            }
                        )
                    })
                    .count();
                assert_eq!(lookups, 1, "{what}: one table row for `r`");
                assert_eq!(
                    audit.constraints(),
                    (amount_limbs - 1) + 1 + usize::from(!left),
                    "{what}"
                );
            }
        }
    }

    /// A shift by a constant cuts each limb once where the amount does not divide it, witnesses the
    /// upper piece and checks both, and costs nothing where it moves whole limbs.
    ///
    /// By 70 at 320 bits, every result limb starts six bits into a source limb, so each of the four
    /// limbs that reach the answer is cut at 58 for a left shift and at 6 for a right one, and the
    /// fifth moves past the top or below the bottom whole. By 128 nothing is cut.
    #[test]
    fn a_shift_by_a_constant_cuts_each_limb_at_most_once() {
        let bits = 320usize;
        for kind in [BinaryArithOpKind::UShl, BinaryArithOpKind::UShr] {
            let left = kind == BinaryArithOpKind::UShl;
            let mut ssa = operation_with(kind, bits, Type::int(bits));
            with_constant_rhs(&mut ssa, IntBits::from_u128(bits, 70));
            run_pass(&mut ssa);
            let audit = Audit::of(&ssa);
            let (upper, lower) = if left { (6, 58) } else { (58, 6) };
            assert_eq!(
                audit.written_checks(),
                vec![Some(upper); 4],
                "{kind:?} by 70: one upper piece per limb it cuts"
            );
            let written = audit.written();
            let rest: Vec<usize> = audit
                .checks()
                .into_iter()
                .filter(|(root, _, _)| !written.contains(root))
                .map(|(_, bits, _)| bits)
                .collect();
            assert_eq!(
                rest,
                vec![lower; 4],
                "{kind:?} by 70: and the piece below it"
            );
            assert_eq!(audit.constraints(), 0, "{kind:?} by 70");

            let mut ssa = operation_with(kind, bits, Type::int(bits));
            with_constant_rhs(&mut ssa, IntBits::from_u128(bits, 128));
            run_pass(&mut ssa);
            let audit = Audit::of(&ssa);
            assert!(audit.written().is_empty(), "{kind:?} by 128");
            assert!(audit.checks().is_empty(), "{kind:?} by 128");
        }
    }

    /// The limb-wise shift needs a limb times a power of two below its place to be one element,
    /// which bn254's 64-bit limb leaves room for and goldilocks' 32-bit one, whose `2^64` passes
    /// its modulus, does not. There the funnel refuses the shift rather than reaching a split that
    /// is not unique.
    #[test]
    fn the_limb_wise_shift_needs_a_limb_times_its_place_in_one_element() {
        assert!(limb_shift_fits(bn254()));
        let goldilocks = (BigInt::one() << 64) - (BigInt::one() << 32) + BigInt::one();
        assert_eq!(limb_bits_for_modulus(&goldilocks, LimbBudget::DEFAULT), 32);
        assert!(!limb_shift_fits_modulus(&goldilocks));
    }

    /// A pure amount witnesses nothing about itself: no bits of `q`, no `r`, no power of two and no
    /// table row. Its check is one comparison, and each limb's split is by a constant, so the only
    /// columns are the limbs' high halves.
    ///
    /// A witnessed guard changes only the comparison, which it guards: the amount is not zeroed
    /// where the guard is off, so nothing about it is witnessed there either.
    #[test]
    fn a_pure_amount_is_split_by_a_constant() {
        let h = witness_limb_bits(bn254());
        let bits = 320usize;
        let widths = limb_widths(bits, h);
        for kind in [BinaryArithOpKind::UShl, BinaryArithOpKind::UShr] {
            for guarded in [false, true] {
                let what = format!("{kind:?}, guarded: {guarded}");
                let left = kind == BinaryArithOpKind::UShl;
                let mut ssa = operation_with(kind, bits, Type::int(bits));
                let condition = guarded.then(|| guard_every_instruction(&mut ssa));
                run_pass(&mut ssa);
                let audit = Audit::of(&ssa);
                let expected: Vec<Option<usize>> = widths
                    .iter()
                    .map(|width| Some(if left { width - 1 } else { *width }))
                    .collect();
                assert_eq!(audit.written_checks(), expected, "{what}");
                assert_eq!(audit.constraints(), 0, "{what}");
                assert!(
                    !audit
                        .ops
                        .iter()
                        .any(|op| matches!(op, OpCode::Lookup { .. })),
                    "{what}: no table row"
                );
                let asserts: Vec<Option<ValueId>> = audit
                    .ops
                    .iter()
                    .filter_map(|op| match op {
                        OpCode::AssertCmp { .. } => Some(None),
                        OpCode::Guard { condition, inner }
                            if matches!(inner.as_ref(), OpCode::AssertCmp { .. }) =>
                        {
                            Some(Some(*condition))
                        }
                        _ => None,
                    })
                    .collect();
                assert_eq!(
                    asserts,
                    vec![condition],
                    "{what}: the amount is compared, under the guard where there is one"
                );
            }
        }
    }

    /// Wrap every instruction of `ssa`'s entry block in a guard on a new witnessed condition, which
    /// is returned.
    fn guard_every_instruction(ssa: &mut HLSSA) -> ValueId {
        let main = ssa.get_unique_entrypoint_id();
        let condition = ssa.fresh_value();
        let entry = ssa.get_function_mut(main).get_entry_mut();
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
        condition
    }

    /// A witnessed operand's known limbs drop out of a product as a constant's do.
    ///
    /// `(a as int320) · b` plans no term against `a`'s four upper limbs, which the widening knows to
    /// be zero. What is left is one partial product per limb of `b`, which is cheaper summed than
    /// the nine evaluations a product of two full operands pays, so the product constrains no
    /// evaluation and no overflow term at all.
    #[test]
    fn a_known_limb_of_a_witnessed_operand_drops_out_of_the_product() {
        let bits = 320usize;
        let mut ssa = HLSSA::with_main("main".to_string());
        let main = ssa.get_unique_entrypoint_id();
        let (narrow, wide, widened, product) = (
            ssa.fresh_value(),
            ssa.fresh_value(),
            ssa.fresh_value(),
            ssa.fresh_value(),
        );
        let mut builder = HLSSABuilder::new(&mut ssa);
        builder.modify_function(main, |fb| {
            fb.function
                .add_return_type(Type::witness_of(Type::int(bits)));
            let entry = fb.function.get_entry_id();
            let block = fb.function.get_block_mut(entry);
            block.push_parameter(narrow, Type::witness_of(Type::int(64)));
            block.push_parameter(wide, Type::witness_of(Type::int(bits)));
            let mut block = fb.test_block(entry);
            block.emit(OpCode::Cast {
                result: widened,
                value: narrow,
                target: CastTarget::Int(bits),
            });
            block.emit(OpCode::BinaryArithOp {
                kind: BinaryArithOpKind::UMul,
                result: product,
                lhs: widened,
                rhs: wide,
            });
            block.terminate_return(vec![product]);
        });
        run_pass(&mut ssa);
        let audit = Audit::of(&ssa);

        let k = limb_widths(bits, witness_limb_bits(bn254())).len();
        let witnessed_products = audit
            .ops
            .iter()
            .filter(|op| {
                let OpCode::BinaryArithOp {
                    kind: BinaryArithOpKind::UMul,
                    lhs,
                    rhs,
                    ..
                } = op
                else {
                    return false;
                };
                [lhs, rhs]
                    .into_iter()
                    .all(|value| ssa.get_const(*value).is_none())
            })
            .count();
        // The one constraint is the widening's own, tying `a`'s limb back to `a`.
        assert_eq!(audit.constraints(), 1, "no evaluation and no overflow term");
        // Per column: the partial product and its hint.
        assert_eq!(
            witnessed_products,
            2 * k,
            "one partial product per limb of `b`"
        );
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
