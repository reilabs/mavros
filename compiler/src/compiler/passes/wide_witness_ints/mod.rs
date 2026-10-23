//! The multi-cell representation: a witnessed integer too wide for one field element becomes a
//! group of limbs.
//!
//! A `WitnessOf(Int(N))` with `N > multi_cell_int_bits` becomes `k = ceil(N/h)` little-endian
//! values, where `h` is [`witness_limb_bits`]. Limbs `0..k-1` are `WitnessOf(Int(h))` and the top
//! one is `WitnessOf(Int(N - (k-1)h))`, so each limb's declared width is correct.
//!
//! A **pure** `Int(N)` **scalar** is untouched: it lives in the hint domain, where both backends
//! compute at any width.
//!
//! A **sequence** of wide elements is transposed: `Array(E, n)` becomes `k` parallel
//! `Array(E_j, n)`, each holding one limb position of every element, and a read or write is that
//! operation once per limb sequence at the caller's index. This is [`element_limb_types`], and it
//! splits **pure** wide elements too: the lookup tape needs one field element per entry, and
//! transposing makes every entry a limb it can already read. A pure element read back is
//! recombined at once by [`Rewriter::recombine_pure`], integer arithmetic at the value's width that
//! costs interpreter work rather than constraints. Users see a whole pure integer, while the
//! read's limbs remain cached for any later consumer in the block that needs them, including
//! arithmetic and lookups. A global holding such a sequence takes a slot per limb sequence.
//!
//! The representation is sound because **every limb is range-checked** to its declared width when
//! it is created or modified. `2^h` is invertible mod `p`, so a single unbounded limb would let a
//! prover solve for any value.
//!
//! Limbs are created in two shapes:
//!
//! - **A single value becoming limbs** ([`Rewriter::decompose`]): each limb is witnessed from the
//!   pure shadow and range-checked to its width, and the limbs are constrained to recombine to the
//!   source value. Without that constraint the limbs would be unrelated to the source.
//! - **A value that is limbs from the start** (a widening cast, a pure-to-witness injection) has
//!   no source to agree with, so its limbs are only range-checked.
//!
//! Everything else either moves limbs or, like the bitwise operations, acts on each limb within its
//! width, so the limbs stay constrained. The exceptions, below, are the carry chain, the schoolbook
//! product, the division built from those two, the shift, the spread and unspread, and the sign
//! and magnitude.
//!
//! # The Carry Chain
//!
//! The unsigned sum, difference and ordering are one gadget: `a < b` **is** the borrow out of the
//! top limb of `a - b`. For a limb of width `w`, the answer is `a + b + c_in - 2^w c_out` for a sum
//! and `a - b - c_in + 2^w c_out` for a difference, with `c_out` a witnessed carry checked to be a
//! bit; range-checking the answer at `w` pins the carry. A checked sum or difference lets no carry
//! out of the top limb, an ordering's answer is that carry, and an asserted ordering fixes it at
//! one.
//!
//! The chain runs past [`widest_cell_sum_bits`], one bit below the threshold above: at that width
//! each operand still fits an element but their sum does not, so the operands are decomposed into
//! limbs first and a sum or difference is recombined into its element.
//!
//! The signed three use the same chain. A signed sum or difference keeps the unsigned limbs and top
//! carry, checked by one linear relation between that carry and the sign bits of both operands and
//! the answer. A signed ordering is the unsigned ordering with the operands' top bits flipped
//! (offset binary).
//!
//! # The Schoolbook Product
//!
//! An unsigned product runs here wherever the single cell cannot hold it, past
//! [`single_cell_product_fits`]: from the width where the single-cell division's `q·d + r`, which
//! reaches past the product, passes the modulus, except for the double lane's two-limb product. A
//! signed one runs here past [`single_cell_signed_product_fits`], with no such exception, as the
//! product of the operands' magnitudes (see the sign, below). Each answer column sums its partial
//! products and incoming carries, and is reduced into its limb, range-checked at the limb width,
//! plus a witnessed carry range-checked at the width its bound needs. The top column carries
//! nothing out, so it is range-checked at the top limb's width, and every partial product landing
//! past it is held to zero, one constraint per left limb. [`plan_product`] decides where the
//! reductions fall and why each identity holds as integers rather than modulo `p`.
//!
//! # The Division
//!
//! An unsigned division or remainder runs here wherever its single-cell lowering's `q·d + r` could
//! pass the modulus, which is where the product does. The quotient and remainder are witnessed as
//! limbs from a pure-side division and pinned by two checks: `q·d + r == n`, the schoolbook with
//! `r` as one more term in each column and every column **held to the dividend's limb** rather than
//! range-checked; and `r < d`, the carry chain with its top borrow forced out, which also refuses a
//! zero divisor. Together they admit exactly the Euclidean pair. An operand's known limbs bound
//! both answers, and a limb they cannot reach is a known zero rather than a column. A signed
//! division is the unsigned one on the operands' magnitudes, past
//! [`single_cell_signed_product_fits`].
//!
//! # The Shift
//!
//! A shift under either reading runs here wherever the single cell cannot hold it, past
//! [`single_cell_shift_fits`]. A left shift is one map on the pattern, and a signed right shift is
//! the unsigned one with the sign filled in. An amount known at compile time only moves bits, so
//! the operand is cut where a run of its bits stops landing inside one answer limb, and each piece
//! is range-checked at its width; moving an operand held as limbs by a whole number of limbs costs
//! nothing. Any other amount is `q·h + r`, with `r` and `2^r` read from the powers-of-two table.
//! Every limb is split at `2^r` into two range-checked halves that land on either side of a limb
//! boundary without overlapping, and a barrel over the bits of `q` moves whole limbs. The amount is
//! bounded by the decomposition where that reaches the width, and by an explicit range check where
//! it does not. A pure amount is known wherever the constraints are built, so its decomposition is
//! pure, its bound a comparison, and the split and barrel linear.
//!
//! # The Spread
//!
//! A spread and an unspread are limb-local at the half-limb: a half-limb of a spread's operand is
//! one limb of its answer, and a limb of an unspread's operand is a half-limb of each stream. So
//! each becomes the narrow form the witness lowering builds, once per half-limb or limb, and those
//! narrow lookups bound the pieces, which [`Rewriter::cut_unbounded`] cuts unbounded. A piece the
//! read straddles is taken at the bits the read reaches, a limb an unspread reads one bit of is
//! range-checked to that bit, and a limb wholly above the read is held to zero, so bits above a
//! read are rejected as the narrow lowering rejects them. Under a guard, each of those checks holds
//! only where the guard does.
//!
//! Every pure evaluator masks the bits above a read instead, so the same spread rejects an operand
//! as a witness and accepts it once known, as after constant folding. A program's verdict can
//! depend on that, and the rejection is kept because callers range-check through it.
//!
//! # The Sign
//!
//! A signed value's sign is the top bit of its top limb, which [`Rewriter::sign_bit`] cuts out as a
//! witnessed bit, with the rest of the limb range-checked one bit narrower. The chain checks above
//! are linear in it, as are a sign extension into the representation (its limbs past the source's
//! are that bit times all ones) and the fill of a right shift by a known amount.
//!
//! The rest is sign-magnitude. A limb complemented where the sign is one,
//! [`Rewriter::complement`], is `x + s·(2^w - 1 - 2x)`: one product, and below `2^w` with no
//! separate check. A value negated where the sign is one, [`Rewriter::negate_if`], is each limb
//! complemented and the sign added through the chain, dropping its top carry. A product, quotient
//! and remainder run the unsigned gadgets on the operands' magnitudes; the answer's magnitude is
//! held to `2^(N - 1) - 1 + s` for its sign `s` by one constraint on its top bit, then negated by
//! it. A right shift by any other amount is the unsigned one between two complements by the
//! operand's sign.
//!
//! # Strategy
//!
//! Types come from a single [`TypeInfo`] snapshot of the original, type-consistent IR and are never
//! recomputed mid-pass. Each function is handled in two phases, as `elide_tuples` does for tuples:
//!
//! 1. **Plan:** build a `value_map: ValueId -> Vec<ValueId>` from every value to its limbs. A value
//!    with no wide witness integer in its type maps to itself, so nothing narrow is churned.
//! 2. **Rewrite:** flatten returns, block parameters, instructions and terminators through that
//!    map, emitting the gadgets above where an instruction does more than move limbs.

use std::collections::BTreeSet;

use mavros_artifacts::FieldConfig;
use mavros_int_semantics::IntBits;
use num_bigint::BigUint;
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
            ceil_log2, max_pow2_table_size, single_cell_product_fits, single_cell_shift_fits,
            single_cell_signed_product_fits, widest_cell_sum_bits, widest_injective_int_bits,
            witness_half_limb_bits, witness_limb_bits,
        },
        overflow_guard::abs_as_u,
        shift_guard::emit_pure_shift_amount_check,
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
        let globals = Globals::split(ssa, field);

        let type_info = store.get::<TypeInfo>();
        let cfg = store.get::<FlowAnalysis>();

        let function_ids: Vec<FunctionId> = ssa.get_function_ids().collect();

        for fid in function_ids {
            let reachable: Vec<BlockId> = cfg
                .get_function_cfg(fid)
                .get_domination_pre_order()
                .collect();
            // Only reachable blocks are rewritten, so an orphan would keep its wide values and its
            // global slots unrenumbered. None reaches here: the pipeline reclaims orphans before
            // the lowerings just ahead of this pass, which type every block against reachable-only
            // type info.
            debug_assert_eq!(
                reachable.len(),
                ssa.get_function(fid).get_blocks().count(),
                "ICE: an unreachable block reached the multi-cell representation"
            );
            let value_map = plan_function(ssa, fid, &reachable, type_info);
            // A function with no limbs can still hold an operation whose operands each fit an
            // element but whose sum or product does not; the chain, schoolbook and division lower
            // those too.
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
            let reads_a_moved_global = || {
                globals.moved
                    && reachable.iter().any(|bid| {
                        ssa.get_function(fid)
                            .get_block(*bid)
                            .get_instructions()
                            .any(|op| Globals::slot(op).is_some())
                    })
            };
            if value_map.values().all(|limbs| limbs.len() == 1)
                && !lowers_here()
                && !reads_a_moved_global()
            {
                continue;
            }
            rewrite_function(ssa, fid, &reachable, &value_map, type_info, field, &globals);
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

/// The types a type expands into: one per limb for a witnessed wide integer, itself otherwise.
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
        // A sequence of wide elements is transposed: one sequence per limb position, holding that
        // limb of every element. Each array's element type is a single limb, at most one host cell
        // wide, so the lookup tape reads it with its existing cell tag and needs no notion of a
        // wide element.
        //
        // The alternative, one sequence of `n * k` limbs, would index it at `i * k + j`, putting
        // arithmetic between the caller's index and the tape's. Here every limb sequence is
        // indexed by the caller's index.
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
/// A sequence is stored as one sequence per limb, so a **pure** wide element splits too: kept
/// whole, a constant wide sequence would hold each entry as one field element, capping its width at
/// the modulus. Reassembling an element costs interpreter work, not constraints.
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

    // A wide value's expansion must reach the values that merely **carry** it, which their types
    // do not identify.
    //
    // A witness-indexed array read moves its element through a field element: the lowering reads
    // the hint, strips it to the pure domain, casts that to `Field`, writes it as a witness, pins
    // it with `constrain_lookup`, and casts back to the element's width. Only the first and last
    // of those are typed as the wide integer. The middle ones are a pure `int`, a `Field` and a
    // `WitnessOf(Field)`, each of which `limb_types` treats as a single value, as it is everywhere
    // else.
    //
    // So the expansion is propagated, to a fixed point, along the instructions that carry a value
    // without reading it. This is sound, not a heuristic: a value the modulus cannot carry has no
    // single field element **in either domain**, so its field image is one element per limb
    // whether stripped or not; and every step below preserves the value.
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
                        let Some(bits) = fti
                            .get_value_type(*value)
                            .int_width()
                            .filter(|bits| *bits > multi_cell_int_bits(field))
                        else {
                            continue;
                        };
                        (*result, limb_widths(bits, witness_limb_bits(field)).len())
                    }
                    // No conversion, so the result stands for whatever the source does.
                    //
                    // `WitnessOf` and `ValueOf` are deliberately **not** here: the rewriter
                    // decomposes and recombines at them explicitly. A strip yields one pure wide
                    // value, as the pure lane holds it whole.
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

/// The limbs of a value; an unmapped value is one limb, the value itself.
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
/// A constructor's element type and result type are computed separately. Where they disagree, a
/// `zip` would silently build fewer limb sequences than the plan minted values for.
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
    globals: &Globals,
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
        let mut read_limbs = HashMap::default();
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
                read_limbs: &mut read_limbs,
                known: &mut known,
                globals,
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

/// What decides both answers of a division: dividend, divisor, guard, and whether it is signed.
type Division = (ValueId, ValueId, Option<ValueId>, bool);

/// One answer of a division, as far as it has been built.
#[derive(Clone)]
enum Answer {
    /// Limbs as the field elements [`Rewriter::deliver`] takes.
    Built(Vec<ValueId>),

    /// A signed answer's magnitude as limbs, each with whether it is witnessed, and its sign, which
    /// [`Rewriter::answer`] negates the magnitude by the first time it is asked for.
    Signed(Vec<(ValueId, bool)>, Sign),
}

/// One instruction's worth of rewriting, plus the minting the gadgets need.
///
/// `ssa` is borrowed immutably, which is enough to mint a value or intern a constant: the function
/// being rewritten is out of the SSA while it is rewritten.
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

    /// The limbs a pure integer read in this block was recombined from, by the read's result.
    ///
    /// A user that wants the limbs back, as the field image a witness-indexed lookup moves through
    /// does, takes these rather than splitting what was just joined. The split uses a logical
    /// right shift (except at offset zero) and a narrowing per limb at the value's full width,
    /// undoing the recombination the read just performed.
    read_limbs: &'a mut HashMap<ValueId, Vec<ValueId>>,

    /// The values this function's rewrite has minted whose pattern is known at compile time but
    /// that are not constants themselves, by value.
    known: &'a mut HashMap<ValueId, IntBits>,

    /// Where each global's slots moved to.
    globals: &'a Globals,

    out: Vec<OpCode>,
}

/// The global slots after each global holding wide integers has one slot per limb.
///
/// A global holding wide integers as sequence elements takes the same transposed shape as a local:
/// one sequence per limb position, each in a separate slot. Every later slot moves along, so every
/// global opcode is renumbered, whatever it holds.
struct Globals {
    /// The first new slot of each old one.
    offsets: Vec<usize>,

    /// The old slots' types, which say how many new slots each became.
    old_types: Vec<Type>,

    /// Whether any slot moved; if none did, nothing needs renumbering.
    moved: bool,
}

impl Globals {
    /// Split every global in `ssa` that holds a wide integer, and record where each slot went.
    fn split(ssa: &mut HLSSA, field: FieldConfig) -> Self {
        let old_types: Vec<Type> = ssa.get_global_types().to_vec();
        let mut offsets = Vec::with_capacity(old_types.len());
        let mut new_types = Vec::new();
        for ty in &old_types {
            offsets.push(new_types.len());
            new_types.extend(limb_types(ty, field));
        }
        let moved = new_types.len() != old_types.len();
        if moved {
            ssa.set_global_types(new_types);
        }
        Self {
            offsets,
            old_types,
            moved,
        }
    }

    /// The old slot a global opcode names, if `op` is one.
    fn slot(op: &OpCode) -> Option<usize> {
        match op {
            OpCode::ReadGlobal { offset, .. } => Some(*offset as usize),
            OpCode::InitGlobal { global, .. } | OpCode::DropGlobal { global } => Some(*global),
            _ => None,
        }
    }
}

impl Rewriter<'_> {
    fn limb_bits(&self) -> usize {
        witness_limb_bits(self.field)
    }

    fn limbs(&self, value: ValueId) -> Vec<ValueId> {
        limbs_of(self.value_map, value)
    }

    /// The limbs of an operand, splitting a **pure** one the plan never saw.
    ///
    /// `check_widths` makes both operands of a non-shift the same width, and a mixed pure/witness
    /// pair is legal, so one witnessed operand is sufficient to require a witness lowering. A
    /// constant is split at compile time, so its limbs are constants too. A pure value that a
    /// sequence read in this block recombined hands back the limbs it was recombined from.
    fn operand_limbs(&mut self, value: ValueId, expected: usize) -> Vec<ValueId> {
        let mapped = self.limbs(value);
        if mapped.len() == expected {
            return mapped;
        }
        if let Some(limbs) = self.read_limbs.get(&value)
            && limbs.len() == expected
        {
            return limbs.clone();
        }
        assert_eq!(
            mapped.len(),
            1,
            "ICE: an operand of {} limbs met one of {expected}",
            mapped.len()
        );

        let bits = self
            .types
            .get_value_type(value)
            .int_width()
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

    /// `value >> offset` on the pure side.
    ///
    /// Up to the injective width it is a division by `2^offset`: `Simplifier` turns a shift by a
    /// constant there into a `BitRange`, and it runs after the rule that lowers `BitRange`, so the
    /// window would reach codegen and be refused.
    ///
    /// Past that width it is a shift, which the simplifier leaves alone. Wasm emits it inline,
    /// where a division calls a runtime helper per limb and, at 16384 bits, makes the function too
    /// large to compile.
    fn shifted_down(&mut self, value: ValueId, bits: usize, offset: usize) -> ValueId {
        if offset == 0 {
            return value;
        }
        if bits > widest_injective_int_bits(self.field) {
            let amount = self.int_const(IntBits::from_u128(bits, offset as u128));
            return self.bin(BinaryArithOpKind::UShr, value, amount);
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

/// The rewriter is an emitter, so the shared lowering helpers can build into it. Everything it
/// emits goes through [`Rewriter::push`], which tracks the constants a cast carries.
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
    /// A single witnessed `bits`-wide value, split into range-checked limbs constrained to
    /// recombine to it.
    ///
    /// As `bits <= multi_cell_int_bits`, the source has an element and the limb sum reaches it
    /// without wrapping. A value is decomposed once per block per width; later requests reuse the
    /// limbs.
    fn decompose(&mut self, value: ValueId, bits: usize) -> Vec<ValueId> {
        if let Some(limbs) = self.decomposed.get(&(value, bits)) {
            return limbs.clone();
        }
        let pure = self.cast(value, CastTarget::ValueOf);
        let (limbs, sum) = self.witness_limbs_of(pure, bits);
        let value_field = self.cast(value, CastTarget::Field);
        self.constrain_equal(value_field, sum, None);

        self.decomposed.insert((value, bits), limbs.clone());
        limbs
    }

    /// The limbs of a pure `bits`-wide `value` at the limb widths, each written to a witness
    /// column and range-checked at its width, with their place-weighted field sum. The caller ties
    /// that sum to the witnessed source.
    fn witness_limbs_of(&mut self, pure: ValueId, bits: usize) -> (Vec<ValueId>, ValueId) {
        let widths = limb_widths(bits, self.limb_bits());
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
        (limbs, sum)
    }

    /// A witnessed field element read as an integer of `to_bits` held as limbs, into `results`.
    ///
    /// An element below the modulus fits any width the representation holds, so this is its
    /// canonical value, zero-filled. It is decomposed at the field's width by
    /// [`Self::witness_limbs_of`]: limbs from the pure shadow, each range-checked at its width,
    /// constrained to sum to the element. That sum is taken modulo `p` and the limbs span past it,
    /// so `v` and `v + p` both satisfy it where `v + p` fits the limbs. The carry chain with its
    /// top borrow forced out holds the limbs below the modulus, leaving only the canonical value.
    fn lower_field_into_limbs(&mut self, value: ValueId, to_bits: usize, results: &[ValueId]) {
        let field_bits = self.field.field_bit_size() as usize;
        let widths = limb_widths(field_bits, self.limb_bits());
        let shadow = self.cast(value, CastTarget::ValueOf);
        let pure = self.cast(shadow, CastTarget::Int(field_bits));
        let (limbs, sum) = self.witness_limbs_of(pure, field_bits);
        self.constrain_equal(value, sum, None);

        let modulus = IntBits::from_biguint(
            field_bits,
            &field_modulus(self.field)
                .to_biguint()
                .expect("a modulus is positive"),
        );
        let witnessed: Vec<(ValueId, bool)> = limbs.iter().map(|limb| (*limb, true)).collect();
        let bound: Vec<(ValueId, bool)> = widths
            .iter()
            .enumerate()
            .map(|(index, width)| {
                let limb = modulus.bit_range(index * self.limb_bits(), *width);
                (self.int_const(limb), false)
            })
            .collect();
        self.carry_chain_over(true, &witnessed, &bound, widths, CarryOut::One, None);

        let target_widths = limb_widths(to_bits, self.limb_bits());
        assert_eq!(
            target_widths.len(),
            results.len(),
            "ICE: an int{to_bits} is {} limbs, not {}",
            target_widths.len(),
            results.len()
        );
        for (index, (result, width)) in results.iter().zip(target_widths).enumerate() {
            match limbs.get(index) {
                Some(limb) => self.push(OpCode::Cast {
                    result: *result,
                    value: *limb,
                    target: CastTarget::Int(width),
                }),
                None => self.define_zero(*result, width),
            }
        }
    }

    /// Limb `index` of a pure `bits`-wide `value`, at `width`, written to a witness column and
    /// range-checked at `checked` bits. Returns the limb and its field element.
    ///
    /// Written from the hint, not injected: an injection keeps the pure shadow live into R1CS,
    /// where a witness strip is an ICE.
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

    /// A **pure** wide value injected into the representation as witnessed, range-checked limbs,
    /// written into `results`.
    ///
    /// [`Self::decompose`] cannot serve past the field's width: it ties the limbs to the source's
    /// field element, which a value wider than the modulus lacks. The source is a hint, so the
    /// limbs **are** the value, each bounded by a range check. A constant enters as one witnessed
    /// constant per limb, pinned by being one.
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

    /// The low `bits` of a limb list as a single witnessed value of that width; the inverse of
    /// [`Self::decompose`].
    ///
    /// It needs no separate constraint: a linear combination of pinned values is pinned.
    ///
    /// `bits` must be at most [`multi_cell_int_bits`], and the assertion is what makes this sound:
    /// the sum is formed in **one field element**, so a target the modulus cannot carry would wrap
    /// silently. A wider target uses [`Self::pure_limbs_at`] instead.
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
    /// The **pure** counterpart of [`Self::recombine`], which sums field elements and so wraps once
    /// the value passes the modulus. This is integer arithmetic at the value's width, which both
    /// compiled backends have at every width, and costs interpreter work rather than constraints.
    /// It lets a sequence hold elements the field cannot carry.
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
    /// A narrowing cast, not a `BitRange`, which keeps its source's type: a limb list at the
    /// target's widths needs each limb typed at its position's width. The cast lowers to the same
    /// mask, witness and range check.
    fn truncate_limb(&mut self, limb: ValueId, width: usize) -> ValueId {
        self.cast(limb, CastTarget::Int(width))
    }

    /// A witnessed zero of `width` bits, which is a constant and therefore pinned by being one.
    fn zero_limb(&mut self, width: usize) -> ValueId {
        let result = self.fresh();
        self.define_zero(result, width);
        result
    }

    /// The limbs of a wide value as **pure** values, cut to the shape a `to_bits` target needs.
    ///
    /// For narrowing out of the representation to a target wider than the field carries, where
    /// [`Self::recombine`] cannot serve: it sums field elements, and a sum reaching `2^to_bits`
    /// wraps once the target passes the modulus. [`Self::recombine_pure`] reads these back with
    /// integer arithmetic at the target's width.
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
    /// with the two range checks of the cut.
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
    /// Each limb of the answer as a field element, range-checked at its width, the top one
    /// unless [`TopLimb::CutByCaller`] says the caller bounds it.
    answer: Vec<ValueId>,

    /// The width of each limb, little-endian.
    widths: Vec<usize>,

    /// The carry out of the top limb, where the chain witnesses one.
    carry_out: Option<ValueId>,
}

impl Rewriter<'_> {
    /// The unsigned sum, difference and ordering at a width whose sum one element cannot carry,
    /// through the carry [`Chain`] whatever form the operands are in, as the chain handles
    /// overflow. Returns whether `op` was lowered.
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
    /// A signed ordering is the unsigned ordering of the operands with their top bits flipped
    /// (offset binary), which maps the signed range onto `[0, 2^N)` in order. Only the top limb
    /// changes ([`Self::flip_top_bit`]).
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

    /// A signed sum or difference: the unsigned one's limbs, with overflow read off the sign bits
    /// rather than the carry out of the top limb.
    ///
    /// The chain says `a + b = r + 2^N·c` for a sum and `a - b = r - 2^N·c` for a difference, with
    /// `c` the witnessed carry or borrow out of the top limb. Reading each pattern `x` as
    /// `x - 2^N·s_x` for its sign bit `s_x`, the signed sum is `r + 2^N·(c - s_a - s_b)` and the
    /// signed difference `r - 2^N·(c + s_a - s_b)`. Either fits exactly when it equals `r`'s signed
    /// reading, `r - 2^N·s_r`, so the one check is `c + s_r == s_a + s_b` for a sum and
    /// `c + s_a == s_r + s_b` for a difference: linear in four pinned bits.
    ///
    /// The answer's sign is cut out of its top limb without the guard, as the operands' are: an
    /// honest carry keeps every answer limb in range wherever the operands are, and they are on
    /// every path. Only the check is guarded. The cut bounds the top limb at its width, so the
    /// chain does not check it again ([`TopLimb::CutByCaller`]), except where that limb is one bit
    /// wide and so is the sign bit itself, with no cut.
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

    /// The top bit of a `width`-bit limb, as a field element that is zero or one.
    ///
    /// The sign-bit primitive. A known limb answers at compile time and a pure one on the pure
    /// side, so neither costs a constraint. A witnessed one is cut at `width - 1` by
    /// [`Self::cut_segment`], which witnesses the bit, range-checks it at one bit, and range-checks
    /// the rest of the limb at `width - 1`. Together these pin the bit and bound the limb below
    /// `2^width`: the other bit value puts the rest out of range. A limb is cut once per block, and
    /// later requests reuse the bit ([`Self::signs`]). A one-bit limb is its top bit and is not
    /// bounded here.
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

    /// A `width`-bit limb with its top bit flipped, as [`Self::carry_chain_over`] reads it.
    ///
    /// A known or pure limb flips on the pure side. A witnessed one is `x + 2^(w - 1) - s·2^w` for
    /// its top bit `s`: linear in the [`Self::sign_bit`] cut, and below `2^w`.
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
    /// Under a guard the answer is zero where the guard is off: the chain's range checks are off
    /// there, so the limbs it would leave are unbounded.
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

    /// Hand the limbs of an answer to `result`, each a field element range-checked at its width,
    /// and zero where a guard is off.
    ///
    /// A result held as one element fits one, as the operands do: only their sum or product needed
    /// limbs. A limb known at compile time, as every limb of `0 / d` is, is delivered as a
    /// **witnessed** integer constant.
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
    /// For a limb of width `w`, the answer is `a + b + c_in - 2^w * c_out` for a sum and
    /// `a - b - c_in + 2^w * c_out` for a difference, range-checked at `w`. This pins `c_out`: the
    /// other value moves the answer by `2^w`, to `2^w` or past it, or below zero, where it wraps to
    /// an element no range check at `w` admits. Either answer is within `2^(w + 1)` of zero, far
    /// inside the modulus, so the identity holds as one field element whichever carry the prover
    /// picks.
    ///
    /// Every check the chain makes is under `guard`. The decomposition of an operand that is still
    /// one element is not: its checks restate that operand's bounds, which hold whether the guard
    /// is on or off.
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

    /// [`Self::carry_chain`] over operands already cut into limbs at `widths`, each limb paired
    /// with whether it is witnessed, which decides how its hint is read. One list may mix both: a
    /// known constant limb is a pure value beside witnessed ones.
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
    /// A sum's carry is its bit `w`; a difference borrows exactly when the subtrahend exceeds the
    /// minuend. Neither reaches `2^(w + 1)`, so neither wraps.
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
    /// A witnessed operand still held as one value is below the representation threshold and is
    /// split by [`Self::decompose`], which ties its limbs to it. A pure one is cut on the pure
    /// side, at no constraint cost.
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
    /// Each limb at its width.
    limbs: Vec<Limb>,

    /// Whether the operand is witnessed, which decides how the hints of its [`Limb::Value`]s are
    /// read. A witnessed operand can still have known limbs, which are [`Limb::Constant`]s.
    witnessed: bool,

    /// The largest value each limb can take: the largest its width holds, or a constant limb's
    /// value.
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
    /// where the product is evaluated, whose identity is also its overflow check.
    overflow: Vec<(usize, Vec<usize>)>,

    /// Whether the columns are witnessed whole and pinned by evaluating the product, instead of
    /// summed from partial products.
    evaluated: bool,

    /// The width the pure side computes its hints at, which no accumulator's bound reaches.
    hint_bits: usize,
}

/// The schoolbook plan for operands whose limbs are bounded by `lhs` and `rhs`, plus an `addend`
/// with the given limb bounds (empty for none), at the answer's limb `widths`, on a field whose
/// widest injective width is `injective`.
///
/// The addend lets a division state `q·d + r` as one sum: limb `i` of `r` is one more
/// non-negative term in column `i`, so every argument below still holds.
///
/// [`None`] where one partial product and the headroom it is reduced in do not fit an element.
///
/// **Every bound is an integer bound, kept under the field.** A column sums non-negative partial
/// products and carries while the sum stays below `2^injective`, so the field element is the
/// integer sum. Before the next term would pass that, the column is reduced: its low limb is
/// range-checked at the limb width and the rest leaves as a carry range-checked at `carry_bits`.
/// The identity `low + 2^w·carry` is below `2^injective`, so it holds as integers too. A carry's
/// bound is its range check's, `2^carry_bits - 1`, not the honest maximum.
///
/// **The overflow check is on the operands.** All partial products are non-negative, so the
/// product is below `2^N` exactly when every partial product at or past the top column's place is
/// zero and the top column fits its width. The first half is `a_i · Σ b_j == 0` over the `b_j`
/// that land there with `a_i`; it holds as integers while the sum times the limb is below the
/// modulus, and a zero sum of non-negative terms is zero term by term. This costs one constraint
/// per left limb rather than one per partial product.
///
/// **Where summing costs more, the product is evaluated instead** (`evaluate`). Each partial
/// product of two witnessed limbs is a separate row, so the schoolbook pays about `k^2 / 2` rows
/// on two full operands, while evaluating pays `2k - 1` regardless. The caller counts the rows
/// summing would pay with [`witnessed_partial_products`] and evaluates only where both operands are
/// witnessed and that count is larger. The columns `c_0 .. c_(k-1)` are witnessed whole and
/// `(Σ a_i x^i)(Σ b_j x^j) = Σ c_m x^m` is constrained at `2k - 1` distinct points, which makes it
/// a polynomial identity of degree `2k - 2`, so every coefficient agrees modulo `p`.
///
/// Every column's integer sum is non-negative, and the plan requires each to be below
/// `2^injective`, so a coefficient equal modulo `p` is equal as an integer: `c_m` is column `m`,
/// and the columns past the answer being zero is the overflow check. Where a column could pass the
/// modulus, the plan falls back to the schoolbook, which reduces as often as it needs.
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

/// Whether [`plan_product`] plans a `bits`-wide schoolbook on `field` at full operand ranges, with
/// a full-range addend or without one.
#[cfg(test)]
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

/// The bound of each limb at `widths` when nothing narrower is known: the largest value its width
/// holds.
fn full_bounds(widths: &[usize]) -> Vec<BigUint> {
    widths
        .iter()
        .map(|width| (BigUint::one() << *width) - 1u8)
        .collect()
}

/// How many products of two witnessed limbs summing `lhs · rhs` costs: one per partial product in
/// the answer, plus one per left limb for the overflow check.
///
/// Evaluating costs `2k - 1`, fewer whenever every limb is witnessed and `k >= 2`. A known limb
/// multiplies for free, so a widened narrow value is cheaper summed. A witnessed guard is left
/// out: it adds to both sides, and on two dense operands they tie at two limbs and evaluating wins
/// past that.
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
    /// An unsigned product the single cell cannot hold. Returns whether `op` was one and was
    /// lowered.
    ///
    /// A schoolbook at the witness limb, following [`plan_product`], which states the soundness
    /// argument. Under a guard the range and overflow checks are off where the guard is off, and
    /// the answer is zero there; the carries are written either way.
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

    /// `lhs · rhs (+ addend)` at `widths`, following [`plan_product`], as the field element of each
    /// answer limb.
    ///
    /// A `pinned` schoolbook holds each column's final value equal to the target's limb, already
    /// bounded, instead of range-checking it. So it states `lhs · rhs + addend == target` exactly
    /// and returns the target's limbs.
    ///
    /// Under a guard the checks are off where the guard is off. A pinned schoolbook is a
    /// division's, whose hints are zero there, so only the pins need the guard.
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

    /// An operand of the schoolbook as its limbs at `bits`, with their bounds.
    ///
    /// A constant is cut at compile time, so limbs it does not reach are known zeros and drop out
    /// of the plan. A witnessed operand can have known limbs too (above a widened value, emptied by
    /// a constant shift, or known from an earlier gadget); each is taken as its constant, which is
    /// pinned by being one.
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
    /// Under a guard the left limbs are scaled by it, so where it is off the product and every
    /// column are zero: the identity holds whatever operands the branch not taken left.
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

    /// Every column of the answer, as field elements, following `plan`. `columns` are an evaluated
    /// product's witnessed columns, which a [`Step::Column`] adds.
    ///
    /// Where `product` is pinned, a column's final value is held equal to the target's limb, which
    /// bounds it as tightly as the range check it replaces; a column that never needed reducing is
    /// held to that limb too. The range checks are under `checks` and the equalities under `pins`.
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
    /// An unsigned division or remainder whose product the single cell cannot hold; returns `true`
    /// if `op` was one and was lowered.
    ///
    /// The quotient `q` and remainder `r` are witnessed as limbs from a pure-side division and
    /// pinned by `q·d + r == n` and `r < d`. The first is one schoolbook, with `r` added to its
    /// columns and each column held to the dividend's limb; the second is the carry chain with its
    /// top borrow forced out, which a zero divisor cannot pass. Together they admit exactly the
    /// Euclidean pair.
    ///
    /// Under a guard, the pure side divides zero by one where the guard is off, so `q`, `r` and
    /// everything the schoolbook builds from them are zero there: its checks hold whatever the
    /// untaken branch left behind, and only `q·d + r == n` and `r < d` are guarded. The answer is
    /// those zeros, range-checked like any limb, so it needs no select. A value divided by itself,
    /// `d / d`, witnesses nothing: its answers are the constants one and zero, so where the guard
    /// is off they are selected to zero instead.
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
        // A signed answer negates the magnitude by a chain checked only under the guard, so where
        // the guard is off the answer is selected to zero, as `d / d`'s is.
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

    /// The quotient and the remainder of `n / d` as limbs, witnessed from the pure side's
    /// `pure_dividend / pure_divisor`, the operands' whole values there.
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
        // Bound the answers by `q <= n / d` and `r < d`; only known limbs, and most of all a
        // constant divisor, tighten the width's bound. The chain below refuses a zero divisor, so
        // the bounds need only hold for `d >= 1`.
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

    /// A pure `bits`-wide `hint` witnessed as limbs at `widths`, each range-checked no wider than
    /// `max` allows for it. A limb `max` does not reach is a constant zero, which drops out of the
    /// schoolbook's plan. Only the caller's constraints tie the limbs to anything.
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
/// reading, guarded or not, with a witnessed operand, past [`single_cell_product_fits`], where the
/// single-cell lowering's `q·d + r` could wrap the modulus.
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

    /// A shift by a known amount at or past the width, which refuses the program wherever the guard
    /// is on. The answer is zero, so it is in range where the guard is off.
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

    /// The limb widths a value of `bits` is held at here: its limbs' widths, or `bits` where it is
    /// still one element.
    fn shift_widths(&self, value: ValueId, bits: usize) -> Vec<usize> {
        if self.limbs(value).len() > 1 {
            limb_widths(bits, self.limb_bits())
        } else {
            vec![bits]
        }
    }

    /// `lhs` shifted by an `amount` below `bits` that is known at compile time.
    ///
    /// Every answer bit is an operand bit, so the operand is cut wherever a run of its bits stops
    /// landing inside one answer limb: at its limb boundaries, where its bits meet the answer's
    /// limb boundaries, and where the discarded bits begin. Each piece of a cut limb is witnessed
    /// and range-checked at its width, except the lowest, which is the limb minus the others and
    /// is range-checked too: the limb and the recombined pieces are below `2^w`, so the difference
    /// is either the lowest piece or an element near `p`, which that check rejects. Each answer
    /// limb is then a sum of pieces.
    ///
    /// A limb nothing cuts costs nothing, so an amount that is a whole number of limbs is free.
    ///
    /// The answer is one field element per limb of `answer_widths` (the result's). Where `signed`
    /// asks for it and the shift moves anything, the operand's sign is also returned, for
    /// [`Self::fill_with_sign`]: the piece left by a cut at its top bit or, where the top limb is
    /// known, its top bit.
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

        // A known top limb's sign is read off its pattern. Cutting it would leave a known answer
        // limb as arithmetic on constants, which `deliver` hands on as a pure value.
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

    /// One witnessed `width`-bit `segment` starting at `start`, cut at `bounds` (which run from
    /// `start` to its end). Returns each piece's start, width and field element.
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
    /// **The amount** is `n = q·h + r`, with `r` below the limb width `h` and `q` the number of
    /// whole limbs it moves. Its low limb is witnessed as `h·Σ e_t·2^t + r` with every `e_t` a bit
    /// and `r` read out of the powers-of-two table beside its power `f = 2^r`, which bounds `r`
    /// below `h`; its other limbs are held to zero. With `s` bits of `q`, that is `n < h·2^s`
    /// already, and `n < bits` exactly where `h·2^s` is the width; elsewhere `bits - 1 - n` is
    /// range-checked as well.
    ///
    /// **Each limb** `x_i` is split as `x_i·m = hi_i·2^h + lo_i` with both halves range-checked,
    /// where `m` is `f` for `<<` and `2^(h - r)` for `>>`, pinned by `m·f == 2^h`. As `x_i·m` is
    /// below `2^2h`, which [`witness_limb_bits`] keeps injective, the split is unique. Answer
    /// limb `j` before the whole limbs move is `lo_j + hi_(j-1)` for `<<` or `hi_j + lo_(j+1)` for
    /// `>>`; the two halves cover disjoint bits, so neither sum carries.
    ///
    /// **The whole limbs** move by `q` through a barrel: stage `t` moves every limb `2^t` places
    /// where `e_t` is set, each of its limbs being `a + e_t·(b - a)`. For a `<<` whose top limb is
    /// narrower than `h`, that limb is not split: its `x·m` enters the barrel whole, and a final
    /// cut keeps the answer's low `w` bits and range-checks the `h` above them.
    ///
    /// Under a guard a witnessed amount's lowest limb is zero where the guard is off, so every
    /// check holds there, and the operand's limbs are in range on every path. Holding the amount's
    /// other limbs to zero is the one check that needs the guard. A pure amount is decomposed on
    /// the pure side, by [`Self::pure_shift_amount`], and every product above is by a constant.
    ///
    /// **A sign-filling right shift**, where `signed`, is the operand [`Self::complement`]ed by its
    /// sign, shifted as above, and the answer complemented back. For a negative operand that is
    /// `!(!lhs >> n)`, and `!lhs` is not negative, so complementing back fills the vacated bits
    /// with ones.
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

        // The top limb of a `<<` has to be below its width. It was never split, so it can be
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

    /// The amount of a [`Self::shift_by_amount`]: the `stages` bits of how many whole limbs it
    /// moves, and the multiplier that splits a limb, as a field element and as a pure hint at twice
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

        if !self.is_witness(rhs) {
            return self.pure_shift_amount(rhs, bits, stages, left, guard);
        }
        // The amount is as wide as the value, so it is held in limbs exactly where the value is.
        let (low, rest, low_bits) = if bits > multi_cell_int_bits(self.field) {
            let widths = limb_widths(bits, limb_bits);
            let limbs = self.operand_limbs(rhs, widths.len());
            (limbs[0], limbs[1..].to_vec(), widths[0])
        } else {
            (rhs, Vec::new(), bits)
        };

        // Every limb above the lowest is zero, or the amount is past any width: every width is
        // below `2^h`, the most the lowest limb holds.
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

        // `n = h·Σ e_t·2^t + r`, each `e_t` a bit and `r` a key of the table.
        let within_field = self.cast(hints.within, CastTarget::Field);
        let within_column = self.write_witness(within_field);
        let mut sum = within_column;
        let mut e = Vec::with_capacity(stages);
        for (stage, hint) in hints.stages.into_iter().enumerate() {
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

        // Where `h·2^stages` is not the width, the decomposition alone does not bound the amount
        // below it, so it is checked explicitly.
        if limb_bits << stages != bits {
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

    /// The pure side of an amount `n = q·h + r` of `amount_bits`, decomposed the same way by both
    /// routes: each of the `stages` bits of `q` as a one-bit integer, `r`, and `2^r` at twice the
    /// limb width.
    ///
    /// Only the low `log2(h) + stages` bits are read, so a larger amount is taken modulo
    /// `h·2^stages`; the caller's check rules that out.
    fn amount_hints(&mut self, amount: ValueId, amount_bits: usize, stages: usize) -> AmountHints {
        let limb_bits = self.limb_bits();
        let log_limb = limb_bits.trailing_zeros() as usize;
        let stages = (0..stages)
            .map(|stage| {
                let bit = self.shifted_down(amount, amount_bits, log_limb + stage);
                self.cast(bit, CastTarget::Int(1))
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

    /// [`Self::shift_amount`] for a pure amount, which is known wherever the constraints are built,
    /// so nothing is witnessed: its check is [`emit_pure_shift_amount_check`]'s comparison, and its
    /// bits and multiplier are pure, so each split and barrel stage is linear.
    ///
    /// Under a guard only the comparison is guarded. The amount is not zeroed where the guard is
    /// off; the split and barrel still hold, and the answer there is the operand shifted by that
    /// amount modulo `h·2^stages`.
    fn pure_shift_amount(
        &mut self,
        rhs: ValueId,
        bits: usize,
        stages: usize,
        left: bool,
        guard: Option<ValueId>,
    ) -> (Vec<ValueId>, ValueId, ValueId) {
        emit_pure_shift_amount_check(self, guard, rhs, bits);

        let hints = self.amount_hints(rhs, bits, stages);
        let e = hints
            .stages
            .into_iter()
            .map(|bit| self.cast(bit, CastTarget::Field))
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
    /// Each bit of how many whole limbs the amount moves, as a one-bit integer.
    stages: Vec<ValueId>,

    /// The amount within a limb, `r`, at `log2(h)` bits.
    within: ValueId,

    /// `2^r`, at twice the limb width.
    power: ValueId,
}

/// The shift the representation lowers, if `op` is one, under either reading, guarded or not, with
/// a witnessed operand, at a width the single cell does not take.
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
    /// The unsigned schoolbook over the operands' magnitudes, whose overflow check keeps the
    /// product below `2^N`; then [`Self::check_fits`] under the product's sign; then the product
    /// negated by that sign. Every check past the operands is guarded, except the cut of the
    /// product's top bit, which reads a product that is zero where the guard is off; the answer is
    /// zero there too, as the unsigned product's is.
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

        // Off the guard the schoolbook's checks are off, so a top column the untaken branch
        // overflowed is past its width. The negation reads that column through a cast to a limb's
        // width, which truncates on the pure side while the constraints carry it whole, so the
        // negation's products would disagree with their columns. The product is zero there
        // instead.
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
    /// The unsigned division of the magnitudes, which refuses a zero divisor, then Noir's
    /// truncation toward zero: the quotient's sign is the XOR of the operands' signs, and the
    /// remainder takes the dividend's. The quotient's magnitude is held to fit under its sign for
    /// both answers, as the model refuses `INT_MIN % -1` along with `INT_MIN / -1`. The remainder's
    /// always fits, being below the divisor's magnitude and so below `2^(N - 1)`.
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

    /// Fill the bits vacated by a right shift by a known nonzero `amount` with the operand's
    /// `sign`, which [`Self::shift_by_constant`] cut out, in an `answer` with limbs at `widths`.
    ///
    /// The unsigned answer has those bits clear, so filling adds `s` times each limb's share of
    /// `2^N - 2^(N - amount)`: linear in the sign, and each limb stays below its width. A sign
    /// known to be set fills a known limb at compile time, so it stays a constant, which
    /// [`Self::deliver`] hands on as a witnessed constant.
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

    /// The magnitude of a `bits`-wide operand read as signed: its limbs at `widths`, each with
    /// whether it is witnessed; its sign; and, for a pure operand, the whole magnitude the limbs
    /// were cut from, which a division's hint reads.
    ///
    /// `|INT_MIN|` is `2^(N - 1)`, which `N` unsigned bits hold, so every magnitude fits its
    /// width. A constant's is taken at compile time, so limbs it does not reach stay known zeros;
    /// a pure operand's is taken on the pure side, at no cost; and a witnessed one is its limbs
    /// [`Self::negate_if`] by its sign.
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
    /// `x` is and `s` is a bit: it needs no separate check. It costs one product where both the
    /// limb and the sign are witnessed, and is linear otherwise.
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
    /// Each limb is [`Self::complement`]ed by the sign, then the carry chain adds the sign to the
    /// low limb, its top carry witnessed and dropped: `!x + 1` is `2^N - x` for every nonzero `x`,
    /// and for zero it carries out of the top limb and leaves zero, as it should. The chain's
    /// checks are under `guard`.
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

    /// Hold a magnitude, as `limbs` at `widths`, to at most `2^(N - 1) - 1 + s` for the sign `s` it
    /// is about to take: below `2^(N - 1)`, or exactly `2^(N - 1)` (`INT_MIN`) where `s` is one.
    ///
    /// The magnitude's top bit `t` is cut out of its top limb, leaving `rest`, and one constraint
    /// asks `t·(rest + Σ low limbs + (1 - s)) == 0`. The sum is of non-negative integers far below
    /// the modulus, so it is zero exactly when each term is: either `t` is zero, or every other bit
    /// is and `s` is one. The constraint is under `guard`; the cut is not, as every caller passes
    /// limbs in range on every path.
    ///
    /// A product's schoolbook still checks its top column: the cut reads it through a cast that
    /// truncates on the pure side, so without that check an overflowing product would be refused
    /// only by the cut's constraint, after witness generation has run on.
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

    /// Limbs at `widths` as a schoolbook [`Factor`]: a known limb is a constant with its value as
    /// its bound, and any other is bounded by its width.
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
    /// `deliver` hands a known limb on as a witnessed constant, but recognizes one only as a
    /// constant: a cast of one would reach a witnessed result as a pure value, as every limb of
    /// `0 % d` would.
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

// THE SPREAD
// ================================================================================================

impl Rewriter<'_> {
    /// A witnessed spread whose answer is held as limbs, under `guard` where there is one.
    ///
    /// A spread is limb-local: each half-limb of the operand spreads into one answer limb, by the
    /// narrow spread the witness lowering builds, its lookup keyed by the half-limb. The lookups
    /// also bound the half-limbs [`Self::cut_unbounded`] leaves unbounded, even for an answer used
    /// only in part, as dead-code elimination keeps every witnessed spread until its lookup is
    /// built. Only the half-limbs the read reaches are cut, the one it straddles spreads only the
    /// bits the read takes, and anything above the read inside a cut is refused. A limb known to
    /// be zero spreads to zero and costs nothing.
    fn lower_spread(
        &mut self,
        guard: Option<ValueId>,
        result: ValueId,
        value: ValueId,
        value_bits: usize,
    ) {
        let bits = self
            .types
            .get_value_type(value)
            .int_width()
            .unwrap_or_else(|| ice!("a spread of a non-integer reached the representation"));
        let containers = limb_widths(bits, witness_half_limb_bits(self.field));
        let results = self.limbs(result);
        assert_eq!(
            results.len(),
            containers.len(),
            "ICE: an int{bits} spread is a limb per half-limb, not {} for {}",
            results.len(),
            containers.len()
        );

        // `None` is a piece of a limb known to be zero, which `cut_unbounded` would cut into as
        // many pieces as the read reaches half-limbs.
        let mut pieces: Vec<Option<(ValueId, usize)>> = Vec::with_capacity(containers.len());
        if self.wide_width(value).is_some() {
            let half = witness_half_limb_bits(self.field);
            for (limb, width, read) in self.limbs_and_reads(value, bits, value_bits) {
                let zero = self.known(limb).is_some_and(|pattern| pattern.is_zero());
                match read {
                    0 if zero => {}
                    0 => self.constrain_zero(limb, guard),
                    read if zero => pieces.extend(std::iter::repeat_n(None, read.div_ceil(half))),
                    read => pieces.extend(
                        self.cut_unbounded(limb, width, read, guard)
                            .into_iter()
                            .map(Some),
                    ),
                }
            }
        } else {
            pieces = self
                .cut_unbounded(value, bits, value_bits, guard)
                .into_iter()
                .map(Some)
                .collect();
        }

        for (index, (result, container)) in results.into_iter().zip(containers).enumerate() {
            match pieces.get(index) {
                Some(&Some((piece, read))) => self.push_guarded(
                    guard,
                    OpCode::Spread {
                        result,
                        value: piece,
                        value_bits: read,
                    },
                ),
                Some(None) | None => self.define_zero(result, 2 * container),
            }
        }
    }

    /// A witnessed unspread of an operand held as limbs, under `guard` where there is one.
    ///
    /// Limb-local in the other direction: each operand limb unspreads into a half-limb of each
    /// stream, by the narrow unspread the witness lowering already builds, whose lookups bound
    /// both. The half-limbs are then reassembled into the streams, which may be limbs or one
    /// element each.
    ///
    /// A limb the read straddles unspreads only the bits the read takes. A limb the read takes a
    /// single bit of has no odd bit to unspread, so it is range-checked to one bit and is itself
    /// the even half-limb. A limb wholly above the read is held to zero. A limb known to be zero
    /// unspreads to zero and costs nothing.
    fn lower_unspread(
        &mut self,
        guard: Option<ValueId>,
        result_odd: ValueId,
        result_even: ValueId,
        value: ValueId,
        value_bits: usize,
    ) {
        let bits = self.wide_width(value).unwrap_or_else(|| {
            ice!(
                "an unspread whose operand is one element answers streams that are one element too"
            )
        });
        let mut odd_parts = Vec::new();
        let mut even_parts = Vec::new();

        for (limb, width, read) in self.limbs_and_reads(value, bits, value_bits) {
            let (odd_width, even_width) = (width / 2, width.div_ceil(2));
            let zero = self.known(limb).is_some_and(|pattern| pattern.is_zero());
            let (odd, even) = match read {
                _ if zero => (None, self.zero_limb(even_width)),
                0 => {
                    self.constrain_zero(limb, guard);
                    (None, self.zero_limb(even_width))
                }
                1 => {
                    let limb_field = self.cast(limb, CastTarget::Field);
                    // Under a guard the bit is witnessed and tied to the limb under it: where the
                    // guard is off the limb may hold more than a bit, which an `Int` of the even
                    // half-limb's width would truncate.
                    let bit = match guard {
                        None => limb_field,
                        Some(_) => {
                            let pure = self.cast(limb, CastTarget::ValueOf);
                            let low = self.cast(pure, CastTarget::Int(1));
                            let low_field = self.cast(low, CastTarget::Field);
                            let bit = self.write_witness(low_field);
                            self.constrain_equal(limb_field, bit, guard);
                            bit
                        }
                    };
                    self.push_guarded(
                        guard,
                        OpCode::Rangecheck {
                            value: bit,
                            max_bits: 1,
                        },
                    );
                    (None, self.cast(bit, CastTarget::Int(even_width)))
                }
                _ => {
                    let (odd, even) = (self.fresh(), self.fresh());
                    self.push_guarded(
                        guard,
                        OpCode::Unspread {
                            result_odd: odd,
                            result_even: even,
                            value: limb,
                            value_bits: read,
                        },
                    );
                    (Some(odd), even)
                }
            };
            // A one-bit top limb has an even bit and no odd one, so it adds no odd half-limb.
            if odd_width > 0 {
                let odd = odd.unwrap_or_else(|| self.zero_limb(odd_width));
                odd_parts.push((odd, odd_width));
            }
            even_parts.push((even, even_width));
        }

        self.assemble_half_limbs(result_odd, &odd_parts);
        self.assemble_half_limbs(result_even, &even_parts);
    }

    /// The limbs of `value`, an `int(bits)` held as limbs, low first, each with its width and the
    /// number of its bits that a read of the low `read` bits takes, which is zero for a limb wholly
    /// above the read.
    fn limbs_and_reads(
        &self,
        value: ValueId,
        bits: usize,
        read: usize,
    ) -> Vec<(ValueId, usize, usize)> {
        let limb_bits = self.limb_bits();
        self.limbs(value)
            .into_iter()
            .zip(limb_widths(bits, limb_bits))
            .enumerate()
            .map(|(index, (limb, width))| {
                (
                    limb,
                    width,
                    read.saturating_sub(index * limb_bits).min(width),
                )
            })
            .collect()
    }

    /// The half-limbs of a `bits`-wide witnessed `value` that the low `read` bits reach, low first,
    /// each typed at its width and paired with how many of its bits the read takes. **None is
    /// bounded**: the caller bounds each.
    ///
    /// The high pieces are witnessed from the pure shadow and the lowest is the source minus them,
    /// so the pieces recombine to the source; bits above the read land in the lowest. Once each is
    /// bounded at its read, their sum is below `2^read`.
    ///
    /// Without a guard the lowest piece is derived, and its narrow lookup bounds it. Under a guard
    /// it is witnessed too, and tied to the source under the guard: where the guard is off nothing
    /// bounds a derived piece, so one carrying the bits above the read would sit in an `Int` it
    /// does not fit, and the VM's truncating cast would disagree with the constraint system on the
    /// witness.
    fn cut_unbounded(
        &mut self,
        value: ValueId,
        bits: usize,
        read: usize,
        guard: Option<ValueId>,
    ) -> Vec<(ValueId, usize)> {
        let half = witness_half_limb_bits(self.field);
        let containers = limb_widths(bits, half);
        let reads = limb_widths(read, half);
        if containers.len() == 1 {
            return vec![(value, read)];
        }

        let pure = self.cast(value, CastTarget::ValueOf);
        let source = self.cast(value, CastTarget::Field);
        let first = if guard.is_some() { 0 } else { 1 };
        let mut rest = source;
        let mut sum = None;
        let mut pieces = Vec::with_capacity(reads.len());
        for (index, piece_read) in reads.iter().enumerate().skip(first) {
            let hint = self.shifted_down(pure, bits, index * half);
            let narrowed = self.cast(hint, CastTarget::Int(*piece_read));
            let hint_field = self.cast(narrowed, CastTarget::Field);
            let written = self.write_witness(hint_field);
            let placed = if index == 0 {
                written
            } else {
                let place = self.field_const(self.field.two_pow(index * half));
                self.bin(BinaryArithOpKind::UMul, written, place)
            };
            if guard.is_some() {
                sum = Some(match sum {
                    None => placed,
                    Some(acc) => self.bin(BinaryArithOpKind::UAdd, acc, placed),
                });
            } else {
                rest = self.bin(BinaryArithOpKind::USub, rest, placed);
            }
            pieces.push((
                self.cast(written, CastTarget::Int(containers[index])),
                *piece_read,
            ));
        }
        match sum {
            Some(sum) => self.constrain_equal(source, sum, guard),
            None => {
                let lowest = self.cast(rest, CastTarget::Int(containers[0]));
                pieces.insert(0, (lowest, reads[0]));
            }
        }
        pieces
    }

    /// Half-limbs, low first, recombined into `result`, which spans their total width and is held
    /// as limbs or as one element.
    ///
    /// Every piece but the last is a full half-limb, so each lands at a multiple of one; each is
    /// already bounded, so the sum is below `2^width` and needs no separate constraint.
    fn assemble_half_limbs(&mut self, result: ValueId, parts: &[(ValueId, usize)]) {
        let half = witness_half_limb_bits(self.field);
        let targets = self.limbs(result);
        let target_widths: Vec<usize> = limb_types(self.types.get_value_type(result), self.field)
            .iter()
            .map(|ty| ty.int_width().expect("a stream is an integer"))
            .collect();
        let groups: Vec<&[(ValueId, usize)]> = if targets.len() == 1 {
            vec![parts]
        } else {
            parts.chunks(self.limb_bits() / half).collect()
        };
        assert_eq!(
            groups.len(),
            targets.len(),
            "ICE: {} groups of half-limbs were assembled into a stream of {} limbs",
            groups.len(),
            targets.len()
        );

        for ((target, target_width), group) in targets.into_iter().zip(target_widths).zip(groups) {
            let width: usize = group.iter().map(|(_, width)| width).sum();
            assert_eq!(
                width, target_width,
                "ICE: half-limbs of {width} bits were assembled into an int{target_width}"
            );
            let mut sum = None;
            for (index, (part, _)) in group.iter().enumerate() {
                let part_field = self.cast(*part, CastTarget::Field);
                let placed = if index == 0 {
                    part_field
                } else {
                    let place = self.field_const(self.field.two_pow(index * half));
                    self.bin(BinaryArithOpKind::UMul, part_field, place)
                };
                sum = Some(match sum {
                    None => placed,
                    Some(acc) => self.bin(BinaryArithOpKind::UAdd, acc, placed),
                });
            }
            self.push(OpCode::Cast {
                result: target,
                value: sum.expect("a limb is at least one half-limb"),
                target: CastTarget::Int(width),
            });
        }
    }

    /// Hold a witnessed integer to zero, where `guard` holds.
    fn constrain_zero(&mut self, value: ValueId, guard: Option<ValueId>) {
        let value_field = self.cast(value, CastTarget::Field);
        let zero = self.field_const(self.field.zero());
        self.constrain_equal(value_field, zero, guard);
    }

    /// Define `result` as a witnessed zero of `width` bits, which is a constant and pinned by being
    /// one: [`Self::zero_limb`] into a value the caller names.
    fn define_zero(&mut self, result: ValueId, width: usize) {
        let zero = self.int_const(IntBits::zero(width));
        self.push(OpCode::Cast {
            result,
            value: zero,
            target: CastTarget::WitnessOf,
        });
    }
}

// PER-INSTRUCTION REWRITING
// ================================================================================================

impl Rewriter<'_> {
    /// Rewrite one instruction into the limb-wise instructions that replace it.
    fn lower(&mut self, op: &OpCode) {
        if self.lower_global(op) {
            return;
        }
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

            // A witnessed spread or unspread, under its guard where it has one, whose lookups and
            // zero checks hold only where the guard does.
            OpCode::Spread { value, .. } | OpCode::Unspread { value, .. }
                if self.is_witness(*value) =>
            {
                self.lower_spread_family(None, op);
            }
            OpCode::Guard { condition, inner }
                if matches!(
                    **inner,
                    OpCode::Spread { value, .. } | OpCode::Unspread { value, .. }
                        if self.is_witness(value)
                ) =>
            {
                let guard = self.one(*condition);
                self.lower_spread_family(Some(guard), inner);
            }

            // An assertion in a branch on a witness, which holds only where the branch is taken.
            OpCode::Guard { condition, inner } if matches!(**inner, OpCode::AssertCmp { .. }) => {
                let OpCode::AssertCmp { kind, lhs, rhs } = **inner else {
                    ice_unreachable!("matched above");
                };
                let guard = self.one(*condition);
                self.lower_assert_compare(kind, lhs, rhs, Some(guard));
            }

            // Everything below moves limbs without reading them, so each becomes its narrow form
            // once per limb.
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

            // The witness-indexed read: `witness_array_access` built its lookup before this pass,
            // so the index and flag are narrow and pinned. One lookup per limb sequence against
            // them selects each limb by the same `sum(hit) == 1` argument.
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

            // A blob-backed sequence: one blob per limb. The elements are constants, so the split
            // is arithmetic on their patterns, which lets a constant sequence hold elements too
            // wide for one field element.
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
                let bits = element_type
                    .int_width()
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

            // A witness-indexed read inside a branch on a witness, whose hint the array lowering
            // guards so that an index past the end on the branch not taken does not trap: one
            // guarded read per limb sequence, under the same condition.
            OpCode::Guard { condition, inner } if matches!(**inner, OpCode::ArrayGet { .. }) => {
                let OpCode::ArrayGet {
                    result,
                    array,
                    index,
                } = **inner
                else {
                    ice_unreachable!("matched above");
                };
                let condition = self.one(*condition);
                self.lower_array_get(Some(condition), result, array, index);
            }

            // One read per limb sequence, at the caller's index. The index is narrow and shared
            // rather than re-derived, so a witnessed one is looked up once per limb against the
            // value the array lowering pinned.
            OpCode::ArrayGet {
                result,
                array,
                index,
            } => self.lower_array_get(None, *result, *array, *index),

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

            // The slice operations that survive this far. Every operation that changes a limb
            // sequence's length is applied to all `k`, so the length is read off any one, and the
            // index is shared.
            //
            // `SlicePop`, `SliceInsert` and `SliceRemove` have no arm:
            // `InstructionLowering::slice_ops` replaces each with an assert, a get and a copy loop
            // before this pass.
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

            // The bitwise three, the one arithmetic family with **no cross-limb interaction**:
            // bit `i` of the answer depends only on bit `i` of each operand, so each answer limb is
            // the same operation on the matching operand limbs. There is no carry and nothing to
            // reconstruct: each limb is already held to its width, and an operation that sets no
            // bit absent from both operands cannot break that.
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

            // The complement, the unary member of the same family and limb-wise for the same
            // reason: bit `i` of the answer depends only on bit `i` of the operand. Each limb is
            // complemented at its width, so the top one, which may be narrower than a full limb,
            // gains no bits beyond the value's width.
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

            // A whole pure integer written as a witnessed one held as limbs, such as an
            // unconstrained call's answer: each limb is a new column, written from the value and
            // bounded at its width, as nothing else ties the limbs to it.
            OpCode::WriteWitness {
                result: Some(result),
                value,
                pinned,
            } if self.limbs(*value).len() == 1 && self.limbs(*result).len() > 1 => {
                // A pinned write is an entry point's input, which `prepare_entry_point` keeps
                // within one element.
                debug_assert!(!pinned, "ICE: a pinned write of a wide value");
                let bits = self
                    .wide_width(*result)
                    .expect("ICE: a write into the representation without a wide result");
                let widths = limb_widths(bits, self.limb_bits());
                for (index, (result, width)) in
                    self.limbs(*result).into_iter().zip(widths).enumerate()
                {
                    let (limb, _) = self.witness_limb(*value, bits, index, width, width);
                    self.push(OpCode::Cast {
                        result,
                        value: limb,
                        target: CastTarget::Nop,
                    });
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

            // A wide witnessed value reaching anything else is a shape this pass does not
            // represent.
            other => ice!(
                "{other:?} reached the multi-cell representation with a wide operand, which is a shape it does not represent"
            ),
        }
    }

    /// A global opcode, renumbered to its slots' new place and fanned out over its limbs, a limb
    /// per slot. A drop drops only the slots that hold something on the heap.
    fn lower_global(&mut self, op: &OpCode) -> bool {
        if !self.globals.moved {
            return false;
        }
        match op {
            OpCode::ReadGlobal {
                result,
                offset,
                result_type,
            } => {
                let base = self.globals.offsets[*offset as usize];
                let results = self.limbs(*result);
                let types = limb_types(result_type, self.field);
                assert_eq!(
                    results.len(),
                    types.len(),
                    "ICE: a global read of {} slots answers {} limbs",
                    types.len(),
                    results.len()
                );
                for (slot, (result, result_type)) in results.into_iter().zip(types).enumerate() {
                    self.push(OpCode::ReadGlobal {
                        result,
                        offset: (base + slot) as u64,
                        result_type,
                    });
                }
            }
            OpCode::InitGlobal { global, value } => {
                let base = self.globals.offsets[*global];
                let values = self.limbs(*value);
                let slots = limb_types(&self.globals.old_types[*global], self.field).len();
                assert_eq!(
                    values.len(),
                    slots,
                    "ICE: a global of {slots} slots is initialised with {} limbs",
                    values.len()
                );
                for (slot, value) in values.into_iter().enumerate() {
                    self.push(OpCode::InitGlobal {
                        global: base + slot,
                        value,
                    });
                }
            }
            OpCode::DropGlobal { global } => {
                let base = self.globals.offsets[*global];
                for (slot, ty) in limb_types(&self.globals.old_types[*global], self.field)
                    .into_iter()
                    .enumerate()
                {
                    if ty.is_heap_allocated() {
                        self.push(OpCode::DropGlobal {
                            global: base + slot,
                        });
                    }
                }
            }
            _ => return false,
        }
        true
    }

    /// One read per limb sequence of `array`, at the caller's index and under `guard` if any.
    ///
    /// A pure element is recombined into `result` at once for users that expect a whole integer.
    /// The limbs remain in `read_limbs` for any later operand consumer in this block, including
    /// witness-indexed lookups, without splitting the whole value again. Recombination costs
    /// interpreter work, not constraints, as a pure value is known wherever the constraints are built.
    fn lower_array_get(
        &mut self,
        guard: Option<ValueId>,
        result: ValueId,
        array: ValueId,
        index: ValueId,
    ) {
        let index = self.one(index);
        let arrays = self.limbs(array);
        let results = self.limbs(result);
        let whole = results.len() == 1 && arrays.len() > 1;
        let reads: Vec<ValueId> = if whole {
            arrays.iter().map(|_| self.fresh()).collect()
        } else {
            results
        };
        for (read, array) in paired(reads.clone(), arrays) {
            let get = OpCode::ArrayGet {
                result: read,
                array,
                index,
            };
            self.push(match guard {
                Some(condition) => OpCode::Guard {
                    condition,
                    inner: Box::new(get),
                },
                None => get,
            });
        }
        if whole {
            let bits = self
                .types
                .get_value_type(result)
                .int_width()
                .unwrap_or_else(|| {
                    ice!("a whole element read out of limb sequences is not an integer")
                });
            let combined = self.recombine_pure(&reads, bits);
            self.push(OpCode::Cast {
                result,
                value: combined,
                target: CastTarget::Nop,
            });
            self.read_limbs.insert(result, reads);
        }
    }

    /// A spread or an unspread, under `guard` where there is one.
    fn lower_spread_family(&mut self, guard: Option<ValueId>, op: &OpCode) {
        match op {
            OpCode::Spread {
                result,
                value,
                value_bits,
            } => self.lower_spread(guard, *result, *value, *value_bits),
            OpCode::Unspread {
                result_odd,
                result_even,
                value,
                value_bits,
            } => self.lower_unspread(guard, *result_odd, *result_even, *value, *value_bits),
            other => ice_unreachable!("{other:?} is not a spread or an unspread"),
        }
    }

    /// A cast, which is where a value enters and leaves the representation.
    fn lower_cast(&mut self, result: ValueId, value: ValueId, target: &CastTarget) {
        let source_type = self.types.get_value_type(value);
        let source_bits = source_type.int_width();

        match target {
            CastTarget::Int(to_bits) => {
                // The other side of the field boundary: a wide value read back out of the field
                // elements its limbs were pinned as. Each limb returns at its width, so there
                // is nothing to recombine and no place value to mint.
                if source_bits.is_none() {
                    let limbs = self.limbs(value);
                    let results = self.limbs(result);
                    if limbs.len() == 1 && results.len() > 1 {
                        self.lower_field_into_limbs(value, *to_bits, &results);
                        return;
                    }
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
                    // Out of it. A target the field carries is one element again, so the low limbs
                    // recombine into it and the linear combination keeps them pinned. A target it
                    // cannot carry has no element to sum into, so the limbs recombine as integer
                    // arithmetic at the target's width, as the witness strip does.
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

            // A wide value's field image is one element per limb; the array lowering's lookup
            // chain moves it through a field. Each limb is already held to its width, so each is
            // an element.
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

    /// A sign extension whose target is held as limbs: the source's limbs, with its sign bit `s`
    /// filling every bit above them.
    ///
    /// Every limb of the answer is linear in `s`. Below the source's top limb they are the source's
    /// limbs; the limb holding the source's top limb (`w_s` bits of a `w`-bit limb) is that limb
    /// plus `s·(2^w - 2^w_s)`; every limb above it is `s·(2^w - 1)`. Each is below `2^w`, as its
    /// pieces are and cover disjoint bits, so the cut that pins `s` is the whole cost.
    ///
    /// A sign known at compile time (read off a known top limb) fills at compile time: that limb
    /// with its fill and every limb above it are constants.
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

    /// Equality: the conjunction of per-limb equalities.
    ///
    /// A pair of limbs both known at compile time is decided here rather than compared: the
    /// comparison's lowering would spend a gadget on a witnessed constant. One pair known to differ
    /// decides the whole equality, and if every pair is known equal, so is the value.
    fn lower_compare(&mut self, kind: CmpKind, result: ValueId, lhs: ValueId, rhs: ValueId) {
        assert!(
            matches!(kind, CmpKind::Eq),
            "ICE: a {kind:?} of a wide witnessed integer reached the multi-cell representation; the carry chain takes every ordering"
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
            "ICE: a {kind:?} assertion of a wide witnessed integer reached the multi-cell representation; the carry chain takes every ordering"
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
    let bits = types.get_value_type(lhs).int_width()?;
    (types.get_value_type(lhs).is_witness_of() || types.get_value_type(rhs).is_witness_of())
        .then_some(bits)
}

// UTILITIES
// ================================================================================================

/// The widest witnessed integer carried as a single field element.
///
/// A wider value has no element to hold it, so it is held as limbs. This is
/// [`widest_injective_int_bits`], so it is field-derived: 253 on bn254, 63 on goldilocks.
///
/// A gadget that needs more headroom than one element leaves does not move this; it asks for a
/// bound locally, as the carry chain asks [`widest_cell_sum_bits`], one bit narrower.
/// `the_representation_threshold_is_where_an_element_stops_being_injective` fails if this moves
/// without its reason moving too.
pub fn multi_cell_int_bits(field: FieldConfig) -> usize {
    widest_injective_int_bits(field)
}

#[cfg(test)]
mod tests;
