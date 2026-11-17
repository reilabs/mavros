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

mod globals;
mod pass;
mod plan;
mod rewriter;
mod types;
mod utils;

pub use pass::WideWitnessInts;
pub use utils::multi_cell_int_bits;

#[cfg(test)]
mod test_utils;
#[cfg(test)]
mod tests;
