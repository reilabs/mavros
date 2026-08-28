//! The generated Noir corpus that holds the _lowered_ path to this model.
//!
//! Every other conformance test in the workspace checks an evaluator by calling it directly, which
//! only ever reaches the folding path: a constant in, a constant out. That leaves the far longer
//! route untested: an operand that is a witness flows through guard emission, the witness lowering,
//! R1CS construction, the VM's opcode dispatch and the WASM backend, and none of those are
//! reachable from a unit test that hands two constants to a folder.
//!
//! So this module renders Noir programs instead. Each one takes its operands as `main` parameters,
//! which makes them witnesses and defeats constant folding outright, and asserts the answer this
//! model gives:
//!
//! ```text
//! fn main(u8_0: u8, u8_1: u8) {
//!     assert_eq(u8_0 + u8_1, 42);
//! }
//! ```
//!
//! The `assert_eq` makes the value observable to the test runner's **existing** R1CS-satisfaction
//! oracle in every lane, so a divergence between any two backends shows up as a red row with no new
//! machinery in the runner at all.
//!
//! This crate creates test fixtures this way because the expected answers must come from [`eval`]
//! and nowhere else. A corpus written by hand records what someone believed; a corpus rendered here
//! records what the specification says, and regenerating it after a semantic change produces a
//! reviewable diff of exactly which programs changed meaning.
//!
//! Noir's own type system bounds this, and the gaps are as follows:
//!
//! - **Widths:** Noir's `IntegerBitSize` is exactly `{8, 16, 32, 64, 128}`, so [`NOIR_WIDTHS`] is
//!   the whole set. The model's width `1` and its [`corners::ODD_WIDTHS`] are unwritable in a Noir
//!   program and stay unit-test-only.
//! - **Mixed Operand Widths:** Noir's elaborator unifies a shift's amount with its value, so
//!   `rhs_bits == bits` in every program here. The `s2 > s1` axis arises only from shifts the
//!   compiler builds itself.
//! - **`i128`:** Noir has it while Mavros caps a signed _reading_ at [`crate::MAX_SIGNED_BITS`], so
//!   a signed 128-bit program is rejected at compile time rather than run. See that constant's doc
//!   for more info.
//! - **`u128 <<`:** `witness_bitwise::product_headroom_or_bail` refuses to lower it, so the left
//!   shift stops at 64 bits while every other operation runs the full unsigned set.
//!
//! The **rejecting** half renders one program per `(operation, reading, reason)` at the narrowest
//! width the model rejects at — see [`first_rejection`] — because a rejection reason is a property
//! of the operation while a program is a `STATUS.md` row. On its own that covers the _reasons_ and
//! not the width-dependent arithmetic each check is built out of. [`Case::Widest`] closes that, for
//! the checks whose bound is _derived from_ the width. [`Case::NegativeAmount`] covers a check
//! whose only live width is one neither of the other two would pick.
//!
//! The accepting half is what checks every width for the operations themselves, and
//! `overflow_guard`'s own conformance sweep is what covers the one predicate no corpus can reach,
//! the discharge that deletes a check outright.

use crate::{
    IntBits, IntOp, MAX_BITS, MAX_SIGNED_BITS, Outcome, Reject, Sign, SignedValue, corners, eval,
    residue,
};

use std::collections::{BTreeMap, BTreeSet};

// SHAPE OF THE CORPUS
// ================================================================================================

/// The integer widths a Noir program can name.
///
/// Taken from `noirc_frontend`'s `IntegerBitSize`, which has exactly these five variants. It is
/// deliberately _not_ [`corners::WIDTHS`]: the model sweeps widths that no source program can ask
/// for, and a generated test at one of those would not compile.
pub const NOIR_WIDTHS: [usize; 5] = [8, 16, 32, 64, 128];

/// How many operand pairs each `(op, sign, width)` cell contributes.
///
/// A bound, not a cross-product. Every cell has hundreds of accepting corner pairs and the corpus
/// is charged one `STATUS.md` row per test either way, so the pairs are _sampled_ across the whole
/// accepting set rather than truncated from its front — see [`spread`].
///
/// The ceiling is real rather than a matter of taste. A generated `main` lowers to **one** WASM
/// function, and a WASM function has a hard cap on its local count; at twelve pairs per cell the
/// `>>` program crossed it and the WASM lanes failed with "too many locals" while every other lane
/// passed. Six leaves the widest program at a bit under half that budget, so a new width or a new
/// operation does not silently walk back into it.
const PAIRS_PER_CELL: usize = 6;

/// How many pairs a cell contributes when the operation's reading cannot change its answer.
///
/// `And`/`Or`/`Xor` are bit-for-bit identical under both readings, so a signed cell for one of them
/// pins that the _type_ lowers, not that the arithmetic is right. A handful is enough for that.
const PAIRS_PER_READING_BLIND_CELL: usize = 3;

// GENERATED TESTS
// ================================================================================================

/// One rendered Noir test package.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GeneratedTest {
    /// The package name, which is also its directory name.
    pub name: String,

    /// The contents of `src/main.nr`.
    pub main_nr: String,

    /// The contents of `Prover.toml`.
    pub prover_toml: String,

    /// Whether this program must be _rejected_, which decides the corpus directory it belongs in.
    pub expect_failure: bool,
}

impl GeneratedTest {
    /// The contents of `Nargo.toml`, which differs between packages only in the name.
    #[must_use]
    pub fn nargo_toml(&self) -> String {
        format!(
            "[package]\nname = \"{}\"\ntype = \"bin\"\nauthors = [\"\"]\n\n[dependencies]\n",
            self.name
        )
    }
}

/// Every program whose operations this model _accepts_, one per operation.
#[must_use]
pub fn accepting_tests() -> Vec<GeneratedTest> {
    let mut groups: Vec<Vec<IntOp>> = Vec::new();
    for &op in &IntOp::ALL {
        match groups.iter_mut().find(|g| g[0].name() == op.name()) {
            Some(group) => group.push(op),
            None => groups.push(vec![op]),
        }
    }
    groups.iter().map(|ops| accepting_test(ops)).collect()
}

/// Every program this model _rejects_, one per `(operation, reading, reason)`.
///
/// These cannot be merged the way the accepting ones are: a rejection aborts the program, so a
/// second rejecting operation in the same file would never be reached and its guard could be
/// deleted without any test noticing.
#[must_use]
pub fn rejecting_tests() -> Vec<GeneratedTest> {
    let mut out = Vec::new();
    for &op in &IntOp::ALL {
        for sign in renderings(op) {
            for reason in ALL_REASONS {
                for case in Case::ALL {
                    if let Some(test) = rejecting_test(op, sign, reason, case) {
                        out.push(test);
                    }
                }
            }
        }
    }
    out
}

/// Every reason the model can give, which is the axis the rejecting corpus is indexed by.
const ALL_REASONS: [Reject; 4] = [
    Reject::Overflow,
    Reject::DivByZero,
    Reject::DivOverflow,
    Reject::ShiftAmount,
];

/// Which input a rejecting program is rendered for.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Case {
    /// The narrowest, earliest pair the model rejects for the reason — see [`first_rejection`].
    ///
    /// Every `(operation, reading, reason)` that rejects anything at all has one of these.
    Narrowest,

    /// The same cell again at the **widest** width it rejects at, where the check's bound is one
    /// the width decides.
    ///
    /// [`Case::Narrowest`] settles which rejections exist; this one settles that each is still
    /// enforced where its bound is largest. The distinction it turns on is whether a wrong bound
    /// would even be visible at another width:
    ///
    /// - [`Reject::Overflow`] is a magnitude bound. `mul_overflows_nonzero` divides `mask(bits)`
    ///   and `signed_mul_magnitude_overflows` divides `2^(bits−1) − 1 + sign`, so both are a
    ///   different number at every width, and `sign_bit` reads a different bit.
    /// - [`Reject::ShiftAmount`] compares the amount against `bits` itself.
    /// - [`Reject::DivOverflow`] tests against `INT_MIN`, which is a width-derived constant.
    /// - [`Reject::DivByZero`] is **not** included. Its check is `rhs == 0`, an equality against
    ///   the type's own zero, and that is the same test at every width — a second program would
    ///   buy a `STATUS.md` row and no coverage.
    ///
    /// A cell whose widest rejecting width _is_ its narrowest renders nothing here rather than a
    /// duplicate of the program above.
    Widest,

    /// A shift by an amount whose _signed_ reading is negative, at the widest signed width.
    NegativeAmount,
}

impl Case {
    const ALL: [Case; 3] = [Case::Narrowest, Case::Widest, Case::NegativeAmount];

    /// The suffix this case adds to the generated package name, so the two never collide.
    const fn suffix(self) -> &'static str {
        match self {
            Case::Narrowest => "",
            Case::Widest => "_widest",
            Case::NegativeAmount => "_negative",
        }
    }
}

// THE ACCEPTING CORPUS
// ================================================================================================

fn accepting_test(ops: &[IntOp]) -> GeneratedTest {
    let cells: Vec<Cell> = ops
        .iter()
        .flat_map(|&op| renderings(op).into_iter().map(move |sign| (op, sign)))
        .flat_map(|(op, sign)| {
            widths_for(op, sign)
                .into_iter()
                .filter_map(move |bits| accepting_cell(op, sign, bits))
        })
        .collect();
    let op = ops[0];

    let mut pool = Pool::default();
    for cell in &cells {
        for &(lhs, rhs, _) in &cell.pairs {
            pool.record(cell.sign, cell.bits, lhs);
            pool.record(cell.sign, cell.bits, rhs);
        }
    }

    let pool = pool.finish();

    let mut body = String::new();
    for cell in &cells {
        body.push_str(&format!(
            "\n    // {} at {} bits: {} pair(s).\n",
            reading_word(cell.sign),
            cell.bits,
            cell.pairs.len()
        ));
        for &(lhs, rhs, expected) in &cell.pairs {
            body.push_str(&format!(
                "    assert_eq({} {} {}, {});\n",
                pool.name(cell.sign, cell.bits, lhs),
                op.symbol(),
                pool.name(cell.sign, cell.bits, rhs),
                literal(cell.sign, cell.bits, expected),
            ));
        }
    }

    let name = format!("int_semantics_{}", op.name());
    let main_nr = format!(
        "{}fn main(\n{}) {{{}}}\n",
        accepting_header(op, &cells),
        pool.signature(),
        body
    );

    GeneratedTest {
        name,
        main_nr,
        prover_toml: pool.prover_toml(),
        expect_failure: false,
    }
}

/// The accepting `(lhs, rhs, expected)` triples for one `(op, sign, width)` cell.
fn accepting_cell(op: IntOp, sign: Sign, bits: usize) -> Option<Cell> {
    let lhs_values = corners::values(bits);
    let rhs_values = if op.is_shift() {
        in_range_amounts(bits)
    } else {
        corners::values(bits)
    };

    let mut accepted = Vec::new();
    for &lhs in &lhs_values {
        for &rhs in &rhs_values {
            if let Outcome::Value(expected) = eval(op, &pattern(bits, lhs), &pattern(bits, rhs)) {
                accepted.push((lhs, rhs, host(&expected)));
            }
        }
    }

    let wanted = if op.reading_matters() {
        PAIRS_PER_CELL
    } else {
        PAIRS_PER_READING_BLIND_CELL
    };
    let pairs = spread(accepted, wanted);
    if pairs.is_empty() {
        None
    } else {
        Some(Cell { sign, bits, pairs })
    }
}

struct Cell {
    sign: Sign,
    bits: usize,
    pairs: Vec<(u128, u128, u128)>,
}

// THE REJECTING CORPUS
// ================================================================================================

fn rejecting_test(op: IntOp, sign: Sign, reason: Reject, case: Case) -> Option<GeneratedTest> {
    let (bits, lhs, rhs) = rejection_for(op, sign, reason, case)?;

    // The assertion is written against the answer a _total_ backend produces for this input, so
    // that removing the guard makes the program pass rather than fail. Where `residue` declines to
    // specify the answer there is nothing to assert -- see `sink` below.
    let (checked, sink) = match residue(op, &pattern(bits, lhs), &pattern(bits, rhs)) {
        Some(expected) => (
            format!(
                "    assert_eq(a {} b, {});\n",
                op.symbol(),
                literal(sign, bits, host(&expected))
            ),
            String::new(),
        ),

        // The header says why there is nothing to assert here. `zero` is a witness, so the multiply
        // cannot fold away and `r` stays live to preserve rejection, while the assertion holds
        // whatever `r` turns out to be. The sink parameter comes from the same arm as the body that
        // uses it, so the two cannot come apart.
        None => (
            format!(
                "    let r = a {} b;\n    assert_eq(r * zero, 0);\n",
                op.symbol()
            ),
            format!(", zero: {}", type_name(sign, bits)),
        ),
    };

    let mut prover_toml = format!(
        "a = \"{}\"\nb = \"{}\"\n",
        literal(sign, bits, lhs),
        literal(sign, bits, rhs)
    );
    if !sink.is_empty() {
        prover_toml.push_str("zero = \"0\"\n");
    }

    let ty = type_name(sign, bits);
    let main_nr = format!(
        "{}fn main(a: {ty}, b: {ty}{sink}) {{\n{checked}}}\n",
        rejecting_header(op, sign, bits, lhs, rhs, reason, case),
    );

    Some(GeneratedTest {
        name: format!(
            "int_semantics_{}_{}_{}{}_fails",
            op.name(),
            reading_letter(sign),
            reason_name(reason),
            case.suffix()
        ),
        main_nr,
        prover_toml,
        expect_failure: true,
    })
}

/// A `bits`-wide pattern from the host word this generator carries its values as.
///
/// The generator is host-typed throughout — it sorts, deduplicates and formats these as Noir
/// literals — so an [`IntBits`] exists here only for the length of a call into the model.
fn pattern(bits: usize, value: u128) -> IntBits {
    IntBits::from_u128(bits, value)
}

/// The host word a model answer denotes, the other half of [`pattern`].
fn host(value: &IntBits) -> u128 {
    u128::try_from(value).expect("a pattern no wider than MAX_BITS fits a host word")
}

/// The operands one rejecting program is built from, or `None` where the case has no program.
fn rejection_for(op: IntOp, sign: Sign, reason: Reject, case: Case) -> Option<(usize, u128, u128)> {
    match case {
        Case::Narrowest => first_rejection(op, sign, reason),
        Case::Widest => widest_rejection(op, sign, reason),
        Case::NegativeAmount => negative_amount_rejection(op, sign, reason),
    }
}

/// The earliest corner pair rejected for `reason` at the **widest** width that rejects at all.
///
/// [`first_rejection`] with the width axis reversed, and two gates on top. The reason has to be
/// one whose check bound the width decides — see [`Case::Widest`] — and the width found has to
/// differ from the narrowest one, since rendering the same program under two names would cost a
/// row for nothing.
fn widest_rejection(op: IntOp, sign: Sign, reason: Reject) -> Option<(usize, u128, u128)> {
    if reason == Reject::DivByZero {
        return None;
    }

    let narrowest = first_rejection(op, sign, reason)?.0;
    let mut widths = widths_for(op, sign);
    widths.reverse();
    let (bits, lhs, rhs) = search_rejection(op, reason, &widths)?;
    (bits != narrowest).then_some((bits, lhs, rhs))
}

/// The earliest corner pair rejected for a **negative amount** at the widest signed width.
///
/// Only a signed shift has one: the reading is what makes an amount negative, and the width is
/// pinned to the widest rather than searched for the reason [`Case::NegativeAmount`] gives. The
/// amounts are filtered by their _reading_ rather than by magnitude.
fn negative_amount_rejection(op: IntOp, sign: Sign, reason: Reject) -> Option<(usize, u128, u128)> {
    if !(op.is_shift() && sign == Sign::Signed && reason == Reject::ShiftAmount) {
        return None;
    }

    let bits = *widths_for(op, sign).last()?;
    let amounts: Vec<u128> = corners::shift_amounts(bits, bits)
        .into_iter()
        .filter(|&a| pattern(bits, a).to_signed() < SignedValue::from(0u8))
        .collect();

    // Zero last, for the reason `first_rejection` gives.
    let mut lhs_values = corners::values(bits);
    lhs_values.sort_by_key(|&v| v == 0);

    for &lhs in &lhs_values {
        for &rhs in &amounts {
            if eval(op, &pattern(bits, lhs), &pattern(bits, rhs)) == Outcome::Rejected(reason) {
                return Some((bits, lhs, rhs));
            }
        }
    }
    None
}

/// The narrowest, earliest corner pair this model rejects for `reason`, if there is one.
///
/// Searching rather than listing is the point: the corpus states which rejections the model has,
/// so a `Reject` variant that gains or loses a reachable input changes the set of generated
/// directories instead of drifting away from a hand-written list.
fn first_rejection(op: IntOp, sign: Sign, reason: Reject) -> Option<(usize, u128, u128)> {
    search_rejection(op, reason, &widths_for(op, sign))
}

/// The first corner pair rejected for `reason`, scanning `widths` in the order given.
///
/// The order is the caller's whole contribution: [`first_rejection`] passes the widths ascending
/// and [`widest_rejection`] descending, and neither has an opinion about anything else.
fn search_rejection(op: IntOp, reason: Reject, widths: &[usize]) -> Option<(usize, u128, u128)> {
    for &bits in widths {
        let rhs_values = if op.is_shift() {
            corners::shift_amounts(bits, bits)
        } else {
            corners::values(bits)
        };
        // A zero left operand annihilates most operations, so a rejection found with one makes the
        // weakest possible test: `0 << 8` and `0 / 0` fail their guard, but they would also produce
        // the asserted answer if the guard were deleted and the arithmetic were wrong. Trying zero
        // last costs nothing and leaves the guard as the only thing the program can trip on.
        let mut lhs_values = corners::values(bits);
        lhs_values.sort_by_key(|&v| v == 0);
        for &lhs in &lhs_values {
            for &rhs in &rhs_values {
                if eval(op, &pattern(bits, lhs), &pattern(bits, rhs)) == Outcome::Rejected(reason) {
                    return Some((bits, lhs, rhs));
                }
            }
        }
    }
    None
}

// THE OPERAND POOL
// ================================================================================================

/// The witness parameters a generated program declares.
///
/// Values are pooled per `(reading, width)` and named by their position in the _sorted_ set, so the
/// same corpus renders byte-identically however the cells that use them are ordered.
#[derive(Default)]
struct Pool {
    used: BTreeSet<(usize, bool, u128)>,
}

/// A pool with its parameter names decided, which is what the renderers actually read.
struct NamedPool {
    /// Every group, in declaration order, as `(reading, width, values in reading order)`.
    groups: Vec<(Sign, usize, Vec<u128>)>,

    /// The parameter name each pooled value is declared under, keyed by `(width, signed, value)`.
    names: BTreeMap<(usize, bool, u128), String>,
}

impl Pool {
    fn record(&mut self, sign: Sign, bits: usize, value: u128) {
        self.used.insert((bits, sign == Sign::Signed, value));
    }

    /// Freeze the pool, assigning each value its parameter name.
    fn finish(self) -> NamedPool {
        // Sorted by reading first, so the declaration order matches the order the body visits its
        // cells in: every unsigned group, then every signed one.
        let mut keys: Vec<(bool, usize)> = self
            .used
            .iter()
            .map(|&(bits, signed, _)| (signed, bits))
            .collect();
        keys.sort_unstable();
        keys.dedup();

        let mut groups = Vec::new();
        let mut names = BTreeMap::new();
        for (signed, bits) in keys {
            let sign = if signed { Sign::Signed } else { Sign::Unsigned };
            let mut values: Vec<u128> = self
                .used
                .iter()
                .filter(|&&(b, s, _)| b == bits && s == signed)
                .map(|&(_, _, v)| v)
                .collect();
            // Sorting signed values by their _reading_ rather than their pattern keeps the
            // generated `Prover.toml` monotonic, which is what makes it readable.
            if signed {
                values.sort_by_key(|&v| pattern(bits, v).to_signed());
            } else {
                values.sort_unstable();
            }
            for (index, &value) in values.iter().enumerate() {
                names.insert(
                    (bits, signed, value),
                    format!("{}_{index}", type_name(sign, bits)),
                );
            }
            groups.push((sign, bits, values));
        }

        NamedPool { groups, names }
    }
}

impl NamedPool {
    fn name(&self, sign: Sign, bits: usize, value: u128) -> &str {
        self.names
            .get(&(bits, sign == Sign::Signed, value))
            .expect("ICE: a pair used a value the pool never recorded")
    }

    fn signature(&self) -> String {
        let mut out = String::new();
        for (sign, bits, values) in &self.groups {
            for index in 0..values.len() {
                let ty = type_name(*sign, *bits);
                out.push_str(&format!("    {ty}_{index}: {ty},\n"));
            }
        }
        out
    }

    fn prover_toml(&self) -> String {
        let mut out = String::new();
        for (sign, bits, values) in &self.groups {
            for (index, &value) in values.iter().enumerate() {
                out.push_str(&format!(
                    "{}_{index} = \"{}\"\n",
                    type_name(*sign, *bits),
                    literal(*sign, *bits, value)
                ));
            }
        }
        out
    }
}

// RENDERING HELPERS
// ================================================================================================

/// A `bits`-wide pattern as a Noir literal, under `sign`'s reading.
fn literal(sign: Sign, bits: usize, value: u128) -> String {
    match sign {
        Sign::Unsigned => value.to_string(),
        Sign::Signed => pattern(bits, value).to_signed().to_string(),
    }
}

fn type_name(sign: Sign, bits: usize) -> String {
    let prefix = reading_letter(sign);
    format!("{prefix}{bits}")
}

const fn reading_letter(sign: Sign) -> char {
    match sign {
        Sign::Unsigned => 'u',
        Sign::Signed => 'i',
    }
}

const fn reading_word(sign: Sign) -> &'static str {
    match sign {
        Sign::Unsigned => "unsigned",
        Sign::Signed => "signed",
    }
}

const fn reason_name(reason: Reject) -> &'static str {
    match reason {
        Reject::Overflow => "overflow",
        Reject::DivByZero => "by_zero",
        Reject::DivOverflow => "min_over_neg_one",
        Reject::ShiftAmount => "amount",
    }
}

/// The Noir **types** worth declaring an operation's operands as.
///
/// A `Sign` here is a fact about the generated _source_, not about the model: it picks `u8` over
/// `i8` in a signature and how a literal is spelled, which is all `Pool` and [`literal`] use it
/// for.
/// An operation names its own reading, so the type and the reading coincide wherever there is one.
/// Where there is not, both types are worth generating anyway, because a signed _type_ takes its
/// own route through the lowering even when the operation cannot tell the difference. How many
/// pairs that is worth is [`PAIRS_PER_READING_BLIND_CELL`].
///
/// The result is twenty `(op, sign)` pairs: twelve operations that name a reading, once each, plus
/// `And`/`Or`/`Xor`/`Shl` twice each.
fn renderings(op: IntOp) -> Vec<Sign> {
    op.sign()
        .map_or_else(|| Sign::ALL.to_vec(), |sign| vec![sign])
}

/// The widths a generated program may use for `(op, sign)`.
fn widths_for(op: IntOp, sign: Sign) -> Vec<usize> {
    NOIR_WIDTHS
        .iter()
        .copied()
        .filter(|&bits| {
            // Mavros caps a signed reading at 64 bits, so `i128` is rejected at compile time.
            if sign == Sign::Signed && bits > MAX_SIGNED_BITS {
                return false;
            }
            // `witness_bitwise::product_headroom_or_bail` refuses a 128-bit left shift, so the
            // program would not compile. Every other operation runs the full unsigned set.
            !(op == IntOp::Shl && bits == MAX_BITS)
        })
        .collect()
}

/// Shift amounts this model accepts at `bits`, taken from the corner set rather than `0..bits`.
///
/// The corners are where the boundary bugs live: `bits - 1` is the largest legal amount and
/// `bits / 2` sits where a decomposition changes shape.
fn in_range_amounts(bits: usize) -> Vec<u128> {
    corners::shift_amounts(bits, bits)
        .into_iter()
        .filter(|&a| a < bits as u128)
        .collect()
}

/// Take `wanted` items spread evenly across `items`, keeping their order.
///
/// Evenly rather than from the front, because the accepting set is generated by a nested loop over
/// sorted corners: its first `n` entries all share a left operand, and would sample one row of the
/// matrix instead of the matrix.
fn spread<T>(items: Vec<T>, wanted: usize) -> Vec<T> {
    if items.len() <= wanted {
        return items;
    }
    let len = items.len();
    let mut kept: Vec<T> = Vec::with_capacity(wanted);
    for (index, item) in items.into_iter().enumerate() {
        if kept.len() < wanted && index == kept.len() * len / wanted {
            kept.push(item);
        }
    }
    kept
}

// FILE HEADERS
// ================================================================================================

fn generated_banner() -> String {
    "// GENERATED FILE -- do not edit by hand.\n\
     //\n\
     // Rendered by `mavros-int-semantics`'s `corpus` module from the model itself. Regenerate with\n\
     // `MAVROS_BLESS=1 cargo test -p mavros-int-semantics`, and read the diff: a program whose\n\
     // expected answer moved is a program whose *meaning* moved.\n"
        .to_string()
}

fn accepting_header(op: IntOp, cells: &[Cell]) -> String {
    let mut out = generated_banner();
    out.push_str(&format!(
        "//\n// Every operand here is a `main` parameter, and therefore a witness. That is the point: a\n\
         // constant pair would be folded before the guards, the witness lowering, the VM and the WASM\n\
         // backend ever saw it, and those are the five evaluators a unit test cannot reach. The\n\
         // `assert_eq` then makes each answer observable to the R1CS-satisfaction oracle in every lane.\n\
         //\n\
         // Operation: `{}`. Cells, as (reading, width, pairs):\n",
        op.symbol()
    ));
    for cell in cells {
        out.push_str(&format!(
            "//   * {}, {} bits, {} pair(s)\n",
            reading_word(cell.sign),
            cell.bits,
            cell.pairs.len()
        ));
    }
    out.push('\n');
    out
}

fn rejecting_header(
    op: IntOp,
    sign: Sign,
    bits: usize,
    lhs: u128,
    rhs: u128,
    reason: Reject,
    case: Case,
) -> String {
    let mut out = generated_banner();
    out.push_str(&format!(
        "//\n// This program must be REJECTED. `{}` on {} {}-bit operands `{}` and `{}` is\n\
         // `Reject::{reason:?}` in the model, so a run that produces a proof means the guard that owes\n\
         // this rejection is missing.\n\
         //\n\
         // Both operands are witnesses, which is the half the hand-written `noir_failure_tests` do not\n\
         // cover: those put the operands behind a function so the *pure* check is under test, whereas\n\
         // here the check is owed by the witness lowering instead.\n",
        op.symbol(),
        reading_word(sign),
        bits,
        literal(sign, bits, lhs),
        literal(sign, bits, rhs),
    ));
    if case == Case::Widest {
        out.push_str(
            "//\n// The width here is the *widest* this cell rejects at rather than the narrowest, and there\n\
             // is a sibling program at the narrowest. The pair exists because this rejection's bound is\n\
             // one the width decides -- a magnitude limit, the amount bound itself, or `INT_MIN` -- so a\n\
             // check that is right at eight bits carries no evidence that it is right here.\n",
        );
    }
    if case == Case::NegativeAmount {
        out.push_str(
            "//\n// The amount here is *negative* rather than merely too large, and the width is the widest\n\
             // signed one rather than the narrowest rejecting one. That pairing is deliberate: it is the\n\
             // only shape that reaches `shift_guard`'s negative-amount conjunct, which is dead at every\n\
             // narrower width because the widening cast ahead of the bound test clears the sign bit.\n",
        );
    }
    if residue(op, &pattern(bits, lhs), &pattern(bits, rhs)).is_some() {
        out.push_str(
            "//\n// The assertion is written against the answer a total backend produces anyway, so removing\n\
             // the guard makes this program PASS -- which an expect-failure row reports as a failure.\n",
        );
    } else {
        out.push_str(
            "//\n// The model declines to specify an answer for this input, so there is nothing to assert\n\
             // against. The multiply by a witness zero keeps the result live -- a dead witness operation\n\
             // loses its rejection -- without claiming a value the model does not have.\n",
        );
    }
    out.push('\n');
    out
}

// TESTS
// ================================================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Every accepting assertion states the answer [`eval`] gives, at the cell's own width.
    ///
    /// The bless test compares the tree against the generator, which a generator that renders the
    /// wrong answer satisfies just as well. This is the check that the answers are the model's.
    #[test]
    fn every_expected_answer_is_the_models() {
        let mut checked = 0;
        for &op in &IntOp::ALL {
            for sign in renderings(op) {
                for bits in widths_for(op, sign) {
                    let Some(cell) = accepting_cell(op, sign, bits) else { continue };
                    for (lhs, rhs, expected) in cell.pairs {
                        assert_eq!(
                            eval(op, &pattern(bits, lhs), &pattern(bits, rhs)),
                            Outcome::Value(pattern(bits, expected)),
                            "{op:?} {sign:?} at {bits} bits: {lhs:#x} and {rhs:#x}"
                        );
                        checked += 1;
                    }
                }
            }
        }
        assert!(
            checked > 300,
            "only {checked} pairs -- the corpus has quietly emptied out"
        );
    }

    /// Every rejecting program's operands are ones the model rejects, for the reason it is named
    /// after.
    #[test]
    fn every_rejecting_program_is_one_the_model_rejects() {
        for &op in &IntOp::ALL {
            for sign in renderings(op) {
                for reason in ALL_REASONS {
                    for case in Case::ALL {
                        let Some((bits, lhs, rhs)) = rejection_for(op, sign, reason, case) else {
                            continue;
                        };
                        assert_eq!(
                            eval(op, &pattern(bits, lhs), &pattern(bits, rhs)),
                            Outcome::Rejected(reason)
                        );
                    }
                }
            }
        }
    }

    /// Every reason the model can give is represented by at least one program.
    ///
    /// A `Reject` variant that stops being reachable would otherwise drop out of the corpus in
    /// silence, taking its coverage with it.
    #[test]
    fn every_rejection_reason_reaches_the_corpus() {
        let names: Vec<String> = rejecting_tests().into_iter().map(|t| t.name).collect();
        for reason in ALL_REASONS {
            let tag = reason_name(reason);
            assert!(
                names.iter().any(|n| n.contains(tag)),
                "no generated program covers Reject::{reason:?}"
            );
        }
    }

    /// The negative-amount case reaches the corpus, at a width where the check it targets is live.
    ///
    /// Without this the case could quietly stop generating (a width filter, a corner set that lost
    /// its negative amounts) and the only conjunct in `shift_guard` with no other coverage would go
    /// back to having none with every row still green.
    #[test]
    fn the_negative_shift_amount_case_reaches_the_corpus() {
        for op in [IntOp::Shl, IntOp::SShr] {
            let (bits, _, rhs) = negative_amount_rejection(op, Sign::Signed, Reject::ShiftAmount)
                .unwrap_or_else(|| panic!("{op:?} generates no negative-amount program"));
            assert_eq!(
                bits, MAX_SIGNED_BITS,
                "the conjunct is dead below {MAX_SIGNED_BITS} bits"
            );
            assert!(
                pattern(bits, rhs).to_signed() < SignedValue::from(0u8),
                "the amount is not negative"
            );
        }

        let names: Vec<String> = rejecting_tests().into_iter().map(|t| t.name).collect();
        for op in [IntOp::Shl, IntOp::SShr] {
            let want = format!("int_semantics_{}_i_amount_negative_fails", op.name());
            assert!(names.contains(&want), "{want} was not generated");
        }
    }

    /// No two generated tests share a directory name.
    ///
    /// They are written by name, so a collision would silently overwrite one with the other and
    /// lose its coverage rather than fail.
    #[test]
    fn the_generated_names_are_unique() {
        let mut names: Vec<String> = accepting_tests()
            .into_iter()
            .chain(rejecting_tests())
            .map(|t| t.name)
            .collect();
        let total = names.len();
        names.sort();
        names.dedup();
        assert_eq!(names.len(), total, "two generated tests share a name");
    }

    /// No generated program names a width Noir does not have, or a signed width Mavros cannot read.
    #[test]
    fn no_program_names_a_width_that_does_not_exist() {
        for &op in &IntOp::ALL {
            for sign in renderings(op) {
                for bits in widths_for(op, sign) {
                    assert!(
                        NOIR_WIDTHS.contains(&bits),
                        "{bits} is not a Noir integer width"
                    );
                    if sign == Sign::Signed {
                        assert!(
                            bits <= MAX_SIGNED_BITS,
                            "i{bits} is wider than Mavros can read"
                        );
                    }
                }
            }
        }
    }

    /// Every parameter a generated program declares is used by its body.
    ///
    /// The pool is built from the pairs, so an unused parameter means the two have come apart —
    /// and it would cost a witness column per occurrence for nothing.
    #[test]
    fn every_parameter_a_program_declares_is_used() {
        let mut declared = 0;
        for test in accepting_tests() {
            let after_header = test
                .main_nr
                .split_once("fn main(\n")
                .expect("ICE: a program with no `main`")
                .1;
            let (signature, body) = after_header
                .split_once(") {")
                .expect("ICE: a `main` with no body");
            for line in signature.lines().filter(|l| l.contains(": ")) {
                let param = line
                    .trim()
                    .split(':')
                    .next()
                    .expect("ICE: a declaration with no name");
                assert!(
                    body.contains(param),
                    "{}: `{param}` is declared but never used",
                    test.name
                );
                declared += 1;
            }
        }
        assert!(
            declared > 100,
            "only {declared} parameters -- the signature parse has drifted"
        );
    }

    /// [`spread`] samples across the whole input rather than truncating its front.
    ///
    /// This is the property the accepting corpus depends on: its candidate pairs come out of a
    /// nested loop over sorted corners, so the first `n` of them all share a left operand.
    #[test]
    fn spread_samples_the_whole_range_not_its_front() {
        let items: Vec<usize> = (0..100).collect();
        let kept = spread(items, 5);
        assert_eq!(kept.len(), 5);
        assert!(
            kept[0] < 20 && *kept.last().unwrap() > 70,
            "sampled {kept:?}"
        );
        assert!(
            kept.windows(2).all(|w| w[0] < w[1]),
            "order not preserved: {kept:?}"
        );

        // Fewer items than asked for is the whole list, unchanged.
        assert_eq!(spread(vec![1, 2, 3], 10), vec![1, 2, 3]);
    }
}
