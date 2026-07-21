//! An analysis of post-sealing opportunities for compacting the R1CS without changing the meaning
//! of the constraint system.
//!
//! Much of the redundancy in the sealed constraint system exists only post-unroll (per-iteration
//! selector constants, subcircuits duplicated by repeated constrained calls minting fresh witness
//! advice), so no SSA pass can see it. This module measures how many algebraic rows and columns a
//! post-seal compaction pass could remove, without transforming anything; the numbers surface per
//! test in STATUS.md and aggregate in the CI growth report.
//!
//! ## Soundness
//!
//! A *satisfying assignment* is a witness vector with `w_0 = 1` satisfying every row
//! `(A·w)(B·w) − C·w = 0`. `w_0 = 1` is a system-wide convention: rows are homogeneous, so no row
//! can express it; it is established by the entry-point wrapper's pinned write and by the
//! proof system pinning the constant wire. The analysis models a hypothetical pass that applies
//! its final substitution — merged columns read from their surviving representative, pinned columns
//! become constants folded into column 0, and generally-eliminated columns become affine
//! combinations of surviving columns — to _all_ rows including the non-algebraic sections, deletes
//! the dead rows, and deletes the eliminated columns.
//!
//! The unifying principle is that any **linear** (degree ≤ 1) constraint over the witnesses lets
//! Gaussian elimination solve for one unprotected column and delete it with its row. Every equality
//! the substitution records is a value-preserving `w_j := (affine LC)` derived from such a
//! relation, and holds in every satisfying assignment of the original system, by induction over
//! discovery order:
//!
//! - A tautological row holds identically, for any witness with `w_0 = 1`.
//! - A row whose product `A·B` is identically constant is linear: `A ≡ 0` gives `C·w = 0`, and
//!   `A ≡ k·w_0` gives `(k·B − C)·w = 0` (symmetrically for `B`). Solving such a relation for its
//!   lowest-index unprotected column `w_j` (invertible coefficient, since canonicalization strips
//!   zeros) records `w_j = −(1/k)·(rest)`. Its degenerate shapes are a pin (`rest` constant) and a
//!   merge (`rest` a single unit-coefficient column); the general shape is an affine substitution.
//! - Two rows sharing a product (`A` and `B` identical after canonicalization) subtract to
//!   `(C_1 − C_2)·w = 0` — the products cancel exactly — an entailed linear relation solved the
//!   same way. Canonical forms produced under _different_ substitution versions are each valid
//!   constraints in their own right, so identical products license the subtraction with soundness
//!   never depending on version synchronicity.
//! - A row identical to a live earlier row is the same constraint.
//!
//! This results in a bijection between the solution sets of the original and compacted systems. In
//! the forward direction, restrict a solution to the surviving columns. In the backward direction,
//! lift by `w_merged := w_root`, `w_pinned := const`, and `w_subbed := (its affine LC over
//! surviving columns)`. This lift satisfies every dropped row, by induction over _reverse_ death
//! order. A dropped row's final image equals the final image of its death-time canonical form,
//! because substitution versions compose (applying the final subst after any intermediate one
//! equals applying it directly: union-find refines monotonically, pins and subs are never
//! re-valued, eliminated columns never re-enter a canonical row, and column 0 is never remapped).
//!
//! That image is a tautology (the patterns are preserved by further substitution, which is linear),
//! folds to `0 = 0` at exactly the pinned value (its own pin), or — for a duplicate — equals the
//! image of its twin, which either survives or dies strictly later, so the chain terminates at a
//! surviving row or a self-discharging death. Crucially, **an unsatisfiable system remains
//! unsatisfiable**: contradictory pins leave a live `0·B − c·w_0 = 0` (`c ≠ 0`) residue that
//! matches no removal pattern.
//!
//! The backward direction is the adversarial one, but as any adversarial solution to the compacted
//! system lifts to a solution of the original system, **no new statements become provable**.
//!
//! Column 0 and the positional in/return block (`protected_cols`) are **never** removable.
//! Protected columns are the lowest indices and [`UnionFind`] keeps the `Ord`-smallest root, so a
//! class containing a protected column would have a protected root. Pins reject protected roots,
//! and phase 3 skips protected defining columns entirely, so union-find classes never contain a
//! protected column at all and unions only ever eliminate roots at or above `protected_cols`. The
//! public statement (the projection onto the protected columns) is therefore unchanged.
//!
//! ## Estimation
//!
//! The counters cannot over-count: a row dies at most once, paired with exactly one counter
//! (`removable_rows == duplicate_rows + tautology_rows + pinned_cols + linear_elim_rows`; a pin and
//! a standalone linear elimination each kill their own row, while a merge or same-`(A,B)`
//! elimination kills no row — the row later dies as a duplicate).
//!
//! Each elimination removes _exactly_ one distinct surviving column, and an eliminated column never
//! re-enters a canonical row, so pins, merges, and general substitutions count disjoint columns and
//! `removable_cols == pinned_cols + merged_cols + eliminated_cols`. The one subtlety is potential
//! double-counting *dependent* linear rows: the elimination phases apply the substitution LIVE, so
//! a row made redundant by an earlier elimination this pass re-canonicalizes to `0 = 0` and
//! eliminates nothing further. Everything else errs conservative (a column whose only relations
//! involve protected or already-eliminated columns is simply kept), so the analysis only ever
//! under-counts.
//!
//! The fixpoint runs over the **algebraic section only**. Table and lookup rows are never droppable
//! (they are positional, and look degenerate before challenges are filled), and they cannot
//! influence the algebraic section. Algebraic rows reference only algebraic columns, because the
//! non-algebraic columns are allocated later, inside `R1CGen::seal()`. Non-algebraic rows may
//! reference algebraic columns, but extra constraints only shrink the solution set, so equalities
//! derived from the algebraic rows alone remain valid for the full system; the hypothetical pass
//! rewrites those references through the same substitution with unchanged values.
//!
//! This stays true for a *multi-term* substitution: the lookup rows are a logUp (log-derivative)
//! argument whose key and value positions are already arbitrary linear combinations of algebraic
//! columns (the committed helper/multiplicity/challenge columns are all non-algebraic), so
//! expanding a column into an LC there keeps every row a valid R1C with unchanged value.
//!
//! ## Determinism
//!
//! the stats land in the `END:R1CS` stage line, which the determinism harness fingerprints. All
//! iteration is in row/column index order and merges go through [`UnionFind`], whose smaller-root
//! tie-break keeps the partition deterministic.

use ark_ff::AdditiveGroup;

use crate::collections::{HashMap, UnionFind};
use crate::compiler::Field;

use super::hlssa_to_r1cs::{LC, R1CS};

// ANALYSIS
// ================================================================================================
//
/// Measure how much a post-seal compaction could remove from `r1cs`'s algebraic section.
///
/// `protected_cols` is the number of leading witness columns that must survive untouched: column 0
/// (the constant one) plus the positional input/return block, i.e. `1 + flattened field count of
/// the entry point's params and returns`.
pub fn analyze(r1cs: &R1CS, protected_cols: usize) -> CompactionStats {
    // With column 0 unprotected, `pin_candidate` could "pin" the constant-one column itself,
    // silently corrupting every constant fold below.
    assert!(
        protected_cols >= 1,
        "column 0 (the constant one) must always be protected"
    );

    // Columns are narrowed to `u32` for interning; wider indices would silently alias.
    assert!(
        r1cs.witness_layout.algebraic_size <= u32::MAX as usize,
        "algebraic column indices must fit in u32"
    );

    let algebraic_rows = r1cs.constraints_layout.algebraic_size;
    let mut stats = CompactionStats {
        algebraic_rows,
        algebraic_cols: r1cs.witness_layout.algebraic_size,
        ..CompactionStats::default()
    };
    let protected = protected_cols as u32;

    let mut interner = CoeffInterner::default();
    let mut subst = Substitution::new();
    let mut live = vec![true; algebraic_rows];

    // Canonical form of each row, kept as `None` while the row is dead. Persisted across rounds
    // rather than rebuilt from the original constraints each round: a live row's stored form is
    // re-folded through the substitution as it has evolved (via `apply_canon`), which by
    // substitution-version composition equals folding the original row afresh, but lets untouched
    // rows short-circuit to a cheap clone instead of a full re-fold. The `None`/`Some` split still
    // exactly mirrors `live`, so the pattern phases below are byte-for-byte unchanged.
    let mut canon: Vec<Option<CanonRow>> = vec![None; algebraic_rows];

    loop {
        stats.rounds += 1;
        let mut changed = false;

        // (Re)canonicalize every live row under the substitution as of this round's start. The
        // pattern phases below all work on these forms; effects of this round's pins/merges are
        // only visible after the next round's re-canonicalization, which is what drives the cascade
        // (merge -> rows become identical -> duplicate drop next round).
        //
        // A row is folded from its original constraint the first time it is canonicalized (its
        // stored form is `None`); thereafter the stored form is re-folded through the current
        // substitution. Dead rows are reset to `None` so the `Some`/`None` split keeps tracking
        // `live` exactly.
        for idx in 0..algebraic_rows {
            if !live[idx] {
                canon[idx] = None;
                continue;
            }
            let (a, b, c) = match canon[idx].take() {
                Some(row) => (
                    subst.apply_canon(&row.a, &mut interner),
                    subst.apply_canon(&row.b, &mut interner),
                    subst.apply_canon(&row.c, &mut interner),
                ),
                None => {
                    let r1c = &r1cs.constraints[idx];
                    (
                        subst.apply(&r1c.a, &mut interner),
                        subst.apply(&r1c.b, &mut interner),
                        subst.apply(&r1c.c, &mut interner),
                    )
                }
            };
            let (a, b) = if b < a { (b, a) } else { (a, b) };
            canon[idx] = Some(CanonRow { a, b, c });
        }

        // Phase 1+2: drop tautologies, pin constant columns. Row index order; a column pins at most
        // once per run (later rows that would re-pin it resolve to tautologies next round).
        for (idx, row) in canon.iter().enumerate() {
            let Some(row) = row else { continue };
            if is_tautology(row, &interner) {
                live[idx] = false;
                stats.tautology_rows += 1;
                changed = true;
            } else if let Some((col, value)) = pin_candidate(row, protected, &interner) {
                if !subst.is_pinned(col) {
                    let root = subst.uf.find(col);
                    subst.pins.insert(root, value);
                    live[idx] = false;
                    stats.pinned_cols += 1;
                    changed = true;
                }
            }
        }

        // Phase 2.5: eliminate one column from each standalone linear relation. A row whose product
        // `A·B` is linear — a multiplicand is identically constant, either empty (`≡0`) or a lone
        // `k·w₀` — reduces to a linear constraint `L·w = 0` (`L = C` when the product is 0, or
        // `L = C − k·B` when `A ≡ k·w₀`, and symmetrically). Solving it for one unprotected column
        // removes that column and consumes the row. Applied LIVE (like phase 3, not the round-start
        // forms) so two *dependent* linear rows can never eliminate two columns: once the first
        // eliminates `w_j`, the second re-canonicalizes to `0 = 0` and yields nothing. This
        // generalizes the single-term phase-2 pin to an arbitrary `C` (Claim 1) and folds a
        // constant multiplicand into the relation (Claim 2).
        for (idx, row) in canon.iter().enumerate() {
            let Some(row) = row else { continue };
            if !live[idx] {
                continue;
            }
            let a = subst.apply_canon(&row.a, &mut interner);
            let b = subst.apply_canon(&row.b, &mut interner);
            // `(k, product_lc)`: the product is `k·product_lc` (with `product_lc = None` meaning
            // the product is identically 0). `None` overall means a genuinely quadratic row — skip
            // it.
            let linear = if a.is_empty() || b.is_empty() {
                Some((Field::ZERO, None))
            } else if let Some(k) = lone_constant(&a, &interner) {
                Some((k, Some(b.clone())))
            } else if let Some(k) = lone_constant(&b, &interner) {
                Some((k, Some(a.clone())))
            } else {
                None
            };
            let Some((k, product_lc)) = linear else { continue };
            let c = subst.apply_canon(&row.c, &mut interner);
            // L = C − k·product_lc, folded.
            let mut folded = std::collections::BTreeMap::<u32, Field>::new();
            for &(col, coeff_id) in &c {
                *folded.entry(col).or_insert(Field::ZERO) += interner.value(coeff_id);
            }
            if let Some(lc) = &product_lc {
                for &(col, coeff_id) in lc {
                    *folded.entry(col).or_insert(Field::ZERO) -= k * interner.value(coeff_id);
                }
            }
            let terms: Vec<(u32, Field)> = folded
                .into_iter()
                .filter(|(_, v)| *v != Field::ZERO)
                .collect();
            match eliminate_linear(&mut subst, &terms, protected) {
                Elim::None => {}
                Elim::Pin => {
                    live[idx] = false;
                    stats.pinned_cols += 1;
                    changed = true;
                }
                Elim::Merge => {
                    live[idx] = false;
                    stats.merged_cols += 1;
                    stats.linear_elim_rows += 1;
                    changed = true;
                }
                Elim::Sub => {
                    live[idx] = false;
                    stats.eliminated_cols += 1;
                    stats.linear_elim_rows += 1;
                    changed = true;
                }
            }
        }

        // Phase 3: eliminate a column via each pair of rows sharing the same product. Rows are
        // grouped by their live-canonical `(A, B)`; the first row in index order is the group's
        // reference. For every later row `R` in the group, `A·B` cancels in `R − reference`, so
        // `D = C_R − C_reference = 0` is an entailed linear relation — solving it removes one
        // unprotected column, after which `R` becomes a duplicate of the reference and dies in
        // phase 4 (so no row is killed here). A single-term equal-coefficient `D` routes to a
        // union-find merge (the old defining-row rule, now the degenerate case); a multi-term or
        // scaled `D` is Marcin's generalization (Claim 3).
        //
        // Version skew between reference and `R` (canonicalized at different substitution versions)
        // is harmless: identical `(A, B)` means identical product forms, so both rows entail
        // `A·B·w = C·w` and their difference is entailed. `R`'s `C` is re-canonicalized live here,
        // but the stored reference `C` is a *snapshot* from when the group was created and may name
        // columns eliminated since, so `D` is not necessarily over surviving columns —
        // `eliminate_linear` re-folds it through the current substitution, which resolves those
        // stale columns (an already-satisfied `D` collapses to nothing). Applied in index order so
        // the fold-style cascade closes in one pass.
        let mut groups: HashMap<(CanonLc, CanonLc), CanonLc> = HashMap::default();
        for (idx, row) in canon.iter().enumerate() {
            // Rows killed above no longer prove anything this round.
            let Some(row) = row else { continue };
            if !live[idx] {
                continue;
            }
            let a = subst.apply_canon(&row.a, &mut interner);
            let b = subst.apply_canon(&row.b, &mut interner);
            let c = subst.apply_canon(&row.c, &mut interner);
            let (a, b) = if b < a { (b, a) } else { (a, b) };
            match groups.entry((a, b)) {
                std::collections::hash_map::Entry::Vacant(slot) => {
                    slot.insert(c);
                }
                std::collections::hash_map::Entry::Occupied(slot) => {
                    let terms = lc_difference(&c, slot.get(), &interner);
                    // No row is consumed here; the row dies as a duplicate in phase 4. So a Pin —
                    // possible when the difference fixes a column to a constant — is counted under
                    // `eliminated_cols`, not `pinned_cols` (which is reserved for pins that kill
                    // their own row, keeping the row-death partition exact).
                    match eliminate_linear(&mut subst, &terms, protected) {
                        Elim::None => {}
                        Elim::Merge => {
                            stats.merged_cols += 1;
                            changed = true;
                        }
                        Elim::Pin | Elim::Sub => {
                            stats.eliminated_cols += 1;
                            changed = true;
                        }
                    }
                }
            }
        }

        // Phase 4: drop exact duplicates among the rows still live this round.
        let mut seen: HashMap<&CanonRow, ()> = HashMap::default();
        for (idx, row) in canon.iter().enumerate() {
            let Some(row) = row else { continue };
            if !live[idx] {
                continue;
            }
            if seen.contains_key(row) {
                live[idx] = false;
                stats.duplicate_rows += 1;
                changed = true;
            } else {
                seen.insert(row, ());
            }
        }

        if !changed {
            break;
        }
    }

    stats.removable_rows = live.iter().filter(|l| !**l).count();
    stats.removable_cols = stats.pinned_cols + stats.merged_cols + stats.eliminated_cols;
    debug_assert_eq!(
        stats.removable_rows,
        stats.duplicate_rows + stats.tautology_rows + stats.pinned_cols + stats.linear_elim_rows,
        "row-death counters must partition the removable rows",
    );
    stats
}

// SHARED TYPES
// ================================================================================================

/// A linear combination in canonical form: substitution applied, like terms folded, zero
/// coefficients dropped, sorted by column.
///
/// Coefficients are interned so rows hash and compare on `(u32, u32)` pairs.
type CanonLc = Vec<(u32, u32)>;

// COMPACTION STATISTICS
// ================================================================================================

/// What a compaction pass could remove from the algebraic section, plus a per-mechanism
/// breakdown.
///
/// `removable_rows == duplicate_rows + tautology_rows + pinned_cols + linear_elim_rows`. Each pin
/// and each standalone linear elimination kills _exactly_ the row that established it; column
/// merges and same-`(A,B)` eliminations kill no row of their own — the row later dies as a
/// duplicate) and `removable_cols == pinned_cols + merged_cols + eliminated_cols`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CompactionStats {
    pub algebraic_rows: usize,
    pub removable_rows: usize,
    pub algebraic_cols: usize,
    pub removable_cols: usize,
    pub duplicate_rows: usize,
    pub tautology_rows: usize,
    pub pinned_cols: usize,
    pub merged_cols: usize,

    /// Columns eliminated by a general affine substitution `w_j = Σ kᵢ·wᵢ + m·w₀`.
    pub eliminated_cols: usize,

    /// Rows killed by a *standalone* linear relation being solved for a merged/subbed column (the
    /// row is consumed by the elimination).
    ///
    /// Pin-routed standalone eliminations are counted under `pinned_cols` instead, so this stays
    /// disjoint.
    pub linear_elim_rows: usize,
    pub rounds: usize,
}

// COEFFICIENT INTERNING
// ================================================================================================

/// Interner for field coefficients. Ids are assigned in first-seen order, which is deterministic
/// because rows are always walked in index order.
#[derive(Default)]
struct CoeffInterner {
    ids: HashMap<Field, u32>,
    values: Vec<Field>,
}

impl CoeffInterner {
    fn intern(&mut self, value: Field) -> u32 {
        *self.ids.entry(value).or_insert_with(|| {
            self.values.push(value);
            (self.values.len() - 1) as u32
        })
    }

    fn value(&self, id: u32) -> Field {
        self.values[id as usize]
    }
}

// ANALYTIC SUBSTITUTION
// ================================================================================================

/// The evolving substitution over the algebraic columns. Three tiers, in increasing generality:
///
/// - `uf` — column merges `w_i = w_j` (union-find, the min-index representative survives);
/// - `pins` — columns fixed to a constant `w_j = c` (a pinned column folds into the constant column
///   0), keyed by the column's union-find root;
/// - `subs` — general affine eliminations `w_j = Σ kᵢ·wᵢ + m·w₀`, keyed by root, with the value an
///   LC over *surviving* columns (and column 0). This is the representation Marcin's generalized
///   linear-elimination cases need; `uf`/`pins` are its degenerate one-term shapes, kept because
///   they are cheaper and (for `uf`) keep deep merge chains near-O(1) via path compression.
///
/// A column is in at most one tier: once eliminated it never reappears in a canonical row, so it is
/// never re-pinned/merged/subbed. The tiers are resolved together in [`Substitution::fold_terms`].
struct Substitution {
    uf: UnionFind<u32>,
    pins: HashMap<u32, Field>,
    subs: HashMap<u32, Vec<(u32, Field)>>,
}

impl Substitution {
    fn new() -> Self {
        Substitution {
            uf: UnionFind::default(),
            pins: HashMap::default(),
            subs: HashMap::default(),
        }
    }

    fn is_pinned(&mut self, col: u32) -> bool {
        let root = self.uf.find(col);
        self.pins.contains_key(&root)
    }

    /// Canonicalize one LC under the current substitution.
    fn apply(&mut self, lc: &LC, interner: &mut CoeffInterner) -> CanonLc {
        self.fold_terms(lc.iter().map(|&(col, coeff)| (col as u32, coeff)), interner)
    }

    /// Re-canonicalize an already-canonical LC under the substitution as it has evolved since
    /// the LC was built (merges may collide columns; new pins fold into column 0). Fast path:
    /// an LC the substitution no longer touches is returned as a plain copy.
    fn apply_canon(&mut self, lc: &CanonLc, interner: &mut CoeffInterner) -> CanonLc {
        let untouched = lc.iter().all(|&(col, _)| {
            self.uf.find(col) == col
                && !self.pins.contains_key(&col)
                && !self.subs.contains_key(&col)
        });
        if untouched {
            return lc.clone();
        }
        let resolved: Vec<(u32, Field)> = lc
            .iter()
            .map(|&(col, coeff_id)| (col, interner.value(coeff_id)))
            .collect();
        self.fold_terms(resolved.into_iter(), interner)
    }

    /// Fold field-valued terms through the current substitution, resolving every column to its
    /// surviving union-find root (pinned columns fold into column 0, affine-subbed columns expand
    /// into their stored LC). Returns field-valued terms in column order with zero coefficients
    /// stripped — over surviving columns only.
    ///
    /// Fold like terms with a sorted map so the output order is column order. A worklist (rather
    /// than a plain loop) so an affine-substituted column expands: its `subs` LC is pushed back
    /// scaled by the incoming coefficient, and those terms are themselves resolved. Because an
    /// eliminated column never reappears as a surviving root (it is always folded away), the subst
    /// is acyclic and this terminates; `subs` LCs are stored over surviving columns so the
    /// expansion is shallow.
    fn fold_field_terms(&mut self, terms: impl Iterator<Item = (u32, Field)>) -> Vec<(u32, Field)> {
        // Resolve every term to a surviving root (pins fold into column 0, subbed columns expand
        // via the worklist), gathering the resolved terms into a flat `Vec`, then sort-and-coalesce
        // by column. This is a value-for-value replacement for the previous per-call `BTreeMap`
        // accumulation — field addition is commutative and associative, so summing each column's
        // gathered coefficients in sorted-adjacency order yields the identical canonical form — but
        // it trades a fresh allocating tree per fold for a single contiguous buffer, which is the
        // dominant per-fold cost at corpus scale.
        let mut gathered: Vec<(u32, Field)> = Vec::new();
        let mut stack: Vec<(u32, Field)> = terms.collect();
        while let Some((col, coeff)) = stack.pop() {
            if coeff == Field::ZERO {
                continue;
            }
            let root = self.uf.find(col);
            if let Some(&value) = self.pins.get(&root) {
                gathered.push((0, coeff * value));
            } else if let Some(sub) = self.subs.get(&root) {
                for &(sub_col, sub_coeff) in sub {
                    stack.push((sub_col, coeff * sub_coeff));
                }
            } else {
                gathered.push((root, coeff));
            }
        }

        gathered.sort_unstable_by_key(|&(col, _)| col);
        let mut folded: Vec<(u32, Field)> = Vec::with_capacity(gathered.len());
        for (col, coeff) in gathered {
            match folded.last_mut() {
                Some(last) if last.0 == col => last.1 += coeff,
                _ => folded.push((col, coeff)),
            }
        }
        folded.retain(|(_, coeff)| *coeff != Field::ZERO);

        folded
    }

    /// Canonicalize field-valued terms and intern their coefficients into a [`CanonLc`].
    fn fold_terms(
        &mut self,
        terms: impl Iterator<Item = (u32, Field)>,
        interner: &mut CoeffInterner,
    ) -> CanonLc {
        self.fold_field_terms(terms)
            .into_iter()
            .map(|(col, coeff)| (col, interner.intern(coeff)))
            .collect()
    }
}

// CANONICAL ROW FORM
// ================================================================================================

/// One algebraic row in canonical form. `a`/`b` are stored with the lexicographically smaller LC
/// first so product commutativity cannot hide a duplicate or a pattern match.
#[derive(Clone, PartialEq, Eq, Hash)]
struct CanonRow {
    a: CanonLc,
    b: CanonLc,
    c: CanonLc,
}

// INTERNAL FUNCTIONALITY
// ================================================================================================

/// The folded difference `c − c_ref` of two canonical LCs, as field-valued terms (zeros stripped,
/// column-sorted).
///
/// Used to derive the linear relation between two rows that share a product.
fn lc_difference(c: &CanonLc, c_ref: &CanonLc, interner: &CoeffInterner) -> Vec<(u32, Field)> {
    let mut folded = std::collections::BTreeMap::<u32, Field>::new();
    for &(col, coeff_id) in c {
        *folded.entry(col).or_insert(Field::ZERO) += interner.value(coeff_id);
    }
    for &(col, coeff_id) in c_ref {
        *folded.entry(col).or_insert(Field::ZERO) -= interner.value(coeff_id);
    }
    folded
        .into_iter()
        .filter(|(_, v)| *v != Field::ZERO)
        .collect()
}

/// Is this LC a lone constant term `k·w_0`? Returns `k`.
fn lone_constant(lc: &CanonLc, interner: &CoeffInterner) -> Option<Field> {
    match lc.as_slice() {
        [(0, coeff_id)] => Some(interner.value(*coeff_id)),
        _ => None,
    }
}

/// A row that holds for every witness assignment.
///
/// Conservative pattern set (matches the measurement prototype):
///
/// - `A ≡ 0 ∧ C ≡ 0` (an empty product side zeroes the product; `b < a` normalization in
///   [`analyze`] puts an empty LC in `a`);
/// - `A ≡ k·w_0 ∧ C = k·B` termwise (e.g. `v·1 − v = 0`), either orientation;
/// - both sides constant with `C` the matching constant product.
fn is_tautology(row: &CanonRow, interner: &CoeffInterner) -> bool {
    if row.a.is_empty() && row.c.is_empty() {
        return true;
    }
    let scaled_match = |k: Field, side: &CanonLc, c: &CanonLc| {
        c.len() == side.len()
            && c.iter()
                .zip(side)
                .all(|(&(c_col, c_coeff), &(s_col, s_coeff))| {
                    c_col == s_col && interner.value(c_coeff) == k * interner.value(s_coeff)
                })
    };
    if let Some(k) = lone_constant(&row.a, interner) {
        if let Some(m) = lone_constant(&row.b, interner) {
            // Constant product: `k·m − C = 0` holds iff C is exactly the constant `k·m`.
            return match row.c.as_slice() {
                [(0, c_id)] => interner.value(*c_id) == k * m,
                _ => false,
            };
        }
        if scaled_match(k, &row.b, &row.c) {
            return true;
        }
    }
    if let Some(k) = lone_constant(&row.b, interner) {
        if scaled_match(k, &row.a, &row.c) {
            return true;
        }
    }
    false
}

/// A degenerate row that pins an unprotected column to a constant: one product side is the empty
/// LC (so the product is 0) and `C` is `k·w_j` or `k·w_j + m·w_0`, giving `w_j = −m/k`.
fn pin_candidate(row: &CanonRow, protected: u32, interner: &CoeffInterner) -> Option<(u32, Field)> {
    // `b < a` normalization puts an empty product side in `a`.
    if !row.a.is_empty() {
        return None;
    }
    let (col, k, m) = match row.c.as_slice() {
        [(col, k_id)] => (*col, interner.value(*k_id), Field::ZERO),
        [(0, m_id), (col, k_id)] => (*col, interner.value(*k_id), interner.value(*m_id)),
        _ => return None,
    };
    if col < protected {
        return None;
    }
    // Canonicalization never emits zero coefficients, so `k` is invertible.
    Some((col, -(m / k)))
}

/// Which representation tier an elimination used (see [`Substitution`]).
enum Elim {
    /// The relation has no unprotected column — nothing eliminated.
    ///
    /// A pure-constant residue is left live (a `0 = 0` tautology is dropped elsewhere; an
    /// unsatisfiable `c = 0`, `c ≠ 0`, must stay so the compacted system stays unsatisfiable).
    None,
    Pin,
    Merge,
    Sub,
}

/// Solve the linear relation `Σ terms·w = 0` for its lowest-index unprotected column and record the
/// elimination in `subst`, returning which tier was used.
///
/// `terms` need not be canonical for the *current* substitution: they are re-folded through it here
/// (via [`Substitution::fold_field_terms`]) before anything else. This matters because a caller can
/// build `terms` from an LC that predates the latest merges/pins/subs — phase 3 differences a
/// freshly-canonicalized row against a *reference* `C` that was canonicalized at an earlier subst
/// version, so its raw delta may still name columns that have since been eliminated. Re-folding
/// maps every column to a surviving root, so an already-satisfied relation collapses to empty (⇒
/// [`Elim::None`], no phantom column removed) and the recorded substitution can never be
/// self-referential.
///
/// The lowest-index unprotected column is a deterministic pivot; because protected columns are the
/// lowest indices it is never one of them. Its coefficient is invertible (folding strips zeros; the
/// field is prime), and the relation is entailed, so `w_pivot := −(1/k)·(rest)` is a
/// value-preserving substitution. It is sound for the algebraic rows, and — because the table /
/// lookup rows are logUp linear forms in these columns rather than single-column commitments —
/// expanding a column into a multi-term LC there keeps every row a valid R1C with unchanged value.
fn eliminate_linear(subst: &mut Substitution, terms: &[(u32, Field)], protected: u32) -> Elim {
    // Normalize the relation against the current substitution first (see the doc above). After this
    // every column is a distinct surviving root, sorted by column.
    let terms = subst.fold_field_terms(terms.iter().copied());
    let Some(pivot_pos) = terms.iter().position(|&(col, _)| col >= protected) else {
        return Elim::None;
    };
    let (pivot, pivot_coeff) = terms[pivot_pos];
    let rhs: Vec<(u32, Field)> = terms
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != pivot_pos)
        .map(|(_, &(col, coeff))| (col, -(coeff / pivot_coeff)))
        .collect();
    let root = subst.uf.find(pivot);
    match rhs.as_slice() {
        // A constant (or zero) right-hand side is a pin.
        [] => {
            subst.pins.insert(root, Field::ZERO);
            Elim::Pin
        }
        [(0, value)] => {
            subst.pins.insert(root, *value);
            Elim::Pin
        }
        // A single other column with unit coefficient is a plain equality — kept in union-find so
        // deep merge chains stay near-O(1). `union` keeps the min-index (hence protected) root.
        // Because the folded `terms` hold distinct surviving roots, `root != other_root` always
        // holds here, so this eliminates exactly one column; the equality guard is a last-line
        // defence that a would-be no-op union never counts as a removed column.
        [(other, coeff)] if *coeff == Field::from(1u64) => {
            let other_root = subst.uf.find(*other);
            if root == other_root {
                Elim::None
            } else {
                subst.uf.union(root, other_root);
                Elim::Merge
            }
        }
        // Anything else (multi-term, or a single scaled/negated column) is a general affine
        // substitution.
        _ => {
            subst.subs.insert(root, rhs);
            Elim::Sub
        }
    }
}

// TESTS
// ================================================================================================

#[cfg(test)]
mod tests {
    use mavros_artifacts::{ConstraintsLayout, R1C, WitnessLayout};

    use super::*;

    fn field(v: i64) -> Field {
        if v < 0 {
            -Field::from((-v) as u64)
        } else {
            Field::from(v as u64)
        }
    }

    fn lc(terms: &[(usize, i64)]) -> LC {
        terms.iter().map(|&(col, v)| (col, field(v))).collect()
    }

    /// An R1CS whose algebraic section is exactly `rows` over `cols` columns, with empty
    /// non-algebraic sections (the analysis never looks at them).
    fn system(cols: usize, rows: Vec<R1C>) -> R1CS {
        R1CS {
            witness_layout: WitnessLayout {
                algebraic_size: cols,
                multiplicities_size: 0,
                challenges_size: 0,
                tables_data_size: 0,
                lookups_data_size: 0,
            },
            constraints_layout: ConstraintsLayout {
                algebraic_size: rows.len(),
                tables_data_size: 0,
                lookups_data_size: 0,
            },
            constraints: rows,
        }
    }

    fn r1c(a: &[(usize, i64)], b: &[(usize, i64)], c: &[(usize, i64)]) -> R1C {
        R1C {
            a: lc(a),
            b: lc(b),
            c: lc(c),
        }
    }

    /// A dense multiplication chain: `w_{i+1} = w_i * w_i`. Nothing is removable.
    #[test]
    fn dense_system_has_no_slack() {
        let rows = (1..5)
            .map(|i| r1c(&[(i, 1)], &[(i, 1)], &[(i + 1, 1)]))
            .collect();
        let stats = analyze(&system(6, rows), 2);
        assert_eq!(stats.removable_rows, 0);
        assert_eq!(stats.removable_cols, 0);
        assert_eq!(stats.rounds, 1);
    }

    #[test]
    fn zero_product_zero_c_is_tautology() {
        // (0) * (w1) - (0) = 0, in both product orientations.
        let rows = vec![r1c(&[], &[(1, 1)], &[]), r1c(&[(1, 1)], &[], &[])];
        let stats = analyze(&system(2, rows), 2);
        assert_eq!(stats.tautology_rows, 2);
        assert_eq!(stats.removable_rows, 2);
        assert_eq!(stats.removable_cols, 0);
    }

    #[test]
    fn constant_times_lc_matching_c_is_tautology() {
        // (2·w0) * (3·w2 + 5·w3) - (6·w2 + 10·w3) = 0 and the commuted form.
        let rows = vec![
            r1c(&[(0, 2)], &[(2, 3), (3, 5)], &[(2, 6), (3, 10)]),
            r1c(&[(2, 3), (3, 5)], &[(0, 2)], &[(2, 6), (3, 10)]),
        ];
        let stats = analyze(&system(4, rows), 2);
        assert_eq!(stats.tautology_rows, 2);
        assert_eq!(stats.removable_rows, 2);
    }

    #[test]
    fn constant_times_constant_is_tautology_only_when_it_holds() {
        let rows = vec![
            // (2) * (3) - (6) = 0: holds identically.
            r1c(&[(0, 2)], &[(0, 3)], &[(0, 6)]),
            // (2) * (3) - (7) = 0: unsatisfiable, must NOT be dropped.
            r1c(&[(0, 2)], &[(0, 3)], &[(0, 7)]),
        ];
        let stats = analyze(&system(1, rows), 1);
        assert_eq!(stats.tautology_rows, 1);
        assert_eq!(stats.removable_rows, 1);
    }

    #[test]
    fn degenerate_row_pins_column_and_cascades() {
        let rows = vec![
            // (0) * (w1) - (2·w2 + 6·w0) = 0  =>  w2 = -3 (pin).
            r1c(&[], &[(1, 1)], &[(2, 2), (0, 6)]),
            // (w1) * (w2) - (w3) = 0: once w2 is pinned to -3 this is w3 = -3·w1, a linear relation
            // in the protected input w1, so w3 is now eliminable too (Claim 2 on the constant B).
            r1c(&[(1, 1)], &[(2, 1)], &[(3, 1)]),
            // (0) * (w1) - (w2 + 3·w0) = 0: same fact again; a tautology once w2 is pinned.
            r1c(&[], &[(1, 1)], &[(2, 1), (0, 3)]),
        ];
        let stats = analyze(&system(4, rows), 2);
        assert_eq!(stats.pinned_cols, 1);
        assert_eq!(stats.eliminated_cols, 1);
        assert_eq!(stats.tautology_rows, 1);
        assert_eq!(stats.removable_rows, 3);
        assert_eq!(stats.removable_cols, 2);
        assert!(stats.rounds >= 2);
    }

    /// Two rows pinning the same column to different values make the system unsatisfiable. The
    /// second pin must not be recorded, and the second row must stay live forever (it
    /// re-canonicalizes to the unsatisfiable residue `0·B − c = 0`, `c ≠ 0`) so the compacted
    /// system stays unsatisfiable too.
    #[test]
    fn contradictory_pins_keep_the_unsat_residue_live() {
        let rows = vec![
            // (0) * (w1) - (w2 + 3·w0) = 0  =>  w2 = -3.
            r1c(&[], &[(1, 1)], &[(2, 1), (0, 3)]),
            // (0) * (w1) - (w2 + 5·w0) = 0  =>  w2 = -5: contradicts the first pin. Under it,
            // this row becomes 0·w1 - 2·w0 = 0, which no pattern may ever remove.
            r1c(&[], &[(1, 1)], &[(2, 1), (0, 5)]),
        ];
        let stats = analyze(&system(3, rows), 2);
        assert_eq!(stats.pinned_cols, 1);
        assert_eq!(stats.tautology_rows, 0);
        assert_eq!(stats.duplicate_rows, 0);
        assert_eq!(stats.merged_cols, 0);
        assert_eq!(stats.removable_rows, 1);
        assert_eq!(stats.removable_cols, 1);
    }

    #[test]
    fn protected_columns_are_never_pinned() {
        // Would pin w1, but w1 is inside the protected block.
        let rows = vec![r1c(&[], &[(2, 1)], &[(1, 1), (0, 3)])];
        let stats = analyze(&system(3, rows), 2);
        assert_eq!(stats.removable_rows, 0);
        assert_eq!(stats.removable_cols, 0);
    }

    #[test]
    fn identical_defining_rows_merge_and_duplicates_drop() {
        // Two copies of the same subcircuit: w3 and w4 are both defined as w1*w2, then used
        // identically. Merging w4 into w3 makes rows 2/3 duplicates of rows 0/1.
        let rows = vec![
            r1c(&[(1, 1)], &[(2, 1)], &[(3, 1)]),
            r1c(&[(3, 1)], &[(3, 1)], &[(5, 1)]),
            r1c(&[(1, 1)], &[(2, 1)], &[(4, 1)]),
            r1c(&[(4, 1)], &[(4, 1)], &[(6, 1)]),
        ];
        let stats = analyze(&system(7, rows), 3);
        // Round 1 merges w4 -> w3 AND w6 -> w5 (phase 3 applies the substitution live, so row
        // 3's residue already sees w4 -> w3); round 2 drops rows 2 and 3 as duplicates.
        assert_eq!(stats.merged_cols, 2);
        assert_eq!(stats.duplicate_rows, 2);
        assert_eq!(stats.removable_rows, 2);
        assert_eq!(stats.removable_cols, 2);
    }

    #[test]
    fn same_ab_unequal_coefficient_diff_elides_via_sub() {
        // Same (A,B); C differs as w3 vs 2·w4. The old rule required equal coefficients and merged
        // nothing here — but 2·w4 − w3 = 0 is the valid equality w3 = 2·w4, so Claim 3 eliminates a
        // column via a general (scaled) substitution and the second row drops as a duplicate.
        let rows = vec![
            r1c(&[(1, 1)], &[(2, 1)], &[(3, 1)]),
            r1c(&[(1, 1)], &[(2, 1)], &[(4, 2)]),
        ];
        let stats = analyze(&system(5, rows), 3);
        assert_eq!(stats.merged_cols, 0);
        assert_eq!(stats.eliminated_cols, 1);
        assert_eq!(stats.duplicate_rows, 1);
        assert_eq!(stats.removable_rows, 1);
        assert_eq!(stats.removable_cols, 1);
    }

    #[test]
    fn same_ab_multi_term_c_difference_elides_one_column() {
        // Identical product (A,B) but C differs by two terms: (w4 + w5) vs w6. The difference
        // (w4 + w5) − w6 = 0 removes one column (Claim 3, multi-term) and the second row duplicates.
        let rows = vec![
            r1c(&[(1, 1)], &[(2, 1)], &[(4, 1), (5, 1)]),
            r1c(&[(1, 1)], &[(2, 1)], &[(6, 1)]),
        ];
        let stats = analyze(&system(7, rows), 3);
        assert_eq!(stats.eliminated_cols, 1);
        assert_eq!(stats.duplicate_rows, 1);
        assert_eq!(stats.removable_rows, 1);
        assert_eq!(stats.removable_cols, 1);
    }

    #[test]
    fn rows_with_different_products_do_not_pair() {
        // Same B but different A (w3 vs w4): the products differ, so the rows are never grouped and
        // nothing is derived (as under the old defining-occurrence rule, which this replaces).
        let rows = vec![
            r1c(&[(3, 1)], &[(2, 1)], &[(3, 1)]),
            r1c(&[(4, 1)], &[(2, 1)], &[(4, 1)]),
        ];
        let stats = analyze(&system(5, rows), 3);
        assert_eq!(stats.merged_cols, 0);
        assert_eq!(stats.eliminated_cols, 0);
        assert_eq!(stats.removable_rows, 0);
    }

    #[test]
    fn exact_duplicate_rows_drop() {
        let rows = vec![
            r1c(&[(1, 1)], &[(2, 1)], &[(3, 1)]),
            r1c(&[(1, 1)], &[(2, 1)], &[(3, 1)]),
            // Commuted product: still the same constraint after normalization.
            r1c(&[(2, 1)], &[(1, 1)], &[(3, 1)]),
        ];
        let stats = analyze(&system(4, rows), 2);
        assert_eq!(stats.duplicate_rows, 2);
        assert_eq!(stats.removable_rows, 2);
    }

    #[test]
    fn folded_and_reordered_terms_canonicalize_alike() {
        // Same constraint written messily: split coefficients, shuffled order, an explicit zero
        // term. Canonicalization must fold them into duplicates.
        let rows = vec![
            r1c(&[(1, 1)], &[(2, 1), (3, 2)], &[(4, 1)]),
            r1c(&[(1, 1)], &[(3, 2), (2, 1), (5, 0)], &[(4, 1)]),
            r1c(&[(1, 1)], &[(3, 1), (2, 1), (3, 1)], &[(4, 1)]),
        ];
        let stats = analyze(&system(6, rows), 2);
        assert_eq!(stats.duplicate_rows, 2);
        assert_eq!(stats.removable_rows, 2);
    }

    #[test]
    fn table_rows_are_ignored() {
        // Same duplicated rows as `exact_duplicate_rows_drop`, but declared as table rows:
        // nothing is algebraic, nothing may be counted.
        let mut r1cs = system(
            4,
            vec![
                r1c(&[(1, 1)], &[(2, 1)], &[(3, 1)]),
                r1c(&[(1, 1)], &[(2, 1)], &[(3, 1)]),
            ],
        );
        r1cs.constraints_layout.algebraic_size = 0;
        r1cs.constraints_layout.tables_data_size = 2;
        let stats = analyze(&r1cs, 2);
        assert_eq!(stats.algebraic_rows, 0);
        assert_eq!(stats.removable_rows, 0);
        assert_eq!(stats.removable_cols, 0);
    }

    /// A chain w_{i+1} = w_i + 1 rooted at a protected input, duplicated on two disjoint column
    /// runs. Every row is linear (its multiplicand is the constant w0), so phase 2.5 eliminates all
    /// of them — and because it applies the substitution LIVE, the whole chain closes in a single
    /// in-order pass (one round per dependency level would make deep circuits quadratic).
    #[test]
    fn terminates_on_adversarial_linear_chain() {
        let mut rows = Vec::new();
        let n = 6;
        for i in 0..n {
            // (w0) * (chain_i + 1) - (chain_{i+1}) = 0 twice, on disjoint column chains starting
            // from the same protected input w1.
            let left_prev = if i == 0 { 1 } else { 2 + i - 1 };
            let right_prev = if i == 0 { 1 } else { 2 + n + i - 1 };
            rows.push(r1c(&[(0, 1)], &[(left_prev, 1), (0, 1)], &[(2 + i, 1)]));
            rows.push(r1c(
                &[(0, 1)],
                &[(right_prev, 1), (0, 1)],
                &[(2 + n + i, 1)],
            ));
        }
        let stats = analyze(&system(2 + 2 * n, rows), 2);
        assert_eq!(stats.eliminated_cols, 2 * n);
        assert_eq!(stats.linear_elim_rows, 2 * n);
        assert_eq!(stats.removable_rows, 2 * n);
        assert_eq!(stats.removable_cols, 2 * n);
        assert!(stats.rounds <= 3);
    }

    /// Row order must not affect the resulting counts.
    #[test]
    fn stats_are_order_independent() {
        let rows = vec![
            r1c(&[(1, 1)], &[(2, 1)], &[(3, 1)]),
            r1c(&[(3, 1)], &[(3, 1)], &[(5, 1)]),
            r1c(&[(1, 1)], &[(2, 1)], &[(4, 1)]),
            r1c(&[(4, 1)], &[(4, 1)], &[(6, 1)]),
            r1c(&[], &[(1, 1)], &[]),
            r1c(&[], &[(2, 1)], &[(7, 2), (0, 6)]),
        ];
        let baseline = analyze(&system(8, rows.clone()), 3);
        let mut reversed = rows;
        reversed.reverse();
        let permuted = analyze(&system(8, reversed), 3);
        assert_eq!(baseline.removable_rows, permuted.removable_rows);
        assert_eq!(baseline.removable_cols, permuted.removable_cols);
        assert_eq!(baseline.merged_cols, permuted.merged_cols);
        assert_eq!(baseline.pinned_cols, permuted.pinned_cols);
    }

    #[test]
    fn multi_term_linear_row_elides_one_column() {
        // (0)*(w1) - (w3 + w4 + w5) = 0 is the relation w3 + w4 + w5 = 0. Solving it for the
        // lowest-index unprotected column (w3) removes one column and consumes the row. (Claim 1.)
        let rows = vec![r1c(&[], &[(1, 1)], &[(3, 1), (4, 1), (5, 1)])];
        let stats = analyze(&system(6, rows), 3);
        assert_eq!(stats.eliminated_cols, 1);
        assert_eq!(stats.linear_elim_rows, 1);
        assert_eq!(stats.removable_rows, 1);
        assert_eq!(stats.removable_cols, 1);
    }

    #[test]
    fn dependent_linear_rows_elide_one_column_not_two() {
        // w3 + w4 = 0 and 2·w3 + 2·w4 = 0 are dependent. The live substitution must reduce the
        // second to 0 = 0, so only ONE column is removed (the anti-double-count guarantee).
        let rows = vec![
            r1c(&[], &[(1, 1)], &[(3, 1), (4, 1)]),
            r1c(&[], &[(1, 1)], &[(3, 2), (4, 2)]),
        ];
        let stats = analyze(&system(5, rows), 3);
        assert_eq!(stats.removable_cols, 1);
        assert_eq!(stats.removable_rows, 2);
    }

    #[test]
    fn standalone_equality_routes_to_merge() {
        // (0)*(w1) - (w3 - w4) = 0 is w3 = w4: a plain equality merges (union-find), not a sub.
        let rows = vec![r1c(&[], &[(1, 1)], &[(3, 1), (4, -1)])];
        let stats = analyze(&system(5, rows), 3);
        assert_eq!(stats.merged_cols, 1);
        assert_eq!(stats.eliminated_cols, 0);
        assert_eq!(stats.linear_elim_rows, 1);
        assert_eq!(stats.removable_rows, 1);
        assert_eq!(stats.removable_cols, 1);
    }

    #[test]
    fn constant_a_non_cancelling_row_elides_a_column() {
        // (2·w0) * (w3 + w4) - (w5) = 0 is the relation 2·w3 + 2·w4 − w5 = 0 (Claim 2: the constant
        // multiplicand makes it linear). It is NOT a tautology (C ≠ 2·B), so it removes a column.
        let rows = vec![r1c(&[(0, 2)], &[(3, 1), (4, 1)], &[(5, 1)])];
        let stats = analyze(&system(6, rows), 3);
        assert_eq!(stats.eliminated_cols, 1);
        assert_eq!(stats.linear_elim_rows, 1);
        assert_eq!(stats.removable_rows, 1);
        assert_eq!(stats.removable_cols, 1);
    }

    #[test]
    fn unsatisfiable_linear_pair_is_preserved() {
        // w3 + w4 = 0 and w3 + w4 = 1. The first eliminates a column; under that substitution the
        // second becomes 0 = 1 and MUST stay live so the compacted system stays unsatisfiable.
        let rows = vec![
            r1c(&[], &[(1, 1)], &[(3, 1), (4, 1)]),
            r1c(&[], &[(1, 1)], &[(3, 1), (4, 1), (0, -1)]),
        ];
        let stats = analyze(&system(5, rows), 3);
        assert_eq!(stats.removable_cols, 1);
        assert_eq!(stats.removable_rows, 1);
    }

    #[test]
    fn generalized_stats_are_order_independent() {
        // Exercises all three generalized paths (multi-term standalone, constant-A, same-(A,B)
        // multi-term difference), including a shared column (w4) across two relations, under a row
        // permutation. The eliminable rank is order-invariant even though the pivots chosen differ.
        let rows = vec![
            r1c(&[], &[(1, 1)], &[(3, 1), (4, 1), (5, 1)]), // w3 + w4 + w5 = 0        (Claim 1)
            r1c(&[(0, 2)], &[(4, 1)], &[(6, 1)]),           // 2·w4 − w6 = 0           (Claim 2)
            r1c(&[(1, 1)], &[(2, 1)], &[(7, 1), (8, 1)]),   // w1·w2 = w7 + w8   }
            r1c(&[(1, 1)], &[(2, 1)], &[(9, 1)]),           // w1·w2 = w9        } (Claim 3)
        ];
        let baseline = analyze(&system(10, rows.clone()), 3);
        let mut reversed = rows;
        reversed.reverse();
        let permuted = analyze(&system(10, reversed), 3);
        assert_eq!(baseline.removable_rows, permuted.removable_rows);
        assert_eq!(baseline.removable_cols, permuted.removable_cols);
        assert_eq!(baseline.eliminated_cols, permuted.eliminated_cols);
    }

    #[test]
    fn all_protected_linear_row_is_kept() {
        // w1 + w2 = 0 over only protected columns: a real constraint on the public statement, not
        // an elimination.
        let rows = vec![r1c(&[], &[(2, 1)], &[(1, 1), (2, 1)])];
        let stats = analyze(&system(3, rows), 3);
        assert_eq!(stats.removable_rows, 0);
        assert_eq!(stats.removable_cols, 0);
    }

    /// Phase 3 stores each group's reference `C` as a snapshot, then applies eliminations live. A
    /// later merge in the same pass can stale that snapshot, so the raw `C_R − C_ref` difference
    /// names an already-merged column. `eliminate_linear` must re-fold the difference: here it
    /// collapses to `0 = 0` (the relation is already satisfied), so NO extra column is counted.
    ///
    /// Regression: before the fix the stale difference `w3 − w4` (with `w4` already merged into
    /// `w3`) drove a no-op union that still incremented `merged_cols`, reporting
    /// `removable_cols == 2` where only one column (`w4`) is truly removable.
    #[test]
    fn stale_reference_merge_does_not_overcount_columns() {
        let rows = vec![
            r1c(&[(5, 1)], &[(6, 1)], &[(4, 1)]), // group A: w5·w6 = w4   (ref C snapshot = [w4])
            r1c(&[(1, 1)], &[(2, 1)], &[(3, 1)]), // group B: w1·w2 = w3   (ref C snapshot = [w3])
            r1c(&[(1, 1)], &[(2, 1)], &[(4, 1)]), // group B hit: w4 − w3 ⇒ merge w4 → w3 (real)
            r1c(&[(5, 1)], &[(6, 1)], &[(3, 1)]), // group A hit: [w3] − STALE [w4] ⇒ 0 = 0, no col
        ];
        let stats = analyze(&system(7, rows.clone()), 2);
        assert_eq!(stats.merged_cols, 1, "only w4 → w3 is a real merge");
        assert_eq!(stats.eliminated_cols, 0);
        assert_eq!(stats.removable_cols, 1);
        // Rows 2 and 3 both become duplicates once w4 folds to w3.
        assert_eq!(stats.duplicate_rows, 2);
        assert_eq!(stats.removable_rows, 2);

        // The over-count must not depend on row order either.
        let mut reversed = rows;
        reversed.reverse();
        let permuted = analyze(&system(7, reversed), 2);
        assert_eq!(permuted.removable_cols, 1);
        assert_eq!(permuted.removable_rows, 2);
    }

    /// The multi-term sibling of the case above: a stale reference difference that routes to the
    /// affine `subs` tier. Before the fix, `eliminate_linear` recorded `subs[w3] = w6 − w5` while
    /// `w5` was already merged into `w3`, i.e. a self-referential substitution, and the next
    /// round's `fold_terms` looped forever. Re-folding the difference first resolves `w5 → w3`,
    /// yielding the well-formed `subs[w3] = w6/2`; the analysis must terminate and count exactly
    /// the two columns (`w5` merged, `w3` subbed) that are genuinely removable.
    #[test]
    fn stale_reference_multi_term_terminates_without_self_reference() {
        let rows = vec![
            r1c(&[(7, 1)], &[(8, 1)], &[(3, 1), (5, 1)]), /* group X: w7·w8 = w3 + w5  (ref snapshot) */
            r1c(&[(1, 1)], &[(2, 1)], &[(3, 1)]),         // group Y: w1·w2 = w3
            r1c(&[(1, 1)], &[(2, 1)], &[(5, 1)]),         // group Y hit: w5 − w3 ⇒ merge w5 → w3
            r1c(&[(7, 1)], &[(8, 1)], &[(6, 1)]),         // group X hit: [w6] − STALE [w3+w5]
        ];
        let stats = analyze(&system(9, rows), 2);
        assert_eq!(stats.merged_cols, 1, "w5 → w3");
        assert_eq!(stats.eliminated_cols, 1, "w3 → w6/2");
        assert_eq!(stats.removable_cols, 2);
        assert_eq!(stats.duplicate_rows, 2);
        assert_eq!(stats.removable_rows, 2);
    }
}
