//! Dynamic lookup-table sizing for rangechecks and spreads.
//!
//! Every rangecheck and spread is ultimately served by one or more lookup tables. A table of
//! size `2^s` checks a key against `[0, 2^s)`; a `w`-bit value too wide for a single table is
//! split into chunks.
//!
//! This analysis reads the whole-program lookup-width histograms (from [`Summary`]) and picks an
//! optimal *set* of table sizes — separately for rangechecks (`R`, width-1 tables) and spreads
//! (`P`, width-2 tables) — minimizing total R1CS constraints. Because a spread table's key column
//! is itself a rangecheck, rangechecks may also be served by a spread table (at 2 constraints per
//! lookup instead of 1), so the two decisions are optimized jointly.
//!
//! ## Cost model (R1CS constraints)
//! * Rangecheck table at size `s`: `2^s + 1` allocation, `1` constraint per lookup.
//! * Spread table at size `s`: `2^s + 1` allocation, `2` constraints per lookup.
//!
//! ## Simulation tricks
//! * A full chunk of exactly `s` bits is one lookup in an `s`-bit table.
//! * A residual chunk of `r < s` bits is bounded with the "2-larger" trick: look up `x` and
//!   `(2^r − 1) − x` in the `s`-bit table (two lookups). This keeps the range *exactly* `r` bits,
//!   which is required for spreads since they double as rangechecks.

use std::fmt::Debug;

use crate::{
    collections::HashMap,
    compiler::{
        analysis::instrumenter::Summary,
        pass_manager::{Analysis, AnalysisId, AnalysisStore},
        ssa::hlssa::HLSSA,
    },
};

/// Maximum number of distinct table sizes the optimizer will allocate per category. The optimal
/// set is small because allocation cost grows exponentially with size; if a chosen set ever uses
/// the full budget we log a warning so this can be bumped.
const MAX_TABLES_PER_CATEGORY: usize = 3;

/// Hard ceiling on a single table's bit-width. The affordability bound (a table can't save more
/// lookups than its `2^s` allocation costs) keeps realistic sizes far below this; the ceiling only
/// guarantees the backends' fixed-size table caches (`vm::VM::{rgchk_tables,spread_tables}`) are
/// never indexed out of range.
const MAX_TABLE_BITS: u8 = 24;

/// Which physical table a chunk lookup targets.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum TableKind {
    /// A width-1 rangecheck table (`Rangecheck(s)`), 1 constraint per lookup.
    Range,
    /// A width-2 spread table (`Spread(s)`), 2 constraints per lookup.
    Spread,
}

impl TableKind {
    fn per_lookup_constraints(self) -> usize {
        match self {
            TableKind::Range => 1,
            TableKind::Spread => 2,
        }
    }
}

/// One chunk of a decomposed rangecheck/spread lookup, lowest-order chunk first.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Chunk {
    /// Number of value bits this chunk covers.
    pub width: u8,
    /// Size (in bits) of the table this chunk is looked up in; the table has `2^table_size` rows.
    pub table_size: u8,
    /// Which table the chunk is looked up in.
    pub table: TableKind,
    /// When true, `width < table_size` and the chunk is bounded with the "2-larger" trick (two
    /// lookups instead of one).
    pub partial: bool,
}

impl Chunk {
    fn lookups(self) -> usize {
        if self.partial { 2 } else { 1 }
    }

    fn constraints(self) -> usize {
        self.lookups() * self.table.per_lookup_constraints()
    }
}

/// A provider is one available table the decomposition DP may draw chunks from.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct Provider {
    size: u8,
    kind: TableKind,
}

/// The chosen table sizes for the whole program. Produced by [`optimize`] and consumed by the
/// lookup-spilling pass (to emit chunked lookups) and by R1CS generation (which materializes one
/// table per distinct size it sees).
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct LookupSizing {
    /// Rangecheck (width-1) table sizes, sorted ascending.
    pub rangecheck_tables: Vec<u8>,
    /// Spread (width-2) table sizes, sorted ascending.
    pub spread_tables: Vec<u8>,
}

impl LookupSizing {
    /// Providers available for a rangecheck: every range table (cost 1) plus every spread table
    /// whose size isn't already a range table (cost 2, via the shared key column). Sorted for
    /// deterministic decompositions.
    fn rangecheck_providers(&self) -> Vec<Provider> {
        let mut providers: Vec<Provider> = self
            .rangecheck_tables
            .iter()
            .map(|&size| Provider {
                size,
                kind: TableKind::Range,
            })
            .collect();
        for &size in &self.spread_tables {
            if !self.rangecheck_tables.contains(&size) {
                providers.push(Provider {
                    size,
                    kind: TableKind::Spread,
                });
            }
        }
        sort_providers(&mut providers);
        providers
    }

    /// Providers available for a spread: the spread tables only.
    fn spread_providers(&self) -> Vec<Provider> {
        let mut providers: Vec<Provider> = self
            .spread_tables
            .iter()
            .map(|&size| Provider {
                size,
                kind: TableKind::Spread,
            })
            .collect();
        sort_providers(&mut providers);
        providers
    }

    /// How to decompose a `width`-bit rangecheck into chunk lookups (lowest chunk first).
    /// `width == 1` is handled algebraically by the caller and returns an empty plan.
    pub fn decompose_rangecheck(&self, width: u8) -> Vec<Chunk> {
        if width <= 1 {
            return vec![];
        }
        decompose(width, &self.rangecheck_providers())
            .unwrap_or_else(|| panic!("no rangecheck table can serve a {width}-bit rangecheck"))
            .1
    }

    /// How to decompose a `width`-bit spread into chunk lookups (lowest chunk first).
    pub fn decompose_spread(&self, width: u8) -> Vec<Chunk> {
        decompose(width, &self.spread_providers())
            .unwrap_or_else(|| panic!("no spread table can serve a {width}-bit spread"))
            .1
    }
}

/// Deterministic provider ordering: largest size first, with `Range` preferred over `Spread` on
/// ties (cheaper per lookup). The DP relies on this for reproducible decompositions.
fn sort_providers(providers: &mut [Provider]) {
    providers.sort_by(|a, b| b.size.cmp(&a.size).then(a.kind.cmp(&b.kind)));
}

/// Minimum-constraint decomposition of a `width`-bit value over the available `providers`,
/// returned as `(total_constraints, chunks)` with the lowest-order chunk first. Returns `None`
/// only when `providers` is empty and `width > 0` (nothing can serve the lookup).
///
/// The recurrence covers `x` bits either by peeling a full chunk of size `s ≤ x` from some
/// provider (recursing on `x − s`) or by covering all of `x` as a single partial chunk in a
/// provider strictly larger than `x` (the 2-larger trick). A partial is always terminal, so a
/// plan is "some full chunks then at most one partial top chunk", matching what spilling emits.
/// Minimum decomposition cost for `width` over `providers`, without building the plan. This is the
/// cost-only twin of [`decompose`] (same recurrence, returning only `.0`): a pure integer DP with
/// no per-state `Vec<Chunk>` allocation/cloning. The optimizer evaluates this across millions of
/// `(R, P)` candidate pairs, so the plan-building variant (used only by the actual spilling, a
/// handful of times) is far too expensive here.
fn decompose_cost(width: u8, providers: &[Provider]) -> Option<usize> {
    if width == 0 {
        return Some(0);
    }
    if providers.is_empty() {
        return None;
    }
    let w = width as usize;
    let mut best: Vec<Option<usize>> = vec![None; w + 1];
    best[0] = Some(0);
    for x in 1..=w {
        let mut chosen: Option<usize> = None;
        let mut consider = |cost: usize| {
            chosen = Some(chosen.map_or(cost, |b: usize| b.min(cost)));
        };
        // Peel a full chunk of size s <= x (one lookup) off the top; recurse on the low x - s bits.
        for p in providers {
            let s = p.size as usize;
            if s > x {
                continue;
            }
            if let Some(sub) = best[x - s] {
                consider(sub + p.kind.per_lookup_constraints());
            }
        }
        // Cover all x bits as one partial chunk (two lookups) in a strictly-larger table.
        for p in providers {
            if (p.size as usize) > x {
                consider(2 * p.kind.per_lookup_constraints());
            }
        }
        best[x] = chosen;
    }
    best[w]
}

fn decompose(width: u8, providers: &[Provider]) -> Option<(usize, Vec<Chunk>)> {
    if width == 0 {
        return Some((0, vec![]));
    }
    if providers.is_empty() {
        return None;
    }
    let w = width as usize;
    let mut best: Vec<Option<(usize, Vec<Chunk>)>> = vec![None; w + 1];
    best[0] = Some((0, vec![]));

    for x in 1..=w {
        let mut chosen: Option<(usize, Vec<Chunk>)> = None;
        let mut consider = |cost: usize, make_plan: &dyn Fn() -> Vec<Chunk>| {
            if chosen.as_ref().map_or(true, |(c, _)| cost < *c) {
                chosen = Some((cost, make_plan()));
            }
        };

        // Peel a full chunk of size s <= x off the top; recurse on the low x - s bits.
        for p in providers {
            let s = p.size as usize;
            if s > x {
                continue;
            }
            if let Some((sub_cost, sub_plan)) = best[x - s].clone() {
                let chunk = Chunk {
                    width: p.size,
                    table_size: p.size,
                    table: p.kind,
                    partial: false,
                };
                consider(sub_cost + chunk.constraints(), &|| {
                    let mut plan = sub_plan.clone();
                    plan.push(chunk);
                    plan
                });
            }
        }

        // Cover all x bits as one partial chunk in a strictly-larger table (2-larger trick).
        for p in providers {
            if (p.size as usize) > x {
                let chunk = Chunk {
                    width: x as u8,
                    table_size: p.size,
                    table: p.kind,
                    partial: true,
                };
                consider(chunk.constraints(), &|| vec![chunk]);
            }
        }

        best[x] = chosen;
    }

    best[w]
        .clone()
        .map(|(cost, plan)| (cost, canonicalize_plan(plan)))
}

/// Put the plan in a deterministic, spilling-friendly order: full chunks first (widest first),
/// then the single partial chunk (if any). There is at most one partial because the DP only ever
/// uses a partial as a terminal cover.
fn canonicalize_plan(plan: Vec<Chunk>) -> Vec<Chunk> {
    let mut fulls: Vec<Chunk> = plan.iter().copied().filter(|c| !c.partial).collect();
    let partials: Vec<Chunk> = plan.iter().copied().filter(|c| c.partial).collect();
    debug_assert!(partials.len() <= 1, "a plan has at most one partial chunk");
    fulls.sort_by(|a, b| {
        b.width
            .cmp(&a.width)
            .then(b.table_size.cmp(&a.table_size))
            .then(a.table.cmp(&b.table))
    });
    fulls.extend(partials);
    fulls
}

/// Allocation cost (R1CS constraints) for a table of the given kind and size.
fn allocation_constraints(size: u8, kind: TableKind) -> usize {
    let rows = 1usize << size;
    match kind {
        TableKind::Range => rows + 1,
        TableKind::Spread => rows + 1,
    }
}

/// Sorted `(width, count)` histogram, deterministic for reproducible optimization.
fn sorted_histogram(hist: &HashMap<u8, usize>) -> Vec<(u8, usize)> {
    let mut v: Vec<(u8, usize)> = hist.iter().map(|(&b, &c)| (b, c)).collect();
    v.sort_unstable();
    v
}

/// Total per-lookup rangecheck cost over the histogram for the given providers (excludes table
/// allocation; 1-bit checks are algebraic and excluded). `None` if some width can't be served.
fn rangecheck_recurring_cost(rc_hist: &[(u8, usize)], providers: &[Provider]) -> Option<usize> {
    let mut total = 0;
    for &(width, count) in rc_hist {
        if width <= 1 || count == 0 {
            continue;
        }
        let cost = decompose_cost(width, providers)?;
        total += cost * count;
    }
    Some(total)
}

/// Total per-lookup spread cost over the histogram (excludes table allocation). A multi-chunk
/// spread reconstructs both its key and its spread from the lowest chunk by subtracting the higher
/// chunks (free linear combinations), so it needs no recombination constraint. `None` if some
/// width can't be served.
fn spread_recurring_cost(sp_hist: &[(u8, usize)], providers: &[Provider]) -> Option<usize> {
    let mut total = 0;
    for &(width, count) in sp_hist {
        if count == 0 {
            continue;
        }
        let cost = decompose_cost(width, providers)?;
        total += cost * count;
    }
    Some(total)
}

/// Number of bits needed to represent `n` (i.e. `floor(log2(n)) + 1`), `0` for `n == 0`.
fn bit_width(n: usize) -> u8 {
    (usize::BITS - n.leading_zeros()) as u8
}

/// Lookups under the legacy byte-table baseline; bounds how large a table can grow before its
/// `2^s` allocation can no longer pay for itself (it can save at most this many lookups).
fn baseline_total_lookups(rc_hist: &[(u8, usize)], sp_hist: &[(u8, usize)]) -> usize {
    let mut total = 0;
    for &(w, c) in rc_hist {
        let per = match w {
            0 | 1 => 0,
            8 => 1,
            _ => (w as usize) / 8 + if (w as usize) % 8 > 0 { 2 } else { 0 },
        };
        total += per * c;
    }
    for &(b, c) in sp_hist {
        let b = b as usize;
        let per = if b >= 16 {
            // split into chunks of at most 8 bits, like the legacy spilling
            b.div_ceil(8)
        } else {
            1
        };
        total += per * c;
    }
    total
}

/// Candidate table sizes the optimizer may pick from: `2 ..= s_max`, where `s_max` is the smaller
/// of the widest request (no larger table is ever useful) and the affordability bound derived
/// from the total lookup volume (no larger table can pay for its allocation).
fn candidate_sizes(rc_hist: &[(u8, usize)], sp_hist: &[(u8, usize)]) -> Vec<u8> {
    let max_width = rc_hist
        .iter()
        .chain(sp_hist.iter())
        .filter(|(_, c)| *c > 0)
        .map(|(w, _)| *w)
        .max()
        .unwrap_or(0);
    if max_width < 2 {
        return vec![];
    }
    let afford = bit_width(baseline_total_lookups(rc_hist, sp_hist)).saturating_add(1);
    let s_max = max_width.min(afford).min(MAX_TABLE_BITS).max(2);
    (2..=s_max).collect()
}

/// All subset bitmasks over candidate indices `0..n` with popcount `0 ..= max_size` (includes the
/// empty set). Masks make the optimizer's hot loop allocation-free: subset membership, the
/// spread-only difference, and memo keys are all cheap integer ops.
fn bounded_subset_masks(n: usize, max_size: usize) -> Vec<u32> {
    assert!(n <= 32, "candidate set too large for u32 subset masks");
    let mut out = vec![0u32];
    fn rec(n: usize, start: usize, max_size: usize, cur: u32, depth: usize, out: &mut Vec<u32>) {
        if depth == max_size {
            return;
        }
        for i in start..n {
            let next = cur | (1u32 << i);
            out.push(next);
            rec(n, i + 1, max_size, next, depth + 1, out);
        }
    }
    rec(n, 0, max_size, 0, 0, &mut out);
    out
}

/// The candidate sizes selected by `mask`.
fn mask_sizes(mask: u32, candidates: &[u8]) -> Vec<u8> {
    (0..candidates.len())
        .filter(|i| mask & (1u32 << i) != 0)
        .map(|i| candidates[i])
        .collect()
}

/// Build the canonical rangecheck provider set: range tables (cost 1) for `range_mask`, plus
/// spread-only tables (cost 2) for `spread_only_mask` (already disjoint from `range_mask`).
fn providers_from_masks(range_mask: u32, spread_only_mask: u32, candidates: &[u8]) -> Vec<Provider> {
    let mut providers: Vec<Provider> = (0..candidates.len())
        .filter(|i| range_mask & (1u32 << i) != 0)
        .map(|i| Provider {
            size: candidates[i],
            kind: TableKind::Range,
        })
        .collect();
    for i in 0..candidates.len() {
        if spread_only_mask & (1u32 << i) != 0 {
            providers.push(Provider {
                size: candidates[i],
                kind: TableKind::Spread,
            });
        }
    }
    sort_providers(&mut providers);
    providers
}

fn spread_providers_for(p: &[u8]) -> Vec<Provider> {
    let mut providers: Vec<Provider> = p
        .iter()
        .map(|&size| Provider {
            size,
            kind: TableKind::Spread,
        })
        .collect();
    sort_providers(&mut providers);
    providers
}

/// Pick the table-size sets `(R, P)` minimizing total R1CS constraints for the given whole-program
/// lookup-width histograms.
///
/// Spread tables (`P`) serve both spreads and (optionally) rangechecks, so we enumerate `P` and,
/// for each, the rangecheck-only tables `R`, scoring by the full joint cost. The candidate space
/// is small (a handful of sizes, ≤[`MAX_TABLES_PER_CATEGORY`] tables each), so this exhaustive
/// search is cheap and globally optimal within the cardinality budget.
pub fn optimize(
    rangecheck_hist: &HashMap<u8, usize>,
    spread_hist: &HashMap<u8, usize>,
) -> LookupSizing {
    let rc_hist = sorted_histogram(rangecheck_hist);
    let sp_hist = sorted_histogram(spread_hist);

    let has_rangechecks = rc_hist.iter().any(|&(w, c)| w >= 2 && c > 0);
    let has_spreads = sp_hist.iter().any(|&(_, c)| c > 0);
    if !has_rangechecks && !has_spreads {
        return LookupSizing::default();
    }

    let candidates = candidate_sizes(&rc_hist, &sp_hist);
    let n = candidates.len();

    // Enumerate table-size subsets as bitmasks over candidate indices. Costs are memoized: spreads
    // by the spread mask, rangechecks by the provider profile `(range sizes, spread-only sizes)`
    // packed into a `u64` — so each (R, P) pair only pays cheap integer ops, no allocation.
    let masks = bounded_subset_masks(n, MAX_TABLES_PER_CATEGORY);
    let alloc_for = |mask: u32, kind: TableKind| -> usize {
        (0..n)
            .filter(|i| mask & (1u32 << i) != 0)
            .map(|i| allocation_constraints(candidates[i], kind))
            .sum()
    };

    let mut spread_memo: HashMap<u32, Option<usize>> = HashMap::default();
    let mut rangecheck_memo: HashMap<u64, Option<usize>> = HashMap::default();

    let mut best: Option<(usize, u32, u32)> = None;

    for &p_mask in &masks {
        if has_spreads && p_mask == 0 {
            continue; // spreads need at least one table
        }

        let spread_alloc = alloc_for(p_mask, TableKind::Spread);
        let spread_recurring = *spread_memo.entry(p_mask).or_insert_with(|| {
            spread_recurring_cost(&sp_hist, &spread_providers_for(&mask_sizes(p_mask, &candidates)))
        });
        let Some(spread_recurring) = spread_recurring else {
            continue;
        };

        for &r_mask in &masks {
            if has_rangechecks && r_mask == 0 && p_mask == 0 {
                continue; // rangechecks need at least one provider
            }
            // The rangecheck cost depends only on the provider profile: range sizes (`r_mask`) plus
            // spread-only sizes (`p_mask & !r_mask`). Many (R, P) pairs share a profile, so key the
            // memo on it directly.
            let spread_only = p_mask & !r_mask;
            let key = ((r_mask as u64) << 32) | spread_only as u64;
            let rc_recurring = *rangecheck_memo.entry(key).or_insert_with(|| {
                rangecheck_recurring_cost(
                    &rc_hist,
                    &providers_from_masks(r_mask, spread_only, &candidates),
                )
            });
            let Some(rc_recurring) = rc_recurring else {
                continue;
            };
            let range_alloc = alloc_for(r_mask, TableKind::Range);

            let total = spread_alloc + spread_recurring + range_alloc + rc_recurring;
            if best.as_ref().map_or(true, |(c, _, _)| total < *c) {
                best = Some((total, r_mask, p_mask));
            }
        }
    }

    let (_, r_mask, p_mask) =
        best.expect("at least one feasible configuration exists when lookups are present");
    let mut rangecheck_tables = mask_sizes(r_mask, &candidates);
    let mut spread_tables = mask_sizes(p_mask, &candidates);
    rangecheck_tables.sort_unstable();
    spread_tables.sort_unstable();

    if rangecheck_tables.len() == MAX_TABLES_PER_CATEGORY
        || spread_tables.len() == MAX_TABLES_PER_CATEGORY
    {
        tracing::warn!(
            "lookup sizing used the full table budget ({MAX_TABLES_PER_CATEGORY}); \
             consider raising MAX_TABLES_PER_CATEGORY (R={rangecheck_tables:?}, P={spread_tables:?})"
        );
    }

    LookupSizing {
        rangecheck_tables,
        spread_tables,
    }
}

impl Analysis for LookupSizing {
    fn dependencies() -> Vec<AnalysisId> {
        vec![Summary::id()]
    }

    fn compute(_ssa: &HLSSA, store: &AnalysisStore) -> Self {
        let summary = store.get::<Summary>();
        let sizing = optimize(
            &summary.global_rangecheck_lookups,
            &summary.global_spread_lookups,
        );
        tracing::info!(
            "lookup sizing: rangecheck tables {:?}, spread tables {:?}",
            sizing.rangecheck_tables,
            sizing.spread_tables
        );
        sizing
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hist(entries: &[(u8, usize)]) -> HashMap<u8, usize> {
        entries.iter().copied().collect()
    }

    /// The constraint total the optimizer believes a configuration costs (mirrors `optimize`).
    fn cost_of(sizing: &LookupSizing, rc: &[(u8, usize)], sp: &[(u8, usize)]) -> usize {
        let alloc: usize = sizing
            .rangecheck_tables
            .iter()
            .map(|&s| allocation_constraints(s, TableKind::Range))
            .sum::<usize>()
            + sizing
                .spread_tables
                .iter()
                .map(|&s| allocation_constraints(s, TableKind::Spread))
                .sum::<usize>();
        let rc_cost = rangecheck_recurring_cost(rc, &sizing.rangecheck_providers()).unwrap();
        let sp_cost = spread_recurring_cost(sp, &sizing.spread_providers()).unwrap();
        alloc + rc_cost + sp_cost
    }

    #[test]
    fn no_lookups_means_no_tables() {
        let sizing = optimize(&hist(&[]), &hist(&[]));
        assert!(sizing.rangecheck_tables.is_empty());
        assert!(sizing.spread_tables.is_empty());
    }

    #[test]
    fn one_bit_rangechecks_need_no_table() {
        let sizing = optimize(&hist(&[(1, 1000)]), &hist(&[]));
        assert!(sizing.rangecheck_tables.is_empty());
        assert!(sizing.spread_tables.is_empty());
    }

    #[test]
    fn full_chunk_decomposition_is_single_lookup() {
        let sizing = LookupSizing {
            rangecheck_tables: vec![8],
            spread_tables: vec![],
        };
        let plan = sizing.decompose_rangecheck(8);
        assert_eq!(plan.len(), 1);
        assert!(!plan[0].partial);
        assert_eq!(plan[0].width, 8);
        assert_eq!(plan[0].table, TableKind::Range);
    }

    #[test]
    fn partial_chunk_uses_two_larger_trick() {
        let sizing = LookupSizing {
            rangecheck_tables: vec![8],
            spread_tables: vec![],
        };
        // 4 bits in an 8-bit table: one partial chunk (2 lookups).
        let plan = sizing.decompose_rangecheck(4);
        assert_eq!(plan.len(), 1);
        assert!(plan[0].partial);
        assert_eq!(plan[0].width, 4);
        assert_eq!(plan[0].table_size, 8);
    }

    #[test]
    fn wide_rangecheck_splits_into_full_chunks() {
        let sizing = LookupSizing {
            rangecheck_tables: vec![16],
            spread_tables: vec![],
        };
        // 32 bits = two full 16-bit chunks, no partial.
        let plan = sizing.decompose_rangecheck(32);
        assert_eq!(plan.len(), 2);
        assert!(plan.iter().all(|c| !c.partial && c.width == 16));
    }

    #[test]
    fn chunk_widths_sum_to_request_width() {
        let sizing = LookupSizing {
            rangecheck_tables: vec![5, 8],
            spread_tables: vec![16],
        };
        for w in 2u8..=64 {
            let plan = sizing.decompose_rangecheck(w);
            let sum: u32 = plan.iter().map(|c| c.width as u32).sum();
            assert_eq!(sum, w as u32, "widths must sum exactly for w={w}");
        }
        for w in 1u8..=32 {
            let plan = sizing.decompose_spread(w);
            let sum: u32 = plan.iter().map(|c| c.width as u32).sum();
            assert_eq!(sum, w as u32, "spread widths must sum exactly for w={w}");
        }
    }

    #[test]
    fn many_16bit_rangechecks_prefer_a_16bit_table() {
        // Lots of 16-bit checks: a 16-bit table (1 lookup each) beats an 8-bit table (2 each)
        // once the lookup savings outweigh the 2^16 allocation.
        let rc = [(16u8, 200_000usize)];
        let sizing = optimize(&hist(&rc), &hist(&[]));
        assert!(
            sizing.rangecheck_tables.contains(&16),
            "expected a 16-bit table, got {:?}",
            sizing.rangecheck_tables
        );
        // Sanity: the chosen config is no worse than a plain 8-bit table.
        let byte = LookupSizing {
            rangecheck_tables: vec![8],
            spread_tables: vec![],
        };
        assert!(cost_of(&sizing, &rc, &[]) <= cost_of(&byte, &rc, &[]));
    }

    #[test]
    fn few_8bit_rangechecks_avoid_overallocating() {
        // Only a handful of 8-bit checks: a full 2^8 table never pays for itself, so the
        // optimizer prefers a smaller table (two cheap lookups per check) and never does worse
        // than the byte baseline.
        let rc = [(8u8, 10usize)];
        let sizing = optimize(&hist(&rc), &hist(&[]));
        assert_eq!(sizing.rangecheck_tables.len(), 1);
        assert!(
            sizing.rangecheck_tables[0] <= 8,
            "should not allocate a table larger than the request width, got {:?}",
            sizing.rangecheck_tables
        );
        let byte = LookupSizing {
            rangecheck_tables: vec![8],
            spread_tables: vec![],
        };
        assert!(cost_of(&sizing, &rc, &[]) <= cost_of(&byte, &rc, &[]));
    }

    #[test]
    fn sha_like_spreads_unify_to_one_wider_table() {
        // SHA-style: many 16-bit spreads plus 16-/32-bit rangechecks. The byte baseline splits
        // every spread into two 8-bit chunks; a 16-bit spread table does each in one lookup and
        // can also serve the rangechecks via its key column.
        let sp = [(16u8, 100_000usize)];
        let rc = [(32u8, 50_000usize), (16u8, 50_000usize)];
        let sizing = optimize(&hist(&rc), &hist(&sp));

        let chosen = cost_of(&sizing, &rc, &sp);
        let byte = LookupSizing {
            rangecheck_tables: vec![8],
            spread_tables: vec![8],
        };
        assert!(
            chosen < cost_of(&byte, &rc, &sp),
            "dynamic sizing {:?} (cost {chosen}) should beat byte baseline (cost {})",
            sizing,
            cost_of(&byte, &rc, &sp)
        );
    }

    #[test]
    fn optimizer_beats_or_matches_byte_baseline_across_workloads() {
        let workloads: &[(&[(u8, usize)], &[(u8, usize)])] = &[
            (&[(8, 5000)], &[]),
            (&[(16, 5000)], &[]),
            (&[(32, 1000), (8, 4000)], &[]),
            (&[(64, 200)], &[(16, 9000)]),
            (&[(12, 3000), (24, 1500)], &[(11, 2000)]),
        ];
        for (rc, sp) in workloads {
            let sizing = optimize(&hist(rc), &hist(sp));
            let byte = LookupSizing {
                rangecheck_tables: if rc.iter().any(|&(w, _)| w >= 2) {
                    vec![8]
                } else {
                    vec![]
                },
                spread_tables: if sp.is_empty() { vec![] } else { vec![8] },
            };
            assert!(
                cost_of(&sizing, rc, sp) <= cost_of(&byte, rc, sp),
                "workload rc={rc:?} sp={sp:?}: chosen {sizing:?} worse than byte baseline"
            );
        }
    }
}
