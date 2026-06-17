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

/// Rangecheck providers for table-size sets `(R, P)`: range tables (cost 1) plus spread-only sizes
/// (cost 2, via the shared key column).
fn providers_from_sizes(r: &[u8], p: &[u8]) -> Vec<Provider> {
    let mut providers: Vec<Provider> = r
        .iter()
        .map(|&size| Provider {
            size,
            kind: TableKind::Range,
        })
        .collect();
    for &size in p {
        if !r.contains(&size) {
            providers.push(Provider {
                size,
                kind: TableKind::Spread,
            });
        }
    }
    sort_providers(&mut providers);
    providers
}

/// Total R1CS cost (allocation + recurring) of a concrete `(R, P)` table-size configuration, or
/// `None` if it can't serve every lookup. Used to score the native-floor config against the
/// bounded search.
fn config_cost(
    rc_hist: &[(u8, usize)],
    sp_hist: &[(u8, usize)],
    r: &[u8],
    p: &[u8],
    has_spreads: bool,
    has_rangechecks: bool,
) -> Option<usize> {
    if has_spreads && p.is_empty() {
        return None;
    }
    if has_rangechecks && r.is_empty() && p.is_empty() {
        return None;
    }
    let spread_alloc: usize = p
        .iter()
        .map(|&s| allocation_constraints(s, TableKind::Spread))
        .sum();
    let spread_recurring = spread_recurring_cost(sp_hist, &spread_providers_for(p))?;
    let rc_recurring = rangecheck_recurring_cost(rc_hist, &providers_from_sizes(r, p))?;
    let range_alloc: usize = r
        .iter()
        .map(|&s| allocation_constraints(s, TableKind::Range))
        .sum();
    Some(spread_alloc + spread_recurring + range_alloc + rc_recurring)
}

/// A candidate table-size configuration: the chosen range (`r`) and spread (`p`) table sizes.
///
/// The whole search is deterministic — the chosen tables are a pure function of the input
/// histograms. The only nondeterminism source this crate guards against is hash-map iteration
/// order, and the search never iterates the input maps: it reads them once via [`sorted_histogram`]
/// into sorted vectors and then works entirely with `Vec`s and integer costs (no randomness,
/// clocks, or floats). Every step is therefore a deterministic function, so neighbour order and
/// best-improving tie-breaks are identical run to run. (`r`/`p` are sorted before returning, for a
/// canonical result.)
#[derive(Clone, Default, PartialEq, Eq)]
struct Config {
    r: Vec<u8>,
    p: Vec<u8>,
}

/// `true` if cost `a` is strictly cheaper than `b`, treating `None` (an unservable configuration)
/// as `+∞`.
fn cheaper(a: Option<usize>, b: Option<usize>) -> bool {
    match (a, b) {
        (Some(x), Some(y)) => x < y,
        (Some(_), None) => true,
        (None, _) => false,
    }
}

/// Pick the table-size sets `(R, P)` minimizing total R1CS constraints for the given whole-program
/// lookup-width histograms.
///
/// This is an uncapacitated-facility-location problem: each table is a "facility" with a fixed
/// opening cost (its `2^s + 1` allocation) and each lookup is a "client" served at the cost of its
/// cheapest decomposition over the open tables. Opening a table can only lower (never raise) every
/// client's serving cost, so the savings are submodular — which is exactly the regime where greedy
/// construction plus add/drop/swap local search find near-optimal solutions while touching the
/// *entire* candidate space (every size, both kinds), rather than a fixed-cardinality slice of it.
///
/// We run local search from several seeds and keep the cheapest result:
/// * greedy-from-empty — repeatedly open whichever table most reduces total cost;
/// * one dedicated table per distinct lookup width (the allocate-on-demand strategy);
/// * the legacy 8-bit byte tables.
///
/// Seeding from the latter two guarantees we never do worse than either baseline.
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
    if candidates.is_empty() {
        return LookupSizing::default();
    }
    let s_max = *candidates.last().expect("candidates is non-empty");

    let cost = |cfg: &Config| {
        config_cost(
            &rc_hist,
            &sp_hist,
            &cfg.r,
            &cfg.p,
            has_spreads,
            has_rangechecks,
        )
    };

    // Seeds for the local search (see the function doc).
    let distinct = Config {
        r: rc_hist
            .iter()
            .filter(|&&(w, c)| w >= 2 && c > 0 && w <= s_max)
            .map(|&(w, _)| w)
            .collect(),
        p: sp_hist
            .iter()
            .filter(|&&(w, c)| c > 0 && w <= s_max)
            .map(|&(w, _)| w)
            .collect(),
    };
    let byte = {
        let b = 8u8.min(s_max);
        Config {
            r: if has_rangechecks { vec![b] } else { vec![] },
            p: if has_spreads { vec![b] } else { vec![] },
        }
    };
    let seeds = [greedy_construct(&candidates, &cost), distinct, byte];

    let mut best: Option<Config> = None;
    for seed in seeds {
        let refined = local_search(seed, &candidates, &cost);
        if best
            .as_ref()
            .is_none_or(|b| cheaper(cost(&refined), cost(b)))
        {
            best = Some(refined);
        }
    }

    let mut best = best.expect("at least one seed is feasible when lookups are present");
    best.r.sort_unstable();
    best.p.sort_unstable();
    LookupSizing {
        rangecheck_tables: best.r,
        spread_tables: best.p,
    }
}

/// Build a configuration greedily: starting from no tables, repeatedly open whichever single table
/// (range or spread, any candidate size) reduces total cost the most, until none does. The first
/// step necessarily reaches a servable configuration (any one table can chunk every width).
fn greedy_construct(candidates: &[u8], cost: &dyn Fn(&Config) -> Option<usize>) -> Config {
    let mut cfg = Config::default();
    loop {
        let current = cost(&cfg);
        let mut best: Option<Config> = None;
        for cand in neighbors_add(&cfg, candidates) {
            let cand_cost = cost(&cand);
            let beats = best.as_ref().map_or(current, cost);
            if cheaper(cand_cost, beats) {
                best = Some(cand);
            }
        }
        match best {
            Some(next) => cfg = next,
            None => return cfg,
        }
    }
}

/// Refine a configuration to a local optimum under add/drop/swap moves (best-improving). Each move
/// either opens a table, closes one, or replaces one with a different size.
fn local_search(
    mut cfg: Config,
    candidates: &[u8],
    cost: &dyn Fn(&Config) -> Option<usize>,
) -> Config {
    loop {
        let current = cost(&cfg);
        let mut best: Option<Config> = None;
        for cand in neighbors(&cfg, candidates) {
            let cand_cost = cost(&cand);
            let beats = best.as_ref().map_or(current, cost);
            if cheaper(cand_cost, beats) {
                best = Some(cand);
            }
        }
        match best {
            Some(next) => cfg = next,
            None => return cfg,
        }
    }
}

/// All configurations reachable by opening one more table (one per candidate size, each kind).
fn neighbors_add(cfg: &Config, candidates: &[u8]) -> Vec<Config> {
    let mut out = Vec::new();
    for &s in candidates {
        if !cfg.r.contains(&s) {
            let mut c = cfg.clone();
            c.r.push(s);
            out.push(c);
        }
        if !cfg.p.contains(&s) {
            let mut c = cfg.clone();
            c.p.push(s);
            out.push(c);
        }
    }
    out
}

/// All configurations one add/drop/swap move away from `cfg`.
fn neighbors(cfg: &Config, candidates: &[u8]) -> Vec<Config> {
    let mut out = neighbors_add(cfg, candidates);
    // Drops.
    for i in 0..cfg.r.len() {
        let mut c = cfg.clone();
        c.r.remove(i);
        out.push(c);
    }
    for i in 0..cfg.p.len() {
        let mut c = cfg.clone();
        c.p.remove(i);
        out.push(c);
    }
    // Swaps (replace one open table with a different size of the same kind).
    for i in 0..cfg.r.len() {
        for &s in candidates {
            if !cfg.r.contains(&s) {
                let mut c = cfg.clone();
                c.r[i] = s;
                out.push(c);
            }
        }
    }
    for i in 0..cfg.p.len() {
        for &s in candidates {
            if !cfg.p.contains(&s) {
                let mut c = cfg.clone();
                c.p[i] = s;
                out.push(c);
            }
        }
    }
    out
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
