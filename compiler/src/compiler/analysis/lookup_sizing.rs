//! Dynamic lookup-table sizing for rangechecks and spreads.
//!
//! Every rangecheck and spread is ultimately served by one or more lookup tables. A table of
//! size `2^s` checks a key against `[0, 2^s)`; a `w`-bit value too wide for a single table is
//! split into chunks.
//!
//! This analysis reads the whole-program lookup-width histograms (from [`Summary`]) and picks an
//! optimal *set* of table sizes — separately for rangechecks (`R`, width-1 tables) and spreads
//! (`P`, width-2 tables) — minimizing a joint cost `witnesses + CONSTRAINT_WEIGHT · constraints`.
//! Witnesses dominate the prover's commitment, but constraints are not free, so both are counted
//! (see [`CONSTRAINT_WEIGHT`]). Because a spread table's key column is itself a rangecheck,
//! rangechecks may also be served by a spread table (at 2 witnesses per lookup instead of 1), so
//! the two decisions are optimized jointly.
//!
//! ## Cost model
//! Each element contributes both witnesses and constraints, combined via [`CONSTRAINT_WEIGHT`]:
//! * Rangecheck table at size `s`: `2·2^s` allocation witnesses (a multiplicity + an inverse per
//!   row) and `2^s + 1` allocation constraints (a folded per-row constraint + the sum constraint);
//!   `1` witness and `1` constraint per lookup.
//! * Spread table at size `s`: same allocation, `2` witnesses and `2` constraints per lookup.
//! * Each chunk of a multi-chunk decomposition except the derived lowest one also costs its
//!   extraction witness(es) — 1 for a rangecheck chunk, 2 for a spread chunk (no extra constraint).
//! * A width-1 bit chunk needs no table: it costs only its `b·(b−1)=0` bound — 1 constraint
//!   unconditionally, or a gating witness + 2 constraints when flag-conditional (see
//!   [`chunk_witnesses`]/[`chunk_constraints`]).
//!
//! ## Simulation tricks
//! * A full chunk of exactly `s` bits is one lookup in an `s`-bit table.
//! * A residual chunk of `r < s` bits is bounded with the "2-larger" trick: look up `x` and
//!   `(2^r − 1) − x` in a larger table. The complement `(2^r − 1) − x` only has to be
//!   range-bounded (to rule out field wraparound), so when an `r`-or-larger *rangecheck* table
//!   exists it is a 1-witness rangecheck rather than a second 2-witness spread lookup — a spread
//!   partial costs `2 + 1` rather than `2 + 2`. The primary lookup keeps its kind (a spread must
//!   stay a spread to produce `spread(x)`).

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

/// A table of `MAX_TABLE_BITS` bits is indexed into the backends' fixed-size table caches by its
/// bit-width, so the cache must have a slot for that width. Guarantee it at compile time.
const _: () = assert!(
    (MAX_TABLE_BITS as usize) < crate::vm::bytecode::NUM_TABLE_SIZE_SLOTS,
    "MAX_TABLE_BITS exceeds the VM's fixed-size table cache; widen rgchk_tables/spread_tables",
);

/// How many witnesses one R1CS constraint is worth in the joint cost the optimizer minimizes
/// (`witnesses + CONSTRAINT_WEIGHT · constraints`). Witnesses dominate the prover's commitment, but
/// constraints are not free (sumcheck rounds, matrix nonzeros), so a witness saving that costs
/// several constraints is usually a bad trade. `1` weights them equally; `0` recovers the
/// witness-only metric (and lets free unconditional bit decomposition undercut every table).
const CONSTRAINT_WEIGHT: usize = 1;

/// Which physical table a chunk lookup targets.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum TableKind {
    /// A width-1 rangecheck table (`Rangecheck(s)`), 1 witness per lookup.
    Range,
    /// A width-2 spread table (`Spread(s)`), 2 witnesses per lookup.
    Spread,
}

impl TableKind {
    fn per_lookup_witnesses(self) -> usize {
        match self {
            TableKind::Range => 1,
            TableKind::Spread => 2,
        }
    }
}

/// Witnesses for one `width`-bit chunk in a `kind` table: the lookup witnesses
/// (`per_lookup_witnesses`), an extraction witness for every chunk except the derived lowest one
/// (2 for a spread, whose `spread(key)` hint is also witnessed), and a partial chunk's complement
/// ([`complement_cost`]). A width-1 chunk needs no table: its `b·(b−1)=0` bound is free when
/// unconditional, or costs one gating witness when `gated` (the degree-3 `flag·b·(b−1)=0` an R1CS
/// constraint can't express without an intermediate `b²`). See the module-level cost model.
fn chunk_witnesses(
    width: u8,
    kind: TableKind,
    partial: bool,
    max_range_size: u8,
    extracted: bool,
    gated: bool,
) -> usize {
    if width <= 1 {
        return usize::from(gated) + usize::from(extracted);
    }
    let extraction = if extracted {
        if kind == TableKind::Spread { 2 } else { 1 }
    } else {
        0
    };
    let complement = if partial {
        complement_cost(width, kind, max_range_size)
    } else {
        0
    };
    kind.per_lookup_witnesses() + extraction + complement
}

/// Constraints for one chunk, the twin of [`chunk_witnesses`]. Extraction is witness-only (a fresh
/// witness pinned by a linear reconstruction), so it doesn't appear here. A width-1 bit costs its
/// algebraic bound: 1 constraint unconditionally, or 2 when `gated` (via the `b²` witness).
fn chunk_constraints(
    width: u8,
    kind: TableKind,
    partial: bool,
    max_range_size: u8,
    gated: bool,
) -> usize {
    if width <= 1 {
        return if gated { 2 } else { 1 };
    }
    let complement = if partial {
        complement_cost(width, kind, max_range_size)
    } else {
        0
    };
    kind.per_lookup_witnesses() + complement
}

/// A partial chunk's "2-larger" complement, costing one lookup (so one witness *and* one constraint
/// alike): a 1-cost rangecheck when a range table of size ≥ `width` exists, otherwise a lookup in
/// the chunk's own `kind` table (a spread additionally carries its spread-hint slot).
fn complement_cost(width: u8, kind: TableKind, max_range_size: u8) -> usize {
    if max_range_size >= width {
        TableKind::Range.per_lookup_witnesses()
    } else {
        kind.per_lookup_witnesses() + usize::from(kind == TableKind::Spread)
    }
}

/// Joint cost of one chunk: `witnesses + CONSTRAINT_WEIGHT · constraints`. This is what the
/// decomposition DP and the optimizer minimize.
fn chunk_cost(
    width: u8,
    kind: TableKind,
    partial: bool,
    max_range_size: u8,
    extracted: bool,
    gated: bool,
) -> usize {
    chunk_witnesses(width, kind, partial, max_range_size, extracted, gated)
        + CONSTRAINT_WEIGHT * chunk_constraints(width, kind, partial, max_range_size, gated)
}

/// One chunk of a decomposed rangecheck/spread lookup.
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
    fn rangecheck_providers(&self) -> Vec<Provider> {
        rangecheck_providers_from_sizes(&self.rangecheck_tables, &self.spread_tables)
    }

    fn spread_providers(&self) -> Vec<Provider> {
        spread_providers_from_sizes(&self.spread_tables)
    }

    /// The largest rangecheck table available — bounds partial chunks' complements at 1 witness.
    fn max_range_size(&self) -> u8 {
        self.rangecheck_tables.iter().copied().max().unwrap_or(0)
    }

    /// How to decompose a `width`-bit rangecheck into chunk lookups (lowest chunk first). `gated`
    /// reflects whether the lookup is flag-conditional, which changes the price (and so the chosen
    /// shape) of width-1 bit chunks. `width == 1` is handled algebraically by the caller and returns
    /// an empty plan.
    pub fn decompose_rangecheck(&self, width: u8, gated: bool) -> Vec<Chunk> {
        if width <= 1 {
            return vec![];
        }
        decompose(
            width,
            &self.rangecheck_providers(),
            self.max_range_size(),
            gated,
        )
        .1
    }

    /// How to decompose a `width`-bit spread into chunk lookups (lowest chunk first). `gated` — see
    /// [`Self::decompose_rangecheck`].
    pub fn decompose_spread(&self, width: u8, gated: bool) -> Vec<Chunk> {
        decompose(
            width,
            &self.spread_providers(),
            self.max_range_size(),
            gated,
        )
        .1
    }
}

/// DP table of the minimum decomposition cost (joint witnesses + constraints) for *every* width
/// `0..=max_width` over `providers` — the cost-only twin of [`decompose`], a pure integer DP with
/// no per-state `Vec<Chunk>`. One pass fills the whole table, so the optimizer (which evaluates this
/// across millions of `(R, P)` candidate pairs) computes it once per (provider set, conditionality)
/// and indexes it per histogram width rather than re-running the DP per width.
///
/// The recurrence covers `x` bits by peeling a full provider-sized chunk off the top (recursing on
/// the low `x − s` bits; the chunk reaching the base pays no extraction), peeling a single width-1
/// bit chunk (`x − 1`), or covering all of `x` as one partial chunk in a strictly-larger table (the
/// 2-larger trick). The always-available bit chunk makes every width servable.
fn decompose_costs(
    max_width: u8,
    providers: &[Provider],
    max_range_size: u8,
    gated: bool,
) -> Vec<usize> {
    let w = max_width as usize;
    let mut best = vec![0usize; w + 1]; // best[0] = 0
    for x in 1..=w {
        let mut chosen = usize::MAX;
        for p in providers {
            let s = p.size as usize;
            if s <= x {
                let extracted = x - s > 0;
                let cost = chunk_cost(p.size, p.kind, false, max_range_size, extracted, gated);
                chosen = chosen.min(best[x - s] + cost);
            }
        }
        let extracted = x - 1 > 0;
        let bit = chunk_cost(1, TableKind::Range, false, max_range_size, extracted, gated);
        chosen = chosen.min(best[x - 1] + bit);
        for p in providers {
            if (p.size as usize) > x {
                chosen = chosen.min(chunk_cost(
                    x as u8,
                    p.kind,
                    true,
                    max_range_size,
                    false,
                    gated,
                ));
            }
        }
        best[x] = chosen;
    }
    best
}

/// Optimal decomposition of a `width`-bit value over `providers`, as `(cost, chunks)` with the
/// lowest-order chunk first. The recurrence covers `x` bits by peeling a full provider-sized chunk
/// off the top (recursing on `x − s`), peeling a single width-1 bit chunk (recursing on `x − 1`),
/// or covering all of `x` as one partial chunk in a strictly-larger table (the 2-larger trick,
/// always terminal). The cost-only twin used by the optimizer is [`decompose_costs`].
fn decompose(
    width: u8,
    providers: &[Provider],
    max_range_size: u8,
    gated: bool,
) -> (usize, Vec<Chunk>) {
    if width == 0 {
        return (0, vec![]);
    }
    let w = width as usize;
    let mut best: Vec<Option<(usize, Vec<Chunk>)>> = vec![None; w + 1];
    best[0] = Some((0, vec![]));

    for x in 1..=w {
        let mut chosen: Option<(usize, Vec<Chunk>)> = None;
        let mut consider = |cost: usize, make_plan: &dyn Fn() -> Vec<Chunk>| {
            if chosen.as_ref().is_none_or(|(c, _)| cost < *c) {
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
                let extracted = x - s > 0;
                let cost =
                    sub_cost + chunk_cost(p.size, p.kind, false, max_range_size, extracted, gated);
                consider(cost, &|| {
                    let mut plan = sub_plan.clone();
                    plan.push(chunk);
                    plan
                });
            }
        }

        // Peel a single bit off the top (a width-1 chunk needs no table); recurse on x - 1.
        if let Some((sub_cost, sub_plan)) = best[x - 1].clone() {
            let chunk = Chunk {
                width: 1,
                table_size: 1,
                table: TableKind::Range,
                partial: false,
            };
            let extracted = x - 1 > 0;
            let cost =
                sub_cost + chunk_cost(1, TableKind::Range, false, max_range_size, extracted, gated);
            consider(cost, &|| {
                let mut plan = sub_plan.clone();
                plan.push(chunk);
                plan
            });
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
                let cost = chunk_cost(x as u8, p.kind, true, max_range_size, false, gated);
                consider(cost, &|| vec![chunk]);
            }
        }

        best[x] = chosen;
    }

    let (cost, plan) = best[w]
        .clone()
        .expect("the width-1 bit chunk serves every width");
    (cost, canonicalize_plan(plan))
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

/// Allocation witnesses for a size-`s` table: a multiplicity witness and an inverse witness for each
/// of its `2^s` rows. Both range and spread tables use this layout.
fn allocation_witnesses(size: u8) -> usize {
    2 * (1usize << size)
}

/// Allocation *constraints* for a table of the given size: one folded constraint per row (both
/// operands of an entry are compile-time constants) plus one per-table sum constraint.
fn allocation_constraints(size: u8) -> usize {
    (1usize << size) + 1
}

/// Joint allocation cost of a table: `witnesses + CONSTRAINT_WEIGHT · constraints`.
fn allocation_cost(size: u8) -> usize {
    allocation_witnesses(size) + CONSTRAINT_WEIGHT * allocation_constraints(size)
}

/// Project a `(width, is_unconditional) -> count` histogram into a sorted `(width, count)` one,
/// optionally restricted to one conditionality (`Some(true)` = unconditional, `Some(false)` =
/// gated, `None` = both). Sorted for deterministic, reproducible optimization.
fn project(hist: &HashMap<(u8, bool), usize>, conditionality: Option<bool>) -> Vec<(u8, usize)> {
    let mut by_width: HashMap<u8, usize> = HashMap::default();
    for (&(width, uncond), &count) in hist {
        if conditionality.is_none_or(|c| c == uncond) {
            *by_width.entry(width).or_default() += count;
        }
    }
    let mut v: Vec<(u8, usize)> = by_width.into_iter().collect();
    v.sort_unstable();
    v
}

/// Total per-lookup rangecheck cost over the histogram for the given providers (excludes table
/// allocation; 1-bit checks are algebraic and excluded). `max_range_size` is the largest rangecheck
/// table available, used to price partial chunks' complements.
fn rangecheck_recurring_cost(
    rc_hist: &[(u8, usize)],
    providers: &[Provider],
    max_range_size: u8,
    gated: bool,
) -> usize {
    recurring_cost(rc_hist, providers, max_range_size, gated, true)
}

/// Total per-lookup spread cost over the histogram (excludes table allocation). A multi-chunk
/// spread reconstructs both its key and its spread from the lowest chunk by subtracting the higher
/// chunks (free linear combinations), so it needs no recombination constraint. A partial chunk's
/// complement is a plain rangecheck when a rangecheck table of sufficient size exists, hence the
/// `max_range_size`.
fn spread_recurring_cost(
    sp_hist: &[(u8, usize)],
    providers: &[Provider],
    max_range_size: u8,
    gated: bool,
) -> usize {
    recurring_cost(sp_hist, providers, max_range_size, gated, false)
}

/// Sum of per-lookup decomposition costs over a width histogram. Builds the cost-for-all-widths DP
/// table once (see [`decompose_costs`]) and indexes it per width, rather than decomposing each width
/// separately. `skip_width_one` drops 1-bit lookups, which rangechecks lower inline for free.
fn recurring_cost(
    hist: &[(u8, usize)],
    providers: &[Provider],
    max_range_size: u8,
    gated: bool,
    skip_width_one: bool,
) -> usize {
    let served = |&&(width, count): &&(u8, usize)| count > 0 && (width > 1 || !skip_width_one);
    let Some(max_width) = hist.iter().filter(served).map(|&(width, _)| width).max() else {
        return 0;
    };
    let costs = decompose_costs(max_width, providers, max_range_size, gated);
    hist.iter()
        .filter(served)
        .map(|&(width, count)| costs[width as usize] * count)
        .sum()
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

/// Providers available for a spread lookup: the spread tables only.
fn spread_providers_from_sizes(p: &[u8]) -> Vec<Provider> {
    p.iter()
        .map(|&size| Provider {
            size,
            kind: TableKind::Spread,
        })
        .collect()
}

/// Providers available for a rangecheck given range sizes `r` and spread sizes `p`: every range
/// table (cost 1) plus every spread-only size (cost 2, via the shared key column).
fn rangecheck_providers_from_sizes(r: &[u8], p: &[u8]) -> Vec<Provider> {
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
    providers
}

/// The whole-program lookup-width histograms the optimizer prices against, split by conditionality
/// (sorted slices). Unconditional and gated lookups share the chosen tables but price width-1 bit
/// chunks differently (see [`chunk_cost`]).
struct Workload<'a> {
    rc_uncond: &'a [(u8, usize)],
    rc_gated: &'a [(u8, usize)],
    sp_uncond: &'a [(u8, usize)],
    sp_gated: &'a [(u8, usize)],
}

/// Total joint cost (allocation + recurring, each `witnesses + CONSTRAINT_WEIGHT · constraints`) of
/// a concrete `(R, P)` table-size configuration. Allocation is counted once; the unconditional and
/// gated lookups are priced separately over the shared tables.
fn config_cost(w: &Workload, r: &[u8], p: &[u8]) -> usize {
    // The largest rangecheck table available to bound partial chunks' complements (only true range
    // tables count — a spread table's key column would cost 2, not 1).
    let max_range_size = r.iter().copied().max().unwrap_or(0);
    let spread_providers = spread_providers_from_sizes(p);
    let rc_providers = rangecheck_providers_from_sizes(r, p);
    let alloc: usize = r.iter().chain(p).map(|&s| allocation_cost(s)).sum();
    let spread_recurring =
        spread_recurring_cost(w.sp_uncond, &spread_providers, max_range_size, false)
            + spread_recurring_cost(w.sp_gated, &spread_providers, max_range_size, true);
    let rc_recurring = rangecheck_recurring_cost(w.rc_uncond, &rc_providers, max_range_size, false)
        + rangecheck_recurring_cost(w.rc_gated, &rc_providers, max_range_size, true);
    alloc + spread_recurring + rc_recurring
}

/// A candidate table-size configuration: the chosen range (`r`) and spread (`p`) table sizes.
#[derive(Clone, Default, PartialEq, Eq)]
struct Config {
    r: Vec<u8>,
    p: Vec<u8>,
}

/// Pick the table-size sets `(R, P)` minimizing the joint cost for the given whole-program
/// lookup-width histograms.
///
/// Each histogram is keyed by `(width, is_unconditional)`; [`project`] splits it into the total
/// (for candidate sizing) and the per-conditionality slices the cost model prices separately.
///
/// Deterministic: the chosen tables are a pure function of the input histograms. The search reads
/// the maps once into sorted `Vec`s and then works entirely with integer costs (no randomness,
/// clocks, or floats), so neighbour order and tie-breaks are identical run to run.
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
fn optimize(
    rangecheck_hist: &HashMap<(u8, bool), usize>,
    spread_hist: &HashMap<(u8, bool), usize>,
) -> LookupSizing {
    let rc_hist = project(rangecheck_hist, None);
    let sp_hist = project(spread_hist, None);

    let has_rangechecks = rc_hist.iter().any(|&(w, c)| w >= 2 && c > 0);
    let has_spreads = sp_hist.iter().any(|&(_, c)| c > 0);
    if !has_rangechecks && !has_spreads {
        return LookupSizing::default();
    }

    // Unconditional lookups (free width-1 bits) and gated ones (a width-1 bit costs a gating
    // witness) share the chosen tables but price bit chunks differently.
    let rc_uncond = project(rangecheck_hist, Some(true));
    let sp_uncond = project(spread_hist, Some(true));
    let rc_gated = project(rangecheck_hist, Some(false));
    let sp_gated = project(spread_hist, Some(false));

    let candidates = candidate_sizes(&rc_hist, &sp_hist);
    if candidates.is_empty() {
        return LookupSizing::default();
    }
    let s_max = *candidates.last().expect("candidates is non-empty");

    let workload = Workload {
        rc_uncond: &rc_uncond,
        rc_gated: &rc_gated,
        sp_uncond: &sp_uncond,
        sp_gated: &sp_gated,
    };
    let cost = |cfg: &Config| config_cost(&workload, &cfg.r, &cfg.p);

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
        if best.as_ref().is_none_or(|b| cost(&refined) < cost(b)) {
            best = Some(refined);
        }
    }

    let mut best = best.expect("the seed list is non-empty");
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
fn greedy_construct(candidates: &[u8], cost: &dyn Fn(&Config) -> usize) -> Config {
    let mut cfg = Config::default();
    loop {
        let current = cost(&cfg);
        let mut best: Option<Config> = None;
        for cand in neighbors_add(&cfg, candidates) {
            let cand_cost = cost(&cand);
            let beats = best.as_ref().map_or(current, cost);
            if cand_cost < beats {
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
fn local_search(mut cfg: Config, candidates: &[u8], cost: &dyn Fn(&Config) -> usize) -> Config {
    loop {
        let current = cost(&cfg);
        let mut best: Option<Config> = None;
        for cand in neighbors(&cfg, candidates) {
            let cand_cost = cost(&cand);
            let beats = best.as_ref().map_or(current, cost);
            if cand_cost < beats {
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

    /// A `(width, is_unconditional) -> count` histogram from `gated` and `uncond` `(width, count)`
    /// entries.
    fn hist(gated: &[(u8, usize)], uncond: &[(u8, usize)]) -> HashMap<(u8, bool), usize> {
        gated
            .iter()
            .map(|&(w, c)| ((w, false), c))
            .chain(uncond.iter().map(|&(w, c)| ((w, true), c)))
            .collect()
    }

    /// Joint cost of a configuration treating every lookup as gated. Used by the byte-baseline
    /// comparison tests, which only feed gated histograms to `optimize`, so this matches what the
    /// optimizer computed for them.
    fn cost_of(sizing: &LookupSizing, rc: &[(u8, usize)], sp: &[(u8, usize)]) -> usize {
        let alloc: usize = sizing
            .rangecheck_tables
            .iter()
            .map(|&s| allocation_cost(s))
            .sum::<usize>()
            + sizing
                .spread_tables
                .iter()
                .map(|&s| allocation_cost(s))
                .sum::<usize>();
        let max_range = sizing.max_range_size();
        let rc_cost =
            rangecheck_recurring_cost(rc, &sizing.rangecheck_providers(), max_range, true);
        let sp_cost = spread_recurring_cost(sp, &sizing.spread_providers(), max_range, true);
        alloc + rc_cost + sp_cost
    }

    #[test]
    fn no_lookups_means_no_tables() {
        let sizing = optimize(&hist(&[], &[]), &hist(&[], &[]));
        assert!(sizing.rangecheck_tables.is_empty());
        assert!(sizing.spread_tables.is_empty());
    }

    #[test]
    fn one_bit_rangechecks_need_no_table() {
        let sizing = optimize(&hist(&[(1, 1000)], &[]), &hist(&[], &[]));
        assert!(sizing.rangecheck_tables.is_empty());
        assert!(sizing.spread_tables.is_empty());
    }

    #[test]
    fn full_chunk_decomposition_is_single_lookup() {
        let sizing = LookupSizing {
            rangecheck_tables: vec![8],
            spread_tables: vec![],
        };
        let plan = sizing.decompose_rangecheck(8, true);
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
        let plan = sizing.decompose_rangecheck(4, true);
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
        let plan = sizing.decompose_rangecheck(32, true);
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
            let plan = sizing.decompose_rangecheck(w, true);
            let sum: u32 = plan.iter().map(|c| c.width as u32).sum();
            assert_eq!(sum, w as u32, "widths must sum exactly for w={w}");
        }
        for w in 1u8..=32 {
            let plan = sizing.decompose_spread(w, true);
            let sum: u32 = plan.iter().map(|c| c.width as u32).sum();
            assert_eq!(sum, w as u32, "spread widths must sum exactly for w={w}");
        }
    }

    #[test]
    fn many_16bit_rangechecks_prefer_a_16bit_table() {
        // Lots of 16-bit checks: a 16-bit table (1 lookup each) beats an 8-bit table (2 each)
        // once the lookup savings outweigh the 2^16 allocation.
        let rc = [(16u8, 200_000usize)];
        let sizing = optimize(&hist(&rc, &[]), &hist(&[], &[]));
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
        let sizing = optimize(&hist(&rc, &[]), &hist(&[], &[]));
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
        let sizing = optimize(&hist(&rc, &[]), &hist(&sp, &[]));

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
    fn unconditional_bits_are_free_gated_bits_cost_a_witness() {
        // No tables available, so a 3-bit value must be fully bit-decomposed. Both costs are the
        // joint `witnesses + W·constraints`; gated bits add a gating witness and a second
        // constraint each, so the gated decomposition is strictly more expensive.
        let uncond = decompose_costs(3, &[], 0, false)[3];
        let gated = decompose_costs(3, &[], 0, true)[3];
        assert!(
            uncond < gated,
            "unconditional decomposition ({uncond}) must be cheaper than gated ({gated})"
        );
    }

    #[test]
    fn unconditional_lookups_can_skip_a_table() {
        // Very few 8-bit checks, all unconditional: free bit decomposition (3×7 = 21 extraction
        // witnesses) undercuts the cheapest table (a 2^4 table costs 32 to allocate alone), so the
        // optimizer opens none.
        let rc = [(8u8, 3usize)];
        let sizing = optimize(&hist(&[], &rc), &hist(&[], &[]));
        assert!(
            sizing.rangecheck_tables.is_empty(),
            "all-unconditional very-low-volume checks should bit-decompose, not allocate: {:?}",
            sizing.rangecheck_tables
        );
    }

    #[test]
    fn gated_low_volume_lookups_still_allocate() {
        // The same low-volume workload, but gated: bit decomposition now costs 3×15 = 45, so a
        // small table wins and the optimizer allocates one (contrast `unconditional_lookups_*`).
        let rc = [(8u8, 3usize)];
        let sizing = optimize(&hist(&rc, &[]), &hist(&[], &[]));
        assert!(
            !sizing.rangecheck_tables.is_empty(),
            "gated checks should allocate rather than pay per-bit gating witnesses"
        );
    }

    #[test]
    fn many_unconditional_lookups_still_want_a_table() {
        // At high volume the per-lookup extraction witnesses of bit decomposition outweigh a
        // table's one-time allocation, so even unconditional lookups open a table.
        let rc = [(8u8, 200_000usize)];
        let sizing = optimize(&hist(&[], &rc), &hist(&[], &[]));
        assert!(
            !sizing.rangecheck_tables.is_empty(),
            "high-volume checks should still allocate a table"
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
            let sizing = optimize(&hist(rc, &[]), &hist(sp, &[]));
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
