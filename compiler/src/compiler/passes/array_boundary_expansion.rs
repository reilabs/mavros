//! Interprocedural array boundary expansion, performing whole-program scalar replacement of array
//! aggregates across call boundaries.
//!
//! Where array SROA peels a fixed-size array into one value per cell, it only does this for arrays
//! that never cross a function boundary: a parameter, a `Call` arg/result, or a `Return` is a
//! *collapse trigger*, so an array passed into a helper or returned from one is never
//! scalar-replaced even when both sides only ever constant-index it. This pass severs those call
//! boundaries so the existing intra-function passes can finish the job.
//!
//! ## Why this is a Separate Pass
//!
//! This is deliberately a *separate* pass rather than an extension of `array_sroa`, for the same
//! reason `arg_promotion` is separate from `mem2reg`.
//!
//! `array_sroa`'s correctness rests on a **boundary guarantee**: a `Split` value never appears at a
//! parameter, `Call`, `Return`, or global, which is what lets it be a simple one-level
//! intra-function value rewriter (its `single()` tripwire panics if that guarantee is ever
//! violated).
//!
//! Teaching `array_sroa` to rewrite signatures, call sites and returns directly would dismantle
//! that invariant and reinvent the caller/callee coordination this pass performs. Instead, this
//! pass *only severs boundaries* — emitting local `MkSeq`/`ArrayGet` so the array still exists
//! locally but no longer crosses the boundary — and leaves the actual peeling to a downstream
//! `ArraySroa`, with `dce` reclaiming any cells that turn out unused. Each pass stays simple and
//! composable.
//!
//! ## Reuse, not Re-Threading
//!
//! Like `arg_promotion`, this pass rewrites only the *boundary* of procedures; it reconstructs the
//! original array value locally so the bodies in the middle are untouched and still typecheck.
//!
//! - **Callee (Parameter):** The `Array<T, N>` formal `i` of `g` becomes `N` by-value `T` formals;
//!   a `MkSeq` of those `N` cells, *reusing the original parameter id as its result*, is prepended
//!   at entry. The reconstructed array is now purely local and constant-indexed, so the follow-up
//!   `ArraySroa` reclassifies it `Split` and peels it.
//! - **Callee (Return):** Before each `Return`, the `N` cells of the returned array are `ArrayGet`d
//!   and returned as `N` scalars; the return-type signature is widened to match.
//! - **Caller (Argument):** At every `Call{Static(g)}` site, the argument array's `N` cells are
//!   `ArrayGet`d before the call and passed by value.
//! - **Caller (Result):** the call's expanded result is `MkSeq`d back into the original result id
//!   after the call.
//!
//! A trailing `ArraySroa` then peels both reconstructions, and a trailing `DCE` drops any cell a
//! side never used (a dead by-value formal, its matching call argument at every site, and the dead
//! caller-side `ArrayGet`) on an interprocedural level.
//!
//! ## Parameter and Return Expansion
//!
//! Expansion fires only where the points-to layer proves it both safe and profitable, as given by
//! the following conditions.
//!
//! - **Owned & Enumerable:** `g` is type-analyzable, not an entry point, not the globals init or
//!   deinit function, not address-taken, and has at least one static call site — so *every* call
//!   site is a `Call{Static(g)}` this pass rewrites.
//! - **Shape:** A *bare* (non-witnessed) fixed-size `Array<T, N>` of bare *scalar* cells
//!   (`Field`/`u*`/`i*`), with `MIN_CELLS <= N <= SIZE_CAP`.
//! - **Callee-Side Splittability:** The parameter (respectively, every `Return`'s value at that
//!   position) would be `Split` were it not for crossing *this* boundary — i.e. it hits no *hard*
//!   collapse trigger and no *other* boundary.
//! - **Caller-Side Profitability:** At every site the array would itself be `Split` once the call
//!   boundary is severed — for a parameter, the argument array
//!   ([`PointsTo::boundary_splittable_arg`]) so the inserted per-cell `ArrayGet`s peel; for a
//!   return, the result array ([`PointsTo::boundary_splittable_result`]) so the reconstructing
//!   `MkSeq` peels. Without this an expansion would *widen* the boundary (extra `ArrayGet`s / a
//!   surviving `MkSeq`) for no net peel.
//!
//! Because array values are value-semantic, there is no caller co-argument aliasing hazard (the one
//! `arg_promotion`'s C4 guards against for refs), so no aliasing check is needed here.
//!
//! ## Deferred Improvements
//!
//! The following are deferred improvements for future work.
//!
//! - **Composition:** A parameter forwarded onward to another expandable call, or a returned value
//!   that is itself a parameter, needs a call-graph fixpoint; blocked by the single-boundary
//!   condition.
//! - **Witnessed / Nested / Ref-Cell Arrays:** Blocked by the shape condition.
//! - **Globals:** Array-typed globals are a separate module-level effort; they remain hard collapse
//!   triggers here.
//! - **Multi-Consumer Args:** an argument array also passed to a *non-expanded* call survives as a
//!   second consumer, so its caller `MkSeq` does not fully peel and bytecode can still grow
//!   slightly. The caller-side guards bound this to a latent size risk, never an unsoundness;
//!   tightening to "every consumer is expanded" is a follow-up.

use itertools::Itertools;

use crate::{
    collections::{HashMap, HashSet},
    compiler::{
        analysis::{points_to::PointsTo, types::TypeInfo},
        pass_manager::{Analysis, AnalysisId, AnalysisStore, Pass},
        ssa::{
            BlockId, FunctionId, Terminator, ValueId,
            hlssa::{
                CallTarget, Constant, HLFunction, HLSSA, LocatedOpCode, OpCode, SequenceTargetType,
                Type, TypeExpr,
            },
        },
    },
};

// CONSTANTS
// ================================================================================================

/// Smallest cell count worth expanding (`N < 2` is pure churn — one arg for one arg).
const MIN_CELLS: usize = 2;

/// Largest cell count worth expanding, bounding signature/argument-list blow-up.
const SIZE_CAP: usize = 16;

// ARRAY BOUNDARY EXPANSION PASS
// ================================================================================================

pub struct ArrayBoundaryExpansion {}

impl ArrayBoundaryExpansion {
    pub fn new() -> Self {
        ArrayBoundaryExpansion {}
    }
}

impl Pass for ArrayBoundaryExpansion {
    fn name(&self) -> &'static str {
        "array_boundary_expansion"
    }

    fn needs(&self) -> Vec<AnalysisId> {
        // `PointsTo` pulls in `FlowAnalysis`/`TypeInfo` transitively (its `dependencies()`).
        vec![TypeInfo::id(), PointsTo::id()]
    }

    fn run(&self, ssa: &mut HLSSA, store: &AnalysisStore) {
        self.do_run(ssa, store.get::<TypeInfo>(), store.get::<PointsTo>());
    }

    fn preserves(&self) -> Vec<AnalysisId> {
        // New params, returns, and call-site instructions invalidate every cached analysis.
        vec![]
    }
}

impl ArrayBoundaryExpansion {
    pub fn do_run(&self, ssa: &mut HLSSA, types: &TypeInfo, points_to: &PointsTo) {
        let selection = select(ssa, types, points_to);
        if selection.is_empty() {
            return;
        }
        let plan = build_plan(ssa, &selection);
        apply(ssa, &plan);
    }
}

// PHASE 0: SELECTION
// ================================================================================================

/// The expandable `(callee, slot) -> (cell count, element type)` sets, for parameters and returns.
#[derive(Default)]
struct Selection {
    /// Per callee, `param index -> (N, element type)`.
    params: HashMap<FunctionId, HashMap<usize, (usize, Type)>>,

    /// Per callee, `return position -> (N, element type)`.
    returns: HashMap<FunctionId, HashMap<usize, (usize, Type)>>,
}

impl Selection {
    fn is_empty(&self) -> bool {
        self.params.is_empty() && self.returns.is_empty()
    }
}

/// `Some((element type, N))` iff `ty` is a bare fixed-size array of bare scalar cells within the
/// size bounds — the only shape v1 expands.
fn expandable_array(ty: &Type) -> Option<(Type, usize)> {
    let TypeExpr::Array(_, n) = &ty.expr else {
        return None;
    };
    let n = *n;
    if !(MIN_CELLS..=SIZE_CAP).contains(&n) {
        return None;
    }
    let elem = ty.get_array_element();
    if matches!(elem.expr, TypeExpr::Field | TypeExpr::U(_) | TypeExpr::I(_)) {
        Some((elem, n))
    } else {
        None
    }
}

/// Whether `g` returns at `pos` from at least one `Return`, and every such returned value is
/// splittable-modulo-the-return-boundary.
fn returns_all_splittable(
    func: &HLFunction,
    points_to: &PointsTo,
    g: FunctionId,
    pos: usize,
) -> bool {
    let mut saw_return = false;
    for (_, block) in func.get_blocks() {
        if let Some(Terminator::Return(values)) = block.get_terminator() {
            saw_return = true;
            match values.get(pos) {
                Some(v) if points_to.boundary_splittable_return(g, *v) => {}
                _ => return false,
            }
        }
    }
    saw_return
}

/// Move each callee's surviving slots from `cands` into `out`: skip a callee with no static call
/// site, and drop any `(callee, slot)` in `drops`. Shared by parameter and return selection.
fn retain_kept(
    cands: HashMap<FunctionId, HashMap<usize, (usize, Type)>>,
    drops: &HashSet<(FunctionId, usize)>,
    has_site: &HashSet<FunctionId>,
    out: &mut HashMap<FunctionId, HashMap<usize, (usize, Type)>>,
) {
    for (g, slots) in cands {
        if !has_site.contains(&g) {
            continue;
        }
        let kept: HashMap<usize, (usize, Type)> = slots
            .into_iter()
            .filter(|(slot, _)| !drops.contains(&(g, *slot)))
            .collect();
        if !kept.is_empty() {
            out.insert(g, kept);
        }
    }
}

fn select(ssa: &HLSSA, types: &TypeInfo, points_to: &PointsTo) -> Selection {
    let entry_points: HashSet<FunctionId> = ssa.get_entry_points().iter().copied().collect();
    let globals_init = ssa.get_globals_init_fn();
    let globals_deinit = ssa.get_globals_deinit_fn();

    // A function whose pointer is taken cannot have all its call sites enumerated (and could be
    // reached via a dynamic call), so it is never expanded.
    let mut address_taken: HashSet<FunctionId> = HashSet::default();
    ssa.for_each_const(|_, cv| {
        if let Constant::FnPtr(g) = &**cv {
            address_taken.insert(*g);
        }
    });
    let excluded = |g: FunctionId| {
        !types.has_function(g)
            || entry_points.contains(&g)
            || address_taken.contains(&g)
            || Some(g) == globals_init
            || Some(g) == globals_deinit
    };

    // Callee-local candidates.
    let mut cand_params: HashMap<FunctionId, HashMap<usize, (usize, Type)>> = HashMap::default();
    let mut cand_returns: HashMap<FunctionId, HashMap<usize, (usize, Type)>> = HashMap::default();
    for g in ssa.get_function_ids() {
        if excluded(g) {
            continue;
        }
        let func = ssa.get_function(g);

        let mut pmap: HashMap<usize, (usize, Type)> = HashMap::default();
        for (i, (pid, ty)) in func.get_entry().get_parameters().enumerate() {
            if let Some((elem, n)) = expandable_array(ty) {
                if points_to.boundary_splittable_param(g, *pid) {
                    pmap.insert(i, (n, elem));
                }
            }
        }
        if !pmap.is_empty() {
            cand_params.insert(g, pmap);
        }

        let mut rmap: HashMap<usize, (usize, Type)> = HashMap::default();
        for (pos, ty) in func.get_returns().iter().enumerate() {
            if let Some((elem, n)) = expandable_array(ty) {
                if returns_all_splittable(func, points_to, g, pos) {
                    rmap.insert(pos, (n, elem));
                }
            }
        }
        if !rmap.is_empty() {
            cand_returns.insert(g, rmap);
        }
    }

    if cand_params.is_empty() && cand_returns.is_empty() {
        return Selection::default();
    }

    // Call-site agreement: every static call site is scanned. A parameter is dropped if any site's
    // argument would not itself peel (or its caller is un-analyzable); a callee with no static call
    // site is dropped entirely. Return expansion is structurally sound regardless of the caller,
    // but each site still gets a caller-side *profitability* check (`boundary_splittable_result`,
    // so the reconstructing `MkSeq` peels rather than widening the boundary) plus a defensive arity
    // check.
    let mut param_drop: HashSet<(FunctionId, usize)> = HashSet::default();
    let mut return_drop: HashSet<(FunctionId, usize)> = HashSet::default();
    let mut has_site: HashSet<FunctionId> = HashSet::default();
    for f in ssa.get_function_ids() {
        let analyzable = types.has_function(f);
        for (_, block) in ssa.get_function(f).get_blocks() {
            for instr in block.get_instructions() {
                let OpCode::Call {
                    function: CallTarget::Static(g),
                    args,
                    results,
                    ..
                } = instr
                else {
                    continue;
                };
                let pcand = cand_params.get(g);
                let rcand = cand_returns.get(g);
                if pcand.is_none() && rcand.is_none() {
                    continue;
                }
                has_site.insert(*g);
                if let Some(pmap) = pcand {
                    for &i in pmap.keys() {
                        let ok = analyzable
                            && i < args.len()
                            && points_to.boundary_splittable_arg(f, args[i]);
                        if !ok {
                            param_drop.insert((*g, i));
                        }
                    }
                }
                if let Some(rmap) = rcand {
                    for &pos in rmap.keys() {
                        // Profitability: the reconstructing `MkSeq` the caller emits must itself
                        // peel, i.e. the call-result array is only constant-indexed in the caller.
                        let ok = analyzable
                            && pos < results.len()
                            && points_to.boundary_splittable_result(f, results[pos]);
                        if !ok {
                            return_drop.insert((*g, pos));
                        }
                    }
                }
            }
        }
    }

    let mut sel = Selection::default();
    retain_kept(cand_params, &param_drop, &has_site, &mut sel.params);
    retain_kept(cand_returns, &return_drop, &has_site, &mut sel.returns);
    sel
}

// PHASE 1: PLAN CONSTRUCTION
// ================================================================================================

/// The whole-program expansion plan, with every fresh id pre-minted (so the mutation phase holds
/// `&mut` without needing to mint) in a deterministic structural order.
#[derive(Default)]
struct Plan {
    /// Pre-minted constant `u32` index values: `index_consts[k]` is the constant `k`. Minted here
    /// because constant interning needs `&HLSSA`, unavailable while `apply` holds `&mut`.
    index_consts: Vec<ValueId>,

    /// Per expanded callee, its parameter/return rewrites and per-`Return`-block cell ids.
    callees: HashMap<FunctionId, CalleePlan>,

    /// Per `(caller, block)`, the call-site rewrites in instruction order.
    callers: HashMap<(FunctionId, BlockId), Vec<CallRewrite>>,
}

struct CalleePlan {
    /// Expanded parameters, ascending by index.
    params: Vec<ParamExpand>,

    /// Expanded return positions, ascending.
    returns: Vec<ReturnExpand>,

    /// Per `Return` block, the fresh per-cell `ArrayGet` result ids: the outer `Vec` is aligned
    /// with `returns` (ascending position), each inner `Vec` of length that position's `N`.
    return_cells: HashMap<BlockId, Vec<Vec<ValueId>>>,
}

struct ParamExpand {
    index: usize,
    n: usize,
    elem: Type,

    /// Fresh ids for the `N` new by-value formals (cell order `0..N`).
    cells: Vec<ValueId>,
}

struct ReturnExpand {
    position: usize,
    n: usize,
    elem: Type,
}

struct CallRewrite {
    /// The expanded callee (matched against the call during the mutation sweep).
    callee: FunctionId,

    /// Expanded arguments, ascending by parameter index.
    args: Vec<ArgExpand>,

    /// Expanded results, ascending by return position.
    results: Vec<ResultExpand>,
}

struct ArgExpand {
    index: usize,
    n: usize,
    /// Fresh ids for the pre-call per-cell `ArrayGet` results (the new by-value arguments).
    cells: Vec<ValueId>,
}

struct ResultExpand {
    position: usize,
    n: usize,
    elem: Type,
    /// Fresh ids for the per-cell call results (reconstructed into the original result id after).
    cells: Vec<ValueId>,
}

fn build_plan(ssa: &HLSSA, sel: &Selection) -> Plan {
    let mut plan = Plan::default();

    // Pre-mint index constants up to the largest expansion.
    let max_n = sel
        .params
        .values()
        .chain(sel.returns.values())
        .flat_map(|m| m.values().map(|(n, _)| *n))
        .max()
        .unwrap_or(0);
    plan.index_consts = (0..max_n)
        .map(|k| ssa.add_const(Constant::U(32, k as u128)))
        .collect();

    // Callee plans (sorted callees; within each, params then return cells).
    for g in sel
        .params
        .keys()
        .chain(sel.returns.keys())
        .copied()
        .sorted_unstable()
        .dedup()
    {
        let func = ssa.get_function(g);

        let mut params: Vec<ParamExpand> = Vec::new();
        if let Some(pmap) = sel.params.get(&g) {
            for i in pmap.keys().copied().sorted_unstable() {
                let (n, elem) = pmap[&i].clone();
                let cells = ssa.fresh_values(n);
                params.push(ParamExpand {
                    index: i,
                    n,
                    elem,
                    cells,
                });
            }
        }

        let mut returns: Vec<ReturnExpand> = Vec::new();
        if let Some(rmap) = sel.returns.get(&g) {
            for pos in rmap.keys().copied().sorted_unstable() {
                let (n, elem) = rmap[&pos].clone();
                returns.push(ReturnExpand {
                    position: pos,
                    n,
                    elem,
                });
            }
        }

        let mut return_cells: HashMap<BlockId, Vec<Vec<ValueId>>> = HashMap::default();
        if !returns.is_empty() {
            let return_blocks = func
                .get_blocks()
                .filter(|(_, b)| matches!(b.get_terminator(), Some(Terminator::Return(_))))
                .map(|(bid, _)| *bid)
                .sorted_unstable();
            for bid in return_blocks {
                let per_pos: Vec<Vec<ValueId>> =
                    returns.iter().map(|re| ssa.fresh_values(re.n)).collect();
                return_cells.insert(bid, per_pos);
            }
        }

        plan.callees.insert(
            g,
            CalleePlan {
                params,
                returns,
                return_cells,
            },
        );
    }

    // Caller plans (sorted functions, blocks, instructions in order).
    for f in ssa.get_function_ids().sorted_unstable() {
        let func = ssa.get_function(f);
        for bid in func.get_blocks().map(|(bid, _)| *bid).sorted_unstable() {
            let mut rewrites: Vec<CallRewrite> = Vec::new();
            for instr in func.get_block(bid).get_instructions() {
                let OpCode::Call {
                    function: CallTarget::Static(g),
                    ..
                } = instr
                else {
                    continue;
                };
                let pmap = sel.params.get(g);
                let rmap = sel.returns.get(g);
                if pmap.is_none() && rmap.is_none() {
                    continue;
                }

                let mut args: Vec<ArgExpand> = Vec::new();
                if let Some(pmap) = pmap {
                    for i in pmap.keys().copied().sorted_unstable() {
                        let (n, _) = pmap[&i];
                        let cells = ssa.fresh_values(n);
                        args.push(ArgExpand { index: i, n, cells });
                    }
                }
                let mut results: Vec<ResultExpand> = Vec::new();
                if let Some(rmap) = rmap {
                    for pos in rmap.keys().copied().sorted_unstable() {
                        let (n, elem) = rmap[&pos].clone();
                        let cells = ssa.fresh_values(n);
                        results.push(ResultExpand {
                            position: pos,
                            n,
                            elem,
                            cells,
                        });
                    }
                }
                rewrites.push(CallRewrite {
                    callee: *g,
                    args,
                    results,
                });
            }
            if !rewrites.is_empty() {
                plan.callers.insert((f, bid), rewrites);
            }
        }
    }

    plan
}

// PHASE 2: APPLY TRANSFORM
// ================================================================================================

fn apply(ssa: &mut HLSSA, plan: &Plan) {
    for g in plan.callees.keys().copied().sorted_unstable() {
        rewrite_callee(
            ssa.get_function_mut(g),
            &plan.callees[&g],
            &plan.index_consts,
        );
    }

    for (f, bid) in plan.callers.keys().copied().sorted_unstable() {
        rewrite_caller_block(
            ssa.get_function_mut(f),
            bid,
            &plan.callers[&(f, bid)],
            &plan.index_consts,
        );
    }
}

/// Expand a callee's array parameters (into per-cell formals reconstructed at entry) and array
/// returns (into per-cell returns loaded before each `Return`).
fn rewrite_callee(func: &mut HLFunction, cp: &CalleePlan, index_consts: &[ValueId]) {
    // 1. Parameters: replace each array formal with its `N` cells, and prepend a reconstructing
    //    `MkSeq` whose result is the original parameter id (so every body use stays valid).
    if !cp.params.is_empty() {
        let entry_id = func.get_entry_id();
        let entry = func.get_block_mut(entry_id);
        let old_params = entry.take_parameters();
        let mut new_params = Vec::with_capacity(old_params.len());
        let mut reconstructs: Vec<LocatedOpCode> = Vec::with_capacity(cp.params.len());
        for (idx, (pid, ty)) in old_params.into_iter().enumerate() {
            if let Some(pe) = cp.params.iter().find(|p| p.index == idx) {
                for cell in &pe.cells {
                    new_params.push((*cell, pe.elem.clone()));
                }
                reconstructs.push(LocatedOpCode::without(OpCode::MkSeq {
                    result: pid,
                    elems: pe.cells.clone(),
                    seq_type: SequenceTargetType::Array(pe.n),
                    elem_type: pe.elem.clone(),
                }));
            } else {
                new_params.push((pid, ty));
            }
        }
        entry.put_parameters(new_params);

        let old_instrs = entry.take_instructions();
        let mut new_instrs = Vec::with_capacity(old_instrs.len() + reconstructs.len());
        new_instrs.extend(reconstructs);
        new_instrs.extend(old_instrs);
        entry.put_instructions(new_instrs);
    }

    // 2. Returns: at every `Return` block, load each cell of every expanded returned array and
    //    splice them into the return list; then widen the return-type signature.
    if !cp.returns.is_empty() {
        for bid in cp.return_cells.keys().copied().sorted_unstable() {
            let cells_per_pos = &cp.return_cells[&bid];
            let block = func.get_block_mut(bid);

            // Snapshot the returned values before mutating the block.
            let values = match block.get_terminator() {
                Some(Terminator::Return(vals)) => vals.clone(),
                _ => unreachable!("return_cells recorded only for Return blocks"),
            };

            // Emit per-cell `ArrayGet`s from each expanded returned array (defined earlier in the
            // block, so it is in scope here at the end of the body).
            for (pos_idx, re) in cp.returns.iter().enumerate() {
                let arr = values[re.position];
                let cells = &cells_per_pos[pos_idx];
                for k in 0..re.n {
                    block.push_instruction(OpCode::ArrayGet {
                        result: cells[k],
                        array: arr,
                        index: index_consts[k],
                    });
                }
            }

            // Rebuild the `Return` value list in declared order, splicing each expanded position.
            let mut new_vals: Vec<ValueId> = Vec::with_capacity(values.len());
            for (pos, v) in values.iter().enumerate() {
                if let Some(pos_idx) = cp.returns.iter().position(|re| re.position == pos) {
                    new_vals.extend(cells_per_pos[pos_idx].iter().copied());
                } else {
                    new_vals.push(*v);
                }
            }
            match block.get_terminator_mut() {
                Terminator::Return(vals) => *vals = new_vals,
                _ => unreachable!("snapshot proved this is a Return block"),
            }
        }

        // Widen the function's return-type vector: each expanded position becomes `N` cell types.
        let old_returns = func.take_returns();
        for (pos, ty) in old_returns.into_iter().enumerate() {
            if let Some(re) = cp.returns.iter().find(|r| r.position == pos) {
                for _ in 0..re.n {
                    func.add_return_type(re.elem.clone());
                }
            } else {
                func.add_return_type(ty);
            }
        }
    }
}

/// Rewrite the expanded-callee call sites in one caller block, in instruction order.
fn rewrite_caller_block(
    func: &mut HLFunction,
    bid: BlockId,
    rewrites: &[CallRewrite],
    index_consts: &[ValueId],
) {
    let block = func.get_block_mut(bid);
    let old = block.take_instructions();
    let mut new_instrs = Vec::with_capacity(old.len());
    let mut next = 0;
    for instr in old {
        if let OpCode::Call {
            function: CallTarget::Static(g),
            ..
        } = instr.as_ref()
        {
            // The recorded rewrites cover exactly the expanded-callee calls, in this order. Order
            // matching is sound because the callee-rewrite phase (run before any caller rewrite in
            // `apply`) only ever *inserts* `MkSeq`/`ArrayGet`, never a `Call`, so the `Call`
            // subsequence of every block is identical to the one `build_plan` recorded against.
            if next < rewrites.len() && rewrites[next].callee == *g {
                let (instr, location) = instr.take();
                let mut rewritten = Vec::new();
                apply_call_rewrite(instr, &rewrites[next], index_consts, &mut rewritten);
                new_instrs.extend(
                    rewritten
                        .into_iter()
                        .map(|instruction| LocatedOpCode::new(instruction, location.clone())),
                );
                next += 1;
                continue;
            }
        }
        new_instrs.push(instr);
    }
    debug_assert_eq!(
        next,
        rewrites.len(),
        "all recorded array-expansion call rewrites consumed"
    );
    block.put_instructions(new_instrs);
}

/// Emit pre-call `ArrayGet`s / the spliced `Call` / post-call `MkSeq`s for one expanded call.
fn apply_call_rewrite(
    instr: OpCode,
    cr: &CallRewrite,
    index_consts: &[ValueId],
    out: &mut Vec<OpCode>,
) {
    let OpCode::Call {
        results,
        function,
        args,
        unconstrained,
    } = instr
    else {
        unreachable!("apply_call_rewrite on a non-Call");
    };

    // 1. Read each expanded argument array's cells before the call.
    for ae in &cr.args {
        let arr = args[ae.index];
        for k in 0..ae.n {
            out.push(OpCode::ArrayGet {
                result: ae.cells[k],
                array: arr,
                index: index_consts[k],
            });
        }
    }

    // 2. Splice the argument list: each expanded array arg becomes its `N` cells.
    let mut new_args: Vec<ValueId> = Vec::with_capacity(args.len());
    for (i, a) in args.iter().enumerate() {
        if let Some(ae) = cr.args.iter().find(|e| e.index == i) {
            new_args.extend(ae.cells.iter().copied());
        } else {
            new_args.push(*a);
        }
    }

    // 3. Splice the result list: each expanded array result becomes its `N` cells; remember the
    //    reconstruction into the original result id.
    let mut new_results: Vec<ValueId> = Vec::with_capacity(results.len());
    let mut reconstructs: Vec<OpCode> = Vec::new();
    for (j, r) in results.iter().enumerate() {
        if let Some(re) = cr.results.iter().find(|e| e.position == j) {
            new_results.extend(re.cells.iter().copied());
            reconstructs.push(OpCode::MkSeq {
                result: *r,
                elems: re.cells.clone(),
                seq_type: SequenceTargetType::Array(re.n),
                elem_type: re.elem.clone(),
            });
        } else {
            new_results.push(*r);
        }
    }

    out.push(OpCode::Call {
        results: new_results,
        function,
        args: new_args,
        unconstrained,
    });
    out.extend(reconstructs);
}

// TESTS
// ================================================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::{
        analysis::{flow_analysis::FlowAnalysis, types::Types},
        passes::{
            array_sroa::ArraySroa,
            dead_code_elimination::{Config, DCE},
            mem2reg::Mem2Reg,
        },
        ssa::hlssa::builder::{HLEmitter, HLSSABuilder},
    };

    fn fr(n: u64) -> ark_bn254::Fr {
        ark_bn254::Fr::from(n)
    }

    /// Build the analyses on the current IR and run only this pass (mirrors the pass-manager).
    fn run_expansion(ssa: &mut HLSSA) {
        let flow = FlowAnalysis::run(ssa);
        let types = Types::new().run(ssa, &flow);
        let pt = PointsTo::run(ssa, &flow, &types);
        ArrayBoundaryExpansion::new().do_run(ssa, &types, &pt);
    }

    fn run_sroa(ssa: &mut HLSSA) {
        let flow = FlowAnalysis::run(ssa);
        let types = Types::new().run(ssa, &flow);
        let pt = PointsTo::run(ssa, &flow, &types);
        ArraySroa::new().do_run(ssa, &flow, &types, &pt);
    }

    fn run_mem2reg(ssa: &mut HLSSA) {
        let flow = FlowAnalysis::run(ssa);
        let types = Types::new().run(ssa, &flow);
        let pt = PointsTo::run(ssa, &flow, &types);
        Mem2Reg::new().do_run(ssa, &flow, &types, &pt);
    }

    fn run_dce(ssa: &mut HLSSA) {
        let flow = FlowAnalysis::run(ssa);
        DCE::new(Config::pre_r1c()).do_run(ssa, &flow);
    }

    /// `(MkSeq count, ArrayGet count)` in a function body.
    fn seq_get_counts(ssa: &HLSSA, fid: FunctionId) -> (usize, usize) {
        let mut t = (0, 0);
        for (_, block) in ssa.get_function(fid).get_blocks() {
            for inst in block.get_instructions() {
                match inst {
                    OpCode::MkSeq { .. } => t.0 += 1,
                    OpCode::ArrayGet { .. } => t.1 += 1,
                    _ => {}
                }
            }
        }
        t
    }

    fn param_types(ssa: &HLSSA, fid: FunctionId) -> Vec<Type> {
        ssa.get_function(fid).get_param_types()
    }

    fn return_count(ssa: &HLSSA, fid: FunctionId) -> usize {
        ssa.get_function(fid).get_returns().len()
    }

    /// `(arg count, result count)` of the first `Call` in a function, if any.
    fn first_call_arity(ssa: &HLSSA, fid: FunctionId) -> Option<(usize, usize)> {
        for (_, block) in ssa.get_function(fid).get_blocks() {
            for inst in block.get_instructions() {
                if let OpCode::Call { args, results, .. } = inst {
                    return Some((args.len(), results.len()));
                }
            }
        }
        None
    }

    /// A scalar array *parameter* that is only constant-indexed in the callee is expanded into
    /// per-cell formals, and the follow-up ArraySroa peels the reconstruction on both sides.
    #[test]
    fn param_array_expands_and_peels() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let main_id = ssa.get_unique_entrypoint_id();
        let sink;
        {
            let mut sb = HLSSABuilder::new(&mut ssa);
            sink = sb.ssa().add_function("sink".to_string());
            // sink(a: Array<Field,2>) -> Field: return a[0] + a[1]
            sb.modify_function(sink, |b| {
                b.function.add_return_type(Type::field());
                let entry = b.function.get_entry_id();
                let mut e = b.block(entry);
                let a = e.add_parameter(Type::field().array_of(2));
                let i0 = e.u_const(32, 0);
                let i1 = e.u_const(32, 1);
                let x = e.array_get(a, i0);
                let y = e.array_get(a, i1);
                let s = e.add(x, y);
                e.terminate_return(vec![s]);
            });
            // main(): arr = [3, 4]; return sink(arr)
            sb.modify_function(main_id, |b| {
                b.function.add_return_type(Type::field());
                let entry = b.function.get_entry_id();
                let mut e = b.block(entry);
                let c3 = e.field_const(fr(3));
                let c4 = e.field_const(fr(4));
                let arr = e.mk_seq(vec![c3, c4], SequenceTargetType::Array(2), Type::field());
                let r = e.call(sink, vec![arr], 1)[0];
                e.terminate_return(vec![r]);
            });
        }
        run_expansion(&mut ssa);

        // sink's array param became two scalar params; the call passes two scalar args.
        let pts = param_types(&ssa, sink);
        assert_eq!(pts.len(), 2, "array param expanded into 2 cells");
        assert!(
            pts.iter().all(|t| !t.is_array_or_slice()),
            "cells are scalars"
        );
        assert_eq!(
            first_call_arity(&ssa, main_id),
            Some((2, 1)),
            "call passes 2 args"
        );

        // The follow-up ArraySroa peels both reconstructions away.
        run_sroa(&mut ssa);
        assert_eq!(seq_get_counts(&ssa, sink), (0, 0), "sink fully peeled");
        assert_eq!(
            seq_get_counts(&ssa, main_id).0,
            0,
            "main's array literal peeled"
        );
    }

    /// A scalar array *return* that is built from constants in the callee is expanded into per-cell
    /// returns, and ArraySroa peels both sides.
    #[test]
    fn return_array_expands_and_peels() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let main_id = ssa.get_unique_entrypoint_id();
        let make;
        {
            let mut sb = HLSSABuilder::new(&mut ssa);
            make = sb.ssa().add_function("make".to_string());
            // make() -> Array<Field,2>: return [7, 9]
            sb.modify_function(make, |b| {
                b.function.add_return_type(Type::field().array_of(2));
                let entry = b.function.get_entry_id();
                let mut e = b.block(entry);
                let c7 = e.field_const(fr(7));
                let c9 = e.field_const(fr(9));
                let arr = e.mk_seq(vec![c7, c9], SequenceTargetType::Array(2), Type::field());
                e.terminate_return(vec![arr]);
            });
            // main(): a = make(); return a[0] + a[1]
            sb.modify_function(main_id, |b| {
                b.function.add_return_type(Type::field());
                let entry = b.function.get_entry_id();
                let mut e = b.block(entry);
                let a = e.call(make, vec![], 1)[0];
                let i0 = e.u_const(32, 0);
                let i1 = e.u_const(32, 1);
                let x = e.array_get(a, i0);
                let y = e.array_get(a, i1);
                let s = e.add(x, y);
                e.terminate_return(vec![s]);
            });
        }
        run_expansion(&mut ssa);

        assert_eq!(
            return_count(&ssa, make),
            2,
            "array return expanded into 2 cells"
        );
        assert_eq!(
            first_call_arity(&ssa, main_id),
            Some((0, 2)),
            "call yields 2 results"
        );

        run_sroa(&mut ssa);
        assert_eq!(seq_get_counts(&ssa, make), (0, 0), "make fully peeled");
        assert_eq!(seq_get_counts(&ssa, main_id), (0, 0), "main fully peeled");
    }

    /// A cell the callee never reads becomes a dead formal after the peel; the follow-up DCE drops
    /// it interprocedurally (the formal *and* the matching argument at the call site). The used
    /// cell is anchored by a global store so DCE keeps it (this toy's only other consumer, the
    /// entry-point return, is itself pruned at this stage).
    #[test]
    fn unused_cell_is_eliminated() {
        let mut ssa = HLSSA::with_main("main".to_string());
        ssa.set_global_types(vec![Type::field()]);
        let main_id = ssa.get_unique_entrypoint_id();
        let sink;
        {
            let mut sb = HLSSABuilder::new(&mut ssa);
            sink = sb.ssa().add_function("sink".to_string());
            // sink(a: Array<Field,2>) -> Field: return a[0]   (cell 1 unused)
            sb.modify_function(sink, |b| {
                b.function.add_return_type(Type::field());
                let entry = b.function.get_entry_id();
                let mut e = b.block(entry);
                let a = e.add_parameter(Type::field().array_of(2));
                let i0 = e.u_const(32, 0);
                let x = e.array_get(a, i0);
                e.terminate_return(vec![x]);
            });
            // main(): arr = [3, 4]; global[0] = sink(arr); return
            sb.modify_function(main_id, |b| {
                let entry = b.function.get_entry_id();
                let mut e = b.block(entry);
                let c3 = e.field_const(fr(3));
                let c4 = e.field_const(fr(4));
                let arr = e.mk_seq(vec![c3, c4], SequenceTargetType::Array(2), Type::field());
                let r = e.call(sink, vec![arr], 1)[0];
                e.init_global(0, r);
                e.terminate_return(vec![]);
            });
        }
        run_expansion(&mut ssa);
        run_sroa(&mut ssa);
        run_mem2reg(&mut ssa);
        run_dce(&mut ssa);

        assert_eq!(
            param_types(&ssa, sink).len(),
            1,
            "dead cell-1 formal removed by DCE"
        );
        assert_eq!(
            first_call_arity(&ssa, main_id),
            Some((1, 1)),
            "matching dead argument removed at the call site"
        );
    }

    /// Negative — a dynamically-indexed array parameter hits a hard collapse trigger, so it is not
    /// expanded.
    #[test]
    fn dynamic_index_param_is_retained() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let main_id = ssa.get_unique_entrypoint_id();
        let sink;
        {
            let mut sb = HLSSABuilder::new(&mut ssa);
            sink = sb.ssa().add_function("sink".to_string());
            // sink(a: Array<Field,2>, i: u32) -> Field: return a[i]
            sb.modify_function(sink, |b| {
                b.function.add_return_type(Type::field());
                let entry = b.function.get_entry_id();
                let mut e = b.block(entry);
                let a = e.add_parameter(Type::field().array_of(2));
                let i = e.add_parameter(Type::u(32));
                let x = e.array_get(a, i);
                e.terminate_return(vec![x]);
            });
            // main(): arr = [3, 4]; return sink(arr, 0)
            sb.modify_function(main_id, |b| {
                b.function.add_return_type(Type::field());
                let entry = b.function.get_entry_id();
                let mut e = b.block(entry);
                let c3 = e.field_const(fr(3));
                let c4 = e.field_const(fr(4));
                let arr = e.mk_seq(vec![c3, c4], SequenceTargetType::Array(2), Type::field());
                let i0 = e.u_const(32, 0);
                let r = e.call(sink, vec![arr, i0], 1)[0];
                e.terminate_return(vec![r]);
            });
        }
        run_expansion(&mut ssa);
        assert!(
            param_types(&ssa, sink)[0].is_array_or_slice(),
            "dynamically-indexed array param retained"
        );
    }

    /// Negative — an array passed onward to another (non-expandable) call is a second boundary on
    /// the group, so the parameter is not expanded (composition deferred).
    #[test]
    fn onward_passed_param_is_retained() {
        let mut ssa = HLSSA::with_main("main".to_string());
        ssa.set_global_types(vec![Type::field().array_of(2)]);
        let main_id = ssa.get_unique_entrypoint_id();
        let (g, sink);
        {
            let mut sb = HLSSABuilder::new(&mut ssa);
            g = sb.ssa().add_function("g".to_string());
            sink = sb.ssa().add_function("sink".to_string());
            // sink(a: Array<Field,2>): global[0] = a; return   (leaks ⇒ not itself expandable)
            sb.modify_function(sink, |b| {
                let entry = b.function.get_entry_id();
                let mut e = b.block(entry);
                let a = e.add_parameter(Type::field().array_of(2));
                e.init_global(0, a);
                e.terminate_return(vec![]);
            });
            // g(a: Array<Field,2>): sink(a); return   (forwards its array param onward)
            sb.modify_function(g, |b| {
                let entry = b.function.get_entry_id();
                let mut e = b.block(entry);
                let a = e.add_parameter(Type::field().array_of(2));
                e.call(sink, vec![a], 0);
                e.terminate_return(vec![]);
            });
            // main(): arr = [3, 4]; g(arr); return
            sb.modify_function(main_id, |b| {
                let entry = b.function.get_entry_id();
                let mut e = b.block(entry);
                let c3 = e.field_const(fr(3));
                let c4 = e.field_const(fr(4));
                let arr = e.mk_seq(vec![c3, c4], SequenceTargetType::Array(2), Type::field());
                e.call(g, vec![arr], 0);
                e.terminate_return(vec![]);
            });
        }
        run_expansion(&mut ssa);
        assert!(
            param_types(&ssa, g)[0].is_array_or_slice(),
            "onward-passed array param retained (second boundary)"
        );
    }

    /// The shape gate: only bare fixed-size arrays of bare scalars within the size bounds expand.
    #[test]
    fn expandable_array_shape_gate() {
        assert!(expandable_array(&Type::field().array_of(2)).is_some());
        assert!(expandable_array(&Type::u(32).array_of(4)).is_some());
        assert!(
            expandable_array(&Type::field().array_of(1)).is_none(),
            "below MIN_CELLS"
        );
        assert!(
            expandable_array(&Type::field().array_of(SIZE_CAP + 1)).is_none(),
            "above SIZE_CAP"
        );
        assert!(expandable_array(&Type::field()).is_none(), "not an array");
        assert!(
            expandable_array(&Type::field().array_of(2).array_of(2)).is_none(),
            "nested array cell"
        );
        assert!(
            expandable_array(&Type::field().ref_of().array_of(2)).is_none(),
            "ref cell deferred"
        );
    }
}
