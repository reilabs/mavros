//! Interprocedural reference-parameter promotion.
//!
//! Where `mem2reg` promotes a function's own non-escaping local `Alloc`s to SSA values, this pass
//! promotes memory passed *across call boundaries* through `Ref<T>` parameters. A promotable
//! `Ref<T>` parameter is turned into a **by-value `T` input** parameter plus (when the callee
//! writes through it) a **by-value `T` out** return; every call site reads the pointee before the
//! call and writes the result back after. That severs the boundary so a follow-up `mem2reg` then
//! promotes the now-local allocations on *both* sides.
//!
//! ## Reuse, not Re-Threading
//!
//! This pass does **not** reimplement value threading or phi placement, instead rewriting only the
//! boundary of procedures.
//!
//! - **Callee:** The `Ref<T>` parameter `i` of `g` becomes a fresh by-value `T` parameter; a fresh
//!   local `Alloc<T>` is materialized at entry and seeded from it (`Store a <- pv`). Every use of
//!   the old ref parameter is repointed at `a`; and for a *written* parameter the final value is
//!   loaded before each `Return` and appended to the return list (one appended return per written
//!   param, in ascending parameter order).
//! - **Caller:** At every `Call{Static(g)}` site, each promoted argument's pointee is `Load`ed
//!   before the call and passed by value, and (for a written param) the appended result is `Store`d
//!   back into the original ref after the call.
//!
//! A trailing `mem2reg` then promotes the materialized callee alloc and the now-local caller alloc
//! wherever possible.
//!
//! This reuse is sound **only because** promotion requires (condition C3 below) that the callee's
//! ref parameter is already used exclusively as a `Load`/`Store` pointer — which guarantees the
//! materialized alloc satisfies `mem2reg`'s own promotability, so the follow-up pass is guaranteed
//! to clean it up.
//!
//! ## Promotability of Parameters
//!
//! A parameter `i` of function `g` can be promoted if all of the following conditions hold.
//!
//! - **Owned & Enumerable:** `g` is not an entry point, not the globals init/deinit function, not
//!   reachable via a dynamic call, and has at least one static call site, so *every* call site is a
//!   `Call{Static(g)}` this pass can rewrite.
//! - **C1, Pointee Shape:** `Ref<T>` with `T` a scalar (`Field`/`u*`/`i*`).
//! - **C2, Non-Leakage:** The parameter's pointee does not escape inside `g`.
//! - **C3, Callee Shape:** Inside `g`, the ref is used only as a `Load`/`Store` pointer, and every
//!   such access points unambiguously to it.
//! - **C4, Call-Site Agreement:** At every site, `args[i]` is reached by exactly one non-escaping
//!   object `o`, and `o` does not alias any other argument — conservatively: each other reference
//!   argument whose pointee contains no pointers must not point to `o` (a root-level points-to
//!   check suffices), and any argument whose type itself contains pointers (a `Ref<Ref<..>>`/
//!   array-of-ref that could transitively reach `o`) blocks promotion. The decision is
//!   all-or-nothing per `(g, i)`: if any site fails, `i` is not promoted.
//!
//! The transform fires only where these prove it safe, so imprecision only ever blocks a promotion,
//! never produces an unsound one.
//!
//! ## Deferred Improvements
//!
//! These are sound-but-suboptimal cases; none represent correctness gaps, as the imprecision only
//! ever blocks (or fails to skip) a promotion, never produces an unsound one. Most are unhandled
//! cases blocked by a condition above; the last is a missing cost guard on a promotion that *does*
//! fire.
//!
//! - **Aggregate Pointees** An aggregate passed by value across a call boundary is itself an
//!   ArraySroa collapse trigger, so it can never be re-peeled; promoting it trades one ref-pass for
//!   an N-element value copy plus an N-element return (a net loss). Handling them well needs
//!   per-element boundary *flattening* (à la `elide_tuples` for tuples), not by-value aggregate
//!   passing. Blocked by C1.
//! - **Ref-Bearing Pointees** `Ref<T>` where `T` itself contains pointers (`Ref<Ref<..>>`,
//!   array-of-ref); promoting these would mint ref-typed phis the analysis never saw. Blocked by
//!   C1.
//! - **Onward-Passed Parameters** A ref forwarded to a sub-callee (used as a call argument, not
//!   just a `Load`/`Store` pointer); handling it needs a fixpoint over the call graph. Blocked by
//!   C3.
//! - **Out-Only ABI** A parameter the callee only writes still receives a (dead) by-value input; an
//!   out-only ABI (drop the input, keep only the return) would avoid that movement.
//! - **Redundant Points-to Solves** `preserves() = []` plus the trailing cleanup `mem2reg` makes
//!   `pre_wti` re-run the whole-program Andersen solve several times; an incremental update or a
//!   "rewrote nothing" fast-path could skip the extra passes.
//! - **Caller-Side Promotability:** C4 proves the caller object `o` is a non-escaping singleton,
//!   but *not* that `o` is itself `mem2reg`-clean in the caller (the way C3 guarantees the
//!   materialized alloc is clean in the callee). If `o` does not scalarize there, the inserted
//!   pre-call `Load` / post-call `Store` survive; and since the deref moves from one site in the
//!   callee to *every* call site, a callee with many callers can become a net static-size
//!   **increase**. Promotion stays sound — this is a missing cost guard: mirror
//!   `placeholder_is_clean` for `o` in the caller (or gate on the call-site count) before
//!   committing. Empirically 0 corpus regressions so far, so it is a latent risk, not an observed
//!   one.

use crate::{
    collections::{HashMap, HashSet},
    compiler::{
        analysis::{
            points_to::{PointerUse, PointsTo, object::AbstractObject},
            types::{FunctionTypeInfo, TypeInfo},
        },
        pass_manager::{Analysis, AnalysisId, AnalysisStore, Pass},
        passes::fix_double_jumps::{ReplaceScope, ValueReplacements},
        ssa::{
            BlockId, FunctionId, Terminator, ValueId,
            hlssa::{
                CallTarget, Constant, HLFunction, HLSSA, LocatedOpCode, OpCode, Type, TypeExpr,
            },
        },
    },
};

// ARG PROMOTION PASS
// ================================================================================================

pub struct ArgPromotion {}

impl ArgPromotion {
    pub fn new() -> Self {
        ArgPromotion {}
    }
}

impl Pass for ArgPromotion {
    fn name(&self) -> &'static str {
        "arg_promotion"
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

impl ArgPromotion {
    pub fn do_run(&self, ssa: &mut HLSSA, types: &TypeInfo, points_to: &PointsTo) {
        let selected = select_params(ssa, types, points_to);
        if selected.is_empty() {
            return;
        }
        let plan = build_plan(ssa, &selected);
        apply(ssa, &plan);
    }
}

// PLAN
// ================================================================================================

/// The whole-program promotion plan, with every fresh `ValueId` pre-minted (so the mutation phase
/// holds `&mut` without needing to mint).
#[derive(Default)]
struct Plan {
    /// Per promoted callee, its parameter rewrites and per-`Return`-block out-value loads.
    callees: HashMap<FunctionId, CalleePlan>,

    /// Per `(caller, block)`, the call-site rewrites in instruction order (one per `Call` to a
    /// promoted callee).
    callers: HashMap<(FunctionId, BlockId), Vec<CallRewrite>>,
}

struct CalleePlan {
    /// Promoted parameters, ascending by index.
    params: Vec<ParamPlan>,

    /// For each `Return` block, the fresh `Load` result ids to append — one per *written* (in/out)
    /// parameter, in ascending parameter order.
    return_loads: HashMap<BlockId, Vec<ValueId>>,
}

struct ParamPlan {
    /// The entry-block parameter position (= the points-to parameter index).
    index: usize,

    /// The pointee type `T` (the new by-value parameter / return / alloc-element type).
    pointee: Type,

    /// Whether the callee writes through the parameter (⇒ a by-value out-return is appended).
    in_out: bool,

    /// Fresh id for the new by-value parameter.
    new_param: ValueId,

    /// Fresh id for the materialized local alloc.
    alloc: ValueId,
}

struct CallRewrite {
    /// The promoted callee (used to match the rewrite against the call during the mutation sweep).
    callee: FunctionId,

    /// One entry per promoted parameter of `callee`, ascending by index.
    args: Vec<ArgRewrite>,
}

struct ArgRewrite {
    /// The promoted parameter position.
    index: usize,

    /// Fresh id for the value loaded before the call (the new by-value argument).
    load_in: ValueId,

    /// Fresh id for the out-value received after the call (`Some` iff the parameter is in/out).
    store_out: Option<ValueId>,
}

// PHASE 0: PARAMETER SELECTION
// ================================================================================================

/// Select the promotable `(callee, parameter index) -> in_out` set.
fn select_params(
    ssa: &HLSSA,
    types: &TypeInfo,
    points_to: &PointsTo,
) -> HashMap<FunctionId, HashMap<usize, bool>> {
    let entry_points: HashSet<FunctionId> = ssa.get_entry_points().iter().copied().collect();
    let globals_init = ssa.get_globals_init_fn();
    let globals_deinit = ssa.get_globals_deinit_fn();

    // A function whose pointer is taken cannot have all its call sites enumerated (and could be
    // reached via a dynamic call), so it is never promoted.
    let mut address_taken: HashSet<FunctionId> = HashSet::default();
    ssa.for_each_const(|_, cv| {
        if let Constant::FnPtr(g) = &**cv {
            address_taken.insert(*g);
        }
    });

    // Candidate `(g, i)` by the callee-local conditions C1–C3 + the out-direction decision.
    let mut candidates: HashMap<FunctionId, HashMap<usize, bool>> = HashMap::default();
    for g in ssa.get_function_ids() {
        if !types.has_function(g)
            || entry_points.contains(&g)
            || address_taken.contains(&g)
            || Some(g) == globals_init
            || Some(g) == globals_deinit
        {
            continue;
        }
        let func = ssa.get_function(g);

        // C1/C2-surviving candidates of `g`: (param index, its pointee placeholder object).
        let mut cands: Vec<(usize, AbstractObject)> = Vec::new();
        for (i, (_, ty)) in func.get_entry().get_parameters().enumerate() {
            let ref_ty = ty.peel_witness();
            if !ref_ty.is_ref() {
                continue;
            }

            // C1: the pointee is a scalar (`Field`/`u*`/`i*`). Aggregate (array/slice) and ref
            // pointees are deferred: an aggregate passed *by value* across a call boundary is
            // itself an ArraySroa collapse trigger, so it can never be re-peeled — promoting it
            // trades one ref-pass for an N-element value copy + N-element return, a net bytecode
            // loss.
            if !matches!(
                ref_ty.get_pointed().peel_witness().expr,
                TypeExpr::Field | TypeExpr::U(_) | TypeExpr::I(_)
            ) {
                continue;
            }

            // For a `Ref<T>` the placeholder's only ref-level path is empty (descent stops at the
            // first `Ref`), so this is exactly the parameter's pointee object.
            let placeholder = AbstractObject::Placeholder(g, i, Vec::new());

            // C2: the callee does not let the pointee escape (== `leaks_param`).
            if points_to.escapes(&placeholder) {
                continue;
            }

            cands.push((i, placeholder));
        }

        // C3 (clean) and the out-direction (writes) for every surviving candidate, computed in a
        // single scan of the body rather than one scan per parameter.
        let mut params: HashMap<usize, bool> = HashMap::default();
        if !cands.is_empty() {
            let has_return = func
                .get_blocks()
                .any(|(_, b)| matches!(b.get_terminator(), Some(Terminator::Return(_))));
            for ((i, _), (clean, writes)) in cands
                .iter()
                .zip(classify_params(points_to, g, func, &cands))
            {
                // C3: inside `g`, the ref is used only as a `Load`/`Store` pointer, unambiguously.
                if !clean {
                    continue;
                }

                // Out-direction: append a by-value return only if the callee writes the pointee
                // *and* can return (a diverging callee never resumes the caller, so its writes are
                // unobservable — promote in-only).
                params.insert(*i, has_return && writes);
            }
        }
        if !params.is_empty() {
            candidates.insert(g, params);
        }
    }

    // Nothing is even promotable-shaped: skip the whole-program call-site scan entirely (every
    // `candidates.get(g)` below would miss, leaving `selected` empty regardless).
    if candidates.is_empty() {
        return HashMap::default();
    }

    // C4: validate every static call site. A parameter is dropped if any site disagrees, and a
    // candidate with no static call site at all is dropped entirely.
    let mut to_remove: HashSet<(FunctionId, usize)> = HashSet::default();
    let mut has_site: HashSet<FunctionId> = HashSet::default();
    for f in ssa.get_function_ids() {
        let fti = types.has_function(f).then(|| types.get_function(f));
        for (_, block) in ssa.get_function(f).get_blocks() {
            for instr in block.get_instructions() {
                let OpCode::Call {
                    function: CallTarget::Static(g),
                    args,
                    ..
                } = instr
                else {
                    continue;
                };
                let Some(params) = candidates.get(g) else {
                    continue;
                };
                has_site.insert(*g);
                for &i in params.keys() {
                    let ok = match fti {
                        Some(fti) => call_site_ok(points_to, fti, f, args, i),
                        // An un-analyzed caller has no usable points-to; be conservative.
                        None => false,
                    };
                    if !ok {
                        to_remove.insert((*g, i));
                    }
                }
            }
        }
    }

    let mut selected: HashMap<FunctionId, HashMap<usize, bool>> = HashMap::default();
    for (g, params) in candidates {
        if !has_site.contains(&g) {
            continue;
        }
        let kept: HashMap<usize, bool> = params
            .into_iter()
            .filter(|(i, _)| !to_remove.contains(&(g, *i)))
            .collect();
        if !kept.is_empty() {
            selected.insert(g, kept);
        }
    }
    selected
}

/// Classify every C1/C2-surviving candidate placeholder of `g` in a single scan of the body,
/// returning one `(clean, writes)` pair per entry of `cands`, in the same order.
///
/// This fuses what `placeholder_is_clean` (C3) and `writes_through` (out-direction) to avoid a
/// once-per-parameter check. Each candidate is matched against each instruction's already-fetched
/// points-to set, so a function-body value is fetched once rather than once per candidate.
///
/// - `clean` mirrors `mem2reg::promotable_allocs`'s conditions 3 & 4 keyed on the placeholder: the
///   ref is used only as a `Load`/`Store` pointer, and every such access points unambiguously to
///   it.
/// - `writes` records whether `g` ever stores through the placeholder; for a `clean` candidate C3
///   guarantees any such store points to exactly it. (`writes` is meaningless for a non-`clean`
///   candidate, which the caller drops anyway.)
fn classify_params(
    points_to: &PointsTo,
    g: FunctionId,
    func: &HLFunction,
    cands: &[(usize, AbstractObject)],
) -> Vec<(bool, bool)> {
    let mut clean = vec![true; cands.len()];
    let mut writes = vec![false; cands.len()];

    // Shares `mem2reg`'s pointer-use classifier ([`PointsTo::classify_pointer_uses`]), keyed on the
    // candidate placeholders rather than local allocs. An ambiguous deref (`Deref`) or a
    // non-pointer use (`Consume`) of a value that may reach a candidate makes it un-`clean` —
    // establishing that the ref is used only as an unambiguous `Load`/`Store` pointer (C3); a
    // `Write` through it records the out-direction. For a `clean` candidate, C3 then guarantees any
    // such write points to exactly it. (`writes` is meaningless for a non-`clean` candidate, which
    // the caller drops.)
    points_to.classify_pointer_uses(g, func, |kind, pts| match kind {
        PointerUse::Deref | PointerUse::Consume => {
            for (k, (_, p)) in cands.iter().enumerate() {
                if pts.contains(p) {
                    clean[k] = false;
                }
            }
        }
        PointerUse::Write => {
            for (k, (_, p)) in cands.iter().enumerate() {
                if pts.contains(p) {
                    writes[k] = true;
                }
            }
        }
    });

    clean.into_iter().zip(writes).collect()
}

/// C4 at one call site: `args[i]` must be reached by a single non-escaping object `o`, and no other
/// argument may alias `o`.
fn call_site_ok(
    points_to: &PointsTo,
    fti: &FunctionTypeInfo,
    f: FunctionId,
    args: &[ValueId],
    i: usize,
) -> bool {
    if i >= args.len() || !points_to.is_singleton_reached(f, args[i]) {
        return false;
    }
    let o = points_to
        .points_to(f, args[i])
        .iter()
        .next()
        .expect("is_singleton_reached guarantees one object");
    for (j, &arg_j) in args.iter().enumerate() {
        if j == i {
            continue;
        }
        let jty = fti.get_value_type(arg_j).peel_witness();
        if jty.is_ref() && !jty.get_pointed().contains_ptrs() {
            // A reference whose pointee holds no pointers can alias `o` only at its root level: its
            // pointee object can never be `o`, and carries nothing that transitively reaches it.
            if points_to.points_to(f, arg_j).contains(o) {
                return false;
            }
        } else if jty.contains_ptrs() {
            // A `Ref<Ref<..>>` / array-of-ref could transitively reach `o`, which a root-level
            // points-to query cannot rule out — conservatively block.
            return false;
        }
        // A plain non-pointer value cannot alias `o`.
    }
    true
}

// PHASE 0: PLAN CONSTRUCTION
// ================================================================================================

/// Build the full plan from the selection, minting every fresh id in a deterministic structural
/// order (functions ascending, parameters ascending, blocks ascending, instructions in order).
fn build_plan(ssa: &HLSSA, selected: &HashMap<FunctionId, HashMap<usize, bool>>) -> Plan {
    let mut plan = Plan::default();

    let mut callee_ids: Vec<FunctionId> = selected.keys().copied().collect();
    callee_ids.sort();
    for g in callee_ids {
        let params = &selected[&g];
        let func = ssa.get_function(g);
        let param_types: Vec<Type> = func
            .get_entry()
            .get_parameters()
            .map(|(_, ty)| ty.clone())
            .collect();

        let mut idxs: Vec<usize> = params.keys().copied().collect();
        idxs.sort_unstable();
        let mut param_plans: Vec<ParamPlan> = Vec::with_capacity(idxs.len());
        for i in idxs {
            param_plans.push(ParamPlan {
                index: i,
                pointee: param_types[i].peel_witness().get_pointed(),
                in_out: params[&i],
                new_param: ssa.fresh_value(),
                alloc: ssa.fresh_value(),
            });
        }
        let in_out_count = param_plans.iter().filter(|p| p.in_out).count();

        let mut return_blocks: Vec<BlockId> = func
            .get_blocks()
            .filter(|(_, b)| matches!(b.get_terminator(), Some(Terminator::Return(_))))
            .map(|(bid, _)| *bid)
            .collect();
        return_blocks.sort();
        let mut return_loads: HashMap<BlockId, Vec<ValueId>> = HashMap::default();
        for bid in return_blocks {
            let lvs: Vec<ValueId> = (0..in_out_count).map(|_| ssa.fresh_value()).collect();
            return_loads.insert(bid, lvs);
        }

        plan.callees.insert(
            g,
            CalleePlan {
                params: param_plans,
                return_loads,
            },
        );
    }

    let mut caller_ids: Vec<FunctionId> = ssa.get_function_ids().collect();
    caller_ids.sort();
    for f in caller_ids {
        let func = ssa.get_function(f);
        let mut block_ids: Vec<BlockId> = func.get_blocks().map(|(bid, _)| *bid).collect();
        block_ids.sort();
        for bid in block_ids {
            let mut rewrites: Vec<CallRewrite> = Vec::new();
            for instr in func.get_block(bid).get_instructions() {
                let OpCode::Call {
                    function: CallTarget::Static(g),
                    ..
                } = instr
                else {
                    continue;
                };
                let Some(params) = selected.get(g) else {
                    continue;
                };
                let mut idxs: Vec<usize> = params.keys().copied().collect();
                idxs.sort_unstable();
                let args = idxs
                    .into_iter()
                    .map(|i| ArgRewrite {
                        index: i,
                        load_in: ssa.fresh_value(),
                        store_out: params[&i].then(|| ssa.fresh_value()),
                    })
                    .collect();
                rewrites.push(CallRewrite { callee: *g, args });
            }
            if !rewrites.is_empty() {
                plan.callers.insert((f, bid), rewrites);
            }
        }
    }

    plan
}

// PHASES 1, 2: APPLY TRANSFORM
// ================================================================================================

fn apply(ssa: &mut HLSSA, plan: &Plan) {
    // Phase 1: rewrite each promoted callee's signature + body.
    let mut callee_ids: Vec<FunctionId> = plan.callees.keys().copied().collect();
    callee_ids.sort();
    for g in callee_ids {
        rewrite_callee(ssa.get_function_mut(g), &plan.callees[&g]);
    }

    // Phase 2: rewrite every recorded call site.
    let mut caller_keys: Vec<(FunctionId, BlockId)> = plan.callers.keys().copied().collect();
    caller_keys.sort();
    for (f, bid) in caller_keys {
        rewrite_caller_block(ssa.get_function_mut(f), bid, &plan.callers[&(f, bid)]);
    }
}

/// Phase 1: turn each promoted `Ref<T>` parameter of `func` into a by-value in/out parameter.
fn rewrite_callee(func: &mut HLFunction, cp: &CalleePlan) {
    let entry_id = func.get_entry_id();

    // 1. Replace each promoted ref parameter with a by-value parameter; remember old id -> alloc.
    let mut repl = ValueReplacements::new();
    {
        let entry = func.get_block_mut(entry_id);
        let old_params = entry.take_parameters();
        let mut new_params = Vec::with_capacity(old_params.len());
        for (idx, (pid, ty)) in old_params.into_iter().enumerate() {
            if let Some(pp) = cp.params.iter().find(|p| p.index == idx) {
                repl.insert(pid, pp.alloc);
                new_params.push((pp.new_param, pp.pointee.clone()));
            } else {
                new_params.push((pid, ty));
            }
        }
        entry.put_parameters(new_params);

        // 2. Materialize one local alloc per promoted parameter, seeded from the value-parameter.
        let old_instrs = entry.take_instructions();
        let mut new_instrs = Vec::with_capacity(old_instrs.len() + cp.params.len() * 2);
        for pp in &cp.params {
            // The alloc carries its initial value, so seed it directly from the by-value parameter
            // with no follow-up store needed.
            new_instrs.push(LocatedOpCode::without(OpCode::Alloc {
                result: pp.alloc,
                value: pp.new_param,
            }));
        }
        new_instrs.extend(old_instrs);
        entry.put_instructions(new_instrs);
    }

    // 3. Repoint every use of the old ref parameters at their materialized allocs.
    repl.apply_to_function(func, ReplaceScope::Operands);

    // 4. For written parameters, load the final value before each Return and append it (by-value
    //    out). C3 guarantees the alloc is `mem2reg`-promotable, so the follow-up pass scalarizes
    //    it.
    let in_out: Vec<&ParamPlan> = cp.params.iter().filter(|p| p.in_out).collect();
    if in_out.is_empty() {
        return;
    }
    let mut return_blocks: Vec<BlockId> = cp.return_loads.keys().copied().collect();
    return_blocks.sort();
    for bid in return_blocks {
        let lvs = &cp.return_loads[&bid];
        let block = func.get_block_mut(bid);
        for (k, pp) in in_out.iter().enumerate() {
            block.push_instruction(OpCode::Load {
                result: lvs[k],
                ptr: pp.alloc,
            });
        }
        match block.get_terminator_mut() {
            Terminator::Return(vals) => vals.extend(lvs.iter().copied()),
            _ => unreachable!("return_loads recorded only for Return blocks"),
        }
    }
    for pp in in_out {
        func.add_return_type(pp.pointee.clone());
    }
}

/// Phase 2: rewrite the promoted-callee call sites in one caller block, in instruction order.
fn rewrite_caller_block(func: &mut HLFunction, bid: BlockId, rewrites: &[CallRewrite]) {
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
            // The recorded rewrites cover exactly the promoted-callee calls, in this order.
            if next < rewrites.len() && rewrites[next].callee == *g {
                let (instr, location) = instr.take();
                let mut rewritten = Vec::new();
                apply_call_rewrite(instr, &rewrites[next], &mut rewritten);
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
    debug_assert_eq!(next, rewrites.len(), "all recorded call rewrites consumed");
    block.put_instructions(new_instrs);
}

/// Emit `Load`-before / (modified) `Call` / `Store`-after for one promoted-callee call.
fn apply_call_rewrite(instr: OpCode, cr: &CallRewrite, out: &mut Vec<OpCode>) {
    let OpCode::Call {
        results,
        function,
        args,
        unconstrained,
    } = instr
    else {
        unreachable!("apply_call_rewrite on a non-Call");
    };

    let mut new_args = args.clone();
    let mut new_results = results;

    // Read each promoted pointee before the call; pass it by value.
    for ar in &cr.args {
        out.push(OpCode::Load {
            result: ar.load_in,
            ptr: args[ar.index],
        });
        new_args[ar.index] = ar.load_in;
    }
    // Receive each written-back pointee as an extra result (ascending parameter order, matching the
    // callee's appended returns).
    for ar in &cr.args {
        if let Some(vout) = ar.store_out {
            new_results.push(vout);
        }
    }
    out.push(OpCode::Call {
        results: new_results,
        function,
        args: new_args,
        unconstrained,
    });
    // Write each returned value back into the original ref after the call.
    for ar in &cr.args {
        if let Some(vout) = ar.store_out {
            out.push(OpCode::Store {
                ptr: args[ar.index],
                value: vout,
            });
        }
    }
}

// TESTS
// ================================================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::{
        analysis::{flow_analysis::FlowAnalysis, types::Types},
        passes::mem2reg::Mem2Reg,
        ssa::hlssa::{
            SequenceTargetType,
            builder::{HLEmitter, HLSSABuilder},
        },
        util::test::{falloc, fr},
    };

    /// Build the analyses on the current IR and run only ArgPromotion (mirrors the pass-manager).
    fn run_argpromo(ssa: &mut HLSSA) {
        let flow = FlowAnalysis::run(ssa);
        let types = Types::new().run(ssa, &flow);
        let pt = PointsTo::run(ssa, &flow, &types);
        ArgPromotion::new().do_run(ssa, &types, &pt);
    }

    /// Run a fresh Mem2Reg (analyses recomputed, as `preserves()=[]` forces in the pipeline).
    fn run_mem2reg(ssa: &mut HLSSA) {
        let flow = FlowAnalysis::run(ssa);
        let types = Types::new().run(ssa, &flow);
        let pt = PointsTo::run(ssa, &flow, &types);
        Mem2Reg::new().do_run(ssa, &flow, &types, &pt);
    }

    /// `(allocs, stores, loads, calls)` remaining in a function body.
    fn op_counts(ssa: &HLSSA, fid: FunctionId) -> (usize, usize, usize, usize) {
        let mut t = (0, 0, 0, 0);
        for (_, block) in ssa.get_function(fid).get_blocks() {
            for inst in block.get_instructions() {
                match inst {
                    OpCode::Alloc { .. } => t.0 += 1,
                    OpCode::Store { .. } => t.1 += 1,
                    OpCode::Load { .. } => t.2 += 1,
                    OpCode::Call { .. } => t.3 += 1,
                    _ => {}
                }
            }
        }
        t
    }

    /// `(param types, return count)` — the function signature.
    fn sig(ssa: &HLSSA, fid: FunctionId) -> (Vec<Type>, usize) {
        let f = ssa.get_function(fid);
        (f.get_param_types(), f.get_returns().len())
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

    /// A read+write `Ref<Field>` parameter is promoted to a by-value in/out parameter, and the
    /// follow-up Mem2Reg scalarizes both the callee alloc and the caller local.
    #[test]
    fn scalar_param_promotes_in_and_out() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let main_id = ssa.get_unique_entrypoint_id();
        let g;
        {
            let mut sb = HLSSABuilder::new(&mut ssa);
            g = sb.ssa().add_function("g".to_string());
            // g(p: Ref<Field>): *p = *p + 1; return
            sb.modify_function(g, |b| {
                let entry = b.function.get_entry_id();
                let mut e = b.block(entry);
                let p = e.add_parameter(Type::field().ref_of());
                let v = e.load(p);
                let one = e.field_const(fr(1));
                let w = e.add(v, one);
                e.store(p, w);
                e.terminate_return(vec![]);
            });
            // main(): a = alloc; *a = 5; g(a); return *a
            sb.modify_function(main_id, |b| {
                b.function.add_return_type(Type::field());
                let entry = b.function.get_entry_id();
                let mut e = b.block(entry);
                let a = falloc(&mut e);
                let c = e.field_const(fr(5));
                e.store(a, c);
                e.call(g, vec![a], 0);
                let r = e.load(a);
                e.terminate_return(vec![r]);
            });
        }
        run_argpromo(&mut ssa);
        let (gp, gr) = sig(&ssa, g);
        assert_eq!(gp.len(), 1);
        assert!(!gp[0].is_ref(), "g's ref param became by-value");
        assert_eq!(gr, 1, "an out-return was appended");
        assert_eq!(
            first_call_arity(&ssa, main_id),
            Some((1, 1)),
            "call keeps its arg, gains one out-result"
        );
        // The follow-up Mem2Reg promotes both the caller local and the callee alloc.
        run_mem2reg(&mut ssa);
        assert_eq!(
            op_counts(&ssa, main_id),
            (0, 0, 0, 1),
            "caller local scalarized"
        );
        assert_eq!(op_counts(&ssa, g).0, 0, "callee alloc scalarized");
    }

    /// Negative — an array (aggregate) pointee is deferred: passing it by value across the call
    /// boundary cannot be re-peeled, so the `Ref<Array<Field, N>>` parameter is retained.
    #[test]
    fn array_pointee_is_deferred() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let main_id = ssa.get_unique_entrypoint_id();
        let arr_ty = Type::field().array_of(3);
        let g;
        {
            let mut sb = HLSSABuilder::new(&mut ssa);
            g = sb.ssa().add_function("g".to_string());
            // g(p: Ref<Array<Field,3>>): let a = *p; *p = a[0 := 7]; return
            sb.modify_function(g, {
                let arr_ty = arr_ty.clone();
                move |b| {
                    let entry = b.function.get_entry_id();
                    let mut e = b.block(entry);
                    let p = e.add_parameter(arr_ty.ref_of());
                    let arr = e.load(p);
                    let i0 = e.u_const(32, 0);
                    let x = e.field_const(fr(7));
                    let updated = e.array_set(arr, i0, x);
                    e.store(p, updated);
                    e.terminate_return(vec![]);
                }
            });
            // main(): a = alloc; *a = [1,2,3]; g(a); return
            sb.modify_function(main_id, |b| {
                let entry = b.function.get_entry_id();
                let mut e = b.block(entry);
                let c1 = e.field_const(fr(1));
                let c2 = e.field_const(fr(2));
                let c3 = e.field_const(fr(3));
                let arr = e.mk_seq(
                    vec![c1, c2, c3],
                    SequenceTargetType::Array(3),
                    Type::field(),
                );
                let a = e.alloc(arr); // Ref<Array<Field,3>>, seeded with arr (store folded into the alloc)
                e.call(g, vec![a], 0);
                e.terminate_return(vec![]);
            });
        }
        run_argpromo(&mut ssa);
        let (gp, gr) = sig(&ssa, g);
        assert!(
            gp[0].is_ref(),
            "array ref param retained (aggregate pointee deferred)"
        );
        assert_eq!(gr, 0, "no out-return appended");
    }

    /// A read-only `Ref<Field>` parameter promotes in-only: a by-value in parameter, no out-return,
    /// no caller store-back.
    #[test]
    fn read_only_param_promotes_in_only() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let main_id = ssa.get_unique_entrypoint_id();
        let g;
        {
            let mut sb = HLSSABuilder::new(&mut ssa);
            g = sb.ssa().add_function("g".to_string());
            // g(p: Ref<Field>) -> Field: return *p
            sb.modify_function(g, |b| {
                b.function.add_return_type(Type::field());
                let entry = b.function.get_entry_id();
                let mut e = b.block(entry);
                let p = e.add_parameter(Type::field().ref_of());
                let v = e.load(p);
                e.terminate_return(vec![v]);
            });
            // main(): a = alloc; *a = 5; return g(a)
            sb.modify_function(main_id, |b| {
                b.function.add_return_type(Type::field());
                let entry = b.function.get_entry_id();
                let mut e = b.block(entry);
                let a = falloc(&mut e);
                let c = e.field_const(fr(5));
                e.store(a, c);
                let r = e.call(g, vec![a], 1)[0];
                e.terminate_return(vec![r]);
            });
        }
        run_argpromo(&mut ssa);
        let (gp, gr) = sig(&ssa, g);
        assert!(!gp[0].is_ref(), "ref param became by-value");
        assert_eq!(gr, 1, "no out-return appended (read-only)");
        assert_eq!(
            first_call_arity(&ssa, main_id),
            Some((1, 1)),
            "call result count unchanged (no write-back)"
        );
    }

    /// Two read+write parameters both promote; the appended returns/results align by parameter
    /// order.
    #[test]
    fn multi_param_promotes() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let main_id = ssa.get_unique_entrypoint_id();
        let g;
        {
            let mut sb = HLSSABuilder::new(&mut ssa);
            g = sb.ssa().add_function("g".to_string());
            // g(p0: Ref<F>, p1: Ref<F>): *p0 = *p0 + *p1; *p1 = 0; return
            sb.modify_function(g, |b| {
                let entry = b.function.get_entry_id();
                let mut e = b.block(entry);
                let p0 = e.add_parameter(Type::field().ref_of());
                let p1 = e.add_parameter(Type::field().ref_of());
                let v0 = e.load(p0);
                let v1 = e.load(p1);
                let s = e.add(v0, v1);
                e.store(p0, s);
                let zero = e.field_const(fr(0));
                e.store(p1, zero);
                e.terminate_return(vec![]);
            });
            // main(): a = alloc; b = alloc; *a = 1; *b = 2; g(a, b); return *a
            sb.modify_function(main_id, |b| {
                b.function.add_return_type(Type::field());
                let entry = b.function.get_entry_id();
                let mut e = b.block(entry);
                let a = falloc(&mut e);
                let bb = falloc(&mut e);
                let c1 = e.field_const(fr(1));
                let c2 = e.field_const(fr(2));
                e.store(a, c1);
                e.store(bb, c2);
                e.call(g, vec![a, bb], 0);
                let r = e.load(a);
                e.terminate_return(vec![r]);
            });
        }
        run_argpromo(&mut ssa);
        let (gp, gr) = sig(&ssa, g);
        assert!(
            gp.iter().all(|t| !t.is_ref()),
            "both ref params became by-value"
        );
        assert_eq!(gr, 2, "two out-returns appended");
        assert_eq!(
            first_call_arity(&ssa, main_id),
            Some((2, 2)),
            "two args kept, two out-results added"
        );
        run_mem2reg(&mut ssa);
        assert_eq!(op_counts(&ssa, main_id).0, 0);
        assert_eq!(op_counts(&ssa, g).0, 0);
    }

    /// A callee promoted only when *every* call site agrees: two distinct callers both pass clean
    /// locals, so the parameter promotes and both call sites are rewritten.
    #[test]
    fn two_call_sites_promote() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let main_id = ssa.get_unique_entrypoint_id();
        let (g, h);
        {
            let mut sb = HLSSABuilder::new(&mut ssa);
            g = sb.ssa().add_function("g".to_string());
            h = sb.ssa().add_function("h".to_string());
            // g(p: Ref<Field>): *p = *p + 1; return
            sb.modify_function(g, |b| {
                let entry = b.function.get_entry_id();
                let mut e = b.block(entry);
                let p = e.add_parameter(Type::field().ref_of());
                let v = e.load(p);
                let one = e.field_const(fr(1));
                let w = e.add(v, one);
                e.store(p, w);
                e.terminate_return(vec![]);
            });
            // h(): c = alloc; *c = 9; g(c); return
            sb.modify_function(h, |b| {
                let entry = b.function.get_entry_id();
                let mut e = b.block(entry);
                let c = falloc(&mut e);
                let nine = e.field_const(fr(9));
                e.store(c, nine);
                e.call(g, vec![c], 0);
                e.terminate_return(vec![]);
            });
            // main(): a = alloc; *a = 5; g(a); h(); return
            sb.modify_function(main_id, |b| {
                let entry = b.function.get_entry_id();
                let mut e = b.block(entry);
                let a = falloc(&mut e);
                let five = e.field_const(fr(5));
                e.store(a, five);
                e.call(g, vec![a], 0);
                e.call(h, vec![], 0);
                e.terminate_return(vec![]);
            });
        }
        run_argpromo(&mut ssa);
        let (gp, gr) = sig(&ssa, g);
        assert!(!gp[0].is_ref(), "param promoted across two call sites");
        assert_eq!(gr, 1);
        // Both call sites of g rewritten to (1 arg, 1 out-result).
        assert_eq!(first_call_arity(&ssa, main_id), Some((1, 1)));
        assert_eq!(first_call_arity(&ssa, h), Some((1, 1)));
    }

    /// Recursion: `g` recurses through a *local* (not its own ref parameter), so the parameter
    /// still promotes and the self-call site is rewritten without crashing or diverging the
    /// one-shot pass.
    #[test]
    fn recursion_through_local_promotes() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let main_id = ssa.get_unique_entrypoint_id();
        let g;
        {
            let mut sb = HLSSABuilder::new(&mut ssa);
            g = sb.ssa().add_function("g".to_string());
            // g(p: Ref<Field>): let q = alloc; *q = *p; g(q); *p = *q; return
            sb.modify_function(g, |b| {
                let entry = b.function.get_entry_id();
                let mut e = b.block(entry);
                let p = e.add_parameter(Type::field().ref_of());
                let v = e.load(p);
                let q = falloc(&mut e);
                e.store(q, v);
                e.call(g, vec![q], 0);
                let r = e.load(q);
                e.store(p, r);
                e.terminate_return(vec![]);
            });
            // main(): a = alloc; *a = 0; g(a); return
            sb.modify_function(main_id, |b| {
                let entry = b.function.get_entry_id();
                let mut e = b.block(entry);
                let a = falloc(&mut e);
                let z = e.field_const(fr(0));
                e.store(a, z);
                e.call(g, vec![a], 0);
                e.terminate_return(vec![]);
            });
        }
        run_argpromo(&mut ssa); // must not panic
        let (gp, gr) = sig(&ssa, g);
        assert!(!gp[0].is_ref(), "param promoted despite recursion");
        assert_eq!(gr, 1, "out-return appended");
        // The self-call inside g was rewritten too (its arg loaded, an out-result added).
        assert_eq!(first_call_arity(&ssa, g), Some((1, 1)));
        run_mem2reg(&mut ssa);
        assert_eq!(
            op_counts(&ssa, g).0,
            0,
            "callee + recursion local scalarized"
        );
    }

    /// Negative — a leaking parameter (stashed in a global) is retained: it both escapes (C2) and
    /// is used as a non-pointer operand (C3).
    #[test]
    fn leaking_param_is_retained() {
        let mut ssa = HLSSA::with_main("main".to_string());
        ssa.set_global_types(vec![Type::field().ref_of()]);
        let main_id = ssa.get_unique_entrypoint_id();
        let g;
        {
            let mut sb = HLSSABuilder::new(&mut ssa);
            g = sb.ssa().add_function("g".to_string());
            // g(p: Ref<Field>): global[0] = p; return
            sb.modify_function(g, |b| {
                let entry = b.function.get_entry_id();
                let mut e = b.block(entry);
                let p = e.add_parameter(Type::field().ref_of());
                e.init_global(0, p);
                e.terminate_return(vec![]);
            });
            // main(): a = alloc; *a = 5; g(a); return
            sb.modify_function(main_id, |b| {
                let entry = b.function.get_entry_id();
                let mut e = b.block(entry);
                let a = falloc(&mut e);
                let c = e.field_const(fr(5));
                e.store(a, c);
                e.call(g, vec![a], 0);
                e.terminate_return(vec![]);
            });
        }
        run_argpromo(&mut ssa);
        assert!(
            sig(&ssa, g).0[0].is_ref(),
            "leaking param retained as a ref"
        );
    }

    /// Negative — two arguments that may alias (`g(&a, &a)`) block promotion of both parameters
    /// (by-value in/out would lose the shared-cell semantics).
    #[test]
    fn aliasing_args_are_retained() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let main_id = ssa.get_unique_entrypoint_id();
        let g;
        {
            let mut sb = HLSSABuilder::new(&mut ssa);
            g = sb.ssa().add_function("g".to_string());
            // g(p0: Ref<F>, p1: Ref<F>): *p0 = *p0 + *p1; *p1 = 0; return
            sb.modify_function(g, |b| {
                let entry = b.function.get_entry_id();
                let mut e = b.block(entry);
                let p0 = e.add_parameter(Type::field().ref_of());
                let p1 = e.add_parameter(Type::field().ref_of());
                let v0 = e.load(p0);
                let v1 = e.load(p1);
                let s = e.add(v0, v1);
                e.store(p0, s);
                let zero = e.field_const(fr(0));
                e.store(p1, zero);
                e.terminate_return(vec![]);
            });
            // main(): a = alloc; *a = 5; g(a, a); return
            sb.modify_function(main_id, |b| {
                let entry = b.function.get_entry_id();
                let mut e = b.block(entry);
                let a = falloc(&mut e);
                let c = e.field_const(fr(5));
                e.store(a, c);
                e.call(g, vec![a, a], 0);
                e.terminate_return(vec![]);
            });
        }
        run_argpromo(&mut ssa);
        assert!(
            sig(&ssa, g).0.iter().all(|t| t.is_ref()),
            "aliasing args block promotion of both params"
        );
    }

    /// Negative — a `Ref<Ref<Field>>` co-argument that could transitively reach the promoted
    /// pointee makes the type-directed C4 check conservatively retain the parameter.
    #[test]
    fn refref_coarg_blocks_promotion() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let main_id = ssa.get_unique_entrypoint_id();
        let g;
        {
            let mut sb = HLSSABuilder::new(&mut ssa);
            g = sb.ssa().add_function("g".to_string());
            // g(p0: Ref<F>, pp: Ref<Ref<F>>): *p0 = *p0 + 1; return
            sb.modify_function(g, |b| {
                let entry = b.function.get_entry_id();
                let mut e = b.block(entry);
                let p0 = e.add_parameter(Type::field().ref_of());
                let _pp = e.add_parameter(Type::field().ref_of().ref_of());
                let v = e.load(p0);
                let one = e.field_const(fr(1));
                let w = e.add(v, one);
                e.store(p0, w);
                e.terminate_return(vec![]);
            });
            // main(): a = alloc F; *a = 5; rr = alloc Ref<F>; *rr = a; g(a, rr); return
            sb.modify_function(main_id, |b| {
                let entry = b.function.get_entry_id();
                let mut e = b.block(entry);
                let a = falloc(&mut e);
                let c = e.field_const(fr(5));
                e.store(a, c);
                let rr = e.alloc(a); // Ref<Ref<Field>>, seeded with `a`
                e.store(rr, a);
                e.call(g, vec![a, rr], 0);
                e.terminate_return(vec![]);
            });
        }
        run_argpromo(&mut ssa);
        assert!(
            sig(&ssa, g).0[0].is_ref(),
            "Ref<Ref> co-arg conservatively blocks promotion of the scalar ref param"
        );
    }

    /// Negative — a parameter forwarded to another callee (used as a call argument, not just a
    /// Load/Store pointer) is retained (C3 / onward-passing deferral).
    #[test]
    fn onward_passed_param_is_retained() {
        let mut ssa = HLSSA::with_main("main".to_string());
        ssa.set_global_types(vec![Type::field().ref_of()]);
        let main_id = ssa.get_unique_entrypoint_id();
        let (g, sink);
        {
            let mut sb = HLSSABuilder::new(&mut ssa);
            g = sb.ssa().add_function("g".to_string());
            sink = sb.ssa().add_function("sink".to_string());
            // sink(p: Ref<Field>): global[0] = p; return  (leaks, so not itself promotable)
            sb.modify_function(sink, |b| {
                let entry = b.function.get_entry_id();
                let mut e = b.block(entry);
                let p = e.add_parameter(Type::field().ref_of());
                e.init_global(0, p);
                e.terminate_return(vec![]);
            });
            // g(p: Ref<Field>): sink(p); return  (forwards its ref param onward)
            sb.modify_function(g, |b| {
                let entry = b.function.get_entry_id();
                let mut e = b.block(entry);
                let p = e.add_parameter(Type::field().ref_of());
                e.call(sink, vec![p], 0);
                e.terminate_return(vec![]);
            });
            // main(): a = alloc; *a = 5; g(a); return
            sb.modify_function(main_id, |b| {
                let entry = b.function.get_entry_id();
                let mut e = b.block(entry);
                let a = falloc(&mut e);
                let c = e.field_const(fr(5));
                e.store(a, c);
                e.call(g, vec![a], 0);
                e.terminate_return(vec![]);
            });
        }
        run_argpromo(&mut ssa);
        assert!(
            sig(&ssa, g).0[0].is_ref(),
            "onward-passed param retained (used as a call arg, not a Load/Store ptr)"
        );
    }

    // The entry-point guardrail (`!entry_points.contains(&g)`) is not unit-tested: at this pipeline
    // stage the points-to/flow analyses assume a single, value-typed entry point so a promotable
    // ref-param entry point cannot be constructed here. It is defensive against a future
    // multi-entry stage.

    /// Negative — an address-taken callee (its `FnPtr` exists) is never promoted, since not all
    /// call sites can be enumerated.
    #[test]
    fn address_taken_callee_is_retained() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let main_id = ssa.get_unique_entrypoint_id();
        let g;
        {
            let mut sb = HLSSABuilder::new(&mut ssa);
            g = sb.ssa().add_function("g".to_string());
            // g(p: Ref<Field>): *p = *p + 1; return
            sb.modify_function(g, |b| {
                let entry = b.function.get_entry_id();
                let mut e = b.block(entry);
                let p = e.add_parameter(Type::field().ref_of());
                let v = e.load(p);
                let one = e.field_const(fr(1));
                let w = e.add(v, one);
                e.store(p, w);
                e.terminate_return(vec![]);
            });
            // main(): take g's address, then call it normally.
            sb.modify_function(main_id, |b| {
                let entry = b.function.get_entry_id();
                let mut e = b.block(entry);
                let _fp = e.emit_constant(Constant::FnPtr(g));
                let a = falloc(&mut e);
                let c = e.field_const(fr(5));
                e.store(a, c);
                e.call(g, vec![a], 0);
                e.terminate_return(vec![]);
            });
        }
        run_argpromo(&mut ssa);
        assert!(
            sig(&ssa, g).0[0].is_ref(),
            "address-taken callee retained (call sites not enumerable)"
        );
    }
}
