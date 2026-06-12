//! The implementations for the phases of witness taint inference.
//!
//! [`compute_summaries`] (phase 1) builds a polymorphic [`FunctionSummary`] per function.
//! [`specialize_contexts`] (phase 2) instantiates those summaries from `main` into per-context
//! clones. See each function's own documentation for the details.

use std::collections::{BTreeMap, BTreeSet, VecDeque};

use crate::collections::{HashMap, HashSet};
use crate::compiler::{
    analysis::{
        flow_analysis::FlowAnalysis,
        types::{FunctionTypeInfo, TypeInfo, Types},
        witness_info::{FunctionWitnessType, WitnessShape, WitnessType},
        witness_taint_inference::{
            FunctionSummary, WitnessTaint,
            builder::{build_graph, compute_block_conditions},
            position::{Descent, Owner, Position, paths_of_type},
        },
    },
    ssa::{
        BlockId, FunctionId, ValueId,
        hlssa::{CallTarget, HLFunction, HLSSA, OpCode, Type, TypeExpr},
        traits::Instruction,
    },
    util::ice_non_elided_tuple,
};

// PHASE 1: POLYMORPHIC SUMMARIES
// ================================================================================================

/// Compute a polymorphic [`FunctionSummary`] per function via an ascending fixpoint over the call
/// graph, along with each function's final `≥` graph.
///
/// Each summary starts empty and grows monotonically; whenever one grows, its callers are
/// re-queued. Because the transfer functions are monotone over a finite, 2-point lattice, this
/// converges to the least exact fixpoint. Recursion — including mutual recursion — is handled
/// entirely by this fixpoint.
///
/// The returned graphs are the ones built on each function's *last* worklist pop. Those are the
/// final-summary graphs: a callee summary change always re-queues its callers, so when the
/// worklist drains, every function was last rebuilt after the final change of each of its callees'
/// summaries. Returning them saves phase 2 a full rebuild pass.
fn compute_summaries(
    ssa: &HLSSA,
    flow: &FlowAnalysis,
    types: &TypeInfo,
    block_conds: &HashMap<FunctionId, HashMap<BlockId, Vec<ValueId>>>,
    fids: &[FunctionId],
) -> (
    HashMap<FunctionId, FunctionSummary>,
    HashMap<FunctionId, WitnessTaint>,
) {
    let mut summaries: HashMap<FunctionId, FunctionSummary> = fids
        .iter()
        .map(|f| (*f, FunctionSummary::default()))
        .collect();

    // The formal input/output skeleton is summary-independent (the transitive read-globals depend
    // only on the call graph, not the evolving summaries): compute it once per function rather
    // than on every worklist pop.
    let read_globals = transitive_read_globals(ssa, flow, fids);
    let global_types = ssa.get_global_types();
    let skeletons: HashMap<FunctionId, SummarySkeleton> = fids
        .iter()
        .map(|f| {
            (
                *f,
                summary_skeleton(ssa.get_function(*f), &read_globals[f], global_types),
            )
        })
        .collect();

    // Seed the worklist callee-first (post-order from main), so summaries are computed before
    // their callers consume them and non-recursive call graphs converge in a single pass. Append
    // any analyzed function unreachable from main so none are dropped.
    let mut order: Vec<FunctionId> = flow
        .get_call_graph()
        .get_post_order(ssa.get_main_id())
        .filter(|f| summaries.contains_key(f))
        .collect();
    let mut queued: HashSet<FunctionId> = order.iter().copied().collect();
    for f in fids {
        if queued.insert(*f) {
            order.push(*f);
        }
    }
    let mut worklist: VecDeque<FunctionId> = order.into();

    let mut graphs: HashMap<FunctionId, WitnessTaint> = HashMap::default();
    while let Some(f) = worklist.pop_front() {
        queued.remove(&f);
        let func = ssa.get_function(f);
        let g = build_graph(
            func,
            types.get_function(f),
            flow.get_function_cfg(f),
            &block_conds[&f],
            &summaries,
        );
        let new = extract_summary(&g, &skeletons[&f]);
        graphs.insert(f, g);
        if summaries[&f] != new {
            summaries.insert(f, new);
            for c in flow.get_call_graph().get_callers(f) {
                if summaries.contains_key(&c) && queued.insert(c) {
                    worklist.push_back(c);
                }
            }
        }
    }

    (summaries, graphs)
}

/// A function's formal summary endpoints — the inputs and outputs of [`extract_summary`].
///
/// There is no inherent source/sink split between function parameters and returns; what matters is
/// who determines a level's taint:
///
/// - `inputs` are the levels the *caller* determines: every parameter level, the Deref-descended
///   levels of every return (a returned ref is as much "caller chooses" as a ref parameter — the
///   caller may write through it), the globals the function reads (transitively, through any
///   callee), the cfg flag, and `Top`.
/// - `outputs` are the levels whose taint the *callee* communicates back: every return level and
///   the Deref-descended levels of every parameter (the callee may write through a ref argument).
///
/// Deref-descended levels are both: they name shared memory that either side can write.
///
/// Depends only on the function (signature plus the globals it transitively reads), never on the
/// evolving summaries, so phase 1 computes it once per function.
struct SummarySkeleton {
    inputs: Vec<Position>,
    outputs: Vec<Position>,
}

fn summary_skeleton(
    func: &HLFunction,
    read_globals: &BTreeSet<usize>,
    global_types: &[Type],
) -> SummarySkeleton {
    let params = param_level_positions(func);
    let returns = return_level_positions(func);

    let mut inputs: Vec<Position> = params.clone();
    inputs.extend(returns.iter().cloned().filter(Position::is_deref_descended));
    let mut outputs: Vec<Position> = params
        .into_iter()
        .filter(Position::is_deref_descended)
        .collect();
    outputs.extend(returns);

    // Globals the function reads — directly or through any callee — are summary inputs, so
    // `output ≥ Global(g)` edges propagate to callers when the callee summary is instantiated
    // (Global maps to itself across the call). Callee reads count too: instantiating a callee
    // summary plants its `Global` sources in *this* function's graph, and a source that is not
    // also an input here would be dropped from this function's own summary — the dependence
    // would be lost two or more call levels above the read.
    for g in read_globals {
        let owner = Owner::Global(*g);
        for p in paths_of_type(&global_types[*g]) {
            inputs.push(Position::root(owner.clone()).extend(&p));
        }
    }

    inputs.push(Position::root(Owner::Cfg));
    inputs.push(Position::top());
    SummarySkeleton { inputs, outputs }
}

/// For each function, the globals it reads transitively: its own `ReadGlobal`s plus its callees',
/// closed over the call graph (a small fixpoint, since recursion is allowed).
///
/// Depends only on the code and the call graph, never on the evolving summaries. `BTreeSet` keeps
/// the skeleton's input order deterministic.
fn transitive_read_globals(
    ssa: &HLSSA,
    flow: &FlowAnalysis,
    fids: &[FunctionId],
) -> HashMap<FunctionId, BTreeSet<usize>> {
    let mut read_globals: HashMap<FunctionId, BTreeSet<usize>> = fids
        .iter()
        .map(|f| {
            let mut globals = BTreeSet::new();
            for (_, block) in ssa.get_function(*f).get_blocks() {
                for instr in block.get_instructions() {
                    if let OpCode::ReadGlobal { offset, .. } = instr {
                        globals.insert(*offset as usize);
                    }
                }
            }
            (*f, globals)
        })
        .collect();

    let call_graph = flow.get_call_graph();
    let mut worklist: VecDeque<FunctionId> = fids.iter().copied().collect();
    let mut queued: HashSet<FunctionId> = worklist.iter().copied().collect();
    while let Some(f) = worklist.pop_front() {
        queued.remove(&f);
        let callee_globals: Vec<usize> = call_graph
            .get_callees(f)
            .filter(|c| read_globals.contains_key(c))
            .flat_map(|c| read_globals[&c].iter().copied().collect::<Vec<_>>())
            .collect();
        let globals = read_globals.get_mut(&f).unwrap();
        let mut grown = false;
        for g in callee_globals {
            grown |= globals.insert(g);
        }
        if grown {
            for c in call_graph.get_callers(f) {
                if read_globals.contains_key(&c) && queued.insert(c) {
                    worklist.push_back(c);
                }
            }
        }
    }
    read_globals
}

/// Every level of every parameter of `func`, as formal `Param` positions.
fn param_level_positions(func: &HLFunction) -> Vec<Position> {
    let mut out = Vec::new();
    for (i, (_, ty)) in func.get_entry().get_parameters().enumerate() {
        for p in paths_of_type(ty) {
            out.push(Position {
                owner: Owner::Param(i),
                path: p,
            });
        }
    }
    out
}

/// Every level of every return of `func`, as formal `Return` positions.
fn return_level_positions(func: &HLFunction) -> Vec<Position> {
    let mut out = Vec::new();
    for (j, ty) in func.get_returns().iter().enumerate() {
        for p in paths_of_type(ty) {
            out.push(Position {
                owner: Owner::Return(j),
                path: p,
            });
        }
    }
    out
}

/// Extract the summary: for each formal input, which formal outputs it reaches.
fn extract_summary(g: &WitnessTaint, skeleton: &SummarySkeleton) -> FunctionSummary {
    // One labeled traversal: each node learns which inputs taint it, then every (output, input)
    // edge is read off the outputs' bitsets — instead of one full BFS per input.
    let reaching = g.reaching_sources(&skeleton.inputs);

    let mut edges: BTreeSet<(Position, Position)> = BTreeSet::new();
    for output in &skeleton.outputs {
        let Some(bits) = reaching.get(output) else {
            continue;
        };
        for (i, input) in skeleton.inputs.iter().enumerate() {
            if bits.contains(i) && input != output {
                edges.insert((output.clone(), input.clone()));
            }
        }
    }
    FunctionSummary { edges }
}

// SPECIALIZATION CONTEXT
// ================================================================================================

/// A concrete specialization context
///
/// This contains the original function, its concrete argument taint shapes, the caller-written
/// taint of its returns' Deref-descended levels, and whether it is called under witness control
/// flow.
#[derive(Clone, PartialEq, Eq, Hash)]
struct Ctx {
    fid: FunctionId,
    arg_shapes: Vec<WitnessShape>,

    /// The caller-determined taint of each return. The [`DerefShape`] newtype guarantees that
    /// only Deref-descended levels (the memory a returned ref names, which the caller may write)
    /// carry information, so contexts never split on callee-determined output bits.
    ret_shapes: Vec<DerefShape>,
    cfg_witness: WitnessType,
}

/// A return's caller-determined taint: a [`WitnessShape`] carrying information only at
/// Deref-descended levels — everything at or above the first `Ref` is `Pure`.
///
/// The invariant is guaranteed by construction: the only constructors are [`DerefShape::pure`]
/// (trivially all-Pure) and [`DerefShape::from_solution`] (which never reads a non-Deref level),
/// so a [`Ctx`] literally cannot hold callee-determined output taint.
#[derive(Clone, PartialEq, Eq, Hash)]
struct DerefShape(WitnessShape);

impl DerefShape {
    /// The all-Pure shape of `ty` — the no-caller-writes case (e.g. the entry context, which has
    /// no caller to write through its returned refs).
    fn pure(ty: &Type) -> Self {
        DerefShape(pure_shape(ty))
    }

    /// Build only the Deref-descended levels of a value's shape from the solved witness set;
    /// every level at or above the first `Ref` is `Pure` by construction.
    ///
    /// Deref-descended levels name shared mutable memory the caller may write, so this is exactly
    /// the caller-determined taint of a call result — the shape-level counterpart of the
    /// skeleton's `path.contains(&Descent::Deref)` input filter.
    fn from_solution(owner: Owner, ty: &Type, witness: &HashSet<Position>) -> Self {
        DerefShape(go_shape_from(&owner, &mut Vec::new(), ty, witness, false))
    }
}

/// Run the whole analysis, mutating `ssa` (clones + call rewiring + entry point) and returning the
/// per-specialized-function witness types.
pub fn run(ssa: &mut HLSSA, flow: &FlowAnalysis) -> HashMap<FunctionId, FunctionWitnessType> {
    let types = Types::new().run(ssa, flow);

    // Per-block dominating witness-branch conditions per function (summary-independent).
    let fids: Vec<FunctionId> = ssa
        .get_function_ids()
        .filter(|f| types.has_function(*f))
        .collect();
    let mut block_conds: HashMap<FunctionId, HashMap<BlockId, Vec<ValueId>>> = HashMap::default();
    for f in &fids {
        let func = ssa.get_function(*f);
        block_conds.insert(
            *f,
            compute_block_conditions(func, flow.get_function_cfg(*f)),
        );
    }

    // Once summaries are frozen, each function's `≥` graph is fixed (the witness-globals set and
    // per-context argument shapes enter only through solver *seeds*, never the graph). Phase 1
    // already built every graph against the final summaries — the summaries themselves are baked
    // into those graphs as call-site edges — so the graphs are shared across all
    // `compute_witness_globals` iterations and all phase-2 contexts of the same function.
    let (_summaries, graphs) = compute_summaries(ssa, flow, &types, &block_conds, &fids);

    let witness_globals = compute_witness_globals(ssa, &types, &graphs, &fids);
    specialize_contexts(ssa, &types, &graphs, &block_conds, &witness_globals)
}

// WITNESS GLOBAL HANDLING
// ================================================================================================

/// Determine which global slots may hold a witness value, program-wide.
///
/// Globals are program-wide state, not per-call formals, so their witness-ness is decided once here
/// rather than threaded through call summaries. A global is witness if any `InitGlobal` writes a
/// witness value (or writes under a witness branch *within* the initializer). To stay sound without
/// context-sensitivity, each initializing function is solved with its parameters seeded Witness —
/// the worst case for an argument-derived global write — and with the writer's own dominating
/// witness branch conditions accounted for. The fixpoint accounts for an init that reads another
/// global.
///
/// We do not seed the writer's cfg flag: initializers run unconditionally (asserted below — only
/// the dedicated `globals_init` function may contain `InitGlobal`), so their flag is Pure. Today
/// this returns the empty set: every `InitGlobal` lives in `globals_init`, runs unconditionally,
/// and writes a compile-time constant. This will support mutable globals with very few changes as
/// a result.
fn compute_witness_globals(
    ssa: &HLSSA,
    types: &TypeInfo,
    graphs: &HashMap<FunctionId, WitnessTaint>,
    fids: &[FunctionId],
) -> HashSet<usize> {
    let global_types = ssa.get_global_types();
    let writers: Vec<FunctionId> = fids
        .iter()
        .copied()
        .filter(|f| {
            ssa.get_function(*f).get_blocks().any(|(_, block)| {
                block
                    .get_instructions()
                    .any(|instr| matches!(instr, OpCode::InitGlobal { .. }))
            })
        })
        .collect();

    // The Cfg-seed skip below is sound only while initializers run unconditionally. That holds
    // today by construction — `InitGlobal` is only ever emitted into the dedicated `globals_init`
    // function, which the entry wrapper calls exactly once, never under a witness branch — but
    // nothing else enforces it, so fail loudly if it breaks: a writer outside `globals_init`
    // could run under a *caller's* witness control flow, and skipping its Cfg seed would
    // under-taint the global program-wide.
    assert!(
        writers
            .iter()
            .all(|f| Some(*f) == ssa.get_globals_init_fn()),
        "ICE: InitGlobal outside the dedicated globals_init function ({:?}); compute_witness_globals assumes global initializers run unconditionally",
        writers
    );

    let mut witness: HashSet<usize> = HashSet::default();
    loop {
        let mut changed = false;
        for f in &writers {
            let func = ssa.get_function(*f);
            let ftypes = types.get_function(*f);
            // The evolving `witness` set enters only through the seeds below, so the shared
            // pre-built graph is reused across fixpoint iterations.
            let g = &graphs[f];

            // Worst-case seeds: Top, every parameter level, every Deref-descended return level
            // (a caller could write a witness through a returned ref before the global is read —
            // vacuous today since `globals_init` returns nothing), and known witness globals.
            //
            // We deliberately do NOT seed the writer's cfg flag (`Cfg`). Global initializers run
            // unconditionally — `globals_init` is called exactly once at program entry, never under a
            // witness branch — so their cfg flag is structurally Pure. Seeding it Witness made *every*
            // initialized global witness, because each `InitGlobal` records `global ≥ Cfg` (see
            // `add_cf_taint_to`); that spurious taint then forced witness loop bounds and witness array
            // reads downstream. A witness branch *inside* an initializer is still captured: that path
            // adds `global ≥ cond` for the real branch conditions, and `cond` becomes witness through
            // the Top/param seeds.
            let mut seeds: Vec<Position> = vec![Position::top()];
            seeds.extend(param_level_positions(func));
            seeds.extend(
                return_level_positions(func)
                    .into_iter()
                    .filter(Position::is_deref_descended),
            );
            for gid in &witness {
                let owner = Owner::Global(*gid);
                for p in paths_of_type(&global_types[*gid]) {
                    seeds.push(Position::root(owner.clone()).extend(&p));
                }
            }
            let solved = g.solve(seeds);

            for (_, block) in func.get_blocks() {
                for instr in block.get_instructions() {
                    if let OpCode::InitGlobal { global, value } = instr {
                        let value_ty = ftypes.get_value_type(*value);
                        let owner = Owner::Global(*global);
                        let any_witness = paths_of_type(value_ty)
                            .into_iter()
                            .any(|p| solved.contains(&Position::root(owner.clone()).extend(&p)));
                        if any_witness && witness.insert(*global) {
                            changed = true;
                        }
                    }
                }
            }
        }
        if !changed {
            break;
        }
    }
    witness
}

// PHASE 2: CONCRETE INSTANTIATION AND CLONING
// ================================================================================================

/// The concrete witness solution for one specialization context, before it is materialized onto a
/// clone as a [`FunctionWitnessType`].
struct ContextSolution {
    /// The inferred `WitnessShape` of every SSA value (keyed by the *original* function's value
    /// ids; re-keyed onto the clone during materialization).
    value_shapes: HashMap<ValueId, WitnessShape>,

    /// Each block's cfg-witness: whether it runs under witness-dependent control flow.
    block_cfg_witness: HashMap<BlockId, WitnessType>,

    /// The witness shape of each return value, positionally.
    return_shapes: Vec<WitnessShape>,

    /// Each constrained static call site, keyed by `(block, instruction index)`, paired with the
    /// concrete callee context it resolves to. Cloning preserves block ids and instruction order,
    /// so rewiring looks each call up by its site (`BTreeMap` keeps clone-discovery order
    /// deterministic).
    calls: BTreeMap<(BlockId, usize), Ctx>,
}

/// Walk the reachable `(function, concrete arg shapes, concrete return-deref shapes, concrete
/// cfg)` contexts (see [`Ctx`]) from `main`, solving each once against the phase-1 summaries (no
/// inter-context fixpoint needed — the summary is the exact transfer function for the context).
///
/// Each distinct context becomes a clone: its per-value shapes are re-keyed onto the clone, a
/// [`FunctionWitnessType`] is registered for it, and `Call` targets are rewired to the matching
/// clone. The clone-per-context is required because `UntaintControlFlow` bakes context-specific
/// `WitnessOf` types and a context-specific cfg-flag parameter into each body.
fn specialize_contexts(
    ssa: &mut HLSSA,
    types: &TypeInfo,
    graphs: &HashMap<FunctionId, WitnessTaint>,
    block_conds: &HashMap<FunctionId, HashMap<BlockId, Vec<ValueId>>>,
    witness_globals: &HashSet<usize>,
) -> HashMap<FunctionId, FunctionWitnessType> {
    let main_id = ssa.get_main_id();
    // Program-wide global slot types, captured before the SSA is mutated (clones/rewiring below).
    let global_types: Vec<Type> = ssa.get_global_types().to_vec();
    let main_args: Vec<WitnessShape> = ssa
        .get_function(main_id)
        .get_entry()
        .get_parameters()
        .map(|(_, ty)| pure_shape(ty))
        .collect();

    // The entry has no caller to write through its returned refs: all-Pure.
    let main_rets: Vec<DerefShape> = ssa
        .get_function(main_id)
        .get_returns()
        .iter()
        .map(DerefShape::pure)
        .collect();
    let main_ctx = Ctx {
        fid: main_id,
        arg_shapes: main_args,
        ret_shapes: main_rets,
        cfg_witness: WitnessType::Pure,
    };

    // Discover contexts (BFS), solving each once and cloning it.
    let mut ctx_clone: HashMap<Ctx, (FunctionId, HashMap<ValueId, ValueId>)> = HashMap::default();
    let mut ctx_result: HashMap<Ctx, ContextSolution> = HashMap::default();
    let mut worklist: VecDeque<Ctx> = VecDeque::new();

    let (cid, remap) = ssa.duplicate_function_with_remap(main_id);
    ctx_clone.insert(main_ctx.clone(), (cid, remap));
    worklist.push_back(main_ctx.clone());
    let main_clone_id = cid;

    while let Some(ctx) = worklist.pop_front() {
        let result = solve_context(
            &FunctionData {
                func: ssa.get_function(ctx.fid),
                types: types.get_function(ctx.fid),
                graph: &graphs[&ctx.fid],
                block_conds: &block_conds[&ctx.fid],
            },
            &ctx,
            witness_globals,
            &global_types,
        );
        for cc in result.calls.values() {
            if !ctx_clone.contains_key(cc) {
                let (cid, remap) = ssa.duplicate_function_with_remap(cc.fid);
                ctx_clone.insert(cc.clone(), (cid, remap));
                worklist.push_back(cc.clone());
            }
        }
        ctx_result.insert(ctx, result);
    }

    // Materialize: re-key shapes onto each clone, register FunctionWitnessType, rewire call targets.
    let mut functions: HashMap<FunctionId, FunctionWitnessType> = HashMap::default();
    for (ctx, result) in &ctx_result {
        let (clone_id, remap) = &ctx_clone[ctx];

        let mut value_witness_types: HashMap<ValueId, WitnessShape> = HashMap::default();
        for (ovid, shape) in &result.value_shapes {
            let cvid = remap.get(ovid).copied().unwrap_or(*ovid);
            value_witness_types.insert(cvid, shape.clone());
        }

        functions.insert(
            *clone_id,
            FunctionWitnessType {
                returns_witness: result.return_shapes.clone(),
                cfg_witness: ctx.cfg_witness,
                parameters: ctx.arg_shapes.clone(),
                block_cfg_witness: result.block_cfg_witness.clone(),
                value_witness_types,
            },
        );

        // Rewire each constrained static call to the clone of the callee context, looking it up
        // by its `(block, instruction index)` site — the clone preserves both.
        let mut clone_func = ssa.take_function(*clone_id);
        let clone_bids: Vec<BlockId> = clone_func.get_blocks().map(|(bid, _)| *bid).collect();
        let mut rewired = 0usize;
        for bid in clone_bids {
            for (idx, instr) in clone_func
                .get_block_mut(bid)
                .get_instructions_mut()
                .enumerate()
            {
                if let OpCode::Call {
                    function: CallTarget::Static(t),
                    unconstrained: false,
                    ..
                } = instr
                {
                    let cc = result
                        .calls
                        .get(&(bid, idx))
                        .expect("ICE: constrained call site without a solved context");
                    *t = ctx_clone[cc].0;
                    rewired += 1;
                }
            }
        }
        assert_eq!(
            rewired,
            result.calls.len(),
            "ICE: Leftover call contexts in Witness Taint Inference after rewiring"
        );
        ssa.put_function(*clone_id, clone_func);
    }

    ssa.set_entry_point(main_clone_id);
    functions
}

/// The per-function immutable inputs to [`solve_context`] — everything keyed by the context's
/// `FunctionId`, bundled so the context-specific inputs ([`Ctx`]) stand out at the call site.
#[derive(Clone, Copy)]
struct FunctionData<'a> {
    func: &'a HLFunction,
    types: &'a FunctionTypeInfo,
    graph: &'a WitnessTaint,
    block_conds: &'a HashMap<BlockId, Vec<ValueId>>,
}

/// Solve one concrete context.
///
/// Seed the function's pre-built `≥` graph with the concrete witness inputs (the witness levels of
/// each argument, the caller-written deref levels of each return, the cfg flag when the call is
/// tainted, and every witness global), then read off the per-value / per-return / per-block shapes
/// and the constrained call sites paired with the concrete callee contexts they resolve to. The
/// graph is shared by every context of the same function — contexts differ only in their seeds.
fn solve_context(
    f: &FunctionData,
    ctx: &Ctx,
    witness_globals: &HashSet<usize>,
    global_types: &[Type],
) -> ContextSolution {
    let FunctionData {
        func,
        types,
        graph,
        block_conds,
    } = *f;

    // Seeds: Top, the witness levels of each argument, the caller-written deref levels of each
    // return, the cfg flag if the call is tainted, and every witness global slot (empty today; see
    // `compute_witness_globals`).
    let mut seeds: Vec<Position> = vec![Position::top()];
    for (i, shape) in ctx.arg_shapes.iter().enumerate() {
        seed_shape(Owner::Param(i), shape, &mut Vec::new(), &mut seeds);
    }

    // `DerefShape` guarantees these seeds are Deref-descended (caller-determined levels only).
    for (j, shape) in ctx.ret_shapes.iter().enumerate() {
        seed_shape(Owner::Return(j), &shape.0, &mut Vec::new(), &mut seeds);
    }

    if ctx.cfg_witness == WitnessType::Witness {
        seeds.push(Position::root(Owner::Cfg));
    }

    for gid in witness_globals {
        let owner = Owner::Global(*gid);
        for p in paths_of_type(&global_types[*gid]) {
            seeds.push(Position::root(owner.clone()).extend(&p));
        }
    }
    let witness = graph.solve(seeds);

    // Per-value shapes.
    let mut value_shapes: HashMap<ValueId, WitnessShape> = HashMap::default();
    for (v, ty) in func.get_entry().get_parameters() {
        value_shapes.insert(*v, shape_from(Owner::Value(*v), ty, &witness));
    }
    for (_, block) in func.get_blocks() {
        for (v, ty) in block.get_parameters() {
            value_shapes.insert(*v, shape_from(Owner::Value(*v), ty, &witness));
        }
        for instr in block.get_instructions() {
            for r in instr.get_results() {
                let ty = types.get_value_type(*r);
                value_shapes.insert(*r, shape_from(Owner::Value(*r), ty, &witness));
            }
        }
    }

    let return_shapes: Vec<WitnessShape> = func
        .get_returns()
        .iter()
        .enumerate()
        .map(|(j, ty)| shape_from(Owner::Return(j), ty, &witness))
        .collect();

    // Per-block cfg-witness: the incoming flag, or any dominating witness branch condition.
    let mut block_cfg_witness: HashMap<BlockId, WitnessType> = HashMap::default();
    for (bid, _) in func.get_blocks() {
        let mut info = ctx.cfg_witness;
        if let Some(conds) = block_conds.get(bid) {
            for cond in conds {
                if witness.contains(&Position::root(Owner::Value(*cond))) {
                    info = WitnessType::Witness;
                }
            }
        }
        block_cfg_witness.insert(*bid, info);
    }

    // Constrained static call sites, keyed by site, with their concrete callee contexts.
    let mut calls: BTreeMap<(BlockId, usize), Ctx> = BTreeMap::new();
    for (bid, block) in func.get_blocks() {
        let block_cw = *block_cfg_witness.get(bid).unwrap();
        for (idx, instr) in block.get_instructions().enumerate() {
            if let OpCode::Call {
                function: CallTarget::Static(callee),
                args,
                results,
                unconstrained: false,
            } = instr
            {
                let arg_shapes: Vec<WitnessShape> = args
                    .iter()
                    .map(|a| {
                        value_shapes
                            .get(a)
                            .cloned()
                            .unwrap_or_else(|| pure_shape(types.get_value_type(*a)))
                    })
                    .collect();

                // The result value's deref levels picked up the caller's writes through the
                // returned ref (via unification with its aliases) during this solve; restricted to
                // those levels it is the caller-determined return input of the callee context.
                let ret_shapes: Vec<DerefShape> = results
                    .iter()
                    .map(|r| {
                        DerefShape::from_solution(
                            Owner::Value(*r),
                            types.get_value_type(*r),
                            &witness,
                        )
                    })
                    .collect();
                calls.insert(
                    (*bid, idx),
                    Ctx {
                        fid: *callee,
                        arg_shapes,
                        ret_shapes,
                        cfg_witness: block_cw,
                    },
                );
            }
        }
    }

    ContextSolution {
        value_shapes,
        block_cfg_witness,
        return_shapes,
        calls,
    }
}

// SHAPE HELPERS
// ================================================================================================

/// The all-`Pure` shape skeleton of a type: [`shape_from`] with an empty witness set (the owner is
/// irrelevant — no position is ever found). One shared walker, so a new `TypeExpr` variant has a
/// single place to be handled.
fn pure_shape(ty: &Type) -> WitnessShape {
    shape_from(Owner::Top, ty, &HashSet::default())
}

/// Build a value's `WitnessShape` from the solved witness set: each level is Witness iff its
/// position is in the set.
fn shape_from(owner: Owner, ty: &Type, witness: &HashSet<Position>) -> WitnessShape {
    go_shape_from(&owner, &mut Vec::new(), ty, witness, true)
}

/// Shared recursion of [`shape_from`] / [`DerefShape::from_solution`]: when `read` is false a
/// level is `Pure` without consulting the witness set, and descending through a `Ref` turns
/// reading on.
fn go_shape_from(
    owner: &Owner,
    path: &mut Vec<Descent>,
    ty: &Type,
    witness: &HashSet<Position>,
    read: bool,
) -> WitnessShape {
    let ty = ty.peel_witness();
    let info = if read
        && witness.contains(&Position {
            owner: owner.clone(),
            path: path.clone(),
        }) {
        WitnessType::Witness
    } else {
        WitnessType::Pure
    };
    match &ty.expr {
        TypeExpr::Field
        | TypeExpr::U(_)
        | TypeExpr::I(_)
        | TypeExpr::Function
        | TypeExpr::Blob(..) => WitnessShape::Scalar(info),
        TypeExpr::Array(inner, _) | TypeExpr::Slice(inner) => {
            path.push(Descent::Elem);
            let c = go_shape_from(owner, path, inner, witness, read);
            path.pop();
            WitnessShape::Array(info, Box::new(c))
        }
        TypeExpr::Ref(inner) => {
            path.push(Descent::Deref);
            let c = go_shape_from(owner, path, inner, witness, true);
            path.pop();
            WitnessShape::Ref(info, Box::new(c))
        }
        TypeExpr::WitnessOf(_) => panic!("ICE: WitnessOf during witness-taint inference"),
        TypeExpr::Tuple(_) => ice_non_elided_tuple(),
    }
}

/// Seed the `owner` positions that are concretely Witness in `shape`.
fn seed_shape(
    owner: Owner,
    shape: &WitnessShape,
    path: &mut Vec<Descent>,
    seeds: &mut Vec<Position>,
) {
    if shape.toplevel_info() == WitnessType::Witness {
        seeds.push(Position {
            owner: owner.clone(),
            path: path.clone(),
        });
    }
    match shape {
        WitnessShape::Scalar(_) => {}
        WitnessShape::Array(_, inner) => {
            path.push(Descent::Elem);
            seed_shape(owner, inner, path, seeds);
            path.pop();
        }
        WitnessShape::Ref(_, inner) => {
            path.push(Descent::Deref);
            seed_shape(owner, inner, path, seeds);
            path.pop();
        }
    }
}
