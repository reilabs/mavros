//! Implements a promotion from stack-based load-store traffic into pure SSA values.
//!
//! This is a standard Cytron-style (TOPLAS '91) dominance-frontier iteration that differs only in
//! its use of block parameters instead of phi edges.
//!
//! Whether an allocation can be promoted is determined using the [`PointsTo`] analysis that runs
//! upstream. A local allocation is promoted iff it does not escape and every access to it goes
//! through a pointer that points to _only_ it. The pass, thus, runs partially, promoting clean
//! locals while leaving escaping refs, ref parameters, and aliased allocations as ordinary memory
//! traffic in the same function.
//!
//! This analysis is purely _intraprocedural_ and does not perform promotions through arguments or
//! returns. For an _interprocedural_ pass that handles this, see `arg_promotion.rs`.

use std::collections::VecDeque;

use tracing::{Level, debug, instrument};

use crate::{
    collections::{HashMap, HashSet},
    compiler::{
        analysis::{
            flow_analysis::{CFG, FlowAnalysis},
            points_to::{PointerUse, PointsTo, object::AbstractObject},
            shared::call_string::Context,
            types::{FunctionTypeInfo, TypeInfo},
        },
        pass_manager::{Analysis, AnalysisId, AnalysisStore, Pass},
        passes::shared::value_replacements::ValueReplacements,
        ssa::{
            BlockId, FunctionId, Terminator, ValueId,
            hlssa::{
                HLFunction, HLSSA, OpCode,
                builder::{HLFunctionBuilder, HLSSABuilder},
            },
        },
    },
};

// MEM2REG PASS
// ================================================================================================

pub struct Mem2Reg {}

impl Pass for Mem2Reg {
    fn name(&self) -> &'static str {
        "mem2reg"
    }

    fn needs(&self) -> Vec<AnalysisId> {
        vec![FlowAnalysis::id(), TypeInfo::id(), PointsTo::id()]
    }

    fn run(&self, ssa: &mut HLSSA, store: &AnalysisStore) {
        self.do_run(
            ssa,
            store.get::<FlowAnalysis>(),
            store.get::<TypeInfo>(),
            store.get::<PointsTo>(),
        );
    }
}

impl Mem2Reg {
    pub fn new() -> Self {
        Self {}
    }

    pub fn do_run(
        &self,
        ssa: &mut HLSSA,
        cfg: &FlowAnalysis,
        type_info: &TypeInfo,
        points_to: &PointsTo,
    ) {
        let fids: Vec<_> = ssa.get_function_ids().collect();
        let mut sb = HLSSABuilder::new(ssa);
        for function_id in fids {
            let function_type_info = type_info.get_function(function_id);
            let func_cfg = cfg.get_function_cfg(function_id);
            sb.modify_function(function_id, |fb| {
                // Per-`Alloc` promotability, from the points-to analysis. This promotes each clean
                // local allocation independently — even in a function that also contains escaping
                // refs, ref params, or calls.
                let promotable =
                    self.promotable_allocs(fb.function, function_type_info, function_id, points_to);
                if promotable.is_empty() {
                    debug!(
                        "Skipping mem2reg for function: {:?} (no promotable allocations)",
                        fb.function.get_name()
                    );
                } else {
                    self.run_function(
                        fb,
                        func_cfg,
                        function_type_info,
                        function_id,
                        points_to,
                        &promotable,
                    );
                }
            });
        }
    }

    #[instrument(skip_all, level = Level::DEBUG, fields(function = %fb.function.get_name()))]
    fn run_function(
        &self,
        fb: &mut HLFunctionBuilder<'_>,
        cfg: &CFG,
        type_info: &FunctionTypeInfo,
        fid: FunctionId,
        points_to: &PointsTo,
        promotable: &HashSet<ValueId>,
    ) {
        let mut promotable = promotable.clone();
        let (writes, defs) =
            self.find_pointer_writes_and_defs(fb.function, fid, points_to, &promotable);
        let mut phi_blocks = self.find_phi_blocks(&writes, &defs, cfg);

        // Defensive: a promoted alloc read before it is written on some path would leave
        // `remove_ptrs` with no threaded value (an ICE). The frontend's initialize-before-use rule
        // makes this unreachable for valid input, but partial promotion widens the candidate set,
        // so drop any such alloc — leaving it as ordinary memory traffic — rather than crash.
        let uninitialized =
            self.uninitialized_allocs(fb.function, cfg, &phi_blocks, fid, points_to, &promotable);
        for alloc in &uninitialized {
            promotable.remove(alloc);
            phi_blocks.remove(alloc); // per-alloc keyed, so the remaining placements stay valid
        }

        let phi_args = self.initialize_phis(fb, &phi_blocks, type_info);
        self.remove_ptrs(fb.function, cfg, &phi_args, fid, points_to, &promotable);
    }

    /// Walk the CFG in domination pre-order, threading a per-block state `S` that is seeded from
    /// the immediate dominator's outgoing state (an entry/unreachable block starts from
    /// `S::default()`). `visit` receives each block id and that incoming state, already seeded, and
    /// updates it in place; the resulting state is stored and becomes the seed for every block this
    /// one immediately dominates.
    ///
    /// This is the single shared dataflow skeleton of [`Self::remove_ptrs`] (whose state threads
    /// the promoted-alloc → live-value map) and [`Self::uninitialized_allocs`] (whose state tracks
    /// which allocs have an available value). Both must visit blocks in the same order and inherit
    /// from the immediate dominator identically — keeping that traversal in one place is what
    /// guarantees they cannot drift, since a divergence would turn `remove_ptrs`'s "no threaded
    /// value" `expect` into an ICE on a program the guard was supposed to have pruned.
    fn propagate_in_domination_order<S: Clone + Default>(
        cfg: &CFG,
        mut visit: impl FnMut(BlockId, &mut S),
    ) {
        let mut state_at: HashMap<BlockId, S> = HashMap::default();
        for block_id in cfg.get_domination_pre_order() {
            // Fetch the outgoing state of the immediate dominator (already finalized, since
            // dominators precede the blocks they dominate in this order); the entry block and any
            // block with no recorded dominator start fresh.
            let mut state = cfg
                .get_immediate_dominator(block_id)
                .and_then(|parent| state_at.get(&parent).cloned())
                .unwrap_or_default();
            visit(block_id, &mut state);
            state_at.insert(block_id, state);
        }
    }

    fn remove_ptrs(
        &self,
        function: &mut HLFunction,
        cfg: &CFG,
        phi_map: &HashMap<BlockId, Vec<(ValueId, ValueId)>>,
        fid: FunctionId,
        points_to: &PointsTo,
        promotable: &HashSet<ValueId>,
    ) {
        // The threaded-value map is keyed on the *promoted alloc* (the abstract object's identity),
        // not the syntactic pointer — a Store/Load ptr may be a load-derived ref or a singleton
        // phi whose points-to is exactly `{alloc}` but which is not literally the alloc result.
        //
        // Traverse in domination pre-order (via `propagate_in_domination_order`): for each ptr the
        // last value is then already defined when we enter the block — either defined by some
        // dominator (inherited as the `values` seed) or carried in as a phi parameter.
        let mut value_replacements = ValueReplacements::new();

        Self::propagate_in_domination_order(
            cfg,
            |block_id, values: &mut HashMap<ValueId, ValueId>| {
                // Add phi parameters (keyed on the promoted alloc the phi carries)
                for (param, alloc) in phi_map.get(&block_id).unwrap_or(&vec![]) {
                    values.insert(*alloc, *param);
                }

                let instructions = function.get_block_mut(block_id).take_instructions();
                let mut new_instructions = Vec::new();

                for mut instruction in instructions {
                    // `&instruction` is borrowed only for this match; the borrow ends before the
                    // fall-through keep path rewrites and pushes the instruction. Each promoted
                    // Store/Load resolves its alloc exactly once.
                    match instruction.as_ref() {
                        // A promoted alloc's defining instruction is dropped; a non-promoted alloc
                        // (escaping, ref-pointee, or aliased) falls through and is kept verbatim.
                        OpCode::Alloc { result, value } if promotable.contains(result) => {
                            // Thread the alloc's initial value; a later store overwrites it.
                            values.insert(*result, *value);
                            continue;
                        }
                        // A Store whose ptr resolves to a promoted alloc threads its value;
                        // otherwise it stays a real instruction (the pass is now partial).
                        OpCode::Store { ptr, value } => {
                            if let Some(alloc) =
                                Self::resolved_alloc(points_to, fid, *ptr, promotable)
                            {
                                values.insert(alloc, *value);
                                continue;
                            }
                        }
                        // A Load whose ptr resolves to a promoted alloc is replaced by the threaded
                        // value; otherwise it stays a real instruction.
                        OpCode::Load { result, ptr } => {
                            if let Some(alloc) =
                                Self::resolved_alloc(points_to, fid, *ptr, promotable)
                            {
                                let replacement = values.get(&alloc).expect(
                                "ICE: promoted alloc read with no threaded value (an uninitialized \
                                 read should have been pruned by `uninitialized_allocs`)",
                            );
                                value_replacements.insert(*result, *replacement);
                                continue;
                            }
                        }
                        _ => {}
                    }
                    // A non-promoted access or any other instruction is kept, with operands
                    // rewritten.
                    value_replacements.replace_instruction(&mut instruction);
                    new_instructions.push(instruction);
                }

                function
                    .get_block_mut(block_id)
                    .put_instructions(new_instructions);

                let mut terminator = function.get_block_mut(block_id).take_terminator().unwrap();
                value_replacements.replace_terminator(&mut terminator);

                match &mut terminator {
                    Terminator::Jmp(tgt, params) => {
                        let tmp = vec![];
                        let additional_params = phi_map.get(tgt).unwrap_or(&tmp);
                        for (_, val) in additional_params {
                            let param_val = values.get(val).unwrap_or_else(|| {
                                panic!("ICE: block {} has no value for v{}", block_id.0, val.0)
                            });
                            params.push(value_replacements.get_replacement(*param_val));
                        }
                    }
                    Terminator::JmpIf(_cond, t1, t2) => {
                        if phi_map.contains_key(t1) {
                            let jumper = function.add_block();
                            let params = phi_map
                                .get(t1)
                                .unwrap()
                                .iter()
                                .map(|(_, val)| {
                                    let v = *values.get(val).unwrap_or_else(|| {
                                        panic!(
                                            "ICE: block {} has no value for v{}",
                                            block_id.0, val.0
                                        )
                                    });
                                    value_replacements.get_replacement(v)
                                })
                                .collect::<Vec<_>>();
                            function.terminate_block_with_jmp(jumper, *t1, params);
                            *t1 = jumper;
                        }
                        if phi_map.contains_key(t2) {
                            let jumper = function.add_block();
                            let params = phi_map
                                .get(t2)
                                .unwrap()
                                .iter()
                                .map(|(_, val)| {
                                    let v = *values.get(val).unwrap_or_else(|| {
                                        panic!(
                                            "ICE: block {} has no value for v{}",
                                            block_id.0, val.0
                                        )
                                    });
                                    value_replacements.get_replacement(v)
                                })
                                .collect::<Vec<_>>();
                            function.terminate_block_with_jmp(jumper, *t2, params);
                            *t2 = jumper;
                        }
                    }
                    _ => {}
                }
                function.get_block_mut(block_id).set_terminator(terminator);
            },
        );
    }

    // For each block, returns the vector of (param_id, value_id), where param_id is the id of a new
    // parameter, and value_id is the id of the pointer that is being replaced.
    fn initialize_phis(
        &self,
        fb: &mut HLFunctionBuilder<'_>,
        phi_blocks: &HashMap<ValueId, HashSet<BlockId>>,
        type_info: &FunctionTypeInfo,
    ) -> HashMap<BlockId, Vec<(ValueId, ValueId)>> {
        let mut result: HashMap<BlockId, Vec<(ValueId, ValueId)>> = HashMap::default();
        for (value, blocks) in phi_blocks {
            for block in blocks {
                let param = fb
                    .block(*block)
                    .add_parameter(type_info.get_value_type(*value).get_pointed());
                result.entry(*block).or_default().push((param, *value));
            }
        }
        result
    }

    // Both maps are keyed on the *promoted alloc* (the abstract object), not the syntactic pointer:
    // a write through any ptr that points to exactly a promoted alloc contributes a write *for that
    // object*, so phis are placed for the object rather than for whatever value spelled the
    // pointer.
    fn find_pointer_writes_and_defs(
        &self,
        function: &HLFunction,
        fid: FunctionId,
        points_to: &PointsTo,
        promotable: &HashSet<ValueId>,
    ) -> (
        HashMap<ValueId, HashSet<BlockId>>,
        HashMap<ValueId, BlockId>,
    ) {
        let mut writes: HashMap<ValueId, HashSet<BlockId>> = HashMap::default();
        let mut defs: HashMap<ValueId, BlockId> = HashMap::default();
        for (block_id, block) in function.get_blocks() {
            for instruction in block.get_instructions() {
                match instruction {
                    OpCode::Store { ptr: lhs, value: _ } => {
                        if let Some(alloc) = Self::resolved_alloc(points_to, fid, *lhs, promotable)
                        {
                            writes.entry(alloc).or_default().insert(*block_id);
                        }
                    }
                    OpCode::Alloc {
                        result: lhs,
                        value: _,
                    } if promotable.contains(lhs) => {
                        defs.insert(*lhs, *block_id);
                        writes.entry(*lhs).or_default().insert(*block_id);
                    }
                    _ => {}
                }
            }
        }
        (writes, defs)
    }

    fn find_phi_blocks(
        &self,
        writes: &HashMap<ValueId, HashSet<BlockId>>,
        defs: &HashMap<ValueId, BlockId>,
        cfg: &CFG,
    ) -> HashMap<ValueId, HashSet<BlockId>> {
        let mut result: HashMap<ValueId, HashSet<BlockId>> = HashMap::default();

        for (var, writes) in writes {
            let mut queue = VecDeque::<BlockId>::new();
            let mut visited = HashSet::<BlockId>::default();
            queue.extend(writes);

            while let Some(block) = queue.pop_front() {
                if visited.contains(&block) {
                    continue;
                }
                visited.insert(block);

                for new_block in cfg.get_dominance_frontier(block) {
                    if !cfg.dominates(*defs.get(var).unwrap(), new_block) {
                        continue; // If the pointer is not defined here, there's no need for a phi.
                    }
                    debug!(
                        "Block {}\tneeds phi for v{}\tbecause it's in the dominance frontier of {}\twhich contains a write",
                        new_block.0, var.0, block.0
                    );
                    result.entry(*var).or_default().insert(new_block);
                    queue.push_back(new_block);
                }
            }
        }

        result
    }

    /// The promotable allocations that would be *read before written* on some path — those for
    /// which [`Self::remove_ptrs`] would find no threaded value and panic.
    ///
    /// It mirrors `remove_ptrs`'s domination-order propagation exactly, but tracks a per-block set
    /// of allocations whose value is *available* (defined by a dominating store or carried in by a
    /// phi parameter) rather than the values themselves. So it flags precisely the allocations that
    /// would otherwise hit the `expect` in `remove_ptrs` — both at a `Load` and when filling a
    /// successor phi's argument. The frontend forbids reading an uninitialized local, so this is
    /// normally empty; it is a graceful guard against an ICE, not an expected path.
    fn uninitialized_allocs(
        &self,
        function: &HLFunction,
        cfg: &CFG,
        phi_blocks: &HashMap<ValueId, HashSet<BlockId>>,
        fid: FunctionId,
        points_to: &PointsTo,
        promotable: &HashSet<ValueId>,
    ) -> HashSet<ValueId> {
        // Invert the per-alloc phi placement into a per-block view: which allocations gain an
        // available value (a phi parameter) on entry to each block.
        let mut phi_at: HashMap<BlockId, HashSet<ValueId>> = HashMap::default();
        for (alloc, blocks) in phi_blocks {
            for block in blocks {
                phi_at.entry(*block).or_default().insert(*alloc);
            }
        }

        // Shares `remove_ptrs`'s domination-order propagation (`propagate_in_domination_order`):
        // `available` (the per-block state) is the set of allocs with a defined value — the
        // analogue of `remove_ptrs`'s threaded-value map domain. An alloc read while absent (or
        // owed to a successor phi while absent) is a read-before-write that would make `remove_ptrs`
        // hit its "no threaded value" panic, so it is flagged here instead.
        let mut uninitialized: HashSet<ValueId> = HashSet::default();

        Self::propagate_in_domination_order(cfg, |block_id, available: &mut HashSet<ValueId>| {
            // This block's own phi parameters make their allocs available on entry.
            if let Some(allocs) = phi_at.get(&block_id) {
                available.extend(allocs.iter().copied());
            }

            for instruction in function.get_block(block_id).get_instructions() {
                match instruction {
                    OpCode::Store { ptr, .. } => {
                        if let Some(alloc) = Self::resolved_alloc(points_to, fid, *ptr, promotable)
                        {
                            available.insert(alloc);
                        }
                    }
                    OpCode::Alloc { result, .. } => {
                        // An `Alloc` carries its initial value, so the cell is defined (available)
                        // at the allocation site — mirroring the write recorded for phi placement.
                        // Without this, a promotable alloc whose only initializing write is the
                        // folded init value (no separate `Store`) would be wrongly flagged as a
                        // read-before-write.
                        if promotable.contains(result) {
                            available.insert(*result);
                        }
                    }
                    OpCode::Load { ptr, .. } => {
                        if let Some(alloc) = Self::resolved_alloc(points_to, fid, *ptr, promotable)
                        {
                            if !available.contains(&alloc) {
                                uninitialized.insert(alloc);
                            }
                        }
                    }
                    _ => {}
                }
            }

            // Jumping into a phi block consumes a value for each of the target's phi allocations —
            // the same `values.get(..)` that `remove_ptrs` performs when filling jump arguments.
            let mut targets: Vec<BlockId> = Vec::new();
            match function.get_block(block_id).get_terminator() {
                Some(Terminator::Jmp(target, _)) => targets.push(*target),
                Some(Terminator::JmpIf(_, t1, t2)) => {
                    targets.push(*t1);
                    targets.push(*t2);
                }
                _ => {}
            }
            for target in targets {
                if let Some(allocs) = phi_at.get(&target) {
                    for alloc in allocs {
                        if !available.contains(alloc) {
                            uninitialized.insert(*alloc);
                        }
                    }
                }
            }
        });

        uninitialized
    }

    /// The set of local allocations (by their result `ValueId`) that can be promoted to SSA values.
    ///
    /// Provides a precise, per-`Alloc` predicate driven by the points-to analysis. An allocation
    /// `a` is promotable iff:
    ///
    /// 1. its pointee is scalar (no nested ref) — promotion threads a pointee-typed value through
    ///    block parameters, and a ref-bearing pointee would mint ref-typed phis the analysis never
    ///    saw (a later extension, after array SROA);
    /// 2. it does not escape ([`PointsTo::escapes`]);
    /// 3. *every* `Store`/`Load` whose pointer may point to `a` points to *only* `a` — so all of
    ///    `a`'s accesses are unambiguous and a strong (kill) SSA update is sound; and
    /// 4. the ref value of `a` (and of anything that may point to `a`) is used *only* as the pointer
    ///    operand of a `Load`/`Store` — never consumed as a call argument, a return value, a stored
    ///    value, an array element, etc. Otherwise removing `a`'s `Alloc` would dangle that use.
    ///
    /// Conditions 3 and 4 are quantified over accesses/uses that *may touch* `a`: a Store/Load
    /// through a pointer whose points-to is `{a, b}` (a multi-object deref) disqualifies *both*;
    /// and a non-pointer use of any value that may point to `a` disqualifies `a`. In the
    /// scalar-pointee scope of this increment, condition 4 implies no value other than `a` itself
    /// can point to `a`, so every promoted access's pointer *is* `a` — but the machinery is kept
    /// object-keyed for robustness and future scope growth.
    fn promotable_allocs(
        &self,
        function: &HLFunction,
        type_info: &FunctionTypeInfo,
        fid: FunctionId,
        points_to: &PointsTo,
    ) -> HashSet<ValueId> {
        // Candidates: scalar-pointee, non-escaping local allocations (conditions 1 and 2).
        let mut candidates: HashSet<ValueId> = HashSet::default();
        for (_, block) in function.get_blocks() {
            for instruction in block.get_instructions() {
                if let OpCode::Alloc { result, .. } = instruction {
                    if type_info
                        .get_value_type(*result)
                        .get_pointed()
                        .contains_ptrs()
                    {
                        continue; // ref-bearing pointee — out of scope for scalar promotion
                    }
                    let object = AbstractObject::Alloc(fid, *result, Context::empty());
                    if points_to.escapes(&object) {
                        continue;
                    }
                    candidates.insert(*result);
                }
            }
        }

        // Disqualify candidates by aliased/opaque access (condition 3) or by any non-pointer use of
        // a value that may point to them (condition 4), via the shared pointer-use classifier. Both
        // an ambiguous deref (`Deref`) and a non-pointer use (`Consume`) disqualify every candidate
        // the use may touch; the `Write` direction is irrelevant here (mem2reg, unlike
        // arg_promotion, does not need the out-direction).
        let mut disqualified: HashSet<ValueId> = HashSet::default();
        points_to.classify_pointer_uses(fid, function, |kind, pts| {
            if matches!(kind, PointerUse::Deref | PointerUse::Consume) {
                for o in pts {
                    if let Some(a) = Self::local_alloc(o, fid) {
                        if candidates.contains(&a) {
                            disqualified.insert(a);
                        }
                    }
                }
            }
        });

        candidates.retain(|a| !disqualified.contains(a));
        candidates
    }

    /// If `ptr` points to exactly one promoted allocation, that allocation's result `ValueId`.
    ///
    /// Total and unambiguous on exactly the accesses [`Self::promotable_allocs`] kept: every
    /// Store/Load that may touch a promoted alloc points to *only* it, so it resolves; an access
    /// through a multi-object, opaque, or non-promoted pointer resolves to `None` and is left as a
    /// real instruction.
    fn resolved_alloc(
        points_to: &PointsTo,
        fid: FunctionId,
        ptr: ValueId,
        promotable: &HashSet<ValueId>,
    ) -> Option<ValueId> {
        let pts = points_to.points_to(fid, ptr);
        if pts.len() != 1 {
            return None;
        }
        Self::local_alloc(pts.iter().next().unwrap(), fid).filter(|a| promotable.contains(a))
    }

    /// The result `ValueId` of a local `Alloc` object in function `fid`, if `o` is one.
    fn local_alloc(o: &AbstractObject, fid: FunctionId) -> Option<ValueId> {
        match o {
            AbstractObject::Alloc(f, a, _) if *f == fid => Some(*a),
            _ => None,
        }
    }
}

// TESTS
// ================================================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::{
        analysis::types::Types,
        ssa::hlssa::{Type, builder::HLEmitter},
        util::test::{falloc, fr},
    };

    /// Build the analyses on the pre-pass IR and run mem2reg, mirroring the pass-manager wiring.
    fn run_pass(ssa: &mut HLSSA) {
        let flow = FlowAnalysis::run(ssa);
        let types = Types::new().run(ssa, &flow);
        let pt = PointsTo::run(ssa, &flow, &types);
        Mem2Reg::new().do_run(ssa, &flow, &types, &pt);
    }

    /// `(allocs, stores, loads)` remaining in a function body.
    fn op_counts(ssa: &HLSSA, fid: FunctionId) -> (usize, usize, usize) {
        let f = ssa.get_function(fid);
        let (mut allocs, mut stores, mut loads) = (0, 0, 0);
        for (_, block) in f.get_blocks() {
            for inst in block.get_instructions() {
                match inst {
                    OpCode::Alloc { .. } => allocs += 1,
                    OpCode::Store { .. } => stores += 1,
                    OpCode::Load { .. } => loads += 1,
                    _ => {}
                }
            }
        }
        (allocs, stores, loads)
    }

    /// A ref returned from the function escapes (old gate would veto the whole function), but a
    /// sibling local stays clean — so the local is promoted while the returned one is retained.
    #[test]
    fn escaping_ref_does_not_block_sibling_local() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let main_id = ssa.get_unique_entrypoint_id();
        {
            let mut sb = HLSSABuilder::new(&mut ssa);
            sb.modify_function(main_id, |b| {
                b.function.add_return_type(Type::field().ref_of());
                let entry = b.function.get_entry_id();
                let mut e = b.test_block(entry);
                let kept = falloc(&mut e);
                let returned = falloc(&mut e);
                let c = e.field_const(fr(1));
                e.store(kept, c);
                e.store(returned, c);
                e.terminate_return(vec![returned]);
            });
        }
        run_pass(&mut ssa);
        // `kept` promoted (alloc + store gone); `returned` escapes and is retained.
        assert_eq!(op_counts(&ssa, main_id), (1, 1, 0));
    }

    /// A ref passed to a (non-leaking) callee is consumed as a call argument, so it cannot be
    /// promoted — but its presence no longer vetoes a clean sibling local, which is promoted.
    #[test]
    fn call_with_ref_arg_does_not_block_sibling_local() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let main_id = ssa.get_unique_entrypoint_id();
        {
            let mut sb = HLSSABuilder::new(&mut ssa);
            let reader = sb.ssa().add_function("reader".to_string());
            // reader(p): let _ = *p; return
            sb.modify_function(reader, |b| {
                let entry = b.function.get_entry_id();
                let mut e = b.test_block(entry);
                let p = e.add_parameter(Type::field().ref_of());
                let _ = e.load(p);
                e.terminate_return(vec![]);
            });
            // main(): other = alloc; *other = c; reader(other); kept = alloc; *kept = c; return
            // *kept
            sb.modify_function(main_id, |b| {
                b.function.add_return_type(Type::field());
                let entry = b.function.get_entry_id();
                let mut e = b.test_block(entry);
                let other = falloc(&mut e);
                let c = e.field_const(fr(2));
                e.store(other, c);
                e.call(reader, vec![other], 0);
                let kept = falloc(&mut e);
                e.store(kept, c);
                let v = e.load(kept);
                e.terminate_return(vec![v]);
            });
        }
        run_pass(&mut ssa);
        // `kept` promoted; `other` retained (alloc + store) alongside the surviving call.
        assert_eq!(op_counts(&ssa, main_id), (1, 1, 0));
    }

    /// A function with a ref parameter (old gate would veto it) still has its clean local promoted;
    /// the load through the parameter is retained.
    #[test]
    fn ref_param_function_promotes_clean_local() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let main_id = ssa.get_unique_entrypoint_id();
        let with_ref_param;
        {
            let mut sb = HLSSABuilder::new(&mut ssa);
            let f = sb.ssa().add_function("with_ref_param".to_string());
            with_ref_param = f;
            // with_ref_param(p: Ref<Field>) -> Field: a = alloc; *a = 5; let _ = *p; return *a
            sb.modify_function(f, |b| {
                b.function.add_return_type(Type::field());
                let entry = b.function.get_entry_id();
                let mut e = b.test_block(entry);
                let p = e.add_parameter(Type::field().ref_of());
                let a = falloc(&mut e);
                let c = e.field_const(fr(5));
                e.store(a, c);
                let _ = e.load(p);
                let v = e.load(a);
                e.terminate_return(vec![v]);
            });
            sb.modify_function(main_id, |b| {
                let entry = b.function.get_entry_id();
                let mut e = b.test_block(entry);
                e.terminate_return(vec![]);
            });
        }
        run_pass(&mut ssa);
        // `a` promoted (its alloc/store/load gone); only the `*p` load remains.
        assert_eq!(op_counts(&ssa, with_ref_param), (0, 0, 1));
    }

    /// Two distinct refs merged through a block-parameter phi are a multi-object deref: neither is
    /// promotable (a strong update is unsound), so both allocations and the load are retained.
    #[test]
    fn multi_object_deref_blocks_promotion() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let main_id = ssa.get_unique_entrypoint_id();
        {
            let mut sb = HLSSABuilder::new(&mut ssa);
            sb.modify_function(main_id, |b| {
                b.function.add_return_type(Type::field());
                let entry = b.function.get_entry_id();
                let mut e = b.test_block(entry);
                let cond = e.add_parameter(Type::bool());
                let ra = falloc(&mut e);
                let rb = falloc(&mut e);
                let c = e.field_const(fr(5));
                e.store(ra, c);
                e.store(rb, c);
                let merged = e.build_if_else(
                    cond,
                    vec![Type::field().ref_of()],
                    |_| vec![ra],
                    |_| vec![rb],
                )[0];
                let r = e.load(merged);
                e.terminate_return(vec![r]);
            });
        }
        run_pass(&mut ssa);
        assert_eq!(op_counts(&ssa, main_id), (2, 2, 1));
    }

    /// A clean local whose pointee value differs across control flow is promoted: all memory
    /// traffic is removed and a block-parameter phi carries the merged value (if no phi were
    /// placed, `remove_ptrs` would panic on an uninitialized pointer value).
    #[test]
    fn control_flow_value_merge_promotes_with_phi() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let main_id = ssa.get_unique_entrypoint_id();
        {
            let mut sb = HLSSABuilder::new(&mut ssa);
            sb.modify_function(main_id, |b| {
                b.function.add_return_type(Type::field());
                let entry = b.function.get_entry_id();
                let mut e = b.test_block(entry);
                let cond = e.add_parameter(Type::bool());
                let a = falloc(&mut e);
                let c0 = e.field_const(fr(7));
                let c1 = e.field_const(fr(9));
                e.store(a, c0);
                e.build_if_else(
                    cond,
                    vec![],
                    |then| {
                        then.store(a, c1);
                        vec![]
                    },
                    |_| vec![],
                );
                let v = e.load(a);
                e.terminate_return(vec![v]);
            });
        }
        run_pass(&mut ssa);
        assert_eq!(op_counts(&ssa, main_id), (0, 0, 0));
    }

    /// A ref stored *into* another ref is consumed as a stored value (old gate would veto the
    /// function), so it is not promotable — but a clean sibling local still is.
    #[test]
    fn ref_stored_as_value_does_not_block_sibling_local() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let main_id = ssa.get_unique_entrypoint_id();
        {
            let mut sb = HLSSABuilder::new(&mut ssa);
            sb.modify_function(main_id, |b| {
                b.function.add_return_type(Type::field());
                let entry = b.function.get_entry_id();
                let mut e = b.test_block(entry);
                let a = falloc(&mut e);
                let pp = e.alloc(a); // Ref<Ref<Field>>, ref pointee, seeded with `a`
                e.store(pp, a); // *pp = a — `a`'s ref is consumed as a value
                let kept = falloc(&mut e);
                let c = e.field_const(fr(3));
                e.store(kept, c);
                let v = e.load(kept);
                e.terminate_return(vec![v]);
            });
        }
        run_pass(&mut ssa);
        // `kept` promoted; `pp` (ref pointee) and `a` (stored value) and `*pp = a` retained.
        assert_eq!(op_counts(&ssa, main_id), (2, 1, 0));
    }

    /// Post-#255 an `Alloc` carries its initial value, so a local whose only initializing write is
    /// that folded init value (no separate `Store`) is *not* a read-before-write: it promotes to
    /// the init value, threading it to the load. Exercises the alloc-seeds-the-value path with no
    /// store. (`uninitialized_allocs` therefore never fires here; the guard remains defensive.)
    #[test]
    fn alloc_init_value_promotes_without_store() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let main_id = ssa.get_unique_entrypoint_id();
        {
            let mut sb = HLSSABuilder::new(&mut ssa);
            sb.modify_function(main_id, |b| {
                b.function.add_return_type(Type::field());
                let entry = b.function.get_entry_id();
                let mut e = b.test_block(entry);
                let a = falloc(&mut e); // cell seeded with its init value, never stored to
                let v = e.load(a); // reads the alloc's initial value
                e.terminate_return(vec![v]);
            });
        }
        run_pass(&mut ssa);
        // `a` promotes to its init value: alloc and load both removed.
        assert_eq!(op_counts(&ssa, main_id), (0, 0, 0));
    }

    /// An aggregate (array) pointee is in scope: `contains_ptrs(Array<Field, N>)` is false, so a
    /// clean local `Ref<Array<Field, N>>` whose value differs across control flow is promoted — all
    /// memory traffic removed, with a *whole-array* block-parameter phi carrying the merged value.
    /// Exercises the aggregate-typed phi/threading path the scalar tests never reach.
    #[test]
    fn array_pointee_promotes_with_aggregate_phi() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let main_id = ssa.get_unique_entrypoint_id();
        let arr_ty = Type::field().array_of(3);
        {
            let mut sb = HLSSABuilder::new(&mut ssa);
            sb.modify_function(main_id, |b| {
                b.function.add_return_type(arr_ty.clone());
                let entry = b.function.get_entry_id();
                let mut e = b.test_block(entry);
                let cond = e.add_parameter(Type::bool());
                let arr0 = e.add_parameter(arr_ty.clone());
                let arr1 = e.add_parameter(arr_ty.clone());
                let a = e.alloc(arr0); // seed the cell with arr0 (the unconditional first store folded into the alloc)
                e.build_if_else(
                    cond,
                    vec![],
                    |then| {
                        then.store(a, arr1);
                        vec![]
                    },
                    |_| vec![],
                );
                let v = e.load(a);
                e.terminate_return(vec![v]);
            });
        }
        run_pass(&mut ssa);
        // `a` promoted: alloc/stores/load all gone, merged array carried by a block-parameter phi.
        assert_eq!(op_counts(&ssa, main_id), (0, 0, 0));
    }
}
