//! Merges structurally identical functions into one, rewriting every call site to point to the
//! single survivor and deleting the now dead functions.
//!
//! # Why this pass exists
//!
//! The final program SSA is the witgen program and the AD program merged into one multi-entry
//! SSA (`Driver::prepare_program_ssa`), so many functions exist twice — most dramatically the
//! generated `globals_init`, whose clone alone accounts for up to half of the bytecode of
//! global-heavy programs. Both codegen backends (bytecode and LLSSA lowering) emit _every_ function
//! in the SSA, so the duplicates must be physically removed, not merely left unreachable.
//!
//! # Identity
//!
//! Two functions are identical when their canonical serializations agree and their callees are
//! pairwise in the same group (bisimilar, not necessarily byte-identical — see below). The
//! canonical form renames each function-local `ValueId` to a fresh id assigned in first-encounter
//! order over a block-id-sorted traversal, while _constant_ ids stay raw as they have shared
//! identifiers program-wide. `BlockId`s are function-local and never remapped by the merge, so they
//! compare raw.
//!
//! Function _names_ are excluded (clones get renamed, e.g. `ad_main`); source locations are
//! deliberately _included_ — assert and constraint failures report locations, so this pass only
//! folds clones that are byte-equivalent including diagnostics. Location-blind folding (merging
//! user-written duplicates from different source points) is a possible follow-up knob.
//!
//! Callee identity uses partition refinement: functions are first grouped by their canonical form
//! with call targets abstracted, then groups are repeatedly split by the group ids of their callees
//! until stable. Refinement only ever splits, so it terminates; the surviving co-grouping is a
//! bisimulation — members call same-group callees at every site — which is what makes merging
//! chains of clones (and identical self-recursive pairs) sound.
//!
//! # Preconditions
//!
//! The pass rewrites only `CallTarget::Static` references (via `map_call_targets`); it does not
//! touch `CallTarget::Dynamic` or `Constant::FnPtr(FunctionId)`. So it may only run once
//! defunctionalization has removed both — otherwise deleting a folded function could leave a
//! dangling function-pointer reference. This holds in `program_tail`: both halves are
//! defunctionalized before the program merge (`driver.rs` asserts no `FnPtr` constant survives).
//! `do_run` re-checks this with a debug-only assertion so a future pipeline reorder fails loudly
//! rather than miscompiling.
//!
//! # Soundness
//!
//! Merging never changes behavior: a call to the deleted copy becomes a call to a function with
//! an identical body, and the *call itself* is preserved, so per-call effects (witness minting,
//! constraint emission, globals initialization) happen exactly as before. Entry points are
//! externally invoked by id and are never deleted; they are also never used as a redirect _target_,
//! because the LLVM/WASM backend gives entry points a distinct calling convention (declared
//! `fn(VM*)`, with parameters loaded from the public-input region rather than passed as arguments),
//! so an internal call into an entry would mismatch its signature. Redirect targets are therefore
//! always non-entry survivors: a group's entry points survive on their own, and a non-entry
//! duplicate with no non-entry survivor to fold into is kept rather than pointed at an entry.
//!
//! The pass operates on the program SSA built by `Driver::prepare_program_ssa`, which R1CS
//! generation does not consume (R1CS is generated from a separate `witness_spilled_ssa`), so
//! rows/cols cannot change — the payoff is program size (bytecode and WASM) and downstream
//! compile time.

use crate::{
    collections::{HashMap, HashSet},
    compiler::{
        pass_manager::{AnalysisStore, Pass},
        ssa::{
            Function, FunctionId, Instruction, Terminator, ValueId,
            hlssa::{HLSSA, OpCode, Type},
        },
    },
};

// MERGE IDENTICAL FUNCTIONS
// ================================================================================================

#[derive(Default)]
pub struct MergeIdenticalFunctions {}

impl Pass for MergeIdenticalFunctions {
    fn name(&self) -> &'static str {
        "merge_identical_functions"
    }

    fn run(&self, ssa: &mut HLSSA, _store: &AnalysisStore) {
        self.do_run(ssa);
    }
}

impl MergeIdenticalFunctions {
    pub fn new() -> Self {
        Self {}
    }

    pub fn do_run(&self, ssa: &mut HLSSA) {
        // Precondition: defunctionalization must have run, so no `Constant::FnPtr` (nor the dynamic
        // calls it feeds) remains. This pass only rewrites `CallTarget::Static`, so a surviving
        // function pointer to a folded function would be left dangling. `driver.rs` guarantees this
        // before the program merge; re-check it here so a future pipeline reorder fails loudly.
        #[cfg(debug_assertions)]
        {
            use crate::compiler::ssa::hlssa::Constant;
            fn contains_fn_ptr(c: &Constant) -> bool {
                match c {
                    Constant::FnPtr(_) => true,
                    Constant::Blob(blob) => blob.elements.iter().any(contains_fn_ptr),
                    Constant::U(..) | Constant::I(..) | Constant::Field(_) => false,
                }
            }
            let mut has_fn_ptr = false;
            ssa.for_each_const(|_, cv| has_fn_ptr = has_fn_ptr || contains_fn_ptr(cv.as_ref()));
            assert!(
                !has_fn_ptr,
                "merge_identical_functions requires defunctionalization first: a FnPtr constant \
                 survived, which this pass would leave dangling when folding its target"
            );
        }

        // Constant ids are program-global and shared across the merged halves; they stay raw in the
        // canonical form while every other id is renamed positionally.
        let mut const_ids: HashSet<ValueId> = HashSet::default();
        ssa.for_each_const(|id, _| {
            const_ids.insert(*id);
        });

        let mut fids: Vec<FunctionId> = ssa.get_function_ids().collect();
        fids.sort_unstable_by_key(|fid| fid.0);

        let mut shapes: Vec<(FunctionId, String, Vec<FunctionId>)> = Vec::new();
        for &fid in &fids {
            let (shape, callees) = canonical_form(ssa.get_function(fid), &const_ids);
            shapes.push((fid, shape, callees));
        }

        // Round 0: group by canonical form with call targets abstracted. Refinement: split groups
        // on the group ids of their callees until the partition stops changing.
        let mut group: HashMap<FunctionId, usize> = HashMap::default();
        let mut group_count = {
            let mut intern: HashMap<&str, usize> = HashMap::default();
            for (fid, shape, _) in &shapes {
                let next = intern.len();
                let id = *intern.entry(shape.as_str()).or_insert(next);
                group.insert(*fid, id);
            }
            intern.len()
        };

        // The shape strings are only needed for round-0 grouping; refinement and the rewrite below
        // need just each function's callee list. Drop the strings now to release O(program-text)
        // memory before the fixpoint. (A digest instead of exact strings would risk a collision =
        // an unsound false merge, so grouping keeps the full strings for round 0.)
        let callee_lists: Vec<(FunctionId, Vec<FunctionId>)> = shapes
            .into_iter()
            .map(|(fid, _shape, callees)| (fid, callees))
            .collect();

        loop {
            let mut intern: HashMap<(usize, Vec<usize>), usize> = HashMap::default();
            let mut next_group: HashMap<FunctionId, usize> = HashMap::default();
            for (fid, callees) in &callee_lists {
                let key = (group[fid], callees.iter().map(|c| group[c]).collect());
                let next = intern.len();
                let id = *intern.entry(key).or_insert(next);
                next_group.insert(*fid, id);
            }
            let stable = intern.len() == group_count;
            group_count = intern.len();
            group = next_group;
            if stable {
                break;
            }
        }

        // Per final group: entry points always survive on their own; every non-entry duplicate
        // redirects to the smallest-id *non-entry* survivor and is deleted. An entry point is never
        // used as a redirect target — the LLVM/WASM backend compiles entries with a distinct
        // calling convention (declared `fn(VM*)`, parameters loaded from the public-input region
        // rather than passed as arguments), so an internal call into one would mismatch its
        // signature. When a group has no non-entry member, or its only non-entry is the survivor,
        // nothing folds there.
        let entry_points: HashSet<FunctionId> = ssa.get_entry_points().iter().copied().collect();
        let mut members: HashMap<usize, Vec<FunctionId>> = HashMap::default();
        for &fid in &fids {
            members.entry(group[&fid]).or_default().push(fid);
        }

        // One survivor per group: the smallest-id non-entry member, else (all-entry group) the
        // smallest-id member. `members` values are in sorted-`fids` order, so `find`/`first` yield
        // the smallest id and this stays deterministic regardless of map iteration order.
        let mut survivor_of: HashMap<usize, FunctionId> = HashMap::default();
        for (&gid, mates) in &members {
            let survivor = mates
                .iter()
                .find(|m| !entry_points.contains(m))
                .or_else(|| mates.first())
                .copied()
                .unwrap();
            survivor_of.insert(gid, survivor);
        }

        let mut redirect: HashMap<FunctionId, FunctionId> = HashMap::default();
        for &fid in &fids {
            if entry_points.contains(&fid) {
                continue;
            }
            // `survivor` is guaranteed non-entry: `fid` is non-entry, so its group has a non-entry
            // member and `find` picked one.
            let survivor = survivor_of[&group[&fid]];
            if fid != survivor {
                redirect.insert(fid, survivor);
            }
        }
        if redirect.is_empty() {
            return;
        }

        // The SSA-level globals init/deinit registrations hold `FunctionId`s too; keep them
        // pointing at survivors if their functions get folded.
        if let Some(survivor) = ssa.get_globals_init_fn().and_then(|f| redirect.get(&f)) {
            ssa.set_globals_init_fn(*survivor);
        }
        if let Some(survivor) = ssa.get_globals_deinit_fn().and_then(|f| redirect.get(&f)) {
            ssa.set_globals_deinit_fn(*survivor);
        }

        for (fid, callees) in &callee_lists {
            if redirect.contains_key(fid) {
                ssa.delete_function(*fid)
                    .expect("function scheduled for merging must exist");
                continue;
            }
            if !callees.iter().any(|callee| redirect.contains_key(callee)) {
                continue;
            }
            let function = ssa.get_function_mut(*fid);
            for (_, block) in function.get_blocks_mut() {
                for instruction in block.get_instructions_mut() {
                    instruction
                        .map_call_targets(&mut |callee| *redirect.get(&callee).unwrap_or(&callee));
                }
            }
        }

        // Postcondition (debug only): the delete + rewrite above must leave no reference to a
        // folded function. The folded (deleted) set is exactly `redirect`'s keys; assert that no
        // surviving function statically calls one, and that the globals registrations don't either.
        // `delete_function` validates nothing, so this catches a future edit that forgets a
        // reference class before it silently ships a dangling call.
        #[cfg(debug_assertions)]
        {
            for (fid, _) in &callee_lists {
                if redirect.contains_key(fid) {
                    continue; // deleted above
                }
                let function = ssa.get_function(*fid);
                for (_, block) in function.get_blocks() {
                    for op in block.get_instructions() {
                        for callee in op.get_static_call_targets() {
                            debug_assert!(
                                !redirect.contains_key(&callee),
                                "merge_identical_functions left a dangling call from {fid:?} to \
                                 folded {callee:?}"
                            );
                        }
                    }
                }
            }
            if let Some(f) = ssa.get_globals_init_fn() {
                debug_assert!(
                    !redirect.contains_key(&f),
                    "merge_identical_functions left globals_init at folded {f:?}"
                );
            }
            if let Some(f) = ssa.get_globals_deinit_fn() {
                debug_assert!(
                    !redirect.contains_key(&f),
                    "merge_identical_functions left globals_deinit at folded {f:?}"
                );
            }
        }
    }
}

// CANONICALIZER
// ================================================================================================

/// A function-local `ValueId` renamer.
///
/// constants stay raw, everything else maps to its first-encounter index (in the far end of the id
/// space, so canonical ids can never collide with a real constant id).
struct Canonicalizer<'a> {
    const_ids: &'a HashSet<ValueId>,
    rename: HashMap<ValueId, ValueId>,
}

impl Canonicalizer<'_> {
    fn canon(&mut self, id: ValueId) -> ValueId {
        if self.const_ids.contains(&id) {
            return id;
        }
        let next = ValueId(u64::MAX - self.rename.len() as u64);
        *self.rename.entry(id).or_insert(next)
    }
}

/// The canonical serialization of `function` (see the module doc for what it does and does not
/// include) plus its static call targets in traversal order, which the caller abstracts through the
/// partition instead of comparing raw.
///
/// Identity is decided by exact string equality of this serialization, which relies on `Debug` being
/// injective over every opcode / type / source-location value that can appear: two *distinct* values
/// must never format to the same string, or two different functions would be merged (unsound). This
/// holds today because value-carrying immediates (constants) appear here only as their raw const
/// `ValueId`s, never as formatted values. A future opcode field or `Type` with a lossy `Debug` would
/// break it — prefer a purpose-built encoding over a lossy `Debug` if that ever arises.
fn canonical_form(
    function: &Function<OpCode, Type>,
    const_ids: &HashSet<ValueId>,
) -> (String, Vec<FunctionId>) {
    use std::fmt::Write;

    let mut canonicalizer = Canonicalizer {
        const_ids,
        rename: HashMap::default(),
    };
    let mut callees: Vec<FunctionId> = Vec::new();

    let mut shape = format!(
        "entry {:?} returns {:?}",
        function.get_entry_id(),
        function.get_returns()
    );

    let mut block_ids: Vec<_> = function.get_blocks().map(|(id, _)| *id).collect();
    block_ids.sort_unstable_by_key(|id| id.0);
    for block_id in block_ids {
        let block = function.get_block(block_id);
        write!(shape, "\nblock {:?} params [", block_id).unwrap();
        for (param, typ) in block.get_parameters() {
            write!(shape, "{:?}: {:?}, ", canonicalizer.canon(*param), typ).unwrap();
        }
        shape.push(']');

        for (op, location) in block.get_instructions_with_source_locations() {
            let mut op = op.clone();
            op.map_call_targets(&mut |callee| {
                callees.push(callee);
                FunctionId(u64::MAX)
            });
            for input in op.get_inputs_mut() {
                *input = canonicalizer.canon(*input);
            }
            for result in op.get_results_mut() {
                *result = canonicalizer.canon(*result);
            }
            write!(shape, "\n{op:?} @ {location:?}").unwrap();
        }

        match block.get_terminator() {
            Some(Terminator::Jmp(target, args)) => {
                let args: Vec<ValueId> = args.iter().map(|v| canonicalizer.canon(*v)).collect();
                write!(shape, "\njmp {target:?} {args:?}").unwrap();
            }
            Some(Terminator::JmpIf(cond, then_target, else_target)) => {
                let cond = canonicalizer.canon(*cond);
                write!(shape, "\njmp_if {cond:?} {then_target:?} {else_target:?}").unwrap();
            }
            Some(Terminator::Return(values)) => {
                let values: Vec<ValueId> = values.iter().map(|v| canonicalizer.canon(*v)).collect();
                write!(shape, "\nreturn {values:?}").unwrap();
            }
            None => shape.push_str("\nno terminator"),
        }
    }

    (shape, callees)
}

// TESTS
// ================================================================================================

#[cfg(test)]
mod tests {
    use super::MergeIdenticalFunctions;
    use crate::compiler::Field;
    use crate::compiler::ssa::{
        FunctionId, Instruction, Located, SourceLocation, SourcePosition, ValueId,
        hlssa::{CallTarget, Constant, HLSSA, OpCode, Type, builder::HLSSABuilder},
    };

    /// All static call targets in `fid`'s body, in block-id-sorted stream order.
    fn call_targets(ssa: &HLSSA, fid: FunctionId) -> Vec<FunctionId> {
        let function = ssa.get_function(fid);
        let mut block_ids: Vec<_> = function.get_blocks().map(|(id, _)| *id).collect();
        block_ids.sort_unstable_by_key(|id| id.0);
        let mut targets = Vec::new();
        for block_id in block_ids {
            for op in function.get_block(block_id).get_instructions() {
                targets.extend(op.get_static_call_targets());
            }
        }
        targets
    }

    fn has_function(ssa: &HLSSA, fid: FunctionId) -> bool {
        ssa.get_function_ids().any(|id| id == fid)
    }

    /// A leaf `fn(x: Field) -> Field { return !x }`.
    fn add_leaf(sb: &mut HLSSABuilder, name: &str) -> FunctionId {
        let fid = sb.ssa().add_function(name.to_string());
        sb.modify_function(fid, |fb| {
            fb.function.add_return_type(Type::field());
            let entry = fb.function.get_entry_id();
            let result = fb.fresh_value();
            let mut block = fb.test_block(entry);
            let x = block.add_parameter(Type::field());
            block.emit_instruction(OpCode::Not { result, value: x });
            block.terminate_return(vec![result]);
        });
        fid
    }

    /// Gives `fid` the body `fn(x: Field) -> Field { return callee(x) }`.
    fn set_caller_body(sb: &mut HLSSABuilder, fid: FunctionId, callee: FunctionId) {
        sb.modify_function(fid, |fb| {
            fb.function.add_return_type(Type::field());
            let entry = fb.function.get_entry_id();
            let result = fb.fresh_value();
            let mut block = fb.test_block(entry);
            let x = block.add_parameter(Type::field());
            block.emit_instruction(OpCode::Call {
                results: vec![result],
                function: CallTarget::Static(callee),
                args: vec![x],
                unconstrained: false,
            });
            block.terminate_return(vec![result]);
        });
    }

    /// A `fn(x: Field) -> Field { return callee(x) }` wrapper.
    fn add_caller(sb: &mut HLSSABuilder, name: &str, callee: FunctionId) -> FunctionId {
        let fid = sb.ssa().add_function(name.to_string());
        set_caller_body(sb, fid, callee);
        fid
    }

    /// Point `main` at the given callees (one call each) and return main's id.
    fn wire_main(sb: &mut HLSSABuilder, callees: &[FunctionId]) -> FunctionId {
        let main_id = sb.ssa().get_unique_entrypoint_id();
        sb.modify_function(main_id, |fb| {
            let entry = fb.function.get_entry_id();
            let results: Vec<ValueId> = callees.iter().map(|_| fb.fresh_value()).collect();
            // FIELD-ASSUMPTION: L1-direct-ref (2 sites)
            let arg = fb.emit_const(Constant::Field(Field::from(7u64)));
            let mut block = fb.test_block(entry);
            for (callee, result) in callees.iter().zip(results) {
                block.emit_instruction(OpCode::Call {
                    results: vec![result],
                    function: CallTarget::Static(*callee),
                    args: vec![arg],
                    unconstrained: false,
                });
            }
            block.terminate_return(vec![]);
        });
        main_id
    }

    fn run(ssa: &mut HLSSA) {
        MergeIdenticalFunctions::new().do_run(ssa);
    }

    #[test]
    fn identical_leaves_merge_to_min_id_survivor() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let mut sb = HLSSABuilder::new(&mut ssa);
        let f = add_leaf(&mut sb, "f");
        let g = add_leaf(&mut sb, "g");
        let main_id = wire_main(&mut sb, &[f, g]);

        run(&mut ssa);

        assert!(has_function(&ssa, f));
        assert!(!has_function(&ssa, g));
        assert_eq!(call_targets(&ssa, main_id), vec![f, f]);
    }

    #[test]
    fn wrappers_of_different_leaves_split_by_refinement() {
        // `c1` and `c2` are byte-identical up to the abstracted call target, so round-0 grouping
        // puts them together; only the callee-group refinement can (and must) tell them apart.
        let mut ssa = HLSSA::with_main("main".to_string());
        let mut sb = HLSSABuilder::new(&mut ssa);
        let f = add_leaf(&mut sb, "f");
        // A leaf with a different body: returns its argument unchanged.
        let h = {
            let fid = sb.ssa().add_function("h".to_string());
            sb.modify_function(fid, |fb| {
                fb.function.add_return_type(Type::field());
                let entry = fb.function.get_entry_id();
                let mut block = fb.test_block(entry);
                let x = block.add_parameter(Type::field());
                block.terminate_return(vec![x]);
            });
            fid
        };
        let c1 = add_caller(&mut sb, "c1", f);
        let c2 = add_caller(&mut sb, "c2", h);
        let main_id = wire_main(&mut sb, &[c1, c2]);

        run(&mut ssa);

        assert!(has_function(&ssa, c1) && has_function(&ssa, c2));
        assert_eq!(call_targets(&ssa, main_id), vec![c1, c2]);
        assert_eq!(call_targets(&ssa, c1), vec![f]);
        assert_eq!(call_targets(&ssa, c2), vec![h]);
    }

    #[test]
    fn caller_chain_merges_via_refinement() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let mut sb = HLSSABuilder::new(&mut ssa);
        let f1 = add_leaf(&mut sb, "f1");
        let f2 = add_leaf(&mut sb, "f2");
        let c1 = add_caller(&mut sb, "c1", f1);
        let c2 = add_caller(&mut sb, "c2", f2);
        let main_id = wire_main(&mut sb, &[c1, c2]);

        run(&mut ssa);

        assert!(has_function(&ssa, f1) && has_function(&ssa, c1));
        assert!(!has_function(&ssa, f2) && !has_function(&ssa, c2));
        assert_eq!(call_targets(&ssa, main_id), vec![c1, c1]);
        assert_eq!(call_targets(&ssa, c1), vec![f1]);
    }

    #[test]
    fn differing_constant_prevents_merge() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let mut sb = HLSSABuilder::new(&mut ssa);
        let mut leaf_with_const = |name: &str, value: u128| {
            let fid = sb.ssa().add_function(name.to_string());
            sb.modify_function(fid, |fb| {
                fb.function.add_return_type(Type::u(32));
                let entry = fb.function.get_entry_id();
                let result = fb.fresh_value();
                let c = fb.emit_const(Constant::U(32, value));
                let mut block = fb.test_block(entry);
                let _x = block.add_parameter(Type::u(32));
                block.emit_instruction(OpCode::Not { result, value: c });
                block.terminate_return(vec![result]);
            });
            fid
        };
        let f = leaf_with_const("f", 1);
        let g = leaf_with_const("g", 2);
        let main_id = wire_main(&mut sb, &[f, g]);

        run(&mut ssa);

        assert!(has_function(&ssa, f) && has_function(&ssa, g));
        assert_eq!(call_targets(&ssa, main_id), vec![f, g]);
    }

    #[test]
    fn differing_source_location_prevents_merge() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let mut sb = HLSSABuilder::new(&mut ssa);
        let mut leaf_at_line = |name: &str, line: u64| {
            let fid = sb.ssa().add_function(name.to_string());
            sb.modify_function(fid, |fb| {
                fb.function.add_return_type(Type::field());
                let entry = fb.function.get_entry_id();
                let result = fb.fresh_value();
                let mut block = fb.block(entry);
                let x = {
                    let v = block.ssa.fresh_value();
                    block.block.push_parameter(v, Type::field());
                    v
                };
                let location = SourceLocation::new(
                    "src/main.nr".to_string(),
                    SourcePosition::new(line, 1),
                    SourcePosition::new(line, 5),
                );
                block.emit_located_instruction(Located::new(
                    OpCode::Not { result, value: x },
                    location,
                ));
                block.terminate_return(vec![result]);
            });
            fid
        };
        let f = leaf_at_line("f", 3);
        let g = leaf_at_line("g", 4);
        let main_id = wire_main(&mut sb, &[f, g]);

        run(&mut ssa);

        assert!(has_function(&ssa, f) && has_function(&ssa, g));
        assert_eq!(call_targets(&ssa, main_id), vec![f, g]);
    }

    #[test]
    fn entry_point_is_never_a_redirect_target() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let mut sb = HLSSABuilder::new(&mut ssa);
        let f = add_leaf(&mut sb, "f");
        let g = add_leaf(&mut sb, "g");
        let main_id = wire_main(&mut sb, &[f]);
        ssa.add_entry_point(g);

        run(&mut ssa);

        // `f` (non-entry) and `g` (entry) are byte-identical and `main` calls `f`. Folding `f` into
        // the entry `g` would make `g` internally callable, which the LLVM / WASM backend cannot
        // honor (entries load their params from memory under a `fn(VM*)` signature). So `f` is kept
        // — an entry is never a redirect target — and `main` keeps calling `f`.
        assert!(has_function(&ssa, f) && has_function(&ssa, g));
        assert_eq!(call_targets(&ssa, main_id), vec![f]);
    }

    #[test]
    fn two_identical_entry_points_both_survive() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let mut sb = HLSSABuilder::new(&mut ssa);
        let f = add_leaf(&mut sb, "f");
        let g = add_leaf(&mut sb, "g");
        let main_id = wire_main(&mut sb, &[f, g]);
        ssa.add_entry_point(f);
        ssa.add_entry_point(g);

        run(&mut ssa);

        assert!(has_function(&ssa, f) && has_function(&ssa, g));
        assert_eq!(call_targets(&ssa, main_id), vec![f, g]);
    }

    #[test]
    fn duplicates_fold_onto_a_non_entry_survivor_not_the_entry() {
        // Three byte-identical leaves in one group: two non-entry (`a`, `b`) and one entry (`e`).
        // `b` folds onto the smaller-id non-entry `a` — never onto the entry `e`, which survives on
        // its own. `main` calls both non-entries, so both call sites end up at `a`.
        let mut ssa = HLSSA::with_main("main".to_string());
        let mut sb = HLSSABuilder::new(&mut ssa);
        let a = add_leaf(&mut sb, "a");
        let b = add_leaf(&mut sb, "b");
        let e = add_leaf(&mut sb, "e");
        let main_id = wire_main(&mut sb, &[a, b]);
        ssa.add_entry_point(e);

        run(&mut ssa);

        assert!(has_function(&ssa, a) && has_function(&ssa, e));
        assert!(!has_function(&ssa, b));
        assert_eq!(call_targets(&ssa, main_id), vec![a, a]);
    }

    #[test]
    fn identical_self_recursive_pair_merges() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let mut sb = HLSSABuilder::new(&mut ssa);
        let f = sb.ssa().add_function("f".to_string());
        let g = sb.ssa().add_function("g".to_string());
        for fid in [f, g] {
            sb.modify_function(fid, |fb| {
                fb.function.add_return_type(Type::field());
                let entry = fb.function.get_entry_id();
                let result = fb.fresh_value();
                let mut block = fb.test_block(entry);
                let x = block.add_parameter(Type::field());
                block.emit_instruction(OpCode::Call {
                    results: vec![result],
                    function: CallTarget::Static(fid),
                    args: vec![x],
                    unconstrained: false,
                });
                block.terminate_return(vec![result]);
            });
        }
        let main_id = wire_main(&mut sb, &[f, g]);

        run(&mut ssa);

        assert!(has_function(&ssa, f));
        assert!(!has_function(&ssa, g));
        assert_eq!(call_targets(&ssa, main_id), vec![f, f]);
        assert_eq!(call_targets(&ssa, f), vec![f]);
    }

    #[test]
    fn identical_mutually_recursive_pairs_merge() {
        // Two mutually recursive pairs with identical wrapper bodies. The surviving co-grouping is
        // a bisimulation, so all four collapse onto one self-recursive survivor: every body has the
        // same shape and every callee stays within the group.
        let mut ssa = HLSSA::with_main("main".to_string());
        let mut sb = HLSSABuilder::new(&mut ssa);
        let a1 = sb.ssa().add_function("a1".to_string());
        let a2 = sb.ssa().add_function("a2".to_string());
        let b1 = sb.ssa().add_function("b1".to_string());
        let b2 = sb.ssa().add_function("b2".to_string());
        for (fid, callee) in [(a1, a2), (a2, a1), (b1, b2), (b2, b1)] {
            set_caller_body(&mut sb, fid, callee);
        }
        let main_id = wire_main(&mut sb, &[a1, b1]);

        run(&mut ssa);

        assert!(has_function(&ssa, a1));
        assert!(!has_function(&ssa, a2) && !has_function(&ssa, b1) && !has_function(&ssa, b2));
        assert_eq!(call_targets(&ssa, main_id), vec![a1, a1]);
        assert_eq!(call_targets(&ssa, a1), vec![a1]);
    }

    #[test]
    fn refinement_splits_through_a_recursion_cycle() {
        // `a1 ↔ a2` mutually recurse, but `a2`'s body differs (extra `Not`); `b1 → b2 → b2` share
        // `a1`'s wrapper shape. Round 0 groups {a1, b1, b2}; refinement must split `a1` away (its
        // callee sits in the odd group) while still folding `b2` into `b1`.
        let mut ssa = HLSSA::with_main("main".to_string());
        let mut sb = HLSSABuilder::new(&mut ssa);
        let a1 = sb.ssa().add_function("a1".to_string());
        let a2 = sb.ssa().add_function("a2".to_string());
        let b1 = sb.ssa().add_function("b1".to_string());
        let b2 = sb.ssa().add_function("b2".to_string());
        set_caller_body(&mut sb, a1, a2);
        set_caller_body(&mut sb, b1, b2);
        set_caller_body(&mut sb, b2, b2);

        // a2: `fn(x) -> Field { return a1(!x) }` — a different shape than the plain wrapper.
        sb.modify_function(a2, |fb| {
            fb.function.add_return_type(Type::field());
            let entry = fb.function.get_entry_id();
            let notted = fb.fresh_value();
            let result = fb.fresh_value();
            let mut block = fb.test_block(entry);
            let x = block.add_parameter(Type::field());
            block.emit_instruction(OpCode::Not {
                result: notted,
                value: x,
            });
            block.emit_instruction(OpCode::Call {
                results: vec![result],
                function: CallTarget::Static(a1),
                args: vec![notted],
                unconstrained: false,
            });
            block.terminate_return(vec![result]);
        });
        let main_id = wire_main(&mut sb, &[a1, b1]);

        run(&mut ssa);

        assert!(has_function(&ssa, a1) && has_function(&ssa, a2) && has_function(&ssa, b1));
        assert!(!has_function(&ssa, b2));
        assert_eq!(call_targets(&ssa, main_id), vec![a1, b1]);
        assert_eq!(call_targets(&ssa, a1), vec![a2]);
        assert_eq!(call_targets(&ssa, a2), vec![a1]);
        assert_eq!(call_targets(&ssa, b1), vec![b1]);
    }

    #[test]
    fn differing_jmp_args_prevent_merge() {
        // Identical instructions, blocks, and terminator structure; the only difference is which
        // value the entry passes to the continuation block.
        let mut ssa = HLSSA::with_main("main".to_string());
        let mut sb = HLSSABuilder::new(&mut ssa);
        let mut two_block_leaf = |name: &str, pass_result: bool| {
            let fid = sb.ssa().add_function(name.to_string());
            sb.modify_function(fid, |fb| {
                fb.function.add_return_type(Type::field());
                let entry = fb.function.get_entry_id();
                let cont = fb.function.add_block();
                let result = fb.fresh_value();
                {
                    let mut block = fb.test_block(entry);
                    let x = block.add_parameter(Type::field());
                    block.emit_instruction(OpCode::Not { result, value: x });
                    let arg = if pass_result { result } else { x };
                    block.terminate_jmp(cont, vec![arg]);
                }
                let mut block = fb.test_block(cont);
                let p = block.add_parameter(Type::field());
                block.terminate_return(vec![p]);
            });
            fid
        };
        let f = two_block_leaf("f", true);
        let g = two_block_leaf("g", false);
        let h = two_block_leaf("h", true);
        let main_id = wire_main(&mut sb, &[f, g, h]);

        run(&mut ssa);

        // `h` is byte-equivalent to `f` and folds into it; `g` differs only in the jmp argument
        // and must survive.
        assert!(has_function(&ssa, f) && has_function(&ssa, g));
        assert!(!has_function(&ssa, h));
        assert_eq!(call_targets(&ssa, main_id), vec![f, g, f]);
    }

    #[test]
    fn guard_inner_values_are_canonicalized() {
        // `g`'s value ids are skewed relative to `f`'s, and the skewed ids appear only inside the
        // guard's inner op — the canonicalizer must rename through the guard to see the bodies as
        // equal.
        let mut ssa = HLSSA::with_main("main".to_string());
        let mut sb = HLSSABuilder::new(&mut ssa);
        let mut guarded_leaf = |name: &str, skew: bool| {
            let fid = sb.ssa().add_function(name.to_string());
            sb.modify_function(fid, |fb| {
                if skew {
                    let _ = fb.fresh_value();
                }
                fb.function.add_return_type(Type::field());
                let entry = fb.function.get_entry_id();
                let result = fb.fresh_value();
                let condition = fb.emit_const(Constant::U(1, 1));
                let mut block = fb.test_block(entry);
                let x = block.add_parameter(Type::field());
                block.emit_instruction(OpCode::Guard {
                    condition,
                    inner: Box::new(OpCode::Not { result, value: x }),
                });
                block.terminate_return(vec![result]);
            });
            fid
        };
        let f = guarded_leaf("f", false);
        let g = guarded_leaf("g", true);
        let main_id = wire_main(&mut sb, &[f, g]);

        run(&mut ssa);

        assert!(has_function(&ssa, f));
        assert!(!has_function(&ssa, g));
        assert_eq!(call_targets(&ssa, main_id), vec![f, f]);
    }

    #[test]
    fn globals_init_registrations_follow_redirect() {
        // The SSA-level globals init/deinit registrations must be remapped when the function they
        // name folds into a smaller-id duplicate.
        let mut ssa = HLSSA::with_main("main".to_string());
        let mut sb = HLSSABuilder::new(&mut ssa);
        let f = add_leaf(&mut sb, "f");
        let g = add_leaf(&mut sb, "globals_init");
        let main_id = wire_main(&mut sb, &[f, g]);
        ssa.set_globals_init_fn(g);
        ssa.set_globals_deinit_fn(g);

        run(&mut ssa);

        assert!(!has_function(&ssa, g));
        assert_eq!(ssa.get_globals_init_fn(), Some(f));
        assert_eq!(ssa.get_globals_deinit_fn(), Some(f));
        assert_eq!(call_targets(&ssa, main_id), vec![f, f]);
    }

    #[test]
    fn call_inside_guard_is_retargeted() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let mut sb = HLSSABuilder::new(&mut ssa);
        let f = add_leaf(&mut sb, "f");
        let g = add_leaf(&mut sb, "g");
        let main_id = sb.ssa().get_unique_entrypoint_id();
        sb.modify_function(main_id, |fb| {
            let entry = fb.function.get_entry_id();
            let result = fb.fresh_value();
            let condition = fb.emit_const(Constant::U(1, 1));
            let arg = fb.emit_const(Constant::Field(Field::from(7u64)));
            let mut block = fb.test_block(entry);
            block.emit_instruction(OpCode::Guard {
                condition,
                inner: Box::new(OpCode::Call {
                    results: vec![result],
                    function: CallTarget::Static(g),
                    args: vec![arg],
                    unconstrained: false,
                }),
            });
            block.terminate_return(vec![]);
        });

        run(&mut ssa);

        assert!(!has_function(&ssa, g));
        assert_eq!(call_targets(&ssa, main_id), vec![f]);
    }
}
