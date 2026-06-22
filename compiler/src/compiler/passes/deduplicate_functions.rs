//! Deduplicates structurally-identical functions in the SSA.
//!
//! Monomorphization, defunctionalization, and the witgen/AD split all tend to emit many copies of
//! "the same" function: bodies that differ only in their name, in the absolute value/block ids the
//! id allocator happened to hand out, and in source locations. Codegen then emits a distinct
//! routine for each, bloating the bytecode and the WASM module.
//!
//! # Approach (LLVM `MergeFunctions`)
//!
//! This follows the design of LLVM's `MergeFunctions` pass (see
//! <https://llvm.org/docs/MergeFunctions.html>): functions are bucketed by a structural key, an
//! exact comparison confirms members of a bucket are genuinely identical, one representative is
//! kept per equivalence class, and all references to the others are redirected to it before they
//! are deleted.
//!
//! Two refinements from the literature that matter here:
//!
//! * **Canonical key.** Absolute `ValueId`s and `BlockId`s are local fictions of the id allocator,
//!   so two equal functions disagree on them. We build the key by renumbering a *clone* of the
//!   function into a canonical namespace — blocks in CFG pre-order, then each value (block
//!   parameter / instruction result) in first-definition order — and rendering the result with the
//!   existing IR printer (which already encodes every opcode's payload faithfully). Constants are
//!   addressed by globally-unique, value-interned ids, so they are *left untouched*: identical
//!   constant values keep identical ids and therefore compare equal, while different values do not.
//!   Source locations and the function name are dropped from the key. Because the printer is the
//!   canonical, lossless textual form of the IR, equal keys imply semantic equality.
//!
//! * **Fixpoint.** Call targets are part of the key, so two functions that differ only because they
//!   call two *other* functions that are themselves equal will not match until those callees have
//!   been merged. We therefore iterate to a fixpoint (LLVM's "Deferred" set serves the same
//!   purpose); each round strictly shrinks the function set, so this terminates.
//!
//! Entry points and the globals init/deinit functions are never deleted — they are referenced
//! out-of-band (the entry-offset table, the globals hooks) rather than through call instructions —
//! so they are always chosen as the representative of their class and never redirected away.
//!
//! This runs just before codegen, after the witgen and AD halves have been merged into the single
//! multi-entry program and the CFG has been cleaned up, which is where the duplication is densest.

use crate::{
    collections::{HashMap, HashSet},
    compiler::{
        pass_manager::{AnalysisStore, Pass},
        ssa::{
            BlockId, DefaultSSAAnnotator, FunctionId, Instruction, Terminator, ValueId,
            hlssa::{HLFunction, HLSSA},
        },
    },
};

/// Base offset for canonical local value ids. Chosen far above any id the allocator could
/// realistically mint so that renumbered locals (params/results) never collide with the global ids
/// of constants, which we deliberately leave untouched. A program would need ~10^18 values to reach
/// this, which is impossible in practice.
const CANON_VALUE_BASE: u64 = 1 << 60;

pub struct DeduplicateFunctions {}

impl DeduplicateFunctions {
    pub fn new() -> Self {
        Self {}
    }
}

impl Default for DeduplicateFunctions {
    fn default() -> Self {
        Self::new()
    }
}

impl Pass for DeduplicateFunctions {
    fn name(&self) -> &'static str {
        "deduplicate_functions"
    }

    fn run(&self, ssa: &mut HLSSA, _store: &AnalysisStore) {
        let mut total_removed = 0usize;
        loop {
            let removed = Self::run_once(ssa);
            if removed == 0 {
                break;
            }
            total_removed += removed;
        }
        if total_removed > 0 {
            tracing::info!(
                message = %"Deduplicated identical functions",
                removed = total_removed
            );
        }
    }
}

impl DeduplicateFunctions {
    /// Performs a single merge round. Returns the number of functions removed (zero once a fixpoint
    /// is reached).
    fn run_once(ssa: &mut HLSSA) -> usize {
        // Functions that must survive: they are referenced out-of-band rather than via calls, so
        // deleting (or redirecting) them would break codegen.
        let mut protected: HashSet<FunctionId> =
            ssa.get_entry_points().iter().copied().collect();
        if let Some(g) = ssa.get_globals_init_fn() {
            protected.insert(g);
        }
        if let Some(g) = ssa.get_globals_deinit_fn() {
            protected.insert(g);
        }

        // Bucket every function by its canonical key. Iterate in id order so each bucket lists its
        // members in ascending id order (making representative selection deterministic).
        let mut ids: Vec<FunctionId> = ssa.get_function_ids().collect();
        ids.sort_by_key(|f| f.0);

        let mut buckets: HashMap<String, Vec<FunctionId>> = HashMap::default();
        for fid in &ids {
            let key = Self::canonical_key(ssa, *fid);
            buckets.entry(key).or_default().push(*fid);
        }

        // For each class of >= 2 functions, keep one representative and map the rest onto it. Prefer
        // a protected function as representative, and never schedule a protected function for
        // deletion.
        let mut replacement: HashMap<FunctionId, FunctionId> = HashMap::default();
        for group in buckets.values() {
            if group.len() < 2 {
                continue;
            }
            let keep = group
                .iter()
                .copied()
                .filter(|f| protected.contains(f))
                .min_by_key(|f| f.0)
                .unwrap_or_else(|| group.iter().copied().min_by_key(|f| f.0).unwrap());
            for &member in group {
                if member == keep || protected.contains(&member) {
                    continue;
                }
                replacement.insert(member, keep);
            }
        }

        if replacement.is_empty() {
            return 0;
        }

        // Redirect every static call target onto its representative across the whole program...
        for (_, function) in ssa.iter_functions_mut() {
            for (_, block) in function.get_blocks_mut() {
                for instruction in block.get_instructions_mut() {
                    instruction
                        .map_call_targets(&mut |callee| replacement.get(&callee).copied().unwrap_or(callee));
                }
            }
        }

        // ...then drop the now-unreferenced duplicates.
        for merged in replacement.keys() {
            ssa.delete_function(*merged);
        }

        replacement.len()
    }

    /// Builds a canonical, location- and name-independent textual key for a function.
    ///
    /// Equal keys imply the functions are structurally identical up to renaming of local values and
    /// blocks (constants and call targets compare by identity).
    fn canonical_key(ssa: &HLSSA, function_id: FunctionId) -> String {
        let mut function = ssa.get_function(function_id).clone();

        let order = Self::block_order(&function);
        let block_index: HashMap<BlockId, u64> = order
            .iter()
            .enumerate()
            .map(|(i, b)| (*b, i as u64))
            .collect();
        let value_map = Self::value_renumbering(&function, &order);

        function.set_name(String::new());

        let blocks = function.take_blocks();
        let mut new_blocks = HashMap::default();
        for (old_block_id, mut block) in blocks {
            // Drop source locations: re-wrapping the payloads clears them.
            let ops = block.take_instructions();
            block.put_instructions(ops);

            // Renumber every value reference into the canonical namespace.
            for (v, _) in block.get_parameters_mut() {
                Self::remap_value(v, &value_map);
            }
            for instruction in block.get_instructions_mut() {
                for operand in instruction.get_operands_mut() {
                    Self::remap_value(operand, &value_map);
                }
            }
            if block.get_terminator().is_some() {
                match block.get_terminator_mut() {
                    Terminator::Jmp(target, args) => {
                        for v in args.iter_mut() {
                            Self::remap_value(v, &value_map);
                        }
                        *target = BlockId(block_index[target]);
                    }
                    Terminator::JmpIf(cond, true_block, false_block) => {
                        Self::remap_value(cond, &value_map);
                        *true_block = BlockId(block_index[true_block]);
                        *false_block = BlockId(block_index[false_block]);
                    }
                    Terminator::Return(values) => {
                        for v in values.iter_mut() {
                            Self::remap_value(v, &value_map);
                        }
                    }
                }
            }

            new_blocks.insert(BlockId(block_index[&old_block_id]), block);
        }
        function.put_blocks(new_blocks);
        function.set_entry_block(BlockId(block_index[&function.get_entry_id()]));

        // Render with the canonical printer. Call targets are keyed purely by their (already
        // redirected) function id; the function id we pass for the header is fixed so it does not
        // perturb the key.
        let func_name = |f: FunctionId| f.0.to_string();
        function.to_string(&func_name, FunctionId(0), &DefaultSSAAnnotator)
    }

    /// Maps a value id through the canonical renumbering, leaving constants (and any value that is
    /// not a local definition) untouched.
    fn remap_value(value: &mut ValueId, value_map: &HashMap<ValueId, u64>) {
        if let Some(canonical) = value_map.get(value) {
            *value = ValueId(*canonical);
        }
    }

    /// Visits blocks in CFG pre-order from the entry, falling back to any unreachable blocks in id
    /// order so the traversal is total and deterministic.
    fn block_order(function: &HLFunction) -> Vec<BlockId> {
        let mut order = Vec::new();
        let mut seen: HashSet<BlockId> = HashSet::default();
        let mut stack = vec![function.get_entry_id()];
        while let Some(block_id) = stack.pop() {
            if !seen.insert(block_id) {
                continue;
            }
            order.push(block_id);
            // Push successors reversed so the leftmost edge is visited first.
            for succ in Self::successors(function, block_id).into_iter().rev() {
                stack.push(succ);
            }
        }

        let mut leftover: Vec<BlockId> = function
            .get_blocks()
            .map(|(b, _)| *b)
            .filter(|b| !seen.contains(b))
            .collect();
        leftover.sort_by_key(|b| b.0);
        order.extend(leftover);
        order
    }

    fn successors(function: &HLFunction, block_id: BlockId) -> Vec<BlockId> {
        match function.get_block(block_id).get_terminator() {
            Some(Terminator::Jmp(target, _)) => vec![*target],
            Some(Terminator::JmpIf(_, true_block, false_block)) => vec![*true_block, *false_block],
            Some(Terminator::Return(_)) | None => vec![],
        }
    }

    /// Assigns canonical ids to every local definition (block parameters, then instruction results)
    /// in canonical block order. Constants are not definitions and are intentionally absent.
    fn value_renumbering(
        function: &HLFunction,
        order: &[BlockId],
    ) -> HashMap<ValueId, u64> {
        let mut value_map: HashMap<ValueId, u64> = HashMap::default();
        let mut next = 0u64;
        let assign = |v: ValueId, value_map: &mut HashMap<ValueId, u64>, next: &mut u64| {
            if !value_map.contains_key(&v) {
                value_map.insert(v, CANON_VALUE_BASE + *next);
                *next += 1;
            }
        };
        for block_id in order {
            let block = function.get_block(*block_id);
            for (v, _) in block.get_parameters() {
                assign(*v, &mut value_map, &mut next);
            }
            for instruction in block.get_instructions() {
                for result in instruction.get_results() {
                    assign(*result, &mut value_map, &mut next);
                }
            }
        }
        value_map
    }
}
