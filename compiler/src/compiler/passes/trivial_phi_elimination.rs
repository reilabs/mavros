//! Removes *trivial phis*: block parameters that receive the same value from every predecessor.
//!
//! After [`ElideTuples`](super::elide_tuples) converts an aggregate into one value (or one memory
//! cell) per leaf, threading that aggregate through control flow produces one block-parameter phi
//! per leaf, either directly, or once [`Mem2Reg`](super::mem2reg) promotes the per-leaf cells. At
//! any given merge only the few leaves that actually differ between the incoming edges carry
//! distinct values; every other leaf receives the identical value from all predecessors.
//!
//! Nothing else in the pipeline removes them: `DeduplicatePhis` only merges predecessors whose
//! *entire* argument list matches, and DCE only drops *unused* parameters — these stay live,
//! threaded straight into the next merge. Left in place they become redundant move traffic in
//! codegen, which can result in needless increases in program size.
//!
//! # Pass Mechanics
//!
//! For a parameter `p_i` whose incoming argument is the same value `V` on every predecessor edge
//! (ignoring edges that feed `p_i` straight back into itself, i.e. loop back-edges), this pass:
//!
//! - Records the substitution `p_i -> V`,
//! - Drops slot `i` from the block's parameter list and from every predecessor's `Jmp` arguments,
//! - Rewrites all uses of `p_i` to `V`.
//!
//! Collapsing one merge can expose newly-trivial phis elsewhere (the agreement only becomes visible
//! once an upstream substitution lands), so the sweep iterates to a fixpoint per function.

use crate::{
    collections::{HashMap, HashSet},
    compiler::{
        pass_manager::{AnalysisId, AnalysisStore, Pass},
        passes::fix_double_jumps::{ReplaceScope, ValueReplacements},
        ssa::{
            BlockId, Terminator, ValueId,
            hlssa::{HLFunction, HLSSA},
        },
    },
};

pub struct TrivialPhiElimination {}

impl Pass for TrivialPhiElimination {
    fn name(&self) -> &'static str {
        "trivial_phi_elimination"
    }

    fn needs(&self) -> Vec<AnalysisId> {
        // Predecessors and their arguments are read straight off the terminators, so no cached
        // CFG analysis is required.
        vec![]
    }

    fn run(&self, ssa: &mut HLSSA, _store: &AnalysisStore) {
        self.do_run(ssa);
    }
}

impl TrivialPhiElimination {
    pub fn new() -> Self {
        Self {}
    }

    pub fn do_run(&self, ssa: &mut HLSSA) {
        for (_, function) in ssa.iter_functions_mut() {
            let entry = function.get_entry_id();
            while Self::run_once(function, entry) {}
        }
    }

    /// Runs a single sweep. Returns `true` if any parameter was removed (so the caller re-runs).
    fn run_once(function: &mut HLFunction, entry: BlockId) -> bool {
        // 1. Index the `Jmp` edges by target: every parameterized block is entered only via `Jmp`
        //    (a `JmpIf` carries no arguments), so record those targets separately to assert the
        //    invariant and to stay clear of them.
        let mut incoming_args: HashMap<BlockId, Vec<Vec<ValueId>>> = HashMap::default();
        let mut jmpif_targets: HashSet<BlockId> = HashSet::default();
        for (_, block) in function.get_blocks() {
            match block.get_terminator() {
                Some(Terminator::Jmp(target, args)) => {
                    incoming_args.entry(*target).or_default().push(args.clone());
                }
                Some(Terminator::JmpIf(_, t, f)) => {
                    jmpif_targets.insert(*t);
                    jmpif_targets.insert(*f);
                }
                _ => {}
            }
        }

        // 2. For each block, find the parameter slots whose incoming value agrees across all edges.
        let mut value_replacements = ValueReplacements::new();
        let mut removed: HashMap<BlockId, HashSet<usize>> = HashMap::default();
        for (bid, block) in function.get_blocks() {
            // The entry block's parameters are supplied by the caller, not by edges, so they are
            // never phis.
            if *bid == entry {
                continue;
            }
            let params: Vec<ValueId> = block.get_parameter_values().copied().collect();
            if params.is_empty() {
                continue;
            }
            let Some(edges) = incoming_args.get(bid) else {
                continue;
            };
            debug_assert!(
                !jmpif_targets.contains(bid),
                "block_{} carries parameters but is targeted by a JmpIf (which passes no arguments)",
                bid.0
            );

            let mut slots = HashSet::default();
            for (i, &p_i) in params.iter().enumerate() {
                // The single value all predecessors agree on, or `None` while still unset. Edges
                // that pass `p_i` back into itself (loop back-edges) don't count against agreement.
                let mut unique: Option<ValueId> = None;
                let mut trivial = true;
                for args in edges {
                    let v = args[i];
                    if v == p_i {
                        continue;
                    }
                    match unique {
                        None => unique = Some(v),
                        Some(u) if u == v => {}
                        Some(_) => {
                            trivial = false;
                            break;
                        }
                    }
                }
                if trivial {
                    if let Some(v) = unique {
                        value_replacements.insert(p_i, v);
                        slots.insert(i);
                    }
                    // `unique == None` means every edge fed `p_i` to itself — a dead loop phi with
                    // no driving value. There is no value to replace it with, so leave it for DCE.
                }
            }
            if !slots.is_empty() {
                removed.insert(*bid, slots);
            }
        }

        if removed.is_empty() {
            return false;
        }

        // 3a. Shrink the parameter lists of the affected blocks.
        for (bid, slots) in &removed {
            let block = function.get_block_mut(*bid);
            let kept: Vec<_> = block
                .take_parameters()
                .into_iter()
                .enumerate()
                .filter(|(i, _)| !slots.contains(i))
                .map(|(_, p)| p)
                .collect();
            block.put_parameters(kept);
        }

        // 3b. Drop the matching argument slots from every predecessor `Jmp`.
        for (_, block) in function.get_blocks_mut() {
            if let Terminator::Jmp(target, args) = block.get_terminator_mut() {
                if let Some(slots) = removed.get(target) {
                    let kept: Vec<_> = std::mem::take(args)
                        .into_iter()
                        .enumerate()
                        .filter(|(i, _)| !slots.contains(i))
                        .map(|(_, v)| v)
                        .collect();
                    *args = kept;
                }
            }
        }

        // 4. Rewrite every remaining use of a removed parameter to the agreed-upon value. The
        //    substitution chases transitively, so chains across collapsed merges resolve in one go.
        value_replacements.apply_to_function(function, ReplaceScope::Operands);

        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::ssa::hlssa::{HLSSA, Type};

    fn field() -> Type {
        Type::field()
    }

    /// A diamond where both arms agree on one merge parameter but differ on another: the agreed
    /// parameter is collapsed, the differing one survives.
    #[test]
    fn collapses_agreeing_merge_parameter() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let (cond, shared, then_val, else_val) =
            (ValueId(10), ValueId(11), ValueId(20), ValueId(21));
        let (m_shared, m_diff) = (ValueId(30), ValueId(31));

        let f = ssa.get_unique_entrypoint_mut();
        let then_b = f.add_block();
        let else_b = f.add_block();
        let merge = f.add_block();

        // entry: branch on `cond` to the two arms.
        f.get_entry_mut()
            .set_terminator(Terminator::JmpIf(cond, then_b, else_b));
        // Both arms pass the same `shared` value, but distinct second values.
        f.get_block_mut(then_b)
            .set_terminator(Terminator::Jmp(merge, vec![shared, then_val]));
        f.get_block_mut(else_b)
            .set_terminator(Terminator::Jmp(merge, vec![shared, else_val]));
        // merge takes both, returns both.
        let merge_block = f.get_block_mut(merge);
        merge_block.push_parameter(m_shared, field());
        merge_block.push_parameter(m_diff, field());
        merge_block.set_terminator(Terminator::Return(vec![m_shared, m_diff]));

        TrivialPhiElimination::new().do_run(&mut ssa);

        let f = ssa.get_unique_entrypoint();
        // The agreeing parameter is gone; the differing one remains.
        let params: Vec<_> = f.get_block(merge).get_parameter_values().copied().collect();
        assert_eq!(params, vec![m_diff]);
        // Predecessor jumps drop the collapsed slot.
        assert!(matches!(
            f.get_block(then_b).get_terminator(),
            Some(Terminator::Jmp(t, args)) if *t == merge && *args == vec![then_val]
        ));
        assert!(matches!(
            f.get_block(else_b).get_terminator(),
            Some(Terminator::Jmp(t, args)) if *t == merge && *args == vec![else_val]
        ));
        // Uses of the collapsed parameter are rewritten to `shared`.
        assert!(matches!(
            f.get_block(merge).get_terminator(),
            Some(Terminator::Return(vals)) if *vals == vec![shared, m_diff]
        ));
    }

    /// A loop whose latch passes the header parameter straight back: the parameter is loop-invariant
    /// and collapses to its pre-header value.
    #[test]
    fn collapses_loop_invariant_parameter() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let (init, cond, param) = (ValueId(10), ValueId(11), ValueId(50));

        let f = ssa.get_unique_entrypoint_mut();
        let header = f.add_block();
        let body = f.add_block();
        let exit = f.add_block();

        // pre-header (entry) seeds the parameter with `init`.
        f.get_entry_mut()
            .set_terminator(Terminator::Jmp(header, vec![init]));
        // header carries the loop-variant parameter and branches body/exit.
        let header_block = f.get_block_mut(header);
        header_block.push_parameter(param, field());
        header_block.set_terminator(Terminator::JmpIf(cond, body, exit));
        // latch feeds the parameter straight back — no mutation.
        f.get_block_mut(body)
            .set_terminator(Terminator::Jmp(header, vec![param]));
        // exit observes the parameter.
        f.get_block_mut(exit)
            .set_terminator(Terminator::Return(vec![param]));

        TrivialPhiElimination::new().do_run(&mut ssa);

        let f = ssa.get_unique_entrypoint();
        // The self-referential parameter collapses to `init`.
        assert!(f.get_block(header).get_parameter_values().next().is_none());
        assert!(matches!(
            f.get_block(exit).get_terminator(),
            Some(Terminator::Return(vals)) if *vals == vec![init]
        ));
        // Both edges into the header lose their now-empty argument.
        assert!(matches!(
            f.get_entry().get_terminator(),
            Some(Terminator::Jmp(t, args)) if *t == header && args.is_empty()
        ));
        assert!(matches!(
            f.get_block(body).get_terminator(),
            Some(Terminator::Jmp(t, args)) if *t == header && args.is_empty()
        ));
    }

    /// A merge whose predecessors disagree on a parameter must keep it untouched.
    #[test]
    fn keeps_genuinely_distinct_parameter() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let (cond, then_val, else_val, param) =
            (ValueId(10), ValueId(20), ValueId(21), ValueId(30));

        let f = ssa.get_unique_entrypoint_mut();
        let then_b = f.add_block();
        let else_b = f.add_block();
        let merge = f.add_block();

        f.get_entry_mut()
            .set_terminator(Terminator::JmpIf(cond, then_b, else_b));
        f.get_block_mut(then_b)
            .set_terminator(Terminator::Jmp(merge, vec![then_val]));
        f.get_block_mut(else_b)
            .set_terminator(Terminator::Jmp(merge, vec![else_val]));
        let merge_block = f.get_block_mut(merge);
        merge_block.push_parameter(param, field());
        merge_block.set_terminator(Terminator::Return(vec![param]));

        TrivialPhiElimination::new().do_run(&mut ssa);

        let f = ssa.get_unique_entrypoint();
        let params: Vec<_> = f.get_block(merge).get_parameter_values().copied().collect();
        assert_eq!(params, vec![param]);
        assert!(matches!(
            f.get_block(merge).get_terminator(),
            Some(Terminator::Return(vals)) if *vals == vec![param]
        ));
    }
}
