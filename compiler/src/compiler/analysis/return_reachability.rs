//! Conservative return reachability, before DCE can discard unused unconstrained calls.
//!
//! The least fixed point admits a function only when some CFG path returns using callees
//! already admitted. Recursive functions with a base case are admitted; unconditional call
//! cycles are not. Unknown conditions, dynamic calls and guarded calls may return.
//!
//! Known gap: this proves only that an entry has no returning path. A conditionally reached
//! non-returning call is accepted if another path returns, even when runtime inputs select
//! the non-returning branch. It is not a termination proof for every execution.

use crate::{
    collections::HashSet,
    compiler::{
        analysis::{flow_analysis::FlowAnalysis, shared::fixpoint::call_graph_fixpoint},
        diagnostic::Diagnostic,
        ssa::{
            SourceLocation, Terminator,
            hlssa::{CallTarget, HLSSA, OpCode},
        },
    },
};

pub(crate) fn non_returning_entries(ssa: &HLSSA, flow: &FlowAnalysis) -> Vec<Diagnostic> {
    let functions: Vec<_> = ssa.get_function_ids().collect();
    let returning = call_graph_fixpoint(
        ssa,
        flow,
        &functions,
        |_| false,
        |id, returning| {
            let function = ssa.get_function(id);
            let mut pending = vec![function.get_entry_id()];
            let mut seen = HashSet::default();
            while let Some(block) = pending.pop() {
                if !seen.insert(block) {
                    continue;
                }
                let block = function.get_block(block);
                if block.get_instructions().any(|op| match op {
                    OpCode::Call {
                        function: CallTarget::Static(callee),
                        ..
                    } => !returning.get(callee).copied().unwrap_or(true),
                    _ => false,
                }) {
                    continue;
                }
                match block.get_terminator() {
                    Some(Terminator::Return(_)) => return true,
                    // Incomplete IR is not a proof of non-return.
                    None => return true,
                    Some(Terminator::Jmp(target, _)) => pending.push(*target),
                    Some(Terminator::JmpIf(_, a, b)) => pending.extend([*a, *b]),
                }
            }
            false
        },
    );
    ssa.get_entry_points()
        .iter()
        .filter(|id| !returning[id])
        .map(|id| {
            let function = ssa.get_function(*id);
            let mut fallback = None;
            let mut call_location = None;
            for block in flow.get_function_cfg(*id).get_domination_pre_order() {
                for (op, location) in function
                    .get_block(block)
                    .get_instructions_with_source_locations()
                {
                    fallback.get_or_insert_with(|| location.clone());
                    if matches!(op, OpCode::Call { function: CallTarget::Static(callee), .. }
                    if !returning.get(callee).copied().unwrap_or(true))
                    {
                        call_location = Some(location.clone());
                        break;
                    }
                }
                if call_location.is_some() {
                    break;
                }
            }
            // Terminators have no source spans. An instruction-free CFG loop can only use a
            // synthetic location; source recursion reports the offending call's actual span.
            Diagnostic::error(
                "entry point has no returning path (unconditional recursion or loop)",
                call_location
                    .or(fallback)
                    .unwrap_or_else(|| SourceLocation::synthetic("return_reachability")),
            )
            .with_note(format!("entry point: {}", function.get_name()))
        })
        .collect()
}

#[cfg(test)]
fn entry_can_return(ssa: &HLSSA) -> bool {
    non_returning_entries(ssa, &FlowAnalysis::run(ssa)).is_empty()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::ssa::hlssa::{
        Type,
        builder::{HLEmitter, HLSSABuilder},
    };

    #[test]
    fn recursive_cycles_need_a_returning_path() {
        for base_case in [false, true] {
            let mut ssa = HLSSA::new();
            let main = ssa.get_unique_entrypoint_id();
            let callee = ssa.add_function("callee".into());
            let mut builder = HLSSABuilder::new(&mut ssa);
            builder.modify_function(main, |b| {
                let entry = b.function.get_entry_id();
                let mut e = b.test_block(entry);
                let cond = e.add_parameter(Type::int(1));
                e.call_unconstrained(callee, vec![cond], 0);
                e.terminate_return(vec![]);
            });
            builder.modify_function(callee, |b| {
                let entry = b.function.get_entry_id();
                let cond = b.test_block(entry).add_parameter(Type::int(1));
                let recurse = b.add_block(|_| {});
                {
                    let mut e = b.test_block(recurse);
                    e.call_unconstrained(main, vec![cond], 0);
                    e.terminate_return(vec![]);
                }
                let done = b.add_block(|e| e.terminate_return(vec![]));
                let mut e = b.test_block(entry);
                if base_case {
                    e.terminate_jmp_if(cond, recurse, done);
                } else {
                    e.terminate_jmp(recurse, vec![]);
                }
            });
            assert_eq!(entry_can_return(&ssa), base_case);
        }
    }

    #[test]
    fn an_unreachable_recursive_function_does_not_reject_main() {
        let mut ssa = HLSSA::new();
        let main = ssa.get_unique_entrypoint_id();
        let dead = ssa.add_function("dead".into());
        let mut builder = HLSSABuilder::new(&mut ssa);
        builder.modify_function(main, |b| {
            let entry = b.function.get_entry_id();
            b.test_block(entry).terminate_return(vec![]);
        });
        builder.modify_function(dead, |b| {
            let entry = b.function.get_entry_id();
            let mut e = b.test_block(entry);
            e.call_unconstrained(dead, vec![], 0);
            e.terminate_return(vec![]);
        });
        assert!(entry_can_return(&ssa));
    }
}
