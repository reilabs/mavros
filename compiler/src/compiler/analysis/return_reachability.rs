//! Conservative return reachability, before DCE can discard unused unconstrained calls.
//! The least fixed point admits a function only when some CFG path returns using callees
//! already admitted. Recursive functions with a base case are admitted; unconditional call
//! cycles are not. Unknown conditions, dynamic calls and guarded calls may return.

use crate::{
    collections::HashSet,
    compiler::ssa::{
        FunctionId, Terminator,
        hlssa::{CallTarget, HLSSA, OpCode},
    },
};

pub fn entry_can_return(ssa: &HLSSA) -> bool {
    let mut returning = HashSet::default();
    loop {
        let before = returning.len();
        for (id, function) in ssa.iter_functions() {
            if returning.contains(id) {
                continue;
            }
            let mut pending = vec![function.get_entry_id()];
            let mut seen = HashSet::default();
            while let Some(block) = pending.pop() {
                if !seen.insert(block) {
                    continue;
                }
                let block = function.get_block(block);
                if block.get_instructions().any(|op| {
                    matches!(op,
                    OpCode::Call { function: CallTarget::Static(callee), .. }
                    if !returning.contains(callee))
                }) {
                    continue;
                }
                match block.get_terminator() {
                    Some(Terminator::Return(_)) => {
                        returning.insert(*id);
                        break;
                    }
                    Some(Terminator::Jmp(target, _)) => pending.push(*target),
                    Some(Terminator::JmpIf(_, a, b)) => pending.extend([*a, *b]),
                    None => {
                        returning.insert(*id);
                        break;
                    } // Incomplete IR is not a proof.
                }
            }
        }
        if returning.len() == before {
            return ssa
                .get_entry_points()
                .iter()
                .all(|id: &FunctionId| returning.contains(id));
        }
    }
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
