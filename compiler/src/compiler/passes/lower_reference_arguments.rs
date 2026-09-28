//! Materialize reference arguments at constrained-to-unconstrained call boundaries.
//!
//! Witness representation conversions operate on values, not references. Load each reference
//! before crossing the boundary and pass its value to a wrapper which reconstructs a local
//! reference before calling the original function. Keep the original signature for ordinary
//! calls, where references (including mutable aliases) must retain their identity.
//!
//! Run after defunctionalization and before witness inference. Noir rejects mutable and nested
//! references at this boundary; this pass handles its supported immutable reference arguments.

use crate::{
    collections::HashMap,
    compiler::{
        pass_manager::{AnalysisStore, Pass},
        ssa::{
            FunctionId, SourceLocation,
            hlssa::{
                CallTarget, HLSSA, OpCode, TypeExpr,
                builder::{HLEmitter, HLSSABuilder},
            },
        },
    },
};

pub struct LowerReferenceArguments;

impl Pass for LowerReferenceArguments {
    fn name(&self) -> &'static str {
        "lower_reference_arguments"
    }

    fn run(&self, ssa: &mut HLSSA, _store: &AnalysisStore) {
        // Snapshot the original functions: generated wrappers contain only ordinary calls.
        let callers: Vec<_> = ssa.get_function_ids().collect();
        let mut wrappers: HashMap<FunctionId, (FunctionId, Vec<bool>)> = HashMap::default();
        for caller in &callers {
            let callees: Vec<_> = ssa
                .get_function(*caller)
                .get_blocks()
                .flat_map(|(_, block)| block.get_instructions())
                .filter_map(|instruction| match instruction {
                    OpCode::Call {
                        function: CallTarget::Static(callee),
                        unconstrained: true,
                        ..
                    } => Some(*callee),
                    _ => None,
                })
                .collect();
            for callee in callees {
                if wrappers.contains_key(&callee) {
                    continue;
                }
                let function = ssa.get_function(callee);
                let params = function.get_param_types();
                let refs: Vec<_> = params
                    .iter()
                    .map(|ty| matches!(ty.expr, TypeExpr::Ref(_)))
                    .collect();
                if !refs.iter().any(|is_ref| *is_ref) {
                    continue;
                }
                let returns = function.get_returns().to_vec();
                let name = format!("{}_ref_wrapper", function.get_name());
                let (wrapper, ()) = HLSSABuilder::new(ssa).add_function(name, |fb| {
                    for ty in &returns {
                        fb.function.add_return_type(ty.clone());
                    }
                    let entry = fb.function.get_entry_id();
                    let mut e = fb
                        .block(entry)
                        .with_source_location(SourceLocation::synthetic(
                            "lower_reference_arguments",
                        ));
                    let args = params
                        .iter()
                        .map(|ty| match &ty.expr {
                            TypeExpr::Ref(inner) => {
                                let value = e.add_parameter(inner.as_ref().clone());
                                e.alloc(value)
                            }
                            _ => e.add_parameter(ty.clone()),
                        })
                        .collect();
                    let results = e.call(callee, args, returns.len());
                    e.terminate_return(results);
                });
                wrappers.insert(callee, (wrapper, refs));
            }
        }

        if wrappers.is_empty() {
            return;
        }
        let mut builder = HLSSABuilder::new(ssa);
        for caller in callers {
            builder.modify_function(caller, |fb| {
                for (_, block) in fb.function.get_blocks_mut() {
                    let mut instructions = Vec::new();
                    for mut instruction in block.take_instructions() {
                        let location = instruction.location().clone();
                        if let OpCode::Call {
                            function: CallTarget::Static(callee),
                            args,
                            unconstrained: true,
                            ..
                        } = &mut *instruction
                            && let Some((wrapper, refs)) = wrappers.get(callee)
                        {
                            assert_eq!(args.len(), refs.len());
                            for (arg, is_ref) in args.iter_mut().zip(refs) {
                                if *is_ref {
                                    let result = fb.ssa.fresh_value();
                                    instructions.push(
                                        OpCode::Load { result, ptr: *arg }.locate(location.clone()),
                                    );
                                    *arg = result;
                                }
                            }
                            *callee = *wrapper;
                        }
                        instructions.push(instruction);
                    }
                    block.put_instructions(instructions);
                }
            });
        }
    }
}
