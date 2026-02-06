use std::collections::{HashMap, HashSet, VecDeque};

use crate::compiler::{constraint_solver::ConstraintSolver, ir::r#type::Empty, ssa::{FunctionId, OpCode, SSA}, taint_analysis::{ConstantTaint, FunctionTaint, Taint, TaintAnalysis, TaintType}};

#[derive(Eq, Hash, PartialEq, Clone, Debug)]
struct Signature {
    cfg_taint: Taint,
    param_taints: Vec<TaintType>,
    return_taints: Vec<TaintType>,
}

#[derive(Debug)]
struct WorkItem {
    function_id: FunctionId,
    target_function_id: FunctionId,
    signature: Signature,
}

pub struct Monomorphization {
    queue: VecDeque<WorkItem>,
    signature_map: HashMap<(FunctionId, Signature), FunctionId>,
}

impl Monomorphization {
    pub fn new() -> Self {
        Self {
            queue: VecDeque::new(),
            signature_map: HashMap::new(),
        }
    }

    pub fn run(&mut self, ssa: &mut SSA<Empty>, taint_analysis: &mut TaintAnalysis) -> Result<(), String> {
        let unspecialized_fns = ssa.get_function_ids().collect::<Vec<_>>();
        // Track functions called as unconstrained - these shouldn't be removed
        let mut unconstrained_fns: HashSet<FunctionId> = HashSet::new();

        let entry_point = ssa.get_main_id();
        let entry_point_taint = taint_analysis.get_function_taint(entry_point);
        let entry_point_signature = self.monomorphize_main_signature(entry_point_taint);
        let main_specialized_id =
            self.request_specialization(ssa, entry_point, entry_point_signature);
        ssa.set_entry_point(main_specialized_id);

        while let Some(work_item) = self.queue.pop_front() {
            // println!("Processing work item: {:?}", work_item);

            let WorkItem {
                function_id,
                target_function_id,
                signature,
            } = work_item;

            let function_taint = taint_analysis.get_function_taint(function_id);

            let mut constraint_solver = ConstraintSolver::new(&function_taint);
            constraint_solver.add_assumption(
                &TaintType::Primitive(function_taint.cfg_taint.clone()),
                &TaintType::Primitive(signature.cfg_taint.clone()),
            );
            for (original_param, specialized_param) in signature
                .param_taints
                .iter()
                .zip(function_taint.parameters.iter())
            {
                constraint_solver.add_assumption(original_param, specialized_param);
            }

            for (original_return, specialized_return) in signature
                .return_taints
                .iter()
                .zip(function_taint.returns_taint.iter())
            {
                constraint_solver.add_assumption(original_return, specialized_return);
            }

            constraint_solver.solve();
            let target_function_taint = function_taint.update_from_unification(&constraint_solver.unification);
            taint_analysis.set_function_taint(target_function_id, target_function_taint);
            let fn_taint = taint_analysis.get_function_taint(target_function_id);

            let mut func = ssa.take_function(target_function_id);

            for (block_id, block) in func.get_blocks_mut() {
                for instruction in block.get_instructions_mut() {
                    match instruction {
                        OpCode::Call { results: returns, function: func_id, args, is_unconstrained } => {
                            // Skip specialization for unconstrained calls - they run in Brillig VM
                            // and don't need taint-based specialization.
                            if *is_unconstrained {
                                unconstrained_fns.insert(*func_id);
                                continue;
                            }

                            let cfg_taint = fn_taint.block_cfg_taints.get(block_id).unwrap();
                            let args_taints = args.iter().map(|arg| fn_taint.value_taints.get(arg).unwrap().clone()).collect();
                            let ret_taints = returns.iter().map(|arg| fn_taint.value_taints.get(arg).unwrap().clone()).collect();
                            let signature = Signature {
                                cfg_taint: cfg_taint.clone(),
                                param_taints: args_taints,
                                return_taints: ret_taints,
                            };

                            let specialized_func_id = self.request_specialization(ssa, *func_id, signature);
                            *func_id = specialized_func_id;
                        }
                        _ => {}
                    }
                }
            }

            ssa.put_function(target_function_id, func);
        }

        for fn_id in unspecialized_fns {
            // Keep unconstrained functions - they're still referenced by unconstrained calls
            // but solve their constraints with all-Pure assumptions
            if unconstrained_fns.contains(&fn_id) {
                let function_taint = taint_analysis.get_function_taint(fn_id);
                let pure_taint = TaintType::Primitive(Taint::Constant(ConstantTaint::Pure));

                // Create a signature with all Pure taints
                let pure_signature = Signature {
                    cfg_taint: Taint::Constant(ConstantTaint::Pure),
                    param_taints: function_taint.parameters.iter().map(|_| pure_taint.clone()).collect(),
                    return_taints: function_taint.returns_taint.iter().map(|_| pure_taint.clone()).collect(),
                };

                // Solve constraints with Pure assumptions
                let mut constraint_solver = ConstraintSolver::new(&function_taint);
                constraint_solver.add_assumption(
                    &TaintType::Primitive(function_taint.cfg_taint.clone()),
                    &TaintType::Primitive(pure_signature.cfg_taint.clone()),
                );
                for (original_param, specialized_param) in pure_signature
                    .param_taints
                    .iter()
                    .zip(function_taint.parameters.iter())
                {
                    constraint_solver.add_assumption(original_param, specialized_param);
                }
                for (original_return, specialized_return) in pure_signature
                    .return_taints
                    .iter()
                    .zip(function_taint.returns_taint.iter())
                {
                    constraint_solver.add_assumption(original_return, specialized_return);
                }

                constraint_solver.solve();
                let resolved_taint = function_taint.update_from_unification(&constraint_solver.unification);
                taint_analysis.set_function_taint(fn_id, resolved_taint);
                continue;
            }
            ssa.take_function(fn_id);
            taint_analysis.remove_function_taint(fn_id);
        }

        Ok(())
    }

    fn request_specialization(
        &mut self,
        ssa: &mut SSA<Empty>,
        function_id: FunctionId,
        signature: Signature,
    ) -> FunctionId {
        if let Some(specialized_id) = self.signature_map.get(&(function_id, signature.clone())) {
            return *specialized_id;
        }

        let original_function = ssa.get_function(function_id);
        let specialized_function = original_function.clone();
        let specialized_id = ssa.insert_function(specialized_function);
        self.signature_map
            .insert((function_id, signature.clone()), specialized_id);
        self.queue.push_back(WorkItem {
            function_id,
            target_function_id: specialized_id,
            signature,
        });
        specialized_id
    }

    fn monomorphize_main_signature(&self, taint: &FunctionTaint) -> Signature {
        Signature {
            cfg_taint: taint.cfg_taint.clone(),
            param_taints: taint
                .parameters
                .iter()
                .map(|p| self.monomorphize_main_taint(p))
                .collect(),
            return_taints: taint
                .returns_taint
                .iter()
                .map(|p| self.monomorphize_main_taint(p))
                .collect(),
        }
    }

    fn monomorphize_main_taint(&self, taint: &TaintType) -> TaintType {
        match taint {
            TaintType::Primitive(_) => {
                TaintType::Primitive(Taint::Constant(ConstantTaint::Witness))
            }
            TaintType::NestedImmutable(_, inner) => TaintType::NestedImmutable(
                Taint::Constant(ConstantTaint::Pure),
                Box::new(self.monomorphize_main_taint(inner)),
            ),
            TaintType::Tuple(_, child_taints) => TaintType::Tuple(
                Taint::Constant(ConstantTaint::Pure), 
                child_taints.iter().map(|child_taint| self.monomorphize_main_taint(child_taint)).collect()
            ),
            _ => panic!("Pointer in main signature: {:?}", taint),
        }
    }
}
