use std::collections::{HashMap, VecDeque};

use crate::compiler::{witness_constraint_solver::WitnessConstraintSolver, ssa::{CallTarget, FunctionId, OpCode, SSA}, witness_info::{FunctionWitnessType, WitnessInfo, WitnessType}, witness_type_inference::WitnessTypeInference};

#[derive(Eq, Hash, PartialEq, Clone, Debug)]
struct Signature {
    cfg_witness: WitnessInfo,
    param_witnesses: Vec<WitnessType>,
    return_witnesses: Vec<WitnessType>,
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

    pub fn run(&mut self, ssa: &mut SSA, witness_inference: &mut WitnessTypeInference) -> Result<(), String> {
        let unspecialized_fns = ssa.get_function_ids().collect::<Vec<_>>();
        let entry_point = ssa.get_main_id();
        let entry_point_wt = witness_inference.get_function_witness_type(entry_point);
        let entry_point_signature = self.monomorphize_main_signature(entry_point_wt);
        let main_specialized_id =
            self.request_specialization(ssa, entry_point, entry_point_signature);
        ssa.set_entry_point(main_specialized_id);

        while let Some(work_item) = self.queue.pop_front() {
            let WorkItem {
                function_id,
                target_function_id,
                signature,
            } = work_item;

            let function_wt = witness_inference.get_function_witness_type(function_id);

            let mut constraint_solver = WitnessConstraintSolver::new(&function_wt);
            constraint_solver.add_assumption(
                &WitnessType::Scalar(function_wt.cfg_witness.clone()),
                &WitnessType::Scalar(signature.cfg_witness.clone()),
            );
            for (specialized_param, original_param) in signature
                .param_witnesses
                .iter()
                .zip(function_wt.parameters.iter())
            {
                constraint_solver.add_assumption(specialized_param, original_param);
            }

            for (specialized_return, original_return) in signature
                .return_witnesses
                .iter()
                .zip(function_wt.returns_witness.iter())
            {
                constraint_solver.add_assumption(specialized_return, original_return);
            }

            constraint_solver.solve();
            let target_function_wt = function_wt.update_from_unification(&constraint_solver.unification);
            witness_inference.set_function_witness_type(target_function_id, target_function_wt);
            let fn_wt = witness_inference.get_function_witness_type(target_function_id);

            let mut func = ssa.take_function(target_function_id);

            for (block_id, block) in func.get_blocks_mut() {
                for instruction in block.get_instructions_mut() {
                    match instruction {
                        OpCode::Call { results: returns, function: CallTarget::Static(func_id), args } => {
                            let cfg_witness = fn_wt.block_cfg_witness.get(block_id).unwrap();
                            let args_witnesses = args.iter().map(|arg| fn_wt.value_witness_types.get(arg).unwrap().clone()).collect();
                            let ret_witnesses = returns.iter().map(|arg| fn_wt.value_witness_types.get(arg).unwrap().clone()).collect();
                            let signature = Signature {
                                cfg_witness: cfg_witness.clone(),
                                param_witnesses: args_witnesses,
                                return_witnesses: ret_witnesses,
                            };

                            let specialized_func_id = self.request_specialization(ssa, *func_id, signature);
                            *func_id = specialized_func_id;
                        }
                        OpCode::Call { function: CallTarget::Dynamic(_), .. } => {
                            panic!("Dynamic call targets are not supported in monomorphization")
                        }
                        _ => {}
                    }
                }
            }

            ssa.put_function(target_function_id, func);
        }

        for fn_id in unspecialized_fns {
            ssa.take_function(fn_id);
            witness_inference.remove_function_witness_type(fn_id);
        }

        Ok(())
    }

    fn request_specialization(
        &mut self,
        ssa: &mut SSA,
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

    fn monomorphize_main_signature(&self, wt: &FunctionWitnessType) -> Signature {
        Signature {
            cfg_witness: wt.cfg_witness.clone(),
            param_witnesses: wt
                .parameters
                .iter()
                .map(|p| self.monomorphize_main_witness(p))
                .collect(),
            return_witnesses: wt
                .returns_witness
                .iter()
                .map(|p| self.monomorphize_main_witness(p))
                .collect(),
        }
    }

    fn monomorphize_main_witness(&self, wt: &WitnessType) -> WitnessType {
        match wt {
            WitnessType::Scalar(_) => {
                WitnessType::Scalar(WitnessInfo::Witness)
            }
            WitnessType::Array(_, inner) => WitnessType::Array(
                WitnessInfo::Pure,
                Box::new(self.monomorphize_main_witness(inner)),
            ),
            WitnessType::Tuple(_, child_wts) => WitnessType::Tuple(
                WitnessInfo::Pure,
                child_wts.iter().map(|child_wt| self.monomorphize_main_witness(child_wt)).collect()
            ),
            _ => panic!("Pointer in main signature: {:?}", wt),
        }
    }
}
