use crate::compiler::constraint_solver::ConstraintSolver;
use crate::compiler::flow_analysis::FlowAnalysis;
use crate::compiler::ir::r#type::{Type, TypeExpr};
use crate::compiler::ssa::{
    BlockId, CallTarget, FunctionId, OpCode, SSA, SsaAnnotator, Terminator, TupleIdx, ValueId,
};
use crate::compiler::taint_analysis::{
    ConstantTaint, TaintAnalysis, TaintType, Taint, TypeVariable,
};
use crate::compiler::witness_constraint_solver::WitnessConstraintSolver;
use crate::compiler::witness_info::{
    FunctionWitnessType, WitnessInfo, WitnessJudgement, WitnessType,
};
use std::collections::HashMap;

#[derive(Clone)]
pub struct WitnessTypeInference {
    functions: HashMap<FunctionId, FunctionWitnessType>,
    last_ty_var: usize,
}

impl WitnessTypeInference {
    pub fn new() -> Self {
        WitnessTypeInference {
            functions: HashMap::new(),
            last_ty_var: 0,
        }
    }

    pub fn to_string(&self) -> String {
        self.functions
            .iter()
            .map(|(id, func)| format!("fn_{}: {}", id.0, func.to_string()))
            .collect::<Vec<_>>()
            .join("\n")
    }

    pub fn get_function_witness_type(&self, func_id: FunctionId) -> &FunctionWitnessType {
        self.functions.get(&func_id).unwrap()
    }

    pub fn set_function_witness_type(
        &mut self,
        func_id: FunctionId,
        wt: FunctionWitnessType,
    ) {
        self.functions.insert(func_id, wt);
    }

    pub fn remove_function_witness_type(&mut self, func_id: FunctionId) {
        self.functions.remove(&func_id);
    }

    pub fn run(&mut self, ssa: &SSA, flow_analysis: &FlowAnalysis) -> Result<(), String> {
        let sccs = flow_analysis
            .get_call_graph()
            .get_sccs_reverse_topological(ssa.get_main_id());

        for scc in sccs {
            if scc.len() == 1
                && !flow_analysis
                    .get_call_graph()
                    .is_self_recursive(scc[0])
            {
                self.analyze_function(ssa, flow_analysis, scc[0]);
            } else {
                self.analyze_scc(ssa, flow_analysis, &scc);
            }
        }
        Ok(())
    }

    fn analyze_scc(
        &mut self,
        ssa: &SSA,
        flow_analysis: &FlowAnalysis,
        scc: &[FunctionId],
    ) {
        // Create placeholder FunctionWitnessType for each SCC member with fresh type variables
        for &func_id in scc {
            let func = ssa.get_function(func_id);
            let cfg_ty_var = self.fresh_ty_var();
            let mut placeholder = FunctionWitnessType {
                returns_witness: vec![],
                cfg_witness: WitnessInfo::Variable(cfg_ty_var),
                parameters: vec![],
                judgements: vec![],
                block_cfg_witness: HashMap::new(),
                value_witness_types: HashMap::new(),
            };

            // Create fresh type vars for parameters
            for (_, tp) in func.get_entry().get_parameters() {
                let wt = self.construct_free_witness_for_type(tp);
                placeholder.parameters.push(wt);
            }

            // Create fresh type vars for returns
            for ret in func.get_returns() {
                let wt = self.construct_free_witness_for_type(ret);
                placeholder.returns_witness.push(wt);
            }

            self.functions.insert(func_id, placeholder);
        }

        // Now analyze each SCC member — calls to SCC members will use
        // instantiate_from() on placeholders, creating inter-linked constraints
        for &func_id in scc {
            self.analyze_function(ssa, flow_analysis, func_id);
        }
    }

    fn analyze_function(
        &mut self,
        ssa: &SSA,
        flow_analysis: &FlowAnalysis,
        func_id: FunctionId,
    ) {
        let func = ssa.get_function(func_id);

        let cfg = flow_analysis.get_function_cfg(func_id);
        let block_queue = cfg.get_blocks_bfs();
        let cfg_ty_var = self.fresh_ty_var();
        let mut function_wt = FunctionWitnessType {
            returns_witness: vec![],
            cfg_witness: WitnessInfo::Variable(cfg_ty_var),
            parameters: vec![],
            judgements: vec![],
            block_cfg_witness: HashMap::new(),
            value_witness_types: HashMap::new(),
        };

        // initialize block params
        for (_, block) in func.get_blocks() {
            for (value, tp) in block.get_parameters() {
                let wt = self.construct_free_witness_for_type(tp);
                function_wt.value_witness_types.insert(*value, wt);
            }
        }

        // initialize function parameters
        for (value, _) in func.get_entry().get_parameters() {
            function_wt
                .parameters
                .push(function_wt.value_witness_types.get(value).unwrap().clone());
        }

        // initialize returns
        for ret in func.get_returns() {
            let wt = self.construct_free_witness_for_type(ret);
            function_wt.returns_witness.push(wt);
        }

        // initialize block cfg witness
        for (block_id, _) in func.get_blocks() {
            let cfg_witness = WitnessInfo::Variable(self.fresh_ty_var());
            function_wt
                .block_cfg_witness
                .insert(*block_id, cfg_witness.clone());
            function_wt
                .judgements
                .push(WitnessJudgement::Le(function_wt.cfg_witness.clone(), cfg_witness));
        }

        for (value_id, _) in func.iter_consts() {
            function_wt
                .value_witness_types
                .insert(*value_id, WitnessType::Scalar(WitnessInfo::Pure));
        }

        for block_id in block_queue {
            let block = func.get_block(block_id);

            let cfg_witness = function_wt.block_cfg_witness.get(&block_id).unwrap();

            for instruction in block.get_instructions() {
                match instruction {
                    OpCode::BinaryArithOp {
                        kind: _,
                        result: r,
                        lhs,
                        rhs,
                    }
                    | OpCode::Cmp {
                        kind: _,
                        result: r,
                        lhs,
                        rhs,
                    } => {
                        let lhs_wt = function_wt.value_witness_types.get(lhs).unwrap();
                        let rhs_wt = function_wt.value_witness_types.get(rhs).unwrap();
                        let result_wt = lhs_wt.union(rhs_wt);
                        function_wt.value_witness_types.insert(*r, result_wt);
                    }
                    OpCode::Select {
                        result: r,
                        cond,
                        if_t: then,
                        if_f: otherwise,
                    } => {
                        let cond_wt = function_wt.value_witness_types.get(cond).unwrap();
                        let then_wt = function_wt.value_witness_types.get(then).unwrap();
                        let otherwise_wt =
                            function_wt.value_witness_types.get(otherwise).unwrap();
                        let result_wt = cond_wt.union(then_wt).union(otherwise_wt);
                        function_wt.value_witness_types.insert(*r, result_wt);
                    }
                    OpCode::Alloc {
                        result: r,
                        elem_type: t,
                    } => {
                        let free = self.construct_free_witness_for_type(t);
                        function_wt.value_witness_types.insert(
                            *r,
                            WitnessType::Ref(WitnessInfo::Pure, Box::new(free)),
                        );
                    }
                    OpCode::Store { ptr, value: v } => {
                        let ptr_wt = function_wt.value_witness_types.get(ptr).unwrap();
                        let value_wt = function_wt.value_witness_types.get(v).unwrap();
                        match ptr_wt {
                            WitnessType::Ref(_, inner) => {
                                function_wt.judgements.push(WitnessJudgement::Le(
                                    cfg_witness.clone(),
                                    inner.toplevel_info(),
                                ));
                                self.deep_le(value_wt, inner, &mut function_wt.judgements);
                            }
                            _ => panic!("Unexpected witness type for ptr"),
                        }
                    }
                    OpCode::Load { result: r, ptr } => {
                        let ptr_wt = function_wt.value_witness_types.get(ptr).unwrap();
                        match ptr_wt {
                            WitnessType::Ref(ptr_info, inner) => {
                                function_wt.value_witness_types.insert(
                                    *r,
                                    inner.with_toplevel_info(
                                        inner.toplevel_info().union(ptr_info),
                                    ),
                                );
                            }
                            _ => panic!("Unexpected witness type for ptr"),
                        }
                    }
                    OpCode::ReadGlobal {
                        result: r,
                        offset: _,
                        result_type: tp,
                    } => {
                        let result_wt = self.construct_pure_witness_for_type(tp);
                        function_wt.value_witness_types.insert(*r, result_wt);
                    }
                    OpCode::AssertEq { lhs: _, rhs: _ } => {}
                    OpCode::AssertR1C { a: _, b: _, c: _ } => {}
                    OpCode::InitGlobal { .. } => {}
                    OpCode::DropGlobal { .. } => {}
                    OpCode::ArrayGet {
                        result: r,
                        array: arr,
                        index: idx,
                    } => {
                        let arr_wt = function_wt.value_witness_types.get(arr).unwrap();
                        let idx_wt = function_wt.value_witness_types.get(idx).unwrap();
                        let elem_wt = arr_wt.child_witness_type().unwrap();
                        let result_wt = elem_wt.with_toplevel_info(
                            arr_wt
                                .toplevel_info()
                                .union(&idx_wt.toplevel_info())
                                .union(&elem_wt.toplevel_info()),
                        );
                        function_wt.value_witness_types.insert(*r, result_wt);
                    }
                    OpCode::ArraySet {
                        result: r,
                        array: arr,
                        index: idx,
                        value,
                    } => {
                        let arr_wt = function_wt.value_witness_types.get(arr).unwrap();
                        let idx_wt = function_wt.value_witness_types.get(idx).unwrap();
                        let value_wt = function_wt.value_witness_types.get(value).unwrap();
                        let arr_elem_wt = arr_wt.child_witness_type().unwrap();
                        let result_arr_wt = idx_wt.union(&arr_elem_wt).union(value_wt);
                        let result_wt = WitnessType::Array(
                            arr_wt.toplevel_info(),
                            Box::new(result_arr_wt),
                        );
                        function_wt.value_witness_types.insert(*r, result_wt);
                    }
                    OpCode::SlicePush {
                        dir: _,
                        result: r,
                        slice: sl,
                        values,
                    } => {
                        let slice_wt = function_wt.value_witness_types.get(sl).unwrap();
                        let slice_elem_wt = slice_wt.child_witness_type().unwrap();
                        let mut result_elem_wt = slice_elem_wt.clone();
                        for value in values {
                            let value_wt =
                                function_wt.value_witness_types.get(value).unwrap();
                            result_elem_wt = result_elem_wt.union(value_wt);
                        }
                        let result_wt = WitnessType::Array(
                            slice_wt.toplevel_info().clone(),
                            Box::new(result_elem_wt),
                        );
                        function_wt.value_witness_types.insert(*r, result_wt);
                    }
                    OpCode::SliceLen {
                        result: r,
                        slice: _sl,
                    } => {
                        let result_wt = WitnessType::Scalar(WitnessInfo::Pure);
                        function_wt.value_witness_types.insert(*r, result_wt);
                    }
                    OpCode::Call {
                        results: outputs,
                        function: CallTarget::Static(func),
                        args: inputs,
                    } => {
                        let return_types = ssa.get_function(*func).get_returns();
                        for (output, typ) in outputs.iter().zip(return_types.iter()) {
                            function_wt.value_witness_types.insert(
                                *output,
                                self.construct_free_witness_for_type(typ),
                            );
                        }
                        let outputs_wt = outputs
                            .iter()
                            .map(|v| function_wt.value_witness_types.get(v).unwrap())
                            .collect::<Vec<_>>();
                        let inputs_wt = inputs
                            .iter()
                            .map(|v| function_wt.value_witness_types.get(v).unwrap())
                            .collect::<Vec<_>>();
                        let mut func_wt =
                            self.functions.get(&func).unwrap().clone();
                        func_wt.instantiate_from(&mut self.last_ty_var);
                        for (output, ret) in
                            outputs_wt.iter().zip(func_wt.returns_witness.iter())
                        {
                            self.deep_le(ret, output, &mut function_wt.judgements);
                            self.deep_le(output, ret, &mut function_wt.judgements);
                        }
                        for (input, param) in
                            inputs_wt.iter().zip(func_wt.parameters.iter())
                        {
                            self.deep_le(input, param, &mut function_wt.judgements);
                            self.deep_le(param, input, &mut function_wt.judgements);
                        }
                        function_wt.judgements.extend(func_wt.judgements);
                        function_wt.judgements.push(WitnessJudgement::Le(
                            cfg_witness.clone(),
                            func_wt.cfg_witness.clone(),
                        ));
                    }
                    OpCode::Call {
                        function: CallTarget::Dynamic(_),
                        ..
                    } => {
                        panic!(
                            "Dynamic call targets are not supported in witness type inference"
                        )
                    }
                    OpCode::MkSeq {
                        result,
                        elems: inputs,
                        seq_type: _,
                        elem_type: tp,
                    } => {
                        let inputs_wt = inputs
                            .iter()
                            .map(|v| function_wt.value_witness_types.get(v).unwrap())
                            .collect::<Vec<_>>();
                        let result_wt = inputs_wt
                            .iter()
                            .fold(self.construct_pure_witness_for_type(tp), |acc, t| {
                                acc.union(t)
                            });
                        function_wt.value_witness_types.insert(
                            *result,
                            WitnessType::Array(WitnessInfo::Pure, Box::new(result_wt)),
                        );
                    }
                    OpCode::Cast {
                        result,
                        value,
                        target: _,
                    } => {
                        let value_wt =
                            function_wt.value_witness_types.get(value).unwrap().clone();
                        function_wt.value_witness_types.insert(*result, value_wt);
                    }
                    OpCode::Truncate {
                        result,
                        value,
                        to_bits: _,
                        from_bits: _,
                    } => {
                        let value_wt =
                            function_wt.value_witness_types.get(value).unwrap().clone();
                        function_wt.value_witness_types.insert(*result, value_wt);
                    }
                    OpCode::Not { result, value } => {
                        let value_wt =
                            function_wt.value_witness_types.get(value).unwrap().clone();
                        function_wt.value_witness_types.insert(*result, value_wt);
                    }
                    OpCode::ToBits {
                        result,
                        value,
                        endianness: _,
                        count: _,
                    } => {
                        let value_wt =
                            function_wt.value_witness_types.get(value).unwrap().clone();
                        let result_wt =
                            WitnessType::Array(WitnessInfo::Pure, Box::new(value_wt));
                        function_wt.value_witness_types.insert(*result, result_wt);
                    }
                    OpCode::ToRadix {
                        result,
                        value,
                        radix: _,
                        endianness: _,
                        count: _,
                    } => {
                        let value_wt =
                            function_wt.value_witness_types.get(value).unwrap().clone();
                        let result_wt =
                            WitnessType::Array(WitnessInfo::Pure, Box::new(value_wt));
                        function_wt.value_witness_types.insert(*result, result_wt);
                    }
                    OpCode::Rangecheck {
                        value: _,
                        max_bits: _,
                    } => {}
                    OpCode::MemOp { kind: _, value: _ } => {}
                    OpCode::WriteWitness { .. }
                    | OpCode::Constrain { .. }
                    | OpCode::FreshWitness {
                        result: _,
                        result_type: _,
                    }
                    | OpCode::BumpD {
                        matrix: _,
                        variable: _,
                        sensitivity: _,
                    }
                    | OpCode::NextDCoeff { result: _ }
                    | OpCode::MulConst {
                        result: _,
                        const_val: _,
                        var: _,
                    }
                    | OpCode::Lookup {
                        target: _,
                        keys: _,
                        results: _,
                    }
                    | OpCode::DLookup {
                        target: _,
                        keys: _,
                        results: _,
                    }
                    | OpCode::Todo { .. }
                    | OpCode::ValueOf { .. } => {
                        panic!(
                            "Should not be present at this stage {:?}",
                            instruction
                        );
                    }
                    OpCode::TupleProj {
                        result,
                        tuple,
                        idx,
                    } => {
                        if let TupleIdx::Static(child_index) = idx {
                            let tuple_wt =
                                function_wt.value_witness_types.get(tuple).unwrap();
                            if let WitnessType::Tuple(_, children) = tuple_wt {
                                let elem_wt = &children[*child_index];
                                let result_wt = elem_wt.with_toplevel_info(
                                    tuple_wt
                                        .toplevel_info()
                                        .union(&elem_wt.toplevel_info()),
                                );
                                function_wt
                                    .value_witness_types
                                    .insert(*result, result_wt);
                            } else {
                                panic!("Witness type should be of tuple type")
                            }
                        } else {
                            panic!("Tuple index should be static at this stage")
                        }
                    }
                    OpCode::MkTuple {
                        result,
                        elems: inputs,
                        element_types: _,
                    } => {
                        let inputs_wt = inputs
                            .iter()
                            .map(|v| {
                                function_wt
                                    .value_witness_types
                                    .get(v)
                                    .unwrap()
                                    .clone()
                            })
                            .collect::<Vec<_>>();
                        function_wt.value_witness_types.insert(
                            *result,
                            WitnessType::Tuple(WitnessInfo::Pure, inputs_wt),
                        );
                    }
                }
            }

            if let Some(terminator) = block.get_terminator() {
                match terminator {
                    Terminator::Return(values) => {
                        let returns_wt = function_wt.returns_witness.clone();
                        let actual_returns_wt = values
                            .iter()
                            .map(|v| function_wt.value_witness_types.get(v).unwrap())
                            .collect::<Vec<_>>();
                        for (declared, actual) in
                            returns_wt.iter().zip(actual_returns_wt.iter())
                        {
                            self.deep_le(actual, declared, &mut function_wt.judgements);
                        }
                    }
                    Terminator::Jmp(target, params) => {
                        let target_params = func
                            .get_block(*target)
                            .get_parameters()
                            .map(|(v, _)| *v)
                            .collect::<Vec<_>>();
                        let target_param_wts = target_params
                            .iter()
                            .map(|v| function_wt.value_witness_types.get(v).unwrap())
                            .collect::<Vec<_>>();
                        let param_wts = params
                            .iter()
                            .map(|v| function_wt.value_witness_types.get(v).unwrap())
                            .collect::<Vec<_>>();
                        for (target_param, param) in
                            target_param_wts.iter().zip(param_wts.iter())
                        {
                            self.deep_le(
                                param,
                                target_param,
                                &mut function_wt.judgements,
                            );
                        }
                    }
                    Terminator::JmpIf(cond, _, _) => {
                        let cond_wt =
                            function_wt.value_witness_types.get(cond).unwrap();
                        if cfg.is_loop_entry(block_id) {
                            function_wt.judgements.push(WitnessJudgement::Le(
                                cond_wt.toplevel_info(),
                                WitnessInfo::Pure,
                            ));
                        } else {
                            let merge = cfg.get_merge_point(block_id);
                            let merge_inputs = func
                                .get_block(merge)
                                .get_parameters()
                                .map(|(v, _)| *v)
                                .collect::<Vec<_>>();
                            for input in merge_inputs {
                                let input_wt = function_wt
                                    .value_witness_types
                                    .get(&input)
                                    .unwrap();
                                function_wt.judgements.push(WitnessJudgement::Le(
                                    cond_wt.toplevel_info(),
                                    input_wt.toplevel_info(),
                                ));
                            }

                            let body_blocks = cfg.get_if_body(block_id);
                            for block in body_blocks {
                                let local_witness = function_wt
                                    .block_cfg_witness
                                    .get(&block)
                                    .unwrap();
                                function_wt.judgements.push(WitnessJudgement::Le(
                                    cond_wt.toplevel_info(),
                                    local_witness.clone(),
                                ));

                                function_wt.judgements.push(WitnessJudgement::Le(
                                    cfg_witness.clone(),
                                    local_witness.clone(),
                                ));
                            }
                        }
                    }
                }
            }
        }

        self.functions.insert(func_id, function_wt);
    }

    fn deep_le(
        &self,
        lhs: &WitnessType,
        rhs: &WitnessType,
        judgements: &mut Vec<WitnessJudgement>,
    ) {
        match (lhs, rhs) {
            (WitnessType::Scalar(lhs), WitnessType::Scalar(rhs)) => {
                judgements.push(WitnessJudgement::Le(lhs.clone(), rhs.clone()));
            }
            (WitnessType::Array(lhs, inner_lhs), WitnessType::Array(rhs, inner_rhs)) => {
                judgements.push(WitnessJudgement::Le(lhs.clone(), rhs.clone()));
                self.deep_le(inner_lhs, inner_rhs, judgements);
            }
            (WitnessType::Ref(lhs, inner_lhs), WitnessType::Ref(rhs, inner_rhs)) => {
                judgements.push(WitnessJudgement::Le(lhs.clone(), rhs.clone()));
                self.deep_le(inner_lhs, inner_rhs, judgements);
            }
            (WitnessType::Tuple(lhs, inner_lhs), WitnessType::Tuple(rhs, inner_rhs)) => {
                judgements.push(WitnessJudgement::Le(lhs.clone(), rhs.clone()));
                for (l, r) in inner_lhs.iter().zip(inner_rhs.iter()) {
                    self.deep_le(l, r, judgements);
                }
            }
            _ => {
                panic!(
                    "Cannot compare different witness types: {:?} vs {:?}",
                    lhs, rhs
                )
            }
        }
    }

    fn fresh_ty_var(&mut self) -> TypeVariable {
        let var = TypeVariable(self.last_ty_var);
        self.last_ty_var += 1;
        var
    }

    fn construct_free_witness_for_type(&mut self, typ: &Type) -> WitnessType {
        match &typ.expr {
            TypeExpr::U(_) | TypeExpr::Field => {
                WitnessType::Scalar(WitnessInfo::Variable(self.fresh_ty_var()))
            }
            TypeExpr::Array(i, _) => WitnessType::Array(
                WitnessInfo::Variable(self.fresh_ty_var()),
                Box::new(self.construct_free_witness_for_type(i)),
            ),
            TypeExpr::Slice(i) => WitnessType::Array(
                WitnessInfo::Variable(self.fresh_ty_var()),
                Box::new(self.construct_free_witness_for_type(i)),
            ),
            TypeExpr::Ref(i) => WitnessType::Ref(
                WitnessInfo::Variable(self.fresh_ty_var()),
                Box::new(self.construct_free_witness_for_type(i)),
            ),
            TypeExpr::WitnessOf(_) => {
                panic!("ICE: WitnessOf should not be present at this stage");
            }
            TypeExpr::Tuple(elements) => WitnessType::Tuple(
                WitnessInfo::Variable(self.fresh_ty_var()),
                elements
                    .iter()
                    .map(|e| self.construct_free_witness_for_type(e))
                    .collect(),
            ),
            TypeExpr::Function => WitnessType::Scalar(WitnessInfo::Pure),
        }
    }

    fn construct_pure_witness_for_type(&mut self, typ: &Type) -> WitnessType {
        match &typ.expr {
            TypeExpr::U(_) | TypeExpr::Field => WitnessType::Scalar(WitnessInfo::Pure),
            TypeExpr::Array(i, _) => WitnessType::Array(
                WitnessInfo::Pure,
                Box::new(self.construct_pure_witness_for_type(i)),
            ),
            TypeExpr::Slice(i) => WitnessType::Array(
                WitnessInfo::Pure,
                Box::new(self.construct_pure_witness_for_type(i)),
            ),
            TypeExpr::Ref(i) => WitnessType::Ref(
                WitnessInfo::Pure,
                Box::new(self.construct_pure_witness_for_type(i)),
            ),
            TypeExpr::WitnessOf(_) => {
                panic!("ICE: WitnessOf should not be present at this stage");
            }
            TypeExpr::Tuple(elements) => WitnessType::Tuple(
                WitnessInfo::Pure,
                elements
                    .iter()
                    .map(|e| self.construct_pure_witness_for_type(e))
                    .collect(),
            ),
            TypeExpr::Function => WitnessType::Scalar(WitnessInfo::Pure),
        }
    }

    /// Compare results of this analysis with the old TaintAnalysis.
    /// Panics with a descriptive message on any mismatch.
    pub fn compare_with_taint_analysis(&self, taint: &TaintAnalysis, ssa: &SSA) {
        for (func_id, _) in ssa.iter_functions() {
            let func_id = *func_id;

            let taint_func = taint.get_function_taint(func_id);
            let witness_func = self.get_function_witness_type(func_id);

            // Solve both constraint systems
            let mut taint_solver = ConstraintSolver::new(taint_func);
            taint_solver.solve();
            let resolved_taint = taint_func.update_from_unification(&taint_solver.unification);

            let mut witness_solver = WitnessConstraintSolver::new(witness_func);
            witness_solver.solve();
            let resolved_witness =
                witness_func.update_from_unification(&witness_solver.unification);

            let func_name = ssa.get_function(func_id).get_name().to_string();

            // Compare cfg_taint / cfg_witness
            Self::compare_info(
                &resolved_taint.cfg_taint,
                &resolved_witness.cfg_witness,
                &format!("fn {} (id={}) cfg", func_name, func_id.0),
            );

            // Compare block_cfg_taints / block_cfg_witness
            for (block_id, taint_info) in &resolved_taint.block_cfg_taints {
                let witness_info = resolved_witness
                    .block_cfg_witness
                    .get(block_id)
                    .unwrap_or_else(|| {
                        panic!(
                            "fn {} (id={}): block {} missing from witness inference block_cfg_witness",
                            func_name, func_id.0, block_id.0
                        )
                    });
                Self::compare_info(
                    taint_info,
                    witness_info,
                    &format!(
                        "fn {} (id={}) block {} cfg",
                        func_name, func_id.0, block_id.0
                    ),
                );
            }

            // Compare value_taints / value_witness_types
            for (value_id, taint_type) in &resolved_taint.value_taints {
                let witness_type = resolved_witness
                    .value_witness_types
                    .get(value_id)
                    .unwrap_or_else(|| {
                        panic!(
                            "fn {} (id={}): value {} missing from witness inference",
                            func_name, func_id.0, value_id.0
                        )
                    });
                Self::compare_types(
                    taint_type,
                    witness_type,
                    &format!(
                        "fn {} (id={}) value {}",
                        func_name, func_id.0, value_id.0
                    ),
                );
            }
        }
    }

    fn compare_info(taint: &Taint, witness: &WitnessInfo, context: &str) {
        let taint_is_witness = match taint {
            Taint::Constant(ConstantTaint::Witness) => true,
            Taint::Constant(ConstantTaint::Pure) => false,
            other => panic!(
                "{}: taint not fully resolved: {:?}",
                context, other
            ),
        };
        let witness_is_witness = match witness {
            WitnessInfo::Witness => true,
            WitnessInfo::Pure => false,
            other => panic!(
                "{}: witness info not fully resolved: {:?}",
                context, other
            ),
        };
        if taint_is_witness != witness_is_witness {
            panic!(
                "MISMATCH at {}: taint={:?}, witness={:?}",
                context, taint, witness
            );
        }
    }

    fn compare_types(taint_type: &TaintType, witness_type: &WitnessType, context: &str) {
        match (taint_type, witness_type) {
            (TaintType::Primitive(t), WitnessType::Scalar(w)) => {
                Self::compare_info(t, w, context);
            }
            (TaintType::NestedImmutable(t, inner_t), WitnessType::Array(w, inner_w)) => {
                Self::compare_info(t, w, &format!("{} (array toplevel)", context));
                Self::compare_types(inner_t, inner_w, &format!("{} (array inner)", context));
            }
            (TaintType::NestedMutable(t, inner_t), WitnessType::Ref(w, inner_w)) => {
                Self::compare_info(t, w, &format!("{} (ref toplevel)", context));
                Self::compare_types(inner_t, inner_w, &format!("{} (ref inner)", context));
            }
            (TaintType::Tuple(t, children_t), WitnessType::Tuple(w, children_w)) => {
                Self::compare_info(t, w, &format!("{} (tuple toplevel)", context));
                if children_t.len() != children_w.len() {
                    panic!(
                        "MISMATCH at {} (tuple arity): taint has {} children, witness has {}",
                        context,
                        children_t.len(),
                        children_w.len()
                    );
                }
                for (i, (ct, cw)) in children_t.iter().zip(children_w.iter()).enumerate() {
                    Self::compare_types(ct, cw, &format!("{} (tuple field {})", context, i));
                }
            }
            _ => {
                panic!(
                    "MISMATCH at {}: structural mismatch taint={:?}, witness={:?}",
                    context, taint_type, witness_type
                );
            }
        }
    }
}

impl SsaAnnotator for WitnessTypeInference {
    fn annotate_value(&self, function_id: FunctionId, value_id: ValueId) -> String {
        let Some(function_wt) = self.functions.get(&function_id) else {
            return "".to_string();
        };
        function_wt.annotate_value(function_id, value_id)
    }

    fn annotate_block(&self, function_id: FunctionId, block_id: BlockId) -> String {
        let Some(function_wt) = self.functions.get(&function_id) else {
            return "".to_string();
        };
        function_wt.annotate_block(function_id, block_id)
    }

    fn annotate_function(&self, function_id: FunctionId) -> String {
        let Some(function_wt) = self.functions.get(&function_id) else {
            return "".to_string();
        };
        function_wt.annotate_function(function_id)
    }
}
