use std::collections::{HashMap, HashSet};

use petgraph::{
    algo::{condensation, toposort},
    graph::DiGraph,
};

use super::super::witness_info::WitnessShape;
use super::signature::{
    BoundaryLayout, CallSite, DependencyGraph, FunctionSummary, SpecKey, VarShape, VariableId,
    array_element, map_ports_to_vars, port_shape_to_var_shape, ref_inner, seed_shape,
    tuple_element, var_shape_for_type,
};
use crate::compiler::{
    analysis::flow_analysis::FlowAnalysis,
    ssa::{
        BlockId, FunctionId, Instruction, Terminator, ValueId,
        hlssa::{CallTarget, Constant, HLSSA, OpCode},
    },
};

#[derive(Clone, Debug)]
pub(super) struct BodyBuild {
    pub(super) graph: DependencyGraph,
    pub(super) value_shapes: HashMap<ValueId, VarShape>,
    pub(super) block_cfg_vars: HashMap<BlockId, VariableId>,
    pub(super) entry_params: Vec<VarShape>,
    pub(super) return_shapes: Vec<VarShape>,
    pub(super) cfg_var: VariableId,
    pub(super) always: VariableId,
    boundary_nodes: Vec<VariableId>,
    port_by_boundary_node: HashMap<VariableId, usize>,
    pub(super) constrained_calls: Vec<CallSite>,
}

/// Builds the dependency graph used by both summary inference and concrete specialization.
///
/// Summary inference gives boundary ports fresh variables and then collapses reachability back to
/// boundary edges. Spec solving gives original SSA values fresh variables, seeds the requested key,
/// and reads the variables reachable from `always`.
pub(super) struct BodyBuilder<'a> {
    ssa: &'a HLSSA,
    flow_analysis: &'a FlowAnalysis,
    layouts: &'a HashMap<FunctionId, BoundaryLayout>,
    summaries: &'a HashMap<FunctionId, FunctionSummary>,
    original_func_id: FunctionId,
    function_id: FunctionId,
    graph: DependencyGraph,
    value_shapes: HashMap<ValueId, VarShape>,
    block_cfg_vars: HashMap<BlockId, VariableId>,
    entry_params: Vec<VarShape>,
    return_shapes: Vec<VarShape>,
    cfg_var: VariableId,
    always: VariableId,
    boundary_nodes: Vec<VariableId>,
    port_by_boundary_node: HashMap<VariableId, usize>,
    constrained_calls: Vec<CallSite>,
}

impl<'a> BodyBuilder<'a> {
    pub(super) fn for_summary(
        function_id: FunctionId,
        ssa: &'a HLSSA,
        flow_analysis: &'a FlowAnalysis,
        layouts: &'a HashMap<FunctionId, BoundaryLayout>,
        summaries: &'a HashMap<FunctionId, FunctionSummary>,
    ) -> Self {
        let mut graph = DependencyGraph::new();
        let layout = &layouts[&function_id];
        let boundary_nodes = (0..layout.port_count)
            .map(|_| graph.fresh_var())
            .collect::<Vec<_>>();
        let port_by_boundary_node = boundary_nodes
            .iter()
            .enumerate()
            .map(|(port, var)| (*var, port))
            .collect::<HashMap<_, _>>();
        let always = graph.fresh_var();
        let entry_params = layout
            .parameters
            .iter()
            .map(|shape| port_shape_to_var_shape(shape, &boundary_nodes))
            .collect();
        let return_shapes = layout
            .returns
            .iter()
            .map(|shape| port_shape_to_var_shape(shape, &boundary_nodes))
            .collect();
        let cfg_var = boundary_nodes[layout.cfg.0];

        Self {
            ssa,
            flow_analysis,
            layouts,
            summaries,
            original_func_id: function_id,
            function_id,
            graph,
            value_shapes: HashMap::new(),
            block_cfg_vars: HashMap::new(),
            entry_params,
            return_shapes,
            cfg_var,
            always,
            boundary_nodes,
            port_by_boundary_node,
            constrained_calls: Vec::new(),
        }
    }

    pub(super) fn for_spec(
        key: &SpecKey,
        ssa: &'a HLSSA,
        flow_analysis: &'a FlowAnalysis,
        layouts: &'a HashMap<FunctionId, BoundaryLayout>,
        summaries: &'a HashMap<FunctionId, FunctionSummary>,
    ) -> Self {
        let mut graph = DependencyGraph::new();
        let always = graph.fresh_var();
        let function = ssa.get_function(key.original_func_id);
        let entry_params = function
            .get_entry()
            .get_parameters()
            .map(|(_, typ)| var_shape_for_type(typ, &mut graph))
            .collect::<Vec<_>>();
        let return_shapes = ssa
            .get_function(key.original_func_id)
            .get_returns()
            .iter()
            .map(|typ| var_shape_for_type(typ, &mut graph))
            .collect::<Vec<_>>();
        let cfg_var = graph.fresh_var();

        for (shape, concrete) in entry_params.iter().zip(&key.parameters) {
            seed_shape(&mut graph, always, shape, concrete);
        }
        for (shape, concrete) in return_shapes.iter().zip(&key.returns) {
            seed_shape(&mut graph, always, shape, concrete);
        }
        if key.cfg_witness.is_witness() {
            graph.add_edge(always, cfg_var);
        }

        Self {
            ssa,
            flow_analysis,
            layouts,
            summaries,
            original_func_id: key.original_func_id,
            function_id: key.original_func_id,
            graph,
            value_shapes: HashMap::new(),
            block_cfg_vars: HashMap::new(),
            entry_params,
            return_shapes,
            cfg_var,
            always,
            boundary_nodes: Vec::new(),
            port_by_boundary_node: HashMap::new(),
            constrained_calls: Vec::new(),
        }
    }

    pub(super) fn build(mut self) -> BodyBuild {
        self.seed_constants();
        self.seed_block_parameters();

        let block_order = self
            .flow_analysis
            .get_function_cfg(self.original_func_id)
            .get_blocks_bfs()
            .collect::<Vec<_>>();
        let blocks = {
            let function = self.ssa.get_function(self.function_id);
            block_order
                .iter()
                .map(|block_id| {
                    let block = function.get_block(*block_id);
                    (
                        *block_id,
                        block.get_instructions().cloned().collect::<Vec<_>>(),
                        block.get_terminator().cloned(),
                    )
                })
                .collect::<Vec<_>>()
        };

        for (block_id, instructions, terminator) in blocks {
            for (idx, instruction) in instructions.iter().enumerate() {
                self.scan_instruction(
                    CallSite {
                        caller_func_id: self.function_id,
                        block_id,
                        instruction_idx: idx,
                    },
                    instruction,
                );
            }

            if let Some(terminator) = terminator {
                self.scan_terminator(block_id, &terminator);
            }
        }

        BodyBuild {
            graph: self.graph,
            value_shapes: self.value_shapes,
            block_cfg_vars: self.block_cfg_vars,
            entry_params: self.entry_params,
            return_shapes: self.return_shapes,
            cfg_var: self.cfg_var,
            always: self.always,
            boundary_nodes: self.boundary_nodes,
            port_by_boundary_node: self.port_by_boundary_node,
            constrained_calls: self.constrained_calls,
        }
    }

    fn seed_constants(&mut self) {
        let consts = self
            .ssa
            .const_snapshot()
            .iter()
            .map(|(value, constant)| (*value, constant.as_ref().clone()))
            .collect::<Vec<_>>();

        for (value, constant) in consts {
            let shape = match constant {
                Constant::U(_, _) | Constant::I(_, _) | Constant::Field(_) | Constant::FnPtr(_) => {
                    WitnessShape::Scalar(self.graph.fresh_var())
                }
            };
            self.value_shapes.insert(value, shape);
        }
    }

    fn seed_block_parameters(&mut self) {
        let entry_block = self.ssa.get_function(self.function_id).get_entry_id();
        let block_order = self
            .flow_analysis
            .get_function_cfg(self.original_func_id)
            .get_blocks_bfs()
            .collect::<Vec<_>>();
        let params = {
            let function = self.ssa.get_function(self.function_id);
            block_order
                .iter()
                .map(|block_id| {
                    (
                        *block_id,
                        function
                            .get_block(*block_id)
                            .get_parameters()
                            .cloned()
                            .collect::<Vec<_>>(),
                    )
                })
                .collect::<Vec<_>>()
        };

        for (block_id, params) in params {
            let block_cfg = if block_id == entry_block {
                self.cfg_var
            } else {
                self.graph.fresh_var()
            };
            self.block_cfg_vars.insert(block_id, block_cfg);
            self.graph.add_edge(self.cfg_var, block_cfg);

            if block_id == entry_block {
                for ((value, _), shape) in params.iter().zip(&self.entry_params) {
                    self.value_shapes.insert(*value, shape.clone());
                }
            } else {
                for (value, typ) in params {
                    let shape = var_shape_for_type(&typ, &mut self.graph);
                    self.value_shapes.insert(value, shape);
                }
            }
        }
    }

    /// Registers the witness-flow rule for one SSA instruction.
    ///
    /// Each opcode either allocates fresh variables for its results, adds dependency edges between
    /// input and output positions, or records a constrained static call site for key solving.
    fn scan_instruction(&mut self, call_site: CallSite, instruction: &OpCode) {
        match instruction {
            OpCode::BinaryArithOp {
                result, lhs, rhs, ..
            }
            | OpCode::Cmp {
                result, lhs, rhs, ..
            } => {
                let result_var = self.graph.fresh_var();
                self.insert_value_shape(*result, WitnessShape::Scalar(result_var));
                let lhs_top = self.shape(*lhs).toplevel_info();
                let rhs_top = self.shape(*rhs).toplevel_info();
                let result_top = self.shape(*result).toplevel_info();
                self.graph.add_edge(lhs_top, result_top);
                self.graph.add_edge(rhs_top, result_top);
            }
            OpCode::Select {
                result,
                cond,
                if_t,
                if_f,
            } => {
                let then_shape = self.shape(*if_t).clone();
                let else_shape = self.shape(*if_f).clone();
                let result_shape = self.fresh_like_shape(&then_shape);
                self.insert_value_shape(*result, result_shape);
                let result_shape = self.shape(*result).clone();
                self.flow_shape(&then_shape, &result_shape);
                self.flow_shape(&else_shape, &result_shape);
                self.graph.add_edge(
                    self.shape(*cond).toplevel_info(),
                    result_shape.toplevel_info(),
                );
                self.flow_ref_positions(&result_shape, &then_shape);
                self.flow_ref_positions(&result_shape, &else_shape);
            }
            OpCode::Alloc { result, elem_type } => {
                let inner = var_shape_for_type(elem_type, &mut self.graph);
                let top = self.graph.fresh_var();
                self.insert_value_shape(*result, WitnessShape::Ref(top, Box::new(inner)));
            }
            OpCode::Store { ptr, value } => {
                let block_cfg = self.block_cfg_var(call_site.block_id);
                let ptr_shape = self.shape(*ptr).clone();
                let ptr_inner = ref_inner(&ptr_shape);
                let value_shape = self.shape(*value).clone();
                self.graph
                    .add_edge(ptr_shape.toplevel_info(), ptr_inner.toplevel_info());
                self.graph.add_edge(block_cfg, ptr_inner.toplevel_info());
                self.flow_shape(&value_shape, &ptr_inner);
                self.flow_ref_positions(&ptr_inner, &value_shape);
            }
            OpCode::Load { result, ptr } => {
                let ptr_shape = self.shape(*ptr).clone();
                let ptr_inner = ref_inner(&ptr_shape);
                let result_shape = self.fresh_like_shape(&ptr_inner);
                self.insert_value_shape(*result, result_shape);
                let result_shape = self.shape(*result).clone();
                self.graph
                    .add_edge(ptr_shape.toplevel_info(), result_shape.toplevel_info());
                self.flow_shape(&ptr_inner, &result_shape);
                self.flow_ref_positions(&result_shape, &ptr_inner);
            }
            OpCode::ReadGlobal {
                result,
                result_type,
                ..
            } => {
                let shape = var_shape_for_type(result_type, &mut self.graph);
                self.insert_value_shape(*result, shape);
            }
            OpCode::Assert { .. }
            | OpCode::AssertCmp { .. }
            | OpCode::AssertR1C { .. }
            | OpCode::InitGlobal { .. }
            | OpCode::DropGlobal { .. }
            | OpCode::Rangecheck { .. }
            | OpCode::MemOp { .. } => {}
            OpCode::ArrayGet {
                result,
                array,
                index,
            } => {
                let array_shape = self.shape(*array).clone();
                let elem = array_element(&array_shape);
                let result_shape = self.fresh_like_shape(&elem);
                self.insert_value_shape(*result, result_shape);
                let result_shape = self.shape(*result).clone();
                self.flow_shape(&elem, &result_shape);
                self.flow_ref_positions(&result_shape, &elem);
                self.top_to_leaves(array_shape.toplevel_info(), &result_shape);
                self.top_to_leaves(self.shape(*index).toplevel_info(), &result_shape);
            }
            OpCode::ArraySet {
                result,
                array,
                index,
                value,
            } => {
                let array_shape = self.shape(*array).clone();
                let array_elem = array_element(&array_shape);
                let result_shape = self.fresh_like_shape(&array_shape);
                self.insert_value_shape(*result, result_shape);
                let result_shape = self.shape(*result).clone();
                let result_elem = array_element(&result_shape);
                let value_shape = self.shape(*value).clone();
                self.flow_shape(&array_shape, &result_shape);
                self.flow_shape(&value_shape, &result_elem);
                self.top_to_leaves(self.shape(*index).toplevel_info(), &result_elem);
                self.flow_ref_positions(&result_elem, &array_elem);
                self.flow_ref_positions(&result_elem, &value_shape);
            }
            OpCode::SlicePush {
                result,
                slice,
                values,
                ..
            } => {
                let slice_shape = self.shape(*slice).clone();
                let slice_elem = array_element(&slice_shape);
                let result_shape = self.fresh_like_shape(&slice_shape);
                self.insert_value_shape(*result, result_shape);
                let result_shape = self.shape(*result).clone();
                let result_elem = array_element(&result_shape);
                self.flow_shape(&slice_shape, &result_shape);
                for value in values {
                    let value_shape = self.shape(*value).clone();
                    self.flow_shape(&value_shape, &result_elem);
                    self.flow_ref_positions(&result_elem, &value_shape);
                }
                self.flow_ref_positions(&result_elem, &slice_elem);
            }
            OpCode::SliceLen { result, .. } => {
                let result_var = self.graph.fresh_var();
                self.insert_value_shape(*result, WitnessShape::Scalar(result_var));
            }
            OpCode::Call {
                results,
                function: CallTarget::Static(callee_id),
                args,
                unconstrained,
            } => {
                let return_types = self.ssa.get_function(*callee_id).get_returns().to_vec();
                for (result, return_type) in results.iter().zip(return_types.iter()) {
                    let shape = var_shape_for_type(return_type, &mut self.graph);
                    self.insert_value_shape(*result, shape);
                }

                if !unconstrained {
                    let cfg_var = self.block_cfg_var(call_site.block_id);
                    self.project_call_summary(*callee_id, args, results, cfg_var);
                    self.constrained_calls.push(call_site);
                }
            }
            OpCode::Call {
                function: CallTarget::Dynamic(_),
                ..
            } => {
                panic!("Dynamic call targets are not supported in witness type inference");
            }
            OpCode::MkSeq {
                result,
                elems,
                elem_type,
                ..
            } => {
                let elem_shape = var_shape_for_type(elem_type, &mut self.graph);
                let top = self.graph.fresh_var();
                self.insert_value_shape(*result, WitnessShape::Array(top, Box::new(elem_shape)));
                let result_elem = array_element(self.shape(*result));
                for elem in elems {
                    let elem_shape = self.shape(*elem).clone();
                    self.flow_shape(&elem_shape, &result_elem);
                    self.flow_ref_positions(&result_elem, &elem_shape);
                }
            }
            OpCode::MkRepeated {
                result,
                element,
                elem_type,
                ..
            } => {
                let elem_shape = var_shape_for_type(elem_type, &mut self.graph);
                let top = self.graph.fresh_var();
                self.insert_value_shape(*result, WitnessShape::Array(top, Box::new(elem_shape)));
                let result_elem = array_element(self.shape(*result));
                let element_shape = self.shape(*element).clone();
                self.flow_shape(&element_shape, &result_elem);
                self.flow_ref_positions(&result_elem, &element_shape);
            }
            OpCode::Unspread {
                result_odd,
                result_even,
                value,
                ..
            } => {
                let value_shape = self.shape(*value).clone();
                let odd_shape = self.fresh_like_shape(&value_shape);
                let even_shape = self.fresh_like_shape(&value_shape);
                self.insert_value_shape(*result_odd, odd_shape);
                self.insert_value_shape(*result_even, even_shape);
                let odd_shape = self.shape(*result_odd).clone();
                let even_shape = self.shape(*result_even).clone();
                self.flow_shape(&value_shape, &odd_shape);
                self.flow_shape(&value_shape, &even_shape);
                self.flow_ref_positions(&odd_shape, &value_shape);
                self.flow_ref_positions(&even_shape, &value_shape);
            }
            OpCode::Spread { result, value, .. }
            | OpCode::Cast { result, value, .. }
            | OpCode::SExt { result, value, .. }
            | OpCode::BitRange { result, value, .. }
            | OpCode::Not { result, value } => {
                let value_shape = self.shape(*value).clone();
                let result_shape = self.fresh_like_shape(&value_shape);
                self.insert_value_shape(*result, result_shape);
                let result_shape = self.shape(*result).clone();
                self.flow_shape(&value_shape, &result_shape);
                self.flow_ref_positions(&result_shape, &value_shape);
            }
            OpCode::ToBits { result, value, .. } | OpCode::ToRadix { result, value, .. } => {
                let value_shape = self.shape(*value).clone();
                let elem_shape = self.fresh_like_shape(&value_shape);
                let top = self.graph.fresh_var();
                self.insert_value_shape(*result, WitnessShape::Array(top, Box::new(elem_shape)));
                let result_elem = array_element(self.shape(*result));
                self.flow_shape(&value_shape, &result_elem);
                self.flow_ref_positions(&result_elem, &value_shape);
            }
            OpCode::TupleProj { result, tuple, idx } => {
                let tuple_shape = self.shape(*tuple).clone();
                let (tuple_top, child) = tuple_element(&tuple_shape, *idx);
                let result_shape = self.fresh_like_shape(&child);
                self.insert_value_shape(*result, result_shape);
                let result_shape = self.shape(*result).clone();
                self.graph.add_edge(tuple_top, result_shape.toplevel_info());
                self.flow_shape(&child, &result_shape);
                self.flow_ref_positions(&result_shape, &child);
            }
            OpCode::MkTuple { result, elems, .. } => {
                let elem_shapes = elems
                    .iter()
                    .map(|value| self.shape(*value).clone())
                    .collect::<Vec<_>>();
                let children = elem_shapes
                    .iter()
                    .map(|shape| self.fresh_like_shape(shape))
                    .collect::<Vec<_>>();
                let top = self.graph.fresh_var();
                self.insert_value_shape(*result, WitnessShape::Tuple(top, children));
                let WitnessShape::Tuple(_, children) = self.shape(*result).clone() else {
                    unreachable!()
                };
                for (elem_shape, child) in elem_shapes.iter().zip(children.iter()) {
                    self.flow_shape(elem_shape, child);
                    self.flow_ref_positions(child, elem_shape);
                }
            }
            OpCode::WriteWitness { result, .. } => {
                if let Some(result) = result {
                    let var = self.graph.fresh_var();
                    self.insert_value_shape(*result, WitnessShape::Scalar(var));
                    self.mark_witness(var);
                }
            }
            OpCode::Constrain { .. }
            | OpCode::FreshWitness { .. }
            | OpCode::BumpD { .. }
            | OpCode::NextDCoeff { .. }
            | OpCode::MulConst { .. }
            | OpCode::Lookup { .. }
            | OpCode::DLookup { .. }
            | OpCode::Todo { .. }
            | OpCode::ValueOf { .. } => {
                panic!("Should not be present at this stage {:?}", instruction);
            }
            OpCode::Guard { .. } => {
                panic!(
                    "Unsupported opcode during witness type inference: {:?}",
                    instruction
                );
            }
        }
    }

    /// Registers witness-flow rules for control-flow edges, block arguments, and returns.
    fn scan_terminator(&mut self, block_id: BlockId, terminator: &Terminator) {
        match terminator {
            Terminator::Return(values) => {
                let return_shapes = self.return_shapes.clone();
                for (value, return_shape) in values.iter().zip(return_shapes.iter()) {
                    let value_shape = self.shape(*value).clone();
                    self.flow_shape(&value_shape, return_shape);
                    self.flow_ref_positions(return_shape, &value_shape);
                }
            }
            Terminator::Jmp(target, params) => {
                let target_params = self
                    .ssa
                    .get_function(self.function_id)
                    .get_block(*target)
                    .get_parameters()
                    .map(|(value, _)| *value)
                    .collect::<Vec<_>>();
                for (arg, param) in params.iter().zip(target_params.iter()) {
                    let arg_shape = self.shape(*arg).clone();
                    let param_shape = self.shape(*param).clone();
                    self.flow_shape(&arg_shape, &param_shape);
                    self.flow_ref_positions(&param_shape, &arg_shape);
                }
            }
            Terminator::JmpIf(cond, _if_true, _if_false) => {
                let cond_top = self.shape(*cond).toplevel_info();
                let block_cfg = self.block_cfg_var(block_id);
                let cfg = self.flow_analysis.get_function_cfg(self.original_func_id);
                for body_block_id in cfg.get_if_body(block_id) {
                    let body_cfg = self.block_cfg_var(body_block_id);
                    self.graph.add_edge(cond_top, body_cfg);
                    self.graph.add_edge(block_cfg, body_cfg);
                }

                let merge = cfg.get_merge_point(block_id);
                let merge_params = self
                    .ssa
                    .get_function(self.function_id)
                    .get_block(merge)
                    .get_parameters()
                    .map(|(value, _)| *value)
                    .collect::<Vec<_>>();
                for param in merge_params {
                    let param_shape = self.shape(param).clone();
                    self.top_to_leaves(cond_top, &param_shape);
                }
            }
        }
    }

    /// Projects a completed callee summary through one call site's arguments, results, and CFG.
    ///
    /// This is where return-to-argument effects are made visible to callers: the summary already
    /// contains the bidirectional boundary facts, and projection simply maps each callee port to
    /// the caller variable occupying that position.
    fn project_call_summary(
        &mut self,
        callee_id: FunctionId,
        args: &[ValueId],
        results: &[ValueId],
        cfg_var: VariableId,
    ) {
        let layout = &self.layouts[&callee_id];
        let summary = &self.summaries[&callee_id];
        let mut port_vars = vec![None; layout.port_count];

        for (arg, port_shape) in args.iter().zip(&layout.parameters) {
            let arg_shape = self.shape(*arg).clone();
            map_ports_to_vars(port_shape, &arg_shape, &mut port_vars);
        }
        for (result, port_shape) in results.iter().zip(&layout.returns) {
            let result_shape = self.shape(*result).clone();
            map_ports_to_vars(port_shape, &result_shape, &mut port_vars);
        }
        port_vars[layout.cfg.0] = Some(cfg_var);

        for (source, targets) in summary.edges.iter().enumerate() {
            let source_var = if source == layout.always() {
                self.always
            } else {
                port_vars[source].unwrap_or_else(|| {
                    panic!(
                        "Missing projected call source port {:?} for {:?}",
                        source, callee_id
                    )
                })
            };
            for target in targets {
                let target_var = port_vars[*target].unwrap_or_else(|| {
                    panic!(
                        "Missing projected call target port {:?} for {:?}",
                        target, callee_id
                    )
                });
                self.graph.add_edge(source_var, target_var);
            }
        }
    }

    fn insert_value_shape(&mut self, value: ValueId, shape: VarShape) {
        self.value_shapes.entry(value).or_insert(shape);
    }

    fn shape(&self, value: ValueId) -> &VarShape {
        self.value_shapes
            .get(&value)
            .unwrap_or_else(|| panic!("Missing witness shape for value {:?}", value))
    }

    fn block_cfg_var(&self, block_id: BlockId) -> VariableId {
        *self
            .block_cfg_vars
            .get(&block_id)
            .unwrap_or_else(|| panic!("Missing cfg witness variable for block {:?}", block_id))
    }

    fn mark_witness(&mut self, var: VariableId) {
        self.graph.add_edge(self.always, var);
    }

    fn fresh_like_shape(&mut self, shape: &VarShape) -> VarShape {
        match shape {
            WitnessShape::Scalar(_) => WitnessShape::Scalar(self.graph.fresh_var()),
            WitnessShape::Array(_, inner) => {
                let top = self.graph.fresh_var();
                let inner = self.fresh_like_shape(inner);
                WitnessShape::Array(top, Box::new(inner))
            }
            WitnessShape::Ref(_, inner) => {
                let top = self.graph.fresh_var();
                let inner = self.fresh_like_shape(inner);
                WitnessShape::Ref(top, Box::new(inner))
            }
            WitnessShape::Tuple(_, children) => {
                let top = self.graph.fresh_var();
                let children = children
                    .iter()
                    .map(|child| self.fresh_like_shape(child))
                    .collect();
                WitnessShape::Tuple(top, children)
            }
        }
    }

    fn flow_shape(&mut self, source: &VarShape, target: &VarShape) {
        match (source, target) {
            (WitnessShape::Scalar(source), WitnessShape::Scalar(target)) => {
                self.graph.add_edge(*source, *target);
            }
            (
                WitnessShape::Array(source, source_inner),
                WitnessShape::Array(target, target_inner),
            )
            | (WitnessShape::Ref(source, source_inner), WitnessShape::Ref(target, target_inner)) => {
                self.graph.add_edge(*source, *target);
                self.flow_shape(source_inner, target_inner);
            }
            (
                WitnessShape::Tuple(source, source_children),
                WitnessShape::Tuple(target, target_children),
            ) => {
                self.graph.add_edge(*source, *target);
                for (source_child, target_child) in
                    source_children.iter().zip(target_children.iter())
                {
                    self.flow_shape(source_child, target_child);
                }
            }
            _ => panic!(
                "Cannot flow mismatched witness shapes: {:?} -> {:?}",
                source, target
            ),
        }
    }

    /// Copies dependencies for nested references inside arrays and tuples.
    ///
    /// Ordinary aggregate projection is one-way. Reference payloads are different: once a ref is
    /// extracted or rebuilt through an aggregate, later writes through that ref must be visible at
    /// the original aggregate position too.
    fn flow_ref_positions(&mut self, source: &VarShape, target: &VarShape) {
        match (source, target) {
            (WitnessShape::Ref(_, _), WitnessShape::Ref(_, _)) => self.flow_shape(source, target),
            (WitnessShape::Array(_, source_inner), WitnessShape::Array(_, target_inner)) => {
                self.flow_ref_positions(source_inner, target_inner);
            }
            (WitnessShape::Tuple(_, source_children), WitnessShape::Tuple(_, target_children)) => {
                for (source_child, target_child) in
                    source_children.iter().zip(target_children.iter())
                {
                    self.flow_ref_positions(source_child, target_child);
                }
            }
            (WitnessShape::Scalar(_), WitnessShape::Scalar(_)) => {}
            _ => panic!(
                "Cannot flow ref positions through mismatched witness shapes: {:?} -> {:?}",
                source, target
            ),
        }
    }

    fn top_to_leaves(&mut self, source: VariableId, target: &VarShape) {
        match target {
            WitnessShape::Scalar(target) | WitnessShape::Ref(target, _) => {
                self.graph.add_edge(source, *target);
            }
            WitnessShape::Array(_, inner) => self.top_to_leaves(source, inner),
            WitnessShape::Tuple(_, children) => {
                for child in children {
                    self.top_to_leaves(source, child);
                }
            }
        }
    }
}

/// Infers closed boundary summaries for all original functions.
///
/// Callee SCCs are processed before caller SCCs. Inside one SCC we run a local fixed point because
/// recursive functions may expose new boundary edges to each other.
pub(super) fn infer_summaries(
    ssa: &HLSSA,
    flow_analysis: &FlowAnalysis,
    function_ids: &[FunctionId],
    layouts: &HashMap<FunctionId, BoundaryLayout>,
) -> HashMap<FunctionId, FunctionSummary> {
    let mut summaries = function_ids
        .iter()
        .map(|function_id| (*function_id, FunctionSummary::new(&layouts[function_id])))
        .collect::<HashMap<_, _>>();

    for scc in summary_scc_order(ssa, function_ids) {
        loop {
            let mut changed = false;
            for function_id in &scc {
                let build =
                    BodyBuilder::for_summary(*function_id, ssa, flow_analysis, layouts, &summaries)
                        .build();
                let summary = summarize_body(&build, &layouts[function_id]);
                changed |= summaries.get_mut(function_id).unwrap().absorb(&summary);
            }
            if !changed {
                break;
            }
        }
    }

    summaries
}

/// Returns call-graph SCCs in callee-before-caller order.
fn summary_scc_order(ssa: &HLSSA, function_ids: &[FunctionId]) -> Vec<Vec<FunctionId>> {
    let mut call_graph = DiGraph::<FunctionId, ()>::new();
    let mut func_to_node = HashMap::new();
    let function_set = function_ids.iter().copied().collect::<HashSet<_>>();

    for function_id in function_ids {
        let node = call_graph.add_node(*function_id);
        func_to_node.insert(*function_id, node);
    }

    for caller_id in function_ids {
        let caller = func_to_node[caller_id];
        let function = ssa.get_function(*caller_id);
        for (_, block) in function.get_blocks() {
            for instruction in block.get_instructions() {
                for callee_id in instruction.get_static_call_targets() {
                    if function_set.contains(&callee_id) {
                        call_graph.add_edge(caller, func_to_node[&callee_id], ());
                    }
                }
            }
        }
    }

    let condensed = condensation(call_graph, true);
    let mut ordered = toposort(&condensed, None).expect("condensed call graph must be acyclic");
    ordered.reverse();
    ordered
        .into_iter()
        .map(|scc_idx| {
            let mut scc = condensed[scc_idx].clone();
            scc.sort_by_key(|function_id| function_id.0);
            scc
        })
        .collect()
}

/// Collapses a body dependency graph to the boundary-port summary consumed by callers.
fn summarize_body(build: &BodyBuild, layout: &BoundaryLayout) -> FunctionSummary {
    let mut summary = FunctionSummary::new(layout);
    let sources = build
        .boundary_nodes
        .iter()
        .copied()
        .enumerate()
        .chain(std::iter::once((layout.always(), build.always)))
        .collect::<Vec<_>>();

    for (source_port, source_var) in sources {
        let reached = build.graph.reachable_from([source_var]);
        for (boundary_var, target_port) in &build.port_by_boundary_node {
            if reached.contains(boundary_var) {
                summary.add_edge(source_port, *target_port);
            }
        }
    }

    summary
}
