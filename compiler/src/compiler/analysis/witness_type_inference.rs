//! Performs whole program analysis to determine which values are potentially witness tainted, which
//! are _only_ witnesses, and which are only non-witness values.

use std::collections::{HashMap, HashSet, VecDeque};

use super::witness_info::{FunctionWitnessType, WitnessShape, WitnessType};
use crate::compiler::{
    analysis::flow_analysis::FlowAnalysis,
    ssa::{
        BlockId, FunctionId, Instruction, SSAAnotator, Terminator, ValueId,
        hlssa::{CallTarget, Constant, HLSSA, OpCode, Type, TypeExpr},
    },
};
use petgraph::{
    algo::{condensation, toposort},
    graph::DiGraph,
};

type VarShape = WitnessShape<VariableId>;
type PortShape = WitnessShape<PortId>;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct VariableId(usize);

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct PortId(usize);

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct SpecKey {
    original_func_id: FunctionId,
    parameters: Vec<WitnessShape>,
    returns: Vec<WitnessShape>,
    cfg_witness: WitnessType,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct CallSite {
    caller_func_id: FunctionId,
    block_id: BlockId,
    instruction_idx: usize,
}

#[derive(Clone, Debug)]
struct PendingCall {
    call_site: CallSite,
    callee_id: FunctionId,
    args: Vec<ValueId>,
    results: Vec<ValueId>,
    cfg_var: VariableId,
}

#[derive(Clone, Debug)]
struct SpecInstance {
    original_func_id: FunctionId,
    specialized_func_id: FunctionId,
    key: SpecKey,
    data: Option<SpecData>,
}

#[derive(Clone, Debug)]
struct SpecData {
    entry_params: Vec<VarShape>,
    return_shapes: Vec<VarShape>,
    cfg_var: VariableId,
    value_shapes: HashMap<ValueId, VarShape>,
    block_cfg_vars: HashMap<BlockId, VariableId>,
    witness_vars: HashSet<VariableId>,
}

#[derive(Clone, Debug)]
struct BoundaryLayout {
    parameters: Vec<PortShape>,
    returns: Vec<PortShape>,
    cfg: PortId,
    port_count: usize,
}

impl BoundaryLayout {
    fn new(function_id: FunctionId, ssa: &HLSSA) -> Self {
        let function = ssa.get_function(function_id);
        let mut next_port = 0usize;
        let parameters = function
            .get_entry()
            .get_parameters()
            .map(|(_, typ)| port_shape_for_type(typ, &mut next_port))
            .collect();
        let returns = function
            .get_returns()
            .iter()
            .map(|typ| port_shape_for_type(typ, &mut next_port))
            .collect();
        let cfg = PortId(next_port);
        next_port += 1;

        Self {
            parameters,
            returns,
            cfg,
            port_count: next_port,
        }
    }

    fn always(&self) -> usize {
        self.port_count
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct FunctionSummary {
    edges: Vec<HashSet<usize>>,
}

impl FunctionSummary {
    fn new(layout: &BoundaryLayout) -> Self {
        Self {
            edges: vec![HashSet::new(); layout.port_count + 1],
        }
    }

    fn add_edge(&mut self, source: usize, target: usize) -> bool {
        self.edges[source].insert(target)
    }

    fn absorb(&mut self, other: &FunctionSummary) -> bool {
        let mut changed = false;
        for (source, targets) in other.edges.iter().enumerate() {
            for target in targets {
                changed |= self.add_edge(source, *target);
            }
        }
        changed
    }

    fn close(&self, layout: &BoundaryLayout, mut active: HashSet<usize>) -> HashSet<usize> {
        let always = layout.always();
        active.insert(always);

        let mut queue = VecDeque::new();
        for source in active.iter().copied() {
            queue.push_back(source);
        }

        while let Some(source) = queue.pop_front() {
            for target in &self.edges[source] {
                if active.insert(*target) {
                    queue.push_back(*target);
                }
            }
        }

        active
    }
}

#[derive(Clone, Debug)]
struct DependencyGraph {
    edges: HashMap<VariableId, Vec<VariableId>>,
    next_variable: usize,
}

impl DependencyGraph {
    fn new() -> Self {
        Self {
            edges: HashMap::new(),
            next_variable: 0,
        }
    }

    fn fresh_var(&mut self) -> VariableId {
        let var = VariableId(self.next_variable);
        self.next_variable += 1;
        var
    }

    fn add_edge(&mut self, source: VariableId, target: VariableId) {
        if source == target {
            return;
        }
        let targets = self.edges.entry(source).or_default();
        if !targets.contains(&target) {
            targets.push(target);
        }
    }

    fn reachable_from(&self, sources: impl IntoIterator<Item = VariableId>) -> HashSet<VariableId> {
        let mut reached = HashSet::new();
        let mut queue = VecDeque::new();

        for source in sources {
            if reached.insert(source) {
                queue.push_back(source);
            }
        }

        while let Some(source) = queue.pop_front() {
            for target in self.edges.get(&source).cloned().unwrap_or_default() {
                if reached.insert(target) {
                    queue.push_back(target);
                }
            }
        }

        reached
    }
}

#[derive(Clone, Debug)]
struct BodyBuild {
    graph: DependencyGraph,
    value_shapes: HashMap<ValueId, VarShape>,
    block_cfg_vars: HashMap<BlockId, VariableId>,
    entry_params: Vec<VarShape>,
    return_shapes: Vec<VarShape>,
    cfg_var: VariableId,
    always: VariableId,
    boundary_nodes: Vec<VariableId>,
    port_by_boundary_node: HashMap<VariableId, usize>,
    pending_calls: Vec<PendingCall>,
}

struct BodyBuilder<'a> {
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
    pending_calls: Vec<PendingCall>,
}

impl<'a> BodyBuilder<'a> {
    fn for_summary(
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
            pending_calls: Vec::new(),
        }
    }

    fn for_spec(
        key: &SpecKey,
        specialized_func_id: FunctionId,
        ssa: &'a HLSSA,
        flow_analysis: &'a FlowAnalysis,
        layouts: &'a HashMap<FunctionId, BoundaryLayout>,
        summaries: &'a HashMap<FunctionId, FunctionSummary>,
    ) -> Self {
        let mut graph = DependencyGraph::new();
        let always = graph.fresh_var();
        let function = ssa.get_function(specialized_func_id);
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
            function_id: specialized_func_id,
            graph,
            value_shapes: HashMap::new(),
            block_cfg_vars: HashMap::new(),
            entry_params,
            return_shapes,
            cfg_var,
            always,
            boundary_nodes: Vec::new(),
            port_by_boundary_node: HashMap::new(),
            pending_calls: Vec::new(),
        }
    }

    fn build(mut self) -> BodyBuild {
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
            pending_calls: self.pending_calls,
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
                    self.pending_calls.push(PendingCall {
                        call_site,
                        callee_id: *callee_id,
                        args: args.clone(),
                        results: results.clone(),
                        cfg_var,
                    });
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

#[derive(Clone, Debug)]
pub struct WitnessTypeInference {
    functions: HashMap<FunctionId, FunctionWitnessType>,
}

impl WitnessTypeInference {
    pub fn new() -> Self {
        WitnessTypeInference {
            functions: HashMap::new(),
        }
    }

    pub fn try_get_function_witness_type(
        &self,
        func_id: FunctionId,
    ) -> Option<&FunctionWitnessType> {
        self.functions.get(&func_id)
    }

    pub fn set_function_witness_type(&mut self, func_id: FunctionId, wt: FunctionWitnessType) {
        self.functions.insert(func_id, wt);
    }

    pub fn remove_function_witness_type(&mut self, func_id: FunctionId) {
        self.functions.remove(&func_id);
    }

    pub fn run(&mut self, ssa: &mut HLSSA, flow_analysis: &FlowAnalysis) -> Result<(), String> {
        let function_ids = ssa.get_function_ids().collect::<Vec<_>>();
        let layouts = function_ids
            .iter()
            .map(|function_id| (*function_id, BoundaryLayout::new(*function_id, ssa)))
            .collect::<HashMap<_, _>>();
        let summaries = infer_summaries(ssa, flow_analysis, &function_ids, &layouts);

        let main_id = ssa.get_main_id();
        let main_func = ssa.get_function(main_id);
        let main_key = SpecKey {
            original_func_id: main_id,
            parameters: main_func
                .get_entry()
                .get_parameters()
                .map(|(_, tp)| pure_shape_for_type(tp))
                .collect(),
            returns: main_func
                .get_returns()
                .iter()
                .map(pure_shape_for_type)
                .collect(),
            cfg_witness: WitnessType::Pure,
        };

        let mut engine = SpecializationEngine::new(ssa, flow_analysis, layouts, summaries);
        let root_spec = engine.ensure_spec(main_key);
        engine.run();
        self.functions = engine.finish(root_spec);

        Ok(())
    }
}

struct SpecializationEngine<'a> {
    ssa: &'a mut HLSSA,
    flow_analysis: &'a FlowAnalysis,
    layouts: HashMap<FunctionId, BoundaryLayout>,
    summaries: HashMap<FunctionId, FunctionSummary>,
    specs: Vec<SpecInstance>,
    specs_by_key: HashMap<SpecKey, usize>,
    spec_queue: VecDeque<usize>,
    resolved_calls: HashMap<CallSite, usize>,
}

impl<'a> SpecializationEngine<'a> {
    fn new(
        ssa: &'a mut HLSSA,
        flow_analysis: &'a FlowAnalysis,
        layouts: HashMap<FunctionId, BoundaryLayout>,
        summaries: HashMap<FunctionId, FunctionSummary>,
    ) -> Self {
        Self {
            ssa,
            flow_analysis,
            layouts,
            summaries,
            specs: Vec::new(),
            specs_by_key: HashMap::new(),
            spec_queue: VecDeque::new(),
            resolved_calls: HashMap::new(),
        }
    }

    fn run(&mut self) {
        while let Some(spec_idx) = self.spec_queue.pop_front() {
            self.scan_spec(spec_idx);
        }
    }

    fn ensure_spec(&mut self, key: SpecKey) -> usize {
        let key = self.close_key(key);
        if let Some(idx) = self.specs_by_key.get(&key) {
            return *idx;
        }

        let specialized_func_id = self.ssa.duplicate_function(key.original_func_id);
        let spec_idx = self.specs.len();
        self.specs.push(SpecInstance {
            original_func_id: key.original_func_id,
            specialized_func_id,
            key: key.clone(),
            data: None,
        });
        self.specs_by_key.insert(key, spec_idx);
        self.spec_queue.push_back(spec_idx);
        spec_idx
    }

    fn scan_spec(&mut self, spec_idx: usize) {
        if self.specs[spec_idx].data.is_some() {
            return;
        }

        let spec = self.specs[spec_idx].clone();
        let build = BodyBuilder::for_spec(
            &spec.key,
            spec.specialized_func_id,
            &*self.ssa,
            self.flow_analysis,
            &self.layouts,
            &self.summaries,
        )
        .build();
        let witness_vars = build.graph.reachable_from([build.always]);
        let data = SpecData {
            entry_params: build.entry_params,
            return_shapes: build.return_shapes,
            cfg_var: build.cfg_var,
            value_shapes: build.value_shapes,
            block_cfg_vars: build.block_cfg_vars,
            witness_vars,
        };

        let actual_key = self.key_from_spec_data(&spec, &data);
        assert_eq!(
            self.close_key(actual_key),
            spec.key,
            "Specialization key was not closed for {:?}",
            spec.key
        );

        for call in build.pending_calls {
            let callee_key = self.key_from_call(&data, &call);
            let callee_spec = self.ensure_spec(callee_key);
            self.resolved_calls.insert(call.call_site, callee_spec);
        }

        self.specs[spec_idx].data = Some(data);
    }

    fn finish(mut self, root_spec: usize) -> HashMap<FunctionId, FunctionWitnessType> {
        let spec_indices = (0..self.specs.len()).collect::<Vec<_>>();
        for spec_idx in spec_indices {
            self.rewrite_calls(spec_idx);
        }

        let root_func = self.specs[root_spec].specialized_func_id;
        self.ssa.set_entry_point(root_func);

        let mut functions = HashMap::new();
        for spec in &self.specs {
            let data = spec
                .data
                .as_ref()
                .unwrap_or_else(|| panic!("Unscanned witness specialization {:?}", spec.key));
            let function_id = spec.specialized_func_id;
            let function = self.ssa.get_function(function_id);
            let block_order = self
                .flow_analysis
                .get_function_cfg(spec.original_func_id)
                .get_blocks_bfs()
                .collect::<Vec<_>>();

            let mut block_cfg_witness = HashMap::new();
            let mut value_witness_types = HashMap::new();
            for block_id in block_order {
                let cfg_var = data.block_cfg_vars[&block_id];
                block_cfg_witness.insert(block_id, witness_of_var(cfg_var, &data.witness_vars));

                let block = function.get_block(block_id);
                for (value, _) in block.get_parameters() {
                    insert_concrete_value_shape(*value, data, &mut value_witness_types);
                }
                for instruction in block.get_instructions() {
                    for value in instruction.get_inputs().chain(instruction.get_results()) {
                        insert_concrete_value_shape(*value, data, &mut value_witness_types);
                    }
                }
                if let Some(terminator) = block.get_terminator() {
                    match terminator {
                        Terminator::Jmp(_, values) | Terminator::Return(values) => {
                            for value in values {
                                insert_concrete_value_shape(*value, data, &mut value_witness_types);
                            }
                        }
                        Terminator::JmpIf(cond, _, _) => {
                            insert_concrete_value_shape(*cond, data, &mut value_witness_types);
                        }
                    }
                }
            }

            functions.insert(
                function_id,
                FunctionWitnessType {
                    returns_witness: data
                        .return_shapes
                        .iter()
                        .map(|shape| concretize_shape(shape, &data.witness_vars))
                        .collect(),
                    cfg_witness: witness_of_var(data.cfg_var, &data.witness_vars),
                    parameters: data
                        .entry_params
                        .iter()
                        .map(|shape| concretize_shape(shape, &data.witness_vars))
                        .collect(),
                    block_cfg_witness,
                    value_witness_types,
                },
            );
        }

        functions
    }

    fn rewrite_calls(&mut self, spec_idx: usize) {
        let spec = self.specs[spec_idx].clone();
        let block_order = self
            .flow_analysis
            .get_function_cfg(spec.original_func_id)
            .get_blocks_bfs()
            .collect::<Vec<_>>();
        let mut rewrites = HashMap::<(BlockId, usize), FunctionId>::new();

        for block_id in block_order {
            let block = self
                .ssa
                .get_function(spec.specialized_func_id)
                .get_block(block_id);
            for (idx, instruction) in block.get_instructions().enumerate() {
                let OpCode::Call {
                    function: CallTarget::Static(_),
                    unconstrained,
                    ..
                } = instruction
                else {
                    continue;
                };
                if *unconstrained {
                    continue;
                }

                let call_site = CallSite {
                    caller_func_id: spec.specialized_func_id,
                    block_id,
                    instruction_idx: idx,
                };
                let callee_spec = self
                    .resolved_calls
                    .get(&call_site)
                    .unwrap_or_else(|| panic!("Unresolved witness call site {:?}", call_site));
                rewrites.insert(
                    (block_id, idx),
                    self.specs[*callee_spec].specialized_func_id,
                );
            }
        }

        let function = self.ssa.get_function_mut(spec.specialized_func_id);
        for ((block_id, idx), callee_id) in rewrites {
            let instruction = function
                .get_block_mut(block_id)
                .get_instructions_mut()
                .nth(idx)
                .unwrap_or_else(|| {
                    panic!("Missing call instruction {} in block {:?}", idx, block_id)
                });
            let OpCode::Call { function, .. } = instruction else {
                panic!("Call rewrite target is no longer a call");
            };
            let CallTarget::Static(target) = function else {
                panic!("Call rewrite target is no longer a static call");
            };
            *target = callee_id;
        }
    }

    fn close_key(&self, key: SpecKey) -> SpecKey {
        let layout = &self.layouts[&key.original_func_id];
        let summary = &self.summaries[&key.original_func_id];
        let mut active = HashSet::new();

        for (shape, concrete) in layout.parameters.iter().zip(&key.parameters) {
            collect_witness_ports(shape, concrete, &mut active);
        }
        for (shape, concrete) in layout.returns.iter().zip(&key.returns) {
            collect_witness_ports(shape, concrete, &mut active);
        }
        if key.cfg_witness.is_witness() {
            active.insert(layout.cfg.0);
        }

        let closed = summary.close(layout, active);
        SpecKey {
            original_func_id: key.original_func_id,
            parameters: layout
                .parameters
                .iter()
                .map(|shape| concrete_shape_from_ports(shape, &closed))
                .collect(),
            returns: layout
                .returns
                .iter()
                .map(|shape| concrete_shape_from_ports(shape, &closed))
                .collect(),
            cfg_witness: if closed.contains(&layout.cfg.0) {
                WitnessType::Witness
            } else {
                WitnessType::Pure
            },
        }
    }

    fn key_from_spec_data(&self, spec: &SpecInstance, data: &SpecData) -> SpecKey {
        SpecKey {
            original_func_id: spec.original_func_id,
            parameters: data
                .entry_params
                .iter()
                .map(|shape| concretize_shape(shape, &data.witness_vars))
                .collect(),
            returns: data
                .return_shapes
                .iter()
                .map(|shape| concretize_shape(shape, &data.witness_vars))
                .collect(),
            cfg_witness: witness_of_var(data.cfg_var, &data.witness_vars),
        }
    }

    fn key_from_call(&self, data: &SpecData, call: &PendingCall) -> SpecKey {
        SpecKey {
            original_func_id: call.callee_id,
            parameters: call
                .args
                .iter()
                .map(|value| concretize_shape(&data.value_shapes[value], &data.witness_vars))
                .collect(),
            returns: call
                .results
                .iter()
                .map(|value| concretize_shape(&data.value_shapes[value], &data.witness_vars))
                .collect(),
            cfg_witness: witness_of_var(call.cfg_var, &data.witness_vars),
        }
    }
}

fn infer_summaries(
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

fn insert_concrete_value_shape(
    value: ValueId,
    data: &SpecData,
    output: &mut HashMap<ValueId, WitnessShape>,
) {
    if let Some(shape) = data.value_shapes.get(&value) {
        output.insert(value, concretize_shape(shape, &data.witness_vars));
    }
}

fn witness_of_var(var: VariableId, witness_vars: &HashSet<VariableId>) -> WitnessType {
    if witness_vars.contains(&var) {
        WitnessType::Witness
    } else {
        WitnessType::Pure
    }
}

fn seed_shape(
    graph: &mut DependencyGraph,
    always: VariableId,
    shape: &VarShape,
    concrete: &WitnessShape,
) {
    match (shape, concrete) {
        (WitnessShape::Scalar(var), WitnessShape::Scalar(wt)) => {
            if wt.is_witness() {
                graph.add_edge(always, *var);
            }
        }
        (WitnessShape::Array(var, inner), WitnessShape::Array(wt, concrete_inner))
        | (WitnessShape::Ref(var, inner), WitnessShape::Ref(wt, concrete_inner)) => {
            if wt.is_witness() {
                graph.add_edge(always, *var);
            }
            seed_shape(graph, always, inner, concrete_inner);
        }
        (WitnessShape::Tuple(var, children), WitnessShape::Tuple(wt, concrete_children)) => {
            if wt.is_witness() {
                graph.add_edge(always, *var);
            }
            for (child, concrete_child) in children.iter().zip(concrete_children.iter()) {
                seed_shape(graph, always, child, concrete_child);
            }
        }
        _ => panic!(
            "Cannot seed mismatched witness shapes: {:?} vs {:?}",
            shape, concrete
        ),
    }
}

fn concretize_shape(shape: &VarShape, witness_vars: &HashSet<VariableId>) -> WitnessShape {
    match shape {
        WitnessShape::Scalar(var) => WitnessShape::Scalar(witness_of_var(*var, witness_vars)),
        WitnessShape::Array(var, inner) => WitnessShape::Array(
            witness_of_var(*var, witness_vars),
            Box::new(concretize_shape(inner, witness_vars)),
        ),
        WitnessShape::Ref(var, inner) => WitnessShape::Ref(
            witness_of_var(*var, witness_vars),
            Box::new(concretize_shape(inner, witness_vars)),
        ),
        WitnessShape::Tuple(var, children) => WitnessShape::Tuple(
            witness_of_var(*var, witness_vars),
            children
                .iter()
                .map(|child| concretize_shape(child, witness_vars))
                .collect(),
        ),
    }
}

fn collect_witness_ports(shape: &PortShape, concrete: &WitnessShape, output: &mut HashSet<usize>) {
    match (shape, concrete) {
        (WitnessShape::Scalar(port), WitnessShape::Scalar(wt)) => {
            if wt.is_witness() {
                output.insert(port.0);
            }
        }
        (WitnessShape::Array(port, inner), WitnessShape::Array(wt, concrete_inner))
        | (WitnessShape::Ref(port, inner), WitnessShape::Ref(wt, concrete_inner)) => {
            if wt.is_witness() {
                output.insert(port.0);
            }
            collect_witness_ports(inner, concrete_inner, output);
        }
        (WitnessShape::Tuple(port, children), WitnessShape::Tuple(wt, concrete_children)) => {
            if wt.is_witness() {
                output.insert(port.0);
            }
            for (child, concrete_child) in children.iter().zip(concrete_children.iter()) {
                collect_witness_ports(child, concrete_child, output);
            }
        }
        _ => panic!(
            "Cannot collect mismatched witness ports: {:?} vs {:?}",
            shape, concrete
        ),
    }
}

fn concrete_shape_from_ports(shape: &PortShape, active: &HashSet<usize>) -> WitnessShape {
    match shape {
        WitnessShape::Scalar(port) => WitnessShape::Scalar(witness_of_port(*port, active)),
        WitnessShape::Array(port, inner) => WitnessShape::Array(
            witness_of_port(*port, active),
            Box::new(concrete_shape_from_ports(inner, active)),
        ),
        WitnessShape::Ref(port, inner) => WitnessShape::Ref(
            witness_of_port(*port, active),
            Box::new(concrete_shape_from_ports(inner, active)),
        ),
        WitnessShape::Tuple(port, children) => WitnessShape::Tuple(
            witness_of_port(*port, active),
            children
                .iter()
                .map(|child| concrete_shape_from_ports(child, active))
                .collect(),
        ),
    }
}

fn witness_of_port(port: PortId, active: &HashSet<usize>) -> WitnessType {
    if active.contains(&port.0) {
        WitnessType::Witness
    } else {
        WitnessType::Pure
    }
}

fn map_ports_to_vars(shape: &PortShape, vars: &VarShape, output: &mut [Option<VariableId>]) {
    match (shape, vars) {
        (WitnessShape::Scalar(port), WitnessShape::Scalar(var)) => output[port.0] = Some(*var),
        (WitnessShape::Array(port, inner), WitnessShape::Array(var, var_inner))
        | (WitnessShape::Ref(port, inner), WitnessShape::Ref(var, var_inner)) => {
            output[port.0] = Some(*var);
            map_ports_to_vars(inner, var_inner, output);
        }
        (WitnessShape::Tuple(port, children), WitnessShape::Tuple(var, var_children)) => {
            output[port.0] = Some(*var);
            for (child, var_child) in children.iter().zip(var_children.iter()) {
                map_ports_to_vars(child, var_child, output);
            }
        }
        _ => panic!(
            "Cannot project mismatched witness shapes: {:?} vs {:?}",
            shape, vars
        ),
    }
}

fn port_shape_to_var_shape(shape: &PortShape, boundary_nodes: &[VariableId]) -> VarShape {
    match shape {
        WitnessShape::Scalar(port) => WitnessShape::Scalar(boundary_nodes[port.0]),
        WitnessShape::Array(port, inner) => WitnessShape::Array(
            boundary_nodes[port.0],
            Box::new(port_shape_to_var_shape(inner, boundary_nodes)),
        ),
        WitnessShape::Ref(port, inner) => WitnessShape::Ref(
            boundary_nodes[port.0],
            Box::new(port_shape_to_var_shape(inner, boundary_nodes)),
        ),
        WitnessShape::Tuple(port, children) => WitnessShape::Tuple(
            boundary_nodes[port.0],
            children
                .iter()
                .map(|child| port_shape_to_var_shape(child, boundary_nodes))
                .collect(),
        ),
    }
}

fn port_shape_for_type(typ: &Type, next_port: &mut usize) -> PortShape {
    let port = PortId(*next_port);
    *next_port += 1;
    match &typ.expr {
        TypeExpr::U(_) | TypeExpr::I(_) | TypeExpr::Field | TypeExpr::Function => {
            WitnessShape::Scalar(port)
        }
        TypeExpr::Array(inner, _) | TypeExpr::Slice(inner) => {
            let inner = port_shape_for_type(inner, next_port);
            WitnessShape::Array(port, Box::new(inner))
        }
        TypeExpr::Ref(inner) => {
            let inner = port_shape_for_type(inner, next_port);
            WitnessShape::Ref(port, Box::new(inner))
        }
        TypeExpr::Tuple(elements) => {
            let children = elements
                .iter()
                .map(|typ| port_shape_for_type(typ, next_port))
                .collect();
            WitnessShape::Tuple(port, children)
        }
        TypeExpr::WitnessOf(_) => {
            panic!("ICE: WitnessOf should not be present at this stage");
        }
    }
}

fn var_shape_for_type(typ: &Type, graph: &mut DependencyGraph) -> VarShape {
    match &typ.expr {
        TypeExpr::U(_) | TypeExpr::I(_) | TypeExpr::Field | TypeExpr::Function => {
            WitnessShape::Scalar(graph.fresh_var())
        }
        TypeExpr::Array(inner, _) | TypeExpr::Slice(inner) => {
            let top = graph.fresh_var();
            let inner = var_shape_for_type(inner, graph);
            WitnessShape::Array(top, Box::new(inner))
        }
        TypeExpr::Ref(inner) => {
            let top = graph.fresh_var();
            let inner = var_shape_for_type(inner, graph);
            WitnessShape::Ref(top, Box::new(inner))
        }
        TypeExpr::Tuple(elements) => {
            let top = graph.fresh_var();
            let children = elements
                .iter()
                .map(|typ| var_shape_for_type(typ, graph))
                .collect();
            WitnessShape::Tuple(top, children)
        }
        TypeExpr::WitnessOf(_) => {
            panic!("ICE: WitnessOf should not be present at this stage");
        }
    }
}

fn pure_shape_for_type(typ: &Type) -> WitnessShape {
    match &typ.expr {
        TypeExpr::U(_) | TypeExpr::I(_) | TypeExpr::Field | TypeExpr::Function => {
            WitnessShape::Scalar(WitnessType::Pure)
        }
        TypeExpr::Array(inner, _) | TypeExpr::Slice(inner) => {
            WitnessShape::Array(WitnessType::Pure, Box::new(pure_shape_for_type(inner)))
        }
        TypeExpr::Ref(inner) => {
            WitnessShape::Ref(WitnessType::Pure, Box::new(pure_shape_for_type(inner)))
        }
        TypeExpr::Tuple(elements) => WitnessShape::Tuple(
            WitnessType::Pure,
            elements.iter().map(pure_shape_for_type).collect(),
        ),
        TypeExpr::WitnessOf(_) => {
            panic!("ICE: WitnessOf should not be present at this stage");
        }
    }
}

fn array_element(shape: &VarShape) -> VarShape {
    match shape {
        WitnessShape::Array(_, inner) => *inner.clone(),
        other => panic!(
            "Array element access on non-array witness shape: {:?}",
            other
        ),
    }
}

fn ref_inner(shape: &VarShape) -> VarShape {
    match shape {
        WitnessShape::Ref(_, inner) => *inner.clone(),
        other => panic!("Ref inner access on non-ref witness shape: {:?}", other),
    }
}

fn tuple_element(shape: &VarShape, idx: usize) -> (VariableId, VarShape) {
    match shape {
        WitnessShape::Tuple(top, children) => (*top, children[idx].clone()),
        other => panic!(
            "Tuple element access on non-tuple witness shape: {:?}",
            other
        ),
    }
}

impl SSAAnotator for WitnessTypeInference {
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
