use std::collections::{HashMap, HashSet, VecDeque};

use super::super::witness_info::{WitnessShape, WitnessType};
use crate::compiler::ssa::{
    BlockId, FunctionId,
    hlssa::{HLSSA, Type, TypeExpr},
};

pub(super) type VarShape = WitnessShape<VariableId>;
pub(super) type PortShape = WitnessShape<PortId>;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(super) struct VariableId(pub(super) usize);

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(super) struct PortId(pub(super) usize);

/// Concrete specialization boundary: parameter, return, and CFG witness shapes.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub(super) struct SpecKey {
    pub(super) original_func_id: FunctionId,
    pub(super) parameters: Vec<WitnessShape>,
    pub(super) returns: Vec<WitnessShape>,
    pub(super) cfg_witness: WitnessType,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(super) struct CallSite {
    pub(super) caller_func_id: FunctionId,
    pub(super) block_id: BlockId,
    pub(super) instruction_idx: usize,
}

/// Stable port layout for a function boundary.
///
/// Every top-level or nested position in the function parameters and returns gets a port. `cfg`
/// is an extra port for control-flow taint, and `always()` is the synthetic source used for values
/// that are unconditionally witness.
#[derive(Clone, Debug)]
pub(super) struct BoundaryLayout {
    pub(super) parameters: Vec<PortShape>,
    pub(super) returns: Vec<PortShape>,
    pub(super) cfg: PortId,
    pub(super) port_count: usize,
}

impl BoundaryLayout {
    pub(super) fn new(function_id: FunctionId, ssa: &HLSSA) -> Self {
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

    pub(super) fn always(&self) -> usize {
        self.port_count
    }
}

/// Boundary-to-boundary witness propagation summary for one original function.
///
/// An edge `a -> b` means that if boundary port `a` is witness, then `b` must be witness too.
/// `layout.always()` is a synthetic source for unconditional witness production.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct FunctionSummary {
    pub(super) edges: Vec<HashSet<usize>>,
}

impl FunctionSummary {
    pub(super) fn new(layout: &BoundaryLayout) -> Self {
        Self {
            edges: vec![HashSet::new(); layout.port_count + 1],
        }
    }

    pub(super) fn add_edge(&mut self, source: usize, target: usize) -> bool {
        self.edges[source].insert(target)
    }

    pub(super) fn absorb(&mut self, other: &FunctionSummary) -> bool {
        let mut changed = false;
        for (source, targets) in other.edges.iter().enumerate() {
            for target in targets {
                changed |= self.add_edge(source, *target);
            }
        }
        changed
    }

    /// Computes the closure of a requested specialization key under this function's summary.
    pub(super) fn close(
        &self,
        layout: &BoundaryLayout,
        mut active: HashSet<usize>,
    ) -> HashSet<usize> {
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
pub(super) struct DependencyGraph {
    edges: HashMap<VariableId, Vec<VariableId>>,
    next_variable: usize,
}

impl DependencyGraph {
    pub(super) fn new() -> Self {
        Self {
            edges: HashMap::new(),
            next_variable: 0,
        }
    }

    pub(super) fn fresh_var(&mut self) -> VariableId {
        let var = VariableId(self.next_variable);
        self.next_variable += 1;
        var
    }

    pub(super) fn add_edge(&mut self, source: VariableId, target: VariableId) {
        if source == target {
            return;
        }
        let targets = self.edges.entry(source).or_default();
        if !targets.contains(&target) {
            targets.push(target);
        }
    }

    pub(super) fn reachable_from(
        &self,
        sources: impl IntoIterator<Item = VariableId>,
    ) -> HashSet<VariableId> {
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

pub(super) fn witness_of_var(var: VariableId, witness_vars: &HashSet<VariableId>) -> WitnessType {
    if witness_vars.contains(&var) {
        WitnessType::Witness
    } else {
        WitnessType::Pure
    }
}

/// Seeds concrete witness requirements into a variable graph.
///
/// Spec solving starts from a closed `SpecKey`; every witness position in that key becomes an
/// edge from `always`, so normal reachability computes all implied witness variables.
pub(super) fn seed_shape(
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

/// Reads a variable-shaped value back into a concrete pure/witness shape.
pub(super) fn concretize_shape(
    shape: &VarShape,
    witness_vars: &HashSet<VariableId>,
) -> WitnessShape {
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

/// Converts a concrete key fragment into the set of witness ports requested by that key.
pub(super) fn collect_witness_ports(
    shape: &PortShape,
    concrete: &WitnessShape,
    output: &mut HashSet<usize>,
) {
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

pub(super) fn concrete_shape_from_ports(
    shape: &PortShape,
    active: &HashSet<usize>,
) -> WitnessShape {
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

/// Projects a callee boundary layout onto the caller's value variables at one call site.
pub(super) fn map_ports_to_vars(
    shape: &PortShape,
    vars: &VarShape,
    output: &mut [Option<VariableId>],
) {
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

pub(super) fn port_shape_to_var_shape(
    shape: &PortShape,
    boundary_nodes: &[VariableId],
) -> VarShape {
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

pub(super) fn var_shape_for_type(typ: &Type, graph: &mut DependencyGraph) -> VarShape {
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

pub(super) fn pure_shape_for_type(typ: &Type) -> WitnessShape {
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

pub(super) fn array_element(shape: &VarShape) -> VarShape {
    match shape {
        WitnessShape::Array(_, inner) => *inner.clone(),
        other => panic!(
            "Array element access on non-array witness shape: {:?}",
            other
        ),
    }
}

pub(super) fn ref_inner(shape: &VarShape) -> VarShape {
    match shape {
        WitnessShape::Ref(_, inner) => *inner.clone(),
        other => panic!("Ref inner access on non-ref witness shape: {:?}", other),
    }
}

pub(super) fn tuple_element(shape: &VarShape, idx: usize) -> (VariableId, VarShape) {
    match shape {
        WitnessShape::Tuple(top, children) => (*top, children[idx].clone()),
        other => panic!(
            "Tuple element access on non-tuple witness shape: {:?}",
            other
        ),
    }
}
