use petgraph::Direction;
use petgraph::algo::dominators::{self, Dominators};
use petgraph::graph::{DiGraph, EdgeIndex, NodeIndex};
use petgraph::visit::{Bfs, DfsPostOrder, EdgeRef, Walker};
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::PathBuf;

use crate::compiler::ssa::{BlockId, CallTarget, FunctionId, OpCode, SSA, Terminator};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum JumpType {
    Entry,
    Jmp,
    JmpIf,
    Return,
}

#[derive(Debug, Clone)]
pub struct CFGData {
    cfg: DiGraph<(), JumpType>,
    dominators: Dominators<NodeIndex<u32>>,
    _dominator_post_order: Vec<NodeIndex<u32>>,
    node_dominator_post_order: HashMap<NodeIndex<u32>, usize>,
    dominator_pre_order: Vec<NodeIndex<u32>>,
    node_dominator_pre_order: HashMap<NodeIndex<u32>, usize>,
}

impl CFGData {
    pub fn from_graph(cfg: DiGraph<(), JumpType>, entry: NodeIndex<u32>) -> Self {
        let dominators = dominators::simple_fast(&cfg, entry);
        let mut preorder = Vec::new();
        let mut postorder = Vec::new();
        let mut preorder_map = HashMap::new();
        let mut postorder_map = HashMap::new();
        Self::compute_dominator_traversals(
            entry,
            &dominators,
            &mut preorder,
            &mut postorder,
            &mut preorder_map,
            &mut postorder_map,
        );

        Self {
            cfg,
            dominators,
            _dominator_post_order: postorder,
            node_dominator_post_order: postorder_map,
            dominator_pre_order: preorder,
            node_dominator_pre_order: preorder_map,
        }
    }

    fn compute_dominator_traversals(
        entry: NodeIndex<u32>,
        dominators: &Dominators<NodeIndex<u32>>,
        preorder: &mut Vec<NodeIndex<u32>>,
        postorder: &mut Vec<NodeIndex<u32>>,
        preorder_map: &mut HashMap<NodeIndex<u32>, usize>,
        postorder_map: &mut HashMap<NodeIndex<u32>, usize>,
    ) {
        preorder_map.insert(entry, preorder.len());
        preorder.push(entry);
        for child in dominators.immediately_dominated_by(entry) {
            if entry == child {
                continue;
            }
            Self::compute_dominator_traversals(
                child,
                &dominators,
                preorder,
                postorder,
                preorder_map,
                postorder_map,
            );
        }
        postorder_map.insert(entry, postorder.len());
        postorder.push(entry);
    }

    pub fn dominates(&self, a: NodeIndex<u32>, b: NodeIndex<u32>) -> bool {
        let Some(a_pre_order) = self.node_dominator_pre_order.get(&a) else {
            return false;
        };
        let Some(b_pre_order) = self.node_dominator_pre_order.get(&b) else {
            return false;
        };
        let Some(a_post_order) = self.node_dominator_post_order.get(&a) else {
            return false;
        };
        let Some(b_post_order) = self.node_dominator_post_order.get(&b) else {
            return false;
        };
        a_pre_order <= b_pre_order && a_post_order >= b_post_order
    }

    pub fn get_dominance_frontier(&self, node: NodeIndex<u32>) -> HashSet<NodeIndex<u32>> {
        let mut result = HashSet::new();
        for candidate in self.cfg.node_indices() {
            if self.dominates(node, candidate) {
                continue;
            }
            for predecessor in self.cfg.edges_directed(candidate, Direction::Incoming) {
                if self.dominates(node, predecessor.source()) {
                    result.insert(candidate);
                    break;
                }
            }
        }
        result
    }
}

struct CFGBuilder {
    cfg: DiGraph<(), JumpType>,
    block_to_node: HashMap<BlockId, NodeIndex<u32>>,
    node_to_block: HashMap<NodeIndex<u32>, BlockId>,
    entry_node: NodeIndex<u32>,
    return_node: NodeIndex<u32>,
}

impl CFGBuilder {
    pub fn new() -> Self {
        let mut cfg = DiGraph::new();
        let entry_node = cfg.add_node(());
        let return_node = cfg.add_node(());
        Self {
            cfg,
            block_to_node: HashMap::new(),
            node_to_block: HashMap::new(),
            entry_node,
            return_node,
        }
    }

    pub fn add_jump(&mut self, source: BlockId, target: BlockId, jump_type: JumpType) {
        let source_node = *self
            .block_to_node
            .entry(source)
            .or_insert_with(|| self.cfg.add_node(()));
        self.node_to_block.insert(source_node, source);
        let target_node = *self
            .block_to_node
            .entry(target)
            .or_insert_with(|| self.cfg.add_node(()));
        self.node_to_block.insert(target_node, target);
        self.cfg.add_edge(source_node, target_node, jump_type);
    }

    pub fn set_entry(&mut self, block_id: BlockId) {
        let node = *self.block_to_node.get(&block_id).unwrap();
        self.cfg.add_edge(self.entry_node, node, JumpType::Entry);
    }

    pub fn add_block(&mut self, block_id: BlockId) {
        let node = *self
            .block_to_node
            .entry(block_id)
            .or_insert_with(|| self.cfg.add_node(()));
        self.node_to_block.insert(node, block_id);
    }

    pub fn add_return(&mut self, block_id: BlockId) {
        let node = *self
            .block_to_node
            .entry(block_id)
            .or_insert_with(|| self.cfg.add_node(()));
        self.node_to_block.insert(node, block_id);
        self.cfg.add_edge(node, self.return_node, JumpType::Return);
    }

    pub fn build(self) -> CFG {
        let cfg_data = CFGData::from_graph(self.cfg.clone(), self.entry_node);
        let mut reverse_cfg = self.cfg;
        reverse_cfg.reverse();
        let reverse_cfg_data = CFGData::from_graph(reverse_cfg, self.return_node);
        CFG {
            entry_node: self.entry_node,
            return_node: self.return_node,
            block_to_node: self.block_to_node,
            node_to_block: self.node_to_block,
            cfg: cfg_data,
            reverse_cfg: reverse_cfg_data,
        }
    }
}

#[derive(Debug, Clone)]
pub struct CFG {
    entry_node: NodeIndex<u32>,
    return_node: NodeIndex<u32>,
    block_to_node: HashMap<BlockId, NodeIndex<u32>>,
    node_to_block: HashMap<NodeIndex<u32>, BlockId>,
    cfg: CFGData,
    reverse_cfg: CFGData,
}

impl CFG {
    pub fn get_blocks_bfs(&self) -> impl Iterator<Item = BlockId> {
        Bfs::new(&self.cfg.cfg, self.entry_node)
            .iter(&self.cfg.cfg)
            .filter_map(|node| self.node_to_block.get(&node).cloned())
    }

    pub fn is_loop_entry(&self, block_id: BlockId) -> bool {
        let node = *self.block_to_node.get(&block_id).unwrap();
        for predecessor in self.predecessors(node) {
            if self.cfg.dominates(node, predecessor) {
                return true;
            }
        }
        false
    }

    fn predecessors(&self, node: NodeIndex<u32>) -> impl Iterator<Item = NodeIndex<u32>> {
        self.cfg
            .cfg
            .edges_directed(node, Direction::Incoming)
            .map(|edge| edge.source())
    }

    pub fn get_predecessors(&self, block_id: BlockId) -> impl Iterator<Item = BlockId> {
        let node = *self.block_to_node.get(&block_id).unwrap();
        self.predecessors(node)
            .filter_map(|node| self.node_to_block.get(&node).cloned())
    }

    fn successors(&self, node: NodeIndex<u32>) -> impl Iterator<Item = NodeIndex<u32>> {
        self.cfg
            .cfg
            .edges_directed(node, Direction::Outgoing)
            .map(|edge| edge.target())
    }

    pub fn get_successors(&self, block_id: BlockId) -> impl Iterator<Item = BlockId> {
        let node = *self.block_to_node.get(&block_id).unwrap();
        self.successors(node)
            .filter_map(|node| self.node_to_block.get(&node).cloned())
    }

    fn get_merge_point_node(&self, node: NodeIndex<u32>) -> NodeIndex {
        self.reverse_cfg
            .dominators
            .immediate_dominator(node)
            .unwrap()
    }

    pub fn get_merge_point(&self, block_id: BlockId) -> BlockId {
        let node = *self.block_to_node.get(&block_id).unwrap();
        *self
            .node_to_block
            .get(&self.get_merge_point_node(node))
            .unwrap()
    }

    pub fn get_if_body(&self, block_id: BlockId) -> Vec<BlockId> {
        let entry = *self.block_to_node.get(&block_id).unwrap();
        let merge = self.get_merge_point_node(entry);
        let mut result = Vec::new();
        for node in self.cfg.cfg.node_indices() {
            if node == entry || node == merge {
                continue;
            }
            if self.cfg.dominates(entry, node) && self.reverse_cfg.dominates(merge, node) {
                result.push(*self.node_to_block.get(&node).unwrap());
            }
        }
        result
    }

    pub fn get_jumps_into_merge_from_branch(
        &self,
        branch: BlockId,
        merge: BlockId,
    ) -> Vec<BlockId> {
        let branch_id = *self.block_to_node.get(&branch).unwrap();
        let merge_id = *self.block_to_node.get(&merge).unwrap();
        let mut result = Vec::new();
        for predecessor in self.predecessors(merge_id) {
            if self.cfg.dominates(branch_id, predecessor) {
                result.push(*self.node_to_block.get(&predecessor).unwrap());
            }
        }
        result
    }

    // Finds unconditional jumps into blocks that only have one incoming edge.
    // Returns a list of (source, target) pairs.
    pub fn find_redundant_jumps(&self) -> Vec<(BlockId, BlockId)> {
        let mut result = Vec::new();
        for node in self.cfg.cfg.node_indices() {
            let in_edges = self
                .cfg
                .cfg
                .edges_directed(node, Direction::Incoming)
                .collect::<Vec<_>>();
            if in_edges.len() != 1 {
                continue;
            }
            let in_edge = in_edges[0];
            if in_edge.weight() != &JumpType::Jmp {
                continue;
            }
            let source = *self.node_to_block.get(&in_edge.source()).unwrap();
            let target = *self.node_to_block.get(&node).unwrap();
            result.push((source, target));
        }
        result
    }

    pub fn get_dominance_frontier(&self, block_id: BlockId) -> HashSet<BlockId> {
        let node = *self.block_to_node.get(&block_id).unwrap();
        let r = self.cfg.get_dominance_frontier(node);
        r.into_iter()
            .filter_map(|node| self.node_to_block.get(&node).cloned())
            .collect()
    }

    pub fn get_post_dominance_frontier(&self, block_id: BlockId) -> HashSet<BlockId> {
        let node = *self.block_to_node.get(&block_id).unwrap();
        let r = self.reverse_cfg.get_dominance_frontier(node);
        r.into_iter()
            .filter_map(|node| self.node_to_block.get(&node).cloned())
            .collect()
    }

    pub fn get_domination_pre_order(&self) -> impl Iterator<Item = BlockId> {
        self.cfg
            .dominator_pre_order
            .iter()
            .filter_map(|node| self.node_to_block.get(node).copied())
    }

    pub fn get_immediate_dominator(&self, block_id: BlockId) -> Option<BlockId> {
        let node = *self.block_to_node.get(&block_id).unwrap();
        let dom = self.cfg.dominators.immediate_dominator(node);
        if let Some(dom) = dom {
            self.node_to_block.get(&dom).copied()
        } else {
            None
        }
    }

    pub fn dominates(&self, a: BlockId, b: BlockId) -> bool {
        let a_node = *self.block_to_node.get(&a).unwrap();
        let b_node = *self.block_to_node.get(&b).unwrap();
        self.cfg.dominates(a_node, b_node)
    }

    pub fn post_dominates(&self, a: BlockId, b: BlockId) -> bool {
        let a_node = *self.block_to_node.get(&a).unwrap();
        let b_node = *self.block_to_node.get(&b).unwrap();
        self.reverse_cfg.dominates(a_node, b_node)
    }

    pub fn get_jumps_into(&self, block_id: BlockId) -> Vec<BlockId> {
        let node = *self.block_to_node.get(&block_id).unwrap();
        let r = self.cfg.cfg.edges_directed(node, Direction::Incoming);
        r.into_iter()
            .filter_map(|edge| self.node_to_block.get(&edge.source()).cloned())
            .collect()
    }

    pub fn get_post_dominator(&self, block_id: BlockId) -> BlockId {
        let node = *self.block_to_node.get(&block_id).unwrap();
        let r = self.reverse_cfg.dominators.immediate_dominator(node);
        if let Some(dom) = r {
            *self.node_to_block.get(&dom).unwrap()
        } else {
            panic!("ICE: block has no post-dominator");
        }
    }

    fn is_loop_backedge(&self, edge: EdgeIndex) -> bool {
        let (source, target) = self.cfg.cfg.edge_endpoints(edge).unwrap();
        self.cfg.dominates(target, source)
    }

    pub fn get_return_blocks(&self) -> impl Iterator<Item = BlockId> {
        self.cfg
            .cfg
            .edges_directed(self.return_node, Direction::Incoming)
            .filter_map(|edge| self.node_to_block.get(&edge.source()).cloned())
    }

    pub fn get_edges(&self) -> impl Iterator<Item = (BlockId, BlockId)> {
        self.cfg.cfg.edge_indices().filter_map(|edge| {
            let (source, target) = self.cfg.cfg.edge_endpoints(edge).unwrap();
            self.node_to_block
                .get(&source)
                .and_then(|src| self.node_to_block.get(&target).map(|tgt| (*src, *tgt)))
        })
    }
}

impl CFG {
    pub fn generate_image(
        &self,
        output_path: PathBuf,
        ssa: &crate::compiler::ssa::SSA,
        function: &crate::compiler::ssa::Function,
        func_id: FunctionId,
        label: String,
    ) -> Result<(), Box<dyn std::error::Error>>
    {
        use std::fs;
        use std::process::Command;

        // Generate DOT content
        let mut dot_content = String::new();
        dot_content.push_str("digraph CFG {\n");
        dot_content.push_str("  rankdir=TB;\n");
        dot_content.push_str(&format!("  label=\"{}\";\n", label));
        dot_content.push_str("  labelloc=t;\n");
        dot_content.push_str("  node [shape=box, fontname=\"monospace\"];\n\n");

        let constants = function
            .iter_consts()
            .map(|(id, cst)| format!("v{} = {:?}", id.0, cst))
            .collect::<Vec<_>>()
            .join("\\l");

        // Add special nodes (entry and return)
        dot_content.push_str(&format!(
            "  entry [label=\"ENTRY:\\n{constants}\\l\", style=filled, fillcolor=lightgreen];\n"
        ));
        dot_content.push_str(
            "  return [label=\"RETURN\", shape=ellipse, style=filled, fillcolor=lightcoral];\n\n",
        );

        // Add nodes with block content
        for (_, block_id) in &self.node_to_block {
            let block = function.get_block(*block_id);
            let block_content = block.to_string(
                ssa,
                func_id,
                *block_id,
                &crate::compiler::ssa::DefaultSsaAnnotator,
            );

            // Escape special characters for DOT
            let escaped_content = block_content
                .replace("\"", "\\\"")
                .replace("\n", "\\l")
                .replace("\r", "\\r")
                .replace("\t", "\\t");

            dot_content.push_str(&format!(
                "  block_{} [label=\"{}\\l\"];\n",
                block_id.0, escaped_content
            ));
        }

        dot_content.push_str("\n");

        // Add edges with different colors for different jump types
        for edge in self.cfg.cfg.edge_indices() {
            let (source, target) = self.cfg.cfg.edge_endpoints(edge).unwrap();
            let edge_weight = &self.cfg.cfg[edge];

            // Handle special nodes (entry and return)
            let source_name = if source == self.entry_node {
                "entry".to_string()
            } else if source == self.return_node {
                "return".to_string()
            } else {
                let source_block = self.node_to_block[&source];
                format!("block_{}", source_block.0)
            };

            let target_name = if target == self.entry_node {
                "entry".to_string()
            } else if target == self.return_node {
                "return".to_string()
            } else {
                let target_block = self.node_to_block[&target];
                format!("block_{}", target_block.0)
            };

            let edge_style = match edge_weight {
                crate::compiler::flow_analysis::JumpType::Jmp => {
                    if self.is_loop_backedge(edge) {
                        "[color=blue, constraint=false]"
                    } else {
                        "[color=black]"
                    }
                }
                crate::compiler::flow_analysis::JumpType::JmpIf => "[color=red]",
                crate::compiler::flow_analysis::JumpType::Entry => "[color=green, style=dashed]",
                crate::compiler::flow_analysis::JumpType::Return => "[color=orange, style=dashed]",
            };

            dot_content.push_str(&format!(
                "  {} -> {} {};\n",
                source_name, target_name, edge_style
            ));
        }

        dot_content.push_str("}\n");

        // Write DOT content to a temporary file
        let temp_dot_path = format!("temp_cfg_{}.dot", func_id.0);
        fs::write(&temp_dot_path, dot_content)?;

        // Use system dot command to generate PNG
        let output = Command::new("dot")
            .args(&["-Tpng", &temp_dot_path, "-o", output_path.to_str().unwrap()])
            .output()?;

        // Clean up temporary file
        let _ = fs::remove_file(&temp_dot_path);

        if output.status.success() {
            Ok(())
        } else {
            let error_msg = String::from_utf8_lossy(&output.stderr);
            Err(format!("Graphviz dot command failed: {}", error_msg).into())
        }
    }
}

#[derive(Debug, Clone)]
pub struct CallGraph {
    call_graph: DiGraph<(), ()>,
    func_to_node: HashMap<FunctionId, NodeIndex<u32>>,
    node_to_func: HashMap<NodeIndex<u32>, FunctionId>,
}

impl CallGraph {
    pub fn new() -> Self {
        CallGraph {
            call_graph: DiGraph::new(),
            func_to_node: HashMap::new(),
            node_to_func: HashMap::new(),
        }
    }

    pub fn add_call(&mut self, caller: FunctionId, callee: FunctionId) {
        let caller_node = *self.func_to_node.get(&caller).unwrap();
        let callee_node = *self.func_to_node.get(&callee).unwrap();
        self.call_graph.add_edge(caller_node, callee_node, ());
    }

    pub fn add_function(&mut self, func_id: FunctionId) {
        let node = *self
            .func_to_node
            .entry(func_id)
            .or_insert_with(|| self.call_graph.add_node(()));
        self.node_to_func.insert(node, func_id);
    }

    pub fn detect_loops(&self) -> Vec<Vec<FunctionId>> {
        petgraph::algo::kosaraju_scc(&self.call_graph)
            .into_iter()
            .filter(|scc| scc.len() > 1)
            .map(|scc| {
                scc.into_iter()
                    .map(|node| self.node_to_func[&node])
                    .collect()
            })
            .collect()
    }

    pub fn get_post_order(&self, main_fn_id: FunctionId) -> impl Iterator<Item = FunctionId> {
        DfsPostOrder::new(&self.call_graph, self.func_to_node[&main_fn_id])
            .iter(&self.call_graph)
            .filter_map(|node| self.node_to_func.get(&node).cloned())
    }
}

impl CallGraph {
    pub fn generate_image(
        &self,
        output_path: PathBuf,
        ssa: &SSA,
    ) -> Result<(), Box<dyn std::error::Error>>
    {
        use std::fs;
        use std::process::Command;

        // Generate DOT content
        let mut dot_content = String::new();
        dot_content.push_str("digraph CallGraph {\n");
        dot_content.push_str("  rankdir=TB;\n");
        dot_content.push_str("  node [shape=box];\n\n");

        // Add nodes with function_name@function_id labels
        for (_, func_id) in &self.node_to_func {
            let function = ssa.get_function(*func_id);
            let function_name = function.get_name();
            dot_content.push_str(&format!(
                "  func_{} [label=\"{}@{}\"];\n",
                func_id.0, function_name, func_id.0
            ));
        }

        dot_content.push_str("\n");

        // Add edges
        for edge in self.call_graph.edge_indices() {
            let (source, target) = self.call_graph.edge_endpoints(edge).unwrap();
            let source_func = self.node_to_func[&source];
            let target_func = self.node_to_func[&target];
            dot_content.push_str(&format!(
                "  func_{} -> func_{};\n",
                source_func.0, target_func.0
            ));
        }

        dot_content.push_str("}\n");

        // Write DOT content to a temporary file
        let temp_dot_path = "temp_call_graph.dot";
        fs::write(temp_dot_path, dot_content)?;

        // Use system dot command to generate PNG
        let output = Command::new("dot")
            .args(&["-Tpng", temp_dot_path, "-o", output_path.to_str().unwrap()])
            .output()?;

        // Clean up temporary file
        let _ = fs::remove_file(temp_dot_path);

        if output.status.success() {
            Ok(())
        } else {
            let error_msg = String::from_utf8_lossy(&output.stderr);
            Err(format!("Graphviz dot command failed: {}", error_msg).into())
        }
    }
}

#[derive(Debug, Clone)]
pub struct FlowAnalysis {
    call_graph: CallGraph,
    pub function_cfgs: HashMap<FunctionId, CFG>,
}

impl FlowAnalysis {
    pub fn run(ssa: &SSA) -> Self {
        let mut call_graph = CallGraph::new();
        let mut function_cfgs = HashMap::new();

        for (func_id, _) in ssa.iter_functions() {
            call_graph.add_function(*func_id);
        }

        for (func_id, func) in ssa.iter_functions() {
            let mut cfg = CFGBuilder::new();

            for (block_id, _) in func.get_blocks() {
                cfg.add_block(*block_id);
            }

            cfg.set_entry(func.get_entry_id());

            for (block_id, block) in func.get_blocks() {
                if let Some(instruction) = block.get_terminator() {
                    match instruction {
                        Terminator::Jmp(target, _) => {
                            cfg.add_jump(*block_id, *target, JumpType::Jmp);
                        }
                        Terminator::JmpIf(_, t1, t2) => {
                            cfg.add_jump(*block_id, *t1, JumpType::JmpIf);
                            cfg.add_jump(*block_id, *t2, JumpType::JmpIf);
                        }
                        Terminator::Return(_) => {
                            cfg.add_return(*block_id);
                        }
                    }
                }
                for instruction in block.get_instructions() {
                    match instruction {
                        OpCode::Call { results: _, function: CallTarget::Static(tgt_id), args: _ } => {
                            call_graph.add_call(*func_id, *tgt_id);
                        }
                        OpCode::Call { function: CallTarget::Dynamic(_), .. } => {
                            panic!("Dynamic calls should be eliminated by defunctionalization before flow analysis");
                        }
                        _ => {}
                    }
                }
            }
            function_cfgs.insert(*func_id, cfg.build());
        }

        Self {
            call_graph,
            function_cfgs,
        }
    }

    pub fn get_call_graph(&self) -> &CallGraph {
        &self.call_graph
    }

    pub fn get_function_cfg(&self, function_id: FunctionId) -> &CFG {
        &self.function_cfgs[&function_id]
    }

    pub fn generate_images(
        &self,
        debug_output_dir: PathBuf,
        ssa: &SSA,
        phase_label: String,
    ) {
        if !debug_output_dir.exists() {
            fs::create_dir(&debug_output_dir).unwrap();
        }

        self.call_graph
            .generate_image(debug_output_dir.join("call_graph.png"), ssa)
            .unwrap();
        for (func_id, _cfg) in &self.function_cfgs {
            self.function_cfgs[func_id]
                .generate_image(
                    debug_output_dir.join(format!(
                        "cfg_{}@{}.png",
                        ssa.get_function(*func_id).get_name(),
                        func_id.0
                    )),
                    ssa,
                    ssa.get_function(*func_id),
                    *func_id,
                    format!(
                        "CFG for {} {}",
                        ssa.get_function(*func_id).get_name(),
                        phase_label
                    ),
                )
                .unwrap();
        }
    }

    // pub fn to_graphviz(&self) -> String {
    //     let mut dot = String::new();
    //     dot.push_str("digraph FlowAnalysis {\n");
    //     dot.push_str("  rankdir=TB;\n");
    //     dot.push_str("  compound=true;\n");
    //     dot.push_str("  node [shape=box];\n\n");

    //     for (func_id, _) in &self.function_cfgs {
    //         dot.push_str(&format!(
    //             "  entry_{} [label=\"\", shape=point, style=invis];\n",
    //             func_id.0
    //         ));
    //         dot.push_str(&format!(
    //             "  return_{} [label=\"\", shape=point, style=invis];\n",
    //             func_id.0
    //         ));
    //     }

    //     dot.push_str("\n");

    //     dot.push_str("  { rank=source;\n");
    //     for (func_id, _) in &self.function_cfgs {
    //         dot.push_str(&format!("    entry_{};\n", func_id.0));
    //     }
    //     dot.push_str("  }\n\n");

    //     dot.push_str("  { rank=sink;\n");
    //     for (func_id, _) in &self.function_cfgs {
    //         dot.push_str(&format!("    return_{};\n", func_id.0));
    //     }
    //     dot.push_str("  }\n\n");

    //     dot.push_str("  subgraph cluster_call_graph {\n");
    //     dot.push_str("    label=\"Call Graph\";\n");
    //     dot.push_str("    style=filled;\n");
    //     dot.push_str("    color=lightgrey;\n");
    //     dot.push_str("    node [style=filled,color=white];\n");

    //     for (func_id, node_index) in &self.func_to_node {
    //         dot.push_str(&format!(
    //             "    func_{} [label=\"fn_{}\"];\n",
    //             func_id.0, func_id.0
    //         ));
    //     }

    //     for edge in self.call_graph.edge_indices() {
    //         let (source, target) = self.call_graph.edge_endpoints(edge).unwrap();
    //         let source_func = self.node_to_func[&source];
    //         let target_func = self.node_to_func[&target];
    //         dot.push_str(&format!(
    //             "    func_{} -> func_{};\n",
    //             source_func.0, target_func.0
    //         ));
    //     }

    //     dot.push_str("  }\n\n");

    //     for (func_id, cfg) in &self.function_cfgs {
    //         dot.push_str(&format!("  subgraph cluster_cfg_{} {{\n", func_id.0));
    //         dot.push_str(&format!("    label=\"CFG fn_{}\";\n", func_id.0));
    //         dot.push_str("    style=filled;\n");
    //         dot.push_str("    color=lightblue;\n");
    //         dot.push_str("    node [style=filled,color=white];\n");

    //         for (block_id, node_index) in &cfg.block_to_node {
    //             dot.push_str(&format!(
    //                 "    block_{}_{} [label=\"block_{}\"];\n",
    //                 func_id.0, block_id.0, block_id.0
    //             ));
    //         }

    //         for edge in cfg.cfg.edge_indices() {
    //             let (source, target) = cfg.cfg.edge_endpoints(edge).unwrap();
    //             let edge_weight = &cfg.cfg[edge];
    //             match edge_weight {
    //                 JumpType::Jmp => {
    //                     let source_block = cfg.node_to_block[&source];
    //                     let target_block = cfg.node_to_block[&target];
    //                     dot.push_str(&format!(
    //                         "    block_{}_{} -> block_{}_{};\n",
    //                         func_id.0, source_block.0, func_id.0, target_block.0
    //                     ));
    //                 }
    //                 JumpType::JmpIf => {
    //                     let source_block = cfg.node_to_block[&source];
    //                     let target_block = cfg.node_to_block[&target];
    //                     dot.push_str(&format!(
    //                         "    block_{}_{} -> block_{}_{} [color=red];\n",
    //                         func_id.0, source_block.0, func_id.0, target_block.0
    //                     ));
    //                 }
    //                 _ => {} // Skip Entry and Return edges - they'll be handled outside
    //             }
    //         }

    //         dot.push_str("  }\n\n");
    //     }

    //     // Add entry and return edges outside the subgraphs
    //     for (func_id, cfg) in &self.function_cfgs {
    //         for edge in cfg.cfg.edge_indices() {
    //             let (source, target) = cfg.cfg.edge_endpoints(edge).unwrap();
    //             let edge_weight = &cfg.cfg[edge];
    //             match edge_weight {
    //                 JumpType::Return => {
    //                     let source_block = cfg.node_to_block[&source];
    //                     dot.push_str(&format!(
    //                         "  block_{}_{} -> return_{} [style=dashed, color=red];\n",
    //                         func_id.0, source_block.0, func_id.0
    //                     ));
    //                 }
    //                 JumpType::Entry => {
    //                     let target_block = cfg.node_to_block[&target];
    //                     dot.push_str(&format!(
    //                         "  entry_{} -> block_{}_{} [style=dashed, color=green];\n",
    //                         func_id.0, func_id.0, target_block.0
    //                     ));
    //                 }
    //                 _ => {} // Skip Jmp and JmpIf edges - they're handled inside subgraphs
    //             }
    //         }
    //     }

    //     dot.push_str("}\n");
    //     dot
    // }

    // pub fn save_as_png(&self, filename: &str) -> Result<(), String> {
    //     let dot_content = self.to_graphviz();

    //     let temp_dot_file = format!("{}.dot", filename);
    //     let mut file = fs::File::create(&temp_dot_file)
    //         .map_err(|e| format!("Failed to create temporary dot file: {}", e))?;

    //     file.write_all(dot_content.as_bytes())
    //         .map_err(|e| format!("Failed to write dot content: {}", e))?;

    //     let output = Command::new("dot")
    //         .args(&["-Tpng", &temp_dot_file, "-o", &format!("{}.png", filename)])
    //         .output()
    //         .map_err(|e| format!("Failed to execute dot command: {}", e))?;

    //     if !output.status.success() {
    //         let stderr = String::from_utf8_lossy(&output.stderr);
    //         return Err(format!("dot command failed: {}", stderr));
    //     }

    //     fs::remove_file(&temp_dot_file)
    //         .map_err(|e| format!("Failed to remove temporary dot file: {}", e))?;

    //     Ok(())
    // }
}
