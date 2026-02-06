use crate::compiler::taint_analysis::{
    ConstantTaint, FunctionTaint, Judgement, Taint, TaintType, TypeVariable,
};
use crate::compiler::union_find::UnionFind;
use petgraph::algo::tarjan_scc;
use petgraph::graph::{DiGraph};
use std::collections::HashMap;

/// Constraint solver for taint analysis
#[derive(Clone)]
pub struct ConstraintSolver {
    pub unification: UnionFind,
    judgements: Vec<Judgement>,
}

impl ConstraintSolver {
    pub fn new(taint_analysis: &FunctionTaint) -> Self {
        ConstraintSolver {
            unification: UnionFind::new(),
            judgements: taint_analysis.get_judgements().clone(),
        }
    }

    pub fn add_assumption(&mut self, left_type: &TaintType, right_type: &TaintType) {
        self.push_deep_eq(left_type, right_type);
    }

    /// Main entry point for constraint solving
    pub fn solve(&mut self) {
        // println!("Solving constraints...");
        // println!("{}", self.judgements_string());
        // println!("Number of judgements: {}", self.num_judgements());
        self.simplify_unions_algebraically();
        // println!("\n\n=== Simplified unions algebraically ===");
        // println!("{}", self.judgements_string());
        // println!("Number of judgements: {}", self.num_judgements());
        // TODO: Implement constraint solving logic
        self.inline_equalities();
        // println!("\n\n=== Inlined equalities ===");
        // println!("{}", self.judgements_string());
        // println!("Number of judgements: {}", self.num_judgements());
        self.blow_up_le_of_meet();
        // println!("\n\n=== Blow up le of meet ===");
        // println!("{}", self.judgements_string());
        // println!("Number of judgements: {}", self.num_judgements());
        self.simplify_unions_algebraically();
        // println!("\n\n=== Simplified unions algebraically ===");
        // println!("{}", self.judgements_string());
        // println!("Number of judgements: {}", self.num_judgements());

        let mut current_judgements = self.num_judgements();
        loop {
            self.simplify_le_const();
            // println!("\n\n=== Simplified le const ===");
            // println!("{}", self.judgements_string());
            // println!("Number of judgements: {}", self.num_judgements());
            self.inline_equalities();
            // println!("\n\n=== Inlined equalities ===");
            // println!("{}", self.judgements_string());
            // println!("Number of judgements: {}", self.num_judgements());
            self.simplify_unions_algebraically();
            // println!("\n\n=== Simplified unions algebraically ===");
            // println!("{}", self.judgements_string());
            // println!("Number of judgements: {}", self.num_judgements());
            if self.num_judgements() == current_judgements {
                break;
            }
            current_judgements = self.num_judgements();
        }
        // println!("\n\n=== EQ / LE CONST LOOP ===");
        // println!("{}", self.judgements_string());
        // println!("Number of judgements: {}", self.num_judgements());

        self.run_defaulting();
        // println!("\n\n=== Defaulting ===");
        // println!("{}", self.judgements_string());
        // println!("Number of judgements: {}", self.num_judgements());

        self.simplify_unions_algebraically();
        self.inline_equalities();
        self.simplify_unions_algebraically();
        self.inline_equalities();

        // println!("\n\n=== Inlined equalities ===");
        // println!("{}", self.judgements_string());
        // println!("Number of judgements: {}", self.num_judgements());

        if self.num_judgements() > 0 {
            println!("About to fail:\n{}", self.judgements_string());
            panic!("Failed to solve constraints");
        }

        // self.unify_cycles();
        // println!("\n\n=== Unified cycles ===");
        // println!("{}", self.judgements_string());
        // println!("Number of judgements: {}", self.num_judgements());
        // self.cleanup_judgements();
        // println!("\n\n=== Cleaned up judgements ===");
        // println!("{}", self.judgements_string());
        // println!("Number of judgements: {}", self.num_judgements());
    }

    pub fn judgements_string(&self) -> String {
        self.judgements
            .iter()
            .map(|j| match j {
                Judgement::Eq(l, r) => {
                    let l_substituted = self.unification.substitute_variables(l);
                    let r_substituted = self.unification.substitute_variables(r);
                    format!(
                        "{} = {}",
                        l_substituted.to_string(),
                        r_substituted.to_string()
                    )
                }
                Judgement::Le(l, r) => {
                    let l_substituted = self.unification.substitute_variables(l);
                    let r_substituted = self.unification.substitute_variables(r);
                    format!(
                        "{} ≤ {}",
                        l_substituted.to_string(),
                        r_substituted.to_string()
                    )
                }
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    pub fn num_judgements(&self) -> usize {
        self.judgements.len()
    }

    fn inline_equalities(&mut self) {
        let mut new_judgements = Vec::new();
        for judgement in &self.judgements {
            match judgement {
                Judgement::Eq(Taint::Variable(l), Taint::Variable(r)) => {
                    self.unification.union(*l, *r);
                }
                Judgement::Eq(Taint::Variable(l), Taint::Constant(t))
                | Judgement::Eq(Taint::Constant(t), Taint::Variable(l)) => {
                    // TODO: Occurs check
                    self.unification.set_taint_for_variable(*l, *t);
                }
                Judgement::Eq(t1, t2) => {
                    let t1_substituted = self.unification.substitute_variables(t1);
                    let t2_substituted = self.unification.substitute_variables(t2);
                    if t1_substituted != t2_substituted {
                        new_judgements.push(Judgement::Eq(t1_substituted, t2_substituted));
                    }
                }
                _ => new_judgements.push(judgement.clone()),
            }
        }
        self.judgements = new_judgements;
    }

    fn flatten_unions(&self, taint: &Taint) -> Vec<Taint> {
        match taint {
            Taint::Union(l, r) => {
                let mut result = Vec::new();
                result.extend(self.flatten_unions(l));
                result.extend(self.flatten_unions(r));
                result
            }
            _ => vec![taint.clone()],
        }
    }

    fn simplify_union_algebraically(&self, taint: Taint) -> Taint {
        match taint {
            Taint::Union(l, r) => {
                let l_simplified =
                    self.simplify_union_algebraically(self.unification.substitute_variables(&l));
                let r_simplified =
                    self.simplify_union_algebraically(self.unification.substitute_variables(&r));
                match (l_simplified, r_simplified) {
                    (Taint::Constant(ConstantTaint::Pure), r) => r,
                    (l, Taint::Constant(ConstantTaint::Pure)) => l,
                    (Taint::Constant(ConstantTaint::Witness), _) => {
                        Taint::Constant(ConstantTaint::Witness)
                    }
                    (_, Taint::Constant(ConstantTaint::Witness)) => {
                        Taint::Constant(ConstantTaint::Witness)
                    }
                    (l, r) => Taint::Union(Box::new(l), Box::new(r)),
                }
            }
            _ => taint,
        }
    }

    fn blow_up_le_of_meet(&mut self) {
        let mut new_judgements = Vec::new();
        for judgement in &self.judgements {
            match judgement {
                Judgement::Le(l, r) => {
                    let flattened_unions = self.flatten_unions(l);
                    for union in flattened_unions {
                        new_judgements.push(Judgement::Le(union, r.clone()));
                    }
                }
                _ => new_judgements.push(judgement.clone()),
            }
        }
        self.judgements = new_judgements;
    }

    fn simplify_unions_algebraically(&mut self) {
        let mut new_judgements = Vec::new();
        for judgement in &self.judgements {
            match judgement {
                Judgement::Le(l, r) => {
                    let l_simplified = self.simplify_union_algebraically(l.clone());
                    let r_simplified = self.simplify_union_algebraically(r.clone());
                    new_judgements.push(Judgement::Le(l_simplified, r_simplified));
                }
                Judgement::Eq(l, r) => {
                    let l_simplified = self.simplify_union_algebraically(l.clone());
                    let r_simplified = self.simplify_union_algebraically(r.clone());
                    new_judgements.push(Judgement::Eq(l_simplified, r_simplified));
                }
            }
        }
        self.judgements = new_judgements;
    }

    fn simplify_le_const(&mut self) {
        let mut new_judgements = Vec::new();
        for judgement in &self.judgements {
            match self.unification.substitute_judgement(judgement) {
                Judgement::Le(Taint::Constant(ConstantTaint::Pure), _) => {}
                Judgement::Le(Taint::Constant(ConstantTaint::Witness), r) => {
                    new_judgements.push(Judgement::Eq(
                        Taint::Constant(ConstantTaint::Witness),
                        r.clone(),
                    ));
                }
                Judgement::Le(l, Taint::Constant(ConstantTaint::Pure)) => {
                    new_judgements.push(Judgement::Eq(
                        l.clone(),
                        Taint::Constant(ConstantTaint::Pure),
                    ));
                }
                Judgement::Le(_, Taint::Constant(ConstantTaint::Witness)) => {}
                _ => new_judgements.push(judgement.clone()),
            }
        }
        self.judgements = new_judgements;
    }

    /// Generate a Graphviz DOT representation of variable inequalities
    pub fn generate_inequality_graph(&self) -> String {
        let mut dot = String::new();
        dot.push_str("digraph InequalityGraph {\n");
        dot.push_str("  rankdir=TB;\n");
        dot.push_str("  node [shape=box];\n\n");

        // Collect all variable inequalities
        for judgement in &self.judgements {
            match judgement {
                Judgement::Le(l, r) => {
                    let l_substituted = self.unification.substitute_variables(l);
                    let r_substituted = self.unification.substitute_variables(r);

                    // Only include variable-to-variable inequalities
                    if let (Taint::Variable(left_var), Taint::Variable(right_var)) =
                        (&l_substituted, &r_substituted)
                    {
                        dot.push_str(&format!(
                            "  V{} -> V{} [label=\"\"];\n",
                            left_var.0, right_var.0
                        ));
                    }
                }
                _ => {}
            }
        }

        dot.push_str("}\n");
        dot
    }

    /// Detect cycles in the inequality graph and unify variables in cycles
    pub fn unify_cycles(&mut self) {
        let mut graph = DiGraph::new();
        let mut var_to_node = HashMap::new();
        let mut node_to_var = HashMap::new();

        // Build the graph from inequalities
        for judgement in &self.judgements {
            match judgement {
                Judgement::Le(l, r) => {
                    let l_substituted = self.unification.substitute_variables(l);
                    let r_substituted = self.unification.substitute_variables(r);

                    if let (Taint::Variable(left_var), Taint::Variable(right_var)) =
                        (&l_substituted, &r_substituted)
                    {
                        // Add nodes if they don't exist
                        let left_node = *var_to_node.entry(*left_var).or_insert_with(|| {
                            let node = graph.add_node(*left_var);
                            node_to_var.insert(node, *left_var);
                            node
                        });

                        let right_node = *var_to_node.entry(*right_var).or_insert_with(|| {
                            let node = graph.add_node(*right_var);
                            node_to_var.insert(node, *right_var);
                            node
                        });

                        // Add edge
                        graph.add_edge(left_node, right_node, ());
                    }
                }
                _ => {}
            }
        }

        // Find strongly connected components using petgraph's tarjan_scc
        let sccs = tarjan_scc(&graph);

        // Unify variables in each cycle
        for scc in sccs {
            if scc.len() > 1 {
                let cycle_vars: Vec<TypeVariable> =
                    scc.iter().map(|&node| node_to_var[&node]).collect();
                println!("Found cycle: {:?}", cycle_vars);
                // Unify all variables in the cycle with the first one
                let representative = cycle_vars[0];
                for &var in &cycle_vars[1..] {
                    self.unification.union(representative, var);
                }
            }
        }
    }

    // /// Clean up judgements by removing reflexive inequalities and duplicates
    // fn cleanup_judgements(&mut self) {
    //     let mut new_judgements = Vec::new();
    //     let mut seen = std::collections::HashSet::new();

    //     for judgement in &self.judgements {
    //         match judgement {
    //             Judgement::Le(l, r) => {
    //                 let l_substituted = self.unification.substitute_variables(l);
    //                 let r_substituted = self.unification.substitute_variables(r);

    //                 // Skip reflexive inequalities (X ≤ X)
    //                 if l_substituted == r_substituted {
    //                     continue;
    //                 }

    //                 // Create a canonical form for deduplication
    //                 let canonical = format!(
    //                     "{} ≤ {}",
    //                     l_substituted.to_string(),
    //                     r_substituted.to_string()
    //                 );
    //                 if seen.insert(canonical) {
    //                     new_judgements.push(judgement.clone());
    //                 }
    //             }
    //             Judgement::Eq(l, r) => {
    //                 let l_substituted = self.unification.substitute_variables(l);
    //                 let r_substituted = self.unification.substitute_variables(r);

    //                 // Skip reflexive equalities (X = X)
    //                 if l_substituted == r_substituted {
    //                     continue;
    //                 }

    //                 // Create a canonical form for deduplication
    //                 let canonical = format!(
    //                     "{} = {}",
    //                     l_substituted.to_string(),
    //                     r_substituted.to_string()
    //                 );
    //                 if seen.insert(canonical) {
    //                     new_judgements.push(judgement.clone());
    //                 }
    //             }
    //         }
    //     }

    //     self.judgements = new_judgements;
    // }

    fn push_deep_eq(&mut self, left_type: &TaintType, right_type: &TaintType) {
        match (left_type, right_type) {
            (TaintType::Primitive(l), TaintType::Primitive(r)) => {
                self.judgements.push(Judgement::Eq(l.clone(), r.clone()));
            }
            (TaintType::NestedImmutable(l, inner_l), TaintType::NestedImmutable(r, inner_r)) => {
                self.judgements.push(Judgement::Eq(l.clone(), r.clone()));
                self.push_deep_eq(inner_l, inner_r);
            }
            (TaintType::NestedMutable(l, inner_l), TaintType::NestedMutable(r, inner_r)) => {
                self.judgements.push(Judgement::Eq(l.clone(), r.clone()));
                self.push_deep_eq(inner_l, inner_r);
            }
            (TaintType::Tuple(l, child_taints_l), TaintType::Tuple(r, child_taints_r)) => {
                self.judgements.push(Judgement::Eq(l.clone(), r.clone()));
                assert_eq!(child_taints_l.len(), child_taints_r.len(), "Tuple arity mismatch in deep equality");
                for (child_l, child_r) in child_taints_l.iter().zip(child_taints_r.iter()) {
                    self.push_deep_eq(child_l, child_r);
                }
            }
            _ => panic!("Cannot unify different taint types {:?} and {:?}", left_type, right_type),
        }
    }

    fn run_defaulting(&mut self) {
        let has_constants = self
            .judgements
            .iter()
            .any(|j| self.unification.substitute_judgement(j).has_constants());
        if !has_constants {
            let mut new_judgements = Vec::new();
            for judgement in &self.judgements {
                match judgement {
                    Judgement::Le(l, r) => {
                        new_judgements.push(Judgement::Eq(
                            l.clone(),
                            Taint::Constant(ConstantTaint::Pure),
                        ));
                        new_judgements.push(Judgement::Eq(
                            r.clone(),
                            Taint::Constant(ConstantTaint::Pure),
                        ));
                    }
                    Judgement::Eq(l, r) => {
                        new_judgements.push(Judgement::Eq(
                            l.clone(),
                            Taint::Constant(ConstantTaint::Pure),
                        ));
                        new_judgements.push(Judgement::Eq(
                            r.clone(),
                            Taint::Constant(ConstantTaint::Pure),
                        ));
                    }
                }
            }
            self.judgements = new_judgements;
        }
    }
}
