use crate::compiler::taint_analysis::{ConstantTaint, Judgement, Taint, TaintType, TypeVariable};
use std::cell::RefCell;
use std::collections::HashMap;

/// Union-Find data structure for type variables with taint mapping
#[derive(Debug, Clone)]
pub struct UnionFind {
    parent: RefCell<HashMap<TypeVariable, TypeVariable>>,
    rank: RefCell<HashMap<TypeVariable, usize>>,
    taint_mapping: RefCell<HashMap<TypeVariable, ConstantTaint>>,
}

impl UnionFind {
    pub fn new() -> Self {
        UnionFind {
            parent: RefCell::new(HashMap::new()),
            rank: RefCell::new(HashMap::new()),
            taint_mapping: RefCell::new(HashMap::new()),
        }
    }

    /// Find the representative (root) of the equivalence class containing x
    pub fn find(&self, x: TypeVariable) -> TypeVariable {
        let mut parent = self.parent.borrow_mut();
        let mut rank = self.rank.borrow_mut();

        if !parent.contains_key(&x) {
            parent.insert(x, x);
            rank.insert(x, 0);
            return x;
        }

        let mut current = x;
        let mut path = Vec::new();

        while parent[&current] != current {
            path.push(current);
            current = parent[&current];
        }

        for node in path {
            parent.insert(node, current);
        }

        current
    }

    /// Union two equivalence classes
    pub fn union(&mut self, x: TypeVariable, y: TypeVariable) {
        let root_x = self.find(x);
        let root_y = self.find(y);

        if root_x == root_y {
            return;
        }

        let mut parent = self.parent.borrow_mut();
        let mut rank = self.rank.borrow_mut();
        let rank_x = rank[&root_x];
        let rank_y = rank[&root_y];

        let new_root;
        if rank_x < rank_y {
            parent.insert(root_x, root_y);
            new_root = root_y;
        } else if rank_x > rank_y {
            parent.insert(root_y, root_x);
            new_root = root_x;
        } else {
            parent.insert(root_y, root_x);
            rank.insert(root_x, rank_x + 1);
            new_root = root_x;
        }

        let taint_x = self.taint_mapping.borrow().get(&root_x).cloned();
        let taint_y = self.taint_mapping.borrow().get(&root_y).cloned();

        let mut mapping = self.taint_mapping.borrow_mut();
        match (taint_x, taint_y) {
            (Some(taint_x), Some(taint_y)) => {
                if taint_x != taint_y {
                    panic!("Taints are not the same: {:?} and {:?}", taint_x, taint_y);
                }
                mapping.insert(new_root, taint_x);
            }
            (Some(taint_x), None) => {
                mapping.insert(new_root, taint_x);
            }
            (None, Some(taint_y)) => {
                mapping.insert(new_root, taint_y);
            }
            (None, None) => {
                // If neither had taint, no mapping is created
            }
        }
    }

    pub fn same_class(&self, x: TypeVariable, y: TypeVariable) -> bool {
        self.find(x) == self.find(y)
    }

    pub fn get_class(&self, x: TypeVariable) -> Vec<TypeVariable> {
        let parent = self.parent.borrow();
        let representative = self.find(x);
        let mut result = Vec::new();
        for (var, _) in parent.iter() {
            if self.find(*var) == representative {
                result.push(*var);
            }
        }
        result
    }

    fn set_taint_for_representative(&mut self, representative: TypeVariable, taint: ConstantTaint) {
        let mut mapping = self.taint_mapping.borrow_mut();
        let old_taint = mapping.get(&representative).cloned();
        if old_taint.is_some() && old_taint.unwrap() != taint {
            panic!("Taints are not the same: {:?} and {:?}", old_taint, taint);
        }
        mapping.insert(representative, taint);
    }

    pub fn set_taint_for_variable(&mut self, variable: TypeVariable, taint: ConstantTaint) {
        let representative = self.find(variable);
        self.set_taint_for_representative(representative, taint);
    }

    fn get_taint_for_representative(&self, representative: TypeVariable) -> Option<Taint> {
        let mapping = self.taint_mapping.borrow();
        mapping.get(&representative).cloned().map(|t| Taint::Constant(t))
    }

    pub fn get_taint_for_variable(&self, variable: TypeVariable) -> Option<Taint> {
        let representative = self.find(variable);
        self.get_taint_for_representative(representative)
    }

    pub fn substitute_variables(&self, taint: &Taint) -> Taint {
        match taint {
            Taint::Constant(constant) => Taint::Constant(*constant),
            Taint::Variable(var) => {
                let representative = self.find(*var);
                if let Some(representative_taint) = self.get_taint_for_representative(representative) {
                    self.substitute_variables(&representative_taint)
                } else {
                    Taint::Variable(representative)
                }
            }
            Taint::Union(left, right) => {
                let left_substituted = self.substitute_variables(left);
                let right_substituted = self.substitute_variables(right);
                Taint::Union(Box::new(left_substituted), Box::new(right_substituted))
            }
        }
    }

    pub fn substitute_taint_type(&self, taint_type: &TaintType) -> TaintType {
        match taint_type {
            TaintType::Primitive(taint) => TaintType::Primitive(self.substitute_variables(taint)),
            TaintType::NestedImmutable(taint, inner) => TaintType::NestedImmutable(
                self.substitute_variables(taint),
                Box::new(self.substitute_taint_type(inner)),
            ),
            TaintType::NestedMutable(taint, inner) => TaintType::NestedMutable(
                self.substitute_variables(taint),
                Box::new(self.substitute_taint_type(inner)),
            ),
            TaintType::Tuple(taint, child_taints) => TaintType::Tuple(
                self.substitute_variables(taint),
                child_taints.iter().map(|child_taint| self.substitute_taint_type(child_taint)).collect()
            ),
        }
    }

    pub fn substitute_judgement(&self, judgement: &Judgement) -> Judgement {
        match judgement {
            Judgement::Le(l, r) => {
                let l_substituted = self.substitute_variables(l);
                let r_substituted = self.substitute_variables(r);
                Judgement::Le(l_substituted, r_substituted)
            }
            Judgement::Eq(l, r) => {
                let l_substituted = self.substitute_variables(l);
                let r_substituted = self.substitute_variables(r);
                Judgement::Eq(l_substituted, r_substituted)
            }
        }
    }
}
