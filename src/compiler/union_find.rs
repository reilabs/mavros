use crate::compiler::witness_info::{ConstantWitness, TypeVariable, WitnessInfo, WitnessJudgement, WitnessType};
use std::cell::RefCell;
use std::collections::HashMap;

/// Union-Find data structure for type variables with witness mapping
#[derive(Debug, Clone)]
pub struct UnionFind {
    parent: RefCell<HashMap<TypeVariable, TypeVariable>>,
    rank: RefCell<HashMap<TypeVariable, usize>>,
    witness_mapping: RefCell<HashMap<TypeVariable, ConstantWitness>>,
}

impl UnionFind {
    pub fn new() -> Self {
        UnionFind {
            parent: RefCell::new(HashMap::new()),
            rank: RefCell::new(HashMap::new()),
            witness_mapping: RefCell::new(HashMap::new()),
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

        let witness_x = self.witness_mapping.borrow().get(&root_x).cloned();
        let witness_y = self.witness_mapping.borrow().get(&root_y).cloned();

        let mut mapping = self.witness_mapping.borrow_mut();
        match (witness_x, witness_y) {
            (Some(wx), Some(wy)) => {
                if wx != wy {
                    panic!(
                        "Witness values are not the same: {:?} and {:?}",
                        wx, wy
                    );
                }
                mapping.insert(new_root, wx);
            }
            (Some(wx), None) => {
                mapping.insert(new_root, wx);
            }
            (None, Some(wy)) => {
                mapping.insert(new_root, wy);
            }
            (None, None) => {}
        }
    }

    pub fn set_witness(&mut self, representative: TypeVariable, witness: ConstantWitness) {
        let mut mapping = self.witness_mapping.borrow_mut();
        let old_witness = mapping.get(&representative).cloned();
        if old_witness.is_some() && old_witness.unwrap() != witness {
            panic!(
                "Witness values are not the same: {:?} and {:?}",
                old_witness, witness
            );
        }
        mapping.insert(representative, witness);
    }

    pub fn get_witness(&self, representative: TypeVariable) -> Option<WitnessInfo> {
        let mapping = self.witness_mapping.borrow();
        mapping
            .get(&representative)
            .cloned()
            .map(WitnessInfo::from_constant)
    }

    pub fn get_witness_for_variable(&self, variable: TypeVariable) -> Option<WitnessInfo> {
        let representative = self.find(variable);
        self.get_witness(representative)
    }

    pub fn substitute_variables(&self, info: &WitnessInfo) -> WitnessInfo {
        match info {
            WitnessInfo::Pure => WitnessInfo::Pure,
            WitnessInfo::Witness => WitnessInfo::Witness,
            WitnessInfo::Variable(var) => {
                let representative = self.find(*var);
                if let Some(representative_info) = self.get_witness(representative) {
                    self.substitute_variables(&representative_info)
                } else {
                    WitnessInfo::Variable(representative)
                }
            }
            WitnessInfo::Join(left, right) => {
                let left_substituted = self.substitute_variables(left);
                let right_substituted = self.substitute_variables(right);
                WitnessInfo::Join(Box::new(left_substituted), Box::new(right_substituted))
            }
        }
    }

    pub fn substitute_witness_type(&self, wt: &WitnessType) -> WitnessType {
        match wt {
            WitnessType::Scalar(info) => WitnessType::Scalar(self.substitute_variables(info)),
            WitnessType::Array(info, inner) => WitnessType::Array(
                self.substitute_variables(info),
                Box::new(self.substitute_witness_type(inner)),
            ),
            WitnessType::Ref(info, inner) => WitnessType::Ref(
                self.substitute_variables(info),
                Box::new(self.substitute_witness_type(inner)),
            ),
            WitnessType::Tuple(info, children) => WitnessType::Tuple(
                self.substitute_variables(info),
                children
                    .iter()
                    .map(|child| self.substitute_witness_type(child))
                    .collect(),
            ),
        }
    }

    pub fn substitute_judgement(&self, judgement: &WitnessJudgement) -> WitnessJudgement {
        match judgement {
            WitnessJudgement::Le(l, r) => {
                let l_substituted = self.substitute_variables(l);
                let r_substituted = self.substitute_variables(r);
                WitnessJudgement::Le(l_substituted, r_substituted)
            }
            WitnessJudgement::Eq(l, r) => {
                let l_substituted = self.substitute_variables(l);
                let r_substituted = self.substitute_variables(r);
                WitnessJudgement::Eq(l_substituted, r_substituted)
            }
        }
    }
}
