use itertools::Itertools;

use crate::compiler::union_find::UnionFind;
use crate::compiler::ssa::{BlockId, FunctionId, SsaAnnotator, ValueId};
use std::collections::{HashMap, HashSet};

#[derive(PartialEq, Eq, Debug, Clone, Copy, Hash)]
pub struct TypeVariable(pub usize);

impl std::fmt::Display for TypeVariable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "V{}", self.0)
    }
}

#[derive(PartialEq, Eq, Debug, Clone, Copy, Hash)]
pub enum ConstantWitness {
    Pure,
    Witness,
}

impl std::fmt::Display for ConstantWitness {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ConstantWitness::Pure => write!(f, "P"),
            ConstantWitness::Witness => write!(f, "W"),
        }
    }
}

impl ConstantWitness {
    pub fn is_pure(&self) -> bool {
        match self {
            ConstantWitness::Pure => true,
            ConstantWitness::Witness => false,
        }
    }

    pub fn is_witness(&self) -> bool {
        match self {
            ConstantWitness::Pure => false,
            ConstantWitness::Witness => true,
        }
    }
}

#[derive(PartialEq, Eq, Debug, Clone, Hash)]
pub enum WitnessInfo {
    Pure,
    Witness,
    Variable(TypeVariable),
    Join(Box<WitnessInfo>, Box<WitnessInfo>),
}

impl WitnessInfo {
    pub fn to_string(&self) -> String {
        match self {
            WitnessInfo::Pure => "P".to_string(),
            WitnessInfo::Witness => "W".to_string(),
            WitnessInfo::Variable(var) => format!("V{}", var.0),
            WitnessInfo::Join(left, right) => {
                format!("{} ∪ {}", left.to_string(), right.to_string())
            }
        }
    }

    pub fn union(&self, other: &WitnessInfo) -> WitnessInfo {
        WitnessInfo::Join(Box::new(self.clone()), Box::new(other.clone()))
    }

    pub fn gather_vars(&self, result: &mut HashSet<TypeVariable>) {
        match self {
            WitnessInfo::Variable(var) => {
                result.insert(*var);
            }
            WitnessInfo::Join(left, right) => {
                left.gather_vars(result);
                right.gather_vars(result);
            }
            WitnessInfo::Pure | WitnessInfo::Witness => {}
        }
    }

    pub fn substitute(&mut self, varmap: &HashMap<TypeVariable, TypeVariable>) {
        match self {
            WitnessInfo::Variable(var) => {
                if let Some(subst) = varmap.get(var) {
                    *self = WitnessInfo::Variable(*subst);
                }
            }
            WitnessInfo::Join(left, right) => {
                left.substitute(varmap);
                right.substitute(varmap);
            }
            WitnessInfo::Pure | WitnessInfo::Witness => {}
        }
    }

    pub fn has_constants(&self) -> bool {
        match self {
            WitnessInfo::Pure | WitnessInfo::Witness => true,
            WitnessInfo::Variable(_) => false,
            WitnessInfo::Join(left, right) => left.has_constants() || right.has_constants(),
        }
    }

    pub fn simplify_and_default(&self) -> WitnessInfo {
        match self {
            WitnessInfo::Pure => WitnessInfo::Pure,
            WitnessInfo::Witness => WitnessInfo::Witness,
            WitnessInfo::Variable(_) => WitnessInfo::Pure,
            WitnessInfo::Join(left, right) => {
                match (left.simplify_and_default(), right.simplify_and_default()) {
                    (WitnessInfo::Pure, r) => r,
                    (WitnessInfo::Witness, _) => WitnessInfo::Witness,
                    (l, WitnessInfo::Pure) => l,
                    (_, WitnessInfo::Witness) => WitnessInfo::Witness,
                    _ => panic!("This should be impossible here"),
                }
            }
        }
    }

    pub fn expect_constant(&self) -> ConstantWitness {
        match self {
            WitnessInfo::Pure => ConstantWitness::Pure,
            WitnessInfo::Witness => ConstantWitness::Witness,
            _ => panic!("Expected constant witness info, got {:?}", self),
        }
    }

    pub fn to_constant(&self) -> Option<ConstantWitness> {
        match self {
            WitnessInfo::Pure => Some(ConstantWitness::Pure),
            WitnessInfo::Witness => Some(ConstantWitness::Witness),
            _ => None,
        }
    }

    pub fn from_constant(c: ConstantWitness) -> WitnessInfo {
        match c {
            ConstantWitness::Pure => WitnessInfo::Pure,
            ConstantWitness::Witness => WitnessInfo::Witness,
        }
    }
}

impl std::fmt::Display for WitnessInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_string())
    }
}

#[derive(PartialEq, Eq, Debug, Clone, Hash)]
pub enum WitnessType {
    Scalar(WitnessInfo),
    Array(WitnessInfo, Box<WitnessType>),
    Ref(WitnessInfo, Box<WitnessType>),
    Tuple(WitnessInfo, Vec<WitnessType>),
}

impl WitnessType {
    pub fn to_string(&self) -> String {
        match self {
            WitnessType::Scalar(info) => info.to_string(),
            WitnessType::Array(info, inner) => {
                format!("[{} of {}]", info.to_string(), inner.to_string())
            }
            WitnessType::Ref(info, inner) => {
                format!("[*{} of {}]", info.to_string(), inner.to_string())
            }
            WitnessType::Tuple(info, children) => {
                format!(
                    "({} of <{}>)",
                    info.to_string(),
                    children
                        .iter()
                        .map(|child| child.to_string())
                        .join(", ")
                )
            }
        }
    }

    pub fn union(&self, other: &WitnessType) -> WitnessType {
        match (self, other) {
            (WitnessType::Scalar(t1), WitnessType::Scalar(t2)) => {
                WitnessType::Scalar(t1.union(t2))
            }
            (WitnessType::Array(t1, inner1), WitnessType::Array(t2, inner2)) => {
                let inner_union = inner1.union(inner2);
                WitnessType::Array(t1.union(t2), Box::new(inner_union))
            }
            (WitnessType::Ref(t1, inner1), WitnessType::Array(t2, inner2)) => {
                let inner_union = inner1.union(inner2);
                WitnessType::Ref(t1.union(t2), Box::new(inner_union))
            }
            (WitnessType::Tuple(t1, children1), WitnessType::Tuple(t2, children2)) => {
                let children_union = children1
                    .iter()
                    .zip(children2.iter())
                    .map(|(c1, c2)| c1.union(c2))
                    .collect();
                WitnessType::Tuple(t1.union(t2), children_union)
            }
            _ => panic!("Cannot union different witness types"),
        }
    }

    pub fn toplevel_info(&self) -> WitnessInfo {
        match self {
            WitnessType::Scalar(info) => info.clone(),
            WitnessType::Array(info, _) => info.clone(),
            WitnessType::Ref(info, _) => info.clone(),
            WitnessType::Tuple(info, _) => info.clone(),
        }
    }

    pub fn child_witness_type(&self) -> Option<WitnessType> {
        match self {
            WitnessType::Array(_, inner) => Some(*inner.clone()),
            WitnessType::Ref(_, inner) => Some(*inner.clone()),
            WitnessType::Scalar(_) => None,
            WitnessType::Tuple(_, _) => {
                panic!("Error: child_witness_type shouldn't be called for Tuple values")
            }
        }
    }

    pub fn with_toplevel_info(&self, toplevel: WitnessInfo) -> WitnessType {
        match self {
            WitnessType::Scalar(_) => WitnessType::Scalar(toplevel),
            WitnessType::Array(_, inner) => WitnessType::Array(toplevel, inner.clone()),
            WitnessType::Ref(_, inner) => WitnessType::Ref(toplevel, inner.clone()),
            WitnessType::Tuple(_, inner) => WitnessType::Tuple(toplevel, inner.clone()),
        }
    }

    pub fn gather_vars(&self, result: &mut HashSet<TypeVariable>) {
        match self {
            WitnessType::Scalar(info) => {
                info.gather_vars(result);
            }
            WitnessType::Array(t, inner) => {
                t.gather_vars(result);
                inner.gather_vars(result);
            }
            WitnessType::Ref(t, inner) => {
                t.gather_vars(result);
                inner.gather_vars(result);
            }
            WitnessType::Tuple(t, children) => {
                t.gather_vars(result);
                children
                    .iter()
                    .for_each(|inner| inner.gather_vars(result));
            }
        }
    }

    pub fn substitute(&mut self, varmap: &HashMap<TypeVariable, TypeVariable>) {
        match self {
            WitnessType::Scalar(info) => {
                info.substitute(varmap);
            }
            WitnessType::Array(t, inner) => {
                t.substitute(varmap);
                inner.substitute(varmap);
            }
            WitnessType::Ref(t, inner) => {
                t.substitute(varmap);
                inner.substitute(varmap);
            }
            WitnessType::Tuple(t, children) => {
                t.substitute(varmap);
                children
                    .iter_mut()
                    .for_each(|inner| inner.substitute(varmap));
            }
        }
    }

    pub fn simplify_and_default(&self) -> WitnessType {
        match self {
            WitnessType::Scalar(info) => WitnessType::Scalar(info.simplify_and_default()),
            WitnessType::Array(info, inner) => WitnessType::Array(
                info.simplify_and_default(),
                Box::new(inner.simplify_and_default()),
            ),
            WitnessType::Ref(info, inner) => WitnessType::Ref(
                info.simplify_and_default(),
                Box::new(inner.simplify_and_default()),
            ),
            WitnessType::Tuple(info, children) => WitnessType::Tuple(
                info.simplify_and_default(),
                children
                    .iter()
                    .map(|child| child.simplify_and_default())
                    .collect(),
            ),
        }
    }
}

#[derive(Clone)]
pub enum WitnessJudgement {
    Eq(WitnessInfo, WitnessInfo),
    Le(WitnessInfo, WitnessInfo),
}

impl WitnessJudgement {
    pub fn gather_vars(&self, all_vars: &mut HashSet<TypeVariable>) {
        match self {
            WitnessJudgement::Eq(left, right) => {
                left.gather_vars(all_vars);
                right.gather_vars(all_vars);
            }
            WitnessJudgement::Le(left, right) => {
                left.gather_vars(all_vars);
                right.gather_vars(all_vars);
            }
        }
    }

    pub fn substitute(&mut self, varmap: &HashMap<TypeVariable, TypeVariable>) {
        match self {
            WitnessJudgement::Eq(left, right) => {
                left.substitute(varmap);
                right.substitute(varmap);
            }
            WitnessJudgement::Le(left, right) => {
                left.substitute(varmap);
                right.substitute(varmap);
            }
        }
    }

    pub fn has_constants(&self) -> bool {
        match self {
            WitnessJudgement::Eq(left, right) => left.has_constants() || right.has_constants(),
            WitnessJudgement::Le(left, right) => left.has_constants() || right.has_constants(),
        }
    }
}

impl std::fmt::Display for WitnessJudgement {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WitnessJudgement::Eq(a, b) => write!(f, "{} = {}", a.to_string(), b.to_string()),
            WitnessJudgement::Le(a, b) => write!(f, "{} ≤ {}", a.to_string(), b.to_string()),
        }
    }
}

impl std::fmt::Debug for WitnessJudgement {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WitnessJudgement::Eq(a, b) => write!(f, "{} = {}", a.to_string(), b.to_string()),
            WitnessJudgement::Le(a, b) => write!(f, "{} ≤ {}", a.to_string(), b.to_string()),
        }
    }
}

#[derive(Clone)]
pub struct FunctionWitnessType {
    pub returns_witness: Vec<WitnessType>,
    pub cfg_witness: WitnessInfo,
    pub parameters: Vec<WitnessType>,
    pub judgements: Vec<WitnessJudgement>,
    pub block_cfg_witness: HashMap<BlockId, WitnessInfo>,
    pub value_witness_types: HashMap<ValueId, WitnessType>,
}

impl FunctionWitnessType {
    pub fn to_string(&self) -> String {
        format!(
            "returns: {:?}\nparameters: {:?}\njudgements: {:?}\nvalue_witness_types: {:?}\ncfg_witness: {:?}",
            self.returns_witness, self.parameters, self.judgements, self.value_witness_types, self.cfg_witness
        )
    }

    pub fn instantiate_from(&mut self, last_ty_var: &mut usize) {
        let mut all_vars = HashSet::new();
        self.gather_return_vars(&mut all_vars);
        self.gather_param_vars(&mut all_vars);
        self.gather_cfg_var(&mut all_vars);
        self.gather_judgement_vars(&mut all_vars);
        let mut varmap = HashMap::new();
        for var in all_vars {
            let fresh = TypeVariable(*last_ty_var);
            *last_ty_var += 1;
            varmap.insert(var, fresh);
        }
        self.substitute_return_vars(&varmap);
        self.substitute_param_vars(&varmap);
        self.substitute_judgements(&varmap);
        self.substitute_cfg_witness(&varmap);
        self.substitute_block_cfg_witness(&varmap);
    }

    fn gather_return_vars(&self, all_vars: &mut HashSet<TypeVariable>) {
        for wt in &self.returns_witness {
            wt.gather_vars(all_vars);
        }
    }

    fn gather_param_vars(&self, all_vars: &mut HashSet<TypeVariable>) {
        for wt in &self.parameters {
            wt.gather_vars(all_vars);
        }
    }

    fn gather_cfg_var(&self, all_vars: &mut HashSet<TypeVariable>) {
        self.cfg_witness.gather_vars(all_vars);
    }

    fn gather_judgement_vars(&self, all_vars: &mut HashSet<TypeVariable>) {
        for judgement in &self.judgements {
            judgement.gather_vars(all_vars);
        }
    }

    fn substitute_return_vars(&mut self, varmap: &HashMap<TypeVariable, TypeVariable>) {
        for wt in &mut self.returns_witness {
            wt.substitute(varmap);
        }
    }

    fn substitute_param_vars(&mut self, varmap: &HashMap<TypeVariable, TypeVariable>) {
        for wt in &mut self.parameters {
            wt.substitute(varmap);
        }
    }

    fn substitute_judgements(&mut self, varmap: &HashMap<TypeVariable, TypeVariable>) {
        for judgement in &mut self.judgements {
            judgement.substitute(varmap);
        }
    }

    fn substitute_cfg_witness(&mut self, varmap: &HashMap<TypeVariable, TypeVariable>) {
        self.cfg_witness.substitute(varmap);
    }

    fn substitute_block_cfg_witness(&mut self, varmap: &HashMap<TypeVariable, TypeVariable>) {
        for (_, info) in &mut self.block_cfg_witness {
            info.substitute(varmap);
        }
    }

    pub fn get_judgements(&self) -> &Vec<WitnessJudgement> {
        &self.judgements
    }

    pub fn update_from_unification(&self, unification: &UnionFind) -> Self {
        let mut new_wt = self.clone();

        new_wt.judgements = Vec::new();
        new_wt.cfg_witness = unification
            .substitute_variables(&new_wt.cfg_witness)
            .simplify_and_default();
        new_wt.returns_witness = new_wt
            .returns_witness
            .iter()
            .map(|wt| {
                unification
                    .substitute_witness_type(wt)
                    .simplify_and_default()
            })
            .collect();
        new_wt.parameters = new_wt
            .parameters
            .iter()
            .map(|wt| {
                unification
                    .substitute_witness_type(wt)
                    .simplify_and_default()
            })
            .collect();
        new_wt.value_witness_types = new_wt
            .value_witness_types
            .iter()
            .map(|(value_id, wt)| {
                (
                    *value_id,
                    unification
                        .substitute_witness_type(wt)
                        .simplify_and_default(),
                )
            })
            .collect();
        new_wt.block_cfg_witness = new_wt
            .block_cfg_witness
            .iter()
            .map(|(block_id, info)| {
                (
                    *block_id,
                    unification
                        .substitute_variables(&info)
                        .simplify_and_default(),
                )
            })
            .collect();
        new_wt
    }

    pub fn get_value_witness_type(&self, value_id: ValueId) -> &WitnessType {
        self.value_witness_types.get(&value_id).unwrap()
    }

    pub fn get_block_witness(&self, block_id: BlockId) -> &WitnessInfo {
        self.block_cfg_witness.get(&block_id).unwrap()
    }
}

impl SsaAnnotator for FunctionWitnessType {
    fn annotate_value(&self, _: FunctionId, value_id: ValueId) -> String {
        let Some(wt) = self.value_witness_types.get(&value_id) else {
            return "".to_string();
        };
        wt.to_string()
    }

    fn annotate_block(&self, _: FunctionId, block_id: BlockId) -> String {
        let Some(info) = self.block_cfg_witness.get(&block_id) else {
            return "".to_string();
        };
        format!("cfg_witness: {}", info.to_string())
    }

    fn annotate_function(&self, _: FunctionId) -> String {
        let return_types = self
            .returns_witness
            .iter()
            .map(|t| t.to_string())
            .join(", ");
        format!(
            "returns: [{}], cfg_witness: {}",
            return_types,
            self.cfg_witness.to_string()
        )
    }
}
