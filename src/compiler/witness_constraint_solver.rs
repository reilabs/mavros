use crate::compiler::witness_info::{
    FunctionWitnessType, WitnessInfo, WitnessJudgement, WitnessType,
};
use crate::compiler::witness_union_find::WitnessUnionFind;

/// Constraint solver for witness type inference
#[derive(Clone)]
pub struct WitnessConstraintSolver {
    pub unification: WitnessUnionFind,
    judgements: Vec<WitnessJudgement>,
}

impl WitnessConstraintSolver {
    pub fn new(func_witness_type: &FunctionWitnessType) -> Self {
        WitnessConstraintSolver {
            unification: WitnessUnionFind::new(),
            judgements: func_witness_type.get_judgements().clone(),
        }
    }

    pub fn add_assumption(&mut self, left_type: &WitnessType, right_type: &WitnessType) {
        self.push_deep_eq(left_type, right_type);
    }

    /// Main entry point for constraint solving
    pub fn solve(&mut self) {
        self.simplify_unions_algebraically();
        self.inline_equalities();
        self.blow_up_le_of_meet();
        self.simplify_unions_algebraically();

        let mut current_judgements = self.num_judgements();
        loop {
            self.simplify_le_const();
            self.inline_equalities();
            self.simplify_unions_algebraically();
            if self.num_judgements() == current_judgements {
                break;
            }
            current_judgements = self.num_judgements();
        }

        self.run_defaulting();

        self.simplify_unions_algebraically();
        self.inline_equalities();
        self.simplify_unions_algebraically();
        self.inline_equalities();

        if self.num_judgements() > 0 {
            println!("About to fail:\n{}", self.judgements_string());
            panic!("Failed to solve witness constraints");
        }
    }

    pub fn judgements_string(&self) -> String {
        self.judgements
            .iter()
            .map(|j| match j {
                WitnessJudgement::Eq(l, r) => {
                    let l_substituted = self.unification.substitute_variables(l);
                    let r_substituted = self.unification.substitute_variables(r);
                    format!(
                        "{} = {}",
                        l_substituted.to_string(),
                        r_substituted.to_string()
                    )
                }
                WitnessJudgement::Le(l, r) => {
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
                WitnessJudgement::Eq(WitnessInfo::Variable(l), WitnessInfo::Variable(r)) => {
                    self.unification.union(*l, *r);
                }
                WitnessJudgement::Eq(WitnessInfo::Variable(l), info)
                | WitnessJudgement::Eq(info, WitnessInfo::Variable(l))
                    if info.to_constant().is_some() =>
                {
                    self.unification.set_witness(*l, info.to_constant().unwrap());
                }
                WitnessJudgement::Eq(t1, t2) => {
                    let t1_substituted = self.unification.substitute_variables(t1);
                    let t2_substituted = self.unification.substitute_variables(t2);
                    if t1_substituted != t2_substituted {
                        new_judgements
                            .push(WitnessJudgement::Eq(t1_substituted, t2_substituted));
                    }
                }
                _ => new_judgements.push(judgement.clone()),
            }
        }
        self.judgements = new_judgements;
    }

    fn flatten_unions(&self, info: &WitnessInfo) -> Vec<WitnessInfo> {
        match info {
            WitnessInfo::Join(l, r) => {
                let mut result = Vec::new();
                result.extend(self.flatten_unions(l));
                result.extend(self.flatten_unions(r));
                result
            }
            _ => vec![info.clone()],
        }
    }

    fn simplify_union_algebraically(&self, info: WitnessInfo) -> WitnessInfo {
        match info {
            WitnessInfo::Join(l, r) => {
                let l_simplified = self
                    .simplify_union_algebraically(self.unification.substitute_variables(&l));
                let r_simplified = self
                    .simplify_union_algebraically(self.unification.substitute_variables(&r));
                match (l_simplified, r_simplified) {
                    (WitnessInfo::Pure, r) => r,
                    (l, WitnessInfo::Pure) => l,
                    (WitnessInfo::Witness, _) => WitnessInfo::Witness,
                    (_, WitnessInfo::Witness) => WitnessInfo::Witness,
                    (l, r) => WitnessInfo::Join(Box::new(l), Box::new(r)),
                }
            }
            _ => info,
        }
    }

    fn blow_up_le_of_meet(&mut self) {
        let mut new_judgements = Vec::new();
        for judgement in &self.judgements {
            match judgement {
                WitnessJudgement::Le(l, r) => {
                    let flattened_unions = self.flatten_unions(l);
                    for union in flattened_unions {
                        new_judgements.push(WitnessJudgement::Le(union, r.clone()));
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
                WitnessJudgement::Le(l, r) => {
                    let l_simplified = self.simplify_union_algebraically(l.clone());
                    let r_simplified = self.simplify_union_algebraically(r.clone());
                    new_judgements.push(WitnessJudgement::Le(l_simplified, r_simplified));
                }
                WitnessJudgement::Eq(l, r) => {
                    let l_simplified = self.simplify_union_algebraically(l.clone());
                    let r_simplified = self.simplify_union_algebraically(r.clone());
                    new_judgements.push(WitnessJudgement::Eq(l_simplified, r_simplified));
                }
            }
        }
        self.judgements = new_judgements;
    }

    fn simplify_le_const(&mut self) {
        let mut new_judgements = Vec::new();
        for judgement in &self.judgements {
            match self.unification.substitute_judgement(judgement) {
                WitnessJudgement::Le(WitnessInfo::Pure, _) => {}
                WitnessJudgement::Le(WitnessInfo::Witness, r) => {
                    new_judgements.push(WitnessJudgement::Eq(WitnessInfo::Witness, r.clone()));
                }
                WitnessJudgement::Le(l, WitnessInfo::Pure) => {
                    new_judgements.push(WitnessJudgement::Eq(l.clone(), WitnessInfo::Pure));
                }
                WitnessJudgement::Le(_, WitnessInfo::Witness) => {}
                _ => new_judgements.push(judgement.clone()),
            }
        }
        self.judgements = new_judgements;
    }

    fn push_deep_eq(&mut self, left_type: &WitnessType, right_type: &WitnessType) {
        match (left_type, right_type) {
            (WitnessType::Scalar(l), WitnessType::Scalar(r)) => {
                self.judgements
                    .push(WitnessJudgement::Eq(l.clone(), r.clone()));
            }
            (WitnessType::Array(l, inner_l), WitnessType::Array(r, inner_r)) => {
                self.judgements
                    .push(WitnessJudgement::Eq(l.clone(), r.clone()));
                self.push_deep_eq(inner_l, inner_r);
            }
            (WitnessType::Ref(l, inner_l), WitnessType::Ref(r, inner_r)) => {
                self.judgements
                    .push(WitnessJudgement::Eq(l.clone(), r.clone()));
                self.push_deep_eq(inner_l, inner_r);
            }
            (WitnessType::Tuple(l, children_l), WitnessType::Tuple(r, children_r)) => {
                self.judgements
                    .push(WitnessJudgement::Eq(l.clone(), r.clone()));
                assert_eq!(
                    children_l.len(),
                    children_r.len(),
                    "Tuple arity mismatch in deep equality"
                );
                for (child_l, child_r) in children_l.iter().zip(children_r.iter()) {
                    self.push_deep_eq(child_l, child_r);
                }
            }
            _ => panic!(
                "Cannot unify different witness types {:?} and {:?}",
                left_type, right_type
            ),
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
                    WitnessJudgement::Le(l, r) => {
                        new_judgements
                            .push(WitnessJudgement::Eq(l.clone(), WitnessInfo::Pure));
                        new_judgements
                            .push(WitnessJudgement::Eq(r.clone(), WitnessInfo::Pure));
                    }
                    WitnessJudgement::Eq(l, r) => {
                        new_judgements
                            .push(WitnessJudgement::Eq(l.clone(), WitnessInfo::Pure));
                        new_judgements
                            .push(WitnessJudgement::Eq(r.clone(), WitnessInfo::Pure));
                    }
                }
            }
            self.judgements = new_judgements;
        }
    }
}
