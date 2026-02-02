use itertools::Itertools;

use crate::compiler::flow_analysis::FlowAnalysis;
use crate::compiler::ir::r#type::{CommutativeMonoid, Empty, Type, TypeExpr};
use crate::compiler::ssa::{BlockId, FunctionId, OpCode, SSA, SsaAnnotator, Terminator, TupleIdx, ValueId};
use crate::compiler::union_find::UnionFind;
use std::collections::{HashMap, HashSet};

#[derive(PartialEq, Eq, Debug, Clone, Copy, Hash)]
pub struct TypeVariable(pub usize);

impl std::fmt::Display for TypeVariable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "V{}", self.0)
    }
}

#[derive(PartialEq, Eq, Debug, Clone, Copy, Hash)]
pub enum ConstantTaint {
    Pure,
    Witness,
}

impl CommutativeMonoid for ConstantTaint {
    fn empty() -> Self {
        ConstantTaint::Pure
    }

    fn op(&self, other: &Self) -> Self {
        match (self, other) {
            (ConstantTaint::Pure, ConstantTaint::Pure) => ConstantTaint::Pure,
            _ => ConstantTaint::Witness,
        }
    }
}

impl std::fmt::Display for ConstantTaint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ConstantTaint::Pure => write!(f, "P"),
            ConstantTaint::Witness => write!(f, "W"),
        }
    }
}

impl ConstantTaint {
    pub fn is_pure(&self) -> bool {
        match self {
            ConstantTaint::Pure => true,
            ConstantTaint::Witness => false,
        }
    }

    pub fn is_witness(&self) -> bool {
        match self {
            ConstantTaint::Pure => false,
            ConstantTaint::Witness => true,
        }
    }
}

// Throughout this, we loosely interpret taints as records
// with Witness any record (forall A. {A}) and `Pure` having
// a single row, i.e. Pure = forall A. { pure := () | A }. This is handy
// because we can now use set-theoretic operations and also
// because we can always cast a Pure taint to a Witness taint,
// but not the other way around, so the natural subtyping relation
// plays nice with taints. So we have Pure < Witness.
#[derive(PartialEq, Eq, Debug, Clone, Hash)]
pub enum Taint {
    Constant(ConstantTaint),
    Variable(TypeVariable),
    Union(Box<Taint>, Box<Taint>),
}

impl Taint {
    pub fn to_string(&self) -> String {
        match self {
            Taint::Constant(constant) => constant.to_string(),
            Taint::Variable(var) => format!("V{}", var.0),
            Taint::Union(left, right) => format!("{} ∪ {}", left.to_string(), right.to_string()),
        }
    }

    pub fn union(&self, other: &Taint) -> Taint {
        Taint::Union(Box::new(self.clone()), Box::new(other.clone()))
    }

    pub fn gather_vars(&self, result: &mut HashSet<TypeVariable>) {
        match self {
            Taint::Variable(var) => {
                result.insert(*var);
            }
            Taint::Union(left, right) => {
                left.gather_vars(result);
                right.gather_vars(result);
            }
            Taint::Constant(_) => {}
        }
    }

    pub fn substitute(&mut self, varmap: &HashMap<TypeVariable, TypeVariable>) {
        match self {
            Taint::Variable(var) => {
                if let Some(subst) = varmap.get(var) {
                    *self = Taint::Variable(*subst);
                }
            }
            Taint::Union(left, right) => {
                left.substitute(varmap);
                right.substitute(varmap);
            }
            Taint::Constant(_) => {}
        }
    }

    pub fn has_constants(&self) -> bool {
        match self {
            Taint::Constant(_) => true,
            Taint::Variable(_) => false,
            Taint::Union(left, right) => left.has_constants() || right.has_constants(),
        }
    }

    pub fn simplify_and_default(&self) -> Taint {
        match self {
            Taint::Constant(constant) => Taint::Constant(*constant),
            Taint::Variable(_) => Taint::Constant(ConstantTaint::Pure), // TODO is this correct?
            Taint::Union(left, right) => {
                match (left.simplify_and_default(), right.simplify_and_default()) {
                    (Taint::Constant(ConstantTaint::Pure), r) => r,
                    (Taint::Constant(ConstantTaint::Witness), _) => {
                        Taint::Constant(ConstantTaint::Witness)
                    }
                    (l, Taint::Constant(ConstantTaint::Pure)) => l,
                    (_, Taint::Constant(ConstantTaint::Witness)) => {
                        Taint::Constant(ConstantTaint::Witness)
                    }
                    _ => panic!("This should be impossible here"),
                }
            }
        }
    }

    pub fn expect_constant(&self) -> ConstantTaint {
        match self {
            Taint::Constant(constant) => *constant,
            _ => panic!("Expected constant taint, got {:?}", self),
        }
    }
}

impl std::fmt::Display for Taint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_string())
    }
}

#[derive(PartialEq, Eq, Debug, Clone, Hash)]
pub enum TaintType {
    Primitive(Taint),
    NestedImmutable(Taint, Box<TaintType>),
    NestedMutable(Taint, Box<TaintType>),
    Tuple(Taint, Vec<TaintType>),
}

impl TaintType {
    pub fn to_string(&self) -> String {
        match self {
            TaintType::Primitive(taint) => taint.to_string(),
            TaintType::NestedImmutable(taint, inner) => {
                format!("[{} of {}]", taint.to_string(), inner.to_string())
            }
            TaintType::NestedMutable(taint, inner) => {
                format!("[*{} of {}]", taint.to_string(), inner.to_string())
            }
            TaintType::Tuple(taint, child_taints) => {
                format!("({} of <{}>)", taint.to_string(), child_taints.iter().map(|child_taint| child_taint.to_string()).join(", "))
            }
        }
    }

    pub fn union(&self, other: &TaintType) -> TaintType {
        match (self, other) {
            (TaintType::Primitive(t1), TaintType::Primitive(t2)) => {
                TaintType::Primitive(t1.union(t2))
            }
            (TaintType::NestedImmutable(t1, inner1), TaintType::NestedImmutable(t2, inner2)) => {
                let inner_union = inner1.union(inner2);
                TaintType::NestedImmutable(t1.union(t2), Box::new(inner_union))
            }
            (TaintType::NestedMutable(t1, inner1), TaintType::NestedImmutable(t2, inner2)) => {
                let inner_union = inner1.union(inner2);
                TaintType::NestedMutable(t1.union(t2), Box::new(inner_union))
            }
            (TaintType::Tuple(t1, children1), TaintType::Tuple(t2, children2)) => {
                let children_union = children1
                    .iter()
                    .zip(children2.iter())
                    .map(|(c1, c2)| c1.union(c2))
                    .collect();
                TaintType::Tuple(t1.union(t2), children_union)
            }
            _ => panic!("Cannot union different taint types"),
        }
    }

    pub fn toplevel_taint(&self) -> Taint {
        match self {
            TaintType::Primitive(taint) => taint.clone(),
            TaintType::NestedImmutable(taint, _) => taint.clone(),
            TaintType::NestedMutable(taint, _) => taint.clone(),
            TaintType::Tuple(taint, _) => {taint.clone()}
        }
    }

    pub fn child_taint_type(&self) -> Option<TaintType> {
        match self {
            TaintType::NestedImmutable(_, inner) => Some(*inner.clone()),
            TaintType::NestedMutable(_, inner) => Some(*inner.clone()),
            TaintType::Primitive(_) => None,
            TaintType::Tuple(_, _) => panic!("Error: child_taint_type shouldn't be called for Tuple values")
        }
    }

    pub fn with_toplevel_taint(&self, toplevel: Taint) -> TaintType {
        match self {
            TaintType::Primitive(_) => TaintType::Primitive(toplevel),
            TaintType::NestedImmutable(_, inner) => {
                TaintType::NestedImmutable(toplevel, inner.clone())
            }
            TaintType::NestedMutable(_, inner) => TaintType::NestedMutable(toplevel, inner.clone()),
            TaintType::Tuple(_, inner) => TaintType::Tuple(toplevel, inner.clone()),
        }
    }

    pub fn gather_vars(&self, result: &mut HashSet<TypeVariable>) {
        match self {
            TaintType::Primitive(taint) => {
                taint.gather_vars(result);
            }
            TaintType::NestedImmutable(t, inner) => {
                t.gather_vars(result);
                inner.gather_vars(result);
            }
            TaintType::NestedMutable(t, inner) => {
                t.gather_vars(result);
                inner.gather_vars(result);
            }
            TaintType::Tuple(t, field_taints) => {
                t.gather_vars(result);
                field_taints.iter().for_each(|inner| inner.gather_vars(result));
            }
        }
    }

    pub fn substitute(&mut self, varmap: &HashMap<TypeVariable, TypeVariable>) {
        match self {
            TaintType::Primitive(taint) => {
                taint.substitute(varmap);
            }
            TaintType::NestedImmutable(t, inner) => {
                t.substitute(varmap);
                inner.substitute(varmap);
            }
            TaintType::NestedMutable(t, inner) => {
                t.substitute(varmap);
                inner.substitute(varmap);
            }
            TaintType::Tuple(t, field_taints) => {
                t.substitute(varmap);
                field_taints.iter_mut().for_each(|inner| inner.substitute(varmap));
            }
        }
    }

    pub fn simplify_and_default(&self) -> TaintType {
        match self {
            TaintType::Primitive(taint) => TaintType::Primitive(taint.simplify_and_default()),
            TaintType::NestedImmutable(taint, inner) => TaintType::NestedImmutable(
                taint.simplify_and_default(),
                Box::new(inner.simplify_and_default()),
            ),
            TaintType::NestedMutable(taint, inner) => TaintType::NestedMutable(
                taint.simplify_and_default(),
                Box::new(inner.simplify_and_default()),
            ),
            TaintType::Tuple(taint, child_taints) => TaintType::Tuple(
                taint.simplify_and_default(),
                child_taints.iter().map(|child_taint| child_taint.simplify_and_default()).collect()
            )
        }
    }
}
#[derive(Clone)]
pub struct TaintAnalysis {
    functions: HashMap<FunctionId, FunctionTaint>,
    last_ty_var: usize,
}

impl TaintAnalysis {
    pub fn to_string(&self) -> String {
        self.functions
            .iter()
            .map(|(id, func)| format!("fn_{}: {}", id.0, func.to_string()))
            .collect::<Vec<_>>()
            .join("\n")
    }

    // fn get_taint_type(&self, func_id: FunctionId, value_id: ValueId) -> Option<&TaintType> {
    //     let Some(func) = self.functions.get(&func_id) else {
    //         return None;
    //     };
    //     func.value_taints.get(&value_id)
    // }

    pub fn get_function_taint(&self, func_id: FunctionId) -> &FunctionTaint {
        self.functions.get(&func_id).unwrap()
    }

    pub fn set_function_taint(&mut self, func_id: FunctionId, taint: FunctionTaint) {
        self.functions.insert(func_id, taint);
    }

    pub fn remove_function_taint(&mut self, func_id: FunctionId) {
        self.functions.remove(&func_id);
    }
}

#[derive(Clone)]
pub struct FunctionTaint {
    pub returns_taint: Vec<TaintType>,
    pub cfg_taint: Taint,
    pub parameters: Vec<TaintType>,
    pub judgements: Vec<Judgement>,
    pub block_cfg_taints: HashMap<BlockId, Taint>,
    pub value_taints: HashMap<ValueId, TaintType>,
}

impl FunctionTaint {
    pub fn to_string(&self) -> String {
        format!(
            "returns: {:?}\nparameters: {:?}\njudgements: {:?}\nvalue_taints: {:?}\ncfg_taint: {:?}",
            self.returns_taint, self.parameters, self.judgements, self.value_taints, self.cfg_taint
        )
    }

    pub fn instantiate_from(&mut self, analysis: &mut TaintAnalysis) {
        let mut all_vars = HashSet::new();
        self.gather_return_vars(&mut all_vars);
        self.gather_param_vars(&mut all_vars);
        self.gather_cfg_var(&mut all_vars);
        self.gather_judgement_vars(&mut all_vars);
        let mut varmap = HashMap::new();
        for var in all_vars {
            varmap.insert(var, analysis.fresh_ty_var());
        }
        self.substitute_return_vars(&varmap);
        self.substitute_param_vars(&varmap);
        self.substitute_judgements(&varmap);
        self.substitute_cfg_taint(&varmap);
        self.substitute_block_cfg_taints(&varmap);
    }

    fn gather_return_vars(&self, all_vars: &mut HashSet<TypeVariable>) {
        for taint in &self.returns_taint {
            taint.gather_vars(all_vars);
        }
    }

    fn gather_param_vars(&self, all_vars: &mut HashSet<TypeVariable>) {
        for taint in &self.parameters {
            taint.gather_vars(all_vars);
        }
    }

    fn gather_cfg_var(&self, all_vars: &mut HashSet<TypeVariable>) {
        self.cfg_taint.gather_vars(all_vars);
    }

    fn gather_judgement_vars(&self, all_vars: &mut HashSet<TypeVariable>) {
        for judgement in &self.judgements {
            judgement.gather_vars(all_vars);
        }
    }

    fn substitute_return_vars(&mut self, varmap: &HashMap<TypeVariable, TypeVariable>) {
        for taint in &mut self.returns_taint {
            taint.substitute(varmap);
        }
    }

    fn substitute_param_vars(&mut self, varmap: &HashMap<TypeVariable, TypeVariable>) {
        for taint in &mut self.parameters {
            taint.substitute(varmap);
        }
    }

    fn substitute_judgements(&mut self, varmap: &HashMap<TypeVariable, TypeVariable>) {
        for judgement in &mut self.judgements {
            judgement.substitute(varmap);
        }
    }

    fn substitute_cfg_taint(&mut self, varmap: &HashMap<TypeVariable, TypeVariable>) {
        self.cfg_taint.substitute(varmap);
    }

    fn substitute_block_cfg_taints(&mut self, varmap: &HashMap<TypeVariable, TypeVariable>) {
        for (_, taint) in &mut self.block_cfg_taints {
            taint.substitute(varmap);
        }
    }

    pub fn get_judgements(&self) -> &Vec<Judgement> {
        &self.judgements
    }

    pub fn update_from_unification(&self, unification: &UnionFind) -> Self {
        let mut new_taint = self.clone();

        new_taint.judgements = Vec::new();
        new_taint.cfg_taint = unification
            .substitute_variables(&new_taint.cfg_taint)
            .simplify_and_default();
        new_taint.returns_taint = new_taint
            .returns_taint
            .iter()
            .map(|taint| {
                unification
                    .substitute_taint_type(taint)
                    .simplify_and_default()
            })
            .collect();
        new_taint.parameters = new_taint
            .parameters
            .iter()
            .map(|taint| {
                unification
                    .substitute_taint_type(taint)
                    .simplify_and_default()
            })
            .collect();
        new_taint.value_taints = new_taint
            .value_taints
            .iter()
            .map(|(value_id, taint)| {
                (
                    *value_id,
                    unification
                        .substitute_taint_type(taint)
                        .simplify_and_default(),
                )
            })
            .collect();
        new_taint.block_cfg_taints = new_taint
            .block_cfg_taints
            .iter()
            .map(|(block_id, taint)| {
                (
                    *block_id,
                    unification
                        .substitute_variables(&taint)
                        .simplify_and_default(),
                )
            })
            .collect();
        new_taint
    }

    pub fn get_value_taint(&self, value_id: ValueId) -> &TaintType {
        self.value_taints.get(&value_id).unwrap()
    }

    pub fn get_block_taint(&self, block_id: BlockId) -> &Taint {
        self.block_cfg_taints.get(&block_id).unwrap()
    }
}

impl SsaAnnotator for FunctionTaint {
    fn annotate_value(&self, _: FunctionId, value_id: ValueId) -> String {
        let Some(taint) = self.value_taints.get(&value_id) else {
            return "".to_string();
        };
        taint.to_string()
    }

    fn annotate_block(&self, _: FunctionId, block_id: BlockId) -> String {
        let Some(taint) = self.block_cfg_taints.get(&block_id) else {
            return "".to_string();
        };
        format!("cfg_taint: {}", taint.to_string())
    }

    fn annotate_function(&self, _: FunctionId) -> String {
        let return_taints = self.returns_taint.iter().map(|t| t.to_string()).join(", ");
        format!(
            "returns: [{}], cfg_taint: {}",
            return_taints,
            self.cfg_taint.to_string()
        )
    }
}

#[derive(Clone)]
pub enum Judgement {
    Eq(Taint, Taint),
    Le(Taint, Taint),
}

impl Judgement {
    pub fn gather_vars(&self, all_vars: &mut HashSet<TypeVariable>) {
        match self {
            Judgement::Eq(left, right) => {
                left.gather_vars(all_vars);
                right.gather_vars(all_vars);
            }
            Judgement::Le(left, right) => {
                left.gather_vars(all_vars);
                right.gather_vars(all_vars);
            }
        }
    }

    pub fn substitute(&mut self, varmap: &HashMap<TypeVariable, TypeVariable>) {
        match self {
            Judgement::Eq(left, right) => {
                left.substitute(varmap);
                right.substitute(varmap);
            }
            Judgement::Le(left, right) => {
                left.substitute(varmap);
                right.substitute(varmap);
            }
        }
    }

    pub fn has_constants(&self) -> bool {
        match self {
            Judgement::Eq(left, right) => left.has_constants() || right.has_constants(),
            Judgement::Le(left, right) => left.has_constants() || right.has_constants(),
        }
    }
}

impl TaintAnalysis {
    pub fn new() -> Self {
        TaintAnalysis {
            functions: HashMap::new(),
            last_ty_var: 0,
        }
    }

    pub fn run(&mut self, ssa: &SSA<Empty>, cfg: &FlowAnalysis) -> Result<(), String> {
        let fns_post_order = cfg.get_call_graph().get_post_order(ssa.get_main_id());
        for fn_id in fns_post_order {
            self.analyze_function(ssa, cfg, fn_id);
        }
        Ok(())
    }

    fn analyze_function(&mut self, ssa: &SSA<Empty>, cfg: &FlowAnalysis, func_id: FunctionId) {
        let func = ssa.get_function(func_id);

        let cfg = cfg.get_function_cfg(func_id);
        let block_queue = cfg.get_blocks_bfs();
        let cfg_ty_var = self.fresh_ty_var();
        let mut function_taint = FunctionTaint {
            returns_taint: vec![],
            cfg_taint: Taint::Variable(cfg_ty_var),
            parameters: vec![],
            judgements: vec![],
            block_cfg_taints: HashMap::new(),
            value_taints: HashMap::new(),
        };

        // initialize block params
        for (_, block) in func.get_blocks() {
            for (value, tp) in block.get_parameters() {
                let taint = self.construct_free_taint_for_type(tp);
                function_taint.value_taints.insert(*value, taint);
            }
        }

        // initialize function parameters
        for (value, _) in func.get_entry().get_parameters() {
            function_taint
                .parameters
                .push(function_taint.value_taints.get(value).unwrap().clone());
        }

        // initialize returns
        for ret in func.get_returns() {
            let taint = self.construct_free_taint_for_type(ret);
            function_taint.returns_taint.push(taint);
        }

        // initialize block cfg taints
        for (block_id, _) in func.get_blocks() {
            let cfg_taint = Taint::Variable(self.fresh_ty_var());
            function_taint
                .block_cfg_taints
                .insert(*block_id, cfg_taint.clone());
            // Every block runs under the global function CFG taint, so its local must be
            // a supertype.
            function_taint
                .judgements
                .push(Judgement::Le(function_taint.cfg_taint.clone(), cfg_taint));
        }

        for (value_id, _) in func.iter_consts() {
            function_taint.value_taints.insert(
                *value_id,
                TaintType::Primitive(Taint::Constant(ConstantTaint::Pure)),
            );
        }

        for block_id in block_queue {
            let block = func.get_block(block_id);

            let cfg_taint = function_taint.block_cfg_taints.get(&block_id).unwrap();

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
                        let lhs_taint = function_taint.value_taints.get(lhs).unwrap();
                        let rhs_taint = function_taint.value_taints.get(rhs).unwrap();
                        let result_taint = lhs_taint.union(rhs_taint);
                        function_taint.value_taints.insert(*r, result_taint);
                    }
                    OpCode::Select {
                        result: r,
                        cond,
                        if_t: then,
                        if_f: otherwise,
                    } => {
                        let cond_taint = function_taint.value_taints.get(cond).unwrap();
                        let then_taint = function_taint.value_taints.get(then).unwrap();
                        let otherwise_taint = function_taint.value_taints.get(otherwise).unwrap();
                        let result_taint = cond_taint.union(then_taint).union(otherwise_taint);
                        function_taint.value_taints.insert(*r, result_taint);
                    }
                    OpCode::Alloc {
                        result: r,
                        elem_type: t,
                        result_annotation: _,
                    } => {
                        let free = self.construct_free_taint_for_type(t);
                        function_taint.value_taints.insert(
                            *r,
                            TaintType::NestedMutable(
                                Taint::Constant(ConstantTaint::Pure),
                                Box::new(free),
                            ),
                        );
                    }
                    OpCode::Store { ptr, value: v } => {
                        let ptr_taint = function_taint.value_taints.get(ptr).unwrap();
                        let value_taint = function_taint.value_taints.get(v).unwrap();
                        match ptr_taint {
                            TaintType::NestedMutable(_, inner) => {
                                // the pointer taint will be affected by the cfg taint
                                function_taint
                                    .judgements
                                    .push(Judgement::Le(cfg_taint.clone(), inner.toplevel_taint()));
                                // pointer needs to be big enough to accomodate the new value
                                self.deep_le(value_taint, inner, &mut function_taint.judgements);
                            }
                            _ => panic!("Unexpected taint for ptr"),
                        }
                    }
                    OpCode::Load { result: r, ptr } => {
                        let ptr_taint = function_taint.value_taints.get(ptr).unwrap();
                        match ptr_taint {
                            TaintType::NestedMutable(ptr_taint, inner) => {
                                function_taint.value_taints.insert(
                                    *r,
                                    inner.with_toplevel_taint(
                                        inner.toplevel_taint().union(ptr_taint),
                                    ),
                                );
                            }
                            _ => panic!("Unexpected taint for ptr"),
                        }
                    }
                    OpCode::ReadGlobal {
                        result: r,
                        offset: _,
                        result_type: tp,
                    } => {
                        let result_taint = self.construct_pure_taint_for_type(tp);
                        function_taint.value_taints.insert(*r, result_taint);
                    }
                    OpCode::AssertEq { lhs: _, rhs: _ } => {}
                    OpCode::AssertR1C { a: _, b: _, c: _ } => {}
                    OpCode::ArrayGet {
                        result: r,
                        array: arr,
                        index: idx,
                    } => {
                        let arr_taint = function_taint.value_taints.get(arr).unwrap();
                        let idx_taint = function_taint.value_taints.get(idx).unwrap();
                        let elem_taint = arr_taint.child_taint_type().unwrap();
                        let result_taint = elem_taint.with_toplevel_taint(
                            arr_taint
                                .toplevel_taint()
                                .union(&idx_taint.toplevel_taint())
                                .union(&elem_taint.toplevel_taint()),
                        );
                        function_taint.value_taints.insert(*r, result_taint);
                    }
                    OpCode::ArraySet {
                        result: r,
                        array: arr,
                        index: idx,
                        value,
                    } => {
                        let arr_taint = function_taint.value_taints.get(arr).unwrap();
                        let idx_taint = function_taint.value_taints.get(idx).unwrap();
                        let value_taint = function_taint.value_taints.get(value).unwrap();
                        let arr_elem_taint = arr_taint.child_taint_type().unwrap();
                        let result_arr_taint = idx_taint.union(&arr_elem_taint).union(value_taint);
                        let result_taint = TaintType::NestedImmutable(
                            arr_taint.toplevel_taint(),
                            Box::new(result_arr_taint),
                        );
                        function_taint.value_taints.insert(*r, result_taint);
                    }
                    OpCode::SlicePush {
                        dir: _,
                        result: r,
                        slice: sl,
                        values,
                    } => {
                        let slice_taint = function_taint.value_taints.get(sl).unwrap();

                        let slice_elem_taint = slice_taint.child_taint_type().unwrap();
                        // Union all value taints together
                        let mut result_elem_taint = slice_elem_taint.clone();
                        for value in values {
                            let value_taint = function_taint.value_taints.get(value).unwrap();
                            result_elem_taint = result_elem_taint.union(value_taint);
                        }
                        let result_taint = TaintType::NestedImmutable(
                            slice_taint.toplevel_taint().clone(),
                            Box::new(result_elem_taint),
                        );
                        function_taint.value_taints.insert(*r, result_taint);
                    }
                    OpCode::SliceLen {
                        result: r,
                        slice: sl,
                    } => {
                        let _slice_taint = function_taint.value_taints.get(sl).unwrap();
                        // Slice must always be Pure taint
                        // assert_eq!(
                        //     slice_taint.toplevel_taint(),
                        //     Taint::Constant(ConstantTaint::Pure)
                        // );
                        // Result is always Pure u32
                        let result_taint = TaintType::Primitive(Taint::Constant(ConstantTaint::Pure));
                        function_taint.value_taints.insert(*r, result_taint);
                    }
                    OpCode::Call {
                        results: outputs,
                        function: func,
                        args: inputs,
                    } => {
                        let return_types = ssa.get_function(*func).get_returns();
                        for (output, typ) in outputs.iter().zip(return_types.iter()) {
                            function_taint
                                .value_taints
                                .insert(*output, self.construct_free_taint_for_type(typ));
                        }
                        let outputs_taint = outputs
                            .iter()
                            .map(|v| function_taint.value_taints.get(v).unwrap())
                            .collect::<Vec<_>>();
                        let inputs_taint = inputs
                            .iter()
                            .map(|v| function_taint.value_taints.get(v).unwrap())
                            .collect::<Vec<_>>();
                        let mut func_taint = self.functions.get(&func).unwrap().clone();
                        func_taint.instantiate_from(self);
                        for (output, ret) in
                            outputs_taint.iter().zip(func_taint.returns_taint.iter())
                        {
                            self.deep_le(ret, output, &mut function_taint.judgements);
                            self.deep_le(output, ret, &mut function_taint.judgements);
                        }
                        for (input, param) in inputs_taint.iter().zip(func_taint.parameters.iter())
                        {
                            self.deep_le(input, param, &mut function_taint.judgements);
                            self.deep_le(param, input, &mut function_taint.judgements);

                        }
                        function_taint.judgements.extend(func_taint.judgements);
                        function_taint.judgements.push(Judgement::Le(
                            cfg_taint.clone(),
                            func_taint.cfg_taint.clone(),
                        ));
                    }
                    OpCode::MkSeq {
                        result,
                        elems: inputs,
                        seq_type: _,
                        elem_type: tp,
                    } => {
                        let inputs_taint = inputs
                            .iter()
                            .map(|v| function_taint.value_taints.get(v).unwrap())
                            .collect::<Vec<_>>();
                        let result_taint = inputs_taint
                            .iter()
                            .fold(self.construct_pure_taint_for_type(tp), |acc, t| {
                                acc.union(t)
                            });
                        function_taint.value_taints.insert(
                            *result,
                            TaintType::NestedImmutable(
                                Taint::Constant(ConstantTaint::Pure),
                                Box::new(result_taint),
                            ),
                        );
                    }
                    OpCode::Cast {
                        result,
                        value,
                        target: _,
                    } => {
                        let value_taint = function_taint.value_taints.get(value).unwrap().clone();
                        function_taint.value_taints.insert(*result, value_taint);
                    }
                    OpCode::Truncate {
                        result,
                        value,
                        to_bits: _,
                        from_bits: _,
                    } => {
                        let value_taint = function_taint.value_taints.get(value).unwrap().clone();
                        function_taint.value_taints.insert(*result, value_taint);
                    }
                    OpCode::Not { result, value } => {
                        let value_taint = function_taint.value_taints.get(value).unwrap().clone();
                        function_taint.value_taints.insert(*result, value_taint);
                    }
                    OpCode::ToBits {
                        result,
                        value,
                        endianness: _,
                        count: _,
                    } => {
                        let value_taint = function_taint.value_taints.get(value).unwrap().clone();
                        let result_taint = TaintType::NestedImmutable(
                            Taint::Constant(ConstantTaint::Pure),
                            Box::new(value_taint),
                        );
                        function_taint.value_taints.insert(*result, result_taint);
                    }
                    OpCode::ToRadix {
                        result,
                        value,
                        radix: _radix,
                        endianness: _,
                        count: _,
                    } => {
                        let value_taint = function_taint.value_taints.get(value).unwrap().clone();
                        let result_taint = TaintType::NestedImmutable(
                            Taint::Constant(ConstantTaint::Pure),
                            Box::new(value_taint),
                        );
                        function_taint.value_taints.insert(*result, result_taint);
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
                    | OpCode::PureToWitnessRef {
                        result: _,
                        value: _,
                        result_annotation: _,
                    }
                    | OpCode::UnboxField {
                        result: _,
                        value: _,
                    }
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
                    | OpCode::Todo { .. } => {
                        panic!("Should not be present at this stage {:?}", instruction);
                    }
                    OpCode::TupleProj {
                        result,
                        tuple,
                        idx,
                    } => {
                        if let TupleIdx::Static(child_index) = idx {
                            let tuple_taint = function_taint.value_taints.get(tuple).unwrap();
                            if let TaintType::Tuple(_, child_taints) = tuple_taint {
                                let elem_taint = &child_taints[*child_index];
                                let result_taint = elem_taint.with_toplevel_taint(
                                tuple_taint.toplevel_taint()
                                        .union(&elem_taint.toplevel_taint()),
                                );
                                function_taint.value_taints.insert(*result, result_taint);
                            } else {
                                panic!("Taint should be of tuple type")
                            }
                        } else {
                            panic!("Tuple index should be static at this stage")
                        }
                    },
                    OpCode::MkTuple {
                        result,
                        elems: inputs,
                        element_types: _,
                    } => {
                        let inputs_taint = inputs
                            .iter()
                            .map(|v| function_taint.value_taints.get(v).unwrap().clone())
                            .collect::<Vec<_>>();
                        function_taint.value_taints.insert(
                            *result,
                            TaintType::Tuple(
                                Taint::Constant(ConstantTaint::Pure),
                                inputs_taint,
                            ),
                        );
                    }
                }
            }

            if let Some(terminator) = block.get_terminator() {
                match terminator {
                    Terminator::Return(values) => {
                        let returns_taint = function_taint.returns_taint.clone();
                        let actual_returns_taint = values
                            .iter()
                            .map(|v| function_taint.value_taints.get(v).unwrap())
                            .collect::<Vec<_>>();
                        for (declared, actual) in
                            returns_taint.iter().zip(actual_returns_taint.iter())
                        {
                            // the actual return type must be usable in place of the declared type
                            self.deep_le(actual, declared, &mut function_taint.judgements);
                        }
                    }
                    Terminator::Jmp(target, params) => {
                        let target_params = func
                            .get_block(*target)
                            .get_parameters()
                            .map(|(v, _)| *v)
                            .collect::<Vec<_>>();
                        let target_param_taints = target_params
                            .iter()
                            .map(|v| function_taint.value_taints.get(v).unwrap())
                            .collect::<Vec<_>>();
                        let param_taints = params
                            .iter()
                            .map(|v| function_taint.value_taints.get(v).unwrap())
                            .collect::<Vec<_>>();
                        for (target_param, param) in
                            target_param_taints.iter().zip(param_taints.iter())
                        {
                            // every passed param must be a subtype of declared params
                            self.deep_le(param, target_param, &mut function_taint.judgements);
                        }
                    }
                    Terminator::JmpIf(cond, _, _) => {
                        let cond_taint = function_taint.value_taints.get(cond).unwrap();
                        if cfg.is_loop_entry(block_id) {
                            // we assert that the condition is pure
                            function_taint.judgements.push(Judgement::Le(
                                cond_taint.toplevel_taint(),
                                Taint::Constant(ConstantTaint::Pure),
                            ));
                        } else {
                            // we need to taint the inputs of the merge point with the condition taint
                            let merge = cfg.get_merge_point(block_id);
                            let merge_inputs = func
                                .get_block(merge)
                                .get_parameters()
                                .map(|(v, _)| *v)
                                .collect::<Vec<_>>();
                            for input in merge_inputs {
                                let input_taint = function_taint.value_taints.get(&input).unwrap();
                                function_taint.judgements.push(Judgement::Le(
                                    cond_taint.toplevel_taint(),
                                    input_taint.toplevel_taint(),
                                ));
                            }

                            let body_blocks = cfg.get_if_body(block_id);
                            for block in body_blocks {
                                let local_taint =
                                    function_taint.block_cfg_taints.get(&block).unwrap();
                                function_taint.judgements.push(Judgement::Le(
                                    cond_taint.toplevel_taint(),
                                    local_taint.clone(),
                                ));

                                function_taint
                                    .judgements
                                    .push(Judgement::Le(cfg_taint.clone(), local_taint.clone()));
                            }
                        }
                    }
                }
            }
        }

        self.functions.insert(func_id, function_taint);
    }

    fn deep_le(&self, lhs: &TaintType, rhs: &TaintType, judgements: &mut Vec<Judgement>) {
        match (lhs, rhs) {
            (TaintType::Primitive(lhs), TaintType::Primitive(rhs)) => {
                judgements.push(Judgement::Le(lhs.clone(), rhs.clone()));
            }
            (
                TaintType::NestedImmutable(lhs, inner_lhs),
                TaintType::NestedImmutable(rhs, inner_rhs),
            ) => {
                judgements.push(Judgement::Le(lhs.clone(), rhs.clone()));
                self.deep_le(inner_lhs, inner_rhs, judgements);
            }
            (
                TaintType::NestedMutable(lhs, inner_lhs),
                TaintType::NestedMutable(rhs, inner_rhs),
            ) => {
                judgements.push(Judgement::Le(lhs.clone(), rhs.clone()));
                self.deep_le(inner_lhs, inner_rhs, judgements);
            }
            (
                TaintType::Tuple(lhs, inner_lhs),
                TaintType::Tuple(rhs, inner_rhs),
            ) => {
                judgements.push(Judgement::Le(lhs.clone(), rhs.clone()));
                for (l, r) in inner_lhs.iter().zip(inner_rhs.iter()) {
                    self.deep_le(l, r, judgements);
                }
            }
            _ => {
                panic!("Cannot compare different taint types: {:?} vs {:?}", lhs, rhs)
            }
        }
    }

    fn fresh_ty_var(&mut self) -> TypeVariable {
        let var = TypeVariable(self.last_ty_var);
        self.last_ty_var += 1;
        var
    }

    fn construct_free_taint_for_type(&mut self, typ: &Type<Empty>) -> TaintType {
        match &typ.expr {
            TypeExpr::U(_) | TypeExpr::Field => {
                TaintType::Primitive(Taint::Variable(self.fresh_ty_var()))
            }
            TypeExpr::Array(i, _) => TaintType::NestedImmutable(
                Taint::Variable(self.fresh_ty_var()),
                Box::new(self.construct_free_taint_for_type(i)),
            ),
            TypeExpr::Slice(i) => TaintType::NestedImmutable(
                Taint::Variable(self.fresh_ty_var()),
                Box::new(self.construct_free_taint_for_type(i)),
            ),
            TypeExpr::Ref(i) => TaintType::NestedMutable(
                Taint::Variable(self.fresh_ty_var()),
                Box::new(self.construct_free_taint_for_type(i)),
            ),
            TypeExpr::WitnessRef => {
                panic!("ICE: WitnessVal should not be present at this stage");
            }
            TypeExpr::Tuple(elements) => TaintType::Tuple(
                Taint::Variable(self.fresh_ty_var()),
                elements
                    .iter()
                    .map(|e| self.construct_free_taint_for_type(e))
                    .collect(),
            )
        }
    }

    fn construct_pure_taint_for_type(&mut self, typ: &Type<Empty>) -> TaintType {
        match &typ.expr {
            TypeExpr::U(_) | TypeExpr::Field => {
                TaintType::Primitive(Taint::Constant(ConstantTaint::Pure))
            }
            TypeExpr::Array(i, _) => TaintType::NestedImmutable(
                Taint::Constant(ConstantTaint::Pure),
                Box::new(self.construct_pure_taint_for_type(i)),
            ),
            TypeExpr::Slice(i) => TaintType::NestedImmutable(
                Taint::Constant(ConstantTaint::Pure),
                Box::new(self.construct_pure_taint_for_type(i)),
            ),
            TypeExpr::Ref(i) => TaintType::NestedMutable(
                Taint::Constant(ConstantTaint::Pure),
                Box::new(self.construct_pure_taint_for_type(i)),
            ),
            TypeExpr::WitnessRef => {
                panic!("ICE: WitnessVal should not be present at this stage");
            }
            TypeExpr::Tuple(elements) => TaintType::Tuple(
                Taint::Constant(ConstantTaint::Pure),
                elements
                    .iter()
                    .map(|e| self.construct_pure_taint_for_type(e))
                    .collect(),
            )
        }
    }
}

impl SsaAnnotator for TaintAnalysis {
    fn annotate_value(&self, function_id: FunctionId, value_id: ValueId) -> String {
        let Some(function_taint) = self.functions.get(&function_id) else {
            return "".to_string();
        };
        function_taint.annotate_value(function_id, value_id)
    }

    fn annotate_block(&self, function_id: FunctionId, block_id: BlockId) -> String {
        let Some(function_taint) = self.functions.get(&function_id) else {
            return "".to_string();
        };
        function_taint.annotate_block(function_id, block_id)
    }

    fn annotate_function(&self, function_id: FunctionId) -> String {
        let Some(function_taint) = self.functions.get(&function_id) else {
            return "".to_string();
        };
        function_taint.annotate_function(function_id)
    }
}

// struct UnificationContext {
//     last_variable: usize,
//     judgements: Vec<Judgement>,
// }

// impl UnificationContext {
//     pub fn new() -> Self {
//         UnificationContext {
//             last_variable: 0,
//             judgements: Vec::new(),
//         }
//     }

//     pub fn fresh_variable(&mut self) -> Taint {
//         let var = TypeVariable(self.last_variable);
//         self.last_variable += 1;
//         Taint::Variable(var)
//     }
// }

// // TODO: this will break on recursion
// impl TaintAnalysis {
//     pub fn get_function_analysis_mut(&mut self, fun_id: FunctionId) -> &mut FunctionTaintAnalysis {
//         self.functions.entry(fun_id).or_insert_with(|| {
//             let returns_taint = vec![];
//             FunctionTaintAnalysis::new(returns_taint)
//         })
//     }

//     pub fn get_function_analysis(&self, fun_id: FunctionId) -> &FunctionTaintAnalysis {
//         self.functions
//             .get(&fun_id)
//             .expect("Function analysis not found")
//     }

//     pub fn new() -> Self {
//         TaintAnalysis {
//             functions: HashMap::new(),
//             unification_context: UnificationContext::new(),
//         }
//     }
//     pub fn run(&mut self, ssa: &SSA) -> Result<(), String> {
//         for (fun_id, fun) in ssa.iter_functions() {
//             self.initialize_function_returns(*fun_id, fun);
//             self.initialize_block_inputs(*fun_id, fun);
//         }

//         for (fun_id, fun) in ssa.iter_functions() {
//             self.analyze_instructions(ssa, *fun_id, fun);
//         }

//         Ok(())
//     }

//     fn initialize_function_returns(&mut self, fun_id: FunctionId, fun: &Function) {
//         let returns_taint = fun
//             .get_returns()
//             .iter()
//             .map(|typ| self.construct_free_taint_for_type(typ))
//             .collect();
//         let analysis = self.get_function_analysis_mut(fun_id);
//         analysis.returns_taint = returns_taint;
//     }

//     fn initialize_block_inputs(&mut self, fun_id: FunctionId, fun: &Function) {
//         for (block_id, block) in fun.get_blocks() {
//             let param_values: Vec<_> = block.get_parameters().map(|(v, _)| *v).collect();
//             for (value, tp) in block.get_parameters() {
//                 let taint = self.construct_free_taint_for_type(tp);
//                 let analysis = self.get_function_analysis_mut(fun_id);
//                 let block_analysis = analysis.get_block_analysis_mut(*block_id);
//                 block_analysis.set_value_taint(*value, taint);
//             }
//             // Set parameter values after setting taints
//             let analysis = self.get_function_analysis_mut(fun_id);
//             let block_analysis = analysis.get_block_analysis_mut(*block_id);
//             block_analysis.set_parameter_values(param_values);
//         }
//     }

//     fn analyze_instructions(&mut self, ssa: &SSA, fun_id: FunctionId, fun: &Function) {
//         let analysis = self.get_function_analysis_mut(fun_id);
//         analysis.entry_block = fun.get_entry_id();
//         for (block_id, block) in fun.get_blocks() {
//             self.analyze_block_instructions(ssa, fun_id, *block_id, block);
//         }
//     }

//     fn analyze_block_instructions(
//         &mut self,
//         ssa: &SSA,
//         fun_id: FunctionId,
//         block_id: BlockId,
//         block: &Block,
//     ) {
//         for instruction in block.get_instructions() {
//             self.analyze_instruction(ssa, fun_id, block_id, instruction);
//         }
//         if let Some(terminator) = block.get_terminator() {
//             self.analyze_terminator(ssa, fun_id, block_id, terminator);
//         }
//     }

//     fn analyze_instruction(
//         &mut self,
//         ssa: &SSA,
//         fun_id: FunctionId,
//         block_id: BlockId,
//         instruction: &OpCode,
//     ) {
//         match instruction {
//             OpCode::FieldConst(r, _) | OpCode::BConst(r, _) | OpCode::UConst(r, _) => {
//                 self.set_taint_type(fun_id, block_id, *r, Primitive(Pure));
//             }

//             OpCode::Eq(r, lhs, rhs) | OpCode::Add(r, lhs, rhs) | OpCode::Lt(r, lhs, rhs) => {
//                 let lhs = self
//                     .get_taint_type(fun_id, block_id, *lhs)
//                     .expect("LHS not found");
//                 let rhs = self
//                     .get_taint_type(fun_id, block_id, *rhs)
//                     .expect("RHS not found");
//                 self.set_taint_type(fun_id, block_id, *r, lhs.meet(&rhs));
//             }

//             OpCode::Alloc(r, t) => {
//                 let free = self.construct_free_taint_for_type(t);
//                 let fv = self.unification_context.fresh_variable();
//                 self.set_taint_type(fun_id, block_id, *r, NestedMutable(fv, Box::new(free)));
//             }
//             OpCode::Store(ptr, v) => {
//                 let ptr_taint = self
//                     .get_taint_type(fun_id, block_id, *ptr)
//                     .expect("Pointer taint not found");
//                 let value_taint = self
//                     .get_taint_type(fun_id, block_id, *v)
//                     .expect("Value taint not found");
//                 match ptr_taint {
//                     NestedMutable(_, inner) => {
//                         self.emit_le(inner.toplevel_taint(), value_taint.toplevel_taint()); // TODO should it be deep?
//                     }
//                     _ => {
//                         panic!("Unexpected taint for ptr");
//                     }
//                 }
//             }
//             OpCode::Load(r, ptr) => {
//                 let ptr_taint = self
//                     .get_taint_type(fun_id, block_id, *ptr)
//                     .expect("Pointer taint not found");
//                 match ptr_taint {
//                     NestedMutable(ptr_taint, inner) => {
//                         self.set_taint_type(fun_id, block_id, *r, *inner.clone()); // TODO this should meet with ptr taint
//                     }
//                     _ => {
//                         panic!("Unexpected taint for ptr");
//                     }
//                 }
//             }
//             OpCode::AssertEq(_, _) => {}
//             OpCode::Call(outputs, func, inputs) => {
//                 // Get SSA types for parameters and arguments
//                 let callee_func = ssa.get_function(*func);
//                 let param_types: Vec<_> = callee_func
//                     .get_entry()
//                     .get_parameters()
//                     .map(|(_, t)| t)
//                     .collect();
//                 let caller_function = ssa.get_function(fun_id);
//                 let arg_types: Vec<_> = inputs
//                     .iter()
//                     .map(|v| {
//                         caller_function.get_value_type(*v).unwrap_or(&Type::Field) // fallback to Field for debug
//                     })
//                     .collect();

//                 let input_taints = inputs
//                     .iter()
//                     .map(|v| {
//                         self.get_taint_type(fun_id, block_id, *v)
//                             .expect("Input taint not found")
//                     })
//                     .collect::<Vec<_>>();
//                 let param_taints = self.get_function_analysis(*func).get_input_taints();

//                 // Debug prints
//                 println!("CALL: param_types = {:?}", param_types);
//                 println!("CALL: arg_types = {:?}", arg_types);
//                 println!("CALL: param_taints = {:?}", param_taints);
//                 println!("CALL: input_taints = {:?}", input_taints);

//                 let instantiations = self.prepare_instantiations(&param_taints, &input_taints);

//                 let declared_outputs = self.get_function_analysis(*func).returns_taint.clone();

//                 for (output, declared_output) in outputs.iter().zip(declared_outputs.iter()) {
//                     let output_taint = self.instantiate_type(declared_output, &instantiations);
//                     self.set_taint_type(fun_id, block_id, *output, output_taint);
//                 }
//             }
//             OpCode::ArrayGet(r, arr, ix) => {
//                 let arr_taint = self
//                     .get_taint_type(fun_id, block_id, *arr)
//                     .expect("Array taint not found");
//                 let ix_taint = self
//                     .get_taint_type(fun_id, block_id, *ix)
//                     .expect("Index taint not found");
//                 let item_taint = arr_taint
//                     .child_taint_type()
//                     .expect("Array taint should have child taint");
//                 let result_taint_type = item_taint.with_toplevel_taint(
//                     arr_taint
//                         .toplevel_taint()
//                         .meet(&ix_taint.toplevel_taint())
//                         .meet(&item_taint.toplevel_taint()),
//                 );
//                 self.set_taint_type(fun_id, block_id, *r, result_taint_type);
//             }
//         }
//     }

//     fn analyze_terminator(
//         &mut self,
//         ssa: &SSA,
//         fun_id: FunctionId,
//         block_id: BlockId,
//         terminator: &Terminator,
//     ) {
//         match terminator {
//             Terminator::Return(values) => {
//                 let returns_taint = self.get_function_analysis(fun_id).returns_taint.clone();
//                 let actual_returns_taint = values
//                     .iter()
//                     .map(|v| {
//                         self.get_taint_type(fun_id, block_id, *v)
//                             .expect("Return taint not found")
//                     })
//                     .collect::<Vec<_>>();
//                 for (declared, actual) in returns_taint.iter().zip(actual_returns_taint.iter()) {
//                     self.deep_eq(declared.clone(), actual.clone());
//                 }
//             }
//             Terminator::Jmp(target, params) => {
//                 self.analyze_jump(ssa, fun_id, block_id, *target, params);
//             }
//             Terminator::JmpIf(cond, t1, t2) => {
//                 self.analyze_jump(ssa, fun_id, block_id, *t1, &[]);
//                 self.analyze_jump(ssa, fun_id, block_id, *t2, &[]);
//             }
//         }
//     }

//     fn analyze_jump(
//         &mut self,
//         ssa: &SSA,
//         fun_id: FunctionId,
//         block_id: BlockId,
//         target: BlockId,
//         values: &[ValueId],
//     ) {
//         let target_params = ssa.get_function(fun_id).get_block(target).get_parameters();
//         let param_taints = target_params
//             .map(|(v, _)| {
//                 self.get_taint_type(fun_id, target, *v)
//                     .expect("Target taint not found")
//             })
//             .collect::<Vec<_>>();
//         let value_taints = values
//             .iter()
//             .map(|v| {
//                 self.get_taint_type(fun_id, block_id, *v)
//                     .expect("Param taint not found")
//             })
//             .collect::<Vec<_>>();

//         let instantiations = self.prepare_instantiations(&param_taints, &value_taints);

//         for (pt, vt) in instantiations {
//             self.emit_le(Taint::Variable(pt), vt);
//         }
//         // for (val, param) in value_taints.iter().zip(param_taints.iter()) {
//         //     match (val, param) {
//         //         // Pointers may get tainted in the caller. Therefore, we assert that the current
//         //         // type of the caller is no greater than the type inferred in the callee.
//         //         (
//         //             TaintType::NestedMutable(val_taint, _),
//         //             TaintType::NestedMutable(Taint::Variable(tv), _),
//         //         ) => {
//         //             self.emit_le(
//         //                 val_taint.clone(),
//         //                 Taint::Instantiate(tv.clone(), instantiations.clone()),
//         //             );
//         //         }
//         //         _ => (),
//         //     }
//         // }
//     }

//     fn instantiate_type(
//         &mut self,
//         typ: &TaintType,
//         instantiations: &[(TypeVariable, Taint)],
//     ) -> TaintType {
//         match typ {
//             TaintType::Primitive(t) => {
//                 TaintType::Primitive(Taint::Instantiate(Box::new(t.clone()), instantiations.to_vec()))
//             }
//             TaintType::NestedImmutable(t, inner) => {
//                 let instantiated_inner = self.instantiate_type(inner, instantiations);
//                 TaintType::NestedImmutable(
//                     Taint::Instantiate(Box::new(t.clone()), instantiations.to_vec()),
//                     Box::new(instantiated_inner),
//                 )
//             }
//             TaintType::NestedMutable(t, inner) => {
//                 let instantiated_inner = self.instantiate_type(inner, instantiations);
//                 TaintType::NestedMutable(
//                     Taint::Instantiate(Box::new(t.clone()), instantiations.to_vec()),
//                     Box::new(instantiated_inner),
//                 )
//             }
//             _ => panic!("Instantiated type must be fully free"),
//         }
//     }

//     fn prepare_instantiations(
//         &self,
//         params: &[TaintType],
//         values: &[TaintType],
//     ) -> Vec<(TypeVariable, Taint)> {
//         params
//             .iter()
//             .zip(values.iter())
//             .flat_map(|(p, v)| self.prepare_instantiations_for_type(p, v))
//             .collect()
//     }

//     fn prepare_instantiations_for_type(
//         &self,
//         param: &TaintType,
//         value: &TaintType,
//     ) -> Vec<(TypeVariable, Taint)> {
//         match (param, value) {
//             (TaintType::Primitive(Taint::Variable(tv)), TaintType::Primitive(v_taint)) => {
//                 vec![(*tv, v_taint.clone())]
//             }
//             (
//                 TaintType::NestedImmutable(Taint::Variable(tv), inner_param),
//                 TaintType::NestedImmutable(v_taint, inner_value),
//             ) => {
//                 let mut instantiations = vec![(*tv, v_taint.clone())];
//                 instantiations
//                     .extend(self.prepare_instantiations_for_type(inner_param, inner_value));
//                 instantiations
//             }
//             (
//                 TaintType::NestedMutable(Taint::Variable(tv), inner_param),
//                 TaintType::NestedMutable(v_taint, inner_value),
//             ) => {
//                 let mut instantiations = vec![(*tv, v_taint.clone())];
//                 instantiations
//                     .extend(self.prepare_instantiations_for_type(inner_param, inner_value));
//                 instantiations
//             }
//             _ => panic!(
//                 "Malformed params for instantiation {} and {}",
//                 param.to_string(),
//                 value.to_string()
//             ),
//         }
//     }

//     fn construct_free_taint_for_type(&mut self, typ: &Type) -> TaintType {
//         match typ {
//             Type::U32 | Type::Field | Type::Bool => {
//                 TaintType::Primitive(self.unification_context.fresh_variable())
//             }
//             Type::Array(i, _) => TaintType::NestedImmutable(
//                 self.unification_context.fresh_variable(),
//                 Box::new(self.construct_free_taint_for_type(i)),
//             ),
//             Type::Ref(i) => TaintType::NestedMutable(
//                 self.unification_context.fresh_variable(),
//                 Box::new(self.construct_free_taint_for_type(i)),
//             ),
//         }
//     }

//     fn construct_pure_taint_for_type(&self, typ: &Type) -> TaintType {
//         match typ {
//             Type::U32 | Type::Field | Type::Bool => TaintType::Primitive(Pure),
//             Type::Array(i, _) => {
//                 TaintType::NestedImmutable(Pure, Box::new(self.construct_pure_taint_for_type(i)))
//             }
//             Type::Ref(i) => {
//                 TaintType::NestedMutable(Pure, Box::new(self.construct_pure_taint_for_type(i)))
//             }
//         }
//     }

//     fn construct_taint_for_main_param(typ: &Type) -> TaintType {
//         match typ {
//             Type::U32 | Type::Field | Type::Bool => TaintType::Primitive(Taint::Witness),
//             Type::Array(i, _) => TaintType::NestedImmutable(
//                 Taint::Pure,
//                 Box::new(Self::construct_taint_for_main_param(i)),
//             ),
//             Type::Ref(i) => TaintType::NestedMutable(
//                 Taint::Pure,
//                 Box::new(Self::construct_taint_for_main_param(i)),
//             ),
//         }
//     }

//     fn construct_main_signature(ssa: &SSA) -> Vec<TaintType> {
//         ssa.get_main()
//             .get_param_types()
//             .iter()
//             .map(Self::construct_taint_for_main_param)
//             .collect()
//     }

//     pub fn get_taint_type(
//         &self,
//         func_id: FunctionId,
//         block_id: BlockId,
//         value_id: ValueId,
//     ) -> Option<TaintType> {
//         let Some(analysis) = self.functions.get(&func_id) else {
//             return None;
//         };
//         let Some(block_analysis) = analysis.blocks.get(&block_id) else {
//             return None;
//         };
//         block_analysis.get_value_taint(value_id).cloned()
//     }

//     fn set_taint_type(
//         &mut self,
//         func_id: FunctionId,
//         block_id: BlockId,
//         value_id: ValueId,
//         taint: TaintType,
//     ) {
//         let analysis = self.get_function_analysis_mut(func_id);
//         let block_analysis = analysis.get_block_analysis_mut(block_id);
//         block_analysis.set_value_taint(value_id, taint);
//     }

//     pub fn annotate_value(
//         &self,
//         func_id: FunctionId,
//         block_id: BlockId,
//         value_id: ValueId,
//     ) -> String {
//         let Some(taint) = self.get_taint_type(func_id, block_id, value_id) else {
//             return "".to_string();
//         };
//         taint.to_string()
//     }

//     fn emit_le(&mut self, left: Taint, right: Taint) {
//         self.unification_context
//             .judgements
//             .push(Judgement::Le(left, right));
//     }

//     fn deep_eq(&mut self, left: TaintType, right: TaintType) {
//         let left_toplevel = left.toplevel_taint();
//         let right_toplevel = right.toplevel_taint();
//         self.emit_eq(left_toplevel, right_toplevel);
//         if let (Some(l), Some(r)) = (left.child_taint_type(), right.child_taint_type()) {
//             self.deep_eq(l, r);
//         }
//     }

//     fn emit_eq(&mut self, left: Taint, right: Taint) {
//         self.unification_context
//             .judgements
//             .push(Judgement::Eq(left, right));
//     }

//     pub fn judgements_string(&self) -> String {
//         self.unification_context
//             .judgements
//             .iter()
//             .map(|j| match j {
//                 Judgement::Eq(l, r) => format!("{} = {}", l.to_string(), r.to_string()),
//                 Judgement::Le(l, r) => format!("{} ≤ {}", l.to_string(), r.to_string()),
//             })
//             .collect::<Vec<_>>()
//             .join("\n")
//     }

//     pub fn get_judgements(&self) -> Vec<String> {
//         self.unification_context
//             .judgements
//             .iter()
//             .map(|j| format!("{}", j))
//             .collect()
//     }

//     pub fn get_judgements_data(&self) -> &[Judgement] {
//         &self.unification_context.judgements
//     }
// }

impl std::fmt::Display for Judgement {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Judgement::Eq(a, b) => write!(f, "{} = {}", a.to_string(), b.to_string()),
            Judgement::Le(a, b) => write!(f, "{} ≤ {}", a.to_string(), b.to_string()),
        }
    }
}

impl std::fmt::Debug for Judgement {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Judgement::Eq(a, b) => write!(f, "{} = {}", a.to_string(), b.to_string()),
            Judgement::Le(a, b) => write!(f, "{} ≤ {}", a.to_string(), b.to_string()),
        }
    }
}
