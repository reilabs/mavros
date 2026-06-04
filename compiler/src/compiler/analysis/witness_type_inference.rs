//! Performs whole program analysis to determine which values are potentially witness tainted, which
//! are _only_ witnesses, and which are only non-witness values.

use std::collections::{HashMap, HashSet, VecDeque};

use super::witness_info::{FunctionWitnessType, WitnessShape, WitnessType};
use crate::compiler::{
    analysis::{
        flow_analysis::FlowAnalysis,
        value_definitions::{FunctionValueDefinitions, ValueDefinition},
    },
    ssa::{
        BlockId, FunctionId, Instruction, SSAAnotator, Terminator, ValueId,
        hlssa::{CallTarget, Constant, HLSSA, OpCode, Type, TypeExpr},
    },
};

// ---------------------------------------------------------------------------
// Specialization key and value
// ---------------------------------------------------------------------------

#[derive(Eq, Hash, PartialEq, Clone, Debug)]
struct SpecKey {
    original_func_id: FunctionId,
    arg_types: Vec<WitnessShape>,
    cfg_witness: WitnessType,
}

#[derive(Clone, Debug)]
struct SpecValue {
    specialized_func_id: FunctionId,
    return_types: Vec<WitnessShape>,
    return_constraints: Vec<WitnessShape>,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct RuleId {
    function_id: FunctionId,
    block_id: BlockId,
    kind: RuleKind,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
enum RuleKind {
    Instruction(usize),
    Terminator,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
enum Event {
    NewSpec(SpecKey),
    SpecChanged(FunctionId),
    ReturnConstraintChanged(FunctionId),
    ValueChanged(ValueId),
    BlockCfgChanged(FunctionId, BlockId),
    Rule(RuleId),
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

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
        let main_id = ssa.get_main_id();
        let main_func = ssa.get_function(main_id);

        let main_arg_types: Vec<WitnessShape> = main_func
            .get_entry()
            .get_parameters()
            .map(|(_, tp)| Self::construct_pure_witness_for_type(tp))
            .collect();
        let main_return_types: Vec<WitnessShape> = main_func
            .get_returns()
            .iter()
            .map(Self::construct_pure_witness_for_type)
            .collect();

        let main_key = SpecKey {
            original_func_id: main_id,
            arg_types: main_arg_types.clone(),
            cfg_witness: WitnessType::Pure,
        };

        let main_specialized_id = ssa.duplicate_function(main_id);
        ssa.set_entry_point(main_specialized_id);

        let mut engine = WtiEngine::new(ssa, flow_analysis);
        engine.install_existing_spec(main_key, main_specialized_id, main_return_types);
        engine.run();
        self.functions = engine.finish();

        Ok(())
    }

    // ---------------------------------------------------------------------------
    // Helpers
    // ---------------------------------------------------------------------------

    fn resolve_key(key: &SpecKey, redirects: &HashMap<SpecKey, SpecKey>) -> SpecKey {
        let mut current = key.clone();
        let mut seen = HashSet::new();
        while let Some(next) = redirects.get(&current) {
            assert!(
                seen.insert(current.clone()),
                "Cycle in witness specialization redirects at {:?}",
                current
            );
            current = next.clone();
        }
        current
    }

    fn array_element(array_wt: &WitnessShape) -> WitnessShape {
        match array_wt {
            WitnessShape::Array(_, elem) => *elem.clone(),
            other => panic!(
                "Array element access on non-array witness type: {:?}",
                other
            ),
        }
    }

    fn tuple_element(tuple_wt: &WitnessShape, idx: usize) -> (WitnessType, WitnessShape) {
        match tuple_wt {
            WitnessShape::Tuple(tuple_info, children) => (*tuple_info, children[idx].clone()),
            other => panic!(
                "Tuple element access on non-tuple witness type: {:?}",
                other
            ),
        }
    }

    fn construct_pure_witness_for_type(typ: &Type) -> WitnessShape {
        match &typ.expr {
            TypeExpr::U(_) | TypeExpr::I(_) | TypeExpr::Field => {
                WitnessShape::Scalar(WitnessType::Pure)
            }
            TypeExpr::Array(i, _) => WitnessShape::Array(
                WitnessType::Pure,
                Box::new(Self::construct_pure_witness_for_type(i)),
            ),
            TypeExpr::Slice(i) => WitnessShape::Array(
                WitnessType::Pure,
                Box::new(Self::construct_pure_witness_for_type(i)),
            ),
            TypeExpr::Ref(i) => WitnessShape::Ref(
                WitnessType::Pure,
                Box::new(Self::construct_pure_witness_for_type(i)),
            ),
            TypeExpr::WitnessOf(_) => {
                panic!("ICE: WitnessOf should not be present at this stage");
            }
            TypeExpr::Tuple(elements) => WitnessShape::Tuple(
                WitnessType::Pure,
                elements
                    .iter()
                    .map(Self::construct_pure_witness_for_type)
                    .collect(),
            ),
            TypeExpr::Function => WitnessShape::Scalar(WitnessType::Pure),
        }
    }
}

struct WtiEngine<'a> {
    ssa: &'a mut HLSSA,
    flow_analysis: &'a FlowAnalysis,
    functions: HashMap<FunctionId, FunctionWitnessType>,

    specializations: HashMap<SpecKey, SpecValue>,
    spec_by_func: HashMap<FunctionId, SpecKey>,
    spec_remap: HashMap<SpecKey, SpecKey>,

    value_wt: HashMap<ValueId, WitnessShape>,
    block_cfg: HashMap<(FunctionId, BlockId), WitnessType>,
    value_function: HashMap<ValueId, FunctionId>,
    value_defs: HashMap<FunctionId, FunctionValueDefinitions>,
    entry_arg: HashMap<ValueId, (FunctionId, usize)>,

    value_deps: HashMap<ValueId, Vec<ValueId>>,
    effect_deps: HashMap<ValueId, Vec<RuleId>>,
    block_rules: HashMap<(FunctionId, BlockId), Vec<RuleId>>,
    return_rules: HashMap<FunctionId, Vec<RuleId>>,
    spec_callers: HashMap<FunctionId, HashSet<RuleId>>,
    pending_callers: HashMap<SpecKey, HashSet<RuleId>>,

    events: VecDeque<Event>,
    queued_events: HashSet<Event>,
}

impl<'a> WtiEngine<'a> {
    fn new(ssa: &'a mut HLSSA, flow_analysis: &'a FlowAnalysis) -> Self {
        let mut value_wt = HashMap::new();
        ssa.for_each_const(|vid, val| {
            let shape = match val.as_ref() {
                Constant::U(_, _) | Constant::I(_, _) | Constant::Field(_) | Constant::FnPtr(_) => {
                    WitnessShape::Scalar(WitnessType::Pure)
                }
            };
            value_wt.insert(*vid, shape);
        });

        Self {
            ssa,
            flow_analysis,
            functions: HashMap::new(),
            specializations: HashMap::new(),
            spec_by_func: HashMap::new(),
            spec_remap: HashMap::new(),
            value_wt,
            block_cfg: HashMap::new(),
            value_function: HashMap::new(),
            value_defs: HashMap::new(),
            entry_arg: HashMap::new(),
            value_deps: HashMap::new(),
            effect_deps: HashMap::new(),
            block_rules: HashMap::new(),
            return_rules: HashMap::new(),
            spec_callers: HashMap::new(),
            pending_callers: HashMap::new(),
            events: VecDeque::new(),
            queued_events: HashSet::new(),
        }
    }

    fn install_existing_spec(
        &mut self,
        key: SpecKey,
        specialized_func_id: FunctionId,
        return_types: Vec<WitnessShape>,
    ) {
        let key = self.resolve_key(&key);
        if self.specializations.contains_key(&key) {
            return;
        }

        self.specializations.insert(
            key.clone(),
            SpecValue {
                specialized_func_id,
                return_types: return_types.clone(),
                return_constraints: return_types,
            },
        );
        self.spec_by_func.insert(specialized_func_id, key.clone());
        self.register_function(&key, specialized_func_id);
        self.flush_pending_callers(&key);
    }

    fn run(&mut self) {
        while let Some(event) = self.events.pop_front() {
            self.queued_events.remove(&event);
            match event {
                Event::NewSpec(key) => self.install_new_spec(key),
                Event::SpecChanged(function_id) => {
                    for rule in self
                        .spec_callers
                        .get(&function_id)
                        .cloned()
                        .unwrap_or_default()
                    {
                        self.enqueue(Event::Rule(rule));
                    }
                }
                Event::ReturnConstraintChanged(function_id) => {
                    for rule in self
                        .return_rules
                        .get(&function_id)
                        .cloned()
                        .unwrap_or_default()
                    {
                        self.enqueue(Event::Rule(rule));
                    }
                }
                Event::ValueChanged(value) => self.value_changed(value),
                Event::BlockCfgChanged(function_id, block_id) => {
                    for rule in self
                        .block_rules
                        .get(&(function_id, block_id))
                        .cloned()
                        .unwrap_or_default()
                    {
                        self.enqueue(Event::Rule(rule));
                    }
                }
                Event::Rule(rule) => self.run_rule(rule),
            }
        }
    }

    fn finish(mut self) -> HashMap<FunctionId, FunctionWitnessType> {
        let specs: Vec<(SpecKey, SpecValue)> = self
            .specializations
            .iter()
            .map(|(key, value)| (key.clone(), value.clone()))
            .collect();

        for (spec_key, spec_value) in specs {
            let function_id = spec_value.specialized_func_id;
            if self.spec_by_func.get(&function_id) != Some(&spec_key) {
                continue;
            }

            let cfg = self
                .flow_analysis
                .get_function_cfg(spec_key.original_func_id);
            let block_order: Vec<BlockId> = cfg.get_blocks_bfs().collect();
            let mut specialized_func = self.ssa.take_function(function_id);

            for block_id in &block_order {
                let block_cw = self
                    .block_cfg
                    .get(&(function_id, *block_id))
                    .copied()
                    .unwrap_or(spec_key.cfg_witness);
                let block = specialized_func.get_block_mut(*block_id);
                for instruction in block.get_instructions_mut() {
                    if let OpCode::Call {
                        function: CallTarget::Static(callee_id),
                        args,
                        unconstrained,
                        ..
                    } = instruction
                    {
                        if *unconstrained {
                            continue;
                        }
                        let callee_key = self.resolve_key(&SpecKey {
                            original_func_id: *callee_id,
                            arg_types: args.iter().map(|v| self.witness(*v).clone()).collect(),
                            cfg_witness: block_cw,
                        });
                        let callee_spec =
                            self.specializations.get(&callee_key).unwrap_or_else(|| {
                                panic!("Missing witness specialization for {:?}", callee_key)
                            });
                        *callee_id = callee_spec.specialized_func_id;
                    }
                }
            }

            self.ssa.put_function(function_id, specialized_func);

            let func = self.ssa.get_function(function_id);
            let mut block_cfg_witness = HashMap::new();
            let mut value_witness_types = HashMap::new();
            for block_id in block_order {
                if let Some(block_wt) = self.block_cfg.get(&(function_id, block_id)) {
                    block_cfg_witness.insert(block_id, *block_wt);
                }
                let block = func.get_block(block_id);
                for (value, _) in block.get_parameters() {
                    if let Some(wt) = self.value_wt.get(value) {
                        value_witness_types.insert(*value, wt.clone());
                    }
                }
                for instruction in block.get_instructions() {
                    for value in instruction.get_inputs().chain(instruction.get_results()) {
                        if let Some(wt) = self.value_wt.get(value) {
                            value_witness_types.insert(*value, wt.clone());
                        }
                    }
                }
                if let Some(terminator) = block.get_terminator() {
                    match terminator {
                        Terminator::Jmp(_, values) | Terminator::Return(values) => {
                            for value in values {
                                if let Some(wt) = self.value_wt.get(value) {
                                    value_witness_types.insert(*value, wt.clone());
                                }
                            }
                        }
                        Terminator::JmpIf(cond, _, _) => {
                            if let Some(wt) = self.value_wt.get(cond) {
                                value_witness_types.insert(*cond, wt.clone());
                            }
                        }
                    }
                }
            }

            self.functions.insert(
                function_id,
                FunctionWitnessType {
                    returns_witness: spec_value.return_types,
                    cfg_witness: spec_key.cfg_witness,
                    parameters: spec_key.arg_types,
                    block_cfg_witness,
                    value_witness_types,
                },
            );
        }

        self.functions
    }

    fn install_new_spec(&mut self, key: SpecKey) {
        let key = self.resolve_key(&key);
        if self.specializations.contains_key(&key) {
            self.flush_pending_callers(&key);
            return;
        }

        let callee_func = self.ssa.get_function(key.original_func_id);
        let return_types: Vec<WitnessShape> = callee_func
            .get_returns()
            .iter()
            .map(WitnessTypeInference::construct_pure_witness_for_type)
            .collect();
        let specialized_id = self.ssa.duplicate_function(key.original_func_id);
        self.install_existing_spec(key, specialized_id, return_types);
    }

    fn register_function(&mut self, key: &SpecKey, function_id: FunctionId) {
        let cfg = self.flow_analysis.get_function_cfg(key.original_func_id);
        let block_order: Vec<BlockId> = cfg.get_blocks_bfs().collect();
        let entry_id = self.ssa.get_function(function_id).get_entry_id();
        self.value_defs.insert(
            function_id,
            FunctionValueDefinitions::from_function(self.ssa.get_function(function_id)),
        );

        let blocks: Vec<_> = {
            let func = self.ssa.get_function(function_id);
            block_order
                .iter()
                .map(|block_id| {
                    let block = func.get_block(*block_id);
                    (
                        *block_id,
                        block.get_parameters().cloned().collect::<Vec<_>>(),
                        block.get_instructions().cloned().collect::<Vec<_>>(),
                        block.get_terminator().cloned(),
                    )
                })
                .collect()
        };

        for (block_id, params, instructions, terminator) in blocks {
            self.block_cfg
                .entry((function_id, block_id))
                .or_insert(key.cfg_witness);

            for (idx, (value, typ)) in params.iter().enumerate() {
                self.value_function.insert(*value, function_id);
                if block_id == entry_id {
                    self.entry_arg.insert(*value, (function_id, idx));
                    self.value_wt.insert(*value, key.arg_types[idx].clone());
                } else {
                    self.value_wt.entry(*value).or_insert_with(|| {
                        WitnessTypeInference::construct_pure_witness_for_type(typ)
                    });
                }
            }

            for (idx, instruction) in instructions.iter().enumerate() {
                let rule = RuleId {
                    function_id,
                    block_id,
                    kind: RuleKind::Instruction(idx),
                };
                self.block_rules
                    .entry((function_id, block_id))
                    .or_default()
                    .push(rule);
                self.register_instruction_deps(rule, instruction);
                self.enqueue(Event::Rule(rule));
            }

            if let Some(terminator) = terminator {
                let rule = RuleId {
                    function_id,
                    block_id,
                    kind: RuleKind::Terminator,
                };
                self.block_rules
                    .entry((function_id, block_id))
                    .or_default()
                    .push(rule);
                self.register_terminator_deps(rule, &terminator);
                self.enqueue(Event::Rule(rule));
            }
        }
    }

    fn register_instruction_deps(&mut self, rule: RuleId, instruction: &OpCode) {
        let inputs: Vec<ValueId> = instruction.get_inputs().copied().collect();
        let results: Vec<ValueId> = instruction.get_results().copied().collect();

        for result in &results {
            self.value_function.insert(*result, rule.function_id);
            self.add_value_dep(*result, *result);
        }
        for input in &inputs {
            for result in &results {
                self.add_value_dep(*input, *result);
            }
        }

        match instruction {
            OpCode::Store { ptr, value } => {
                self.add_effect_dep(*ptr, rule);
                self.add_effect_dep(*value, rule);
            }
            OpCode::Call { results, args, .. } => {
                for arg in args {
                    self.add_effect_dep(*arg, rule);
                }
                for result in results {
                    self.add_effect_dep(*result, rule);
                    self.add_value_dep(*result, *result);
                }
            }
            _ => {}
        }
    }

    fn register_terminator_deps(&mut self, rule: RuleId, terminator: &Terminator) {
        match terminator {
            Terminator::Return(values) => {
                self.return_rules
                    .entry(rule.function_id)
                    .or_default()
                    .push(rule);
                for value in values {
                    self.add_effect_dep(*value, rule);
                }
            }
            Terminator::Jmp(target, params) => {
                let target_params: Vec<ValueId> = self
                    .ssa
                    .get_function(rule.function_id)
                    .get_block(*target)
                    .get_parameters()
                    .map(|(value, _)| *value)
                    .collect();
                for value in params.iter().chain(target_params.iter()) {
                    self.add_effect_dep(*value, rule);
                }
            }
            Terminator::JmpIf(cond, _, _) => {
                self.add_effect_dep(*cond, rule);
            }
        }
    }

    fn value_changed(&mut self, value: ValueId) {
        self.retype_entry_arg(value);

        for target in self.value_deps.get(&value).cloned().unwrap_or_default() {
            if let Some(rule) = self.defining_rule(target) {
                self.enqueue(Event::Rule(rule));
            }
        }
        for rule in self.effect_deps.get(&value).cloned().unwrap_or_default() {
            self.enqueue(Event::Rule(rule));
        }
    }

    fn run_rule(&mut self, rule: RuleId) {
        if !self.spec_by_func.contains_key(&rule.function_id) {
            return;
        }

        match rule.kind {
            RuleKind::Instruction(idx) => {
                let instruction = self
                    .ssa
                    .get_function(rule.function_id)
                    .get_block(rule.block_id)
                    .get_instruction(idx)
                    .clone();
                let block_cw = self.block_witness(rule.function_id, rule.block_id);
                self.run_instruction(rule, block_cw, instruction);
            }
            RuleKind::Terminator => {
                let Some(terminator) = self
                    .ssa
                    .get_function(rule.function_id)
                    .get_block(rule.block_id)
                    .get_terminator()
                    .cloned()
                else {
                    return;
                };
                let block_cw = self.block_witness(rule.function_id, rule.block_id);
                self.run_terminator(rule, block_cw, terminator);
            }
        }
    }

    fn run_instruction(&mut self, rule: RuleId, block_cw: WitnessType, instruction: OpCode) {
        match instruction {
            OpCode::BinaryArithOp {
                kind: _,
                result,
                lhs,
                rhs,
            }
            | OpCode::Cmp {
                kind: _,
                result,
                lhs,
                rhs,
            } => {
                let result_wt = WitnessShape::Scalar(
                    self.witness(lhs)
                        .toplevel_info()
                        .join(self.witness(rhs).toplevel_info()),
                );
                self.join_value(result, result_wt);
            }
            OpCode::Select {
                result,
                cond,
                if_t,
                if_f,
            } => {
                let cond_wt = self.witness(cond);
                let then_wt = self.witness(if_t);
                let else_wt = self.witness(if_f);
                let result_wt = then_wt.join(else_wt).with_toplevel_info(
                    cond_wt
                        .toplevel_info()
                        .join(then_wt.toplevel_info())
                        .join(else_wt.toplevel_info()),
                );
                let result_wt = self.join_value(result, result_wt);
                if result_wt.contains_ref() {
                    self.join_value(if_t, result_wt.clone());
                    self.join_value(if_f, result_wt);
                }
            }
            OpCode::Alloc { result, elem_type } => {
                let inner = WitnessTypeInference::construct_pure_witness_for_type(&elem_type);
                self.join_value(
                    result,
                    WitnessShape::Ref(WitnessType::Pure, Box::new(inner)),
                );
            }
            OpCode::Store { ptr, value } => {
                let val_wt = self.witness(value).clone();
                let store_wt = val_wt.with_toplevel_info(val_wt.toplevel_info().join(block_cw));
                self.join_ref_inner(ptr, store_wt.clone());
                if store_wt.contains_ref() {
                    let stored_wt = self.read_ref_inner(ptr);
                    self.join_value(value, stored_wt);
                }
            }
            OpCode::Load { result, ptr } => {
                let result_wt = self.read_ref_inner(ptr);
                let result_wt = self.join_value(result, result_wt);
                if result_wt.contains_ref() {
                    self.join_ref_inner(ptr, result_wt);
                }
            }
            OpCode::ReadGlobal {
                result,
                offset: _,
                result_type,
            } => {
                self.join_value(
                    result,
                    WitnessTypeInference::construct_pure_witness_for_type(&result_type),
                );
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
                let arr_wt = self.witness(array);
                let idx_wt = self.witness(index);
                let result_wt = WitnessTypeInference::array_element(arr_wt)
                    .with_witness_in_leaves(arr_wt.toplevel_info().join(idx_wt.toplevel_info()));
                let result_wt = self.join_value(result, result_wt);
                if result_wt.contains_ref() {
                    self.join_array_element(array, &result_wt);
                }
            }
            OpCode::ArraySet {
                result,
                array,
                index,
                value,
            } => {
                let arr_wt = self.witness(array);
                let idx_wt = self.witness(index);
                let val_wt = self.witness(value);
                let arr_elem_wt = WitnessTypeInference::array_element(arr_wt);
                let arr_top = arr_wt.toplevel_info();
                let result_elem_wt = arr_elem_wt.join(val_wt).with_toplevel_info(
                    arr_elem_wt
                        .toplevel_info()
                        .join(val_wt.toplevel_info())
                        .join(idx_wt.toplevel_info()),
                );
                self.join_value(
                    result,
                    WitnessShape::Array(arr_top, Box::new(result_elem_wt)),
                );
                let result_elem_wt = WitnessTypeInference::array_element(self.witness(result));
                if result_elem_wt.contains_ref() {
                    self.join_array_element(array, &result_elem_wt);
                    self.join_value(value, result_elem_wt);
                }
            }
            OpCode::SlicePush {
                dir: _,
                result,
                slice,
                values,
            } => {
                let slice_wt = self.witness(slice);
                let slice_top = slice_wt.toplevel_info();
                let mut result_elem_wt = WitnessTypeInference::array_element(slice_wt);
                for value in &values {
                    result_elem_wt = result_elem_wt.join(self.witness(*value));
                }
                self.join_value(
                    result,
                    WitnessShape::Array(slice_top, Box::new(result_elem_wt)),
                );
                let result_elem_wt = WitnessTypeInference::array_element(self.witness(result));
                if result_elem_wt.contains_ref() {
                    self.join_array_element(slice, &result_elem_wt);
                    for value in values {
                        self.join_value(value, result_elem_wt.clone());
                    }
                }
            }
            OpCode::SliceLen { result, slice: _ } => {
                self.join_value(result, WitnessShape::Scalar(WitnessType::Pure));
            }
            OpCode::Call {
                results,
                function: CallTarget::Static(callee_id),
                args,
                unconstrained,
            } => {
                self.run_call(rule, block_cw, results, callee_id, args, unconstrained);
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
                seq_type: _,
                elem_type,
            } => {
                let result_elem_wt = elems.iter().fold(
                    WitnessTypeInference::construct_pure_witness_for_type(&elem_type),
                    |acc, elem| acc.join(self.witness(*elem)),
                );
                self.join_value(
                    result,
                    WitnessShape::Array(WitnessType::Pure, Box::new(result_elem_wt)),
                );
                let result_elem_wt = WitnessTypeInference::array_element(self.witness(result));
                if result_elem_wt.contains_ref() {
                    for elem in elems {
                        self.join_value(elem, result_elem_wt.clone());
                    }
                }
            }
            OpCode::MkRepeated {
                result,
                element,
                seq_type: _,
                count: _,
                elem_type,
            } => {
                let result_elem_wt =
                    WitnessTypeInference::construct_pure_witness_for_type(&elem_type)
                        .join(self.witness(element));
                self.join_value(
                    result,
                    WitnessShape::Array(WitnessType::Pure, Box::new(result_elem_wt)),
                );
                let result_elem_wt = WitnessTypeInference::array_element(self.witness(result));
                if result_elem_wt.contains_ref() {
                    self.join_value(element, result_elem_wt);
                }
            }
            OpCode::Unspread {
                result_odd,
                result_even,
                value,
                ..
            } => {
                let val_wt = self.witness(value).clone();
                let odd_wt = self.join_value(result_odd, val_wt.clone());
                let even_wt = self.join_value(result_even, val_wt);
                if odd_wt.contains_ref() {
                    self.join_value(value, odd_wt);
                }
                if even_wt.contains_ref() {
                    self.join_value(value, even_wt);
                }
            }
            OpCode::Spread { result, value, .. }
            | OpCode::Cast { result, value, .. }
            | OpCode::SExt { result, value, .. }
            | OpCode::BitRange { result, value, .. }
            | OpCode::Not { result, value } => {
                let input_wt = self.witness(value).clone();
                let result_wt = self.join_value(result, input_wt);
                if result_wt.contains_ref() {
                    self.join_value(value, result_wt);
                }
            }
            OpCode::ToBits { result, value, .. } | OpCode::ToRadix { result, value, .. } => {
                let input_wt = self.witness(value).clone();
                self.join_value(
                    result,
                    WitnessShape::Array(WitnessType::Pure, Box::new(input_wt)),
                );
                let result_elem_wt = WitnessTypeInference::array_element(self.witness(result));
                if result_elem_wt.contains_ref() {
                    self.join_value(value, result_elem_wt);
                }
            }
            OpCode::TupleProj { result, tuple, idx } => {
                let (tuple_top, elem_wt) =
                    WitnessTypeInference::tuple_element(self.witness(tuple), idx);
                let result_wt = elem_wt.with_toplevel_info(tuple_top.join(elem_wt.toplevel_info()));
                let result_wt = self.join_value(result, result_wt);
                if result_wt.contains_ref() {
                    self.join_tuple_element(tuple, idx, &result_wt);
                }
            }
            OpCode::MkTuple { result, elems, .. } => {
                let children: Vec<WitnessShape> =
                    elems.iter().map(|v| self.witness(*v).clone()).collect();
                self.join_value(result, WitnessShape::Tuple(WitnessType::Pure, children));
                if let WitnessShape::Tuple(_, children) = self.witness(result).clone() {
                    for (elem, child_wt) in elems.iter().zip(children.iter()) {
                        if child_wt.contains_ref() {
                            self.join_value(*elem, child_wt.clone());
                        }
                    }
                }
            }
            OpCode::WriteWitness { result, .. } => {
                if let Some(result) = result {
                    self.join_value(result, WitnessShape::Scalar(WitnessType::Witness));
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
            _ => {
                panic!(
                    "Unsupported opcode during witness type inference: {:?}",
                    instruction
                );
            }
        }
    }

    fn run_call(
        &mut self,
        rule: RuleId,
        block_cw: WitnessType,
        results: Vec<ValueId>,
        callee_id: FunctionId,
        args: Vec<ValueId>,
        unconstrained: bool,
    ) {
        if unconstrained {
            self.join_pure_returns(callee_id, &results);
            return;
        }

        let callee_key = self.resolve_key(&SpecKey {
            original_func_id: callee_id,
            arg_types: args.iter().map(|v| self.witness(*v).clone()).collect(),
            cfg_witness: block_cw,
        });

        let Some(callee_spec) = self.specializations.get(&callee_key).cloned() else {
            self.pending_callers
                .entry(callee_key.clone())
                .or_default()
                .insert(rule);
            self.enqueue(Event::NewSpec(callee_key));
            self.join_pure_returns(callee_id, &results);
            return;
        };

        self.spec_callers
            .entry(callee_spec.specialized_func_id)
            .or_default()
            .insert(rule);

        for (result, ret_wt) in results.iter().zip(&callee_spec.return_types) {
            self.join_value(*result, ret_wt.clone());
        }
        for (arg, arg_wt) in args.iter().zip(&callee_key.arg_types) {
            self.join_value(*arg, arg_wt.clone());
        }
        self.join_return_constraints(callee_spec.specialized_func_id, &results);
    }

    fn run_terminator(&mut self, rule: RuleId, block_cw: WitnessType, terminator: Terminator) {
        match terminator {
            Terminator::Return(values) => {
                let spec_key = self.spec_by_func.get(&rule.function_id).unwrap().clone();
                let constraints = self
                    .specializations
                    .get(&spec_key)
                    .unwrap()
                    .return_constraints
                    .clone();
                let mut ret_wts = Vec::new();
                for (value, constraint) in values.iter().zip(constraints.iter()) {
                    if constraint.contains_ref() {
                        self.join_value(*value, constraint.clone());
                    }
                    ret_wts.push(self.witness(*value).join(constraint));
                }
                self.join_return_types(rule.function_id, ret_wts);
            }
            Terminator::Jmp(target, params) => {
                let target_params: Vec<ValueId> = self
                    .ssa
                    .get_function(rule.function_id)
                    .get_block(target)
                    .get_parameters()
                    .map(|(value, _)| *value)
                    .collect();
                for (target_value, param) in target_params.iter().zip(params.iter()) {
                    let param_wt = self.witness(*param).clone();
                    let joined = self.join_value(*target_value, param_wt);
                    if joined.contains_ref() {
                        self.join_value(*param, joined);
                    }
                }
            }
            Terminator::JmpIf(cond, _if_true, _if_false) => {
                let cond_toplevel = self.witness(cond).toplevel_info();
                let spec_key = self.spec_by_func.get(&rule.function_id).unwrap();
                let cfg = self
                    .flow_analysis
                    .get_function_cfg(spec_key.original_func_id);
                let merge = cfg.get_merge_point(rule.block_id);
                let merge_params: Vec<ValueId> = self
                    .ssa
                    .get_function(rule.function_id)
                    .get_block(merge)
                    .get_parameters()
                    .map(|(value, _)| *value)
                    .collect();
                for param_id in merge_params {
                    let joined = self.witness(param_id).with_witness_in_leaves(cond_toplevel);
                    self.join_value(param_id, joined);
                }

                for body_block_id in cfg.get_if_body(rule.block_id) {
                    self.join_block_cfg(
                        rule.function_id,
                        body_block_id,
                        cond_toplevel.join(block_cw),
                    );
                }
            }
        }
    }

    fn retype_entry_arg(&mut self, value: ValueId) {
        let Some((function_id, idx)) = self.entry_arg.get(&value).copied() else {
            return;
        };
        let Some(old_key) = self.spec_by_func.get(&function_id).cloned() else {
            return;
        };
        if !old_key.arg_types[idx].contains_ref() {
            return;
        }

        let joined = old_key.arg_types[idx].join(self.witness(value));
        if joined == old_key.arg_types[idx] {
            return;
        }

        let mut new_arg_types = old_key.arg_types.clone();
        new_arg_types[idx] = joined;
        let new_key = self.resolve_key(&SpecKey {
            original_func_id: old_key.original_func_id,
            arg_types: new_arg_types,
            cfg_witness: old_key.cfg_witness,
        });
        self.remap_spec(function_id, old_key, new_key);
    }

    fn remap_spec(&mut self, function_id: FunctionId, old_key: SpecKey, new_key: SpecKey) {
        if old_key == new_key {
            return;
        }
        self.spec_remap.insert(old_key.clone(), new_key.clone());
        if let Some(pending) = self.pending_callers.remove(&old_key) {
            self.pending_callers
                .entry(new_key.clone())
                .or_default()
                .extend(pending);
        }

        let old_value = self.specializations.remove(&old_key).unwrap();
        if let Some(existing) = self.specializations.get_mut(&new_key) {
            let existing_function_id = existing.specialized_func_id;
            existing.return_types =
                Self::join_shapes(&existing.return_types, &old_value.return_types);
            existing.return_constraints =
                Self::join_shapes(&existing.return_constraints, &old_value.return_constraints);
            self.spec_by_func.remove(&function_id);
            if let Some(callers) = self.spec_callers.remove(&function_id) {
                self.spec_callers
                    .entry(existing_function_id)
                    .or_default()
                    .extend(callers);
            }
            self.enqueue(Event::SpecChanged(existing_function_id));
            self.enqueue(Event::ReturnConstraintChanged(existing_function_id));
        } else {
            self.specializations.insert(new_key.clone(), old_value);
            self.spec_by_func.insert(function_id, new_key);
            self.enqueue(Event::SpecChanged(function_id));
            self.enqueue(Event::ReturnConstraintChanged(function_id));
            for rule in self
                .block_rules
                .iter()
                .filter_map(|((fid, _), rules)| {
                    if *fid == function_id {
                        Some(rules.clone())
                    } else {
                        None
                    }
                })
                .flatten()
                .collect::<Vec<_>>()
            {
                self.enqueue(Event::Rule(rule));
            }
        }
    }

    fn join_return_types(&mut self, function_id: FunctionId, ret_wts: Vec<WitnessShape>) {
        let spec_key = self.spec_by_func.get(&function_id).unwrap().clone();
        let spec = self.specializations.get_mut(&spec_key).unwrap();
        let joined = Self::join_shapes(&spec.return_types, &ret_wts);
        if joined != spec.return_types {
            spec.return_types = joined;
            self.enqueue(Event::SpecChanged(function_id));
        }
    }

    fn join_return_constraints(&mut self, function_id: FunctionId, results: &[ValueId]) {
        let spec_key = self.spec_by_func.get(&function_id).unwrap().clone();
        let result_types: Vec<WitnessShape> = results
            .iter()
            .map(|result| self.witness(*result).clone())
            .collect();
        let spec = self.specializations.get_mut(&spec_key).unwrap();
        assert_eq!(
            spec.return_constraints.len(),
            result_types.len(),
            "Cannot join return constraints of different lengths"
        );

        let joined: Vec<WitnessShape> = spec
            .return_constraints
            .iter()
            .zip(result_types.iter())
            .map(|(current, seen)| {
                if current.contains_ref() || seen.contains_ref() {
                    current.join(seen)
                } else {
                    current.clone()
                }
            })
            .collect();
        if joined != spec.return_constraints {
            spec.return_constraints = joined;
            self.enqueue(Event::ReturnConstraintChanged(function_id));
        }
    }

    fn join_pure_returns(&mut self, function_id: FunctionId, results: &[ValueId]) {
        let return_types = self.ssa.get_function(function_id).get_returns().to_vec();
        for (result, ret_type) in results.iter().zip(return_types.iter()) {
            self.join_value(
                *result,
                WitnessTypeInference::construct_pure_witness_for_type(ret_type),
            );
        }
    }

    fn flush_pending_callers(&mut self, key: &SpecKey) {
        let key = self.resolve_key(key);
        if let Some(callers) = self.pending_callers.remove(&key) {
            for rule in callers {
                self.enqueue(Event::Rule(rule));
            }
        }
    }

    fn defining_rule(&self, value: ValueId) -> Option<RuleId> {
        let function_id = *self.value_function.get(&value)?;
        match self.value_defs.get(&function_id)?.get_definition(value)? {
            ValueDefinition::Instruction(block_id, idx, _) => Some(RuleId {
                function_id,
                block_id: *block_id,
                kind: RuleKind::Instruction(*idx),
            }),
            ValueDefinition::Param(_, _, _) => None,
        }
    }

    fn add_value_dep(&mut self, source: ValueId, target: ValueId) {
        self.value_deps.entry(source).or_default().push(target);
    }

    fn add_effect_dep(&mut self, source: ValueId, rule: RuleId) {
        self.effect_deps.entry(source).or_default().push(rule);
    }

    fn enqueue(&mut self, event: Event) {
        if self.queued_events.insert(event.clone()) {
            self.events.push_back(event);
        }
    }

    fn resolve_key(&self, key: &SpecKey) -> SpecKey {
        WitnessTypeInference::resolve_key(key, &self.spec_remap)
    }

    fn block_witness(&self, function_id: FunctionId, block_id: BlockId) -> WitnessType {
        *self
            .block_cfg
            .get(&(function_id, block_id))
            .unwrap_or_else(|| panic!("Missing block witness for {:?} {:?}", function_id, block_id))
    }

    fn join_block_cfg(&mut self, function_id: FunctionId, block_id: BlockId, wt: WitnessType) {
        let current = self
            .block_cfg
            .get(&(function_id, block_id))
            .copied()
            .unwrap_or(WitnessType::Pure);
        let joined = current.join(wt);
        if joined != current {
            self.block_cfg.insert((function_id, block_id), joined);
            self.enqueue(Event::BlockCfgChanged(function_id, block_id));
        }
    }

    fn witness(&self, value: ValueId) -> &WitnessShape {
        self.value_wt
            .get(&value)
            .unwrap_or_else(|| panic!("Missing witness type for value {:?}", value))
    }

    fn join_value(&mut self, value: ValueId, wt: WitnessShape) -> WitnessShape {
        let joined = self
            .value_wt
            .get(&value)
            .map(|existing| existing.join(&wt))
            .unwrap_or(wt);
        if self.value_wt.get(&value) != Some(&joined) {
            self.value_wt.insert(value, joined.clone());
            self.enqueue(Event::ValueChanged(value));
        }
        joined
    }

    fn read_ref_inner(&self, ptr: ValueId) -> WitnessShape {
        match self.witness(ptr) {
            WitnessShape::Ref(ptr_info, inner) => {
                inner.with_toplevel_info(inner.toplevel_info().join(*ptr_info))
            }
            other => panic!("Load from non-ref witness type: {:?}", other),
        }
    }

    fn join_ref_inner(&mut self, ptr: ValueId, inner_wt: WitnessShape) {
        let updated = match self.witness(ptr).clone() {
            WitnessShape::Ref(ptr_info, inner) => {
                WitnessShape::Ref(ptr_info, Box::new(inner.join(&inner_wt)))
            }
            other => panic!("Store to non-ref witness type: {:?}", other),
        };
        self.join_value(ptr, updated);
    }

    fn join_array_element(&mut self, array: ValueId, elem_wt: &WitnessShape) {
        let updated = match self.witness(array).clone() {
            WitnessShape::Array(array_info, elem) => {
                WitnessShape::Array(array_info, Box::new(elem.join(elem_wt)))
            }
            other => panic!("Array element merge on non-array witness type: {:?}", other),
        };
        self.join_value(array, updated);
    }

    fn join_tuple_element(&mut self, tuple: ValueId, idx: usize, elem_wt: &WitnessShape) {
        let updated = match self.witness(tuple).clone() {
            WitnessShape::Tuple(tuple_info, mut children) => {
                children[idx] = children[idx].join(elem_wt);
                WitnessShape::Tuple(tuple_info, children)
            }
            other => panic!("Tuple element merge on non-tuple witness type: {:?}", other),
        };
        self.join_value(tuple, updated);
    }

    fn join_shapes(lhs: &[WitnessShape], rhs: &[WitnessShape]) -> Vec<WitnessShape> {
        assert_eq!(
            lhs.len(),
            rhs.len(),
            "Cannot join witness shape vectors of different lengths"
        );
        lhs.iter().zip(rhs.iter()).map(|(a, b)| a.join(b)).collect()
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
