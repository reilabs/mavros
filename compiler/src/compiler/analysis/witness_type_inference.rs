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

type VarShape = WitnessShape<VariableId>;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct VariableId(u32);

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct SpecKey {
    original_func_id: FunctionId,
    arg_types: Vec<WitnessShape>,
    cfg_witness: WitnessType,
}

#[derive(Clone, Debug)]
struct SpecInstance {
    original_func_id: FunctionId,
    specialized_func_id: FunctionId,
    entry_params: Vec<VarShape>,
    return_shapes: Vec<VarShape>,
    cfg_var: VariableId,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct CallSite {
    caller_func_id: FunctionId,
    block_id: BlockId,
    instruction_idx: usize,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
enum SpecRequest {
    Call(CallSite),
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
        let main_id = ssa.get_main_id();
        let main_func = ssa.get_function(main_id);
        let main_key = SpecKey {
            original_func_id: main_id,
            arg_types: main_func
                .get_entry()
                .get_parameters()
                .map(|(_, tp)| Self::construct_pure_witness_for_type(tp))
                .collect(),
            cfg_witness: WitnessType::Pure,
        };

        let mut engine = WtiEngine::new(ssa, flow_analysis);
        let root_spec = engine.ensure_spec(main_key);
        engine.run();
        self.functions = engine.finish(root_spec);

        Ok(())
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

    specs: Vec<SpecInstance>,
    specs_by_original: HashMap<FunctionId, Vec<usize>>,

    value_shapes: HashMap<ValueId, VarShape>,
    block_cfg_vars: HashMap<(FunctionId, BlockId), VariableId>,
    graph: HashMap<VariableId, Vec<VariableId>>,
    assignments: HashMap<VariableId, WitnessType>,
    spec_deps: HashMap<VariableId, Vec<SpecRequest>>,
    resolved_calls: HashMap<CallSite, usize>,

    spec_queue: VecDeque<SpecRequest>,
    queued_specs: HashSet<SpecRequest>,
    var_queue: VecDeque<VariableId>,
    queued_vars: HashSet<VariableId>,
    next_variable: u32,
}

impl<'a> WtiEngine<'a> {
    fn new(ssa: &'a mut HLSSA, flow_analysis: &'a FlowAnalysis) -> Self {
        let mut engine = Self {
            ssa,
            flow_analysis,
            specs: Vec::new(),
            specs_by_original: HashMap::new(),
            value_shapes: HashMap::new(),
            block_cfg_vars: HashMap::new(),
            graph: HashMap::new(),
            assignments: HashMap::new(),
            spec_deps: HashMap::new(),
            resolved_calls: HashMap::new(),
            spec_queue: VecDeque::new(),
            queued_specs: HashSet::new(),
            var_queue: VecDeque::new(),
            queued_vars: HashSet::new(),
            next_variable: 0,
        };

        let consts: Vec<_> = engine
            .ssa
            .const_snapshot()
            .iter()
            .map(|(vid, constant)| (*vid, constant.as_ref().clone()))
            .collect();
        for (vid, constant) in consts {
            let shape = match constant {
                Constant::U(_, _) | Constant::I(_, _) | Constant::Field(_) | Constant::FnPtr(_) => {
                    WitnessShape::Scalar(engine.fresh_var())
                }
            };
            engine.value_shapes.insert(vid, shape);
        }

        engine
    }

    fn run(&mut self) {
        while !self.var_queue.is_empty() || !self.spec_queue.is_empty() {
            while let Some(var) = self.var_queue.pop_front() {
                self.queued_vars.remove(&var);
                for dst in self.graph.get(&var).cloned().unwrap_or_default() {
                    self.mark_witness(dst);
                }
                for request in self.spec_deps.get(&var).cloned().unwrap_or_default() {
                    self.queue_spec(request);
                }
            }

            if let Some(request) = self.spec_queue.pop_front() {
                self.queued_specs.remove(&request);
                self.process_spec_request(request);
            }
        }
    }

    fn finish(mut self, root_spec: usize) -> HashMap<FunctionId, FunctionWitnessType> {
        let mut groups = HashMap::<SpecKey, Vec<usize>>::new();
        for idx in 0..self.specs.len() {
            groups.entry(self.current_key(idx)).or_default().push(idx);
        }

        let root_func = self.specs[root_spec].specialized_func_id;
        let mut canonical_by_func = HashMap::<FunctionId, FunctionId>::new();
        let mut canonical_by_key = HashMap::<SpecKey, FunctionId>::new();
        let mut surviving_specs = HashSet::<usize>::new();
        let mut duplicate_funcs = Vec::<FunctionId>::new();

        for (key, mut spec_indices) in groups {
            spec_indices.sort_by_key(|idx| self.specs[*idx].specialized_func_id.0);
            let representative = spec_indices
                .iter()
                .copied()
                .find(|idx| self.specs[*idx].specialized_func_id == root_func)
                .unwrap_or(spec_indices[0]);
            let representative_func = self.specs[representative].specialized_func_id;

            canonical_by_key.insert(key, representative_func);
            surviving_specs.insert(representative);
            for idx in spec_indices {
                let func_id = self.specs[idx].specialized_func_id;
                canonical_by_func.insert(func_id, representative_func);
                if idx != representative {
                    duplicate_funcs.push(func_id);
                }
            }
        }

        let rewritten_funcs: Vec<usize> = surviving_specs.iter().copied().collect();
        for spec_idx in rewritten_funcs {
            self.rewrite_calls(spec_idx, &canonical_by_key);
        }

        for func_id in duplicate_funcs {
            self.ssa.delete_function(func_id);
        }

        let canonical_root = canonical_by_func[&root_func];
        self.ssa.set_entry_point(canonical_root);

        let mut functions = HashMap::new();
        for spec_idx in surviving_specs {
            let spec = &self.specs[spec_idx];
            let function_id = spec.specialized_func_id;
            let key = self.current_key(spec_idx);
            let func = self.ssa.get_function(function_id);
            let block_order: Vec<BlockId> = self
                .flow_analysis
                .get_function_cfg(spec.original_func_id)
                .get_blocks_bfs()
                .collect();

            let mut block_cfg_witness = HashMap::new();
            let mut value_witness_types = HashMap::new();
            for block_id in block_order {
                let cfg_var = self.block_cfg_vars[&(function_id, block_id)];
                block_cfg_witness.insert(block_id, self.assignment(cfg_var));

                let block = func.get_block(block_id);
                for (value, _) in block.get_parameters() {
                    self.insert_concrete_value_shape(*value, &mut value_witness_types);
                }
                for instruction in block.get_instructions() {
                    for value in instruction.get_inputs().chain(instruction.get_results()) {
                        self.insert_concrete_value_shape(*value, &mut value_witness_types);
                    }
                }
                if let Some(terminator) = block.get_terminator() {
                    match terminator {
                        Terminator::Jmp(_, values) | Terminator::Return(values) => {
                            for value in values {
                                self.insert_concrete_value_shape(*value, &mut value_witness_types);
                            }
                        }
                        Terminator::JmpIf(cond, _, _) => {
                            self.insert_concrete_value_shape(*cond, &mut value_witness_types);
                        }
                    }
                }
            }

            functions.insert(
                function_id,
                FunctionWitnessType {
                    returns_witness: spec
                        .return_shapes
                        .iter()
                        .map(|shape| self.concretize_shape(shape))
                        .collect(),
                    cfg_witness: key.cfg_witness,
                    parameters: key.arg_types,
                    block_cfg_witness,
                    value_witness_types,
                },
            );
        }

        functions
    }

    fn ensure_spec(&mut self, key: SpecKey) -> usize {
        if let Some(indices) = self.specs_by_original.get(&key.original_func_id) {
            for idx in indices.clone() {
                if self.current_key(idx) == key {
                    return idx;
                }
            }
        }

        let specialized_func_id = self.ssa.duplicate_function(key.original_func_id);
        let entry_func = self.ssa.get_function(specialized_func_id);
        let entry_block = entry_func.get_entry_id();
        let entry_param_types: Vec<Type> = entry_func
            .get_entry()
            .get_parameters()
            .map(|(_, typ)| typ.clone())
            .collect();
        let entry_params: Vec<VarShape> = entry_param_types
            .iter()
            .map(|typ| self.fresh_shape_for_type(typ))
            .collect();
        let return_types = self
            .ssa
            .get_function(key.original_func_id)
            .get_returns()
            .to_vec();
        let return_shapes: Vec<VarShape> = return_types
            .iter()
            .map(|typ| self.fresh_shape_for_type(typ))
            .collect();
        let cfg_var = self.fresh_var();

        let spec_idx = self.specs.len();
        self.specs.push(SpecInstance {
            original_func_id: key.original_func_id,
            specialized_func_id,
            entry_params: entry_params.clone(),
            return_shapes,
            cfg_var,
        });
        self.specs_by_original
            .entry(key.original_func_id)
            .or_default()
            .push(spec_idx);

        for (shape, concrete) in entry_params.iter().zip(&key.arg_types) {
            self.seed_shape(shape, concrete);
        }
        if key.cfg_witness.is_witness() {
            self.mark_witness(cfg_var);
        }

        let block_order: Vec<BlockId> = self
            .flow_analysis
            .get_function_cfg(key.original_func_id)
            .get_blocks_bfs()
            .collect();
        for block_id in &block_order {
            let block_cfg = if *block_id == entry_block {
                cfg_var
            } else {
                self.fresh_var()
            };
            self.block_cfg_vars
                .insert((specialized_func_id, *block_id), block_cfg);
            self.add_edge(cfg_var, block_cfg);
        }

        self.scan_spec(spec_idx, block_order);
        spec_idx
    }

    fn scan_spec(&mut self, spec_idx: usize, block_order: Vec<BlockId>) {
        let spec = self.specs[spec_idx].clone();
        let entry_block = self
            .ssa
            .get_function(spec.specialized_func_id)
            .get_entry_id();
        let blocks: Vec<_> = {
            let func = self.ssa.get_function(spec.specialized_func_id);
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

        for (block_id, params, _, _) in &blocks {
            if *block_id == entry_block {
                for ((value, _), shape) in params.iter().zip(&spec.entry_params) {
                    self.value_shapes.insert(*value, shape.clone());
                }
            } else {
                for (value, typ) in params {
                    let shape = self.fresh_shape_for_type(typ);
                    self.value_shapes.insert(*value, shape);
                }
            }
        }

        for (block_id, _, instructions, terminator) in blocks {
            for (idx, instruction) in instructions.iter().enumerate() {
                self.scan_instruction(
                    CallSite {
                        caller_func_id: spec.specialized_func_id,
                        block_id,
                        instruction_idx: idx,
                    },
                    instruction,
                );
            }

            if let Some(terminator) = terminator {
                self.scan_terminator(spec_idx, block_id, &terminator);
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
                let result_var = self.fresh_var();
                self.insert_value_shape(*result, WitnessShape::Scalar(result_var));
                let lhs_top = self.shape(*lhs).toplevel_info();
                let rhs_top = self.shape(*rhs).toplevel_info();
                let result_top = self.shape(*result).toplevel_info();
                self.add_edge(lhs_top, result_top);
                self.add_edge(rhs_top, result_top);
            }
            OpCode::Select {
                result,
                cond,
                if_t,
                if_f,
            } => {
                let then_shape = self.shape(*if_t).clone();
                let else_shape = self.shape(*if_f).clone();
                let cond_top = self.shape(*cond).toplevel_info();
                let result_shape = self.fresh_like_shape(&then_shape);
                self.insert_value_shape(*result, result_shape);
                let result_shape = self.shape(*result).clone();
                self.flow_shape(&then_shape, &result_shape);
                self.flow_shape(&else_shape, &result_shape);
                self.add_edge(cond_top, result_shape.toplevel_info());
                self.flow_ref_positions(&result_shape, &then_shape);
                self.flow_ref_positions(&result_shape, &else_shape);
            }
            OpCode::Alloc { result, elem_type } => {
                let inner = self.fresh_shape_for_type(elem_type);
                let top = self.fresh_var();
                self.insert_value_shape(*result, WitnessShape::Ref(top, Box::new(inner)));
            }
            OpCode::Store { ptr, value } => {
                let block_cfg = self.block_cfg_var(call_site.caller_func_id, call_site.block_id);
                let ptr_top = self.shape(*ptr).toplevel_info();
                let ptr_inner = self.ref_inner(self.shape(*ptr));
                let value_shape = self.shape(*value).clone();
                self.add_edge(ptr_top, ptr_inner.toplevel_info());
                self.add_edge(block_cfg, ptr_inner.toplevel_info());
                self.flow_shape(&value_shape, &ptr_inner);
                self.flow_ref_positions(&ptr_inner, &value_shape);
            }
            OpCode::Load { result, ptr } => {
                let ptr_top = self.shape(*ptr).toplevel_info();
                let ptr_inner = self.ref_inner(self.shape(*ptr));
                let result_shape = self.fresh_like_shape(&ptr_inner);
                self.insert_value_shape(*result, result_shape);
                let result_shape = self.shape(*result).clone();
                self.add_edge(ptr_top, result_shape.toplevel_info());
                self.flow_shape(&ptr_inner, &result_shape);
                self.flow_ref_positions(&result_shape, &ptr_inner);
            }
            OpCode::ReadGlobal {
                result,
                result_type,
                ..
            } => {
                let shape = self.fresh_shape_for_type(result_type);
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
                let elem = self.array_element(self.shape(*array));
                let array_top = self.shape(*array).toplevel_info();
                let index_top = self.shape(*index).toplevel_info();
                let result_shape = self.fresh_like_shape(&elem);
                self.insert_value_shape(*result, result_shape);
                let result_shape = self.shape(*result).clone();
                self.flow_shape(&elem, &result_shape);
                self.flow_ref_positions(&result_shape, &elem);
                self.top_to_leaves(array_top, &result_shape);
                self.top_to_leaves(index_top, &result_shape);
            }
            OpCode::ArraySet {
                result,
                array,
                index,
                value,
            } => {
                let array_shape = self.shape(*array).clone();
                let array_elem = self.array_element(&array_shape);
                let result_shape = self.fresh_like_shape(&array_shape);
                self.insert_value_shape(*result, result_shape);
                let result_shape = self.shape(*result).clone();
                let result_elem = self.array_element(&result_shape);
                let value_shape = self.shape(*value).clone();
                let index_top = self.shape(*index).toplevel_info();
                self.flow_shape(&array_shape, &result_shape);
                self.flow_shape(&value_shape, &result_elem);
                self.top_to_leaves(index_top, &result_elem);
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
                let slice_elem = self.array_element(&slice_shape);
                let result_shape = self.fresh_like_shape(&slice_shape);
                self.insert_value_shape(*result, result_shape);
                let result_shape = self.shape(*result).clone();
                let result_elem = self.array_element(&result_shape);
                self.flow_shape(&slice_shape, &result_shape);
                for value in values {
                    let value_shape = self.shape(*value).clone();
                    self.flow_shape(&value_shape, &result_elem);
                    self.flow_ref_positions(&result_elem, &value_shape);
                }
                self.flow_ref_positions(&result_elem, &slice_elem);
            }
            OpCode::SliceLen { result, .. } => {
                let result_var = self.fresh_var();
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
                    let shape = self.fresh_shape_for_type(return_type);
                    self.insert_value_shape(*result, shape);
                }
                if !unconstrained {
                    let request = SpecRequest::Call(call_site);
                    for arg in args {
                        let arg_shape = self.shape(*arg).clone();
                        self.register_spec_dep_for_shape(&arg_shape, request.clone());
                    }
                    let cfg_var = self.block_cfg_var(call_site.caller_func_id, call_site.block_id);
                    self.register_spec_dep(cfg_var, request.clone());
                    self.queue_spec(request);
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
                let elem_shape = self.fresh_shape_for_type(elem_type);
                let top = self.fresh_var();
                self.insert_value_shape(*result, WitnessShape::Array(top, Box::new(elem_shape)));
                let result_elem = self.array_element(self.shape(*result));
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
                let elem_shape = self.fresh_shape_for_type(elem_type);
                let top = self.fresh_var();
                self.insert_value_shape(*result, WitnessShape::Array(top, Box::new(elem_shape)));
                let result_elem = self.array_element(self.shape(*result));
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
                let top = self.fresh_var();
                self.insert_value_shape(*result, WitnessShape::Array(top, Box::new(elem_shape)));
                let result_elem = self.array_element(self.shape(*result));
                self.flow_shape(&value_shape, &result_elem);
                self.flow_ref_positions(&result_elem, &value_shape);
            }
            OpCode::TupleProj { result, tuple, idx } => {
                let (tuple_top, child) = self.tuple_element(self.shape(*tuple), *idx);
                let result_shape = self.fresh_like_shape(&child);
                self.insert_value_shape(*result, result_shape);
                let result_shape = self.shape(*result).clone();
                self.add_edge(tuple_top, result_shape.toplevel_info());
                self.flow_shape(&child, &result_shape);
                self.flow_ref_positions(&result_shape, &child);
            }
            OpCode::MkTuple { result, elems, .. } => {
                let elem_shapes: Vec<_> = elems
                    .iter()
                    .map(|value| self.shape(*value).clone())
                    .collect();
                let children: Vec<_> = elem_shapes
                    .iter()
                    .map(|shape| self.fresh_like_shape(shape))
                    .collect();
                let top = self.fresh_var();
                self.insert_value_shape(*result, WitnessShape::Tuple(top, children));
                if let WitnessShape::Tuple(_, children) = self.shape(*result).clone() {
                    for (elem_shape, child) in elem_shapes.iter().zip(children.iter()) {
                        self.flow_shape(&elem_shape, child);
                        self.flow_ref_positions(child, &elem_shape);
                    }
                }
            }
            OpCode::WriteWitness { result, .. } => {
                if let Some(result) = result {
                    let var = self.fresh_var();
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

    fn scan_terminator(&mut self, spec_idx: usize, block_id: BlockId, terminator: &Terminator) {
        let spec = self.specs[spec_idx].clone();
        match terminator {
            Terminator::Return(values) => {
                for (value, return_shape) in values.iter().zip(&spec.return_shapes) {
                    let value_shape = self.shape(*value).clone();
                    self.flow_shape(&value_shape, return_shape);
                    self.flow_ref_positions(return_shape, &value_shape);
                }
            }
            Terminator::Jmp(target, params) => {
                let target_params: Vec<ValueId> = self
                    .ssa
                    .get_function(spec.specialized_func_id)
                    .get_block(*target)
                    .get_parameters()
                    .map(|(value, _)| *value)
                    .collect();
                for (arg, param) in params.iter().zip(target_params.iter()) {
                    let arg_shape = self.shape(*arg).clone();
                    let param_shape = self.shape(*param).clone();
                    self.flow_shape(&arg_shape, &param_shape);
                    self.flow_ref_positions(&param_shape, &arg_shape);
                }
            }
            Terminator::JmpIf(cond, _if_true, _if_false) => {
                let cond_top = self.shape(*cond).toplevel_info();
                let block_cfg = self.block_cfg_var(spec.specialized_func_id, block_id);
                let cfg = self.flow_analysis.get_function_cfg(spec.original_func_id);
                for body_block_id in cfg.get_if_body(block_id) {
                    let body_cfg = self.block_cfg_var(spec.specialized_func_id, body_block_id);
                    self.add_edge(cond_top, body_cfg);
                    self.add_edge(block_cfg, body_cfg);
                }

                let merge = cfg.get_merge_point(block_id);
                let merge_params: Vec<ValueId> = self
                    .ssa
                    .get_function(spec.specialized_func_id)
                    .get_block(merge)
                    .get_parameters()
                    .map(|(value, _)| *value)
                    .collect();
                for param in merge_params {
                    let param_shape = self.shape(param).clone();
                    self.top_to_leaves(cond_top, &param_shape);
                }
            }
        }
    }

    fn process_spec_request(&mut self, request: SpecRequest) {
        match request {
            SpecRequest::Call(call_site) => self.process_call_request(call_site),
        }
    }

    fn process_call_request(&mut self, call_site: CallSite) {
        let instruction = self
            .ssa
            .get_function(call_site.caller_func_id)
            .get_block(call_site.block_id)
            .get_instruction(call_site.instruction_idx)
            .clone();

        let OpCode::Call {
            results,
            function: CallTarget::Static(callee_id),
            args,
            unconstrained,
        } = instruction
        else {
            return;
        };
        if unconstrained {
            return;
        }

        let key = SpecKey {
            original_func_id: callee_id,
            arg_types: args
                .iter()
                .map(|arg| self.concretize_shape(self.shape(*arg)))
                .collect(),
            cfg_witness: self
                .assignment(self.block_cfg_var(call_site.caller_func_id, call_site.block_id)),
        };
        let callee_spec = self.ensure_spec(key);
        self.resolved_calls.insert(call_site, callee_spec);

        for (arg, param_shape) in args
            .iter()
            .zip(self.specs[callee_spec].entry_params.clone().iter())
        {
            let arg_shape = self.shape(*arg).clone();
            self.flow_shape(&arg_shape, param_shape);
            self.flow_ref_positions(param_shape, &arg_shape);
        }
        for (result, return_shape) in results
            .iter()
            .zip(self.specs[callee_spec].return_shapes.clone().iter())
        {
            let result_shape = self.shape(*result).clone();
            self.flow_shape(return_shape, &result_shape);
            self.flow_ref_positions(&result_shape, return_shape);
        }
    }

    fn rewrite_calls(&mut self, spec_idx: usize, canonical_by_key: &HashMap<SpecKey, FunctionId>) {
        let spec = self.specs[spec_idx].clone();
        let block_order: Vec<BlockId> = self
            .flow_analysis
            .get_function_cfg(spec.original_func_id)
            .get_blocks_bfs()
            .collect();
        let mut rewrites = HashMap::<(BlockId, usize), FunctionId>::new();

        for block_id in &block_order {
            let block_cfg =
                self.assignment(self.block_cfg_var(spec.specialized_func_id, *block_id));
            let block = self
                .ssa
                .get_function(spec.specialized_func_id)
                .get_block(*block_id);
            for (idx, instruction) in block.get_instructions().enumerate() {
                let OpCode::Call {
                    function: CallTarget::Static(callee_id),
                    args,
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
                    block_id: *block_id,
                    instruction_idx: idx,
                };
                let callee_spec = self
                    .resolved_calls
                    .get(&call_site)
                    .unwrap_or_else(|| panic!("Unresolved witness call site {:?}", call_site));
                let key = self.current_key(*callee_spec);
                assert_eq!(*callee_id, key.original_func_id);
                assert_eq!(block_cfg, key.cfg_witness);
                assert_eq!(args.len(), key.arg_types.len());
                let specialized = *canonical_by_key
                    .get(&key)
                    .unwrap_or_else(|| panic!("Missing witness specialization for {:?}", key));
                rewrites.insert((*block_id, idx), specialized);
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

    fn current_key(&self, spec_idx: usize) -> SpecKey {
        let spec = &self.specs[spec_idx];
        SpecKey {
            original_func_id: spec.original_func_id,
            arg_types: spec
                .entry_params
                .iter()
                .map(|shape| self.concretize_shape(shape))
                .collect(),
            cfg_witness: self.assignment(spec.cfg_var),
        }
    }

    fn insert_concrete_value_shape(
        &self,
        value: ValueId,
        output: &mut HashMap<ValueId, WitnessShape>,
    ) {
        if let Some(shape) = self.value_shapes.get(&value) {
            output.insert(value, self.concretize_shape(shape));
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

    fn block_cfg_var(&self, function_id: FunctionId, block_id: BlockId) -> VariableId {
        *self
            .block_cfg_vars
            .get(&(function_id, block_id))
            .unwrap_or_else(|| {
                panic!(
                    "Missing block cfg variable for {:?} {:?}",
                    function_id, block_id
                )
            })
    }

    fn fresh_var(&mut self) -> VariableId {
        let var = VariableId(self.next_variable);
        self.next_variable += 1;
        self.assignments.insert(var, WitnessType::Pure);
        var
    }

    fn assignment(&self, var: VariableId) -> WitnessType {
        *self
            .assignments
            .get(&var)
            .unwrap_or_else(|| panic!("Missing assignment for {:?}", var))
    }

    fn mark_witness(&mut self, var: VariableId) {
        if self.assignment(var).is_witness() {
            return;
        }
        self.assignments.insert(var, WitnessType::Witness);
        if self.queued_vars.insert(var) {
            self.var_queue.push_back(var);
        }
    }

    fn add_edge(&mut self, source: VariableId, target: VariableId) {
        let edges = self.graph.entry(source).or_default();
        if edges.contains(&target) {
            return;
        }
        edges.push(target);
        if self.assignment(source).is_witness() {
            self.mark_witness(target);
        }
    }

    fn queue_spec(&mut self, request: SpecRequest) {
        if self.queued_specs.insert(request.clone()) {
            self.spec_queue.push_back(request);
        }
    }

    fn register_spec_dep(&mut self, var: VariableId, request: SpecRequest) {
        let deps = self.spec_deps.entry(var).or_default();
        if !deps.contains(&request) {
            deps.push(request);
        }
    }

    fn register_spec_dep_for_shape(&mut self, shape: &VarShape, request: SpecRequest) {
        for var in Self::shape_vars(shape) {
            self.register_spec_dep(var, request.clone());
        }
    }

    fn seed_shape(&mut self, shape: &VarShape, concrete: &WitnessShape) {
        match (shape, concrete) {
            (WitnessShape::Scalar(var), WitnessShape::Scalar(wt)) => {
                if wt.is_witness() {
                    self.mark_witness(*var);
                }
            }
            (WitnessShape::Array(var, inner), WitnessShape::Array(wt, concrete_inner))
            | (WitnessShape::Ref(var, inner), WitnessShape::Ref(wt, concrete_inner)) => {
                if wt.is_witness() {
                    self.mark_witness(*var);
                }
                self.seed_shape(inner, concrete_inner);
            }
            (WitnessShape::Tuple(var, children), WitnessShape::Tuple(wt, concrete_children)) => {
                if wt.is_witness() {
                    self.mark_witness(*var);
                }
                for (child, concrete_child) in children.iter().zip(concrete_children.iter()) {
                    self.seed_shape(child, concrete_child);
                }
            }
            _ => panic!(
                "Cannot seed mismatched witness shapes: {:?} vs {:?}",
                shape, concrete
            ),
        }
    }

    fn fresh_shape_for_type(&mut self, typ: &Type) -> VarShape {
        match &typ.expr {
            TypeExpr::U(_) | TypeExpr::I(_) | TypeExpr::Field => {
                WitnessShape::Scalar(self.fresh_var())
            }
            TypeExpr::Array(inner, _) | TypeExpr::Slice(inner) => {
                let top = self.fresh_var();
                let inner = self.fresh_shape_for_type(inner);
                WitnessShape::Array(top, Box::new(inner))
            }
            TypeExpr::Ref(inner) => {
                let top = self.fresh_var();
                let inner = self.fresh_shape_for_type(inner);
                WitnessShape::Ref(top, Box::new(inner))
            }
            TypeExpr::Tuple(elements) => {
                let top = self.fresh_var();
                let children = elements
                    .iter()
                    .map(|typ| self.fresh_shape_for_type(typ))
                    .collect();
                WitnessShape::Tuple(top, children)
            }
            TypeExpr::WitnessOf(_) => {
                panic!("ICE: WitnessOf should not be present at this stage");
            }
            TypeExpr::Function => WitnessShape::Scalar(self.fresh_var()),
        }
    }

    fn fresh_like_shape(&mut self, shape: &VarShape) -> VarShape {
        match shape {
            WitnessShape::Scalar(_) => WitnessShape::Scalar(self.fresh_var()),
            WitnessShape::Array(_, inner) => {
                let top = self.fresh_var();
                let inner = self.fresh_like_shape(inner);
                WitnessShape::Array(top, Box::new(inner))
            }
            WitnessShape::Ref(_, inner) => {
                let top = self.fresh_var();
                let inner = self.fresh_like_shape(inner);
                WitnessShape::Ref(top, Box::new(inner))
            }
            WitnessShape::Tuple(_, children) => {
                let top = self.fresh_var();
                let children = children
                    .iter()
                    .map(|child| self.fresh_like_shape(child))
                    .collect();
                WitnessShape::Tuple(top, children)
            }
        }
    }

    fn concretize_shape(&self, shape: &VarShape) -> WitnessShape {
        match shape {
            WitnessShape::Scalar(var) => WitnessShape::Scalar(self.assignment(*var)),
            WitnessShape::Array(var, inner) => WitnessShape::Array(
                self.assignment(*var),
                Box::new(self.concretize_shape(inner)),
            ),
            WitnessShape::Ref(var, inner) => WitnessShape::Ref(
                self.assignment(*var),
                Box::new(self.concretize_shape(inner)),
            ),
            WitnessShape::Tuple(var, children) => WitnessShape::Tuple(
                self.assignment(*var),
                children
                    .iter()
                    .map(|child| self.concretize_shape(child))
                    .collect(),
            ),
        }
    }

    fn flow_shape(&mut self, source: &VarShape, target: &VarShape) {
        match (source, target) {
            (WitnessShape::Scalar(source), WitnessShape::Scalar(target)) => {
                self.add_edge(*source, *target);
            }
            (
                WitnessShape::Array(source, source_inner),
                WitnessShape::Array(target, target_inner),
            )
            | (WitnessShape::Ref(source, source_inner), WitnessShape::Ref(target, target_inner)) => {
                self.add_edge(*source, *target);
                self.flow_shape(source_inner, target_inner);
            }
            (
                WitnessShape::Tuple(source, source_children),
                WitnessShape::Tuple(target, target_children),
            ) => {
                self.add_edge(*source, *target);
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
                self.add_edge(source, *target);
            }
            WitnessShape::Array(_, inner) => self.top_to_leaves(source, inner),
            WitnessShape::Tuple(_, children) => {
                for child in children {
                    self.top_to_leaves(source, child);
                }
            }
        }
    }

    fn shape_vars(shape: &VarShape) -> Vec<VariableId> {
        let mut vars = Vec::new();
        Self::collect_shape_vars(shape, &mut vars);
        vars
    }

    fn collect_shape_vars(shape: &VarShape, vars: &mut Vec<VariableId>) {
        match shape {
            WitnessShape::Scalar(var) => vars.push(*var),
            WitnessShape::Array(var, inner) | WitnessShape::Ref(var, inner) => {
                vars.push(*var);
                Self::collect_shape_vars(inner, vars);
            }
            WitnessShape::Tuple(var, children) => {
                vars.push(*var);
                for child in children {
                    Self::collect_shape_vars(child, vars);
                }
            }
        }
    }

    fn array_element(&self, shape: &VarShape) -> VarShape {
        match shape {
            WitnessShape::Array(_, inner) => *inner.clone(),
            other => panic!(
                "Array element access on non-array witness shape: {:?}",
                other
            ),
        }
    }

    fn ref_inner(&self, shape: &VarShape) -> VarShape {
        match shape {
            WitnessShape::Ref(_, inner) => *inner.clone(),
            other => panic!("Ref inner access on non-ref witness shape: {:?}", other),
        }
    }

    fn tuple_element(&self, shape: &VarShape, idx: usize) -> (VariableId, VarShape) {
        match shape {
            WitnessShape::Tuple(top, children) => (*top, children[idx].clone()),
            other => panic!(
                "Tuple element access on non-tuple witness shape: {:?}",
                other
            ),
        }
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
