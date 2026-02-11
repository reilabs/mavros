use std::collections::{HashMap, HashSet, VecDeque};

use tracing::{debug, instrument, Level};

use crate::compiler::analysis::types::{FunctionTypeInfo, TypeInfo};
use crate::compiler::passes::fix_double_jumps::ValueReplacements;
use crate::compiler::{
    flow_analysis::{CFG, FlowAnalysis},
    ir::r#type::{Type, TypeExpr},
    pass_manager::{DataPoint, Pass, PassInfo, PassManager},
    ssa::{BlockId, Function, OpCode, SSA, Terminator, ValueId},
};

pub struct Mem2Reg {}

impl Pass for Mem2Reg {
    fn run(&self, ssa: &mut SSA, pass_manager: &PassManager) {
        self.do_run(ssa, pass_manager.get_cfg(), pass_manager.get_type_info());
    }

    fn pass_info(&self) -> PassInfo {
        PassInfo {
            name: "mem2reg",
            needs: vec![DataPoint::CFG, DataPoint::Types],
        }
    }
}

impl Mem2Reg {
    pub fn new() -> Self {
        Self {}
    }

    pub fn do_run(&self, ssa: &mut SSA, cfg: &FlowAnalysis, type_info: &TypeInfo) {
        for (function_id, function) in ssa.iter_functions_mut() {
            let function_type_info = type_info.get_function(*function_id);
            if self.escape_safe(function, function_type_info) {
                self.run_function(function, cfg.get_function_cfg(*function_id), function_type_info);
            } else {
                debug!(
                    "Skipping mem2reg for function: {:?} because it failed escape analysis",
                    function.get_name()
                );
            }
        }
    }

    #[instrument(skip_all, level = Level::DEBUG, fields(function = %function.get_name()))]
    fn run_function(&self, function: &mut Function, cfg: &CFG, type_info: &FunctionTypeInfo) {
        let (writes, defs) = self.find_pointer_writes_and_defs(function);
        let phi_blocks = self.find_phi_blocks(&writes, &defs, cfg);
        let phi_args = self.initialize_phis(function, &phi_blocks, type_info);
        self.remove_ptrs(function, cfg, &phi_args);
    }

    fn remove_ptrs(
        &self,
        function: &mut Function,
        cfg: &CFG,
        phi_map: &HashMap<BlockId, Vec<(ValueId, ValueId)>>,
    ) {
        let mut ptr_values = HashMap::<BlockId, HashMap<ValueId, ValueId>>::new();
        let mut value_replacements = ValueReplacements::new();

        // Traverse the CFG in domination pre-order
        // This ensures that for each ptr the last value is already defined when we enter the block:
        // either it is defined by some dominator, or it is a phi parameter.
        for block_id in cfg.get_domination_pre_order() {
            // fetch all possible values from the parent block
            let mut values = if let Some(parent) = cfg.get_immediate_dominator(block_id) {
                if let Some(parent_values) = ptr_values.get(&parent) {
                    parent_values.clone()
                } else {
                    HashMap::new()
                }
            } else {
                HashMap::new()
            };

            // add phi parameters
            for (param, ptr) in phi_map.get(&block_id).unwrap_or(&vec![]) {
                values.insert(*ptr, *param);
            }

            let instructions = function.get_block_mut(block_id).take_instructions();
            let mut new_instructions = Vec::new();

            for mut instruction in instructions {
                match instruction {
                    OpCode::Alloc { result: _, elem_type: _ } => {}
                    OpCode::Store { ptr: lhs, value: rhs } => {
                        values.insert(lhs, rhs);
                    }
                    OpCode::Load { result: lhs, ptr: rhs } => {
                        let replacement = values.get(&rhs).expect("Uninitialized ptr value");
                        value_replacements.insert(lhs, *replacement);
                    }
                    _ => {
                        value_replacements.replace_instruction(&mut instruction);
                        new_instructions.push(instruction);
                    }
                }
            }

            function
                .get_block_mut(block_id)
                .put_instructions(new_instructions);

            let mut terminator = function.get_block_mut(block_id).take_terminator().unwrap();
            value_replacements.replace_terminator(&mut terminator);

            match &mut terminator {
                Terminator::Jmp(tgt, params) => {
                    let tmp = vec![];
                    let additional_params = phi_map.get(tgt).unwrap_or(&tmp);
                    for (_, val) in additional_params {
                        let param_val = values.get(val).expect(&format!(
                            "ICE: block {} has no value for v{}",
                            block_id.0, val.0
                        ));
                        params.push(value_replacements.get_replacement(*param_val));
                    }
                }
                Terminator::JmpIf(_cond, t1, t2) => {
                    if phi_map.contains_key(t1) {
                        let jumper = function.add_block();
                        let params = phi_map
                            .get(t1)
                            .unwrap()
                            .iter()
                            .map(|(_, val)| {
                                let v = *values.get(val).expect(&format!(
                                    "ICE: block {} has no value for v{}",
                                    block_id.0, val.0
                                ));
                                value_replacements.get_replacement(v)
                            })
                            .collect::<Vec<_>>();
                        function.terminate_block_with_jmp(jumper, *t1, params);
                        *t1 = jumper;
                    }
                    if phi_map.contains_key(t2) {
                        let jumper = function.add_block();
                        let params = phi_map
                            .get(t2)
                            .unwrap()
                            .iter()
                            .map(|(_, val)| {
                                let v = *values.get(val).expect(&format!(
                                    "ICE: block {} has no value for v{}",
                                    block_id.0, val.0
                                ));
                                value_replacements.get_replacement(v)
                            })
                            .collect::<Vec<_>>();
                        function.terminate_block_with_jmp(jumper, *t2, params);
                        *t2 = jumper;
                    }
                }
                _ => {}
            }
            function.get_block_mut(block_id).set_terminator(terminator);
            ptr_values.insert(block_id, values);
        }
    }

    // returns for each block the vector of (param_id, value_id), where param_id is the id of a new parameter,
    // and value_id is the id of the pointer that is being replaced
    fn initialize_phis(
        &self,
        function: &mut Function,
        phi_blocks: &HashMap<ValueId, HashSet<BlockId>>,
        type_info: &FunctionTypeInfo,
    ) -> HashMap<BlockId, Vec<(ValueId, ValueId)>> {
        let mut result: HashMap<BlockId, Vec<(ValueId, ValueId)>> = HashMap::new();
        for (value, blocks) in phi_blocks {
            for block in blocks {
                let param = function.add_parameter(
                    *block,
                    type_info.get_value_type(*value).get_pointed(),
                );
                result.entry(*block).or_default().push((param, *value));
            }
        }
        result
    }

    fn find_pointer_writes_and_defs(
        &self,
        function: &Function,
    ) -> (HashMap<ValueId, HashSet<BlockId>>, HashMap<ValueId, BlockId>) {
        let mut writes: HashMap<ValueId, HashSet<BlockId>> = HashMap::new();
        let mut defs: HashMap<ValueId, BlockId> = HashMap::new();
        for (block_id, block) in function.get_blocks() {
            for instruction in block.get_instructions() {
                match instruction {
                    OpCode::Store { ptr: lhs, value: _ } => {
                        writes.entry(*lhs).or_default().insert(*block_id);
                    }
                    OpCode::Alloc { result: lhs, elem_type: _ } => {
                        defs.insert(*lhs, *block_id);
                    }
                    _ => {}
                }
            }
        }
        (writes, defs)
    }

    fn find_phi_blocks(
        &self,
        writes: &HashMap<ValueId, HashSet<BlockId>>,
        defs: &HashMap<ValueId, BlockId>,
        cfg: &CFG,
    ) -> HashMap<ValueId, HashSet<BlockId>> {
        let mut result: HashMap<ValueId, HashSet<BlockId>> = HashMap::new();

        for (var, writes) in writes {
            let mut queue = VecDeque::<BlockId>::new();
            let mut visited = HashSet::<BlockId>::new();
            queue.extend(writes);

            while let Some(block) = queue.pop_front() {
                if visited.contains(&block) {
                    continue;
                }
                visited.insert(block);

                for new_block in cfg.get_dominance_frontier(block) {
                    if !cfg.dominates(*defs.get(var).unwrap(), new_block) {
                        continue; // If the pointer is not defined here, there's no need for a phi.
                    }
                    debug!(
                        "Block {}\tneeds phi for v{}\tbecause it's in the dominance frontier of {}\twhich contains a write",
                        new_block.0, var.0, block.0
                    );
                    result.entry(*var).or_default().insert(new_block);
                    queue.push_back(new_block);
                }
            }
        }

        result
    }

    // This is _very_ crude. We give up on mem2reg for the entire function
    // if we detect _any_ pointer escaping or entering the function, or being
    // written to another pointer. Obviously this needs a better implementation.
    fn escape_safe(&self, function: &Function, type_info: &FunctionTypeInfo) -> bool {
        for (_, block) in function.get_blocks() {
            for (_, typ) in block.get_parameters() {
                if self.type_contains_ptr(typ) {
                    return false;
                }
            }

            for instruction in block.get_instructions() {
                match instruction {
                    OpCode::ArraySet { result: _, array: _, index: _, value: val } => {
                        let vtyp = type_info.get_value_type(*val);
                        if self.type_contains_ptr(vtyp) {
                            return false;
                        }
                    }
                    OpCode::SlicePush { result: _, slice: _, values: vals, dir: _ } => {
                        for val in vals {
                            let vtyp = type_info.get_value_type(*val);
                            if self.type_contains_ptr(vtyp) {
                                return false;
                            }
                        }
                    }
                    OpCode::ArrayGet { result: _, array: _, index: val } => {
                        let vtyp = type_info.get_value_type(*val);
                        if self.type_contains_ptr(vtyp) {
                            return false;
                        }
                    }
                    OpCode::Store { ptr: _, value: r } => {
                        let rtyp = type_info.get_value_type(*r);
                        if self.type_contains_ptr(rtyp) {
                            return false;
                        }
                    }
                    OpCode::MkSeq { result: _, elems: _, seq_type: _, elem_type: typ } => {
                        if self.type_contains_ptr(typ) {
                            return false;
                        }
                    }
                    OpCode::Load { result: r, ptr: _ } => {
                        let rtyp = type_info.get_value_type(*r);
                        if self.type_contains_ptr(rtyp) {
                            return false;
                        }
                    }
                    OpCode::Call { results: rets, function: _, args } => {
                        for v in rets.iter().chain(args.iter()) {
                            let vtyp = type_info.get_value_type(*v);
                            if self.type_contains_ptr(vtyp) {
                                return false;
                            }
                        }
                    }
                    _ => {}
                }
            }

            match block.get_terminator() {
                Some(Terminator::Return(vals)) => {
                    for v in vals {
                        let vtyp = type_info.get_value_type(*v);
                        if self.type_contains_ptr(vtyp) {
                            return false;
                        }
                    }
                }
                _ => {}
            }
        }

        true
    }

    fn type_contains_ptr(&self, typ: &Type) -> bool {
        match typ {
            Type {
                expr: TypeExpr::Ref(_),
                ..
            } => true,
            Type {
                expr: TypeExpr::Array(inner, _),
                ..
            } => self.type_contains_ptr(inner),
            _ => false,
        }
    }
}
