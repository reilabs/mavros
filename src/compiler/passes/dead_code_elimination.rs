use std::collections::{HashMap, HashSet};

use crate::compiler::{
    flow_analysis::{CFG, FlowAnalysis},
    pass_manager::{Analysis, AnalysisId, AnalysisStore, Pass},
    ssa::{
        BlockId, CallTarget, FunctionId, HLFunction, HLSSA, Instruction, OpCode, Terminator,
        ValueId,
    },
};

pub struct DCE {
    config: Config,
}

#[derive(Debug)]
enum WorkItem {
    LiveBlock(FunctionId, BlockId),
    LiveValue(FunctionId, ValueId),
    LiveInstruction(FunctionId, BlockId, usize),
    LiveReturnSlot(FunctionId, usize),
}

enum ValueDefinition {
    Param(BlockId, usize),
    Instruction(BlockId, usize),
}

pub struct Config {
    pub witness_shape_frozen: bool,
    /// When true, all blocks are marked as live, preventing removal of empty intermediate blocks.
    /// This is a workaround for untaint_control_flow not handling multiple merge predecessors.
    /// TODO: Remove this option once untaint_control_flow properly handles multiple jumps into merge blocks.
    pub preserve_all_blocks: bool,
    /// When true, the main function's entry block params are always kept alive.
    /// Used in the witgen pipeline where the runtime writes inputs to the frame
    /// based on the ABI, so the frame must always have room for all declared inputs.
    pub preserve_main_entry_params: bool,
}

impl Config {
    pub fn pre_r1c() -> Self {
        Self {
            witness_shape_frozen: false,
            preserve_all_blocks: false,
            preserve_main_entry_params: true,
        }
    }

    pub fn post_r1c() -> Self {
        Self {
            witness_shape_frozen: true,
            preserve_all_blocks: false,
            preserve_main_entry_params: false,
        }
    }

    pub fn witgen() -> Self {
        Self {
            witness_shape_frozen: true,
            preserve_all_blocks: false,
            preserve_main_entry_params: true,
        }
    }

    pub fn preserve_blocks() -> Self {
        Self {
            witness_shape_frozen: false,
            preserve_all_blocks: true,
            preserve_main_entry_params: true,
        }
    }
}

impl Pass for DCE {
    fn name(&self) -> &'static str {
        "dce"
    }

    fn needs(&self) -> Vec<AnalysisId> {
        vec![FlowAnalysis::id()]
    }

    fn run(&self, ssa: &mut HLSSA, store: &AnalysisStore) {
        self.do_run(ssa, store.get::<FlowAnalysis>());
    }
}

impl DCE {
    pub fn new(config: Config) -> Self {
        Self { config }
    }

    pub fn do_run(&self, ssa: &mut HLSSA, cfg: &FlowAnalysis) {
        let main_id = ssa.get_main_id();
        let function_ids: Vec<FunctionId> = ssa.get_function_ids().collect();

        let mut definitions_by_function: HashMap<FunctionId, HashMap<ValueId, ValueDefinition>> =
            HashMap::new();
        let mut static_calls_by_callee: HashMap<FunctionId, Vec<(FunctionId, BlockId, usize)>> =
            HashMap::new();

        for function_id in &function_ids {
            let function = ssa.get_function(*function_id);
            definitions_by_function.insert(*function_id, self.generate_definitions(function));

            for (block_id, block) in function.get_blocks() {
                for (i, instruction) in block.get_instructions().enumerate() {
                    if let OpCode::Call {
                        function: CallTarget::Static(callee),
                        ..
                    } = instruction
                    {
                        static_calls_by_callee.entry(*callee).or_default().push((
                            *function_id,
                            *block_id,
                            i,
                        ));
                    }
                }
            }
        }

        let mut live_values: HashMap<FunctionId, HashSet<ValueId>> = HashMap::new();
        let mut live_blocks: HashMap<FunctionId, HashSet<BlockId>> = HashMap::new();
        let mut live_instructions: HashMap<FunctionId, HashMap<BlockId, HashSet<usize>>> =
            HashMap::new();
        let mut live_params: HashMap<FunctionId, HashMap<BlockId, HashSet<usize>>> = HashMap::new();
        let mut live_entry_params: HashMap<FunctionId, HashSet<usize>> = HashMap::new();
        let mut live_branches: HashMap<FunctionId, HashSet<BlockId>> = HashMap::new();
        let mut live_return_slots: HashMap<FunctionId, HashSet<usize>> = HashMap::new();

        let mut worklist: Vec<WorkItem> = vec![];

        for function_id in &function_ids {
            let function = ssa.get_function(*function_id);
            worklist.push(WorkItem::LiveBlock(*function_id, function.get_entry_id()));

            // In the witgen pipeline, the main function's entry params are externally
            // provided by the runtime and must always be considered live.
            if self.config.preserve_main_entry_params && *function_id == main_id {
                let entry_block = function.get_block(function.get_entry_id());
                for (_i, (val, _ty)) in entry_block.get_parameters().enumerate() {
                    worklist.push(WorkItem::LiveValue(*function_id, *val));
                }
            }

            if self.config.preserve_all_blocks {
                for (block_id, _) in function.get_blocks() {
                    worklist.push(WorkItem::LiveBlock(*function_id, *block_id));
                }
            }

            for (block_id, block) in function.get_blocks() {
                for (i, instruction) in block.get_instructions().enumerate() {
                    match instruction {
                        OpCode::Call {
                            unconstrained: true,
                            ..
                        } => {
                            // Unconstrained calls are NOT initially live — DCE'd when results unused
                        }
                        OpCode::Call { .. } | OpCode::Store { .. } => {
                            worklist.push(WorkItem::LiveInstruction(*function_id, *block_id, i));
                        }
                        OpCode::AssertEq { lhs, rhs } => {
                            if lhs != rhs {
                                worklist.push(WorkItem::LiveInstruction(
                                    *function_id,
                                    *block_id,
                                    i,
                                ));
                            }
                        }
                        OpCode::AssertR1C { .. }
                        | OpCode::Constrain { .. }
                        | OpCode::Lookup { .. }
                        | OpCode::DLookup { .. }
                        | OpCode::NextDCoeff { .. }
                        | OpCode::BumpD { .. }
                        | OpCode::MemOp { .. }
                        | OpCode::Rangecheck { .. }
                        | OpCode::Todo { .. }
                        | OpCode::InitGlobal { .. }
                        | OpCode::DropGlobal { .. } => {
                            worklist.push(WorkItem::LiveInstruction(*function_id, *block_id, i));
                        }
                        OpCode::WriteWitness { pinned, .. } => {
                            if self.config.witness_shape_frozen || *pinned {
                                worklist.push(WorkItem::LiveInstruction(
                                    *function_id,
                                    *block_id,
                                    i,
                                ));
                            }
                        }
                        OpCode::FreshWitness { .. } => {
                            if self.config.witness_shape_frozen {
                                worklist.push(WorkItem::LiveInstruction(
                                    *function_id,
                                    *block_id,
                                    i,
                                ));
                            }
                        }
                        OpCode::ToBits { .. } | OpCode::ToRadix { .. } => {
                            if !self.config.witness_shape_frozen {
                                worklist.push(WorkItem::LiveInstruction(
                                    *function_id,
                                    *block_id,
                                    i,
                                ));
                            }
                        }
                        OpCode::Load { .. }
                        | OpCode::BinaryArithOp { .. }
                        | OpCode::Cmp { .. }
                        | OpCode::Alloc { .. }
                        | OpCode::Select { .. }
                        | OpCode::ArrayGet { .. }
                        | OpCode::ArraySet { .. }
                        | OpCode::TupleProj { .. }
                        | OpCode::SlicePush { .. }
                        | OpCode::SliceLen { .. }
                        | OpCode::MkSeq { .. }
                        | OpCode::Cast { .. }
                        | OpCode::Truncate { .. }
                        | OpCode::Not { .. }
                        | OpCode::MulConst { .. }
                        | OpCode::ReadGlobal { .. }
                        | OpCode::MkTuple { .. }
                        | OpCode::ValueOf { .. }
                        | OpCode::Const { .. } => {}
                    }
                }

                if let Some(Terminator::Return(values)) = block.get_terminator() {
                    worklist.push(WorkItem::LiveBlock(*function_id, *block_id));

                    if *function_id == main_id {
                        for (i, value) in values.iter().enumerate() {
                            worklist.push(WorkItem::LiveReturnSlot(*function_id, i));
                            worklist.push(WorkItem::LiveValue(*function_id, *value));
                        }
                    }
                }
            }
        }

        while let Some(item) = worklist.pop() {
            match item {
                WorkItem::LiveBlock(function_id, block_id) => {
                    if self.block_live(&live_blocks, function_id, block_id) {
                        continue;
                    }
                    live_blocks.entry(function_id).or_default().insert(block_id);

                    let function_cfg = cfg.get_function_cfg(function_id);
                    let function = ssa.get_function(function_id);

                    if let Some(Terminator::JmpIf(condition, _, _)) =
                        function.get_block(block_id).get_terminator()
                    {
                        worklist.push(WorkItem::LiveValue(function_id, *condition));
                    }

                    for pd in function_cfg.get_post_dominance_frontier(block_id) {
                        worklist.push(WorkItem::LiveBlock(function_id, pd));
                        live_branches.entry(function_id).or_default().insert(pd);

                        match function.get_block(pd).get_terminator() {
                            Some(Terminator::JmpIf(condition, _, _)) => {
                                worklist.push(WorkItem::LiveValue(function_id, *condition));
                            }
                            _ => panic!("ICE: It's a frontier, must end with a conditional"),
                        }
                    }

                    if function.get_block(block_id).has_parameters() {
                        for predecessor in function_cfg.get_jumps_into(block_id) {
                            worklist.push(WorkItem::LiveBlock(function_id, predecessor));
                        }
                    }
                }
                WorkItem::LiveValue(function_id, value_id) => {
                    if live_values
                        .entry(function_id)
                        .or_default()
                        .contains(&value_id)
                    {
                        continue;
                    }
                    live_values.entry(function_id).or_default().insert(value_id);

                    let definitions = definitions_by_function
                        .get(&function_id)
                        .expect("function definitions missing");
                    let Some(definition) = definitions.get(&value_id) else {
                        continue;
                    };

                    match definition {
                        ValueDefinition::Param(block_id, i) => {
                            if self.param_live(&live_params, function_id, *block_id, *i) {
                                continue;
                            }

                            live_params
                                .entry(function_id)
                                .or_default()
                                .entry(*block_id)
                                .or_default()
                                .insert(*i);

                            let function = ssa.get_function(function_id);
                            if *block_id == function.get_entry_id() {
                                if live_entry_params.entry(function_id).or_default().insert(*i) {
                                    if let Some(callsites) =
                                        static_calls_by_callee.get(&function_id)
                                    {
                                        for (caller_fn, caller_block, caller_i) in callsites {
                                            let caller = ssa.get_function(*caller_fn);
                                            if let OpCode::Call { args, .. } = caller
                                                .get_block(*caller_block)
                                                .get_instruction(*caller_i)
                                            {
                                                assert!(
                                                    *i < args.len(),
                                                    "ICE: live callee entry param index out of bounds at callsite"
                                                );
                                                worklist.push(WorkItem::LiveValue(
                                                    *caller_fn, args[*i],
                                                ));
                                            }
                                        }
                                    }
                                }
                            }

                            worklist.push(WorkItem::LiveBlock(function_id, *block_id));

                            let function_cfg = cfg.get_function_cfg(function_id);
                            for pred in function_cfg.get_jumps_into(*block_id) {
                                let jumpin_block = function.get_block(pred);
                                match jumpin_block.get_terminator() {
                                    Some(Terminator::Jmp(_, params)) => {
                                        assert!(
                                            *i < params.len(),
                                            "ICE: phi param index out of bounds in predecessor jump"
                                        );
                                        worklist.push(WorkItem::LiveValue(function_id, params[*i]));
                                    }
                                    _ => panic!(
                                        "ICE: the block has phis, so jumps into it must be Jmps"
                                    ),
                                }
                            }
                        }
                        ValueDefinition::Instruction(block_id, i) => {
                            let function = ssa.get_function(function_id);
                            let instruction = function.get_block(*block_id).get_instruction(*i);

                            if let OpCode::Call {
                                results,
                                function: CallTarget::Static(callee),
                                ..
                            } = instruction
                            {
                                if let Some(result_idx) =
                                    results.iter().position(|result| *result == value_id)
                                {
                                    worklist.push(WorkItem::LiveReturnSlot(*callee, result_idx));
                                }
                            }

                            worklist.push(WorkItem::LiveInstruction(function_id, *block_id, *i));
                        }
                    }
                }
                WorkItem::LiveInstruction(function_id, block_id, i) => {
                    if self.instruction_live(&live_instructions, function_id, block_id, i) {
                        continue;
                    }

                    live_instructions
                        .entry(function_id)
                        .or_default()
                        .entry(block_id)
                        .or_default()
                        .insert(i);

                    worklist.push(WorkItem::LiveBlock(function_id, block_id));

                    let function = ssa.get_function(function_id);
                    let instruction = function.get_block(block_id).get_instruction(i);
                    for input in instruction.get_inputs() {
                        worklist.push(WorkItem::LiveValue(function_id, *input));
                    }
                }
                WorkItem::LiveReturnSlot(function_id, slot) => {
                    if !live_return_slots
                        .entry(function_id)
                        .or_default()
                        .insert(slot)
                    {
                        continue;
                    }

                    let function = ssa.get_function(function_id);
                    for (block_id, block) in function.get_blocks() {
                        if let Some(Terminator::Return(values)) = block.get_terminator() {
                            assert!(
                                slot < values.len(),
                                "ICE: return slot index out of bounds for return terminator"
                            );
                            worklist.push(WorkItem::LiveBlock(function_id, *block_id));
                            worklist.push(WorkItem::LiveValue(function_id, values[slot]));
                        }
                    }
                }
            }
        }

        for function_id in function_ids {
            let function_cfg = cfg.get_function_cfg(function_id);
            let mut function = ssa.take_function(function_id);
            let entry_id = function.get_entry_id();

            for block_id in function_cfg.get_domination_pre_order() {
                let mut block = function.take_block(block_id);
                if !self.block_live(&live_blocks, function_id, block_id) {
                    continue;
                }

                let instructions = block.take_instructions();
                let mut new_instructions = vec![];

                for (i, mut instruction) in instructions.into_iter().enumerate() {
                    if !self.instruction_live(&live_instructions, function_id, block_id, i) {
                        continue;
                    }

                    if let OpCode::Call {
                        results,
                        function: CallTarget::Static(callee),
                        args,
                        unconstrained: _,
                    } = &mut instruction
                    {
                        let mut new_args = vec![];
                        for (arg_i, arg) in args.iter().enumerate() {
                            if self.entry_param_live(&live_entry_params, *callee, arg_i) {
                                new_args.push(*arg);
                            }
                        }
                        *args = new_args;

                        let mut new_results = vec![];
                        for (ret_i, result) in results.iter().enumerate() {
                            if self.return_slot_live(&live_return_slots, *callee, ret_i) {
                                new_results.push(*result);
                            }
                        }
                        *results = new_results;
                    }

                    new_instructions.push(instruction);
                }

                block.put_instructions(new_instructions);

                let new_terminator = match block.take_terminator() {
                    Some(Terminator::Jmp(target, params)) => {
                        if self.block_live(&live_blocks, function_id, target) {
                            let mut new_params = vec![];
                            for (i, param) in params.into_iter().enumerate() {
                                if self.block_param_live(
                                    &live_params,
                                    &live_entry_params,
                                    function_id,
                                    entry_id,
                                    target,
                                    i,
                                ) {
                                    new_params.push(param);
                                }
                            }
                            Terminator::Jmp(target, new_params)
                        } else {
                            let new_target = self.closest_live_post_dominator(
                                function_cfg,
                                block_id,
                                live_blocks.get(&function_id).unwrap_or(&HashSet::new()),
                            );
                            Terminator::Jmp(new_target, vec![])
                        }
                    }
                    Some(Terminator::JmpIf(condition, then, otherwise)) => {
                        if live_branches
                            .get(&function_id)
                            .unwrap_or(&HashSet::new())
                            .contains(&block_id)
                        {
                            Terminator::JmpIf(
                                condition,
                                self.closest_live_block(
                                    function_cfg,
                                    then,
                                    live_blocks.get(&function_id).unwrap_or(&HashSet::new()),
                                ),
                                self.closest_live_block(
                                    function_cfg,
                                    otherwise,
                                    live_blocks.get(&function_id).unwrap_or(&HashSet::new()),
                                ),
                            )
                        } else {
                            Terminator::Jmp(
                                self.closest_live_post_dominator(
                                    function_cfg,
                                    block_id,
                                    live_blocks.get(&function_id).unwrap_or(&HashSet::new()),
                                ),
                                vec![],
                            )
                        }
                    }
                    Some(Terminator::Return(values)) => {
                        let mut new_values = vec![];
                        for (i, value) in values.into_iter().enumerate() {
                            if self.return_slot_live(&live_return_slots, function_id, i) {
                                new_values.push(value);
                            }
                        }
                        Terminator::Return(new_values)
                    }
                    None => panic!("ICE: block has no terminator"),
                };

                block.set_terminator(new_terminator);

                let params = block.take_parameters();
                let mut new_params = vec![];
                for (i, param) in params.into_iter().enumerate() {
                    if self.block_param_live(
                        &live_params,
                        &live_entry_params,
                        function_id,
                        entry_id,
                        block_id,
                        i,
                    ) {
                        new_params.push(param);
                    }
                }
                block.put_parameters(new_params);

                function.put_block(block_id, block);
            }

            let old_returns = function.take_returns();
            for (i, return_type) in old_returns.into_iter().enumerate() {
                if self.return_slot_live(&live_return_slots, function_id, i) {
                    function.add_return_type(return_type);
                }
            }

            ssa.put_function(function_id, function);
        }
    }

    fn generate_definitions(&self, function: &HLFunction) -> HashMap<ValueId, ValueDefinition> {
        let mut definitions = HashMap::new();

        for (block_id, block) in function.get_blocks() {
            for (i, (val, _)) in block.get_parameters().enumerate() {
                definitions.insert(*val, ValueDefinition::Param(*block_id, i));
            }

            for (i, instruction) in block.get_instructions().enumerate() {
                for val in instruction.get_results() {
                    definitions.insert(*val, ValueDefinition::Instruction(*block_id, i));
                }
            }
        }

        definitions
    }

    fn block_live(
        &self,
        live_blocks: &HashMap<FunctionId, HashSet<BlockId>>,
        function_id: FunctionId,
        block_id: BlockId,
    ) -> bool {
        live_blocks
            .get(&function_id)
            .unwrap_or(&HashSet::new())
            .contains(&block_id)
    }

    fn instruction_live(
        &self,
        live_instructions: &HashMap<FunctionId, HashMap<BlockId, HashSet<usize>>>,
        function_id: FunctionId,
        block_id: BlockId,
        i: usize,
    ) -> bool {
        live_instructions
            .get(&function_id)
            .and_then(|blocks| blocks.get(&block_id))
            .unwrap_or(&HashSet::new())
            .contains(&i)
    }

    fn param_live(
        &self,
        live_params: &HashMap<FunctionId, HashMap<BlockId, HashSet<usize>>>,
        function_id: FunctionId,
        block_id: BlockId,
        i: usize,
    ) -> bool {
        live_params
            .get(&function_id)
            .and_then(|blocks| blocks.get(&block_id))
            .unwrap_or(&HashSet::new())
            .contains(&i)
    }

    fn entry_param_live(
        &self,
        live_entry_params: &HashMap<FunctionId, HashSet<usize>>,
        function_id: FunctionId,
        i: usize,
    ) -> bool {
        live_entry_params
            .get(&function_id)
            .unwrap_or(&HashSet::new())
            .contains(&i)
    }

    fn return_slot_live(
        &self,
        live_return_slots: &HashMap<FunctionId, HashSet<usize>>,
        function_id: FunctionId,
        i: usize,
    ) -> bool {
        live_return_slots
            .get(&function_id)
            .unwrap_or(&HashSet::new())
            .contains(&i)
    }

    fn block_param_live(
        &self,
        live_params: &HashMap<FunctionId, HashMap<BlockId, HashSet<usize>>>,
        live_entry_params: &HashMap<FunctionId, HashSet<usize>>,
        function_id: FunctionId,
        entry_id: BlockId,
        block_id: BlockId,
        i: usize,
    ) -> bool {
        if block_id == entry_id {
            return self.entry_param_live(live_entry_params, function_id, i);
        }
        self.param_live(live_params, function_id, block_id, i)
    }

    fn closest_live_block(
        &self,
        cfg: &CFG,
        block_id: BlockId,
        live_blocks: &HashSet<BlockId>,
    ) -> BlockId {
        if live_blocks.contains(&block_id) {
            return block_id;
        }
        self.closest_live_post_dominator(cfg, block_id, live_blocks)
    }

    fn closest_live_post_dominator(
        &self,
        cfg: &CFG,
        block_id: BlockId,
        live_blocks: &HashSet<BlockId>,
    ) -> BlockId {
        let mut current_block = cfg.get_post_dominator(block_id);
        while !live_blocks.contains(&current_block) {
            current_block = cfg.get_post_dominator(current_block);
        }
        current_block
    }
}
