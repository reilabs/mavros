use std::collections::{HashMap, HashSet};

use crate::compiler::{
    flow_analysis::{CFG, FlowAnalysis},
    pass_manager::Pass,
    ssa::{BlockId, Function, OpCode, SSA, Terminator, ValueId},
};

pub struct DCE {
    config: Config,
}

#[derive(Debug)]
enum WorkItem {
    LiveBlock(BlockId),
    LiveValue(ValueId),
    LiveInstruction(BlockId, usize),
}

enum ValueDefinition {
    Param(BlockId, usize),
    Instruction(BlockId, usize),
    Const(ValueId),
}

pub struct Config {
    pub witness_shape_frozen: bool,
    /// When true, all blocks are marked as live, preventing removal of empty intermediate blocks.
    /// This is a workaround for untaint_control_flow not handling multiple merge predecessors.
    /// TODO: Remove this option once untaint_control_flow properly handles multiple jumps into merge blocks.
    pub preserve_all_blocks: bool,
}

impl Config {
    pub fn pre_r1c() -> Self {
        Self {
            witness_shape_frozen: false,
            preserve_all_blocks: false,
        }
    }

    pub fn post_r1c() -> Self {
        Self {
            witness_shape_frozen: true,
            preserve_all_blocks: false,
        }
    }

    pub fn preserve_blocks() -> Self {
        Self {
            witness_shape_frozen: false,
            preserve_all_blocks: true,
        }
    }
}

impl<V: Clone> Pass<V> for DCE {
    fn run(&self, ssa: &mut SSA<V>, pass_manager: &crate::compiler::pass_manager::PassManager<V>) {
        self.do_run(ssa, pass_manager.get_cfg());
    }

    fn pass_info(&self) -> crate::compiler::pass_manager::PassInfo {
        crate::compiler::pass_manager::PassInfo {
            name: "dce",
            needs: vec![crate::compiler::pass_manager::DataPoint::CFG],
        }
    }
}

impl DCE {
    pub fn new(config: Config) -> Self {
        Self { config }
    }

    pub fn do_run<V: Clone>(&self, ssa: &mut SSA<V>, cfg: &FlowAnalysis) {
        for (function_id, function) in ssa.iter_functions_mut() {
            let cfg = cfg.get_function_cfg(*function_id);
            let definitions = self.generate_definitions(function);

            let mut live_values: HashSet<ValueId> = HashSet::new();
            let mut live_blocks: HashSet<BlockId> = HashSet::new();
            let mut live_instructions: HashMap<BlockId, HashSet<usize>> = HashMap::new();
            let mut live_params: HashMap<BlockId, HashSet<usize>> = HashMap::new();
            let mut live_branches: HashSet<BlockId> = HashSet::new();
            let mut live_consts: HashSet<ValueId> = HashSet::new();

            let mut worklist: Vec<WorkItem> = vec![];

            for (parameter, _) in function.get_entry().get_parameters() {
                // All function parameters are live
                worklist.push(WorkItem::LiveValue(*parameter));
            }

            // Workaround: mark all blocks as live to preserve empty intermediate blocks.
            // TODO: Remove once untaint_control_flow handles multiple jumps into merge blocks.
            if self.config.preserve_all_blocks {
                for (block_id, _) in function.get_blocks() {
                    worklist.push(WorkItem::LiveBlock(*block_id));
                }
            }

            for (block_id, block) in function.get_blocks() {
                for (i, instruction) in block.get_instructions().enumerate() {
                    match instruction {
                        // calls, stores, constraints and witness stores are critical side-effects
                        OpCode::Call { .. } | OpCode::Store { .. } => {
                            worklist.push(WorkItem::LiveInstruction(*block_id, i));
                        }
                        // assert_eq is critical if it's not definitionally true
                        OpCode::AssertEq { lhs, rhs } => {
                            if lhs != rhs {
                                worklist.push(WorkItem::LiveInstruction(*block_id, i));
                            }
                        }
                        OpCode::AssertR1C { a: _, b: _, c: _ } => {
                            worklist.push(WorkItem::LiveInstruction(*block_id, i));
                        }
                        OpCode::Constrain { .. } => {
                            worklist.push(WorkItem::LiveInstruction(*block_id, i));
                        }
                        OpCode::Lookup { .. } => {
                            worklist.push(WorkItem::LiveInstruction(*block_id, i));
                        }
                        OpCode::DLookup { .. } => {
                            worklist.push(WorkItem::LiveInstruction(*block_id, i));
                        }
                        OpCode::NextDCoeff { result: _ } => {
                            // This also has the side-effect of bumping the counter, so we need to keep it live.
                            worklist.push(WorkItem::LiveInstruction(*block_id, i));
                        }
                        OpCode::BumpD {
                            matrix: _,
                            variable: _,
                            sensitivity: _,
                        } => {
                            worklist.push(WorkItem::LiveInstruction(*block_id, i));
                        }
                        OpCode::WriteWitness { .. } => {
                            // Witness stores are critical after the constraint system is generated.
                            // Previously, they only matter if the result is used.
                            if self.config.witness_shape_frozen {
                                worklist.push(WorkItem::LiveInstruction(*block_id, i));
                            }
                        }
                        OpCode::FreshWitness {
                            result: _,
                            result_type: _,
                        } => {
                            if self.config.witness_shape_frozen {
                                worklist.push(WorkItem::LiveInstruction(*block_id, i));
                            }
                        }
                        OpCode::MemOp { kind: _, value: _ } => {
                            // TODO: this is over-conservative - accidentally alives the value
                            // even if it's not used otherwise. Should be fixed in the future,
                            // but we're inserting these late in the pipeline, so main passes
                            // we'll be fine.
                            worklist.push(WorkItem::LiveInstruction(*block_id, i));
                        }
                        OpCode::Rangecheck {
                            value: _,
                            max_bits: _,
                        } => {
                            worklist.push(WorkItem::LiveInstruction(*block_id, i));
                        }
                        OpCode::ToBits { .. } | OpCode::ToRadix { .. } => {
                            // These are live while we still have witgen and R1CS in the same program
                            // because they can be used as range checks.
                            // After the witness shape is frozen, they can be removed, because the
                            // rangechecks are made explicit by `ExplicitWitness`.
                            if !self.config.witness_shape_frozen {
                                worklist.push(WorkItem::LiveInstruction(*block_id, i));
                            }
                        }
                        OpCode::Todo { .. } => {
                            worklist.push(WorkItem::LiveInstruction(*block_id, i));
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
                        | OpCode::ReadGlobal {
                            result: _,
                            offset: _,
                            result_type: _,
                        }
                        | OpCode::MkTuple { .. } => {},
                    }
                }

                if let Some(terminator) = block.get_terminator() {
                    match terminator {
                        Terminator::Return(values) => {
                            worklist.push(WorkItem::LiveBlock(*block_id));
                            for value in values {
                                worklist.push(WorkItem::LiveValue(*value));
                            }
                        }
                        _ => {}
                    }
                }
            }

            while let Some(item) = worklist.pop() {
                match item {
                    WorkItem::LiveBlock(block_id) => {
                        if live_blocks.contains(&block_id) {
                            continue;
                        }
                        live_blocks.insert(block_id);

                        // All blocks in the post-domination frontier are live,
                        // because they decide whether or not this block is entered.
                        // Also, the condition on which they enter is live.
                        for pd in cfg.get_post_dominance_frontier(block_id) {
                            worklist.push(WorkItem::LiveBlock(pd));
                            live_branches.insert(pd);

                            match function.get_block(pd).get_terminator() {
                                Some(Terminator::JmpIf(condition, _, _)) => {
                                    worklist.push(WorkItem::LiveValue(*condition));
                                }
                                _ => panic!("ICE: It's a frontier, must end with a conditional"),
                            }
                        }

                        // If the block has parameters, all predecessor become
                        // live, as they define the phi inputs.
                        if function.get_block(block_id).has_parameters() {
                            for predecessor in cfg.get_jumps_into(block_id) {
                                worklist.push(WorkItem::LiveBlock(predecessor));
                            }
                        }
                    }
                    WorkItem::LiveValue(value_id) => {
                        if live_values.contains(&value_id) {
                            continue;
                        }
                        live_values.insert(value_id);

                        let definition = definitions.get(&value_id).unwrap();
                        match definition {
                            ValueDefinition::Param(block_id, i) => {
                                if live_params
                                    .get(block_id)
                                    .unwrap_or(&HashSet::new())
                                    .contains(i)
                                {
                                    continue;
                                }
                                live_params
                                    .entry(*block_id)
                                    .or_insert(HashSet::new())
                                    .insert(*i);

                                // The defining block is live
                                worklist.push(WorkItem::LiveBlock(*block_id));
                                // For all jumps into it, the respective argument is live
                                for block_id in cfg.get_jumps_into(*block_id) {
                                    let jumpin_block = function.get_block(block_id);
                                    match jumpin_block.get_terminator() {
                                        Some(Terminator::Jmp(_, params)) => {
                                            worklist.push(WorkItem::LiveValue(params[*i]));
                                        }
                                        _ => panic!(
                                            "ICE: the block has phis, so jumps into it must be Jmps"
                                        ),
                                    }
                                }
                            }
                            ValueDefinition::Instruction(block_id, i) => {
                                // The instruction is live
                                worklist.push(WorkItem::LiveInstruction(*block_id, *i));
                            }
                            ValueDefinition::Const(value_id) => {
                                live_consts.insert(*value_id);
                            }
                        }
                    }
                    WorkItem::LiveInstruction(block_id, i) => {
                        if live_instructions
                            .get(&block_id)
                            .unwrap_or(&HashSet::new())
                            .contains(&i)
                        {
                            continue;
                        }
                        live_instructions
                            .entry(block_id)
                            .or_insert(HashSet::new())
                            .insert(i);

                        // The containing block is live
                        worklist.push(WorkItem::LiveBlock(block_id));

                        let instruction = function.get_block(block_id).get_instruction(i);
                        for input in instruction.get_inputs() {
                            worklist.push(WorkItem::LiveValue(*input));
                        }
                    }
                }
            }

            for value_id in function.iter_consts().map(|(v, _)| *v).collect::<Vec<_>>() {
                if !live_consts.contains(&value_id) {
                    function.remove_const(value_id);
                }
            }

            for block_id in cfg.get_domination_pre_order() {
                let mut block = function.take_block(block_id);
                if !live_blocks.contains(&block_id) {
                    continue;
                }

                let instructions = block.take_instructions();
                let mut new_instructions = vec![];

                for (i, instruction) in instructions.into_iter().enumerate() {
                    if live_instructions
                        .get(&block_id)
                        .unwrap_or(&HashSet::new())
                        .contains(&i)
                    {
                        new_instructions.push(instruction);
                    }
                }

                block.put_instructions(new_instructions);

                let new_terminator = match block.take_terminator() {
                    Some(Terminator::Jmp(target, params)) => {
                        if live_blocks.contains(&target) {
                            let mut new_params = vec![];
                            for (i, param) in params.into_iter().enumerate() {
                                if live_params
                                    .get(&target)
                                    .unwrap_or(&HashSet::new())
                                    .contains(&i)
                                {
                                    new_params.push(param);
                                }
                            }
                            Terminator::Jmp(target, new_params)
                        } else {
                            let new_target =
                                self.closest_live_post_dominator(cfg, block_id, &live_blocks);
                            // we can safely jump without params – blocks with params are post-dominated
                            // by their phi-setting blocks, so we wouldn't find one here.
                            Terminator::Jmp(new_target, vec![])
                        }
                    }
                    Some(Terminator::JmpIf(condition, then, otherwise)) => {
                        if live_branches.contains(&block_id) {
                            Terminator::JmpIf(
                                condition,
                                self.closest_live_block(cfg, then, &live_blocks),
                                self.closest_live_block(cfg, otherwise, &live_blocks),
                            )
                        } else {
                            Terminator::Jmp(
                                self.closest_live_post_dominator(cfg, block_id, &live_blocks),
                                // Again, the jump is parameterless – live blocks with params are
                                // post-dominated by the phi-setting blocks which are all live, but this
                                // block ended with a conditional jump, so it wasn't setting phis.
                                vec![],
                            )
                        }
                    }
                    Some(Terminator::Return(values)) => {
                        let mut new_values = vec![];
                        for value in values.into_iter() {
                            if live_values.contains(&value) {
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
                    if live_params
                        .get(&block_id)
                        .unwrap_or(&HashSet::new())
                        .contains(&i)
                    {
                        new_params.push(param);
                    }
                }
                block.put_parameters(new_params);

                function.put_block(block_id, block);
            }
        }
    }

    fn generate_definitions<V: Clone>(
        &self,
        ssa: &Function<V>,
    ) -> HashMap<ValueId, ValueDefinition> {
        let mut definitions = HashMap::new();

        for (value_id, _) in ssa.iter_consts() {
            definitions.insert(*value_id, ValueDefinition::Const(*value_id));
        }

        for (block_id, block) in ssa.get_blocks() {
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

    fn closest_live_block(
        &self,
        cfg: &CFG,
        block_id: BlockId,
        live_blocks: &HashSet<BlockId>,
    ) -> BlockId {
        if live_blocks.contains(&block_id) {
            return block_id;
        }
        return self.closest_live_post_dominator(cfg, block_id, live_blocks);
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
