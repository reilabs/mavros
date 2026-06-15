//! Computes the set of SSA values that are live-in and live-out for each block in the SSA using the
//! conventional, block-based approach.

use itertools::Itertools;
use std::collections::VecDeque;
use tracing::{Level, instrument, trace};

use crate::{
    collections::{HashMap, HashSet},
    compiler::{
        analysis::flow_analysis::{CFG, FlowAnalysis},
        ssa::{
            BlockId, FunctionId, Instruction, Terminator, ValueId,
            hlssa::{HLFunction, HLSSA},
        },
    },
};

pub enum InstructionPosition {
    Offset(usize),
    Terminator,
}

pub struct InstructionPointer {
    pub position: InstructionPosition,
    pub block: BlockId,
}

pub struct BlockLiveness {
    pub live_in: HashSet<ValueId>,
    pub live_out: HashSet<ValueId>,
}

pub struct FunctionLiveness {
    pub block_liveness: HashMap<BlockId, BlockLiveness>,
}

pub struct Liveness {
    pub function_liveness: HashMap<FunctionId, FunctionLiveness>,
}

pub struct LivenessAnalysis {}

impl LivenessAnalysis {
    pub fn new() -> Self {
        Self {}
    }

    #[instrument(skip_all, name = "LivenessAnalysis::run")]
    pub fn run(&self, ssa: &HLSSA, cfg: &FlowAnalysis) -> Liveness {
        let mut result = Liveness {
            function_liveness: HashMap::default(),
        };

        for (function_id, function) in ssa.iter_functions() {
            trace!("Function {}", function.get_name());
            let function_liveness = self.run_function(function, cfg.get_function_cfg(*function_id));
            result
                .function_liveness
                .insert(*function_id, function_liveness);
        }

        result
    }

    #[instrument(skip_all, level = Level::TRACE, name = "LivenessAnalysis::run_function", fields(function = function.get_name()))]
    fn run_function(&self, function: &HLFunction, cfg: &CFG) -> FunctionLiveness {
        let mut gens = HashMap::<BlockId, HashSet<ValueId>>::default();
        let mut kills = HashMap::<BlockId, HashSet<ValueId>>::default();

        for (block_id, block) in function.get_blocks() {
            let mut k = HashSet::default();
            let mut g = HashSet::default();
            match block.get_terminator().unwrap() {
                Terminator::Return(vs) => {
                    for v in vs {
                        g.insert(*v);
                    }
                }
                Terminator::Jmp(_, params) => {
                    for v in params {
                        g.insert(*v);
                    }
                }
                Terminator::JmpIf(cond, _, _) => {
                    g.insert(*cond);
                }
            }

            for instr in block.get_instructions().rev() {
                for value_id in instr.get_inputs() {
                    g.insert(*value_id);
                }
                for value_id in instr.get_results() {
                    k.insert(*value_id);
                    g.remove(value_id);
                }
            }

            for (value_id, _) in block.get_parameters() {
                k.insert(*value_id);
            }

            gens.insert(*block_id, g);
            kills.insert(*block_id, k);
        }

        let mut result = HashMap::<BlockId, BlockLiveness>::default();
        let mut queue = VecDeque::new();

        for ret in cfg.get_return_blocks() {
            queue.push_back(ret);
        }

        while let Some(block_id) = queue.pop_front() {
            let visited = result.contains_key(&block_id);
            result.entry(block_id).or_insert(BlockLiveness {
                live_in: HashSet::default(),
                live_out: HashSet::default(),
            });
            let original_live_in = &result.get(&block_id).unwrap().live_in;

            let mut new_live_out: HashSet<ValueId> = HashSet::default();

            for block_id in cfg.get_successors(block_id) {
                new_live_out.extend(
                    &result
                        .get(&block_id)
                        .unwrap_or(&BlockLiveness {
                            live_in: HashSet::default(),
                            live_out: HashSet::default(),
                        })
                        .live_in,
                );
            }

            let mut new_live_in = new_live_out.clone();
            new_live_in.extend(gens.get(&block_id).unwrap_or(&HashSet::default()));
            let kills = kills.get(&block_id).unwrap();
            new_live_in.retain(|v| !kills.contains(v));

            if !visited || original_live_in != &new_live_in {
                for pred in cfg.get_predecessors(block_id) {
                    queue.push_back(pred);
                }
            }

            result.insert(
                block_id,
                BlockLiveness {
                    live_in: new_live_in,
                    live_out: new_live_out,
                },
            );
        }

        for (block_id, block_liveness) in result.iter().sorted_by_key(|(block_id, _)| block_id.0) {
            trace!("block {}", block_id.0);
            trace!(
                "  live_in: {}",
                block_liveness
                    .live_in
                    .iter()
                    .map(|v| v.0.to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            );
            trace!(
                "  live_out: {}",
                block_liveness
                    .live_out
                    .iter()
                    .map(|v| v.0.to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            );
        }

        FunctionLiveness {
            block_liveness: result,
        }
    }
}
