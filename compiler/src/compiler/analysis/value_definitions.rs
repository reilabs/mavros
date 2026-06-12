//! Builds a mapping from ever SSA `ValueId` back to the place at which it is defined on a
//! per-function basis.
//!
//! This is mainly consumed by the simplifier, avoiding the need to perform an `O(n)` scan of every
//! block to find a value's defining instruction.
//!
//! Only *structural* definitions (block parameters and instruction results) are recorded here.
//! Constants are deliberately excluded: they live in the SSA's interior-mutable constants pool,
//! which is always up to date (including values minted mid-pass), so callers resolve them live via
//! [`crate::compiler::ssa::SSA::get_const`] rather than from a stale snapshot.

use crate::{
    collections::HashMap,
    compiler::ssa::{
        BlockId, Instruction, ValueId,
        hlssa::{HLFunction, OpCode, Type},
    },
};

pub enum ValueDefinition {
    Param(BlockId, usize, Type),
    Instruction(BlockId, usize, OpCode),
}

pub struct FunctionValueDefinitions {
    definitions: HashMap<ValueId, ValueDefinition>,
}

impl FunctionValueDefinitions {
    pub fn new() -> Self {
        Self {
            definitions: HashMap::default(),
        }
    }

    /// Looks up the structural definition of a value, or `None` if it has none (e.g. a constant,
    /// or a value minted after this snapshot was taken).
    pub fn get_definition(&self, value_id: ValueId) -> Option<&ValueDefinition> {
        self.definitions.get(&value_id)
    }

    pub fn insert(&mut self, value_id: ValueId, definition: ValueDefinition) {
        self.definitions.insert(value_id, definition);
    }

    pub fn from_function(function: &HLFunction) -> Self {
        let mut definitions = Self::new();

        for (block_id, block) in function.get_blocks() {
            for (i, (val, typ)) in block.get_parameters().enumerate() {
                definitions.insert(*val, ValueDefinition::Param(*block_id, i, typ.clone()));
            }

            for (i, instruction) in block.get_instructions().enumerate() {
                for val in instruction.get_results() {
                    definitions.insert(
                        *val,
                        ValueDefinition::Instruction(*block_id, i, instruction.clone()),
                    );
                }
            }
        }

        definitions
    }
}
