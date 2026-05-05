use std::collections::HashMap;

use crate::compiler::{
    ir::r#type::Type,
    ssa::{BlockId, FunctionId, HLFunction, HLSSA, Instruction, OpCode, ValueId},
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
            definitions: HashMap::new(),
        }
    }

    pub fn get_definition(&self, value_id: ValueId) -> &ValueDefinition {
        self.definitions.get(&value_id).unwrap()
    }

    pub fn try_get_definition(&self, value_id: ValueId) -> Option<&ValueDefinition> {
        self.definitions.get(&value_id)
    }

    pub fn insert(&mut self, value_id: ValueId, definition: ValueDefinition) {
        self.definitions.insert(value_id, definition);
    }

    pub fn from_ssa(ssa: &HLFunction) -> Self {
        let mut definitions = Self::new();

        for (block_id, block) in ssa.get_blocks() {
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

pub struct ValueDefinitions {
    functions: HashMap<FunctionId, FunctionValueDefinitions>,
}

impl ValueDefinitions {
    pub fn new() -> Self {
        Self {
            functions: HashMap::new(),
        }
    }

    pub fn from_ssa(ssa: &HLSSA) -> Self {
        let mut definitions = Self::new();

        for (function_id, function) in ssa.iter_functions() {
            definitions
                .functions
                .insert(*function_id, FunctionValueDefinitions::from_ssa(function));
        }

        definitions
    }

    pub fn get_function(&self, function_id: FunctionId) -> &FunctionValueDefinitions {
        self.functions.get(&function_id).unwrap()
    }
}

use crate::compiler::pass_manager::{Analysis, AnalysisStore};

impl Analysis for ValueDefinitions {
    fn compute(ssa: &HLSSA, _store: &AnalysisStore) -> Self {
        ValueDefinitions::from_ssa(ssa)
    }
}
