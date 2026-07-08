//! A map of value replacements applied as a rewrite sweep over a function.

use crate::{
    collections::HashMap,
    compiler::ssa::{
        Instruction, Terminator, ValueId,
        hlssa::{HLFunction, OpCode},
    },
};

// VALUE REPLACEMENTS
// ================================================================================================

pub struct ValueReplacements {
    replacements: HashMap<ValueId, ValueId>,
}

impl ValueReplacements {
    pub fn new() -> Self {
        Self {
            replacements: HashMap::default(),
        }
    }

    pub fn insert(&mut self, replaced: ValueId, replacement: ValueId) {
        self.replacements.insert(replaced, replacement);
    }

    pub fn is_empty(&self) -> bool {
        self.replacements.is_empty()
    }

    pub fn replace_instruction(&self, instruction: &mut OpCode) {
        for operand in instruction.get_operands_mut() {
            *operand = self.get_replacement(*operand);
        }
    }

    pub fn replace_inputs(&self, instruction: &mut OpCode) {
        for input in instruction.get_inputs_mut() {
            *input = self.get_replacement(*input);
        }
    }

    pub fn replace_terminator(&self, terminator: &mut Terminator) {
        match terminator {
            Terminator::Jmp(_, params) => {
                for param in params {
                    *param = self.get_replacement(*param);
                }
            }
            Terminator::JmpIf(cond, _, _) => {
                *cond = self.get_replacement(*cond);
            }
            Terminator::Return(vals) => {
                for val in vals {
                    *val = self.get_replacement(*val);
                }
            }
        }
    }

    pub fn get_replacement(&self, value: ValueId) -> ValueId {
        let mut current = value;
        for _ in 0..=self.replacements.len() {
            match self.replacements.get(&current) {
                Some(&next) if next != current => current = next,
                _ => return current,
            }
        }
        panic!("ValueReplacements: cycle starting at {:?}", value)
    }

    /// Walks every instruction and terminator in `function`, applying the collected replacements.
    ///
    /// `scope` selects whether instruction results are rewritten alongside inputs.
    pub fn apply_to_function(&self, function: &mut HLFunction, scope: ReplaceScope) {
        self.apply_to_function_where(function, scope, |_| true);
    }

    /// [`Self::apply_to_function`] restricted to the instructions `keep` admits; terminators are
    /// always rewritten.
    ///
    /// This exists for replacement maps that must skip a class of consumers — e.g. SCS's
    /// anticipated aliases, which must never rewrite an `Assert`/`AssertCmp` input.
    pub fn apply_to_function_where(
        &self,
        function: &mut HLFunction,
        scope: ReplaceScope,
        keep: impl Fn(&OpCode) -> bool,
    ) {
        for (_, block) in function.get_blocks_mut() {
            for instr in block.get_instructions_mut() {
                if !keep(instr) {
                    continue;
                }
                match scope {
                    ReplaceScope::Inputs => self.replace_inputs(instr),
                    ReplaceScope::Operands => self.replace_instruction(instr),
                }
            }
            self.replace_terminator(block.get_terminator_mut());
        }
    }
}

// REPLACE SCOPE
// ================================================================================================

/// Selects which value references a [`ValueReplacements`] sweep rewrites.
pub enum ReplaceScope {
    /// Only instruction _inputs_ (the values an instruction reads).
    Inputs,

    /// All instruction _operands_, both inputs and results.
    Operands,
}
