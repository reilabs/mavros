//! Contains the traits necessary for working with the SSA.

use std::fmt::{Debug, Display};

use crate::compiler::ssa::{BlockId, FunctionId, ValueId};

/// Functionality that enables the attaching of arbitrary annotations to the SSA.
pub trait SSAAnotator: Debug {
    fn annotate_value(&self, _function_id: FunctionId, _value_id: ValueId) -> String {
        "".to_string()
    }
    fn annotate_function(&self, _function_id: FunctionId) -> String {
        "".to_string()
    }
    fn annotate_block(&self, _function_id: FunctionId, _block_id: BlockId) -> String {
        "".to_string()
    }
}

/// Functionality that each instruction needs to provide to work within the bounds of the SSA.
pub trait Instruction: Clone + Debug + 'static {
    fn get_inputs(&self) -> impl Iterator<Item = &ValueId>;
    fn get_results(&self) -> impl Iterator<Item = &ValueId>;
    fn get_inputs_mut(&mut self) -> impl Iterator<Item = &mut ValueId>;
    fn get_operands_mut(&mut self) -> impl Iterator<Item = &mut ValueId>;

    /// Gets the call targets that are known statically, predominantly to aid in building graphs.
    fn get_static_call_targets(&self) -> Vec<FunctionId>;

    /// Display an instruction as a string.
    ///
    /// Takes closures for function name resolution and value annotation (so the trait doesn't
    /// depend on the SSA).
    fn display_instruction(
        &self,
        func_name: &dyn Fn(FunctionId) -> String,
        annotate_value: &dyn Fn(ValueId) -> String,
    ) -> String;
}

/// The type of values in a given SSA.
pub trait SSAType: Clone + Debug + Display + PartialEq + Eq + 'static {}
