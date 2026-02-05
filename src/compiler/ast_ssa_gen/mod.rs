//! Direct AST to SSA conversion, bypassing Noir's SSA generation.
//!
//! This module converts the monomorphized AST directly to mavros SSA format,
//! avoiding the intermediate Noir SSA representation.

mod expression_converter;
mod type_converter;

use std::collections::HashMap;

use noirc_frontend::monomorphization::ast::{
    FuncId as AstFuncId, Function as AstFunction, Program,
};

use crate::compiler::ir::r#type::Empty;
use crate::compiler::ssa::{FunctionId, Function, SSA};

use expression_converter::ExpressionConverter;
use type_converter::AstTypeConverter;

/// Converts a monomorphized AST Program to SSA.
pub struct AstSsaConverter {
    /// Maps AST function IDs to SSA function IDs
    function_mapper: HashMap<AstFuncId, FunctionId>,
    /// Type converter
    type_converter: AstTypeConverter,
}

impl AstSsaConverter {
    pub fn new() -> Self {
        Self {
            function_mapper: HashMap::new(),
            type_converter: AstTypeConverter::new(),
        }
    }

    /// Convert an entire program to SSA.
    pub fn convert_program(&mut self, program: &Program) -> SSA<Empty> {
        let mut ssa = SSA::new();

        // Phase 1: Register all functions to handle mutual recursion
        for func in &program.functions {
            if func.id == Program::main_id() {
                // Main function already exists in SSA
                self.function_mapper.insert(func.id, ssa.get_main_id());
            } else {
                let ssa_id = ssa.add_function(func.name.clone());
                self.function_mapper.insert(func.id, ssa_id);
            }
        }

        // Phase 2: Convert each function
        for ast_func in &program.functions {
            let ssa_func_id = *self.function_mapper.get(&ast_func.id).unwrap();
            let converted_func = self.convert_function(ast_func);
            *ssa.get_function_mut(ssa_func_id) = converted_func;
        }

        // TODO: Handle globals if needed
        // For now, we'll leave globals empty since just_add doesn't use them

        ssa
    }

    /// Convert a single function to SSA.
    fn convert_function(&self, ast_func: &AstFunction) -> Function<Empty> {
        let mut function = Function::empty(ast_func.name.clone());
        let entry_block = function.get_entry_id();

        // Add return types
        let return_type = self.type_converter.convert_type(&ast_func.return_type);
        if !matches!(ast_func.return_type, noirc_frontend::monomorphization::ast::Type::Unit) {
            function.add_return_type(return_type);
        }

        // Create expression converter
        let mut expr_converter = ExpressionConverter::new(&self.function_mapper, entry_block);

        // Add function parameters as block parameters
        for (local_id, mutable, _name, param_type, _visibility) in &ast_func.parameters {
            let converted_type = self.type_converter.convert_type(param_type);
            let value_id = function.add_parameter(entry_block, converted_type.clone());

            if *mutable {
                // For mutable parameters, allocate a pointer and store the value
                expr_converter.bind_local_mut(*local_id, value_id, converted_type, &mut function);
            } else {
                expr_converter.bind_local(*local_id, value_id);
            }
        }

        // Convert the function body
        let result = expr_converter.convert_expression(&ast_func.body, &mut function);

        // Add return terminator
        let current_block = expr_converter.current_block();
        match result {
            expression_converter::ExprResult::Value(v) => {
                function.terminate_block_with_return(current_block, vec![v]);
            }
            expression_converter::ExprResult::Values(vs) => {
                function.terminate_block_with_return(current_block, vs);
            }
            expression_converter::ExprResult::Unit => {
                function.terminate_block_with_return(current_block, vec![]);
            }
        }

        function
    }
}

impl SSA<Empty> {
    /// Create SSA directly from a monomorphized AST program.
    pub fn from_ast(program: &Program) -> SSA<Empty> {
        let mut converter = AstSsaConverter::new();
        converter.convert_program(program)
    }
}
