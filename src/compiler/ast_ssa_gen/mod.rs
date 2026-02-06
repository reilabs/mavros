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
    /// Maps AST function IDs to SSA function IDs (constrained context)
    constrained_mapper: HashMap<AstFuncId, FunctionId>,
    /// Maps AST function IDs to SSA function IDs (unconstrained context).
    /// For natively unconstrained functions, same ID as constrained_mapper.
    /// For constrained functions, points to a separate unconstrained variant.
    unconstrained_mapper: HashMap<AstFuncId, FunctionId>,
    /// Type converter
    type_converter: AstTypeConverter,
}

impl AstSsaConverter {
    pub fn new() -> Self {
        Self {
            constrained_mapper: HashMap::new(),
            unconstrained_mapper: HashMap::new(),
            type_converter: AstTypeConverter::new(),
        }
    }

    /// Convert an entire program to SSA.
    pub fn convert_program(&mut self, program: &Program) -> SSA<Empty> {
        let mut ssa = SSA::new();

        // Phase 1: Register all functions to handle mutual recursion.
        // For each constrained function, also register an unconstrained variant
        // so that calls from unconstrained context propagate is_unconstrained=true.
        for func in &program.functions {
            let ssa_id = if func.id == Program::main_id() {
                ssa.get_main_id()
            } else {
                ssa.add_function(func.name.clone())
            };
            self.constrained_mapper.insert(func.id, ssa_id);

            if func.unconstrained {
                // Natively unconstrained: same ID in both contexts
                self.unconstrained_mapper.insert(func.id, ssa_id);
            } else {
                // Constrained: create a separate unconstrained variant
                let variant_id = ssa.add_function(format!("{}_unconstrained", func.name));
                self.unconstrained_mapper.insert(func.id, variant_id);
            }
        }

        // Phase 2: Convert each function
        for ast_func in &program.functions {
            if ast_func.unconstrained {
                // Natively unconstrained: convert once with is_unconstrained=true
                let ssa_func_id = *self.constrained_mapper.get(&ast_func.id).unwrap();
                let converted = self.convert_function(ast_func, &self.unconstrained_mapper, true);
                *ssa.get_function_mut(ssa_func_id) = converted;
            } else {
                // Constrained version
                let constrained_id = *self.constrained_mapper.get(&ast_func.id).unwrap();
                let converted = self.convert_function(ast_func, &self.constrained_mapper, false);
                *ssa.get_function_mut(constrained_id) = converted;

                // Unconstrained variant (for calls from unconstrained context)
                let unconstrained_id = *self.unconstrained_mapper.get(&ast_func.id).unwrap();
                let converted = self.convert_function(ast_func, &self.unconstrained_mapper, true);
                *ssa.get_function_mut(unconstrained_id) = converted;
            }
        }

        // TODO: Handle globals if needed
        // For now, we'll leave globals empty since just_add doesn't use them

        ssa
    }

    /// Convert a single function to SSA.
    fn convert_function(
        &self,
        ast_func: &AstFunction,
        function_mapper: &HashMap<AstFuncId, FunctionId>,
        in_unconstrained: bool,
    ) -> Function<Empty> {
        let name = if in_unconstrained && !ast_func.unconstrained {
            format!("{}_unconstrained", ast_func.name)
        } else {
            ast_func.name.clone()
        };
        let mut function = Function::empty(name);
        let entry_block = function.get_entry_id();

        // Add return types
        let return_type = self.type_converter.convert_type(&ast_func.return_type);
        if !matches!(ast_func.return_type, noirc_frontend::monomorphization::ast::Type::Unit) {
            function.add_return_type(return_type);
        }

        // Create expression converter
        let mut expr_converter = ExpressionConverter::new(function_mapper, entry_block, in_unconstrained);

        // Add function parameters as block parameters
        for (local_id, mutable, _name, param_type, _visibility) in &ast_func.parameters {
            let converted_type = self.type_converter.convert_type(param_type);
            let value_id = function.add_parameter(entry_block, converted_type.clone());

            if *mutable {
                // For mutable parameters, allocate a pointer and store the value
                expr_converter.bind_local_mut(*local_id, value_id, converted_type, &mut function);
            } else {
                expr_converter.bind_local(*local_id, vec![value_id]);
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
                // Multiple values (tuple) - materialize into a single tuple value
                let types: Vec<_> = vs.iter()
                    .map(|_| crate::compiler::ir::r#type::Type::field(Empty))
                    .collect();
                let tuple = function.push_mk_tuple(current_block, vs, types);
                function.terminate_block_with_return(current_block, vec![tuple]);
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
