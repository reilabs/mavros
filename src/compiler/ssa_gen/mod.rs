//! SSA generation from Noir's monomorphized program.
//!
//! This module converts the monomorphized program directly to mavros SSA format,
//! bypassing the intermediate Noir SSA representation.

mod expression_converter;
mod type_converter;

use std::collections::{HashMap, HashSet};

use noirc_frontend::monomorphization::ast::{
    Definition, Expression, FuncId as AstFuncId, Function as AstFunction,
    GlobalId, Program,
};

use crate::compiler::ir::r#type::Empty;
use crate::compiler::ssa::{FunctionId, Function, SSA};

use expression_converter::ExpressionConverter;
use type_converter::TypeConverter;

/// Converts a monomorphized Program to SSA.
pub struct SsaConverter {
    /// Maps AST function IDs to SSA function IDs (constrained context)
    constrained_mapper: HashMap<AstFuncId, FunctionId>,
    /// Maps AST function IDs to SSA function IDs (unconstrained context).
    /// For natively unconstrained functions, same ID as constrained_mapper.
    /// For constrained functions, points to a separate unconstrained variant.
    unconstrained_mapper: HashMap<AstFuncId, FunctionId>,
    /// Maps GlobalId to global slot index
    global_slots: HashMap<GlobalId, usize>,
    /// Type converter
    type_converter: TypeConverter,
}

impl SsaConverter {
    pub fn new() -> Self {
        Self {
            constrained_mapper: HashMap::new(),
            unconstrained_mapper: HashMap::new(),
            global_slots: HashMap::new(),
            type_converter: TypeConverter::new(),
        }
    }

    /// Convert an entire program to SSA.
    pub fn convert_program(&mut self, program: &Program) -> SSA<Empty> {
        let mut ssa = SSA::new();

        let unconstrained_functions = program
            .functions
            .iter()
            .filter(|f| f.unconstrained)
            .map(|f| f.id)
            .collect::<HashSet<_>>();
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

        // Phase 2: Convert globals (must be before function conversion so global_slots are available)
        if !program.globals.is_empty() {
            self.convert_globals(program, &mut ssa);
        }

        // Phase 3: Convert each function
        for ast_func in &program.functions {
            if ast_func.unconstrained {
                // Natively unconstrained: convert once with is_unconstrained=true
                let ssa_func_id = *self.constrained_mapper.get(&ast_func.id).unwrap();
                let converted = self.convert_function(ast_func, &self.unconstrained_mapper, true, &unconstrained_functions);
                *ssa.get_function_mut(ssa_func_id) = converted;
            } else {
                // Constrained version
                let constrained_id = *self.constrained_mapper.get(&ast_func.id).unwrap();
                let converted = self.convert_function(ast_func, &self.constrained_mapper, false, &unconstrained_functions);
                *ssa.get_function_mut(constrained_id) = converted;

                // Unconstrained variant (for calls from unconstrained context)
                let unconstrained_id = *self.unconstrained_mapper.get(&ast_func.id).unwrap();
                let converted = self.convert_function(ast_func, &self.unconstrained_mapper, true, &unconstrained_functions);
                *ssa.get_function_mut(unconstrained_id) = converted;
            }
        }

        ssa
    }

    /// Collect all GlobalIds referenced transitively by an expression.
    fn collect_global_deps(expr: &Expression, deps: &mut Vec<GlobalId>) {
        match expr {
            Expression::Ident(ident) => {
                if let Definition::Global(gid) = &ident.definition {
                    deps.push(*gid);
                }
            }
            Expression::Literal(lit) => {
                use noirc_frontend::monomorphization::ast::Literal;
                match lit {
                    Literal::Array(arr) | Literal::Vector(arr) => {
                        for e in &arr.contents {
                            Self::collect_global_deps(e, deps);
                        }
                    }
                    Literal::Repeated { element, .. } => {
                        Self::collect_global_deps(element, deps);
                    }
                    _ => {}
                }
            }
            Expression::Block(exprs) => {
                for e in exprs {
                    Self::collect_global_deps(e, deps);
                }
            }
            Expression::Binary(bin) => {
                Self::collect_global_deps(&bin.lhs, deps);
                Self::collect_global_deps(&bin.rhs, deps);
            }
            Expression::Unary(un) => {
                Self::collect_global_deps(&un.rhs, deps);
            }
            Expression::Cast(cast) => {
                Self::collect_global_deps(&cast.lhs, deps);
            }
            Expression::Tuple(elems) => {
                for e in elems {
                    Self::collect_global_deps(e, deps);
                }
            }
            Expression::Call(call) => {
                Self::collect_global_deps(&call.func, deps);
                for arg in &call.arguments {
                    Self::collect_global_deps(arg, deps);
                }
            }
            Expression::If(if_expr) => {
                Self::collect_global_deps(&if_expr.condition, deps);
                Self::collect_global_deps(&if_expr.consequence, deps);
                if let Some(alt) = &if_expr.alternative {
                    Self::collect_global_deps(alt, deps);
                }
            }
            Expression::Let(let_expr) => {
                Self::collect_global_deps(&let_expr.expression, deps);
            }
            Expression::Semi(inner) | Expression::Clone(inner) | Expression::Drop(inner) => {
                Self::collect_global_deps(inner, deps);
            }
            Expression::Index(idx) => {
                Self::collect_global_deps(&idx.collection, deps);
                Self::collect_global_deps(&idx.index, deps);
            }
            Expression::ExtractTupleField(tuple_expr, _) => {
                Self::collect_global_deps(tuple_expr, deps);
            }
            _ => {}
        }
    }

    /// Convert globals: assign slot indices, build init/deinit functions.
    fn convert_globals(&mut self, program: &Program, ssa: &mut SSA<Empty>) {
        let unconstrained_functions = program
            .functions
            .iter()
            .filter(|f| f.unconstrained)
            .map(|f| f.id)
            .collect::<HashSet<_>>();
        // Assign slot indices
        let mut global_types = Vec::new();
        let mut ordered_ids: Vec<GlobalId> = Vec::new();

        // Topological sort: process globals in dependency order.
        // Build adjacency: each global depends on other globals referenced in its initializer.
        let all_ids: Vec<GlobalId> = program.globals.keys().copied().collect();
        let mut visited = std::collections::HashSet::new();
        let mut in_stack = std::collections::HashSet::new();

        fn topo_visit(
            gid: GlobalId,
            program: &Program,
            visited: &mut std::collections::HashSet<GlobalId>,
            in_stack: &mut std::collections::HashSet<GlobalId>,
            ordered: &mut Vec<GlobalId>,
        ) {
            if visited.contains(&gid) {
                return;
            }
            assert!(!in_stack.contains(&gid), "Cyclic global dependency on {:?}", gid);
            in_stack.insert(gid);
            let (_name, _typ, init_expr) = &program.globals[&gid];
            let mut deps = Vec::new();
            SsaConverter::collect_global_deps(init_expr, &mut deps);
            for dep in deps {
                topo_visit(dep, program, visited, in_stack, ordered);
            }
            in_stack.remove(&gid);
            visited.insert(gid);
            ordered.push(gid);
        }

        for gid in &all_ids {
            topo_visit(*gid, program, &mut visited, &mut in_stack, &mut ordered_ids);
        }

        // Assign slot indices in dependency order
        for gid in &ordered_ids {
            let (_name, typ, _expr) = &program.globals[gid];
            let converted_type = self.type_converter.convert_type(typ);
            let idx = global_types.len();
            global_types.push(converted_type);
            self.global_slots.insert(*gid, idx);
        }

        // Build init function
        let init_fn_id = ssa.add_function("globals_init".to_string());
        {
            let init_fn = ssa.get_function_mut(init_fn_id);
            let entry = init_fn.get_entry_id();

            // We need an ExpressionConverter to evaluate initializer expressions
            let mut expr_converter = ExpressionConverter::new_with_globals(
                &self.constrained_mapper,
                entry,
                false,
                &self.global_slots,
                &unconstrained_functions,
            );

            for gid in &ordered_ids {
                let (_name, _typ, init_expr) = &program.globals[gid];
                let value = expr_converter.convert_expression(init_expr, init_fn).unwrap();
                let idx = self.global_slots[gid];
                init_fn.push_init_global(entry, idx, value);
            }

            init_fn.terminate_block_with_return(entry, vec![]);
        }

        // Build deinit function
        let deinit_fn_id = ssa.add_function("globals_deinit".to_string());
        {
            let deinit_fn = ssa.get_function_mut(deinit_fn_id);
            let entry = deinit_fn.get_entry_id();
            for (i, typ) in global_types.iter().enumerate() {
                if typ.is_heap_allocated() {
                    deinit_fn.push_drop_global(entry, i);
                }
            }
            deinit_fn.terminate_block_with_return(entry, vec![]);
        }

        ssa.set_global_types(global_types);
        ssa.set_globals_init_fn(init_fn_id);
        ssa.set_globals_deinit_fn(deinit_fn_id);
    }

    /// Convert a single function to SSA.
    fn convert_function(
        &self,
        ast_func: &AstFunction,
        function_mapper: &HashMap<AstFuncId, FunctionId>,
        in_unconstrained: bool,
        unconstrained_functions: &HashSet<AstFuncId>,
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
        let mut expr_converter = ExpressionConverter::new_with_globals(function_mapper, entry_block, in_unconstrained, &self.global_slots, unconstrained_functions);

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
        let return_values = result.into_iter().collect();
        function.terminate_block_with_return(current_block, return_values);

        function
    }
}

impl SSA<Empty> {
    /// Create SSA directly from a monomorphized program.
    pub fn from_program(program: &Program) -> SSA<Empty> {
        let mut converter = SsaConverter::new();
        converter.convert_program(program)
    }
}
