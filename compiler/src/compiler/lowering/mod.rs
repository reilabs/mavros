//! SSA generation from Noir's monomorphized program.
//!
//! This module converts the monomorphized program directly to mavros SSA format,
//! bypassing the intermediate Noir SSA representation.

mod derive_pedersen_generators;
mod expression_converter;
mod type_converter;

use noirc_frontend::monomorphization::ast::{
    Definition, Expression, FuncId as AstFuncId, Function as AstFunction, GlobalId, Program,
    Type as AstType,
};
use noirc_frontend::monomorphization::visitor::visit_expr;

use crate::{
    collections::{HashMap, HashSet},
    compiler::ssa::{
        FunctionId, SourceLocation,
        hlssa::{
            HLFunction, HLSSA,
            builder::{HLEmitter, HLFunctionBuilder, HLSSABuilder},
        },
    },
};

use expression_converter::ExpressionConverter;
use fm::FileManager;
use type_converter::TypeConverter;

/// Converts a monomorphized Program to SSA.
pub struct SSAConverter {
    /// Maps AST function IDs to SSA function IDs (constrained context).
    constrained_mapper: HashMap<AstFuncId, FunctionId>,

    /// Maps AST function IDs to SSA function IDs (unconstrained context).
    ///
    /// For natively unconstrained functions, it will yield the same ID as constrained_mapper. For
    /// constrained functions, it points to a separate unconstrained variant.
    unconstrained_mapper: HashMap<AstFuncId, FunctionId>,

    /// Set of AST function IDs that are natively unconstrained.
    natively_unconstrained: HashSet<AstFuncId>,

    /// Functions which must be inlined so constant-only builtins see their concrete call-site
    /// arguments, matching the point at which Noir's own SSA pipeline lowers them.
    constant_builtin_inline_functions: HashSet<AstFuncId>,

    /// Maps GlobalId to global slot index.
    global_slots: HashMap<GlobalId, usize>,

    /// Maps GlobalId to its interned constant ValueId when the initializer is a constant.
    global_constants: HashMap<GlobalId, crate::compiler::ssa::ValueId>,

    /// Utility for converting between types.
    type_converter: TypeConverter,
}

impl SSAConverter {
    pub fn new() -> Self {
        Self {
            constrained_mapper: HashMap::default(),
            unconstrained_mapper: HashMap::default(),
            natively_unconstrained: HashSet::default(),
            constant_builtin_inline_functions: HashSet::default(),
            global_slots: HashMap::default(),
            global_constants: HashMap::default(),
            type_converter: TypeConverter::new(),
        }
    }

    /// Convert an entire program to SSA.
    /// Returns the SSA and whether main is unconstrained.
    pub fn convert_program(
        &mut self,
        program: &Program,
        file_manager: Option<&FileManager>,
    ) -> (HLSSA, bool) {
        let mut ssa = HLSSA::new();
        self.constant_builtin_inline_functions =
            Self::functions_reaching_builtin(program, "derive_pedersen_generators");

        // Phase 1: Register all functions to handle mutual recursion.
        // For each constrained function, also register an unconstrained variant
        // so that calls from unconstrained context propagate is_unconstrained=true.
        for func in &program.functions {
            let ssa_id = if func.id == Program::main_id() {
                ssa.get_unique_entrypoint_id()
            } else {
                ssa.add_function(func.name.clone())
            };
            self.constrained_mapper.insert(func.id, ssa_id);

            if func.unconstrained {
                // Natively unconstrained: same ID in both contexts
                self.unconstrained_mapper.insert(func.id, ssa_id);
                self.natively_unconstrained.insert(func.id);
            } else {
                // Constrained: create a separate unconstrained variant
                let variant_id = ssa.add_function(format!("{}_unconstrained", func.name));
                self.unconstrained_mapper.insert(func.id, variant_id);
            }
        }

        // Phase 2: Convert globals (must be before function conversion so global_slots are available)
        if !program.globals.is_empty() {
            self.convert_globals(program, &mut ssa, file_manager);
        }

        // Phase 3: Convert each function
        for ast_func in &program.functions {
            // Static and resolved higher-order calls to these functions are expanded by
            // ExpressionConverter. Their abstract standalone copies cannot resolve the
            // constant-only builtin and are intentionally left empty for reachability pruning.
            if self.is_targeted_inline_function(ast_func) {
                continue;
            }
            if ast_func.unconstrained {
                // Natively unconstrained: convert once with is_unconstrained=true
                let ssa_func_id = *self.constrained_mapper.get(&ast_func.id).unwrap();
                let converted = self.convert_function(
                    &mut ssa,
                    ast_func,
                    &program.functions,
                    &self.unconstrained_mapper,
                    true,
                    file_manager,
                );
                *ssa.get_function_mut(ssa_func_id) = converted;
            } else {
                // Constrained version
                let constrained_id = *self.constrained_mapper.get(&ast_func.id).unwrap();
                let converted = self.convert_function(
                    &mut ssa,
                    ast_func,
                    &program.functions,
                    &self.constrained_mapper,
                    false,
                    file_manager,
                );
                *ssa.get_function_mut(constrained_id) = converted;

                // Unconstrained variant (for calls from unconstrained context)
                let unconstrained_id = *self.unconstrained_mapper.get(&ast_func.id).unwrap();
                let converted = self.convert_function(
                    &mut ssa,
                    ast_func,
                    &program.functions,
                    &self.unconstrained_mapper,
                    true,
                    file_manager,
                );
                *ssa.get_function_mut(unconstrained_id) = converted;
            }
        }

        // These IDs were needed while lowering function values, but every supported call to them
        // has now been expanded. Remove their intentionally body-less registrations before the
        // ordinary SSA pipeline inspects function bodies.
        for ast_func in &program.functions {
            if self.is_targeted_inline_function(ast_func) {
                ssa.delete_function(self.constrained_mapper[&ast_func.id]);
                ssa.delete_function(self.unconstrained_mapper[&ast_func.id]);
            }
        }

        let main_is_unconstrained = program
            .functions
            .iter()
            .find(|f| f.id == Program::main_id())
            .map(|f| f.unconstrained)
            .unwrap_or(false);

        (ssa, main_is_unconstrained)
    }

    fn is_targeted_inline_function(&self, function: &AstFunction) -> bool {
        function.id != Program::main_id()
            && !function.unconstrained
            && self
                .constant_builtin_inline_functions
                .contains(&function.id)
    }

    /// Find the transitive callers of a builtin in the monomorphized call graph.
    fn functions_reaching_builtin(program: &Program, builtin: &str) -> HashSet<AstFuncId> {
        let mut callees = HashMap::<AstFuncId, HashSet<AstFuncId>>::default();
        let mut higher_order_calls = Vec::<(AstFuncId, HashSet<AstFuncId>)>::new();
        let mut reaching = HashSet::default();

        fn contains_function(typ: &AstType) -> bool {
            match typ {
                AstType::Function(..) => true,
                AstType::Tuple(elements) => elements.iter().any(contains_function),
                _ => false,
            }
        }

        for function in &program.functions {
            let function_callees = callees.entry(function.id).or_default();
            visit_expr(&function.body, &mut |expression| {
                if let Expression::Call(call) = expression
                    && let Expression::Ident(ident) = call.func.as_ref()
                {
                    match &ident.definition {
                        Definition::Function(callee) => {
                            function_callees.insert(*callee);
                            let mut function_arguments = HashSet::default();
                            for argument in &call.arguments {
                                if argument
                                    .return_type()
                                    .is_some_and(|typ| contains_function(typ.as_ref()))
                                {
                                    visit_expr(argument, &mut |argument_expression| {
                                        if let Expression::Ident(ident) = argument_expression
                                            && let Definition::Function(function) =
                                                &ident.definition
                                        {
                                            function_arguments.insert(*function);
                                        }
                                        true
                                    });
                                }
                            }
                            if !function_arguments.is_empty() {
                                higher_order_calls.push((*callee, function_arguments));
                            }
                        }
                        Definition::Builtin(name) if name == builtin => {
                            reaching.insert(function.id);
                        }
                        _ => {}
                    }
                }
                true
            });
        }

        // Treat a higher-order helper as calling the function values passed to it. This reuses
        // the same transitive closure below and ensures helpers specialized at their call sites do
        // not leave abstract function-pointer bodies for defunctionalization.
        for (callee, function_arguments) in higher_order_calls {
            callees
                .entry(callee)
                .or_default()
                .extend(function_arguments);
        }

        loop {
            let mut changed = false;
            for (caller, function_callees) in &callees {
                if !reaching.contains(caller)
                    && function_callees
                        .iter()
                        .any(|callee| reaching.contains(callee))
                {
                    changed |= reaching.insert(*caller);
                }
            }
            if !changed {
                return reaching;
            }
        }
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
    fn convert_globals(
        &mut self,
        program: &Program,
        ssa: &mut HLSSA,
        file_manager: Option<&FileManager>,
    ) {
        // Assign slot indices
        let mut global_types = Vec::new();
        let mut ordered_ids: Vec<GlobalId> = Vec::new();

        // Topological sort: process globals in dependency order.
        // Build adjacency: each global depends on other globals referenced in its initializer.
        // `Program.globals` is a `BTreeMap`, so key order is already deterministic; the sort
        // makes slot assignment robust even if that container ever changes.
        let mut all_ids: Vec<GlobalId> = program.globals.keys().copied().collect();
        all_ids.sort();
        let mut visited = crate::collections::HashSet::default();
        let mut in_stack = crate::collections::HashSet::default();

        fn topo_visit(
            gid: GlobalId,
            program: &Program,
            visited: &mut crate::collections::HashSet<GlobalId>,
            in_stack: &mut crate::collections::HashSet<GlobalId>,
            ordered: &mut Vec<GlobalId>,
        ) {
            if visited.contains(&gid) {
                return;
            }
            assert!(
                !in_stack.contains(&gid),
                "Cyclic global dependency on {:?}",
                gid
            );
            in_stack.insert(gid);
            let (_name, _typ, init_expr) = &program.globals[&gid];
            let mut deps = Vec::new();
            SSAConverter::collect_global_deps(init_expr, &mut deps);
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
        let mut sb = HLSSABuilder::new(ssa);
        sb.modify_function(init_fn_id, |b| {
            let entry = b.function.get_entry_id();

            let mut current_block = entry;

            for gid in &ordered_ids {
                // Use a fresh converter for each initializer so we can record a constant global
                // after one initializer and let later initializers read it directly.
                let mut expr_converter = ExpressionConverter::new_with_globals(
                    &self.constrained_mapper,
                    &program.functions,
                    &self.constant_builtin_inline_functions,
                    &self.natively_unconstrained,
                    false,
                    &self.global_slots,
                    &self.global_constants,
                    current_block,
                    file_manager,
                );
                let (_name, _typ, init_expr) = &program.globals[gid];
                let value = expr_converter.convert_expression(init_expr, b).unwrap();
                current_block = expr_converter.current_block();
                let idx = self.global_slots[gid];
                let location = expr_converter.expression_source_location(init_expr);
                b.block(current_block)
                    .with_source_location(location)
                    .init_global(idx, value);
                if b.ssa.is_const(value) {
                    self.global_constants.insert(*gid, value);
                }
            }

            b.block(current_block).terminate_return(vec![]);
        });

        // Build deinit function
        let deinit_fn_id = sb.ssa().add_function("globals_deinit".to_string());
        sb.modify_function(deinit_fn_id, |b| {
            let entry = b.function.get_entry_id();
            let location = SourceLocation::synthetic("globals_deinit");
            for (i, typ) in global_types.iter().enumerate() {
                if typ.is_heap_allocated() {
                    b.block(entry)
                        .with_source_location(location.clone())
                        .drop_global(i);
                }
            }
            b.block(entry).terminate_return(vec![]);
        });

        ssa.set_global_types(global_types);
        ssa.set_globals_init_fn(init_fn_id);
        ssa.set_globals_deinit_fn(deinit_fn_id);
    }

    /// Convert a single function to SSA.
    fn convert_function(
        &self,
        ssa: &mut HLSSA,
        ast_func: &AstFunction,
        functions: &[AstFunction],
        function_mapper: &HashMap<AstFuncId, FunctionId>,
        in_unconstrained: bool,
        file_manager: Option<&FileManager>,
    ) -> HLFunction {
        let name = if in_unconstrained && !ast_func.unconstrained {
            format!("{}_unconstrained", ast_func.name)
        } else {
            ast_func.name.clone()
        };
        let mut function = HLFunction::empty(name);
        let entry_block = function.get_entry_id();

        // Add return types
        let return_type = self.type_converter.convert_type(&ast_func.return_type);
        if !matches!(
            ast_func.return_type,
            noirc_frontend::monomorphization::ast::Type::Unit
        ) {
            function.add_return_type(return_type);
        }

        let mut b = HLFunctionBuilder::new(&mut function, ssa);

        // Create expression converter
        let mut expr_converter = ExpressionConverter::new_with_globals(
            function_mapper,
            functions,
            &self.constant_builtin_inline_functions,
            &self.natively_unconstrained,
            in_unconstrained,
            &self.global_slots,
            &self.global_constants,
            entry_block,
            file_manager,
        );

        // Add function parameters as block parameters
        for (local_id, mutable, _name, param_type, _visibility) in &ast_func.parameters {
            let converted_type = self.type_converter.convert_type(param_type);
            let value_id = b.block(entry_block).add_parameter(converted_type);

            if *mutable {
                // For mutable parameters, allocate a pointer and store the value. The alloc is
                // attributed to the function body, the closest source anchor a parameter has.
                let location = expr_converter.expression_source_location(&ast_func.body);
                expr_converter.bind_local_mut(*local_id, value_id, &mut b, location);
            } else {
                expr_converter.bind_local(*local_id, value_id);
            }
        }

        // Convert the function body
        let result = expr_converter.convert_expression(&ast_func.body, &mut b);

        // Add return terminator
        let return_values = result.into_iter().collect();
        b.block(expr_converter.current_block())
            .terminate_return(return_values);

        function
    }
}

impl HLSSA {
    /// Create SSA directly from a monomorphized program.
    pub fn from_program(program: &Program) -> (HLSSA, bool) {
        Self::from_program_with_file_manager(program, None)
    }

    /// Create SSA directly from a monomorphized program, preserving source locations when
    /// a Noir file manager is available.
    pub fn from_program_with_file_manager(
        program: &Program,
        file_manager: Option<&FileManager>,
    ) -> (HLSSA, bool) {
        let mut converter = SSAConverter::new();
        converter.convert_program(program, file_manager)
    }
}
