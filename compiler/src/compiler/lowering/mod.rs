//! SSA generation from Noir's monomorphized program.
//!
//! This module converts the monomorphized program directly to mavros SSA format,
//! bypassing the intermediate Noir SSA representation.
//!
//! At the lowering boundary each Noir value occupies one HLSSA value: tuples and structs
//! are materialized, and unit is the empty tuple, just like an empty struct. Ordinary calls
//! and function signatures therefore have one result even for unit. Statement-shaped
//! expressions and effect-only builtins may internally produce no result; `convert_value`
//! evaluates them before supplying the empty tuple at a value boundary, and `expression_type`
//! supplies the matching unit type. Tuple elision later removes empty tuples from parameters,
//! results, and storage. Sequence lowering must preserve bounds checks and slice lengths
//! before that erasure; materializing unit alone is not sufficient. The entry-point wrapper
//! uses the Noir ABI to decide whether `main` has a return to bind to the witness: an internal
//! unit result does not add one.

mod expression_converter;
mod type_converter;

#[cfg(test)]
mod tests;

use noirc_frontend::monomorphization::{
    ast::{
        Definition, Expression, FuncId as AstFuncId, Function as AstFunction, GlobalId, InlineType,
        Program, Type as AstType,
    },
    visitor::visit_expr,
};

use crate::{
    collections::{HashMap, HashSet},
    compiler::{
        diagnostic::Diagnostic,
        ssa::{
            FunctionId, SourceLocation,
            hlssa::{
                HLFunction, HLSSA,
                builder::{HLEmitter, HLFunctionBuilder, HLSSABuilder},
            },
        },
    },
};

use expression_converter::ExpressionConverter;
use fm::FileManager;
use mavros_artifacts::FieldConfig;
use std::cell::RefCell;
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

    /// Maps GlobalId to global slot index.
    global_slots: HashMap<GlobalId, usize>,

    /// Maps GlobalId to its interned constant ValueId when the initializer is a constant.
    global_constants: HashMap<GlobalId, crate::compiler::ssa::ValueId>,

    /// The parameter types each function is called with, its environment first when it is a
    /// closure with captures; see [`SSAConverter::declared_parameters`].
    declared_parameters: HashMap<AstFuncId, Vec<AstType>>,

    /// Utility for converting between types.
    type_converter: TypeConverter,

    /// Constructs refused while lowering.
    diagnostics: RefCell<Vec<Diagnostic>>,
}

impl SSAConverter {
    pub fn new() -> Self {
        Self {
            constrained_mapper: HashMap::default(),
            unconstrained_mapper: HashMap::default(),
            natively_unconstrained: HashSet::default(),
            global_slots: HashMap::default(),
            global_constants: HashMap::default(),
            declared_parameters: HashMap::default(),
            type_converter: TypeConverter::new(),
            diagnostics: RefCell::new(Vec::new()),
        }
    }

    /// Convert an entire program to SSA.
    /// Returns the SSA and whether main is unconstrained, or every construct lowering refused.
    pub fn convert_program(
        &mut self,
        program: &Program,
        file_manager: Option<&FileManager>,
        field: FieldConfig,
    ) -> Result<(HLSSA, bool), Vec<Diagnostic>> {
        let mut ssa = HLSSA::with_main_in(field, "main".to_string());
        self.declared_parameters = Self::declared_parameters(program);

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
            if ast_func.unconstrained {
                // Natively unconstrained: convert once with is_unconstrained=true
                let ssa_func_id = *self.constrained_mapper.get(&ast_func.id).unwrap();
                let converted = self.convert_function(
                    &mut ssa,
                    ast_func,
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
                    &self.unconstrained_mapper,
                    true,
                    file_manager,
                );
                *ssa.get_function_mut(unconstrained_id) = converted;
            }
        }

        let main_is_unconstrained = program
            .functions
            .iter()
            .find(|f| f.id == Program::main_id())
            .map(|f| f.unconstrained)
            .unwrap_or(false);

        let diagnostics = self.diagnostics.take();
        if !diagnostics.is_empty() {
            return Err(diagnostics);
        }
        Ok((ssa, main_is_unconstrained))
    }

    /// The parameter types each function is called with, read from the identifiers that name it.
    ///
    /// The monomorphizer expands a tuple or struct pattern parameter into one parameter per leaf,
    /// while a caller passes one argument per declared parameter, and a closure with captures takes
    /// its environment as an extra leading parameter.
    fn declared_parameters(program: &Program) -> HashMap<AstFuncId, Vec<AstType>> {
        let mut declared = HashMap::default();
        let mut record = |expr: &Expression| {
            if let Expression::Ident(ident) = expr
                && let Definition::Function(id) = ident.definition
            {
                let signature = ExpressionConverter::function_signature(ident);
                let (params, env) = (signature.params, signature.env);
                let env = (*env != AstType::Unit).then(|| env.clone());
                declared
                    .entry(id)
                    .or_insert_with(|| env.into_iter().chain(params.iter().cloned()).collect());
            }
            true
        };
        for function in &program.functions {
            visit_expr(&function.body, &mut record);
        }
        declared
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
                    &self.natively_unconstrained,
                    false,
                    &self.global_slots,
                    &self.global_constants,
                    current_block,
                    file_manager,
                    &self.diagnostics,
                );
                let (_name, _typ, init_expr) = &program.globals[gid];
                let value = expr_converter.convert_value(init_expr, b);
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

            assert!(!b.block(current_block).is_terminated());
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
            assert!(!b.block(entry).is_terminated());
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

        // Every Noir function returns one materialized value, including unit.
        function.add_return_type(self.type_converter.convert_type(&ast_func.return_type));

        let mut b = HLFunctionBuilder::new(&mut function, ssa);

        // Create expression converter
        let mut expr_converter = ExpressionConverter::new_with_globals(
            function_mapper,
            &self.natively_unconstrained,
            in_unconstrained,
            &self.global_slots,
            &self.global_constants,
            entry_block,
            file_manager,
            &self.diagnostics,
        );

        // The frontend rejects folding from source. Retain this defense for hand-built
        // monomorphic programs: lowering recursive circuit calls would feed them into cost
        // analysis.
        if matches!(ast_func.inline_type, InlineType::Fold) {
            expr_converter.reject_unsupported(
                format!(
                    "#[fold] functions are not supported by Mavros: `{}`",
                    ast_func.name
                ),
                ExpressionConverter::expression_location(&ast_func.body),
            );
        }

        // One block parameter per declared parameter, and a pattern parameter's leaves bound to
        // projections of it. Both are located at the function body, the closest source anchor a
        // parameter has.
        let location = expr_converter.expression_source_location(&ast_func.body);
        for pattern in self.parameter_patterns(ast_func) {
            let typ = match &pattern {
                ParameterPattern::Leaf(index) => &ast_func.parameters[*index].3,
                ParameterPattern::Tuple(typ, _) => typ,
            };
            let value_id = b
                .block(entry_block)
                .add_parameter(self.type_converter.convert_type(typ));
            Self::bind_parameter(
                ast_func,
                &pattern,
                value_id,
                entry_block,
                &mut expr_converter,
                &mut b,
                &location,
            );
        }

        // Function bodies follow the same one-value contract as call expressions.
        let result = expr_converter.convert_value(&ast_func.body, &mut b);
        assert!(!b.block(expr_converter.current_block()).is_terminated());
        b.block(expr_converter.current_block())
            .terminate_return(vec![result]);

        function
    }
}

/// How the monomorphizer's parameters of a function make up one declared parameter.
enum ParameterPattern {
    /// The parameter at this index is the declared one.
    Leaf(usize),
    /// A tuple or struct pattern of this declared type, one pattern per field. A zero-sized
    /// declared parameter whose monomorphized parameter an earlier zero-leaf pattern matched is
    /// one with no fields.
    Tuple(AstType, Vec<ParameterPattern>),
}

impl SSAConverter {
    /// The declared parameters of `ast_func`.
    ///
    /// A function referenced by no identifier, such as `main`, keeps its parameters as they are.
    ///
    /// # Panics
    ///
    /// If its parameters do not spell the types its callers pass. The monomorphizer only expands
    /// patterns into leaves, so they always do.
    fn parameter_patterns(&self, ast_func: &AstFunction) -> Vec<ParameterPattern> {
        let params = &ast_func.parameters;
        let Some(declared) = self.declared_parameters.get(&ast_func.id) else {
            return (0..params.len()).map(ParameterPattern::Leaf).collect();
        };
        let mut next = 0;
        let patterns = declared
            .iter()
            .map(|typ| Self::match_pattern(typ, params, &mut next))
            .collect::<Option<Vec<_>>>();
        match patterns {
            Some(patterns) if next == params.len() => patterns,
            _ => ice!(
                "`{}` is called with parameters {declared:?}, which its parameters {:?} do not \
                 spell",
                ast_func.name,
                params.iter().map(|param| &param.3).collect::<Vec<_>>()
            ),
        }
    }

    /// The pattern of a declared parameter of type `typ` whose leaves start at parameter `next`.
    fn match_pattern(
        typ: &AstType,
        params: &noirc_frontend::monomorphization::ast::Parameters,
        next: &mut usize,
    ) -> Option<ParameterPattern> {
        if params.get(*next).is_some_and(|param| *param.3 == *typ) {
            *next += 1;
            return Some(ParameterPattern::Leaf(*next - 1));
        }
        let fields = match typ {
            AstType::Tuple(fields) => fields.as_slice(),
            AstType::Unit => &[],
            _ => return None,
        };
        let fields = fields
            .iter()
            .map(|field| Self::match_pattern(field, params, next))
            .collect::<Option<Vec<_>>>()?;
        Some(ParameterPattern::Tuple(typ.clone(), fields))
    }

    /// Binds the leaves of `pattern`, held in `value_id`, to their locals.
    fn bind_parameter(
        ast_func: &AstFunction,
        pattern: &ParameterPattern,
        value_id: crate::compiler::ssa::ValueId,
        entry_block: crate::compiler::ssa::BlockId,
        expr_converter: &mut ExpressionConverter<'_>,
        b: &mut HLFunctionBuilder<'_>,
        location: &SourceLocation,
    ) {
        match pattern {
            ParameterPattern::Leaf(index) => {
                let (local_id, mutable, ..) = &ast_func.parameters[*index];
                if *mutable {
                    expr_converter.bind_local_mut(*local_id, value_id, b, location.clone());
                } else {
                    expr_converter.bind_local(*local_id, value_id);
                }
            }
            ParameterPattern::Tuple(_, fields) => {
                for (index, field) in fields.iter().enumerate() {
                    let field_value = b
                        .block(entry_block)
                        .with_source_location(location.clone())
                        .tuple_proj(value_id, index);
                    Self::bind_parameter(
                        ast_func,
                        field,
                        field_value,
                        entry_block,
                        expr_converter,
                        b,
                        location,
                    );
                }
            }
        }
    }
}

impl HLSSA {
    /// Create SSA over bn254 directly from a monomorphized program.
    pub fn from_program(program: &Program) -> Result<(HLSSA, bool), Vec<Diagnostic>> {
        Self::from_program_with_file_manager(program, None, FieldConfig::bn254())
    }

    /// Create SSA over `field` directly from a monomorphized program, preserving source locations
    /// when a Noir file manager is available.
    pub fn from_program_with_file_manager(
        program: &Program,
        file_manager: Option<&FileManager>,
        field: FieldConfig,
    ) -> Result<(HLSSA, bool), Vec<Diagnostic>> {
        let mut converter = SSAConverter::new();
        converter.convert_program(program, file_manager, field)
    }
}
