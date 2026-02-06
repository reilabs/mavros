//! Converts monomorphized AST expressions to SSA instructions.

use std::collections::{HashMap, HashSet};

use noirc_frontend::ast::BinaryOpKind;
use noirc_frontend::monomorphization::ast::{
    Assign, Binary, Definition, Expression, For, FuncId as AstFuncId, Ident, If, Index, LValue, Let, LocalId,
};

use crate::compiler::ir::r#type::{Empty, Type};
use crate::compiler::ssa::{BlockId, CastTarget, Endianness, Function, FunctionId, Radix, SeqType, TupleIdx, ValueId};

use super::type_converter::AstTypeConverter;

/// Result of converting an expression.
/// Some expressions produce values, others don't (e.g., constraints).
pub enum ExprResult {
    /// Expression produced a value
    Value(ValueId),
    /// Expression produced multiple values (e.g., tuple)
    Values(Vec<ValueId>),
    /// Expression produced no value (e.g., assert, unit)
    Unit,
}

impl ExprResult {
    /// Get the single value, panicking if not exactly one value.
    pub fn into_value(self) -> ValueId {
        match self {
            ExprResult::Value(v) => v,
            ExprResult::Values(vs) if vs.len() == 1 => vs[0],
            ExprResult::Values(_) => panic!("Expected single value, got multiple"),
            ExprResult::Unit => panic!("Expected value, got unit"),
        }
    }

}

/// A step in a nested lvalue access path (e.g., `arr[i].field[j]`).
#[derive(Debug)]
enum AccessStep {
    /// Array index: `arr[idx]`
    Index(ValueId),
    /// Tuple/struct field: `obj.field_N`
    Field(usize),
}

/// Loop context for break/continue support.
struct LoopContext {
    loop_header: BlockId,
    exit_block: BlockId,
    loop_index: ValueId,
    index_bit_size: usize,
}

/// Converts expressions within a single function.
pub struct ExpressionConverter<'a> {
    /// Maps LocalId to ValueId(s) for variable bindings.
    /// Tuples/structs are stored as multiple values (flattened).
    /// For mutable variables, this stores the pointer(s) to the value(s).
    /// For immutable variables, this stores the value(s) directly.
    bindings: HashMap<LocalId, Vec<ValueId>>,
    /// Tracks which LocalIds are mutable (their binding is a pointer)
    mutable_locals: HashSet<LocalId>,
    /// Maps AST FuncId to SSA FunctionId
    function_mapper: &'a HashMap<AstFuncId, FunctionId>,
    /// Type converter
    type_converter: AstTypeConverter,
    /// Current block we're building
    current_block: BlockId,
    /// Stack of enclosing loop contexts for break/continue
    loop_stack: Vec<LoopContext>,
    /// Whether the current function is unconstrained
    in_unconstrained: bool,
}

impl<'a> ExpressionConverter<'a> {
    pub fn new(
        function_mapper: &'a HashMap<AstFuncId, FunctionId>,
        entry_block: BlockId,
        in_unconstrained: bool,
    ) -> Self {
        Self {
            bindings: HashMap::new(),
            mutable_locals: HashSet::new(),
            function_mapper,
            type_converter: AstTypeConverter::new(),
            current_block: entry_block,
            loop_stack: Vec::new(),
            in_unconstrained,
        }
    }

    /// Bind an immutable local variable to value(s)
    pub fn bind_local(&mut self, local_id: LocalId, values: Vec<ValueId>) {
        self.bindings.insert(local_id, values);
    }

    /// Bind a mutable local variable - allocates a pointer and stores the initial value
    pub fn bind_local_mut(
        &mut self,
        local_id: LocalId,
        value_id: ValueId,
        typ: Type<Empty>,
        function: &mut Function<Empty>,
    ) {
        let ptr = function.push_alloc(self.current_block, typ, Empty);
        function.push_store(self.current_block, ptr, value_id);
        self.bindings.insert(local_id, vec![ptr]);
        self.mutable_locals.insert(local_id);
    }

    /// Get flattened types for an expression (for tuple/struct allocation)
    fn get_flattened_types(&self, expr: &Expression) -> Vec<Type<Empty>> {
        if let Some(typ) = expr.return_type() {
            self.flatten_type(&typ)
        } else {
            vec![Type::field(Empty)]
        }
    }

    /// Flatten a type into its component types
    fn flatten_type(&self, typ: &noirc_frontend::monomorphization::ast::Type) -> Vec<Type<Empty>> {
        use noirc_frontend::monomorphization::ast::Type as AstType;
        match typ {
            AstType::Tuple(types) => {
                types.iter().flat_map(|t| self.flatten_type(t)).collect()
            }
            _ => vec![self.type_converter.convert_type(typ)],
        }
    }

    /// Check whether a Noir type contains an array anywhere in its structure.
    fn noir_type_contains_array(typ: &noirc_frontend::monomorphization::ast::Type) -> bool {
        use noirc_frontend::monomorphization::ast::Type as AstType;
        match typ {
            AstType::Array(_, _) => true,
            AstType::Tuple(fields) => fields.iter().any(|f| Self::noir_type_contains_array(f)),
            _ => false,
        }
    }

    /// Calculate return size for function calls (tuples count as 1, unit as 0)
    fn return_size(&self, typ: &noirc_frontend::monomorphization::ast::Type) -> usize {
        use noirc_frontend::monomorphization::ast::Type as AstType;
        match typ {
            AstType::Unit => 0,
            _ => 1,
        }
    }

    /// Get the current block
    pub fn current_block(&self) -> BlockId {
        self.current_block
    }

    /// Convert an expression to SSA instructions.
    pub fn convert_expression(
        &mut self,
        expr: &Expression,
        function: &mut Function<Empty>,
    ) -> ExprResult {
        match expr {
            Expression::Ident(ident) => self.convert_ident(ident, function),
            Expression::Binary(binary) => self.convert_binary(binary, function),
            Expression::Let(let_expr) => self.convert_let(let_expr, function),
            Expression::Block(exprs) => self.convert_block(exprs, function),
            Expression::Constrain(constraint_expr, _location, _msg) => {
                self.convert_constrain(constraint_expr, function)
            }
            Expression::Semi(inner) => {
                // Semi expressions evaluate the inner expression and discard the result
                self.convert_expression(inner, function);
                ExprResult::Unit
            }
            Expression::Literal(lit) => self.convert_literal(lit, function),
            Expression::Tuple(exprs) => self.convert_tuple(exprs, function),
            Expression::Call(call) => self.convert_call(call, function),
            Expression::Assign(assign) => self.convert_assign(assign, function),
            Expression::For(for_expr) => self.convert_for(for_expr, function),
            Expression::Index(index) => self.convert_index(index, function),
            Expression::ExtractTupleField(tuple_expr, idx) => {
                self.convert_extract_tuple_field(tuple_expr, *idx, function)
            }
            Expression::Cast(cast) => self.convert_cast(cast, function),
            Expression::If(if_expr) => self.convert_if(if_expr, function),
            Expression::Unary(unary) => self.convert_unary(unary, function),
            Expression::Clone(inner) => self.convert_expression(inner, function),
            Expression::Drop(_) => ExprResult::Unit,
            Expression::Break => {
                let ctx = self.loop_stack.last().expect("break outside of loop");
                let exit_block = ctx.exit_block;
                function.terminate_block_with_jmp(self.current_block, exit_block, vec![]);
                // Create a dead block for any subsequent code
                self.current_block = function.add_block();
                ExprResult::Unit
            }
            Expression::Continue => {
                let ctx = self.loop_stack.last().expect("continue outside of loop");
                let loop_header = ctx.loop_header;
                let loop_index = ctx.loop_index;
                let index_bit_size = ctx.index_bit_size;
                // Increment index and jump back to header
                let one = function.push_u_const(index_bit_size, 1);
                let next_index = function.push_add(self.current_block, loop_index, one);
                function.terminate_block_with_jmp(self.current_block, loop_header, vec![next_index]);
                // Create a dead block for any subsequent code
                self.current_block = function.add_block();
                ExprResult::Unit
            }
            _ => todo!("Expression type not yet supported: {:?}", std::mem::discriminant(expr)),
        }
    }

    fn convert_ident(
        &mut self,
        ident: &Ident,
        function: &mut Function<Empty>,
    ) -> ExprResult {
        match &ident.definition {
            Definition::Local(local_id) => {
                let values = self.bindings.get(local_id)
                    .unwrap_or_else(|| panic!("Undefined local variable: {:?}", local_id))
                    .clone();

                // For mutable variables, we need to load from the pointer(s)
                let values = if self.mutable_locals.contains(local_id) {
                    values.iter()
                        .map(|ptr| function.push_load(self.current_block, *ptr))
                        .collect()
                } else {
                    values
                };

                // If we have multiple values (flattened tuple binding), reconstruct the tuple
                // so callers always get a properly typed materialized value
                if values.len() == 1 {
                    ExprResult::Value(values[0])
                } else {
                    // Reconstruct tuple from flattened values
                    let types = self.flatten_type(&ident.typ);
                    let tuple = function.push_mk_tuple(self.current_block, values, types);
                    ExprResult::Value(tuple)
                }
            }
            Definition::Function(func_id) => {
                let ssa_func_id = self.function_mapper.get(func_id)
                    .unwrap_or_else(|| panic!("Undefined function: {:?}", func_id));
                // Return a function pointer constant
                let value_id = function.push_fn_ptr_const(*ssa_func_id);
                ExprResult::Value(value_id)
            }
            Definition::Builtin(name) => {
                todo!("Builtin function not yet supported: {}", name)
            }
            Definition::LowLevel(name) => {
                todo!("LowLevel function not yet supported: {}", name)
            }
            Definition::Oracle(name) => {
                todo!("Oracle function not yet supported: {}", name)
            }
            Definition::Global(_) => {
                todo!("Global variables not yet supported")
            }
        }
    }

    fn convert_binary(
        &mut self,
        binary: &Binary,
        function: &mut Function<Empty>,
    ) -> ExprResult {
        let lhs = self.convert_expression(&binary.lhs, function).into_value();
        let rhs = self.convert_expression(&binary.rhs, function).into_value();

        let result = match binary.operator {
            BinaryOpKind::Add => function.push_add(self.current_block, lhs, rhs),
            BinaryOpKind::Subtract => function.push_sub(self.current_block, lhs, rhs),
            BinaryOpKind::Multiply => function.push_mul(self.current_block, lhs, rhs),
            BinaryOpKind::Divide => function.push_div(self.current_block, lhs, rhs),
            BinaryOpKind::Equal => function.push_eq(self.current_block, lhs, rhs),
            BinaryOpKind::NotEqual => {
                let eq = function.push_eq(self.current_block, lhs, rhs);
                function.push_not(self.current_block, eq)
            }
            BinaryOpKind::Less => function.push_lt(self.current_block, lhs, rhs),
            BinaryOpKind::LessEqual => {
                // a <= b is equivalent to !(a > b) which is !(b < a)
                let gt = function.push_lt(self.current_block, rhs, lhs);
                function.push_not(self.current_block, gt)
            }
            BinaryOpKind::Greater => function.push_lt(self.current_block, rhs, lhs),
            BinaryOpKind::GreaterEqual => {
                // a >= b is equivalent to !(a < b)
                let lt = function.push_lt(self.current_block, lhs, rhs);
                function.push_not(self.current_block, lt)
            }
            BinaryOpKind::And => function.push_and(self.current_block, lhs, rhs),
            BinaryOpKind::Or => {
                // TODO: Support Or directly in SSA
                // a || b = !(!a && !b)
                let not_a = function.push_not(self.current_block, lhs);
                let not_b = function.push_not(self.current_block, rhs);
                let and = function.push_and(self.current_block, not_a, not_b);
                function.push_not(self.current_block, and)
            }
            BinaryOpKind::Xor => {
                // TODO: Support Xor directly in SSA
                // a ^ b = (a || b) && !(a && b)
                let not_a = function.push_not(self.current_block, lhs);
                let not_b = function.push_not(self.current_block, rhs);
                let nand_ab = function.push_and(self.current_block, not_a, not_b);
                let or_result = function.push_not(self.current_block, nand_ab);
                let and_result = function.push_and(self.current_block, lhs, rhs);
                let not_and = function.push_not(self.current_block, and_result);
                function.push_and(self.current_block, or_result, not_and)
            }
            BinaryOpKind::Modulo => {
                todo!("Modulo operator not yet supported")
            }
            BinaryOpKind::ShiftLeft | BinaryOpKind::ShiftRight => {
                todo!("Shift operators not yet supported")
            }
        };

        ExprResult::Value(result)
    }

    fn convert_let(
        &mut self,
        let_expr: &Let,
        function: &mut Function<Empty>,
    ) -> ExprResult {
        let value = self.convert_expression(&let_expr.expression, function).into_value();

        if let_expr.mutable {
            // Always use a single pointer for the whole value.
            // Nested field/index updates are handled by the descend-modify-ascend
            // pattern in convert_assign.
            let ast_type = let_expr.expression.return_type()
                .expect("Mutable let binding must have a typed expression");
            let typ = self.type_converter.convert_type(&ast_type);

            // For plain tuples/structs without arrays, use flattened pointers
            // (one per field) since downstream passes expect this form.
            // For everything else, use a single pointer with the zipper pattern.
            let types = self.get_flattened_types(&let_expr.expression);
            if types.len() > 1 && !Self::noir_type_contains_array(&ast_type) {
                let ptrs: Vec<_> = types.iter().enumerate()
                    .map(|(i, typ)| {
                        let field = function.push_tuple_proj(self.current_block, value, TupleIdx::Static(i));
                        let ptr = function.push_alloc(self.current_block, typ.clone(), Empty);
                        function.push_store(self.current_block, ptr, field);
                        ptr
                    })
                    .collect();
                self.bindings.insert(let_expr.id, ptrs);
            } else {
                let ptr = function.push_alloc(self.current_block, typ, Empty);
                function.push_store(self.current_block, ptr, value);
                self.bindings.insert(let_expr.id, vec![ptr]);
            }
            self.mutable_locals.insert(let_expr.id);
        } else {
            // Immutable - store single materialized value
            self.bindings.insert(let_expr.id, vec![value]);
        }
        ExprResult::Unit
    }

    fn convert_block(
        &mut self,
        exprs: &[Expression],
        function: &mut Function<Empty>,
    ) -> ExprResult {
        let mut last_result = ExprResult::Unit;
        for expr in exprs {
            last_result = self.convert_expression(expr, function);
        }
        last_result
    }

    fn convert_assign(
        &mut self,
        assign: &Assign,
        function: &mut Function<Empty>,
    ) -> ExprResult {
        use noirc_frontend::monomorphization::ast::Type as AstType;

        let new_value = self.convert_expression(&assign.expression, function).into_value();

        // Flatten the lvalue into a root pointer(s) + access path
        let (root_ptrs, root_type, steps) = self.flatten_lvalue(&assign.lvalue, function);

        if steps.is_empty() {
            // Simple variable assignment — store directly
            if root_ptrs.len() == 1 {
                function.push_store(self.current_block, root_ptrs[0], new_value);
            } else {
                // Flattened tuple binding — project each field and store
                for (i, ptr) in root_ptrs.iter().enumerate() {
                    let field = function.push_tuple_proj(self.current_block, new_value, TupleIdx::Static(i));
                    function.push_store(self.current_block, *ptr, field);
                }
            }
            return ExprResult::Unit;
        }

        assert_eq!(root_ptrs.len(), 1, "Nested lvalue access requires single-pointer binding");
        let root_ptr = root_ptrs[0];

        // Phase 1: Descend — load root, then extract at each level
        let mut values = Vec::with_capacity(steps.len() + 1);
        let mut types = Vec::with_capacity(steps.len() + 1);

        values.push(function.push_load(self.current_block, root_ptr));
        types.push(root_type);

        for step in &steps {
            let parent = *values.last().unwrap();
            let child = match step {
                AccessStep::Index(idx) => {
                    function.push_array_get(self.current_block, parent, *idx)
                }
                AccessStep::Field(field_idx) => {
                    function.push_tuple_proj(self.current_block, parent, TupleIdx::Static(*field_idx))
                }
            };
            let child_type = match (step, types.last().unwrap()) {
                (AccessStep::Index(_), AstType::Array(_, elem)) => elem.as_ref().clone(),
                (AccessStep::Field(idx), AstType::Tuple(fields)) => fields[*idx].clone(),
                (step, ty) => panic!("Type mismatch in lvalue: step {:?} on type {:?}", step, ty),
            };
            values.push(child);
            types.push(child_type);
        }

        // Phase 2: Replace leaf
        *values.last_mut().unwrap() = new_value;

        // Phase 3: Ascend — reconstruct from leaf to root
        for k in (0..steps.len()).rev() {
            let modified_child = values[k + 1];
            let parent = values[k];
            values[k] = match &steps[k] {
                AccessStep::Index(idx) => {
                    function.push_array_set(self.current_block, parent, *idx, modified_child)
                }
                AccessStep::Field(field_idx) => {
                    self.synthesize_tuple_set(
                        parent, *field_idx, modified_child, &types[k], function,
                    )
                }
            };
        }

        // Phase 4: Store reconstructed root
        function.push_store(self.current_block, root_ptr, values[0]);
        ExprResult::Unit
    }

    /// Synthesize a tuple "set" by projecting all fields, replacing one, and rebuilding.
    fn synthesize_tuple_set(
        &mut self,
        tuple: ValueId,
        target_field: usize,
        new_value: ValueId,
        noir_type: &noirc_frontend::monomorphization::ast::Type,
        function: &mut Function<Empty>,
    ) -> ValueId {
        use noirc_frontend::monomorphization::ast::Type as AstType;

        let fields = match noir_type {
            AstType::Tuple(fields) => fields,
            _ => panic!("synthesize_tuple_set called on non-tuple type: {:?}", noir_type),
        };

        let mut elements = Vec::with_capacity(fields.len());
        let mut element_types = Vec::with_capacity(fields.len());

        for (i, field_type) in fields.iter().enumerate() {
            let elem = if i == target_field {
                new_value
            } else {
                function.push_tuple_proj(self.current_block, tuple, TupleIdx::Static(i))
            };
            elements.push(elem);
            element_types.push(self.type_converter.convert_type(field_type));
        }

        function.push_mk_tuple(self.current_block, elements, element_types)
    }

    /// Read an LValue as a value (for Dereference: get the pointer that the lvalue holds).
    fn convert_lvalue_to_value(
        &mut self,
        lvalue: &LValue,
        function: &mut Function<Empty>,
    ) -> ValueId {
        match lvalue {
            LValue::Ident(ident) => {
                self.convert_ident(ident, function).into_value()
            }
            LValue::Dereference { reference, .. } => {
                let ptr = self.convert_lvalue_to_value(reference, function);
                function.push_load(self.current_block, ptr)
            }
            _ => panic!("Unsupported lvalue in dereference position: {:?}", std::mem::discriminant(lvalue)),
        }
    }

    /// Flatten an LValue into (root_pointers, root_noir_type, access_steps).
    /// Walks the LValue tree to the root Ident and collects steps in root-to-leaf order.
    /// Returns multiple pointers for flattened tuple/struct bindings (no nested access),
    /// or a single pointer for types that use the zipper pattern.
    fn flatten_lvalue(
        &mut self,
        lvalue: &LValue,
        function: &mut Function<Empty>,
    ) -> (Vec<ValueId>, noirc_frontend::monomorphization::ast::Type, Vec<AccessStep>) {
        match lvalue {
            LValue::Ident(ident) => {
                match &ident.definition {
                    Definition::Local(local_id) => {
                        if self.mutable_locals.contains(local_id) {
                            let ptrs = self.bindings.get(local_id)
                                .unwrap_or_else(|| panic!("Undefined mutable local: {:?}", local_id))
                                .clone();
                            (ptrs, ident.typ.clone(), vec![])
                        } else {
                            panic!("Cannot assign to immutable local variable: {}", ident.name)
                        }
                    }
                    _ => panic!("Cannot assign to non-local: {:?}", ident.definition),
                }
            }
            LValue::Index { array, index, .. } => {
                let (root_ptrs, root_type, mut steps) = self.flatten_lvalue(array, function);
                let idx_value = self.convert_expression(index, function).into_value();
                steps.push(AccessStep::Index(idx_value));
                (root_ptrs, root_type, steps)
            }
            LValue::MemberAccess { object, field_index } => {
                let (root_ptrs, root_type, mut steps) = self.flatten_lvalue(object, function);
                steps.push(AccessStep::Field(*field_index));
                (root_ptrs, root_type, steps)
            }
            LValue::Dereference { reference, element_type } => {
                // *b where b holds a pointer — evaluate the inner lvalue as an
                // expression to get the pointer value, then use it as root
                let ptr = self.convert_lvalue_to_value(reference, function);
                (vec![ptr], element_type.clone(), vec![])
            }
            LValue::Clone(inner) => {
                self.flatten_lvalue(inner, function)
            }
        }
    }

    fn convert_for(
        &mut self,
        for_expr: &For,
        function: &mut Function<Empty>,
    ) -> ExprResult {
        // Evaluate start and end range in the current block
        let start = self.convert_expression(&for_expr.start_range, function).into_value();
        let end = self.convert_expression(&for_expr.end_range, function).into_value();

        // Create blocks for the loop structure
        let loop_header = function.add_block();
        let loop_body = function.add_block();
        let exit_block = function.add_block();

        // Convert the index type
        let index_type = self.type_converter.convert_type(&for_expr.index_type);

        // Add the loop index as a parameter to the header block
        let loop_index = function.add_parameter(loop_header, index_type);

        // Jump from current block to loop header with start value
        function.terminate_block_with_jmp(self.current_block, loop_header, vec![start]);

        // In the loop header: check if index < end
        let cond = function.push_lt(loop_header, loop_index, end);
        function.terminate_block_with_jmp_if(loop_header, cond, loop_body, exit_block);

        // In the loop body: bind the index variable and execute the block
        self.current_block = loop_body;
        self.bindings.insert(for_expr.index_variable, vec![loop_index]);

        let index_bit_size = self.type_converter.convert_type(&for_expr.index_type).get_bit_size();

        // Push loop context for break/continue
        self.loop_stack.push(LoopContext {
            loop_header,
            exit_block,
            loop_index,
            index_bit_size,
        });

        // Execute the loop body
        self.convert_expression(&for_expr.block, function);

        self.loop_stack.pop();

        // Increment the index and jump back to header
        // (only if current block is not already terminated by break/continue)
        if !function.block_is_terminated(self.current_block) {
            let one = function.push_u_const(index_bit_size, 1);
            let next_index = function.push_add(self.current_block, loop_index, one);
            function.terminate_block_with_jmp(self.current_block, loop_header, vec![next_index]);
        }

        // Continue in the exit block
        self.current_block = exit_block;

        // For loops don't produce a value
        ExprResult::Unit
    }

    /// Try to evaluate a boolean expression to a compile-time constant.
    /// Used to fold `if is_unconstrained()` / `if !is_unconstrained()`.
    fn try_eval_const_bool(&self, expr: &Expression) -> Option<bool> {
        match expr {
            Expression::Unary(unary) if matches!(unary.operator, noirc_frontend::ast::UnaryOp::Not) && !unary.skip => {
                self.try_eval_const_bool(&unary.rhs).map(|b| !b)
            }
            Expression::Call(call) => {
                if let Expression::Ident(ident) = call.func.as_ref() {
                    if let Definition::Builtin(name) = &ident.definition {
                        if name == "is_unconstrained" {
                            return Some(self.in_unconstrained);
                        }
                    }
                }
                None
            }
            _ => None,
        }
    }

    fn convert_if(
        &mut self,
        if_expr: &If,
        function: &mut Function<Empty>,
    ) -> ExprResult {
        use noirc_frontend::monomorphization::ast::Type as AstType;

        // Fold constant boolean conditions (e.g. if !is_unconstrained())
        // to avoid emitting dead branches that contain unsupported operations.
        if let Some(known) = self.try_eval_const_bool(&if_expr.condition) {
            if known {
                return self.convert_expression(&if_expr.consequence, function);
            } else if let Some(alt) = &if_expr.alternative {
                return self.convert_expression(alt, function);
            } else {
                return ExprResult::Unit;
            }
        }

        let condition = self.convert_expression(&if_expr.condition, function).into_value();

        let then_block = function.add_block();
        let else_block = function.add_block();
        let merge_block = function.add_block();

        function.terminate_block_with_jmp_if(self.current_block, condition, then_block, else_block);

        let is_unit = matches!(if_expr.typ, AstType::Unit);

        // Then branch
        self.current_block = then_block;
        let then_result = self.convert_expression(&if_expr.consequence, function);
        let then_value = if is_unit { None } else { Some(then_result.into_value()) };
        let then_exit = self.current_block;

        // Else branch
        self.current_block = else_block;
        let else_value = if let Some(alt) = &if_expr.alternative {
            let else_result = self.convert_expression(alt, function);
            if is_unit { None } else { Some(else_result.into_value()) }
        } else {
            None
        };
        let else_exit = self.current_block;

        if is_unit {
            function.terminate_block_with_jmp(then_exit, merge_block, vec![]);
            function.terminate_block_with_jmp(else_exit, merge_block, vec![]);
            self.current_block = merge_block;
            ExprResult::Unit
        } else {
            let result_type = self.type_converter.convert_type(&if_expr.typ);
            let merge_param = function.add_parameter(merge_block, result_type);
            function.terminate_block_with_jmp(then_exit, merge_block, vec![then_value.unwrap()]);
            function.terminate_block_with_jmp(else_exit, merge_block, vec![else_value.unwrap()]);
            self.current_block = merge_block;
            ExprResult::Value(merge_param)
        }
    }

    fn convert_unary(
        &mut self,
        unary: &noirc_frontend::monomorphization::ast::Unary,
        function: &mut Function<Empty>,
    ) -> ExprResult {
        if unary.skip {
            return self.convert_expression(&unary.rhs, function);
        }
        match unary.operator {
            noirc_frontend::ast::UnaryOp::Reference { .. } => {
                // &mut x on a let-mut local: return the ptr directly (no load)
                if let Expression::Ident(ident) = unary.rhs.as_ref() {
                    if let Definition::Local(local_id) = &ident.definition {
                        if self.mutable_locals.contains(local_id) {
                            let ptrs = self.bindings.get(local_id).unwrap().clone();
                            assert_eq!(ptrs.len(), 1, "&mut on flattened tuple binding not supported");
                            return ExprResult::Value(ptrs[0]);
                        }
                    }
                }
                // Non-mutable-local: evaluate and alloc a fresh Ref
                let value = self.convert_expression(&unary.rhs, function).into_value();
                let inner_type = self.type_converter.convert_type(
                    &unary.rhs.return_type().expect("Reference operand must have a type")
                );
                let ptr = function.push_alloc(self.current_block, inner_type, Empty);
                function.push_store(self.current_block, ptr, value);
                ExprResult::Value(ptr)
            }
            noirc_frontend::ast::UnaryOp::Dereference { .. } => {
                // *x — load from the pointer
                let value = self.convert_expression(&unary.rhs, function).into_value();
                ExprResult::Value(function.push_load(self.current_block, value))
            }
            _ => {
                let value = self.convert_expression(&unary.rhs, function).into_value();
                let result = match unary.operator {
                    noirc_frontend::ast::UnaryOp::Not => {
                        function.push_not(self.current_block, value)
                    }
                    noirc_frontend::ast::UnaryOp::Minus => {
                        let zero = function.push_field_const(ark_bn254::Fr::from(0u64));
                        function.push_sub(self.current_block, zero, value)
                    }
                    _ => unreachable!(),
                };
                ExprResult::Value(result)
            }
        }
    }

    fn convert_index(
        &mut self,
        index: &Index,
        function: &mut Function<Empty>,
    ) -> ExprResult {
        let collection = self.convert_expression(&index.collection, function).into_value();
        let idx = self.convert_expression(&index.index, function).into_value();
        let result = function.push_array_get(self.current_block, collection, idx);
        ExprResult::Value(result)
    }

    fn convert_extract_tuple_field(
        &mut self,
        tuple_expr: &Expression,
        idx: usize,
        function: &mut Function<Empty>,
    ) -> ExprResult {
        // Expressions always return materialized tuples, so just use projection
        let tuple = self.convert_expression(tuple_expr, function).into_value();
        let result = function.push_tuple_proj(self.current_block, tuple, TupleIdx::Static(idx));
        ExprResult::Value(result)
    }

    fn convert_cast(
        &mut self,
        cast: &noirc_frontend::monomorphization::ast::Cast,
        function: &mut Function<Empty>,
    ) -> ExprResult {
        use noirc_frontend::monomorphization::ast::Type as AstType;

        let value = self.convert_expression(&cast.lhs, function).into_value();

        let src_bits = match cast.lhs.return_type().as_deref() {
            Some(AstType::Field) => 254,
            Some(AstType::Integer(_, bit_size)) => bit_size.bit_size() as usize,
            Some(AstType::Bool) => 1,
            _ => 0,
        };

        let (target, target_bits) = match &cast.r#type {
            AstType::Field => (CastTarget::Field, 254),
            AstType::Integer(_, bit_size) => {
                let bits = bit_size.bit_size() as usize;
                (CastTarget::U(bits), bits)
            }
            AstType::Bool => (CastTarget::U(1), 1),
            _ => panic!("Unsupported cast target type: {:?}", cast.r#type),
        };

        // Narrowing cast: truncate first, then cast
        let value = if src_bits > 0 && target_bits < src_bits {
            function.push_truncate(self.current_block, value, target_bits, src_bits)
        } else {
            value
        };

        let result = function.push_cast(self.current_block, value, target);
        ExprResult::Value(result)
    }

    fn convert_constrain(
        &mut self,
        constraint_expr: &Expression,
        function: &mut Function<Empty>,
    ) -> ExprResult {
        // Special case: if the constraint is a binary equality, emit AssertEq directly
        if let Expression::Binary(binary) = constraint_expr {
            if binary.operator == BinaryOpKind::Equal {
                let lhs = self.convert_expression(&binary.lhs, function).into_value();
                let rhs = self.convert_expression(&binary.rhs, function).into_value();
                function.push_assert_eq(self.current_block, lhs, rhs);
                return ExprResult::Unit;
            }
        }

        // General case: the constraint expression must evaluate to true (1)
        let result = self.convert_expression(constraint_expr, function).into_value();
        let one = function.push_u_const(1, 1);
        function.push_assert_eq(self.current_block, result, one);
        ExprResult::Unit
    }

    fn convert_literal(
        &mut self,
        lit: &noirc_frontend::monomorphization::ast::Literal,
        function: &mut Function<Empty>,
    ) -> ExprResult {
        use noirc_frontend::monomorphization::ast::Literal;

        match lit {
            Literal::Bool(b) => {
                let value = if *b { 1 } else { 0 };
                ExprResult::Value(function.push_u_const(1, value))
            }
            Literal::Integer(signed_field, typ, _location) => {
                use noirc_frontend::monomorphization::ast::Type as AstType;

                match typ {
                    AstType::Field => {
                        // Convert SignedField to ark_bn254::Fr directly (both backed by same type)
                        let field_element = signed_field.to_field_element();
                        let field_val = field_element.into_repr();
                        ExprResult::Value(function.push_field_const(field_val))
                    }
                    AstType::Integer(signedness, bit_size) => {
                        use noirc_frontend::shared::Signedness;
                        if *signedness == Signedness::Signed {
                            todo!("Signed integer literals not yet supported");
                        }
                        let bits: usize = bit_size.bit_size() as usize;
                        // Get the value as u128
                        let value = signed_field.to_u128();
                        ExprResult::Value(function.push_u_const(bits, value))
                    }
                    AstType::Bool => {
                        let value = signed_field.to_u128();
                        ExprResult::Value(function.push_u_const(1, value))
                    }
                    _ => panic!("Unexpected type for integer literal: {:?}", typ),
                }
            }
            Literal::Unit => ExprResult::Unit,
            Literal::Array(array_lit) | Literal::Slice(array_lit) => {
                self.convert_array_literal(array_lit, function)
            }
            Literal::Str(_) => todo!("String literals not yet supported"),
            Literal::FmtStr(_, _, _) => todo!("Format string literals not yet supported"),
        }
    }

    fn convert_array_literal(
        &mut self,
        array_lit: &noirc_frontend::monomorphization::ast::ArrayLiteral,
        function: &mut Function<Empty>,
    ) -> ExprResult {
        // Get the element type from the array/slice type
        let (arr_len, elem_ast_type) = match &array_lit.typ {
            noirc_frontend::monomorphization::ast::Type::Array(len, elem_type) => (Some(*len), elem_type.as_ref()),
            noirc_frontend::monomorphization::ast::Type::Slice(elem_type) => (None, elem_type.as_ref()),
            _ => panic!("Expected array/slice type for array literal, got {:?}", array_lit.typ),
        };

        // Convert each element, materializing tuples if needed
        let elements: Vec<ValueId> = array_lit.contents
            .iter()
            .map(|e| {
                let result = self.convert_expression(e, function);
                self.materialize_if_tuple(result, elem_ast_type, function)
            })
            .collect();

        let seq_type = match arr_len {
            Some(len) => SeqType::Array(len as usize),
            None => SeqType::Slice,
        };
        let elem_type = self.type_converter.convert_type(elem_ast_type);

        let result = function.push_mk_array(
            self.current_block,
            elements,
            seq_type,
            elem_type,
        );
        ExprResult::Value(result)
    }

    /// Materialize a tuple value if the result has multiple values
    fn materialize_if_tuple(
        &self,
        result: ExprResult,
        typ: &noirc_frontend::monomorphization::ast::Type,
        function: &mut Function<Empty>,
    ) -> ValueId {
        match result {
            ExprResult::Value(v) => v,
            ExprResult::Values(values) => {
                // Need to construct a tuple from the values
                let types = self.flatten_type(typ);
                function.push_mk_tuple(self.current_block, values, types)
            }
            ExprResult::Unit => panic!("Cannot materialize unit value"),
        }
    }

    fn convert_tuple(
        &mut self,
        exprs: &[Expression],
        function: &mut Function<Empty>,
    ) -> ExprResult {
        if exprs.is_empty() {
            // Empty struct/tuple — still a value (e.g. A {})
            let tuple = function.push_mk_tuple(self.current_block, vec![], vec![]);
            return ExprResult::Value(tuple);
        }

        // Convert each element to a single materialized value
        let values: Vec<ValueId> = exprs
            .iter()
            .map(|e| self.convert_expression(e, function).into_value())
            .collect();

        // Get types for each element
        // Note: For Definition::Function idents, the Noir type may be
        // Tuple([Function, Function]) (constrained + unconstrained pair),
        // but convert_ident produces a single scalar FnPtr value.
        // Use Type::function() to match the actual value produced.
        let types: Vec<_> = exprs
            .iter()
            .map(|e| {
                if let Expression::Ident(ident) = e {
                    if matches!(&ident.definition, Definition::Function(_)) {
                        return Type::function(Empty);
                    }
                }
                let return_type = e.return_type().expect("Tuple element must have a type");
                self.type_converter.convert_type(&return_type)
            })
            .collect();

        // Always construct a materialized tuple
        let tuple = function.push_mk_tuple(self.current_block, values, types);
        ExprResult::Value(tuple)
    }

    fn convert_call(
        &mut self,
        call: &noirc_frontend::monomorphization::ast::Call,
        function: &mut Function<Empty>,
    ) -> ExprResult {
        // Determine the function being called
        match call.func.as_ref() {
            Expression::Ident(ident) => {
                match &ident.definition {
                    Definition::Function(func_id) => {
                        let args: Vec<ValueId> = call.arguments
                            .iter()
                            .map(|arg| self.convert_expression(arg, function).into_value())
                            .collect();

                        let ssa_func_id = self.function_mapper.get(func_id)
                            .unwrap_or_else(|| panic!("Undefined function: {:?}", func_id));

                        // Return size is 1 for tuples (they're returned as a single value)
                        // and 0 for unit
                        let return_type = &call.return_type;
                        let return_size = self.return_size(return_type);

                        let results = function.push_call(self.current_block, *ssa_func_id, args, return_size);

                        if results.is_empty() {
                            ExprResult::Unit
                        } else {
                            // Always a single value (tuples are materialized)
                            ExprResult::Value(results[0])
                        }
                    }
                    // Builtin/LowLevel calls handle their own argument conversion
                    // since some arguments (e.g. string messages) must be skipped
                    Definition::Builtin(name) => {
                        self.convert_builtin_call(name, call, function)
                    }
                    Definition::LowLevel(name) => {
                        self.convert_lowlevel_call(name, call, function)
                    }
                    _ => todo!("Call to {:?} not yet supported", ident.definition),
                }
            }
            _ => {
                // Indirect call through a function pointer
                let args: Vec<ValueId> = call.arguments
                    .iter()
                    .map(|arg| self.convert_expression(arg, function).into_value())
                    .collect();

                let fn_ptr = self.convert_expression(&call.func, function).into_value();
                let return_type = &call.return_type;
                let return_size = self.return_size(return_type);

                let results = function.push_call_indirect(self.current_block, fn_ptr, args, return_size);

                if results.is_empty() {
                    ExprResult::Unit
                } else {
                    // Always a single value (tuples are materialized)
                    ExprResult::Value(results[0])
                }
            }
        }
    }

    fn convert_builtin_call(
        &mut self,
        name: &str,
        call: &noirc_frontend::monomorphization::ast::Call,
        function: &mut Function<Empty>,
    ) -> ExprResult {
        match name {
            "assert_eq" => {
                let lhs = self.convert_expression(&call.arguments[0], function).into_value();
                let rhs = self.convert_expression(&call.arguments[1], function).into_value();
                function.push_assert_eq(self.current_block, lhs, rhs);
                ExprResult::Unit
            }
            "static_assert" => {
                // static_assert(condition, message) - drop the string message
                let cond = self.convert_expression(&call.arguments[0], function).into_value();
                let t = function.push_u_const(1, 1);
                function.push_assert_eq(self.current_block, cond, t);
                ExprResult::Unit
            }
            "array_len" => {
                let arg_type = call.arguments[0].return_type()
                    .expect("array_len argument must have a known type");
                match arg_type.as_ref() {
                    noirc_frontend::monomorphization::ast::Type::Array(len, _) => {
                        let value = function.push_u_const(32, *len as u128);
                        ExprResult::Value(value)
                    }
                    noirc_frontend::monomorphization::ast::Type::Slice(_) => {
                        let slice = self.convert_expression(&call.arguments[0], function).into_value();
                        let value = function.push_slice_len(self.current_block, slice);
                        ExprResult::Value(value)
                    }
                    _ => panic!("array_len called on non-array/slice type: {:?}", arg_type),
                }
            }
            "to_le_radix" => {
                // to_le_radix(value, radix) -> [u8; N]
                let input = self.convert_expression(&call.arguments[0], function).into_value();
                let radix = self.convert_expression(&call.arguments[1], function).into_value();
                let output_size = match call.return_type {
                    noirc_frontend::monomorphization::ast::Type::Array(len, _) => len as usize,
                    _ => panic!("to_le_radix must return an array, got {:?}", call.return_type),
                };
                let result = function.push_to_radix(
                    self.current_block,
                    input,
                    Radix::Dyn(radix),
                    Endianness::Little,
                    output_size,
                );
                ExprResult::Value(result)
            }
            "to_be_radix" => {
                // to_be_radix(value, radix) -> [u8; N]
                let input = self.convert_expression(&call.arguments[0], function).into_value();
                let radix = self.convert_expression(&call.arguments[1], function).into_value();
                let output_size = match call.return_type {
                    noirc_frontend::monomorphization::ast::Type::Array(len, _) => len as usize,
                    _ => panic!("to_be_radix must return an array, got {:?}", call.return_type),
                };
                let result = function.push_to_radix(
                    self.current_block,
                    input,
                    Radix::Dyn(radix),
                    Endianness::Big,
                    output_size,
                );
                ExprResult::Value(result)
            }
            "apply_range_constraint" => {
                // apply_range_constraint(value, bit_size) - range check
                let _value = self.convert_expression(&call.arguments[0], function).into_value();
                let _bit_size = self.convert_expression(&call.arguments[1], function).into_value();
                // TODO: emit range constraint instruction
                ExprResult::Unit
            }
            "is_unconstrained" => {
                let value = function.push_u_const(1, if self.in_unconstrained { 1 } else { 0 });
                ExprResult::Value(value)
            }
            "as_witness" => {
                // No-op hint, just evaluate the argument and discard
                self.convert_expression(&call.arguments[0], function);
                ExprResult::Unit
            }
            "to_le_bits" => {
                let input = self.convert_expression(&call.arguments[0], function).into_value();
                let output_size = match call.return_type {
                    noirc_frontend::monomorphization::ast::Type::Array(len, _) => len as usize,
                    _ => panic!("to_le_bits must return an array, got {:?}", call.return_type),
                };
                let result = function.push_to_bits(self.current_block, input, Endianness::Little, output_size);
                ExprResult::Value(result)
            }
            "to_be_bits" => {
                let input = self.convert_expression(&call.arguments[0], function).into_value();
                let output_size = match call.return_type {
                    noirc_frontend::monomorphization::ast::Type::Array(len, _) => len as usize,
                    _ => panic!("to_be_bits must return an array, got {:?}", call.return_type),
                };
                let result = function.push_to_bits(self.current_block, input, Endianness::Big, output_size);
                ExprResult::Value(result)
            }
            _ => todo!("Builtin function '{}' not yet supported", name),
        }
    }

    fn convert_lowlevel_call(
        &mut self,
        name: &str,
        _call: &noirc_frontend::monomorphization::ast::Call,
        _function: &mut Function<Empty>,
    ) -> ExprResult {
        match name {
            _ => todo!("LowLevel function '{}' not yet supported", name),
        }
    }
}
