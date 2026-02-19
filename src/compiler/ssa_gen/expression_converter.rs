//! Converts monomorphized AST expressions to SSA instructions.

use std::collections::{HashMap, HashSet};

use noirc_frontend::ast::BinaryOpKind;
use noirc_frontend::monomorphization::ast::{
    Assign, Binary, Definition, Expression, For, FuncId as AstFuncId, GlobalId, Ident, If, Index,
    LValue, Let, LocalId,
};

use crate::compiler::ir::r#type::Type;
use crate::compiler::ssa::{
    BlockId, CastTarget, Endianness, Function, FunctionId, Radix, SeqType, TupleIdx, ValueId,
};

use super::type_converter::TypeConverter;

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
    /// Maps LocalId to ValueId for variable bindings.
    /// For mutable variables, this stores the pointer to the value.
    /// For immutable variables, this stores the value directly.
    bindings: HashMap<LocalId, ValueId>,
    /// Tracks which LocalIds are mutable (their binding is a pointer)
    mutable_locals: HashSet<LocalId>,
    /// Maps AST FuncId to SSA FunctionId
    function_mapper: &'a HashMap<AstFuncId, FunctionId>,
    /// Type converter
    type_converter: TypeConverter,
    /// Current block we're building
    current_block: BlockId,
    /// Stack of enclosing loop contexts for break/continue
    loop_stack: Vec<LoopContext>,
    /// Whether the current function is unconstrained
    in_unconstrained: bool,
    /// Maps GlobalId to global slot index
    global_slots: &'a HashMap<GlobalId, usize>,
    /// Maps array size to replacement AstFuncId for poseidon2_permutation
    poseidon2_replacements: &'a HashMap<u32, AstFuncId>,
}

impl<'a> ExpressionConverter<'a> {
    pub fn new_with_globals(
        function_mapper: &'a HashMap<AstFuncId, FunctionId>,
        entry_block: BlockId,
        in_unconstrained: bool,
        global_slots: &'a HashMap<GlobalId, usize>,
        poseidon2_replacements: &'a HashMap<u32, AstFuncId>,
    ) -> Self {
        Self {
            bindings: HashMap::new(),
            mutable_locals: HashSet::new(),
            function_mapper,
            type_converter: TypeConverter::new(),
            current_block: entry_block,
            loop_stack: Vec::new(),
            in_unconstrained,
            global_slots,
            poseidon2_replacements,
        }
    }

    /// Bind an immutable local variable to a value
    pub fn bind_local(&mut self, local_id: LocalId, value: ValueId) {
        self.bindings.insert(local_id, value);
    }

    /// Bind a mutable local variable - allocates a pointer and stores the initial value
    pub fn bind_local_mut(
        &mut self,
        local_id: LocalId,
        value_id: ValueId,
        typ: Type,
        function: &mut Function,
    ) {
        let ptr = function.push_alloc(self.current_block, typ);
        function.push_store(self.current_block, ptr, value_id);
        self.bindings.insert(local_id, ptr);
        self.mutable_locals.insert(local_id);
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
        function: &mut Function,
    ) -> Option<ValueId> {
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
                None
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
            Expression::Drop(_) => None,
            Expression::Break => {
                let ctx = self.loop_stack.last().expect("break outside of loop");
                let exit_block = ctx.exit_block;
                function.terminate_block_with_jmp(self.current_block, exit_block, vec![]);
                // Create a dead block for any subsequent code
                self.current_block = function.add_block();
                None
            }
            Expression::Continue => {
                let ctx = self.loop_stack.last().expect("continue outside of loop");
                let loop_header = ctx.loop_header;
                let loop_index = ctx.loop_index;
                let index_bit_size = ctx.index_bit_size;
                // Increment index and jump back to header
                let one = function.push_u_const(index_bit_size, 1);
                let next_index = function.push_add(self.current_block, loop_index, one);
                function.terminate_block_with_jmp(
                    self.current_block,
                    loop_header,
                    vec![next_index],
                );
                // Create a dead block for any subsequent code
                self.current_block = function.add_block();
                None
            }
            _ => todo!(
                "Expression type not yet supported: {:?}",
                std::mem::discriminant(expr)
            ),
        }
    }

    fn convert_ident(&mut self, ident: &Ident, function: &mut Function) -> Option<ValueId> {
        match &ident.definition {
            Definition::Local(local_id) => {
                let value = *self
                    .bindings
                    .get(local_id)
                    .unwrap_or_else(|| panic!("Undefined local variable: {:?}", local_id));

                // For mutable variables, we need to load from the pointer
                let value = if self.mutable_locals.contains(local_id) {
                    function.push_load(self.current_block, value)
                } else {
                    value
                };

                Some(value)
            }
            Definition::Function(func_id) => {
                let ssa_func_id = self
                    .function_mapper
                    .get(func_id)
                    .unwrap_or_else(|| panic!("Undefined function: {:?}", func_id));
                // Return a function pointer constant
                let value_id = function.push_fn_ptr_const(*ssa_func_id);
                Some(value_id)
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
            Definition::Global(global_id) => {
                let slot = *self
                    .global_slots
                    .get(global_id)
                    .unwrap_or_else(|| panic!("Undefined global: {:?}", global_id));
                let typ = self.type_converter.convert_type(&ident.typ);
                let value = function.push_read_global(self.current_block, slot as u64, typ);
                Some(value)
            }
        }
    }

    fn convert_binary(&mut self, binary: &Binary, function: &mut Function) -> Option<ValueId> {
        let lhs = self.convert_expression(&binary.lhs, function).unwrap();
        let rhs = self.convert_expression(&binary.rhs, function).unwrap();

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

        Some(result)
    }

    fn convert_let(&mut self, let_expr: &Let, function: &mut Function) -> Option<ValueId> {
        let result = self.convert_expression(&let_expr.expression, function);
        let value = result.unwrap_or_else(|| {
            // Unit binding (e.g., `let unit = ()`) — create an empty tuple
            function.push_mk_tuple(self.current_block, vec![], vec![])
        });

        if let_expr.mutable {
            // Use a single pointer for the whole value.
            // Nested field/index updates are handled by the descend-modify-ascend
            // pattern in convert_assign.
            let ast_type = let_expr
                .expression
                .return_type()
                .expect("Mutable let binding must have a typed expression");
            let typ = self.type_converter.convert_type(&ast_type);
            let ptr = function.push_alloc(self.current_block, typ);
            function.push_store(self.current_block, ptr, value);
            self.bindings.insert(let_expr.id, ptr);
            self.mutable_locals.insert(let_expr.id);
        } else {
            // Immutable - store single materialized value
            self.bindings.insert(let_expr.id, value);
        }
        None
    }

    fn convert_block(&mut self, exprs: &[Expression], function: &mut Function) -> Option<ValueId> {
        let mut last_result = None;
        for expr in exprs {
            last_result = self.convert_expression(expr, function);
        }
        last_result
    }

    fn convert_assign(&mut self, assign: &Assign, function: &mut Function) -> Option<ValueId> {
        use noirc_frontend::monomorphization::ast::Type as AstType;

        let new_value = self
            .convert_expression(&assign.expression, function)
            .unwrap();

        // Flatten the lvalue into a root pointer + access path
        let (root_ptr, root_type, steps) = self.flatten_lvalue(&assign.lvalue, function);

        if steps.is_empty() {
            // Simple variable assignment — store directly
            function.push_store(self.current_block, root_ptr, new_value);
            return None;
        }

        // Phase 1: Descend — load root, then extract at each level
        let mut values = Vec::with_capacity(steps.len() + 1);
        let mut types = Vec::with_capacity(steps.len() + 1);

        values.push(function.push_load(self.current_block, root_ptr));
        types.push(root_type);

        for step in &steps {
            let parent = *values.last().unwrap();
            let child = match step {
                AccessStep::Index(idx) => function.push_array_get(self.current_block, parent, *idx),
                AccessStep::Field(field_idx) => function.push_tuple_proj(
                    self.current_block,
                    parent,
                    TupleIdx::Static(*field_idx),
                ),
            };
            let child_type = match (step, types.last().unwrap()) {
                (AccessStep::Index(_), AstType::Array(_, elem))
                | (AccessStep::Index(_), AstType::Vector(elem)) => elem.as_ref().clone(),
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
                AccessStep::Field(field_idx) => self.synthesize_tuple_set(
                    parent,
                    *field_idx,
                    modified_child,
                    &types[k],
                    function,
                ),
            };
        }

        // Phase 4: Store reconstructed root
        function.push_store(self.current_block, root_ptr, values[0]);
        None
    }

    /// Synthesize a tuple "set" by projecting all fields, replacing one, and rebuilding.
    fn synthesize_tuple_set(
        &mut self,
        tuple: ValueId,
        target_field: usize,
        new_value: ValueId,
        noir_type: &noirc_frontend::monomorphization::ast::Type,
        function: &mut Function,
    ) -> ValueId {
        use noirc_frontend::monomorphization::ast::Type as AstType;

        let fields = match noir_type {
            AstType::Tuple(fields) => fields,
            _ => panic!(
                "synthesize_tuple_set called on non-tuple type: {:?}",
                noir_type
            ),
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
    fn convert_lvalue_to_value(&mut self, lvalue: &LValue, function: &mut Function) -> ValueId {
        match lvalue {
            LValue::Ident(ident) => self.convert_ident(ident, function).unwrap(),
            LValue::Dereference { reference, .. } => {
                let ptr = self.convert_lvalue_to_value(reference, function);
                function.push_load(self.current_block, ptr)
            }
            _ => panic!(
                "Unsupported lvalue in dereference position: {:?}",
                std::mem::discriminant(lvalue)
            ),
        }
    }

    /// Flatten an LValue into (root_pointer, root_noir_type, access_steps).
    /// Walks the LValue tree to the root Ident and collects steps in root-to-leaf order.
    fn flatten_lvalue(
        &mut self,
        lvalue: &LValue,
        function: &mut Function,
    ) -> (
        ValueId,
        noirc_frontend::monomorphization::ast::Type,
        Vec<AccessStep>,
    ) {
        match lvalue {
            LValue::Ident(ident) => match &ident.definition {
                Definition::Local(local_id) => {
                    if self.mutable_locals.contains(local_id) {
                        let ptr = *self
                            .bindings
                            .get(local_id)
                            .unwrap_or_else(|| panic!("Undefined mutable local: {:?}", local_id));
                        (ptr, ident.typ.clone(), vec![])
                    } else {
                        panic!("Cannot assign to immutable local variable: {}", ident.name)
                    }
                }
                _ => panic!("Cannot assign to non-local: {:?}", ident.definition),
            },
            LValue::Index { array, index, .. } => {
                let (root_ptr, root_type, mut steps) = self.flatten_lvalue(array, function);
                let idx_value = self.convert_expression(index, function).unwrap();
                steps.push(AccessStep::Index(idx_value));
                (root_ptr, root_type, steps)
            }
            LValue::MemberAccess {
                object,
                field_index,
            } => {
                let (root_ptr, root_type, mut steps) = self.flatten_lvalue(object, function);
                steps.push(AccessStep::Field(*field_index));
                (root_ptr, root_type, steps)
            }
            LValue::Dereference {
                reference,
                element_type,
            } => {
                // *b where b holds a pointer — evaluate the inner lvalue as an
                // expression to get the pointer value, then use it as root
                let ptr = self.convert_lvalue_to_value(reference, function);
                (ptr, element_type.clone(), vec![])
            }
            LValue::Clone(inner) => self.flatten_lvalue(inner, function),
        }
    }

    fn convert_for(&mut self, for_expr: &For, function: &mut Function) -> Option<ValueId> {
        // Evaluate start and end range in the current block
        let start = self
            .convert_expression(&for_expr.start_range, function)
            .unwrap();
        let end = self
            .convert_expression(&for_expr.end_range, function)
            .unwrap();

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
        self.bindings.insert(for_expr.index_variable, loop_index);

        let index_bit_size = self
            .type_converter
            .convert_type(&for_expr.index_type)
            .get_bit_size();

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
        None
    }

    /// Try to evaluate a boolean expression to a compile-time constant.
    /// Used to fold `if is_unconstrained()` / `if !is_unconstrained()`.
    fn try_eval_const_bool(&self, expr: &Expression) -> Option<bool> {
        match expr {
            Expression::Unary(unary)
                if matches!(unary.operator, noirc_frontend::ast::UnaryOp::Not) && !unary.skip =>
            {
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

    fn convert_if(&mut self, if_expr: &If, function: &mut Function) -> Option<ValueId> {
        use noirc_frontend::monomorphization::ast::Type as AstType;

        // Fold constant boolean conditions (e.g. if !is_unconstrained())
        // to avoid emitting dead branches that contain unsupported operations.
        if let Some(known) = self.try_eval_const_bool(&if_expr.condition) {
            if known {
                return self.convert_expression(&if_expr.consequence, function);
            } else if let Some(alt) = &if_expr.alternative {
                return self.convert_expression(alt, function);
            } else {
                return None;
            }
        }

        let condition = self
            .convert_expression(&if_expr.condition, function)
            .unwrap();

        let then_block = function.add_block();
        let else_block = function.add_block();
        let merge_block = function.add_block();

        function.terminate_block_with_jmp_if(self.current_block, condition, then_block, else_block);

        let is_unit = matches!(if_expr.typ, AstType::Unit);

        // Then branch
        self.current_block = then_block;
        let then_result = self.convert_expression(&if_expr.consequence, function);
        let then_value = if is_unit {
            None
        } else {
            Some(then_result.unwrap())
        };
        let then_exit = self.current_block;

        // Else branch
        self.current_block = else_block;
        let else_value = if let Some(alt) = &if_expr.alternative {
            let else_result = self.convert_expression(alt, function);
            if is_unit {
                None
            } else {
                Some(else_result.unwrap())
            }
        } else {
            None
        };
        let else_exit = self.current_block;

        if is_unit {
            function.terminate_block_with_jmp(then_exit, merge_block, vec![]);
            function.terminate_block_with_jmp(else_exit, merge_block, vec![]);
            self.current_block = merge_block;
            None
        } else {
            let result_type = self.type_converter.convert_type(&if_expr.typ);
            let merge_param = function.add_parameter(merge_block, result_type);
            function.terminate_block_with_jmp(then_exit, merge_block, vec![then_value.unwrap()]);
            function.terminate_block_with_jmp(else_exit, merge_block, vec![else_value.unwrap()]);
            self.current_block = merge_block;
            Some(merge_param)
        }
    }

    fn convert_unary(
        &mut self,
        unary: &noirc_frontend::monomorphization::ast::Unary,
        function: &mut Function,
    ) -> Option<ValueId> {
        if unary.skip {
            return self.convert_expression(&unary.rhs, function);
        }
        match unary.operator {
            noirc_frontend::ast::UnaryOp::Reference { .. } => {
                // &mut x on a let-mut local: return the ptr directly (no load)
                if let Expression::Ident(ident) = unary.rhs.as_ref() {
                    if let Definition::Local(local_id) = &ident.definition {
                        if self.mutable_locals.contains(local_id) {
                            let ptr = *self.bindings.get(local_id).unwrap();
                            return Some(ptr);
                        }
                    }
                }
                // &mut expr.field — would require splicing a pointer into an
                // existing allocation (aliasing). Not yet supported.
                if let Expression::ExtractTupleField(inner, _) = unary.rhs.as_ref() {
                    let is_deref = matches!(inner.as_ref(), Expression::Unary(u) if matches!(u.operator, noirc_frontend::ast::UnaryOp::Dereference { .. }));
                    if !is_deref {
                        todo!(
                            "&mut on tuple field requiring pointer splicing not yet supported: {:?}",
                            unary.rhs
                        );
                    }
                }

                // General case: evaluate the expression, alloc a fresh Ref, store into it.
                let value = self.convert_expression(&unary.rhs, function).unwrap();
                let inner_type = self.type_converter.convert_type(
                    &unary
                        .rhs
                        .return_type()
                        .expect("Reference operand must have a type"),
                );
                let ptr = function.push_alloc(self.current_block, inner_type);
                function.push_store(self.current_block, ptr, value);
                Some(ptr)
            }
            _ => {
                let value = self.convert_expression(&unary.rhs, function).unwrap();
                let result = match unary.operator {
                    noirc_frontend::ast::UnaryOp::Dereference { .. } => {
                        // *x — load from the pointer
                        function.push_load(self.current_block, value)
                    }
                    noirc_frontend::ast::UnaryOp::Not => {
                        function.push_not(self.current_block, value)
                    }
                    noirc_frontend::ast::UnaryOp::Minus => {
                        let zero = function.push_field_const(ark_bn254::Fr::from(0u64));
                        function.push_sub(self.current_block, zero, value)
                    }
                    _ => unreachable!(),
                };
                Some(result)
            }
        }
    }

    fn convert_index(&mut self, index: &Index, function: &mut Function) -> Option<ValueId> {
        let collection = self
            .convert_expression(&index.collection, function)
            .unwrap();
        let idx = self.convert_expression(&index.index, function).unwrap();
        let result = function.push_array_get(self.current_block, collection, idx);
        Some(result)
    }

    fn convert_extract_tuple_field(
        &mut self,
        tuple_expr: &Expression,
        idx: usize,
        function: &mut Function,
    ) -> Option<ValueId> {
        // Expressions always return materialized tuples, so just use projection
        let tuple = self.convert_expression(tuple_expr, function).unwrap();
        let result = function.push_tuple_proj(self.current_block, tuple, TupleIdx::Static(idx));
        Some(result)
    }

    fn convert_cast(
        &mut self,
        cast: &noirc_frontend::monomorphization::ast::Cast,
        function: &mut Function,
    ) -> Option<ValueId> {
        use noirc_frontend::monomorphization::ast::Type as AstType;

        let value = self.convert_expression(&cast.lhs, function).unwrap();

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
        Some(result)
    }

    fn convert_constrain(
        &mut self,
        constraint_expr: &Expression,
        function: &mut Function,
    ) -> Option<ValueId> {
        // Special case: if the constraint is a binary equality, emit AssertEq directly
        if let Expression::Binary(binary) = constraint_expr {
            if binary.operator == BinaryOpKind::Equal {
                let lhs = self.convert_expression(&binary.lhs, function).unwrap();
                let rhs = self.convert_expression(&binary.rhs, function).unwrap();
                function.push_assert_eq(self.current_block, lhs, rhs);
                return None;
            }
        }

        // General case: the constraint expression must evaluate to true (1)
        let result = self.convert_expression(constraint_expr, function).unwrap();
        let one = function.push_u_const(1, 1);
        function.push_assert_eq(self.current_block, result, one);
        None
    }

    fn convert_literal(
        &mut self,
        lit: &noirc_frontend::monomorphization::ast::Literal,
        function: &mut Function,
    ) -> Option<ValueId> {
        use noirc_frontend::monomorphization::ast::Literal;

        match lit {
            Literal::Bool(b) => {
                let value = if *b { 1 } else { 0 };
                Some(function.push_u_const(1, value))
            }
            Literal::Integer(signed_field, typ, _location) => {
                use noirc_frontend::monomorphization::ast::Type as AstType;

                match typ {
                    AstType::Field => {
                        // Convert SignedField to ark_bn254::Fr directly (both backed by same type)
                        let field_element = signed_field.to_field_element();
                        let field_val = field_element.into_repr();
                        Some(function.push_field_const(field_val))
                    }
                    AstType::Integer(signedness, bit_size) => {
                        use noirc_frontend::shared::Signedness;
                        if *signedness == Signedness::Signed {
                            todo!("Signed integer literals not yet supported");
                        }
                        let bits: usize = bit_size.bit_size() as usize;
                        // Get the value as u128
                        let value = signed_field.to_u128();
                        Some(function.push_u_const(bits, value))
                    }
                    AstType::Bool => {
                        let value = signed_field.to_u128();
                        Some(function.push_u_const(1, value))
                    }
                    _ => panic!("Unexpected type for integer literal: {:?}", typ),
                }
            }
            Literal::Unit => None,
            Literal::Array(array_lit) | Literal::Vector(array_lit) => {
                self.convert_array_literal(array_lit, function)
            }
            Literal::Repeated {
                element,
                length,
                is_vector,
                typ,
            } => {
                use noirc_frontend::monomorphization::ast::Type as AstType;
                let elem_val = self.convert_expression(element, function).unwrap();
                let elements: Vec<ValueId> =
                    std::iter::repeat(elem_val).take(*length as usize).collect();
                let elem_ast_type = match typ {
                    AstType::Array(_, elem_type) => elem_type.as_ref(),
                    AstType::Vector(elem_type) => elem_type.as_ref(),
                    _ => panic!(
                        "Expected array/vector type for repeated literal, got {:?}",
                        typ
                    ),
                };
                let seq_type = if *is_vector {
                    SeqType::Slice
                } else {
                    SeqType::Array(*length as usize)
                };
                let elem_type = self.type_converter.convert_type(elem_ast_type);
                let result =
                    function.push_mk_array(self.current_block, elements, seq_type, elem_type);
                Some(result)
            }
            Literal::Str(s) => {
                // str<N>: array of u8 (UTF-8 bytes)
                let elem_type = Type::u(8);
                let len = s.len();
                let elems: Vec<ValueId> = s
                    .bytes()
                    .map(|b| function.push_u_const(8, b as u128))
                    .collect();
                let arr = function.push_mk_array(
                    self.current_block,
                    elems,
                    SeqType::Array(len),
                    elem_type,
                );
                Some(arr)
            }
            Literal::FmtStr(fragments, _count, captures) => {
                // fmtstr<N, T>: Tuple(Array(U(32), N), ...T_fields)
                use noirc_frontend::token::FmtStrFragment;

                // Build the codepoint array from fragments
                let mut codepoints = Vec::new();
                for fragment in fragments {
                    let text = match fragment {
                        FmtStrFragment::String(s) => s.clone(),
                        FmtStrFragment::Interpolation(name, _) => format!("{{{name}}}"),
                    };
                    for c in text.chars() {
                        codepoints.push(function.push_u_const(32, c as u128));
                    }
                }
                let cp_len = codepoints.len();
                let cp_array = function.push_mk_array(
                    self.current_block,
                    codepoints,
                    SeqType::Array(cp_len),
                    Type::u(32),
                );

                // Convert captures (always a Tuple expression) and flatten
                let mut tuple_elems = vec![cp_array];
                let mut elem_types = vec![Type::u(32).array_of(cp_len)];
                if let Expression::Tuple(capture_exprs) = captures.as_ref() {
                    for expr in capture_exprs {
                        let val = self.convert_expression(expr, function).unwrap();
                        tuple_elems.push(val);
                        let typ = expr.return_type().expect("FmtStr capture must have a type");
                        elem_types.push(self.type_converter.convert_type(&typ));
                    }
                }

                let result = function.push_mk_tuple(self.current_block, tuple_elems, elem_types);
                Some(result)
            }
        }
    }

    fn convert_array_literal(
        &mut self,
        array_lit: &noirc_frontend::monomorphization::ast::ArrayLiteral,
        function: &mut Function,
    ) -> Option<ValueId> {
        // Get the element type from the array/slice type
        let (arr_len, elem_ast_type) = match &array_lit.typ {
            noirc_frontend::monomorphization::ast::Type::Array(len, elem_type) => {
                (Some(*len), elem_type.as_ref())
            }
            noirc_frontend::monomorphization::ast::Type::Vector(elem_type) => {
                (None, elem_type.as_ref())
            }
            _ => panic!(
                "Expected array/slice type for array literal, got {:?}",
                array_lit.typ
            ),
        };

        let elements: Vec<ValueId> = array_lit
            .contents
            .iter()
            .map(|e| self.convert_expression(e, function).unwrap())
            .collect();

        let seq_type = match arr_len {
            Some(len) => SeqType::Array(len as usize),
            None => SeqType::Slice,
        };
        let elem_type = self.type_converter.convert_type(elem_ast_type);

        let result = function.push_mk_array(self.current_block, elements, seq_type, elem_type);
        Some(result)
    }

    fn convert_tuple(&mut self, exprs: &[Expression], function: &mut Function) -> Option<ValueId> {
        if exprs.is_empty() {
            // Empty struct/tuple — still a value (e.g. A {})
            let tuple = function.push_mk_tuple(self.current_block, vec![], vec![]);
            return Some(tuple);
        }

        // Convert each element to a single materialized value
        let values: Vec<ValueId> = exprs
            .iter()
            .map(|e| self.convert_expression(e, function).unwrap())
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
                        return Type::function();
                    }
                }
                let return_type = e.return_type().expect("Tuple element must have a type");
                self.type_converter.convert_type(&return_type)
            })
            .collect();

        // Always construct a materialized tuple
        let tuple = function.push_mk_tuple(self.current_block, values, types);
        Some(tuple)
    }

    fn convert_call(
        &mut self,
        call: &noirc_frontend::monomorphization::ast::Call,
        function: &mut Function,
    ) -> Option<ValueId> {
        // Determine the function being called
        match call.func.as_ref() {
            Expression::Ident(ident) => {
                match &ident.definition {
                    Definition::Function(func_id) => {
                        let args: Vec<ValueId> = call
                            .arguments
                            .iter()
                            .map(|arg| self.convert_expression(arg, function).unwrap())
                            .collect();

                        let ssa_func_id = self
                            .function_mapper
                            .get(func_id)
                            .unwrap_or_else(|| panic!("Undefined function: {:?}", func_id));

                        // Return size is 1 for tuples (they're returned as a single value)
                        // and 0 for unit
                        let return_type = &call.return_type;
                        let return_size = self.return_size(return_type);

                        let results =
                            function.push_call(self.current_block, *ssa_func_id, args, return_size);

                        if results.is_empty() {
                            None
                        } else {
                            // Always a single value (tuples are materialized)
                            Some(results[0])
                        }
                    }
                    // Builtin/LowLevel calls handle their own argument conversion
                    // since some arguments (e.g. string messages) must be skipped
                    Definition::Builtin(name) => self.convert_builtin_call(name, call, function),
                    Definition::LowLevel(name) => self.convert_lowlevel_call(name, call, function),
                    _ => todo!("Call to {:?} not yet supported", ident.definition),
                }
            }
            _ => {
                // Indirect call through a function pointer
                let args: Vec<ValueId> = call
                    .arguments
                    .iter()
                    .map(|arg| self.convert_expression(arg, function).unwrap())
                    .collect();

                let fn_ptr = self.convert_expression(&call.func, function).unwrap();
                let return_type = &call.return_type;
                let return_size = self.return_size(return_type);

                let results =
                    function.push_call_indirect(self.current_block, fn_ptr, args, return_size);

                if results.is_empty() {
                    None
                } else {
                    // Always a single value (tuples are materialized)
                    Some(results[0])
                }
            }
        }
    }

    fn convert_builtin_call(
        &mut self,
        name: &str,
        call: &noirc_frontend::monomorphization::ast::Call,
        function: &mut Function,
    ) -> Option<ValueId> {
        match name {
            "assert_eq" => {
                let lhs = self
                    .convert_expression(&call.arguments[0], function)
                    .unwrap();
                let rhs = self
                    .convert_expression(&call.arguments[1], function)
                    .unwrap();
                function.push_assert_eq(self.current_block, lhs, rhs);
                None
            }
            "static_assert" => {
                // static_assert(condition, message) - drop the string message
                let cond = self
                    .convert_expression(&call.arguments[0], function)
                    .unwrap();
                let t = function.push_u_const(1, 1);
                function.push_assert_eq(self.current_block, cond, t);
                None
            }
            "array_len" => {
                let arg_type = call.arguments[0]
                    .return_type()
                    .expect("array_len argument must have a known type");
                match arg_type.as_ref() {
                    noirc_frontend::monomorphization::ast::Type::Array(len, _) => {
                        // Evaluate the argument for side effects (e.g., it may be a
                        // function call that emits constraints), then return the
                        // compile-time-known length.
                        self.convert_expression(&call.arguments[0], function);
                        let value = function.push_u_const(32, *len as u128);
                        Some(value)
                    }
                    noirc_frontend::monomorphization::ast::Type::Vector(_) => {
                        let slice = self
                            .convert_expression(&call.arguments[0], function)
                            .unwrap();
                        let value = function.push_slice_len(self.current_block, slice);
                        Some(value)
                    }
                    _ => panic!("array_len called on non-array/slice type: {:?}", arg_type),
                }
            }
            "to_le_radix" => {
                // to_le_radix(value, radix) -> [u8; N]
                let input = self
                    .convert_expression(&call.arguments[0], function)
                    .unwrap();
                let radix = self
                    .convert_expression(&call.arguments[1], function)
                    .unwrap();
                let output_size = match call.return_type {
                    noirc_frontend::monomorphization::ast::Type::Array(len, _) => len as usize,
                    _ => panic!(
                        "to_le_radix must return an array, got {:?}",
                        call.return_type
                    ),
                };
                let result = function.push_to_radix(
                    self.current_block,
                    input,
                    Radix::Dyn(radix),
                    Endianness::Little,
                    output_size,
                );
                Some(result)
            }
            "to_be_radix" => {
                // to_be_radix(value, radix) -> [u8; N]
                let input = self
                    .convert_expression(&call.arguments[0], function)
                    .unwrap();
                let radix = self
                    .convert_expression(&call.arguments[1], function)
                    .unwrap();
                let output_size = match call.return_type {
                    noirc_frontend::monomorphization::ast::Type::Array(len, _) => len as usize,
                    _ => panic!(
                        "to_be_radix must return an array, got {:?}",
                        call.return_type
                    ),
                };
                let result = function.push_to_radix(
                    self.current_block,
                    input,
                    Radix::Dyn(radix),
                    Endianness::Big,
                    output_size,
                );
                Some(result)
            }
            "apply_range_constraint" => {
                let value = self
                    .convert_expression(&call.arguments[0], function)
                    .unwrap();
                let bit_size = match &call.arguments[1] {
                    Expression::Literal(
                        noirc_frontend::monomorphization::ast::Literal::Integer(sf, _, _),
                    ) => sf.to_u128() as usize,
                    other => panic!(
                        "apply_range_constraint bit_size must be a constant, got {:?}",
                        other
                    ),
                };
                function.push_rangecheck(self.current_block, value, bit_size);
                None
            }
            "is_unconstrained" => {
                let value = function.push_u_const(1, if self.in_unconstrained { 1 } else { 0 });
                Some(value)
            }
            "as_witness" => {
                // No-op hint, just evaluate the argument and discard
                self.convert_expression(&call.arguments[0], function);
                None
            }
            "to_le_bits" => {
                let input = self
                    .convert_expression(&call.arguments[0], function)
                    .unwrap();
                let output_size = match call.return_type {
                    noirc_frontend::monomorphization::ast::Type::Array(len, _) => len as usize,
                    _ => panic!(
                        "to_le_bits must return an array, got {:?}",
                        call.return_type
                    ),
                };
                let result = function.push_to_bits(
                    self.current_block,
                    input,
                    Endianness::Little,
                    output_size,
                );
                Some(result)
            }
            "to_be_bits" => {
                let input = self
                    .convert_expression(&call.arguments[0], function)
                    .unwrap();
                let output_size = match call.return_type {
                    noirc_frontend::monomorphization::ast::Type::Array(len, _) => len as usize,
                    _ => panic!(
                        "to_be_bits must return an array, got {:?}",
                        call.return_type
                    ),
                };
                let result =
                    function.push_to_bits(self.current_block, input, Endianness::Big, output_size);
                Some(result)
            }
            "as_vector" => {
                let array = self
                    .convert_expression(&call.arguments[0], function)
                    .unwrap();
                Some(function.push_cast(self.current_block, array, CastTarget::ArrayToSlice))
            }
            _ => todo!("Builtin function '{}' not yet supported", name),
        }
    }

    fn convert_lowlevel_call(
        &mut self,
        name: &str,
        call: &noirc_frontend::monomorphization::ast::Call,
        function: &mut Function,
    ) -> Option<ValueId> {
        match name {
            "poseidon2_permutation" => {
                let array_size = match &call.return_type {
                    noirc_frontend::monomorphization::ast::Type::Array(n, _) => *n as u32,
                    _ => panic!("poseidon2_permutation expected array return type"),
                };
                let replacement_id = self
                    .poseidon2_replacements
                    .get(&array_size)
                    .unwrap_or_else(|| {
                        panic!("No poseidon2 replacement for size {}", array_size)
                    });
                let ssa_func_id = self
                    .function_mapper
                    .get(replacement_id)
                    .unwrap_or_else(|| panic!("Replacement function not in function_mapper"));

                let args: Vec<ValueId> = call
                    .arguments
                    .iter()
                    .map(|arg| self.convert_expression(arg, function).unwrap())
                    .collect();

                let return_size = self.return_size(&call.return_type);
                let results =
                    function.push_call(self.current_block, *ssa_func_id, args, return_size);

                if results.is_empty() {
                    None
                } else {
                    Some(results[0])
                }
            }
            _ => todo!("LowLevel function '{}' not yet supported", name),
        }
    }
}
