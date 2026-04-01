//! Converts monomorphized AST expressions to SSA instructions.

use std::collections::{HashMap, HashSet};

use noirc_frontend::ast::BinaryOpKind;
use noirc_frontend::monomorphization::ast::{
    Assign, Binary, Definition, Expression, For, FuncId as AstFuncId, GlobalId, Ident, If, Index,
    LValue, Let, LocalId, While,
};

use crate::compiler::block_builder::{HLEmitter, HLFunctionBuilder};
use crate::compiler::ir::r#type::Type;
use crate::compiler::ssa::{
    BlockId, CastTarget, ConstValue, Endianness, FunctionId, OpCode, Radix, SeqType, ValueId,
};

use super::type_converter::TypeConverter;

/// A LowLevel function replacement: either a single function or a family dispatched by array size.
pub enum LowLevelReplacement {
    Single(AstFuncId),
    ByArraySize(HashMap<u32, AstFuncId>),
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
    /// Only for `for` loops — (index_value, bit_size), used by Continue to increment.
    for_loop_index: Option<(ValueId, usize)>,
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
    /// Set of AST function IDs that are natively unconstrained
    natively_unconstrained: &'a HashSet<AstFuncId>,
    /// Type converter
    type_converter: TypeConverter,
    /// Stack of enclosing loop contexts for break/continue
    loop_stack: Vec<LoopContext>,
    /// Whether the current function is unconstrained
    in_unconstrained: bool,
    /// Maps GlobalId to global slot index
    global_slots: &'a HashMap<GlobalId, usize>,
    /// Dedup cache for constants
    const_cache: HashMap<ConstValue, ValueId>,
    /// Maps LowLevel function name to its replacement
    lowlevel_replacements: &'a HashMap<String, LowLevelReplacement>,
    /// Current block being emitted into
    current_block: BlockId,
}

impl<'a> ExpressionConverter<'a> {
    pub fn new_with_globals(
        function_mapper: &'a HashMap<AstFuncId, FunctionId>,
        natively_unconstrained: &'a HashSet<AstFuncId>,
        in_unconstrained: bool,
        global_slots: &'a HashMap<GlobalId, usize>,
        lowlevel_replacements: &'a HashMap<String, LowLevelReplacement>,
        entry_block: BlockId,
    ) -> Self {
        Self {
            bindings: HashMap::new(),
            mutable_locals: HashSet::new(),
            function_mapper,
            natively_unconstrained,
            type_converter: TypeConverter::new(),
            loop_stack: Vec::new(),
            in_unconstrained,
            global_slots,
            lowlevel_replacements,
            const_cache: HashMap::new(),
            current_block: entry_block,
        }
    }

    pub fn current_block(&self) -> BlockId {
        self.current_block
    }

    fn get_or_create_const(&mut self, b: &mut HLFunctionBuilder<'_>, cv: ConstValue) -> ValueId {
        if let Some(&vid) = self.const_cache.get(&cv) {
            return vid;
        }
        let vid = b.fresh_value();
        let entry = b.function().get_entry_id();
        b.block(entry).emit(OpCode::Const {
            result: vid,
            value: cv.clone(),
        });
        self.const_cache.insert(cv, vid);
        vid
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
        b: &mut HLFunctionBuilder<'_>,
    ) {
        let mut e = b.block(self.current_block);
        let ptr = e.alloc(typ);
        e.store(ptr, value_id);
        drop(e);
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

    /// Convert an expression to SSA instructions.
    pub fn convert_expression(
        &mut self,
        expr: &Expression,
        b: &mut HLFunctionBuilder<'_>,
    ) -> Option<ValueId> {
        match expr {
            Expression::Ident(ident) => self.convert_ident(ident, b),
            Expression::Binary(binary) => self.convert_binary(binary, b),
            Expression::Let(let_expr) => self.convert_let(let_expr, b),
            Expression::Block(exprs) => self.convert_block(exprs, b),
            Expression::Constrain(constraint_expr, _location, _msg) => {
                self.convert_constrain(constraint_expr, b)
            }
            Expression::Semi(inner) => {
                // Semi expressions evaluate the inner expression and discard the result
                self.convert_expression(inner, b);
                None
            }
            Expression::Literal(lit) => self.convert_literal(lit, b),
            Expression::Tuple(exprs) => self.convert_tuple(exprs, b),
            Expression::Call(call) => self.convert_call(call, b),
            Expression::Assign(assign) => self.convert_assign(assign, b),
            Expression::For(for_expr) => self.convert_for(for_expr, b),
            Expression::Index(index) => self.convert_index(index, b),
            Expression::ExtractTupleField(tuple_expr, idx) => {
                self.convert_extract_tuple_field(tuple_expr, *idx, b)
            }
            Expression::Cast(cast) => self.convert_cast(cast, b),
            Expression::If(if_expr) => self.convert_if(if_expr, b),
            Expression::Unary(unary) => self.convert_unary(unary, b),
            Expression::Clone(inner) => self.convert_expression(inner, b),
            Expression::Drop(_) => None,
            Expression::Break => {
                let ctx = self.loop_stack.last().expect("break outside of loop");
                let exit_block = ctx.exit_block;
                b.block(self.current_block)
                    .terminate_jmp(exit_block, vec![]);
                // Create a dead block for any subsequent code
                let dead = b.add_block(|_| {});
                self.current_block = dead;
                None
            }
            Expression::Continue => {
                let ctx = self.loop_stack.last().expect("continue outside of loop");
                let loop_header = ctx.loop_header;
                let for_loop_index = ctx.for_loop_index;
                if let Some((loop_index, index_bit_size)) = for_loop_index {
                    // For loop: increment index and jump back to header
                    let one = self.get_or_create_const(b, ConstValue::U(index_bit_size, 1));
                    let mut e = b.block(self.current_block);
                    let next_index = e.add(loop_index, one);
                    e.terminate_jmp(loop_header, vec![next_index]);
                } else {
                    // While/loop: just jump back to header with no args
                    b.block(self.current_block)
                        .terminate_jmp(loop_header, vec![]);
                }
                // Create a dead block for any subsequent code
                let dead = b.add_block(|_| {});
                self.current_block = dead;
                None
            }
            Expression::While(w) => self.convert_while(w, b),
            Expression::Loop(body) => self.convert_loop(body, b),
            _ => todo!(
                "Expression type not yet supported: {:?}",
                std::mem::discriminant(expr)
            ),
        }
    }

    fn convert_ident(&mut self, ident: &Ident, b: &mut HLFunctionBuilder<'_>) -> Option<ValueId> {
        match &ident.definition {
            Definition::Local(local_id) => {
                let value = *self
                    .bindings
                    .get(local_id)
                    .unwrap_or_else(|| panic!("Undefined local variable: {:?}", local_id));

                // For mutable variables, we need to load from the pointer
                let value = if self.mutable_locals.contains(local_id) {
                    b.block(self.current_block).load(value)
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
                let value_id = self.get_or_create_const(b, ConstValue::FnPtr(*ssa_func_id));
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
                let value = b.block(self.current_block).read_global(slot as u64, typ);
                Some(value)
            }
        }
    }

    fn convert_binary(
        &mut self,
        binary: &Binary,
        b: &mut HLFunctionBuilder<'_>,
    ) -> Option<ValueId> {
        let lhs = self.convert_expression(&binary.lhs, b).unwrap();
        let rhs = self.convert_expression(&binary.rhs, b).unwrap();

        let mut e = b.block(self.current_block);
        let result = match binary.operator {
            BinaryOpKind::Add => e.add(lhs, rhs),
            BinaryOpKind::Subtract => e.sub(lhs, rhs),
            BinaryOpKind::Multiply => e.mul(lhs, rhs),
            BinaryOpKind::Divide => e.div(lhs, rhs),
            BinaryOpKind::Equal => e.eq(lhs, rhs),
            BinaryOpKind::NotEqual => {
                let eq = e.eq(lhs, rhs);
                e.not(eq)
            }
            BinaryOpKind::Less => e.lt(lhs, rhs),
            BinaryOpKind::LessEqual => {
                let gt = e.lt(rhs, lhs);
                e.not(gt)
            }
            BinaryOpKind::Greater => e.lt(rhs, lhs),
            BinaryOpKind::GreaterEqual => {
                let lt = e.lt(lhs, rhs);
                e.not(lt)
            }
            BinaryOpKind::And => e.and(lhs, rhs),
            BinaryOpKind::Or => {
                let not_a = e.not(lhs);
                let not_b = e.not(rhs);
                let and = e.and(not_a, not_b);
                e.not(and)
            }
            BinaryOpKind::Xor => {
                let not_a = e.not(lhs);
                let not_b = e.not(rhs);
                let nand_ab = e.and(not_a, not_b);
                let or_result = e.not(nand_ab);
                let and_result = e.and(lhs, rhs);
                let not_and = e.not(and_result);
                e.and(or_result, not_and)
            }
            BinaryOpKind::Modulo => e.modulo(lhs, rhs),
            BinaryOpKind::ShiftLeft | BinaryOpKind::ShiftRight => {
                todo!("Shift operators not yet supported")
            }
        };

        Some(result)
    }

    fn convert_let(&mut self, let_expr: &Let, b: &mut HLFunctionBuilder<'_>) -> Option<ValueId> {
        let result = self.convert_expression(&let_expr.expression, b);
        let value = result.unwrap_or_else(|| {
            // Unit binding (e.g., `let unit = ()`) — create an empty tuple
            b.block(self.current_block).mk_tuple(vec![], vec![])
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
            let mut e = b.block(self.current_block);
            let ptr = e.alloc(typ);
            e.store(ptr, value);
            drop(e);
            self.bindings.insert(let_expr.id, ptr);
            self.mutable_locals.insert(let_expr.id);
        } else {
            // Immutable - store single materialized value
            self.bindings.insert(let_expr.id, value);
        }
        None
    }

    fn convert_block(
        &mut self,
        exprs: &[Expression],
        b: &mut HLFunctionBuilder<'_>,
    ) -> Option<ValueId> {
        let mut last_result = None;
        for expr in exprs {
            last_result = self.convert_expression(expr, b);
        }
        last_result
    }

    fn convert_assign(
        &mut self,
        assign: &Assign,
        b: &mut HLFunctionBuilder<'_>,
    ) -> Option<ValueId> {
        use noirc_frontend::monomorphization::ast::Type as AstType;

        let new_value = self.convert_expression(&assign.expression, b).unwrap();

        // Flatten the lvalue into a root pointer + access path
        let (root_ptr, root_type, steps) = self.flatten_lvalue(&assign.lvalue, b);

        if steps.is_empty() {
            // Simple variable assignment — store directly
            b.block(self.current_block).store(root_ptr, new_value);
            return None;
        }

        // Phase 1: Descend — load root, then extract at each level
        let mut values = Vec::with_capacity(steps.len() + 1);
        let mut types = Vec::with_capacity(steps.len() + 1);

        {
            let mut e = b.block(self.current_block);
            values.push(e.load(root_ptr));
            types.push(root_type);

            for step in &steps {
                let parent = *values.last().unwrap();
                let child = match step {
                    AccessStep::Index(idx) => e.array_get(parent, *idx),
                    AccessStep::Field(field_idx) => e.tuple_proj(parent, *field_idx),
                };
                let child_type = match (step, types.last().unwrap()) {
                    (AccessStep::Index(_), AstType::Array(_, elem))
                    | (AccessStep::Index(_), AstType::Vector(elem)) => elem.as_ref().clone(),
                    (AccessStep::Field(idx), AstType::Tuple(fields)) => fields[*idx].clone(),
                    (step, ty) => {
                        panic!("Type mismatch in lvalue: step {:?} on type {:?}", step, ty)
                    }
                };
                values.push(child);
                types.push(child_type);
            }
        }

        // Phase 2: Replace leaf
        *values.last_mut().unwrap() = new_value;

        // Phase 3: Ascend — reconstruct from leaf to root
        for k in (0..steps.len()).rev() {
            let modified_child = values[k + 1];
            let parent = values[k];
            values[k] = match &steps[k] {
                AccessStep::Index(idx) => {
                    b.block(self.current_block)
                        .array_set(parent, *idx, modified_child)
                }
                AccessStep::Field(field_idx) => {
                    self.synthesize_tuple_set(parent, *field_idx, modified_child, &types[k], b)
                }
            };
        }

        // Phase 4: Store reconstructed root
        b.block(self.current_block).store(root_ptr, values[0]);
        None
    }

    /// Synthesize a tuple "set" by projecting all fields, replacing one, and rebuilding.
    fn synthesize_tuple_set(
        &mut self,
        tuple: ValueId,
        target_field: usize,
        new_value: ValueId,
        noir_type: &noirc_frontend::monomorphization::ast::Type,
        b: &mut HLFunctionBuilder<'_>,
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

        let mut e = b.block(self.current_block);
        for (i, field_type) in fields.iter().enumerate() {
            let elem = if i == target_field {
                new_value
            } else {
                e.tuple_proj(tuple, i)
            };
            elements.push(elem);
            element_types.push(self.type_converter.convert_type(field_type));
        }

        e.mk_tuple(elements, element_types)
    }

    /// Read an LValue as a value (for Dereference: get the pointer that the lvalue holds).
    fn convert_lvalue_to_value(
        &mut self,
        lvalue: &LValue,
        b: &mut HLFunctionBuilder<'_>,
    ) -> ValueId {
        match lvalue {
            LValue::Ident(ident) => self.convert_ident(ident, b).unwrap(),
            LValue::Dereference { reference, .. } => {
                let ptr = self.convert_lvalue_to_value(reference, b);
                b.block(self.current_block).load(ptr)
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
        b: &mut HLFunctionBuilder<'_>,
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
                let (root_ptr, root_type, mut steps) = self.flatten_lvalue(array, b);
                let idx_value = self.convert_expression(index, b).unwrap();
                steps.push(AccessStep::Index(idx_value));
                (root_ptr, root_type, steps)
            }
            LValue::MemberAccess {
                object,
                field_index,
            } => {
                let (root_ptr, root_type, mut steps) = self.flatten_lvalue(object, b);
                steps.push(AccessStep::Field(*field_index));
                (root_ptr, root_type, steps)
            }
            LValue::Dereference {
                reference,
                element_type,
            } => {
                // *b where b holds a pointer — evaluate the inner lvalue as an
                // expression to get the pointer value, then use it as root
                let ptr = self.convert_lvalue_to_value(reference, b);
                (ptr, element_type.clone(), vec![])
            }
            LValue::Clone(inner) => self.flatten_lvalue(inner, b),
        }
    }

    fn convert_for(&mut self, for_expr: &For, b: &mut HLFunctionBuilder<'_>) -> Option<ValueId> {
        // Evaluate start and end range in the current block
        let start = self.convert_expression(&for_expr.start_range, b).unwrap();
        let end = self.convert_expression(&for_expr.end_range, b).unwrap();

        // Create blocks for the loop structure
        let loop_header = b.add_block(|_| {});
        let loop_body = b.add_block(|_| {});
        let exit_block = b.add_block(|_| {});

        // Convert the index type
        let index_type = self.type_converter.convert_type(&for_expr.index_type);

        // Build header: parameter, condition, branch
        let loop_index = {
            let mut header = b.block(loop_header);
            let loop_index = header.add_parameter(index_type);
            let cond = header.lt(loop_index, end);
            header.terminate_jmp_if(cond, loop_body, exit_block);
            loop_index
        };

        // Jump from current block to loop header with start value
        b.block(self.current_block)
            .terminate_jmp(loop_header, vec![start]);

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
            for_loop_index: Some((loop_index, index_bit_size)),
        });

        // Execute the loop body
        self.convert_expression(&for_expr.block, b);

        self.loop_stack.pop();

        // Increment the index and jump back to header
        // (only if current block is not already terminated by break/continue)
        if !b.block(self.current_block).is_terminated() {
            let one = self.get_or_create_const(b, ConstValue::U(index_bit_size, 1));
            let mut body_end = b.block(self.current_block);
            let next_index = body_end.add(loop_index, one);
            body_end.terminate_jmp(loop_header, vec![next_index]);
        }

        // Continue in the exit block
        self.current_block = exit_block;

        // For loops don't produce a value
        None
    }

    fn convert_while(
        &mut self,
        while_expr: &While,
        b: &mut HLFunctionBuilder<'_>,
    ) -> Option<ValueId> {
        // Create blocks: loop_header evaluates condition, loop_body runs body, exit_block continues
        let loop_header = b.add_block(|_| {});
        let loop_body = b.add_block(|_| {});
        let exit_block = b.add_block(|_| {});

        // Jump from current block to loop header
        b.block(self.current_block)
            .terminate_jmp(loop_header, vec![]);

        // In loop header: evaluate condition, branch
        self.current_block = loop_header;
        let cond = self.convert_expression(&while_expr.condition, b).unwrap();
        b.block(loop_header)
            .terminate_jmp_if(cond, loop_body, exit_block);

        // In loop body: push context, convert body, pop context, jump back to header
        self.current_block = loop_body;
        self.loop_stack.push(LoopContext {
            loop_header,
            exit_block,
            for_loop_index: None,
        });

        self.convert_expression(&while_expr.body, b);

        self.loop_stack.pop();

        // Jump back to header (if not already terminated by break/continue)
        if !b.block(self.current_block).is_terminated() {
            b.block(self.current_block)
                .terminate_jmp(loop_header, vec![]);
        }

        // Continue in exit block
        self.current_block = exit_block;

        // While loops don't produce a value
        None
    }

    fn convert_loop(
        &mut self,
        body: &Expression,
        b: &mut HLFunctionBuilder<'_>,
    ) -> Option<ValueId> {
        // loop { body } — only exits via break
        let loop_block = b.add_block(|_| {});
        let exit_block = b.add_block(|_| {});

        // Jump from current block to loop block
        b.block(self.current_block)
            .terminate_jmp(loop_block, vec![]);

        // In loop block: push context, convert body, pop context, jump back
        self.current_block = loop_block;
        self.loop_stack.push(LoopContext {
            loop_header: loop_block,
            exit_block,
            for_loop_index: None,
        });

        self.convert_expression(body, b);

        self.loop_stack.pop();

        // Jump back to loop block (if not already terminated by break/continue)
        if !b.block(self.current_block).is_terminated() {
            b.block(self.current_block)
                .terminate_jmp(loop_block, vec![]);
        }

        // Continue in exit block (only reachable via break)
        self.current_block = exit_block;

        // Loop doesn't produce a value
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

    fn convert_if(&mut self, if_expr: &If, b: &mut HLFunctionBuilder<'_>) -> Option<ValueId> {
        use noirc_frontend::monomorphization::ast::Type as AstType;

        // Fold constant boolean conditions (e.g. if !is_unconstrained())
        // to avoid emitting dead branches that contain unsupported operations.
        if let Some(known) = self.try_eval_const_bool(&if_expr.condition) {
            if known {
                return self.convert_expression(&if_expr.consequence, b);
            } else if let Some(alt) = &if_expr.alternative {
                return self.convert_expression(alt, b);
            } else {
                return None;
            }
        }

        let condition = self.convert_expression(&if_expr.condition, b).unwrap();

        let then_block = b.add_block(|_| {});
        let else_block = b.add_block(|_| {});
        let merge_block = b.add_block(|_| {});

        b.block(self.current_block)
            .terminate_jmp_if(condition, then_block, else_block);

        let is_unit = matches!(if_expr.typ, AstType::Unit);

        // Then branch
        self.current_block = then_block;
        let then_result = self.convert_expression(&if_expr.consequence, b);
        let then_value = if is_unit {
            None
        } else {
            Some(then_result.unwrap())
        };
        let then_exit = self.current_block;

        // Else branch
        self.current_block = else_block;
        let else_value = if let Some(alt) = &if_expr.alternative {
            let else_result = self.convert_expression(alt, b);
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
            b.block(then_exit).terminate_jmp(merge_block, vec![]);
            b.block(else_exit).terminate_jmp(merge_block, vec![]);
            self.current_block = merge_block;
            None
        } else {
            let result_type = self.type_converter.convert_type(&if_expr.typ);
            let merge_param = b.block(merge_block).add_parameter(result_type);
            b.block(then_exit)
                .terminate_jmp(merge_block, vec![then_value.unwrap()]);
            b.block(else_exit)
                .terminate_jmp(merge_block, vec![else_value.unwrap()]);
            self.current_block = merge_block;
            Some(merge_param)
        }
    }

    fn convert_unary(
        &mut self,
        unary: &noirc_frontend::monomorphization::ast::Unary,
        b: &mut HLFunctionBuilder<'_>,
    ) -> Option<ValueId> {
        if unary.skip {
            return self.convert_expression(&unary.rhs, b);
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
                let value = self.convert_expression(&unary.rhs, b).unwrap();
                let inner_type = self.type_converter.convert_type(
                    &unary
                        .rhs
                        .return_type()
                        .expect("Reference operand must have a type"),
                );
                let mut e = b.block(self.current_block);
                let ptr = e.alloc(inner_type);
                e.store(ptr, value);
                Some(ptr)
            }
            _ => {
                let value = self.convert_expression(&unary.rhs, b).unwrap();
                let result = match unary.operator {
                    noirc_frontend::ast::UnaryOp::Dereference { .. } => {
                        b.block(self.current_block).load(value)
                    }
                    noirc_frontend::ast::UnaryOp::Not => b.block(self.current_block).not(value),
                    noirc_frontend::ast::UnaryOp::Minus => {
                        use noirc_frontend::monomorphization::ast::Type as AstType;
                        let zero_const = match unary.rhs.return_type().as_deref() {
                            Some(AstType::Integer(noirc_frontend::shared::Signedness::Signed, bit_size)) => {
                                ConstValue::U(bit_size.bit_size() as usize, 0)
                            }
                            Some(AstType::Integer(noirc_frontend::shared::Signedness::Unsigned, bit_size)) => {
                                ConstValue::U(bit_size.bit_size() as usize, 0)
                            }
                            _ => ConstValue::Field(ark_bn254::Fr::from(0u64)),
                        };
                        let zero = self.get_or_create_const(b, zero_const);
                        b.block(self.current_block).sub(zero, value)
                    }
                    _ => unreachable!(),
                };
                Some(result)
            }
        }
    }

    fn convert_index(&mut self, index: &Index, b: &mut HLFunctionBuilder<'_>) -> Option<ValueId> {
        let collection = self.convert_expression(&index.collection, b).unwrap();
        let idx = self.convert_expression(&index.index, b).unwrap();
        let result = b.block(self.current_block).array_get(collection, idx);
        Some(result)
    }

    fn convert_extract_tuple_field(
        &mut self,
        tuple_expr: &Expression,
        idx: usize,
        b: &mut HLFunctionBuilder<'_>,
    ) -> Option<ValueId> {
        // Expressions always return materialized tuples, so just use projection
        let tuple = self.convert_expression(tuple_expr, b).unwrap();
        let result = b.block(self.current_block).tuple_proj(tuple, idx);
        Some(result)
    }

    fn convert_cast(
        &mut self,
        cast: &noirc_frontend::monomorphization::ast::Cast,
        b: &mut HLFunctionBuilder<'_>,
    ) -> Option<ValueId> {
        use noirc_frontend::monomorphization::ast::Type as AstType;

        let value = self.convert_expression(&cast.lhs, b).unwrap();

        let src_bits = match cast.lhs.return_type().as_deref() {
            Some(AstType::Field) => 254,
            Some(AstType::Integer(_, bit_size)) => bit_size.bit_size() as usize,
            Some(AstType::Bool) => 1,
            _ => 0,
        };

        let (target, target_bits) = match &cast.r#type {
            AstType::Field => (CastTarget::Field, 254),
            AstType::Integer(signedness, bit_size) => {
                use noirc_frontend::shared::Signedness;
                let bits = bit_size.bit_size() as usize;
                match signedness {
                    Signedness::Unsigned => (CastTarget::U(bits), bits),
                    Signedness::Signed => {
                        if src_bits > 0 && bits > src_bits {
                            panic!("Signed widening casts not yet implemented");
                        }
                        (CastTarget::I(bits), bits)
                    }
                }
            }
            AstType::Bool => (CastTarget::U(1), 1),
            _ => panic!("Unsupported cast target type: {:?}", cast.r#type),
        };

        let mut e = b.block(self.current_block);
        // Narrowing cast: truncate first, then cast
        let value = if src_bits > 0 && target_bits < src_bits {
            e.truncate(value, target_bits, src_bits)
        } else {
            value
        };

        let result = e.cast_to(target, value);
        Some(result)
    }

    fn convert_constrain(
        &mut self,
        constraint_expr: &Expression,
        b: &mut HLFunctionBuilder<'_>,
    ) -> Option<ValueId> {
        // Special case: if the constraint is a binary equality, emit AssertEq directly
        if let Expression::Binary(binary) = constraint_expr {
            if binary.operator == BinaryOpKind::Equal {
                let lhs = self.convert_expression(&binary.lhs, b).unwrap();
                let rhs = self.convert_expression(&binary.rhs, b).unwrap();
                b.block(self.current_block).assert_eq(lhs, rhs);
                return None;
            }
        }

        // General case: the constraint expression must evaluate to true (1)
        let result = self.convert_expression(constraint_expr, b).unwrap();
        let one = self.get_or_create_const(b, ConstValue::U(1, 1));
        b.block(self.current_block).assert_eq(result, one);
        None
    }

    fn convert_literal(
        &mut self,
        lit: &noirc_frontend::monomorphization::ast::Literal,
        b: &mut HLFunctionBuilder<'_>,
    ) -> Option<ValueId> {
        use noirc_frontend::monomorphization::ast::Literal;

        match lit {
            Literal::Bool(bv) => {
                let value = if *bv { 1 } else { 0 };
                Some(self.get_or_create_const(b, ConstValue::U(1, value)))
            }
            Literal::Integer(signed_field, typ, _location) => {
                use noirc_frontend::monomorphization::ast::Type as AstType;

                match typ {
                    AstType::Field => {
                        // Convert SignedField to ark_bn254::Fr directly (both backed by same type)
                        let field_element = signed_field.to_field_element();
                        let field_val = field_element.into_repr();
                        Some(self.get_or_create_const(b, ConstValue::Field(field_val)))
                    }
                    AstType::Integer(signedness, bit_size) => {
                        use noirc_frontend::shared::Signedness;
                        let bits: usize = bit_size.bit_size() as usize;
                        if *signedness == Signedness::Signed {
                            // 2's complement: get the signed value, mask to n bits
                            let signed_val = signed_field.to_i128();
                            let twos_complement = (signed_val as u128) & ((1u128 << bits) - 1);
                            let unsigned_val = self.get_or_create_const(b, ConstValue::U(bits, twos_complement));
                            // Cast to I(bits) so type inference sees this as signed
                            let result = b.block(self.current_block).cast_to(CastTarget::I(bits), unsigned_val);
                            Some(result)
                        } else {
                            // Get the value as u128
                            let value = signed_field.to_u128();
                            Some(self.get_or_create_const(b, ConstValue::U(bits, value)))
                        }
                    }
                    AstType::Bool => {
                        let value = signed_field.to_u128();
                        Some(self.get_or_create_const(b, ConstValue::U(1, value)))
                    }
                    _ => panic!("Unexpected type for integer literal: {:?}", typ),
                }
            }
            Literal::Unit => None,
            Literal::Array(array_lit) | Literal::Vector(array_lit) => {
                self.convert_array_literal(array_lit, b)
            }
            Literal::Repeated {
                element,
                length,
                is_vector,
                typ,
            } => {
                use noirc_frontend::monomorphization::ast::Type as AstType;
                let elem_val = self.convert_expression(element, b).unwrap();
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
                let result = b
                    .block(self.current_block)
                    .mk_seq(elements, seq_type, elem_type);
                Some(result)
            }
            Literal::Str(s) => {
                // str<N>: array of u8 (UTF-8 bytes)
                let elem_type = Type::u(8);
                let len = s.len();
                let elems: Vec<ValueId> = s
                    .bytes()
                    .map(|byte| self.get_or_create_const(b, ConstValue::U(8, byte as u128)))
                    .collect();
                let arr = b
                    .block(self.current_block)
                    .mk_seq(elems, SeqType::Array(len), elem_type);
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
                        codepoints.push(self.get_or_create_const(b, ConstValue::U(32, c as u128)));
                    }
                }
                let cp_len = codepoints.len();
                let cp_array = b.block(self.current_block).mk_seq(
                    codepoints,
                    SeqType::Array(cp_len),
                    Type::u(32),
                );

                // Convert captures (always a Tuple expression) and flatten
                let mut tuple_elems = vec![cp_array];
                let mut elem_types = vec![Type::u(32).array_of(cp_len)];
                if let Expression::Tuple(capture_exprs) = captures.as_ref() {
                    for expr in capture_exprs {
                        let val = self.convert_expression(expr, b).unwrap();
                        tuple_elems.push(val);
                        let typ = expr.return_type().expect("FmtStr capture must have a type");
                        elem_types.push(self.type_converter.convert_type(&typ));
                    }
                }

                let result = b
                    .block(self.current_block)
                    .mk_tuple(tuple_elems, elem_types);
                Some(result)
            }
        }
    }

    fn convert_array_literal(
        &mut self,
        array_lit: &noirc_frontend::monomorphization::ast::ArrayLiteral,
        b: &mut HLFunctionBuilder<'_>,
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
            .map(|e| self.convert_expression(e, b).unwrap())
            .collect();

        let seq_type = match arr_len {
            Some(len) => SeqType::Array(len as usize),
            None => SeqType::Slice,
        };
        let elem_type = self.type_converter.convert_type(elem_ast_type);

        let result = b
            .block(self.current_block)
            .mk_seq(elements, seq_type, elem_type);
        Some(result)
    }

    fn convert_tuple(
        &mut self,
        exprs: &[Expression],
        b: &mut HLFunctionBuilder<'_>,
    ) -> Option<ValueId> {
        if exprs.is_empty() {
            // Empty struct/tuple — still a value (e.g. A {})
            return Some(b.block(self.current_block).mk_tuple(vec![], vec![]));
        }

        // Convert each element to a single materialized value
        let values: Vec<ValueId> = exprs
            .iter()
            .map(|e| self.convert_expression(e, b).unwrap())
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
        let tuple = b.block(self.current_block).mk_tuple(values, types);
        Some(tuple)
    }

    fn convert_call(
        &mut self,
        call: &noirc_frontend::monomorphization::ast::Call,
        b: &mut HLFunctionBuilder<'_>,
    ) -> Option<ValueId> {
        // Determine the function being called
        match call.func.as_ref() {
            Expression::Ident(ident) => {
                match &ident.definition {
                    Definition::Function(func_id) => {
                        let args: Vec<ValueId> = call
                            .arguments
                            .iter()
                            .map(|arg| self.convert_expression(arg, b).unwrap())
                            .collect();

                        let ssa_func_id = self
                            .function_mapper
                            .get(func_id)
                            .unwrap_or_else(|| panic!("Undefined function: {:?}", func_id));

                        // Return size is 1 for tuples (they're returned as a single value)
                        // and 0 for unit
                        let return_type = &call.return_type;
                        let return_size = self.return_size(return_type);

                        // Constrained calling unconstrained: emit unconstrained call
                        let is_unconstrained_call =
                            !self.in_unconstrained && self.natively_unconstrained.contains(func_id);

                        let results = if is_unconstrained_call {
                            b.block(self.current_block).call_unconstrained(
                                *ssa_func_id,
                                args,
                                return_size,
                            )
                        } else {
                            b.block(self.current_block)
                                .call(*ssa_func_id, args, return_size)
                        };

                        if results.is_empty() {
                            None
                        } else {
                            // Always a single value (tuples are materialized)
                            Some(results[0])
                        }
                    }
                    // Builtin/LowLevel calls handle their own argument conversion
                    // since some arguments (e.g. string messages) must be skipped
                    Definition::Builtin(name) => self.convert_builtin_call(name, call, b),
                    Definition::LowLevel(name) => self.convert_lowlevel_call(name, call, b),
                    _ => todo!("Call to {:?} not yet supported", ident.definition),
                }
            }
            _ => {
                // Indirect call through a function pointer
                let args: Vec<ValueId> = call
                    .arguments
                    .iter()
                    .map(|arg| self.convert_expression(arg, b).unwrap())
                    .collect();

                let fn_ptr = self.convert_expression(&call.func, b).unwrap();
                let return_type = &call.return_type;
                let return_size = self.return_size(return_type);

                let results = b
                    .block(self.current_block)
                    .call_indirect(fn_ptr, args, return_size);

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
        b: &mut HLFunctionBuilder<'_>,
    ) -> Option<ValueId> {
        match name {
            "assert_eq" => {
                let lhs = self.convert_expression(&call.arguments[0], b).unwrap();
                let rhs = self.convert_expression(&call.arguments[1], b).unwrap();
                b.block(self.current_block).assert_eq(lhs, rhs);
                None
            }
            "static_assert" => {
                // static_assert(condition, message) - drop the string message
                let cond = self.convert_expression(&call.arguments[0], b).unwrap();
                let t = self.get_or_create_const(b, ConstValue::U(1, 1));
                b.block(self.current_block).assert_eq(cond, t);
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
                        self.convert_expression(&call.arguments[0], b);
                        let value = self.get_or_create_const(b, ConstValue::U(32, *len as u128));
                        Some(value)
                    }
                    noirc_frontend::monomorphization::ast::Type::Vector(_) => {
                        let slice = self.convert_expression(&call.arguments[0], b).unwrap();
                        let value = b.block(self.current_block).slice_len(slice);
                        Some(value)
                    }
                    _ => panic!("array_len called on non-array/slice type: {:?}", arg_type),
                }
            }
            "to_le_radix" => {
                // to_le_radix(value, radix) -> [u8; N]
                let input = self.convert_expression(&call.arguments[0], b).unwrap();
                let radix = self.convert_expression(&call.arguments[1], b).unwrap();
                let output_size = match call.return_type {
                    noirc_frontend::monomorphization::ast::Type::Array(len, _) => len as usize,
                    _ => panic!(
                        "to_le_radix must return an array, got {:?}",
                        call.return_type
                    ),
                };
                let result = b.block(self.current_block).to_radix(
                    input,
                    Radix::Dyn(radix),
                    Endianness::Little,
                    output_size,
                );
                Some(result)
            }
            "to_be_radix" => {
                // to_be_radix(value, radix) -> [u8; N]
                let input = self.convert_expression(&call.arguments[0], b).unwrap();
                let radix = self.convert_expression(&call.arguments[1], b).unwrap();
                let output_size = match call.return_type {
                    noirc_frontend::monomorphization::ast::Type::Array(len, _) => len as usize,
                    _ => panic!(
                        "to_be_radix must return an array, got {:?}",
                        call.return_type
                    ),
                };
                let result = b.block(self.current_block).to_radix(
                    input,
                    Radix::Dyn(radix),
                    Endianness::Big,
                    output_size,
                );
                Some(result)
            }
            "apply_range_constraint" => {
                let value = self.convert_expression(&call.arguments[0], b).unwrap();
                let bit_size = match &call.arguments[1] {
                    Expression::Literal(
                        noirc_frontend::monomorphization::ast::Literal::Integer(sf, _, _),
                    ) => sf.to_u128() as usize,
                    other => panic!(
                        "apply_range_constraint bit_size must be a constant, got {:?}",
                        other
                    ),
                };
                b.block(self.current_block).rangecheck(value, bit_size);
                None
            }
            "is_unconstrained" => {
                let value = self.get_or_create_const(
                    b,
                    ConstValue::U(1, if self.in_unconstrained { 1 } else { 0 }),
                );
                Some(value)
            }
            "as_witness" => {
                // No-op hint, just evaluate the argument and discard
                self.convert_expression(&call.arguments[0], b);
                None
            }
            "to_le_bits" => {
                let input = self.convert_expression(&call.arguments[0], b).unwrap();
                let output_size = match call.return_type {
                    noirc_frontend::monomorphization::ast::Type::Array(len, _) => len as usize,
                    _ => panic!(
                        "to_le_bits must return an array, got {:?}",
                        call.return_type
                    ),
                };
                let result =
                    b.block(self.current_block)
                        .to_bits(input, Endianness::Little, output_size);
                Some(result)
            }
            "to_be_bits" => {
                let input = self.convert_expression(&call.arguments[0], b).unwrap();
                let output_size = match call.return_type {
                    noirc_frontend::monomorphization::ast::Type::Array(len, _) => len as usize,
                    _ => panic!(
                        "to_be_bits must return an array, got {:?}",
                        call.return_type
                    ),
                };
                let result =
                    b.block(self.current_block)
                        .to_bits(input, Endianness::Big, output_size);
                Some(result)
            }
            "as_vector" => {
                let array = self.convert_expression(&call.arguments[0], b).unwrap();
                Some(
                    b.block(self.current_block)
                        .cast_to(CastTarget::ArrayToSlice, array),
                )
            }
            _ => todo!("Builtin function '{}' not yet supported", name),
        }
    }

    fn convert_lowlevel_call(
        &mut self,
        name: &str,
        call: &noirc_frontend::monomorphization::ast::Call,
        b: &mut HLFunctionBuilder<'_>,
    ) -> Option<ValueId> {
        let replacement = self
            .lowlevel_replacements
            .get(name)
            .unwrap_or_else(|| panic!("LowLevel function '{}' has no replacement", name));

        let replacement_id = match replacement {
            LowLevelReplacement::Single(func_id) => func_id,
            LowLevelReplacement::ByArraySize(size_map) => {
                let array_size = match &call.return_type {
                    noirc_frontend::monomorphization::ast::Type::Array(n, _) => *n as u32,
                    _ => panic!("{} expected array return type", name),
                };
                size_map
                    .get(&array_size)
                    .unwrap_or_else(|| panic!("No {} replacement for size {}", name, array_size))
            }
        };

        let ssa_func_id = self
            .function_mapper
            .get(replacement_id)
            .unwrap_or_else(|| panic!("Replacement function not in function_mapper"));

        let args: Vec<ValueId> = call
            .arguments
            .iter()
            .map(|arg| self.convert_expression(arg, b).unwrap())
            .collect();

        let return_size = self.return_size(&call.return_type);
        let results = b
            .block(self.current_block)
            .call(*ssa_func_id, args, return_size);

        if results.is_empty() {
            None
        } else {
            Some(results[0])
        }
    }
}
