//! Converts monomorphized AST expressions to SSA instructions.

use noirc_frontend::{
    ast::BinaryOpKind,
    monomorphization::ast::{
        Assign, Binary, Definition, Expression, For, FuncId as AstFuncId, GlobalId, Ident, If,
        Index, LValue, Let, LocalId, Type as AstType, While,
    },
};

use crate::{
    collections::{HashMap, HashSet},
    compiler::{
        lowering::type_converter::TypeConverter,
        ssa::{
            BlockId, FunctionId, ValueId,
            hlssa::{
                Blob, CastTarget, Constant, Endianness, MAX_SUPPORTED_SIGNED_BITS, Radix,
                SequenceTargetType, Type, TypeExpr,
                builder::{HLEmitter, HLFunctionBuilder},
            },
        },
    },
};

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
    ///
    /// For mutable variables, this stores the pointer to the value. For immutable variables, this
    /// stores the value directly.
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

    /// Current block being emitted into
    current_block: BlockId,
}

impl<'a> ExpressionConverter<'a> {
    pub fn new_with_globals(
        function_mapper: &'a HashMap<AstFuncId, FunctionId>,
        natively_unconstrained: &'a HashSet<AstFuncId>,
        in_unconstrained: bool,
        global_slots: &'a HashMap<GlobalId, usize>,
        entry_block: BlockId,
    ) -> Self {
        Self {
            bindings: HashMap::default(),
            mutable_locals: HashSet::default(),
            function_mapper,
            natively_unconstrained,
            type_converter: TypeConverter::new(),
            loop_stack: Vec::new(),
            in_unconstrained,
            global_slots,
            current_block: entry_block,
        }
    }

    pub fn current_block(&self) -> BlockId {
        self.current_block
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
        b: &mut HLFunctionBuilder<'_>,
    ) {
        let mut e = b.block(self.current_block);
        let ptr = e.alloc(value_id);
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
                    let one = b.emit_const(Constant::U(index_bit_size, 1));
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
                let value_id = b.emit_const(Constant::FnPtr(*ssa_func_id));
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
            BinaryOpKind::Or => e.or(lhs, rhs),
            BinaryOpKind::Xor => e.xor(lhs, rhs),
            BinaryOpKind::Modulo => e.modulo(lhs, rhs),
            BinaryOpKind::ShiftLeft => e.shl(lhs, rhs),
            BinaryOpKind::ShiftRight => e.shr(lhs, rhs),
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
            let mut e = b.block(self.current_block);
            let ptr = e.alloc(value);
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
        let new_value = self.convert_expression(&assign.expression, b).unwrap();
        self.with_lvalue_ref(
            &assign.lvalue,
            b,
            &|this: &mut Self, ptr, b: &mut HLFunctionBuilder<'_>| {
                b.block(this.current_block).store(ptr, new_value);
            },
        );
        None
    }

    fn with_lvalue_ref(
        &mut self,
        lvalue: &LValue,
        b: &mut HLFunctionBuilder<'_>,
        f: &dyn Fn(&mut Self, ValueId, &mut HLFunctionBuilder<'_>),
    ) {
        if let Some(ptr) = self.try_lvalue_ref(lvalue, b) {
            f(self, ptr, b);
            return;
        }

        match lvalue {
            LValue::Ident(_) => panic!("Cannot assign to non-addressable lvalue: {:?}", lvalue),
            LValue::MemberAccess {
                object,
                field_index,
            } => {
                self.with_lvalue_ref(
                    object,
                    b,
                    &|this: &mut Self, tuple_ref, b: &mut HLFunctionBuilder<'_>| {
                        let field_ref = b
                            .block(this.current_block)
                            .tuple_ref_proj(tuple_ref, *field_index);
                        f(this, field_ref, b);
                    },
                );
            }
            LValue::Index {
                array,
                index,
                element_type: _,
                ..
            } => {
                let array_value = self.read_lvalue(array, b);
                let idx = self.convert_expression(index, b).unwrap();
                let element = b.block(self.current_block).array_get(array_value, idx);
                let element_ref = {
                    let mut e = b.block(self.current_block);
                    e.alloc(element)
                };

                f(self, element_ref, b);

                let element = b.block(self.current_block).load(element_ref);
                let updated = b
                    .block(self.current_block)
                    .array_set(array_value, idx, element);
                self.with_lvalue_ref(
                    array,
                    b,
                    &|this: &mut Self, ptr, b: &mut HLFunctionBuilder<'_>| {
                        b.block(this.current_block).store(ptr, updated);
                    },
                );
            }
            LValue::Dereference { .. } => unreachable!("dereference lvalues have refs"),
            LValue::Clone(inner) => self.with_lvalue_ref(inner, b, f),
        }
    }

    fn read_lvalue(&mut self, lvalue: &LValue, b: &mut HLFunctionBuilder<'_>) -> ValueId {
        match lvalue {
            LValue::Ident(ident) => self.convert_ident(ident, b).unwrap(),
            LValue::Dereference { reference, .. } => {
                let ptr = self.read_lvalue(reference, b);
                b.block(self.current_block).load(ptr)
            }
            LValue::MemberAccess {
                object,
                field_index,
            } => {
                if let Some(tuple_ref) = self.try_lvalue_ref(object, b) {
                    let field_ref = b
                        .block(self.current_block)
                        .tuple_ref_proj(tuple_ref, *field_index);
                    return b.block(self.current_block).load(field_ref);
                }

                let tuple = self.read_lvalue(object, b);
                b.block(self.current_block).tuple_proj(tuple, *field_index)
            }
            LValue::Index { array, index, .. } => {
                let array_value = self.read_lvalue(array, b);
                let idx = self.convert_expression(index, b).unwrap();
                b.block(self.current_block).array_get(array_value, idx)
            }
            LValue::Clone(inner) => self.read_lvalue(inner, b),
        }
    }

    fn try_lvalue_ref(
        &mut self,
        lvalue: &LValue,
        b: &mut HLFunctionBuilder<'_>,
    ) -> Option<ValueId> {
        match lvalue {
            LValue::Ident(ident) => match &ident.definition {
                Definition::Local(local_id) if self.mutable_locals.contains(local_id) => {
                    Some(*self.bindings.get(local_id).unwrap())
                }
                _ => None,
            },
            LValue::Dereference { reference, .. } => Some(self.read_lvalue(reference, b)),
            LValue::MemberAccess {
                object,
                field_index,
            } => {
                let tuple_ref = self.try_lvalue_ref(object, b)?;
                Some(
                    b.block(self.current_block)
                        .tuple_ref_proj(tuple_ref, *field_index),
                )
            }
            LValue::Index { .. } => None,
            LValue::Clone(inner) => self.try_lvalue_ref(inner, b),
        }
    }

    fn convert_for(&mut self, for_expr: &For, b: &mut HLFunctionBuilder<'_>) -> Option<ValueId> {
        // Evaluate start and end range in the current block
        let start = self.convert_expression(&for_expr.start_range, b).unwrap();
        let end_raw = self.convert_expression(&for_expr.end_range, b).unwrap();

        let index_type = self.type_converter.convert_type(&for_expr.index_type);

        // if range is inclusive, bump by one
        let end = if for_expr.inclusive {
            let one = b.emit_const(Constant::U(index_type.get_bit_size(), 1));
            b.block(self.current_block).add(end_raw, one)
        } else {
            end_raw
        };

        // Create blocks for the loop structure
        let loop_header = b.add_block(|_| {});
        let loop_body = b.add_block(|_| {});
        let exit_block = b.add_block(|_| {});

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
            let one = b.emit_const(Constant::U(index_bit_size, 1));
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
            if matches!(
                unary.operator,
                noirc_frontend::ast::UnaryOp::Reference { .. }
            ) {
                if let Some(ptr) = self.try_expression_ref(unary.rhs.as_ref(), b) {
                    return Some(ptr);
                }
            }
            return self.convert_expression(&unary.rhs, b);
        }
        match unary.operator {
            noirc_frontend::ast::UnaryOp::Reference { .. } => {
                if let Some(ptr) = self.try_expression_ref(unary.rhs.as_ref(), b) {
                    return Some(ptr);
                }

                // General case: evaluate the expression, alloc a fresh Ref, store into it.
                let value = self.convert_expression(&unary.rhs, b).unwrap();
                let mut e = b.block(self.current_block);
                let ptr = e.alloc(value);
                Some(ptr)
            }
            noirc_frontend::ast::UnaryOp::Dereference { .. } => {
                let value = self.convert_expression(&unary.rhs, b).unwrap();
                Some(b.block(self.current_block).load(value))
            }
            _ => {
                let value = self.convert_expression(&unary.rhs, b).unwrap();
                let result = match unary.operator {
                    noirc_frontend::ast::UnaryOp::Dereference { .. } => unreachable!(),
                    noirc_frontend::ast::UnaryOp::Not => b.block(self.current_block).not(value),
                    noirc_frontend::ast::UnaryOp::Minus => {
                        use noirc_frontend::monomorphization::ast::Type as AstType;
                        let zero_const = match unary.rhs.return_type().as_deref() {
                            Some(AstType::Integer(
                                noirc_frontend::shared::Signedness::Signed,
                                bit_size,
                            )) => Constant::I(bit_size.bit_size() as usize, 0),
                            Some(AstType::Integer(
                                noirc_frontend::shared::Signedness::Unsigned,
                                bit_size,
                            )) => Constant::U(bit_size.bit_size() as usize, 0),
                            _ => Constant::Field(ark_bn254::Fr::from(0u64)),
                        };
                        let zero = b.emit_const(zero_const);
                        b.block(self.current_block).sub(zero, value)
                    }
                    _ => unreachable!(),
                };
                Some(result)
            }
        }
    }

    fn try_expression_ref(
        &mut self,
        expr: &Expression,
        b: &mut HLFunctionBuilder<'_>,
    ) -> Option<ValueId> {
        match expr {
            Expression::Ident(ident) => match &ident.definition {
                Definition::Local(local_id) if self.mutable_locals.contains(local_id) => {
                    Some(*self.bindings.get(local_id).unwrap())
                }
                Definition::Local(local_id)
                    if Self::reference_pointee_type(&ident.typ).is_some() =>
                {
                    Some(*self.bindings.get(local_id).unwrap())
                }
                _ => None,
            },
            Expression::Unary(unary) => {
                if unary.skip {
                    self.try_expression_ref(unary.rhs.as_ref(), b)
                } else if matches!(
                    unary.operator,
                    noirc_frontend::ast::UnaryOp::Dereference { .. }
                ) {
                    Some(self.convert_expression(unary.rhs.as_ref(), b).unwrap())
                } else {
                    None
                }
            }
            Expression::ExtractTupleField(tuple_expr, idx) => {
                let tuple_ref = self.try_tuple_expression_ref(tuple_expr.as_ref(), b)?;
                Some(b.block(self.current_block).tuple_ref_proj(tuple_ref, *idx))
            }
            Expression::Clone(inner) => self.try_expression_ref(inner.as_ref(), b),
            _ => {
                if Self::reference_pointee_type(&Self::expression_type(expr)?).is_some() {
                    self.convert_expression(expr, b)
                } else {
                    None
                }
            }
        }
    }

    fn try_tuple_expression_ref(
        &mut self,
        expr: &Expression,
        b: &mut HLFunctionBuilder<'_>,
    ) -> Option<ValueId> {
        match Self::expression_type(expr)? {
            AstType::Reference(inner, _) if matches!(inner.as_ref(), AstType::Tuple(_)) => {
                self.convert_expression(expr, b)
            }
            AstType::Tuple(_) => self.try_expression_ref(expr, b),
            _ => None,
        }
    }

    fn reference_pointee_type(typ: &AstType) -> Option<AstType> {
        match typ {
            AstType::Reference(inner, _) => Some(*inner.clone()),
            _ => None,
        }
    }

    fn expression_type(expr: &Expression) -> Option<AstType> {
        match expr {
            Expression::Clone(inner) => Self::expression_type(inner.as_ref()),
            Expression::Block(exprs) => exprs.last().and_then(Self::expression_type),
            Expression::Tuple(exprs) => exprs
                .iter()
                .map(Self::expression_type)
                .collect::<Option<Vec<_>>>()
                .map(AstType::Tuple),
            Expression::ExtractTupleField(tuple_expr, idx) => {
                match Self::expression_type(tuple_expr.as_ref())? {
                    AstType::Reference(inner, mutable) => match inner.as_ref() {
                        AstType::Tuple(fields) => {
                            Some(AstType::Reference(Box::new(fields[*idx].clone()), mutable))
                        }
                        _ => None,
                    },
                    AstType::Tuple(fields) => Some(fields[*idx].clone()),
                    _ => None,
                }
            }
            _ => expr.return_type().map(|typ| typ.into_owned()),
        }
    }

    fn convert_index(&mut self, index: &Index, b: &mut HLFunctionBuilder<'_>) -> Option<ValueId> {
        let mut collection = self.convert_expression(&index.collection, b).unwrap();
        // If the collection is a reference, load through it first
        if matches!(
            Self::expression_type(&index.collection),
            Some(noirc_frontend::monomorphization::ast::Type::Reference(_, _))
        ) {
            collection = b.block(self.current_block).load(collection);
        }
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
        let value = self.convert_expression(tuple_expr, b).unwrap();
        if matches!(
            Self::expression_type(tuple_expr),
            Some(AstType::Reference(inner, _)) if matches!(inner.as_ref(), AstType::Tuple(_))
        ) {
            return Some(b.block(self.current_block).tuple_ref_proj(value, idx));
        }
        let result = b.block(self.current_block).tuple_proj(value, idx);
        Some(result)
    }

    fn convert_cast(
        &mut self,
        cast: &noirc_frontend::monomorphization::ast::Cast,
        b: &mut HLFunctionBuilder<'_>,
    ) -> Option<ValueId> {
        use noirc_frontend::monomorphization::ast::Type as AstType;
        use noirc_frontend::shared::Signedness;

        let value = self.convert_expression(&cast.lhs, b).unwrap();

        let (src_bits, src_signed) = match cast.lhs.return_type().as_deref() {
            Some(AstType::Field) => (254, false),
            Some(AstType::Integer(signedness, bit_size)) => (
                bit_size.bit_size() as usize,
                *signedness == Signedness::Signed,
            ),
            Some(AstType::Bool) => (1, false),
            _ => (0, false),
        };

        let (target, target_bits) = match &cast.r#type {
            AstType::Field => (CastTarget::Field, 254),
            AstType::Integer(signedness, bit_size) => {
                let bits = bit_size.bit_size() as usize;
                match signedness {
                    Signedness::Unsigned => (CastTarget::U(bits), bits),
                    Signedness::Signed => (CastTarget::I(bits), bits),
                }
            }
            AstType::Bool => (CastTarget::U(1), 1),
            _ => panic!("Unsupported cast target type: {:?}", cast.r#type),
        };

        let mut e = b.block(self.current_block);

        // Narrowing cast: select the low bits first, then cast.
        let value = if src_bits > 0 && target_bits < src_bits {
            e.bit_range(value, 0, target_bits)
        } else {
            value
        };

        // Signed widening: sign-extend before casting
        let value = if src_signed && src_bits > 0 && target_bits > src_bits {
            e.sext(value, src_bits, target_bits)
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
        let result = self.convert_expression(constraint_expr, b).unwrap();
        b.block(self.current_block).assert_bool(result);
        None
    }

    fn convert_literal(
        &mut self,
        lit: &noirc_frontend::monomorphization::ast::Literal,
        b: &mut HLFunctionBuilder<'_>,
    ) -> Option<ValueId> {
        use noirc_frontend::monomorphization::ast::Literal;

        match lit {
            Literal::Bool(_) | Literal::Integer(_, _, _) => {
                Some(b.emit_const(Self::scalar_literal_to_constant(lit).unwrap()))
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
                let elem_ast_type = match typ {
                    AstType::Array(_, elem_type) => elem_type.as_ref(),
                    AstType::Vector(elem_type) => elem_type.as_ref(),
                    _ => panic!(
                        "Expected array/vector type for Repeated literal, got {:?}",
                        typ
                    ),
                };
                let element_val = self.convert_expression(element, b).unwrap();
                let len = *length as usize;
                let seq_type = if *is_vector {
                    SequenceTargetType::Slice
                } else {
                    SequenceTargetType::Array(len)
                };
                let elem_type = self.type_converter.convert_type(elem_ast_type);
                let result =
                    b.block(self.current_block)
                        .mk_repeated(element_val, seq_type, len, elem_type);
                Some(result)
            }
            Literal::Str(s) => {
                // str<N>: array of u8 (UTF-8 bytes)
                let elem_type = Type::u(8);
                let elems = s.bytes().map(|byte| Constant::U(8, byte as u128)).collect();
                let blob = b.emit_const(Constant::Blob(Blob::new(elem_type.clone(), elems)));
                let arr = b.block(self.current_block).mk_seq_of_blob(elem_type, blob);
                Some(arr)
            }
            Literal::FmtStr(fragments, _count, captures) => {
                // fmtstr<N, T>: Tuple(Array(U(32), N), ...T_fields)
                use noirc_frontend::token::FmtStrFragment;

                // Build the codepoint array from fragments
                let mut codepoint_constants = Vec::new();
                for fragment in fragments {
                    let text = match fragment {
                        FmtStrFragment::String(s) => s.clone(),
                        FmtStrFragment::Interpolation(name, _) => format!("{{{name}}}"),
                    };
                    for c in text.chars() {
                        codepoint_constants.push(Constant::U(32, c as u128));
                    }
                }
                let cp_len = codepoint_constants.len();
                let blob =
                    b.emit_const(Constant::Blob(Blob::new(Type::u(32), codepoint_constants)));
                let cp_array = b
                    .block(self.current_block)
                    .mk_seq_of_blob(Type::u(32), blob);

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

        let seq_type = match arr_len {
            Some(len) => SequenceTargetType::Array(len as usize),
            None => SequenceTargetType::Slice,
        };
        let elem_type = self.type_converter.convert_type(elem_ast_type);

        if let Some(elements) = self.const_scalar_array_elements(array_lit, &elem_type) {
            debug_assert!(matches!(seq_type, SequenceTargetType::Array(_)));
            let blob = b.emit_const(Constant::Blob(Blob::new(elem_type.clone(), elements)));
            let result = b.block(self.current_block).mk_seq_of_blob(elem_type, blob);
            return Some(result);
        }

        let elements: Vec<ValueId> = array_lit
            .contents
            .iter()
            .map(|e| self.convert_expression(e, b).unwrap())
            .collect();

        let result = b
            .block(self.current_block)
            .mk_seq(elements, seq_type, elem_type);
        Some(result)
    }

    fn const_scalar_array_elements(
        &self,
        array_lit: &noirc_frontend::monomorphization::ast::ArrayLiteral,
        elem_type: &Type,
    ) -> Option<Vec<Constant>> {
        if !Self::is_const_seq_scalar_type(elem_type) {
            return None;
        }
        if !matches!(
            &array_lit.typ,
            noirc_frontend::monomorphization::ast::Type::Array(_, _)
        ) {
            return None;
        }

        let mut constants = Vec::with_capacity(array_lit.contents.len());
        for expr in &array_lit.contents {
            let constant = Self::scalar_expr_to_constant(expr)?;
            if !Self::constant_matches_type(&constant, elem_type) {
                return None;
            }
            constants.push(constant);
        }
        Some(constants)
    }

    fn scalar_expr_to_constant(expr: &Expression) -> Option<Constant> {
        match expr {
            Expression::Literal(lit) => Self::scalar_literal_to_constant(lit),
            _ => None,
        }
    }

    fn scalar_literal_to_constant(
        lit: &noirc_frontend::monomorphization::ast::Literal,
    ) -> Option<Constant> {
        use noirc_frontend::monomorphization::ast::{Literal, Type as AstType};

        match lit {
            Literal::Bool(bv) => {
                let value = if *bv { 1 } else { 0 };
                Some(Constant::U(1, value))
            }
            Literal::Integer(signed_field, typ, _location) => match typ {
                AstType::Field => {
                    let field_element = signed_field.to_field_element();
                    let field_val = field_element.into_repr();
                    Some(Constant::Field(field_val))
                }
                AstType::Integer(signedness, bit_size) => {
                    use noirc_frontend::shared::Signedness;
                    let bits: usize = bit_size.bit_size() as usize;
                    if *signedness == Signedness::Signed {
                        assert!(
                            bits <= MAX_SUPPORTED_SIGNED_BITS,
                            "signed integers wider than i{MAX_SUPPORTED_SIGNED_BITS} are unsupported"
                        );
                        let signed_val = signed_field.to_i128();
                        let twos_complement = (signed_val as u128) & ((1u128 << bits) - 1);
                        Some(Constant::I(bits, twos_complement))
                    } else {
                        let value = signed_field.to_u128();
                        Some(Constant::U(bits, value))
                    }
                }
                AstType::Bool => {
                    let value = signed_field.to_u128();
                    Some(Constant::U(1, value))
                }
                _ => panic!("Unexpected type for integer literal: {:?}", typ),
            },
            _ => None,
        }
    }

    fn is_const_seq_scalar_type(typ: &Type) -> bool {
        matches!(typ.expr, TypeExpr::Field | TypeExpr::U(_) | TypeExpr::I(_))
    }

    fn constant_matches_type(constant: &Constant, typ: &Type) -> bool {
        match (constant, &typ.expr) {
            (Constant::Field(_), TypeExpr::Field) => true,
            (Constant::U(bits, _), TypeExpr::U(type_bits)) => bits == type_bits,
            (Constant::I(bits, _), TypeExpr::I(type_bits)) => bits == type_bits,
            _ => false,
        }
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
                    Definition::Function(func_id) => self.convert_static_call(func_id, call, b),
                    // Builtin/LowLevel calls handle their own argument conversion
                    // since some arguments (e.g. string messages) must be skipped
                    Definition::Builtin(name) => self.convert_builtin_call(name, call, b),
                    Definition::LowLevel(name) => self.convert_lowlevel_call(name, call, b),
                    Definition::Oracle(name) if name == "print" => None,
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

    fn convert_static_call(
        &mut self,
        func_id: &AstFuncId,
        call: &noirc_frontend::monomorphization::ast::Call,
        b: &mut HLFunctionBuilder<'_>,
    ) -> Option<ValueId> {
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
        let return_size = self.return_size(&call.return_type);

        // Constrained calling unconstrained: emit unconstrained call
        let is_unconstrained_call =
            !self.in_unconstrained && self.natively_unconstrained.contains(func_id);
        let results = if is_unconstrained_call {
            b.block(self.current_block)
                .call_unconstrained(*ssa_func_id, args, return_size)
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
                b.block(self.current_block).assert_bool(cond);
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
                        let value = b.emit_const(Constant::U(32, *len as u128));
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
            "field_less_than" => {
                let lhs = self.convert_expression(&call.arguments[0], b).unwrap();
                let rhs = self.convert_expression(&call.arguments[1], b).unwrap();
                Some(b.block(self.current_block).lt(lhs, rhs))
            }
            "is_unconstrained" => {
                let value = b
                    .ssa
                    .add_const(Constant::U(1, if self.in_unconstrained { 1 } else { 0 }));
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
        // Handle mavros intrinsics directly
        if let Some(result) = self.try_convert_mavros_intrinsic(name, call, b) {
            return Some(result);
        }

        todo!("LowLevel function '{}' not yet supported", name)
    }

    /// Handle mavros-specific foreign functions directly in SSA,
    /// without requiring a Noir replacement crate.
    fn try_convert_mavros_intrinsic(
        &mut self,
        name: &str,
        call: &noirc_frontend::monomorphization::ast::Call,
        b: &mut HLFunctionBuilder<'_>,
    ) -> Option<ValueId> {
        match name {
            "unsafe_cast" => {
                use noirc_frontend::monomorphization::ast::Type as AstType;
                use noirc_frontend::shared::Signedness;

                let target = match &call.return_type {
                    AstType::Field => CastTarget::Field,
                    AstType::Bool => CastTarget::U(1),
                    AstType::Integer(signedness, bit_size) => {
                        let bits = bit_size.bit_size() as usize;
                        match signedness {
                            Signedness::Unsigned => CastTarget::U(bits),
                            Signedness::Signed => CastTarget::I(bits),
                        }
                    }
                    other => panic!("unsafe_cast: unsupported target type {:?}", other),
                };

                let value = self.convert_expression(&call.arguments[0], b).unwrap();
                let result = b.block(self.current_block).cast_to(target, value);
                Some(result)
            }
            "spread_inner" => {
                let bits = Self::extract_const_u32(&call.arguments[1]);
                assert!(
                    bits >= 1 && bits <= 16,
                    "spread: bits must be 1..=16, got {bits}"
                );
                let value = self.convert_expression(&call.arguments[0], b).unwrap();
                let result = b.block(self.current_block).spread(value, bits as u8);
                Some(result)
            }
            "unspread_inner" => {
                let bits = Self::extract_const_u32(&call.arguments[1]);
                assert!(
                    bits >= 1 && bits <= 16,
                    "unspread: bits must be 1..=16, got {bits}"
                );
                let value = self.convert_expression(&call.arguments[0], b).unwrap();
                let (odd, even) = b.block(self.current_block).unspread(value, bits as u8);
                let result = b
                    .block(self.current_block)
                    .mk_tuple(vec![odd, even], vec![Type::u(32), Type::u(32)]);
                Some(result)
            }
            _ => None,
        }
    }

    /// Extract a compile-time constant u32 from a monomorphized expression.
    fn extract_const_u32(expr: &Expression) -> u32 {
        match expr {
            Expression::Literal(noirc_frontend::monomorphization::ast::Literal::Integer(
                signed_field,
                _typ,
                _location,
            )) => signed_field.to_u128() as u32,
            _ => panic!("Expected a constant integer argument, got {:?}", expr),
        }
    }
}
