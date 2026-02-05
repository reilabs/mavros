//! Converts monomorphized AST expressions to SSA instructions.

use std::collections::HashMap;

use noirc_frontend::ast::BinaryOpKind;
use noirc_frontend::monomorphization::ast::{
    Binary, Definition, Expression, FuncId as AstFuncId, Ident, Let, LocalId,
};

use crate::compiler::ir::r#type::Empty;
use crate::compiler::ssa::{BlockId, Function, FunctionId, ValueId};

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

/// Converts expressions within a single function.
pub struct ExpressionConverter<'a> {
    /// Maps LocalId to ValueId for variable bindings
    bindings: HashMap<LocalId, ValueId>,
    /// Maps AST FuncId to SSA FunctionId
    function_mapper: &'a HashMap<AstFuncId, FunctionId>,
    /// Type converter
    type_converter: AstTypeConverter,
    /// Current block we're building
    current_block: BlockId,
}

impl<'a> ExpressionConverter<'a> {
    pub fn new(
        function_mapper: &'a HashMap<AstFuncId, FunctionId>,
        entry_block: BlockId,
    ) -> Self {
        Self {
            bindings: HashMap::new(),
            function_mapper,
            type_converter: AstTypeConverter::new(),
            current_block: entry_block,
        }
    }

    /// Bind a local variable to a value
    pub fn bind_local(&mut self, local_id: LocalId, value_id: ValueId) {
        self.bindings.insert(local_id, value_id);
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
            _ => todo!("Expression type not yet supported: {:?}", std::mem::discriminant(expr)),
        }
    }

    fn convert_ident(
        &mut self,
        ident: &Ident,
        _function: &mut Function<Empty>,
    ) -> ExprResult {
        match &ident.definition {
            Definition::Local(local_id) => {
                let value_id = self.bindings.get(local_id)
                    .unwrap_or_else(|| panic!("Undefined local variable: {:?}", local_id));
                ExprResult::Value(*value_id)
            }
            Definition::Function(func_id) => {
                let ssa_func_id = self.function_mapper.get(func_id)
                    .unwrap_or_else(|| panic!("Undefined function: {:?}", func_id));
                // Return a function pointer constant
                let value_id = _function.push_fn_ptr_const(*ssa_func_id);
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
                // a || b = !(!a && !b)
                let not_a = function.push_not(self.current_block, lhs);
                let not_b = function.push_not(self.current_block, rhs);
                let and = function.push_and(self.current_block, not_a, not_b);
                function.push_not(self.current_block, and)
            }
            BinaryOpKind::Xor => {
                // a ^ b = (a || b) && !(a && b)
                // But for bits: a ^ b = (a + b) - 2 * (a * b) = a + b - 2ab
                // Simpler: a ^ b = (a != b)
                // For boolean: not(eq(a, b))
                let eq = function.push_eq(self.current_block, lhs, rhs);
                function.push_not(self.current_block, eq)
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
        self.bindings.insert(let_expr.id, value);
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

    fn convert_constrain(
        &mut self,
        constraint_expr: &Expression,
        function: &mut Function<Empty>,
    ) -> ExprResult {
        // The constraint expression should be either:
        // 1. A binary equality check (a == b)
        // 2. A call to assert_eq
        // 3. A boolean expression that should be true

        match constraint_expr {
            Expression::Binary(binary) if binary.operator == BinaryOpKind::Equal => {
                // Direct equality constraint: assert(a == b)
                let lhs = self.convert_expression(&binary.lhs, function).into_value();
                let rhs = self.convert_expression(&binary.rhs, function).into_value();
                function.push_assert_eq(self.current_block, lhs, rhs);
                ExprResult::Unit
            }
            Expression::Call(call) => {
                // Check if it's assert_eq
                if let Expression::Ident(ident) = call.func.as_ref() {
                    if ident.name == "assert_eq" && call.arguments.len() == 2 {
                        let lhs = self.convert_expression(&call.arguments[0], function).into_value();
                        let rhs = self.convert_expression(&call.arguments[1], function).into_value();
                        function.push_assert_eq(self.current_block, lhs, rhs);
                        return ExprResult::Unit;
                    }
                }
                // Other call-based constraints
                let result = self.convert_expression(constraint_expr, function).into_value();
                let one = function.push_u_const(1, 1);
                function.push_assert_eq(self.current_block, result, one);
                ExprResult::Unit
            }
            _ => {
                // General boolean constraint: the expression should evaluate to true (1)
                let result = self.convert_expression(constraint_expr, function).into_value();
                let one = function.push_u_const(1, 1);
                function.push_assert_eq(self.current_block, result, one);
                ExprResult::Unit
            }
        }
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
                use std::str::FromStr;
                use noirc_frontend::monomorphization::ast::Type as AstType;

                match typ {
                    AstType::Field => {
                        // Convert SignedField to ark_bn254::Fr via string representation
                        let field_element = signed_field.to_field_element();
                        let field_str = field_element.to_string();
                        let field_val = ark_bn254::Fr::from_str(&field_str).unwrap();
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
            Literal::Array(_) | Literal::Slice(_) => {
                todo!("Array/Slice literals not yet supported")
            }
            Literal::Str(_) => todo!("String literals not yet supported"),
            Literal::FmtStr(_, _, _) => todo!("Format string literals not yet supported"),
        }
    }

    fn convert_tuple(
        &mut self,
        exprs: &[Expression],
        function: &mut Function<Empty>,
    ) -> ExprResult {
        let values: Vec<ValueId> = exprs
            .iter()
            .map(|e| self.convert_expression(e, function).into_value())
            .collect();

        if values.is_empty() {
            return ExprResult::Unit;
        }

        // Get types for the tuple elements
        let types: Vec<_> = exprs
            .iter()
            .filter_map(|e| e.return_type())
            .map(|t| self.type_converter.convert_type(&t))
            .collect();

        let result = function.push_mk_tuple(self.current_block, values, types);
        ExprResult::Value(result)
    }

    fn convert_call(
        &mut self,
        call: &noirc_frontend::monomorphization::ast::Call,
        function: &mut Function<Empty>,
    ) -> ExprResult {
        // Convert arguments
        let args: Vec<ValueId> = call.arguments
            .iter()
            .map(|arg| self.convert_expression(arg, function).into_value())
            .collect();

        // Determine the function being called
        match call.func.as_ref() {
            Expression::Ident(ident) => {
                match &ident.definition {
                    Definition::Function(func_id) => {
                        let ssa_func_id = self.function_mapper.get(func_id)
                            .unwrap_or_else(|| panic!("Undefined function: {:?}", func_id));

                        // Determine return size
                        let return_type = &call.return_type;
                        let return_size = self.type_converter.convert_type(return_type).calculate_type_size();

                        let results = function.push_call(self.current_block, *ssa_func_id, args, return_size);

                        if results.is_empty() {
                            ExprResult::Unit
                        } else if results.len() == 1 {
                            ExprResult::Value(results[0])
                        } else {
                            ExprResult::Values(results)
                        }
                    }
                    Definition::Builtin(name) => {
                        self.convert_builtin_call(name, &args, call, function)
                    }
                    Definition::LowLevel(name) => {
                        self.convert_lowlevel_call(name, &args, call, function)
                    }
                    _ => todo!("Call to {:?} not yet supported", ident.definition),
                }
            }
            _ => {
                // Indirect call through a function pointer
                let fn_ptr = self.convert_expression(&call.func, function).into_value();
                let return_type = &call.return_type;
                let return_size = self.type_converter.convert_type(return_type).calculate_type_size();

                let results = function.push_call_indirect(self.current_block, fn_ptr, args, return_size);

                if results.is_empty() {
                    ExprResult::Unit
                } else if results.len() == 1 {
                    ExprResult::Value(results[0])
                } else {
                    ExprResult::Values(results)
                }
            }
        }
    }

    fn convert_builtin_call(
        &mut self,
        name: &str,
        args: &[ValueId],
        _call: &noirc_frontend::monomorphization::ast::Call,
        function: &mut Function<Empty>,
    ) -> ExprResult {
        match name {
            "assert_eq" => {
                if args.len() != 2 {
                    panic!("assert_eq expects 2 arguments, got {}", args.len());
                }
                function.push_assert_eq(self.current_block, args[0], args[1]);
                ExprResult::Unit
            }
            _ => todo!("Builtin function '{}' not yet supported", name),
        }
    }

    fn convert_lowlevel_call(
        &mut self,
        name: &str,
        args: &[ValueId],
        _call: &noirc_frontend::monomorphization::ast::Call,
        function: &mut Function<Empty>,
    ) -> ExprResult {
        match name {
            "assert_constant" => {
                // assert_constant is a compile-time check, no runtime code needed
                ExprResult::Unit
            }
            _ => todo!("LowLevel function '{}' not yet supported", name),
        }
    }
}
