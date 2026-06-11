use acvm::{AcirField, FieldElement};
use num_bigint::BigInt;
use num_traits::Zero;

use noirc_frontend::ast::{BinaryOpKind, UnaryOp};
use noirc_frontend::monomorphization::ast::{
    Definition, Expression, GlobalId, LValue, Literal, Type,
};
use noirc_frontend::signed_field::SignedField;

use super::error::InterpretError;
use super::value::{IntValue, Value, field_to_bigint};
use super::{Flow, Frame, Interpreter};

/// A step along an lvalue access path, rooted at a local variable.
enum Step {
    Index(usize),
    Field(usize),
}

impl<'p> Interpreter<'p> {
    /// Evaluate an expression in value position, rejecting a stray `break`/`continue`.
    pub(super) fn eval_expr_value(
        &mut self,
        expr: &'p Expression,
        env: &mut Frame,
    ) -> Result<Value, InterpretError> {
        match self.eval(expr, env)? {
            Flow::Normal(value) => Ok(value),
            Flow::Break | Flow::Continue => Err(InterpretError::Internal(
                "break/continue in value position".to_string(),
            )),
        }
    }

    pub(super) fn eval(
        &mut self,
        expr: &'p Expression,
        env: &mut Frame,
    ) -> Result<Flow, InterpretError> {
        let value = match expr {
            Expression::Ident(ident) => match &ident.definition {
                Definition::Local(local) => env.get(local).cloned().ok_or_else(|| {
                    InterpretError::Internal(format!("unbound local '{}'", ident.name))
                })?,
                Definition::Global(global) => self.eval_global(*global)?,
                Definition::Function(id) => Value::Function(*id),
                Definition::Builtin(name)
                | Definition::LowLevel(name)
                | Definition::Oracle(name) => {
                    return Err(InterpretError::Unsupported(format!(
                        "reference to intrinsic '{name}'"
                    )));
                }
            },

            Expression::Literal(literal) => self.eval_literal(literal, env)?,

            Expression::Block(expressions) => {
                let mut last = Value::Unit;
                for expression in expressions {
                    match self.eval(expression, env)? {
                        Flow::Normal(value) => last = value,
                        other => return Ok(other),
                    }
                }
                last
            }

            Expression::Unary(unary) => {
                if unary.skip {
                    return self.eval(&unary.rhs, env);
                }
                let rhs = self.eval_expr_value(&unary.rhs, env)?;
                self.eval_unary(&unary.operator, rhs)?
            }

            Expression::Binary(binary) => {
                let lhs = self.eval_expr_value(&binary.lhs, env)?;
                let rhs = self.eval_expr_value(&binary.rhs, env)?;
                self.eval_binary(binary.operator, lhs, rhs)?
            }

            Expression::Index(index) => {
                // Evaluate the index before reading the collection: the index expression may
                // mutate the very array being indexed (e.g. `b[{ b[0] = ...; 0 }]`), and the load
                // must see those stores, matching Noir's SSA (array_get reads the current array).
                let i = self.eval_expr_value(&index.index, env)?.as_index()?;
                let collection = self.eval_expr_value(&index.collection, env)?;
                match collection {
                    Value::Array(elements) => elements.get(i).cloned().ok_or_else(|| {
                        InterpretError::Type(format!(
                            "index {i} out of bounds (len {})",
                            elements.len()
                        ))
                    })?,
                    other => {
                        return Err(InterpretError::Type(format!("cannot index {other:?}")));
                    }
                }
            }

            Expression::Cast(cast) => {
                let value = self.eval_expr_value(&cast.lhs, env)?;
                self.eval_cast(value, &cast.r#type)?
            }

            Expression::For(for_) => {
                let (signed, bits) = int_type(&for_.index_type).ok_or_else(|| {
                    InterpretError::Type("for-loop index is not an integer type".to_string())
                })?;
                let start = self.eval_expr_value(&for_.start_range, env)?;
                let end = self.eval_expr_value(&for_.end_range, env)?;
                let start = start.as_int()?.value.clone();
                let end = end.as_int()?.value.clone();
                let end = if for_.inclusive { end + 1 } else { end };
                let mut i = start;
                while i < end {
                    env.insert(
                        for_.index_variable,
                        Value::Int(IntValue::canonical(signed, bits, i.clone())),
                    );
                    match self.eval(&for_.block, env)? {
                        Flow::Break => break,
                        Flow::Continue | Flow::Normal(_) => {}
                    }
                    i += 1;
                }
                Value::Unit
            }

            Expression::Loop(body) => {
                loop {
                    match self.eval(body, env)? {
                        Flow::Break => break,
                        Flow::Continue | Flow::Normal(_) => {}
                    }
                }
                Value::Unit
            }

            Expression::While(while_) => {
                while self.eval_expr_value(&while_.condition, env)?.as_bool()? {
                    match self.eval(&while_.body, env)? {
                        Flow::Break => break,
                        Flow::Continue | Flow::Normal(_) => {}
                    }
                }
                Value::Unit
            }

            Expression::If(if_) => {
                if self.eval_expr_value(&if_.condition, env)?.as_bool()? {
                    return self.eval(&if_.consequence, env);
                }
                match &if_.alternative {
                    Some(alternative) => return self.eval(alternative, env),
                    None => Value::Unit,
                }
            }

            Expression::Tuple(elements) => {
                let mut values = Vec::with_capacity(elements.len());
                for element in elements {
                    values.push(self.eval_expr_value(element, env)?);
                }
                Value::Tuple(values)
            }

            Expression::ExtractTupleField(tuple, field) => {
                let value = self.eval_expr_value(tuple, env)?;
                match value {
                    Value::Tuple(elements) => elements.get(*field).cloned().ok_or_else(|| {
                        InterpretError::Type(format!("tuple field {field} out of bounds"))
                    })?,
                    other => {
                        return Err(InterpretError::Type(format!(
                            "cannot extract field from {other:?}"
                        )));
                    }
                }
            }

            Expression::Call(call) => {
                let func_id = self.resolve_callee(&call.func, env)?;
                let mut args = Vec::with_capacity(call.arguments.len());
                for argument in &call.arguments {
                    args.push(self.eval_expr_value(argument, env)?);
                }
                self.call_function(func_id, args)?
            }

            Expression::Let(let_) => {
                let value = self.eval_expr_value(&let_.expression, env)?;
                env.insert(let_.id, value);
                Value::Unit
            }

            Expression::Constrain(condition, location, message) => {
                if !self.eval_expr_value(condition, env)?.as_bool()? {
                    let message = match message {
                        Some(boxed) => self.render_assert_message(&boxed.0, env),
                        None => None,
                    };
                    return Err(InterpretError::AssertionFailed {
                        location: *location,
                        message,
                    });
                }
                Value::Unit
            }

            Expression::Assign(assign) => {
                let value = self.eval_expr_value(&assign.expression, env)?;
                self.assign_lvalue(&assign.lvalue, value, env)?;
                Value::Unit
            }

            Expression::Semi(inner) => match self.eval(inner, env)? {
                Flow::Normal(_) => Value::Unit,
                other => return Ok(other),
            },

            Expression::Clone(inner) => self.eval_expr_value(inner, env)?,

            Expression::Drop(inner) => match self.eval(inner, env)? {
                Flow::Normal(_) => Value::Unit,
                other => return Ok(other),
            },

            Expression::Break => return Ok(Flow::Break),
            Expression::Continue => return Ok(Flow::Continue),

            Expression::Match(_) => {
                return Err(InterpretError::Unsupported("match expression".to_string()));
            }
        };
        Ok(Flow::Normal(value))
    }

    /// Evaluate a global on first reference and cache it. Globals are evaluated lazily so that
    /// stdlib-injected globals which the program never touches (and which may use constructs the
    /// interpreter does not yet support) are never evaluated.
    fn eval_global(&mut self, id: GlobalId) -> Result<Value, InterpretError> {
        if let Some(value) = self.globals.get(&id) {
            return Ok(value.clone());
        }
        let program = self.program;
        let (_, _, expr) = program
            .globals
            .get(&id)
            .ok_or_else(|| InterpretError::Internal(format!("unknown global {id:?}")))?;
        let mut frame = Frame::new();
        let value = self.eval_expr_value(expr, &mut frame)?;
        self.globals.insert(id, value.clone());
        Ok(value)
    }

    fn eval_literal(
        &mut self,
        literal: &'p Literal,
        env: &mut Frame,
    ) -> Result<Value, InterpretError> {
        let value = match literal {
            Literal::Integer(signed_field, typ, _) => match typ {
                Type::Field => Value::Field(signed_field.to_field_element()),
                Type::Integer(signedness, bits) => {
                    let signed = signedness.is_signed();
                    let value = signed_field_to_bigint(signed_field, signed)?;
                    Value::Int(IntValue::canonical(signed, bits.bit_size(), value))
                }
                other => {
                    return Err(InterpretError::Type(format!(
                        "integer literal with non-numeric type {other:?}"
                    )));
                }
            },
            Literal::Bool(b) => Value::Bool(*b),
            Literal::Unit => Value::Unit,
            Literal::Str(s) => Value::Str(s.clone()),
            Literal::Array(array) | Literal::Vector(array) => {
                let mut elements = Vec::with_capacity(array.contents.len());
                for element in &array.contents {
                    elements.push(self.eval_expr_value(element, env)?);
                }
                Value::Array(elements)
            }
            Literal::Repeated {
                element, length, ..
            } => {
                let value = self.eval_expr_value(element, env)?;
                Value::Array(vec![value; *length as usize])
            }
            Literal::FmtStr(..) => {
                return Err(InterpretError::Unsupported("format string".to_string()));
            }
        };
        Ok(value)
    }

    fn resolve_callee(
        &mut self,
        func: &'p Expression,
        env: &mut Frame,
    ) -> Result<noirc_frontend::monomorphization::ast::FuncId, InterpretError> {
        if let Expression::Ident(ident) = func {
            match &ident.definition {
                Definition::Function(id) => return Ok(*id),
                Definition::Builtin(name) | Definition::LowLevel(name) => {
                    return Err(InterpretError::Unsupported(format!(
                        "intrinsic/builtin call '{name}'"
                    )));
                }
                Definition::Oracle(name) => {
                    return Err(InterpretError::Unsupported(format!("oracle call '{name}'")));
                }
                Definition::Local(_) | Definition::Global(_) => {}
            }
        }
        match self.eval_expr_value(func, env)? {
            Value::Function(id) => Ok(id),
            other => Err(InterpretError::Type(format!(
                "call of non-function {other:?}"
            ))),
        }
    }

    fn eval_unary(&self, op: &UnaryOp, rhs: Value) -> Result<Value, InterpretError> {
        match op {
            UnaryOp::Minus => match rhs {
                Value::Int(int) => Ok(Value::Int(IntValue::checked(
                    int.signed, int.bits, -int.value, "negation",
                )?)),
                Value::Field(field) => Ok(Value::Field(FieldElement::zero() - field)),
                other => Err(InterpretError::Type(format!("cannot negate {other:?}"))),
            },
            UnaryOp::Not => match rhs {
                Value::Bool(b) => Ok(Value::Bool(!b)),
                Value::Int(int) => {
                    // Bitwise NOT: complement the two's-complement bit pattern.
                    let mask = (BigInt::from(1) << int.bits as usize) - 1;
                    let complemented = mask - int.unsigned_repr();
                    Ok(Value::Int(IntValue::canonical(
                        int.signed,
                        int.bits,
                        complemented,
                    )))
                }
                other => Err(InterpretError::Type(format!(
                    "cannot apply `!` to {other:?}"
                ))),
            },
            UnaryOp::Reference { .. } => Err(InterpretError::Unsupported(
                "reference (`&`/`&mut`)".to_string(),
            )),
            UnaryOp::Dereference { .. } => {
                Err(InterpretError::Unsupported("dereference (`*`)".to_string()))
            }
        }
    }

    fn eval_binary(
        &self,
        op: BinaryOpKind,
        lhs: Value,
        rhs: Value,
    ) -> Result<Value, InterpretError> {
        use BinaryOpKind::*;
        match (lhs, rhs) {
            (Value::Int(a), Value::Int(b)) => eval_int_binary(op, a, b),
            (Value::Field(a), Value::Field(b)) => match op {
                Add => Ok(Value::Field(a + b)),
                Subtract => Ok(Value::Field(a - b)),
                Multiply => Ok(Value::Field(a * b)),
                Divide => {
                    if b == FieldElement::zero() {
                        return Err(InterpretError::DivisionByZero);
                    }
                    Ok(Value::Field(a * b.inverse()))
                }
                Equal => Ok(Value::Bool(a == b)),
                NotEqual => Ok(Value::Bool(a != b)),
                _ => Err(InterpretError::Type(
                    "ordering/bitwise operators are not defined on Field".to_string(),
                )),
            },
            (Value::Bool(a), Value::Bool(b)) => match op {
                And => Ok(Value::Bool(a && b)),
                Or => Ok(Value::Bool(a || b)),
                Xor => Ok(Value::Bool(a ^ b)),
                Equal => Ok(Value::Bool(a == b)),
                NotEqual => Ok(Value::Bool(a != b)),
                // Noir orders bools as false < true (same as Rust's `bool: Ord`).
                Less => Ok(Value::Bool(a < b)),
                LessEqual => Ok(Value::Bool(a <= b)),
                Greater => Ok(Value::Bool(a > b)),
                GreaterEqual => Ok(Value::Bool(a >= b)),
                _ => Err(InterpretError::Type(format!(
                    "operator {op:?} not defined on bool"
                ))),
            },
            (lhs, rhs) => Err(InterpretError::Type(format!(
                "binary operator {op:?} on mismatched operands {lhs:?}, {rhs:?}"
            ))),
        }
    }

    fn eval_cast(&self, value: Value, target: &Type) -> Result<Value, InterpretError> {
        match target {
            Type::Field => match value {
                // Noir rejects signed-to-Field casts at type-check (`UnsupportedFieldCast`);
                // inventing a semantics here would silently bless an ill-typed AST.
                Value::Int(int) if int.signed => Err(InterpretError::Type(
                    "cast of a signed integer to Field (rejected by Noir's type checker)"
                        .to_string(),
                )),
                Value::Int(int) => Ok(Value::Field(int.to_field())),
                Value::Bool(b) => Ok(Value::Field(if b {
                    FieldElement::one()
                } else {
                    FieldElement::zero()
                })),
                Value::Field(f) => Ok(Value::Field(f)),
                other => Err(InterpretError::Type(format!(
                    "cannot cast {other:?} to Field"
                ))),
            },
            Type::Integer(signedness, bits) => {
                let signed = signedness.is_signed();
                let width = bits.bit_size();
                let raw = match value {
                    Value::Int(int) => int.value,
                    Value::Bool(b) => BigInt::from(b as u8),
                    // Noir casts Field -> integer by truncating mod 2^bits (see ssa_gen
                    // `insert_safe_cast`). `canonical` does the truncation; `field_to_bigint`
                    // avoids `to_u128`'s panic for field values >= 2^128.
                    Value::Field(f) => field_to_bigint(&f),
                    other => {
                        return Err(InterpretError::Type(format!(
                            "cannot cast {other:?} to an integer"
                        )));
                    }
                };
                Ok(Value::Int(IntValue::canonical(signed, width, raw)))
            }
            Type::Bool => match value {
                Value::Bool(b) => Ok(Value::Bool(b)),
                // Noir rejects numeric-to-bool casts at type-check (`CannotCastNumericToBool`);
                // a `!= 0` semantics here would silently bless an ill-typed AST.
                other => Err(InterpretError::Type(format!(
                    "cannot cast {other:?} to bool (rejected by Noir's type checker)"
                ))),
            },
            other => Err(InterpretError::Unsupported(format!("cast to {other:?}"))),
        }
    }

    /// Best-effort rendering of an assertion's failure message (only static strings for now).
    fn render_assert_message(&mut self, expr: &'p Expression, env: &mut Frame) -> Option<String> {
        match self.eval_expr_value(expr, env) {
            Ok(Value::Str(s)) => Some(s),
            _ => None,
        }
    }

    fn assign_lvalue(
        &mut self,
        lvalue: &'p LValue,
        new: Value,
        env: &mut Frame,
    ) -> Result<(), InterpretError> {
        let (root, path) = self.lvalue_path(lvalue, env)?;
        let current = env
            .get(&root)
            .cloned()
            .ok_or_else(|| InterpretError::Internal("assignment to unbound local".to_string()))?;
        let updated = update_path(current, &path, new)?;
        env.insert(root, updated);
        Ok(())
    }

    fn lvalue_path(
        &mut self,
        lvalue: &'p LValue,
        env: &mut Frame,
    ) -> Result<(noirc_frontend::monomorphization::ast::LocalId, Vec<Step>), InterpretError> {
        match lvalue {
            LValue::Ident(ident) => match &ident.definition {
                Definition::Local(local) => Ok((*local, Vec::new())),
                _ => Err(InterpretError::Unsupported(
                    "assignment to a non-local binding".to_string(),
                )),
            },
            LValue::Index { array, index, .. } => {
                let i = self.eval_expr_value(index, env)?.as_index()?;
                let (root, mut path) = self.lvalue_path(array, env)?;
                path.push(Step::Index(i));
                Ok((root, path))
            }
            LValue::MemberAccess {
                object,
                field_index,
            } => {
                let (root, mut path) = self.lvalue_path(object, env)?;
                path.push(Step::Field(*field_index));
                Ok((root, path))
            }
            LValue::Clone(inner) => self.lvalue_path(inner, env),
            LValue::Dereference { .. } => Err(InterpretError::Unsupported(
                "assignment through a dereference".to_string(),
            )),
        }
    }
}

/// Functionally update a nested value at `path`, returning the new root value.
fn update_path(value: Value, path: &[Step], new: Value) -> Result<Value, InterpretError> {
    let Some((step, rest)) = path.split_first() else {
        return Ok(new);
    };
    match (step, value) {
        (Step::Index(i), Value::Array(mut elements)) => {
            if *i >= elements.len() {
                return Err(InterpretError::Type(format!(
                    "assignment index {i} out of bounds (len {})",
                    elements.len()
                )));
            }
            let child = std::mem::replace(&mut elements[*i], Value::Unit);
            elements[*i] = update_path(child, rest, new)?;
            Ok(Value::Array(elements))
        }
        (Step::Field(i), Value::Tuple(mut elements)) => {
            if *i >= elements.len() {
                return Err(InterpretError::Type(format!(
                    "assignment field {i} out of bounds"
                )));
            }
            let child = std::mem::replace(&mut elements[*i], Value::Unit);
            elements[*i] = update_path(child, rest, new)?;
            Ok(Value::Tuple(elements))
        }
        (_, other) => Err(InterpretError::Type(format!(
            "lvalue path does not match value shape: {other:?}"
        ))),
    }
}

fn eval_int_binary(op: BinaryOpKind, a: IntValue, b: IntValue) -> Result<Value, InterpretError> {
    use BinaryOpKind::*;
    // Both operands of an integer binary op share one type in a well-typed AST (this Noir's
    // `Shl`/`Shr` traits are `fn shl(self, other: Self)`, so shifts included). Computing a
    // mismatch with the lhs's type would be a silent wrong value, so reject it — defence in
    // depth against an ill-typed AST reaching the oracle.
    if a.signed != b.signed || a.bits != b.bits {
        return Err(InterpretError::Type(format!(
            "binary {op:?} on mismatched integer types: (signed={},{}b) vs (signed={},{}b)",
            a.signed, a.bits, b.signed, b.bits
        )));
    }
    // Comparators read the canonical (sign-aware) values directly.
    match op {
        Equal => return Ok(Value::Bool(a.value == b.value)),
        NotEqual => return Ok(Value::Bool(a.value != b.value)),
        Less => return Ok(Value::Bool(a.value < b.value)),
        LessEqual => return Ok(Value::Bool(a.value <= b.value)),
        Greater => return Ok(Value::Bool(a.value > b.value)),
        GreaterEqual => return Ok(Value::Bool(a.value >= b.value)),
        _ => {}
    }

    let signed = a.signed;
    let bits = a.bits;
    let int = match op {
        Add => IntValue::checked(signed, bits, a.value + b.value, "addition")?,
        Subtract => IntValue::checked(signed, bits, a.value - b.value, "subtraction")?,
        Multiply => IntValue::checked(signed, bits, a.value * b.value, "multiplication")?,
        Divide => {
            if b.value.is_zero() {
                return Err(InterpretError::DivisionByZero);
            }
            IntValue::checked(signed, bits, a.value / b.value, "division")?
        }
        Modulo => {
            if b.value.is_zero() {
                return Err(InterpretError::DivisionByZero);
            }
            // Noir's `%` is Rust `checked_rem`, which overflows (errors) on `i_MIN % -1` even
            // though the mathematical remainder is 0. Mirror that single edge case.
            let (min, _) = IntValue::range(signed, bits);
            if signed && a.value == min && b.value == -BigInt::from(1) {
                return Err(InterpretError::Overflow("modulo".to_string()));
            }
            IntValue::canonical(signed, bits, a.value % b.value)
        }
        And => IntValue::canonical(signed, bits, a.unsigned_repr() & b.unsigned_repr()),
        Or => IntValue::canonical(signed, bits, a.unsigned_repr() | b.unsigned_repr()),
        Xor => IntValue::canonical(signed, bits, a.unsigned_repr() ^ b.unsigned_repr()),
        ShiftLeft => {
            // Noir's `<<` follows Rust's `checked_shl` (see the SSA interpreter `BinaryOp::Shl`):
            // only a shift amount >= the operand width is an error; the value itself wraps to the
            // width (high bits are dropped). Mavros's R1CS lowering is stricter (it range-checks
            // the shifted result), but that is a Mavros lowering choice, not what the AST means.
            let amount = shift_amount(&b)?;
            if amount >= bits as usize {
                return Err(InterpretError::Overflow(
                    "shift-left amount >= bit width".to_string(),
                ));
            }
            IntValue::canonical(signed, bits, a.unsigned_repr() << amount)
        }
        ShiftRight => {
            // Rust's `checked_shr`: amount must be < width; unsigned is logical and signed is
            // arithmetic. Shifting the canonical (sign-aware) value handles both: a negative
            // BigInt shifts toward negative infinity.
            let amount = shift_amount(&b)?;
            if amount >= bits as usize {
                return Err(InterpretError::Overflow(
                    "shift-right amount >= bit width".to_string(),
                ));
            }
            IntValue::canonical(signed, bits, a.value >> amount)
        }
        Equal | NotEqual | Less | LessEqual | Greater | GreaterEqual => unreachable!(),
    };
    Ok(Value::Int(int))
}

fn shift_amount(b: &IntValue) -> Result<usize, InterpretError> {
    let repr = b.unsigned_repr();
    let (_, digits) = repr.to_u64_digits();
    match digits.as_slice() {
        [] => Ok(0),
        [single] => Ok(*single as usize),
        _ => Err(InterpretError::Type("shift amount too large".to_string())),
    }
}

fn int_type(typ: &Type) -> Option<(bool, u8)> {
    match typ {
        Type::Integer(signedness, bits) => Some((signedness.is_signed(), bits.bit_size())),
        _ => None,
    }
}

fn signed_field_to_bigint(value: &SignedField, signed: bool) -> Result<BigInt, InterpretError> {
    if signed {
        value
            .try_to_signed::<i128>()
            .map(BigInt::from)
            .ok_or_else(|| InterpretError::Type("signed literal out of i128 range".to_string()))
    } else {
        value
            .try_to_unsigned::<u128>()
            .map(BigInt::from)
            .ok_or_else(|| InterpretError::Type("unsigned literal out of u128 range".to_string()))
    }
}

/// These tests pin the integer semantics to Mavros's R1CS lowering in
/// `passes/instruction_lowering/witness_integer_arith.rs`, so a green interpreter result
/// implies the same arithmetic Mavros constrains:
/// - Add/Sub/Mul are **checked** — overflow is an unsatisfiable constraint there, an error here.
/// - Signed Div/Mod **truncate toward zero**, remainder takes the dividend's sign
///   (`q_sign = sign_l ^ sign_r`, `r_sign = sign_l` in `lower_signed_divmod`).
/// - Unsigned Div/Mod give `q = floor(a/b)`, `r in [0, divisor)`.
#[cfg(test)]
mod semantics_tests {
    use super::*;

    fn int(signed: bool, bits: u8, v: i128) -> IntValue {
        IntValue::canonical(signed, bits, BigInt::from(v))
    }

    fn eval(op: BinaryOpKind, a: IntValue, b: IntValue) -> Result<Value, InterpretError> {
        eval_int_binary(op, a, b)
    }

    fn as_i128(value: Value) -> i128 {
        let Value::Int(i) = value else {
            panic!("expected integer, got {value:?}");
        };
        let digits = i.value.to_string();
        digits.parse().expect("fits i128")
    }

    #[test]
    fn signed_division_truncates_toward_zero() {
        // Matches `lower_signed_divmod`: magnitude division with q_sign = sign_l ^ sign_r.
        let cases = [
            (-7, 2, -3),
            (7, -2, -3),
            (-7, -2, 3),
            (7, 2, 3),
            (-8, 3, -2),
        ];
        for (a, b, expected) in cases {
            let got =
                as_i128(eval(BinaryOpKind::Divide, int(true, 8, a), int(true, 8, b)).unwrap());
            assert_eq!(got, expected, "{a} / {b}");
        }
    }

    #[test]
    fn signed_remainder_takes_dividend_sign() {
        // Matches `lower_signed_divmod`: r_sign = sign_l (the dividend's sign).
        let cases = [(-7, 2, -1), (7, -2, 1), (-7, -2, -1), (7, 2, 1)];
        for (a, b, expected) in cases {
            let got =
                as_i128(eval(BinaryOpKind::Modulo, int(true, 8, a), int(true, 8, b)).unwrap());
            assert_eq!(got, expected, "{a} % {b}");
        }
    }

    #[test]
    fn signed_modulo_overflows_on_min_mod_neg_one() {
        // Noir's `%` is Rust checked_rem: i_MIN % -1 overflows (errors) despite a math result of 0.
        assert!(matches!(
            eval(BinaryOpKind::Modulo, int(true, 8, -128), int(true, 8, -1)),
            Err(InterpretError::Overflow(_))
        ));
        // A normal signed modulo still works.
        assert_eq!(
            as_i128(eval(BinaryOpKind::Modulo, int(true, 8, -7), int(true, 8, 3)).unwrap()),
            -1
        );
    }

    #[test]
    fn unsigned_division_is_floor() {
        let got =
            as_i128(eval(BinaryOpKind::Divide, int(false, 8, 200), int(false, 8, 7)).unwrap());
        assert_eq!(got, 28); // floor(200/7)
        let rem =
            as_i128(eval(BinaryOpKind::Modulo, int(false, 8, 200), int(false, 8, 7)).unwrap());
        assert_eq!(rem, 4); // 200 - 28*7
    }

    #[test]
    fn unsigned_bitwise_matches_bit_patterns() {
        // 0b1100 & 0b1010 = 0b1000; | = 0b1110; ^ = 0b0110 (witness_bitwise.rs lower_word_bitwise).
        assert_eq!(
            as_i128(eval(BinaryOpKind::And, int(false, 8, 12), int(false, 8, 10)).unwrap()),
            8
        );
        assert_eq!(
            as_i128(eval(BinaryOpKind::Or, int(false, 8, 12), int(false, 8, 10)).unwrap()),
            14
        );
        assert_eq!(
            as_i128(eval(BinaryOpKind::Xor, int(false, 8, 12), int(false, 8, 10)).unwrap()),
            6
        );
    }

    #[test]
    fn shift_right_logical_and_arithmetic() {
        // Unsigned shr is logical (floor); signed shr is arithmetic (sign-preserving).
        assert_eq!(
            as_i128(
                eval(
                    BinaryOpKind::ShiftRight,
                    int(false, 8, 200),
                    int(false, 8, 2)
                )
                .unwrap()
            ),
            50
        );
        // Shift operands share one type in a well-typed AST (Noir's `Shl`/`Shr` are
        // `fn(self, other: Self)`; the SSA interpreter only handles same-type pairs), so the
        // signed shift's amount is the same signed type.
        assert_eq!(
            as_i128(eval(BinaryOpKind::ShiftRight, int(true, 8, -8), int(true, 8, 1)).unwrap()),
            -4
        );
        // Only a shift amount >= the width is an error (Rust checked_shr).
        assert!(matches!(
            eval(BinaryOpKind::ShiftRight, int(false, 8, 1), int(false, 8, 8)),
            Err(InterpretError::Overflow(_))
        ));
    }

    #[test]
    fn shift_left_wraps_value_errors_only_on_overshift() {
        // Noir `<<` (Rust checked_shl): the value wraps to the width; only an amount >= the width
        // is an error. So 0xff << 1 == 0xfe, but 1u8 << 8 overflows.
        assert_eq!(
            as_i128(
                eval(
                    BinaryOpKind::ShiftLeft,
                    int(false, 8, 0xff),
                    int(false, 8, 1)
                )
                .unwrap()
            ),
            0xfe
        );
        assert_eq!(
            as_i128(eval(BinaryOpKind::ShiftLeft, int(false, 8, 1), int(false, 8, 7)).unwrap()),
            128
        );
        assert!(matches!(
            eval(BinaryOpKind::ShiftLeft, int(false, 8, 1), int(false, 8, 8)),
            Err(InterpretError::Overflow(_))
        ));
    }

    #[test]
    fn arithmetic_is_checked_not_wrapping() {
        // Mavros range-checks the result; an overflowing value is unsatisfiable there. Here it
        // is an `Overflow` error rather than a silent wrap to 44 (u8) / 144 (i8).
        assert!(matches!(
            eval(BinaryOpKind::Add, int(false, 8, 200), int(false, 8, 100)),
            Err(InterpretError::Overflow(_))
        ));
        assert!(matches!(
            eval(BinaryOpKind::Add, int(true, 8, 100), int(true, 8, 44)),
            Err(InterpretError::Overflow(_))
        ));
        assert!(matches!(
            eval(BinaryOpKind::Subtract, int(false, 8, 0), int(false, 8, 1)),
            Err(InterpretError::Overflow(_))
        ));
    }
}
