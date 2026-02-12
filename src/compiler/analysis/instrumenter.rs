use std::{cell::RefCell, collections::HashMap, rc::Rc};

use ark_ff::{AdditiveGroup, BigInt, BigInteger, PrimeField};
use itertools::Itertools;
use tracing::instrument;

use crate::compiler::{
    Field,
    analysis::{
        symbolic_executor::{self, SymbolicExecutor},
        types::TypeInfo,
    },
    ir::r#type::{Type, TypeExpr},
    ssa::{
        BinaryArithOpKind, CastTarget, CmpKind, Endianness, FunctionId, MemOp, Radix, SSA, SeqType,
        SliceOpDir,
    },
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ScalarKind {
    Field,
    U(usize),
}

impl ScalarKind {
    pub fn from_type(tp: &Type) -> Self {
        match &tp.strip_witness().expr {
            TypeExpr::Field => ScalarKind::Field,
            TypeExpr::U(s) => ScalarKind::U(*s),
            TypeExpr::WitnessOf(inner) => ScalarKind::from_type(inner),
            _ => ScalarKind::Field, // default fallback
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ValueSignature {
    U(usize, u128),
    Field(Field),
    Array(Vec<ValueSignature>),
    PointerTo(Box<ValueSignature>),
    Unknown(ScalarKind),
    WitnessOf(Box<ValueSignature>),
    Tuple(Vec<ValueSignature>),
}

impl ValueSignature {
    pub fn to_value(&self) -> Value {
        match self {
            ValueSignature::U(size, val) => Value::U(*size, *val),
            ValueSignature::Field(field) => Value::Field(field.clone()),
            ValueSignature::Array(vals) => {
                Value::Array(vals.iter().map(|v| v.to_value()).collect())
            }
            ValueSignature::PointerTo(val) => Value::Pointer(Rc::new(RefCell::new(val.to_value()))),
            ValueSignature::Unknown(kind) => Value::Unknown(*kind),
            ValueSignature::WitnessOf(inner) => Value::WitnessOf(Box::new(inner.to_value())),
            ValueSignature::Tuple(elements) => {
                Value::Tuple(elements.iter().map(|e| e.to_value()).collect())
            }
        }
    }

    pub fn pretty_print(&self, full: bool) -> String {
        match self {
            ValueSignature::U(_, v) => format!("{v}"),
            ValueSignature::Field(f) => format!("{}", f),
            ValueSignature::Array(items) => {
                if full {
                    let items = items.iter().map(|v| v.pretty_print(full)).join(", ");
                    format!("[{items}]")
                } else {
                    format!("[...]")
                }
            }
            ValueSignature::PointerTo(p) => format!("&({})", p.as_ref().pretty_print(full)),
            ValueSignature::Unknown(_) => "?".to_string(),
            ValueSignature::WitnessOf(inner) => format!("W({})", inner.pretty_print(full)),
            ValueSignature::Tuple(elements) => {
                if full {
                    let elements = elements.iter().map(|e| e.pretty_print(full)).join(", ");
                    format!("({})", elements)
                } else {
                    format!("(...)")
                }
            }
        }
    }
}

#[derive(Debug, Clone)]
pub enum Value {
    U(usize, u128),
    Field(Field),
    Array(Vec<Value>),
    Pointer(Rc<RefCell<Value>>),
    Unknown(ScalarKind),
    WitnessOf(Box<Value>),
    Tuple(Vec<Value>),
}

impl Value {
    fn cmp_op(
        &self,
        b: &Value,
        cmp_kind: &crate::compiler::ssa::CmpKind,
        instrumenter: &mut dyn OpInstrumenter,
    ) -> Value {
        // Extract inner values from WitnessOf, track if either was witness
        let (a_inner, a_wit) = match self {
            Value::WitnessOf(inner) => (inner.as_ref(), true),
            other => (other, false),
        };
        let (b_inner, b_wit) = match b {
            Value::WitnessOf(inner) => (inner.as_ref(), true),
            other => (other, false),
        };
        let either_wit = a_wit || b_wit;

        // If either inner is Unknown, result is Unknown + record costs
        if matches!(a_inner, Value::Unknown(_)) || matches!(b_inner, Value::Unknown(_)) {
            if either_wit {
                instrumenter.record_constraints(1);
                return Value::WitnessOf(Box::new(Value::Unknown(ScalarKind::U(1))));
            }
            return Value::Unknown(ScalarKind::U(1));
        }

        let result = match (a_inner, b_inner) {
            (Value::U(_, a), Value::U(_, b)) => match cmp_kind {
                CmpKind::Eq => Value::U(1, if a == b { 1 } else { 0 }),
                CmpKind::Lt => Value::U(1, if a < b { 1 } else { 0 }),
            },
            (Value::Field(a), Value::Field(b)) => match cmp_kind {
                CmpKind::Eq => Value::U(1, if a == b { 1 } else { 0 }),
                CmpKind::Lt => Value::U(1, if a < b { 1 } else { 0 }),
            },
            (_, _) => {
                panic!("Cannot compare {:?} and {:?}", self, b);
            }
        };

        if either_wit {
            instrumenter.record_constraints(1);
            Value::WitnessOf(Box::new(result))
        } else {
            result
        }
    }

    fn binary_arith_op(
        &self,
        b: &Value,
        binary_arith_op_kind: &crate::compiler::ssa::BinaryArithOpKind,
        instrumenter: &mut dyn OpInstrumenter,
    ) -> Value {
        // Extract inner values from WitnessOf, track if either was witness
        let (a_inner, a_wit) = match self {
            Value::WitnessOf(inner) => (inner.as_ref(), true),
            other => (other, false),
        };
        let (b_inner, b_wit) = match b {
            Value::WitnessOf(inner) => (inner.as_ref(), true),
            other => (other, false),
        };
        let either_wit = a_wit || b_wit;
        let a_unknown = matches!(a_inner, Value::Unknown(_));
        let b_unknown = matches!(b_inner, Value::Unknown(_));

        // Mul by zero → zero, always free. Return typed zero matching the operands.
        if matches!(binary_arith_op_kind, BinaryArithOpKind::Mul) {
            let a_zero = matches!(a_inner, Value::Field(f) if *f == Field::ZERO)
                || matches!(a_inner, Value::U(_, 0));
            let b_zero = matches!(b_inner, Value::Field(f) if *f == Field::ZERO)
                || matches!(b_inner, Value::U(_, 0));
            if a_zero || b_zero {
                // Determine the correct type for the zero result
                let zero = match (a_inner, b_inner) {
                    (Value::U(s, _), _) | (_, Value::U(s, _)) => Value::U(*s, 0),
                    (Value::Unknown(ScalarKind::U(s)), _)
                    | (_, Value::Unknown(ScalarKind::U(s))) => Value::U(*s, 0),
                    _ => Value::Field(Field::ZERO),
                };
                return zero;
            }
        }

        // If either inner is Unknown, result is Unknown + record costs.
        // In R1CS, Mul only needs a constraint when both operands are variables
        // (Unknown). constant * variable is just a linear combination (free).
        if a_unknown || b_unknown {
            let kind = match (a_inner, b_inner) {
                (Value::Unknown(k), _) => *k,
                (_, Value::Unknown(k)) => *k,
                _ => unreachable!(),
            };
            match binary_arith_op_kind {
                BinaryArithOpKind::Mul => {
                    if a_unknown && b_unknown && a_wit && b_wit {
                        instrumenter.record_constraints(1);
                    }
                }
                BinaryArithOpKind::Div => {
                    if b_unknown && b_wit {
                        instrumenter.record_constraints(1);
                    }
                }
                _ => {}
            }
            if either_wit {
                return Value::WitnessOf(Box::new(Value::Unknown(kind)));
            }
            return Value::Unknown(kind);
        }

        // Both operands are concrete. Compute the result.
        let result = match (a_inner, b_inner) {
            (Value::U(s, a), Value::U(_, b)) => match binary_arith_op_kind {
                BinaryArithOpKind::Add => Value::U(*s, a + b),
                BinaryArithOpKind::Sub => Value::U(*s, a - b),
                BinaryArithOpKind::Mul => Value::U(*s, a * b),
                BinaryArithOpKind::Div => Value::U(*s, a / b),
                BinaryArithOpKind::And => Value::U(*s, a & b),
            },
            (Value::Field(a), Value::Field(b)) => match binary_arith_op_kind {
                BinaryArithOpKind::Add => Value::Field(a + b),
                BinaryArithOpKind::Sub => Value::Field(a - b),
                BinaryArithOpKind::Mul => Value::Field(a * b),
                BinaryArithOpKind::Div => Value::Field(a / b),
                BinaryArithOpKind::And => todo!(),
            },
            (_, _) => panic!("Cannot perform binary arithmetic on {:?} and {:?}", self, b),
        };

        // Both concrete → no Mul constraint (known product).
        // Div by concrete witness is also free (multiply by inverse).
        if either_wit {
            Value::WitnessOf(Box::new(result))
        } else {
            result
        }
    }

    fn blind_from(&mut self) {
        match self {
            Value::WitnessOf(inner) => {
                inner.forget_concrete();
            }
            Value::Unknown(_) => {}
            Value::U(_, _) | Value::Field(_) => {}
            Value::Array(vals) => {
                for val in vals {
                    val.blind_from();
                }
            }
            Value::Pointer(val) => {
                val.borrow_mut().blind_from();
            }
            Value::Tuple(vals) => {
                for val in vals {
                    val.blind_from();
                }
            }
        }
    }

    fn forget_concrete(&mut self) {
        match self {
            Value::U(s, _) => *self = Value::Unknown(ScalarKind::U(*s)),
            Value::Field(_) => *self = Value::Unknown(ScalarKind::Field),
            Value::Unknown(_) => {}
            Value::WitnessOf(inner) => {
                inner.forget_concrete();
            }
            Value::Array(vals) => {
                for val in vals {
                    val.forget_concrete();
                }
            }
            Value::Pointer(val) => {
                val.borrow_mut().forget_concrete();
            }
            Value::Tuple(vals) => {
                for val in vals {
                    val.forget_concrete();
                }
            }
        }
    }

    fn make_unspecialized_sig(&self) -> ValueSignature {
        match self {
            Value::Unknown(kind) => ValueSignature::Unknown(*kind),
            Value::WitnessOf(inner) => {
                ValueSignature::WitnessOf(Box::new(inner.make_unspecialized_sig()))
            }
            Value::U(s, v) => ValueSignature::U(*s, *v),
            Value::Field(f) => ValueSignature::Field(f.clone()),
            Value::Array(vals) => {
                ValueSignature::Array(vals.iter().map(|v| v.make_unspecialized_sig()).collect())
            }
            Value::Pointer(val) => {
                ValueSignature::PointerTo(Box::new(val.borrow().make_unspecialized_sig()))
            }
            Value::Tuple(vals) => {
                ValueSignature::Tuple(vals.iter().map(|v| v.make_unspecialized_sig()).collect())
            }
        }
    }

    fn assert_eq(&self, other: &Value, instrumenter: &mut dyn OpInstrumenter) {
        if self.is_witness() || other.is_witness() {
            instrumenter.record_constraints(1);
        }
    }

    fn rangecheck(&self, max_bits: usize, instrumenter: &mut dyn OpInstrumenter) {
        if self.is_witness() {
            instrumenter.record_rangechecks(max_bits as u8, 1);
        }
    }

    fn is_witness(&self) -> bool {
        match self {
            Value::Unknown(_) => false,
            Value::WitnessOf(_) => true,
            _ => false,
        }
    }

    fn array_get(
        &self,
        index: &Value,
        _tp: &Type,
        instrumenter: &mut dyn OpInstrumenter,
    ) -> Value {
        // If the array itself is Unknown, result is Unknown
        if matches!(self, Value::Unknown(_)) {
            return Value::Unknown(ScalarKind::Field);
        }

        // Extract array from WitnessOf wrapper if present
        let arr = match self {
            Value::WitnessOf(inner) => inner.as_ref(),
            other => other,
        };

        match (arr, index) {
            (Value::Array(vals), Value::U(_, index)) => vals[*index as usize].clone(),
            (Value::Array(vals), Value::Unknown(_))
            | (Value::Array(vals), Value::WitnessOf(_)) => {
                // Unknown or witness index — we don't know which element, return Unknown
                instrumenter.record_lookups(vals.len(), 1, 1);
                Value::Unknown(ScalarKind::Field)
            }
            (Value::Unknown(_), _) => Value::Unknown(ScalarKind::Field),
            _ => panic!(
                "Cannot get array element from {:?} with index {:?}",
                self, index
            ),
        }
    }

    fn tuple_get(
        &self,
        index: usize,
        _tp: &Type,
        _instrumenter: &mut dyn OpInstrumenter,
    ) -> Value {
        match self {
            Value::Unknown(_) => Value::Unknown(ScalarKind::Field),
            Value::WitnessOf(inner) => inner.tuple_get(index, _tp, _instrumenter),
            Value::Tuple(vals) => vals[index as usize].clone(),
            _ => panic!(
                "Cannot get tuple element from {:?} with index {:?}",
                self, index
            ),
        }
    }

    fn array_set(
        &self,
        index: &Value,
        value: &Value,
        _instrumenter: &mut dyn OpInstrumenter,
    ) -> Value {
        match (self, index, value) {
            (Value::Array(vals), Value::U(_, index), value) => {
                let mut new_vals = vals.clone();
                new_vals[*index as usize] = value.clone();
                Value::Array(new_vals)
            }
            // Dynamic index: any element could be replaced, so set all to value
            (Value::Array(vals), _, value) => {
                let new_vals = vals.iter().map(|_| value.clone()).collect();
                Value::Array(new_vals)
            }
            _ => panic!(
                "Cannot set array element of {:?} with index {:?} to {:?}",
                self, index, value
            ),
        }
    }

    fn truncate_op(
        &self,
        _from: usize,
        to: usize,
        _instrumenter: &mut dyn OpInstrumenter,
    ) -> Value {
        match self {
            Value::Unknown(kind) => Value::Unknown(*kind),
            Value::WitnessOf(inner) => {
                Value::WitnessOf(Box::new(inner.truncate_op(_from, to, _instrumenter)))
            }
            Value::U(_, v) => Value::U(to, v & ((1 << to) - 1)),
            Value::Field(f) => {
                let bits = f
                    .into_bigint()
                    .to_bits_le()
                    .into_iter()
                    .take(to)
                    .collect::<Vec<_>>();
                let r = Field::from_bigint(BigInt::from_bits_le(&bits));
                Value::Field(r.unwrap())
            }
            _ => panic!("Cannot truncate {:?}", self),
        }
    }

    fn cast_op(
        &self,
        cast_target: &crate::compiler::ssa::CastTarget,
        _instrumenter: &mut dyn OpInstrumenter,
    ) -> Value {
        match (self, cast_target) {
            (_, CastTarget::WitnessOf) => Value::WitnessOf(Box::new(self.clone())),
            (Value::Unknown(kind), _) => Value::Unknown(*kind),
            (Value::WitnessOf(inner), target) => {
                Value::WitnessOf(Box::new(inner.cast_op(target, _instrumenter)))
            }
            (Value::U(_, v), CastTarget::U(s2)) => Value::U(*s2, *v),
            (Value::U(_, v), CastTarget::Field) => Value::Field(Field::from(*v)),
            (Value::Field(f), CastTarget::Field) => Value::Field(f.clone()),
            (Value::Field(f), CastTarget::U(s)) => {
                let bigint = f.into_bigint();
                Value::U(*s, bigint.0[0] as u128 | ((bigint.0[1] as u128) << 64))
            }
            (_, CastTarget::Nop | CastTarget::ArrayToSlice) => self.clone(),
            _ => panic!("Cannot cast {:?} to {:?}", self, cast_target),
        }
    }

    fn constrain(a: &Value, b: &Value, c: &Value, instrumenter: &mut dyn OpInstrumenter) {
        if a.is_witness() || b.is_witness() || c.is_witness() {
            instrumenter.record_constraints(1);
        }
    }

    fn to_bits(
        &self,
        endianness: &crate::compiler::ssa::Endianness,
        size: usize,
        instrumenter: &mut dyn OpInstrumenter,
    ) -> Value {
        match self {
            Value::Unknown(kind) => Value::Unknown(*kind),
            Value::WitnessOf(inner) => {
                let result = inner.to_bits(endianness, size, instrumenter);
                // Wrap each bit in WitnessOf
                match result {
                    Value::Array(bits) => Value::Array(
                        bits.into_iter()
                            .map(|b| Value::WitnessOf(Box::new(b)))
                            .collect(),
                    ),
                    _ => unreachable!(),
                }
            }
            Value::U(_, v) => {
                let mut r = vec![];
                for i in 0..size {
                    let bit = (v >> i) & 1;
                    let bit = Value::U(1, bit);
                    r.push(bit);
                }
                if *endianness == Endianness::Big {
                    r.reverse();
                }
                Value::Array(r)
            }
            Value::Field(f) => {
                let bigint = f.into_bigint();
                let bits = bigint.to_bits_le().into_iter().take(size);
                let mut bits = bits.map(|b| Value::U(1, b as u128)).collect::<Vec<_>>();
                if *endianness == Endianness::Big {
                    bits.reverse();
                }
                Value::Array(bits)
            }
            _ => panic!("Cannot convert {:?} to bits", self),
        }
    }

    fn to_radix(
        &self,
        radix: &Radix<Value>,
        _endianness: &crate::compiler::ssa::Endianness,
        size: usize,
        instrumenter: &mut dyn OpInstrumenter,
    ) -> Value {
        match self {
            Value::Unknown(_) | Value::WitnessOf(_) => {
                // Witness value decomposed to radix — result is unknown digits
                instrumenter.record_rangechecks(8, size);
                instrumenter.record_constraints(1);
                Value::Array(vec![Value::Unknown(ScalarKind::U(8)); size])
            }
            Value::Field(f) => {
                // Concrete field decomposed to radix
                let radix_val = match radix {
                    Radix::Dyn(Value::U(_, r)) => *r as u128,
                    Radix::Bytes => 256,
                    _ => panic!("Cannot convert {:?} to radix {:?}", self, radix),
                };
                let mut val = f.into_bigint();
                let mut digits = vec![];
                for _ in 0..size {
                    let digit = {
                        let limb = val.0[0] as u128;
                        limb % radix_val
                    };
                    digits.push(Value::U(8, digit));
                    // Divide val by radix_val
                    let mut carry: u128 = 0;
                    for i in (0..val.0.len()).rev() {
                        let cur = (carry << 64) | (val.0[i] as u128);
                        val.0[i] = (cur / radix_val) as u64;
                        carry = cur % radix_val;
                    }
                }
                Value::Array(digits)
            }
            _ => panic!("Cannot convert {:?} to radix {:?}", self, radix),
        }
    }

    fn not_op(&self, _instrumenter: &mut dyn OpInstrumenter) -> Value {
        match self {
            Value::Unknown(kind) => Value::Unknown(*kind),
            Value::WitnessOf(inner) => Value::WitnessOf(Box::new(inner.not_op(_instrumenter))),
            Value::U(s, v) => Value::U(*s, !v),
            _ => panic!("Cannot perform not operation on {:?}", self),
        }
    }

    fn ptr_read(&self, _tp: &Type, _instrumenter: &mut dyn OpInstrumenter) -> Value {
        match self {
            Value::Pointer(val) => val.borrow().clone(),
            _ => panic!("Cannot read from {:?}", self),
        }
    }

    fn ptr_write(&self, val: &Value, _instrumenter: &mut dyn OpInstrumenter) {
        match self {
            Value::Pointer(ptr) => {
                *(ptr.borrow_mut()) = val.clone();
            }
            _ => panic!("Cannot write to {:?}", self),
        }
    }

    fn assert_r1c(a: &Value, b: &Value, c: &Value, get_unspecialized: &mut dyn OpInstrumenter) {
        if a.is_witness() || b.is_witness() || c.is_witness() {
            get_unspecialized.record_constraints(1);
        }
    }

    fn select(
        &self,
        if_true: &Value,
        if_false: &Value,
        _tp: &Type,
        instrumenter: &mut dyn OpInstrumenter,
    ) -> Value {
        match self {
            Value::U(_, 0) => if_true.clone(),
            Value::U(_, _) => if_false.clone(),
            Value::WitnessOf(inner) => {
                // We know the concrete cond value, so we can select deterministically
                match inner.as_ref() {
                    Value::U(_, 0) => if_true.clone(),
                    Value::U(_, _) => if_false.clone(),
                    _ => {
                        // Non-U inner — treat as unknown
                        if if_true.is_witness() || if_false.is_witness() {
                            instrumenter.record_constraints(1);
                        }
                        Value::Unknown(ScalarKind::Field)
                    }
                }
            }
            Value::Unknown(_) => {
                if if_true.is_witness() || if_false.is_witness() {
                    instrumenter.record_constraints(1);
                }
                Value::Unknown(ScalarKind::Field)
            }
            _ => panic!("Cannot select on {:?}", self),
        }
    }
}

#[derive(Debug, Clone)]
pub struct SpecSplitValue {
    unspecialized: Value,
    specialized: Value,
}

impl SpecSplitValue {
    fn blind_unspecialized_from(&mut self) {
        self.unspecialized.blind_from();
    }

    fn blind_from(&mut self) {
        self.unspecialized.blind_from();
        self.specialized.blind_from();
    }
}

impl symbolic_executor::Value<CostAnalysis> for SpecSplitValue {
    fn cmp(
        &self,
        b: &SpecSplitValue,
        cmp_kind: CmpKind,
        _tp: &Type,
        instrumenter: &mut CostAnalysis,
    ) -> SpecSplitValue {
        let unspecialized = self.unspecialized.cmp_op(
            &b.unspecialized,
            &cmp_kind,
            instrumenter.get_unspecialized(),
        );
        let specialized =
            self.specialized
                .cmp_op(&b.specialized, &cmp_kind, instrumenter.get_specialized());
        SpecSplitValue {
            unspecialized,
            specialized,
        }
    }

    fn arith(
        &self,
        b: &SpecSplitValue,
        binary_arith_op_kind: BinaryArithOpKind,
        _tp: &Type,
        instrumenter: &mut CostAnalysis,
    ) -> SpecSplitValue {
        let unspecialized = self.unspecialized.binary_arith_op(
            &b.unspecialized,
            &binary_arith_op_kind,
            instrumenter.get_unspecialized(),
        );
        let specialized = self.specialized.binary_arith_op(
            &b.specialized,
            &binary_arith_op_kind,
            instrumenter.get_specialized(),
        );
        SpecSplitValue {
            unspecialized,
            specialized,
        }
    }

    fn cast(
        &self,
        cast_target: &crate::compiler::ssa::CastTarget,
        _tp: &Type,
        instrumenter: &mut CostAnalysis,
    ) -> SpecSplitValue {
        SpecSplitValue {
            unspecialized: self
                .unspecialized
                .cast_op(cast_target, instrumenter.get_unspecialized()),
            specialized: self
                .specialized
                .cast_op(cast_target, instrumenter.get_specialized()),
        }
    }

    fn truncate(
        &self,
        from: usize,
        to: usize,
        _tp: &Type,
        instrumenter: &mut CostAnalysis,
    ) -> SpecSplitValue {
        SpecSplitValue {
            unspecialized: self.unspecialized.truncate_op(
                from,
                to,
                instrumenter.get_unspecialized(),
            ),
            specialized: self
                .specialized
                .truncate_op(from, to, instrumenter.get_specialized()),
        }
    }

    fn not(&self, _tp: &Type, instrumenter: &mut CostAnalysis) -> SpecSplitValue {
        SpecSplitValue {
            unspecialized: self.unspecialized.not_op(instrumenter.get_unspecialized()),
            specialized: self.specialized.not_op(instrumenter.get_specialized()),
        }
    }

    fn ptr_write(&self, val: &SpecSplitValue, _instrumenter: &mut CostAnalysis) {
        self.unspecialized
            .ptr_write(&val.unspecialized, _instrumenter.get_unspecialized());
        self.specialized
            .ptr_write(&val.specialized, _instrumenter.get_specialized());
    }

    fn ptr_read(&self, tp: &Type, ctx: &mut CostAnalysis) -> SpecSplitValue {
        let mut res = SpecSplitValue {
            unspecialized: self.unspecialized.ptr_read(tp, ctx.get_unspecialized()),
            specialized: self.specialized.ptr_read(tp, ctx.get_specialized()),
        };
        res.blind_unspecialized_from();
        res
    }

    fn mk_array(
        values: Vec<SpecSplitValue>,
        _ctx: &mut CostAnalysis,
        _seq_type: SeqType,
        _elem_type: &Type,
    ) -> SpecSplitValue {
        let (uns, spec) = values
            .into_iter()
            .map(|v| (v.unspecialized, v.specialized))
            .unzip();
        SpecSplitValue {
            unspecialized: Value::Array(uns),
            specialized: Value::Array(spec),
        }
    }

    fn mk_tuple (
        values: Vec<SpecSplitValue>,
        _ctx: &mut CostAnalysis,
        _elem_types: &[Type],
    ) -> SpecSplitValue {
        let (uns, spec) = values
            .into_iter()
            .map(|v| (v.unspecialized, v.specialized))
            .unzip();
        SpecSplitValue {
            unspecialized: Value::Tuple(uns),
            specialized: Value::Tuple(spec),
        }
    }

    fn assert_r1c(
        a: &SpecSplitValue,
        b: &SpecSplitValue,
        c: &SpecSplitValue,
        ctx: &mut CostAnalysis,
    ) {
        Value::assert_r1c(
            &a.unspecialized,
            &b.unspecialized,
            &c.unspecialized,
            ctx.get_unspecialized(),
        );
        Value::assert_r1c(
            &a.specialized,
            &b.specialized,
            &c.specialized,
            ctx.get_specialized(),
        );
    }

    fn array_get(
        &self,
        i: &SpecSplitValue,
        tp: &Type,
        instrumenter: &mut CostAnalysis,
    ) -> SpecSplitValue {
        SpecSplitValue {
            unspecialized: self.unspecialized.array_get(
                &i.unspecialized,
                tp,
                instrumenter.get_unspecialized(),
            ),
            specialized: self.specialized.array_get(
                &i.specialized,
                tp,
                instrumenter.get_specialized(),
            ),
        }
    }

    fn tuple_get(
        &self,
        index: usize,
        tp: &Type,
        instrumenter: &mut CostAnalysis,
    ) -> SpecSplitValue {
        SpecSplitValue {
            unspecialized: self.unspecialized.tuple_get(
                index,
                tp,
                instrumenter.get_unspecialized(),
            ),
            specialized: self.specialized.tuple_get(
                index,
                tp,
                instrumenter.get_specialized(),
            ),
        }
    }

    fn array_set(
        &self,
        i: &SpecSplitValue,
        v: &SpecSplitValue,
        _tp: &Type,
        instrumenter: &mut CostAnalysis,
    ) -> SpecSplitValue {
        SpecSplitValue {
            unspecialized: self.unspecialized.array_set(
                &i.unspecialized,
                &v.unspecialized,
                instrumenter.get_unspecialized(),
            ),
            specialized: self.specialized.array_set(
                &i.specialized,
                &v.specialized,
                instrumenter.get_specialized(),
            ),
        }
    }

    fn select(
        &self,
        if_t: &SpecSplitValue,
        if_f: &SpecSplitValue,
        tp: &Type,
        instrumenter: &mut CostAnalysis,
    ) -> SpecSplitValue {
        SpecSplitValue {
            unspecialized: self.unspecialized.select(
                &if_t.unspecialized,
                &if_f.unspecialized,
                tp,
                instrumenter.get_unspecialized(),
            ),
            specialized: self.specialized.select(
                &if_t.specialized,
                &if_f.specialized,
                tp,
                instrumenter.get_specialized(),
            ),
        }
    }

    fn constrain(
        a: &SpecSplitValue,
        b: &SpecSplitValue,
        c: &SpecSplitValue,
        instrumenter: &mut CostAnalysis,
    ) {
        Value::constrain(
            &a.unspecialized,
            &b.unspecialized,
            &c.unspecialized,
            instrumenter.get_unspecialized(),
        );
        Value::constrain(
            &a.specialized,
            &b.specialized,
            &c.specialized,
            instrumenter.get_specialized(),
        );
    }

    fn assert_eq(&self, b: &SpecSplitValue, instrumenter: &mut CostAnalysis) {
        self.specialized
            .assert_eq(&b.specialized, instrumenter.get_specialized());
        self.unspecialized
            .assert_eq(&b.unspecialized, instrumenter.get_unspecialized());
    }

    fn rangecheck(&self, max_bits: usize, instrumenter: &mut CostAnalysis) {
        self.unspecialized
            .rangecheck(max_bits, instrumenter.get_unspecialized());
        self.specialized
            .rangecheck(max_bits, instrumenter.get_specialized());
    }

    fn to_bits(
        &self,
        endianness: Endianness,
        size: usize,
        _tp: &Type,
        instrumenter: &mut CostAnalysis,
    ) -> SpecSplitValue {
        SpecSplitValue {
            unspecialized: self.unspecialized.to_bits(
                &endianness,
                size,
                instrumenter.get_unspecialized(),
            ),
            specialized: self.specialized.to_bits(
                &endianness,
                size,
                instrumenter.get_specialized(),
            ),
        }
    }

    fn to_radix(
        &self,
        radix: &Radix<SpecSplitValue>,
        endianness: Endianness,
        size: usize,
        _tp: &Type,
        instrumenter: &mut CostAnalysis,
    ) -> SpecSplitValue {
        let spec_radix = match radix {
            Radix::Dyn(v) => Radix::Dyn(v.specialized.clone()),
            Radix::Bytes => Radix::Bytes,
        };
        let unspec_radix = match radix {
            Radix::Dyn(v) => Radix::Dyn(v.unspecialized.clone()),
            Radix::Bytes => Radix::Bytes,
        };
        SpecSplitValue {
            unspecialized: self.unspecialized.to_radix(
                &unspec_radix,
                &endianness,
                size,
                instrumenter.get_unspecialized(),
            ),
            specialized: self.specialized.to_radix(
                &spec_radix,
                &endianness,
                size,
                instrumenter.get_specialized(),
            ),
        }
    }

    fn expect_constant_bool(&self, _ctx: &mut CostAnalysis) -> bool {
        // Use specialized side only — the unspecialized side may have Unknown
        // loop conditions due to forget_concrete on non-witness params
        match &self.specialized {
            Value::U(1, v) => *v != 0,
            _ => panic!(
                "Expected constant bool, got specialized={:?}",
                self.specialized
            ),
        }
    }

    fn of_u(s: usize, v: u128, _ctx: &mut CostAnalysis) -> Self {
        Self {
            unspecialized: Value::U(s, v),
            specialized: Value::U(s, v),
        }
    }

    fn of_field(f: Field, _ctx: &mut CostAnalysis) -> Self {
        Self {
            unspecialized: Value::Field(f),
            specialized: Value::Field(f),
        }
    }

    fn alloc(_elem_type: &Type, _ctx: &mut CostAnalysis) -> Self {
        Self {
            unspecialized: Value::Pointer(Rc::new(RefCell::new(Value::Unknown(ScalarKind::Field)))),
            specialized: Value::Pointer(Rc::new(RefCell::new(Value::Unknown(ScalarKind::Field)))),
        }
    }

    fn write_witness(&self, _tp: Option<&Type>, _ctx: &mut CostAnalysis) -> Self {
        // Wrap value in WitnessOf
        Self {
            unspecialized: Value::WitnessOf(Box::new(self.unspecialized.clone())),
            specialized: Value::WitnessOf(Box::new(self.specialized.clone())),
        }
    }

    fn fresh_witness(result_type: &Type, _ctx: &mut CostAnalysis) -> Self {
        let kind = ScalarKind::from_type(result_type);
        Self {
            unspecialized: Value::WitnessOf(Box::new(Value::Unknown(kind))),
            specialized: Value::WitnessOf(Box::new(Value::Unknown(kind))),
        }
    }

    fn mem_op(&self, _kind: MemOp, _ctx: &mut CostAnalysis) {}
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FunctionSignature {
    id: FunctionId,
    params: Vec<ValueSignature>,
}

impl FunctionSignature {
    pub fn pretty_print(&self, ssa: &SSA, all_params: bool) -> String {
        let fn_body = ssa.get_function(self.id);
        let name = fn_body.get_name();
        let params = self
            .params
            .iter()
            .map(|v| v.pretty_print(all_params))
            .collect::<Vec<_>>()
            .join(", ");
        format!("{}#[{}]", name, params)
    }

    pub fn get_fun_id(&self) -> FunctionId {
        self.id
    }

    pub fn get_params(&self) -> &[ValueSignature] {
        &self.params
    }
}

trait OpInstrumenter {
    fn record_constraints(&mut self, number: usize);
    fn record_rangechecks(&mut self, size: u8, count: usize);
    fn record_lookups(&mut self, keys: usize, results: usize, count: usize);
}

trait FunctionInstrumenter {
    fn get_specialized(&mut self) -> &mut dyn OpInstrumenter;
    fn get_unspecialized(&mut self) -> &mut dyn OpInstrumenter;
    fn record_call(&mut self, sig: FunctionSignature);
    fn seal(self: Box<Self>) -> FunctionCost;
}

#[derive(Debug, Clone)]
struct Instrumenter {
    constraints: usize,
    rangechecks: HashMap<u8, usize>,
    lookups: HashMap<(usize, usize), usize>,
}

impl Instrumenter {
    fn new() -> Self {
        Self {
            constraints: 0,
            rangechecks: HashMap::new(),
            lookups: HashMap::new(),
        }
    }
}

impl OpInstrumenter for Instrumenter {
    fn record_constraints(&mut self, number: usize) {
        self.constraints += number;
    }

    fn record_rangechecks(&mut self, size: u8, count: usize) {
        *self.rangechecks.entry(size).or_insert(0) += count;
    }

    fn record_lookups(&mut self, keys: usize, results: usize, count: usize) {
        *self.lookups.entry((keys, results)).or_insert(0) += count;
    }
}

#[derive(Debug, Clone)]
pub struct FunctionCost {
    calls: HashMap<FunctionSignature, usize>,
    raw: Instrumenter,
    specialized: Instrumenter,
}

impl FunctionInstrumenter for FunctionCost {
    fn get_specialized(&mut self) -> &mut dyn OpInstrumenter {
        &mut self.specialized
    }

    fn get_unspecialized(&mut self) -> &mut dyn OpInstrumenter {
        &mut self.raw
    }

    fn record_call(&mut self, sig: FunctionSignature) {
        *self.calls.entry(sig).or_insert(0) += 1;
    }

    fn seal(self: Box<Self>) -> FunctionCost {
        *self
    }
}

pub struct DummyInstrumenter {}

impl FunctionInstrumenter for DummyInstrumenter {
    fn get_specialized(&mut self) -> &mut dyn OpInstrumenter {
        self
    }

    fn get_unspecialized(&mut self) -> &mut dyn OpInstrumenter {
        self
    }

    fn record_call(&mut self, _: FunctionSignature) {}

    fn seal(self: Box<Self>) -> FunctionCost {
        panic!("DummyInstrumenter cannot be sealed");
    }
}

impl OpInstrumenter for DummyInstrumenter {
    fn record_constraints(&mut self, _: usize) {}
    fn record_rangechecks(&mut self, _: u8, _: usize) {}
    fn record_lookups(&mut self, _: usize, _: usize, _: usize) {}
}

pub struct CostAnalysis {
    entry_point: Option<FunctionSignature>,
    functions: HashMap<FunctionSignature, FunctionCost>,
    cache: HashMap<FunctionSignature, Vec<ValueSignature>>,
    stack: Vec<(FunctionSignature, Box<dyn FunctionInstrumenter>)>,
}

impl symbolic_executor::Context<SpecSplitValue> for CostAnalysis {
    fn on_call(
        &mut self,
        func: FunctionId,
        params: &mut [SpecSplitValue],
        param_types: &[&Type],
    ) -> Option<Vec<SpecSplitValue>> {
        for (pval, _ptype) in params.iter_mut().zip(param_types.iter()) {
            pval.blind_from();
            // Additionally forget concrete values on the unspecialized side
            // so that non-witness concrete parameters (like exponents) create
            // divergence between specialized and unspecialized sides
            pval.unspecialized.forget_concrete();
        }

        // Build signature from the specialized side — this captures concrete
        // non-witness values that the specializer can bake in
        let inputs_sig: Vec<ValueSignature> = params
            .iter()
            .map(|pval| pval.specialized.make_unspecialized_sig())
            .collect();

        let sig = FunctionSignature {
            id: func,
            params: inputs_sig,
        };

        // It's unsafe to use a cache for functions that take pointers,
        // as these could get modified. We can improve in the future by
        // also caching the final results of all input ptrs.
        let ptrs = param_types.iter().any(|tp| tp.contains_ptrs());
        if !ptrs {
            if let Some(cached) = self.cache.get(&sig).cloned() {
                self.register_cached_call(sig.clone());
                return Some(
                    cached
                        .iter()
                        .map(|v| SpecSplitValue {
                            unspecialized: v.to_value(),
                            specialized: v.to_value(),
                        })
                        .collect(),
                );
            }
        }

        self.enter_call(sig);
        None
    }

    fn on_return(&mut self, returns: &mut [SpecSplitValue], return_types: &[Type]) {
        for (rval, _rtype) in returns.iter_mut().zip(return_types.iter()) {
            rval.blind_from();
        }

        let sig = self.exit_call();

        let mut caches = vec![];

        for rval in returns.iter() {
            caches.push(rval.specialized.make_unspecialized_sig());
        }

        self.cache.insert(sig.clone(), caches);
    }

    fn on_jmp(
        &mut self,
        _target: crate::compiler::ssa::BlockId,
        params: &mut [SpecSplitValue],
        param_types: &[&Type],
    ) {
        for (pval, _ptype) in params.iter_mut().zip(param_types.iter()) {
            pval.blind_unspecialized_from();
        }
    }

    fn todo(
        &mut self,
        payload: &str,
        _result_types: &[Type],
    ) -> Vec<SpecSplitValue> {
        panic!("Todo opcode encountered in CostAnalysis: {}", payload);
    }

    fn slice_push(
        &mut self,
        slice: &SpecSplitValue,
        pushed_values: &[SpecSplitValue],
        dir: SliceOpDir,
    ) -> SpecSplitValue {
        assert_eq!(dir, SliceOpDir::Back); // TODO
        let new_unspec = match &slice.unspecialized {
            Value::Array(values) => {
                let mut new_values = values.clone();
                new_values.extend(pushed_values.iter().map(|v| v.unspecialized.clone()));
                Value::Array(new_values)
            }
            _ => panic!("Cannot push to {:?}", slice.unspecialized),
        };
        let new_spec = match &slice.specialized {
            Value::Array(values) => {
                let mut new_values = values.clone();
                new_values.extend(pushed_values.iter().map(|v| v.specialized.clone()));
                Value::Array(new_values)
            }
            _ => panic!("Cannot push to {:?}", slice.specialized),
        };
        SpecSplitValue {
            unspecialized: new_unspec,
            specialized: new_spec,
        }
    }

    fn slice_len(&mut self, slice: &SpecSplitValue) -> SpecSplitValue {
        let unspec = match &slice.unspecialized {
            Value::Array(values) => Value::U(32, values.len() as u128),
            _ => panic!("Cannot get length of {:?}", slice.unspecialized),
        };
        let spec = match &slice.specialized {
            Value::Array(values) => Value::U(32, values.len() as u128),
            _ => panic!("Cannot get length of {:?}", slice.specialized),
        };
        SpecSplitValue {
            unspecialized: unspec,
            specialized: spec,
        }
    }
}

pub struct SpecializationSummary {
    pub calls: usize,
    pub raw_constraints: usize,
    pub specialized_constraints: usize,
    pub specialization_total_savings: usize,
}

pub struct Summary {
    total_constraints: usize,
    total_savings_to_make: usize,
    pub functions: HashMap<FunctionSignature, SpecializationSummary>,
}

impl Summary {
    pub fn pretty_print(&self, ssa: &SSA) -> String {
        let mut r = String::new();
        r += &format!("Total constraints: {}\n", self.total_constraints);
        r += &format!(
            "Total savings to make: {} ({:.1}%)\n",
            self.total_savings_to_make,
            self.total_savings_to_make as f64 / self.total_constraints as f64 * 100.0
        );
        for (sig, summary) in self
            .functions
            .iter()
            .sorted_by_key(|(_, s)| s.specialization_total_savings)
            .rev()
        {
            r += &format!("Function {}\n", sig.pretty_print(ssa, false));
            r += &format!("  Called times: {}\n", summary.calls);
            r += &format!("  Raw constraints: {}\n", summary.raw_constraints);
            r += &format!(
                "  Specialized constraints: {}\n",
                summary.specialized_constraints
            );
            r += &format!(
                "  Specialization total savings: {}\n",
                summary.specialization_total_savings
            );
        }

        r
    }
}

impl CostAnalysis {
    fn register_cached_call(&mut self, sig: FunctionSignature) {
        if !self.stack.is_empty() {
            let (_, cost) = self.stack.last_mut().unwrap();
            cost.record_call(sig.clone());
        }
    }

    fn enter_call(&mut self, sig: FunctionSignature) {
        if !self.stack.is_empty() {
            let (_, cost) = self.stack.last_mut().unwrap();
            cost.record_call(sig.clone());
        }
        if self.entry_point.is_none() {
            self.entry_point = Some(sig.clone());
        }
        if self.functions.contains_key(&sig) {
            self.stack.push((sig, Box::new(DummyInstrumenter {})));
        } else {
            let instrumenter = FunctionCost {
                calls: HashMap::new(),
                raw: Instrumenter::new(),
                specialized: Instrumenter::new(),
            };
            self.stack.push((sig, Box::new(instrumenter)));
        }
    }

    fn exit_call(&mut self) -> FunctionSignature {
        let (sig, instrumenter) = self.stack.pop().unwrap();
        if !self.functions.contains_key(&sig) {
            let instrumenter = instrumenter.seal();
            self.functions.insert(sig.clone(), instrumenter);
        }
        sig
    }

    fn get_specialized(&mut self) -> &mut dyn OpInstrumenter {
        self.stack.last_mut().unwrap().1.as_mut().get_specialized()
    }

    fn get_unspecialized(&mut self) -> &mut dyn OpInstrumenter {
        self.stack
            .last_mut()
            .unwrap()
            .1
            .as_mut()
            .get_unspecialized()
    }

    pub fn seal(self) -> HashMap<FunctionSignature, FunctionCost> {
        self.functions
    }

    pub fn pretty_print(&self, ssa: &SSA) -> String {
        let mut r = String::new();
        for (sig, cost) in self.functions.iter() {
            r += &format!("Function {}\n", sig.pretty_print(ssa, false));
            r += &format!("  Calls:\n");
            for (sig, count) in cost.calls.iter() {
                r += &format!("    {}: {} times\n", sig.pretty_print(ssa, false), count);
            }
            r += &format!("  Raw constraints: {}\n", cost.raw.constraints);
            r += &format!(
                "  Specialized constraints: {}\n",
                cost.specialized.constraints
            );
        }
        r
    }

    pub fn summarize(&self) -> Summary {
        let mut r = Summary {
            functions: HashMap::new(),
            total_constraints: 0,
            total_savings_to_make: 0,
        };
        for (sig, cost) in self.functions.iter() {
            r.functions.insert(
                sig.clone(),
                SpecializationSummary {
                    calls: 0,
                    raw_constraints: cost.raw.constraints,
                    specialized_constraints: cost.specialized.constraints,
                    specialization_total_savings: 0,
                },
            );
        }
        self.walk_call_tree(&mut r, 1, self.entry_point.as_ref().unwrap());

        for (_, summary) in r.functions.iter_mut() {
            summary.specialization_total_savings =
                (summary.raw_constraints - summary.specialized_constraints) * summary.calls;
            r.total_constraints += summary.raw_constraints * summary.calls;
            r.total_savings_to_make += summary.specialization_total_savings;
        }
        r
    }

    fn walk_call_tree(&self, summary: &mut Summary, mul: usize, from_sig: &FunctionSignature) {
        let from = self.functions.get(&from_sig).unwrap();
        let from_summary = summary.functions.get_mut(from_sig).unwrap();
        from_summary.calls += mul;
        for (sig, count) in from.calls.iter() {
            self.walk_call_tree(summary, count * mul, sig);
        }
    }
}

pub struct CostEstimator {}

impl CostEstimator {
    pub fn new() -> Self {
        Self {}
    }

    #[instrument(skip_all, name = "CostEstimator::run")]
    pub fn run(
        &self,
        ssa: &SSA,
        type_info: &TypeInfo,
    ) -> CostAnalysis {
        let main_sig = self.make_main_sig(ssa);
        let mut costs = CostAnalysis {
            functions: HashMap::new(),
            stack: vec![],
            entry_point: Some(main_sig.clone()),
            cache: HashMap::new(),
        };

        self.run_fn_from_signature(ssa, type_info, main_sig, &mut costs);

        costs
    }

    fn run_fn_from_signature(
        &self,
        ssa: &SSA,
        type_info: &TypeInfo,
        sig: FunctionSignature,
        costs: &mut CostAnalysis,
    ) {
        let inputs: Vec<SpecSplitValue> = sig
            .params
            .iter()
            .map(|param| SpecSplitValue {
                // We need to call `to_value` twice, to avoid pointer aliasing.
                unspecialized: param.to_value(),
                specialized: param.to_value(),
            })
            .collect();
        SymbolicExecutor::new().run(ssa, type_info, sig.id, inputs, costs);
    }

    fn type_to_unknown_sig(&self, tp: &Type) -> ValueSignature {
        match &tp.expr {
            TypeExpr::Field => ValueSignature::Unknown(ScalarKind::Field),
            TypeExpr::U(s) => ValueSignature::Unknown(ScalarKind::U(*s)),
            TypeExpr::WitnessOf(inner) => {
                ValueSignature::WitnessOf(Box::new(self.type_to_unknown_sig(inner)))
            }
            TypeExpr::Array(elem, len) => {
                let elem_sig = self.type_to_unknown_sig(elem);
                ValueSignature::Array(vec![elem_sig; *len])
            }
            TypeExpr::Slice(elem) => {
                let elem_sig = self.type_to_unknown_sig(elem);
                ValueSignature::Array(vec![elem_sig; 0])
            }
            TypeExpr::Tuple(elems) => {
                ValueSignature::Tuple(
                    elems.iter().map(|e| self.type_to_unknown_sig(e)).collect()
                )
            }
            TypeExpr::Ref(inner) => {
                ValueSignature::PointerTo(Box::new(self.type_to_unknown_sig(inner)))
            }
            _ => ValueSignature::Unknown(ScalarKind::Field),
        }
    }

    fn make_main_sig(&self, ssa: &SSA) -> FunctionSignature {
        let id = ssa.get_main_id();
        let main_fn = ssa.get_function(id);
        let params = main_fn.get_param_types();
        let params = params
            .iter()
            .map(|param| self.type_to_unknown_sig(param))
            .collect();
        FunctionSignature { id, params }
    }
}
