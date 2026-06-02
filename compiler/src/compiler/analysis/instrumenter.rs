//! Implements a specialization cost-estimation analysis for the compiler.
//!
//! It functions by performing speculative specialization to estimate how many constraints, lookups,
//! and range-checks could be saved by certain specializations. This is done using symbolic
//! execution combined with an instrumenter for the circuit cost, and gives the compiler an idea of
//! how much a function could be shrunk through specialization on concrete inputs.

use std::{cell::RefCell, collections::HashMap, rc::Rc};

use ark_ff::{AdditiveGroup, BigInt, BigInteger, Field as _, PrimeField};
use itertools::Itertools;
use tracing::instrument;

use crate::compiler::{
    Field,
    analysis::{
        symbolic_executor::{self, SymbolicExecutor},
        types::TypeInfo,
    },
    ssa::{
        FunctionId,
        hlssa::{
            BinaryArithOpKind, CastTarget, CmpKind, Endianness, HLSSA, LookupTarget, Radix,
            RefCountOp, SequenceTargetType, SliceOpDir, Type, TypeExpr,
        },
    },
    util::{spread_bits, unspread_bits},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ScalarKind {
    Field,
    U(usize),
    I(usize),
}

impl ScalarKind {
    pub fn from_type(tp: &Type) -> Self {
        match &tp.strip_witness().expr {
            TypeExpr::Field => ScalarKind::Field,
            TypeExpr::U(s) => ScalarKind::U(*s),
            TypeExpr::I(s) => ScalarKind::I(*s),
            TypeExpr::WitnessOf(_) => panic!("WitnessOf is not a scalar type: {:?}", tp),
            _ => panic!("Not a scalar type: {:?}", tp),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ValueSignature {
    U { bits_size: usize, value: u128 },
    I { bits_size: usize, value: u128 },
    Field(Field),
    Array(Vec<ValueSignature>),
    PointerTo(Box<ValueSignature>),
    Unknown(ScalarKind),
    UnknownSlice,
    WitnessOf(Box<ValueSignature>),
    Tuple(Vec<ValueSignature>),
}

impl ValueSignature {
    pub fn to_value(&self) -> Value {
        match self {
            ValueSignature::U { bits_size, value } => Value::U(*bits_size, *value),
            ValueSignature::I { bits_size, value } => Value::I(*bits_size, *value),
            ValueSignature::Field(field) => Value::Field(*field),
            ValueSignature::Array(vals) => {
                Value::array(vals.iter().map(|v| v.to_value()).collect())
            }
            ValueSignature::PointerTo(val) => Value::Pointer(Rc::new(RefCell::new(val.to_value()))),
            ValueSignature::Unknown(kind) => Value::Unknown(*kind),
            ValueSignature::UnknownSlice => Value::UnknownSlice,
            ValueSignature::WitnessOf(inner) => Value::WitnessOf(Box::new(inner.to_value())),
            ValueSignature::Tuple(elements) => {
                Value::Tuple(elements.iter().map(|e| e.to_value()).collect())
            }
        }
    }

    pub fn pretty_print(&self, full: bool) -> String {
        match self {
            ValueSignature::U { value, .. } | ValueSignature::I { value, .. } => {
                format!("{value}")
            }
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
            ValueSignature::UnknownSlice => "?slice".to_string(),
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
pub struct ArrayData {
    values: Vec<Value>,
}

#[derive(Debug, Clone)]
pub enum Value {
    U(usize, u128),
    I(usize, u128),
    Field(Field),
    Array(Rc<RefCell<ArrayData>>),
    Pointer(Rc<RefCell<Value>>),
    Unknown(ScalarKind),
    UnknownSlice,
    WitnessOf(Box<Value>),
    Tuple(Vec<Value>),
}

impl Value {
    fn array(values: Vec<Value>) -> Self {
        Value::Array(Rc::new(RefCell::new(ArrayData { values })))
    }

    fn array_key(array: &Rc<RefCell<ArrayData>>) -> usize {
        Rc::as_ptr(array) as usize
    }

    fn flattened_table_len(&self) -> usize {
        match self {
            Value::Array(values) => values
                .borrow()
                .values
                .iter()
                .map(Value::flattened_table_len)
                .sum(),
            _ => 1,
        }
    }

    fn is_const_one(&self) -> bool {
        match self {
            Value::U(_, v) | Value::I(_, v) => *v == 1,
            Value::Field(f) => *f == Field::ONE,
            Value::WitnessOf(inner) => inner.is_const_one(),
            _ => false,
        }
    }

    fn as_field_const(&self) -> Option<Field> {
        match self {
            Value::U(_, v) | Value::I(_, v) => Some(Field::from(*v)),
            Value::Field(f) => Some(*f),
            Value::WitnessOf(inner) => inner.as_field_const(),
            _ => None,
        }
    }

    fn bit_mask(bits: usize) -> u128 {
        assert!(
            (1..=128).contains(&bits),
            "invalid integer width for bit mask: {bits}"
        );
        if bits == 128 {
            u128::MAX
        } else {
            (1u128 << bits) - 1
        }
    }

    /// Interpret a u128 two's-complement value as signed i128 for `bits` width.
    fn to_signed(val: u128, bits: usize) -> i128 {
        let mask = Self::bit_mask(bits);
        let val = val & mask;
        if bits == 128 {
            return val as i128;
        }
        let sign_bit = 1u128 << (bits - 1);
        if val & sign_bit != 0 {
            (val | !mask) as i128
        } else {
            val as i128
        }
    }

    /// Store a signed result back as u128 masked to `bits` width.
    fn from_signed(val: i128, bits: usize) -> u128 {
        (val as u128) & Self::bit_mask(bits)
    }

    fn unwrap_witness(&self) -> &Value {
        match self {
            Value::WitnessOf(inner) => inner.as_ref(),
            other => other,
        }
    }

    fn unknown_from_type(tp: &Type) -> Value {
        match &tp.expr {
            TypeExpr::Field => Value::Unknown(ScalarKind::Field),
            TypeExpr::U(s) => Value::Unknown(ScalarKind::U(*s)),
            TypeExpr::I(s) => Value::Unknown(ScalarKind::I(*s)),
            TypeExpr::WitnessOf(inner) => {
                Value::WitnessOf(Box::new(Value::unknown_from_type(inner)))
            }
            TypeExpr::Array(elem, n) => {
                let elem_unknown = Value::unknown_from_type(elem);
                Value::array(vec![elem_unknown; *n])
            }
            TypeExpr::Slice(_) => Value::UnknownSlice,
            TypeExpr::Tuple(elems) => {
                Value::Tuple(elems.iter().map(Value::unknown_from_type).collect())
            }
            TypeExpr::Ref(inner) => {
                Value::Pointer(Rc::new(RefCell::new(Value::unknown_from_type(inner))))
            }
            TypeExpr::Function => panic!("Cannot create unknown value for Function type"),
        }
    }

    fn ult_op(&self, b: &Value, instrumenter: &mut dyn OpInstrumenter) -> Value {
        match (self, b) {
            (Value::U(_, a), Value::U(_, b)) => Value::U(1, if a < b { 1 } else { 0 }),
            (Value::I(_, a), Value::I(_, b)) => Value::U(1, if a < b { 1 } else { 0 }),
            (Value::Field(a), Value::Field(b)) => Value::U(1, if a < b { 1 } else { 0 }),
            (Value::WitnessOf(_), _) | (_, Value::WitnessOf(_)) => {
                instrumenter.record_constraints(1);
                Value::WitnessOf(Box::new(
                    self.unwrap_witness()
                        .ult_op(b.unwrap_witness(), instrumenter),
                ))
            }
            (Value::Unknown(_), _) | (_, Value::Unknown(_)) => Value::Unknown(ScalarKind::U(1)),
            _ => panic!("Cannot compare {:?} and {:?}", self, b),
        }
    }

    fn slt_op(&self, b: &Value, bits: usize, instrumenter: &mut dyn OpInstrumenter) -> Value {
        match (self, b) {
            (Value::U(_, a), Value::U(_, b)) => {
                let (sa, sb) = (Self::to_signed(*a, bits), Self::to_signed(*b, bits));
                Value::U(1, if sa < sb { 1 } else { 0 })
            }
            (Value::I(s, a), Value::I(_, b)) => {
                let (sa, sb) = (Self::to_signed(*a, *s), Self::to_signed(*b, *s));
                Value::U(1, if sa < sb { 1 } else { 0 })
            }
            (Value::Field(a), Value::Field(b)) => Value::U(1, if a < b { 1 } else { 0 }),
            (Value::WitnessOf(_), _) | (_, Value::WitnessOf(_)) => {
                instrumenter.record_constraints(1);
                Value::WitnessOf(Box::new(self.unwrap_witness().slt_op(
                    b.unwrap_witness(),
                    bits,
                    instrumenter,
                )))
            }
            (Value::Unknown(_), _) | (_, Value::Unknown(_)) => Value::Unknown(ScalarKind::U(1)),
            _ => panic!("Cannot compare {:?} and {:?}", self, b),
        }
    }

    fn eq_op(&self, b: &Value, instrumenter: &mut dyn OpInstrumenter) -> Value {
        match (self, b) {
            (Value::U(_, a), Value::U(_, b)) | (Value::I(_, a), Value::I(_, b)) => {
                Value::U(1, if a == b { 1 } else { 0 })
            }
            (Value::Field(a), Value::Field(b)) => Value::U(1, if a == b { 1 } else { 0 }),
            (Value::WitnessOf(_), _) | (_, Value::WitnessOf(_)) => {
                instrumenter.record_constraints(1);
                Value::WitnessOf(Box::new(
                    self.unwrap_witness()
                        .eq_op(b.unwrap_witness(), instrumenter),
                ))
            }
            (Value::Unknown(_), _) | (_, Value::Unknown(_)) => Value::Unknown(ScalarKind::U(1)),
            _ => panic!("Cannot compare {:?} and {:?}", self, b),
        }
    }

    fn binary_arith_op(
        &self,
        b: &Value,
        binary_arith_op_kind: &crate::compiler::ssa::hlssa::BinaryArithOpKind,
        instrumenter: &mut dyn OpInstrumenter,
    ) -> Value {
        match binary_arith_op_kind {
            BinaryArithOpKind::Add => match (self, b) {
                (Value::U(s, a), Value::U(_, b)) => {
                    Value::U(*s, a.wrapping_add(*b) & Self::bit_mask(*s))
                }
                (Value::I(s, a), Value::I(_, b)) => Value::I(
                    *s,
                    Self::from_signed(
                        Self::to_signed(*a, *s).wrapping_add(Self::to_signed(*b, *s)),
                        *s,
                    ),
                ),
                (Value::Field(a), Value::Field(b)) => Value::Field(a + b),
                (Value::WitnessOf(_), _) | (_, Value::WitnessOf(_)) => {
                    Value::WitnessOf(Box::new(self.unwrap_witness().binary_arith_op(
                        b.unwrap_witness(),
                        binary_arith_op_kind,
                        instrumenter,
                    )))
                }
                (Value::Unknown(k), _) | (_, Value::Unknown(k)) => Value::Unknown(*k),
                _ => panic!("Cannot perform binary arithmetic on {:?} and {:?}", self, b),
            },
            BinaryArithOpKind::Sub => match (self, b) {
                (Value::U(s, a), Value::U(_, b)) => {
                    Value::U(*s, a.wrapping_sub(*b) & Self::bit_mask(*s))
                }
                (Value::I(s, a), Value::I(_, b)) => Value::I(
                    *s,
                    Self::from_signed(
                        Self::to_signed(*a, *s).wrapping_sub(Self::to_signed(*b, *s)),
                        *s,
                    ),
                ),
                (Value::Field(a), Value::Field(b)) => Value::Field(a - b),
                (Value::WitnessOf(_), _) | (_, Value::WitnessOf(_)) => {
                    Value::WitnessOf(Box::new(self.unwrap_witness().binary_arith_op(
                        b.unwrap_witness(),
                        binary_arith_op_kind,
                        instrumenter,
                    )))
                }
                (Value::Unknown(k), _) | (_, Value::Unknown(k)) => Value::Unknown(*k),
                _ => panic!("Cannot perform binary arithmetic on {:?} and {:?}", self, b),
            },
            BinaryArithOpKind::Mul => match (self, b) {
                (Value::U(s, 0), _) | (_, Value::U(s, 0)) => Value::U(*s, 0),
                (Value::Field(f), _) if *f == Field::ZERO => Value::Field(Field::ZERO),
                (_, Value::Field(f)) if *f == Field::ZERO => Value::Field(Field::ZERO),
                (Value::U(s, a), Value::U(_, b)) => {
                    Value::U(*s, a.wrapping_mul(*b) & Self::bit_mask(*s))
                }
                (Value::I(s, a), Value::I(_, b)) => Value::I(
                    *s,
                    Self::from_signed(
                        Self::to_signed(*a, *s).wrapping_mul(Self::to_signed(*b, *s)),
                        *s,
                    ),
                ),
                (Value::Field(a), Value::Field(b)) => Value::Field(a * b),
                (Value::WitnessOf(a), Value::WitnessOf(b)) => match (a.as_ref(), b.as_ref()) {
                    (Value::Unknown(_), Value::Unknown(_)) => {
                        instrumenter.record_high_degree_mul();
                        Value::WitnessOf(Box::new(a.binary_arith_op(
                            b,
                            binary_arith_op_kind,
                            instrumenter,
                        )))
                    }
                    _ => Value::WitnessOf(Box::new(a.binary_arith_op(
                        b,
                        binary_arith_op_kind,
                        instrumenter,
                    ))),
                },
                (Value::WitnessOf(_), _) | (_, Value::WitnessOf(_)) => {
                    Value::WitnessOf(Box::new(self.unwrap_witness().binary_arith_op(
                        b.unwrap_witness(),
                        binary_arith_op_kind,
                        instrumenter,
                    )))
                }
                (Value::Unknown(k), _) | (_, Value::Unknown(k)) => Value::Unknown(*k),
                _ => panic!("Cannot perform binary arithmetic on {:?} and {:?}", self, b),
            },
            BinaryArithOpKind::Div => match (self, b) {
                (Value::U(s, a), Value::U(_, b)) => Value::U(*s, a / b),
                (Value::I(s, a), Value::I(_, b)) => {
                    let (sa, sb) = (Self::to_signed(*a, *s), Self::to_signed(*b, *s));
                    Value::I(
                        *s,
                        Self::from_signed(if sb == 0 { 0 } else { sa.wrapping_div(sb) }, *s),
                    )
                }
                (Value::Field(a), Value::Field(b)) => Value::Field(a / b),
                (_, Value::WitnessOf(b)) => match b.as_ref() {
                    Value::Unknown(_) => {
                        instrumenter.record_constraints(1);
                        Value::WitnessOf(Box::new(self.unwrap_witness().binary_arith_op(
                            b,
                            binary_arith_op_kind,
                            instrumenter,
                        )))
                    }
                    _ => Value::WitnessOf(Box::new(self.unwrap_witness().binary_arith_op(
                        b,
                        binary_arith_op_kind,
                        instrumenter,
                    ))),
                },
                (Value::WitnessOf(a), b) => Value::WitnessOf(Box::new(a.binary_arith_op(
                    b,
                    binary_arith_op_kind,
                    instrumenter,
                ))),
                (Value::Unknown(k), _) | (_, Value::Unknown(k)) => Value::Unknown(*k),
                _ => panic!("Cannot perform binary arithmetic on {:?} and {:?}", self, b),
            },
            BinaryArithOpKind::Mod => match (self, b) {
                (Value::U(s, a), Value::U(_, b)) => Value::U(*s, a % b),
                (Value::I(s, a), Value::I(_, b)) => {
                    let (sa, sb) = (Self::to_signed(*a, *s), Self::to_signed(*b, *s));
                    Value::I(
                        *s,
                        Self::from_signed(if sb == 0 { 0 } else { sa.wrapping_rem(sb) }, *s),
                    )
                }
                (_, Value::WitnessOf(b)) => match b.as_ref() {
                    Value::Unknown(_) => {
                        instrumenter.record_constraints(1);
                        Value::WitnessOf(Box::new(self.unwrap_witness().binary_arith_op(
                            b,
                            binary_arith_op_kind,
                            instrumenter,
                        )))
                    }
                    _ => Value::WitnessOf(Box::new(self.unwrap_witness().binary_arith_op(
                        b,
                        binary_arith_op_kind,
                        instrumenter,
                    ))),
                },
                (Value::WitnessOf(a), b) => Value::WitnessOf(Box::new(a.binary_arith_op(
                    b,
                    binary_arith_op_kind,
                    instrumenter,
                ))),
                (Value::Unknown(k), _) | (_, Value::Unknown(k)) => Value::Unknown(*k),
                _ => panic!("Cannot perform binary arithmetic on {:?} and {:?}", self, b),
            },
            BinaryArithOpKind::And => match (self, b) {
                (Value::U(s, a), Value::U(_, b)) => Value::U(*s, a & b),
                (Value::I(s, a), Value::I(_, b)) => Value::I(*s, (a & b) & Self::bit_mask(*s)),
                (Value::Field(_), Value::Field(_)) => todo!(),
                (Value::WitnessOf(_), _) | (_, Value::WitnessOf(_)) => {
                    Value::WitnessOf(Box::new(self.unwrap_witness().binary_arith_op(
                        b.unwrap_witness(),
                        binary_arith_op_kind,
                        instrumenter,
                    )))
                }
                (Value::Unknown(k), _) | (_, Value::Unknown(k)) => Value::Unknown(*k),
                _ => panic!("Cannot perform binary arithmetic on {:?} and {:?}", self, b),
            },
            BinaryArithOpKind::Or => match (self, b) {
                (Value::U(s, a), Value::U(_, b)) => Value::U(*s, a | b),
                (Value::I(s, a), Value::I(_, b)) => Value::I(*s, (a | b) & Self::bit_mask(*s)),
                (Value::WitnessOf(_), _) | (_, Value::WitnessOf(_)) => {
                    Value::WitnessOf(Box::new(self.unwrap_witness().binary_arith_op(
                        b.unwrap_witness(),
                        binary_arith_op_kind,
                        instrumenter,
                    )))
                }
                (Value::Unknown(k), _) | (_, Value::Unknown(k)) => Value::Unknown(*k),
                _ => panic!("Cannot perform binary arithmetic on {:?} and {:?}", self, b),
            },
            BinaryArithOpKind::Xor => match (self, b) {
                (Value::U(s, a), Value::U(_, b)) => Value::U(*s, a ^ b),
                (Value::I(s, a), Value::I(_, b)) => Value::I(*s, (a ^ b) & Self::bit_mask(*s)),
                (Value::WitnessOf(_), _) | (_, Value::WitnessOf(_)) => {
                    Value::WitnessOf(Box::new(self.unwrap_witness().binary_arith_op(
                        b.unwrap_witness(),
                        binary_arith_op_kind,
                        instrumenter,
                    )))
                }
                (Value::Unknown(k), _) | (_, Value::Unknown(k)) => Value::Unknown(*k),
                _ => panic!("Cannot perform binary arithmetic on {:?} and {:?}", self, b),
            },
            BinaryArithOpKind::Shl => match (self, b) {
                (Value::U(s, a), Value::U(_, b)) => Value::U(
                    *s,
                    if *b as usize >= *s {
                        0
                    } else {
                        (a << b) & Self::bit_mask(*s)
                    },
                ),
                (Value::I(s, a), Value::I(_, b)) => {
                    let shift = Self::to_signed(*b, *s) as u32;
                    Value::I(
                        *s,
                        if shift as usize >= *s {
                            0
                        } else {
                            (a << shift) & Self::bit_mask(*s)
                        },
                    )
                }
                (Value::WitnessOf(_), _) | (_, Value::WitnessOf(_)) => {
                    Value::WitnessOf(Box::new(self.unwrap_witness().binary_arith_op(
                        b.unwrap_witness(),
                        binary_arith_op_kind,
                        instrumenter,
                    )))
                }
                (Value::Unknown(k), _) | (_, Value::Unknown(k)) => Value::Unknown(*k),
                _ => panic!("Cannot perform binary arithmetic on {:?} and {:?}", self, b),
            },
            BinaryArithOpKind::Shr => match (self, b) {
                (Value::U(s, a), Value::U(_, b)) => Value::U(*s, a >> b),
                (Value::I(s, a), Value::I(_, b)) => {
                    let sa = Self::to_signed(*a, *s);
                    let shift = Self::to_signed(*b, *s);
                    Value::I(
                        *s,
                        if shift < 0 || shift as usize >= *s {
                            // Over-shift: result is 0 or -1 depending on sign
                            Self::from_signed(if sa < 0 { -1 } else { 0 }, *s)
                        } else {
                            Self::from_signed(sa >> (shift as u32), *s)
                        },
                    )
                }
                (Value::WitnessOf(_), _) | (_, Value::WitnessOf(_)) => {
                    Value::WitnessOf(Box::new(self.unwrap_witness().binary_arith_op(
                        b.unwrap_witness(),
                        binary_arith_op_kind,
                        instrumenter,
                    )))
                }
                (Value::Unknown(k), _) | (_, Value::Unknown(k)) => Value::Unknown(*k),
                _ => panic!("Cannot perform binary arithmetic on {:?} and {:?}", self, b),
            },
        }
    }

    fn blind(&mut self) {
        match self {
            Value::WitnessOf(inner) => {
                inner.forget_concrete();
            }
            Value::Unknown(_) | Value::UnknownSlice => {}
            Value::U(_, _) | Value::I(_, _) | Value::Field(_) => {}
            Value::Array(vals) => {
                for val in vals.borrow_mut().values.iter_mut() {
                    val.blind();
                }
            }
            Value::Pointer(val) => {
                val.borrow_mut().blind();
            }
            Value::Tuple(vals) => {
                for val in vals {
                    val.blind();
                }
            }
        }
    }

    fn forget_concrete(&mut self) {
        match self {
            Value::U(s, _) => *self = Value::Unknown(ScalarKind::U(*s)),
            Value::I(s, _) => *self = Value::Unknown(ScalarKind::I(*s)),
            Value::Field(_) => *self = Value::Unknown(ScalarKind::Field),
            Value::Unknown(_) | Value::UnknownSlice => {}
            Value::WitnessOf(inner) => {
                inner.forget_concrete();
            }
            Value::Array(vals) => {
                for val in vals.borrow_mut().values.iter_mut() {
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
            Value::UnknownSlice => ValueSignature::UnknownSlice,
            Value::WitnessOf(inner) => {
                ValueSignature::WitnessOf(Box::new(inner.make_unspecialized_sig()))
            }
            Value::U(s, v) => ValueSignature::U {
                bits_size: *s,
                value: *v,
            },
            Value::I(s, v) => ValueSignature::I {
                bits_size: *s,
                value: *v,
            },
            Value::Field(f) => ValueSignature::Field(*f),
            Value::Array(vals) => ValueSignature::Array(
                vals.borrow()
                    .values
                    .iter()
                    .map(|v| v.make_unspecialized_sig())
                    .collect(),
            ),
            Value::Pointer(val) => {
                ValueSignature::PointerTo(Box::new(val.borrow().make_unspecialized_sig()))
            }
            Value::Tuple(vals) => {
                ValueSignature::Tuple(vals.iter().map(|v| v.make_unspecialized_sig()).collect())
            }
        }
    }

    fn assert_bool(&self, instrumenter: &mut dyn OpInstrumenter) {
        if self.is_witness() {
            instrumenter.record_constraints(1);
        }
    }

    fn assert_cmp(
        _kind: CmpKind,
        a: &Self,
        b: &Self,
        _lhs_type: &Type,
        instrumenter: &mut dyn OpInstrumenter,
    ) {
        if a.is_witness() || b.is_witness() {
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

    fn array_get(&self, index: &Value, tp: &Type, instrumenter: &mut dyn OpInstrumenter) -> Value {
        if matches!(self, Value::Unknown(_) | Value::UnknownSlice) {
            return Value::unknown_from_type(tp);
        }

        let arr = self.unwrap_witness();

        match (arr, index) {
            (Value::Array(vals), Value::U(_, index)) => {
                vals.borrow().values[*index as usize].clone()
            }
            (Value::Array(vals), Value::WitnessOf(inner)) => match inner.as_ref() {
                Value::U(_, index) => vals.borrow().values[*index as usize].clone(),
                _ => {
                    instrumenter.record_lookups(vals.borrow().values.len(), 1, 1);
                    Value::unknown_from_type(tp)
                }
            },
            (Value::Array(vals), Value::Unknown(_)) => {
                instrumenter.record_lookups(vals.borrow().values.len(), 1, 1);
                Value::unknown_from_type(tp)
            }
            (Value::Unknown(_) | Value::UnknownSlice, _) => Value::unknown_from_type(tp),
            _ => panic!(
                "Cannot get array element from {:?} with index {:?}",
                self, index
            ),
        }
    }

    fn tuple_get(&self, index: usize) -> Value {
        match self {
            Value::Unknown(_) => Value::Unknown(ScalarKind::Field),
            Value::WitnessOf(inner) => inner.tuple_get(index),
            Value::Tuple(vals) => vals[index].clone(),
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
                let mut new_vals = vals.borrow().values.clone();
                new_vals[*index as usize] = value.clone();
                Value::array(new_vals)
            }
            (Value::Array(vals), Value::WitnessOf(inner), value) => match inner.as_ref() {
                Value::U(_, index) => {
                    let mut new_vals = vals.borrow().values.clone();
                    new_vals[*index as usize] = value.clone();
                    Value::array(new_vals)
                }
                _ => {
                    let new_vals = vals.borrow().values.iter().map(|_| value.clone()).collect();
                    Value::array(new_vals)
                }
            },
            (Value::Array(vals), _, value) => {
                let new_vals = vals.borrow().values.iter().map(|_| value.clone()).collect();
                Value::array(new_vals)
            }
            (Value::UnknownSlice, _, _) => Value::UnknownSlice,
            _ => panic!(
                "Cannot set array element of {:?} with index {:?} to {:?}",
                self, index, value
            ),
        }
    }

    fn bit_range_op(
        &self,
        offset: usize,
        width: usize,
        _instrumenter: &mut dyn OpInstrumenter,
    ) -> Value {
        match self {
            Value::Unknown(kind) => Value::Unknown(*kind),
            Value::WitnessOf(inner) => {
                Value::WitnessOf(Box::new(inner.bit_range_op(offset, width, _instrumenter)))
            }
            Value::U(bits, v) => Value::U(*bits, (v >> offset) & Self::bit_mask(width)),
            Value::I(bits, v) => Value::I(*bits, (v >> offset) & Self::bit_mask(width)),
            Value::Field(f) => {
                let bits = f
                    .into_bigint()
                    .to_bits_le()
                    .into_iter()
                    .skip(offset)
                    .take(width)
                    .collect::<Vec<_>>();
                let r = Field::from_bigint(BigInt::from_bits_le(&bits));
                Value::Field(r.unwrap())
            }
            _ => panic!("Cannot extract bit range from {:?}", self),
        }
    }

    fn sext_op(&self, from: usize, to: usize, _instrumenter: &mut dyn OpInstrumenter) -> Value {
        match self {
            Value::Unknown(kind) => Value::Unknown(*kind),
            Value::WitnessOf(inner) => {
                Value::WitnessOf(Box::new(inner.sext_op(from, to, _instrumenter)))
            }
            Value::U(_, v) => {
                // Sign-extend: check sign bit at position from-1
                let sign_bit = if from > 0 { (v >> (from - 1)) & 1 } else { 0 };
                if sign_bit == 1 {
                    let mask = ((1u128 << to) - 1) ^ ((1u128 << from) - 1);
                    Value::U(to, v | mask)
                } else {
                    Value::U(to, *v)
                }
            }
            _ => panic!("Cannot sext {:?}", self),
        }
    }

    fn spread_op(&self, instrumenter: &mut dyn OpInstrumenter) -> Value {
        match self {
            Value::U(bits, v) => {
                assert!(
                    *bits <= 64,
                    "Spread only supports integer widths up to 64 bits, got u{}",
                    bits
                );
                Value::U(bits * 2, spread_bits(*v, *bits))
            }
            Value::I(bits, v) => {
                assert!(
                    *bits <= 64,
                    "Spread only supports integer widths up to 64 bits, got i{}",
                    bits
                );
                Value::I(bits * 2, spread_bits(*v, *bits))
            }
            Value::Field(_) => panic!("Spread of field values is unsupported"),
            Value::WitnessOf(inner) => {
                if matches!(inner.as_ref(), Value::Unknown(_)) {
                    instrumenter.record_lookups(256, 1, 1);
                }
                Value::WitnessOf(Box::new(inner.spread_op(instrumenter)))
            }
            Value::Unknown(ScalarKind::U(bits)) => {
                assert!(
                    *bits <= 64,
                    "Spread only supports integer widths up to 64 bits, got u{}",
                    bits
                );
                Value::Unknown(ScalarKind::U(bits * 2))
            }
            Value::Unknown(ScalarKind::I(bits)) => {
                assert!(
                    *bits <= 64,
                    "Spread only supports integer widths up to 64 bits, got i{}",
                    bits
                );
                Value::Unknown(ScalarKind::I(bits * 2))
            }
            Value::Unknown(ScalarKind::Field) => panic!("Spread of field values is unsupported"),
            _ => panic!("Cannot spread {:?}", self),
        }
    }

    fn unspread_op(&self, instrumenter: &mut dyn OpInstrumenter) -> (Value, Value) {
        match self {
            Value::U(bits, v) => {
                assert!(
                    *bits <= 128 && bits % 2 == 0,
                    "Unspread expects an even integer width up to 128 bits, got u{}",
                    bits
                );
                let (odd_val, even_val) = unspread_bits(*v, *bits);
                let half_bits = bits / 2;
                (Value::U(half_bits, odd_val), Value::U(half_bits, even_val))
            }
            Value::I(bits, v) => {
                assert!(
                    *bits <= 128 && bits % 2 == 0,
                    "Unspread expects an even integer width up to 128 bits, got i{}",
                    bits
                );
                let (odd_val, even_val) = unspread_bits(*v, *bits);
                let half_bits = bits / 2;
                (Value::I(half_bits, odd_val), Value::I(half_bits, even_val))
            }
            Value::Field(_) => panic!("Unspread of field values is unsupported"),
            Value::WitnessOf(inner) => {
                if matches!(inner.as_ref(), Value::Unknown(_)) {
                    instrumenter.record_constraints(1);
                }
                let (odd, even) = inner.unspread_op(instrumenter);
                (
                    Value::WitnessOf(Box::new(odd)),
                    Value::WitnessOf(Box::new(even)),
                )
            }
            Value::Unknown(ScalarKind::U(bits)) => {
                assert!(
                    *bits <= 128 && bits % 2 == 0,
                    "Unspread expects an even integer width up to 128 bits, got u{}",
                    bits
                );
                let half_bits = bits / 2;
                (
                    Value::Unknown(ScalarKind::U(half_bits)),
                    Value::Unknown(ScalarKind::U(half_bits)),
                )
            }
            Value::Unknown(ScalarKind::I(bits)) => {
                assert!(
                    *bits <= 128 && bits % 2 == 0,
                    "Unspread expects an even integer width up to 128 bits, got i{}",
                    bits
                );
                let half_bits = bits / 2;
                (
                    Value::Unknown(ScalarKind::I(half_bits)),
                    Value::Unknown(ScalarKind::I(half_bits)),
                )
            }
            Value::Unknown(ScalarKind::Field) => {
                panic!("Unspread of field values is unsupported")
            }
            _ => panic!("Cannot unspread {:?}", self),
        }
    }

    fn cast_op(
        &self,
        cast_target: &crate::compiler::ssa::hlssa::CastTarget,
        _instrumenter: &mut dyn OpInstrumenter,
    ) -> Value {
        match (self, cast_target) {
            (_, CastTarget::WitnessOf) => Value::WitnessOf(Box::new(self.clone())),
            (Value::Unknown(_), CastTarget::U(s)) => Value::Unknown(ScalarKind::U(*s)),
            (Value::Unknown(_), CastTarget::I(s)) => Value::Unknown(ScalarKind::I(*s)),
            (Value::Unknown(_), CastTarget::Field) => Value::Unknown(ScalarKind::Field),
            (Value::Unknown(kind), CastTarget::Nop | CastTarget::ArrayToSlice) => {
                Value::Unknown(*kind)
            }
            (Value::WitnessOf(inner), target) => {
                Value::WitnessOf(Box::new(inner.cast_op(target, _instrumenter)))
            }
            (Value::U(_, v), CastTarget::U(s2)) => Value::U(*s2, *v & Self::bit_mask(*s2)),
            (Value::U(_, v), CastTarget::I(s2)) => Value::I(*s2, *v & Self::bit_mask(*s2)),
            (Value::I(_, v), CastTarget::U(s2)) => Value::U(*s2, *v & Self::bit_mask(*s2)),
            (Value::I(_, v), CastTarget::I(s2)) => Value::I(*s2, *v & Self::bit_mask(*s2)),
            (Value::U(_, v), CastTarget::Field) => Value::Field(Field::from(*v)),
            (Value::I(s, v), CastTarget::Field) => {
                Value::Field(Field::from(Self::to_signed(*v, *s) as u64))
            }
            (Value::Field(f), CastTarget::Field) => Value::Field(*f),
            (Value::Field(f), CastTarget::U(s)) => {
                let bigint = f.into_bigint();
                Value::U(
                    *s,
                    (bigint.0[0] as u128 | ((bigint.0[1] as u128) << 64)) & Self::bit_mask(*s),
                )
            }
            (Value::Field(f), CastTarget::I(s)) => {
                let bigint = f.into_bigint();
                Value::I(
                    *s,
                    (bigint.0[0] as u128 | ((bigint.0[1] as u128) << 64)) & Self::bit_mask(*s),
                )
            }
            (_, CastTarget::Nop | CastTarget::ArrayToSlice) => self.clone(),
            _ => panic!("Cannot cast {:?} to {:?}", self, cast_target),
        }
    }

    fn constrain(_a: &Value, _b: &Value, _c: &Value, instrumenter: &mut dyn OpInstrumenter) {
        match (
            _a.as_field_const(),
            _b.as_field_const(),
            _c.as_field_const(),
        ) {
            (Some(a), Some(b), Some(c)) => assert_eq!(a * b, c),
            _ => instrumenter.record_constrain(),
        }
    }

    fn to_bits(&self, endianness: &crate::compiler::ssa::hlssa::Endianness, size: usize) -> Value {
        match self {
            Value::Unknown(kind) => Value::Unknown(*kind),
            Value::WitnessOf(inner) => {
                let result = inner.to_bits(endianness, size);
                match result {
                    Value::Array(bits) => Value::array(
                        bits.borrow()
                            .values
                            .iter()
                            .cloned()
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
                Value::array(r)
            }
            Value::Field(f) => {
                let bigint = f.into_bigint();
                let bits = bigint.to_bits_le().into_iter().take(size);
                let mut bits = bits.map(|b| Value::U(1, b as u128)).collect::<Vec<_>>();
                if *endianness == Endianness::Big {
                    bits.reverse();
                }
                Value::array(bits)
            }
            _ => panic!("Cannot convert {:?} to bits", self),
        }
    }

    fn to_radix(
        &self,
        radix: &Radix<Value>,
        _endianness: &crate::compiler::ssa::hlssa::Endianness,
        size: usize,
        instrumenter: &mut dyn OpInstrumenter,
    ) -> Value {
        match self {
            Value::WitnessOf(inner) => {
                let result = inner.to_radix(radix, _endianness, size, instrumenter);
                match result {
                    Value::Array(digits) => Value::array(
                        digits
                            .borrow()
                            .values
                            .iter()
                            .cloned()
                            .map(|d| Value::WitnessOf(Box::new(d)))
                            .collect(),
                    ),
                    _ => unreachable!(),
                }
            }
            Value::Unknown(_) => {
                instrumenter.record_rangechecks(8, size);
                instrumenter.record_constraints(1);
                Value::array(vec![Value::Unknown(ScalarKind::U(8)); size])
            }
            Value::Field(f) => {
                let radix_val = match radix {
                    Radix::Dyn(Value::U(_, r)) => *r,
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
                Value::array(digits)
            }
            _ => panic!("Cannot convert {:?} to radix {:?}", self, radix),
        }
    }

    fn not_op(&self, _instrumenter: &mut dyn OpInstrumenter) -> Value {
        match self {
            Value::Unknown(kind) => Value::Unknown(*kind),
            Value::WitnessOf(inner) => Value::WitnessOf(Box::new(inner.not_op(_instrumenter))),
            Value::U(s, v) => Value::U(*s, !v & Self::bit_mask(*s)),
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

    fn assert_r1c(_a: &Value, _b: &Value, _c: &Value, _get_unspecialized: &mut dyn OpInstrumenter) {
    }

    fn select(
        &self,
        if_true: &Value,
        if_false: &Value,
        _tp: &Type,
        instrumenter: &mut dyn OpInstrumenter,
    ) -> Value {
        match self {
            Value::U(_, 0) => if_false.clone(),
            Value::U(_, _) => if_true.clone(),
            Value::WitnessOf(inner) => match inner.as_ref() {
                Value::U(_, 0) => if_false.clone(),
                Value::U(_, _) => if_true.clone(),
                _ => {
                    if self.is_witness() && (if_true.is_witness() || if_false.is_witness()) {
                        instrumenter.record_constraints(1);
                    }
                    let mut result = if_true.clone();
                    result.forget_concrete();
                    result
                }
            },
            Value::Unknown(_) => {
                let mut result = if_true.clone();
                result.forget_concrete();
                result
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
    fn blind_unspecialized(&mut self) {
        self.unspecialized.blind();
    }

    fn blind(&mut self) {
        self.unspecialized.blind();
        self.specialized.blind();
    }
}

impl symbolic_executor::Value<CostAnalysis> for SpecSplitValue {
    fn ult(&self, b: &SpecSplitValue, instrumenter: &mut CostAnalysis) -> SpecSplitValue {
        let unspecialized = self
            .unspecialized
            .ult_op(&b.unspecialized, instrumenter.get_unspecialized());
        let specialized = self
            .specialized
            .ult_op(&b.specialized, instrumenter.get_specialized());
        SpecSplitValue {
            unspecialized,
            specialized,
        }
    }

    fn slt(
        &self,
        b: &SpecSplitValue,
        bits: usize,
        instrumenter: &mut CostAnalysis,
    ) -> SpecSplitValue {
        let unspecialized =
            self.unspecialized
                .slt_op(&b.unspecialized, bits, instrumenter.get_unspecialized());
        let specialized =
            self.specialized
                .slt_op(&b.specialized, bits, instrumenter.get_specialized());
        SpecSplitValue {
            unspecialized,
            specialized,
        }
    }

    fn eq(&self, b: &SpecSplitValue, instrumenter: &mut CostAnalysis) -> SpecSplitValue {
        let unspecialized = self
            .unspecialized
            .eq_op(&b.unspecialized, instrumenter.get_unspecialized());
        let specialized = self
            .specialized
            .eq_op(&b.specialized, instrumenter.get_specialized());
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
        cast_target: &crate::compiler::ssa::hlssa::CastTarget,
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

    fn bit_range(
        &self,
        offset: usize,
        width: usize,
        _tp: &Type,
        instrumenter: &mut CostAnalysis,
    ) -> SpecSplitValue {
        SpecSplitValue {
            unspecialized: self.unspecialized.bit_range_op(
                offset,
                width,
                instrumenter.get_unspecialized(),
            ),
            specialized: self.specialized.bit_range_op(
                offset,
                width,
                instrumenter.get_specialized(),
            ),
        }
    }

    fn sext(
        &self,
        from: usize,
        to: usize,
        _tp: &Type,
        instrumenter: &mut CostAnalysis,
    ) -> SpecSplitValue {
        SpecSplitValue {
            unspecialized: self
                .unspecialized
                .sext_op(from, to, instrumenter.get_unspecialized()),
            specialized: self
                .specialized
                .sext_op(from, to, instrumenter.get_specialized()),
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
        res.blind_unspecialized();
        res
    }

    fn mk_array(
        values: Vec<SpecSplitValue>,
        _ctx: &mut CostAnalysis,
        _seq_type: SequenceTargetType,
        _elem_type: &Type,
    ) -> SpecSplitValue {
        let (uns, spec) = values
            .into_iter()
            .map(|v| (v.unspecialized, v.specialized))
            .unzip();
        SpecSplitValue {
            unspecialized: Value::array(uns),
            specialized: Value::array(spec),
        }
    }

    fn mk_tuple(
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
        _tp: &Type,
        _instrumenter: &mut CostAnalysis,
    ) -> SpecSplitValue {
        SpecSplitValue {
            unspecialized: self.unspecialized.tuple_get(index),
            specialized: self.specialized.tuple_get(index),
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

    fn assert_bool(&self, instrumenter: &mut CostAnalysis) {
        self.specialized.assert_bool(instrumenter.get_specialized());
        self.unspecialized
            .assert_bool(instrumenter.get_unspecialized());
    }

    fn assert_cmp(
        kind: CmpKind,
        a: &Self,
        b: &Self,
        lhs_type: &Type,
        instrumenter: &mut CostAnalysis,
    ) {
        Value::assert_cmp(
            kind,
            &a.specialized,
            &b.specialized,
            lhs_type,
            instrumenter.get_specialized(),
        );
        Value::assert_cmp(
            kind,
            &a.unspecialized,
            &b.unspecialized,
            lhs_type,
            instrumenter.get_unspecialized(),
        );
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
        _instrumenter: &mut CostAnalysis,
    ) -> SpecSplitValue {
        SpecSplitValue {
            unspecialized: self.unspecialized.to_bits(&endianness, size),
            specialized: self.specialized.to_bits(&endianness, size),
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
        let specialized = match &self.specialized {
            Value::U(1, v) => *v != 0,
            _ => panic!(
                "Expected constant bool, got specialized={:?}",
                self.specialized
            ),
        };
        let unspecialized = match &self.unspecialized {
            Value::U(1, v) => *v != 0,
            _ => panic!(
                "Expected constant bool, got unspecialized={:?}",
                self.unspecialized
            ),
        };
        assert_eq!(specialized, unspecialized);
        specialized
    }

    fn of_u(s: usize, v: u128, _ctx: &mut CostAnalysis) -> Self {
        Self {
            unspecialized: Value::U(s, v),
            specialized: Value::U(s, v),
        }
    }

    fn of_i(s: usize, v: u128, _ctx: &mut CostAnalysis) -> Self {
        Self {
            unspecialized: Value::I(s, v),
            specialized: Value::I(s, v),
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

    fn value_of(&self, _ctx: &mut CostAnalysis) -> Self {
        Self {
            unspecialized: self.unspecialized.unwrap_witness().clone(),
            specialized: self.specialized.unwrap_witness().clone(),
        }
    }

    fn mem_op(&self, _kind: RefCountOp, _ctx: &mut CostAnalysis) {}

    fn spread(&self, _bits: u8, instrumenter: &mut CostAnalysis) -> Self {
        Self {
            unspecialized: self
                .unspecialized
                .spread_op(instrumenter.get_unspecialized()),
            specialized: self.specialized.spread_op(instrumenter.get_specialized()),
        }
    }

    fn unspread(&self, _bits: u8, instrumenter: &mut CostAnalysis) -> (Self, Self) {
        let (unspec_odd, unspec_even) = self
            .unspecialized
            .unspread_op(instrumenter.get_unspecialized());
        let (spec_odd, spec_even) = self.specialized.unspread_op(instrumenter.get_specialized());
        (
            Self {
                unspecialized: unspec_odd,
                specialized: spec_odd,
            },
            Self {
                unspecialized: unspec_even,
                specialized: spec_even,
            },
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FunctionSignature {
    id: FunctionId,
    params: Vec<ValueSignature>,
}

impl FunctionSignature {
    pub fn pretty_print(&self, ssa: &HLSSA, all_params: bool) -> String {
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
    fn record_constrain(&mut self);
    fn record_high_degree_mul(&mut self);
    fn record_rangechecks(&mut self, size: u8, count: usize);
    fn record_lookups(&mut self, keys: usize, results: usize, count: usize);
    fn record_lookup(&mut self, target: &LookupTarget<Value>, args: &[Value], flag: &Value);
}

trait FunctionInstrumenter {
    fn get_specialized(&mut self) -> &mut dyn OpInstrumenter;
    fn get_unspecialized(&mut self) -> &mut dyn OpInstrumenter;
    fn record_call(&mut self, sig: FunctionSignature);
    fn seal(self: Box<Self>) -> FunctionCost;
}

#[derive(Debug, Clone)]
struct Instrumenter {
    constrains: usize,
    high_degree_muls: usize,

    /// Original pre-spilling lookup requests, kept for reporting.
    rangecheck_lookups: HashMap<u8, usize>,
    spread_lookups: HashMap<u8, usize>,
    array_lookups: usize,

    /// Extra algebraic constraints introduced by lookup spilling.
    rangecheck_one_constraints: usize,
    spilled_wide_spread_constraints: usize,

    /// Final table lookup requests after applying the same spilling expansion
    /// that runs before R1CS generation.
    final_rangecheck8_lookups: usize,
    final_spread_lookups: HashMap<u8, usize>,
    total_table_lookups: usize,

    /// Array tables are allocated per array value identity.
    allocated_array_tables: HashMap<usize, usize>,
}

impl Instrumenter {
    fn new() -> Self {
        Self {
            constrains: 0,
            high_degree_muls: 0,
            rangecheck_lookups: HashMap::new(),
            spread_lookups: HashMap::new(),
            array_lookups: 0,
            rangecheck_one_constraints: 0,
            spilled_wide_spread_constraints: 0,
            final_rangecheck8_lookups: 0,
            final_spread_lookups: HashMap::new(),
            total_table_lookups: 0,
            allocated_array_tables: HashMap::new(),
        }
    }

    fn record_rangecheck_lookup(&mut self, bits: u8, flag: &Value) {
        assert!(bits >= 1, "rangecheck width must be at least 1 bit");
        *self.rangecheck_lookups.entry(bits).or_insert(0) += 1;

        if bits == 1 {
            self.rangecheck_one_constraints += if flag.is_const_one() { 1 } else { 2 };
            return;
        }

        let bits = bits as usize;
        let final_lookups = if bits == 8 {
            1
        } else {
            let full_bytes = bits / 8;
            let leftover_bits = bits % 8;
            full_bytes + if leftover_bits > 0 { 2 } else { 0 }
        };
        self.final_rangecheck8_lookups += final_lookups;
        self.total_table_lookups += final_lookups;
    }

    fn record_spread_lookup(&mut self, bits: u8) {
        assert!(bits >= 1, "spread width must be at least 1 bit");
        *self.spread_lookups.entry(bits).or_insert(0) += 1;

        if bits >= 16 {
            assert!(
                bits <= 128,
                "wide Spread lookup spilling currently supports widths up to 128 bits, got {bits}"
            );
            self.spilled_wide_spread_constraints += 1;
            let mut offset = 0usize;
            let bits = bits as usize;
            while offset < bits {
                let chunk_bits = (bits - offset).min(8) as u8;
                self.record_final_spread_lookup(chunk_bits);
                offset += chunk_bits as usize;
            }
        } else {
            self.record_final_spread_lookup(bits);
        }
    }

    fn record_final_spread_lookup(&mut self, bits: u8) {
        *self.final_spread_lookups.entry(bits).or_insert(0) += 1;
        self.total_table_lookups += 1;
    }

    fn record_array_lookup(&mut self, array: &Value) {
        let Value::Array(array) = array else {
            panic!("array lookup target must be an array, got {:?}", array);
        };
        self.array_lookups += 1;
        self.total_table_lookups += 1;
        self.allocated_array_tables
            .entry(Value::array_key(array))
            .or_insert_with(|| Value::Array(array.clone()).flattened_table_len());
    }

    fn table_allocation_constraints(&self) -> usize {
        let range_constraints = if self.rangecheck_lookups.keys().any(|bits| *bits >= 2) {
            (1usize << 8) + 1
        } else {
            0
        };
        let spread_constraints = self
            .final_spread_lookups
            .keys()
            .filter(|bits| **bits >= 2)
            .map(|bits| 2 * (1usize << *bits as usize) + 1)
            .sum::<usize>();
        let array_constraints = self
            .allocated_array_tables
            .values()
            .map(|len| 2 * len + 1)
            .sum::<usize>();
        range_constraints + spread_constraints + array_constraints
    }

    fn array_table_allocation_constraints(&self) -> usize {
        self.allocated_array_tables
            .values()
            .map(|len| 2 * len + 1)
            .sum()
    }

    fn lookup_data_constraints(&self) -> usize {
        self.final_rangecheck8_lookups
            + self
                .final_spread_lookups
                .values()
                .map(|count| count * 2)
                .sum::<usize>()
            + self.array_lookups * 2
    }

    fn recurring_constraints(&self) -> usize {
        self.constrains
            + self.high_degree_muls
            + self.rangecheck_one_constraints
            + self.spilled_wide_spread_constraints
            + self.lookup_data_constraints()
    }

    fn total_constraints(&self) -> usize {
        self.recurring_constraints() + self.table_allocation_constraints()
    }

    fn specialization_constraints(&self) -> usize {
        self.recurring_constraints() + self.array_table_allocation_constraints()
    }

    fn allocated_lookup_table_rows(&self) -> usize {
        let range_rows = if self.rangecheck_lookups.keys().any(|bits| *bits >= 2) {
            1usize << 8
        } else {
            0
        };
        let spread_rows = self
            .final_spread_lookups
            .keys()
            .filter(|bits| **bits >= 2)
            .map(|bits| 1usize << *bits as usize)
            .sum::<usize>();
        let array_rows = self.allocated_array_tables.values().sum::<usize>();
        range_rows + spread_rows + array_rows
    }

    fn detail_line(&self) -> String {
        format!(
            "constrain={}, high_deg_mul={}, lookups={}, table_rows={}, table_constraints={}",
            self.constrains,
            self.high_degree_muls,
            self.total_table_lookups,
            self.allocated_lookup_table_rows(),
            self.table_allocation_constraints()
        )
    }
}

impl OpInstrumenter for Instrumenter {
    fn record_constraints(&mut self, _number: usize) {}

    fn record_constrain(&mut self) {
        self.constrains += 1;
    }

    fn record_high_degree_mul(&mut self) {
        self.high_degree_muls += 1;
    }

    fn record_rangechecks(&mut self, _size: u8, _count: usize) {}

    fn record_lookups(&mut self, _keys: usize, _results: usize, _count: usize) {}

    fn record_lookup(&mut self, target: &LookupTarget<Value>, _args: &[Value], flag: &Value) {
        match target {
            LookupTarget::Rangecheck(bits) => self.record_rangecheck_lookup(*bits, flag),
            LookupTarget::DynRangecheck(bound) => {
                let bits = dynamic_rangecheck_bits(bound);
                self.record_rangecheck_lookup(bits, flag);
            }
            LookupTarget::Spread(bits) => self.record_spread_lookup(*bits),
            LookupTarget::Array(array) => self.record_array_lookup(array),
        }
    }
}

fn dynamic_rangecheck_bits(bound: &Value) -> u8 {
    let bound = match bound {
        Value::U(_, v) | Value::I(_, v) => *v,
        Value::Field(f) => {
            let bigint = f.into_bigint();
            bigint.0[0] as u128 | ((bigint.0[1] as u128) << 64)
        }
        Value::WitnessOf(inner) => return dynamic_rangecheck_bits(inner),
        other => panic!("dynamic rangecheck bound must be constant, got {:?}", other),
    };
    assert_eq!(bound, 256, "TODO: support dynamic rangecheck bound {bound}");
    8
}

fn map_lookup_target<T, U>(
    target: &LookupTarget<T>,
    mut map_value: impl FnMut(&T) -> U,
) -> LookupTarget<U> {
    match target {
        LookupTarget::Rangecheck(bits) => LookupTarget::Rangecheck(*bits),
        LookupTarget::DynRangecheck(bound) => LookupTarget::DynRangecheck(map_value(bound)),
        LookupTarget::Array(array) => LookupTarget::Array(map_value(array)),
        LookupTarget::Spread(bits) => LookupTarget::Spread(*bits),
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
    fn record_constrain(&mut self) {}
    fn record_high_degree_mul(&mut self) {}
    fn record_rangechecks(&mut self, _: u8, _: usize) {}
    fn record_lookups(&mut self, _: usize, _: usize, _: usize) {}
    fn record_lookup(&mut self, _: &LookupTarget<Value>, _: &[Value], _: &Value) {}
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
        result_types: &[Type],
        unconstrained: bool,
    ) -> Option<Vec<SpecSplitValue>> {
        if unconstrained {
            fn unknown_value(ty: &Type) -> Value {
                match &ty.expr {
                    TypeExpr::Field => Value::Unknown(ScalarKind::Field),
                    TypeExpr::U(s) => Value::Unknown(ScalarKind::U(*s)),
                    TypeExpr::I(s) => Value::Unknown(ScalarKind::I(*s)),
                    TypeExpr::Array(elem, size) => {
                        Value::array((0..*size).map(|_| unknown_value(elem)).collect())
                    }
                    TypeExpr::Tuple(elems) => {
                        Value::Tuple(elems.iter().map(unknown_value).collect())
                    }
                    TypeExpr::WitnessOf(inner) => Value::WitnessOf(Box::new(unknown_value(inner))),
                    TypeExpr::Ref(inner) => {
                        Value::Pointer(Rc::new(RefCell::new(unknown_value(inner))))
                    }
                    _ => panic!("Unsupported type for unknown value: {:?}", ty),
                }
            }
            return Some(
                result_types
                    .iter()
                    .map(|ty| {
                        let v = unknown_value(ty);
                        SpecSplitValue {
                            unspecialized: v.clone(),
                            specialized: v,
                        }
                    })
                    .collect(),
            );
        }

        for (pval, _ptype) in params.iter_mut().zip(param_types.iter()) {
            pval.blind();
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

    fn on_return(&mut self, returns: &mut [SpecSplitValue], _return_types: &[Type]) {
        for rval in returns.iter_mut() {
            rval.blind();
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
        _param_types: &[&Type],
    ) {
        for pval in params.iter_mut() {
            pval.blind_unspecialized();
        }
    }

    fn lookup(
        &mut self,
        target: LookupTarget<SpecSplitValue>,
        args: Vec<SpecSplitValue>,
        flag: SpecSplitValue,
    ) {
        let unspecialized_target = map_lookup_target(&target, |v| v.unspecialized.clone());
        let specialized_target = map_lookup_target(&target, |v| v.specialized.clone());
        let unspecialized_args = args
            .iter()
            .map(|arg| arg.unspecialized.clone())
            .collect::<Vec<_>>();
        let specialized_args = args
            .iter()
            .map(|arg| arg.specialized.clone())
            .collect::<Vec<_>>();
        let unspecialized_flag = flag.unspecialized;
        let specialized_flag = flag.specialized;

        self.get_unspecialized().record_lookup(
            &unspecialized_target,
            &unspecialized_args,
            &unspecialized_flag,
        );
        self.get_specialized().record_lookup(
            &specialized_target,
            &specialized_args,
            &specialized_flag,
        );
    }

    fn todo(&mut self, payload: &str, _result_types: &[Type]) -> Vec<SpecSplitValue> {
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
                let mut new_values = values.borrow().values.clone();
                new_values.extend(pushed_values.iter().map(|v| v.unspecialized.clone()));
                Value::array(new_values)
            }
            Value::UnknownSlice => Value::UnknownSlice,
            _ => panic!("Cannot push to {:?}", slice.unspecialized),
        };
        let new_spec = match &slice.specialized {
            Value::Array(values) => {
                let mut new_values = values.borrow().values.clone();
                new_values.extend(pushed_values.iter().map(|v| v.specialized.clone()));
                Value::array(new_values)
            }
            Value::UnknownSlice => Value::UnknownSlice,
            _ => panic!("Cannot push to {:?}", slice.specialized),
        };
        SpecSplitValue {
            unspecialized: new_unspec,
            specialized: new_spec,
        }
    }

    fn slice_len(&mut self, slice: &SpecSplitValue) -> SpecSplitValue {
        let unspec = match &slice.unspecialized {
            Value::Array(values) => Value::U(32, values.borrow().values.len() as u128),
            Value::UnknownSlice => Value::Unknown(ScalarKind::U(32)),
            _ => panic!("Cannot get length of {:?}", slice.unspecialized),
        };
        let spec = match &slice.specialized {
            Value::Array(values) => Value::U(32, values.borrow().values.len() as u128),
            Value::UnknownSlice => Value::Unknown(ScalarKind::U(32)),
            _ => panic!("Cannot get length of {:?}", slice.specialized),
        };
        SpecSplitValue {
            unspecialized: unspec,
            specialized: spec,
        }
    }

    fn on_guard(
        &mut self,
        inner: &crate::compiler::ssa::hlssa::OpCode,
        _condition: &SpecSplitValue,
        inputs: Vec<&SpecSplitValue>,
        result_types: Vec<&Type>,
    ) -> Vec<SpecSplitValue> {
        fn unknown_value(ty: &Type) -> Value {
            match &ty.expr {
                TypeExpr::Field => Value::Unknown(ScalarKind::Field),
                TypeExpr::U(s) | TypeExpr::I(s) => Value::Unknown(ScalarKind::U(*s)),
                TypeExpr::Array(elem, size) => {
                    Value::array((0..*size).map(|_| unknown_value(elem)).collect())
                }
                TypeExpr::Tuple(elems) => Value::Tuple(elems.iter().map(unknown_value).collect()),
                TypeExpr::WitnessOf(inner) => Value::WitnessOf(Box::new(unknown_value(inner))),
                TypeExpr::Ref(inner) => Value::Pointer(Rc::new(RefCell::new(unknown_value(inner)))),
                _ => panic!("Unsupported type for unknown value: {:?}", ty),
            }
        }

        // Nuke ptr contents for effectful ptr ops
        if let crate::compiler::ssa::hlssa::OpCode::Store { .. } = inner {
            // First input is the ptr
            if let Some(ptr_val) = inputs.first() {
                if let Value::Pointer(p) = &ptr_val.unspecialized {
                    *p.borrow_mut() = Value::Unknown(ScalarKind::Field);
                }
                if let Value::Pointer(p) = &ptr_val.specialized {
                    *p.borrow_mut() = Value::Unknown(ScalarKind::Field);
                }
            }
        }

        // Create unknown values for all results
        result_types
            .iter()
            .map(|ty| {
                let v = unknown_value(ty);
                SpecSplitValue {
                    unspecialized: v.clone(),
                    specialized: v,
                }
            })
            .collect()
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

#[derive(Default)]
struct AggregatedConstraintCost {
    recurring_constraints: usize,
    array_table_constraints: usize,
    rangecheck_lookups: HashMap<u8, usize>,
    final_spread_lookups: HashMap<u8, usize>,
}

impl AggregatedConstraintCost {
    fn add(&mut self, cost: &Instrumenter, calls: usize) {
        if calls == 0 {
            return;
        }
        self.recurring_constraints += cost.recurring_constraints() * calls;
        self.array_table_constraints += cost.array_table_allocation_constraints() * calls;
        for (bits, count) in cost.rangecheck_lookups.iter() {
            *self.rangecheck_lookups.entry(*bits).or_insert(0) += count * calls;
        }
        for (bits, count) in cost.final_spread_lookups.iter() {
            *self.final_spread_lookups.entry(*bits).or_insert(0) += count * calls;
        }
    }

    fn shared_table_constraints(&self) -> usize {
        let range_constraints = if self.rangecheck_lookups.keys().any(|bits| *bits >= 2) {
            (1usize << 8) + 1
        } else {
            0
        };
        let spread_constraints = self
            .final_spread_lookups
            .keys()
            .filter(|bits| **bits >= 2)
            .map(|bits| 2 * (1usize << *bits as usize) + 1)
            .sum::<usize>();
        range_constraints + spread_constraints
    }

    fn total_constraints(&self) -> usize {
        self.recurring_constraints + self.array_table_constraints + self.shared_table_constraints()
    }
}

impl Summary {
    pub fn pretty_print(&self, ssa: &HLSSA) -> String {
        let mut r = String::new();
        r += &format!("Total constraints: {}\n", self.total_constraints);
        let savings_pct = if self.total_constraints == 0 {
            0.0
        } else {
            self.total_savings_to_make as f64 / self.total_constraints as f64 * 100.0
        };
        r += &format!(
            "Total savings to make: {} ({:.1}%)\n",
            self.total_savings_to_make, savings_pct
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

    pub fn pretty_print(&self, ssa: &HLSSA) -> String {
        let mut r = String::new();
        for (sig, cost) in self.functions.iter() {
            r += &format!("Function {}\n", sig.pretty_print(ssa, false));
            r += &format!("  Calls:\n");
            for (sig, count) in cost.calls.iter() {
                r += &format!("    {}: {} times\n", sig.pretty_print(ssa, false), count);
            }
            r += &format!("  Raw constraints: {}\n", cost.raw.total_constraints());
            r += &format!("  Raw detail: {}\n", cost.raw.detail_line());
            r += &format!(
                "  Specialized constraints: {}\n",
                cost.specialized.total_constraints()
            );
            r += &format!("  Specialized detail: {}\n", cost.specialized.detail_line());
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
                    raw_constraints: cost.raw.specialization_constraints(),
                    specialized_constraints: cost.specialized.specialization_constraints(),
                    specialization_total_savings: 0,
                },
            );
        }
        self.walk_call_tree(&mut r, 1, self.entry_point.as_ref().unwrap());

        let mut aggregate = AggregatedConstraintCost::default();
        for (_, summary) in r.functions.iter_mut() {
            summary.specialization_total_savings = summary
                .raw_constraints
                .saturating_sub(summary.specialized_constraints)
                * summary.calls;
            r.total_savings_to_make += summary.specialization_total_savings;
        }
        for (sig, summary) in r.functions.iter() {
            let cost = self.functions.get(sig).unwrap();
            aggregate.add(&cost.raw, summary.calls);
        }
        r.total_constraints = aggregate.total_constraints();
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
    pub fn run(&self, ssa: &HLSSA, type_info: &TypeInfo) -> CostAnalysis {
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
        ssa: &HLSSA,
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
            TypeExpr::I(s) => ValueSignature::Unknown(ScalarKind::I(*s)),
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
                ValueSignature::Tuple(elems.iter().map(|e| self.type_to_unknown_sig(e)).collect())
            }
            TypeExpr::Ref(inner) => {
                ValueSignature::PointerTo(Box::new(self.type_to_unknown_sig(inner)))
            }
            _ => ValueSignature::Unknown(ScalarKind::Field),
        }
    }

    fn make_main_sig(&self, ssa: &HLSSA) -> FunctionSignature {
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

use crate::compiler::pass_manager::{Analysis, AnalysisId, AnalysisStore};

impl Analysis for Summary {
    fn dependencies() -> Vec<AnalysisId> {
        vec![TypeInfo::id()]
    }

    fn compute(ssa: &HLSSA, store: &AnalysisStore) -> Self {
        let type_info = store.get::<TypeInfo>();
        let cost_estimator = CostEstimator::new();
        let cost_analysis = cost_estimator.run(ssa, type_info);
        cost_analysis.summarize()
    }
}
