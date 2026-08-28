//! Implements a specialization cost-estimation analysis for the compiler.
//!
//! It functions by performing speculative specialization to estimate how many constraints, lookups,
//! and range-checks could be saved by certain specializations. This is done using symbolic
//! execution combined with an instrumenter for the circuit cost, and gives the compiler an idea of
//! how much a function could be shrunk through specialization on concrete inputs.

use std::{cell::RefCell, rc::Rc};

use ark_ff::{BigInt, BigInteger};
use itertools::Itertools;
use mavros_artifacts::FieldConfig;
use mavros_int_semantics::{self as semantics};
use tracing::{debug, instrument};

use crate::{
    collections::HashMap,
    compiler::{
        Field,
        analysis::{
            symbolic_executor::{self, AssertionFailure, SymbolicExecutor},
            types::TypeInfo,
        },
        pass_manager::{Analysis, AnalysisId, AnalysisStore},
        ssa::{
            FunctionId,
            hlssa::{
                ArithGroup, BinaryArithOpKind, CastTarget, CmpKind, Endianness, HLSSA,
                LookupTarget, MAX_SUPPORTED_UNSIGNED_BITS, Radix, RefCountOp, SequenceTargetType,
                SliceOpDir, Type, TypeExpr,
            },
        },
        util::{
            bit_mask, decode_signed, ice_non_elided_tuple, sign_extend_bits, spread_bits,
            unspread_bits,
        },
    },
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ScalarKind {
    Field,
    /// An `n`-bit integer, with no reading attached — as `TypeExpr::Int`, which it is built from.
    Int(usize),
}

impl ScalarKind {
    pub fn from_type(tp: &Type) -> Self {
        match &tp.strip_witness().expr {
            TypeExpr::Field => ScalarKind::Field,
            TypeExpr::Int(s) => ScalarKind::Int(*s),
            TypeExpr::WitnessOf(_) => panic!("WitnessOf is not a scalar type: {:?}", tp),
            _ => panic!("Not a scalar type: {:?}", tp),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ValueSignature {
    Int { bits_size: usize, value: u128 },
    Field(Field),
    Array(Vec<ValueSignature>),
    Blob(Vec<ValueSignature>),
    PointerTo(Box<ValueSignature>),
    Unknown(ScalarKind),
    UnknownSlice,
    WitnessOf(Box<ValueSignature>),
}

impl ValueSignature {
    pub fn to_value(&self) -> Value {
        match self {
            ValueSignature::Int { bits_size, value } => Value::Int(*bits_size, *value),
            ValueSignature::Field(field) => Value::Field(*field),
            ValueSignature::Array(vals) => {
                Value::array(vals.iter().map(|v| v.to_value()).collect())
            }
            ValueSignature::Blob(vals) => Value::Blob(vals.iter().map(|v| v.to_value()).collect()),
            ValueSignature::PointerTo(val) => Value::Pointer(Rc::new(RefCell::new(val.to_value()))),
            ValueSignature::Unknown(kind) => Value::Unknown(*kind),
            ValueSignature::UnknownSlice => Value::UnknownSlice,
            ValueSignature::WitnessOf(inner) => Value::WitnessOf(Box::new(inner.to_value())),
        }
    }

    pub fn pretty_print(&self, full: bool) -> String {
        match self {
            ValueSignature::Int { value, .. } => format!("{value}"),
            ValueSignature::Field(f) => format!("{}", f),
            ValueSignature::Array(items) => {
                if full {
                    let items = items.iter().map(|v| v.pretty_print(full)).join(", ");
                    format!("[{items}]")
                } else {
                    format!("[...]")
                }
            }
            ValueSignature::Blob(items) => {
                if full {
                    let items = items.iter().map(|v| v.pretty_print(full)).join(", ");
                    format!("blob[{items}]")
                } else {
                    "blob[...]".to_string()
                }
            }
            ValueSignature::PointerTo(p) => format!("&({})", p.as_ref().pretty_print(full)),
            ValueSignature::Unknown(_) => "?".to_string(),
            ValueSignature::UnknownSlice => "?slice".to_string(),
            ValueSignature::WitnessOf(inner) => format!("W({})", inner.pretty_print(full)),
        }
    }
}

#[derive(Debug, Clone)]
// FIELD-ASSUMPTION: L4-eval
pub enum Value {
    /// `bits` raw two's-complement bits. Which reading applies is decided by the opcode the
    /// interpreter is executing, never by the value — see [`Value::binary_arith_op`].
    Int(usize, u128),
    Field(Field),
    Array(Rc<Vec<Value>>),
    Blob(Vec<Value>),
    Pointer(Rc<RefCell<Value>>),
    Unknown(ScalarKind),
    UnknownSlice,
    WitnessOf(Box<Value>),
}

impl Value {
    fn array(values: Vec<Value>) -> Self {
        Value::Array(Rc::new(values))
    }

    fn as_field_const(&self, field: FieldConfig) -> Option<Field> {
        match self {
            Value::Int(_, v) => Some(field.constant(*v)),
            Value::Field(f) => Some(*f),
            Value::WitnessOf(inner) => inner.as_field_const(field),
            _ => None,
        }
    }

    /// Interpret a u128 two's-complement value as signed i128 for `bits` width.
    fn to_signed(val: u128, bits: usize) -> i128 {
        decode_signed(bits, val & bit_mask(bits))
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
            TypeExpr::Int(s) => Value::Unknown(ScalarKind::Int(*s)),
            TypeExpr::WitnessOf(inner) => {
                Value::WitnessOf(Box::new(Value::unknown_from_type(inner)))
            }
            TypeExpr::Array(elem, n) => {
                let elem_unknown = Value::unknown_from_type(elem);
                Value::array(vec![elem_unknown; *n])
            }
            TypeExpr::Slice(_) => Value::UnknownSlice,
            TypeExpr::Tuple(_) => ice_non_elided_tuple(),
            TypeExpr::Ref(inner) => {
                Value::Pointer(Rc::new(RefCell::new(Value::unknown_from_type(inner))))
            }
            TypeExpr::Function => panic!("Cannot create unknown value for Function type"),
            TypeExpr::Blob(elem, n) => {
                let elem_unknown = Value::unknown_from_type(elem);
                Value::Blob(vec![elem_unknown; *n])
            }
        }
    }

    /// Interpret one comparison.
    ///
    /// A `Value::Int` is a raw bit pattern with no reading attached, so the arms below take their
    /// reading from the _opcode_, exactly as [`Value::binary_arith_op`] does. `SLt` decodes both
    /// operands as two's complement at the caller's `bits` while `ULt` compares the patterns as
    /// magnitudes and `Eq` needs no reading at all.
    fn cmp_op(&self, b: &Value, kind: CmpKind, bits: Option<usize>) -> Value {
        match (self, b) {
            (Value::Int(_, a), Value::Int(_, b)) => {
                let result = match kind {
                    CmpKind::Eq => a == b,
                    CmpKind::ULt => a < b,
                    CmpKind::SLt => {
                        let bits = bits.expect("ICE: signed comparison without an operand width");
                        Self::to_signed(*a, bits) < Self::to_signed(*b, bits)
                    }
                };
                Value::Int(1, result as u128)
            }
            // A field element has no two's complement reading, so an `SLt` over one never reaches
            // here as the executor rejects it before the call. `ULt` and `Eq` compare the elements
            // themselves.
            (Value::Field(a), Value::Field(b)) => {
                let result = match kind {
                    CmpKind::Eq => a == b,
                    CmpKind::ULt | CmpKind::SLt => a < b,
                };
                Value::Int(1, result as u128)
            }
            (Value::WitnessOf(_), _) | (_, Value::WitnessOf(_)) => Value::WitnessOf(Box::new(
                self.unwrap_witness().cmp_op(b.unwrap_witness(), kind, bits),
            )),
            (Value::Unknown(_), _) | (_, Value::Unknown(_)) => Value::Unknown(ScalarKind::Int(1)),
            _ => panic!("Cannot compare {:?} and {:?}", self, b),
        }
    }

    /// Interpret one binary arithmetic operation.
    ///
    /// The integer arm is the reference model's [`residue`](semantics::residue): the bit pattern a
    /// _total_ evaluator must produce. It runs over dummy signature values for cost estimation, so
    /// it meets operands a real execution would have been rejected for and must still answer
    /// something. On every input Noir accepts, `residue` _is_ the accepted value, so the estimate
    /// is exact here for all intents and purposes.
    ///
    /// The `unwrap_or(0)` covers the two inputs the model deliberately declines to specify: a zero
    /// divisor and a signed `INT_MIN / -1`. LLVM calls both undefined while the VM answers zero, so
    /// there is no agreed answer to hold anyone to, and zero is this interpreter's own local choice
    /// rather than a disagreement with anything. It needs one because a zero denominator is routine
    /// here as a gadget's real divisor is constrained nonzero only on honest runs, and these are
    /// dummy values.
    ///
    /// The non-integer arms stay per-group because they genuinely differ: a zero operand
    /// short-circuits a `Mul` whatever the other side is, a field `Div` is multiplication by a
    /// modular inverse, and most groups have no field meaning at all.
    fn binary_arith_op(
        &self,
        b: &Value,
        binary_arith_op_kind: &crate::compiler::ssa::hlssa::BinaryArithOpKind,
        instrumenter: &mut dyn OpInstrumenter,
    ) -> Value {
        let group = binary_arith_op_kind.group();

        // A zero operand makes a product zero whatever the other side is, so this runs ahead of the
        // dispatch below rather than inside its integer and field arms.
        if group == ArithGroup::Mul {
            match (self, b) {
                (Value::Int(s, 0), _) | (_, Value::Int(s, 0)) => return Value::Int(*s, 0),
                (Value::Field(f), _) if *f == instrumenter.field().zero() => {
                    return Value::Field(instrumenter.field().zero());
                }
                (_, Value::Field(f)) if *f == instrumenter.field().zero() => {
                    return Value::Field(instrumenter.field().zero());
                }
                _ => {}
            }
        }

        match (self, b) {
            // Each operand is read at _its own_ width. For everything but a shift the two agree, as
            // the type analysis widens a result to the wider operand; a shift is the case where
            // they legitimately differ, and an amount's own width is the only thing that says what
            // its pattern means.
            (Value::Int(bits, lhs), Value::Int(rhs_bits, rhs)) => Value::Int(
                *bits,
                semantics::residue(
                    group.into(),
                    binary_arith_op_kind.sign(),
                    *bits,
                    *lhs,
                    *rhs_bits,
                    *rhs,
                )
                .unwrap_or(0),
            ),

            (Value::Field(x), Value::Field(y)) => match group {
                // FIELD-ASSUMPTION: L4-eval
                ArithGroup::Add => Value::Field(x + y),
                ArithGroup::Sub => Value::Field(x - y),
                ArithGroup::Mul => Value::Field(x * y),
                // FIELD-ASSUMPTION: L4-inverse
                ArithGroup::Div => Value::Field(if *y == instrumenter.field().zero() {
                    instrumenter.field().zero()
                } else {
                    x / y
                }),
                ArithGroup::And => todo!(),
                ArithGroup::Rem
                | ArithGroup::Or
                | ArithGroup::Xor
                | ArithGroup::Shl
                | ArithGroup::Shr => {
                    panic!("Cannot perform binary arithmetic on {:?} and {:?}", self, b)
                }
            },

            // The one witness pair that is not purely structural: two _unknown_ witnesses
            // multiplied is the shape that costs a degree, and counting those is core to this
            // analysis.
            (Value::WitnessOf(x), Value::WitnessOf(y)) if group == ArithGroup::Mul => {
                if matches!(
                    (x.as_ref(), y.as_ref()),
                    (Value::Unknown(_), Value::Unknown(_))
                ) {
                    instrumenter.record_high_degree_mul();
                }
                Value::WitnessOf(Box::new(x.binary_arith_op(
                    y,
                    binary_arith_op_kind,
                    instrumenter,
                )))
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
        }
    }

    fn blind(&mut self) {
        match self {
            Value::WitnessOf(inner) => {
                inner.forget_concrete();
            }
            Value::Unknown(_) | Value::UnknownSlice => {}
            Value::Int(_, _) | Value::Field(_) => {}
            Value::Array(vals) => {
                for val in Rc::make_mut(vals).iter_mut() {
                    val.blind();
                }
            }
            Value::Blob(vals) => {
                for val in vals {
                    val.blind();
                }
            }
            Value::Pointer(val) => {
                val.borrow_mut().blind();
            }
        }
    }

    fn forget_concrete(&mut self) {
        match self {
            Value::Int(s, _) => *self = Value::Unknown(ScalarKind::Int(*s)),
            Value::Field(_) => *self = Value::Unknown(ScalarKind::Field),
            Value::Unknown(_) | Value::UnknownSlice => {}
            Value::WitnessOf(inner) => {
                inner.forget_concrete();
            }
            Value::Array(vals) => {
                for val in Rc::make_mut(vals).iter_mut() {
                    val.forget_concrete();
                }
            }
            Value::Blob(vals) => {
                for val in vals {
                    val.forget_concrete();
                }
            }
            Value::Pointer(val) => {
                val.borrow_mut().forget_concrete();
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
            Value::Int(s, v) => ValueSignature::Int {
                bits_size: *s,
                value: *v,
            },
            Value::Field(f) => ValueSignature::Field(*f),
            Value::Array(vals) => {
                ValueSignature::Array(vals.iter().map(|v| v.make_unspecialized_sig()).collect())
            }
            Value::Blob(vals) => {
                ValueSignature::Blob(vals.iter().map(|v| v.make_unspecialized_sig()).collect())
            }
            Value::Pointer(val) => {
                ValueSignature::PointerTo(Box::new(val.borrow().make_unspecialized_sig()))
            }
        }
    }

    fn array_get(&self, index: &Value, tp: &Type, _instrumenter: &mut dyn OpInstrumenter) -> Value {
        if matches!(self, Value::Unknown(_) | Value::UnknownSlice) {
            return Value::unknown_from_type(tp);
        }

        let values: &[Value] = match self.unwrap_witness() {
            Value::Array(values) => values,
            Value::Blob(values) => values,
            Value::Unknown(_) | Value::UnknownSlice => return Value::unknown_from_type(tp),
            _ => panic!(
                "Cannot get array element from {:?} with index {:?}",
                self, index
            ),
        };

        // An out-of-range constant index is not an ICE here. The IR legitimately contains reads
        // that can never execute successfully — a `SlicePop` of a statically empty slice lowers
        // to its (constant-false) bounds assert plus an `ArrayGet` at index 0 — and rejecting
        // such a program is witgen's job, not the cost estimator's. Answering `Unknown` just
        // declines to specialize through the read.
        let at = |i: &u128| match values.get(*i as usize) {
            Some(value) => value.clone(),
            None => Value::unknown_from_type(tp),
        };
        match index {
            Value::Int(_, i) => at(i),
            Value::WitnessOf(inner) => match inner.as_ref() {
                Value::Int(_, i) => at(i),
                _ => Value::unknown_from_type(tp),
            },
            Value::Unknown(_) => Value::unknown_from_type(tp),
            _ => panic!(
                "Cannot get array element from {:?} with index {:?}",
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
            (Value::Array(vals), Value::Int(_, index), value) => {
                let mut new_vals = vals.as_ref().clone();
                new_vals[*index as usize] = value.clone();
                Value::array(new_vals)
            }
            (Value::Array(vals), Value::WitnessOf(inner), value) => match inner.as_ref() {
                Value::Int(_, index) => {
                    let mut new_vals = vals.as_ref().clone();
                    new_vals[*index as usize] = value.clone();
                    Value::array(new_vals)
                }
                _ => {
                    let new_vals = vals.iter().map(|_| value.clone()).collect();
                    Value::array(new_vals)
                }
            },
            (Value::Array(vals), _, value) => {
                let new_vals = vals.iter().map(|_| value.clone()).collect();
                Value::array(new_vals)
            }
            (Value::UnknownSlice, _, _) => Value::UnknownSlice,
            _ => panic!(
                "Cannot set array element of {:?} with index {:?} to {:?}",
                self, index, value
            ),
        }
    }

    // FIELD-ASSUMPTION: L4-decompose
    fn bit_range_op(
        &self,
        offset: usize,
        width: usize,
        instrumenter: &mut dyn OpInstrumenter,
    ) -> Value {
        match self {
            Value::Unknown(kind) => Value::Unknown(*kind),
            Value::WitnessOf(inner) => {
                Value::WitnessOf(Box::new(inner.bit_range_op(offset, width, instrumenter)))
            }
            Value::Int(bits, v) => Value::Int(*bits, (v >> offset) & bit_mask(width)),
            Value::Field(f) => {
                let bits = f
                    .into_bigint()
                    .to_bits_le()
                    .into_iter()
                    .skip(offset)
                    .take(width)
                    .collect::<Vec<_>>();
                let r = instrumenter
                    .field()
                    .from_bigint(BigInt::from_bits_le(&bits));
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
            Value::Int(_, v) => Value::Int(to, sign_extend_bits(*v, from, to)),
            _ => panic!("Cannot sext {:?}", self),
        }
    }

    fn spread_op(&self) -> Value {
        match self {
            Value::Int(bits, v) => {
                assert!(
                    *bits <= 64,
                    "Spread only supports integer widths up to 64 bits, got int{}",
                    bits
                );
                Value::Int(bits * 2, spread_bits(*v, *bits))
            }
            Value::Field(_) => panic!("Spread of field values is unsupported"),
            Value::WitnessOf(inner) => Value::WitnessOf(Box::new(inner.spread_op())),
            Value::Unknown(ScalarKind::Int(bits)) => {
                assert!(
                    *bits <= 64,
                    "Spread only supports integer widths up to 64 bits, got int{}",
                    bits
                );
                Value::Unknown(ScalarKind::Int(bits * 2))
            }
            Value::Unknown(ScalarKind::Field) => panic!("Spread of field values is unsupported"),
            _ => panic!("Cannot spread {:?}", self),
        }
    }

    fn unspread_op(&self) -> (Value, Value) {
        match self {
            Value::Int(bits, v) => {
                assert!(
                    *bits <= MAX_SUPPORTED_UNSIGNED_BITS && bits % 2 == 0,
                    "Unspread expects an even integer width up to {MAX_SUPPORTED_UNSIGNED_BITS} bits, got int{}",
                    bits
                );
                let (odd_val, even_val) = unspread_bits(*v, *bits);
                let half_bits = bits / 2;
                (
                    Value::Int(half_bits, odd_val),
                    Value::Int(half_bits, even_val),
                )
            }
            Value::Field(_) => panic!("Unspread of field values is unsupported"),
            Value::WitnessOf(inner) => {
                let (odd, even) = inner.unspread_op();
                (
                    Value::WitnessOf(Box::new(odd)),
                    Value::WitnessOf(Box::new(even)),
                )
            }
            Value::Unknown(ScalarKind::Int(bits)) => {
                assert!(
                    *bits <= MAX_SUPPORTED_UNSIGNED_BITS && bits % 2 == 0,
                    "Unspread expects an even integer width up to {MAX_SUPPORTED_UNSIGNED_BITS} bits, got int{}",
                    bits
                );
                let half_bits = bits / 2;
                (
                    Value::Unknown(ScalarKind::Int(half_bits)),
                    Value::Unknown(ScalarKind::Int(half_bits)),
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
        instrumenter: &mut dyn OpInstrumenter,
    ) -> Value {
        match (self, cast_target) {
            (_, CastTarget::WitnessOf) => Value::WitnessOf(Box::new(self.clone())),
            (_, CastTarget::ValueOf) => self.unwrap_witness().clone(),
            (Value::Array(values), CastTarget::Map(inner)) => Value::array(
                values
                    .iter()
                    .map(|v| v.cast_op(inner, instrumenter))
                    .collect(),
            ),
            (Value::UnknownSlice, CastTarget::Map(_)) => Value::UnknownSlice,
            (Value::Unknown(_), CastTarget::Int(s)) => Value::Unknown(ScalarKind::Int(*s)),
            (Value::Unknown(_), CastTarget::Field) => Value::Unknown(ScalarKind::Field),
            (Value::Unknown(kind), CastTarget::Nop | CastTarget::ArrayToSlice) => {
                Value::Unknown(*kind)
            }
            (Value::WitnessOf(inner), target) => {
                Value::WitnessOf(Box::new(inner.cast_op(target, instrumenter)))
            }
            (Value::Int(_, v), CastTarget::Int(s2)) => Value::Int(*s2, *v & bit_mask(*s2)),
            // Raw bits, and _only_ raw bits. There used to be a second arm here, reached whenever
            // the operand happened to be tagged `I`, that decoded to two's complement and then
            // `as u64` — which disagreed with every other implementation of this cast
            // (`lattice::eval_cast` takes the raw payload, `hlssa_to_r1cs::of_int` builds
            // `Fr::from(raw)`, and the LLVM path zero-extends) and mangled the value on top of it,
            // since `as u64` on a negative `i128` is not a field element's worth of anything. The
            // cost interpreter has to agree with the circuit it is estimating, so the surviving
            // arm is the one the backends implement.
            (Value::Int(_, v), CastTarget::Field) => {
                Value::Field(instrumenter.field().constant(*v))
            }
            (Value::Field(f), CastTarget::Field) => Value::Field(*f),
            (Value::Field(f), CastTarget::Int(s)) => {
                let bigint = f.into_bigint();
                Value::Int(
                    *s,
                    (bigint.0[0] as u128 | ((bigint.0[1] as u128) << 64)) & bit_mask(*s),
                )
            }
            (_, CastTarget::Nop | CastTarget::ArrayToSlice) => self.clone(),
            _ => panic!("Cannot cast {:?} to {:?}", self, cast_target),
        }
    }

    fn constrain(
        a: &Value,
        b: &Value,
        c: &Value,
        instrumenter: &mut dyn OpInstrumenter,
    ) -> Result<(), AssertionFailure> {
        match (
            a.as_field_const(instrumenter.field()),
            b.as_field_const(instrumenter.field()),
            c.as_field_const(instrumenter.field()),
        ) {
            (Some(a), Some(b), Some(c)) => {
                if a * b != c {
                    // A constraint over compile-time constants that does not hold: the program is
                    // unsatisfiable on every input (e.g. an `execution_failure` test). Surface it
                    // via the assertion-failure channel instead of panicking as R1CS generation is
                    // the canonical reporter and rejects the program.
                    return Err(AssertionFailure::new(format!(
                        "constraint {a:?} * {b:?} = {c:?} is statically false"
                    )));
                }
                // A trivially-true constant constraint is elided by codegen, so it costs nothing.
            }
            _ => instrumenter.record_constrain(),
        }
        Ok(())
    }

    // FIELD-ASSUMPTION: L4-decompose
    fn to_bits(&self, endianness: &Endianness, size: usize) -> Value {
        match self {
            Value::Unknown(_) => Value::array(vec![Value::Unknown(ScalarKind::Int(1)); size]),
            Value::WitnessOf(inner) => {
                let result = inner.to_bits(endianness, size);
                match result {
                    Value::Array(bits) => Value::array(
                        bits.iter()
                            .cloned()
                            .map(|b| Value::WitnessOf(Box::new(b)))
                            .collect(),
                    ),
                    _ => unreachable!("to_bits of a WitnessOf expected an Array result"),
                }
            }
            // Decomposition is a property of the bit pattern, so a signed value decomposes
            // exactly as its unsigned twin does. The bits themselves are always `u1`.
            Value::Int(_, v) => {
                let mut r = (0..size)
                    .map(|i| {
                        Value::Int(
                            1,
                            if i < u128::BITS as usize {
                                (v >> i) & 1
                            } else {
                                0
                            },
                        )
                    })
                    .collect::<Vec<_>>();
                if *endianness == Endianness::Big {
                    r.reverse();
                }
                Value::array(r)
            }
            Value::Field(f) => {
                let bigint = f.into_bigint();
                let raw_bits = bigint.to_bits_le();
                let mut bits = (0..size)
                    .map(|i| Value::Int(1, raw_bits.get(i).copied().unwrap_or(false) as u128))
                    .collect::<Vec<_>>();
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
    ) -> Value {
        match self {
            Value::WitnessOf(inner) => {
                let result = inner.to_radix(radix, _endianness, size);
                match result {
                    Value::Array(digits) => Value::array(
                        digits
                            .iter()
                            .cloned()
                            .map(|d| Value::WitnessOf(Box::new(d)))
                            .collect(),
                    ),
                    _ => unreachable!("to_radix of a WitnessOf expected an Array result"),
                }
            }
            Value::Unknown(_) => Value::array(vec![Value::Unknown(ScalarKind::Int(8)); size]),
            Value::Field(f) => {
                let radix_val = match radix {
                    Radix::Dyn(Value::Int(_, r)) => *r,
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
                    digits.push(Value::Int(8, digit));
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
            Value::Int(s, v) => Value::Int(*s, !v & bit_mask(*s)),
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

    fn assert_r1c(_a: &Value, _b: &Value, _c: &Value, _instrumenter: &mut dyn OpInstrumenter) {}

    fn select(
        &self,
        if_true: &Value,
        if_false: &Value,
        _tp: &Type,
        _instrumenter: &mut dyn OpInstrumenter,
    ) -> Value {
        match self {
            Value::Int(_, 0) => if_false.clone(),
            Value::Int(_, _) => if_true.clone(),
            Value::WitnessOf(inner) => match inner.as_ref() {
                Value::Int(_, 0) => if_false.clone(),
                Value::Int(_, _) => if_true.clone(),
                _ => {
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
    fn cmp(
        &self,
        b: &SpecSplitValue,
        kind: CmpKind,
        bits: Option<usize>,
        _instrumenter: &mut CostAnalysis,
    ) -> SpecSplitValue {
        SpecSplitValue {
            unspecialized: self.unspecialized.cmp_op(&b.unspecialized, kind, bits),
            specialized: self.specialized.cmp_op(&b.specialized, kind, bits),
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

    fn assert_r1c(
        a: &SpecSplitValue,
        b: &SpecSplitValue,
        c: &SpecSplitValue,
        ctx: &mut CostAnalysis,
    ) -> Result<(), AssertionFailure> {
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
        Ok(())
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
    ) -> Result<(), AssertionFailure> {
        Value::constrain(
            &a.unspecialized,
            &b.unspecialized,
            &c.unspecialized,
            instrumenter.get_unspecialized(),
        )?;
        Value::constrain(
            &a.specialized,
            &b.specialized,
            &c.specialized,
            instrumenter.get_specialized(),
        )?;
        Ok(())
    }

    // Witness asserts and rangechecks are lowered to explicit Constrain/Lookup ops before the
    // cost analysis runs (and those are costed via `record_constrain`/`record_lookup`), so any
    // op still in these high-level forms carries no cost of its own. The cost estimator also
    // runs over fully-symbolic inputs, so these never fold to a constant that could fail.
    fn assert_bool(&self, _instrumenter: &mut CostAnalysis) -> Result<(), AssertionFailure> {
        Ok(())
    }

    fn assert_cmp(
        _kind: CmpKind,
        _a: &Self,
        _b: &Self,
        _bits: Option<usize>,
        _instrumenter: &mut CostAnalysis,
    ) -> Result<(), AssertionFailure> {
        Ok(())
    }

    fn rangecheck(
        &self,
        _max_bits: usize,
        _instrumenter: &mut CostAnalysis,
    ) -> Result<(), AssertionFailure> {
        Ok(())
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
        _instrumenter: &mut CostAnalysis,
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
            unspecialized: self
                .unspecialized
                .to_radix(&unspec_radix, &endianness, size),
            specialized: self.specialized.to_radix(&spec_radix, &endianness, size),
        }
    }

    fn expect_constant_bool(&self, _ctx: &mut CostAnalysis) -> bool {
        let specialized = match &self.specialized {
            Value::Int(1, v) => *v != 0,
            _ => panic!(
                "Expected constant bool, got specialized={:?}",
                self.specialized
            ),
        };
        let unspecialized = match &self.unspecialized {
            Value::Int(1, v) => *v != 0,
            _ => panic!(
                "Expected constant bool, got unspecialized={:?}",
                self.unspecialized
            ),
        };
        // The unspecialized world is a blinding-refinement of the specialized one: both are seeded
        // identically and every value op is symmetric, with the only asymmetry being
        // `blind_unspecialized`, which coarsens a concrete leaf to `Unknown` — never to a _different_
        // concrete. So a branch condition that is a concrete bool in both worlds must agree; a
        // divergence here is a specializer/writeback miscompilation, not a benign dead path.
        assert_eq!(
            specialized, unspecialized,
            "ICE: branch condition diverged between the specialized and unspecialized cost-analysis worlds"
        );
        specialized
    }

    fn of_int(s: usize, v: u128, _ctx: &mut CostAnalysis) -> Self {
        Self {
            unspecialized: Value::Int(s, v),
            specialized: Value::Int(s, v),
        }
    }

    fn of_field(f: Field, _ctx: &mut CostAnalysis) -> Self {
        Self {
            unspecialized: Value::Field(f),
            specialized: Value::Field(f),
        }
    }

    fn of_blob(_elem_type: Type, values: Vec<Self>, _ctx: &mut CostAnalysis) -> Self {
        let (unspecialized, specialized) = values
            .into_iter()
            .map(|v| (v.unspecialized, v.specialized))
            .unzip();
        Self {
            unspecialized: Value::Blob(unspecialized),
            specialized: Value::Blob(specialized),
        }
    }

    fn expect_blob(&self, _ctx: &mut CostAnalysis) -> Vec<Self> {
        match (&self.unspecialized, &self.specialized) {
            (Value::Blob(unspecialized), Value::Blob(specialized)) => {
                // The two worlds are structurally identical up to blinding (which only coarsens
                // leaves to `Unknown`, never changes a blob's shape), so their lengths must match;
                // a mismatch is a miscompilation, not a recoverable condition.
                assert_eq!(
                    unspecialized.len(),
                    specialized.len(),
                    "ICE: blob length diverged between the specialized and unspecialized cost-analysis worlds"
                );
                unspecialized
                    .iter()
                    .cloned()
                    .zip(specialized.iter().cloned())
                    .map(|(unspecialized, specialized)| Self {
                        unspecialized,
                        specialized,
                    })
                    .collect()
            }
            _ => panic!(
                "Expected blob, got unspecialized={:?}, specialized={:?}",
                self.unspecialized, self.specialized
            ),
        }
    }

    fn alloc(value: &Self, _ctx: &mut CostAnalysis) -> Self {
        Self {
            unspecialized: Value::Pointer(Rc::new(RefCell::new(value.unspecialized.clone()))),
            specialized: Value::Pointer(Rc::new(RefCell::new(value.specialized.clone()))),
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

    fn mem_op(&self, _kind: RefCountOp, _ctx: &mut CostAnalysis) {}

    fn spread(&self, _bits: u8, _instrumenter: &mut CostAnalysis) -> Self {
        Self {
            unspecialized: self.unspecialized.spread_op(),
            specialized: self.specialized.spread_op(),
        }
    }

    fn unspread(&self, _bits: u8, _instrumenter: &mut CostAnalysis) -> (Self, Self) {
        let (unspec_odd, unspec_even) = self.unspecialized.unspread_op();
        let (spec_odd, spec_even) = self.specialized.unspread_op();
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
    fn field(&self) -> FieldConfig;
    fn record_constrain(&mut self);
    fn record_high_degree_mul(&mut self);
    /// `unconditional` is true when the lookup's flag is the literal constant `1`; such lookups
    /// spill bit chunks to the free `b·(b−1) = 0` form (no gating witness).
    fn record_lookup(&mut self, target: &LookupTarget<Value>, unconditional: bool);
}

trait FunctionInstrumenter {
    fn get_specialized(&mut self) -> &mut dyn OpInstrumenter;
    fn get_unspecialized(&mut self) -> &mut dyn OpInstrumenter;
    fn record_call(&mut self, sig: FunctionSignature);
    fn seal(self: Box<Self>) -> FunctionCost;
}

/// Raw circuit-cost events for one symbolic execution of a function, exactly as the executor
/// saw them. Recording only counts what happened; turning the events into constraint and table
/// costs is interpretation, done on read by the derivation methods below.
#[derive(Debug, Clone)]
struct Instrumenter {
    field: FieldConfig,
    constrains: usize,
    high_degree_muls: usize,

    /// Rangecheck lookup requests by `(width, is_unconditional)` — `true` when the lookup's flag is
    /// the literal constant `1`, which lets its bit chunks spill to the free `b·(b−1)=0` form.
    rangecheck_lookups: HashMap<(u8, bool), usize>,
    /// Spread lookup requests by `(width, is_unconditional)`; see `rangecheck_lookups`.
    spread_lookups: HashMap<(u8, bool), usize>,
    array_lookups: usize,
}

impl OpInstrumenter for Instrumenter {
    fn field(&self) -> FieldConfig {
        self.field
    }

    fn record_constrain(&mut self) {
        self.constrains += 1;
    }

    fn record_high_degree_mul(&mut self) {
        self.high_degree_muls += 1;
    }

    fn record_lookup(&mut self, target: &LookupTarget<Value>, unconditional: bool) {
        match target {
            LookupTarget::Rangecheck(bits) => self.record_rangecheck_lookup(*bits, unconditional),
            LookupTarget::DynRangecheck(_) => {
                // `to_radix` lowers its (asserted radix-256) digit checks to static 8-bit
                // rangechecks, so none survive to cost analysis.
                unreachable!(
                    "DynRangecheck is lowered to a static 8-bit rangecheck before spilling"
                )
            }
            LookupTarget::Spread(bits) => {
                assert!(*bits >= 1, "spread width must be at least 1 bit");
                *self
                    .spread_lookups
                    .entry((*bits, unconditional))
                    .or_insert(0) += 1;
            }
            LookupTarget::Array(_) => self.array_lookups += 1,
        }
    }
}

/// Interpretation of the raw events, mirroring the lookup-spilling expansion that runs before
/// R1CS generation. Everything here is derived; the struct stores no interpreted state.
impl Instrumenter {
    /// A fresh instrumenter for `field`, with every event count zeroed.
    ///
    /// Written out rather than derived from `Default`: [`FieldConfig`] has no `Default`, so that a
    /// carrier cannot acquire a field without being handed one.
    fn new(field: FieldConfig) -> Instrumenter {
        Instrumenter {
            field,
            constrains: 0,
            high_degree_muls: 0,
            rangecheck_lookups: HashMap::default(),
            spread_lookups: HashMap::default(),
            array_lookups: 0,
        }
    }

    fn record_rangecheck_lookup(&mut self, bits: u8, unconditional: bool) {
        assert!(bits >= 1, "rangecheck width must be at least 1 bit");
        *self
            .rangecheck_lookups
            .entry((bits, unconditional))
            .or_insert(0) += 1;
    }

    /// 8-bit rangecheck table lookups after spilling: a w-bit rangecheck (w >= 2) becomes one
    /// lookup per full byte plus two for a leftover partial byte. 1-bit rangechecks never reach
    /// the table (see [`Self::rangecheck_one_constraints`]).
    fn final_rangecheck8_lookups(&self) -> usize {
        self.rangecheck_lookups
            .iter()
            .map(|(&(bits, _), &count)| {
                let bits = bits as usize;
                let per_lookup = match bits {
                    1 => 0,
                    8 => 1,
                    _ => bits / 8 + if bits % 8 > 0 { 2 } else { 0 },
                };
                per_lookup * count
            })
            .sum()
    }

    /// 1-bit rangechecks spill to the algebraic b*(b-1) = 0. (Guarded checks spend one more
    /// constraint to gate that, but the estimate doesn't track lookup flags.)
    fn rangecheck_one_constraints(&self) -> usize {
        self.rangecheck_lookups
            .iter()
            .filter(|((bits, _), _)| *bits == 1)
            .map(|(_, &count)| count)
            .sum()
    }

    /// Spread table lookups after spilling: widths >= 16 are split into chunks of at most
    /// 8 bits each.
    fn final_spread_lookups(&self) -> HashMap<u8, usize> {
        let mut r: HashMap<u8, usize> = HashMap::default();
        for (&(bits, _), &count) in self.spread_lookups.iter() {
            if bits >= 16 {
                assert!(
                    bits <= 128,
                    "wide Spread lookup spilling currently supports widths up to 128 bits, got {bits}"
                );
                let bits = bits as usize;
                let mut offset = 0usize;
                while offset < bits {
                    let chunk_bits = (bits - offset).min(8) as u8;
                    *r.entry(chunk_bits).or_insert(0) += count;
                    offset += chunk_bits as usize;
                }
            } else {
                *r.entry(bits).or_insert(0) += count;
            }
        }
        r
    }

    /// One recombination constraint per wide (>= 16 bit) spread that spilling splits up.
    fn spilled_wide_spread_constraints(&self) -> usize {
        self.spread_lookups
            .iter()
            .filter(|((bits, _), _)| *bits >= 16)
            .map(|(_, count)| *count)
            .sum()
    }

    fn total_table_lookups(&self) -> usize {
        self.final_rangecheck8_lookups()
            + self.final_spread_lookups().values().sum::<usize>()
            + self.array_lookups
    }

    fn uses_rangecheck_table(&self) -> bool {
        self.rangecheck_lookups.keys().any(|(bits, _)| *bits >= 2)
    }

    // Array table _allocation_ is deliberately not costed anywhere below: tables are
    // circuit-global (allocated once per distinct array, whichever function touches it first),
    // so a per-function instrumenter has no sound way to attribute that cost. Only the
    // recurring per-lookup cost is counted.

    fn table_allocation_constraints(&self) -> usize {
        let range_constraints = if self.uses_rangecheck_table() {
            (1usize << 8) + 1
        } else {
            0
        };
        // Both operands of a spread entry are compile-time constants, so each
        // entry is a single folded constraint: 2^bits per-entry constraints
        // plus one sum constraint.
        let spread_constraints = self
            .final_spread_lookups()
            .keys()
            .filter(|bits| **bits >= 2)
            .map(|bits| (1usize << *bits as usize) + 1)
            .sum::<usize>();
        range_constraints + spread_constraints
    }

    fn lookup_data_constraints(&self) -> usize {
        self.final_rangecheck8_lookups()
            + self
                .final_spread_lookups()
                .values()
                .map(|count| count * 2)
                .sum::<usize>()
            + self.array_lookups * 2
    }

    fn recurring_constraints(&self) -> usize {
        self.constrains
            + self.high_degree_muls
            + self.rangecheck_one_constraints()
            + self.spilled_wide_spread_constraints()
            + self.lookup_data_constraints()
    }

    fn total_constraints(&self) -> usize {
        self.recurring_constraints() + self.table_allocation_constraints()
    }

    fn allocated_lookup_table_rows(&self) -> usize {
        let range_rows = if self.uses_rangecheck_table() {
            1usize << 8
        } else {
            0
        };
        let spread_rows = self
            .final_spread_lookups()
            .keys()
            .filter(|bits| **bits >= 2)
            .map(|bits| 1usize << *bits as usize)
            .sum::<usize>();
        range_rows + spread_rows
    }

    fn detail_line(&self) -> String {
        format!(
            "constrain={}, high_deg_mul={}, lookups={}, table_rows={}, table_constraints={}",
            self.constrains,
            self.high_degree_muls,
            self.total_table_lookups(),
            self.allocated_lookup_table_rows(),
            self.table_allocation_constraints()
        )
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

pub struct DummyInstrumenter {
    field: FieldConfig,
}

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
    fn field(&self) -> FieldConfig {
        self.field
    }

    fn record_constrain(&mut self) {}
    fn record_high_degree_mul(&mut self) {}
    fn record_lookup(&mut self, _: &LookupTarget<Value>, _: bool) {}
}

pub struct CostAnalysis {
    field: FieldConfig,
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
                    TypeExpr::Int(s) => Value::Unknown(ScalarKind::Int(*s)),
                    TypeExpr::Array(elem, size) => {
                        Value::array((0..*size).map(|_| unknown_value(elem)).collect())
                    }
                    TypeExpr::Tuple(_) => ice_non_elided_tuple(),
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
        _args: Vec<SpecSplitValue>,
        flag: SpecSplitValue,
    ) {
        // A flag that folds to the constant 1 means the lookup is unconditional, so its spilled bit
        // chunks take the free algebraic form. `as_field_const` treats `U(_,1)`/`Field(1)` alike.
        let field = self.field;
        let one = Some(field.constant(1u64));
        let unspecialized_target = target.map(|v| v.unspecialized.clone());
        self.get_unspecialized().record_lookup(
            &unspecialized_target,
            flag.unspecialized.as_field_const(field) == one,
        );

        let specialized_target = target.map(|v| v.specialized.clone());
        self.get_specialized().record_lookup(
            &specialized_target,
            flag.specialized.as_field_const(field) == one,
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
        let new_unspec = match &slice.unspecialized {
            Value::Array(values) => {
                let pushed = pushed_values.iter().map(|v| v.unspecialized.clone());
                let new_values = match dir {
                    SliceOpDir::Front => pushed.chain(values.as_ref().iter().cloned()).collect(),
                    SliceOpDir::Back => values.as_ref().iter().cloned().chain(pushed).collect(),
                };
                Value::array(new_values)
            }
            Value::UnknownSlice => Value::UnknownSlice,
            _ => panic!("Cannot push to {:?}", slice.unspecialized),
        };
        let new_spec = match &slice.specialized {
            Value::Array(values) => {
                let pushed = pushed_values.iter().map(|v| v.specialized.clone());
                let new_values = match dir {
                    SliceOpDir::Front => pushed.chain(values.as_ref().iter().cloned()).collect(),
                    SliceOpDir::Back => values.as_ref().iter().cloned().chain(pushed).collect(),
                };
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
            Value::Array(values) => Value::Int(32, values.len() as u128),
            Value::UnknownSlice => Value::Unknown(ScalarKind::Int(32)),
            _ => panic!("Cannot get length of {:?}", slice.unspecialized),
        };
        let spec = match &slice.specialized {
            Value::Array(values) => Value::Int(32, values.len() as u128),
            Value::UnknownSlice => Value::Unknown(ScalarKind::Int(32)),
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
                TypeExpr::Field | TypeExpr::Int(_) => Value::Unknown(ScalarKind::from_type(ty)),
                TypeExpr::Array(elem, size) => {
                    Value::array((0..*size).map(|_| unknown_value(elem)).collect())
                }
                TypeExpr::Tuple(_) => ice_non_elided_tuple(),
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
    /// Whole-program rangecheck lookup requests by `(original width, is_unconditional)`
    /// (call-multiplicity weighted, before any spilling into chunks). Consumed by `LookupSizing`,
    /// which prices unconditional lookups' width-1 chunks at zero (their bit-bounds are free).
    pub global_rangecheck_lookups: HashMap<(u8, bool), usize>,
    /// Whole-program spread lookup requests by `(original width, is_unconditional)`.
    pub global_spread_lookups: HashMap<(u8, bool), usize>,
}

#[derive(Default)]
struct AggregatedConstraintCost {
    recurring_constraints: usize,
    rangecheck_lookups: HashMap<u8, usize>,
    final_spread_lookups: HashMap<u8, usize>,
}

impl AggregatedConstraintCost {
    fn add(&mut self, cost: &Instrumenter, calls: usize) {
        if calls == 0 {
            return;
        }
        self.recurring_constraints += cost.recurring_constraints() * calls;
        for (&(bits, _), &count) in cost.rangecheck_lookups.iter() {
            *self.rangecheck_lookups.entry(bits).or_insert(0) += count * calls;
        }
        for (bits, count) in cost.final_spread_lookups() {
            *self.final_spread_lookups.entry(bits).or_insert(0) += count * calls;
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
        self.recurring_constraints + self.shared_table_constraints()
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
        let field = self.field;
        if !self.stack.is_empty() {
            let (_, cost) = self.stack.last_mut().unwrap();
            cost.record_call(sig.clone());
        }
        if self.entry_point.is_none() {
            self.entry_point = Some(sig.clone());
        }
        if self.functions.contains_key(&sig) {
            self.stack
                .push((sig, Box::new(DummyInstrumenter { field })));
        } else {
            let instrumenter = FunctionCost {
                calls: HashMap::default(),
                raw: Instrumenter::new(field),
                specialized: Instrumenter::new(field),
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

    /// Seal any frames left on the stack after a symbolic execution aborted via an
    /// `AssertionFailure` (so `on_return` never ran). The partial costs are irrelevant — a program
    /// that statically violates a constraint is rejected by R1CS generation — but the per-function
    /// maps must stay consistent (the entry point in particular must be present) so
    /// `summarize`/`walk_call_tree` do not panic on a missing function.
    fn finalize_aborted(&mut self) {
        while !self.stack.is_empty() {
            self.exit_call();
        }
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
            functions: HashMap::default(),
            total_constraints: 0,
            total_savings_to_make: 0,
            global_rangecheck_lookups: HashMap::default(),
            global_spread_lookups: HashMap::default(),
        };
        for (sig, cost) in self.functions.iter() {
            r.functions.insert(
                sig.clone(),
                SpecializationSummary {
                    calls: 0,
                    raw_constraints: cost.raw.recurring_constraints(),
                    specialized_constraints: cost.specialized.recurring_constraints(),
                    specialization_total_savings: 0,
                },
            );
        }
        self.walk_call_tree(&mut r, 1, self.entry_point.as_ref().unwrap());

        let mut aggregate = AggregatedConstraintCost::default();
        for (sig, summary) in r.functions.iter_mut() {
            summary.specialization_total_savings = summary
                .raw_constraints
                .saturating_sub(summary.specialized_constraints)
                * summary.calls;
            r.total_savings_to_make += summary.specialization_total_savings;
            let raw = &self.functions[sig].raw;
            aggregate.add(raw, summary.calls);
            // Aggregate the _raw_ (un-spilled) lookup widths for the table-size optimizer.
            // Note `AggregatedConstraintCost` pre-splits spreads into bytes via
            // `final_spread_lookups()`; the optimizer needs the original widths instead.
            for (&key, &count) in raw.rangecheck_lookups.iter() {
                *r.global_rangecheck_lookups.entry(key).or_insert(0) += count * summary.calls;
            }
            for (&key, &count) in raw.spread_lookups.iter() {
                *r.global_spread_lookups.entry(key).or_insert(0) += count * summary.calls;
            }
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
            field: ssa.field(),
            functions: HashMap::default(),
            stack: vec![],
            entry_point: Some(main_sig.clone()),
            cache: HashMap::default(),
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
        // Upstream constant folding can make a constraint's operands compile-time constants, so
        // the cost estimator can reach a statically-violated assertion even on symbolic inputs
        // (e.g. an `execution_failure` program with an unsatisfiable constraint). That is not a
        // bug here: the cost accumulated in `costs` so far is kept, and R1CS generation is the
        // canonical reporter that rejects the program. So we stop costing this function rather
        // than crashing compilation.
        if let Err(failure) = SymbolicExecutor::new().run(ssa, type_info, sig.id, inputs, costs) {
            debug!(
                message = %"cost analysis: statically-violated assertion; stopping cost estimation for this function",
                failure = %failure
            );
            costs.finalize_aborted();
        }
    }

    fn type_to_unknown_sig(&self, tp: &Type) -> ValueSignature {
        match &tp.expr {
            TypeExpr::Field => ValueSignature::Unknown(ScalarKind::Field),
            TypeExpr::Int(s) => ValueSignature::Unknown(ScalarKind::Int(*s)),
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
            TypeExpr::Tuple(_) => ice_non_elided_tuple(),
            TypeExpr::Ref(inner) => {
                ValueSignature::PointerTo(Box::new(self.type_to_unknown_sig(inner)))
            }
            _ => ValueSignature::Unknown(ScalarKind::Field),
        }
    }

    fn make_main_sig(&self, ssa: &HLSSA) -> FunctionSignature {
        let id = ssa.get_unique_entrypoint_id();
        let main_fn = ssa.get_function(id);
        let params = main_fn.get_param_types();
        let params = params
            .iter()
            .map(|param| self.type_to_unknown_sig(param))
            .collect();
        FunctionSignature { id, params }
    }
}

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

#[cfg(test)]
mod tests {
    use super::{BinaryArithOpKind, CmpKind, CostEstimator, DummyInstrumenter, ScalarKind, Value};
    use crate::compiler::{
        analysis::{flow_analysis::FlowAnalysis, types::Types},
        pass_manager::{AnalysisStore, Pass},
        passes::instruction_lowering::InstructionLowering,
        ssa::{
            Terminator,
            hlssa::{Constant, Endianness, HLSSA, OpCode, Radix, Type},
        },
    };
    use mavros_artifacts::FieldConfig;
    use mavros_int_semantics::{IntOp, Sign, corners, residue};

    /// The comparison's reading comes from the opcode, so one pair of operands must compare two
    /// different ways under the two orderings.
    ///
    /// The cost model estimates a circuit it does not build, and it is only useful while it agrees
    /// with the one that _is_ built. There is nothing about a `Value::Int` that can tell these two
    /// answers apart — the payload is the same eight bits either way — so if the reading ever comes
    /// from anywhere but `kind` again, this is the pair that catches it.
    #[test]
    fn a_comparison_reads_its_operands_the_way_the_opcode_says() {
        // 0xFB in eight bits is 251 read as a magnitude and -5 read as two's complement, so the
        // two readings disagree about every comparison of it against a small positive.
        let a = Value::Int(8, 0xFB);
        let b = Value::Int(8, 2);

        let is = |v: Value, expected: u128| matches!(v, Value::Int(1, got) if got == expected);

        assert!(
            is(a.cmp_op(&b, CmpKind::ULt, Some(8)), 0),
            "251 < 2 is false"
        );
        assert!(is(a.cmp_op(&b, CmpKind::SLt, Some(8)), 1), "-5 < 2 is true");
        assert!(is(a.cmp_op(&b, CmpKind::Eq, Some(8)), 0));
        assert!(is(a.cmp_op(&a, CmpKind::Eq, Some(8)), 1));
    }

    /// An out-of-range shift amount **masks** to `bits - 1` rather than saturating.
    #[test]
    fn an_out_of_range_shift_amount_masks_rather_than_saturating() {
        use BinaryArithOpKind::{SShl, SShr, UShl, UShr};

        let mut dummy = DummyInstrumenter {
            field: FieldConfig::bn254(),
        };
        fn shift(
            a: &Value,
            b: &Value,
            kind: BinaryArithOpKind,
            dummy: &mut DummyInstrumenter,
        ) -> u128 {
            match a.binary_arith_op(b, &kind, dummy) {
                Value::Int(_, v) => v,
                other => panic!("expected an integer result, got {other:?}"),
            }
        }

        // An amount wider than the value, whose low three bits are `0`: `256 & 7 == 0`, so every
        // direction and every reading shifts by nothing and returns the value untouched.
        let value = Value::Int(8, 0x5A);
        let wide = Value::Int(32, 256);
        for kind in [SShl, UShl, SShr, UShr] {
            assert_eq!(
                shift(&value, &wide, kind, &mut dummy),
                0x5A,
                "{kind:?} by 256 at eight bits masks to a shift by zero"
            );
        }

        // An in-range amount is untouched by any of this, which is what keeps the assertions above
        // from passing on a shift that had simply stopped working.
        let one = Value::Int(8, 1);
        assert_eq!(shift(&value, &one, UShl, &mut dummy), 0xB4);
        assert_eq!(shift(&value, &one, UShr, &mut dummy), 0x2D);

        // A negative amount is a large magnitude, so it masks to `7` rather than saturating.
        let negative = Value::Int(8, 0xFF);
        assert_eq!(shift(&value, &negative, SShl, &mut dummy), 0x00);
        assert_eq!(shift(&value, &negative, UShr, &mut dummy), 0x00);
        assert_eq!(
            shift(&Value::Int(8, 0xF0), &negative, SShr, &mut dummy),
            0xFF,
            "sign-filling a negative value still saturates it at -1, by shifting seven places"
        );
    }

    /// The integer arm hands the model the operands it was given, in the order it was given them,
    /// under the reading the _opcode_ names.
    #[test]
    fn the_integer_arm_delegates_with_the_operands_and_reading_it_was_given() {
        use BinaryArithOpKind::{SDiv, SRem, SShr, SSub, UDiv, URem, UShr, USub};

        let mut dummy = DummyInstrumenter {
            field: FieldConfig::bn254(),
        };

        for (kind, op, sign) in [
            (USub, IntOp::Sub, Sign::Unsigned),
            (SSub, IntOp::Sub, Sign::Signed),
            (UDiv, IntOp::Div, Sign::Unsigned),
            (SDiv, IntOp::Div, Sign::Signed),
            (URem, IntOp::Rem, Sign::Unsigned),
            (SRem, IntOp::Rem, Sign::Signed),
            (UShr, IntOp::Shr, Sign::Unsigned),
            (SShr, IntOp::Shr, Sign::Signed),
        ] {
            for bits in [8usize, 32] {
                for a in corners::values(bits) {
                    for b in corners::values(bits) {
                        let got = match Value::Int(bits, a).binary_arith_op(
                            &Value::Int(bits, b),
                            &kind,
                            &mut dummy,
                        ) {
                            Value::Int(width, v) => {
                                assert_eq!(width, bits, "{kind:?} changed the result's width");
                                v
                            }
                            other => panic!("expected an integer result, got {other:?}"),
                        };

                        assert_eq!(
                            got,
                            residue(op, sign, bits, a, bits, b).unwrap_or(0),
                            "{kind:?} at {bits} bits disagreed on {a:#x}, {b:#x}"
                        );
                    }
                }
            }
        }
    }

    /// `ToBits` always produces an array, including when the input's concrete value is unknown.
    /// Keeping that shape is especially important for witnessed unknowns: the witness wrapper
    /// maps over the result to mark each output bit as witness-derived.
    #[test]
    fn witnessed_unknown_to_bits_preserves_array_shape() {
        let value = Value::WitnessOf(Box::new(Value::Unknown(ScalarKind::Field)));

        let result = value.to_bits(&Endianness::Little, 4);

        let Value::Array(bits) = result else {
            panic!("ToBits of a witnessed unknown must produce an array");
        };
        assert_eq!(bits.len(), 4);
        assert!(bits.iter().all(|bit| matches!(
            bit,
            Value::WitnessOf(inner)
                if matches!(inner.as_ref(), Value::Unknown(ScalarKind::Int(1)))
        )));
    }

    #[test]
    fn integer_to_bits_zero_pads_past_u128() {
        let Value::Array(bits) = Value::Int(8, 5).to_bits(&Endianness::Little, 130) else {
            panic!("ToBits must produce an array");
        };

        assert_eq!(bits.len(), 130);
        assert!(matches!(bits[0], Value::Int(1, 1)));
        assert!(matches!(bits[1], Value::Int(1, 0)));
        assert!(matches!(bits[2], Value::Int(1, 1)));
        assert!(
            bits[128..]
                .iter()
                .all(|bit| matches!(bit, Value::Int(1, 0)))
        );
    }

    #[test]
    fn field_to_bits_zero_pads_before_big_endian_output() {
        let ssa = HLSSA::with_main("main".to_string());
        let value = Value::Field(ssa.field().constant(5));
        let Value::Array(bits) = value.to_bits(&Endianness::Big, 260) else {
            panic!("ToBits must produce an array");
        };

        assert_eq!(bits.len(), 260);
        assert!(
            bits[..257]
                .iter()
                .all(|bit| matches!(bit, Value::Int(1, 0)))
        );
        assert!(matches!(bits[257], Value::Int(1, 1)));
        assert!(matches!(bits[258], Value::Int(1, 0)));
        assert!(matches!(bits[259], Value::Int(1, 1)));
    }

    /// Witness lowering makes the decomposition cost visible as one one-bit rangecheck per bit
    /// plus one recomposition constraint. `SpecSplitValue::to_bits` must not charge it again.
    #[test]
    fn lowered_witness_to_bits_cost_is_explicit() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let value = ssa.fresh_value();
        let result = ssa.fresh_value();
        let function = ssa.get_unique_entrypoint_mut();
        function.add_return_type(Type::witness_of(Type::int(1)).array_of(3));
        let entry = function.get_entry_mut();
        entry.push_parameter(value, Type::witness_of(Type::field()));
        entry.push_test_instruction(OpCode::ToBits {
            result,
            value,
            endianness: Endianness::Little,
            count: 3,
        });
        entry.set_terminator(Terminator::Return(vec![result]));

        InstructionLowering::witness_integer_ops().run(&mut ssa, &AnalysisStore::new());
        let flow = FlowAnalysis::run(&ssa);
        let type_info = Types::new().run(&ssa, &flow);
        let costs = CostEstimator::new().run(&ssa, &type_info);
        let function_cost = costs
            .functions
            .values()
            .next()
            .expect("main must be costed");

        assert_eq!(function_cost.raw.recurring_constraints(), 4);
        assert_eq!(function_cost.specialized.recurring_constraints(), 4);
    }

    /// `ToRadix` follows the same accounting path: the pure hint is free, while each lowered
    /// byte rangecheck and the recomposition constraint are recorded explicitly.
    #[test]
    fn lowered_witness_to_radix_cost_is_explicit() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let value = ssa.fresh_value();
        let result = ssa.fresh_value();
        let function = ssa.get_unique_entrypoint_mut();
        function.add_return_type(Type::witness_of(Type::int(8)).array_of(3));
        let entry = function.get_entry_mut();
        entry.push_parameter(value, Type::witness_of(Type::field()));
        entry.push_test_instruction(OpCode::ToRadix {
            result,
            value,
            radix: Radix::Bytes,
            endianness: Endianness::Little,
            count: 3,
        });
        entry.set_terminator(Terminator::Return(vec![result]));

        InstructionLowering::witness_integer_ops().run(&mut ssa, &AnalysisStore::new());
        let flow = FlowAnalysis::run(&ssa);
        let type_info = Types::new().run(&ssa, &flow);
        let costs = CostEstimator::new().run(&ssa, &type_info);
        let function_cost = costs
            .functions
            .values()
            .next()
            .expect("main must be costed");

        for instrumenter in [&function_cost.raw, &function_cost.specialized] {
            assert_eq!(instrumenter.constrains, 1);
            assert_eq!(instrumenter.rangecheck_lookups.get(&(8, true)), Some(&3));
        }
    }

    /// Run the cost estimator over `ssa` with freshly-computed dependencies, then `summarize` it —
    /// exactly the path `Summary::compute` drives. The estimator absorbs static assertion failures
    /// internally, so the point of these tests is simply that the whole path does not panic (in
    /// particular `summarize`/`walk_call_tree`, which would trip over a failed function missing
    /// from the cost map).
    fn run_cost_estimator(ssa: &HLSSA) {
        let flow = FlowAnalysis::run(ssa);
        let type_info = Types::new().run(ssa, &flow);
        let _ = CostEstimator::new().run(ssa, &type_info).summarize();
    }

    /// `main` with a single `Constrain { a, b, c }` over compile-time field constants.
    fn ssa_constraining_constants(a: u64, b: u64, c: u64) -> HLSSA {
        let mut ssa = HLSSA::with_main("main".to_string());
        let a = ssa.add_const(Constant::Field(ssa.field().constant(a)));
        let b = ssa.add_const(Constant::Field(ssa.field().constant(b)));
        let c = ssa.add_const(Constant::Field(ssa.field().constant(c)));
        let entry = ssa.get_unique_entrypoint_mut().get_entry_mut();
        entry.push_test_instruction(OpCode::Constrain { a, b, c });
        entry.set_terminator(Terminator::Return(vec![]));
        ssa
    }

    /// A `Constrain` over constants whose product does NOT equal the third operand is statically
    /// unsatisfiable (an `execution_failure`-style program). The cost estimator must surface this
    /// through the `AssertionFailure` channel and keep going, NOT crash compilation. Regression for
    /// the `assert_eq!(a * b, c)` panic that the Click-Cooper writebacks exposed on
    /// `execution_failure/regression_5202`.
    #[test]
    fn statically_false_constraint_does_not_crash_cost_estimation() {
        // 2 * 3 = 6 ≠ 7.
        run_cost_estimator(&ssa_constraining_constants(2, 3, 7));
    }

    /// The satisfiable control: a trivially-true constant constraint is elided (costs nothing) and
    /// the estimator runs cleanly.
    #[test]
    fn statically_true_constant_constraint_does_not_crash_cost_estimation() {
        // 2 * 3 = 6.
        run_cost_estimator(&ssa_constraining_constants(2, 3, 6));
    }
}
