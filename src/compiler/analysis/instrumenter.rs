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
    taint_analysis::ConstantTaint,
};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ValueSignature {
    U(usize, u128),
    Field(Field),
    Array(Vec<ValueSignature>),
    PointerTo(Box<ValueSignature>),
    FWitness,
    UWitness(usize),
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
            ValueSignature::FWitness => Value::FWitness,
            ValueSignature::UWitness(s) => Value::UWitness(*s),
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
            ValueSignature::FWitness => "W".to_string(),
            ValueSignature::UWitness(_) => "W".to_string(),
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
    FWitness,
    UWitness(usize),
    Tuple(Vec<Value>),
}

impl Value {
    fn cmp_op(
        &self,
        b: &Value,
        cmp_kind: &crate::compiler::ssa::CmpKind,
        instrumenter: &mut dyn OpInstrumenter,
    ) -> Value {
        match (self, b) {
            (Value::U(_, a), Value::U(_, b)) => match cmp_kind {
                CmpKind::Eq => Value::U(1, if a == b { 1 } else { 0 }),
                CmpKind::Lt => Value::U(1, if a < b { 1 } else { 0 }),
            },
            (Value::Field(a), Value::Field(b)) => match cmp_kind {
                CmpKind::Eq => Value::U(1, if a == b { 1 } else { 0 }),
                CmpKind::Lt => Value::U(1, if a < b { 1 } else { 0 }),
            },
            (Value::UWitness(i), _) | (_, Value::UWitness(i)) => {
                instrumenter.record_constraints(1);
                instrumenter.record_rangechecks(*i as u8, 1);
                Value::UWitness(1)
            }
            (Value::FWitness, _) | (_, Value::FWitness) => {
                todo!();
            }
            (_, _) => {
                panic!("Cannot compare {:?} and {:?}", self, b);
            }
        }
    }

    fn binary_arith_op(
        &self,
        b: &Value,
        binary_arith_op_kind: &crate::compiler::ssa::BinaryArithOpKind,
        instrumenter: &mut dyn OpInstrumenter,
    ) -> Value {
        match (self, b) {
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
            (Value::FWitness, Value::FWitness) => match binary_arith_op_kind {
                BinaryArithOpKind::Add => Value::FWitness,
                BinaryArithOpKind::Sub => Value::FWitness,
                BinaryArithOpKind::Mul => {
                    instrumenter.record_constraints(1);
                    Value::FWitness
                }
                BinaryArithOpKind::Div => {
                    instrumenter.record_constraints(1);
                    Value::FWitness
                }
                BinaryArithOpKind::And => todo!(),
            },
            (Value::FWitness, Value::Field(a)) | (Value::Field(a), Value::FWitness) => {
                match binary_arith_op_kind {
                    BinaryArithOpKind::Mul => {
                        if a == &Field::ZERO {
                            Value::Field(Field::ZERO)
                        } else {
                            Value::FWitness
                        }
                    }
                    _ => Value::FWitness,
                }
            }
            (Value::UWitness(1), _) | (_, Value::UWitness(1)) => match binary_arith_op_kind {
                BinaryArithOpKind::And => {
                    instrumenter.record_constraints(1);
                    Value::UWitness(1)
                }
                _ => todo!("{:?}", binary_arith_op_kind),
            },
            (Value::UWitness(s), _) | (_, Value::UWitness(s)) => match binary_arith_op_kind {
                BinaryArithOpKind::Mul => {
                    instrumenter.record_constraints(2); // compute result && check the digital decomposition
                    instrumenter.record_rangechecks(*s as u8, 2);
                    Value::UWitness(*s)
                }
                _ => todo!("{:?}", binary_arith_op_kind),
            },
            (_, _) => panic!("Cannot perform binary arithmetic on {:?} and {:?}", self, b),
        }
    }

    fn blind_from(&mut self, tp: &Type<ConstantTaint>) {
        match (&self, tp.get_annotation(), &tp.expr) {
            (Value::UWitness(_), _, _) => {}
            (Value::FWitness, _, _) => {}
            (Value::U(s, _), ConstantTaint::Witness, _) => *self = Value::UWitness(*s),
            (Value::U { .. }, _, _) => {}
            (Value::Field { .. }, ConstantTaint::Witness, _) => *self = Value::FWitness,
            (Value::Field { .. }, _, _) => {}
            (Value::Array(_), ConstantTaint::Witness, _) => {
                panic!("Witness arrays not supported yet")
            }
            (Value::Array(_), _, tp) => {
                let item_tp = match tp {
                    TypeExpr::Array(tp, _) => tp,
                    TypeExpr::Slice(tp) => tp,
                    _ => panic!("Unexpected array type: {:?}", tp),
                };
                match self {
                    Value::Array(vals) => {
                        for val in vals {
                            val.blind_from(&item_tp);
                        }
                    }
                    _ => panic!("Unexpected array type: {:?}", tp),
                }
            }
            (Value::Pointer(_), ConstantTaint::Witness, _) => {
                panic!("Witness pointers not supported yet");
            }
            (Value::Pointer(val), ConstantTaint::Pure, tp) => {
                let item_tp = match tp {
                    TypeExpr::Ref(tp) => tp,
                    _ => panic!("Unexpected pointer type: {:?}", tp),
                };
                val.borrow_mut().blind_from(&item_tp);
            }
            (Value::Tuple(_), ConstantTaint::Witness, _) => {
                panic!("Witness tuples not supported yet")
            }
            (Value::Tuple(_), _, tp) => {
                let element_tps = match tp {
                    TypeExpr::Tuple(elements) => elements,
                    _ => panic!("Unexpected tuple type: {:?}", tp),
                };
                match self {
                    Value::Tuple(vals) => {
                        for (val, element_tp) in vals.iter_mut().zip(element_tps.iter()) {
                            val.blind_from(element_tp);
                        }
                    }
                    _ => panic!("Unexpected tuple type: {:?}", tp),
                }
            }
        }
    }

    fn make_unspecialized_sig(&self, tp: &Type<ConstantTaint>) -> ValueSignature {
        match (self, tp.get_annotation(), &tp.expr) {
            (Value::UWitness(s), _, _) => ValueSignature::UWitness(*s),
            (Value::FWitness, _, _) => ValueSignature::FWitness,
            (Value::U(s, v), ConstantTaint::Pure, _) => ValueSignature::U(*s, *v),
            (Value::U(s, _), ConstantTaint::Witness, _) => ValueSignature::UWitness(*s),
            (Value::Field(f), ConstantTaint::Pure, _) => ValueSignature::Field(f.clone()),
            (Value::Field(_), ConstantTaint::Witness, _) => ValueSignature::FWitness,
            (Value::Array(vals), ConstantTaint::Pure, tp) => {
                let item_tp = match tp {
                    TypeExpr::Array(tp, _) => tp,
                    TypeExpr::Slice(tp) => tp,
                    _ => panic!("Unexpected array type: {:?}", tp),
                };
                ValueSignature::Array(
                    vals.iter()
                        .map(|v| v.make_unspecialized_sig(&item_tp))
                        .collect(),
                )
            }
            (Value::Array(_), ConstantTaint::Witness, _) => {
                panic!("Witness arrays not supported yet")
            }
            (Value::Pointer(val), ConstantTaint::Pure, tp) => {
                let item_tp = match tp {
                    TypeExpr::Ref(tp) => tp,
                    _ => panic!("Unexpected pointer type: {:?}", tp),
                };
                ValueSignature::PointerTo(Box::new(val.borrow().make_unspecialized_sig(&item_tp)))
            }
            (Value::Pointer(_), ConstantTaint::Witness, _) => {
                panic!("Witness pointers not supported yet")
            }
            (Value::Tuple(vals), ConstantTaint::Pure, tp) => {
                let element_tps = match tp {
                    TypeExpr::Tuple(elements) => elements,
                    _ => panic!("Unexpected tuple type: {:?}", tp),
                };
                ValueSignature::Tuple(
                    vals.iter()
                        .zip(element_tps.iter())
                        .map(|(v, element_tp)| v.make_unspecialized_sig(element_tp))
                        .collect(),
                )
            }
            (Value::Tuple(_), ConstantTaint::Witness, _) => {
                panic!("Witness tuples not supported yet")
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
            Value::UWitness(_) => true,
            Value::FWitness => true,
            _ => false,
        }
    }

    fn array_get(
        &self,
        index: &Value,
        tp: &Type<ConstantTaint>,
        instrumenter: &mut dyn OpInstrumenter,
    ) -> Value {
        match (self, index) {
            (Value::Array(vals), Value::U(_, index)) => vals[*index as usize].clone(),
            (Value::Array(vals), Value::UWitness(_)) => {
                // TODO: Measure type width
                instrumenter.record_lookups(vals.len(), 1, 1);
                Value::witness_of(tp)
            }
            _ => panic!(
                "Cannot get array element from {:?} with index {:?}",
                self, index
            ),
        }
    }

    fn tuple_get(
        &self,
        index: usize,
        _tp: &Type<ConstantTaint>,
        _instrumenter: &mut dyn OpInstrumenter,
    ) -> Value {
        match self {
            Value::Tuple(vals) => vals[index as usize].clone(),
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
            (Value::Array(vals), Value::U(_, index), value) => {
                let mut new_vals = vals.clone();
                new_vals[*index as usize] = value.clone();
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
            (Value::U(_, v), CastTarget::U(s2)) => Value::U(*s2, *v),
            (Value::UWitness(_), CastTarget::U(s2)) => Value::UWitness(*s2),
            (Value::U(_, v), CastTarget::Field) => Value::Field(Field::from(*v)),
            (Value::UWitness(_), CastTarget::Field) => Value::FWitness,
            (Value::Field(f), CastTarget::Field) => Value::Field(f.clone()),
            (Value::Field(f), CastTarget::U(s)) => {
                let bigint = f.into_bigint();

                Value::U(*s, bigint.0[0] as u128 | ((bigint.0[1] as u128) << 64))
            }
            (Value::FWitness, CastTarget::U(s)) => Value::UWitness(*s),
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
        _instrumenter: &mut dyn OpInstrumenter,
    ) -> Value {
        match self {
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
        match (self, radix) {
            (Value::FWitness, Radix::Dyn(Value::U(32, 256))) => {
                instrumenter.record_rangechecks(8, size);
                instrumenter.record_constraints(1);
                Value::Array(vec![Value::UWitness(8); size])
            }
            _ => panic!("Cannot convert {:?} to radix {:?}", self, radix),
        }
    }

    fn not_op(&self, _instrumenter: &mut dyn OpInstrumenter) -> Value {
        match self {
            Value::U(s, v) => Value::U(*s, !v),
            Value::UWitness(1) => Value::UWitness(1),
            _ => panic!("Cannot perform not operation on {:?}", self),
        }
    }

    fn ptr_read(&self, _tp: &Type<ConstantTaint>, _instrumenter: &mut dyn OpInstrumenter) -> Value {
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

    fn witness_of(tp: &Type<ConstantTaint>) -> Value {
        match &tp.expr {
            TypeExpr::U(s) => Value::UWitness(*s),
            TypeExpr::Field => Value::FWitness,
            TypeExpr::WitnessRef => Value::FWitness,
            TypeExpr::Array(tp, size) => {
                let mut values = vec![];
                for _ in 0..*size {
                    values.push(Self::witness_of(tp));
                }
                Value::Array(values)
            }
            TypeExpr::Slice(_) => panic!("Cannot witness slice type"),
            TypeExpr::Ref(_) => panic!("Cannot witness pointer type"),
            TypeExpr::Tuple(_elements) => {todo!("Tuples not supported yet")}
        }
    }

    fn select(
        &self,
        if_true: &Value,
        if_false: &Value,
        tp: &Type<ConstantTaint>,
        get_specialized: &mut dyn OpInstrumenter,
    ) -> Value {
        match self {
            Value::U(_, 0) => if_true.clone(),
            Value::U(_, _) => if_false.clone(),
            Value::UWitness(_) => {
                if if_true.is_witness() || if_false.is_witness() {
                    get_specialized.record_constraints(1);
                }
                Self::witness_of(tp)
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
    fn blind_unspecialized_from(&mut self, tp: &Type<ConstantTaint>) {
        self.unspecialized.blind_from(tp);
    }

    fn blind_from(&mut self, tp: &Type<ConstantTaint>) {
        self.unspecialized.blind_from(tp);
        self.specialized.blind_from(tp);
    }

    fn make_unspecialized_sig(&self, tp: &Type<ConstantTaint>) -> ValueSignature {
        self.unspecialized.make_unspecialized_sig(tp)
    }
}

impl symbolic_executor::Value<CostAnalysis, ConstantTaint> for SpecSplitValue {
    fn cmp(
        &self,
        b: &SpecSplitValue,
        cmp_kind: CmpKind,
        _tp: &Type<ConstantTaint>,
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
        _tp: &Type<ConstantTaint>,
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
        _tp: &Type<ConstantTaint>,
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
        _tp: &Type<ConstantTaint>,
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

    fn not(&self, _tp: &Type<ConstantTaint>, instrumenter: &mut CostAnalysis) -> SpecSplitValue {
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

    fn ptr_read(&self, tp: &Type<ConstantTaint>, ctx: &mut CostAnalysis) -> SpecSplitValue {
        let mut res = SpecSplitValue {
            unspecialized: self.unspecialized.ptr_read(tp, ctx.get_unspecialized()),
            specialized: self.specialized.ptr_read(tp, ctx.get_specialized()),
        };
        res.blind_unspecialized_from(tp);
        res
    }

    fn mk_array(
        values: Vec<SpecSplitValue>,
        _ctx: &mut CostAnalysis,
        _seq_type: SeqType,
        _elem_type: &Type<ConstantTaint>,
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
        _elem_types: &[Type<ConstantTaint>],
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
        tp: &Type<ConstantTaint>,
        instrumenter: &mut CostAnalysis,
    ) -> SpecSplitValue {
        let mut res = SpecSplitValue {
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
        };
        res.blind_unspecialized_from(tp);
        res
    }

    fn tuple_get(
        &self,
        index: usize,
        tp: &Type<ConstantTaint>,
        instrumenter: &mut CostAnalysis,
    ) -> SpecSplitValue {
        let mut res = SpecSplitValue {
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
        };
        res.blind_unspecialized_from(tp);
        res
    }

    fn array_set(
        &self,
        i: &SpecSplitValue,
        v: &SpecSplitValue,
        _tp: &Type<ConstantTaint>,
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
        tp: &Type<ConstantTaint>,
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
        _tp: &Type<ConstantTaint>,
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
        _tp: &Type<ConstantTaint>,
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
        match (&self.unspecialized, &self.specialized) {
            (Value::U(1, v), Value::U(1, v2)) => {
                assert_eq!(*v, *v2);
                *v != 0
            }
            _ => panic!(
                "Expected constant bool, got {:?} and {:?}",
                self.unspecialized, self.specialized
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

    fn alloc(_ctx: &mut CostAnalysis) -> Self {
        Self {
            unspecialized: Value::Pointer(Rc::new(RefCell::new(Value::FWitness))),
            specialized: Value::Pointer(Rc::new(RefCell::new(Value::FWitness))),
        }
    }

    fn write_witness(&self, tp: Option<&Type<ConstantTaint>>, _ctx: &mut CostAnalysis) -> Self {
        match tp {
            Some(tp) => {
                let mut res = self.clone();
                res.blind_unspecialized_from(tp);
                res
            }
            None => self.clone(),
        }
    }

    fn fresh_witness(_ctx: &mut CostAnalysis) -> Self {
        Self {
            unspecialized: Value::FWitness,
            specialized: Value::FWitness,
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
    pub fn pretty_print(&self, ssa: &SSA<ConstantTaint>, all_params: bool) -> String {
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

impl symbolic_executor::Context<SpecSplitValue, ConstantTaint> for CostAnalysis {
    fn on_call(
        &mut self,
        func: FunctionId,
        params: &mut [SpecSplitValue],
        param_types: &[&Type<ConstantTaint>],
    ) -> Option<Vec<SpecSplitValue>> {
        let mut inputs_sig = vec![];
        for (pval, ptype) in params.iter_mut().zip(param_types.iter()) {
            pval.blind_from(ptype);
            inputs_sig.push(pval.make_unspecialized_sig(ptype));
        }

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

    fn on_return(&mut self, returns: &mut [SpecSplitValue], return_types: &[Type<ConstantTaint>]) {
        for (rval, rtype) in returns.iter_mut().zip(return_types.iter()) {
            rval.blind_from(rtype);
        }

        let sig = self.exit_call();

        let mut caches = vec![];

        for (rval, rtype) in returns.iter().zip(return_types.iter()) {
            caches.push(rval.make_unspecialized_sig(rtype));
        }

        self.cache.insert(sig.clone(), caches);
    }

    fn on_jmp(
        &mut self,
        _target: crate::compiler::ssa::BlockId,
        params: &mut [SpecSplitValue],
        param_types: &[&Type<ConstantTaint>],
    ) {
        for (pval, ptype) in params.iter_mut().zip(param_types.iter()) {
            pval.blind_unspecialized_from(ptype);
        }
    }

    fn todo(
        &mut self,
        payload: &str,
        _result_types: &[Type<ConstantTaint>],
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
    pub fn pretty_print(&self, ssa: &SSA<ConstantTaint>) -> String {
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

    pub fn pretty_print(&self, ssa: &SSA<ConstantTaint>) -> String {
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
        ssa: &SSA<ConstantTaint>,
        type_info: &TypeInfo<ConstantTaint>,
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
        ssa: &SSA<ConstantTaint>,
        type_info: &TypeInfo<ConstantTaint>,
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

    fn make_main_sig(&self, ssa: &SSA<ConstantTaint>) -> FunctionSignature {
        let id = ssa.get_main_id();
        let main_fn = ssa.get_function(id);
        let params = main_fn.get_param_types();
        let params = params
            .iter()
            .map(|param| self.make_witness_sig(param))
            .collect();
        FunctionSignature { id, params }
    }

    fn make_witness_sig(&self, tp: &Type<ConstantTaint>) -> ValueSignature {
        match &tp.expr {
            TypeExpr::U(size) => ValueSignature::UWitness(*size),
            TypeExpr::Field => ValueSignature::FWitness,
            TypeExpr::Array(internal, size) => {
                ValueSignature::Array(vec![self.make_witness_sig(internal); *size])
            }
            TypeExpr::WitnessRef => ValueSignature::FWitness,
            TypeExpr::Slice(_) => panic!("slice not possible here"),
            TypeExpr::Ref(_) => panic!("ref not possible here"),
            TypeExpr::Tuple(elements) => {
                ValueSignature::Tuple(elements.iter().map(|e| self.make_witness_sig(e)).collect())
            }
        }
    }
}
