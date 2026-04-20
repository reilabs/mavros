use std::{cell::RefCell, rc::Rc};

use crate::compiler::{
    analysis::{
        symbolic_executor::{self, SymbolicExecutor},
        types::TypeInfo,
    },
    ir::r#type::{Type, TypeExpr},
    ssa::{
        self as ssa_mod, BinaryArithOpKind, BlockId, CmpKind, FunctionId, HLSSA, MemOp, Radix,
        SliceOpDir,
    },
};
use ark_ff::{AdditiveGroup, BigInt, BigInteger, Field, PrimeField};
use tracing::{instrument, warn};

pub use mavros_artifacts::{ConstraintsLayout, LC, R1C, R1CS, WitnessLayout};

// #[derive(Clone, Debug, Copy, PartialEq, PartialOrd, Eq, Ord)]
// pub enum WitnessIndex {
//     PreCommitment(usize),
//     ChallengePower(usize, usize),
//     LookupValueInverse(usize),
//     LookupValueInverseAux(usize),
// }

#[derive(Clone, Debug)]
struct ArrayData {
    table_id: Option<usize>,
    data: Vec<Value>,
}

#[derive(Clone, Debug)]
struct TupleData {
    table_id: Option<usize>,
    data: Vec<Value>,
}

#[derive(Clone, Debug)]
pub enum Value {
    Const(ark_bn254::Fr),
    LC(LC),
    Array(Rc<RefCell<ArrayData>>),
    Tuple(Rc<RefCell<TupleData>>),
    Ptr(Rc<RefCell<Value>>),
    Invalid,
}

impl Value {
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

    fn wrap_unsigned(v: u128, bits: usize) -> u128 {
        v & Self::bit_mask(bits)
    }

    fn decode_signed(v: u128, bits: usize) -> i128 {
        assert!(bits > 0 && bits <= 128, "invalid signed width: {bits}");
        let masked = Self::wrap_unsigned(v, bits);
        if bits == 128 {
            masked as i128
        } else {
            let sign_bit = 1u128 << (bits - 1);
            if masked & sign_bit == 0 {
                masked as i128
            } else {
                (masked as i128) - ((1u128 << bits) as i128)
            }
        }
    }

    fn encode_signed(v: i128, bits: usize) -> u128 {
        Self::wrap_unsigned(v as u128, bits)
    }

    pub fn add(&self, other: &Value) -> Value {
        match (self, other) {
            (Value::Const(lhs), Value::Const(rhs)) => Value::Const(lhs + rhs),
            (_, _) => {
                let lhs = self.expect_linear_combination();
                let rhs = other.expect_linear_combination();
                let mut lhs_i = 0;
                let mut rhs_i = 0;
                let mut result = Vec::new();
                while lhs_i < lhs.len() && rhs_i < rhs.len() {
                    if lhs[lhs_i].0 == rhs[rhs_i].0 {
                        let r = lhs[lhs_i].1 + rhs[rhs_i].1;
                        if r != ark_bn254::Fr::ZERO {
                            result.push((lhs[lhs_i].0, r));
                        }
                        lhs_i += 1;
                        rhs_i += 1;
                    } else if lhs[lhs_i].0 < rhs[rhs_i].0 {
                        result.push(lhs[lhs_i]);
                        lhs_i += 1;
                    } else {
                        result.push(rhs[rhs_i]);
                        rhs_i += 1;
                    }
                }
                while lhs_i < lhs.len() {
                    result.push(lhs[lhs_i]);
                    lhs_i += 1;
                }
                while rhs_i < rhs.len() {
                    result.push(rhs[rhs_i]);
                    rhs_i += 1;
                }
                Value::LC(result)
            }
        }
    }

    fn neg(&self) -> Value {
        match self {
            Value::Const(c) => Value::Const(-*c),
            Value::LC(lc) => Value::LC(lc.iter().map(|(i, c)| (*i, -*c)).collect()),
            _ => panic!("expected linear combination"),
        }
    }

    pub fn sub(&self, other: &Value) -> Value {
        self.add(&other.neg())
    }

    pub fn div(&self, other: &Value) -> Value {
        match (self, other) {
            (Value::Const(lhs), Value::Const(rhs)) => Value::Const(lhs / rhs),
            (_, _) => panic!("expected constant"),
        }
    }

    pub fn expect_constant(&self) -> ark_bn254::Fr {
        match self {
            Value::Const(c) => *c,
            _ => panic!("expected constant"),
        }
    }

    pub fn expect_u1(&self) -> bool {
        match self {
            Value::Const(c) => {
                let v: u128 = c.into_bigint().to_string().parse().unwrap_or_else(|e| {
                    panic!("expected u1, but field value is {}: {e}", c.into_bigint())
                });
                assert!(v <= 1, "expected u1, but value is {v}");
                v == 1
            }
            r => panic!("expected u1, got {:?}", r),
        }
    }

    pub fn expect_u8(&self) -> u8 {
        match self {
            Value::Const(c) => {
                let s = c.into_bigint().to_string();
                s.parse()
                    .unwrap_or_else(|e| panic!("expected u8, but field value is {s}: {e}"))
            }
            r => panic!("expected u8, got {:?}", r),
        }
    }

    pub fn expect_u32(&self) -> u32 {
        match self {
            Value::Const(c) => {
                let s = c.into_bigint().to_string();
                s.parse()
                    .unwrap_or_else(|e| panic!("expected u32, but field value is {s}: {e}"))
            }
            r => panic!("expected u32, got {:?}", r),
        }
    }

    pub fn expect_u64(&self) -> u64 {
        match self {
            Value::Const(c) => {
                let s = c.into_bigint().to_string();
                s.parse()
                    .unwrap_or_else(|e| panic!("expected u64, but field value is {s}: {e}"))
            }
            r => panic!("expected u64, got {:?}", r),
        }
    }

    pub fn expect_u128(&self) -> u128 {
        match self {
            Value::Const(c) => {
                let s = c.into_bigint().to_string();
                s.parse()
                    .unwrap_or_else(|e| panic!("expected u128, but field value is {s}: {e}"))
            }
            r => panic!("expected u128, got {:?}", r),
        }
    }

    pub fn mul(&self, other: &Value) -> Value {
        match (self, other) {
            (Value::Const(lhs), Value::Const(rhs)) => Value::Const(lhs * rhs),
            (Value::Const(c), Value::LC(lc)) | (Value::LC(lc), Value::Const(c)) => {
                if *c == ark_bn254::Fr::ZERO {
                    return Value::Const(ark_bn254::Fr::ZERO);
                }
                let mut result = Vec::new();
                for (i, cl) in lc.iter() {
                    result.push((*i, *c * *cl));
                }
                Value::LC(result)
            }
            (_, _) => panic!("expected constant or linear combination and constant"),
        }
    }

    pub fn expect_ptr(&self) -> Rc<RefCell<Value>> {
        match self {
            Value::Ptr(ptr) => ptr.clone(),
            _ => panic!("expected ptr"),
        }
    }

    pub fn lt(&self, other: &Value) -> Value {
        let self_const = self.expect_constant();
        let other_const = other.expect_constant();
        if self_const < other_const {
            Value::Const(ark_bn254::Fr::ONE)
        } else {
            Value::Const(ark_bn254::Fr::ZERO)
        }
    }

    pub fn expect_array(&self) -> Rc<RefCell<ArrayData>> {
        match self {
            Value::Array(array) => array.clone(),
            _ => panic!("expected array"),
        }
    }

    pub fn expect_tuple(&self) -> Rc<RefCell<TupleData>> {
        match self {
            Value::Tuple(fields) => fields.clone(),
            _ => panic!("expected tuple"),
        }
    }

    pub fn expect_linear_combination(&self) -> Vec<(usize, ark_bn254::Fr)> {
        match self {
            Value::Const(c) => vec![(0, *c)],
            Value::LC(lc) => lc.clone(),
            _ => panic!("expected constant or linear combination"),
        }
    }

    pub fn eq(&self, other: &Value) -> Value {
        let self_const = self.expect_constant();
        let other_const = other.expect_constant();
        if self_const == other_const {
            Value::Const(ark_bn254::Fr::ONE)
        } else {
            Value::Const(ark_bn254::Fr::ZERO)
        }
    }

    pub fn mk_array(data: Vec<Value>) -> Value {
        Value::Array(Rc::new(RefCell::new(ArrayData {
            table_id: None,
            data,
        })))
    }

    pub fn mk_tuple(fields: Vec<Value>) -> Value {
        Value::Tuple(Rc::new(RefCell::new(TupleData {
            table_id: None,
            data: fields,
        })))
    }
}

#[derive(Clone, Debug)]
pub struct LookupConstraint {
    pub table_id: usize,
    pub elements: Vec<LC>,
    pub flag: LC,
}

#[derive(Clone, Debug)]
pub enum Table {
    Range(u64),
    OfElems(Vec<LC>),
    Spread(u8),
}

#[derive(Clone)]
pub struct R1CGen {
    constraints: Vec<R1C>,
    tables: Vec<Table>,
    lookups: Vec<LookupConstraint>,
    next_witness: usize,
}

impl symbolic_executor::Context<Value> for R1CGen {
    fn on_call(
        &mut self,
        _func: FunctionId,
        _params: &mut [Value],
        _param_types: &[&Type],
        _result_types: &[Type],
        unconstrained: bool,
    ) -> Option<Vec<Value>> {
        assert!(
            !unconstrained,
            "ICE: unconstrained calls should be DCE'd before R1CS gen"
        );
        None
    }

    fn on_return(&mut self, _returns: &mut [Value], _return_types: &[Type]) {}

    fn on_jmp(&mut self, _target: BlockId, _params: &mut [Value], _param_types: &[&Type]) {}

    fn lookup(
        &mut self,
        target: super::ssa::LookupTarget<Value>,
        keys: Vec<Value>,
        results: Vec<Value>,
        flag: Value,
    ) {
        let flag_lc = flag.expect_linear_combination();
        match target {
            super::ssa::LookupTarget::Rangecheck(i) => {
                // TODO this will become table resolution logic eventually
                assert!(i == 8, "TODO: support other rangecheck sizes");
                if self.tables.is_empty() {
                    self.tables.push(Table::Range(i as u64))
                } else {
                    match self.tables[0] {
                        Table::Range(i1) => assert_eq!(i1, i as u64, "unsupported"),
                        Table::OfElems(_) | Table::Spread(_) => panic!("unsupported"),
                    }
                }
                let els = keys
                    .into_iter()
                    .chain(results.into_iter())
                    .map(|e| e.expect_linear_combination())
                    .collect();
                self.lookups.push(LookupConstraint {
                    table_id: 0,
                    elements: els,
                    flag: flag_lc.clone(),
                });
            }
            super::ssa::LookupTarget::DynRangecheck(v) => {
                // TODO this will become table resolution logic eventually
                let v = v.expect_u32();
                assert!(v == 256, "TODO: support other rangecheck sizes");
                let i = 8;
                if self.tables.is_empty() {
                    self.tables.push(Table::Range(i as u64))
                } else {
                    match self.tables[0] {
                        Table::Range(i1) => assert_eq!(i1, i as u64, "unsupported"),
                        Table::OfElems(_) | Table::Spread(_) => panic!("unsupported"),
                    }
                }
                let els = keys
                    .into_iter()
                    .chain(results.into_iter())
                    .map(|e| e.expect_linear_combination())
                    .collect();
                self.lookups.push(LookupConstraint {
                    table_id: 0,
                    elements: els,
                    flag: flag_lc.clone(),
                });
            }
            super::ssa::LookupTarget::Spread(bits) => {
                let table_id = {
                    let existing = self.tables.iter().position(|t| match t {
                        Table::Spread(n) => *n == bits,
                        _ => false,
                    });
                    if let Some(idx) = existing {
                        idx
                    } else {
                        self.tables.push(Table::Spread(bits));
                        self.tables.len() - 1
                    }
                };
                let els = keys
                    .into_iter()
                    .chain(results.into_iter())
                    .map(|e| e.expect_linear_combination())
                    .collect();
                self.lookups.push(LookupConstraint {
                    table_id,
                    elements: els,
                    flag: flag_lc,
                });
            }
            super::ssa::LookupTarget::Array(arr) => {
                let arr = arr.expect_array();
                let table_id = if arr.borrow().table_id.is_none() {
                    let elems = arr
                        .borrow()
                        .data
                        .iter()
                        .map(|e| e.expect_linear_combination())
                        .collect();
                    self.tables.push(Table::OfElems(elems));
                    let idx = self.tables.len() - 1;
                    arr.borrow_mut().table_id = Some(idx);
                    idx
                } else {
                    arr.borrow().table_id.unwrap()
                };
                let els = keys
                    .into_iter()
                    .chain(results.into_iter())
                    .map(|e| e.expect_linear_combination())
                    .collect();
                self.lookups.push(LookupConstraint {
                    table_id,
                    elements: els,
                    flag: flag_lc,
                });
            }
        }
    }

    fn todo(&mut self, payload: &str, _result_types: &[Type]) -> Vec<Value> {
        panic!("Todo opcode encountered in R1CSGen: {}", payload);
    }

    fn slice_push(&mut self, slice: &Value, values: &[Value], dir: SliceOpDir) -> Value {
        match dir {
            SliceOpDir::Front => {
                let mut r = values.to_vec();
                r.extend(slice.expect_array().borrow().data.iter().map(|v| v.clone()));
                Value::mk_array(r)
            }
            SliceOpDir::Back => {
                let mut r = slice.expect_array().borrow().data.clone();
                r.extend(values.iter().map(|v| v.clone()));
                Value::mk_array(r)
            }
        }
    }

    fn slice_len(&mut self, slice: &Value) -> Value {
        let array = slice.expect_array();
        Value::Const(ark_bn254::Fr::from(array.borrow().data.len() as u128))
    }

    fn on_guard(
        &mut self,
        _inner: &crate::compiler::ssa::OpCode,
        _condition: &Value,
        _inputs: Vec<&Value>,
        _result_types: Vec<&Type>,
    ) -> Vec<Value> {
        panic!("ICE: Guard should not appear in R1CS gen (should be lowered before)")
    }
}

impl symbolic_executor::Value<R1CGen> for Value {
    fn cmp(&self, b: &Self, cmp_kind: CmpKind, _out_type: &Type, _ctx: &mut R1CGen) -> Self {
        match cmp_kind {
            CmpKind::Eq => self.eq(b),
            CmpKind::Lt => self.lt(b),
        }
    }

    fn arith(
        &self,
        b: &Self,
        binary_arith_op_kind: BinaryArithOpKind,
        out_type: &Type,
        _ctx: &mut R1CGen,
    ) -> Self {
        match &out_type.strip_witness().expr {
            TypeExpr::U(bits) => {
                assert!(
                    *bits > 0 && *bits <= 128,
                    "Unsupported unsigned integer size in R1CS arith: u{bits}"
                );
                assert!(
                    matches!((self, b), (Value::Const(_), Value::Const(_))),
                    "Non-constant integer {:?} is not supported in R1CS arith",
                    binary_arith_op_kind
                );
                let a = Self::wrap_unsigned(self.expect_u128(), *bits);
                let b = Self::wrap_unsigned(b.expect_u128(), *bits);
                let shift = b as u32;
                let result = match binary_arith_op_kind {
                    BinaryArithOpKind::Add => a.wrapping_add(b),
                    BinaryArithOpKind::Sub => a.wrapping_sub(b),
                    BinaryArithOpKind::Mul => a.wrapping_mul(b),
                    BinaryArithOpKind::Div => a / b,
                    BinaryArithOpKind::Mod => a % b,
                    BinaryArithOpKind::And => a & b,
                    BinaryArithOpKind::Or => a | b,
                    BinaryArithOpKind::Xor => a ^ b,
                    BinaryArithOpKind::Shl => a.wrapping_shl(shift),
                    BinaryArithOpKind::Shr => a.wrapping_shr(shift),
                };
                Value::Const(ark_bn254::Fr::from(Self::wrap_unsigned(result, *bits)))
            }
            TypeExpr::I(bits) => {
                assert!(
                    *bits > 0 && *bits <= 128,
                    "Unsupported signed integer size in R1CS arith: i{bits}"
                );
                assert!(
                    matches!((self, b), (Value::Const(_), Value::Const(_))),
                    "Non-constant integer {:?} is not supported in R1CS arith",
                    binary_arith_op_kind
                );
                let a = Self::decode_signed(self.expect_u128(), *bits);
                let b = Self::decode_signed(b.expect_u128(), *bits);
                let b_bits = Self::wrap_unsigned(b as u128, *bits) as u32;
                let result = match binary_arith_op_kind {
                    BinaryArithOpKind::Add => Self::encode_signed(a.wrapping_add(b), *bits),
                    BinaryArithOpKind::Sub => Self::encode_signed(a.wrapping_sub(b), *bits),
                    BinaryArithOpKind::Mul => Self::encode_signed(a.wrapping_mul(b), *bits),
                    BinaryArithOpKind::And => {
                        Self::wrap_unsigned(a as u128, *bits)
                            & Self::wrap_unsigned(b as u128, *bits)
                    }
                    BinaryArithOpKind::Or => {
                        Self::wrap_unsigned(a as u128, *bits)
                            | Self::wrap_unsigned(b as u128, *bits)
                    }
                    BinaryArithOpKind::Xor => {
                        Self::wrap_unsigned(a as u128, *bits)
                            ^ Self::wrap_unsigned(b as u128, *bits)
                    }
                    BinaryArithOpKind::Shl => {
                        let raw = Self::wrap_unsigned(a as u128, *bits).wrapping_shl(b_bits);
                        Self::wrap_unsigned(raw, *bits)
                    }
                    BinaryArithOpKind::Shr => {
                        Self::wrap_unsigned(a as u128, *bits).wrapping_shr(b_bits)
                    }
                    BinaryArithOpKind::Div | BinaryArithOpKind::Mod => {
                        panic!("Signed div/mod not yet implemented")
                    }
                };
                Value::Const(ark_bn254::Fr::from(result))
            }
            TypeExpr::Field | TypeExpr::WitnessOf(_) => match binary_arith_op_kind {
                BinaryArithOpKind::Add => self.add(b),
                BinaryArithOpKind::Sub => self.sub(b),
                BinaryArithOpKind::Mul => self.mul(b),
                BinaryArithOpKind::Div => self.div(b),
                BinaryArithOpKind::Mod => {
                    panic!("Modulo is not defined on field elements")
                }
                BinaryArithOpKind::And
                | BinaryArithOpKind::Or
                | BinaryArithOpKind::Xor
                | BinaryArithOpKind::Shl
                | BinaryArithOpKind::Shr => {
                    panic!("Bitwise operations are not supported on field elements")
                }
            },
            _ => panic!("Unsupported type in R1CS arith"),
        }
    }

    fn assert_eq(&self, other: &Self, _ctx: &mut R1CGen) {
        assert_eq!(self.expect_constant(), other.expect_constant());
    }

    fn assert_r1c(a: &Self, b: &Self, c: &Self, _ctx: &mut R1CGen) {
        let a = a.expect_constant();
        let b = b.expect_constant();
        let c = c.expect_constant();
        assert!(a * b == c);
    }

    fn array_get(&self, index: &Self, _out_type: &Type, _ctx: &mut R1CGen) -> Self {
        let index = index.expect_u32();
        let value = self.expect_array().borrow().data[index as usize].clone();
        value
    }

    fn tuple_get(&self, index: usize, _out_type: &Type, _ctx: &mut R1CGen) -> Self {
        let value = self.expect_tuple().borrow().data[index as usize].clone();
        value
    }

    fn array_set(&self, index: &Self, value: &Self, _out_type: &Type, _ctx: &mut R1CGen) -> Self {
        let array = self.expect_array();
        let index = index.expect_u32();
        let mut new_array = array.borrow().data.clone();
        new_array[index as usize] = value.clone();
        Value::mk_array(new_array)
    }

    fn truncate(&self, _from: usize, to: usize, _out_type: &Type, _ctx: &mut R1CGen) -> Self {
        let new_value = self
            .expect_constant()
            .into_bigint()
            .to_bits_le()
            .iter()
            .take(to)
            .cloned()
            .collect::<Vec<_>>();
        Value::Const(ark_bn254::Fr::from_bigint(BigInt::from_bits_le(&new_value)).unwrap())
    }

    fn sext(&self, from: usize, _to: usize, _out_type: &Type, _ctx: &mut R1CGen) -> Self {
        // Sign-extend: if sign bit is set, add (2^to - 2^from) to the value
        let val = self.expect_constant();
        let bits = val.into_bigint().to_bits_le();
        let sign_bit = if from > 0 && from - 1 < bits.len() {
            bits[from - 1]
        } else {
            false
        };
        if sign_bit {
            let extension = ark_bn254::Fr::from(1u128 << _to) - ark_bn254::Fr::from(1u128 << from);
            Value::Const(val + extension)
        } else {
            self.clone()
        }
    }

    fn cast(
        &self,
        _cast_target: &super::ssa::CastTarget,
        _out_type: &Type,
        _ctx: &mut R1CGen,
    ) -> Self {
        self.clone()
    }

    fn constrain(a: &Self, b: &Self, c: &Self, ctx: &mut R1CGen) {
        let a = a.expect_linear_combination();
        let b = b.expect_linear_combination();
        let c = c.expect_linear_combination();
        ctx.constraints.push(R1C { a, b, c });
    }

    fn to_bits(
        &self,
        endianness: super::ssa::Endianness,
        size: usize,
        _out_type: &Type,
        _ctx: &mut R1CGen,
    ) -> Self {
        let value_const = self.expect_constant();
        let mut bits = value_const.into_bigint().to_bits_le();
        // Truncate or pad to the desired output size
        if bits.len() > size {
            bits.truncate(size);
        } else {
            while bits.len() < size {
                bits.push(false);
            }
        }
        // Handle endianness
        let final_bits = match endianness {
            crate::compiler::ssa::Endianness::Little => bits,
            crate::compiler::ssa::Endianness::Big => {
                let mut reversed = bits;
                reversed.reverse();
                reversed
            }
        };
        // Convert bits to array of field elements (0 or 1)
        let mut bit_values = Vec::new();
        for bit in final_bits {
            let bit_value = if bit {
                Value::Const(ark_bn254::Fr::from(1u128))
            } else {
                Value::Const(ark_bn254::Fr::from(0u128))
            };
            bit_values.push(bit_value);
        }
        Value::mk_array(bit_values)
    }

    fn not(&self, out_type: &Type, _ctx: &mut R1CGen) -> Self {
        let value_const = self.expect_constant();
        let bits = value_const.into_bigint().to_bits_le();
        let bit_size = out_type.get_bit_size();
        let mut negated_bits = Vec::new();
        for i in 0..bit_size {
            let bit = if i < bits.len() { bits[i] } else { false };
            negated_bits.push(!bit);
        }
        Value::Const(ark_bn254::Fr::from_bigint(BigInt::from_bits_le(&negated_bits)).unwrap())
    }

    fn of_u(_s: usize, v: u128, _ctx: &mut R1CGen) -> Self {
        Value::Const(ark_bn254::Fr::from(v))
    }

    fn of_i(_s: usize, v: u128, _ctx: &mut R1CGen) -> Self {
        Value::Const(ark_bn254::Fr::from(v))
    }

    fn of_field(f: super::Field, _ctx: &mut R1CGen) -> Self {
        Value::Const(ark_bn254::Fr::from(f))
    }

    fn mk_array(
        a: Vec<Self>,
        _ctx: &mut R1CGen,
        _seq_type: super::ssa::SeqType,
        _elem_type: &Type,
    ) -> Self {
        Value::mk_array(a)
    }

    fn mk_tuple(elems: Vec<Self>, _ctx: &mut R1CGen, _elem_types: &[Type]) -> Self {
        Value::mk_tuple(elems)
    }

    fn alloc(_elem_type: &Type, _ctx: &mut R1CGen) -> Self {
        Value::Ptr(Rc::new(RefCell::new(Value::Invalid)))
    }

    fn ptr_write(&self, value: &Self, _ctx: &mut R1CGen) {
        let ptr = self.expect_ptr();
        *ptr.borrow_mut() = value.clone();
    }

    fn ptr_read(&self, _out_type: &Type, _ctx: &mut R1CGen) -> Self {
        let ptr = self.expect_ptr();
        let value = ptr.borrow().clone();
        value
    }

    fn expect_constant_bool(&self, _ctx: &mut R1CGen) -> bool {
        self.expect_constant() == ark_bn254::Fr::ONE
    }

    fn select(&self, if_t: &Self, if_f: &Self, _out_type: &Type, _ctx: &mut R1CGen) -> Self {
        self.mul(if_t)
            .add(&Value::Const(ark_bn254::Fr::ONE).sub(self).mul(if_f))
    }

    fn write_witness(&self, _tp: Option<&Type>, ctx: &mut R1CGen) -> Self {
        let witness_var = ctx.next_witness();
        Value::LC(vec![(witness_var, ark_bn254::Fr::ONE)])
    }

    fn fresh_witness(_result_type: &Type, ctx: &mut R1CGen) -> Self {
        let witness_var = ctx.next_witness();
        Value::LC(vec![(witness_var, ark_bn254::Fr::ONE)])
    }

    fn value_of(&self, _ctx: &mut R1CGen) -> Self {
        panic!("ICE: ValueOf should not reach R1CS gen")
    }

    fn mem_op(&self, _kind: MemOp, _ctx: &mut R1CGen) {}

    fn rangecheck(&self, max_bits: usize, _ctx: &mut R1CGen) {
        let self_const = self.expect_constant();
        let check = self_const
            .into_bigint()
            .to_bits_le()
            .iter()
            .skip(max_bits)
            .all(|b| !b);
        assert!(check);
    }

    fn to_radix(
        &self,
        _radix: &Radix<Self>,
        _endianness: crate::compiler::ssa::Endianness,
        _size: usize,
        _out_type: &Type,
        _ctx: &mut R1CGen,
    ) -> Self {
        todo!("ToRadix R1CS generation not yet implemented")
    }

    fn spread(&self, bits: u8, _ctx: &mut R1CGen) -> Self {
        let val = self.expect_constant();
        let v: u128 = val.into_bigint().as_ref()[0] as u128;
        let spread_val = ssa_mod::spread_bits(v, bits as usize);
        Value::Const(ark_bn254::Fr::from(spread_val))
    }

    fn unspread(&self, bits: u8, _ctx: &mut R1CGen) -> (Self, Self) {
        let val = self.expect_constant();
        let v: u128 = val.into_bigint().as_ref()[0] as u128;
        let (odd_val, even_val) = ssa_mod::unspread_bits(v, bits as usize * 2);
        (
            Value::Const(ark_bn254::Fr::from(odd_val)),
            Value::Const(ark_bn254::Fr::from(even_val)),
        )
    }
}

impl R1CGen {
    pub fn new() -> Self {
        Self {
            constraints: vec![],
            next_witness: 0,
            tables: vec![],
            lookups: vec![],
        }
    }

    pub fn verify(&self, witness: &[ark_bn254::Fr]) -> bool {
        for (i, r1c) in self.constraints.iter().enumerate() {
            let a = r1c
                .a
                .iter()
                .map(|(i, c)| c * &witness[*i])
                .sum::<ark_bn254::Fr>();
            let b = r1c
                .b
                .iter()
                .map(|(i, c)| c * &witness[*i])
                .sum::<ark_bn254::Fr>();
            let c = r1c
                .c
                .iter()
                .map(|(i, c)| c * &witness[*i])
                .sum::<ark_bn254::Fr>();
            if a * b != c {
                warn!(message = %"R1CS constraint failed to verify", index = i);
                return false;
            }
        }
        return true;
    }

    #[instrument(skip_all, name = "R1CGen::run")]
    pub fn run(&mut self, ssa: &HLSSA, type_info: &TypeInfo) {
        let entry_point = ssa.get_main_id();
        assert!(
            ssa.get_function(entry_point).get_param_types().len() == 0,
            "Main should not have parameters as WitnessWriteToFresh pass should remove them"
        );
        let main_params = vec![];
        let executor = SymbolicExecutor::new();
        executor.run(ssa, type_info, entry_point, main_params, self);
    }

    pub fn get_r1cs(self) -> Vec<R1C> {
        self.constraints
    }

    pub fn get_witness_size(&self) -> usize {
        self.next_witness
    }

    fn next_witness(&mut self) -> usize {
        let result = self.next_witness;
        self.next_witness += 1;
        result
    }

    pub fn seal(self) -> R1CS {
        // Algebraic section
        let mut witness_layout = WitnessLayout {
            algebraic_size: self.next_witness,
            multiplicities_size: 0,
            challenges_size: 0,
            tables_data_size: 0,
            lookups_data_size: 0,
        };
        let mut constraints_layout = ConstraintsLayout {
            algebraic_size: self.constraints.len(),
            tables_data_size: 0,
            lookups_data_size: 0,
        };
        let mut result = self.constraints;

        // multiplicities init + compute the needed challenges
        struct TableInfo {
            multiplicities_witness_off: usize,
            table: Table,
            sum_constraint_idx: usize,
        }
        let mut table_infos = vec![];
        let mut max_width = 0;
        for table in self.tables.into_iter() {
            match table {
                Table::Range(len) => {
                    let len = 1 << len;
                    table_infos.push(TableInfo {
                        multiplicities_witness_off: witness_layout.multiplicities_size
                            + witness_layout.algebraic_size,
                        table,
                        sum_constraint_idx: 0,
                    });
                    max_width = max_width.max(1);
                    witness_layout.multiplicities_size += len;
                }
                Table::OfElems(els) => {
                    let len = els.len();
                    table_infos.push(TableInfo {
                        multiplicities_witness_off: witness_layout.multiplicities_size
                            + witness_layout.algebraic_size,
                        table: Table::OfElems(els),
                        sum_constraint_idx: 0,
                    });
                    max_width = max_width.max(2);
                    witness_layout.multiplicities_size += len;
                }
                Table::Spread(bits) => {
                    let len = 1usize << bits;
                    table_infos.push(TableInfo {
                        multiplicities_witness_off: witness_layout.multiplicities_size
                            + witness_layout.algebraic_size,
                        table,
                        sum_constraint_idx: 0,
                    });
                    max_width = max_width.max(2); // key + spread_value
                    witness_layout.multiplicities_size += len;
                }
            }
        }

        if table_infos.is_empty() {
            return R1CS {
                witness_layout,
                constraints_layout,
                constraints: result,
            };
        }

        // challenges init
        let alpha = witness_layout.challenges_end();
        witness_layout.challenges_size += 1;
        let beta = if max_width > 1 {
            let beta = witness_layout.challenges_end();
            witness_layout.challenges_size += 1;
            beta
        } else {
            usize::MAX // hoping this crashes soon if used
        };

        // tables contents init
        for table_info in table_infos.iter_mut() {
            match &table_info.table {
                Table::Range(bits) => {
                    // for each element i, we need one witness `y = mᵢ / (α - i)`
                    // and one constraint saying `y * (α - i) - mᵢ = 0`
                    let len = 1 << bits;
                    let mut sum_lhs: LC = vec![];
                    for i in 0..len {
                        let y = witness_layout.next_table_data();
                        let m = table_info.multiplicities_witness_off + i;
                        result.push(R1C {
                            a: vec![(y, ark_bn254::Fr::ONE)],
                            b: vec![
                                (alpha, ark_bn254::Fr::ONE),
                                (0, -crate::compiler::Field::from(i as u64)),
                            ],
                            c: vec![(m, ark_bn254::Fr::ONE)],
                        });
                        sum_lhs.push((y, ark_bn254::Fr::ONE));
                    }
                    result.push(R1C {
                        a: sum_lhs,
                        b: vec![(0, ark_bn254::Fr::ONE)],
                        c: vec![], // this is prepared for the looked up values to come into
                    });
                    table_info.sum_constraint_idx = result.len() - 1;
                }
                Table::OfElems(els) => {
                    // for each element (i, v), we need two witness/constraint pairs:
                    // -> x = β * v, with the constraint `β * v - x = 0`
                    // -> y = mᵢ / (α - i - x), with the constraint `y * (α - i - x) - mᵢ = 0`
                    let mut sum_lhs: LC = vec![];
                    for (i, v) in els.iter().enumerate() {
                        let x = witness_layout.next_table_data();
                        let y = witness_layout.next_table_data();
                        let m = table_info.multiplicities_witness_off + i;
                        result.push(R1C {
                            a: vec![(beta, crate::compiler::Field::ONE)],
                            b: v.clone(),
                            c: vec![(x, -crate::compiler::Field::ONE)],
                        });
                        result.push(R1C {
                            a: vec![(y, ark_bn254::Fr::ONE)],
                            b: vec![
                                (alpha, ark_bn254::Fr::ONE),
                                (0, -crate::compiler::Field::from(i as u64)),
                                (x, -crate::compiler::Field::ONE),
                            ],
                            c: vec![(m, crate::compiler::Field::ONE)],
                        });
                        sum_lhs.push((y, ark_bn254::Fr::ONE));
                    }
                    result.push(R1C {
                        a: sum_lhs,
                        b: vec![(0, ark_bn254::Fr::ONE)],
                        c: vec![], // this is prepared for the looked up values to come into
                    });
                    table_info.sum_constraint_idx = result.len() - 1;
                }
                Table::Spread(bits) => {
                    // Spread table: for each entry i in 0..2^bits, value = spread(i)
                    // Width = 2 (key=i, value=spread(i)), same structure as OfElems
                    let len = 1usize << bits;
                    let mut sum_lhs: LC = vec![];
                    for i in 0..len {
                        let spread_val = ssa_mod::spread_bits(i as u128, 32);
                        let v: LC = vec![(0, crate::compiler::Field::from(spread_val))];
                        let x = witness_layout.next_table_data();
                        let y = witness_layout.next_table_data();
                        let m = table_info.multiplicities_witness_off + i;
                        result.push(R1C {
                            a: vec![(beta, crate::compiler::Field::ONE)],
                            b: v,
                            c: vec![(x, -crate::compiler::Field::ONE)],
                        });
                        result.push(R1C {
                            a: vec![(y, ark_bn254::Fr::ONE)],
                            b: vec![
                                (alpha, ark_bn254::Fr::ONE),
                                (0, -crate::compiler::Field::from(i as u64)),
                                (x, -crate::compiler::Field::ONE),
                            ],
                            c: vec![(m, crate::compiler::Field::ONE)],
                        });
                        sum_lhs.push((y, ark_bn254::Fr::ONE));
                    }
                    result.push(R1C {
                        a: sum_lhs,
                        b: vec![(0, ark_bn254::Fr::ONE)],
                        c: vec![],
                    });
                    table_info.sum_constraint_idx = result.len() - 1;
                }
            }
        }

        constraints_layout.tables_data_size = result.len() - constraints_layout.algebraic_size;

        // lookups init
        for lookup in self.lookups.into_iter() {
            // if lookup.elements.len() >= 2 {
            //     todo!("wide tables");
            // }

            let y_wit = match lookup.elements.len() {
                1 => {
                    let y = witness_layout.next_lookups_data();
                    let mut b = vec![(alpha, ark_bn254::Fr::ONE)];
                    for (w, coeff) in lookup.elements[0].iter() {
                        b.push((*w, -*coeff));
                    }
                    // y * (α - key) = flag
                    result.push(R1C {
                        a: vec![(y, ark_bn254::Fr::ONE)],
                        b,
                        c: lookup.flag.clone(),
                    });
                    y
                }
                2 => {
                    let x = witness_layout.next_lookups_data();
                    let y = witness_layout.next_lookups_data();
                    // β * value = -x  (defines x = -β*value)
                    result.push(R1C {
                        a: vec![(beta, crate::compiler::Field::ONE)],
                        b: lookup.elements[1].clone(),
                        c: vec![(x, -crate::compiler::Field::ONE)],
                    });

                    // y * (α - x - key) = flag
                    let mut b = vec![
                        (alpha, ark_bn254::Fr::ONE),
                        (x, -crate::compiler::Field::ONE),
                    ];
                    for (w, coeff) in lookup.elements[0].iter() {
                        b.push((*w, -*coeff));
                    }
                    result.push(R1C {
                        a: vec![(y, ark_bn254::Fr::ONE)],
                        b,
                        c: lookup.flag.clone(),
                    });
                    y
                }
                _ => panic!("unsupported lookup width {}", lookup.elements.len()),
            };

            result[table_infos[lookup.table_id].sum_constraint_idx]
                .c
                .push((y_wit, ark_bn254::Fr::ONE));
        }

        constraints_layout.lookups_data_size =
            result.len() - constraints_layout.algebraic_size - constraints_layout.tables_data_size;

        return R1CS {
            witness_layout,
            constraints_layout,
            constraints: result,
        };
    }
}
