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
pub struct ArrayData {
    table_id: Option<usize>,
    data: Vec<Value>,
}

#[derive(Clone, Debug)]
pub struct TupleData {
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
            (_, Value::Const(rhs)) => {
                let inv = Value::Const(ark_bn254::Fr::ONE / rhs);
                self.mul(&inv)
            }
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
        Value::Tuple(Rc::new(RefCell::new(TupleData { data: fields })))
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
    /// N-dimensional lookup table.
    ///
    /// `values` holds the array's leaves in row-major flat order
    /// (`values[i_1 * d_2 * .. * d_N + .. + i_N]`).
    /// `dims = [d_1, d_2, .., d_N]` records the original shape so the
    /// per-slot key coordinates can be reconstructed at constraint-emission
    /// time. The β-power LogUp denominator at slot `s` is
    /// `α − value(s) − β·i_1(s) − β²·i_2(s) − … − β^N·i_N(s)`.
    OfElems {
        values: Vec<LC>,
        dims: Vec<usize>,
    },
    Spread(u8),
}

/// Flatten an N-D nested `Value::Array` into row-major leaves + shape.
///
/// Walks down through `Value::Array` nodes while every level is itself an
/// array; the first level whose leaves aren't arrays defines the leaf type.
fn flatten_nd_array(arr: &Rc<RefCell<ArrayData>>) -> (Vec<Value>, Vec<usize>) {
    let mut dims = vec![arr.borrow().data.len()];
    // Probe the first leaf to discover further dims.
    let mut cursor = arr.borrow().data.first().cloned();
    while let Some(Value::Array(inner)) = cursor {
        let len = inner.borrow().data.len();
        dims.push(len);
        cursor = inner.borrow().data.first().cloned();
    }
    let mut flat = Vec::new();
    flatten_into(arr, &dims, 0, &mut flat);
    assert_eq!(
        flat.len(),
        dims.iter().product::<usize>(),
        "ICE: flattened length ({}) does not match product of dims ({:?})",
        flat.len(),
        dims
    );
    (flat, dims)
}

fn flatten_into(arr: &Rc<RefCell<ArrayData>>, dims: &[usize], level: usize, out: &mut Vec<Value>) {
    let borrowed = arr.borrow();
    assert_eq!(
        borrowed.data.len(),
        dims[level],
        "ICE: inconsistent N-D array shape at level {}",
        level
    );
    if level + 1 == dims.len() {
        for v in borrowed.data.iter() {
            out.push(v.clone());
        }
    } else {
        for v in borrowed.data.iter() {
            match v {
                Value::Array(inner) => flatten_into(inner, dims, level + 1, out),
                _ => panic!(
                    "ICE: expected nested Value::Array at level {} of N-D flatten",
                    level + 1
                ),
            }
        }
    }
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
                        Table::OfElems { .. } | Table::Spread(_) => panic!("unsupported"),
                    }
                }
                // Rangecheck has no value column — `elements` is the single
                // key whose β-power-0 slot doubles as the lookup target.
                let els = keys
                    .into_iter()
                    .chain(results)
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
                        Table::OfElems { .. } | Table::Spread(_) => panic!("unsupported"),
                    }
                }
                let els = keys
                    .into_iter()
                    .chain(results)
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
                // Convention: value (β⁰) first, then keys (β^j). For Spread,
                // `results = [spread_output]` and `keys = [spread_input]`.
                let els = results
                    .into_iter()
                    .chain(keys)
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
                    let (flat_values, dims) = flatten_nd_array(&arr);
                    let values: Vec<LC> = flat_values
                        .iter()
                        .map(|e| e.expect_linear_combination())
                        .collect();
                    self.tables.push(Table::OfElems { values, dims });
                    let idx = self.tables.len() - 1;
                    arr.borrow_mut().table_id = Some(idx);
                    idx
                } else {
                    arr.borrow().table_id.unwrap()
                };
                // Convention: value (β⁰) first, then keys k_1..k_N (β¹..β^N).
                let els = results
                    .into_iter()
                    .chain(keys)
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
    fn ult(&self, b: &Self, _ctx: &mut R1CGen) -> Self {
        self.lt(b)
    }

    fn slt(&self, b: &Self, bits: usize, _ctx: &mut R1CGen) -> Self {
        let a = Self::decode_signed(self.expect_u128(), bits);
        let b_val = Self::decode_signed(b.expect_u128(), bits);
        if a < b_val {
            Value::Const(ark_bn254::Fr::ONE)
        } else {
            Value::Const(ark_bn254::Fr::ZERO)
        }
    }

    fn eq(&self, b: &Self, _ctx: &mut R1CGen) -> Self {
        self.eq(b)
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

    fn assert_bool(&self, _ctx: &mut R1CGen) {
        let v = self.expect_constant();
        assert!(
            v != ark_bn254::Fr::from(0u64),
            "assert failed: value is zero"
        );
    }

    fn assert_cmp(kind: CmpKind, a: &Self, b: &Self, lhs_type: &Type, _ctx: &mut R1CGen) {
        match kind {
            CmpKind::Eq => {
                assert_eq!(a.expect_constant(), b.expect_constant());
            }
            CmpKind::Lt => {
                let a_val = a.expect_constant();
                let b_val = b.expect_constant();
                match &lhs_type.strip_witness().expr {
                    TypeExpr::I(bits) => {
                        // Signed comparison: interpret as two's complement
                        let a_int = a_val.into_bigint();
                        let b_int = b_val.into_bigint();
                        let half = ark_bn254::Fr::from(1u64 << (bits - 1)).into_bigint();
                        let a_neg = a_int >= half;
                        let b_neg = b_int >= half;
                        let result = match (a_neg, b_neg) {
                            (true, false) => true,  // negative < positive
                            (false, true) => false, // positive >= negative
                            _ => a_int < b_int,     // same sign: compare directly
                        };
                        assert!(
                            result,
                            "assert_cmp lt (signed) failed: {:?} >= {:?}",
                            a_val, b_val
                        );
                    }
                    _ => {
                        assert!(
                            a_val < b_val,
                            "assert_cmp lt failed: {:?} >= {:?}",
                            a_val,
                            b_val
                        );
                    }
                }
            }
        }
    }

    fn assert_r1c(a: &Self, b: &Self, c: &Self, _ctx: &mut R1CGen) {
        let a = a.expect_constant();
        let b = b.expect_constant();
        let c = c.expect_constant();
        assert!(a * b == c);
    }

    fn array_get(&self, index: &Self, _out_type: &Type, _ctx: &mut R1CGen) -> Self {
        let index = index.expect_u32();
        self.expect_array().borrow().data[index as usize].clone()
    }

    fn tuple_get(&self, index: usize, _out_type: &Type, _ctx: &mut R1CGen) -> Self {
        self.expect_tuple().borrow().data[index].clone()
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
        ptr.borrow().clone()
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
        true
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
        // `max_n_keys` is the largest N in any β-power-LogUp table or lookup
        // (number of β^1..β^N terms in the denominator). Rangechecks have N=0.
        let mut max_n_keys: usize = 0;
        for table in self.tables.into_iter() {
            let (len, n_keys) = match &table {
                Table::Range(bits) => (1usize << bits, 0usize),
                Table::OfElems { values, dims } => (values.len(), dims.len()),
                Table::Spread(bits) => (1usize << bits, 1usize),
            };
            table_infos.push(TableInfo {
                multiplicities_witness_off: witness_layout.multiplicities_size
                    + witness_layout.algebraic_size,
                table,
                sum_constraint_idx: 0,
            });
            max_n_keys = max_n_keys.max(n_keys);
            witness_layout.multiplicities_size += len;
        }
        // Lookups can request more keys than any table currently registered
        // (shouldn't happen — every lookup goes via a table — but be defensive).
        for lookup in &self.lookups {
            if lookup.elements.len() >= 2 {
                max_n_keys = max_n_keys.max(lookup.elements.len() - 1);
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

        // β and β-power chain. β^j (for j ≥ 2) is a derived witness with the
        // constraint β · β^{j-1} = β^j. Allocated up-front so per-slot/per-query
        // constraints can reference them as plain witness indices.
        // `beta_powers[j-1] = β^j` slot. Empty if `max_n_keys == 0` (no β at all).
        let mut beta_powers: Vec<usize> = Vec::with_capacity(max_n_keys);
        if max_n_keys >= 1 {
            let beta = witness_layout.challenges_end();
            witness_layout.challenges_size += 1;
            beta_powers.push(beta);
            for j in 2..=max_n_keys {
                let bp = witness_layout.next_table_data();
                // R1C: β · β^{j-1} = β^j
                result.push(R1C {
                    a: vec![(beta_powers[0], crate::compiler::Field::ONE)],
                    b: vec![(beta_powers[j - 2], crate::compiler::Field::ONE)],
                    c: vec![(bp, crate::compiler::Field::ONE)],
                });
                beta_powers.push(bp);
            }
        }
        // β^j (j ≥ 1) lookup; panics if j == 0 (β^0 = 1 is the constant slot 0).
        let beta_pow = |j: usize| -> usize {
            assert!(j >= 1, "β^0 is the constant slot, not a witness");
            beta_powers[j - 1]
        };

        // tables contents init
        for table_info in table_infos.iter_mut() {
            match &table_info.table {
                Table::Range(bits) => {
                    // For each i ∈ 0..2^bits: y · (α − i) = mᵢ
                    let len = 1usize << bits;
                    let mut sum_lhs: LC = vec![];
                    for i in 0..len {
                        let y = witness_layout.next_table_data();
                        let m = table_info.multiplicities_witness_off + i;
                        result.push(R1C {
                            a: vec![(y, crate::compiler::Field::ONE)],
                            b: vec![
                                (alpha, crate::compiler::Field::ONE),
                                (0, -crate::compiler::Field::from(i as u64)),
                            ],
                            c: vec![(m, crate::compiler::Field::ONE)],
                        });
                        sum_lhs.push((y, crate::compiler::Field::ONE));
                    }
                    result.push(R1C {
                        a: sum_lhs,
                        b: vec![(0, crate::compiler::Field::ONE)],
                        c: vec![], // prepared for the looked up values to come into
                    });
                    table_info.sum_constraint_idx = result.len() - 1;
                }
                Table::OfElems { values, dims } => {
                    // Row-major flat layout. For slot s with value v_s (LC) and
                    // coords (i_1(s),…,i_N(s)) where N = dims.len():
                    //   y_s · (α − v_s − β·i_1(s) − β²·i_2(s) − … − β^N·i_N(s)) = m_s
                    // All coords are slot-derived constants → the entire B-side
                    // is a flat LC, no per-slot β multiplications needed.
                    let n_keys = dims.len();
                    // suffix[j] = ∏ d_{k} for k=j..n_keys (product of dims after position j-1)
                    let mut suffix = vec![1usize; n_keys + 1];
                    for j in (0..n_keys).rev() {
                        suffix[j] = suffix[j + 1] * dims[j];
                    }
                    let mut sum_lhs: LC = vec![];
                    for (s, v) in values.iter().enumerate() {
                        let y = witness_layout.next_table_data();
                        let m = table_info.multiplicities_witness_off + s;
                        let mut b = vec![(alpha, crate::compiler::Field::ONE)];
                        // -v_s
                        for (w, coeff) in v.iter() {
                            b.push((*w, -*coeff));
                        }
                        // -β^j · i_j(s) for j in 1..=n_keys (coords as constants)
                        for j in 1..=n_keys {
                            let i_j = (s / suffix[j]) % dims[j - 1];
                            if i_j != 0 {
                                b.push((beta_pow(j), -crate::compiler::Field::from(i_j as u64)));
                            }
                        }
                        result.push(R1C {
                            a: vec![(y, crate::compiler::Field::ONE)],
                            b,
                            c: vec![(m, crate::compiler::Field::ONE)],
                        });
                        sum_lhs.push((y, crate::compiler::Field::ONE));
                    }
                    result.push(R1C {
                        a: sum_lhs,
                        b: vec![(0, crate::compiler::Field::ONE)],
                        c: vec![],
                    });
                    table_info.sum_constraint_idx = result.len() - 1;
                }
                Table::Spread(bits) => {
                    // 2-column table: value column (β⁰) = spread(i), key column (β¹) = i.
                    // Per slot: y · (α − spread(i) − β·i) = m_i. Both terms constant.
                    let len = 1usize << bits;
                    let mut sum_lhs: LC = vec![];
                    for i in 0..len {
                        let spread_val = ssa_mod::spread_bits(i as u128, 32);
                        let y = witness_layout.next_table_data();
                        let m = table_info.multiplicities_witness_off + i;
                        let mut b = vec![
                            (alpha, crate::compiler::Field::ONE),
                            (0, -crate::compiler::Field::from(spread_val)),
                        ];
                        if i != 0 {
                            b.push((beta_pow(1), -crate::compiler::Field::from(i as u64)));
                        }
                        result.push(R1C {
                            a: vec![(y, crate::compiler::Field::ONE)],
                            b,
                            c: vec![(m, crate::compiler::Field::ONE)],
                        });
                        sum_lhs.push((y, crate::compiler::Field::ONE));
                    }
                    result.push(R1C {
                        a: sum_lhs,
                        b: vec![(0, crate::compiler::Field::ONE)],
                        c: vec![],
                    });
                    table_info.sum_constraint_idx = result.len() - 1;
                }
            }
        }

        constraints_layout.tables_data_size = result.len() - constraints_layout.algebraic_size;

        // lookups init
        for lookup in self.lookups.into_iter() {
            let n_elems = lookup.elements.len();
            assert!(n_elems >= 1, "lookup must have at least one element");
            let y_wit = if n_elems == 1 {
                // Rangecheck-style: denom = α − v  (where v is the value being checked).
                let y = witness_layout.next_lookups_data();
                let mut b = vec![(alpha, crate::compiler::Field::ONE)];
                for (w, coeff) in lookup.elements[0].iter() {
                    b.push((*w, -*coeff));
                }
                result.push(R1C {
                    a: vec![(y, crate::compiler::Field::ONE)],
                    b,
                    c: lookup.flag.clone(),
                });
                y
            } else {
                // Value at β⁰ = elements[0]; keys at β^j = elements[j], 1 ≤ j ≤ n_keys.
                // For each key: aux x_j = β^j · k_j (R1C), then
                //   y · (α − value − x_1 − … − x_{n_keys}) = flag.
                let n_keys = n_elems - 1;
                let mut xs = Vec::with_capacity(n_keys);
                for j in 1..=n_keys {
                    let x = witness_layout.next_lookups_data();
                    result.push(R1C {
                        a: vec![(beta_pow(j), crate::compiler::Field::ONE)],
                        b: lookup.elements[j].clone(),
                        c: vec![(x, crate::compiler::Field::ONE)],
                    });
                    xs.push(x);
                }
                let y = witness_layout.next_lookups_data();
                let mut b = vec![(alpha, crate::compiler::Field::ONE)];
                for (w, coeff) in lookup.elements[0].iter() {
                    b.push((*w, -*coeff));
                }
                for x in xs {
                    b.push((x, -crate::compiler::Field::ONE));
                }
                result.push(R1C {
                    a: vec![(y, crate::compiler::Field::ONE)],
                    b,
                    c: lookup.flag.clone(),
                });
                y
            };

            result[table_infos[lookup.table_id].sum_constraint_idx]
                .c
                .push((y_wit, crate::compiler::Field::ONE));
        }

        constraints_layout.lookups_data_size =
            result.len() - constraints_layout.algebraic_size - constraints_layout.tables_data_size;

        R1CS {
            witness_layout,
            constraints_layout,
            constraints: result,
        }
    }
}
