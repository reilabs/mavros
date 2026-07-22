use std::{cell::RefCell, collections::BTreeMap, rc::Rc};

use crate::compiler::{
    analysis::{
        symbolic_executor::{self, AssertionFailure, SymbolicExecutor},
        types::TypeInfo,
    },
    ssa::{
        BlockId, FunctionId,
        hlssa::{
            self, BinaryArithOpKind, CmpKind, HLSSA, MAX_SUPPORTED_SIGNED_BITS,
            MAX_SUPPORTED_UNSIGNED_BITS, Radix, RefCountOp, SliceOpDir, Type, TypeExpr,
        },
    },
    util::{spread_bits, unspread_bits},
};
use ark_ff::{AdditiveGroup, BigInt, BigInteger, Field, PrimeField};
use tracing::{instrument, warn};

pub use mavros_artifacts::{ConstraintsLayout, FlamegraphProfile, LC, R1C, R1CS, WitnessLayout};

/// Per-function circuit-size profiles produced alongside the R1CS.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct R1CSProfile {
    pub constraints: FlamegraphProfile,
    pub witnesses: FlamegraphProfile,
}

// FIELD-ASSUMPTION: L1-direct-ref (57 sites)
// FIELD-ASSUMPTION: L4-two-pow
fn two_pow(exponent: usize) -> ark_bn254::Fr {
    ark_bn254::Fr::from(2).pow([exponent as u64])
}

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
// FIELD-ASSUMPTION: L4-eval
pub enum Value {
    Const(ark_bn254::Fr),
    LC(LC),
    Array(Rc<RefCell<ArrayData>>),
    Blob(Vec<Value>),
    Ptr(Rc<RefCell<Value>>),
    Invalid,
}

impl Value {
    fn bit_mask(bits: usize) -> u128 {
        assert!(
            (1..=MAX_SUPPORTED_UNSIGNED_BITS).contains(&bits),
            "invalid integer width for bit mask: {bits}"
        );
        if bits == MAX_SUPPORTED_UNSIGNED_BITS {
            u128::MAX
        } else {
            (1u128 << bits) - 1
        }
    }

    fn wrap_unsigned(v: u128, bits: usize) -> u128 {
        v & Self::bit_mask(bits)
    }

    fn decode_signed(v: u128, bits: usize) -> i128 {
        assert!(
            bits > 0 && bits <= MAX_SUPPORTED_SIGNED_BITS,
            "signed integers wider than i{MAX_SUPPORTED_SIGNED_BITS} are unsupported"
        );
        let masked = Self::wrap_unsigned(v, bits);
        let sign_bit = 1u128 << (bits - 1);
        if masked & sign_bit == 0 {
            masked as i128
        } else {
            (masked as i128) - ((1u128 << bits) as i128)
        }
    }

    fn encode_signed(v: i128, bits: usize) -> u128 {
        assert!(
            bits <= MAX_SUPPORTED_SIGNED_BITS,
            "signed integers wider than i{MAX_SUPPORTED_SIGNED_BITS} are unsupported"
        );
        Self::wrap_unsigned(v as u128, bits)
    }

    // FIELD-ASSUMPTION: L4-eval
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
            // FIELD-ASSUMPTION: L4-inverse
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

    // FIELD-ASSUMPTION: L4-eval
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

    pub fn expect_blob(&self) -> Vec<Value> {
        match self {
            Value::Blob(elements) => elements.clone(),
            _ => panic!("expected blob"),
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
}

fn flatten_array_into_table(arr: &ArrayData, out: &mut Vec<LC>) {
    for elem in arr.data.iter() {
        match elem {
            Value::Array(inner) => flatten_array_into_table(&inner.borrow(), out),
            _ => out.push(elem.expect_linear_combination()),
        }
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
    function_names: BTreeMap<FunctionId, String>,
    call_stack: Vec<String>,
    profile_root: String,
    profile: R1CSProfile,
}

impl symbolic_executor::Context<Value> for R1CGen {
    fn on_call(
        &mut self,
        func: FunctionId,
        _params: &mut [Value],
        _param_types: &[&Type],
        _result_types: &[Type],
        unconstrained: bool,
    ) -> Option<Vec<Value>> {
        assert!(
            !unconstrained,
            "ICE: unconstrained calls should be DCE'd before R1CS gen"
        );
        self.call_stack.push(
            self.function_names
                .get(&func)
                .cloned()
                .unwrap_or_else(|| format!("fn{}", func.0)),
        );
        None
    }

    fn on_return(&mut self, _returns: &mut [Value], _return_types: &[Type]) {
        self.call_stack
            .pop()
            .expect("ICE: R1CS profiler call stack underflow");
    }

    fn on_jmp(&mut self, _target: BlockId, _params: &mut [Value], _param_types: &[&Type]) {}

    fn lookup(&mut self, target: hlssa::LookupTarget<Value>, args: Vec<Value>, flag: Value) {
        let flag_lc = flag.expect_linear_combination();
        let els: Vec<_> = args
            .into_iter()
            .map(|e| e.expect_linear_combination())
            .collect();
        match target {
            hlssa::LookupTarget::Rangecheck(i) => {
                // Find or create the rangecheck table of this size. The lookup-sizing analysis
                // may select several distinct sizes, so multiple range tables can coexist.
                let existing_table_count = self.tables.len();
                let table_id = self.find_or_create_range_table(i as u64);
                if self.tables.len() != existing_table_count {
                    let length = 1u64 << i;
                    self.record_constraints(length + 1);
                    self.record_witnesses(2 * length);
                }
                self.record_constraints(1);
                self.record_witnesses(1);
                self.lookups.push(LookupConstraint {
                    table_id,
                    elements: els,
                    flag: flag_lc.clone(),
                });
            }
            hlssa::LookupTarget::DynRangecheck(_) => {
                // `to_radix` lowers its (asserted radix-256) digit checks to static 8-bit
                // rangechecks, so no `DynRangecheck` survives to R1CS generation.
                unreachable!(
                    "DynRangecheck is lowered to a static 8-bit rangecheck before R1CS gen"
                )
            }
            hlssa::LookupTarget::Spread(bits) => {
                let table_id = {
                    let existing = self.tables.iter().position(|t| match t {
                        Table::Spread(n) => *n == bits,
                        _ => false,
                    });
                    if let Some(idx) = existing {
                        idx
                    } else {
                        self.tables.push(Table::Spread(bits));
                        let length = 1u64 << bits;
                        self.record_constraints(length + 1);
                        self.record_witnesses(2 * length);
                        self.tables.len() - 1
                    }
                };
                self.record_constraints(2);
                self.record_witnesses(2);
                self.lookups.push(LookupConstraint {
                    table_id,
                    elements: els,
                    flag: flag_lc,
                });
            }
            hlssa::LookupTarget::Array(arr) => {
                let arr = arr.expect_array();
                let table_id = if arr.borrow().table_id.is_none() {
                    let mut elems = Vec::new();
                    flatten_array_into_table(&arr.borrow(), &mut elems);
                    let length = elems.len() as u64;
                    self.tables.push(Table::OfElems(elems));
                    self.record_constraints(2 * length + 1);
                    self.record_witnesses(3 * length);
                    let idx = self.tables.len() - 1;
                    arr.borrow_mut().table_id = Some(idx);
                    idx
                } else {
                    arr.borrow().table_id.unwrap()
                };
                self.record_constraints(2);
                self.record_witnesses(2);
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
        _inner: &crate::compiler::ssa::hlssa::OpCode,
        _condition: &Value,
        _inputs: Vec<&Value>,
        _result_types: Vec<&Type>,
    ) -> Vec<Value> {
        panic!("ICE: Guard should not appear in R1CS gen (should be lowered before)")
    }
}

// FIELD-ASSUMPTION: L4-eval
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
                    *bits > 0 && *bits <= MAX_SUPPORTED_UNSIGNED_BITS,
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
                    *bits > 0 && *bits <= MAX_SUPPORTED_SIGNED_BITS,
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
                    BinaryArithOpKind::Div => Self::encode_signed(a / b, *bits),
                    BinaryArithOpKind::Mod => Self::encode_signed(a % b, *bits),
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

    fn assert_bool(&self, _ctx: &mut R1CGen) -> Result<(), AssertionFailure> {
        let v = self.expect_constant();
        if v == ark_bn254::Fr::from(0u64) {
            return Err(AssertionFailure::new("assert failed: value is zero"));
        }
        Ok(())
    }

    fn assert_cmp(
        kind: CmpKind,
        a: &Self,
        b: &Self,
        lhs_type: &Type,
        _ctx: &mut R1CGen,
    ) -> Result<(), AssertionFailure> {
        match kind {
            CmpKind::Eq => {
                let a_val = a.expect_constant();
                let b_val = b.expect_constant();
                if a_val != b_val {
                    return Err(AssertionFailure::new(format!(
                        "assert_cmp eq failed: {a_val:?} != {b_val:?}"
                    )));
                }
            }
            // FIELD-ASSUMPTION: L4-sign
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
                        if !result {
                            return Err(AssertionFailure::new(format!(
                                "assert_cmp lt (signed) failed: {a_val:?} >= {b_val:?}"
                            )));
                        }
                    }
                    _ => {
                        if a_val >= b_val {
                            return Err(AssertionFailure::new(format!(
                                "assert_cmp lt failed: {a_val:?} >= {b_val:?}"
                            )));
                        }
                    }
                }
            }
        }
        Ok(())
    }

    fn assert_r1c(a: &Self, b: &Self, c: &Self, _ctx: &mut R1CGen) -> Result<(), AssertionFailure> {
        let a = a.expect_constant();
        let b = b.expect_constant();
        let c = c.expect_constant();
        if a * b != c {
            return Err(AssertionFailure::new(format!(
                "assert_r1c failed: {a:?} * {b:?} != {c:?}"
            )));
        }
        Ok(())
    }

    fn array_get(&self, index: &Self, _out_type: &Type, _ctx: &mut R1CGen) -> Self {
        let index = index.expect_u32();
        self.expect_array().borrow().data[index as usize].clone()
    }

    fn array_set(&self, index: &Self, value: &Self, _out_type: &Type, _ctx: &mut R1CGen) -> Self {
        let array = self.expect_array();
        let index = index.expect_u32();
        let mut new_array = array.borrow().data.clone();
        new_array[index as usize] = value.clone();
        Value::mk_array(new_array)
    }

    // FIELD-ASSUMPTION: L4-decompose
    fn bit_range(&self, offset: usize, width: usize, _out_type: &Type, _ctx: &mut R1CGen) -> Self {
        let new_value = self
            .expect_constant()
            .into_bigint()
            .to_bits_le()
            .iter()
            .skip(offset)
            .take(width)
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
            let extension = two_pow(_to) - two_pow(from);
            Value::Const(val + extension)
        } else {
            self.clone()
        }
    }

    fn cast(&self, cast_target: &hlssa::CastTarget, _out_type: &Type, _ctx: &mut R1CGen) -> Self {
        // Witness strips (ValueOf, also under Maps) only feed hint chains and
        // unconstrained call arguments, so they must be dead (and DCE'd) by
        // R1CS generation. The remaining casts — witness injections and Maps
        // thereof included — don't change the symbolic value.
        assert!(
            !cast_target.is_value_of(),
            "ICE: witness strip {cast_target} should not reach R1CS gen"
        );
        self.clone()
    }

    fn constrain(a: &Self, b: &Self, c: &Self, ctx: &mut R1CGen) -> Result<(), AssertionFailure> {
        let a = a.expect_linear_combination();
        let b = b.expect_linear_combination();
        let c = c.expect_linear_combination();
        ctx.constraints.push(R1C { a, b, c });
        ctx.record_constraints(1);
        Ok(())
    }

    fn to_bits(
        &self,
        endianness: hlssa::Endianness,
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
            crate::compiler::ssa::hlssa::Endianness::Little => bits,
            crate::compiler::ssa::hlssa::Endianness::Big => {
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

    fn of_field(f: crate::compiler::Field, _ctx: &mut R1CGen) -> Self {
        Value::Const(ark_bn254::Fr::from(f))
    }

    fn of_blob(_elem_type: Type, elements: Vec<Self>, _ctx: &mut R1CGen) -> Self {
        Value::Blob(elements)
    }

    fn expect_blob(&self, _ctx: &mut R1CGen) -> Vec<Self> {
        self.expect_blob()
    }

    fn mk_array(
        a: Vec<Self>,
        _ctx: &mut R1CGen,
        _seq_type: hlssa::SequenceTargetType,
        _elem_type: &Type,
    ) -> Self {
        Value::mk_array(a)
    }

    fn alloc(value: &Self, _ctx: &mut R1CGen) -> Self {
        Value::Ptr(Rc::new(RefCell::new(value.clone())))
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
        ctx.record_witnesses(1);
        Value::LC(vec![(witness_var, ark_bn254::Fr::ONE)])
    }

    fn fresh_witness(_result_type: &Type, ctx: &mut R1CGen) -> Self {
        let witness_var = ctx.next_witness();
        ctx.record_witnesses(1);
        Value::LC(vec![(witness_var, ark_bn254::Fr::ONE)])
    }

    fn mem_op(&self, _kind: RefCountOp, _ctx: &mut R1CGen) {}

    // FIELD-ASSUMPTION: L4-decompose
    fn rangecheck(&self, max_bits: usize, _ctx: &mut R1CGen) -> Result<(), AssertionFailure> {
        let self_const = self.expect_constant();
        let check = self_const
            .into_bigint()
            .to_bits_le()
            .iter()
            .skip(max_bits)
            .all(|b| !b);
        if !check {
            return Err(AssertionFailure::new(format!(
                "rangecheck failed: {self_const:?} does not fit in {max_bits} bits"
            )));
        }
        Ok(())
    }

    fn to_radix(
        &self,
        radix: &Radix<Self>,
        endianness: crate::compiler::ssa::hlssa::Endianness,
        size: usize,
        _out_type: &Type,
        _ctx: &mut R1CGen,
    ) -> Self {
        match radix {
            Radix::Bytes => {
                let mut bytes = self.expect_constant().into_bigint().to_bytes_le();
                if bytes.len() > size {
                    bytes.truncate(size);
                } else {
                    bytes.resize(size, 0);
                }
                if matches!(endianness, crate::compiler::ssa::hlssa::Endianness::Big) {
                    bytes.reverse();
                }
                Value::mk_array(
                    bytes
                        .into_iter()
                        .map(|byte| Value::Const(ark_bn254::Fr::from(byte)))
                        .collect(),
                )
            }
            Radix::Dyn(_) => todo!("dynamic ToRadix R1CS generation not yet implemented"),
        }
    }

    fn spread(&self, bits: u8, _ctx: &mut R1CGen) -> Self {
        let val = self.expect_constant();
        let v: u128 = val.into_bigint().as_ref()[0] as u128;
        let spread_val = spread_bits(v, bits as usize);
        Value::Const(ark_bn254::Fr::from(spread_val))
    }

    fn unspread(&self, bits: u8, _ctx: &mut R1CGen) -> (Self, Self) {
        let val = self.expect_constant();
        let v: u128 = val.into_bigint().as_ref()[0] as u128;
        let (odd_val, even_val) = unspread_bits(v, bits as usize * 2);
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
            function_names: BTreeMap::new(),
            call_stack: Vec::new(),
            profile_root: "<r1cs>".to_string(),
            profile: R1CSProfile::default(),
        }
    }

    fn record_constraints(&mut self, count: u64) {
        self.profile
            .constraints
            .record(self.call_stack.iter().cloned(), count);
    }

    fn record_witnesses(&mut self, count: u64) {
        self.profile
            .witnesses
            .record(self.call_stack.iter().cloned(), count);
    }

    /// Return the id of the rangecheck table for `bits`-bit values (i.e. `2^bits` rows), creating
    /// it if absent.
    fn find_or_create_range_table(&mut self, bits: u64) -> usize {
        if let Some(idx) = self
            .tables
            .iter()
            .position(|t| matches!(t, Table::Range(b) if *b == bits))
        {
            idx
        } else {
            self.tables.push(Table::Range(bits));
            self.tables.len() - 1
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
    pub fn run(&mut self, ssa: &HLSSA, type_info: &TypeInfo) -> Result<(), AssertionFailure> {
        self.function_names = ssa
            .iter_functions()
            .map(|(id, function)| (*id, function.get_name().to_string()))
            .collect();
        let entry_point = ssa.get_unique_entrypoint_id();
        self.profile_root = ssa.get_function(entry_point).get_name().to_string();
        assert!(
            ssa.get_function(entry_point).get_param_types().len() == 0,
            "Main should not have parameters as WitnessWriteToFresh pass should remove them"
        );
        let main_params = vec![];
        let executor = SymbolicExecutor::new();
        let result = executor.run(ssa, type_info, entry_point, main_params, self);
        debug_assert!(result.is_err() || self.call_stack.is_empty());
        result
    }

    pub fn get_r1cs(self) -> Vec<R1C> {
        self.constraints
    }

    pub fn get_witness_size(&self) -> usize {
        self.next_witness
    }

    /// Number of lookup query sites — one term per site on the lookup side of the LogUp
    /// identity. Together with the table-entry count this is the argument's soundness degree
    /// `D`. Must be read before [`R1CGen::seal`] consumes `self`.
    pub fn num_lookups(&self) -> usize {
        self.lookups.len()
    }

    fn next_witness(&mut self) -> usize {
        let result = self.next_witness;
        self.next_witness += 1;
        result
    }

    pub fn seal(self) -> R1CS {
        self.seal_with_profile().0
    }

    pub fn seal_with_profile(mut self) -> (R1CS, R1CSProfile) {
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
            let r1cs = R1CS {
                witness_layout,
                constraints_layout,
                constraints: result,
            };
            debug_assert_eq!(
                self.profile.constraints.total_weight(),
                r1cs.constraints_layout.size() as u64
            );
            debug_assert_eq!(
                self.profile.witnesses.total_weight(),
                r1cs.witness_layout.size() as u64
            );
            return (r1cs, self.profile);
        }

        // challenges init
        // FIELD-ASSUMPTION: L4-logup-challenges
        // One `alpha` (+ an optional column-folding `beta`) gives ~log2(p) bits of LogUp
        // soundness — sound on bn254, but only ~log2(p)/1 on a small field. A goldilocks
        // target needs K independent (alpha, beta) pairs here (see docs/field-agnosticism.md).
        let alpha = witness_layout.challenges_end();
        witness_layout.challenges_size += 1;
        let beta = if max_width > 1 {
            let beta = witness_layout.challenges_end();
            witness_layout.challenges_size += 1;
            beta
        } else {
            usize::MAX // hoping this crashes soon if used
        };
        self.profile.witnesses.record(
            [self.profile_root.clone(), "<lookup challenges>".to_string()],
            witness_layout.challenges_size as u64,
        );

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
                    // Spread table: for each entry i in 0..2^bits, value = spread(i).
                    // Both operands (key=i, value=spread(i)) are compile-time
                    // constants, so the `x = β·spread(i)` intermediate of the
                    // generic key-value table (`OfElems`) collapses: β·spread(i)
                    // is linear in the witness and folds directly into the
                    // denominator. One witness/constraint per entry instead of two:
                    // -> y = mᵢ / (α - i + β·spread(i)),
                    //    constraint `y · (α - i + β·spread(i)) - mᵢ = 0`
                    let len = 1usize << bits;
                    let mut sum_lhs: LC = vec![];
                    for i in 0..len {
                        let spread_val = spread_bits(i as u128, 32);
                        let y = witness_layout.next_table_data();
                        let m = table_info.multiplicities_witness_off + i;
                        result.push(R1C {
                            a: vec![(y, ark_bn254::Fr::ONE)],
                            b: vec![
                                (alpha, ark_bn254::Fr::ONE),
                                (0, -crate::compiler::Field::from(i as u64)),
                                (beta, crate::compiler::Field::from(spread_val)),
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

        let r1cs = R1CS {
            witness_layout,
            constraints_layout,
            constraints: result,
        };
        debug_assert_eq!(
            self.profile.constraints.total_weight(),
            r1cs.constraints_layout.size() as u64
        );
        debug_assert_eq!(
            self.profile.witnesses.total_weight(),
            r1cs.witness_layout.size() as u64
        );
        (r1cs, self.profile)
    }
}

// LOG-UP SOUNDNESS REPORTING AND ESTIMATION
// ================================================================================================

/// The outcome of sizing the LogUp lookup argument for a requested bits-of-security
/// target. See `docs/field-agnosticism.md` (`L4-logup-challenges`) for the model.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SoundnessReport {
    /// The requested bits of security (from `--logup-soundness`).
    pub requested_bits: u32,

    /// `floor(log2 p)` for the working field — a lower bound on `log2|F|`.
    pub field_bits: u32,

    /// `D = table_entries + num_lookups`, the degree of the cleared-denominator polynomial.
    pub soundness_degree: usize,

    /// Bits of security a single challenge buys: `floor(log2 p) - ceil(log2 D)`.
    pub per_challenge_bits: u32,

    /// `K = ceil(requested_bits / per_challenge_bits)` — the minimum number of challenges.
    pub challenges: u32,

    /// `K * per_challenge_bits` — a lower bound on the security actually delivered.
    pub achieved_bits: u32,

    /// `Some((k, bits))` when the request forces a nearly-empty extra challenge: dropping to
    /// `k = challenges - 1` challenges would still deliver `bits` bits, because the request sits in
    /// the lower half of the final challenge's contribution.
    pub near_optimal_alternative: Option<(u32, u32)>,
}

impl SoundnessReport {
    /// Diagnostic emitted when `challenges > 1` but K-challenge replication is not yet
    /// implemented, so the requested target cannot be honoured on this field.
    pub fn unsupported_message(&self) -> String {
        let mut msg = format!(
            "--logup-soundness={} bits needs {} LogUp challenges on this field \
             (~{} bits each; argument degree D = {}), but multi-challenge LogUp is not yet \
             implemented \u{2014} a single challenge provides only ~{} bits",
            self.requested_bits,
            self.challenges,
            self.per_challenge_bits,
            self.soundness_degree,
            self.per_challenge_bits,
        );
        if let Some((k, bits)) = self.near_optimal_alternative {
            msg.push_str(&format!(
                " (note: lowering --logup-soundness to {bits} would need only {k} challenge(s), \
                 shedding a nearly-empty extra challenge)"
            ));
        }
        msg
    }
}

/// `ceil(log2 n)` for `n >= 1`, computed with integer bit-length arithmetic (never `f64`,
/// so the emitted circuit stays bit-reproducible across platforms). `n <= 1` -> 0.
fn ceil_log2(n: usize) -> u32 {
    if n <= 1 {
        0
    } else {
        usize::BITS - (n - 1).leading_zeros()
    }
}

/// Compute the LogUp challenge count for the working field's bit size.
///
/// FIELD-ASSUMPTION: L4-logup-challenges
///
/// Reads `MODULUS_BIT_SIZE` of the concrete field to size per-challenge soundness. In P3 this
/// should read `FieldConfig::field_bit_size()` rather than the bn254 alias.
pub fn logup_soundness_report(
    requested_bits: u32,
    degree: usize,
) -> Result<SoundnessReport, String> {
    // floor(log2 p): p in [2^(MODULUS_BIT_SIZE-1), 2^MODULUS_BIT_SIZE), so floor(log2 p) =
    // MODULUS_BIT_SIZE - 1. Using the floor (rather than the bit size) keeps `achieved_bits`
    // a lower bound and never over-claims security.
    let field_bits = <crate::compiler::Field as PrimeField>::MODULUS_BIT_SIZE - 1;
    compute_logup_soundness(requested_bits, field_bits, degree)
}

/// Pure LogUp soundness computation (unit-tested with synthetic `field_bits`).
///
/// LogUp proves a log-derivative rational identity at a random challenge; clearing denominators
/// yields a nonzero polynomial of total degree `<= D = table_entries + num_lookups`, so one
/// challenge fails with probability `<= D/|F|` (Schwartz-Zippel). K independent challenges give
/// `(D/|F|)^K`, i.e. `K * (log2|F| - log2 D)` bits. All rounding is toward *under*-estimating
/// security (floor `log2 p`, ceil `log2 D`), so we never report more bits than are actually
/// delivered.
fn compute_logup_soundness(
    requested_bits: u32,
    field_bits: u32,
    degree: usize,
) -> Result<SoundnessReport, String> {
    let d_bits = ceil_log2(degree);
    if field_bits <= d_bits {
        return Err(format!(
            "LogUp soundness: field is too small \u{2014} floor(log2 p) = {field_bits} bits is not \
             larger than ceil(log2 D) = {d_bits} bits for a lookup argument of degree D = {degree}; \
             a single challenge distinguishes < 1 bit, so no number of challenges recovers soundness"
        ));
    }
    let per_challenge_bits = field_bits - d_bits;
    let challenges = requested_bits.div_ceil(per_challenge_bits).max(1);
    let achieved_bits = challenges * per_challenge_bits;

    // Near-optimal: the request lands in the lower half of the final challenge's
    // contribution, so it buys a whole extra challenge for less than half its worth.
    let near_optimal_alternative = if challenges >= 2 {
        let prev = (challenges - 1) * per_challenge_bits;
        let gap = requested_bits.saturating_sub(prev);
        if gap > 0 && gap <= per_challenge_bits / 2 {
            Some((challenges - 1, prev))
        } else {
            None
        }
    } else {
        None
    };

    Ok(SoundnessReport {
        requested_bits,
        field_bits,
        soundness_degree: degree,
        per_challenge_bits,
        challenges,
        achieved_bits,
        near_optimal_alternative,
    })
}

#[cfg(test)]
mod logup_soundness_tests {
    use super::{compute_logup_soundness, logup_soundness_report};

    #[test]
    fn bn254_needs_a_single_challenge_for_realistic_targets() {
        // bn254: floor(log2 p) = 253. A large circuit (D = 2^24 -> ceil(log2 D) = 24)
        // still buys 229 bits/challenge, so every sane request is a single challenge.
        let field_bits = 253;
        let degree = 1 << 24;
        for requested in [1, 80, 128, 223, 229] {
            let r = compute_logup_soundness(requested, field_bits, degree).unwrap();
            assert_eq!(
                r.challenges, 1,
                "requested {requested} should need 1 challenge"
            );
            assert_eq!(r.per_challenge_bits, 229);
            assert!(r.near_optimal_alternative.is_none());
        }
    }

    #[test]
    fn live_bn254_alias_yields_a_single_challenge_at_the_default() {
        // The real field alias (bn254) at the 128-bit default must be a genuine no-op.
        let r = logup_soundness_report(128, 1 << 20).unwrap();
        assert_eq!(r.field_bits, 253);
        assert_eq!(r.challenges, 1);
    }

    #[test]
    fn goldilocks_scales_challenges_with_the_target() {
        // Synthetic goldilocks: floor(log2 p) = 63, D ~= 2^22 -> ceil(log2 D) = 22, so
        // per-challenge = 41 bits (the sound floor of the informal ~42-bit estimate).
        let field_bits = 63;
        let degree = 1 << 22;
        let pc = 41;
        assert_eq!(
            compute_logup_soundness(pc, field_bits, degree)
                .unwrap()
                .per_challenge_bits,
            pc
        );
        // K = ceil(requested / 41).
        assert_eq!(
            compute_logup_soundness(41, field_bits, degree)
                .unwrap()
                .challenges,
            1
        );
        assert_eq!(
            compute_logup_soundness(42, field_bits, degree)
                .unwrap()
                .challenges,
            2
        );
        assert_eq!(
            compute_logup_soundness(123, field_bits, degree)
                .unwrap()
                .challenges,
            3
        );
        assert_eq!(
            compute_logup_soundness(124, field_bits, degree)
                .unwrap()
                .challenges,
            4
        );
        let r128 = compute_logup_soundness(128, field_bits, degree).unwrap();
        assert_eq!(r128.challenges, 4);
        assert_eq!(r128.achieved_bits, 164);
    }

    #[test]
    fn near_optimal_fires_just_above_a_challenge_boundary() {
        let field_bits = 63;
        let degree = 1 << 22; // per-challenge = 41
        // 128 is 5 above the 3-challenge mark (123); 5 <= 41/2, so warn and suggest 123/K=3.
        let r = compute_logup_soundness(128, field_bits, degree).unwrap();
        assert_eq!(r.near_optimal_alternative, Some((3, 123)));
        // 145 is 22 above the 3-challenge mark; 22 > 41/2 = 20, so no suggestion.
        let r = compute_logup_soundness(145, field_bits, degree).unwrap();
        assert_eq!(r.challenges, 4);
        assert!(r.near_optimal_alternative.is_none());
        // Exactly on a boundary (123 = 3*41) is a single-challenge overshoot of 0 -> K=3, none.
        let r = compute_logup_soundness(123, field_bits, degree).unwrap();
        assert!(r.near_optimal_alternative.is_none());
    }

    #[test]
    fn degenerate_when_degree_exceeds_the_field() {
        // D >= |F|: a single evaluation point can't distinguish the instance; hard error.
        assert!(compute_logup_soundness(128, 20, 1 << 22).is_err());
        // Boundary: field_bits == d_bits is still an error (per-challenge would be 0 bits).
        assert!(compute_logup_soundness(128, 22, 1 << 22).is_err());
        assert!(compute_logup_soundness(128, 23, 1 << 22).is_ok());
    }
}
