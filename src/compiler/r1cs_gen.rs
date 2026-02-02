use std::{cell::RefCell, fmt::Display, rc::Rc};

use crate::compiler::{
    analysis::{
        symbolic_executor::{self, SymbolicExecutor},
        types::TypeInfo,
    },
    ir::r#type::{CommutativeMonoid, Type, TypeExpr},
    ssa::{BinaryArithOpKind, BlockId, CmpKind, FunctionId, MemOp, Radix, SSA, SliceOpDir},
};
use ark_ff::{AdditiveGroup, BigInt, BigInteger, Field, PrimeField};
use tracing::{error, instrument, warn};

// #[derive(Clone, Debug, Copy, PartialEq, PartialOrd, Eq, Ord)]
// pub enum WitnessIndex {
//     PreCommitment(usize),
//     ChallengePower(usize, usize),
//     LookupValueInverse(usize),
//     LookupValueInverseAux(usize),
// }

type LC = Vec<(usize, crate::compiler::Field)>;

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
                let v: u128 = c.into_bigint().to_string().parse()
                    .unwrap_or_else(|e| panic!("expected u1, but field value is {}: {e}", c.into_bigint()));
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
                s.parse().unwrap_or_else(|e| panic!("expected u8, but field value is {s}: {e}"))
            }
            r => panic!("expected u8, got {:?}", r),
        }
    }

    pub fn expect_u32(&self) -> u32 {
        match self {
            Value::Const(c) => {
                let s = c.into_bigint().to_string();
                s.parse().unwrap_or_else(|e| panic!("expected u32, but field value is {s}: {e}"))
            }
            r => panic!("expected u32, got {:?}", r),
        }
    }

    pub fn expect_u64(&self) -> u64 {
        match self {
            Value::Const(c) => {
                let s = c.into_bigint().to_string();
                s.parse().unwrap_or_else(|e| panic!("expected u64, but field value is {s}: {e}"))
            }
            r => panic!("expected u64, got {:?}", r),
        }
    }

    pub fn expect_u128(&self) -> u128 {
        match self {
            Value::Const(c) => {
                let s = c.into_bigint().to_string();
                s.parse().unwrap_or_else(|e| panic!("expected u128, but field value is {s}: {e}"))
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
            data: fields 
        })))
    }
}

#[derive(Clone, Debug)]
pub struct R1C {
    pub a: LC,
    pub b: LC,
    pub c: LC,
}

#[derive(Clone, Debug)]
pub struct LookupConstraint {
    pub table_id: usize,
    pub elements: Vec<LC>,
}

#[derive(Clone, Debug)]
pub enum Table {
    Range(u64),
    OfElems(Vec<LC>),
}

fn field_to_string(c: ark_bn254::Fr) -> String {
    if c.into_bigint() > crate::compiler::Field::MODULUS_MINUS_ONE_DIV_TWO {
        format!("-{}", -c)
    } else {
        c.to_string()
    }
}

impl Display for R1C {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let a_str = self
            .a
            .iter()
            .map(|(i, c)| format!("{} * v{}", field_to_string(*c), i))
            .collect::<Vec<_>>()
            .join(" + ");
        let b_str = self
            .b
            .iter()
            .map(|(i, c)| format!("{} * v{}", field_to_string(*c), i))
            .collect::<Vec<_>>()
            .join(" + ");
        let c_str = self
            .c
            .iter()
            .map(|(i, c)| format!("{} * v{}", field_to_string(*c), i))
            .collect::<Vec<_>>()
            .join(" + ");

        write!(f, "({}) * ({}) - ({}) = 0", a_str, b_str, c_str)
    }
}

#[derive(Clone)]
pub struct R1CGen {
    constraints: Vec<R1C>,
    tables: Vec<Table>,
    lookups: Vec<LookupConstraint>,
    next_witness: usize,
}

impl<V: Clone> symbolic_executor::Context<Value, V> for R1CGen {
    fn on_call(
        &mut self,
        _func: FunctionId,
        _params: &mut [Value],
        _param_types: &[&Type<V>],
    ) -> Option<Vec<Value>> {
        None
    }

    fn on_return(&mut self, _returns: &mut [Value], _return_types: &[Type<V>]) {}

    fn on_jmp(&mut self, _target: BlockId, _params: &mut [Value], _param_types: &[&Type<V>]) {}

    fn lookup(
        &mut self,
        target: super::ssa::LookupTarget<Value>,
        keys: Vec<Value>,
        results: Vec<Value>,
    ) {
        match target {
            super::ssa::LookupTarget::Rangecheck(i) => {
                // TODO this will become table resolution logic eventually
                assert!(i == 8, "TODO: support other rangecheck sizes");
                if self.tables.is_empty() {
                    self.tables.push(Table::Range(i as u64))
                } else {
                    match self.tables[0] {
                        Table::Range(i1) => assert_eq!(i1, i as u64, "unsupported"),
                        Table::OfElems(_) => panic!("unsupported"),
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
                        Table::OfElems(_) => panic!("unsupported"),
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
                });
            }
        }
    }

    fn todo(&mut self, payload: &str, _result_types: &[Type<V>]) -> Vec<Value> {
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
}

impl<V: Clone> symbolic_executor::Value<R1CGen, V> for Value {
    fn cmp(&self, b: &Self, cmp_kind: CmpKind, _out_type: &Type<V>, _ctx: &mut R1CGen) -> Self {
        match cmp_kind {
            CmpKind::Eq => self.eq(b),
            CmpKind::Lt => self.lt(b),
        }
    }

    fn arith(
        &self,
        b: &Self,
        binary_arith_op_kind: BinaryArithOpKind,
        out_type: &Type<V>,
        _ctx: &mut R1CGen,
    ) -> Self {
        match &out_type.expr {
            TypeExpr::U(1) => {
                let a = self.expect_u1();
                let b = b.expect_u1();
                let result = match binary_arith_op_kind {
                    BinaryArithOpKind::Add => (a as u32) + (b as u32),
                    BinaryArithOpKind::Sub => (a as u32) - (b as u32),
                    BinaryArithOpKind::Mul => (a as u32) * (b as u32),
                    BinaryArithOpKind::Div => (a as u32) / (b as u32),
                    BinaryArithOpKind::And => (a & b) as u32,
                };
                Value::Const(ark_bn254::Fr::from(result))
            }
            TypeExpr::U(8) => {
                let a = self.expect_u8();
                let b = b.expect_u8();
                let result = match binary_arith_op_kind {
                    BinaryArithOpKind::Add => (a + b) as u32,
                    BinaryArithOpKind::Sub => (a - b) as u32,
                    BinaryArithOpKind::Mul => (a * b) as u32,
                    BinaryArithOpKind::Div => (a / b) as u32,
                    BinaryArithOpKind::And => (a & b) as u32,
                };
                Value::Const(ark_bn254::Fr::from(result))
            }
            TypeExpr::U(32) => {
                let a = self.expect_u32();
                let b = b.expect_u32();
                let result = match binary_arith_op_kind {
                    BinaryArithOpKind::Add => a + b,
                    BinaryArithOpKind::Sub => a - b,
                    BinaryArithOpKind::Mul => a * b,
                    BinaryArithOpKind::Div => a / b,
                    BinaryArithOpKind::And => a & b,
                };
                Value::Const(ark_bn254::Fr::from(result))
            }
            TypeExpr::U(64) => {
                let a = self.expect_u64();
                let b = b.expect_u64();
                let result = match binary_arith_op_kind {
                    BinaryArithOpKind::Add => a + b,
                    BinaryArithOpKind::Sub => a - b,
                    BinaryArithOpKind::Mul => a * b,
                    BinaryArithOpKind::Div => a / b,
                    BinaryArithOpKind::And => a & b,
                };
                Value::Const(ark_bn254::Fr::from(result))
            }
            TypeExpr::U(128) => {
                let a = self.expect_u128();
                let b = b.expect_u128();
                let result = match binary_arith_op_kind {
                    BinaryArithOpKind::Add => a + b,
                    BinaryArithOpKind::Sub => a - b,
                    BinaryArithOpKind::Mul => a * b,
                    BinaryArithOpKind::Div => a / b,
                    BinaryArithOpKind::And => a & b,
                };
                Value::Const(ark_bn254::Fr::from(result))
            }
            TypeExpr::U(size) => {
                panic!("Unsupported unsigned integer size in R1CS arith: u{size}")
            }
            TypeExpr::Field | TypeExpr::WitnessRef => match binary_arith_op_kind {
                BinaryArithOpKind::Add => self.add(b),
                BinaryArithOpKind::Sub => self.sub(b),
                BinaryArithOpKind::Mul => self.mul(b),
                BinaryArithOpKind::Div => self.div(b),
                BinaryArithOpKind::And => {
                    panic!("Bitwise AND is not supported on field elements")
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

    fn array_get(&self, index: &Self, _out_type: &Type<V>, _ctx: &mut R1CGen) -> Self {
        let index = index.expect_u32();
        let value = self.expect_array().borrow().data[index as usize].clone();
        value
    }

    fn tuple_get(&self, index: usize, _out_type: &Type<V>, _ctx: &mut R1CGen) -> Self {
        let value = self.expect_tuple().borrow().data[index as usize].clone();
        value
    }

    fn array_set(
        &self,
        index: &Self,
        value: &Self,
        _out_type: &Type<V>,
        _ctx: &mut R1CGen,
    ) -> Self {
        let array = self.expect_array();
        let index = index.expect_u32();
        let mut new_array = array.borrow().data.clone();
        new_array[index as usize] = value.clone();
        Value::mk_array(new_array)
    }

    fn truncate(&self, _from: usize, to: usize, _out_type: &Type<V>, _ctx: &mut R1CGen) -> Self {
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

    fn cast(
        &self,
        _cast_target: &super::ssa::CastTarget,
        _out_type: &Type<V>,
        _ctx: &mut R1CGen,
    ) -> Self {
        self.clone()
    }

    fn constrain(a: &Self, b: &Self, c: &Self, ctx: &mut R1CGen) {
        let a = a.expect_linear_combination();
        let b = b.expect_linear_combination();
        let c = c.expect_linear_combination();
        ctx.constraints.push(R1C { a: a, b: b, c: c });
    }

    fn to_bits(
        &self,
        endianness: super::ssa::Endianness,
        size: usize,
        _out_type: &Type<V>,
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

    fn not(&self, out_type: &Type<V>, _ctx: &mut R1CGen) -> Self {
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

    fn of_field(f: super::Field, _ctx: &mut R1CGen) -> Self {
        Value::Const(ark_bn254::Fr::from(f))
    }

    fn mk_array(
        a: Vec<Self>,
        _ctx: &mut R1CGen,
        _seq_type: super::ssa::SeqType,
        _elem_type: &Type<V>,
    ) -> Self {
        Value::mk_array(a)
    }

    fn mk_tuple(
        elems: Vec<Self>,
        _ctx: &mut R1CGen,
        _elem_types: &[Type<V>],
    ) -> Self {
        Value::mk_tuple(elems)
    }

    fn alloc(_ctx: &mut R1CGen) -> Self {
        Value::Ptr(Rc::new(RefCell::new(Value::Invalid)))
    }

    fn ptr_write(&self, value: &Self, _ctx: &mut R1CGen) {
        let ptr = self.expect_ptr();
        *ptr.borrow_mut() = value.clone();
    }

    fn ptr_read(&self, _out_type: &Type<V>, _ctx: &mut R1CGen) -> Self {
        let ptr = self.expect_ptr();
        let value = ptr.borrow().clone();
        value
    }

    fn expect_constant_bool(&self, _ctx: &mut R1CGen) -> bool {
        self.expect_constant() == ark_bn254::Fr::ONE
    }

    fn select(&self, if_t: &Self, if_f: &Self, _out_type: &Type<V>, _ctx: &mut R1CGen) -> Self {
        self.mul(if_t)
            .add(&Value::Const(ark_bn254::Fr::ONE).sub(self).mul(if_f))
    }

    fn write_witness(&self, _tp: Option<&Type<V>>, ctx: &mut R1CGen) -> Self {
        let witness_var = ctx.next_witness();
        Value::LC(vec![(witness_var, ark_bn254::Fr::ONE)])
    }

    fn fresh_witness(ctx: &mut R1CGen) -> Self {
        let witness_var = ctx.next_witness();
        Value::LC(vec![(witness_var, ark_bn254::Fr::ONE)])
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
        _out_type: &Type<V>,
        _ctx: &mut R1CGen,
    ) -> Self {
        todo!("ToRadix R1CS generation not yet implemented")
    }
}

#[derive(Clone, Debug, Copy)]
pub struct WitnessLayout {
    pub algebraic_size: usize,
    pub multiplicities_size: usize,

    pub challenges_size: usize,

    pub tables_data_size: usize,
    pub lookups_data_size: usize,
}

impl WitnessLayout {
    pub fn algebraic_start(&self) -> usize {
        0
    }

    pub fn algebraic_end(&self) -> usize {
        self.algebraic_size
    }

    pub fn multiplicities_start(&self) -> usize {
        self.algebraic_end()
    }

    pub fn multiplicities_end(&self) -> usize {
        self.multiplicities_start() + self.multiplicities_size
    }

    pub fn challenges_start(&self) -> usize {
        self.multiplicities_end()
    }

    pub fn challenges_end(&self) -> usize {
        self.challenges_start() + self.challenges_size
    }

    pub fn next_challenge(&mut self) -> usize {
        let challenge_id = self.challenges_end();
        self.challenges_size += 1;
        challenge_id
    }

    pub fn tables_data_start(&self) -> usize {
        self.challenges_end()
    }

    pub fn tables_data_end(&self) -> usize {
        self.tables_data_size + self.tables_data_start()
    }

    pub fn next_table_data(&mut self) -> usize {
        let table_data_id = self.tables_data_end();
        self.tables_data_size += 1;
        table_data_id
    }

    pub fn lookups_data_start(&self) -> usize {
        self.tables_data_end()
    }

    pub fn lookups_data_end(&self) -> usize {
        self.lookups_data_size + self.lookups_data_start()
    }

    pub fn next_lookups_data(&mut self) -> usize {
        let lookups_data_id = self.lookups_data_end();
        self.lookups_data_size += 1;
        lookups_data_id
    }

    pub fn size(&self) -> usize {
        self.algebraic_size
            + self.multiplicities_size
            + self.challenges_size
            + self.tables_data_size
            + self.lookups_data_size
    }

    pub fn pre_commitment_size(&self) -> usize {
        self.algebraic_size + self.multiplicities_size
    }

    pub fn post_commitment_size(&self) -> usize {
        self.challenges_size + self.tables_data_size + self.lookups_data_size
    }
}

#[derive(Clone, Debug, Copy)]
pub struct ConstraintsLayout {
    pub algebraic_size: usize,
    pub tables_data_size: usize,
    pub lookups_data_size: usize,
}

impl ConstraintsLayout {
    pub fn size(&self) -> usize {
        self.algebraic_size + self.tables_data_size + self.lookups_data_size
    }

    pub fn tables_data_start(&self) -> usize {
        self.algebraic_size
    }

    pub fn lookups_data_start(&self) -> usize {
        self.algebraic_size + self.tables_data_size
    }
}

#[derive(Clone, Debug)]
pub struct R1CS {
    pub witness_layout: WitnessLayout,
    pub constraints_layout: ConstraintsLayout,
    pub constraints: Vec<R1C>,
}

impl R1CGen {
    pub fn new() -> Self {
        Self {
            constraints: vec![],
            next_witness: 1, // 0 is reserved for constant one
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
    pub fn run<V: Clone + CommutativeMonoid>(&mut self, ssa: &SSA<V>, type_info: &TypeInfo<V>) {
        let entry_point = ssa.get_main_id();
        assert!(ssa.get_function(entry_point).get_param_types().len() == 0, "Main should not have parameters as WitnessWriteToFresh pass should remove them");
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
                    result.push(R1C {
                        a: vec![(y, ark_bn254::Fr::ONE)],
                        b,
                        c: vec![(0, ark_bn254::Fr::ONE)],
                    });
                    y
                }
                2 => {
                    let x = witness_layout.next_lookups_data();
                    let y = witness_layout.next_lookups_data();
                    result.push(R1C {
                        a: vec![(beta, crate::compiler::Field::ONE)],
                        b: lookup.elements[1].clone(),
                        c: vec![(x, -crate::compiler::Field::ONE)],
                    });

                    result.push(R1C {
                        a: vec![(y, ark_bn254::Fr::ONE)],
                        b: lookup.elements[0].clone(),
                        c: vec![(x, -crate::compiler::Field::ONE)],
                    });
                    let mut b = vec![(alpha, ark_bn254::Fr::ONE), (x, -crate::compiler::Field::ONE)];
                    for (w, coeff) in lookup.elements[0].iter() {
                        b.push((*w, -*coeff));
                    }
                    result.push(R1C {
                        a: vec![(y, ark_bn254::Fr::ONE)],
                        b,
                        c: vec![(0, ark_bn254::Fr::ONE)],
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

impl R1CS {
    pub fn compute_derivatives(
        &self,
        coeffs: &[crate::compiler::Field],
        res_a: &mut [crate::compiler::Field],
        res_b: &mut [crate::compiler::Field],
        res_c: &mut [crate::compiler::Field],
    ) {
        for (r1c, coeff) in self.constraints.iter().zip(coeffs.iter()) {
            for (a_ix, a_coeff) in r1c.a.iter() {
                res_a[*a_ix] += *a_coeff * *coeff;
            }
            for (b_ix, b_coeff) in r1c.b.iter() {
                res_b[*b_ix] += *b_coeff * *coeff;
            }
            for (c_ix, c_coeff) in r1c.c.iter() {
                res_c[*c_ix] += *c_coeff * *coeff;
            }
        }
    }

    pub fn check_witgen_output(
        &self,
        pre_comm_witness: &[crate::compiler::Field],
        post_comm_witness: &[crate::compiler::Field],
        a: &[crate::compiler::Field],
        b: &[crate::compiler::Field],
        c: &[crate::compiler::Field],
    ) -> bool {
        let witness = [pre_comm_witness, post_comm_witness].concat();
        if a.len() != self.constraints_layout.size() {
            error!(message = %"The a vector has the wrong length", expected = self.constraints_layout.size(), actual = a.len());
            return false;
        }
        if b.len() != self.constraints_layout.size() {
            error!(message = %"The b vector has the wrong length", expected = self.constraints_layout.size(), actual = b.len());
            return false;
        }
        if c.len() != self.constraints_layout.size() {
            error!(message = %"The c vector has the wrong length", expected = self.constraints_layout.size(), actual = c.len());
            return false;
        }
        for (i, r1c) in self.constraints.iter().enumerate() {
            let av = r1c
                .a
                .iter()
                .map(|(i, c)| c * &witness[*i])
                .sum::<ark_bn254::Fr>();

            let bv = r1c
                .b
                .iter()
                .map(|(i, c)| c * &witness[*i])
                .sum::<ark_bn254::Fr>();

            let cv = r1c
                .c
                .iter()
                .map(|(i, c)| c * &witness[*i])
                .sum::<ark_bn254::Fr>();
            let mut fail = false;
            if av * bv != cv {
                error!(message = %"R1CS constraint failed to verify", index = i);
                fail = true;
            }
            if av != a[i] {
                error!(message = %"Wrong A value for constraint", index = i, actual = a[i].to_string(), expected = av.to_string());
                fail = true;
            }
            if bv != b[i] {
                error!(message = %"Wrong B value for constraint", index = i, actual = b[i].to_string(), expected = bv.to_string());
                fail = true;
            }
            if cv != c[i] {
                error!(message = %"Wrong C value for constraint", index = i, actual = c[i].to_string(), expected = cv.to_string());
                fail = true;
            }
            if fail {
                return false;
            }
        }
        return true;
    }

    pub fn check_ad_output(
        &self,
        coeffs: &[crate::compiler::Field],
        a: &[crate::compiler::Field],
        b: &[crate::compiler::Field],
        c: &[crate::compiler::Field],
    ) -> bool {
        let mut a = a.to_vec();
        let mut b = b.to_vec();
        let mut c = c.to_vec();
        for (r1c, coeff) in self.constraints.iter().zip(coeffs.iter()) {
            for (a_ix, a_coeff) in r1c.a.iter() {
                a[*a_ix] -= *a_coeff * *coeff;
            }
            for (b_ix, b_coeff) in r1c.b.iter() {
                b[*b_ix] -= *b_coeff * *coeff;
            }
            for (c_ix, c_coeff) in r1c.c.iter() {
                c[*c_ix] -= *c_coeff * *coeff;
            }
        }
        let mut wrongs = 0;
        for i in 0..a.len() {
            if a[i] != crate::compiler::Field::ZERO {
                if wrongs == 0 {
                    error!(message = %"Wrong A deriv for witness", index = i);
                }
                wrongs += 1;
            }
            if b[i] != crate::compiler::Field::ZERO {
                if wrongs == 0 {
                    error!(message = %"Wrong B deriv for witness", index = i);
                }
                wrongs += 1;
            }
            if c[i] != crate::compiler::Field::ZERO {
                if wrongs == 0 {
                    error!(message = %"Wrong C deriv for witness", index = i);
                }
                wrongs += 1;
            }
        }
        if wrongs > 0 {
            error!("{} out of {} wrong derivatives", wrongs, 3 * a.len());
            return false;
        }
        return true;
    }
}
