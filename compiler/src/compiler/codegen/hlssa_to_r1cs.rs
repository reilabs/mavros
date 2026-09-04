use std::{cell::RefCell, collections::BTreeMap, rc::Rc};

use ark_ff::{AdditiveGroup, BigInt, BigInteger, Field, PrimeField};
use tracing::{instrument, warn};

use mavros_artifacts::FieldConfig;
use mavros_int_semantics::{self as semantics, CmpOp, IntBits};

use crate::compiler::{
    analysis::{
        symbolic_executor::{self, AssertionFailure, SymbolicExecutor},
        types::TypeInfo,
    },
    ssa::{
        BlockId, FunctionId,
        hlssa::{
            self, ArithGroup, BinaryArithOpKind, CmpKind, HLSSA, MAX_SUPPORTED_UNSIGNED_BITS,
            Radix, RefCountOp, SliceOpDir, Type, TypeExpr, assert_signed_op_width,
        },
    },
    util::{host_word, spread_bits, unspread_bits},
};

pub use mavros_artifacts::{
    ConstraintsLayout, FlamegraphProfile, FlamegraphStackId, LC, R1C, R1CS, WitnessLayout,
};

/// Per-function circuit-size profiles produced alongside the R1CS.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct R1CSProfile {
    pub constraints: FlamegraphProfile,
    pub witnesses: FlamegraphProfile,
}

#[derive(Debug)]
pub struct R1CSProfileDisabled {
    r1cs: Box<R1CS>,
}

impl R1CSProfileDisabled {
    pub fn into_r1cs(self) -> R1CS {
        *self.r1cs
    }
}

impl std::fmt::Display for R1CSProfileDisabled {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("R1CS profiling was not enabled")
    }
}

impl std::error::Error for R1CSProfileDisabled {}

// FIELD-ASSUMPTION: L1-direct-ref (57 sites)
// FIELD-ASSUMPTION: L4-two-pow
fn two_pow(exponent: usize) -> ark_bn254::Fr {
    ark_bn254::Fr::from(2).pow([exponent as u64])
}

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
    /// Report an undefined `Div`/`Mod` reaching the R1CS constant folds as a compiler bug.
    ///
    /// `LowerPureGuards` asserts every integer `Div`/`Mod` is defined immediately before the op,
    /// so an undefined one makes the program unsatisfiable and `R1CGen` rejects it at that assert —
    /// which precedes the division in the instruction stream — before this fold is ever asked for
    /// a quotient. Reaching here means that ordering broke, so say so instead of surfacing Rust's
    /// bare "attempt to divide by zero" from inside codegen, or — for `INT_MIN / -1` — instead of
    /// silently wrapping to a plausible-looking answer.
    #[track_caller]
    fn ice_undefined_divmod(kind: BinaryArithOpKind, bits: usize, signed: bool) -> ! {
        let sign = if signed { 'i' } else { 'u' };
        panic!(
            "ICE: undefined {kind:?} reached the R1CS constant fold on {sign}{bits}; \
             LowerPureGuards should have rejected the program at the preceding assertion"
        )
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
        // Zero has no inverse, and arkworks' `Div` unwraps one, so both arms below panic on a zero
        // divisor. `LowerPureGuards` now asserts every field division is defined, so `R1CGen`
        // rejects the program at that assertion — which precedes the division — before either arm
        // runs. Say which invariant broke rather than surfacing arkworks' unwrap from inside
        // codegen.
        // FIELD-ASSUMPTION: L4-eval
        if let Value::Const(rhs) = other
            && *rhs == ark_bn254::Fr::ZERO
        {
            panic!(
                "ICE: zero divisor reached the R1CS field division fold; \
                 LowerPureGuards should have rejected the program at the preceding assertion"
            );
        }
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
        let v = self.expect_in_u128("u1");
        assert!(v <= 1, "expected u1, but value is {v}");
        v == 1
    }

    /// The canonical value of a constant as a `u128`, or `None` if it needs more than 128 bits.
    ///
    /// Every `expect_u*` goes through this. They each used to format the constant as a decimal
    /// string and re-parse it, which is correct but costs an allocation and a base-10 conversion
    /// per operand — fine at the handful of calls R1CS generation makes, and far too slow to sweep
    /// a conformance test through.
    ///
    /// Both the read and the test that it lost nothing come from the model, which is what makes
    /// the agreement with `lattice::int_cast_bits` structural rather than a coincidence the two
    /// have to be kept in. Deriving "which limbs are outside 128 bits" here as well is how a
    /// caller and its reader come apart; see `field_limbs_fit`.
    ///
    /// The limb _count_ is a property of the field, and both entry points take the slice whatever
    /// its length: a field whose `BigInt` is narrower than this one simply has fewer, where
    /// `limbs[2]` would have panicked.
    fn const_u128(&self, what: &str) -> Option<u128> {
        match self {
            // FIELD-ASSUMPTION: L4-decompose
            Value::Const(c) => {
                let limbs = c.into_bigint().0;
                IntBits::field_limbs_fit(&limbs, semantics::MAX_BITS)
                    .then(|| host_word(&IntBits::from_field_limbs(&limbs, semantics::MAX_BITS)))
            }
            r => panic!("expected {what}, got {r:?}"),
        }
    }

    /// The canonical value as a `u128`, panicking with the caller's name if it does not fit.
    fn expect_in_u128(&self, what: &str) -> u128 {
        self.const_u128(what).unwrap_or_else(|| {
            let Value::Const(c) = self else { unreachable!("const_u128 panics on a non-constant") };
            panic!("expected {what}, but field value is {}", c.into_bigint())
        })
    }

    /// Narrow to `T`, panicking with the caller's name if the value does not fit it.
    fn expect_narrow<T: TryFrom<u128>>(&self, what: &str) -> T {
        let v = self.expect_in_u128(what);
        T::try_from(v).unwrap_or_else(|_| panic!("expected {what}, but field value is {v}"))
    }

    pub fn expect_u8(&self) -> u8 {
        self.expect_narrow("u8")
    }

    pub fn expect_u32(&self) -> u32 {
        self.expect_narrow("u32")
    }

    pub fn expect_u64(&self) -> u64 {
        self.expect_narrow("u64")
    }

    pub fn expect_u128(&self) -> u128 {
        self.expect_in_u128("u128")
    }

    /// The canonical value as a `bits`-wide pattern.
    ///
    /// This evaluator carries its integers as field elements and reads them out as host words, so a
    /// pattern exists only for the length of a call into the model.
    fn expect_pattern(&self, bits: usize) -> IntBits {
        IntBits::from_u128(bits, self.expect_u128())
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
    /// `(n, 2^n)` for `n` in `0..2^size`, where `size` is `log2` of the shifted operand's width.
    Pow2(u8),
}

impl Table {
    fn row_count(&self) -> usize {
        match self {
            Table::Range(bits) => 1usize << bits,
            Table::OfElems(elements) => elements.len(),
            Table::Spread(bits) => 1usize << bits,
            Table::Pow2(size) => 1usize << size,
        }
    }

    fn width(&self) -> usize {
        match self {
            Table::Range(_) => 1,
            Table::OfElems(_) | Table::Spread(_) | Table::Pow2(_) => 2,
        }
    }

    fn profile_size(&self) -> (u64, u64) {
        let rows = self.row_count() as u64;
        match self {
            Table::Range(_) | Table::Spread(_) | Table::Pow2(_) => (rows + 1, 2 * rows),
            Table::OfElems(_) => (2 * rows + 1, 3 * rows),
        }
    }

    fn profile_name(&self, table_index: usize) -> String {
        match self {
            Table::Range(bits) => format!("<range table: {bits} bits>"),
            Table::OfElems(elements) => {
                format!("<array table #{table_index}: {} rows>", elements.len())
            }
            Table::Spread(bits) => format!("<spread table: {bits} bits>"),
            Table::Pow2(size) => {
                format!("<pow2 table: {}-bit shifts>", 1usize << size)
            }
        }
    }
}

#[derive(Clone)]
pub struct R1CGen {
    /// The field the program operates over. Codegen's arithmetic is still raw `ark_bn254::Fr` (it
    /// becomes per-field in P4); this is here so that the type-width queries it makes read the
    /// configured field rather than a static.
    field: FieldConfig,
    constraints: Vec<R1C>,
    tables: Vec<Table>,
    lookups: Vec<LookupConstraint>,
    next_witness: usize,
    function_names: BTreeMap<FunctionId, String>,
    call_stack: Vec<String>,
    call_stack_ids: Vec<(FlamegraphStackId, FlamegraphStackId)>,
    profile_root: String,
    profile: Option<R1CSProfile>,
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
        if self.profile.is_some() {
            let name = self
                .function_names
                .get(&func)
                .cloned()
                .unwrap_or_else(|| format!("fn{}", func.0));
            self.push_profile_frame(name);
        }
        None
    }

    fn on_return(&mut self, _returns: &mut [Value], _return_types: &[Type]) {
        if self.profile.is_some() {
            self.pop_profile_frame();
        }
    }

    fn on_jmp(&mut self, _target: BlockId, _params: &mut [Value], _param_types: &[&Type]) {}

    fn lookup(&mut self, target: hlssa::LookupTarget<Value>, args: Vec<Value>, flag: Value) {
        let flag_lc = flag.expect_linear_combination();
        let els: Vec<_> = args
            .into_iter()
            .map(|e| e.expect_linear_combination())
            .collect();
        let lookup_size = els.len() as u64;
        let table_id = match target {
            hlssa::LookupTarget::Rangecheck(i) => {
                // Find or create the rangecheck table of this size. The lookup-sizing analysis
                // may select several distinct sizes, so multiple range tables can coexist.
                self.find_or_create_range_table(i as u64)
            }
            hlssa::LookupTarget::DynRangecheck(_) => {
                // `to_radix` lowers its (asserted radix-256) digit checks to static 8-bit
                // rangechecks, so no `DynRangecheck` survives to R1CS generation.
                unreachable!(
                    "DynRangecheck is lowered to a static 8-bit rangecheck before R1CS gen"
                )
            }
            hlssa::LookupTarget::Spread(bits) => {
                let existing = self.tables.iter().position(|t| match t {
                    Table::Spread(n) => *n == bits,
                    _ => false,
                });
                if let Some(idx) = existing {
                    idx
                } else {
                    self.add_table(Table::Spread(bits))
                }
            }
            hlssa::LookupTarget::Pow2(size) => {
                let existing = self.tables.iter().position(|t| match t {
                    Table::Pow2(n) => *n == size,
                    _ => false,
                });
                if let Some(idx) = existing {
                    idx
                } else {
                    self.add_table(Table::Pow2(size))
                }
            }
            hlssa::LookupTarget::Array(arr) => {
                let arr = arr.expect_array();
                if arr.borrow().table_id.is_none() {
                    let mut elems = Vec::new();
                    flatten_array_into_table(&arr.borrow(), &mut elems);
                    let idx = self.add_table(Table::OfElems(elems));
                    arr.borrow_mut().table_id = Some(idx);
                    idx
                } else {
                    arr.borrow().table_id.unwrap()
                }
            }
        };
        self.record_constraints(lookup_size);
        self.record_witnesses(lookup_size);
        self.lookups.push(LookupConstraint {
            table_id,
            elements: els,
            flag: flag_lc,
        });
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
    fn cmp(&self, b: &Self, kind: CmpKind, bits: Option<usize>, _ctx: &mut R1CGen) -> Self {
        match kind {
            CmpKind::Eq => self.eq(b),
            // `ULt` compares the field encodings, which is the magnitude for an unsigned integer.
            CmpKind::ULt => self.lt(b),
            CmpKind::SLt => {
                let bits = bits.expect("ICE: signed comparison without an operand width");
                let less = self
                    .expect_pattern(bits)
                    .compare(CmpOp::SLt, &b.expect_pattern(bits));
                Value::Const(if less {
                    ark_bn254::Fr::ONE
                } else {
                    ark_bn254::Fr::ZERO
                })
            }
        }
    }

    /// Evaluate a binary arithmetic operation on constants.
    ///
    /// Unlike the constant folders in `click_cooper::lattice` and the `Specializer`, this is not
    /// speculative: it is the _final_ evaluation, and it runs **after** the guard IR. So it must
    /// not decline: an operation reaching here has already had its rejection enforced (or proved
    /// unnecessary) by the assertion `LowerPureGuards` planted ahead of it, and something has to be
    /// written. That makes the obligation [`mavros_int_semantics::residue`]'s rather than
    /// [`mavros_int_semantics::eval`]'s, exactly:
    ///
    /// - Accepted inputs produce Noir's value;
    /// - Rejected ones the model still specifies produce the pattern every other backend produces,
    ///   so that a program mavros wrongly accepted is at least wrong the same way everywhere;
    /// - The two the model deliberately leaves unspecified, because there the whole obligation sits
    ///   on the guard IR and reaching this point means that ordering broke.
    fn arith(
        &self,
        b: &Self,
        binary_arith_op_kind: BinaryArithOpKind,
        out_type: &Type,
        _ctx: &mut R1CGen,
    ) -> Self {
        match &out_type.strip_witness().expr {
            TypeExpr::Int(bits) => {
                let bits = *bits;
                assert!(
                    bits > 0 && bits <= MAX_SUPPORTED_UNSIGNED_BITS,
                    "Unsupported integer size in R1CS arith: int{bits}"
                );
                if binary_arith_op_kind.is_signed() {
                    assert_signed_op_width(bits, "R1CS constant arithmetic");
                }
                assert!(
                    matches!((self, b), (Value::Const(_), Value::Const(_))),
                    "Non-constant integer {:?} is not supported in R1CS arith",
                    binary_arith_op_kind
                );

                // Both operands are read at `bits`: HLSSA types an `IntArith` result as
                // `int{max(s1, s2)}` and this evaluator only sees that result type, so a shift
                // amount narrower than the value is indistinguishable from one declared at the
                // value's own width. Reading it wider than it was declared is harmless as the extra
                // bits are zero, and `expect_pattern` masks to `bits` on the way in as the model
                // takes its operands already normalized and does no masking of its own.
                let raw = semantics::residue(
                    binary_arith_op_kind.into(),
                    &self.expect_pattern(bits),
                    &b.expect_pattern(bits),
                )
                .unwrap_or_else(|| {
                    Self::ice_undefined_divmod(
                        binary_arith_op_kind,
                        bits,
                        binary_arith_op_kind.is_signed(),
                    )
                });

                Value::Const(ark_bn254::Fr::from(host_word(&raw)))
            }
            TypeExpr::Field | TypeExpr::WitnessOf(_) => match binary_arith_op_kind.group() {
                ArithGroup::Add => self.add(b),
                ArithGroup::Sub => self.sub(b),
                ArithGroup::Mul => self.mul(b),
                ArithGroup::Div => self.div(b),
                ArithGroup::Rem => {
                    panic!("Modulo is not defined on field elements")
                }
                ArithGroup::And
                | ArithGroup::Or
                | ArithGroup::Xor
                | ArithGroup::Shl
                | ArithGroup::Shr => {
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
        bits: Option<usize>,
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
            CmpKind::ULt => {
                let a_val = a.expect_constant();
                let b_val = b.expect_constant();
                if a_val >= b_val {
                    return Err(AssertionFailure::new(format!(
                        "assert_cmp lt failed: {a_val:?} >= {b_val:?}"
                    )));
                }
            }
            CmpKind::SLt => {
                let bits = bits.expect("ICE: signed comparison without an operand width");
                let a_val = a.expect_constant();
                let b_val = b.expect_constant();

                // Read as two's complement, by the same model call above, so an assertion and the
                // comparison it asserts on cannot disagree.
                if !a
                    .expect_pattern(bits)
                    .compare(CmpOp::SLt, &b.expect_pattern(bits))
                {
                    return Err(AssertionFailure::new(format!(
                        "assert_cmp lt (signed) failed: {a_val:?} >= {b_val:?}"
                    )));
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

    fn not(&self, out_type: &Type, ctx: &mut R1CGen) -> Self {
        let value_const = self.expect_constant();
        let bits = value_const.into_bigint().to_bits_le();
        let bit_size = out_type.get_bit_size(ctx.field());
        let mut negated_bits = Vec::new();
        for i in 0..bit_size {
            let bit = if i < bits.len() { bits[i] } else { false };
            negated_bits.push(!bit);
        }
        Value::Const(ark_bn254::Fr::from_bigint(BigInt::from_bits_le(&negated_bits)).unwrap())
    }

    fn of_int(v: &IntBits, _ctx: &mut R1CGen) -> Self {
        Value::Const(ark_bn254::Fr::from(host_word(v)))
    }

    fn of_field(f: crate::compiler::Field, _ctx: &mut R1CGen) -> Self {
        // Boundary: the middle-end `FieldElement` becomes the raw `ark_bn254::Fr` this R1CS
        // evaluator computes coefficients in.
        Value::Const(f.to_ark())
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
    pub fn new(field: FieldConfig) -> Self {
        Self {
            field,
            constraints: vec![],
            next_witness: 0,
            tables: vec![],
            lookups: vec![],
            function_names: BTreeMap::new(),
            call_stack: Vec::new(),
            call_stack_ids: Vec::new(),
            profile_root: "<r1cs>".to_string(),
            profile: None,
        }
    }

    /// The field the program operates over.
    pub fn field(&self) -> FieldConfig {
        self.field
    }

    pub fn enable_profile(&mut self) {
        self.profile = Some(R1CSProfile::default());
    }

    fn push_profile_frame(&mut self, name: String) {
        self.call_stack.push(name);
        let profile = self
            .profile
            .as_mut()
            .expect("profile frames are recorded only when profiling is enabled");
        let constraint_stack_id = profile
            .constraints
            .intern_stack(self.call_stack.iter().cloned())
            .expect("R1CS profile call stack is non-empty");
        let witness_stack_id = profile
            .witnesses
            .intern_stack(self.call_stack.iter().cloned())
            .expect("R1CS profile call stack is non-empty");
        self.call_stack_ids
            .push((constraint_stack_id, witness_stack_id));
    }

    fn pop_profile_frame(&mut self) {
        self.call_stack
            .pop()
            .expect("ICE: R1CS profiler call stack underflow");
        self.call_stack_ids
            .pop()
            .expect("ICE: R1CS profiler stack ID underflow");
    }

    fn record_constraints(&mut self, count: u64) {
        if let Some(profile) = &mut self.profile {
            let stack_id = self
                .call_stack_ids
                .last()
                .expect("constraints are recorded inside an R1CS call frame")
                .0;
            profile.constraints.record_interned(stack_id, count);
        }
    }

    fn record_witnesses(&mut self, count: u64) {
        if let Some(profile) = &mut self.profile {
            let stack_id = self
                .call_stack_ids
                .last()
                .expect("witnesses are recorded inside an R1CS call frame")
                .1;
            profile.witnesses.record_interned(stack_id, count);
        }
    }

    fn add_table(&mut self, table: Table) -> usize {
        let table_index = self.tables.len();
        let (constraints, witnesses) = table.profile_size();
        if let Some(profile) = &mut self.profile {
            let stack = [
                self.profile_root.clone(),
                "<lookup tables>".to_string(),
                table.profile_name(table_index),
            ];
            profile.constraints.record(stack.clone(), constraints);
            profile.witnesses.record(stack, witnesses);
        }
        self.tables.push(table);
        table_index
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
            self.add_table(Table::Range(bits))
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
        let entry_point = ssa.get_unique_entrypoint_id();
        if self.profile.is_some() {
            self.function_names = ssa
                .iter_functions()
                .map(|(id, function)| (*id, function.get_name().to_string()))
                .collect();
            self.profile_root = ssa.get_function(entry_point).get_name().to_string();
        }
        assert!(
            ssa.get_function(entry_point).get_param_types().len() == 0,
            "Main should not have parameters as WitnessWriteToFresh pass should remove them"
        );
        let main_params = vec![];
        let executor = SymbolicExecutor::new();
        let result = executor.run(ssa, type_info, entry_point, main_params, self);
        debug_assert!(
            self.profile.is_none()
                || result.is_err()
                || (self.call_stack.is_empty() && self.call_stack_ids.is_empty())
        );
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

    pub fn seal(self, guard: Option<(usize, usize)>) -> R1CS {
        self.seal_impl(guard).0
    }

    pub fn seal_with_profile(
        self,
        guard: Option<(usize, usize)>,
    ) -> Result<(R1CS, R1CSProfile), R1CSProfileDisabled> {
        let (r1cs, profile) = self.seal_impl(guard);
        match profile {
            Some(profile) => Ok((r1cs, profile)),
            None => Err(R1CSProfileDisabled {
                r1cs: Box::new(r1cs),
            }),
        }
    }

    fn seal_impl(mut self, guard: Option<(usize, usize)>) -> (R1CS, Option<R1CSProfile>) {
        // Algebraic section
        let mut witness_layout = WitnessLayout {
            guard_index: guard.map(|(index, _)| index),
            return_len: guard.map_or(0, |(_, len)| len),
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
        for table in self.tables {
            let len = table.row_count();
            max_width = max_width.max(table.width());
            table_infos.push(TableInfo {
                multiplicities_witness_off: witness_layout.multiplicities_size
                    + witness_layout.algebraic_size,
                table,
                sum_constraint_idx: 0,
            });
            witness_layout.multiplicities_size += len;
        }

        if table_infos.is_empty() {
            let r1cs = R1CS {
                witness_layout,
                constraints_layout,
                constraints: result,
            };
            if let Some(profile) = &self.profile {
                assert_eq!(
                    profile.constraints.total_weight(),
                    r1cs.constraints_layout.size() as u64
                );
                assert_eq!(
                    profile.witnesses.total_weight(),
                    r1cs.witness_layout.size() as u64
                );
            }
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
        if let Some(profile) = &mut self.profile {
            profile.witnesses.record(
                [self.profile_root.clone(), "<lookup challenges>".to_string()],
                witness_layout.challenges_size as u64,
            );
        }

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
                                (0, -ark_bn254::Fr::from(i as u64)),
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
                            a: vec![(beta, ark_bn254::Fr::ONE)],
                            b: v.clone(),
                            c: vec![(x, -ark_bn254::Fr::ONE)],
                        });
                        result.push(R1C {
                            a: vec![(y, ark_bn254::Fr::ONE)],
                            b: vec![
                                (alpha, ark_bn254::Fr::ONE),
                                (0, -ark_bn254::Fr::from(i as u64)),
                                (x, -ark_bn254::Fr::ONE),
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
                                (0, -ark_bn254::Fr::from(i as u64)),
                                (beta, ark_bn254::Fr::from(spread_val)),
                            ],
                            c: vec![(m, ark_bn254::Fr::ONE)],
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
                Table::Pow2(size) => {
                    // Powers-of-two table: for each entry n in 0..2^size, value = 2^n. Folded
                    // exactly like the spread table above as both operands are compile-time
                    // constants, so beta*2^n folds straight into the denominator:
                    //
                    // -> y = m_n / (alpha - n + beta*2^n),
                    //    constraint `y * (alpha - n + beta*2^n) - m_n = 0`
                    //
                    // FIELD-ASSUMPTION: L4-decompose. The largest value is 2^(2^size - 1), which is
                    // 2^127 at the widest size `MAX_POW2_TABLE_SIZE` admits; that ceiling is set so
                    // the widest row clears the modulus, so no row wraps. `two_pow` would wrap
                    // silently if it ever did.
                    let len = 1usize << size;
                    let mut sum_lhs: LC = vec![];
                    for n in 0..len {
                        let y = witness_layout.next_table_data();
                        let m = table_info.multiplicities_witness_off + n;
                        result.push(R1C {
                            a: vec![(y, ark_bn254::Fr::ONE)],
                            b: vec![
                                (alpha, ark_bn254::Fr::ONE),
                                (0, -ark_bn254::Fr::from(n as u64)),
                                (beta, two_pow(n)),
                            ],
                            c: vec![(m, ark_bn254::Fr::ONE)],
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
                        a: vec![(beta, ark_bn254::Fr::ONE)],
                        b: lookup.elements[1].clone(),
                        c: vec![(x, -ark_bn254::Fr::ONE)],
                    });

                    // y * (α - x - key) = flag
                    let mut b = vec![(alpha, ark_bn254::Fr::ONE), (x, -ark_bn254::Fr::ONE)];
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
        if let Some(profile) = &self.profile {
            assert_eq!(
                profile.constraints.total_weight(),
                r1cs.constraints_layout.size() as u64
            );
            assert_eq!(
                profile.witnesses.total_weight(),
                r1cs.witness_layout.size() as u64
            );
        }
        (r1cs, self.profile)
    }
}

#[cfg(test)]
mod r1cs_profile_tests {
    use super::{ArrayData, FieldConfig, R1CGen, Value, hlssa, symbolic_executor};
    use ark_ff::Field as _;
    use std::{cell::RefCell, rc::Rc};

    fn witness(generator: &mut R1CGen) -> Value {
        let witness = generator.next_witness();
        generator.record_witnesses(1);
        Value::LC(vec![(witness, ark_bn254::Fr::ONE)])
    }

    fn array_table(values: &[u64]) -> Value {
        Value::Array(Rc::new(RefCell::new(ArrayData {
            table_id: None,
            data: values
                .iter()
                .map(|value| Value::Const(ark_bn254::Fr::from(*value)))
                .collect(),
        })))
    }

    fn lookup(generator: &mut R1CGen, target: hlssa::LookupTarget<Value>, args: Vec<Value>) {
        <R1CGen as symbolic_executor::Context<Value>>::lookup(
            generator,
            target,
            args,
            Value::Const(ark_bn254::Fr::ONE),
        );
    }

    #[test]
    fn all_lookup_profiles_match_the_sealed_r1cs_layout() {
        let mut generator = R1CGen::new(FieldConfig::bn254());
        generator.enable_profile();
        generator.profile_root = "main".to_string();
        generator.push_profile_frame("main".to_string());

        let range_value = witness(&mut generator);
        lookup(
            &mut generator,
            hlssa::LookupTarget::Rangecheck(2),
            vec![range_value],
        );

        let spread_key = witness(&mut generator);
        let spread_value = witness(&mut generator);
        lookup(
            &mut generator,
            hlssa::LookupTarget::Spread(2),
            vec![spread_key, spread_value],
        );

        let pow2_amount = witness(&mut generator);
        let pow2_factor = witness(&mut generator);
        lookup(
            &mut generator,
            hlssa::LookupTarget::Pow2(2),
            vec![pow2_amount, pow2_factor],
        );

        for values in [[10, 20, 30], [40, 50, 60]] {
            let table = array_table(&values);
            let index = witness(&mut generator);
            let value = witness(&mut generator);
            lookup(
                &mut generator,
                hlssa::LookupTarget::Array(table),
                vec![index, value],
            );
        }
        generator.pop_profile_frame();

        let (r1cs, profile) = generator.seal_with_profile(None).unwrap();
        assert_eq!(
            profile.constraints.total_weight(),
            r1cs.constraints_layout.size() as u64
        );
        assert_eq!(
            profile.witnesses.total_weight(),
            r1cs.witness_layout.size() as u64
        );

        let constraints = profile.constraints.to_folded();
        assert!(constraints.contains("main;<lookup tables>;<range table: 2 bits>"));
        assert!(constraints.contains("main;<lookup tables>;<spread table: 2 bits>"));
        assert!(constraints.contains("main;<lookup tables>;<pow2 table: 4-bit shifts>"));
        assert!(constraints.contains("main;<lookup tables>;<array table #3: 3 rows>"));
        assert!(constraints.contains("main;<lookup tables>;<array table #4: 3 rows>"));
        assert!(constraints.contains("main 9\n"));
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
/// The bit size comes from the configured field rather than a static constant; what remains
/// field-dependent is the K-challenge replication this rejects (see `Error::LogupSoundnessUnsupported`).
pub fn logup_soundness_report(
    field: FieldConfig,
    requested_bits: u32,
    degree: usize,
) -> Result<SoundnessReport, String> {
    // floor(log2 p): p in [2^(field_bit_size-1), 2^field_bit_size), so floor(log2 p) =
    // field_bit_size - 1. Using the floor (rather than the bit size) keeps `achieved_bits`
    // a lower bound and never over-claims security.
    let field_bits = field.field_bit_size() - 1;
    compute_logup_soundness(requested_bits, field_bits, degree)
}

/// Pure LogUp soundness computation (unit-tested with synthetic `field_bits`).
///
/// LogUp proves a log-derivative rational identity at a random challenge; clearing denominators
/// yields a nonzero polynomial of total degree `<= D = table_entries + num_lookups`, so one
/// challenge fails with probability `<= D/|F|` (Schwartz-Zippel). K independent challenges give
/// `(D/|F|)^K`, i.e. `K * (log2|F| - log2 D)` bits. All rounding is toward _under_-estimating
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
    use mavros_artifacts::FieldConfig;

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
        // The real configured field (bn254) at the 128-bit default must be a genuine no-op.
        let r = logup_soundness_report(FieldConfig::bn254(), 128, 1 << 20).unwrap();
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

#[cfg(test)]
mod comparison_tests {
    use super::{CmpKind, FieldConfig, R1CGen, Value, symbolic_executor};

    fn constant(v: u64) -> Value {
        Value::Const(ark_bn254::Fr::from(v))
    }

    fn cmp(a: &Value, b: &Value, kind: CmpKind, bits: Option<usize>) -> bool {
        let mut generator = R1CGen::new(FieldConfig::bn254());
        <Value as symbolic_executor::Value<R1CGen>>::cmp(a, b, kind, bits, &mut generator)
            .expect_u1()
    }

    fn assert_cmp_holds(a: &Value, b: &Value, kind: CmpKind, bits: Option<usize>) -> bool {
        let mut generator = R1CGen::new(FieldConfig::bn254());
        <Value as symbolic_executor::Value<R1CGen>>::assert_cmp(kind, a, b, bits, &mut generator)
            .is_ok()
    }

    /// Both comparison paths read their operands the way the opcode says, not the way the operand
    /// type says.
    ///
    /// `assert_cmp` is the half that was wrong. It used to pick two's complement by asking whether
    /// the operand was an integer, which was a working proxy for "signed" only while an unsigned
    /// integer wore a different type tag. Once `TypeExpr::U` and `TypeExpr::I` collapsed into one
    /// `Int`, that question started answering "yes" for every integer, and a `ULt` over a byte with
    /// its high bit set read it as negative — passing an assertion that should have failed, which
    /// is the direction that matters: the program is accepted, not rejected.
    #[test]
    fn a_comparison_reads_its_operands_the_way_the_opcode_says() {
        // 0xFB in eight bits is 251 read as a magnitude and -5 read as two's complement, so the
        // two readings disagree about every comparison of it against a small positive.
        let a = constant(0xFB);
        let b = constant(2);

        assert!(!cmp(&a, &b, CmpKind::ULt, Some(8)), "251 < 2 is false");
        assert!(cmp(&a, &b, CmpKind::SLt, Some(8)), "-5 < 2 is true");
        assert!(!cmp(&a, &b, CmpKind::Eq, Some(8)));
        assert!(cmp(&a, &a, CmpKind::Eq, Some(8)));

        assert!(
            !assert_cmp_holds(&a, &b, CmpKind::ULt, Some(8)),
            "an unsigned assertion that 251 < 2 must fail"
        );
        assert!(
            assert_cmp_holds(&a, &b, CmpKind::SLt, Some(8)),
            "a signed assertion that -5 < 2 must hold"
        );
        assert!(!assert_cmp_holds(&a, &b, CmpKind::Eq, Some(8)));
        assert!(assert_cmp_holds(&a, &a, CmpKind::Eq, Some(8)));
    }

    /// A field element has no width and no sign, and its comparisons say so.
    ///
    /// `assert_cmp` selects its arm by the opcode rather than by the operand's kind, so this is
    /// the case that pins a field comparison to the same answers either way.
    #[test]
    fn a_field_comparison_needs_no_width() {
        let a = constant(2);
        let b = constant(0xFB);

        assert!(cmp(&a, &b, CmpKind::ULt, None));
        assert!(assert_cmp_holds(&a, &b, CmpKind::ULt, None));
        assert!(!assert_cmp_holds(&b, &a, CmpKind::ULt, None));
        assert!(assert_cmp_holds(&a, &a, CmpKind::Eq, None));
    }
}

// INT-SEMANTICS CONFORMANCE
// ================================================================================================

#[cfg(test)]
mod int_semantics_conformance {
    use mavros_int_semantics::{IntBits, IntOp, Sign, corners, residue};

    use super::{BinaryArithOpKind, FieldConfig, R1CGen, Type, Value, symbolic_executor};

    /// Every operation, spelled out so a new variant cannot quietly skip the sweep.
    const ALL_ARITH: [BinaryArithOpKind; 17] = [
        BinaryArithOpKind::UAdd,
        BinaryArithOpKind::SAdd,
        BinaryArithOpKind::USub,
        BinaryArithOpKind::SSub,
        BinaryArithOpKind::UMul,
        BinaryArithOpKind::SMul,
        BinaryArithOpKind::UDiv,
        BinaryArithOpKind::SDiv,
        BinaryArithOpKind::URem,
        BinaryArithOpKind::SRem,
        BinaryArithOpKind::UShl,
        BinaryArithOpKind::SShl,
        BinaryArithOpKind::UShr,
        BinaryArithOpKind::SShr,
        BinaryArithOpKind::And,
        BinaryArithOpKind::Or,
        BinaryArithOpKind::Xor,
    ];

    /// `residue` on the host words either side of it.
    ///
    /// This evaluator holds its integers as field elements and reads them out as host words, so a
    /// pattern exists only for the length of a call into the model.
    fn model(op: BinaryArithOpKind, bits: usize, a: u128, b: u128) -> Option<u128> {
        residue(
            op.into(),
            &IntBits::from_u128(bits, a),
            &IntBits::from_u128(bits, b),
        )
        .map(|v| u128::try_from(&v).expect("a narrow answer fits a host word"))
    }

    /// Fold one operation the way `R1CGen` does, and read the answer back as a raw pattern.
    fn fold(op: BinaryArithOpKind, bits: usize, a: u128, b: u128) -> u128 {
        let mut generator = R1CGen::new(FieldConfig::bn254());
        <Value as symbolic_executor::Value<R1CGen>>::arith(
            &Value::Const(ark_bn254::Fr::from(a)),
            &Value::Const(ark_bn254::Fr::from(b)),
            op,
            &Type::int(bits),
            &mut generator,
        )
        .expect_u128()
    }

    /// The widths this evaluator may be asked about, for one reading.
    fn widths(op: BinaryArithOpKind) -> &'static [usize] {
        corners::widths_for(op.is_signed())
    }

    fn operand_pairs(op: BinaryArithOpKind, bits: usize) -> Vec<(u128, u128)> {
        let rhs = if IntOp::from(op).is_shift() {
            corners::shift_amounts(bits, bits)
        } else {
            corners::values(bits)
        };
        corners::values(bits)
            .into_iter()
            .flat_map(|a| rhs.iter().map(move |b| (a, *b)))
            .collect()
    }

    /// Whether the model declines to specify this input, which is the ICE case rather than a value.
    fn unspecified(op: BinaryArithOpKind, bits: usize, a: u128, b: u128) -> bool {
        model(op, bits, a, b).is_none()
    }

    /// The R1CS fold answers exactly what a total backend must answer.
    ///
    /// This is the exact relation, not the refinement the speculative folders owe: this evaluator
    /// runs after the guard IR and cannot decline. See [`Value::arith`] for why that makes the
    /// obligation `residue`'s rather than `eval`'s.
    #[test]
    fn the_r1cs_fold_agrees_with_the_model() {
        let mut checked = 0usize;

        for op in ALL_ARITH {
            for &bits in widths(op) {
                for (a, b) in operand_pairs(op, bits) {
                    let Some(want) = model(op, bits, a, b) else {
                        continue;
                    };
                    let got = fold(op, bits, a, b);
                    assert_eq!(
                        got, want,
                        "{op:?} at {bits} bits: {a:#x} {b:#x} folded to {got:#x}, model says \
                         {want:#x}"
                    );
                    checked += 1;
                }
            }
        }

        // An implementation that answered nothing would satisfy the loop above vacuously, so the
        // count is part of the test rather than a diagnostic.
        assert!(
            checked > 25_000,
            "the sweep only reached {checked} specified points"
        );
    }

    /// The two shapes the model leaves unspecified are exactly the two that ICE.
    ///
    /// Stated as a sweep rather than as a pair of `#[should_panic]` cases so that it is the model
    /// that decides which inputs those are. If `residue` ever gains or loses an unspecified shape,
    /// this fails rather than silently drifting.
    #[test]
    fn the_unspecified_inputs_are_exactly_the_divmod_ones() {
        let mut found = 0usize;

        for op in ALL_ARITH {
            for &bits in widths(op) {
                for (a, b) in operand_pairs(op, bits) {
                    if !unspecified(op, bits, a, b) {
                        continue;
                    }
                    found += 1;

                    assert!(
                        matches!(
                            IntOp::from(op),
                            IntOp::UDiv | IntOp::SDiv | IntOp::URem | IntOp::SRem
                        ),
                        "{op:?} at {bits} bits: {a:#x} {b:#x} is unspecified but is not a division"
                    );

                    let signed = matches!(op.sign(), Sign::Signed);
                    let min = 1u128 << (bits - 1);
                    let is_zero_divisor = b & mavros_int_semantics::mask(bits) == 0;
                    let is_div_overflow = signed
                        && a & mavros_int_semantics::mask(bits) == min
                        && b == !0 >> (128 - bits);
                    assert!(
                        is_zero_divisor || is_div_overflow,
                        "{op:?} at {bits} bits: {a:#x} {b:#x} is unspecified for neither reason"
                    );
                }
            }
        }

        assert!(
            found > 100,
            "the sweep only reached {found} unspecified points"
        );
    }

    /// An unspecified input is a compiler bug, and says so rather than folding to a plausible value.
    #[test]
    #[should_panic(expected = "ICE: undefined")]
    fn a_zero_divisor_ices() {
        fold(BinaryArithOpKind::UDiv, 8, 7, 0);
    }

    #[test]
    #[should_panic(expected = "ICE: undefined")]
    fn signed_division_overflow_ices() {
        fold(BinaryArithOpKind::SDiv, 8, 0x80, 0xFF);
    }

    #[test]
    fn an_out_of_range_shift_amount_reduces_to_the_width() {
        assert_eq!(fold(BinaryArithOpKind::UShl, 8, 1, 8), 1);
        assert_eq!(fold(BinaryArithOpKind::UShl, 8, 1, 9), 2);
        assert_eq!(fold(BinaryArithOpKind::UShr, 8, 0x80, 8), 0x80);

        // The signed right shift is here as the control: it reduces by the width like the other
        // two, so a failure isolated to `<<` points at the shift direction rather than at the
        // reduction.
        assert_eq!(fold(BinaryArithOpKind::SShr, 8, 0xFF, 8), 0xFF);
    }
}
