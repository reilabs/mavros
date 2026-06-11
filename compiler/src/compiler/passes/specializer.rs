//! Creates specialized copies of functions for specific call-site argument values using symbolic
//! execution where beneficial.
//!
//! These specialized functions are wired in using a dispatch function mechanism similar to the one
//! in defunctionalization wherever needed. In some cases, we can call the specialized version
//! directly instead.

use std::collections::{HashMap, HashSet};

use ark_ff::{AdditiveGroup, BigInteger, PrimeField};
use tracing::{info, instrument};

use crate::compiler::{
    Field,
    analysis::{
        instrumenter::{FunctionSignature, SpecializationSummary, Summary, ValueSignature},
        symbolic_executor::{self, SymbolicExecutor},
        types::TypeInfo,
    },
    pass_manager::{Analysis, AnalysisId, AnalysisStore, Pass},
    ssa::{
        BlockId, FunctionId, ValueId,
        hlssa::{
            BinaryArithOpKind, Blob, CastTarget, CmpKind, Constant, Endianness, HLFunction, HLSSA,
            LookupTarget, MAX_SUPPORTED_UNSIGNED_BITS, OpCode, Radix, RefCountOp,
            SequenceTargetType, Type, TypeExpr,
            builder::{HLEmitter, HLFunctionBuilder},
        },
    },
    util::{spread_bits, unspread_bits},
};

fn bit_mask(width: usize) -> Option<u128> {
    if width == 0 || width > 128 {
        None
    } else if width == 128 {
        Some(u128::MAX)
    } else {
        Some((1u128 << width) - 1)
    }
}

fn map_lookup_target(target: LookupTarget<Val>) -> LookupTarget<ValueId> {
    match target {
        LookupTarget::Rangecheck(bits) => LookupTarget::Rangecheck(bits),
        LookupTarget::DynRangecheck(bound) => LookupTarget::DynRangecheck(bound.0),
        LookupTarget::Array(array) => LookupTarget::Array(array.0),
        LookupTarget::Spread(bits) => LookupTarget::Spread(bits),
    }
}

pub struct Specializer {
    pub savings_to_code_ratio: f64,
}

#[derive(Debug, Clone)]
enum ConstVal {
    U(usize, u128),
    I(usize, u128),
    Field(Field),
    Array(Vec<ValueId>),
    Blob(Vec<ValueId>),
    BitsOf(Box<ValueId>, usize, Endianness),
}

fn const_val_as_field(value: &ConstVal) -> Option<Field> {
    match value {
        ConstVal::U(_, v) | ConstVal::I(_, v) => Some(Field::from(*v)),
        ConstVal::Field(f) => Some(*f),
        _ => None,
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
struct Val(ValueId);

struct SpecializationState<'a> {
    /// Shared reference to the SSA, used to mint fresh `ValueId`s via `fresh_value(&self)` while
    /// the symbolic executor holds its own shared `&HLSSA` borrow.
    ssa: &'a HLSSA,

    /// The candidate function body, mutated in place during symbolic execution. The candidate's
    /// `FunctionId` is owned by the caller. This is just the body slot, taken out of the SSA so it
    /// can be modified while the executor holds the SSA shared.
    body: HLFunction,

    /// Constant values created during specialization, usually by constant folding.
    const_vals: HashMap<ValueId, ConstVal>,
}

impl HLEmitter for SpecializationState<'_> {
    fn fresh_value(&mut self) -> ValueId {
        self.ssa.fresh_value()
    }

    fn emit(&mut self, op: OpCode) {
        let entry = self.body.get_entry_id();
        self.body.get_block_mut(entry).push_instruction(op);
    }

    fn emit_constant(&mut self, value: Constant) -> ValueId {
        self.ssa.add_const(value)
    }
}

impl symbolic_executor::Value<SpecializationState<'_>> for Val {
    fn ult(&self, b: &Self, ctx: &mut SpecializationState) -> Self {
        let l_const = ctx.const_vals.get(&self.0).cloned();
        let r_const = ctx.const_vals.get(&b.0).cloned();
        match (l_const, r_const) {
            (Some(ConstVal::U(_, l_val)), Some(ConstVal::U(_, r_val))) => {
                let res_u = if l_val < r_val { 1 } else { 0 };
                let res = ctx.u_const(1, res_u);
                ctx.const_vals.insert(res, ConstVal::U(1, res_u));
                Self(res)
            }
            (None, _) | (_, None) => {
                let res = ctx.cmp(self.0, b.0, CmpKind::Lt);
                Self(res)
            }
            _ => todo!(),
        }
    }

    fn slt(&self, b: &Self, _bits: usize, ctx: &mut SpecializationState) -> Self {
        let l_const = ctx.const_vals.get(&self.0).cloned();
        let r_const = ctx.const_vals.get(&b.0).cloned();
        match (l_const, r_const) {
            (Some(ConstVal::U(_, l_val)), Some(ConstVal::U(_, r_val))) => {
                let res_u = if l_val < r_val { 1 } else { 0 };
                let res = ctx.u_const(1, res_u);
                ctx.const_vals.insert(res, ConstVal::U(1, res_u));
                Self(res)
            }
            (None, _) | (_, None) => {
                let res = ctx.cmp(self.0, b.0, CmpKind::Lt);
                Self(res)
            }
            _ => todo!(),
        }
    }

    fn eq(&self, b: &Self, ctx: &mut SpecializationState) -> Self {
        let l_const = ctx.const_vals.get(&self.0).cloned();
        let r_const = ctx.const_vals.get(&b.0).cloned();
        match (l_const, r_const) {
            (Some(ConstVal::U(_, l_val)), Some(ConstVal::U(_, r_val))) => {
                let res_u = if l_val == r_val { 1 } else { 0 };
                let res = ctx.u_const(1, res_u);
                ctx.const_vals.insert(res, ConstVal::U(1, res_u));
                Self(res)
            }
            (None, _) | (_, None) => {
                let res = ctx.cmp(self.0, b.0, CmpKind::Eq);
                Self(res)
            }
            _ => todo!(),
        }
    }

    fn arith(
        &self,
        b: &Self,
        binary_arith_op_kind: BinaryArithOpKind,
        _out_type: &Type,
        ctx: &mut SpecializationState,
    ) -> Self {
        let a_const = ctx.const_vals.get(&self.0).cloned();
        let b_const = ctx.const_vals.get(&b.0).cloned();
        match (binary_arith_op_kind, a_const, b_const) {
            (BinaryArithOpKind::Add, Some(ConstVal::U(s, a_val)), Some(ConstVal::U(_, b_val))) => {
                let res = a_val + b_val;
                let res_v = ctx.u_const(s, res);
                ctx.const_vals.insert(res_v, ConstVal::U(s, res));
                Self(res_v)
            }
            (BinaryArithOpKind::Sub, Some(ConstVal::U(s, a_val)), Some(ConstVal::U(_, b_val))) => {
                let res = a_val - b_val;
                let res_v = ctx.u_const(s, res);
                ctx.const_vals.insert(res_v, ConstVal::U(s, res));
                Self(res_v)
            }
            (
                BinaryArithOpKind::Mul,
                Some(ConstVal::Field(l_val)),
                Some(ConstVal::Field(r_val)),
            ) => {
                let res = l_val * r_val;
                let res_v = ctx.field_const(res);
                ctx.const_vals.insert(res_v, ConstVal::Field(res));
                Self(res_v)
            }
            (BinaryArithOpKind::Sub, Some(ConstVal::Field(f)), Some(ConstVal::Field(f2))) => {
                let res = f - f2;
                let res_v = ctx.field_const(res);
                ctx.const_vals.insert(res_v, ConstVal::Field(res));
                Self(res_v)
            }
            (BinaryArithOpKind::Add, Some(ConstVal::Field(f)), Some(ConstVal::Field(f2))) => {
                let res = f + f2;
                let res_v = ctx.field_const(res);
                ctx.const_vals.insert(res_v, ConstVal::Field(res));
                Self(res_v)
            }

            (BinaryArithOpKind::Mul, Some(ConstVal::Field(f)), _) if f == ark_ff::Field::ONE => *b,
            (BinaryArithOpKind::Mul, _, Some(ConstVal::Field(f))) if f == ark_ff::Field::ONE => {
                *self
            }
            (BinaryArithOpKind::Mul, Some(ConstVal::Field(f)), _) if f == Field::ZERO => *self,
            (BinaryArithOpKind::Mul, _, Some(ConstVal::Field(f))) if f == Field::ZERO => *b,

            (BinaryArithOpKind::Mul, None, None) => {
                let res = ctx.mul(self.0, b.0);
                Self(res)
            }

            (BinaryArithOpKind::Add, Some(ConstVal::Field(f)), _) if f == Field::ZERO => *b,
            (BinaryArithOpKind::Add, _, Some(ConstVal::Field(f))) if f == Field::ZERO => *self,

            (BinaryArithOpKind::Add, _, _) => Self(ctx.add(self.0, b.0)),
            (BinaryArithOpKind::Sub, _, _) => Self(ctx.sub(self.0, b.0)),
            (BinaryArithOpKind::Mul, _, _) => Self(ctx.mul(self.0, b.0)),
            (BinaryArithOpKind::Div, _, _) => Self(ctx.div(self.0, b.0)),

            (BinaryArithOpKind::Mod, Some(ConstVal::U(s, a_val)), Some(ConstVal::U(_, b_val))) => {
                let res = a_val % b_val;
                let res_v = ctx.u_const(s, res);
                ctx.const_vals.insert(res_v, ConstVal::U(s, res));
                Self(res_v)
            }
            (BinaryArithOpKind::Mod, _, _) => Self(ctx.modulo(self.0, b.0)),

            (BinaryArithOpKind::And, Some(ConstVal::U(s, a_val)), Some(ConstVal::U(_, b_val))) => {
                let res = a_val & b_val;
                let res_v = ctx.u_const(s, res);
                ctx.const_vals.insert(res_v, ConstVal::U(s, res));
                Self(res_v)
            }

            (BinaryArithOpKind::And, _, None) | (BinaryArithOpKind::And, None, _) => {
                let res = ctx.and(self.0, b.0);
                Self(res)
            }

            (BinaryArithOpKind::Or, Some(ConstVal::U(s, a_val)), Some(ConstVal::U(_, b_val))) => {
                let res = a_val | b_val;
                let res_v = ctx.u_const(s, res);
                ctx.const_vals.insert(res_v, ConstVal::U(s, res));
                Self(res_v)
            }

            (BinaryArithOpKind::Or, _, _) => {
                let res = ctx.or(self.0, b.0);
                Self(res)
            }

            (BinaryArithOpKind::Xor, Some(ConstVal::U(s, a_val)), Some(ConstVal::U(_, b_val))) => {
                let res = a_val ^ b_val;
                let res_v = ctx.u_const(s, res);
                ctx.const_vals.insert(res_v, ConstVal::U(s, res));
                Self(res_v)
            }

            (BinaryArithOpKind::Xor, _, _) => {
                let res = ctx.xor(self.0, b.0);
                Self(res)
            }

            (BinaryArithOpKind::Shl, Some(ConstVal::U(s, a_val)), Some(ConstVal::U(_, b_val))) => {
                let res = a_val << b_val;
                let res_v = ctx.u_const(s, res);
                ctx.const_vals.insert(res_v, ConstVal::U(s, res));
                Self(res_v)
            }

            (BinaryArithOpKind::Shl, _, _) => {
                let res = ctx.shl(self.0, b.0);
                Self(res)
            }

            (BinaryArithOpKind::Shr, Some(ConstVal::U(s, a_val)), Some(ConstVal::U(_, b_val))) => {
                let res = a_val >> b_val;
                let res_v = ctx.u_const(s, res);
                ctx.const_vals.insert(res_v, ConstVal::U(s, res));
                Self(res_v)
            }

            (BinaryArithOpKind::Shr, _, _) => {
                let res = ctx.shr(self.0, b.0);
                Self(res)
            }

            (op, a, b) => panic!("Not yet implemented {:?} {:?}", op, (a, b)),
        }
    }

    fn assert_bool(&self, ctx: &mut SpecializationState) {
        let v_const = ctx.const_vals.get(&self.0);
        match v_const {
            Some(ConstVal::U(_, val)) => {
                assert!(*val != 0, "assert failed: value is zero");
            }
            None => {
                HLEmitter::assert_bool(ctx, self.0);
            }
            _ => panic!("Not yet implemented {:?}", v_const),
        }
    }

    fn assert_cmp(
        kind: CmpKind,
        a: &Self,
        b: &Self,
        _lhs_type: &Type,
        ctx: &mut SpecializationState,
    ) {
        let l_const = ctx.const_vals.get(&a.0);
        let r_const = ctx.const_vals.get(&b.0);
        match kind {
            CmpKind::Eq => match (l_const, r_const) {
                (Some(ConstVal::U(_, l_val)), Some(ConstVal::U(_, r_val))) => {
                    assert_eq!(l_val, r_val);
                }
                (None, _) | (_, None) => {
                    HLEmitter::assert_cmp(ctx, kind, a.0, b.0);
                }
                _ => panic!("Not yet implemented {:?}", (l_const, r_const)),
            },
            _ => {
                HLEmitter::assert_cmp(ctx, kind, a.0, b.0);
            }
        }
    }

    fn assert_r1c(a: &Self, b: &Self, c: &Self, ctx: &mut SpecializationState) {
        let a_const = ctx.const_vals.get(&a.0).and_then(const_val_as_field);
        let b_const = ctx.const_vals.get(&b.0).and_then(const_val_as_field);
        let c_const = ctx.const_vals.get(&c.0).and_then(const_val_as_field);
        match (a_const, b_const, c_const) {
            (Some(a), Some(b), Some(c)) => assert_eq!(a * b, c),
            _ => ctx.emit(OpCode::AssertR1C {
                a: a.0,
                b: b.0,
                c: c.0,
            }),
        }
    }

    fn array_get(&self, index: &Self, _out_type: &Type, ctx: &mut SpecializationState) -> Self {
        let a_const = ctx.const_vals.get(&self.0).cloned();
        let index_const = ctx.const_vals.get(&index.0).cloned();
        match (a_const, index_const) {
            (Some(ConstVal::Array(a) | ConstVal::Blob(a)), Some(ConstVal::U(_, index))) => {
                let res = a[index as usize];
                Self(res)
            }
            (Some(ConstVal::BitsOf(v, size, endianness)), Some(ConstVal::U(_, index))) => {
                let v_const = ctx.const_vals.get(v.as_ref()).cloned();
                match v_const {
                    Some(ConstVal::Field(f)) => {
                        let r = f.into_bigint().to_bits_le();
                        let ix = match endianness {
                            Endianness::Little => index as usize,
                            Endianness::Big => size - index as usize - 1,
                        };
                        let res = if r[ix] { 1 } else { 0 };
                        let res_v = ctx.u_const(1, res);
                        ctx.const_vals.insert(res_v, ConstVal::U(1, res));
                        Self(res_v)
                    }
                    _ => panic!("Not yet implemented {:?}", (v_const, endianness)),
                }
            }
            (None, _) | (_, None) => {
                let res = HLEmitter::array_get(ctx, self.0, index.0);
                Self(res)
            }
            (a, i) => panic!("Not yet implemented {:?}", (a, i)),
        }
    }

    fn array_set(
        &self,
        _index: &Self,
        _value: &Self,
        _out_type: &Type,
        _ctx: &mut SpecializationState,
    ) -> Self {
        todo!()
    }

    fn sext(
        &self,
        from: usize,
        to: usize,
        _out_type: &Type,
        ctx: &mut SpecializationState,
    ) -> Self {
        let self_const = ctx.const_vals.get(&self.0).cloned();
        match self_const {
            Some(ConstVal::U(_, v)) => {
                let sign_bit = if from > 0 { (v >> (from - 1)) & 1 } else { 0 };
                let res = if sign_bit == 1 {
                    let mask = bit_mask(to).unwrap() ^ bit_mask(from).unwrap();
                    v | mask
                } else {
                    v
                };
                let res_v = ctx.u_const(to, res);
                ctx.const_vals.insert(res_v, ConstVal::U(to, res));
                Self(res_v)
            }
            _ => {
                let res = ctx.sext(self.0, from, to);
                Self(res)
            }
        }
    }

    fn bit_range(
        &self,
        offset: usize,
        width: usize,
        out_type: &Type,
        ctx: &mut SpecializationState,
    ) -> Self {
        let self_const = ctx.const_vals.get(&self.0).cloned();
        let mask = bit_mask(width);
        match (self_const, &out_type.strip_witness().expr, mask) {
            (Some(ConstVal::U(bits, v)), _, Some(mask)) => {
                let res = (v >> offset) & mask;
                let res_v = ctx.u_const(bits, res);
                ctx.const_vals.insert(res_v, ConstVal::U(bits, res));
                Self(res_v)
            }
            (Some(ConstVal::I(bits, v)), _, Some(mask)) => {
                let res = (v >> offset) & mask;
                let res_v = ctx.i_const(bits, res);
                ctx.const_vals.insert(res_v, ConstVal::I(bits, res));
                Self(res_v)
            }
            (Some(ConstVal::Field(f)), TypeExpr::Field, Some(mask)) if offset < 128 => {
                let v: u128 = f.into_bigint().as_ref()[0] as u128;
                let res = Field::from((v >> offset) & mask);
                let res_v = ctx.field_const(res);
                ctx.const_vals.insert(res_v, ConstVal::Field(res));
                Self(res_v)
            }
            _ => {
                let res = ctx.bit_range(self.0, offset, width);
                Self(res)
            }
        }
    }

    fn cast(
        &self,
        cast_target: &CastTarget,
        _out_type: &Type,
        ctx: &mut SpecializationState,
    ) -> Self {
        let self_const = ctx.const_vals.get(&self.0).cloned();
        match self_const {
            Some(ConstVal::U(_, v)) => match cast_target {
                CastTarget::U(s) | CastTarget::I(s) => {
                    let res = v & bit_mask(*s).unwrap();
                    let res_v = ctx.u_const(*s, res);
                    ctx.const_vals.insert(res_v, ConstVal::U(*s, res));
                    Self(res_v)
                }
                CastTarget::Field => {
                    let res = Field::from(v);
                    let res_v = ctx.field_const(res);
                    ctx.const_vals.insert(res_v, ConstVal::Field(res));
                    Self(res_v)
                }
                CastTarget::Nop | CastTarget::ArrayToSlice | CastTarget::WitnessOf => *self,
            },
            Some(ConstVal::Field(f)) => match cast_target {
                CastTarget::U(s) | CastTarget::I(s) => {
                    let v: u128 = f.into_bigint().as_ref()[0] as u128;
                    let res = v & bit_mask(*s).unwrap();
                    let res_v = ctx.u_const(*s, res);
                    ctx.const_vals.insert(res_v, ConstVal::U(*s, res));
                    Self(res_v)
                }
                CastTarget::Field
                | CastTarget::Nop
                | CastTarget::ArrayToSlice
                | CastTarget::WitnessOf => *self,
            },
            None => {
                let res = ctx.cast_to(*cast_target, self.0);
                Self(res)
            }
            _ => {
                let res = ctx.cast_to(*cast_target, self.0);
                Self(res)
            }
        }
    }

    fn constrain(a: &Self, b: &Self, c: &Self, ctx: &mut SpecializationState) {
        let a_const = ctx.const_vals.get(&a.0).and_then(const_val_as_field);
        let b_const = ctx.const_vals.get(&b.0).and_then(const_val_as_field);
        let c_const = ctx.const_vals.get(&c.0).and_then(const_val_as_field);
        match (a_const, b_const, c_const) {
            (Some(a), Some(b), Some(c)) => assert_eq!(a * b, c),
            _ => HLEmitter::constrain(ctx, a.0, b.0, c.0),
        }
    }

    fn to_bits(
        &self,
        endianness: Endianness,
        size: usize,
        _out_type: &Type,
        ctx: &mut SpecializationState,
    ) -> Self {
        let val = ctx.to_bits(self.0, endianness, size);
        ctx.const_vals
            .insert(val, ConstVal::BitsOf(Box::new(self.0), size, endianness));
        Self(val)
    }

    fn not(&self, _out_type: &Type, ctx: &mut SpecializationState) -> Self {
        let const_val = ctx.const_vals.get(&self.0).cloned();
        match const_val {
            Some(ConstVal::U(s, v)) => {
                let res = !v & bit_mask(s).unwrap();
                let res_v = ctx.u_const(s, res);
                ctx.const_vals.insert(res_v, ConstVal::U(s, res));
                Self(res_v)
            }
            None => {
                let res = HLEmitter::not(ctx, self.0);
                Self(res)
            }
            _ => todo!(),
        }
    }

    fn of_u(s: usize, v: u128, ctx: &mut SpecializationState) -> Self {
        let val = ctx.u_const(s, v);
        ctx.const_vals.insert(val, ConstVal::U(s, v));
        Self(val)
    }

    fn of_i(s: usize, v: u128, ctx: &mut SpecializationState) -> Self {
        let val = ctx.i_const(s, v);
        ctx.const_vals.insert(val, ConstVal::I(s, v));
        Self(val)
    }

    fn of_field(f: Field, ctx: &mut SpecializationState) -> Self {
        let val = ctx.field_const(f);
        ctx.const_vals.insert(val, ConstVal::Field(f));
        Self(val)
    }

    fn of_blob(elem_type: Type, elements: Vec<Self>, ctx: &mut SpecializationState) -> Self {
        fn constant_for(ctx: &SpecializationState<'_>, value: ValueId, typ: &Type) -> Constant {
            match ctx
                .const_vals
                .get(&value)
                .unwrap_or_else(|| panic!("Blob element v{} is not a constant", value.0))
            {
                ConstVal::U(bits, value) => Constant::U(*bits, *value),
                ConstVal::I(bits, value) => Constant::I(*bits, *value),
                ConstVal::Field(value) => Constant::Field(*value),
                ConstVal::Blob(elements) => {
                    let inner = typ.get_array_element();
                    Constant::Blob(Blob::new(
                        inner.clone(),
                        elements
                            .iter()
                            .map(|element| constant_for(ctx, *element, &inner))
                            .collect(),
                    ))
                }
                other => panic!(
                    "Blob element v{} is not a scalar/blob constant: {:?}",
                    value.0, other
                ),
            }
        }

        let element_ids = elements.iter().map(|v| v.0).collect::<Vec<_>>();
        let constants = element_ids
            .iter()
            .map(|element| constant_for(ctx, *element, &elem_type))
            .collect();
        let val = ctx.emit_constant(Constant::Blob(Blob::new(elem_type, constants)));
        ctx.const_vals.insert(val, ConstVal::Blob(element_ids));
        Self(val)
    }

    fn expect_blob(&self, ctx: &mut SpecializationState) -> Vec<Self> {
        match ctx.const_vals.get(&self.0) {
            Some(ConstVal::Blob(elements)) => elements.iter().copied().map(Self).collect(),
            other => panic!("Expected blob, got {:?}", other),
        }
    }

    fn mk_array(
        a: Vec<Self>,
        ctx: &mut SpecializationState,
        seq_type: SequenceTargetType,
        elem_type: &Type,
    ) -> Self {
        let a = a.into_iter().map(|v| v.0).collect::<Vec<_>>();
        let val = ctx.mk_seq(a.clone(), seq_type, elem_type.clone());
        ctx.const_vals.insert(val, ConstVal::Array(a));
        Self(val)
    }

    fn alloc(elem_type: &Type, ctx: &mut SpecializationState) -> Self {
        let val = ctx.alloc(elem_type.clone());
        Self(val)
    }

    fn ptr_write(&self, val: &Self, ctx: &mut SpecializationState) {
        ctx.store(self.0, val.0);
    }

    fn ptr_read(&self, _out_type: &Type, ctx: &mut SpecializationState) -> Self {
        let val = ctx.load(self.0);
        Self(val)
    }

    fn expect_constant_bool(&self, ctx: &mut SpecializationState) -> bool {
        let val = ctx.const_vals.get(&self.0).unwrap();
        match val {
            ConstVal::U(_, v) => *v == 1,
            _ => todo!(),
        }
    }

    fn select(
        &self,
        if_t: &Self,
        if_f: &Self,
        _out_type: &Type,
        ctx: &mut SpecializationState,
    ) -> Self {
        let self_const = ctx.const_vals.get(&self.0);

        match self_const {
            Some(ConstVal::U(_, v)) => {
                let res = if *v == 1 { if_t.0 } else { if_f.0 };
                Self(res)
            }
            None => {
                let res = HLEmitter::select(ctx, self.0, if_t.0, if_f.0);
                Self(res)
            }
            _ => todo!(),
        }
    }

    fn write_witness(&self, tp: Option<&Type>, ctx: &mut SpecializationState) -> Self {
        if ctx.const_vals.contains_key(&self.0) {
            return *self;
        }
        match tp {
            Some(_) => Self(HLEmitter::write_witness(ctx, self.0)),
            None => {
                ctx.emit(OpCode::WriteWitness {
                    result: None,
                    value: self.0,
                    pinned: false,
                });
                *self
            }
        }
    }

    fn fresh_witness(result_type: &Type, ctx: &mut SpecializationState) -> Self {
        let result = ctx.fresh_value();
        ctx.emit(OpCode::FreshWitness {
            result,
            result_type: result_type.clone(),
        });
        Self(result)
    }

    fn value_of(&self, ctx: &mut SpecializationState) -> Self {
        let res = HLEmitter::value_of(ctx, self.0);
        Self(res)
    }

    fn mem_op(&self, kind: RefCountOp, ctx: &mut SpecializationState) {
        HLEmitter::mem_op(ctx, self.0, kind);
    }

    fn rangecheck(&self, max_bits: usize, ctx: &mut SpecializationState) {
        HLEmitter::rangecheck(ctx, self.0, max_bits);
    }

    fn spread(&self, bits: u8, ctx: &mut SpecializationState) -> Self {
        let cst_val = ctx.const_vals.get(&self.0);
        match cst_val {
            Some(ConstVal::U(b, v)) => {
                assert!(
                    *b <= 64,
                    "Spread only supports integer widths up to 64 bits, got u{}",
                    b
                );
                Self::of_u(b * 2, spread_bits(*v, *b), ctx)
            }
            Some(ConstVal::I(b, v)) => {
                assert!(
                    *b <= 64,
                    "Spread only supports integer widths up to 64 bits, got i{}",
                    b
                );
                Self::of_i(b * 2, spread_bits(*v, *b), ctx)
            }
            _ => {
                let res = HLEmitter::spread(ctx, self.0, bits);
                Self(res)
            }
        }
    }

    fn unspread(&self, bits: u8, ctx: &mut SpecializationState) -> (Self, Self) {
        let cst_val = ctx.const_vals.get(&self.0);
        match cst_val {
            Some(ConstVal::U(b, v)) => {
                assert!(
                    *b <= MAX_SUPPORTED_UNSIGNED_BITS && b % 2 == 0,
                    "Unspread expects an even integer width up to {MAX_SUPPORTED_UNSIGNED_BITS} bits, got u{}",
                    b
                );
                let half_bits = b / 2;
                let (odd, even) = unspread_bits(*v, *b);
                (
                    Self::of_u(half_bits, odd, ctx),
                    Self::of_u(half_bits, even, ctx),
                )
            }
            Some(ConstVal::I(b, v)) => {
                assert!(
                    *b <= MAX_SUPPORTED_UNSIGNED_BITS && b % 2 == 0,
                    "Unspread expects an even integer width up to {MAX_SUPPORTED_UNSIGNED_BITS} bits, got i{}",
                    b
                );
                let half_bits = b / 2;
                let (odd, even) = unspread_bits(*v, *b);
                (
                    Self::of_i(half_bits, odd, ctx),
                    Self::of_i(half_bits, even, ctx),
                )
            }
            _ => {
                let (res_and, res_xor) = HLEmitter::unspread(ctx, self.0, bits);
                (Self(res_and), Self(res_xor))
            }
        }
    }

    fn to_radix(
        &self,
        radix: &Radix<Self>,
        endianness: Endianness,
        size: usize,
        _out_type: &Type,
        ctx: &mut SpecializationState,
    ) -> Self {
        let cst_val = ctx.const_vals.get(&self.0);
        match cst_val {
            None => {
                let radix = match radix {
                    Radix::Dyn(v) => Radix::Dyn(v.0),
                    Radix::Bytes => Radix::Bytes,
                };
                let res = HLEmitter::to_radix(ctx, self.0, radix, endianness, size);
                Self(res)
            }
            Some(_) => todo!(),
        }
    }
}

impl symbolic_executor::Context<Val> for SpecializationState<'_> {
    fn on_call(
        &mut self,
        func: FunctionId,
        params: &mut [Val],
        _param_types: &[&Type],
        result_types: &[Type],
        unconstrained: bool,
    ) -> Option<Vec<Val>> {
        if unconstrained {
            // Emit the unconstrained call as-is into the specialized function
            let args: Vec<ValueId> = params.iter().map(|v| v.0).collect();
            let n = result_types.len();
            let results = self.call_unconstrained(func, args, n);
            return Some(results.into_iter().map(Val).collect());
        }
        None
    }

    fn on_return(&mut self, returns: &mut [Val], _return_types: &[Type]) {
        self.body.terminate_block_with_return(
            self.body.get_entry_id(),
            returns.iter().map(|v| v.0).collect(),
        );
    }

    fn on_jmp(&mut self, _target: BlockId, _params: &mut [Val], _param_types: &[&Type]) {}

    fn lookup(&mut self, target: LookupTarget<Val>, args: Vec<Val>, flag: Val) {
        self.emit(OpCode::Lookup {
            target: map_lookup_target(target),
            args: args.into_iter().map(|arg| arg.0).collect(),
            flag: flag.0,
        });
    }

    fn dlookup(&mut self, target: LookupTarget<Val>, args: Vec<Val>, flag: Val) {
        self.emit(OpCode::DLookup {
            target: map_lookup_target(target),
            args: args.into_iter().map(|arg| arg.0).collect(),
            flag: flag.0,
        });
    }

    fn todo(&mut self, payload: &str, _result_types: &[Type]) -> Vec<Val> {
        todo!("Todo opcode: {}", payload);
    }

    fn slice_len(&mut self, slice: &Val) -> Val {
        if let Some(ConstVal::Array(elements)) = self.const_vals.get(&slice.0) {
            let len = elements.len() as u128;
            let val = self.u_const(32, len);
            self.const_vals.insert(val, ConstVal::U(32, len));
            Val(val)
        } else {
            let val = HLEmitter::slice_len(self, slice.0);
            Val(val)
        }
    }

    fn on_guard(
        &mut self,
        inner: &OpCode,
        condition: &Val,
        inputs: Vec<&Val>,
        _result_types: Vec<&Type>,
    ) -> Vec<Val> {
        use crate::compiler::ssa::Instruction;

        // Build a mapping from old ValueIds to new ValueIds
        let orig_inputs: Vec<_> = inner.get_inputs().cloned().collect();
        let orig_results: Vec<_> = inner.get_results().cloned().collect();
        let mut id_map: HashMap<ValueId, ValueId> = HashMap::new();
        for (orig, new_val) in orig_inputs.iter().zip(inputs.iter()) {
            id_map.insert(*orig, new_val.0);
        }
        let mut result_vals = Vec::new();
        for orig_result in &orig_results {
            let fresh = self.fresh_value();
            id_map.insert(*orig_result, fresh);
            result_vals.push(Val(fresh));
        }
        // Clone and remap all operands
        let mut new_inner = inner.clone();
        for op in new_inner.get_operands_mut() {
            if let Some(new_id) = id_map.get(op) {
                *op = *new_id;
            }
        }
        self.emit(OpCode::Guard {
            condition: condition.0,
            inner: Box::new(new_inner),
        });
        result_vals
    }
}

impl Pass for Specializer {
    fn name(&self) -> &'static str {
        "specializer"
    }

    fn needs(&self) -> Vec<AnalysisId> {
        vec![Summary::id(), TypeInfo::id()]
    }

    fn run(&self, ssa: &mut HLSSA, store: &AnalysisStore) {
        let summary = store.get::<Summary>();
        let mut speculative_ids: HashSet<FunctionId> = HashSet::new();
        let mut accepted_ids: HashSet<FunctionId> = HashSet::new();

        for (sig, summary) in summary.functions.iter() {
            if summary.specialization_total_savings > 0 {
                self.try_spec(
                    ssa,
                    store.get::<TypeInfo>(),
                    summary,
                    sig.clone(),
                    &mut speculative_ids,
                    &mut accepted_ids,
                );
            }
        }

        // Drop any speculative candidate (and its `#unspecialized` clone, if it had one)
        // that wasn't accepted. Constants the rejected candidates left behind become
        // unreferenced and are cleaned up by the DCE pass that runs immediately after the
        // specializer.
        ssa.retain_functions(|id, _| !speculative_ids.contains(&id) || accepted_ids.contains(&id));
    }
}

impl Specializer {
    pub fn new(savings_to_code_ratio: f64) -> Self {
        Self {
            savings_to_code_ratio,
        }
    }

    #[instrument(skip_all, name = "Specializer::try_spec", fields(function = %signature.pretty_print(ssa, true), expected_savings = summary.specialization_total_savings))]
    fn try_spec(
        &self,
        ssa: &mut HLSSA,
        type_info: &TypeInfo,
        summary: &SpecializationSummary,
        signature: FunctionSignature,
        speculative_ids: &mut HashSet<FunctionId>,
        accepted_ids: &mut HashSet<FunctionId>,
    ) {
        let name = signature.pretty_print(ssa, true);

        if summary.specialization_total_savings as f64 / self.savings_to_code_ratio < 10.0 {
            info!(
                message = %"Specialization rejected, would need less than 10 codesize to be worth it",
                specialization = %name,
                saved_constraints = summary.specialization_total_savings,
                savings_to_code_ratio = self.savings_to_code_ratio
            );
            return;
        }

        // Snapshot what we need from the original before any mutation: param types, return
        // types, and its name (for the #specialized / #unspecialized derived names).
        let original_param_types: Vec<Type>;
        let original_return_types: Vec<Type>;
        let original_name: String;
        {
            let original_fn = ssa.get_function(signature.get_fun_id());
            original_param_types = original_fn.get_param_types();
            original_return_types = original_fn.get_returns().to_vec();
            original_name = original_fn.get_name().to_string();
        }

        // Mint the candidate's FunctionId up-front. The empty body lives in the SSA at this
        // id until we put the filled-in one back. Track it as speculative so the end-of-pass
        // cleanup drops it if the specialization is ultimately rejected.
        let candidate_id = ssa.add_function(name.clone());
        speculative_ids.insert(candidate_id);

        // Take the empty body out so the state can mutate it while the symbolic executor
        // holds a shared `&HLSSA`.
        let mut body = ssa.take_function(candidate_id);
        for ret in &original_return_types {
            body.add_return_type(ret.clone());
        }

        // Build call params and the initial `const_vals` map. Routes through `add_const`
        // (still `&mut ssa` here) so the constants for `Field`/`U`/`I` signature params are
        // interned eagerly.
        let mut call_params: Vec<Val> = vec![];
        let mut const_vals: HashMap<ValueId, ConstVal> = HashMap::new();
        for (param, sig) in original_param_types
            .iter()
            .zip(signature.get_params().iter())
        {
            match sig {
                ValueSignature::PointerTo(_) => {
                    info!("TODO: Aborting specialization on a pointer value");
                    return;
                }
                ValueSignature::Array(_) => {
                    info!("TODO: Aborting specialization on an array value");
                    return;
                }
                ValueSignature::Blob(_) => {
                    info!("TODO: Aborting specialization on a blob value");
                    return;
                }
                ValueSignature::Unknown(_)
                | ValueSignature::UnknownSlice
                | ValueSignature::WitnessOf(_) => {
                    let id = ssa.fresh_value();
                    body.get_entry_mut().push_parameter(id, param.clone());
                    call_params.push(Val(id));
                }
                ValueSignature::Field(f) => {
                    let val = ssa.add_const(Constant::Field(*f));
                    call_params.push(Val(val));
                    const_vals.insert(val, ConstVal::Field(*f));
                }
                ValueSignature::U { bits_size, value } => {
                    let val = ssa.add_const(Constant::U(*bits_size, *value));
                    call_params.push(Val(val));
                    const_vals.insert(val, ConstVal::U(*bits_size, *value));
                }
                ValueSignature::I { bits_size, value } => {
                    let val = ssa.add_const(Constant::I(*bits_size, *value));
                    call_params.push(Val(val));
                    const_vals.insert(val, ConstVal::I(*bits_size, *value));
                }
            }
        }

        let body = {
            let mut state = SpecializationState {
                ssa: &*ssa,
                body,
                const_vals,
            };

            SymbolicExecutor::new().run(
                &*ssa,
                type_info,
                signature.get_fun_id(),
                call_params,
                &mut state,
            );

            state.body
        };

        let code_bloat = body.code_size();
        let savings_to_code_ratio = summary.specialization_total_savings as f64 / code_bloat as f64;

        // Put the body back unconditionally. On rejection it stays at `candidate_id` only
        // until the end-of-pass `retain_functions` call drops it.
        ssa.put_function(candidate_id, body);

        if savings_to_code_ratio > self.savings_to_code_ratio {
            info!(message = %"Specialization accepted", code_bloat = code_bloat,  savings_to_code_ratio = savings_to_code_ratio, threshold_ratio = self.savings_to_code_ratio);

            // Clone the original via the SSA helper. The clone has fresh `ValueId`s and
            // becomes the dispatcher's fallback target; the original's slot is then
            // overwritten with the dispatcher itself.
            let unspecialized_id = ssa.duplicate_function(signature.get_fun_id());
            ssa.get_function_mut(unspecialized_id)
                .set_name(format!("{}#unspecialized", original_name));

            let dispatcher = self.build_dispatcher_for(
                ssa,
                original_param_types,
                original_return_types,
                &signature,
                format!("{}#specialized", original_name),
                candidate_id,
                unspecialized_id,
            );
            *ssa.get_function_mut(signature.get_fun_id()) = dispatcher;

            accepted_ids.insert(candidate_id);
            accepted_ids.insert(unspecialized_id);
        } else {
            info!(message = %"Specialization rejected", code_bloat = code_bloat,  savings_to_code_ratio = savings_to_code_ratio, threshold_ratio = self.savings_to_code_ratio);
        }
    }

    fn build_dispatcher_for(
        &self,
        ssa: &mut HLSSA,
        params: Vec<Type>,
        returns: Vec<Type>,
        signature: &FunctionSignature,
        fn_name: String,
        specialized_id: FunctionId,
        unspecialized_id: FunctionId,
    ) -> HLFunction {
        let mut dispatcher = HLFunction::empty(fn_name);
        let entry_block = dispatcher.get_entry_id();

        let mut b = HLFunctionBuilder::new(&mut dispatcher, ssa);

        let mut dispatcher_params = vec![];
        {
            let mut entry = b.block(entry_block);
            for param in params {
                dispatcher_params.push(entry.add_parameter(param));
            }
        }

        for return_type in returns.iter() {
            b.function().add_return_type(return_type.clone());
        }

        let mut specialized_params = vec![];
        let should_call_spec;
        {
            let mut entry = b.block(entry_block);
            let mut cond = entry.u_const(1, 1);

            for (pval, psig) in dispatcher_params.iter().zip(signature.get_params().iter()) {
                match psig {
                    ValueSignature::PointerTo(_) => {
                        unreachable!(
                            "ICE: pointer specializations are rejected before dispatcher generation"
                        );
                    }
                    ValueSignature::Array(_) => {
                        unreachable!(
                            "ICE: array specializations are rejected before dispatcher generation"
                        );
                    }
                    ValueSignature::Blob(_) => {
                        unreachable!(
                            "ICE: blob specializations are rejected before dispatcher generation"
                        );
                    }
                    ValueSignature::Unknown(_)
                    | ValueSignature::UnknownSlice
                    | ValueSignature::WitnessOf(_) => {
                        specialized_params.push(*pval);
                    }
                    ValueSignature::Field(v) => {
                        let cst = entry.field_const(*v);
                        let is_eq = entry.eq(*pval, cst);
                        cond = entry.and(cond, is_eq);
                    }
                    ValueSignature::U { bits_size, value } => {
                        let cst = entry.u_const(*bits_size, *value);
                        let is_eq = entry.eq(*pval, cst);
                        cond = entry.and(cond, is_eq);
                    }
                    ValueSignature::I { bits_size, value } => {
                        let cst = entry.i_const(*bits_size, *value);
                        let is_eq = entry.eq(*pval, cst);
                        cond = entry.and(cond, is_eq);
                    }
                }
            }
            should_call_spec = cond;
        }

        let specialized_caller = b.add_block(|_| {});
        let unspecialized_caller = b.add_block(|_| {});

        let mut return_values = vec![];
        let return_block = b.add_block(|ret| {
            for r in returns {
                return_values.push(ret.add_parameter(r));
            }
            ret.terminate_return(return_values.clone());
        });

        {
            let mut cb = b.block(unspecialized_caller);
            let unspecialized_returns =
                cb.call(unspecialized_id, dispatcher_params, return_values.len());
            cb.terminate_jmp(return_block, unspecialized_returns);
        }

        {
            let mut cb = b.block(specialized_caller);
            let specialized_returns =
                cb.call(specialized_id, specialized_params, return_values.len());
            cb.terminate_jmp(return_block, specialized_returns);
        }

        b.block(entry_block).terminate_jmp_if(
            should_call_spec,
            specialized_caller,
            unspecialized_caller,
        );

        dispatcher
    }
}
