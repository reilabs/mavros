use std::collections::HashMap;

use ark_ff::{AdditiveGroup, BigInteger, PrimeField};
use tracing::{info, instrument};

use crate::compiler::{
    Field,
    analysis::{
        instrumenter::{FunctionSignature, SpecializationSummary, Summary, ValueSignature},
        symbolic_executor::{self, SymbolicExecutor},
        types::TypeInfo,
    },
    block_builder::{HLEmitter, HLFunctionBuilder},
    ir::r#type::Type,
    pass_manager::{Analysis, AnalysisId, AnalysisStore, Pass},
    ssa::{
        BinaryArithOpKind, CastTarget, Endianness, FunctionId, HLFunction, HLSSA, MemOp, OpCode,
        Radix, SeqType, ValueId,
    },
};

pub struct Specializer {
    pub savings_to_code_ratio: f64,
}

#[derive(Debug, Clone)]
enum ConstVal {
    U(usize, u128),
    I(usize, u128),
    Field(Field),
    Array(Vec<ValueId>),
    Tuple(Vec<ValueId>),
    BitsOf(Box<ValueId>, usize, Endianness),
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
struct Val(ValueId);

struct SpecializationState {
    function: HLFunction,
    const_vals: HashMap<ValueId, ConstVal>,
}

impl HLEmitter for SpecializationState {
    fn fresh_value(&mut self) -> ValueId {
        self.function.fresh_value()
    }

    fn emit(&mut self, op: OpCode) {
        let entry = self.function.get_entry_id();
        self.function.get_block_mut(entry).push_instruction(op);
    }
}

impl symbolic_executor::Value<SpecializationState> for Val {
    fn cmp(
        &self,
        b: &Self,
        cmp_kind: crate::compiler::ssa::CmpKind,
        _out_type: &crate::compiler::ir::r#type::Type,
        ctx: &mut SpecializationState,
    ) -> Self {
        let l_const = ctx.const_vals.get(&self.0).cloned();
        let r_const = ctx.const_vals.get(&b.0).cloned();
        match (l_const, r_const) {
            (Some(ConstVal::U(_, l_val)), Some(ConstVal::U(_, r_val))) => match cmp_kind {
                crate::compiler::ssa::CmpKind::Lt => {
                    let res_u = if l_val < r_val { 1 } else { 0 };
                    let res = ctx.u_const(1, res_u);
                    ctx.const_vals.insert(res, ConstVal::U(1, res_u));
                    Self(res)
                }
                crate::compiler::ssa::CmpKind::Eq => {
                    let res_u = if l_val == r_val { 1 } else { 0 };
                    let res = ctx.u_const(1, res_u);
                    ctx.const_vals.insert(res, ConstVal::U(1, res_u));
                    Self(res)
                }
            },
            (None, _) | (_, None) => {
                let res = ctx.cmp(self.0, b.0, cmp_kind);
                Self(res)
            }
            _ => todo!(),
        }
    }

    fn arith(
        &self,
        b: &Self,
        binary_arith_op_kind: crate::compiler::ssa::BinaryArithOpKind,
        _out_type: &crate::compiler::ir::r#type::Type,
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

            (BinaryArithOpKind::Mod, Some(ConstVal::U(s, a_val)), Some(ConstVal::U(_, b_val))) => {
                let res = a_val % b_val;
                let res_v = ctx.u_const(s, res);
                ctx.const_vals.insert(res_v, ConstVal::U(s, res));
                Self(res_v)
            }
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

    fn assert_eq(&self, other: &Self, ctx: &mut SpecializationState) {
        let l_const = ctx.const_vals.get(&self.0);
        let r_const = ctx.const_vals.get(&other.0);
        match (l_const, r_const) {
            (Some(ConstVal::U(_, l_val)), Some(ConstVal::U(_, r_val))) => {
                assert_eq!(l_val, r_val);
            }
            (None, _) | (_, None) => {
                HLEmitter::assert_eq(ctx, self.0, other.0);
            }
            _ => panic!("Not yet implemented {:?}", (l_const, r_const)),
        }
    }

    fn assert_r1c(_a: &Self, _b: &Self, _c: &Self, _ctx: &mut SpecializationState) {
        todo!()
    }

    fn array_get(
        &self,
        index: &Self,
        _out_type: &crate::compiler::ir::r#type::Type,
        ctx: &mut SpecializationState,
    ) -> Self {
        let a_const = ctx.const_vals.get(&self.0).cloned();
        let index_const = ctx.const_vals.get(&index.0).cloned();
        match (a_const, index_const) {
            (Some(ConstVal::Array(a)), Some(ConstVal::U(_, index))) => {
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

    fn tuple_get(
        &self,
        index: usize,
        _out_type: &crate::compiler::ir::r#type::Type,
        ctx: &mut SpecializationState,
    ) -> Self {
        let a_const = ctx.const_vals.get(&self.0);
        match a_const {
            Some(ConstVal::Tuple(a)) => {
                let res = a[index as usize];
                Self(res)
            }
            _ => panic!("Not yet implemented {:?}", a_const),
        }
    }

    fn array_set(
        &self,
        _index: &Self,
        _value: &Self,
        _out_type: &crate::compiler::ir::r#type::Type,
        _ctx: &mut SpecializationState,
    ) -> Self {
        todo!()
    }

    fn truncate(
        &self,
        from: usize,
        to: usize,
        _out_type: &crate::compiler::ir::r#type::Type,
        ctx: &mut SpecializationState,
    ) -> Self {
        let self_const = ctx.const_vals.get(&self.0).cloned();
        match self_const {
            Some(ConstVal::U(_, v)) => {
                let res = v & ((1 << to) - 1);
                let res_v = ctx.u_const(to, res);
                ctx.const_vals.insert(res_v, ConstVal::U(to, res));
                Self(res_v)
            }
            Some(ConstVal::Field(f)) => {
                let v: u128 = f.into_bigint().as_ref()[0] as u128;
                let res = v & ((1 << to) - 1);
                let res_v = ctx.u_const(to, res);
                ctx.const_vals.insert(res_v, ConstVal::U(to, res));
                Self(res_v)
            }
            _ => {
                let res = ctx.truncate(self.0, to, from);
                Self(res)
            }
        }
    }

    fn sext(
        &self,
        from: usize,
        to: usize,
        _out_type: &crate::compiler::ir::r#type::Type,
        ctx: &mut SpecializationState,
    ) -> Self {
        let self_const = ctx.const_vals.get(&self.0).cloned();
        match self_const {
            Some(ConstVal::U(_, v)) => {
                let sign_bit = if from > 0 { (v >> (from - 1)) & 1 } else { 0 };
                let res = if sign_bit == 1 {
                    let mask = ((1u128 << to) - 1) ^ ((1u128 << from) - 1);
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

    fn cast(
        &self,
        cast_target: &crate::compiler::ssa::CastTarget,
        _out_type: &crate::compiler::ir::r#type::Type,
        ctx: &mut SpecializationState,
    ) -> Self {
        let self_const = ctx.const_vals.get(&self.0).cloned();
        match self_const {
            Some(ConstVal::U(_, v)) => match cast_target {
                CastTarget::U(s) | CastTarget::I(s) => {
                    let res = v & ((1 << *s) - 1);
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
                CastTarget::Nop | CastTarget::ArrayToSlice | CastTarget::WitnessOf => self.clone(),
            },
            Some(ConstVal::Field(f)) => match cast_target {
                CastTarget::U(s) | CastTarget::I(s) => {
                    let v: u128 = f.into_bigint().as_ref()[0] as u128;
                    let res = v & ((1 << *s) - 1);
                    let res_v = ctx.u_const(*s, res);
                    ctx.const_vals.insert(res_v, ConstVal::U(*s, res));
                    Self(res_v)
                }
                CastTarget::Field
                | CastTarget::Nop
                | CastTarget::ArrayToSlice
                | CastTarget::WitnessOf => self.clone(),
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

    fn constrain(_a: &Self, _b: &Self, _c: &Self, _ctx: &mut SpecializationState) {
        todo!()
    }

    fn to_bits(
        &self,
        endianness: Endianness,
        size: usize,
        _out_type: &crate::compiler::ir::r#type::Type,
        ctx: &mut SpecializationState,
    ) -> Self {
        let val = ctx.to_bits(self.0, endianness, size);
        ctx.const_vals
            .insert(val, ConstVal::BitsOf(Box::new(self.0), size, endianness));
        Self(val)
    }

    fn not(
        &self,
        _out_type: &crate::compiler::ir::r#type::Type,
        ctx: &mut SpecializationState,
    ) -> Self {
        let const_val = ctx.const_vals.get(&self.0).cloned();
        match const_val {
            Some(ConstVal::U(s, v)) => {
                let res = !v & ((1 << s) - 1);
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

    fn mk_array(
        a: Vec<Self>,
        ctx: &mut SpecializationState,
        seq_type: SeqType,
        elem_type: &Type,
    ) -> Self {
        let a = a.into_iter().map(|v| v.0).collect::<Vec<_>>();
        let val = ctx.mk_seq(a.clone(), seq_type, elem_type.clone());
        ctx.const_vals.insert(val, ConstVal::Array(a));
        Self(val)
    }

    fn mk_tuple(elems: Vec<Self>, ctx: &mut SpecializationState, elem_types: &[Type]) -> Self {
        let a = elems.into_iter().map(|v| v.0).collect::<Vec<_>>();
        let val = ctx.mk_tuple(a.clone(), elem_types.to_vec());
        ctx.const_vals.insert(val, ConstVal::Tuple(a));
        Self(val)
    }

    fn alloc(elem_type: &Type, ctx: &mut SpecializationState) -> Self {
        let val = ctx.alloc(elem_type.clone());
        Self(val)
    }

    fn ptr_write(&self, val: &Self, ctx: &mut SpecializationState) {
        ctx.store(self.0, val.0);
    }

    fn ptr_read(
        &self,
        _out_type: &crate::compiler::ir::r#type::Type,
        ctx: &mut SpecializationState,
    ) -> Self {
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
        _out_type: &crate::compiler::ir::r#type::Type,
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

    fn write_witness(
        &self,
        _tp: Option<&crate::compiler::ir::r#type::Type>,
        _ctx: &mut SpecializationState,
    ) -> Self {
        todo!()
    }

    fn fresh_witness(
        _result_type: &crate::compiler::ir::r#type::Type,
        _ctx: &mut SpecializationState,
    ) -> Self {
        todo!()
    }

    fn value_of(&self, ctx: &mut SpecializationState) -> Self {
        let res = HLEmitter::value_of(ctx, self.0);
        Self(res)
    }

    fn mem_op(&self, kind: MemOp, ctx: &mut SpecializationState) {
        HLEmitter::mem_op(ctx, self.0, kind);
    }

    fn rangecheck(&self, max_bits: usize, ctx: &mut SpecializationState) {
        HLEmitter::rangecheck(ctx, self.0, max_bits);
    }

    fn to_radix(
        &self,
        radix: &Radix<Self>,
        endianness: Endianness,
        size: usize,
        _out_type: &crate::compiler::ir::r#type::Type,
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

impl symbolic_executor::Context<Val> for SpecializationState {
    fn on_call(
        &mut self,
        func: crate::compiler::ssa::FunctionId,
        params: &mut [Val],
        _param_types: &[&crate::compiler::ir::r#type::Type],
        result_types: &[crate::compiler::ir::r#type::Type],
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

    fn on_return(
        &mut self,
        returns: &mut [Val],
        _return_types: &[crate::compiler::ir::r#type::Type],
    ) {
        self.function.terminate_block_with_return(
            self.function.get_entry_id(),
            returns.iter().map(|v| v.0).collect(),
        );
    }

    fn on_jmp(
        &mut self,
        _target: crate::compiler::ssa::BlockId,
        _params: &mut [Val],
        _param_types: &[&crate::compiler::ir::r#type::Type],
    ) {
    }

    fn todo(
        &mut self,
        payload: &str,
        _result_types: &[crate::compiler::ir::r#type::Type],
    ) -> Vec<Val> {
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
        inner: &crate::compiler::ssa::OpCode,
        condition: &Val,
        inputs: Vec<&Val>,
        _result_types: Vec<&crate::compiler::ir::r#type::Type>,
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
        for (sig, summary) in summary.functions.iter() {
            if summary.specialization_total_savings > 0 {
                self.try_spec(ssa, store.get::<TypeInfo>(), summary, sig.clone());
            }
        }
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

        let original_fn = ssa.get_function(signature.get_fun_id());

        let mut state = SpecializationState {
            function: HLFunction::empty(name),
            const_vals: HashMap::new(),
        };

        let mut call_params: Vec<Val> = vec![];

        for (param, sig) in original_fn
            .get_param_types()
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
                ValueSignature::Unknown(_) | ValueSignature::WitnessOf(_) => {
                    call_params.push(Val(state
                        .function
                        .add_parameter(state.function.get_entry_id(), param.clone())));
                }
                ValueSignature::Field(f) => {
                    let val = state.field_const(*f);
                    call_params.push(Val(val));
                    state.const_vals.insert(val, ConstVal::Field(*f));
                }
                ValueSignature::U(size, v) => {
                    let val = state.u_const(*size, *v);
                    call_params.push(Val(val));
                    state.const_vals.insert(val, ConstVal::U(*size, *v));
                }
                ValueSignature::I(size, v) => {
                    let val = state.i_const(*size, *v);
                    call_params.push(Val(val));
                    state.const_vals.insert(val, ConstVal::I(*size, *v));
                }
                ValueSignature::Tuple(_) => {
                    info!("TODO: Aborting specialization on a tuple value");
                    return;
                }
            }
        }

        for ret in original_fn.get_returns() {
            state.function.add_return_type(ret.clone());
        }

        SymbolicExecutor::new().run(
            ssa,
            type_info,
            signature.get_fun_id(),
            call_params,
            &mut state,
        );

        let code_bloat = state.function.code_size();
        let savings_to_code_ratio = summary.specialization_total_savings as f64 / code_bloat as f64;

        if savings_to_code_ratio > self.savings_to_code_ratio {
            info!(message = %"Specialization accepted", code_bloat = code_bloat,  savings_to_code_ratio = savings_to_code_ratio, threshold_ratio = self.savings_to_code_ratio);
            let original_fn = ssa.take_function(signature.get_fun_id());
            let new_fn_id = ssa.add_function("".to_string());
            let new_original_id = ssa.add_function("".to_string()); // Temporary

            let original_params = original_fn.get_param_types();
            let original_returns = original_fn.get_returns().to_vec();

            let dispatcher = self.build_dispatcher_for(
                original_params,
                original_returns,
                &signature,
                original_fn.get_name().to_string() + "#specialized",
                new_fn_id,
                new_original_id,
            );
            ssa.put_function(new_original_id, original_fn);
            ssa.put_function(signature.get_fun_id(), dispatcher);
            ssa.put_function(new_fn_id, state.function);
        } else {
            // TODO: run some passes to see if it decreases
            info!(message = %"Specialization rejected", code_bloat = code_bloat,  savings_to_code_ratio = savings_to_code_ratio, threshold_ratio = self.savings_to_code_ratio);
        }
    }

    fn build_dispatcher_for(
        &self,
        params: Vec<Type>,
        returns: Vec<Type>,
        signature: &FunctionSignature,
        fn_name: String,
        specialized_id: FunctionId,
        unspecialized_id: FunctionId,
    ) -> HLFunction {
        let mut dispatcher = HLFunction::empty(fn_name);
        let entry_block = dispatcher.get_entry_id();

        let mut b = HLFunctionBuilder::new(&mut dispatcher);

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
                        todo!();
                    }
                    ValueSignature::Array(_) => {
                        todo!();
                    }
                    ValueSignature::Unknown(_) | ValueSignature::WitnessOf(_) => {
                        specialized_params.push(*pval);
                    }
                    ValueSignature::Field(v) => {
                        let cst = entry.field_const(*v);
                        let is_eq = entry.eq(*pval, cst);
                        cond = entry.and(cond, is_eq);
                    }
                    ValueSignature::U(s, v) => {
                        let cst = entry.u_const(*s, *v);
                        let is_eq = entry.eq(*pval, cst);
                        cond = entry.and(cond, is_eq);
                    }
                    ValueSignature::I(s, v) => {
                        let cst = entry.i_const(*s, *v);
                        let is_eq = entry.eq(*pval, cst);
                        cond = entry.and(cond, is_eq);
                    }
                    ValueSignature::Tuple(_) => {
                        todo!();
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
