use std::collections::HashMap;

use ark_ff::{AdditiveGroup, BigInteger, PrimeField};
use tracing::{info, instrument};

use crate::compiler::{
    Field,
    analysis::{
        instrumenter::{FunctionSignature, SpecializationSummary, ValueSignature},
        symbolic_executor::{self, SymbolicExecutor},
        types::TypeInfo,
    },
    ir::r#type::Type,
    pass_manager::{DataPoint, Pass, PassInfo},
    ssa::{
        BinaryArithOpKind, CastTarget, Endianness, Function, FunctionId, MemOp, Radix, SSA,
        SeqType, ValueId,
    },
    taint_analysis::ConstantTaint,
};

pub struct Specializer {
    pub savings_to_code_ratio: f64,
}

#[derive(Debug)]
enum ConstVal {
    U(usize, u128),
    Field(Field),
    Array(Vec<ValueId>),
    Tuple(Vec<ValueId>),
    BitsOf(Box<ValueId>, usize, Endianness),
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
struct Val(ValueId);

struct SpecializationState {
    function: Function<ConstantTaint>,
    const_vals: HashMap<ValueId, ConstVal>,
}

impl symbolic_executor::Value<SpecializationState, ConstantTaint> for Val {
    fn cmp(
        &self,
        b: &Self,
        cmp_kind: crate::compiler::ssa::CmpKind,
        _out_type: &crate::compiler::ir::r#type::Type<ConstantTaint>,
        ctx: &mut SpecializationState,
    ) -> Self {
        let l_const = ctx.const_vals.get(&self.0);
        let r_const = ctx.const_vals.get(&b.0);
        match (l_const, r_const) {
            (Some(ConstVal::U(_, l_val)), Some(ConstVal::U(_, r_val))) => match cmp_kind {
                crate::compiler::ssa::CmpKind::Lt => {
                    let res_u = if l_val < r_val { 1 } else { 0 };
                    let res = ctx.function.push_u_const(1, res_u);
                    ctx.const_vals.insert(res, ConstVal::U(1, res_u));
                    Self(res)
                }
                crate::compiler::ssa::CmpKind::Eq => {
                    let res_u = if l_val == r_val { 1 } else { 0 };
                    let res = ctx.function.push_u_const(1, res_u);
                    ctx.const_vals.insert(res, ConstVal::U(1, res_u));
                    Self(res)
                }
            },
            (None, _) | (_, None) => {
                let res = ctx
                    .function
                    .push_cmp(ctx.function.get_entry_id(), self.0, b.0, cmp_kind);
                Self(res)
            }
            _ => todo!(),
        }
    }

    fn arith(
        &self,
        b: &Self,
        binary_arith_op_kind: crate::compiler::ssa::BinaryArithOpKind,
        _out_type: &crate::compiler::ir::r#type::Type<ConstantTaint>,
        ctx: &mut SpecializationState,
    ) -> Self {
        let a_const = ctx.const_vals.get(&self.0);
        let b_const = ctx.const_vals.get(&b.0);
        match (binary_arith_op_kind, a_const, b_const) {
            (BinaryArithOpKind::Add, Some(ConstVal::U(s, a_val)), Some(ConstVal::U(_, b_val))) => {
                let res = a_val + b_val;
                let res_v = ctx.function.push_u_const(*s, res);
                ctx.const_vals.insert(res_v, ConstVal::U(*s, res));
                Self(res_v)
            }
            (BinaryArithOpKind::Sub, Some(ConstVal::U(s, a_val)), Some(ConstVal::U(_, b_val))) => {
                let res = a_val - b_val;
                let res_v = ctx.function.push_u_const(*s, res);
                ctx.const_vals.insert(res_v, ConstVal::U(*s, res));
                Self(res_v)
            }
            (
                BinaryArithOpKind::Mul,
                Some(ConstVal::Field(l_val)),
                Some(ConstVal::Field(r_val)),
            ) => {
                let res = l_val * r_val;
                let res_v = ctx.function.push_field_const(res);
                ctx.const_vals.insert(res_v, ConstVal::Field(res));
                Self(res_v)
            }
            (BinaryArithOpKind::Sub, Some(ConstVal::Field(f)), Some(ConstVal::Field(f2))) => {
                let res = f - f2;
                let res_v = ctx.function.push_field_const(res);
                ctx.const_vals.insert(res_v, ConstVal::Field(res));
                Self(res_v)
            }
            (BinaryArithOpKind::Add, Some(ConstVal::Field(f)), Some(ConstVal::Field(f2))) => {
                let res = f + f2;
                let res_v = ctx.function.push_field_const(res);
                ctx.const_vals.insert(res_v, ConstVal::Field(res));
                Self(res_v)
            }

            (BinaryArithOpKind::Mul, Some(ConstVal::Field(f)), _) if *f == ark_ff::Field::ONE => *b,
            (BinaryArithOpKind::Mul, _, Some(ConstVal::Field(f))) if *f == ark_ff::Field::ONE => {
                *self
            }
            (BinaryArithOpKind::Mul, Some(ConstVal::Field(f)), _) if *f == Field::ZERO => *self,
            (BinaryArithOpKind::Mul, _, Some(ConstVal::Field(f))) if *f == Field::ZERO => *b,

            (BinaryArithOpKind::Mul, None, None) => {
                let res = ctx
                    .function
                    .push_mul(ctx.function.get_entry_id(), self.0, b.0);
                Self(res)
            }

            (BinaryArithOpKind::Add, Some(ConstVal::Field(f)), _) if *f == Field::ZERO => *b,
            (BinaryArithOpKind::Add, _, Some(ConstVal::Field(f))) if *f == Field::ZERO => *self,

            (BinaryArithOpKind::And, Some(ConstVal::U(s, a_val)), Some(ConstVal::U(_, b_val))) => {
                let res = a_val & b_val;
                let res_v = ctx.function.push_u_const(*s, res);
                ctx.const_vals.insert(res_v, ConstVal::U(*s, res));
                Self(res_v)
            }

            (BinaryArithOpKind::And, _, None) | (BinaryArithOpKind::And, None, _) => {
                let res = ctx
                    .function
                    .push_and(ctx.function.get_entry_id(), self.0, b.0);
                Self(res)
            }

            _ => panic!(
                "Not yet implemented {:?} {:?}",
                binary_arith_op_kind,
                (a_const, b_const)
            ),
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
                ctx.function.push_assert_eq(ctx.function.get_entry_id(), self.0, other.0);
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
        _out_type: &crate::compiler::ir::r#type::Type<ConstantTaint>,
        ctx: &mut SpecializationState,
    ) -> Self {
        let a_const = ctx.const_vals.get(&self.0);
        let index_const = ctx.const_vals.get(&index.0);
        match (a_const, index_const) {
            (Some(ConstVal::Array(a)), Some(ConstVal::U(_, index))) => {
                let res = a[*index as usize];
                Self(res)
            }
            (Some(ConstVal::BitsOf(v, size, endianness)), Some(ConstVal::U(_, index))) => {
                let v_const = ctx.const_vals.get(v.as_ref());
                match v_const {
                    Some(ConstVal::Field(f)) => {
                        let r = f.into_bigint().to_bits_le();
                        let ix = match endianness {
                            Endianness::Little => *index as usize,
                            Endianness::Big => size - *index as usize - 1,
                        };
                        let res = if r[ix] { 1 } else { 0 };
                        let res_v = ctx.function.push_u_const(1, res);
                        ctx.const_vals.insert(res_v, ConstVal::U(1, res));
                        Self(res_v)
                    }
                    _ => panic!("Not yet implemented {:?}", (v_const, endianness)),
                }
            }
            (None, _) | (_, None) => {
                let res = ctx
                    .function
                    .push_array_get(ctx.function.get_entry_id(), self.0, index.0);
                Self(res)
            }
            _ => panic!("Not yet implemented {:?}", (a_const, index_const)),
        }
    }

    fn tuple_get(
        &self,
        index: usize,
        _out_type: &crate::compiler::ir::r#type::Type<ConstantTaint>,
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
        _out_type: &crate::compiler::ir::r#type::Type<ConstantTaint>,
        _ctx: &mut SpecializationState,
    ) -> Self {
        todo!()
    }

    fn truncate(
        &self,
        _from: usize,
        _to: usize,
        _out_type: &crate::compiler::ir::r#type::Type<ConstantTaint>,
        _ctx: &mut SpecializationState,
    ) -> Self {
        todo!()
    }

    fn cast(
        &self,
        cast_target: &crate::compiler::ssa::CastTarget,
        _out_type: &crate::compiler::ir::r#type::Type<ConstantTaint>,
        ctx: &mut SpecializationState,
    ) -> Self {
        let self_const = ctx.const_vals.get(&self.0);
        match self_const {
            Some(ConstVal::U(_, v)) => match cast_target {
                CastTarget::U(s) => {
                    let res = v & ((1 << *s) - 1);
                    let res_v = ctx.function.push_u_const(*s, res);
                    ctx.const_vals.insert(res_v, ConstVal::U(*s, res));
                    Self(res_v)
                }
                CastTarget::Field => {
                    let res = Field::from(*v);
                    let res_v = ctx.function.push_field_const(res);
                    ctx.const_vals.insert(res_v, ConstVal::Field(res));
                    Self(res_v)
                }
                CastTarget::Nop => {
                    self.clone()
                }
            },
            _ => todo!(),
        }
    }

    fn constrain(_a: &Self, _b: &Self, _c: &Self, _ctx: &mut SpecializationState) {
        todo!()
    }

    fn to_bits(
        &self,
        endianness: Endianness,
        size: usize,
        _out_type: &crate::compiler::ir::r#type::Type<ConstantTaint>,
        ctx: &mut SpecializationState,
    ) -> Self {
        let val = ctx
            .function
            .push_to_bits(ctx.function.get_entry_id(), self.0, endianness, size);
        ctx.const_vals
            .insert(val, ConstVal::BitsOf(Box::new(self.0), size, endianness));
        Self(val)
    }

    fn not(
        &self,
        _out_type: &crate::compiler::ir::r#type::Type<ConstantTaint>,
        ctx: &mut SpecializationState,
    ) -> Self {
        let const_val = ctx.const_vals.get(&self.0);
        match const_val {
            Some(ConstVal::U(s, v)) => {
                let res = !v & ((1 << s) - 1);
                let res_v = ctx.function.push_u_const(*s, res);
                ctx.const_vals.insert(res_v, ConstVal::U(*s, res));
                Self(res_v)
            }
            None => {
                let res = ctx.function.push_not(ctx.function.get_entry_id(), self.0);
                Self(res)
            }
            _ => todo!(),
        }
    }

    fn of_u(s: usize, v: u128, ctx: &mut SpecializationState) -> Self {
        let val = ctx.function.push_u_const(s, v);
        ctx.const_vals.insert(val, ConstVal::U(s, v));
        Self(val)
    }

    fn of_field(f: Field, ctx: &mut SpecializationState) -> Self {
        let val = ctx.function.push_field_const(f);
        ctx.const_vals.insert(val, ConstVal::Field(f));
        Self(val)
    }

    fn mk_array(
        a: Vec<Self>,
        ctx: &mut SpecializationState,
        seq_type: SeqType,
        elem_type: &Type<ConstantTaint>,
    ) -> Self {
        let a = a.into_iter().map(|v| v.0).collect::<Vec<_>>();
        let val = ctx.function.push_mk_array(
            ctx.function.get_entry_id(),
            a.clone(),
            seq_type,
            elem_type.clone(),
        );
        ctx.const_vals.insert(val, ConstVal::Array(a));
        Self(val)
    }

    fn mk_tuple(
        elems: Vec<Self>,
        ctx: &mut SpecializationState,
        elem_types: &[Type<ConstantTaint>],
    ) -> Self {
        let a = elems.into_iter().map(|v| v.0).collect::<Vec<_>>();
        let val = ctx.function.push_mk_tuple(
            ctx.function.get_entry_id(),
            a.clone(),
            elem_types.to_vec(),
        );
        ctx.const_vals.insert(val, ConstVal::Tuple(a));
        Self(val)
    }

    fn alloc(_ctx: &mut SpecializationState) -> Self {
        todo!()
    }

    fn ptr_write(&self, _val: &Self, _ctx: &mut SpecializationState) {
        todo!()
    }

    fn ptr_read(
        &self,
        _out_type: &crate::compiler::ir::r#type::Type<ConstantTaint>,
        _ctx: &mut SpecializationState,
    ) -> Self {
        todo!()
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
        _out_type: &crate::compiler::ir::r#type::Type<ConstantTaint>,
        ctx: &mut SpecializationState,
    ) -> Self {
        let self_const = ctx.const_vals.get(&self.0);

        match self_const {
            Some(ConstVal::U(_, v)) => {
                let res = if *v == 1 { if_t.0 } else { if_f.0 };
                Self(res)
            }
            None => {
                let res =
                    ctx.function
                        .push_select(ctx.function.get_entry_id(), self.0, if_t.0, if_f.0);
                Self(res)
            }
            _ => todo!(),
        }
    }

    fn write_witness(
        &self,
        _tp: Option<&crate::compiler::ir::r#type::Type<ConstantTaint>>,
        _ctx: &mut SpecializationState,
    ) -> Self {
        todo!()
    }

    fn fresh_witness(_ctx: &mut SpecializationState) -> Self {
        todo!()
    }

    fn mem_op(&self, kind: MemOp, ctx: &mut SpecializationState) {
        ctx.function
            .push_mem_op(ctx.function.get_entry_id(), self.0, kind);
    }

    fn rangecheck(&self, max_bits: usize, ctx: &mut SpecializationState) {
        ctx.function
            .push_rangecheck(ctx.function.get_entry_id(), self.0, max_bits);
    }

    fn to_radix(
        &self,
        radix: &Radix<Self>,
        endianness: Endianness,
        size: usize,
        _out_type: &crate::compiler::ir::r#type::Type<ConstantTaint>,
        ctx: &mut SpecializationState,
    ) -> Self {
        let cst_val = ctx.const_vals.get(&self.0);
        match cst_val {
            None => {
                let radix = match radix {
                    Radix::Dyn(v) => Radix::Dyn(v.0),
                    Radix::Bytes => Radix::Bytes,
                };
                let res = ctx.function.push_to_radix(
                    ctx.function.get_entry_id(),
                    self.0,
                    radix,
                    endianness,
                    size,
                );
                Self(res)
            }
            Some(_) => todo!(),
        }
    }
}

impl<T> symbolic_executor::Context<Val, T> for SpecializationState {
    fn on_call(
        &mut self,
        _func: crate::compiler::ssa::FunctionId,
        _params: &mut [Val],
        _param_types: &[&crate::compiler::ir::r#type::Type<T>],
    ) -> Option<Vec<Val>> {
        None
    }

    fn on_return(
        &mut self,
        returns: &mut [Val],
        _return_types: &[crate::compiler::ir::r#type::Type<T>],
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
        _param_types: &[&crate::compiler::ir::r#type::Type<T>],
    ) {
    }

    fn todo(
        &mut self,
        payload: &str,
        _result_types: &[crate::compiler::ir::r#type::Type<T>],
    ) -> Vec<Val> {
        todo!("Todo opcode: {}", payload);
    }
}

impl Pass<ConstantTaint> for Specializer {
    fn run(
        &self,
        ssa: &mut SSA<ConstantTaint>,
        pass_manager: &crate::compiler::pass_manager::PassManager<ConstantTaint>,
    ) {
        let summary = pass_manager.get_constraint_instrumentation();
        for (sig, summary) in summary.functions.iter() {
            if summary.specialization_total_savings > 0 {
                self.try_spec(ssa, pass_manager.get_type_info(), summary, sig.clone());
            }
        }
    }

    fn pass_info(&self) -> crate::compiler::pass_manager::PassInfo {
        PassInfo {
            name: "specializer",
            needs: vec![DataPoint::ConstraintInstrumentation, DataPoint::Types],
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
        ssa: &mut SSA<ConstantTaint>,
        type_info: &TypeInfo<ConstantTaint>,
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
            function: Function::<ConstantTaint>::empty(name),
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
                ValueSignature::UWitness(_) => {
                    call_params.push(Val(state
                        .function
                        .add_parameter(state.function.get_entry_id(), param.clone())));
                }
                ValueSignature::FWitness => {
                    call_params.push(Val(state
                        .function
                        .add_parameter(state.function.get_entry_id(), param.clone())));
                }
                ValueSignature::Field(f) => {
                    let val = state.function.push_field_const(*f);
                    call_params.push(Val(val));
                    state.const_vals.insert(val, ConstVal::Field(*f));
                }
                ValueSignature::U(size, v) => {
                    let val = state.function.push_u_const(*size, *v);
                    call_params.push(Val(val));
                    state.const_vals.insert(val, ConstVal::U(*size, *v));
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
        params: Vec<Type<ConstantTaint>>,
        returns: Vec<Type<ConstantTaint>>,
        signature: &FunctionSignature,
        fn_name: String,
        specialized_id: FunctionId,
        unspecialized_id: FunctionId,
    ) -> Function<ConstantTaint> {
        let mut dispatcher = Function::<ConstantTaint>::empty(fn_name);
        let mut dispatcher_params = vec![];
        for param in params {
            dispatcher_params.push(dispatcher.add_parameter(dispatcher.get_entry_id(), param));
        }

        for return_type in returns.iter() {
            dispatcher.add_return_type(return_type.clone());
        }

        let mut specialized_params = vec![];

        let entry_block = dispatcher.get_entry_id();
        let mut should_call_spec = dispatcher.push_u_const(1, 1);

        for (pval, psig) in dispatcher_params.iter().zip(signature.get_params().iter()) {
            match psig {
                ValueSignature::PointerTo(_) => {
                    todo!();
                }
                ValueSignature::Array(_) => {
                    todo!();
                }
                ValueSignature::UWitness(_) | ValueSignature::FWitness => {
                    specialized_params.push(*pval);
                }
                ValueSignature::Field(v) => {
                    let cst = dispatcher.push_field_const(*v);
                    let is_eq = dispatcher.push_eq(entry_block, *pval, cst);
                    should_call_spec = dispatcher.push_and(entry_block, should_call_spec, is_eq);
                }
                ValueSignature::U(s, v) => {
                    let cst = dispatcher.push_u_const(*s, *v);
                    let is_eq = dispatcher.push_eq(entry_block, *pval, cst);
                    should_call_spec = dispatcher.push_and(entry_block, should_call_spec, is_eq);
                }
                ValueSignature::Tuple(_) => {
                    todo!();
                }
            }
        }

        let specialized_caller = dispatcher.add_block();
        let unspecialized_caller = dispatcher.add_block();
        let return_block = dispatcher.add_block();

        let mut return_values = vec![];
        for ret in returns {
            return_values.push(dispatcher.add_parameter(return_block, ret));
        }
        dispatcher.terminate_block_with_return(return_block, return_values.clone());

        let unspecialized_returns = dispatcher.push_call(
            unspecialized_caller,
            unspecialized_id,
            dispatcher_params,
            return_values.len(),
        );
        dispatcher.terminate_block_with_jmp(
            unspecialized_caller,
            return_block,
            unspecialized_returns,
        );

        let specialized_returns = dispatcher.push_call(
            specialized_caller,
            specialized_id,
            specialized_params,
            return_values.len(),
        );
        dispatcher.terminate_block_with_jmp(specialized_caller, return_block, specialized_returns);

        dispatcher.terminate_block_with_jmp_if(
            entry_block,
            should_call_spec,
            specialized_caller,
            unspecialized_caller,
        );

        // dispatcher.terminate_block_with_return(dispatcher.get_entry_id(), vec![Val(unspecialized_id)]);

        dispatcher
    }
}
