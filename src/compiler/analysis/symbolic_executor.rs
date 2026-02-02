use tracing::{Level, instrument};

use crate::compiler::{
    Field,
    analysis::types::TypeInfo,
    ir::r#type::{CommutativeMonoid, Type},
    ssa::{
        BinaryArithOpKind, BlockId, CastTarget, CmpKind, Const, Endianness, FunctionId, GlobalDef, LookupTarget, MemOp, Radix, SSA, SeqType, SliceOpDir, Terminator
    },
};

pub trait Value<Context, Taint>
where
    Self: Sized + Clone,
{
    fn cmp(&self, b: &Self, cmp_kind: CmpKind, out_type: &Type<Taint>, ctx: &mut Context) -> Self;
    fn arith(
        &self,
        b: &Self,
        binary_arith_op_kind: BinaryArithOpKind,
        out_type: &Type<Taint>,
        ctx: &mut Context,
    ) -> Self;
    fn assert_eq(&self, other: &Self, ctx: &mut Context);
    fn assert_r1c(a: &Self, b: &Self, c: &Self, ctx: &mut Context);
    fn array_get(&self, index: &Self, out_type: &Type<Taint>, ctx: &mut Context) -> Self;
    fn tuple_get(&self, index: usize, out_type: &Type<Taint>, ctx: &mut Context) -> Self;
    fn array_set(
        &self,
        index: &Self,
        value: &Self,
        out_type: &Type<Taint>,
        ctx: &mut Context,
    ) -> Self;
    fn truncate(&self, _from: usize, to: usize, out_type: &Type<Taint>, ctx: &mut Context) -> Self;
    fn cast(&self, cast_target: &CastTarget, out_type: &Type<Taint>, ctx: &mut Context) -> Self;
    fn constrain(a: &Self, b: &Self, c: &Self, ctx: &mut Context);
    fn to_bits(
        &self,
        endianness: Endianness,
        size: usize,
        out_type: &Type<Taint>,
        ctx: &mut Context,
    ) -> Self;
    fn to_radix(
        &self,
        radix: &Radix<Self>,
        endianness: Endianness,
        size: usize,
        out_type: &Type<Taint>,
        ctx: &mut Context,
    ) -> Self;
    fn not(&self, out_type: &Type<Taint>, ctx: &mut Context) -> Self;
    fn of_u(s: usize, v: u128, ctx: &mut Context) -> Self;
    fn of_field(f: Field, ctx: &mut Context) -> Self;
    fn mk_array(
        a: Vec<Self>,
        ctx: &mut Context,
        seq_type: SeqType,
        elem_type: &Type<Taint>,
    ) -> Self;
    fn mk_tuple(
        elems: Vec<Self>,
        ctx: &mut Context,
        elem_types: &[Type<Taint>],
    ) -> Self;
    fn alloc(ctx: &mut Context) -> Self;
    fn ptr_write(&self, val: &Self, ctx: &mut Context);
    fn ptr_read(&self, out_type: &Type<Taint>, ctx: &mut Context) -> Self;
    fn expect_constant_bool(&self, ctx: &mut Context) -> bool;
    fn select(&self, if_t: &Self, if_f: &Self, out_type: &Type<Taint>, ctx: &mut Context) -> Self;
    fn write_witness(&self, tp: Option<&Type<Taint>>, ctx: &mut Context) -> Self;
    fn fresh_witness(ctx: &mut Context) -> Self;
    fn mem_op(&self, kind: MemOp, ctx: &mut Context);
    fn rangecheck(&self, max_bits: usize, ctx: &mut Context);
}

pub trait Context<V, Taint> {
    // If this returns Some, the function execution will be skipped and the returned values will be used instead
    // as call results.
    fn on_call(
        &mut self,
        func: FunctionId,
        params: &mut [V],
        param_types: &[&Type<Taint>],
    ) -> Option<Vec<V>>;
    fn on_return(&mut self, returns: &mut [V], return_types: &[Type<Taint>]);
    fn on_jmp(&mut self, target: BlockId, params: &mut [V], param_types: &[&Type<Taint>]);

    // TODO it looks odd that this is the only opcode implemented here.
    // This is the _new_ structure, so at some point we should migrate all other opcodes here.
    fn lookup(&mut self, _target: LookupTarget<V>, _keys: Vec<V>, _results: Vec<V>) {
        panic!("ICE: backend does not implement lookup");
    }

    fn dlookup(&mut self, _target: LookupTarget<V>, _keys: Vec<V>, _results: Vec<V>) {
        panic!("ICE: backend does not implement dlookup");
    }

    fn todo(&mut self, payload: &str, _result_types: &[Type<Taint>]) -> Vec<V> {
        panic!("Todo opcode encountered: {}", payload);
    }

    fn slice_push(&mut self, _slice: &V, _values: &[V], _dir: SliceOpDir) -> V {
        panic!("ICE: backend does not implement slice_push");
    }

    fn slice_len(&mut self, _slice: &V) -> V {
        panic!("ICE: backend does not implement slice_len");
    }
}

pub struct SymbolicExecutor {}

impl SymbolicExecutor {
    pub fn new() -> Self {
        Self {}
    }

    #[instrument(skip_all, name = "SymbolicExecutor::run", level = Level::DEBUG)]
    pub fn run<V, T, Ctx>(
        &self,
        ssa: &SSA<T>,
        type_info: &TypeInfo<T>,
        entry_point: FunctionId,
        params: Vec<V>,
        context: &mut Ctx,
    ) where
        V: Value<Ctx, T>,
        Ctx: Context<V, T>,
        T: Clone + CommutativeMonoid,
    {
        let mut globals = vec![];

        for global in ssa.get_globals().iter() {
            match global {
                GlobalDef::Const(Const::U(s, v)) => {
                    globals.push(V::of_u(*s, *v, context));
                }
                GlobalDef::Const(Const::Field(f)) => {
                    globals.push(V::of_field(f.clone(), context));
                }
                GlobalDef::Const(Const::WitnessRef(_)) => {
                    todo!()
                }
                GlobalDef::Array(items, typ) => {
                    let items = items
                        .iter()
                        .map(|id| globals[*id].clone())
                        .collect::<Vec<_>>();
                    globals.push(V::mk_array(
                        items.clone(),
                        context,
                        SeqType::Array(items.len()),
                        &typ.as_pure(),
                    ));
                }
            }
        }

        self.run_fn(ssa, type_info, entry_point, params, &globals, context);
    }

    #[instrument(skip_all, name="SymbolicExecutor::run_fn", level = Level::TRACE, fields(function = %ssa.get_function(fn_id).get_name()))]
    fn run_fn<V, T, Ctx>(
        &self,
        ssa: &SSA<T>,
        type_info: &TypeInfo<T>,
        fn_id: FunctionId,
        mut inputs: Vec<V>,
        globals: &[V],
        ctx: &mut Ctx,
    ) -> Vec<V>
    where
        V: Value<Ctx, T>,
        Ctx: Context<V, T>,
        T: Clone,
    {
        let fn_body = ssa.get_function(fn_id);
        let fn_type_info = type_info.get_function(fn_id);
        let entry = fn_body.get_entry();
        let mut scope: Vec<Option<V>> = vec![None; fn_body.get_var_num_bound()];

        for (val, cst) in fn_body.iter_consts() {
            let v = match cst {
                Const::U(s, v) => V::of_u(*s, *v, ctx),
                Const::Field(f) => V::of_field(f.clone(), ctx),
                Const::WitnessRef(_) => todo!(),
            };
            scope[val.0 as usize] = Some(v);
        }

        let call_result = ctx.on_call(
            fn_id,
            &mut inputs,
            &entry.get_parameters().map(|(_, tp)| tp).collect::<Vec<_>>(),
        );

        if let Some(call_result) = call_result {
            return call_result;
        }

        for (pval, ppos) in inputs.iter_mut().zip(entry.get_parameter_values()) {
            scope[ppos.0 as usize] = Some(pval.clone());
        }

        let mut current = Some(entry);

        while let Some(block) = current {
            for instr in block.get_instructions() {
                match instr {
                    crate::compiler::ssa::OpCode::Cmp {
                        kind: cmp_kind,
                        result: r,
                        lhs: a,
                        rhs: b,
                    } => {
                        let a = scope[a.0 as usize].as_ref().unwrap();
                        let b = scope[b.0 as usize].as_ref().unwrap();
                        scope[r.0 as usize] =
                            Some(a.cmp(b, *cmp_kind, &fn_type_info.get_value_type(*r), ctx));
                    }
                    crate::compiler::ssa::OpCode::BinaryArithOp {
                        kind: binary_arith_op_kind,
                        result: r,
                        lhs: a,
                        rhs: b,
                    } => {
                        let a = scope[a.0 as usize].as_ref().unwrap();
                        let b = scope[b.0 as usize].as_ref().unwrap();
                        scope[r.0 as usize] = Some(a.arith(
                            b,
                            *binary_arith_op_kind,
                            &fn_type_info.get_value_type(*r),
                            ctx,
                        ));
                    }
                    crate::compiler::ssa::OpCode::Cast {
                        result: r,
                        value: a,
                        target: cast_target,
                    } => {
                        let a = scope[a.0 as usize].as_ref().unwrap();
                        scope[r.0 as usize] =
                            Some(a.cast(cast_target, &fn_type_info.get_value_type(*r), ctx));
                    }
                    crate::compiler::ssa::OpCode::Truncate {
                        result: r,
                        value: a,
                        to_bits: to,
                        from_bits: from,
                    } => {
                        let a = scope[a.0 as usize].as_ref().unwrap();
                        scope[r.0 as usize] =
                            Some(a.truncate(*from, *to, &fn_type_info.get_value_type(*r), ctx));
                    }
                    crate::compiler::ssa::OpCode::Not {
                        result: r,
                        value: a,
                    } => {
                        let a = scope[a.0 as usize].as_ref().unwrap();
                        scope[r.0 as usize] = Some(a.not(&fn_type_info.get_value_type(*r), ctx));
                    }
                    crate::compiler::ssa::OpCode::MkSeq {
                        result: r,
                        elems: a,
                        seq_type,
                        elem_type,
                    } => {
                        let a = a
                            .iter()
                            .map(|id| scope[id.0 as usize].as_ref().unwrap().clone())
                            .collect::<Vec<_>>();
                        scope[r.0 as usize] = Some(V::mk_array(a, ctx, *seq_type, elem_type));
                    }
                    crate::compiler::ssa::OpCode::Alloc {
                        result: r,
                        elem_type: _,
                        result_annotation: _,
                    } => {
                        scope[r.0 as usize] = Some(V::alloc(ctx));
                    }
                    crate::compiler::ssa::OpCode::Store { ptr, value: val } => {
                        let ptr = scope[ptr.0 as usize].as_ref().unwrap();
                        let val = scope[val.0 as usize].as_ref().unwrap();
                        ptr.ptr_write(val, ctx);
                    }
                    crate::compiler::ssa::OpCode::Load { result: r, ptr } => {
                        let ptr = scope[ptr.0 as usize].as_ref().unwrap();
                        scope[r.0 as usize] =
                            Some(ptr.ptr_read(&fn_type_info.get_value_type(*r), ctx));
                    }
                    crate::compiler::ssa::OpCode::AssertR1C { a, b, c } => {
                        let a = scope[a.0 as usize].as_ref().unwrap();
                        let b = scope[b.0 as usize].as_ref().unwrap();
                        let c = scope[c.0 as usize].as_ref().unwrap();
                        V::assert_r1c(a, b, c, ctx);
                    }
                    crate::compiler::ssa::OpCode::Call {
                        results: returns,
                        function: function_id,
                        args: arguments,
                    } => {
                        let params = arguments
                            .iter()
                            .map(|id| scope[id.0 as usize].as_ref().unwrap().clone())
                            .collect::<Vec<_>>();
                        let outputs =
                            self.run_fn(ssa, type_info, *function_id, params, globals, ctx);
                        for (i, val) in returns.iter().enumerate() {
                            scope[val.0 as usize] = Some(outputs[i].clone());
                        }
                    }
                    crate::compiler::ssa::OpCode::ArrayGet {
                        result: r,
                        array: a,
                        index: i,
                    } => {
                        let a = scope[a.0 as usize].as_ref().unwrap();
                        let i = scope[i.0 as usize].as_ref().unwrap();
                        scope[r.0 as usize] =
                            Some(a.array_get(i, &fn_type_info.get_value_type(*r), ctx));
                    }
                    crate::compiler::ssa::OpCode::ArraySet {
                        result: r,
                        array: arr,
                        index: i,
                        value: v,
                    } => {
                        let a = scope[arr.0 as usize].as_ref().unwrap();
                        let i = scope[i.0 as usize].as_ref().unwrap();
                        let v = scope[v.0 as usize].as_ref().unwrap();
                        scope[r.0 as usize] =
                            Some(a.array_set(i, v, &fn_type_info.get_value_type(*r), ctx));
                    }
                    crate::compiler::ssa::OpCode::SlicePush {
                        result,
                        slice,
                        values,
                        dir,
                    } => {                      
                        let sl = scope[slice.0 as usize].as_ref().unwrap();
                        let vals: Vec<_> = values.iter()
                            .map(|v| scope[v.0 as usize].as_ref().unwrap().clone())
                            .collect();
                        scope[result.0 as usize] = Some(ctx.slice_push(sl, &vals, *dir));
                    }
                    crate::compiler::ssa::OpCode::SliceLen {
                        result: r,
                        slice: sl,
                    } => {
                        let sl = scope[sl.0 as usize].as_ref().unwrap();
                        scope[r.0 as usize] = Some(ctx.slice_len(sl));
                    }
                    crate::compiler::ssa::OpCode::Select {
                        result: r,
                        cond,
                        if_t,
                        if_f,
                    } => {
                        let cond = scope[cond.0 as usize].as_ref().unwrap();
                        let if_t = scope[if_t.0 as usize].as_ref().unwrap();
                        let if_f = scope[if_f.0 as usize].as_ref().unwrap();
                        scope[r.0 as usize] =
                            Some(cond.select(if_t, if_f, &fn_type_info.get_value_type(*r), ctx));
                    }
                    crate::compiler::ssa::OpCode::ToBits {
                        result: r,
                        value: a,
                        endianness,
                        count: size,
                    } => {
                        let a = scope[a.0 as usize].as_ref().unwrap();
                        scope[r.0 as usize] = Some(a.to_bits(
                            *endianness,
                            *size,
                            &fn_type_info.get_value_type(*r),
                            ctx,
                        ));
                    }
                    crate::compiler::ssa::OpCode::ToRadix {
                        result: r,
                        value: a,
                        radix,
                        endianness,
                        count: size,
                    } => {
                        let a = scope[a.0 as usize].as_ref().unwrap();
                        let radix = match radix {
                            Radix::Bytes => Radix::Bytes,
                            Radix::Dyn(radix) => {
                                Radix::Dyn(scope[radix.0 as usize].as_ref().unwrap().clone())
                            }
                        };
                        scope[r.0 as usize] = Some(a.to_radix(
                            &radix,
                            *endianness,
                            *size,
                            &fn_type_info.get_value_type(*r),
                            ctx,
                        ));
                    }
                    crate::compiler::ssa::OpCode::WriteWitness {
                        result: r,
                        value: a,
                        witness_annotation: _,
                    } => {
                        let a = scope[a.0 as usize].as_ref().unwrap();
                        if let Some(r) = r {
                            scope[r.0 as usize] =
                                Some(a.write_witness(Some(fn_type_info.get_value_type(*r)), ctx));
                        } else {
                            a.write_witness(None, ctx);
                        }
                    }
                    crate::compiler::ssa::OpCode::FreshWitness {
                        result: r,
                        result_type: _,
                    } => {
                        scope[r.0 as usize] = Some(V::fresh_witness(ctx));
                    }
                    crate::compiler::ssa::OpCode::Constrain { a, b, c } => {
                        let a = scope[a.0 as usize].as_ref().unwrap();
                        let b = scope[b.0 as usize].as_ref().unwrap();
                        let c = scope[c.0 as usize].as_ref().unwrap();
                        V::constrain(a, b, c, ctx);
                    }
                    crate::compiler::ssa::OpCode::AssertEq { lhs: a, rhs: b } => {
                        let a = scope[a.0 as usize].as_ref().unwrap();
                        let b = scope[b.0 as usize].as_ref().unwrap();
                        V::assert_eq(a, b, ctx);
                    }
                    crate::compiler::ssa::OpCode::MemOp { kind, value } => {
                        let value = scope[value.0 as usize].as_ref().unwrap();
                        value.mem_op(*kind, ctx);
                    }
                    crate::compiler::ssa::OpCode::NextDCoeff { result: _a } => {
                        todo!()
                    }
                    crate::compiler::ssa::OpCode::BumpD {
                        matrix: _matrix,
                        variable: _a,
                        sensitivity: _b,
                    } => {
                        todo!()
                    }
                    crate::compiler::ssa::OpCode::PureToWitnessRef {
                        result: _,
                        value: _,
                        result_annotation: _,
                    } => {
                        todo!()
                    }
                    crate::compiler::ssa::OpCode::UnboxField {
                        result: _,
                        value: _,
                    } => {
                        todo!()
                    }
                    crate::compiler::ssa::OpCode::MulConst {
                        result: _,
                        const_val: _,
                        var: _,
                    } => {
                        todo!()
                    }
                    crate::compiler::ssa::OpCode::Rangecheck { value: v, max_bits } => {
                        let v = scope[v.0 as usize].as_ref().unwrap();
                        v.rangecheck(*max_bits, ctx);
                    }
                    crate::compiler::ssa::OpCode::ReadGlobal {
                        result,
                        offset,
                        result_type: _,
                    } => {
                        let r = globals[*offset as usize].clone();
                        scope[result.0 as usize] = Some(r);
                    }
                    crate::compiler::ssa::OpCode::Lookup {
                        target,
                        keys,
                        results,
                    } => {
                        let target = match target {
                            LookupTarget::Rangecheck(n) => LookupTarget::Rangecheck(*n),
                            LookupTarget::DynRangecheck(v) => LookupTarget::DynRangecheck(
                                scope[v.0 as usize].as_ref().unwrap().clone(),
                            ),
                            LookupTarget::Array(arr) => {
                                LookupTarget::Array(scope[arr.0 as usize].as_ref().unwrap().clone())
                            }
                        };
                        let keys = keys
                            .iter()
                            .map(|id| scope[id.0 as usize].as_ref().unwrap().clone())
                            .collect::<Vec<_>>();
                        let results = results
                            .iter()
                            .map(|id| scope[id.0 as usize].as_ref().unwrap().clone())
                            .collect::<Vec<_>>();
                        ctx.lookup(target, keys, results);
                    }
                    crate::compiler::ssa::OpCode::DLookup {
                        target,
                        keys,
                        results,
                    } => {
                        let target = match target {
                            LookupTarget::Rangecheck(n) => LookupTarget::Rangecheck(*n),
                            LookupTarget::DynRangecheck(v) => LookupTarget::DynRangecheck(
                                scope[v.0 as usize].as_ref().unwrap().clone(),
                            ),
                            LookupTarget::Array(arr) => {
                                LookupTarget::Array(scope[arr.0 as usize].as_ref().unwrap().clone())
                            }
                        };
                        let keys = keys
                            .iter()
                            .map(|id| scope[id.0 as usize].as_ref().unwrap().clone())
                            .collect::<Vec<_>>();
                        let results = results
                            .iter()
                            .map(|id| scope[id.0 as usize].as_ref().unwrap().clone())
                            .collect::<Vec<_>>();
                        ctx.dlookup(target, keys, results);
                    }
                    crate::compiler::ssa::OpCode::TupleProj {
                        result: r,
                        tuple: a,
                        idx: i,
                    } => {
                        if let crate::compiler::ssa::TupleIdx::Static(index) = i {
                            let a = scope[a.0 as usize].as_ref().unwrap();
                            scope[r.0 as usize] =
                                Some(a.tuple_get(*index, &fn_type_info.get_value_type(*r), ctx));
                        } else {
                            panic!("Dynamic tuple indexing should not appear here");
                        }
                    },
                    crate::compiler::ssa::OpCode::MkTuple { 
                        result, 
                        elems, 
                        element_types,
                    } => {
                        let elems = elems
                            .iter()
                            .map(|id| scope[id.0 as usize].as_ref().unwrap().clone())
                            .collect::<Vec<_>>();
                        scope[result.0 as usize] = Some(V::mk_tuple(elems, ctx, element_types));
                    }
                    crate::compiler::ssa::OpCode::Todo {
                        payload,
                        results,
                        result_types,
                    } => {
                        // The context handler should return the result values
                        let result_values = ctx.todo(&payload, result_types);
                        if result_values.len() != results.len() {
                            panic!(
                                "Todo opcode handler returned {} values but {} were expected",
                                result_values.len(),
                                results.len()
                            );
                        }
                        for (result_id, result_value) in results.iter().zip(result_values.iter()) {
                            scope[result_id.0 as usize] = Some(result_value.clone());
                        }
                    }
                }
            }

            match block.get_terminator().unwrap() {
                Terminator::Return(returns) => {
                    let mut outputs = returns
                        .iter()
                        .map(|id| scope[id.0 as usize].as_ref().unwrap().clone())
                        .collect::<Vec<_>>();
                    ctx.on_return(&mut outputs, &fn_body.get_returns());
                    return outputs;
                }
                Terminator::Jmp(target, params) => {
                    let mut params = params
                        .iter()
                        .map(|id| scope[id.0 as usize].as_ref().unwrap().clone())
                        .collect::<Vec<_>>();
                    let target_block = fn_body.get_block(*target);
                    let target_params = target_block.get_parameter_values();
                    ctx.on_jmp(
                        *target,
                        &mut params,
                        &target_block
                            .get_parameters()
                            .map(|(_, tp)| tp)
                            .collect::<Vec<_>>(),
                    );
                    for (i, val) in target_params.zip(params.into_iter()) {
                        scope[i.0 as usize] = Some(val);
                    }
                    current = Some(target_block);
                }
                Terminator::JmpIf(cond, if_true, if_false) => {
                    let cond = scope[cond.0 as usize].as_ref().unwrap();
                    if cond.expect_constant_bool(ctx) {
                        current = Some(fn_body.get_block(*if_true));
                    } else {
                        current = Some(fn_body.get_block(*if_false));
                    }
                }
            }
        }

        panic!("ICE: Unreachable, function did not return");
    }
}
