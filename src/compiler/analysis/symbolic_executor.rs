use tracing::{Level, instrument};

use crate::compiler::{
    Field,
    analysis::types::TypeInfo,
    ir::r#type::Type,
    ssa::{
        BinaryArithOpKind, BlockId, CastTarget, CmpKind, Endianness, FunctionId, HLSSA,
        Instruction, LookupTarget, MemOp, OpCode, Radix, SeqType, SliceOpDir, Terminator,
    },
};

pub trait Value<Context>
where
    Self: Sized + Clone,
{
    fn cmp(&self, b: &Self, cmp_kind: CmpKind, out_type: &Type, ctx: &mut Context) -> Self;
    fn arith(
        &self,
        b: &Self,
        binary_arith_op_kind: BinaryArithOpKind,
        out_type: &Type,
        ctx: &mut Context,
    ) -> Self;
    fn assert_eq(&self, other: &Self, ctx: &mut Context);
    fn assert_r1c(a: &Self, b: &Self, c: &Self, ctx: &mut Context);
    fn array_get(&self, index: &Self, out_type: &Type, ctx: &mut Context) -> Self;
    fn tuple_get(&self, index: usize, out_type: &Type, ctx: &mut Context) -> Self;
    fn array_set(&self, index: &Self, value: &Self, out_type: &Type, ctx: &mut Context) -> Self;
    fn truncate(&self, _from: usize, to: usize, out_type: &Type, ctx: &mut Context) -> Self;
    fn sext(&self, from: usize, to: usize, out_type: &Type, ctx: &mut Context) -> Self;
    fn cast(&self, cast_target: &CastTarget, out_type: &Type, ctx: &mut Context) -> Self;
    fn constrain(a: &Self, b: &Self, c: &Self, ctx: &mut Context);
    fn to_bits(
        &self,
        endianness: Endianness,
        size: usize,
        out_type: &Type,
        ctx: &mut Context,
    ) -> Self;
    fn to_radix(
        &self,
        radix: &Radix<Self>,
        endianness: Endianness,
        size: usize,
        out_type: &Type,
        ctx: &mut Context,
    ) -> Self;
    fn not(&self, out_type: &Type, ctx: &mut Context) -> Self;
    fn of_u(s: usize, v: u128, ctx: &mut Context) -> Self;
    fn of_i(s: usize, v: u128, ctx: &mut Context) -> Self;
    fn of_field(f: Field, ctx: &mut Context) -> Self;
    fn mk_array(a: Vec<Self>, ctx: &mut Context, seq_type: SeqType, elem_type: &Type) -> Self;
    fn mk_tuple(elems: Vec<Self>, ctx: &mut Context, elem_types: &[Type]) -> Self;
    fn alloc(elem_type: &Type, ctx: &mut Context) -> Self;
    fn ptr_write(&self, val: &Self, ctx: &mut Context);
    fn ptr_read(&self, out_type: &Type, ctx: &mut Context) -> Self;
    fn expect_constant_bool(&self, ctx: &mut Context) -> bool;
    fn select(&self, if_t: &Self, if_f: &Self, out_type: &Type, ctx: &mut Context) -> Self;
    fn write_witness(&self, tp: Option<&Type>, ctx: &mut Context) -> Self;
    fn fresh_witness(result_type: &Type, ctx: &mut Context) -> Self;
    fn value_of(&self, ctx: &mut Context) -> Self;
    fn mem_op(&self, kind: MemOp, ctx: &mut Context);
    fn rangecheck(&self, max_bits: usize, ctx: &mut Context);
    fn spread(&self, bits: u8, ctx: &mut Context) -> Self;
    fn unspread(&self, bits: u8, ctx: &mut Context) -> (Self, Self);
}

pub trait Context<V> {
    /// Called when a function call is encountered. If this returns Some, the function
    /// body is skipped and the returned values are used as call results. For unconstrained
    /// calls this MUST return Some, since unconstrained functions may contain JmpIfs on
    /// unknown values that the symbolic executor cannot trace through.
    fn on_call(
        &mut self,
        func: FunctionId,
        params: &mut [V],
        param_types: &[&Type],
        result_types: &[Type],
        unconstrained: bool,
    ) -> Option<Vec<V>>;
    fn on_return(&mut self, returns: &mut [V], return_types: &[Type]);
    fn on_jmp(&mut self, target: BlockId, params: &mut [V], param_types: &[&Type]);

    // TODO it looks odd that this is the only opcode implemented here.
    // This is the _new_ structure, so at some point we should migrate all other opcodes here.
    fn lookup(&mut self, _target: LookupTarget<V>, _keys: Vec<V>, _results: Vec<V>, _flag: V) {
        panic!("ICE: backend does not implement lookup");
    }

    fn dlookup(&mut self, _target: LookupTarget<V>, _keys: Vec<V>, _results: Vec<V>, _flag: V) {
        panic!("ICE: backend does not implement dlookup");
    }

    fn todo(&mut self, payload: &str, _result_types: &[Type]) -> Vec<V> {
        panic!("Todo opcode encountered: {}", payload);
    }

    fn slice_push(&mut self, _slice: &V, _values: &[V], _dir: SliceOpDir) -> V {
        panic!("ICE: backend does not implement slice_push");
    }

    fn slice_len(&mut self, _slice: &V) -> V {
        panic!("ICE: backend does not implement slice_len");
    }

    /// Handle a Guard instruction. Receives the inner opcode, the condition value,
    /// all resolved inner inputs, and result types. Returns values for each result.
    /// The implementer should nuke information on outputs and handle effectful ops
    /// (e.g. Store → nuke ptr contents).
    fn on_guard(
        &mut self,
        inner: &OpCode,
        condition: &V,
        inputs: Vec<&V>,
        result_types: Vec<&Type>,
    ) -> Vec<V>;
}

pub struct SymbolicExecutor {}

impl SymbolicExecutor {
    pub fn new() -> Self {
        Self {}
    }

    #[instrument(skip_all, name = "SymbolicExecutor::run", level = Level::DEBUG)]
    pub fn run<V, Ctx>(
        &self,
        ssa: &HLSSA,
        type_info: &TypeInfo,
        entry_point: FunctionId,
        params: Vec<V>,
        context: &mut Ctx,
    ) where
        V: Value<Ctx>,
        Ctx: Context<V>,
    {
        let mut globals: Vec<Option<V>> = vec![None; ssa.num_globals()];

        self.run_fn(ssa, type_info, entry_point, params, &mut globals, context);
    }

    #[instrument(skip_all, name="SymbolicExecutor::run_fn", level = Level::TRACE, fields(function = %ssa.get_function(fn_id).get_name()))]
    fn run_fn<V, Ctx>(
        &self,
        ssa: &HLSSA,
        type_info: &TypeInfo,
        fn_id: FunctionId,
        mut inputs: Vec<V>,
        globals: &mut Vec<Option<V>>,
        ctx: &mut Ctx,
    ) -> Vec<V>
    where
        V: Value<Ctx>,
        Ctx: Context<V>,
    {
        let fn_body = ssa.get_function(fn_id);
        let fn_type_info = type_info.get_function(fn_id);
        let entry = fn_body.get_entry();
        let mut scope: Vec<Option<V>> = vec![None; fn_body.get_var_num_bound()];

        let call_result = ctx.on_call(
            fn_id,
            &mut inputs,
            &entry.get_parameters().map(|(_, tp)| tp).collect::<Vec<_>>(),
            &fn_body.get_returns(),
            false,
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
                        let lhs_type = fn_type_info.get_value_type(*a);
                        let a = scope[a.0 as usize].as_ref().unwrap();
                        let b = scope[b.0 as usize].as_ref().unwrap();
                        scope[r.0 as usize] =
                            Some(a.cmp(b, *cmp_kind, &lhs_type, ctx));
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
                    crate::compiler::ssa::OpCode::SExt {
                        result: r,
                        value: a,
                        from_bits: from,
                        to_bits: to,
                    } => {
                        let a = scope[a.0 as usize].as_ref().unwrap();
                        scope[r.0 as usize] =
                            Some(a.sext(*from, *to, &fn_type_info.get_value_type(*r), ctx));
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
                        elem_type,
                    } => {
                        scope[r.0 as usize] = Some(V::alloc(elem_type, ctx));
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
                        function: crate::compiler::ssa::CallTarget::Static(function_id),
                        args: arguments,
                        unconstrained,
                    } => {
                        let mut params: Vec<_> = arguments
                            .iter()
                            .map(|id| scope[id.0 as usize].as_ref().unwrap().clone())
                            .collect();
                        let outputs = if *unconstrained {
                            let entry = ssa.get_function(*function_id).get_entry();
                            let param_types: Vec<_> =
                                entry.get_parameters().map(|(_, tp)| tp).collect();
                            let result_types: Vec<_> = returns
                                .iter()
                                .map(|r| fn_type_info.get_value_type(*r).clone())
                                .collect();
                            ctx.on_call(
                                *function_id,
                                &mut params,
                                &param_types,
                                &result_types,
                                true,
                            )
                            .expect("ICE: on_call must return Some for unconstrained calls")
                        } else {
                            // For constrained calls, run_fn handles on_call internally
                            self.run_fn(ssa, type_info, *function_id, params, globals, ctx)
                        };
                        for (i, val) in returns.iter().enumerate() {
                            scope[val.0 as usize] = Some(outputs[i].clone());
                        }
                    }
                    crate::compiler::ssa::OpCode::Call {
                        function: crate::compiler::ssa::CallTarget::Dynamic(_),
                        ..
                    } => {
                        panic!("Dynamic call targets are not supported in symbolic execution")
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
                        let vals: Vec<_> = values
                            .iter()
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
                        ..
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
                        result_type,
                    } => {
                        scope[r.0 as usize] = Some(V::fresh_witness(result_type, ctx));
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
                        let r = globals[*offset as usize]
                            .as_ref()
                            .expect("ReadGlobal: global slot not initialized")
                            .clone();
                        scope[result.0 as usize] = Some(r);
                    }
                    crate::compiler::ssa::OpCode::InitGlobal { global, value } => {
                        globals[*global] = Some(scope[value.0 as usize].as_ref().unwrap().clone());
                    }
                    crate::compiler::ssa::OpCode::DropGlobal { global } => {
                        globals[*global] = None;
                    }
                    crate::compiler::ssa::OpCode::Lookup {
                        target,
                        keys,
                        results,
                        flag,
                    } => {
                        let target = match target {
                            LookupTarget::Rangecheck(n) => LookupTarget::Rangecheck(*n),
                            LookupTarget::Spread(n) => LookupTarget::Spread(*n),
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
                        let flag_value = scope[flag.0 as usize].as_ref().unwrap().clone();
                        ctx.lookup(target, keys, results, flag_value);
                    }
                    crate::compiler::ssa::OpCode::DLookup {
                        target,
                        keys,
                        results,
                        flag,
                    } => {
                        let target = match target {
                            LookupTarget::Rangecheck(n) => LookupTarget::Rangecheck(*n),
                            LookupTarget::Spread(n) => LookupTarget::Spread(*n),
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
                        let flag_value = scope[flag.0 as usize].as_ref().unwrap().clone();
                        ctx.dlookup(target, keys, results, flag_value);
                    }
                    crate::compiler::ssa::OpCode::TupleProj {
                        result: r,
                        tuple: a,
                        idx,
                    } => {
                        let a = scope[a.0 as usize].as_ref().unwrap();
                        scope[r.0 as usize] =
                            Some(a.tuple_get(*idx, &fn_type_info.get_value_type(*r), ctx));
                    }
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
                    crate::compiler::ssa::OpCode::Spread {
                        result,
                        value,
                        bits,
                    } => {
                        let val = scope[value.0 as usize].as_ref().unwrap();
                        scope[result.0 as usize] = Some(val.spread(*bits, ctx));
                    }
                    crate::compiler::ssa::OpCode::Unspread {
                        result_odd,
                        result_even,
                        value,
                        bits,
                    } => {
                        let val = scope[value.0 as usize].as_ref().unwrap();
                        let (odd_val, even_val) = val.unspread(*bits, ctx);
                        scope[result_odd.0 as usize] = Some(odd_val);
                        scope[result_even.0 as usize] = Some(even_val);
                    }
                    crate::compiler::ssa::OpCode::ValueOf { result, value } => {
                        let val = scope[value.0 as usize].clone().unwrap();
                        scope[result.0 as usize] = Some(val.value_of(ctx));
                    }
                    crate::compiler::ssa::OpCode::Const { result, value } => match value {
                        crate::compiler::ssa::ConstValue::U(size, val) => {
                            scope[result.0 as usize] = Some(V::of_u(*size, *val, ctx));
                        }
                        crate::compiler::ssa::ConstValue::I(size, val) => {
                            scope[result.0 as usize] = Some(V::of_i(*size, *val, ctx));
                        }
                        crate::compiler::ssa::ConstValue::Field(val) => {
                            scope[result.0 as usize] = Some(V::of_field(*val, ctx));
                        }
                        crate::compiler::ssa::ConstValue::FnPtr(_) => {
                            todo!("FnPtrConst in symbolic executor");
                        }
                    },
                    crate::compiler::ssa::OpCode::Guard { condition, inner } => {
                        let condition_val = scope[condition.0 as usize].as_ref().unwrap();
                        let inputs: Vec<&V> = inner
                            .get_inputs()
                            .map(|id| scope[id.0 as usize].as_ref().unwrap())
                            .collect();
                        let result_ids: Vec<_> = inner.get_results().cloned().collect();
                        let result_types: Vec<&Type> = result_ids
                            .iter()
                            .map(|id| fn_type_info.get_value_type(*id))
                            .collect();
                        let results = ctx.on_guard(inner, condition_val, inputs, result_types);
                        for (result_id, result_val) in result_ids.iter().zip(results.into_iter()) {
                            scope[result_id.0 as usize] = Some(result_val);
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
