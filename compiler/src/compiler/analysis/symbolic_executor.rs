//! Implements a generic symbolic execution engine over the HL SSA that different clients can plug
//! into by providing their own `Value` and `Context` implementations that specialize it for their
//! use-case.

use tracing::{Level, instrument};

use crate::{
    collections::HashMap,
    compiler::{
        Field,
        analysis::types::TypeInfo,
        ssa::{
            BlockId, FunctionId, Instruction, Terminator, ValueId,
            hlssa::{
                BinaryArithOpKind, CallTarget, CastTarget, CmpKind, Constant, Endianness, HLSSA,
                HLSSAConstantsSnapshot, LookupTarget, OpCode, Radix, RefCountOp,
                SequenceTargetType, SliceOpDir, Type, TypeExpr,
            },
        },
        util::ice_non_elided_tuple,
    },
};

pub trait Value<Context>
where
    Self: Sized + Clone,
{
    fn ult(&self, b: &Self, ctx: &mut Context) -> Self;
    fn slt(&self, b: &Self, bits: usize, ctx: &mut Context) -> Self;
    fn eq(&self, b: &Self, ctx: &mut Context) -> Self;
    fn arith(
        &self,
        b: &Self,
        binary_arith_op_kind: BinaryArithOpKind,
        out_type: &Type,
        ctx: &mut Context,
    ) -> Self;
    fn assert_bool(&self, ctx: &mut Context);
    fn assert_cmp(kind: CmpKind, a: &Self, b: &Self, lhs_type: &Type, ctx: &mut Context);
    fn assert_r1c(a: &Self, b: &Self, c: &Self, ctx: &mut Context);
    fn array_get(&self, index: &Self, out_type: &Type, ctx: &mut Context) -> Self;
    fn array_set(&self, index: &Self, value: &Self, out_type: &Type, ctx: &mut Context) -> Self;
    fn sext(&self, from: usize, to: usize, out_type: &Type, ctx: &mut Context) -> Self;
    fn bit_range(&self, offset: usize, width: usize, out_type: &Type, ctx: &mut Context) -> Self;
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
    fn of_blob(elem_type: Type, elements: Vec<Self>, ctx: &mut Context) -> Self;
    fn expect_blob(&self, ctx: &mut Context) -> Vec<Self>;
    fn mk_array(
        a: Vec<Self>,
        ctx: &mut Context,
        seq_type: SequenceTargetType,
        elem_type: &Type,
    ) -> Self;
    fn alloc(elem_type: &Type, value: &Self, ctx: &mut Context) -> Self;
    fn ptr_write(&self, val: &Self, ctx: &mut Context);
    fn ptr_read(&self, out_type: &Type, ctx: &mut Context) -> Self;
    fn expect_constant_bool(&self, ctx: &mut Context) -> bool;
    fn select(&self, if_t: &Self, if_f: &Self, out_type: &Type, ctx: &mut Context) -> Self;
    fn write_witness(&self, tp: Option<&Type>, ctx: &mut Context) -> Self;
    fn fresh_witness(result_type: &Type, ctx: &mut Context) -> Self;
    fn mem_op(&self, kind: RefCountOp, ctx: &mut Context);
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
    fn lookup(&mut self, _target: LookupTarget<V>, _args: Vec<V>, _flag: V) {
        panic!("ICE: backend does not implement lookup");
    }

    fn dlookup(&mut self, _target: LookupTarget<V>, _args: Vec<V>, _flag: V) {
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

        // Materialize the SSA constants into `V`s once, up front. Constant ids are global and stable
        // across functions, so we build this map a single time and share it by reference with every
        // `run_fn` instead of rebuilding it (via the emitting `of_*` path) on every function entry.
        let constants = ssa.const_snapshot();
        let consts = materialize_constants(&constants, context);

        self.run_fn(
            ssa,
            type_info,
            entry_point,
            params,
            &mut globals,
            &consts,
            context,
        );
    }

    #[instrument(skip_all, name="SymbolicExecutor::run_fn", level = Level::TRACE, fields(function = %ssa.get_function(fn_id).get_name()))]
    fn run_fn<V, Ctx>(
        &self,
        ssa: &HLSSA,
        type_info: &TypeInfo,
        fn_id: FunctionId,
        mut inputs: Vec<V>,
        globals: &mut Vec<Option<V>>,
        consts: &HashMap<ValueId, V>,
        ctx: &mut Ctx,
    ) -> Vec<V>
    where
        V: Value<Ctx>,
        Ctx: Context<V>,
    {
        let fn_body = ssa.get_function(fn_id);
        let fn_type_info = type_info.get_function(fn_id);
        let entry = fn_body.get_entry();

        let mut scope = Scope::new(consts);

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
            scope.insert(*ppos, pval.clone());
        }

        let mut current = Some(entry);

        while let Some(block) = current {
            for instr in block.get_instructions() {
                match instr {
                    OpCode::Cmp {
                        kind: cmp_kind,
                        result: r,
                        lhs: a,
                        rhs: b,
                    } => {
                        let lhs_type = fn_type_info.get_value_type(*a);
                        let a = &scope[a];
                        let b = &scope[b];
                        let stripped = lhs_type.strip_witness();
                        scope.insert(
                            *r,
                            match cmp_kind {
                                CmpKind::Eq => a.eq(b, ctx),
                                CmpKind::Lt => match &stripped.expr {
                                    TypeExpr::I(bits) => a.slt(b, *bits, ctx),
                                    _ => a.ult(b, ctx),
                                },
                            },
                        );
                    }
                    OpCode::BinaryArithOp {
                        kind: binary_arith_op_kind,
                        result: r,
                        lhs: a,
                        rhs: b,
                    } => {
                        let a = &scope[a];
                        let b = &scope[b];
                        scope.insert(
                            *r,
                            a.arith(
                                b,
                                *binary_arith_op_kind,
                                &fn_type_info.get_value_type(*r),
                                ctx,
                            ),
                        );
                    }
                    OpCode::Cast {
                        result: r,
                        value: a,
                        target: cast_target,
                    } => {
                        let a = &scope[a];
                        scope.insert(
                            *r,
                            a.cast(cast_target, &fn_type_info.get_value_type(*r), ctx),
                        );
                    }
                    OpCode::SExt {
                        result: r,
                        value: a,
                        from_bits: from,
                        to_bits: to,
                    } => {
                        let a = &scope[a];
                        scope.insert(
                            *r,
                            a.sext(*from, *to, &fn_type_info.get_value_type(*r), ctx),
                        );
                    }
                    OpCode::BitRange {
                        result: r,
                        value: a,
                        offset,
                        width,
                    } => {
                        let a = &scope[a];
                        scope.insert(
                            *r,
                            a.bit_range(*offset, *width, &fn_type_info.get_value_type(*r), ctx),
                        );
                    }
                    OpCode::Not {
                        result: r,
                        value: a,
                    } => {
                        let a = &scope[a];
                        scope.insert(*r, a.not(&fn_type_info.get_value_type(*r), ctx));
                    }
                    OpCode::MkSeq {
                        result: r,
                        elems: a,
                        seq_type,
                        elem_type,
                    } => {
                        let a = a.iter().map(|id| scope[id].clone()).collect::<Vec<_>>();
                        scope.insert(*r, V::mk_array(a, ctx, *seq_type, elem_type));
                    }
                    OpCode::MkSeqOfBlob {
                        result: r,
                        element_type,
                        blob,
                    } => {
                        let a = scope[blob].expect_blob(ctx);
                        let len = a.len();
                        scope.insert(
                            *r,
                            V::mk_array(a, ctx, SequenceTargetType::Array(len), element_type),
                        );
                    }
                    OpCode::MkRepeated {
                        result: r,
                        element,
                        seq_type,
                        count,
                        elem_type,
                    } => {
                        let elem = scope[element].clone();
                        let a = vec![elem; *count];
                        scope.insert(*r, V::mk_array(a, ctx, *seq_type, elem_type));
                    }
                    OpCode::Alloc {
                        result: r,
                        elem_type,
                        value,
                    } => {
                        let v = scope[value].clone();
                        scope.insert(*r, V::alloc(elem_type, &v, ctx));
                    }
                    OpCode::Store { ptr, value: val } => {
                        let ptr = &scope[ptr];
                        let val = &scope[val];
                        ptr.ptr_write(val, ctx);
                    }
                    OpCode::Load { result: r, ptr } => {
                        let ptr = &scope[ptr];
                        scope.insert(*r, ptr.ptr_read(&fn_type_info.get_value_type(*r), ctx));
                    }
                    OpCode::AssertR1C { a, b, c } => {
                        let a = &scope[a];
                        let b = &scope[b];
                        let c = &scope[c];
                        V::assert_r1c(a, b, c, ctx);
                    }
                    OpCode::Call {
                        results: returns,
                        function: CallTarget::Static(function_id),
                        args: arguments,
                        unconstrained,
                    } => {
                        let mut params: Vec<_> =
                            arguments.iter().map(|id| scope[id].clone()).collect();
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
                            self.run_fn(ssa, type_info, *function_id, params, globals, consts, ctx)
                        };
                        for (i, val) in returns.iter().enumerate() {
                            scope.insert(*val, outputs[i].clone());
                        }
                    }
                    OpCode::Call {
                        function: CallTarget::Dynamic(_),
                        ..
                    } => {
                        panic!("Dynamic call targets are not supported in symbolic execution")
                    }
                    OpCode::ArrayGet {
                        result: r,
                        array: a,
                        index: i,
                    } => {
                        let a = &scope[a];
                        let i = &scope[i];
                        scope.insert(*r, a.array_get(i, &fn_type_info.get_value_type(*r), ctx));
                    }
                    OpCode::ArraySet {
                        result: r,
                        array: arr,
                        index: i,
                        value: v,
                    } => {
                        let a = &scope[arr];
                        let i = &scope[i];
                        let v = &scope[v];
                        scope.insert(*r, a.array_set(i, v, &fn_type_info.get_value_type(*r), ctx));
                    }
                    OpCode::SlicePush {
                        result,
                        slice,
                        values,
                        dir,
                    } => {
                        let sl = &scope[slice];
                        let vals: Vec<_> = values.iter().map(|v| scope[v].clone()).collect();
                        scope.insert(*result, ctx.slice_push(sl, &vals, *dir));
                    }
                    OpCode::SliceLen {
                        result: r,
                        slice: sl,
                    } => {
                        let sl = &scope[sl];
                        scope.insert(*r, ctx.slice_len(sl));
                    }
                    OpCode::Select {
                        result: r,
                        cond,
                        if_t,
                        if_f,
                    } => {
                        let cond = &scope[cond];
                        let if_t = &scope[if_t];
                        let if_f = &scope[if_f];
                        scope.insert(
                            *r,
                            cond.select(if_t, if_f, &fn_type_info.get_value_type(*r), ctx),
                        );
                    }
                    OpCode::ToBits {
                        result: r,
                        value: a,
                        endianness,
                        count: size,
                    } => {
                        let a = &scope[a];
                        scope.insert(
                            *r,
                            a.to_bits(*endianness, *size, &fn_type_info.get_value_type(*r), ctx),
                        );
                    }
                    OpCode::ToRadix {
                        result: r,
                        value: a,
                        radix,
                        endianness,
                        count: size,
                    } => {
                        let a = &scope[a];
                        let radix = match radix {
                            Radix::Bytes => Radix::Bytes,
                            Radix::Dyn(radix) => Radix::Dyn(scope[radix].clone()),
                        };
                        scope.insert(
                            *r,
                            a.to_radix(
                                &radix,
                                *endianness,
                                *size,
                                &fn_type_info.get_value_type(*r),
                                ctx,
                            ),
                        );
                    }
                    OpCode::WriteWitness {
                        result: r,
                        value: a,
                        ..
                    } => {
                        let a = &scope[a];
                        if let Some(r) = r {
                            scope.insert(
                                *r,
                                a.write_witness(Some(fn_type_info.get_value_type(*r)), ctx),
                            );
                        } else {
                            a.write_witness(None, ctx);
                        }
                    }
                    OpCode::FreshWitness {
                        result: r,
                        result_type,
                    } => {
                        scope.insert(*r, V::fresh_witness(result_type, ctx));
                    }
                    OpCode::Constrain { a, b, c } => {
                        let a = &scope[a];
                        let b = &scope[b];
                        let c = &scope[c];
                        V::constrain(a, b, c, ctx);
                    }
                    OpCode::Assert { value } => {
                        let v = &scope[value];
                        V::assert_bool(v, ctx);
                    }
                    OpCode::AssertCmp {
                        kind,
                        lhs: a,
                        rhs: b,
                    } => {
                        let lhs_type = fn_type_info.get_value_type(*a);
                        let a = &scope[a];
                        let b = &scope[b];
                        V::assert_cmp(*kind, a, b, lhs_type, ctx);
                    }
                    OpCode::MemOp { kind, value } => {
                        let value = &scope[value];
                        value.mem_op(*kind, ctx);
                    }
                    OpCode::NextDCoeff { result: _a } => {
                        todo!()
                    }
                    OpCode::BumpD {
                        matrix: _matrix,
                        variable: _a,
                        sensitivity: _b,
                    } => {
                        todo!()
                    }
                    OpCode::MulConst {
                        result: _,
                        const_val: _,
                        var: _,
                    } => {
                        todo!()
                    }
                    OpCode::Rangecheck { value: v, max_bits } => {
                        let v = &scope[v];
                        v.rangecheck(*max_bits, ctx);
                    }
                    OpCode::ReadGlobal {
                        result,
                        offset,
                        result_type: _,
                    } => {
                        let r = globals[*offset as usize]
                            .as_ref()
                            .expect("ReadGlobal: global slot not initialized")
                            .clone();
                        scope.insert(*result, r);
                    }
                    OpCode::InitGlobal { global, value } => {
                        globals[*global] = Some(scope[value].clone());
                    }
                    OpCode::DropGlobal { global } => {
                        globals[*global] = None;
                    }
                    OpCode::Lookup { target, args, flag } => {
                        let target = match target {
                            LookupTarget::Rangecheck(n) => LookupTarget::Rangecheck(*n),
                            LookupTarget::Spread(n) => LookupTarget::Spread(*n),
                            LookupTarget::DynRangecheck(v) => {
                                LookupTarget::DynRangecheck(scope[v].clone())
                            }
                            LookupTarget::Array(arr) => LookupTarget::Array(scope[arr].clone()),
                        };
                        let args = args.iter().map(|id| scope[id].clone()).collect::<Vec<_>>();
                        let flag_value = scope[flag].clone();
                        ctx.lookup(target, args, flag_value);
                    }
                    OpCode::DLookup { target, args, flag } => {
                        let target = match target {
                            LookupTarget::Rangecheck(n) => LookupTarget::Rangecheck(*n),
                            LookupTarget::Spread(n) => LookupTarget::Spread(*n),
                            LookupTarget::DynRangecheck(v) => {
                                LookupTarget::DynRangecheck(scope[v].clone())
                            }
                            LookupTarget::Array(arr) => LookupTarget::Array(scope[arr].clone()),
                        };
                        let args = args.iter().map(|id| scope[id].clone()).collect::<Vec<_>>();
                        let flag_value = scope[flag].clone();
                        ctx.dlookup(target, args, flag_value);
                    }
                    OpCode::TupleProj { .. }
                    | OpCode::TupleRefProj { .. }
                    | OpCode::MkTuple { .. } => ice_non_elided_tuple(),
                    OpCode::Todo {
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
                            scope.insert(*result_id, result_value.clone());
                        }
                    }
                    OpCode::Spread {
                        result,
                        value,
                        bits,
                    } => {
                        let val = &scope[value];
                        scope.insert(*result, val.spread(*bits, ctx));
                    }
                    OpCode::Unspread {
                        result_odd,
                        result_even,
                        value,
                        bits,
                    } => {
                        let val = &scope[value];
                        let (odd_val, even_val) = val.unspread(*bits, ctx);
                        scope.insert(*result_odd, odd_val);
                        scope.insert(*result_even, even_val);
                    }
                    OpCode::Guard { condition, inner } => {
                        let condition_val = &scope[condition];
                        let inputs: Vec<&V> = inner.get_inputs().map(|id| &scope[id]).collect();
                        let result_ids: Vec<_> = inner.get_results().cloned().collect();
                        let result_types: Vec<&Type> = result_ids
                            .iter()
                            .map(|id| fn_type_info.get_value_type(*id))
                            .collect();
                        let results = ctx.on_guard(inner, condition_val, inputs, result_types);
                        for (result_id, result_val) in result_ids.iter().zip(results.into_iter()) {
                            scope.insert(*result_id, result_val);
                        }
                    }
                }
            }

            match block.get_terminator().unwrap() {
                Terminator::Return(returns) => {
                    let mut outputs = returns
                        .iter()
                        .map(|id| scope[id].clone())
                        .collect::<Vec<_>>();
                    ctx.on_return(&mut outputs, &fn_body.get_returns());
                    return outputs;
                }
                Terminator::Jmp(target, params) => {
                    let mut params = params
                        .iter()
                        .map(|id| scope[id].clone())
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
                        scope.insert(*i, val);
                    }
                    current = Some(target_block);
                }
                Terminator::JmpIf(cond, if_true, if_false) => {
                    let cond = &scope[cond];
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

/// Per-function value environment, layered over the shared constant scope.
///
/// Locals (parameters and instruction results) are stored per call, constants are resolved on
/// lookup miss relying on unique value IDs.
struct Scope<'c, V> {
    locals: HashMap<ValueId, V>,
    consts: &'c HashMap<ValueId, V>,
}

impl<'c, V> Scope<'c, V> {
    fn new(consts: &'c HashMap<ValueId, V>) -> Self {
        Self {
            locals: HashMap::default(),
            consts,
        }
    }

    fn insert(&mut self, id: ValueId, v: V) {
        self.locals.insert(id, v);
    }
}

impl<V> std::ops::Index<&ValueId> for Scope<'_, V> {
    type Output = V;

    fn index(&self, id: &ValueId) -> &V {
        self.locals
            .get(id)
            .or_else(|| self.consts.get(id))
            .expect("ICE: value id not in scope")
    }
}

/// Build the constant scope: every interned SSA constant turned into a `V`, exactly once.
///
/// Sources from [`HLSSA::const_snapshot`], which releases the constants lock before returning,
/// rather than [`HLSSA::for_each_const`], which holds the read lock across the walk. The `of_*`
/// calls below may intern constants via `add_const` (which takes the constants write lock), so
/// running them while the read lock is held would deadlock.
fn materialize_constants<V, Ctx>(
    constants: &HLSSAConstantsSnapshot,
    ctx: &mut Ctx,
) -> HashMap<ValueId, V>
where
    V: Value<Ctx>,
{
    let mut consts = HashMap::default();
    for (vid, cv) in constants {
        let v = materialize_constant_value(cv.as_ref(), ctx);
        consts.insert(*vid, v);
    }
    consts
}

fn materialize_constant_value<V, Ctx>(constant: &Constant, ctx: &mut Ctx) -> V
where
    V: Value<Ctx>,
{
    match constant {
        Constant::U(size, val) => V::of_u(*size, *val, ctx),
        Constant::I(size, val) => V::of_i(*size, *val, ctx),
        Constant::Field(val) => V::of_field(*val, ctx),
        Constant::FnPtr(_) => {
            todo!("FnPtrConst in symbolic executor");
        }
        Constant::Blob(blob) => {
            let elements = blob
                .elements
                .iter()
                .map(|element| materialize_constant_value(element, ctx))
                .collect();
            V::of_blob(blob.elem_type.clone(), elements, ctx)
        }
    }
}
