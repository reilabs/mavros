use crate::compiler::{
    ir::r#type::{SSAType, Type},
    ssa::{
        BlockId, CallTarget, CastTarget, CmpKind, Endianness, Function, FunctionId, HLBlock,
        HLFunction, Instruction, LookupTarget, MemOp, OpCode, Radix, SeqType, SliceOpDir,
        Terminator, TupleIdx, ValueId,
    },
};

// ---------------------------------------------------------------------------
// HLEmitter — unified trait for emitting HL SSA instructions
// ---------------------------------------------------------------------------

pub trait HLEmitter {
    fn fresh_value(&mut self) -> ValueId;
    fn emit(&mut self, op: OpCode);

    // -- Arithmetic --

    fn add(&mut self, lhs: ValueId, rhs: ValueId) -> ValueId {
        let r = self.fresh_value();
        self.emit(OpCode::mk_add(r, lhs, rhs));
        r
    }

    fn sub(&mut self, lhs: ValueId, rhs: ValueId) -> ValueId {
        let r = self.fresh_value();
        self.emit(OpCode::mk_sub(r, lhs, rhs));
        r
    }

    fn mul(&mut self, lhs: ValueId, rhs: ValueId) -> ValueId {
        let r = self.fresh_value();
        self.emit(OpCode::mk_mul(r, lhs, rhs));
        r
    }

    fn div(&mut self, lhs: ValueId, rhs: ValueId) -> ValueId {
        let r = self.fresh_value();
        self.emit(OpCode::mk_div(r, lhs, rhs));
        r
    }

    fn and(&mut self, lhs: ValueId, rhs: ValueId) -> ValueId {
        let r = self.fresh_value();
        self.emit(OpCode::mk_and(r, lhs, rhs));
        r
    }

    fn not(&mut self, value: ValueId) -> ValueId {
        let r = self.fresh_value();
        self.emit(OpCode::Not { result: r, value });
        r
    }

    // -- Comparison --

    fn cmp(&mut self, lhs: ValueId, rhs: ValueId, kind: CmpKind) -> ValueId {
        let r = self.fresh_value();
        self.emit(OpCode::Cmp {
            kind,
            result: r,
            lhs,
            rhs,
        });
        r
    }

    fn eq(&mut self, lhs: ValueId, rhs: ValueId) -> ValueId {
        let r = self.fresh_value();
        self.emit(OpCode::mk_eq(r, lhs, rhs));
        r
    }

    fn lt(&mut self, lhs: ValueId, rhs: ValueId) -> ValueId {
        let r = self.fresh_value();
        self.emit(OpCode::mk_lt(r, lhs, rhs));
        r
    }

    // -- Casts --

    fn cast_to_field(&mut self, value: ValueId) -> ValueId {
        let r = self.fresh_value();
        self.emit(OpCode::mk_cast_to_field(r, value));
        r
    }

    fn cast_to(&mut self, target: CastTarget, value: ValueId) -> ValueId {
        let r = self.fresh_value();
        self.emit(OpCode::mk_cast_to(r, target, value));
        r
    }

    fn cast_to_witness_of(&mut self, value: ValueId) -> ValueId {
        let r = self.fresh_value();
        self.emit(OpCode::mk_cast_to(r, CastTarget::WitnessOf, value));
        r
    }

    fn truncate(&mut self, value: ValueId, to_bits: usize, from_bits: usize) -> ValueId {
        let r = self.fresh_value();
        self.emit(OpCode::Truncate {
            result: r,
            value,
            to_bits,
            from_bits,
        });
        r
    }

    // -- Constants --

    fn field_const(&mut self, value: ark_bn254::Fr) -> ValueId {
        let r = self.fresh_value();
        self.emit(OpCode::mk_field_const(r, value));
        r
    }

    fn u_const(&mut self, bits: usize, value: u128) -> ValueId {
        let r = self.fresh_value();
        self.emit(OpCode::mk_u_const(r, bits, value));
        r
    }

    // -- Witness --

    fn value_of(&mut self, value: ValueId) -> ValueId {
        let r = self.fresh_value();
        self.emit(OpCode::mk_value_of(r, value));
        r
    }

    fn write_witness(&mut self, value: ValueId) -> ValueId {
        let r = self.fresh_value();
        self.emit(OpCode::mk_write_witness(r, value));
        r
    }

    fn pinned_write_witness(&mut self, value: ValueId) -> ValueId {
        let r = self.fresh_value();
        self.emit(OpCode::mk_pinned_write_witness(r, value));
        r
    }

    // -- Memory / aggregates --

    fn array_get(&mut self, array: ValueId, index: ValueId) -> ValueId {
        let r = self.fresh_value();
        self.emit(OpCode::mk_array_get(r, array, index));
        r
    }

    fn array_set(&mut self, array: ValueId, index: ValueId, value: ValueId) -> ValueId {
        let r = self.fresh_value();
        self.emit(OpCode::ArraySet {
            result: r,
            array,
            index,
            value,
        });
        r
    }

    fn tuple_proj(&mut self, tuple: ValueId, idx: TupleIdx) -> ValueId {
        let r = self.fresh_value();
        self.emit(OpCode::TupleProj {
            result: r,
            tuple,
            idx,
        });
        r
    }

    fn mk_tuple(&mut self, elems: Vec<ValueId>, element_types: Vec<Type>) -> ValueId {
        let r = self.fresh_value();
        self.emit(OpCode::MkTuple {
            result: r,
            elems,
            element_types,
        });
        r
    }

    fn mk_seq(&mut self, elems: Vec<ValueId>, seq_type: SeqType, elem_type: Type) -> ValueId {
        let r = self.fresh_value();
        self.emit(OpCode::MkSeq {
            result: r,
            elems,
            seq_type,
            elem_type,
        });
        r
    }

    fn select(&mut self, cond: ValueId, if_t: ValueId, if_f: ValueId) -> ValueId {
        let r = self.fresh_value();
        self.emit(OpCode::Select {
            result: r,
            cond,
            if_t,
            if_f,
        });
        r
    }

    fn load(&mut self, ptr: ValueId) -> ValueId {
        let r = self.fresh_value();
        self.emit(OpCode::Load { result: r, ptr });
        r
    }

    fn alloc(&mut self, elem_type: Type) -> ValueId {
        let r = self.fresh_value();
        self.emit(OpCode::Alloc {
            result: r,
            elem_type,
        });
        r
    }

    // -- Slices --

    fn slice_push(&mut self, slice: ValueId, values: Vec<ValueId>, dir: SliceOpDir) -> ValueId {
        let r = self.fresh_value();
        self.emit(OpCode::SlicePush {
            result: r,
            slice,
            values,
            dir,
        });
        r
    }

    fn slice_len(&mut self, slice: ValueId) -> ValueId {
        let r = self.fresh_value();
        self.emit(OpCode::SliceLen { result: r, slice });
        r
    }

    // -- Bits / Radix --

    fn to_bits(&mut self, value: ValueId, endianness: Endianness, count: usize) -> ValueId {
        let r = self.fresh_value();
        self.emit(OpCode::ToBits {
            result: r,
            value,
            endianness,
            count,
        });
        r
    }

    fn to_radix(
        &mut self,
        value: ValueId,
        radix: Radix<ValueId>,
        endianness: Endianness,
        count: usize,
    ) -> ValueId {
        let r = self.fresh_value();
        self.emit(OpCode::ToRadix {
            result: r,
            value,
            radix,
            endianness,
            count,
        });
        r
    }

    // -- No-result instructions --

    fn constrain(&mut self, a: ValueId, b: ValueId, c: ValueId) {
        self.emit(OpCode::mk_constrain(a, b, c));
    }

    fn store(&mut self, ptr: ValueId, value: ValueId) {
        self.emit(OpCode::Store { ptr, value });
    }

    fn assert_eq(&mut self, lhs: ValueId, rhs: ValueId) {
        self.emit(OpCode::AssertEq { lhs, rhs });
    }

    fn rangecheck(&mut self, value: ValueId, max_bits: usize) {
        self.emit(OpCode::Rangecheck { value, max_bits });
    }

    fn mem_op(&mut self, value: ValueId, kind: MemOp) {
        self.emit(OpCode::MemOp { kind, value });
    }

    fn lookup_rngchk(&mut self, target: LookupTarget<ValueId>, value: ValueId) {
        self.emit(OpCode::mk_lookup_rngchk(target, value));
    }

    fn lookup_rngchk_8(&mut self, value: ValueId) {
        self.emit(OpCode::mk_lookup_rngchk_8(value));
    }

    fn lookup_arr(&mut self, array: ValueId, index: ValueId, result: ValueId) {
        self.emit(OpCode::mk_lookup_arr(array, index, result));
    }

    // -- Globals --

    fn read_global(&mut self, index: u64, typ: Type) -> ValueId {
        let r = self.fresh_value();
        self.emit(OpCode::ReadGlobal {
            result: r,
            offset: index,
            result_type: typ,
        });
        r
    }

    fn init_global(&mut self, global: usize, value: ValueId) {
        self.emit(OpCode::InitGlobal { global, value });
    }

    fn drop_global(&mut self, global: usize) {
        self.emit(OpCode::DropGlobal { global });
    }

    // -- Calls --

    fn call(&mut self, fn_id: FunctionId, args: Vec<ValueId>, n: usize) -> Vec<ValueId> {
        let mut results = Vec::with_capacity(n);
        for _ in 0..n {
            results.push(self.fresh_value());
        }
        self.emit(OpCode::Call {
            results: results.clone(),
            function: CallTarget::Static(fn_id),
            args,
        });
        results
    }

    fn call_indirect(&mut self, fn_ptr: ValueId, args: Vec<ValueId>, n: usize) -> Vec<ValueId> {
        let mut results = Vec::with_capacity(n);
        for _ in 0..n {
            results.push(self.fresh_value());
        }
        self.emit(OpCode::Call {
            results: results.clone(),
            function: CallTarget::Dynamic(fn_ptr),
            args,
        });
        results
    }

    // -- Debug --

    fn todo_op(&mut self, payload: String, results: Vec<ValueId>, result_types: Vec<Type>) {
        self.emit(OpCode::Todo {
            payload,
            results,
            result_types,
        });
    }
}

// ---------------------------------------------------------------------------
// InstrBuilder — lightweight instruction emitter
// ---------------------------------------------------------------------------

pub struct InstrBuilder<'a, Op: Instruction, Ty: SSAType> {
    pub function: &'a mut Function<Op, Ty>,
    pub instructions: &'a mut Vec<Op>,
}

impl<'a, Op: Instruction, Ty: SSAType> InstrBuilder<'a, Op, Ty> {
    pub fn new(function: &'a mut Function<Op, Ty>, instructions: &'a mut Vec<Op>) -> Self {
        Self {
            function,
            instructions,
        }
    }

    /// Push a pre-built instruction (passthrough or pre-allocated result).
    pub fn push(&mut self, op: Op) {
        self.instructions.push(op);
    }
}

impl HLEmitter for InstrBuilder<'_, OpCode, Type> {
    fn fresh_value(&mut self) -> ValueId {
        self.function.fresh_value()
    }

    fn emit(&mut self, op: OpCode) {
        self.instructions.push(op);
    }
}

// ---------------------------------------------------------------------------
// FunctionBuilder — function-level coordinator (no current_block)
// ---------------------------------------------------------------------------

pub struct FunctionBuilder<'a> {
    pub function: &'a mut HLFunction,
}

impl<'a> FunctionBuilder<'a> {
    pub fn new(function: &'a mut HLFunction) -> Self {
        Self { function }
    }

    pub fn add_block(&mut self) -> BlockId {
        self.function.add_block()
    }

    pub fn fresh_value(&mut self) -> ValueId {
        self.function.fresh_value()
    }

    /// Escape hatch for direct function access.
    pub fn function(&mut self) -> &mut HLFunction {
        self.function
    }

    /// Get a BlockEmitter for a specific block.
    pub fn block(&mut self, id: BlockId) -> BlockEmitter<'_> {
        BlockEmitter::new(self.function, id)
    }
}

// ---------------------------------------------------------------------------
// BlockEmitter — emits instructions into a specific block, with block-splitting
// ---------------------------------------------------------------------------

/// Holds the current block *taken out* of the function, so `emit()` is a
/// direct `Vec::push` with no HashMap lookup.  The block is put back into the
/// function on `seal_and_switch` or `Drop`.
pub struct BlockEmitter<'a> {
    pub function: &'a mut HLFunction,
    block_id: BlockId,
    block: HLBlock,
}

impl Drop for BlockEmitter<'_> {
    fn drop(&mut self) {
        let block = std::mem::replace(&mut self.block, HLBlock::empty());
        self.function.put_block(self.block_id, block);
    }
}

impl HLEmitter for BlockEmitter<'_> {
    fn fresh_value(&mut self) -> ValueId {
        self.function.fresh_value()
    }

    fn emit(&mut self, op: OpCode) {
        self.block.push_instruction(op);
    }
}

impl<'a> BlockEmitter<'a> {
    pub fn new(function: &'a mut HLFunction, block_id: BlockId) -> Self {
        let block = function.take_block(block_id);
        Self {
            function,
            block_id,
            block,
        }
    }

    pub fn block_id(&self) -> BlockId {
        self.block_id
    }

    pub fn add_block(&mut self) -> BlockId {
        self.function.add_block()
    }

    pub fn add_parameter(&mut self, typ: Type) -> ValueId {
        let v = self.function.fresh_value();
        self.block.push_parameter(v, typ);
        v
    }

    /// Set the terminator on the current block.
    pub fn set_terminator(&mut self, terminator: Terminator) {
        self.block.set_terminator(terminator);
    }

    /// Set terminator on the current block, put it back, and switch to
    /// `next_id` (whose block is taken out of the function).
    pub fn seal_and_switch(&mut self, terminator: Terminator, next_id: BlockId) {
        self.block.set_terminator(terminator);
        let old_block = std::mem::replace(&mut self.block, self.function.take_block(next_id));
        self.function.put_block(self.block_id, old_block);
        self.block_id = next_id;
    }

    pub fn terminate_jmp(&mut self, dest: BlockId, args: Vec<ValueId>) {
        self.block
            .set_terminator(Terminator::Jmp(dest, args));
    }

    pub fn terminate_jmp_if(&mut self, cond: ValueId, then_dest: BlockId, else_dest: BlockId) {
        self.block
            .set_terminator(Terminator::JmpIf(cond, then_dest, else_dest));
    }

    pub fn terminate_return(&mut self, vals: Vec<ValueId>) {
        self.block
            .set_terminator(Terminator::Return(vals));
    }

    pub fn is_terminated(&self) -> bool {
        self.block.get_terminator().is_some()
    }

    /// Build a loop with the three-block structure: header → body → back-edge.
    ///
    /// The caller provides:
    /// - `params`: `(initial_value, type)` for all loop-carried state
    /// - `header`: receives an `InstrBuilder` targeting the header block and the
    ///   loop parameter `ValueId`s; emits the condition check and returns the
    ///   condition `ValueId`
    /// - `body`: receives the emitter (at the body block) and the loop parameter
    ///   `ValueId`s; emits the body and returns the updated parameter values
    ///   (fed back to the header via the back-edge Jmp)
    ///
    /// Returns the loop parameter `ValueId`s as seen from the continuation block.
    pub fn build_loop(
        &mut self,
        params: Vec<(ValueId, Type)>,
        header: impl FnOnce(&mut InstrBuilder<'_, OpCode, Type>, &[ValueId]) -> ValueId,
        body: impl FnOnce(&mut Self, &[ValueId]) -> Vec<ValueId>,
    ) -> Vec<ValueId> {
        // Create blocks
        let header_id = self.add_block();
        let body_id = self.add_block();
        let cont_id = self.add_block();

        // Seal current block → Jmp(header, initial values)
        let init_values: Vec<ValueId> = params.iter().map(|(v, _)| *v).collect();
        self.seal_and_switch(Terminator::Jmp(header_id, init_values), body_id);

        // Build header parameters
        let mut param_ids = vec![];
        for _ in 0..params.len() {
            let v = self.function.fresh_value();
            param_ids.push(v);
        }
        let header_params: Vec<_> = param_ids
            .iter()
            .zip(params.iter())
            .map(|(v, (_, tp))| (*v, tp.clone()))
            .collect();
        self.function
            .get_block_mut(header_id)
            .put_parameters(header_params);

        // Call header closure to emit condition into a temporary instruction buffer
        let mut header_instructions = vec![];
        let cond = {
            let mut b = InstrBuilder::new(self.function, &mut header_instructions);
            header(&mut b, &param_ids)
        };
        self.function
            .get_block_mut(header_id)
            .put_instructions(header_instructions);
        self.function
            .get_block_mut(header_id)
            .set_terminator(Terminator::JmpIf(cond, body_id, cont_id));

        // Call body closure (emitter is at body block, may split it further)
        let updated = body(self, &param_ids);

        // Seal body → Jmp(header, updated values)
        self.seal_and_switch(Terminator::Jmp(header_id, updated), cont_id);

        // Emitter is now at continuation; return param ids
        param_ids
    }

    /// Build a counted loop: `for i in 0..len { body(i, accumulators) → updated_accumulators }`
    ///
    /// Wrapper around `build_loop` that handles the u32 index, condition (`i < len`),
    /// and increment (`i + 1`). Returns only the accumulator values at loop exit.
    pub fn build_counted_loop(
        &mut self,
        len: usize,
        accumulators: Vec<(ValueId, Type)>,
        body: impl FnOnce(&mut Self, ValueId, &[ValueId]) -> Vec<ValueId>,
    ) -> Vec<ValueId> {
        // Emit constants into current block (before the loop)
        let const_0 = self.u_const(32, 0);
        let const_1 = self.u_const(32, 1);
        let const_len = self.u_const(32, len as u128);

        // Loop params: [index, ...accumulators]
        let mut params = vec![(const_0, Type::u(32))];
        params.extend(accumulators);

        let results = self.build_loop(
            params,
            |b, loop_params| b.lt(loop_params[0], const_len),
            |emitter, loop_params| {
                let i_val = loop_params[0];
                let acc_params = &loop_params[1..];
                let updated_accs = body(emitter, i_val, acc_params);
                let next_i = emitter.add(i_val, const_1);
                let mut result = vec![next_i];
                result.extend(updated_accs);
                result
            },
        );

        // Skip index, return accumulator results
        results[1..].to_vec()
    }
}
