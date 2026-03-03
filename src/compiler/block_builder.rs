use std::collections::HashMap;

use crate::compiler::{
    ir::r#type::{SSAType, Type},
    ssa::{
        Block, BlockId, CastTarget, Function, Instruction, LookupTarget, OpCode, SeqType,
        Terminator, TupleIdx, ValueId,
    },
};

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

    /// Allocate a fresh ValueId.
    pub fn fresh_value(&mut self) -> ValueId {
        self.function.fresh_value()
    }
}

// HL-specific convenience methods
impl InstrBuilder<'_, OpCode, Type> {
    // -- Arithmetic --

    pub fn add(&mut self, lhs: ValueId, rhs: ValueId) -> ValueId {
        let r = self.function.fresh_value();
        self.instructions.push(OpCode::mk_add(r, lhs, rhs));
        r
    }

    pub fn sub(&mut self, lhs: ValueId, rhs: ValueId) -> ValueId {
        let r = self.function.fresh_value();
        self.instructions.push(OpCode::mk_sub(r, lhs, rhs));
        r
    }

    pub fn mul(&mut self, lhs: ValueId, rhs: ValueId) -> ValueId {
        let r = self.function.fresh_value();
        self.instructions.push(OpCode::mk_mul(r, lhs, rhs));
        r
    }

    pub fn div(&mut self, lhs: ValueId, rhs: ValueId) -> ValueId {
        let r = self.function.fresh_value();
        self.instructions.push(OpCode::mk_div(r, lhs, rhs));
        r
    }

    pub fn and(&mut self, lhs: ValueId, rhs: ValueId) -> ValueId {
        let r = self.function.fresh_value();
        self.instructions.push(OpCode::mk_and(r, lhs, rhs));
        r
    }

    // -- Comparison --

    pub fn eq(&mut self, lhs: ValueId, rhs: ValueId) -> ValueId {
        let r = self.function.fresh_value();
        self.instructions.push(OpCode::mk_eq(r, lhs, rhs));
        r
    }

    pub fn lt(&mut self, lhs: ValueId, rhs: ValueId) -> ValueId {
        let r = self.function.fresh_value();
        self.instructions.push(OpCode::mk_lt(r, lhs, rhs));
        r
    }

    // -- Casts --

    pub fn cast_to_field(&mut self, value: ValueId) -> ValueId {
        let r = self.function.fresh_value();
        self.instructions.push(OpCode::mk_cast_to_field(r, value));
        r
    }

    pub fn cast_to(&mut self, target: CastTarget, value: ValueId) -> ValueId {
        let r = self.function.fresh_value();
        self.instructions.push(OpCode::mk_cast_to(r, target, value));
        r
    }

    pub fn cast_to_witness_of(&mut self, value: ValueId) -> ValueId {
        let r = self.function.fresh_value();
        self.instructions
            .push(OpCode::mk_cast_to(r, CastTarget::WitnessOf, value));
        r
    }

    // -- Constants --

    pub fn field_const(&mut self, value: ark_bn254::Fr) -> ValueId {
        let r = self.function.fresh_value();
        self.instructions.push(OpCode::mk_field_const(r, value));
        r
    }

    pub fn u_const(&mut self, bits: usize, value: u128) -> ValueId {
        let r = self.function.fresh_value();
        self.instructions.push(OpCode::mk_u_const(r, bits, value));
        r
    }

    // -- Witness --

    pub fn value_of(&mut self, value: ValueId) -> ValueId {
        let r = self.function.fresh_value();
        self.instructions.push(OpCode::mk_value_of(r, value));
        r
    }

    pub fn write_witness(&mut self, value: ValueId) -> ValueId {
        let r = self.function.fresh_value();
        self.instructions.push(OpCode::mk_write_witness(r, value));
        r
    }

    pub fn pinned_write_witness(&mut self, value: ValueId) -> ValueId {
        let r = self.function.fresh_value();
        self.instructions
            .push(OpCode::mk_pinned_write_witness(r, value));
        r
    }

    // -- Memory / aggregates --

    pub fn array_get(&mut self, array: ValueId, index: ValueId) -> ValueId {
        let r = self.function.fresh_value();
        self.instructions
            .push(OpCode::mk_array_get(r, array, index));
        r
    }

    pub fn array_set(&mut self, array: ValueId, index: ValueId, value: ValueId) -> ValueId {
        let r = self.function.fresh_value();
        self.instructions.push(OpCode::ArraySet {
            result: r,
            array,
            index,
            value,
        });
        r
    }

    pub fn tuple_proj(&mut self, tuple: ValueId, idx: TupleIdx) -> ValueId {
        let r = self.function.fresh_value();
        self.instructions.push(OpCode::TupleProj {
            result: r,
            tuple,
            idx,
        });
        r
    }

    pub fn mk_tuple(&mut self, elems: Vec<ValueId>, element_types: Vec<Type>) -> ValueId {
        let r = self.function.fresh_value();
        self.instructions.push(OpCode::MkTuple {
            result: r,
            elems,
            element_types,
        });
        r
    }

    pub fn mk_seq(&mut self, elems: Vec<ValueId>, seq_type: SeqType, elem_type: Type) -> ValueId {
        let r = self.function.fresh_value();
        self.instructions.push(OpCode::MkSeq {
            result: r,
            elems,
            seq_type,
            elem_type,
        });
        r
    }

    pub fn select(&mut self, cond: ValueId, if_t: ValueId, if_f: ValueId) -> ValueId {
        let r = self.function.fresh_value();
        self.instructions.push(OpCode::Select {
            result: r,
            cond,
            if_t,
            if_f,
        });
        r
    }

    pub fn load(&mut self, ptr: ValueId) -> ValueId {
        let r = self.function.fresh_value();
        self.instructions.push(OpCode::Load { result: r, ptr });
        r
    }

    pub fn alloc(&mut self, elem_type: Type) -> ValueId {
        let r = self.function.fresh_value();
        self.instructions.push(OpCode::Alloc {
            result: r,
            elem_type,
        });
        r
    }

    // -- No-result instructions --

    pub fn constrain(&mut self, a: ValueId, b: ValueId, c: ValueId) {
        self.instructions.push(OpCode::mk_constrain(a, b, c));
    }

    pub fn store(&mut self, ptr: ValueId, value: ValueId) {
        self.instructions.push(OpCode::Store { ptr, value });
    }

    pub fn lookup_rngchk(&mut self, target: LookupTarget<ValueId>, value: ValueId) {
        self.instructions
            .push(OpCode::mk_lookup_rngchk(target, value));
    }

    pub fn lookup_rngchk_8(&mut self, value: ValueId) {
        self.instructions.push(OpCode::mk_lookup_rngchk_8(value));
    }

    pub fn lookup_arr(&mut self, array: ValueId, index: ValueId, result: ValueId) {
        self.instructions
            .push(OpCode::mk_lookup_arr(array, index, result));
    }

    pub fn assert_eq(&mut self, lhs: ValueId, rhs: ValueId) {
        self.instructions.push(OpCode::AssertEq { lhs, rhs });
    }
}

// ---------------------------------------------------------------------------
// BlockCursor — block-splitting state machine
// ---------------------------------------------------------------------------

pub struct BlockCursor<'a, Op: Instruction, Ty: SSAType> {
    pub function: &'a mut Function<Op, Ty>,
    current_block_id: BlockId,
    current_block: Block<Op, Ty>,
    instructions: Vec<Op>,
    new_blocks: HashMap<BlockId, Block<Op, Ty>>,
}

impl<'a, Op: Instruction, Ty: SSAType> BlockCursor<'a, Op, Ty> {
    /// Start a cursor for the given block. The block should already have had
    /// its instructions and terminator taken out for processing.
    pub fn new(
        function: &'a mut Function<Op, Ty>,
        block_id: BlockId,
        block: Block<Op, Ty>,
    ) -> Self {
        Self {
            function,
            current_block_id: block_id,
            current_block: block,
            instructions: vec![],
            new_blocks: HashMap::new(),
        }
    }

    /// Get an InstrBuilder for emitting instructions into the current block.
    pub fn instr(&mut self) -> InstrBuilder<'_, Op, Ty> {
        InstrBuilder {
            function: self.function,
            instructions: &mut self.instructions,
        }
    }

    /// Create a new block without switching to it.
    pub fn new_block(&mut self) -> (BlockId, Block<Op, Ty>) {
        self.function.next_virtual_block()
    }

    /// Finalize the current block with the given terminator, then switch the
    /// cursor to `next_id` / `next_block`.
    pub fn seal_and_switch(
        &mut self,
        terminator: Terminator,
        next_id: BlockId,
        next_block: Block<Op, Ty>,
    ) {
        self.current_block
            .put_instructions(std::mem::take(&mut self.instructions));
        self.current_block.set_terminator(terminator);
        let old_block = std::mem::replace(&mut self.current_block, next_block);
        self.new_blocks.insert(self.current_block_id, old_block);
        self.current_block_id = next_id;
    }

    /// Store a separately-built block (e.g. loop headers, loop bodies).
    pub fn insert_block(&mut self, id: BlockId, block: Block<Op, Ty>) {
        self.new_blocks.insert(id, block);
    }

    /// Seal the final block with the given terminator and return all blocks.
    pub fn finish_with_terminator(
        mut self,
        terminator: Terminator,
    ) -> HashMap<BlockId, Block<Op, Ty>> {
        self.current_block
            .put_instructions(std::mem::take(&mut self.instructions));
        self.current_block.set_terminator(terminator);
        self.new_blocks
            .insert(self.current_block_id, self.current_block);
        self.new_blocks
    }

    /// Seal the final block (keeping its existing terminator) and return all blocks.
    pub fn finish(mut self) -> HashMap<BlockId, Block<Op, Ty>> {
        self.current_block
            .put_instructions(std::mem::take(&mut self.instructions));
        self.new_blocks
            .insert(self.current_block_id, self.current_block);
        self.new_blocks
    }

    pub fn current_block_id(&self) -> BlockId {
        self.current_block_id
    }

    /// Build a loop with the three-block structure: header → body → back-edge.
    ///
    /// This is fully generic over `Op`/`Ty`. The caller provides:
    /// - `params`: `(initial_value, type)` for all loop-carried state
    /// - `header`: receives an `InstrBuilder` targeting the header block and the
    ///   loop parameter `ValueId`s; emits the condition check and returns the
    ///   condition `ValueId`
    /// - `body`: receives the cursor (at the body block) and the loop parameter
    ///   `ValueId`s; emits the body and returns the updated parameter values
    ///   (fed back to the header via the back-edge Jmp)
    ///
    /// Returns the loop parameter `ValueId`s as seen from the continuation block.
    pub fn build_loop(
        &mut self,
        params: Vec<(ValueId, Ty)>,
        header: impl FnOnce(&mut InstrBuilder<'_, Op, Ty>, &[ValueId]) -> ValueId,
        body: impl FnOnce(&mut Self, &[ValueId]) -> Vec<ValueId>,
    ) -> Vec<ValueId> {
        // Create blocks
        let (header_id, mut header_block) = self.new_block();
        let (body_id, body_block) = self.new_block();
        let (cont_id, cont_block) = self.new_block();

        // Seal current block → Jmp(header, initial values)
        let init_values: Vec<ValueId> = params.iter().map(|(v, _)| *v).collect();
        self.seal_and_switch(Terminator::Jmp(header_id, init_values), body_id, body_block);

        // Build header parameters
        let mut param_ids = vec![];
        let mut header_params = vec![];
        for (_, tp) in &params {
            let v = self.function.fresh_value();
            param_ids.push(v);
            header_params.push((v, tp.clone()));
        }
        header_block.put_parameters(header_params);

        // Call header closure to emit condition into a temporary instruction buffer
        let mut header_instructions = vec![];
        let cond = {
            let mut b = InstrBuilder::new(self.function, &mut header_instructions);
            header(&mut b, &param_ids)
        };
        header_block.put_instructions(header_instructions);
        header_block.set_terminator(Terminator::JmpIf(cond, body_id, cont_id));
        self.insert_block(header_id, header_block);

        // Call body closure (cursor is at body block, may split it further)
        let updated = body(self, &param_ids);

        // Seal body → Jmp(header, updated values)
        self.seal_and_switch(Terminator::Jmp(header_id, updated), cont_id, cont_block);

        // Cursor is now at continuation; return param ids
        param_ids
    }
}

// HL-specific BlockCursor methods
impl BlockCursor<'_, OpCode, Type> {
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
        let const_0 = self.instr().u_const(32, 0);
        let const_1 = self.instr().u_const(32, 1);
        let const_len = self.instr().u_const(32, len as u128);

        // Loop params: [index, ...accumulators]
        let mut params = vec![(const_0, Type::u(32))];
        params.extend(accumulators);

        let results = self.build_loop(
            params,
            |b, loop_params| b.lt(loop_params[0], const_len),
            |cursor, loop_params| {
                let i_val = loop_params[0];
                let acc_params = &loop_params[1..];
                let updated_accs = body(cursor, i_val, acc_params);
                let next_i = cursor.instr().add(i_val, const_1);
                let mut result = vec![next_i];
                result.extend(updated_accs);
                result
            },
        );

        // Skip index, return accumulator results
        results[1..].to_vec()
    }
}
