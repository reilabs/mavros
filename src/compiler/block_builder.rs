use std::collections::HashMap;

use crate::compiler::{
    ir::r#type::{SSAType, Type},
    ssa::{
        BinaryArithOpKind, Block, BlockId, CastTarget, CmpKind, Function, Instruction,
        LookupTarget, OpCode, SeqType, Terminator, TupleIdx, ValueId,
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
}

// HL-specific BlockCursor methods
impl BlockCursor<'_, OpCode, Type> {
    /// Build a counted loop: `for i in 0..len { body(i, accumulators) → updated_accumulators }`
    ///
    /// Emits the full three-block structure (header/body/continuation), calls `body`
    /// once to generate the loop body IR, and returns the accumulator values at the
    /// loop exit (the header's block parameters as seen from the continuation).
    ///
    /// - `len`: iteration count (emits u32 constants for 0, 1, len)
    /// - `accumulators`: `(initial_value, type)` pairs for loop-carried state
    /// - `body`: receives `(cursor, index, &[accumulator_params])` → updated accumulator values
    pub fn build_counted_loop(
        &mut self,
        len: usize,
        accumulators: Vec<(ValueId, Type)>,
        body: impl FnOnce(&mut Self, ValueId, &[ValueId]) -> Vec<ValueId>,
    ) -> Vec<ValueId> {
        // Emit constants into current block
        let const_0 = self.instr().u_const(32, 0);
        let const_1 = self.instr().u_const(32, 1);
        let const_len = self.instr().u_const(32, len as u128);

        // Create blocks
        let (header_id, mut header) = self.new_block();
        let (body_id, body_block) = self.new_block();
        let (cont_id, cont_block) = self.new_block();

        // Seal current block → Jmp(header, [0, ...initial_accs])
        let mut init_args = vec![const_0];
        init_args.extend(accumulators.iter().map(|(v, _)| *v));
        self.seal_and_switch(Terminator::Jmp(header_id, init_args), body_id, body_block);

        // Build header: params, condition, JmpIf
        let i_param = self.function.fresh_value();
        let mut header_params = vec![(i_param, Type::u(32))];
        let mut acc_params = vec![];
        for (_, tp) in &accumulators {
            let v = self.function.fresh_value();
            acc_params.push(v);
            header_params.push((v, tp.clone()));
        }
        header.put_parameters(header_params);

        let cond = self.function.fresh_value();
        header.push_instruction(OpCode::Cmp {
            kind: CmpKind::Lt,
            result: cond,
            lhs: i_param,
            rhs: const_len,
        });
        header.set_terminator(Terminator::JmpIf(cond, body_id, cont_id));
        self.insert_block(header_id, header);

        // Call body closure (cursor is at body block, may split it further)
        let updated_accs = body(self, i_param, &acc_params);

        // Increment index, seal body → Jmp(header, [next_i, ...updated_accs])
        let next_i = self.instr().add(i_param, const_1);
        let mut back_args = vec![next_i];
        back_args.extend(updated_accs);
        self.seal_and_switch(Terminator::Jmp(header_id, back_args), cont_id, cont_block);

        // Cursor is now at continuation; return acc params
        acc_params
    }
}
