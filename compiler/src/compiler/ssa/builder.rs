use crate::compiler::ssa::{Block, BlockId, Function, Instruction, SSAType, Terminator, ValueId};

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

// ---------------------------------------------------------------------------
// FunctionBuilder — function-level coordinator (no current_block)
// ---------------------------------------------------------------------------

pub struct FunctionBuilder<'a, Op: Instruction, Ty: SSAType> {
    pub function: &'a mut Function<Op, Ty>,
}

impl<'a, Op: Instruction, Ty: SSAType> FunctionBuilder<'a, Op, Ty> {
    pub fn new(function: &'a mut Function<Op, Ty>) -> Self {
        Self { function }
    }

    pub fn add_block(&mut self, f: impl FnOnce(&mut BlockEmitter<'_, Op, Ty>)) -> BlockId {
        let (id, mut emitter) = BlockEmitter::from_new_block(self.function);
        f(&mut emitter);
        id
    }

    pub fn fresh_value(&mut self) -> ValueId {
        self.function.fresh_value()
    }

    /// Escape hatch for direct function access.
    pub fn function(&mut self) -> &mut Function<Op, Ty> {
        self.function
    }

    /// Get a BlockEmitter for a specific block.
    pub fn block(&mut self, id: BlockId) -> BlockEmitter<'_, Op, Ty> {
        BlockEmitter::new(self.function, id)
    }
}

// ---------------------------------------------------------------------------
// BlockEmitter — emits instructions into a specific block, with block-splitting
// ---------------------------------------------------------------------------

/// Holds the current block *taken out* of the function, so `emit()` is a
/// direct `Vec::push` with no HashMap lookup.  The block is put back into the
/// function on `seal_and_switch` or `Drop`.
pub struct BlockEmitter<'a, Op: Instruction, Ty: SSAType> {
    pub function: &'a mut Function<Op, Ty>,
    pub(crate) block_id: BlockId,
    pub(crate) block: Block<Op, Ty>,
}

impl<Op: Instruction, Ty: SSAType> Drop for BlockEmitter<'_, Op, Ty> {
    fn drop(&mut self) {
        let block = std::mem::replace(&mut self.block, Block::empty());
        self.function.put_block(self.block_id, block);
    }
}

impl<'a, Op: Instruction, Ty: SSAType> BlockEmitter<'a, Op, Ty> {
    /// Create an emitter for an existing block (takes it out of the function).
    pub fn new(function: &'a mut Function<Op, Ty>, block_id: BlockId) -> Self {
        let block = function.take_block(block_id);
        Self {
            function,
            block_id,
            block,
        }
    }

    /// Create an emitter for a fresh block (not yet in the function).
    /// The block is inserted into the function on drop.
    pub(crate) fn from_new_block(function: &'a mut Function<Op, Ty>) -> (BlockId, Self) {
        let (block_id, block) = function.next_virtual_block();
        (
            block_id,
            Self {
                function,
                block_id,
                block,
            },
        )
    }

    pub fn block_id(&self) -> BlockId {
        self.block_id
    }

    pub fn add_block(&mut self) -> (BlockId, &mut Block<Op, Ty>) {
        self.function.add_block_mut()
    }

    pub fn add_parameter(&mut self, typ: Ty) -> ValueId {
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
        self.block.set_terminator(Terminator::Jmp(dest, args));
    }

    pub fn terminate_jmp_if(&mut self, cond: ValueId, then_dest: BlockId, else_dest: BlockId) {
        self.block
            .set_terminator(Terminator::JmpIf(cond, then_dest, else_dest));
    }

    pub fn terminate_return(&mut self, vals: Vec<ValueId>) {
        self.block.set_terminator(Terminator::Return(vals));
    }

    pub fn is_terminated(&self) -> bool {
        self.block.get_terminator().is_some()
    }

    /// Build a loop with the three-block structure: header -> body -> back-edge.
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
        params: Vec<(ValueId, Ty)>,
        header: impl FnOnce(&mut InstrBuilder<'_, Op, Ty>, &[ValueId]) -> ValueId,
        body: impl FnOnce(&mut Self, &[ValueId]) -> Vec<ValueId>,
    ) -> Vec<ValueId> {
        // Create blocks
        let (header_id, _) = self.add_block();
        let (body_id, _) = self.add_block();
        let (cont_id, _) = self.add_block();

        // Seal current block -> Jmp(header, initial values)
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

        // Seal body -> Jmp(header, updated values)
        self.seal_and_switch(Terminator::Jmp(header_id, updated), cont_id);

        // Emitter is now at continuation; return param ids
        param_ids
    }

    /// Build an if-then-else diamond:
    ///
    ///   current_block → JmpIf(cond, then_blk, else_blk)
    ///   then_blk → ... → Jmp(merge_blk, then_results)
    ///   else_blk → ... → Jmp(merge_blk, else_results)
    ///   merge_blk(params...) → continues here
    ///
    /// `result_types` defines the merge block parameters.
    /// `then_branch` and `else_branch` each receive the emitter and return values
    /// that are passed to the merge block.
    ///
    /// Returns the merge parameter `ValueId`s.
    pub fn build_if_else(
        &mut self,
        cond: ValueId,
        result_types: Vec<Ty>,
        then_branch: impl FnOnce(&mut Self) -> Vec<ValueId>,
        else_branch: impl FnOnce(&mut Self) -> Vec<ValueId>,
    ) -> Vec<ValueId> {
        let results: Vec<_> = result_types
            .into_iter()
            .map(|ty| (self.function.fresh_value(), ty))
            .collect();
        let merge_params: Vec<_> = results.iter().map(|(v, _)| *v).collect();
        self.build_if_else_into(cond, results, then_branch, else_branch);
        merge_params
    }

    /// Like `build_if_else`, but uses pre-allocated `ValueId`s for the merge
    /// block parameters instead of creating fresh ones.  This is useful when
    /// the caller already has a `ValueId` that downstream code references
    /// (e.g. the result of a Guard being lowered).
    pub fn build_if_else_into(
        &mut self,
        cond: ValueId,
        results: Vec<(ValueId, Ty)>,
        then_branch: impl FnOnce(&mut Self) -> Vec<ValueId>,
        else_branch: impl FnOnce(&mut Self) -> Vec<ValueId>,
    ) {
        let (then_blk, _) = self.add_block();
        let (else_blk, _) = self.add_block();
        let (merge_blk, _) = self.add_block();

        // Add merge parameters with pre-allocated IDs
        for (v, ty) in results {
            self.function.get_block_mut(merge_blk).push_parameter(v, ty);
        }

        // Seal current block → JmpIf
        self.seal_and_switch(Terminator::JmpIf(cond, then_blk, else_blk), then_blk);

        // Build then branch
        let then_results = then_branch(self);
        self.seal_and_switch(Terminator::Jmp(merge_blk, then_results), else_blk);

        // Build else branch
        let else_results = else_branch(self);
        self.seal_and_switch(Terminator::Jmp(merge_blk, else_results), merge_blk);
    }
}
