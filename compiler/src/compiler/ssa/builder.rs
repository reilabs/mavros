use std::{fmt::Debug, hash::Hash};

use crate::compiler::ssa::{
    Block, BlockId, Function, FunctionId, Instruction, Located, SSA, SSAType, Terminator, ValueId,
};

// ---------------------------------------------------------------------------
// InstrBuilder — lightweight instruction emitter
// ---------------------------------------------------------------------------

pub struct InstrBuilder<'a, Op: Instruction, Ty: SSAType, C: Clone + Debug + Eq + Hash> {
    pub function: &'a mut Function<Op, Ty>,
    pub ssa: &'a mut SSA<Op, Ty, C>,
    pub instructions: &'a mut Vec<Located<Op>>,
}

impl<'a, Op: Instruction, Ty: SSAType, C: Clone + Debug + Eq + Hash> InstrBuilder<'a, Op, Ty, C> {
    pub fn new(
        function: &'a mut Function<Op, Ty>,
        ssa: &'a mut SSA<Op, Ty, C>,
        instructions: &'a mut Vec<Located<Op>>,
    ) -> Self {
        Self {
            function,
            ssa,
            instructions,
        }
    }

    /// Push a pre-built instruction (passthrough or pre-allocated result).
    pub fn push(&mut self, instruction: impl Into<Located<Op>>) {
        self.instructions.push(instruction.into());
    }
}

// ---------------------------------------------------------------------------
// FunctionBuilder — function-level coordinator (no current_block)
// ---------------------------------------------------------------------------

pub struct FunctionBuilder<'a, Op: Instruction, Ty: SSAType, C: Clone + Debug + Eq + Hash> {
    pub function: &'a mut Function<Op, Ty>,
    pub ssa: &'a mut SSA<Op, Ty, C>,
}

impl<'a, Op: Instruction, Ty: SSAType, C: Clone + Debug + Eq + Hash>
    FunctionBuilder<'a, Op, Ty, C>
{
    pub fn new(function: &'a mut Function<Op, Ty>, ssa: &'a mut SSA<Op, Ty, C>) -> Self {
        Self { function, ssa }
    }

    pub fn add_block(&mut self, f: impl FnOnce(&mut BlockEmitter<'_, Op, Ty, C>)) -> BlockId {
        let (id, mut emitter) = BlockEmitter::from_new_block(self.function, self.ssa);
        f(&mut emitter);
        id
    }

    pub fn fresh_value(&mut self) -> ValueId {
        self.ssa.fresh_value()
    }

    /// Intern a constant into the SSA and return its `ValueId`.
    pub fn emit_const(&mut self, value: C) -> ValueId {
        self.ssa.add_const(value)
    }

    /// Escape hatch for direct function access.
    pub fn function(&mut self) -> &mut Function<Op, Ty> {
        self.function
    }

    /// Escape hatch for direct SSA access.
    pub fn ssa(&mut self) -> &mut SSA<Op, Ty, C> {
        self.ssa
    }

    /// Get a BlockEmitter for a specific block.
    pub fn block(&mut self, id: BlockId) -> BlockEmitter<'_, Op, Ty, C> {
        BlockEmitter::new(self.function, self.ssa, id)
    }
}

// ---------------------------------------------------------------------------
// BlockEmitter — emits instructions into a specific block, with block-splitting
// ---------------------------------------------------------------------------

/// Holds the current block *taken out* of the function, so `emit()` is a
/// direct `Vec::push` with no HashMap lookup.  The block is put back into the
/// function on `seal_and_switch` or `Drop`.
pub struct BlockEmitter<'a, Op: Instruction, Ty: SSAType, C: Clone + Debug + Eq + Hash> {
    pub function: &'a mut Function<Op, Ty>,
    pub ssa: &'a mut SSA<Op, Ty, C>,
    pub(crate) block_id: BlockId,
    pub(crate) block: Block<Op, Ty>,
}

impl<Op: Instruction, Ty: SSAType, C: Clone + Debug + Eq + Hash> Drop
    for BlockEmitter<'_, Op, Ty, C>
{
    fn drop(&mut self) {
        let block = std::mem::replace(&mut self.block, Block::empty());
        self.function.put_block(self.block_id, block);
    }
}

impl<'a, Op: Instruction, Ty: SSAType, C: Clone + Debug + Eq + Hash> BlockEmitter<'a, Op, Ty, C> {
    /// Create an emitter for an existing block (takes it out of the function).
    pub fn new(
        function: &'a mut Function<Op, Ty>,
        ssa: &'a mut SSA<Op, Ty, C>,
        block_id: BlockId,
    ) -> Self {
        let block = function.take_block(block_id);
        Self {
            function,
            ssa,
            block_id,
            block,
        }
    }

    /// Create an emitter for a fresh block (not yet in the function).
    /// The block is inserted into the function on drop.
    pub(crate) fn from_new_block(
        function: &'a mut Function<Op, Ty>,
        ssa: &'a mut SSA<Op, Ty, C>,
    ) -> (BlockId, Self) {
        let (block_id, block) = function.next_virtual_block();
        (
            block_id,
            Self {
                function,
                ssa,
                block_id,
                block,
            },
        )
    }

    pub fn block_id(&self) -> BlockId {
        self.block_id
    }

    pub fn instruction_count(&self) -> usize {
        self.block.instruction_count()
    }

    pub fn set_instruction_source_locations_from(
        &mut self,
        start: usize,
        source_location: Option<crate::compiler::ssa::SourceLocation>,
    ) {
        self.block
            .set_instruction_source_locations_from(start, source_location);
    }

    pub fn add_block(&mut self) -> (BlockId, &mut Block<Op, Ty>) {
        self.function.add_block_mut()
    }

    pub fn add_parameter(&mut self, typ: Ty) -> ValueId {
        let v = self.ssa.fresh_value();
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

    pub fn emit_instruction(&mut self, instruction: impl Into<Located<Op>>) {
        self.block.push_located_instruction(instruction.into());
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
        header: impl FnOnce(&mut InstrBuilder<'_, Op, Ty, C>, &[ValueId]) -> ValueId,
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
            let v = self.ssa.fresh_value();
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
            let mut b = InstrBuilder::new(self.function, self.ssa, &mut header_instructions);
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
            .map(|ty| (self.ssa.fresh_value(), ty))
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

// ---------------------------------------------------------------------------
// SSABuilder — top-level coordinator that owns `&mut SSA` and dispenses
//              function builders via closure-based scopes.
// ---------------------------------------------------------------------------

pub struct SSABuilder<'a, Op: Instruction, Ty: SSAType, C: Clone + Debug + Eq + Hash> {
    ssa: &'a mut SSA<Op, Ty, C>,
}

impl<'a, Op: Instruction, Ty: SSAType, C: Clone + Debug + Eq + Hash> SSABuilder<'a, Op, Ty, C> {
    pub fn new(ssa: &'a mut SSA<Op, Ty, C>) -> Self {
        Self { ssa }
    }

    /// Build a fresh function from scratch, inserting it into the SSA on
    /// closure return. The function being built is held by a local variable
    /// (NOT in `ssa.functions`) for the closure's lifetime, so the builder
    /// can hand out `&mut Function` and `&mut SSA` without aliasing.
    pub fn add_function<R>(
        &mut self,
        name: String,
        body: impl FnOnce(&mut FunctionBuilder<'_, Op, Ty, C>) -> R,
    ) -> (FunctionId, R) {
        let mut function = Function::<Op, Ty>::empty(name);
        let result = {
            let mut fb = FunctionBuilder::new(&mut function, self.ssa);
            body(&mut fb)
        };
        let id = self.ssa.insert_function(function);
        (id, result)
    }

    /// Edit an existing function: take it out of `ssa.functions`, hand it
    /// to the closure as a `FunctionBuilder`, put it back on return.
    ///
    /// Panics if `fid` does not exist or has already been taken out
    /// (re-entrant `modify_function` for the same `fid`).
    pub fn modify_function<R>(
        &mut self,
        fid: FunctionId,
        body: impl FnOnce(&mut FunctionBuilder<'_, Op, Ty, C>) -> R,
    ) -> R {
        let mut function = self.ssa.take_function(fid);
        let result = {
            let mut fb = FunctionBuilder::new(&mut function, self.ssa);
            body(&mut fb)
        };
        self.ssa.put_function(fid, function);
        result
    }

    /// Convenience: add a parameter to a specific block of a specific function
    /// without spelling out the `modify_function` closure.
    pub fn add_block_parameter(&mut self, fid: FunctionId, blk: BlockId, ty: Ty) -> ValueId {
        self.modify_function(fid, |fb| fb.block(blk).add_parameter(ty))
    }

    /// Allocate a fresh `ValueId` from the SSA-wide counter.
    pub fn fresh_value(&mut self) -> ValueId {
        self.ssa.fresh_value()
    }

    /// Escape hatch for direct SSA access.
    pub fn ssa(&mut self) -> &mut SSA<Op, Ty, C> {
        self.ssa
    }
}

#[cfg(test)]
mod tests {
    use crate::compiler::ssa::{
        Function, Located, SourceLocation, SourcePosition, ValueId,
        builder::InstrBuilder,
        hlssa::{
            HLSSA, OpCode, Type,
            builder::{HLEmitter, HLSSABuilder},
        },
    };

    fn test_location() -> SourceLocation {
        SourceLocation::new(
            "src/main.nr".to_string(),
            SourcePosition::new(7, 11),
            SourcePosition::new(7, 16),
        )
    }

    /// `ValueId`s issued inside two different functions must not collide:
    /// the counter lives on the SSA, not the function.
    #[test]
    fn global_value_ids_are_unique_across_functions() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let mut sb = HLSSABuilder::new(&mut ssa);
        let main_id = sb.ssa().get_unique_entrypoint_id();
        let other_id = sb.ssa().add_function("other".to_string());

        let v_main = sb.modify_function(main_id, |fb| fb.fresh_value());
        let v_other = sb.modify_function(other_id, |fb| fb.fresh_value());

        assert_ne!(v_main, v_other);
    }

    /// `prepare_rebuild` forwards the SSA-wide counter, so allocations after
    /// rebuild continue from where the original counter left off.
    #[test]
    fn prepare_rebuild_preserves_value_counter() {
        let ssa = HLSSA::with_main("main".to_string());
        let _v0 = ssa.fresh_value();
        let _v1 = ssa.fresh_value();
        let _v2 = ssa.fresh_value();
        let bound_before = ssa.value_num_bound();

        let (rebuilt, _functions, _globals) = ssa.prepare_rebuild();
        assert_eq!(rebuilt.value_num_bound(), bound_before);

        let next = rebuilt.fresh_value();
        assert_eq!(next.0 as usize, bound_before);
    }

    /// An empty `modify_function` closure must leave the SSA structurally
    /// unchanged: function count, counter, and the surviving function map
    /// all preserved.
    #[test]
    fn modify_function_noop_roundtrip() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let main_id = ssa.get_unique_entrypoint_id();
        let _ = ssa.add_function("other".to_string());

        // Allocate some values so the counter is non-zero.
        let _v0 = ssa.fresh_value();
        let _v1 = ssa.fresh_value();

        let function_ids_before: Vec<_> = ssa.get_function_ids().collect();
        let counter_before = ssa.value_num_bound();

        let mut sb = HLSSABuilder::new(&mut ssa);
        for fid in &function_ids_before {
            sb.modify_function(*fid, |_fb| {});
        }

        assert_eq!(ssa.value_num_bound(), counter_before);
        let function_ids_after: Vec<_> = ssa.get_function_ids().collect();
        assert_eq!(function_ids_after.len(), function_ids_before.len());
        assert!(ssa.get_function(main_id).get_blocks().count() > 0);
    }

    #[test]
    fn block_emitter_emit_accepts_located_instruction() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let main_id = ssa.get_unique_entrypoint_id();
        let loc = test_location();

        {
            let mut sb = HLSSABuilder::new(&mut ssa);
            sb.modify_function(main_id, |fb| {
                let entry_id = fb.function.get_entry_id();
                let mut block = fb.block(entry_id);
                block.emit(Located::with(
                    OpCode::Not {
                        result: ValueId(1),
                        value: ValueId(0),
                    },
                    loc.clone(),
                ));
            });
        }

        let entry = ssa.get_unique_entrypoint().get_entry();
        assert_eq!(entry.get_instruction_source_location(0), Some(&loc));
    }

    #[test]
    fn instr_builder_accepts_source_locations_for_temporary_buffers() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let mut function = Function::<OpCode, Type>::empty("tmp".to_string());
        let mut instructions = Vec::new();
        let loc = test_location();

        {
            let mut builder = InstrBuilder::new(&mut function, &mut ssa, &mut instructions);
            builder.emit(Located::with(
                OpCode::Not {
                    result: ValueId(1),
                    value: ValueId(0),
                },
                loc.clone(),
            ));
        }

        assert_eq!(instructions.len(), 1);
        assert_eq!(instructions[0].get_location(), Some(&loc));
    }
}
