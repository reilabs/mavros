use crate::compiler::{
    ir::r#type::{SSAType, Type},
    llssa::{FieldArithOp, IntArithOp, IntCmpOp, LLOp, LLStruct, LLType},
    ssa::{
        BinaryArithOpKind, Block, BlockId, CallTarget, CastTarget, CmpKind, ConstValue, Endianness,
        Function, FunctionId, Instruction, LookupTarget, MemOp, OpCode, Radix, SeqType, SliceOpDir,
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
        self.emit(OpCode::BinaryArithOp {
            kind: BinaryArithOpKind::Add,
            result: r,
            lhs,
            rhs,
        });
        r
    }

    fn sub(&mut self, lhs: ValueId, rhs: ValueId) -> ValueId {
        let r = self.fresh_value();
        self.emit(OpCode::BinaryArithOp {
            kind: BinaryArithOpKind::Sub,
            result: r,
            lhs,
            rhs,
        });
        r
    }

    fn mul(&mut self, lhs: ValueId, rhs: ValueId) -> ValueId {
        let r = self.fresh_value();
        self.emit(OpCode::BinaryArithOp {
            kind: BinaryArithOpKind::Mul,
            result: r,
            lhs,
            rhs,
        });
        r
    }

    fn div(&mut self, lhs: ValueId, rhs: ValueId) -> ValueId {
        let r = self.fresh_value();
        self.emit(OpCode::BinaryArithOp {
            kind: BinaryArithOpKind::Div,
            result: r,
            lhs,
            rhs,
        });
        r
    }

    fn and(&mut self, lhs: ValueId, rhs: ValueId) -> ValueId {
        let r = self.fresh_value();
        self.emit(OpCode::BinaryArithOp {
            kind: BinaryArithOpKind::And,
            result: r,
            lhs,
            rhs,
        });
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
        self.emit(OpCode::Cmp {
            kind: CmpKind::Eq,
            result: r,
            lhs,
            rhs,
        });
        r
    }

    fn lt(&mut self, lhs: ValueId, rhs: ValueId) -> ValueId {
        let r = self.fresh_value();
        self.emit(OpCode::Cmp {
            kind: CmpKind::Lt,
            result: r,
            lhs,
            rhs,
        });
        r
    }

    // -- Casts --

    fn cast_to_field(&mut self, value: ValueId) -> ValueId {
        let r = self.fresh_value();
        self.emit(OpCode::Cast {
            result: r,
            value,
            target: CastTarget::Field,
        });
        r
    }

    fn cast_to(&mut self, target: CastTarget, value: ValueId) -> ValueId {
        let r = self.fresh_value();
        self.emit(OpCode::Cast {
            result: r,
            value,
            target,
        });
        r
    }

    fn cast_to_witness_of(&mut self, value: ValueId) -> ValueId {
        let r = self.fresh_value();
        self.emit(OpCode::Cast {
            result: r,
            value,
            target: CastTarget::WitnessOf,
        });
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
        self.emit(OpCode::Const {
            result: r,
            value: ConstValue::Field(value),
        });
        r
    }

    fn u_const(&mut self, bits: usize, value: u128) -> ValueId {
        let r = self.fresh_value();
        self.emit(OpCode::Const {
            result: r,
            value: ConstValue::U(bits, value),
        });
        r
    }

    // -- Witness --

    fn value_of(&mut self, value: ValueId) -> ValueId {
        let r = self.fresh_value();
        self.emit(OpCode::ValueOf { result: r, value });
        r
    }

    fn write_witness(&mut self, value: ValueId) -> ValueId {
        let r = self.fresh_value();
        self.emit(OpCode::WriteWitness {
            result: Some(r),
            value,
            pinned: false,
        });
        r
    }

    fn pinned_write_witness(&mut self, value: ValueId) -> ValueId {
        let r = self.fresh_value();
        self.emit(OpCode::WriteWitness {
            result: Some(r),
            value,
            pinned: true,
        });
        r
    }

    // -- Memory / aggregates --

    fn array_get(&mut self, array: ValueId, index: ValueId) -> ValueId {
        let r = self.fresh_value();
        self.emit(OpCode::ArrayGet {
            result: r,
            array,
            index,
        });
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
        self.emit(OpCode::Constrain { a, b, c });
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
        self.emit(OpCode::Lookup {
            target,
            keys: vec![value],
            results: vec![],
        });
    }

    fn lookup_rngchk_8(&mut self, value: ValueId) {
        self.emit(OpCode::Lookup {
            target: LookupTarget::Rangecheck(8),
            keys: vec![value],
            results: vec![],
        });
    }

    fn lookup_arr(&mut self, array: ValueId, index: ValueId, result: ValueId) {
        self.emit(OpCode::Lookup {
            target: LookupTarget::Array(array),
            keys: vec![index],
            results: vec![result],
        });
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
// LLEmitter — unified trait for emitting LL SSA instructions
// ---------------------------------------------------------------------------

pub trait LLEmitter {
    fn fresh_value(&mut self) -> ValueId;
    fn emit_ll(&mut self, op: LLOp);

    // -- Constants --

    fn int_const(&mut self, bits: u32, value: u64) -> ValueId {
        let r = self.fresh_value();
        self.emit_ll(LLOp::IntConst {
            result: r,
            bits,
            value,
        });
        r
    }

    fn null_ptr(&mut self) -> ValueId {
        let r = self.fresh_value();
        self.emit_ll(LLOp::NullPtr { result: r });
        r
    }

    // -- Integer Arithmetic --

    fn int_arith(&mut self, kind: IntArithOp, a: ValueId, b: ValueId) -> ValueId {
        let r = self.fresh_value();
        self.emit_ll(LLOp::IntArith {
            kind,
            result: r,
            a,
            b,
        });
        r
    }

    fn int_add(&mut self, a: ValueId, b: ValueId) -> ValueId {
        self.int_arith(IntArithOp::Add, a, b)
    }

    fn int_sub(&mut self, a: ValueId, b: ValueId) -> ValueId {
        self.int_arith(IntArithOp::Sub, a, b)
    }

    fn int_mul(&mut self, a: ValueId, b: ValueId) -> ValueId {
        self.int_arith(IntArithOp::Mul, a, b)
    }

    fn not(&mut self, value: ValueId) -> ValueId {
        let r = self.fresh_value();
        self.emit_ll(LLOp::Not { result: r, value });
        r
    }

    // -- Integer Comparison --

    fn int_cmp(&mut self, kind: IntCmpOp, a: ValueId, b: ValueId) -> ValueId {
        let r = self.fresh_value();
        self.emit_ll(LLOp::IntCmp {
            kind,
            result: r,
            a,
            b,
        });
        r
    }

    fn int_eq(&mut self, a: ValueId, b: ValueId) -> ValueId {
        self.int_cmp(IntCmpOp::Eq, a, b)
    }

    fn int_ult(&mut self, a: ValueId, b: ValueId) -> ValueId {
        self.int_cmp(IntCmpOp::ULt, a, b)
    }

    // -- Width conversion --

    fn truncate(&mut self, value: ValueId, to_bits: u32) -> ValueId {
        let r = self.fresh_value();
        self.emit_ll(LLOp::Truncate {
            result: r,
            value,
            to_bits,
        });
        r
    }

    fn zext(&mut self, value: ValueId, to_bits: u32) -> ValueId {
        let r = self.fresh_value();
        self.emit_ll(LLOp::ZExt {
            result: r,
            value,
            to_bits,
        });
        r
    }

    // -- Field Arithmetic --

    fn field_arith(&mut self, kind: FieldArithOp, a: ValueId, b: ValueId) -> ValueId {
        let r = self.fresh_value();
        self.emit_ll(LLOp::FieldArith {
            kind,
            result: r,
            a,
            b,
        });
        r
    }

    fn field_neg(&mut self, src: ValueId) -> ValueId {
        let r = self.fresh_value();
        self.emit_ll(LLOp::FieldNeg { result: r, src });
        r
    }

    fn field_eq(&mut self, a: ValueId, b: ValueId) -> ValueId {
        let r = self.fresh_value();
        self.emit_ll(LLOp::FieldEq { result: r, a, b });
        r
    }

    fn field_to_limbs(&mut self, src: ValueId) -> ValueId {
        let r = self.fresh_value();
        self.emit_ll(LLOp::FieldToLimbs { result: r, src });
        r
    }

    fn field_from_limbs(&mut self, limbs: ValueId) -> ValueId {
        let r = self.fresh_value();
        self.emit_ll(LLOp::FieldFromLimbs { result: r, limbs });
        r
    }

    // -- Aggregate --

    fn mk_struct(&mut self, struct_type: LLStruct, fields: Vec<ValueId>) -> ValueId {
        let r = self.fresh_value();
        self.emit_ll(LLOp::MkStruct {
            result: r,
            struct_type,
            fields,
        });
        r
    }

    fn extract_field(&mut self, value: ValueId, struct_type: LLStruct, field: usize) -> ValueId {
        let r = self.fresh_value();
        self.emit_ll(LLOp::ExtractField {
            result: r,
            value,
            struct_type,
            field,
        });
        r
    }

    fn insert_field(
        &mut self,
        base: ValueId,
        struct_type: LLStruct,
        field: usize,
        value: ValueId,
    ) -> ValueId {
        let r = self.fresh_value();
        self.emit_ll(LLOp::InsertField {
            result: r,
            base,
            struct_type,
            field,
            value,
        });
        r
    }

    // -- Memory --

    fn heap_alloc(&mut self, struct_type: LLStruct, flex_count: Option<ValueId>) -> ValueId {
        let r = self.fresh_value();
        self.emit_ll(LLOp::HeapAlloc {
            result: r,
            struct_type,
            flex_count,
        });
        r
    }

    fn free(&mut self, ptr: ValueId) {
        self.emit_ll(LLOp::Free { ptr });
    }

    fn ll_load(&mut self, ptr: ValueId, ty: LLType) -> ValueId {
        let r = self.fresh_value();
        self.emit_ll(LLOp::Load { result: r, ptr, ty });
        r
    }

    fn ll_store(&mut self, ptr: ValueId, value: ValueId) {
        self.emit_ll(LLOp::Store { ptr, value });
    }

    fn struct_field_ptr(&mut self, ptr: ValueId, struct_type: LLStruct, field: usize) -> ValueId {
        let r = self.fresh_value();
        self.emit_ll(LLOp::StructFieldPtr {
            result: r,
            ptr,
            struct_type,
            field,
        });
        r
    }

    fn array_elem_ptr(&mut self, ptr: ValueId, elem_type: LLStruct, index: ValueId) -> ValueId {
        let r = self.fresh_value();
        self.emit_ll(LLOp::ArrayElemPtr {
            result: r,
            ptr,
            elem_type,
            index,
        });
        r
    }

    fn memcpy(
        &mut self,
        dst: ValueId,
        src: ValueId,
        struct_type: LLStruct,
        count: Option<ValueId>,
    ) {
        self.emit_ll(LLOp::Memcpy {
            dst,
            src,
            struct_type,
            count,
        });
    }

    // -- Selection --

    fn select(&mut self, cond: ValueId, if_t: ValueId, if_f: ValueId) -> ValueId {
        let r = self.fresh_value();
        self.emit_ll(LLOp::Select {
            result: r,
            cond,
            if_t,
            if_f,
        });
        r
    }

    // -- Calls --

    fn call(&mut self, func: FunctionId, args: Vec<ValueId>, num_results: usize) -> Vec<ValueId> {
        let results: Vec<ValueId> = (0..num_results).map(|_| self.fresh_value()).collect();
        self.emit_ll(LLOp::Call {
            results: results.clone(),
            func,
            args,
        });
        results
    }

    // -- Globals --

    fn global_addr(&mut self, global_id: usize) -> ValueId {
        let r = self.fresh_value();
        self.emit_ll(LLOp::GlobalAddr {
            result: r,
            global_id,
        });
        r
    }

    // -- VM / Constraint --

    fn constrain(&mut self, a: ValueId, b: ValueId, c: ValueId) {
        self.emit_ll(LLOp::Constrain { a, b, c });
    }

    fn write_witness(&mut self, value: ValueId) {
        self.emit_ll(LLOp::WriteWitness { value });
    }

    // -- Trap --

    fn trap(&mut self) {
        self.emit_ll(LLOp::Trap);
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

impl HLEmitter for HLInstrBuilder<'_> {
    fn fresh_value(&mut self) -> ValueId {
        self.function.fresh_value()
    }

    fn emit(&mut self, op: OpCode) {
        self.instructions.push(op);
    }
}

impl LLEmitter for LLInstrBuilder<'_> {
    fn fresh_value(&mut self) -> ValueId {
        self.function.fresh_value()
    }

    fn emit_ll(&mut self, op: LLOp) {
        self.instructions.push(op);
    }
}

// ---------------------------------------------------------------------------
// Type aliases
// ---------------------------------------------------------------------------

pub type HLInstrBuilder<'a> = InstrBuilder<'a, OpCode, Type>;
pub type LLInstrBuilder<'a> = InstrBuilder<'a, LLOp, LLType>;
pub type HLFunctionBuilder<'a> = FunctionBuilder<'a, OpCode, Type>;
pub type LLFunctionBuilder<'a> = FunctionBuilder<'a, LLOp, LLType>;
pub type HLBlockEmitter<'a> = BlockEmitter<'a, OpCode, Type>;
pub type LLBlockEmitter<'a> = BlockEmitter<'a, LLOp, LLType>;

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
    block_id: BlockId,
    block: Block<Op, Ty>,
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
}

// -- HLEmitter for BlockEmitter<OpCode, Type> --------------------------------

impl HLEmitter for HLBlockEmitter<'_> {
    fn fresh_value(&mut self) -> ValueId {
        self.function.fresh_value()
    }

    fn emit(&mut self, op: OpCode) {
        self.block.push_instruction(op);
    }
}

impl HLBlockEmitter<'_> {
    /// Build a counted loop: `for i in 0..len { body(i, accumulators) -> updated_accumulators }`
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

// -- LLEmitter for BlockEmitter<LLOp, LLType> --------------------------------

impl LLEmitter for LLBlockEmitter<'_> {
    fn fresh_value(&mut self) -> ValueId {
        self.function.fresh_value()
    }

    fn emit_ll(&mut self, op: LLOp) {
        self.block.push_instruction(op);
    }
}

impl LLBlockEmitter<'_> {
    /// Build a counted loop: `for i in 0..count { body(i, accumulators) -> updated_accumulators }`
    ///
    /// LL variant: uses i64 index with int_ult/int_add.
    pub fn build_counted_loop(
        &mut self,
        count: usize,
        accumulators: Vec<(ValueId, LLType)>,
        body: impl FnOnce(&mut Self, ValueId, &[ValueId]) -> Vec<ValueId>,
    ) -> Vec<ValueId> {
        let const_0 = self.int_const(64, 0);
        let const_1 = self.int_const(64, 1);
        let const_len = self.int_const(64, count as u64);

        let mut params = vec![(const_0, LLType::i64())];
        params.extend(accumulators);

        let results = self.build_loop(
            params,
            |b, loop_params| b.int_ult(loop_params[0], const_len),
            |emitter, loop_params| {
                let i_val = loop_params[0];
                let acc_params = &loop_params[1..];
                let updated_accs = body(emitter, i_val, acc_params);
                let next_i = emitter.int_add(i_val, const_1);
                let mut result = vec![next_i];
                result.extend(updated_accs);
                result
            },
        );

        results[1..].to_vec()
    }
}
