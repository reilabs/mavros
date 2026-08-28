use mavros_artifacts::FieldConfig;

use crate::compiler::ssa::{
    ValueId,
    builder::{BlockEmitter, FunctionBuilder, InstrBuilder, SSABuilder},
    hlssa::{
        BinaryArithOpKind, CallTarget, CastTarget, CmpKind, Constant, Endianness, LocatedOpCode,
        LookupTarget, OpCode, Radix, RefCountOp, SequenceTargetType, SliceOpDir, Type, TypeExpr,
    },
};

// ---------------------------------------------------------------------------
// HLEmitter — unified trait for emitting HL SSA instructions
// ---------------------------------------------------------------------------

pub trait HLEmitter {
    fn fresh_value(&mut self) -> ValueId;
    fn emit(&mut self, instruction: OpCode);
    fn emit_located(&mut self, instruction: LocatedOpCode);

    /// Intern a constant value into the SSA's constants side-table, returning the `ValueId` that
    /// names it. Identical `Constant`s collapse to the same `ValueId`.
    fn emit_constant(&mut self, value: Constant) -> ValueId;

    /// The field the program operates over, for minting/inspecting field values
    /// (e.g. `b.field().two_pow(k)`, `b.field().constant(n)`).
    fn field(&self) -> FieldConfig;

    // -- Arithmetic --

    /// Emit any binary arithmetic operation.
    ///
    /// The named helpers below are thin wrappers on this; use it directly when the operation is
    /// itself a variable, e.g. when a pass is rebuilding an instruction it matched on.
    fn bin(&mut self, kind: BinaryArithOpKind, lhs: ValueId, rhs: ValueId) -> ValueId {
        let r = self.fresh_value();
        self.emit(OpCode::BinaryArithOp {
            kind,
            result: r,
            lhs,
            rhs,
        });
        r
    }

    fn uadd(&mut self, lhs: ValueId, rhs: ValueId) -> ValueId {
        self.bin(BinaryArithOpKind::UAdd, lhs, rhs)
    }

    fn sadd(&mut self, lhs: ValueId, rhs: ValueId) -> ValueId {
        self.bin(BinaryArithOpKind::SAdd, lhs, rhs)
    }

    fn usub(&mut self, lhs: ValueId, rhs: ValueId) -> ValueId {
        self.bin(BinaryArithOpKind::USub, lhs, rhs)
    }

    fn ssub(&mut self, lhs: ValueId, rhs: ValueId) -> ValueId {
        self.bin(BinaryArithOpKind::SSub, lhs, rhs)
    }

    fn umul(&mut self, lhs: ValueId, rhs: ValueId) -> ValueId {
        self.bin(BinaryArithOpKind::UMul, lhs, rhs)
    }

    fn smul(&mut self, lhs: ValueId, rhs: ValueId) -> ValueId {
        self.bin(BinaryArithOpKind::SMul, lhs, rhs)
    }

    fn udiv(&mut self, lhs: ValueId, rhs: ValueId) -> ValueId {
        self.bin(BinaryArithOpKind::UDiv, lhs, rhs)
    }

    fn sdiv(&mut self, lhs: ValueId, rhs: ValueId) -> ValueId {
        self.bin(BinaryArithOpKind::SDiv, lhs, rhs)
    }

    fn urem(&mut self, lhs: ValueId, rhs: ValueId) -> ValueId {
        self.bin(BinaryArithOpKind::URem, lhs, rhs)
    }

    fn srem(&mut self, lhs: ValueId, rhs: ValueId) -> ValueId {
        self.bin(BinaryArithOpKind::SRem, lhs, rhs)
    }

    fn and(&mut self, lhs: ValueId, rhs: ValueId) -> ValueId {
        self.bin(BinaryArithOpKind::And, lhs, rhs)
    }

    fn or(&mut self, lhs: ValueId, rhs: ValueId) -> ValueId {
        self.bin(BinaryArithOpKind::Or, lhs, rhs)
    }

    fn xor(&mut self, lhs: ValueId, rhs: ValueId) -> ValueId {
        self.bin(BinaryArithOpKind::Xor, lhs, rhs)
    }

    fn ushl(&mut self, lhs: ValueId, rhs: ValueId) -> ValueId {
        self.bin(BinaryArithOpKind::UShl, lhs, rhs)
    }

    fn sshl(&mut self, lhs: ValueId, rhs: ValueId) -> ValueId {
        self.bin(BinaryArithOpKind::SShl, lhs, rhs)
    }

    fn ushr(&mut self, lhs: ValueId, rhs: ValueId) -> ValueId {
        self.bin(BinaryArithOpKind::UShr, lhs, rhs)
    }

    fn sshr(&mut self, lhs: ValueId, rhs: ValueId) -> ValueId {
        self.bin(BinaryArithOpKind::SShr, lhs, rhs)
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
        self.cmp(lhs, rhs, CmpKind::Eq)
    }

    fn ult(&mut self, lhs: ValueId, rhs: ValueId) -> ValueId {
        self.cmp(lhs, rhs, CmpKind::ULt)
    }

    fn slt(&mut self, lhs: ValueId, rhs: ValueId) -> ValueId {
        self.cmp(lhs, rhs, CmpKind::SLt)
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

    fn ensure_field(&mut self, value: ValueId, ty: &Type) -> ValueId {
        if ty.strip_witness().is_field() {
            value
        } else {
            self.cast_to_field(value)
        }
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

    fn widen_u(&mut self, value: ValueId, from_bits: usize, to_bits: usize) -> ValueId {
        assert!(
            from_bits <= to_bits,
            "widen_u cannot narrow ({from_bits} -> {to_bits} bits)"
        );
        if from_bits == to_bits {
            value
        } else {
            self.cast_to(CastTarget::Int(to_bits), value)
        }
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

    fn sext(&mut self, value: ValueId, from_bits: usize, to_bits: usize) -> ValueId {
        let r = self.fresh_value();
        self.emit(OpCode::SExt {
            result: r,
            value,
            from_bits,
            to_bits,
        });
        r
    }

    fn bit_range(&mut self, value: ValueId, offset: usize, width: usize) -> ValueId {
        let r = self.fresh_value();
        self.emit(OpCode::BitRange {
            result: r,
            value,
            offset,
            width,
        });
        r
    }

    // -- Constants --

    // FIELD-ASSUMPTION: L2-builder
    // Accepts anything convertible to a field element (including the raw `ark_bn254::Fr` that the
    // Noir frontend and some call sites still hold), so migrating the payload type is a no-op here.
    fn field_const(&mut self, value: impl Into<crate::compiler::Field>) -> ValueId {
        self.emit_constant(Constant::Field(value.into()))
    }

    /// A constant integer of `bits` raw two's-complement bits.
    ///
    /// There is no signed/unsigned pair here because there is no sign to record: what the bits mean
    /// is decided by the opcode that consumes them.
    fn int_const(&mut self, bits: usize, value: u128) -> ValueId {
        self.emit_constant(Constant::Int(bits, value))
    }

    // -- Witness --

    fn value_of(&mut self, value: ValueId) -> ValueId {
        let r = self.fresh_value();
        self.emit(OpCode::Cast {
            result: r,
            value,
            target: CastTarget::ValueOf,
        });
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

    fn tuple_proj(&mut self, tuple: ValueId, idx: usize) -> ValueId {
        let r = self.fresh_value();
        self.emit(OpCode::TupleProj {
            result: r,
            tuple,
            idx,
        });
        r
    }

    fn tuple_ref_proj(&mut self, tuple_ref: ValueId, idx: usize) -> ValueId {
        let r = self.fresh_value();
        self.emit(OpCode::TupleRefProj {
            result: r,
            tuple_ref,
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

    fn mk_seq(
        &mut self,
        elems: Vec<ValueId>,
        seq_type: SequenceTargetType,
        elem_type: Type,
    ) -> ValueId {
        let r = self.fresh_value();
        self.emit(OpCode::MkSeq {
            result: r,
            elems,
            seq_type,
            elem_type,
        });
        r
    }

    fn mk_seq_of_blob(&mut self, element_type: Type, blob: ValueId) -> ValueId {
        let r = self.fresh_value();
        self.emit(OpCode::MkSeqOfBlob {
            result: r,
            element_type,
            blob,
        });
        r
    }

    fn mk_repeated(
        &mut self,
        element: ValueId,
        seq_type: SequenceTargetType,
        count: usize,
        elem_type: Type,
    ) -> ValueId {
        let r = self.fresh_value();
        self.emit(OpCode::MkRepeated {
            result: r,
            element,
            seq_type,
            count,
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

    fn alloc(&mut self, value: ValueId) -> ValueId {
        let r = self.fresh_value();
        self.emit(OpCode::Alloc { result: r, value });
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

    fn slice_pop(&mut self, slice: ValueId, dir: SliceOpDir) -> (ValueId, ValueId) {
        let result_slice = self.fresh_value();
        let result_elem = self.fresh_value();
        self.emit(OpCode::SlicePop {
            dir,
            result_slice,
            result_elem,
            slice,
        });
        (result_slice, result_elem)
    }

    fn slice_insert(&mut self, slice: ValueId, index: ValueId, value: ValueId) -> ValueId {
        let result = self.fresh_value();
        self.emit(OpCode::SliceInsert {
            result,
            slice,
            index,
            value,
        });
        result
    }

    fn slice_remove(&mut self, slice: ValueId, index: ValueId) -> (ValueId, ValueId) {
        let result_slice = self.fresh_value();
        let result_elem = self.fresh_value();
        self.emit(OpCode::SliceRemove {
            result_slice,
            result_elem,
            slice,
            index,
        });
        (result_slice, result_elem)
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

    fn assert_bool(&mut self, value: ValueId) {
        self.emit(OpCode::Assert { value });
    }

    fn assert_constant(&mut self, value: ValueId) {
        self.emit(OpCode::AssertConstant { value });
    }

    fn assert_cmp(&mut self, kind: CmpKind, lhs: ValueId, rhs: ValueId) {
        self.emit(OpCode::AssertCmp { kind, lhs, rhs });
    }

    fn assert_eq(&mut self, lhs: ValueId, rhs: ValueId) {
        self.emit(OpCode::AssertCmp {
            kind: CmpKind::Eq,
            lhs,
            rhs,
        });
    }

    fn rangecheck(&mut self, value: ValueId, max_bits: usize) {
        self.emit(OpCode::Rangecheck { value, max_bits });
    }

    fn mem_op(&mut self, value: ValueId, kind: RefCountOp) {
        self.emit(OpCode::MemOp { kind, value });
    }

    fn spread(&mut self, value: ValueId, bits: u8) -> ValueId {
        let r = self.fresh_value();
        self.emit(OpCode::Spread {
            result: r,
            value,
            bits,
        });
        r
    }

    fn unspread(&mut self, value: ValueId, bits: u8) -> (ValueId, ValueId) {
        let r_and = self.fresh_value();
        let r_xor = self.fresh_value();
        self.emit(OpCode::Unspread {
            result_odd: r_and,
            result_even: r_xor,
            value,
            bits,
        });
        (r_and, r_xor)
    }

    fn lookup_spread(&mut self, bits: u8, key: ValueId, result: ValueId, flag: ValueId) {
        self.emit(OpCode::Lookup {
            target: LookupTarget::Spread(bits),
            args: vec![key, result],
            flag,
        });
    }

    /// Constrain `(amount, factor)` to be a row of the `2^s`-row powers-of-two table, i.e. to
    /// satisfy `factor == 2^amount` with `amount < 2^s`.
    ///
    /// `s` is `log2` of the shifted operand's width, so membership doubles as the shift-amount
    /// bound and the caller owes no separate check. `flag` gates that exactly as it does for every
    /// other lookup: a zero flag makes the row vacuous, which is what lets a guarded shift carry an
    /// out-of-range amount on an inactive path.
    fn lookup_pow2(&mut self, s: u8, amount: ValueId, factor: ValueId, flag: ValueId) {
        self.emit(OpCode::Lookup {
            target: LookupTarget::Pow2(s),
            args: vec![amount, factor],
            flag,
        });
    }

    fn lookup_rngchk(&mut self, target: LookupTarget<ValueId>, value: ValueId, flag: ValueId) {
        self.emit(OpCode::Lookup {
            target,
            args: vec![value],
            flag,
        });
    }

    fn lookup_rngchk_8(&mut self, value: ValueId, flag: ValueId) {
        self.emit(OpCode::Lookup {
            target: LookupTarget::Rangecheck(8),
            args: vec![value],
            flag,
        });
    }

    fn lookup_arr(&mut self, array: ValueId, index: ValueId, result: ValueId, flag: ValueId) {
        self.emit(OpCode::Lookup {
            target: LookupTarget::Array(array),
            args: vec![index, result],
            flag,
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

    fn call(
        &mut self,
        fn_id: crate::compiler::ssa::FunctionId,
        args: Vec<ValueId>,
        n: usize,
    ) -> Vec<ValueId> {
        let mut results = Vec::with_capacity(n);
        for _ in 0..n {
            results.push(self.fresh_value());
        }
        self.emit(OpCode::Call {
            results: results.clone(),
            function: CallTarget::Static(fn_id),
            args,
            unconstrained: false,
        });
        results
    }

    fn call_unconstrained(
        &mut self,
        fn_id: crate::compiler::ssa::FunctionId,
        args: Vec<ValueId>,
        n: usize,
    ) -> Vec<ValueId> {
        let mut results = Vec::with_capacity(n);
        for _ in 0..n {
            results.push(self.fresh_value());
        }
        self.emit(OpCode::Call {
            results: results.clone(),
            function: CallTarget::Static(fn_id),
            args,
            unconstrained: true,
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
            unconstrained: false,
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
// Type aliases
// ---------------------------------------------------------------------------

pub type HLInstrBuilder<'a> = InstrBuilder<'a, OpCode, Type, Constant>;
pub type HLFunctionBuilder<'a> = FunctionBuilder<'a, OpCode, Type, Constant>;
pub type HLBlockEmitter<'a> = BlockEmitter<'a, OpCode, Type, Constant>;
pub type HLSSABuilder<'a> = SSABuilder<'a, OpCode, Type, Constant>;

// ---------------------------------------------------------------------------
// HLEmitter impls
// ---------------------------------------------------------------------------

impl HLEmitter for HLInstrBuilder<'_> {
    fn fresh_value(&mut self) -> ValueId {
        self.ssa.fresh_value()
    }

    fn emit(&mut self, instruction: OpCode) {
        self.push(instruction);
    }

    fn emit_located(&mut self, instruction: LocatedOpCode) {
        self.push_located(instruction);
    }

    fn emit_constant(&mut self, value: Constant) -> ValueId {
        self.ssa.add_const(value)
    }

    fn field(&self) -> FieldConfig {
        self.ssa.field()
    }
}

impl HLEmitter for HLBlockEmitter<'_> {
    fn fresh_value(&mut self) -> ValueId {
        self.ssa.fresh_value()
    }

    fn emit(&mut self, instruction: OpCode) {
        self.emit_instruction(instruction);
    }

    fn emit_located(&mut self, instruction: LocatedOpCode) {
        self.emit_located_instruction(instruction);
    }

    fn emit_constant(&mut self, value: Constant) -> ValueId {
        self.ssa.add_const(value)
    }

    fn field(&self) -> FieldConfig {
        self.ssa.field()
    }
}

impl HLBlockEmitter<'_> {
    pub(crate) fn unwrap_guard(instruction: &OpCode) -> (Option<ValueId>, &OpCode) {
        match instruction {
            OpCode::Guard { condition, inner } => (Some(*condition), inner.as_ref()),
            other => (None, other),
        }
    }

    pub(crate) fn emit_guarded(&mut self, guard: Option<ValueId>, op: OpCode) {
        match guard {
            Some(condition) => self.emit(OpCode::Guard {
                condition,
                inner: Box::new(op),
            }),
            None => self.emit(op),
        }
    }

    pub(crate) fn default_value(&mut self, typ: &Type) -> ValueId {
        match &typ.expr {
            TypeExpr::Field => self.field_const(0u64),
            TypeExpr::Int(size) => self.int_const(*size, 0),
            TypeExpr::WitnessOf(inner) => {
                let inner_default = self.default_value(inner);
                self.cast_to_witness_of(inner_default)
            }
            TypeExpr::Array(inner, size) => self.default_array(inner, *size),
            TypeExpr::Tuple(element_types) => {
                let elems = element_types
                    .iter()
                    .map(|elem_type| self.default_value(elem_type))
                    .collect();
                self.mk_tuple(elems, element_types.clone())
            }
            TypeExpr::Slice(_) | TypeExpr::Ref(_) | TypeExpr::Function | TypeExpr::Blob(..) => {
                panic!("cannot build a default value for type {}", typ)
            }
        }
    }

    fn default_array(&mut self, elem_type: &Type, len: usize) -> ValueId {
        if len == 0 {
            return self.mk_seq(Vec::new(), SequenceTargetType::Array(0), elem_type.clone());
        }
        let elem = self.default_value(elem_type);
        self.mk_repeated(elem, SequenceTargetType::Array(len), len, elem_type.clone())
    }

    /// Build an array with an SSA counted loop.
    ///
    /// `body` receives the current `u32` index and must return the value to store at that index.
    pub fn build_array_loop(
        &mut self,
        len: usize,
        elem_type: Type,
        body: impl FnOnce(&mut Self, ValueId) -> ValueId,
    ) -> ValueId {
        let initial = self.default_array(&elem_type, len);
        if len == 0 {
            return initial;
        }
        let array_type = elem_type.clone().array_of(len);
        let results =
            self.build_counted_loop(len, vec![(initial, array_type)], |emitter, index, accs| {
                let value = body(emitter, index);
                let updated = emitter.array_set(accs[0], index, value);
                vec![updated]
            });
        results[0]
    }

    /// [`Self::build_array_loop`], threading one extra accumulator through the same loop.
    ///
    /// `body` receives the current `u32` index and the accumulator, and returns the value to store
    /// at that index together with the updated accumulator. Use this when a per-slot quantity has
    /// to be folded across the whole scan: this is a real SSA loop, not an unrolled one, so a value
    /// computed inside `body` is not otherwise reachable from outside it.
    ///
    /// When `len == 0` there is no loop to run, so `acc_init` comes straight back out — a caller
    /// folding over the slots gets the identity it started with, which is the right answer for an
    /// empty scan.
    pub fn build_array_loop_with_acc(
        &mut self,
        len: usize,
        elem_type: Type,
        acc: (ValueId, Type),
        body: impl FnOnce(&mut Self, ValueId, ValueId) -> (ValueId, ValueId),
    ) -> (ValueId, ValueId) {
        let (acc_init, acc_type) = acc;
        let initial = self.default_array(&elem_type, len);
        if len == 0 {
            return (initial, acc_init);
        }
        let array_type = elem_type.clone().array_of(len);
        let results = self.build_counted_loop(
            len,
            vec![(initial, array_type), (acc_init, acc_type)],
            |emitter, index, accs| {
                let (value, next_acc) = body(emitter, index, accs[1]);
                let updated = emitter.array_set(accs[0], index, value);
                vec![updated, next_acc]
            },
        );
        (results[0], results[1])
    }

    /// Build a counted loop: `for i in 0..len { body(i, accumulators) -> updated_accumulators }`
    ///
    /// Wrapper around `build_loop` that handles the u32 index, condition (`i < len`), and increment
    /// (`i + 1`). Returns only the accumulator values at loop exit.
    pub fn build_counted_loop(
        &mut self,
        len: usize,
        accumulators: Vec<(ValueId, Type)>,
        body: impl FnOnce(&mut Self, ValueId, &[ValueId]) -> Vec<ValueId>,
    ) -> Vec<ValueId> {
        // Emit constants into current block (before the loop)
        let const_0 = self.int_const(32, 0);
        let const_1 = self.int_const(32, 1);
        let const_len = self.int_const(32, len as u128);

        // Loop params: [index, ...accumulators]
        let mut params = vec![(const_0, Type::int(32))];
        params.extend(accumulators);

        let results = self.build_loop(
            params,
            |b, loop_params| b.ult(loop_params[0], const_len),
            |emitter, loop_params| {
                let i_val = loop_params[0];
                let acc_params = &loop_params[1..];
                let updated_accs = body(emitter, i_val, acc_params);
                let next_i = emitter.uadd(i_val, const_1);
                let mut result = vec![next_i];
                result.extend(updated_accs);
                result
            },
        );

        // Skip index, return accumulator results
        results[1..].to_vec()
    }

    /// Extend a slice with a counted SSA loop, one `push_back` per iteration:
    /// `for i in acc.len()..end { acc = acc.push_back(body(i)) }`.
    pub fn build_slice_extend_loop(
        &mut self,
        end: ValueId,
        acc: (ValueId, Type),
        body: impl FnOnce(&mut Self, ValueId) -> ValueId,
    ) -> ValueId {
        let (acc_init, acc_type) = acc;
        let start = self.slice_len(acc_init);
        let const_1 = self.int_const(32, 1);
        let results = self.build_loop(
            vec![(start, Type::int(32)), (acc_init, acc_type)],
            |b, params| b.ult(params[0], end),
            |b, params| {
                let (i, acc) = (params[0], params[1]);
                let elem = body(b, i);
                let pushed = b.slice_push(acc, vec![elem], SliceOpDir::Back);
                let next_i = b.uadd(i, const_1);
                vec![next_i, pushed]
            },
        );
        results[1]
    }
}

// TESTS
// ================================================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::ssa::SourceLocation;
    use crate::compiler::ssa::hlssa::HLSSA;

    /// Run `widen_u(from_bits -> to_bits)` on a fresh value; returns the result id, the input
    /// id, and the emitted instructions.
    fn widen(from_bits: usize, to_bits: usize) -> (ValueId, ValueId, Vec<LocatedOpCode>) {
        let mut ssa = HLSSA::with_main("main".to_string());
        let fid = ssa.get_unique_entrypoint_id();
        let mut function = ssa.take_function(fid);
        let v = ssa.fresh_value();
        let mut instrs = Vec::new();
        let w = {
            let mut b = HLInstrBuilder::new(
                &mut function,
                &mut ssa,
                &mut instrs,
                SourceLocation::synthetic("test"),
            );
            b.widen_u(v, from_bits, to_bits)
        };
        (w, v, instrs)
    }

    /// The truncation-fix invariant: a narrow comparison operand is brought up with a single
    /// widening cast (never the wide operand brought down, which truncates and aliases).
    #[test]
    fn widen_u_emits_a_single_widening_cast() {
        let (w, v, instrs) = widen(8, 32);
        assert_ne!(w, v);
        assert_eq!(instrs.len(), 1);
        let (op, _) = instrs.into_iter().next().unwrap().take();
        match op {
            OpCode::Cast {
                result,
                value,
                target: CastTarget::Int(32),
            } => {
                assert_eq!(value, v);
                assert_eq!(result, w);
            }
            other => panic!("expected a widening cast to u32, got {other:?}"),
        }
    }

    #[test]
    fn widen_u_is_the_identity_at_equal_width() {
        let (w, v, instrs) = widen(32, 32);
        assert_eq!(w, v);
        assert!(instrs.is_empty());
    }

    #[test]
    #[should_panic(expected = "widen_u cannot narrow")]
    fn widen_u_refuses_to_narrow() {
        let _ = widen(32, 8);
    }

    /// Run `build_array_loop_with_acc` over `len` slots, storing the index at each one and
    /// counting the slots visited. Returns `(array, accumulator, saw_indices)`.
    fn array_loop_with_acc(len: usize) -> (ValueId, ValueId, Vec<ValueId>) {
        let mut ssa = HLSSA::with_main("main".to_string());
        let fid = ssa.get_unique_entrypoint_id();
        let mut function = ssa.take_function(fid);
        let entry = function.get_entry_id();
        let mut builder = HLFunctionBuilder::new(&mut function, &mut ssa);
        let mut b = builder.test_block(entry);

        let zero = b.int_const(32, 0);
        let mut seen = Vec::new();
        let (array, acc) =
            b.build_array_loop_with_acc(len, Type::int(32), (zero, Type::int(32)), |b, i, acc| {
                seen.push(i);
                let one = b.int_const(32, 1);
                (i, b.uadd(acc, one))
            });
        (array, acc, seen)
    }

    #[test]
    fn build_array_loop_with_acc_threads_the_accumulator_through_the_loop() {
        let (array, acc, seen) = array_loop_with_acc(4);
        // One SSA loop, so the body is built once regardless of the trip count -- the count lives
        // in the loop, not in the number of emitted bodies.
        assert_eq!(seen.len(), 1);
        // The accumulator comes back as the loop's own result, distinct from both the array and
        // the initial value that was fed in.
        assert_ne!(acc, array);
        assert_ne!(acc, seen[0]);
    }

    #[test]
    fn build_array_loop_with_acc_returns_the_initial_accumulator_at_zero_length() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let fid = ssa.get_unique_entrypoint_id();
        let mut function = ssa.take_function(fid);
        let entry = function.get_entry_id();
        let mut builder = HLFunctionBuilder::new(&mut function, &mut ssa);
        let mut b = builder.test_block(entry);

        let init = b.int_const(32, 7);
        let mut ran = false;
        let (_, acc) =
            b.build_array_loop_with_acc(0, Type::int(32), (init, Type::int(32)), |_, i, acc| {
                ran = true;
                (i, acc)
            });
        // No loop is built, so the body never runs and the caller gets its identity back --
        // which is what makes an empty scan's fold well defined.
        assert!(!ran, "the body must not run for a zero-length array");
        assert_eq!(acc, init);
    }
}
