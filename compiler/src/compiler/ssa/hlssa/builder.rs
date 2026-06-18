use crate::compiler::ssa::{
    ValueId,
    builder::{BlockEmitter, FunctionBuilder, InstrBuilder, SSABuilder},
    hlssa::{
        BinaryArithOpKind, CallTarget, CastTarget, CmpKind, Constant, Endianness, LookupTarget,
        OpCode, Radix, RefCountOp, SequenceTargetType, SliceOpDir, Type, TypeExpr,
    },
};

// ---------------------------------------------------------------------------
// HLEmitter — unified trait for emitting HL SSA instructions
// ---------------------------------------------------------------------------

pub trait HLEmitter {
    fn fresh_value(&mut self) -> ValueId;
    fn emit(&mut self, op: OpCode);

    /// Intern a constant value into the SSA's constants side-table, returning the `ValueId` that
    /// names it. Identical `Constant`s collapse to the same `ValueId`.
    fn emit_constant(&mut self, value: Constant) -> ValueId;

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

    fn modulo(&mut self, lhs: ValueId, rhs: ValueId) -> ValueId {
        let r = self.fresh_value();
        self.emit(OpCode::BinaryArithOp {
            kind: BinaryArithOpKind::Mod,
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

    fn or(&mut self, lhs: ValueId, rhs: ValueId) -> ValueId {
        let r = self.fresh_value();
        self.emit(OpCode::BinaryArithOp {
            kind: BinaryArithOpKind::Or,
            result: r,
            lhs,
            rhs,
        });
        r
    }

    fn xor(&mut self, lhs: ValueId, rhs: ValueId) -> ValueId {
        let r = self.fresh_value();
        self.emit(OpCode::BinaryArithOp {
            kind: BinaryArithOpKind::Xor,
            result: r,
            lhs,
            rhs,
        });
        r
    }

    fn shl(&mut self, lhs: ValueId, rhs: ValueId) -> ValueId {
        let r = self.fresh_value();
        self.emit(OpCode::BinaryArithOp {
            kind: BinaryArithOpKind::Shl,
            result: r,
            lhs,
            rhs,
        });
        r
    }

    fn shr(&mut self, lhs: ValueId, rhs: ValueId) -> ValueId {
        let r = self.fresh_value();
        self.emit(OpCode::BinaryArithOp {
            kind: BinaryArithOpKind::Shr,
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

    fn field_const(&mut self, value: ark_bn254::Fr) -> ValueId {
        self.emit_constant(Constant::Field(value))
    }

    fn u_const(&mut self, bits: usize, value: u128) -> ValueId {
        self.emit_constant(Constant::U(bits, value))
    }

    fn i_const(&mut self, bits: usize, value: u128) -> ValueId {
        self.emit_constant(Constant::I(bits, value))
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

    fn emit(&mut self, op: OpCode) {
        self.instructions.push(op);
    }

    fn emit_constant(&mut self, value: Constant) -> ValueId {
        self.ssa.add_const(value)
    }
}

impl HLEmitter for HLBlockEmitter<'_> {
    fn fresh_value(&mut self) -> ValueId {
        self.ssa.fresh_value()
    }

    fn emit(&mut self, op: OpCode) {
        self.block.push_instruction(op);
    }

    fn emit_constant(&mut self, value: Constant) -> ValueId {
        self.ssa.add_const(value)
    }
}

impl HLBlockEmitter<'_> {
    pub(crate) fn default_value(&mut self, typ: &Type) -> ValueId {
        match &typ.expr {
            TypeExpr::Field => self.field_const(ark_bn254::Fr::from(0)),
            TypeExpr::U(size) => self.u_const(*size, 0),
            TypeExpr::I(size) => self.i_const(*size, 0),
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
