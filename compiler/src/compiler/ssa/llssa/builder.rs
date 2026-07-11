use crate::compiler::ssa::{
    FunctionId, ValueId,
    builder::{BlockEditor, BlockEmitter, FunctionBuilder, InstrBuilder, SSABuilder},
    hlssa::DMatrix,
    llssa::{Constant, FieldArithOp, IntArithOp, IntCmpOp, LLOp, LLStruct, LocatedLLOp, Type},
};

// ---------------------------------------------------------------------------
// LLEmitter — unified trait for emitting LL SSA instructions
// ---------------------------------------------------------------------------

pub trait LLEmitter {
    fn fresh_value(&mut self) -> ValueId;
    fn emit_ll(&mut self, instruction: LLOp);
    fn emit_located_ll(&mut self, instruction: LocatedLLOp);
    fn vm_ptr(&mut self) -> ValueId;
    fn emit_constant(&mut self, value: Constant) -> ValueId;

    // -- Constant --

    fn emit_int_const(&mut self, bits: u32, value: u64) -> ValueId {
        self.emit_int_const_u128(bits, value as u128)
    }

    fn emit_int_const_u128(&mut self, bits: u32, value: u128) -> ValueId {
        self.emit_constant(Constant::Int { bits, value })
    }

    fn emit_nullptr_const(&mut self) -> ValueId {
        self.emit_constant(Constant::NullPtr)
    }

    fn emit_struct_const(&mut self, layout: LLStruct, values: Vec<Constant>) -> ValueId {
        // Sanity check for builds with assertions enabled.
        debug_assert!(
            layout.accepts(&values),
            "emit_struct_const: values {values:?} are incompatible with layout {layout:?}",
        );
        self.emit_constant(Constant::Struct { layout, values })
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

    fn spread(&mut self, value: ValueId, bits: u8, result_bits: u32) -> ValueId {
        let r = self.fresh_value();
        self.emit_ll(LLOp::Spread {
            result: r,
            value,
            bits,
            result_bits,
        });
        r
    }

    fn unspread(
        &mut self,
        value: ValueId,
        bits: u8,
        odd_bits: u32,
        even_bits: u32,
    ) -> (ValueId, ValueId) {
        let result_odd = self.fresh_value();
        let result_even = self.fresh_value();
        self.emit_ll(LLOp::Unspread {
            result_odd,
            result_even,
            value,
            bits,
            odd_bits,
            even_bits,
        });
        (result_odd, result_even)
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

    fn field_lt(&mut self, a: ValueId, b: ValueId) -> ValueId {
        let r = self.fresh_value();
        self.emit_ll(LLOp::FieldLt { result: r, a, b });
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

    fn ll_load(&mut self, ptr: ValueId, ty: Type) -> ValueId {
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

    fn const_data_ptr(&mut self, elem_type: LLStruct, blob: ValueId) -> ValueId {
        let r = self.fresh_value();
        self.emit_ll(LLOp::ConstDataPtr {
            result: r,
            elem_type,
            blob,
        });
        r
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
        self.write_field_cursor(LLStruct::WITGEN_VM_A, a);
        self.write_field_cursor(LLStruct::WITGEN_VM_B, b);
        self.write_field_cursor(LLStruct::WITGEN_VM_C, c);
    }

    fn write_witness(&mut self, value: ValueId) {
        self.write_field_cursor(LLStruct::WITGEN_VM_WITNESS, value);
    }

    // -- Trap --

    fn trap(&mut self) {
        self.emit_ll(LLOp::Trap);
    }

    // -- AD (Automatic Differentiation) --

    fn next_d_coeff(&mut self) -> ValueId {
        let coeffs_slot = self.ad_vm_field_ptr(LLStruct::AD_VM_COEFFS);
        let coeffs = self.ll_load(coeffs_slot, Type::Ptr);
        let value = self.ll_load(coeffs, Type::Struct(LLStruct::field_elem()));
        let one = self.emit_int_const(32, 1);
        let next_coeffs = self.array_elem_ptr(coeffs, LLStruct::field_elem(), one);
        self.ll_store(coeffs_slot, next_coeffs);
        value
    }

    fn ad_write_const(&mut self, matrix: DMatrix, const_value: ValueId, sensitivity: ValueId) {
        let out_d_slot = self.ad_vm_field_ptr(ad_out_field(matrix));
        let out_d = self.ll_load(out_d_slot, Type::Ptr);
        let product = self.field_arith(FieldArithOp::Mul, const_value, sensitivity);
        let old = self.ll_load(out_d, Type::Struct(LLStruct::field_elem()));
        let new_value = self.field_arith(FieldArithOp::Add, old, product);
        self.ll_store(out_d, new_value);
    }

    fn ad_write_witness(&mut self, matrix: DMatrix, witness_index: ValueId, sensitivity: ValueId) {
        let out_d_slot = self.ad_vm_field_ptr(ad_out_field(matrix));
        let out_d = self.ll_load(out_d_slot, Type::Ptr);
        let target = self.array_elem_ptr(out_d, LLStruct::field_elem(), witness_index);
        let old = self.ll_load(target, Type::Struct(LLStruct::field_elem()));
        let new_value = self.field_arith(FieldArithOp::Add, old, sensitivity);
        self.ll_store(target, new_value);
    }

    fn ad_fresh_witness(&mut self) -> ValueId {
        let slot = self.ad_vm_field_ptr(LLStruct::AD_VM_CURRENT_WIT_OFF);
        let index = self.ll_load(slot, Type::i32());
        let one = self.emit_int_const(32, 1);
        let next = self.int_add(index, one);
        self.ll_store(slot, next);
        index
    }

    // -- Transparent VM struct access --

    fn witgen_vm_field_ptr(&mut self, field: usize) -> ValueId {
        let vm = self.vm_ptr();
        self.struct_field_ptr(vm, LLStruct::witgen_vm(), field)
    }

    fn ad_vm_field_ptr(&mut self, field: usize) -> ValueId {
        let vm = self.vm_ptr();
        self.struct_field_ptr(vm, LLStruct::ad_vm(), field)
    }

    fn write_field_cursor(&mut self, cursor_field: usize, value: ValueId) {
        let cursor_slot = self.witgen_vm_field_ptr(cursor_field);
        let cursor = self.ll_load(cursor_slot, Type::Ptr);
        self.ll_store(cursor, value);
        let one = self.emit_int_const(32, 1);
        let next = self.array_elem_ptr(cursor, LLStruct::field_elem(), one);
        self.ll_store(cursor_slot, next);
    }
}

fn ad_out_field(matrix: DMatrix) -> usize {
    match matrix {
        DMatrix::A => LLStruct::AD_VM_OUT_DA,
        DMatrix::B => LLStruct::AD_VM_OUT_DB,
        DMatrix::C => LLStruct::AD_VM_OUT_DC,
    }
}

// ---------------------------------------------------------------------------
// Type aliases
// ---------------------------------------------------------------------------

pub type LLInstrBuilder<'a> = InstrBuilder<'a, LLOp, Type, Constant>;
pub type LLFunctionBuilder<'a> = FunctionBuilder<'a, LLOp, Type, Constant>;
pub type LLBlockEditor<'a> = BlockEditor<'a, LLOp, Type, Constant>;
pub type LLBlockEmitter<'a> = BlockEmitter<'a, LLOp, Type, Constant>;
pub type LLSSABuilder<'a> = SSABuilder<'a, LLOp, Type, Constant>;

// ---------------------------------------------------------------------------
// LLEmitter impls
// ---------------------------------------------------------------------------

impl LLEmitter for LLInstrBuilder<'_> {
    fn fresh_value(&mut self) -> ValueId {
        self.ssa.fresh_value()
    }

    fn emit_ll(&mut self, instruction: LLOp) {
        self.push(instruction);
    }

    fn emit_located_ll(&mut self, instruction: LocatedLLOp) {
        self.push_located(instruction);
    }

    fn vm_ptr(&mut self) -> ValueId {
        self.function
            .get_entry()
            .get_parameters()
            .next()
            .expect("LLSSA functions must have a VM pointer entry parameter")
            .0
    }

    fn emit_constant(&mut self, value: Constant) -> ValueId {
        self.ssa.add_const(value)
    }
}

impl LLEmitter for LLBlockEmitter<'_> {
    fn fresh_value(&mut self) -> ValueId {
        self.ssa.fresh_value()
    }

    fn emit_ll(&mut self, instruction: LLOp) {
        self.emit_instruction(instruction);
    }

    fn emit_located_ll(&mut self, instruction: LocatedLLOp) {
        self.emit_located_instruction(instruction);
    }

    fn vm_ptr(&mut self) -> ValueId {
        if self.block_id == self.function.get_entry_id() {
            self.block
                .get_parameters()
                .next()
                .expect("LLSSA functions must have a VM pointer entry parameter")
                .0
        } else {
            self.function
                .get_entry()
                .get_parameters()
                .next()
                .expect("LLSSA functions must have a VM pointer entry parameter")
                .0
        }
    }

    fn emit_constant(&mut self, value: Constant) -> ValueId {
        self.ssa.add_const(value)
    }
}

impl LLBlockEmitter<'_> {
    /// Build a counted loop with an i64 upper bound:
    /// `for i in 0..count { body(i, accumulators) -> updated_accumulators }`.
    pub fn build_counted_loop(
        &mut self,
        count: ValueId,
        accumulators: Vec<(ValueId, Type)>,
        body: impl FnOnce(&mut Self, ValueId, &[ValueId]) -> Vec<ValueId>,
    ) -> Vec<ValueId> {
        let const_0 = self.emit_int_const(64, 0);
        let const_1 = self.emit_int_const(64, 1);

        let mut params = vec![(const_0, Type::i64())];
        params.extend(accumulators);

        let results = self.build_loop(
            params,
            |b, loop_params| b.int_ult(loop_params[0], count),
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
