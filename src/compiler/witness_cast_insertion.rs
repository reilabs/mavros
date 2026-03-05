use std::collections::HashMap;

use tracing::{Level, instrument};

use crate::compiler::{
    analysis::types::Types,
    block_builder::{BlockEmitter, HLEmitter},
    flow_analysis::FlowAnalysis,
    ir::r#type::{Type, TypeExpr},
    ssa::{BlockId, HLBlock, HLFunction, HLSSA, OpCode, SeqType, Terminator, TupleIdx, ValueId},
    witness_info::{FunctionWitnessType, WitnessType},
    witness_type_inference::WitnessTypeInference,
};

pub struct WitnessCastInsertion {}

impl WitnessCastInsertion {
    pub fn new() -> Self {
        Self {}
    }

    #[instrument(skip_all, name = "WitnessCastInsertion::run")]
    pub fn run(&mut self, ssa: HLSSA, witness_inference: &WitnessTypeInference) -> HLSSA {
        // Sub-pass 1: Bake WitnessOf into SSA types (prepare_rebuild pattern)
        let ssa = self.apply_types(ssa, witness_inference);

        // Compute type info for cast insertion
        let flow_analysis = FlowAnalysis::run(&ssa);
        let type_info = Types::new().run(&ssa, &flow_analysis);

        // Sub-pass 2: Insert casts at typed-slot boundaries (take_blocks/put_blocks pattern)
        self.insert_casts(ssa, &type_info)
    }

    // -----------------------------------------------------------------------
    // Sub-pass 1: Type Application
    // -----------------------------------------------------------------------

    fn apply_types(&self, ssa: HLSSA, witness_inference: &WitnessTypeInference) -> HLSSA {
        let (mut result_ssa, functions, old_global_types) = ssa.prepare_rebuild();

        // Global types: identity (pure types stay as-is)
        result_ssa.set_global_types(old_global_types);

        for (function_id, function) in functions.into_iter() {
            let function_wt = witness_inference.get_function_witness_type(function_id);
            let new_function = self.apply_types_to_function(function, function_wt);
            result_ssa.put_function(function_id, new_function);
        }

        result_ssa
    }

    fn apply_types_to_function(
        &self,
        function: HLFunction,
        function_wt: &FunctionWitnessType,
    ) -> HLFunction {
        let (mut function, blocks, returns) = function.prepare_rebuild();

        for (block_id, mut block) in blocks.into_iter() {
            let mut new_block = HLBlock::empty();

            let mut new_parameters = Vec::new();
            for (value_id, typ) in block.take_parameters() {
                let wt = function_wt.get_value_witness_type(value_id);
                new_parameters.push((value_id, self.apply_witness_type(typ, wt)));
            }
            new_block.put_parameters(new_parameters);

            let mut new_instructions = Vec::<OpCode>::new();
            for instruction in block.take_instructions() {
                let new = match instruction {
                    OpCode::Alloc {
                        result: r,
                        elem_type: l,
                    } => {
                        let r_wt = function_wt.get_value_witness_type(r);
                        let child = r_wt.child_witness_type().unwrap();
                        let child_typ = self.apply_witness_type(l, &child);
                        OpCode::Alloc {
                            result: r,
                            elem_type: child_typ,
                        }
                    }
                    OpCode::FreshWitness { .. } => instruction,
                    OpCode::MkSeq {
                        result: r,
                        elems: l,
                        seq_type: stp,
                        elem_type: tp,
                    } => {
                        let r_wt = function_wt
                            .get_value_witness_type(r)
                            .child_witness_type()
                            .unwrap();
                        OpCode::MkSeq {
                            result: r,
                            elems: l,
                            seq_type: stp,
                            elem_type: self.apply_witness_type(tp, &r_wt),
                        }
                    }
                    OpCode::MkTuple {
                        result: r,
                        elems: l,
                        element_types: tps,
                    } => {
                        let r_wt = function_wt.get_value_witness_type(r);
                        let child_wts = if let WitnessType::Tuple(_, children) = r_wt {
                            children
                        } else {
                            panic!("MkTuple result should have Tuple witness type")
                        };
                        OpCode::MkTuple {
                            result: r,
                            elems: l,
                            element_types: tps
                                .iter()
                                .zip(child_wts.iter())
                                .map(|(tp, wt)| self.apply_witness_type(tp.clone(), wt))
                                .collect(),
                        }
                    }
                    OpCode::ReadGlobal {
                        result: r,
                        offset: l,
                        result_type: tp,
                    } => OpCode::ReadGlobal {
                        result: r,
                        offset: l,
                        result_type: tp, // globals are pure, keep as-is
                    },
                    OpCode::Todo {
                        payload,
                        results,
                        result_types,
                    } => OpCode::Todo {
                        payload,
                        results,
                        result_types, // keep as-is (pure)
                    },
                    // All other opcodes pass through unchanged
                    other => other,
                };
                new_instructions.push(new);
            }
            new_block.put_instructions(new_instructions);

            new_block.set_terminator(block.take_terminator().unwrap());

            function.put_block(block_id, new_block);
        }

        for (ret, ret_wt) in returns.into_iter().zip(function_wt.returns_witness.iter()) {
            let ret_typ = self.apply_witness_type(ret, ret_wt);
            function.add_return_type(ret_typ);
        }

        function
    }

    fn apply_witness_type(&self, typ: Type, wt: &WitnessType) -> Type {
        match (typ.expr, wt) {
            (TypeExpr::Field, WitnessType::Scalar(info)) => {
                let base = Type::field();
                if info.is_witness() {
                    Type::witness_of(base)
                } else {
                    base
                }
            }
            (TypeExpr::U(size), WitnessType::Scalar(info)) => {
                let base = Type::u(size);
                if info.is_witness() {
                    Type::witness_of(base)
                } else {
                    base
                }
            }
            (TypeExpr::Array(inner, size), WitnessType::Array(top, inner_wt)) => {
                let base = self
                    .apply_witness_type(*inner, inner_wt.as_ref())
                    .array_of(size);
                if top.is_witness() {
                    Type::witness_of(base)
                } else {
                    base
                }
            }
            (TypeExpr::Slice(inner), WitnessType::Array(top, inner_wt)) => {
                let base = self
                    .apply_witness_type(*inner, inner_wt.as_ref())
                    .slice_of();
                if top.is_witness() {
                    Type::witness_of(base)
                } else {
                    base
                }
            }
            (TypeExpr::Ref(inner), WitnessType::Ref(top, inner_wt)) => {
                let base = self.apply_witness_type(*inner, inner_wt.as_ref()).ref_of();
                if top.is_witness() {
                    Type::witness_of(base)
                } else {
                    base
                }
            }
            (TypeExpr::Tuple(child_types), WitnessType::Tuple(top, child_wts)) => {
                let base = Type::tuple_of(
                    child_types
                        .iter()
                        .zip(child_wts.iter())
                        .map(|(child_type, child_wt)| {
                            self.apply_witness_type(child_type.clone(), child_wt)
                        })
                        .collect(),
                );
                if top.is_witness() {
                    Type::witness_of(base)
                } else {
                    base
                }
            }
            (tp, wt) => panic!("Unexpected type {:?} with witness type {:?}", tp, wt),
        }
    }

    // -----------------------------------------------------------------------
    // Sub-pass 2: Cast Insertion
    // -----------------------------------------------------------------------

    fn insert_casts(
        &self,
        mut ssa: HLSSA,
        type_info: &crate::compiler::analysis::types::TypeInfo,
    ) -> HLSSA {
        let function_ids: Vec<_> = ssa.get_function_ids().collect();
        for function_id in function_ids {
            let func_type_info = type_info.get_function(function_id);
            let mut function = ssa.take_function(function_id);
            self.insert_casts_in_function(&mut function, func_type_info);
            ssa.put_function(function_id, function);
        }
        ssa
    }

    #[instrument(skip_all, name = "WitnessCastInsertion::insert_casts_in_function", level = Level::DEBUG, fields(function = function.get_name()))]
    fn insert_casts_in_function(
        &self,
        function: &mut HLFunction,
        type_info: &crate::compiler::analysis::types::FunctionTypeInfo,
    ) {
        // Collect block param types for Jmp boundary comparison
        let block_param_types: HashMap<BlockId, Vec<Type>> = function
            .get_blocks()
            .map(|(bid, block)| {
                let types = block.get_parameters().map(|(_, tp)| tp.clone()).collect();
                (*bid, types)
            })
            .collect();

        let return_types: Vec<Type> = function.get_returns().to_vec();

        let block_ids: Vec<BlockId> = function.get_blocks().map(|(bid, _)| *bid).collect();
        for bid in block_ids {
            let terminator = function.get_block_mut(bid).take_terminator();
            let instructions = function.get_block_mut(bid).take_instructions();

            let mut emitter = BlockEmitter::new(function, bid);

            for instruction in instructions.into_iter() {
                match instruction {
                    OpCode::MkSeq {
                        result: r,
                        elems: vs,
                        seq_type: s,
                        elem_type: ref tp,
                    } => {
                        let target_elem_type = tp.clone();
                        let new_vs = vs
                            .iter()
                            .map(|v| {
                                self.convert_if_needed(
                                    *v,
                                    &target_elem_type,
                                    type_info,
                                    &mut emitter,
                                )
                            })
                            .collect();
                        emitter.emit(OpCode::MkSeq {
                            result: r,
                            elems: new_vs,
                            seq_type: s,
                            elem_type: target_elem_type,
                        });
                    }
                    OpCode::ArraySet {
                        result,
                        array,
                        index,
                        value,
                    } => {
                        let array_type = type_info.get_value_type(array);
                        let expected_elem_type = match &array_type.expr {
                            TypeExpr::Array(inner, _) => inner.as_ref().clone(),
                            TypeExpr::Slice(inner) => inner.as_ref().clone(),
                            _ => panic!("ArraySet on non-array type"),
                        };
                        let converted = self.convert_if_needed(
                            value,
                            &expected_elem_type,
                            type_info,
                            &mut emitter,
                        );
                        emitter.emit(OpCode::ArraySet {
                            result,
                            array,
                            index,
                            value: converted,
                        });
                    }
                    OpCode::SlicePush {
                        dir,
                        result,
                        slice,
                        values,
                    } => {
                        let slice_type = type_info.get_value_type(slice);
                        let expected_elem_type = match &slice_type.expr {
                            TypeExpr::Slice(inner) => inner.as_ref().clone(),
                            _ => panic!("SlicePush on non-slice type"),
                        };
                        let new_values = values
                            .iter()
                            .map(|v| {
                                self.convert_if_needed(
                                    *v,
                                    &expected_elem_type,
                                    type_info,
                                    &mut emitter,
                                )
                            })
                            .collect();
                        emitter.emit(OpCode::SlicePush {
                            dir,
                            result,
                            slice,
                            values: new_values,
                        });
                    }
                    OpCode::Store { ptr, value } => {
                        let ptr_type = type_info.get_value_type(ptr);
                        let target_type = ptr_type.get_pointed();
                        let converted =
                            self.convert_if_needed(value, &target_type, type_info, &mut emitter);
                        emitter.emit(OpCode::Store {
                            ptr,
                            value: converted,
                        });
                    }
                    OpCode::Select {
                        result: r,
                        cond,
                        if_t,
                        if_f,
                    } => {
                        let if_t_type = type_info.get_value_type(if_t);
                        let if_f_type = type_info.get_value_type(if_f);
                        let target_type = if_t_type.get_arithmetic_result_type(if_f_type);
                        let if_t =
                            self.convert_if_needed(if_t, &target_type, type_info, &mut emitter);
                        let if_f =
                            self.convert_if_needed(if_f, &target_type, type_info, &mut emitter);
                        emitter.emit(OpCode::Select {
                            result: r,
                            cond,
                            if_t,
                            if_f,
                        });
                    }
                    op @ (OpCode::Cmp { .. }
                    | OpCode::BinaryArithOp { .. }
                    | OpCode::Cast { .. }
                    | OpCode::Truncate { .. }
                    | OpCode::Not { .. }
                    | OpCode::Alloc { .. }
                    | OpCode::Load { .. }
                    | OpCode::AssertEq { .. }
                    | OpCode::AssertR1C { .. }
                    | OpCode::Call { .. }
                    | OpCode::ArrayGet { .. }
                    | OpCode::SliceLen { .. }
                    | OpCode::ToBits { .. }
                    | OpCode::ToRadix { .. }
                    | OpCode::MemOp { .. }
                    | OpCode::ValueOf { .. }
                    | OpCode::WriteWitness { .. }
                    | OpCode::FreshWitness { .. }
                    | OpCode::NextDCoeff { .. }
                    | OpCode::BumpD { .. }
                    | OpCode::Constrain { .. }
                    | OpCode::Lookup { .. }
                    | OpCode::DLookup { .. }
                    | OpCode::MulConst { .. }
                    | OpCode::Rangecheck { .. }
                    | OpCode::ReadGlobal { .. }
                    | OpCode::TupleProj { .. }
                    | OpCode::MkTuple { .. }
                    | OpCode::Todo { .. }
                    | OpCode::InitGlobal { .. }
                    | OpCode::DropGlobal { .. }
                    | OpCode::Const { .. }) => {
                        emitter.emit(op);
                    }
                }
            }

            // Handle terminator
            if let Some(terminator) = terminator {
                match terminator {
                    Terminator::Jmp(target, args) => {
                        if let Some(param_types) = block_param_types.get(&target) {
                            let new_args: Vec<_> = args
                                .iter()
                                .zip(param_types.iter())
                                .map(|(arg, expected_type)| {
                                    self.convert_if_needed(
                                        *arg,
                                        expected_type,
                                        type_info,
                                        &mut emitter,
                                    )
                                })
                                .collect();
                            emitter.set_terminator(Terminator::Jmp(target, new_args));
                        } else {
                            emitter.set_terminator(Terminator::Jmp(target, args));
                        }
                    }
                    Terminator::Return(values) => {
                        let new_values: Vec<_> = values
                            .iter()
                            .zip(return_types.iter())
                            .map(|(val, expected_type)| {
                                self.convert_if_needed(*val, expected_type, type_info, &mut emitter)
                            })
                            .collect();
                        emitter.set_terminator(Terminator::Return(new_values));
                    }
                    Terminator::JmpIf(cond, if_true, if_false) => {
                        emitter.set_terminator(Terminator::JmpIf(cond, if_true, if_false));
                    }
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // Conversion helpers
    // -----------------------------------------------------------------------

    fn convert_if_needed(
        &self,
        value: ValueId,
        target_type: &Type,
        type_info: &crate::compiler::analysis::types::FunctionTypeInfo,
        emitter: &mut BlockEmitter<'_>,
    ) -> ValueId {
        let value_type = type_info.get_value_type(value);
        if *value_type == *target_type {
            return value;
        }
        self.emit_value_conversion(value, value_type, target_type, emitter)
    }

    fn emit_value_conversion(
        &self,
        value: ValueId,
        source_type: &Type,
        target_type: &Type,
        emitter: &mut BlockEmitter<'_>,
    ) -> ValueId {
        match (&source_type.expr, &target_type.expr) {
            // Scalar: Field → WitnessOf(Field), U(n) → WitnessOf(U(n))
            (TypeExpr::Field, TypeExpr::WitnessOf(_))
            | (TypeExpr::U(_), TypeExpr::WitnessOf(_)) => emitter.cast_to_witness_of(value),
            // Array element conversion
            (TypeExpr::Array(src_inner, src_size), TypeExpr::Array(tgt_inner, tgt_size)) => {
                assert_eq!(
                    src_size, tgt_size,
                    "Array size mismatch in witness cast insertion"
                );
                self.emit_array_conversion_loop(
                    value,
                    src_inner,
                    tgt_inner,
                    *src_size,
                    target_type,
                    emitter,
                )
            }
            // Tuple: decompose, per-field recursive, recompose
            (TypeExpr::Tuple(src_fields), TypeExpr::Tuple(tgt_fields)) => {
                assert_eq!(
                    src_fields.len(),
                    tgt_fields.len(),
                    "Tuple field count mismatch in witness cast insertion"
                );
                let mut converted_elems = vec![];
                for (i, (src_ft, tgt_ft)) in src_fields.iter().zip(tgt_fields.iter()).enumerate() {
                    let proj = emitter.tuple_proj(value, TupleIdx::Static(i));
                    let converted = self.emit_value_conversion(proj, src_ft, tgt_ft, emitter);
                    converted_elems.push(converted);
                }
                emitter.mk_tuple(converted_elems, tgt_fields.clone())
            }
            // WitnessOf→WitnessOf: identity (same runtime repr)
            (TypeExpr::WitnessOf(_), TypeExpr::WitnessOf(_)) => value,
            _ => panic!(
                "witness cast insertion: unsupported conversion {:?} -> {:?}",
                source_type, target_type
            ),
        }
    }

    fn emit_array_conversion_loop(
        &self,
        source_array: ValueId,
        src_elem_type: &Type,
        tgt_elem_type: &Type,
        array_len: usize,
        target_array_type: &Type,
        emitter: &mut BlockEmitter<'_>,
    ) -> ValueId {
        let initial_dst =
            self.create_dummy_array(tgt_elem_type, array_len, target_array_type, emitter);

        let results = emitter.build_counted_loop(
            array_len,
            vec![(initial_dst, target_array_type.clone())],
            |emitter, i_val, accs| {
                let dst_val = accs[0];
                let elem = emitter.array_get(source_array, i_val);
                let converted =
                    self.emit_value_conversion(elem, src_elem_type, tgt_elem_type, emitter);
                let new_dst = emitter.array_set(dst_val, i_val, converted);
                vec![new_dst]
            },
        );

        results[0]
    }

    /// Create a dummy array of the given target type, properly laid out in memory.
    fn create_dummy_array(
        &self,
        elem_type: &Type,
        array_len: usize,
        _array_type: &Type,
        b: &mut impl HLEmitter,
    ) -> ValueId {
        let dummy_elem = self.create_dummy_value(elem_type, b);
        let elems = vec![dummy_elem; array_len];
        b.mk_seq(elems, SeqType::Array(array_len), elem_type.clone())
    }

    /// Create a single dummy value of the given target type.
    fn create_dummy_value(&self, target_type: &Type, b: &mut impl HLEmitter) -> ValueId {
        match &target_type.expr {
            TypeExpr::WitnessOf(_) => {
                let dummy_field = b.field_const(ark_bn254::Fr::from(0u64));
                b.cast_to_witness_of(dummy_field)
            }
            TypeExpr::Array(inner, size) => self.create_dummy_array(inner, *size, target_type, b),
            TypeExpr::Tuple(fields) => {
                let mut dummy_elems = vec![];
                for field_type in fields.iter() {
                    dummy_elems.push(self.create_dummy_value(field_type, b));
                }
                b.mk_tuple(dummy_elems, fields.clone())
            }
            TypeExpr::Field | TypeExpr::U(_) => b.field_const(ark_bn254::Fr::from(0u64)),
            _ => panic!("create_dummy_value: unsupported type {:?}", target_type),
        }
    }
}
