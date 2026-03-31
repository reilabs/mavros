use std::collections::HashMap;

use crate::compiler::{
    analysis::types::TypeInfo,
    block_builder::{HLBlockEmitter, HLEmitter},
    ir::r#type::{Type, TypeExpr},
    pass_manager::{Analysis, AnalysisId, AnalysisStore, Pass},
    passes::fix_double_jumps::ValueReplacements,
    ssa::{BinaryArithOpKind, BlockId, DMatrix, OpCode, SeqType, Terminator, ValueId},
};

pub struct WitnessLowering {}

impl Pass for WitnessLowering {
    fn name(&self) -> &'static str {
        "witness_lowering"
    }

    fn needs(&self) -> Vec<AnalysisId> {
        vec![TypeInfo::id()]
    }

    fn run(&self, ssa: &mut crate::compiler::ssa::HLSSA, store: &AnalysisStore) {
        self.do_run(ssa, store.get::<TypeInfo>());
    }
}

impl WitnessLowering {
    pub fn new() -> Self {
        Self {}
    }

    pub fn do_run(
        &self,
        ssa: &mut crate::compiler::ssa::HLSSA,
        type_info: &crate::compiler::analysis::types::TypeInfo,
    ) {
        for (function_id, function) in ssa.iter_functions_mut() {
            let type_info = type_info.get_function(*function_id);
            for rtp in function.iter_returns_mut() {
                *rtp = self.witness_lowering_in_type(rtp);
            }
            // Collect converted block parameter types before taking blocks
            let block_param_types: HashMap<BlockId, Vec<Type>> = function
                .get_blocks()
                .map(|(bid, block)| {
                    let types = block
                        .get_parameters()
                        .map(|(_, tp)| self.witness_lowering_in_type(tp))
                        .collect();
                    (*bid, types)
                })
                .collect();

            let mut replacements = ValueReplacements::new();
            let block_ids: Vec<BlockId> = function.get_blocks().map(|(bid, _)| *bid).collect();
            for bid in block_ids {
                // Convert block parameters in-place
                let old_params = function.get_block_mut(bid).take_parameters();
                let new_params = old_params
                    .into_iter()
                    .map(|(r, tp)| (r, self.witness_lowering_in_type(&tp)))
                    .collect();
                function.get_block_mut(bid).put_parameters(new_params);

                let terminator = function.get_block_mut(bid).take_terminator();
                let instructions = function.get_block_mut(bid).take_instructions();

                let mut emitter = HLBlockEmitter::new(function, bid);

                for mut instruction in instructions.into_iter() {
                    replacements.replace_instruction(&mut instruction);
                    match instruction {
                        OpCode::Guard { .. } => {
                            panic!("ICE: Guard should be lowered before witness lowering");
                        }
                        OpCode::Cast {
                            result: r,
                            value: v,
                            target: _,
                        } => {
                            let v_type = type_info.get_value_type(v);
                            if v_type.is_witness_of() {
                                replacements.insert(r, v);
                            } else {
                                emitter.emit(instruction);
                            }
                        }
                        OpCode::FreshWitness { .. } => {
                            emitter.emit(instruction);
                        }
                        OpCode::MkSeq {
                            result: r,
                            elems: vs,
                            seq_type: s,
                            elem_type: tp,
                        } => {
                            let new_elem_type = self.witness_lowering_in_type(&tp);
                            let new_vs = vs
                                .iter()
                                .map(|v| {
                                    self.convert_if_needed(
                                        *v,
                                        &new_elem_type,
                                        type_info,
                                        &mut emitter,
                                    )
                                })
                                .collect();
                            emitter.emit(OpCode::MkSeq {
                                result: r,
                                elems: new_vs,
                                seq_type: s,
                                elem_type: new_elem_type,
                            });
                        }
                        OpCode::Alloc {
                            result: r,
                            elem_type: tp,
                        } => {
                            emitter.emit(OpCode::Alloc {
                                result: r,
                                elem_type: self.witness_lowering_in_type(&tp),
                            });
                        }
                        OpCode::Constrain { a, b, c } => {
                            let a = self.ensure_witness_ref(a, type_info, &mut emitter);
                            let b = self.ensure_witness_ref(b, type_info, &mut emitter);
                            let c = self.ensure_witness_ref(c, type_info, &mut emitter);
                            let new_val = emitter.fresh_value();
                            emitter.emit(OpCode::NextDCoeff { result: new_val });
                            emitter.emit(OpCode::BumpD {
                                matrix: DMatrix::A,
                                variable: a,
                                sensitivity: new_val,
                            });
                            emitter.emit(OpCode::BumpD {
                                matrix: DMatrix::B,
                                variable: b,
                                sensitivity: new_val,
                            });
                            emitter.emit(OpCode::BumpD {
                                matrix: DMatrix::C,
                                variable: c,
                                sensitivity: new_val,
                            });
                        }
                        OpCode::Lookup {
                            target,
                            keys,
                            results,
                            flag,
                        } => {
                            let mut new_keys = vec![];
                            for key in keys.iter() {
                                let key_type = type_info.get_value_type(*key);
                                assert!(
                                    key_type.strip_witness().is_field(),
                                    "Keys of lookup must be fields"
                                );
                                if !key_type.is_witness_of() {
                                    new_keys.push(emitter.cast_to_witness_of(*key));
                                } else {
                                    new_keys.push(*key);
                                }
                            }
                            let mut new_results = vec![];
                            for result in results.iter() {
                                let result_type = type_info.get_value_type(*result);
                                assert!(
                                    result_type.strip_witness().is_field(),
                                    "Results of lookup must be fields"
                                );
                                if !result_type.is_witness_of() {
                                    new_results.push(emitter.cast_to_witness_of(*result));
                                } else {
                                    new_results.push(*result);
                                }
                            }
                            let new_flag = {
                                let flag_type = type_info.get_value_type(flag);
                                if !flag_type.is_witness_of() {
                                    emitter.cast_to_witness_of(flag)
                                } else {
                                    flag
                                }
                            };
                            emitter.emit(OpCode::DLookup {
                                target,
                                keys: new_keys,
                                results: new_results,
                                flag: new_flag,
                            });
                        }
                        OpCode::BinaryArithOp {
                            kind,
                            result: r,
                            lhs: a,
                            rhs: b,
                        } => {
                            let a_type = type_info.get_value_type(a);
                            let b_type = type_info.get_value_type(b);
                            match (a, a_type.is_witness_of(), b, b_type.is_witness_of()) {
                                (_, true, _, true) => match kind {
                                    BinaryArithOpKind::Sub => {
                                        let neg_one =
                                            emitter.field_const(ark_bn254::Fr::from(-1i64));
                                        let neg_b = emitter.fresh_value();
                                        emitter.emit(OpCode::MulConst {
                                            result: neg_b,
                                            const_val: neg_one,
                                            var: b,
                                        });
                                        emitter.emit(OpCode::BinaryArithOp {
                                            kind: BinaryArithOpKind::Add,
                                            result: r,
                                            lhs: a,
                                            rhs: neg_b,
                                        });
                                    }
                                    _ => {
                                        emitter.emit(instruction);
                                    }
                                },
                                (_, false, _, false) => {
                                    emitter.emit(instruction);
                                }
                                (wit, true, pure, false) | (pure, false, wit, true) => match kind {
                                    BinaryArithOpKind::Add => {
                                        let pure_refed = emitter.cast_to_witness_of(pure);
                                        emitter.emit(OpCode::BinaryArithOp {
                                            kind,
                                            result: r,
                                            lhs: pure_refed,
                                            rhs: wit,
                                        });
                                    }
                                    BinaryArithOpKind::Mul => {
                                        emitter.emit(OpCode::MulConst {
                                            result: r,
                                            const_val: pure,
                                            var: wit,
                                        });
                                    }
                                    BinaryArithOpKind::Div => {
                                        panic!("Div is not supported for witness-pure arithmetic")
                                    }
                                    BinaryArithOpKind::Sub => {
                                        let pure_refed = emitter.cast_to_witness_of(pure);
                                        let lhs_ref = if a == wit { wit } else { pure_refed };
                                        let rhs_ref = if b == wit { wit } else { pure_refed };
                                        let neg_one =
                                            emitter.field_const(ark_bn254::Fr::from(-1i64));
                                        let neg_rhs = emitter.fresh_value();
                                        emitter.emit(OpCode::MulConst {
                                            result: neg_rhs,
                                            const_val: neg_one,
                                            var: rhs_ref,
                                        });
                                        emitter.emit(OpCode::BinaryArithOp {
                                            kind: BinaryArithOpKind::Add,
                                            result: r,
                                            lhs: lhs_ref,
                                            rhs: neg_rhs,
                                        });
                                    }
                                    BinaryArithOpKind::And => {
                                        panic!("And is not supported for witness-pure arithmetic")
                                    }
                                },
                            }
                        }
                        OpCode::Store { ptr, value } => {
                            let ptr_type = type_info.get_value_type(ptr);
                            let new_ptr_type = self.witness_lowering_in_type(&ptr_type);
                            let converted = self.convert_if_needed(
                                value,
                                &new_ptr_type,
                                type_info,
                                &mut emitter,
                            );
                            emitter.emit(OpCode::Store {
                                ptr,
                                value: converted,
                            });
                        }
                        OpCode::ArraySet {
                            result,
                            array,
                            index,
                            value,
                        } => {
                            let array_type = type_info.get_value_type(array);
                            let new_array_type = self.witness_lowering_in_type(&array_type);
                            let expected_elem_type = match &new_array_type.expr {
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
                            let new_slice_type = self.witness_lowering_in_type(&slice_type);
                            let expected_elem_type = match &new_slice_type.expr {
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
                        OpCode::Select {
                            result: r,
                            cond,
                            if_t,
                            if_f,
                        } => {
                            let result_type = type_info.get_value_type(r);
                            let target_type = self.witness_lowering_in_type(result_type);
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
                        OpCode::Not { .. }
                        | OpCode::Cmp { .. }
                        | OpCode::Truncate { .. }
                        | OpCode::Load { .. }
                        | OpCode::AssertEq { .. }
                        | OpCode::AssertR1C { .. }
                        | OpCode::Call { .. }
                        | OpCode::ArrayGet { .. }
                        | OpCode::SliceLen { .. }
                        | OpCode::ToBits { .. }
                        | OpCode::ToRadix { .. }
                        | OpCode::MemOp { .. }
                        | OpCode::WriteWitness { .. }
                        | OpCode::NextDCoeff { .. }
                        | OpCode::BumpD { .. }
                        | OpCode::DLookup { .. }
                        | OpCode::MulConst { .. }
                        | OpCode::Rangecheck { .. }
                        | OpCode::ReadGlobal { .. }
                        | OpCode::InitGlobal { .. }
                        | OpCode::DropGlobal { .. }
                        | OpCode::TupleProj { .. }
                        | OpCode::Todo { .. }
                        | OpCode::ValueOf { .. }
                        | OpCode::Const { .. } => {
                            emitter.emit(instruction);
                        }
                        OpCode::MkTuple {
                            result,
                            elems,
                            element_types,
                        } => {
                            let new_element_types = element_types
                                .iter()
                                .map(|tp| self.witness_lowering_in_type(tp))
                                .collect();
                            emitter.emit(OpCode::MkTuple {
                                result,
                                elems,
                                element_types: new_element_types,
                            });
                        }
                    };
                }

                // Handle terminator on current (possibly split) block
                if let Some(mut terminator) = terminator {
                    replacements.replace_terminator(&mut terminator);
                    match &mut terminator {
                        Terminator::Jmp(target, args) => {
                            let param_types = &block_param_types[target];
                            for (arg, expected_type) in args.iter_mut().zip(param_types.iter()) {
                                *arg = self.convert_if_needed(
                                    *arg,
                                    expected_type,
                                    type_info,
                                    &mut emitter,
                                );
                            }
                        }
                        Terminator::JmpIf(_, _, _) => {}
                        Terminator::Return(_) => {}
                    }
                    emitter.set_terminator(terminator);
                }
            }
        }
    }

    /// Emit instructions to convert a value from `source_type` to `target_type`.
    /// For scalars (Field/U), emits a CastToWitnessOf instruction inline.
    /// For arrays, generates a loop that converts each element, which splits the
    /// current block and creates new blocks.
    fn emit_value_conversion(
        &self,
        value: ValueId,
        source_type: &Type,
        target_type: &Type,
        emitter: &mut HLBlockEmitter<'_>,
    ) -> ValueId {
        let converted_source = self.witness_lowering_in_type(source_type);
        if converted_source == *target_type {
            return value;
        }

        match (&source_type.expr, &target_type.expr) {
            (TypeExpr::Field, TypeExpr::WitnessOf(_))
            | (TypeExpr::U(_), TypeExpr::WitnessOf(_)) => emitter.cast_to_witness_of(value),
            (TypeExpr::Array(src_inner, src_size), TypeExpr::Array(tgt_inner, tgt_size)) => {
                assert_eq!(
                    src_size, tgt_size,
                    "Array size mismatch in witness_lowering conversion"
                );
                self.emit_array_conversion_loop(
                    value,
                    src_inner,
                    tgt_inner,
                    *src_size,
                    source_type,
                    target_type,
                    emitter,
                )
            }
            (TypeExpr::Tuple(src_fields), TypeExpr::Tuple(tgt_fields)) => {
                assert_eq!(
                    src_fields.len(),
                    tgt_fields.len(),
                    "Tuple field count mismatch in witness_lowering conversion"
                );
                let mut converted_elems = vec![];
                for (i, (src_ft, tgt_ft)) in src_fields.iter().zip(tgt_fields.iter()).enumerate() {
                    let proj = emitter.tuple_proj(value, i);
                    let converted = self.emit_value_conversion(proj, src_ft, tgt_ft, emitter);
                    converted_elems.push(converted);
                }
                emitter.mk_tuple(converted_elems, tgt_fields.clone())
            }
            (TypeExpr::WitnessOf(_), TypeExpr::WitnessOf(_)) => {
                // Both source and target are WitnessOf — same runtime representation.
                value
            }
            _ => panic!(
                "witness_lowering value conversion not supported: {:?} -> {:?}",
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
        _source_array_type: &Type,
        target_array_type: &Type,
        emitter: &mut HLBlockEmitter<'_>,
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

    /// Convert a value to the given target type if its converted type doesn't already match.
    fn convert_if_needed(
        &self,
        value: ValueId,
        target_type: &Type,
        type_info: &crate::compiler::analysis::types::FunctionTypeInfo,
        emitter: &mut HLBlockEmitter<'_>,
    ) -> ValueId {
        let value_type = type_info.get_value_type(value);
        let converted_type = self.witness_lowering_in_type(&value_type);
        if converted_type == *target_type {
            value
        } else {
            self.emit_value_conversion(value, &value_type, target_type, emitter)
        }
    }

    fn ensure_witness_ref(
        &self,
        val: ValueId,
        type_info: &crate::compiler::analysis::types::FunctionTypeInfo,
        b: &mut impl HLEmitter,
    ) -> ValueId {
        let val_type = type_info.get_value_type(val);
        if val_type.is_witness_of() {
            val
        } else {
            b.cast_to_witness_of(val)
        }
    }

    fn witness_lowering_in_type(&self, tp: &Type) -> Type {
        match &tp.expr {
            TypeExpr::Field | TypeExpr::U(_) => {
                if tp.is_witness_of() {
                    Type::witness_of(tp.clone())
                } else {
                    tp.clone()
                }
            }
            TypeExpr::Array(inner, size) => self.witness_lowering_in_type(inner).array_of(*size),
            TypeExpr::Slice(inner) => self.witness_lowering_in_type(inner).slice_of(),
            TypeExpr::Ref(inner) => self.witness_lowering_in_type(inner).ref_of(),
            TypeExpr::WitnessOf(_) => tp.clone(),
            TypeExpr::Function => tp.clone(),
            TypeExpr::Tuple(elements) => {
                let boxed_elements = elements
                    .iter()
                    .map(|elem| self.witness_lowering_in_type(elem))
                    .collect();
                Type::tuple_of(boxed_elements)
            }
        }
    }
}
