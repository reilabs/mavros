//! Lowers witness operations such that there are no more implicit mixed pure / witness operations
//! and so that every value entering an R1CS cosntraint has been explicitly cast to `WitnessOf`.

use crate::{
    collections::HashMap,
    compiler::{
        analysis::{flow_analysis::FlowAnalysis, types::TypeInfo},
        pass_manager::{Analysis, AnalysisId, AnalysisStore, Pass},
        passes::fix_double_jumps::ValueReplacements,
        ssa::{
            BlockId, Terminator, ValueId,
            hlssa::{
                BinaryArithOpKind, CastTarget, DMatrix, HLSSA, OpCode, Type, TypeExpr,
                builder::{HLBlockEmitter, HLEmitter, HLSSABuilder},
            },
        },
        util::ice_non_elided_tuple,
    },
};

pub struct WitnessLowering {}

impl Pass for WitnessLowering {
    fn name(&self) -> &'static str {
        "witness_lowering"
    }

    fn needs(&self) -> Vec<AnalysisId> {
        vec![TypeInfo::id(), FlowAnalysis::id()]
    }

    fn run(&self, ssa: &mut HLSSA, store: &AnalysisStore) {
        self.do_run(ssa, store.get::<TypeInfo>(), store.get::<FlowAnalysis>());
    }
}

impl WitnessLowering {
    pub fn new() -> Self {
        Self {}
    }

    pub fn do_run(
        &self,
        ssa: &mut HLSSA,
        type_info: &crate::compiler::analysis::types::TypeInfo,
        flow_analysis: &FlowAnalysis,
    ) {
        let fids: Vec<_> = ssa.get_function_ids().collect();
        let mut sb = HLSSABuilder::new(ssa);
        for function_id in fids {
            let type_info = type_info.get_function(function_id);
            let cfg = flow_analysis.get_function_cfg(function_id);
            sb.modify_function(function_id, |fb| {
            for rtp in fb.function.iter_returns_mut() {
                *rtp = self.witness_lowering_in_type(rtp);
            }
            // Collect converted block parameter types before taking blocks
            let block_param_types: HashMap<BlockId, Vec<Type>> = fb
                .function
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
            let block_ids: Vec<BlockId> = cfg.get_domination_pre_order().collect();
            for bid in block_ids {
                // Convert block parameters in-place
                let old_params = fb.function.get_block_mut(bid).take_parameters();
                let new_params = old_params
                    .into_iter()
                    .map(|(r, tp)| (r, self.witness_lowering_in_type(&tp)))
                    .collect();
                fb.function.get_block_mut(bid).put_parameters(new_params);

                let terminator = fb.function.get_block_mut(bid).take_terminator();
                let instructions = fb.function.get_block_mut(bid).take_instructions();

                let mut emitter = fb.block(bid);

                for instruction in instructions.into_iter() {
                    let location = instruction.location().clone();
                    let mut instruction = instruction.payload();
                    replacements.replace_instruction(&mut instruction);
                    emitter.emit_with_location(location, |mut emitter| {
                        match instruction {
                        OpCode::Guard { .. } => {
                            panic!("ICE: Guard should be lowered before witness lowering");
                        }
                        OpCode::Cast {
                            result: r,
                            value: v,
                            ref target,
                        } => {
                            let v_type = type_info.get_value_type(v);
                            if v_type.is_witness_of() && matches!(target, CastTarget::WitnessOf) {
                                // Already WitnessOf — don't double-wrap
                                replacements.insert(r, v);
                            } else {
                                emitter.emit(OpCode::Cast {
                                    result: r,
                                    value: v,
                                    target: target.clone(),
                                });
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
                        OpCode::MkSeqOfBlob {
                            result: r,
                            element_type: tp,
                            blob,
                        } => {
                            let new_elem_type = self.witness_lowering_in_type(&tp);
                            emitter.emit(OpCode::MkSeqOfBlob {
                                result: r,
                                element_type: new_elem_type,
                                blob,
                            });
                        }
                        OpCode::MkRepeated {
                            result: r,
                            element,
                            seq_type,
                            count,
                            elem_type: tp,
                        } => {
                            let new_elem_type = self.witness_lowering_in_type(&tp);
                            let new_element = self.convert_if_needed(
                                element,
                                &new_elem_type,
                                type_info,
                                &mut emitter,
                            );
                            emitter.emit(OpCode::MkRepeated {
                                result: r,
                                element: new_element,
                                seq_type,
                                count,
                                elem_type: new_elem_type,
                            });
                        }
                        OpCode::Alloc { result: r, value } => {
                            let elem_type = type_info.get_value_type(value).clone();
                            let new_elem_type = self.witness_lowering_in_type(&elem_type);
                            let converted = self.convert_if_needed(
                                value,
                                &new_elem_type,
                                type_info,
                                &mut emitter,
                            );
                            emitter.emit(OpCode::Alloc {
                                result: r,
                                value: converted,
                            });
                        }
                        OpCode::Constrain { a, b, c } => {
                            let a = self.ensure_witness_ref(a, type_info, emitter);
                            let b = self.ensure_witness_ref(b, type_info, emitter);
                            let c = self.ensure_witness_ref(c, type_info, emitter);
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
                            args,
                            flag,
                        } => {
                            let mut new_args = vec![];
                            for arg in args.iter() {
                                let arg_type = type_info.get_value_type(*arg);
                                assert!(
                                    arg_type.strip_witness().is_field(),
                                    "Lookup args must be fields, got {:?}",
                                    arg_type
                                );
                                if !arg_type.is_witness_of() {
                                    new_args.push(emitter.cast_to_witness_of(*arg));
                                } else {
                                    new_args.push(*arg);
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
                                args: new_args,
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
                                    BinaryArithOpKind::Div if a == wit => {
                                        // wit / pure → MulConst(wit, 1/pure)
                                        let one = emitter.field_const(ark_bn254::Fr::from(1u64));
                                        let inv_pure = emitter.fresh_value();
                                        emitter.emit(OpCode::BinaryArithOp {
                                            kind: BinaryArithOpKind::Div,
                                            result: inv_pure,
                                            lhs: one,
                                            rhs: pure,
                                        });
                                        emitter.emit(OpCode::MulConst {
                                            result: r,
                                            const_val: inv_pure,
                                            var: wit,
                                        });
                                    }
                                    BinaryArithOpKind::Div | BinaryArithOpKind::Mod => {
                                        panic!(
                                            "Div/Mod is not supported for witness-pure arithmetic"
                                        )
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
                                    BinaryArithOpKind::And
                                    | BinaryArithOpKind::Or
                                    | BinaryArithOpKind::Xor
                                    | BinaryArithOpKind::Shl
                                    | BinaryArithOpKind::Shr => {
                                        panic!(
                                            "{:?} is not supported for witness-pure arithmetic",
                                            kind
                                        )
                                    }
                                },
                            }
                        }
                        OpCode::Store { ptr, value } => {
                            let ptr_type = type_info.get_value_type(ptr);
                            let new_ptr_type = self.witness_lowering_in_type(&ptr_type);
                            let elem_type = new_ptr_type.get_pointed();
                            let converted =
                                self.convert_if_needed(value, &elem_type, type_info, &mut emitter);
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
                            let result_slice_type = type_info.get_value_type(result);
                            let new_result_slice_type = self.witness_lowering_in_type(result_slice_type);
                            let expected_elem_type = match &new_result_slice_type.expr {
                                TypeExpr::Slice(inner) => inner.as_ref().clone(),
                                _ => panic!("SlicePush on non-slice type"),
                            };
                            let new_slice = self.convert_if_needed(
                                slice,
                                &new_result_slice_type,
                                type_info,
                                &mut emitter,
                            );
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
                                slice: new_slice,
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
                        | OpCode::SExt { .. }
                        | OpCode::BitRange { .. }
                        | OpCode::Load { .. }
                        | OpCode::Assert { .. }
                        | OpCode::AssertCmp { .. }
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
                        | OpCode::Todo { .. }
                        | OpCode::Spread { .. }
                        | OpCode::Unspread { .. } => {
                            emitter.emit(instruction);
                        }
                        OpCode::MkTuple { .. }
                        | OpCode::TupleProj { .. }
                        | OpCode::TupleRefProj { .. } => ice_non_elided_tuple(),
                        };
                    });
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
            });
        }
    }

    /// Emit instructions to convert a value from `source_type` to `target_type`.
    /// Scalar witness injections become a single `WitnessOf` cast; arrays and
    /// slices become one composite `Map` cast, lowered to an explicit loop by
    /// `LowerMapCasts` right after this pass.
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

        match CastTarget::conversion(&converted_source, target_type) {
            None => value,
            Some(target) => emitter.cast_to(target, value),
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
            TypeExpr::Field | TypeExpr::U(_) | TypeExpr::I(_) => {
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
            TypeExpr::Blob(..) => tp.clone(),
            TypeExpr::Tuple(_) => ice_non_elided_tuple(),
        }
    }
}
