use std::collections::HashMap;

use crate::compiler::{
    ir::r#type::{Type, TypeExpr},
    pass_manager::{DataPoint, Pass},
    passes::fix_double_jumps::ValueReplacements,
    ssa::{BinaryArithOpKind, Block, BlockId, CastTarget, CmpKind, DMatrix, OpCode, SeqType, Terminator, TupleIdx, ValueId},
};

pub struct WitnessLowering {}

impl Pass for WitnessLowering {
    fn run(
        &self,
        ssa: &mut crate::compiler::ssa::SSA,
        pass_manager: &crate::compiler::pass_manager::PassManager,
    ) {
        self.do_run(ssa, pass_manager.get_type_info());
    }

    fn pass_info(&self) -> crate::compiler::pass_manager::PassInfo {
        crate::compiler::pass_manager::PassInfo {
            name: "witness_lowering",
            needs: vec![DataPoint::Types],
        }
    }

    fn invalidates_cfg(&self) -> bool {
        true
    }
}

impl WitnessLowering {
    pub fn new() -> Self {
        Self {}
    }

    pub fn do_run(
        &self,
        ssa: &mut crate::compiler::ssa::SSA,
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
            let mut new_blocks = HashMap::new();
            for (bid, mut block) in function.take_blocks().into_iter() {
                let old_params = block.take_parameters();
                let new_params = old_params
                    .into_iter()
                    .map(|(r, tp)| (r, self.witness_lowering_in_type(&tp)))
                    .collect();
                block.put_parameters(new_params);

                let terminator = block.take_terminator();
                let instructions = block.take_instructions();

                let mut current_block_id = bid;
                let mut current_block = block;
                let mut new_instructions = vec![];

                for mut instruction in instructions.into_iter() {
                    replacements.replace_instruction(&mut instruction);
                    match instruction {
                        OpCode::Cast {
                            result: r,
                            value: v,
                            target: t,
                        } => {
                            let v_type = type_info.get_value_type(v);
                            if v_type.is_witness_of() {
                                // The value is already WitnessOf — the cast strips it to Field.
                                // Instead of emitting a Nop, alias the result to the original
                                // value so ensure_witness_ref sees the WitnessOf type and
                                // doesn't double-wrap.
                                replacements.insert(r, v);
                            } else {
                                new_instructions.push(instruction);
                            }
                        }
                        OpCode::FreshWitness {
                            result: r,
                            result_type: tp,
                        } => {
                            let i = OpCode::FreshWitness {
                                result: r,
                                result_type: Type::witness_of(tp.clone()),
                            };
                            new_instructions.push(i);
                        }
                        OpCode::MkSeq {
                            result: r,
                            elems: vs,
                            seq_type: s,
                            elem_type: tp,
                        } => {
                            let new_elem_type = self.witness_lowering_in_type(&tp);
                            let new_vs = vs.iter().map(|v| {
                                self.convert_if_needed(
                                    *v, &new_elem_type, type_info,
                                    &mut current_block_id, &mut current_block,
                                    &mut new_instructions, function, &mut new_blocks,
                                )
                            }).collect();
                            let i = OpCode::MkSeq {
                                result: r,
                                elems: new_vs,
                                seq_type: s,
                                elem_type: new_elem_type,
                            };
                            new_instructions.push(i);
                        }
                        OpCode::Alloc { result: r, elem_type: tp } => {
                            let i = OpCode::Alloc {
                                result: r,
                                elem_type: self.witness_lowering_in_type(&tp),
                            };
                            new_instructions.push(i);
                        }
                        OpCode::Constrain { a, b, c } => {
                            let a = self.ensure_witness_ref(a, type_info, &mut new_instructions, function);
                            let b = self.ensure_witness_ref(b, type_info, &mut new_instructions, function);
                            let c = self.ensure_witness_ref(c, type_info, &mut new_instructions, function);
                            let new_val = function.fresh_value();
                            new_instructions.push(OpCode::NextDCoeff { result: new_val });
                            new_instructions.push(OpCode::BumpD {
                                matrix: DMatrix::A,
                                variable: a,
                                sensitivity: new_val,
                            });
                            new_instructions.push(OpCode::BumpD {
                                matrix: DMatrix::B,
                                variable: b,
                                sensitivity: new_val,
                            });
                            new_instructions.push(OpCode::BumpD {
                                matrix: DMatrix::C,
                                variable: c,
                                sensitivity: new_val,
                            });
                        }
                        OpCode::Lookup {
                            target,
                            keys,
                            results,
                        } => {
                            let mut new_keys = vec![];
                            for key in keys.iter() {
                                let key_type = type_info.get_value_type(*key);
                                assert!(key_type.strip_witness().is_field(), "Keys of lookup must be fields");
                                if !key_type.is_witness_of() {
                                    let refed = function.fresh_value();
                                    new_instructions.push(OpCode::Cast { result: refed, value: *key, target: CastTarget::WitnessOf });
                                    new_keys.push(refed);
                                } else {
                                    new_keys.push(*key);
                                }
                            }
                            let mut new_results = vec![];
                            for result in results.iter() {
                                let result_type = type_info.get_value_type(*result);
                                assert!(result_type.strip_witness().is_field(), "Results of lookup must be fields");
                                if !result_type.is_witness_of() {
                                    let refed = function.fresh_value();
                                    new_instructions.push(OpCode::Cast { result: refed, value: *result, target: CastTarget::WitnessOf });
                                    new_results.push(refed);
                                } else {
                                    new_results.push(*result);
                                }
                            }
                            new_instructions.push(OpCode::DLookup {
                                target,
                                keys: new_keys,
                                results: new_results,
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
                            match (
                                a,
                                a_type.is_witness_of(),
                                b,
                                b_type.is_witness_of(),
                            ) {
                                (_, true, _, true) => match kind {
                                    BinaryArithOpKind::Sub => {
                                        // Lower Sub(wit, wit) to Add(a, MulConst(-1, b))
                                        let neg_one = function.push_field_const(ark_bn254::Fr::from(-1i64));
                                        let neg_b = function.fresh_value();
                                        new_instructions.push(OpCode::MulConst {
                                            result: neg_b,
                                            const_val: neg_one,
                                            var: b,
                                        });
                                        new_instructions.push(OpCode::BinaryArithOp {
                                            kind: BinaryArithOpKind::Add,
                                            result: r,
                                            lhs: a,
                                            rhs: neg_b,
                                        });
                                    }
                                    _ => {
                                        new_instructions.push(instruction);
                                    }
                                },
                                (_, false, _, false) => {
                                    new_instructions.push(instruction);
                                }
                                (wit, true, pure, false) | (pure, false, wit, true) => match kind {
                                    BinaryArithOpKind::Add => {
                                        let pure_refed = function.fresh_value();
                                        new_instructions.push(OpCode::Cast { result: pure_refed, value: pure, target: CastTarget::WitnessOf });
                                        new_instructions.push(OpCode::BinaryArithOp {
                                            kind: kind,
                                            result: r,
                                            lhs: pure_refed,
                                            rhs: wit,
                                        });
                                    }
                                    BinaryArithOpKind::Mul => {
                                        new_instructions.push(OpCode::MulConst {
                                            result: r,
                                            const_val: pure,
                                            var: wit,
                                        });
                                    }
                                    BinaryArithOpKind::Div => {
                                        panic!("Div is not supported for witness-pure arithmetic")
                                    }
                                    BinaryArithOpKind::Sub => {
                                        // Lower Sub(a, b) where one is pure/one is witness
                                        // to Add(a_ref, MulConst(-1, b_ref))
                                        let pure_refed = function.fresh_value();
                                        new_instructions.push(OpCode::Cast { result: pure_refed, value: pure, target: CastTarget::WitnessOf });
                                        let lhs_ref = if a == wit { wit } else { pure_refed };
                                        let rhs_ref = if b == wit { wit } else { pure_refed };
                                        let neg_one = function.push_field_const(ark_bn254::Fr::from(-1i64));
                                        let neg_rhs = function.fresh_value();
                                        new_instructions.push(OpCode::MulConst {
                                            result: neg_rhs,
                                            const_val: neg_one,
                                            var: rhs_ref,
                                        });
                                        new_instructions.push(OpCode::BinaryArithOp {
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
                                value, &new_ptr_type, type_info,
                                &mut current_block_id, &mut current_block,
                                &mut new_instructions, function, &mut new_blocks,
                            );
                            new_instructions.push(OpCode::Store { ptr, value: converted });
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
                                value, &expected_elem_type, type_info,
                                &mut current_block_id, &mut current_block,
                                &mut new_instructions, function, &mut new_blocks,
                            );
                            new_instructions.push(OpCode::ArraySet {
                                result, array, index, value: converted,
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
                            let new_values = values.iter().map(|v| {
                                self.convert_if_needed(
                                    *v, &expected_elem_type, type_info,
                                    &mut current_block_id, &mut current_block,
                                    &mut new_instructions, function, &mut new_blocks,
                                )
                            }).collect();
                            new_instructions.push(OpCode::SlicePush {
                                dir, result, slice, values: new_values,
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
                            let if_t = self.convert_if_needed(
                                if_t, &target_type, type_info,
                                &mut current_block_id, &mut current_block,
                                &mut new_instructions, function, &mut new_blocks,
                            );
                            let if_f = self.convert_if_needed(
                                if_f, &target_type, type_info,
                                &mut current_block_id, &mut current_block,
                                &mut new_instructions, function, &mut new_blocks,
                            );
                            new_instructions.push(OpCode::Select {
                                result: r, cond, if_t, if_f,
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
                        /* PureToWitnessRef removed */
                        | OpCode::MulConst { .. }
                        | OpCode::Rangecheck { .. }
                        | OpCode::ReadGlobal { .. }
                        | OpCode::InitGlobal { .. }
                        | OpCode::DropGlobal { .. }
                        | OpCode::TupleProj { .. }
                        | OpCode::Todo { .. }
                        | OpCode::ValueOf { .. } => {
                            new_instructions.push(instruction);
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
                            new_instructions.push(OpCode::MkTuple {
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
                                    *arg, expected_type, type_info,
                                    &mut current_block_id, &mut current_block,
                                    &mut new_instructions, function, &mut new_blocks,
                                );
                            }
                        }
                        Terminator::JmpIf(_, _, _) => {}
                        Terminator::Return(_) => {}
                    }
                    current_block.put_instructions(new_instructions);
                    current_block.set_terminator(terminator);
                } else {
                    current_block.put_instructions(new_instructions);
                }
                new_blocks.insert(current_block_id, current_block);
            }
            function.put_blocks(new_blocks);
        }
    }

    /// Emit instructions to convert a value from `source_type` to `target_type`.
    /// For scalars (Field/U), emits a PureToWitnessRef instruction inline.
    /// For arrays, generates a loop that converts each element, which splits the
    /// current block and creates new blocks.
    fn emit_value_conversion(
        &self,
        value: ValueId,
        source_type: &Type,
        target_type: &Type,
        current_block_id: &mut BlockId,
        current_block: &mut Block,
        new_instructions: &mut Vec<OpCode>,
        function: &mut crate::compiler::ssa::Function,
        new_blocks: &mut HashMap<BlockId, Block>,
    ) -> ValueId {
        let converted_source = self.witness_lowering_in_type(source_type);
        if converted_source == *target_type {
            return value;
        }

        match (&source_type.expr, &target_type.expr) {
            (TypeExpr::Field, TypeExpr::WitnessOf(_)) | (TypeExpr::U(_), TypeExpr::WitnessOf(_)) => {
                let refed = function.fresh_value();
                new_instructions.push(OpCode::Cast { result: refed, value: value, target: CastTarget::WitnessOf });
                refed
            }
            (TypeExpr::Array(src_inner, src_size), TypeExpr::Array(tgt_inner, tgt_size)) => {
                assert_eq!(src_size, tgt_size, "Array size mismatch in witness_lowering conversion");
                self.emit_array_conversion_loop(
                    value,
                    src_inner,
                    tgt_inner,
                    *src_size,
                    source_type,
                    target_type,
                    current_block_id,
                    current_block,
                    new_instructions,
                    function,
                    new_blocks,
                )
            }
            (TypeExpr::Tuple(src_fields), TypeExpr::Tuple(tgt_fields)) => {
                assert_eq!(src_fields.len(), tgt_fields.len(), "Tuple field count mismatch in witness_lowering conversion");
                let mut converted_elems = vec![];
                for (i, (src_ft, tgt_ft)) in src_fields.iter().zip(tgt_fields.iter()).enumerate() {
                    let proj = function.fresh_value();
                    new_instructions.push(OpCode::TupleProj {
                        result: proj,
                        tuple: value,
                        idx: TupleIdx::Static(i),
                    });
                    let converted = self.emit_value_conversion(
                        proj,
                        src_ft,
                        tgt_ft,
                        current_block_id,
                        current_block,
                        new_instructions,
                        function,
                        new_blocks,
                    );
                    converted_elems.push(converted);
                }
                let result = function.fresh_value();
                new_instructions.push(OpCode::MkTuple {
                    result,
                    elems: converted_elems,
                    element_types: tgt_fields.clone(),
                });
                result
            }
            (TypeExpr::WitnessOf(_), TypeExpr::WitnessOf(_)) => {
                // Both source and target are WitnessOf — same runtime representation.
                // The inner types may differ (e.g. WitnessOf(Field) vs WitnessOf(U(1)))
                // but all WitnessOf values are witness references in the AD VM.
                value
            }
            _ => panic!(
                "witness_lowering value conversion not supported: {:?} -> {:?}",
                source_type, target_type
            ),
        }
    }

    /// Generate a loop that iterates over array elements and converts each one.
    /// Uses `source_array` directly from the dominating block for reads (no loop param
    /// needed since it doesn't change), and a properly-typed `dst` loop parameter for
    /// writes. The initial `dst` is a dummy array created via MkSeq to ensure correct
    /// memory layout (Field and WitnessRef have different VM sizes).
    fn emit_array_conversion_loop(
        &self,
        source_array: ValueId,
        src_elem_type: &Type,
        tgt_elem_type: &Type,
        array_len: usize,
        _source_array_type: &Type,
        target_array_type: &Type,
        current_block_id: &mut BlockId,
        current_block: &mut Block,
        new_instructions: &mut Vec<OpCode>,
        function: &mut crate::compiler::ssa::Function,
        new_blocks: &mut HashMap<BlockId, Block>,
    ) -> ValueId {
        // Create a properly-typed initial target array filled with dummy elements.
        // This ensures the dst array has the correct memory layout from the start.
        let initial_dst = self.create_dummy_array(
            tgt_elem_type,
            array_len,
            target_array_type,
            new_instructions,
            function,
        );

        // Create loop blocks
        let (loop_header_id, mut loop_header) = function.next_virtual_block();
        let (loop_body_id, loop_body) = function.next_virtual_block();
        let (continuation_id, continuation) = function.next_virtual_block();

        // Constants (u32 for array indexing)
        let const_0 = function.push_u_const(32, 0);
        let const_1 = function.push_u_const(32, 1);
        let const_len = function.push_u_const(32, array_len as u128);

        // Finalize current block: Jmp to loop_header with (i=0, dst=initial_dst)
        // source_array is accessed directly from the dominating block, not as a loop param.
        current_block.put_instructions(std::mem::take(new_instructions));
        current_block.set_terminator(Terminator::Jmp(
            loop_header_id,
            vec![const_0, initial_dst],
        ));
        let old_block = std::mem::replace(current_block, continuation);
        new_blocks.insert(*current_block_id, old_block);
        *current_block_id = continuation_id;

        // Loop header parameters: (i: U32, dst: target_array_type)
        let i_val = function.fresh_value();
        let dst_val = function.fresh_value();
        loop_header.put_parameters(vec![
            (i_val, Type::u(32)),
            (dst_val, target_array_type.clone()),
        ]);

        // Loop header: cond = i < len, jmpif cond body continuation
        let cond_val = function.fresh_value();
        loop_header.push_instruction(OpCode::Cmp {
            kind: CmpKind::Lt,
            result: cond_val,
            lhs: i_val,
            rhs: const_len,
        });
        loop_header.set_terminator(Terminator::JmpIf(cond_val, loop_body_id, continuation_id));
        new_blocks.insert(loop_header_id, loop_header);

        // Loop body: get element from source_array, convert, set into dst
        let mut body_block_id = loop_body_id;
        let mut body_block = loop_body;
        let mut body_instructions = vec![];

        // ArrayGet from source_array (dominates loop, correct source element type)
        let elem_val = function.fresh_value();
        body_instructions.push(OpCode::ArrayGet {
            result: elem_val,
            array: source_array,
            index: i_val,
        });

        // Convert element (may recursively split body block for nested arrays)
        let converted_elem = self.emit_value_conversion(
            elem_val,
            src_elem_type,
            tgt_elem_type,
            &mut body_block_id,
            &mut body_block,
            &mut body_instructions,
            function,
            new_blocks,
        );

        // ArraySet converted element into dst (correct target type and stride)
        let new_dst = function.fresh_value();
        body_instructions.push(OpCode::ArraySet {
            result: new_dst,
            array: dst_val,
            index: i_val,
            value: converted_elem,
        });

        // Increment index
        let next_i = function.fresh_value();
        body_instructions.push(OpCode::BinaryArithOp {
            kind: BinaryArithOpKind::Add,
            result: next_i,
            lhs: i_val,
            rhs: const_1,
        });

        // Jump back to loop header (only i and dst change, no self-copies)
        body_block.put_instructions(body_instructions);
        body_block.set_terminator(Terminator::Jmp(loop_header_id, vec![next_i, new_dst]));
        new_blocks.insert(body_block_id, body_block);

        // At loop exit, dst holds the fully converted array
        dst_val
    }

    /// Create a dummy array of the given target type, properly laid out in memory.
    /// Used to initialize the dst array before the conversion loop.
    fn create_dummy_array(
        &self,
        elem_type: &Type,
        array_len: usize,
        _array_type: &Type,
        new_instructions: &mut Vec<OpCode>,
        function: &mut crate::compiler::ssa::Function,
    ) -> ValueId {
        let dummy_elem = self.create_dummy_value(elem_type, new_instructions, function);
        let elems = vec![dummy_elem; array_len];
        let result = function.fresh_value();
        new_instructions.push(OpCode::MkSeq {
            result,
            elems,
            seq_type: SeqType::Array(array_len),
            elem_type: elem_type.clone(),
        });
        result
    }

    /// Create a single dummy value of the given target type.
    /// For WitnessRef: wraps a zero field constant.
    /// For arrays/tuples: recursively creates dummy elements.
    fn create_dummy_value(
        &self,
        target_type: &Type,
        new_instructions: &mut Vec<OpCode>,
        function: &mut crate::compiler::ssa::Function,
    ) -> ValueId {
        match &target_type.expr {
            TypeExpr::WitnessOf(_) => {
                let dummy_field = function.push_field_const(ark_bn254::Fr::from(0u64));
                let refed = function.fresh_value();
                new_instructions.push(OpCode::Cast { result: refed, value: dummy_field, target: CastTarget::WitnessOf });
                refed
            }
            TypeExpr::Array(inner, size) => {
                self.create_dummy_array(inner, *size, target_type, new_instructions, function)
            }
            TypeExpr::Tuple(fields) => {
                let mut dummy_elems = vec![];
                for field_type in fields.iter() {
                    dummy_elems.push(self.create_dummy_value(field_type, new_instructions, function));
                }
                let result = function.fresh_value();
                new_instructions.push(OpCode::MkTuple {
                    result,
                    elems: dummy_elems,
                    element_types: fields.clone(),
                });
                result
            }
            TypeExpr::Field | TypeExpr::U(_) => {
                // Pure scalar types that don't need conversion — use a zero constant
                function.push_field_const(ark_bn254::Fr::from(0u64))
            }
            _ => panic!("create_dummy_value: unsupported type {:?}", target_type),
        }
    }

    /// Convert a value to the given target type if its converted type doesn't already match.
    fn convert_if_needed(
        &self,
        value: ValueId,
        target_type: &Type,
        type_info: &crate::compiler::analysis::types::FunctionTypeInfo,
        current_block_id: &mut BlockId,
        current_block: &mut Block,
        new_instructions: &mut Vec<OpCode>,
        function: &mut crate::compiler::ssa::Function,
        new_blocks: &mut HashMap<BlockId, Block>,
    ) -> ValueId {
        let value_type = type_info.get_value_type(value);
        let converted_type = self.witness_lowering_in_type(&value_type);
        if converted_type == *target_type {
            value
        } else {
            self.emit_value_conversion(
                value,
                &value_type,
                target_type,
                current_block_id,
                current_block,
                new_instructions,
                function,
                new_blocks,
            )
        }
    }

    fn ensure_witness_ref(
        &self,
        val: ValueId,
        type_info: &crate::compiler::analysis::types::FunctionTypeInfo,
        new_instructions: &mut Vec<OpCode>,
        function: &mut crate::compiler::ssa::Function,
    ) -> ValueId {
        let val_type = type_info.get_value_type(val);
        if val_type.is_witness_of() {
            val
        } else {
            let refed = function.fresh_value();
            new_instructions.push(OpCode::Cast { result: refed, value: val, target: CastTarget::WitnessOf });
            refed
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
            TypeExpr::Array(inner, size) => self
                .witness_lowering_in_type(inner)
                .array_of(*size),
            TypeExpr::Slice(inner) => {
                self.witness_lowering_in_type(inner).slice_of()
            }
            TypeExpr::Ref(inner) => {
                self.witness_lowering_in_type(inner).ref_of()
            }
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
