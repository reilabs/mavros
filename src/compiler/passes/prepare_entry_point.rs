use crate::compiler::{
    block_builder::{BlockEmitter, FunctionBuilder, HLEmitter},
    ir::r#type::{Type, TypeExpr},
    pass_manager::{AnalysisStore, Pass},
    passes::fix_double_jumps::ValueReplacements,
    ssa::{
        BinaryArithOpKind, BlockId, CallTarget, CastTarget, ConstValue, FunctionId, HLFunction,
        HLSSA, OpCode, SeqType, TupleIdx, ValueId,
    },
};

pub struct PrepareEntryPoint {}

impl Pass for PrepareEntryPoint {
    fn name(&self) -> &'static str {
        "prepare_entry_point"
    }

    fn run(&self, ssa: &mut HLSSA, _store: &AnalysisStore) {
        Self::wrap_main(ssa);
        self.rebuild_main_params(ssa);
        Self::insert_witness_writes(ssa);
        Self::process_unconstrained_calls(ssa);
    }
}

impl PrepareEntryPoint {
    pub fn new() -> Self {
        Self {}
    }

    fn wrap_main(ssa: &mut HLSSA) {
        let original_main_id = ssa.get_main_id();
        let original_main = ssa.get_main();
        let param_types = original_main.get_param_types();
        let return_types = original_main.get_returns().to_vec();

        let globals_init_fn = ssa.get_globals_init_fn();
        let globals_deinit_fn = ssa.get_globals_deinit_fn();

        ssa.get_main_mut().set_name("original_main".to_string());

        let wrapper_id = ssa.add_function("wrapper_main".to_string());
        {
            let wrapper = ssa.get_function_mut(wrapper_id);
            let entry_block = wrapper.get_entry_id();
            let mut b = FunctionBuilder::new(wrapper);
            let mut e = b.block(entry_block);

            let mut arg_values = Vec::new();
            for typ in &param_types {
                arg_values.push(e.add_parameter(typ.clone()));
            }

            let mut return_input_values = Vec::new();
            for typ in &return_types {
                return_input_values.push(e.add_parameter(typ.clone()));
            }

            if let Some(init_fn) = globals_init_fn {
                e.call(init_fn, vec![], 0);
            }

            let results = e.call(original_main_id, arg_values, return_types.len());
            for ((result, public_input), return_type) in results
                .iter()
                .zip(return_input_values.iter())
                .zip(return_types.iter())
            {
                Self::assert_eq_deep(&mut e, *result, *public_input, return_type);
            }

            if let Some(deinit_fn) = globals_deinit_fn {
                e.call(deinit_fn, vec![], 0);
            }

            e.terminate_return(vec![]);
        }
        ssa.set_entry_point(wrapper_id);
    }

    fn assert_eq_deep(b: &mut BlockEmitter, result: ValueId, public_input: ValueId, typ: &Type) {
        match &typ.expr {
            TypeExpr::Field | TypeExpr::U(_) => {
                b.assert_eq(result, public_input);
            }
            TypeExpr::Array(inner, size) => {
                for i in 0..*size {
                    let index = b.u_const(32, i as u128);
                    let result_elem = b.array_get(result, index);
                    let input_elem = b.array_get(public_input, index);
                    Self::assert_eq_deep(b, result_elem, input_elem, inner);
                }
            }
            TypeExpr::Tuple(element_types) => {
                for (i, elem_type) in element_types.iter().enumerate() {
                    let result_elem = b.tuple_proj(result, TupleIdx::Static(i));
                    let input_elem = b.tuple_proj(public_input, TupleIdx::Static(i));
                    Self::assert_eq_deep(b, result_elem, input_elem, elem_type);
                }
            }
            _ => {
                b.assert_eq(result, public_input);
            }
        }
    }

    /// Insert pinned WriteWitness instructions in wrapper_main entry.
    /// The first write emits constant one for witness[0], then writes each entry param.
    fn insert_witness_writes(ssa: &mut HLSSA) {
        let main = ssa.get_main_mut();
        let entry_id = main.get_entry_id();

        // Collect entry params
        let params: Vec<ValueId> = main
            .get_entry()
            .get_parameters()
            .map(|(id, _)| *id)
            .collect();

        // witness[0] must be constant one, emitted by the program.
        let witness_one_value = main.fresh_value();
        let one_const_value = main.fresh_value();
        let mut write_witness_instructions = vec![
            OpCode::Const {
                result: one_const_value,
                value: ConstValue::Field(ark_bn254::Fr::from(1u64)),
            },
            OpCode::WriteWitness {
                result: Some(witness_one_value),
                value: one_const_value,
                pinned: true,
            },
        ];

        // Create pinned WriteWitness for each param and build replacements.
        // These must be pinned so they survive DCE even when their results become unused
        // (e.g. when inter-procedural DCE prunes call args to original_main).
        let mut replacements = ValueReplacements::new();
        for param_id in &params {
            let witness_val = main.fresh_value();
            write_witness_instructions.push(OpCode::WriteWitness {
                result: Some(witness_val),
                value: *param_id,
                pinned: true,
            });
            replacements.insert(*param_id, witness_val);
        }

        // Prepend WriteWitness instructions and apply replacements to existing instructions
        let entry_block = main.get_block_mut(entry_id);
        let old_instructions = entry_block.take_instructions();
        let mut new_instructions = write_witness_instructions;
        for mut instruction in old_instructions {
            replacements.replace_instruction(&mut instruction);
            new_instructions.push(instruction);
        }
        entry_block.put_instructions(new_instructions);
        replacements.replace_terminator(entry_block.get_terminator_mut());
    }

    /// Process unconstrained call results: flatten to Fields, WriteWitness,
    /// rangecheck + reconstruct, and replace original results.
    fn process_unconstrained_calls(ssa: &mut HLSSA) {
        // Collect callee return types first (need to borrow ssa immutably)
        let func_ids: Vec<FunctionId> = ssa.get_function_ids().collect();

        // Collect (function_id, block_id, instruction_index, callee_id, result_value_id)
        // for all unconstrained calls
        let mut unconstrained_call_sites: Vec<(FunctionId, BlockId, usize, FunctionId, ValueId)> =
            Vec::new();

        for &fid in &func_ids {
            let function = ssa.get_function(fid);
            for (&bid, block) in function.get_blocks() {
                for (idx, instr) in block.get_instructions().enumerate() {
                    if let OpCode::Call {
                        unconstrained: true,
                        results,
                        function: CallTarget::Static(callee_id),
                        ..
                    } = instr
                    {
                        if !results.is_empty() {
                            unconstrained_call_sites.push((fid, bid, idx, *callee_id, results[0]));
                        }
                    }
                }
            }
        }

        // Process each call site
        for (fid, bid, _idx, callee_id, result_vid) in unconstrained_call_sites {
            // Get callee return types
            let return_types: Vec<Type> = ssa.get_function(callee_id).get_returns().to_vec();
            if return_types.is_empty() {
                continue;
            }
            let return_type = return_types[0].clone();

            let function = ssa.get_function_mut(fid);

            let (reconstructed_val, mut extra_instructions) =
                Self::flatten_witness_reconstruct(&result_vid, &return_type, function);

            // Build replacement map
            let mut replacements = ValueReplacements::new();
            replacements.insert(result_vid, reconstructed_val);

            // Insert extra instructions after the unconstrained Call in the block's instruction list
            // and apply replacements to the ORIGINAL instructions that come after
            let block = function.get_block_mut(bid);
            let old_instructions = block.take_instructions();
            let extra_count = extra_instructions.len();
            let mut new_instructions = Vec::new();
            let mut found_call = false;
            for mut instr in old_instructions {
                if found_call {
                    // Apply replacement to original instructions after the call+extras
                    replacements.replace_instruction(&mut instr);
                }
                let is_the_call = if let OpCode::Call {
                    unconstrained: true,
                    results,
                    ..
                } = &instr
                {
                    !results.is_empty() && results[0] == result_vid
                } else {
                    false
                };
                new_instructions.push(instr);
                if is_the_call {
                    // Insert extra instructions (these should NOT get replacements applied)
                    new_instructions.extend(extra_instructions.drain(..));
                    found_call = true;
                }
            }
            block.put_instructions(new_instructions);
            replacements.replace_terminator(block.get_terminator_mut());
        }
    }

    /// Flatten a call result to field-level, WriteWitness each field, rangecheck + reconstruct.
    /// Returns (reconstructed_value_id, instructions_to_insert).
    fn flatten_witness_reconstruct(
        value_id: &ValueId,
        typ: &Type,
        function: &mut HLFunction,
    ) -> (ValueId, Vec<OpCode>) {
        let mut instructions = Vec::new();

        match &typ.expr {
            TypeExpr::Field => {
                // WriteWitness the field value directly
                let ww_result = function.fresh_value();
                instructions.push(OpCode::WriteWitness { result: Some(ww_result), value: *value_id, pinned: false });
                (ww_result, instructions)
            }
            TypeExpr::U(size) => {
                // Cast to Field, WriteWitness, rangecheck, cast back
                let as_field = function.fresh_value();
                instructions.push(OpCode::Cast { result: as_field, value: *value_id, target: CastTarget::Field });
                let ww_result = function.fresh_value();
                instructions.push(OpCode::WriteWitness { result: Some(ww_result), value: as_field, pinned: false });

                if *size == 1 {
                    // Boolean constraint: x * (x - 1) = 0
                    let zero = function.fresh_value();
                    instructions.push(OpCode::Const { result: zero, value: ConstValue::Field(ark_bn254::Fr::from(0)) });
                    let one = function.fresh_value();
                    instructions.push(OpCode::Const { result: one, value: ConstValue::Field(ark_bn254::Fr::from(1)) });
                    let x_sub_1 = function.fresh_value();
                    let x_times_x_sub_1 = function.fresh_value();
                    instructions.push(OpCode::BinaryArithOp {
                        kind: BinaryArithOpKind::Sub,
                        result: x_sub_1,
                        lhs: ww_result,
                        rhs: one,
                    });
                    instructions.push(OpCode::BinaryArithOp {
                        kind: BinaryArithOpKind::Mul,
                        result: x_times_x_sub_1,
                        lhs: ww_result,
                        rhs: x_sub_1,
                    });
                    instructions.push(OpCode::AssertEq {
                        lhs: x_times_x_sub_1,
                        rhs: zero,
                    });
                } else {
                    instructions.push(OpCode::Rangecheck {
                        value: ww_result,
                        max_bits: *size,
                    });
                }

                let casted = function.fresh_value();
                instructions.push(OpCode::Cast {
                    result: casted,
                    value: ww_result,
                    target: CastTarget::U(*size),
                });
                (casted, instructions)
            }
            TypeExpr::Array(inner, size) => {
                let mut elems = Vec::new();
                for i in 0..*size {
                    let index = function.fresh_value();
                    instructions.push(OpCode::Const { result: index, value: ConstValue::U(32, i as u128) });
                    let elem = function.fresh_value();
                    instructions.push(OpCode::ArrayGet {
                        result: elem,
                        array: *value_id,
                        index,
                    });
                    let (reconstructed_elem, child_instructions) =
                        Self::flatten_witness_reconstruct(&elem, inner, function);
                    instructions.extend(child_instructions);
                    elems.push(reconstructed_elem);
                }
                let reconstructed = function.fresh_value();
                instructions.push(OpCode::MkSeq {
                    result: reconstructed,
                    elems,
                    seq_type: SeqType::Array(*size),
                    elem_type: *inner.clone(),
                });
                (reconstructed, instructions)
            }
            TypeExpr::Tuple(element_types) => {
                let mut elems = Vec::new();
                let mut elem_types = Vec::new();
                for (i, elem_type) in element_types.iter().enumerate() {
                    let proj = function.fresh_value();
                    instructions.push(OpCode::TupleProj {
                        result: proj,
                        tuple: *value_id,
                        idx: TupleIdx::Static(i),
                    });
                    let (reconstructed_elem, child_instructions) =
                        Self::flatten_witness_reconstruct(&proj, elem_type, function);
                    instructions.extend(child_instructions);
                    elems.push(reconstructed_elem);
                    elem_types.push(elem_type.clone());
                }
                let reconstructed = function.fresh_value();
                instructions.push(OpCode::MkTuple {
                    result: reconstructed,
                    elems,
                    element_types: elem_types,
                });
                (reconstructed, instructions)
            }
            _ => {
                // For other types, just WriteWitness directly
                let ww_result = function.fresh_value();
                instructions.push(OpCode::WriteWitness { result: Some(ww_result), value: *value_id, pinned: false });
                (ww_result, instructions)
            }
        }
    }

    fn rebuild_main_params(&self, ssa: &mut HLSSA) {
        let function = ssa.get_main_mut();

        let params: Vec<_> = function.get_entry().get_parameters().cloned().collect();

        let mut new_instructions = Vec::new();
        let mut new_parameters = Vec::new();

        for (value_id, typ) in params.iter() {
            let (_, child_parameters, child_instructions) =
                Self::reconstruct_param(Some(value_id), typ, function);
            new_parameters.extend(child_parameters);
            new_instructions.extend(child_instructions);
        }

        let entry_id = function.get_entry_id();
        let entry_block = function.get_block_mut(entry_id);
        new_instructions.extend(entry_block.take_instructions());

        entry_block.put_parameters(new_parameters);
        entry_block.put_instructions(new_instructions);
    }

    fn reconstruct_param(
        value_id: Option<&ValueId>,
        typ: &Type,
        function: &mut HLFunction,
    ) -> (ValueId, Vec<(ValueId, Type)>, Vec<OpCode>) {
        let mut new_instructions = Vec::new();
        let mut new_parameters = Vec::new();

        let value_id = if let Some(id) = value_id {
            id
        } else {
            &(function.fresh_value())
        };

        match &typ.expr {
            TypeExpr::Field => new_parameters.push((*value_id, typ.clone())),
            TypeExpr::U(size) => {
                let new_field_id = function.fresh_value();
                new_parameters.push((new_field_id, Type::field()));

                if *size == 1 {
                    // Boolean constraint: x * (x - 1) = 0
                    let zero = function.fresh_value();
                    new_instructions.push(OpCode::Const {
                        result: zero,
                        value: ConstValue::Field(ark_bn254::Fr::from(0)),
                    });
                    let one = function.fresh_value();
                    new_instructions.push(OpCode::Const {
                        result: one,
                        value: ConstValue::Field(ark_bn254::Fr::from(1)),
                    });
                    let x_sub_1 = function.fresh_value();
                    let x_times_x_sub_1 = function.fresh_value();
                    new_instructions.push(OpCode::BinaryArithOp {
                        kind: BinaryArithOpKind::Sub,
                        result: x_sub_1,
                        lhs: new_field_id,
                        rhs: one,
                    });
                    new_instructions.push(OpCode::BinaryArithOp {
                        kind: BinaryArithOpKind::Mul,
                        result: x_times_x_sub_1,
                        lhs: new_field_id,
                        rhs: x_sub_1,
                    });
                    new_instructions.push(OpCode::AssertEq {
                        lhs: x_times_x_sub_1,
                        rhs: zero,
                    });
                } else {
                    new_instructions.push(OpCode::Rangecheck {
                        value: new_field_id,
                        max_bits: *size,
                    });
                }

                new_instructions.push(OpCode::Cast {
                    result: *value_id,
                    value: new_field_id,
                    target: CastTarget::U(*size),
                });
            }
            TypeExpr::Array(inner, size) => {
                let mut elems = Vec::new();
                for _ in 0..*size {
                    let (child_id, child_parameters, child_instructions) =
                        Self::reconstruct_param(None, inner, function);
                    elems.push(child_id);
                    new_parameters.extend(child_parameters);
                    new_instructions.extend(child_instructions);
                }
                new_instructions.push(OpCode::MkSeq {
                    result: *value_id,
                    elems,
                    seq_type: SeqType::Array(*size),
                    elem_type: *inner.clone(),
                });
            }
            TypeExpr::Tuple(element_types) => {
                let mut elems = Vec::new();
                let mut elem_types = Vec::new();
                for elem_type in element_types {
                    let (child_id, child_parameters, child_instructions) =
                        Self::reconstruct_param(None, elem_type, function);
                    elems.push(child_id);
                    elem_types.push(elem_type.clone());
                    new_parameters.extend(child_parameters);
                    new_instructions.extend(child_instructions);
                }
                new_instructions.push(OpCode::MkTuple {
                    result: *value_id,
                    elems,
                    element_types: elem_types,
                });
            }
            _ => todo!("Not implemented yet"),
        }

        return (*value_id, new_parameters, new_instructions);
    }
}
