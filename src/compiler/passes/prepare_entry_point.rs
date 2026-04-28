use std::collections::HashMap;

use crate::compiler::{
    block_builder::{HLBlockEmitter, HLEmitter, HLFunctionBuilder},
    ir::r#type::{Type, TypeExpr},
    pass_manager::{AnalysisStore, Pass},
    passes::fix_double_jumps::ValueReplacements,
    ssa::{
        BlockId, CallTarget, CastTarget, ConstValue, FunctionId, HLFunction, HLSSA, OpCode,
        SeqType, ValueId,
    },
};

pub struct PrepareEntryPoint {
    main_is_unconstrained: bool,
}

struct PrepareFnEntry {
    hlssa_type: Type,
    fn_id: FunctionId,
}

struct ReconstructFnEntry {
    hlssa_type: Type,
    fn_id: FunctionId,
}

impl Pass for PrepareEntryPoint {
    fn name(&self) -> &'static str {
        "prepare_entry_point"
    }

    fn run(&self, ssa: &mut HLSSA, _store: &AnalysisStore) {
        Self::wrap_main(ssa, self.main_is_unconstrained);
        self.rebuild_main_params(ssa);
        Self::insert_witness_writes(ssa);
        Self::process_unconstrained_calls(ssa);
    }
}

impl PrepareEntryPoint {
    pub fn new(main_is_unconstrained: bool) -> Self {
        Self {
            main_is_unconstrained,
        }
    }

    fn wrap_main(ssa: &mut HLSSA, main_is_unconstrained: bool) {
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
            let mut b = HLFunctionBuilder::new(wrapper);
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

            let results = if main_is_unconstrained {
                e.call_unconstrained(original_main_id, arg_values, return_types.len())
            } else {
                e.call(original_main_id, arg_values, return_types.len())
            };
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

    fn assert_eq_deep(
        b: &mut HLBlockEmitter<'_>,
        result: ValueId,
        public_input: ValueId,
        typ: &Type,
    ) {
        match &typ.expr {
            TypeExpr::Field | TypeExpr::U(_) | TypeExpr::I(_) => {
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
                    let result_elem = b.tuple_proj(result, i);
                    let input_elem = b.tuple_proj(public_input, i);
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
        // Pre-collect callee return types (need immutable ssa access)
        let mut callee_return_types: HashMap<FunctionId, Vec<Type>> = HashMap::new();
        let func_ids: Vec<FunctionId> = ssa.get_function_ids().collect();
        for &fid in &func_ids {
            for (_, block) in ssa.get_function(fid).get_blocks() {
                for instr in block.get_instructions() {
                    if let OpCode::Call {
                        unconstrained: true,
                        function: CallTarget::Static(callee_id),
                        ..
                    } = instr
                    {
                        callee_return_types
                            .entry(*callee_id)
                            .or_insert_with(|| ssa.get_function(*callee_id).get_returns().to_vec());
                    }
                }
            }
        }

        let mut prepare_fns = Vec::new();
        for return_types in callee_return_types.values() {
            for return_type in return_types {
                Self::get_or_create_prepare_fn(return_type, ssa, &mut prepare_fns);
            }
        }

        // Process each function/block: single pass over instructions, handling
        // all unconstrained calls inline so indices never go stale.
        for &fid in &func_ids {
            let block_ids: Vec<BlockId> = ssa
                .get_function(fid)
                .get_blocks()
                .map(|(id, _)| *id)
                .collect();
            for bid in block_ids {
                let function = ssa.get_function_mut(fid);
                let block = function.get_block_mut(bid);
                let old_instructions = block.take_instructions();

                let mut replacements = ValueReplacements::new();
                let mut new_instructions = Vec::new();

                for mut instr in old_instructions {
                    replacements.replace_instruction(&mut instr);

                    let call_results: Vec<(ValueId, Type)> = if let OpCode::Call {
                        unconstrained: true,
                        results,
                        function: CallTarget::Static(callee_id),
                        ..
                    } = &instr
                    {
                        callee_return_types
                            .get(callee_id)
                            .map(|rt| {
                                results
                                    .iter()
                                    .zip(rt.iter())
                                    .map(|(r, t)| (*r, t.clone()))
                                    .collect()
                            })
                            .unwrap_or_default()
                    } else {
                        vec![]
                    };

                    new_instructions.push(instr);

                    for (result_vid, return_type) in call_results {
                        let reconstructed = function.fresh_value();
                        let prepare_fn = Self::find_prepare_fn(&return_type, &prepare_fns);
                        new_instructions.push(OpCode::Call {
                            results: vec![reconstructed],
                            function: CallTarget::Static(prepare_fn),
                            args: vec![result_vid],
                            unconstrained: false,
                        });
                        replacements.insert(result_vid, reconstructed);
                    }
                }

                let block = function.get_block_mut(bid);
                block.put_instructions(new_instructions);
                replacements.replace_terminator(block.get_terminator_mut());
            }
        }
    }

    fn find_prepare_fn(typ: &Type, prepare_fns: &[PrepareFnEntry]) -> FunctionId {
        prepare_fns
            .iter()
            .find(|entry| entry.hlssa_type == *typ)
            .map(|entry| entry.fn_id)
            .expect("prepare function should have been pre-created")
    }

    fn unique_prepare_fn_name(prepare_fns: &[PrepareFnEntry]) -> String {
        format!("prepare_{}", prepare_fns.len())
    }

    fn unique_reconstruct_fn_name(reconstruct_fns: &[ReconstructFnEntry]) -> String {
        format!("reconstruct_{}", reconstruct_fns.len())
    }

    fn flattened_field_count(typ: &Type) -> usize {
        match &typ.expr {
            TypeExpr::Field | TypeExpr::U(_) | TypeExpr::I(_) => 1,
            TypeExpr::Array(inner, size) => Self::flattened_field_count(inner) * size,
            TypeExpr::Tuple(element_types) => {
                element_types.iter().map(Self::flattened_field_count).sum()
            }
            TypeExpr::WitnessOf(inner) => Self::flattened_field_count(inner),
            _ => todo!("Not implemented yet"),
        }
    }

    fn get_or_create_prepare_fn(
        typ: &Type,
        ssa: &mut HLSSA,
        prepare_fns: &mut Vec<PrepareFnEntry>,
    ) -> FunctionId {
        if let Some(entry) = prepare_fns.iter().find(|entry| entry.hlssa_type == *typ) {
            return entry.fn_id;
        }

        let fn_id = ssa.add_function(Self::unique_prepare_fn_name(prepare_fns));
        prepare_fns.push(PrepareFnEntry {
            hlssa_type: typ.clone(),
            fn_id,
        });

        let child_fns = match &typ.expr {
            TypeExpr::Array(inner, _) => {
                vec![Self::get_or_create_prepare_fn(inner, ssa, prepare_fns)]
            }
            TypeExpr::Tuple(element_types) => element_types
                .iter()
                .map(|elem_type| Self::get_or_create_prepare_fn(elem_type, ssa, prepare_fns))
                .collect(),
            _ => Vec::new(),
        };

        {
            let function = ssa.get_function_mut(fn_id);
            function.add_return_type(typ.clone());
            let entry_block = function.get_entry_id();
            let mut b = HLFunctionBuilder::new(function);
            let mut e = b.block(entry_block);
            let param = e.add_parameter(typ.clone());
            let result = Self::emit_prepare_body(&mut e, param, typ, &child_fns);
            e.terminate_return(vec![result]);
        }

        fn_id
    }

    fn emit_prepare_body(
        e: &mut HLBlockEmitter<'_>,
        value_id: ValueId,
        typ: &Type,
        child_fns: &[FunctionId],
    ) -> ValueId {
        match &typ.expr {
            TypeExpr::Field => e.write_witness(value_id),
            TypeExpr::U(size) | TypeExpr::I(size) => {
                let cast_back = match &typ.expr {
                    TypeExpr::U(s) => CastTarget::U(*s),
                    TypeExpr::I(s) => CastTarget::I(*s),
                    _ => unreachable!(),
                };
                let as_field = e.cast_to_field(value_id);
                let witness = e.write_witness(as_field);

                if *size == 1 {
                    let zero = e.field_const(ark_bn254::Fr::from(0));
                    let one = e.field_const(ark_bn254::Fr::from(1));
                    let x_sub_1 = e.sub(witness, one);
                    let x_times_x_sub_1 = e.mul(witness, x_sub_1);
                    e.assert_eq(x_times_x_sub_1, zero);
                } else {
                    e.rangecheck(witness, *size);
                }

                e.cast_to(cast_back, witness)
            }
            TypeExpr::Array(_, size) => {
                let child_fn = child_fns
                    .first()
                    .copied()
                    .expect("array prepare function should have child function");
                let prepared_array = e.build_counted_loop(
                    *size,
                    vec![(value_id, typ.clone())],
                    |e, index, accumulators| {
                        let current_array = accumulators[0];
                        let elem = e.array_get(current_array, index);
                        let prepared = e.call(child_fn, vec![elem], 1);
                        let updated_array = e.array_set(current_array, index, prepared[0]);
                        vec![updated_array]
                    },
                );
                prepared_array[0]
            }
            TypeExpr::Tuple(element_types) => {
                let mut elems = Vec::with_capacity(element_types.len());
                for (i, child_fn) in child_fns.iter().enumerate() {
                    let elem = e.tuple_proj(value_id, i);
                    let prepared = e.call(*child_fn, vec![elem], 1);
                    elems.push(prepared[0]);
                }
                e.mk_tuple(elems, element_types.clone())
            }
            _ => e.write_witness(value_id),
        }
    }

    fn rebuild_main_params(&self, ssa: &mut HLSSA) {
        let params: Vec<_> = ssa
            .get_main()
            .get_entry()
            .get_parameters()
            .cloned()
            .collect();
        let mut reconstruct_fns = Vec::new();
        for (_, typ) in &params {
            Self::get_or_create_reconstruct_fn(typ, ssa, &mut reconstruct_fns);
        }

        let function = ssa.get_main_mut();

        let mut new_instructions = Vec::new();
        let mut new_parameters = Vec::new();

        for (value_id, typ) in params.iter() {
            let (_, child_parameters, child_instructions) =
                Self::reconstruct_param(Some(*value_id), typ, function, &reconstruct_fns);
            new_parameters.extend(child_parameters);
            new_instructions.extend(child_instructions);
        }

        let entry_id = function.get_entry_id();
        let entry_block = function.get_block_mut(entry_id);
        new_instructions.extend(entry_block.take_instructions());

        entry_block.put_parameters(new_parameters);
        entry_block.put_instructions(new_instructions);
    }

    fn find_reconstruct_fn(typ: &Type, reconstruct_fns: &[ReconstructFnEntry]) -> FunctionId {
        reconstruct_fns
            .iter()
            .find(|entry| entry.hlssa_type == *typ)
            .map(|entry| entry.fn_id)
            .expect("reconstruct function should have been pre-created")
    }

    fn get_or_create_reconstruct_fn(
        typ: &Type,
        ssa: &mut HLSSA,
        reconstruct_fns: &mut Vec<ReconstructFnEntry>,
    ) -> FunctionId {
        if let Some(entry) = reconstruct_fns
            .iter()
            .find(|entry| entry.hlssa_type == *typ)
        {
            return entry.fn_id;
        }

        match &typ.expr {
            TypeExpr::Array(inner, _) => {
                Self::get_or_create_reconstruct_fn(inner, ssa, reconstruct_fns);
            }
            TypeExpr::Tuple(element_types) => {
                for elem_type in element_types {
                    Self::get_or_create_reconstruct_fn(elem_type, ssa, reconstruct_fns);
                }
            }
            _ => {}
        }

        let fn_id = ssa.add_function(Self::unique_reconstruct_fn_name(reconstruct_fns));
        reconstruct_fns.push(ReconstructFnEntry {
            hlssa_type: typ.clone(),
            fn_id,
        });

        {
            let function = ssa.get_function_mut(fn_id);
            function.add_return_type(typ.clone());
            let entry_block = function.get_entry_id();
            let mut b = HLFunctionBuilder::new(function);
            let mut e = b.block(entry_block);

            let input_len = Self::flattened_field_count(typ);
            let input_array = e.add_parameter(Type::field().array_of(input_len));
            let result = Self::emit_reconstruct_body(&mut e, typ, input_array, reconstruct_fns);
            e.terminate_return(vec![result]);
        }

        fn_id
    }

    fn emit_reconstruct_body(
        e: &mut HLBlockEmitter<'_>,
        typ: &Type,
        input_array: ValueId,
        reconstruct_fns: &[ReconstructFnEntry],
    ) -> ValueId {
        match &typ.expr {
            TypeExpr::Field => {
                let zero = e.u_const(32, 0);
                e.array_get(input_array, zero)
            }
            TypeExpr::U(size) | TypeExpr::I(size) => {
                let zero = e.u_const(32, 0);
                let field_param = e.array_get(input_array, zero);
                if *size == 1 {
                    let zero = e.field_const(ark_bn254::Fr::from(0));
                    let one = e.field_const(ark_bn254::Fr::from(1));
                    let x_sub_1 = e.sub(field_param, one);
                    let x_times_x_sub_1 = e.mul(field_param, x_sub_1);
                    e.assert_eq(x_times_x_sub_1, zero);
                } else {
                    e.rangecheck(field_param, *size);
                }

                let cast_back = match &typ.expr {
                    TypeExpr::U(s) => CastTarget::U(*s),
                    TypeExpr::I(s) => CastTarget::I(*s),
                    _ => unreachable!(),
                };
                e.cast_to(cast_back, field_param)
            }
            TypeExpr::Array(inner, size) => {
                let child_fn = Self::find_reconstruct_fn(inner, reconstruct_fns);
                let child_width = Self::flattened_field_count(inner);
                let initial_array = Self::emit_default_array(e, inner, *size);
                let reconstructed_array = e.build_counted_loop(
                    *size,
                    vec![(initial_array, typ.clone())],
                    |e, index, accumulators| {
                        let current_array = accumulators[0];
                        let child_input = Self::emit_reconstruct_child_input_array_at_index(
                            e,
                            input_array,
                            index,
                            child_width,
                        );
                        let reconstructed = e.call(child_fn, vec![child_input], 1);
                        let updated_array = e.array_set(current_array, index, reconstructed[0]);
                        vec![updated_array]
                    },
                );
                reconstructed_array[0]
            }
            TypeExpr::Tuple(element_types) => {
                let mut elems = Vec::with_capacity(element_types.len());
                let mut field_offset = 0;
                for elem_type in element_types {
                    let child_fn = Self::find_reconstruct_fn(elem_type, reconstruct_fns);
                    let child_width = Self::flattened_field_count(elem_type);
                    let child_input = Self::emit_reconstruct_child_input_array(
                        e,
                        input_array,
                        field_offset,
                        child_width,
                    );
                    let reconstructed = e.call(child_fn, vec![child_input], 1);
                    elems.push(reconstructed[0]);
                    field_offset += child_width;
                }
                e.mk_tuple(elems, element_types.clone())
            }
            _ => todo!("Not implemented yet"),
        }
    }

    fn emit_default_value(e: &mut HLBlockEmitter<'_>, typ: &Type) -> ValueId {
        match &typ.expr {
            TypeExpr::Field => e.field_const(ark_bn254::Fr::from(0)),
            TypeExpr::U(size) => e.u_const(*size, 0),
            TypeExpr::I(size) => e.i_const(*size, 0),
            TypeExpr::Array(inner, size) => Self::emit_default_array(e, inner, *size),
            TypeExpr::Tuple(element_types) => {
                let elems = element_types
                    .iter()
                    .map(|elem_type| Self::emit_default_value(e, elem_type))
                    .collect();
                e.mk_tuple(elems, element_types.clone())
            }
            _ => todo!("Not implemented yet"),
        }
    }

    fn emit_default_array(e: &mut HLBlockEmitter<'_>, inner: &Type, size: usize) -> ValueId {
        let elems = if size == 0 {
            Vec::new()
        } else {
            let default_elem = Self::emit_default_value(e, inner);
            vec![default_elem; size]
        };
        e.mk_seq(elems, SeqType::Array(size), inner.clone())
    }

    fn emit_reconstruct_child_input_array(
        e: &mut HLBlockEmitter<'_>,
        input_array: ValueId,
        start: usize,
        len: usize,
    ) -> ValueId {
        let fields = (0..len)
            .map(|i| {
                let index = e.u_const(32, (start + i) as u128);
                e.array_get(input_array, index)
            })
            .collect();
        e.mk_seq(fields, SeqType::Array(len), Type::field())
    }

    fn emit_reconstruct_child_input_array_at_index(
        e: &mut HLBlockEmitter<'_>,
        input_array: ValueId,
        index: ValueId,
        width: usize,
    ) -> ValueId {
        let start = if width == 1 {
            index
        } else {
            let width_value = e.u_const(32, width as u128);
            e.mul(index, width_value)
        };

        let fields = (0..width)
            .map(|i| {
                let field_index = if i == 0 {
                    start
                } else {
                    let offset = e.u_const(32, i as u128);
                    e.add(start, offset)
                };
                e.array_get(input_array, field_index)
            })
            .collect();
        e.mk_seq(fields, SeqType::Array(width), Type::field())
    }

    fn reconstruct_param(
        value_id: Option<ValueId>,
        typ: &Type,
        function: &mut HLFunction,
        reconstruct_fns: &[ReconstructFnEntry],
    ) -> (ValueId, Vec<(ValueId, Type)>, Vec<OpCode>) {
        let mut new_instructions = Vec::new();
        let mut new_parameters = Vec::new();

        let value_id = value_id.unwrap_or_else(|| function.fresh_value());

        match &typ.expr {
            TypeExpr::Field => new_parameters.push((value_id, typ.clone())),
            TypeExpr::U(_) | TypeExpr::I(_) | TypeExpr::Array(_, _) | TypeExpr::Tuple(_) => {
                let input_len = Self::flattened_field_count(typ);
                let mut fields = Vec::with_capacity(input_len);
                for _ in 0..input_len {
                    let field_id = function.fresh_value();
                    new_parameters.push((field_id, Type::field()));
                    fields.push(field_id);
                }

                let input_array = function.fresh_value();
                new_instructions.push(OpCode::MkSeq {
                    result: input_array,
                    elems: fields,
                    seq_type: SeqType::Array(input_len),
                    elem_type: Type::field(),
                });
                let fn_id = Self::find_reconstruct_fn(typ, reconstruct_fns);
                new_instructions.push(OpCode::Call {
                    results: vec![value_id],
                    function: CallTarget::Static(fn_id),
                    args: vec![input_array],
                    unconstrained: false,
                });
            }
            _ => todo!("Not implemented yet"),
        }

        (value_id, new_parameters, new_instructions)
    }
}
