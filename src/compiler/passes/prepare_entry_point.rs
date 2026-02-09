use std::collections::HashMap;
use crate::compiler::{ir::r#type::{Empty, Type, TypeExpr}, pass_manager::{Pass, PassInfo}, ssa::{BinaryArithOpKind, BlockId, CallTarget, CastTarget, Const, Function, FunctionId, GlobalDef, OpCode, SeqType, TupleIdx, ValueId, SSA}};

pub struct PrepareEntryPoint {}

impl Pass<Empty> for PrepareEntryPoint {
    fn run(
        &self,
        ssa: &mut SSA<Empty>,
        _pass_manager: &crate::compiler::pass_manager::PassManager<Empty>,
    ) {
        Self::wrap_main(ssa);
        Self::constrain_unconstrained_returns(ssa);
        self.rebuild_main_params(ssa);
    }

    fn pass_info(&self) -> PassInfo {
        PassInfo {
            name: "prepare_entry_point",
            needs: vec![],
        }
    }
}

impl PrepareEntryPoint {
    pub fn new() -> Self {
        Self {}
    }

    fn wrap_main(ssa: &mut SSA<Empty>) {
        let original_main_id = ssa.get_main_id();
        let original_main = ssa.get_main();
        let param_types = original_main.get_param_types();
        let return_types = original_main.get_returns().to_vec();
        let is_main_unconstrained = original_main.is_unconstrained();

        // Collect globals info before borrowing ssa mutably
        let globals: Vec<GlobalDef> = ssa.get_globals().to_vec();

        ssa.get_main_mut().set_name("original_main".to_string());

        let wrapper_id = ssa.add_function("wrapper_main".to_string());
        let wrapper = ssa.get_function_mut(wrapper_id);

        let entry_block = wrapper.get_entry_id();
        let mut arg_values = Vec::new();
        for typ in &param_types {
            let val = wrapper.add_parameter(entry_block, typ.clone());
            arg_values.push(val);
        }

        let mut return_input_values = Vec::new();
        for typ in &return_types {
            let val = wrapper.add_parameter(entry_block, typ.clone());
            return_input_values.push(val);
        }

        // Emit InitGlobal for each global before the call
        for (i, global) in globals.iter().enumerate() {
            match global {
                GlobalDef::Const(Const::Field(f)) => {
                    let value = wrapper.push_field_const(f.clone());
                    wrapper.push_init_global(entry_block, i, value);
                }
                GlobalDef::Const(Const::U(s, v)) => {
                    let value = wrapper.push_u_const(*s, *v);
                    wrapper.push_init_global(entry_block, i, value);
                }
                GlobalDef::Const(Const::WitnessRef(_)) => {
                    todo!("WitnessRef globals not yet supported in InitGlobal");
                }
                GlobalDef::Const(Const::FnPtr(_)) => {
                    panic!("FnPtr globals not supported in prepare_entry_point");
                }
                GlobalDef::Array(indices, elem_type) => {
                    // Each index refers to an already-initialized global
                    let elems: Vec<ValueId> = indices
                        .iter()
                        .map(|idx| {
                            let global_type = Self::global_type(&globals[*idx]);
                            wrapper.push_read_global(entry_block, *idx as u64, global_type)
                        })
                        .collect();
                    let mk_seq_result = wrapper.fresh_value();
                    let block = wrapper.get_block_mut(entry_block);
                    block.push_instruction(OpCode::MkSeq {
                        result: mk_seq_result,
                        elems,
                        seq_type: SeqType::Array(indices.len()),
                        elem_type: elem_type.clone(),
                    });
                    wrapper.push_init_global(entry_block, i, mk_seq_result);
                }
            }
        }

        let results = wrapper.push_call(
            entry_block,
            original_main_id,
            arg_values,
            return_types.len(),
            is_main_unconstrained,
        );
        for ((result, public_input), return_type) in results.iter().zip(return_input_values.iter()).zip(return_types.iter()) {
            Self::assert_eq_deep(wrapper, entry_block, *result, *public_input, return_type);
        }

        // Emit DropGlobal in reverse for globals that need RC (arrays, tuples)
        for i in (0..globals.len()).rev() {
            if Self::global_needs_drop(&globals[i]) {
                wrapper.push_drop_global(entry_block, i);
            }
        }

        wrapper.terminate_block_with_return(entry_block, vec![]);

        ssa.set_entry_point(wrapper_id);
    }

    fn global_type(global: &GlobalDef) -> Type<Empty> {
        match global {
            GlobalDef::Const(Const::Field(_)) => Type::field(Empty),
            GlobalDef::Const(Const::U(s, _)) => Type::u(*s, Empty),
            GlobalDef::Const(Const::WitnessRef(_)) => Type::witness_ref(Empty),
            GlobalDef::Const(Const::FnPtr(_)) => Type::function(Empty),
            GlobalDef::Array(indices, elem_type) => {
                elem_type.clone().array_of(indices.len(), Empty)
            }
        }
    }

    fn global_needs_drop(global: &GlobalDef) -> bool {
        match global {
            GlobalDef::Const(Const::Field(_)) | GlobalDef::Const(Const::U(_, _)) => false,
            GlobalDef::Const(Const::WitnessRef(_)) => true,
            GlobalDef::Const(Const::FnPtr(_)) => false,
            GlobalDef::Array(_, _) => true,
        }
    }

    fn assert_eq_deep(
        wrapper: &mut Function<Empty>,
        block: BlockId,
        result: ValueId,
        public_input: ValueId,
        typ: &Type<Empty>,
    ) {
        match &typ.expr {
            TypeExpr::Field | TypeExpr::U(_) => {
                wrapper.push_assert_eq(block, result, public_input);
            }
            TypeExpr::Array(inner, size) => {
                for i in 0..*size {
                    let index = wrapper.push_u_const(32, i as u128);
                    let result_elem = wrapper.push_array_get(block, result, index);
                    let input_elem = wrapper.push_array_get(block, public_input, index);
                    Self::assert_eq_deep(wrapper, block, result_elem, input_elem, inner);
                }
            }
            TypeExpr::Tuple(element_types) => {
                for (i, elem_type) in element_types.iter().enumerate() {
                    let result_elem = wrapper.push_tuple_proj(block, result, TupleIdx::Static(i));
                    let input_elem = wrapper.push_tuple_proj(block, public_input, TupleIdx::Static(i));
                    Self::assert_eq_deep(wrapper, block, result_elem, input_elem, elem_type);
                }
            }
            _ => {
                wrapper.push_assert_eq(block, result, public_input);
            }
        }
    }

    fn rebuild_main_params(
        &self,
        ssa: &mut SSA<Empty>,
    ) {
        let function = ssa.get_main_mut();

        let params: Vec<_> = function.get_entry().get_parameters().cloned().collect();

        let mut new_instructions = Vec::new();
        let mut new_parameters = Vec::new();

        for (value_id, typ) in params.iter() {
            let (_, child_parameters, child_instructions) = Self::reconstruct_param(Some(value_id), typ, function);
            new_parameters.extend(child_parameters);
            new_instructions.extend(child_instructions);
        }

        let entry_id = function.get_entry_id();
        let entry_block = function.get_block_mut(entry_id);
        new_instructions.extend(entry_block.take_instructions());

        entry_block.put_parameters(new_parameters);
        entry_block.put_instructions(new_instructions);
    }

    fn reconstruct_param (value_id: Option<&ValueId>, typ: &Type<Empty>, function: &mut Function<Empty>) -> (ValueId, Vec<(ValueId, Type<Empty>)>, Vec<OpCode<Empty>>) {
        let mut new_instructions = Vec::new();
        let mut new_parameters = Vec::new();

        let value_id = if let Some(id) = value_id {
            id
        } else {
            &(function.fresh_value())
        };

        match &typ.expr {
            TypeExpr::Field => {
                new_parameters.push((*value_id, typ.clone()))
            }
            TypeExpr::U(size) => {
                let new_field_id = function.fresh_value();
                new_parameters.push((new_field_id, Type{expr: TypeExpr::Field, annotation: Empty}));

                if *size == 1 {
                    // Boolean constraint: x * (x - 1) = 0
                    let zero = function.push_field_const(ark_bn254::Fr::from(0));
                    let one = function.push_field_const(ark_bn254::Fr::from(1));
                    let x_sub_1 = function.fresh_value();
                    let x_times_x_sub_1 = function.fresh_value();
                    new_instructions.push(
                        OpCode::BinaryArithOp {
                            kind: BinaryArithOpKind::Sub,
                            result: x_sub_1,
                            lhs: new_field_id,
                            rhs: one,
                        }
                    );
                    new_instructions.push(
                        OpCode::BinaryArithOp {
                            kind: BinaryArithOpKind::Mul,
                            result: x_times_x_sub_1,
                            lhs: new_field_id,
                            rhs: x_sub_1,
                        }
                    );
                    new_instructions.push(
                        OpCode::AssertEq {
                            lhs: x_times_x_sub_1,
                            rhs: zero,
                        }
                    );
                } else {
                    new_instructions.push(
                        OpCode::Rangecheck {
                            value: new_field_id,
                            max_bits: *size,
                        }
                    );
                }

                new_instructions.push(
                    OpCode::Cast {
                        result: *value_id,
                        value: new_field_id,
                        target: CastTarget::U(*size),
                    }
                );
            }
            TypeExpr::Array(inner, size) => {
                let mut elems = Vec::new();
                for _ in 0..*size {
                    let (child_id, child_parameters, child_instructions) = Self::reconstruct_param(None, inner, function);
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
                    let (child_id, child_parameters, child_instructions) = Self::reconstruct_param(None, elem_type, function);
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
            _ => todo!("Not implemented yet")
        }

        return (*value_id, new_parameters, new_instructions);
    }

    /// Constrains return values from unconstrained function calls.
    /// For each unconstrained call, the return values are decomposed to fields,
    /// written to witness, range-checked, and reconstructed.
    fn constrain_unconstrained_returns(ssa: &mut SSA<Empty>) {
        let func_ids: Vec<_> = ssa.get_function_ids().collect();

        for func_id in func_ids {
            // First, collect return types for all unconstrained calls in this function
            let called_fn_returns: HashMap<FunctionId, Vec<Type<Empty>>> = {
                let func = ssa.get_function(func_id);
                let mut map = HashMap::new();
                for (_, block) in func.get_blocks() {
                    for instr in block.get_instructions() {
                        if let OpCode::Call { function: CallTarget::Static(called_fn), is_unconstrained: true, .. } = instr {
                            if !map.contains_key(called_fn) {
                                map.insert(*called_fn, ssa.get_function(*called_fn).get_returns().to_vec());
                            }
                        }
                    }
                }
                map
            };

            if called_fn_returns.is_empty() {
                continue;
            }

            // Now process each block
            let func = ssa.get_function_mut(func_id);
            let block_ids: Vec<_> = func.get_blocks().map(|(id, _)| *id).collect();

            for block_id in block_ids {
                let old_instructions = func.get_block_mut(block_id).take_instructions();
                let mut new_instructions = Vec::new();

                for instr in old_instructions {
                    match instr {
                        OpCode::Call { results, function: CallTarget::Static(called_fn), args, is_unconstrained: true } => {
                            let return_types = called_fn_returns.get(&called_fn).unwrap();

                            // For each return: Field types use original ID directly,
                            // other types get a raw ID that will be constrained.
                            let mut call_results = Vec::new();
                            let mut to_constrain = Vec::new();

                            for (orig_id, typ) in results.iter().zip(return_types.iter()) {
                                if Self::needs_constraining(typ) {
                                    let raw_id = func.fresh_value();
                                    call_results.push(raw_id);
                                    to_constrain.push((*orig_id, raw_id, typ.clone()));
                                } else {
                                    // Field types: use original ID directly, no constraining needed
                                    call_results.push(*orig_id);
                                }
                            }

                            // Emit call
                            new_instructions.push(OpCode::Call {
                                results: call_results,
                                function: CallTarget::Static(called_fn),
                                args,
                                is_unconstrained: true,
                            });

                            // Constrain non-Field return values
                            for (orig_id, raw_id, typ) in to_constrain {
                                let constrain_instrs = Self::constrain_value(
                                    orig_id, raw_id, &typ, func
                                );
                                new_instructions.extend(constrain_instrs);
                            }
                        }
                        other => new_instructions.push(other),
                    }
                }

                func.get_block_mut(block_id).put_instructions(new_instructions);
            }
        }
    }

    /// Returns true if the type needs constraining (rangecheck) for unconstrained returns.
    /// Field types don't need constraining, only integer and composite types do.
    fn needs_constraining(typ: &Type<Empty>) -> bool {
        match &typ.expr {
            TypeExpr::Field => false,
            TypeExpr::U(_) => true,
            TypeExpr::Array(inner, _) => Self::needs_constraining(inner),
            TypeExpr::Tuple(elems) => elems.iter().any(|e| Self::needs_constraining(e)),
            _ => false,
        }
    }

    /// Generates instructions to constrain an unconstrained value.
    /// Decomposes the raw value to fields, range-checks, and reconstructs.
    /// Similar to rebuild_main_params but for call return values.
    /// Note: Field types should not be passed here - use needs_constraining() to filter.
    fn constrain_value(
        final_id: ValueId,
        raw_id: ValueId,
        typ: &Type<Empty>,
        func: &mut Function<Empty>,
    ) -> Vec<OpCode<Empty>> {
        let mut instructions = Vec::new();

        match &typ.expr {
            TypeExpr::Field => {
                // Field values should use original ID directly, not come through here.
                // But if they do (e.g., inside arrays), just use the raw value.
                unreachable!("Field types should not need constraining");
            }
            TypeExpr::U(size) => {
                // Cast to field, rangecheck, cast back (same pattern as rebuild_main_params)
                let as_field = func.fresh_value();

                instructions.push(OpCode::Cast {
                    result: as_field,
                    value: raw_id,
                    target: CastTarget::Field,
                });

                if *size == 1 {
                    // Boolean constraint: x * (x - 1) = 0
                    let zero = func.push_field_const(ark_bn254::Fr::from(0));
                    let one = func.push_field_const(ark_bn254::Fr::from(1));
                    let x_sub_1 = func.fresh_value();
                    let x_times_x_sub_1 = func.fresh_value();
                    instructions.push(OpCode::BinaryArithOp {
                        kind: BinaryArithOpKind::Sub,
                        result: x_sub_1,
                        lhs: as_field,
                        rhs: one,
                    });
                    instructions.push(OpCode::BinaryArithOp {
                        kind: BinaryArithOpKind::Mul,
                        result: x_times_x_sub_1,
                        lhs: as_field,
                        rhs: x_sub_1,
                    });
                    instructions.push(OpCode::AssertEq {
                        lhs: x_times_x_sub_1,
                        rhs: zero,
                    });
                } else {
                    instructions.push(OpCode::Rangecheck {
                        value: as_field,
                        max_bits: *size,
                    });
                }

                instructions.push(OpCode::Cast {
                    result: final_id,
                    value: as_field,
                    target: CastTarget::U(*size),
                });
            }
            TypeExpr::Array(inner, size) => {
                // Decompose, constrain each element, reconstruct
                let mut constrained_elems = Vec::new();

                for i in 0..*size {
                    let index = func.push_u_const(32, i as u128);
                    let raw_elem = func.fresh_value();

                    instructions.push(OpCode::ArrayGet {
                        result: raw_elem,
                        array: raw_id,
                        index,
                    });

                    // Field elements don't need constraining - use raw_elem directly
                    if Self::needs_constraining(inner) {
                        let constrained_elem = func.fresh_value();
                        instructions.extend(
                            Self::constrain_value(constrained_elem, raw_elem, inner, func)
                        );
                        constrained_elems.push(constrained_elem);
                    } else {
                        constrained_elems.push(raw_elem);
                    }
                }

                instructions.push(OpCode::MkSeq {
                    result: final_id,
                    elems: constrained_elems,
                    seq_type: SeqType::Array(*size),
                    elem_type: *inner.clone(),
                });
            }
            TypeExpr::Tuple(element_types) => {
                // Decompose, constrain each element, reconstruct
                let mut constrained_elems = Vec::new();
                let mut elem_types = Vec::new();

                for (i, elem_type) in element_types.iter().enumerate() {
                    let raw_elem = func.fresh_value();

                    instructions.push(OpCode::TupleProj {
                        result: raw_elem,
                        tuple: raw_id,
                        idx: TupleIdx::Static(i),
                    });

                    // Field elements don't need constraining - use raw_elem directly
                    if Self::needs_constraining(elem_type) {
                        let constrained_elem = func.fresh_value();
                        instructions.extend(
                            Self::constrain_value(constrained_elem, raw_elem, elem_type, func)
                        );
                        constrained_elems.push(constrained_elem);
                    } else {
                        constrained_elems.push(raw_elem);
                    }
                    elem_types.push(elem_type.clone());
                }

                instructions.push(OpCode::MkTuple {
                    result: final_id,
                    elems: constrained_elems,
                    element_types: elem_types,
                });
            }
            _ => todo!("constrain_value not implemented for type: {:?}", typ)
        }

        instructions
    }
}
