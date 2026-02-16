use crate::compiler::{
    ir::r#type::{Type, TypeExpr},
    pass_manager::{Pass, PassInfo},
    ssa::{
        BinaryArithOpKind, BlockId, CastTarget, Function, OpCode, SSA, SeqType, TupleIdx, ValueId,
    },
};

pub struct PrepareEntryPoint {}

impl Pass for PrepareEntryPoint {
    fn run(&self, ssa: &mut SSA, _pass_manager: &crate::compiler::pass_manager::PassManager) {
        Self::wrap_main(ssa);
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

    fn wrap_main(ssa: &mut SSA) {
        let original_main_id = ssa.get_main_id();
        let original_main = ssa.get_main();
        let param_types = original_main.get_param_types();
        let return_types = original_main.get_returns().to_vec();

        let globals_init_fn = ssa.get_globals_init_fn();
        let globals_deinit_fn = ssa.get_globals_deinit_fn();

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

        // Call globals init function if present
        if let Some(init_fn) = globals_init_fn {
            wrapper.push_call(entry_block, init_fn, vec![], 0);
        }

        let results = wrapper.push_call(
            entry_block,
            original_main_id,
            arg_values,
            return_types.len(),
        );
        for ((result, public_input), return_type) in results
            .iter()
            .zip(return_input_values.iter())
            .zip(return_types.iter())
        {
            Self::assert_eq_deep(wrapper, entry_block, *result, *public_input, return_type);
        }

        // Call globals deinit function if present
        if let Some(deinit_fn) = globals_deinit_fn {
            wrapper.push_call(entry_block, deinit_fn, vec![], 0);
        }

        wrapper.terminate_block_with_return(entry_block, vec![]);

        ssa.set_entry_point(wrapper_id);
    }

    fn assert_eq_deep(
        wrapper: &mut Function,
        block: BlockId,
        result: ValueId,
        public_input: ValueId,
        typ: &Type,
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
                    let input_elem =
                        wrapper.push_tuple_proj(block, public_input, TupleIdx::Static(i));
                    Self::assert_eq_deep(wrapper, block, result_elem, input_elem, elem_type);
                }
            }
            _ => {
                wrapper.push_assert_eq(block, result, public_input);
            }
        }
    }

    fn rebuild_main_params(&self, ssa: &mut SSA) {
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
        function: &mut Function,
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
                    let zero = function.push_field_const(ark_bn254::Fr::from(0));
                    let one = function.push_field_const(ark_bn254::Fr::from(1));
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
