use std::collections::HashMap;

use super::TypeConverter;
use crate::compiler::{
    ir::r#type::{Empty, TypeExpr},
    ssa::{BlockId, CastTarget, Endianness, Function, FunctionId, Radix, SeqType, SliceOpDir, TupleIdx, ValueId},
};
use noirc_evaluator::ssa::ir::{
    basic_block::BasicBlockId,
    function::{Function as NoirFunction, FunctionId as NoirFunctionId},
    instruction::{
        BinaryOp, Endian, Instruction as NoirInstruction, Intrinsic, TerminatorInstruction,
    },
    types::{NumericType, Type as NoirType},
    value::{Value, ValueId as NoirValueId},
};
use tracing::instrument;

use std::str::FromStr;

pub struct FunctionConverter {
    type_converter: TypeConverter,
    value_mapper: HashMap<NoirValueId, ValueId>,
    block_mapper: HashMap<BasicBlockId, BlockId>,
    global_value_mapper: HashMap<NoirValueId, usize>,
}

impl FunctionConverter {
    pub fn new(global_value_mapper: HashMap<NoirValueId, usize>) -> Self {
        FunctionConverter {
            type_converter: TypeConverter::new(),
            value_mapper: HashMap::new(),
            block_mapper: HashMap::new(),
            global_value_mapper,
        }
    }

    #[instrument(skip_all, fields(function = %noir_function.name()))]
    pub fn convert_function(
        &mut self,
        noir_function: &NoirFunction,
        function_mapper: &HashMap<NoirFunctionId, FunctionId>,
    ) -> Function<Empty> {
        let mut custom_function = Function::empty(noir_function.name().to_string());
        let entry_block_id = custom_function.get_entry_id();

        for return_val in noir_function.returns().iter().map(|vs| vs.iter()).flatten() {
            let return_type = &noir_function.dfg.values[*return_val].get_type();
            let converted_return_type = self.type_converter.convert_type(return_type);
            custom_function.add_return_type(converted_return_type);
        }

        for (block_id, _) in noir_function.dfg.basic_blocks_iter() {
            if block_id == noir_function.entry_block {
                self.block_mapper.insert(block_id, entry_block_id);
            } else {
                self.block_mapper
                    .insert(block_id, custom_function.add_block());
            }
        }

        for (block_id, block) in noir_function.dfg.basic_blocks_iter() {
            let custom_block_id = *self.block_mapper.get(&block_id).unwrap();

            for noir_param_id in block.parameters() {
                let param_value = &noir_function.dfg.values[*noir_param_id];
                let param_type = param_value.get_type();
                let converted_type = self.type_converter.convert_type(&param_type);
                let param_id = custom_function.add_parameter(custom_block_id, converted_type);
                self.value_mapper.insert(*noir_param_id, param_id);
            }
            for noir_instruction_id in block.instructions() {
                let noir_instruction = &noir_function.dfg.instructions[*noir_instruction_id];
                match noir_instruction {
                    NoirInstruction::Binary(binary) => {
                        let left_id = binary.lhs;
                        let right_id = binary.rhs;
                        let result_id =
                            noir_function.dfg.instruction_results(*noir_instruction_id)[0];
                        let left_value = &noir_function.dfg.values[left_id];
                        let right_value = &noir_function.dfg.values[right_id];
                        let left_value = self.convert_value(
                            &mut custom_function,
                            custom_block_id,
                            left_id,
                            left_value,
                        );
                        let right_value = self.convert_value(
                            &mut custom_function,
                            custom_block_id,
                            right_id,
                            right_value,
                        );
                        let rr_id = match binary.operator {
                            BinaryOp::Add { unchecked: _ } => {
                                custom_function.push_add(custom_block_id, left_value, right_value)
                            }
                            BinaryOp::Lt => {
                                custom_function.push_lt(custom_block_id, left_value, right_value)
                            }
                            BinaryOp::Mul { unchecked: _ } => {
                                custom_function.push_mul(custom_block_id, left_value, right_value)
                            }
                            BinaryOp::Eq => {
                                custom_function.push_eq(custom_block_id, left_value, right_value)
                            }
                            BinaryOp::Div => {
                                custom_function.push_div(custom_block_id, left_value, right_value)
                            }
                            BinaryOp::Sub { unchecked: _ } => {
                                custom_function.push_sub(custom_block_id, left_value, right_value)
                            }
                            BinaryOp::And => {
                                custom_function.push_and(custom_block_id, left_value, right_value)
                            }
                            _ => panic!("Unsupported binary operation: {:?}", binary.operator),
                        };
                        self.value_mapper.insert(result_id, rr_id);
                    }
                    NoirInstruction::Call { func, arguments } => {
                        let function_value = &noir_function.dfg.values[*func];

                        if let Value::Function(func) = function_value {
                            let result_ids =
                                noir_function.dfg.instruction_results(*noir_instruction_id);

                            let mut converted_args = Vec::new();
                            for arg_id in arguments {
                                let arg_value = &noir_function.dfg.values[*arg_id];
                                let converted_arg = self.convert_value(
                                    &mut custom_function,
                                    custom_block_id,
                                    *arg_id,
                                    arg_value,
                                );
                                converted_args.push(converted_arg);
                            }

                            let function_id = function_mapper.get(func).unwrap();

                            let return_values = custom_function.push_call(
                                custom_block_id,
                                *function_id,
                                converted_args,
                                result_ids.len(),
                            );

                            for (result_id, return_value) in
                                result_ids.iter().zip(return_values.iter())
                            {
                                self.value_mapper.insert(*result_id, *return_value);
                            }
                        } else if let Value::Intrinsic(intrinsic) = function_value {
                            match intrinsic {
                                Intrinsic::AsWitness { .. } => {} // ignore this call, it's a compiler hint and taking hints is a sign of weakness
                                Intrinsic::ToBits(endianness) => {
                                    let result_id = noir_function
                                        .dfg
                                        .instruction_results(*noir_instruction_id)[0];
                                    let result_value = &noir_function.dfg.values[result_id];
                                    let result_type = result_value.get_type();
                                    let result_size = match result_type.as_ref() {
                                        NoirType::Array(_, size) => *size,
                                        _ => panic!(
                                            "ToBits result should be an array, got: {:?}",
                                            result_type
                                        ),
                                    };
                                    let input_id = arguments[0];
                                    let input_value = &noir_function.dfg.values[input_id];
                                    let input_converted = self.convert_value(
                                        &mut custom_function,
                                        custom_block_id,
                                        input_id,
                                        input_value,
                                    );
                                    let endianness = match endianness {
                                        Endian::Big => Endianness::Big,
                                        Endian::Little => Endianness::Little,
                                    };
                                    let to_bits_result = custom_function.push_to_bits(
                                        custom_block_id,
                                        input_converted,
                                        endianness,
                                        result_size as usize,
                                    );
                                    self.value_mapper.insert(result_id, to_bits_result);
                                }
                                Intrinsic::ToRadix(endianness) => {
                                    let result_id = noir_function
                                        .dfg
                                        .instruction_results(*noir_instruction_id)[0];
                                    let result_value = &noir_function.dfg.values[result_id];
                                    let result_type = result_value.get_type();
                                    let result_size = match result_type.as_ref() {
                                        NoirType::Array(_, size) => *size,
                                        _ => panic!(
                                            "ToRadix result should be an array, got: {:?}",
                                            result_type
                                        ),
                                    };
                                    let input_id = arguments[0];
                                    let radix_id = arguments[1];
                                    let input_value = &noir_function.dfg.values[input_id];
                                    let input_converted = self.convert_value(
                                        &mut custom_function,
                                        custom_block_id,
                                        input_id,
                                        input_value,
                                    );
                                    let radix_value = &noir_function.dfg.values[radix_id];
                                    let radix_converted = self.convert_value(
                                        &mut custom_function,
                                        custom_block_id,
                                        radix_id,
                                        radix_value,
                                    );
                                    let endianness = match endianness {
                                        Endian::Big => Endianness::Big,
                                        Endian::Little => Endianness::Little,
                                    };
                                    let to_radix_result = custom_function.push_to_radix(
                                        custom_block_id,
                                        input_converted,
                                        Radix::Dyn(radix_converted),
                                        endianness,
                                        result_size as usize,
                                    );
                                    self.value_mapper.insert(result_id, to_radix_result);
                                }
                                Intrinsic::StaticAssert => {
                                    let to_assert = arguments[0];
                                    let to_assert_value = &noir_function.dfg.values[to_assert];
                                    let to_assert_converted = self.convert_value(
                                        &mut custom_function,
                                        custom_block_id,
                                        to_assert,
                                        to_assert_value,
                                    );
                                    let t = custom_function.push_u_const(1, 1);
                                    custom_function.push_assert_eq(custom_block_id, to_assert_converted, t);
                                }
                                Intrinsic::SlicePushBack => {
                                    let result_ids = noir_function.dfg.instruction_results(*noir_instruction_id);
                                    if result_ids.len() != 2 {
                                        panic!("SlicePushBack should return 2 values (length, slice), got {}", result_ids.len());
                                    }
                                    let length_result_id = result_ids[0];
                                    let slice_result_id = result_ids[1];
                                    
                                    // Arguments: [slice_length, slice_contents, ...elements_to_push]
                                    // We drop slice_length (arguments[0])
                                    if arguments.len() < 2 {
                                        panic!("SlicePushBack requires at least 2 arguments (slice_length, slice_contents)");
                                    }
                                    
                                    let slice_id = arguments[1];
                                    let slice_value = &noir_function.dfg.values[slice_id];
                                    let slice_converted = self.convert_value(
                                        &mut custom_function,
                                        custom_block_id,
                                        slice_id,
                                        slice_value,
                                    );
                                    
                                    // Convert all elements to push (arguments[2..])
                                    let values_to_push: Vec<ValueId> = arguments[2..]
                                        .iter()
                                        .map(|arg_id| {
                                            let arg_value = &noir_function.dfg.values[*arg_id];
                                            self.convert_value(
                                                &mut custom_function,
                                                custom_block_id,
                                                *arg_id,
                                                arg_value,
                                            )
                                        })
                                        .collect();
                                    
                                    // Create the SlicePush instruction to get the new slice
                                    let slice_push_result = custom_function.push_slice_push(
                                        custom_block_id,
                                        slice_converted,
                                        values_to_push,
                                        SliceOpDir::Back,
                                    );
                                    
                                    // Create SliceLen instruction on the new slice to get the length
                                    let slice_len_result = custom_function.push_slice_len(
                                        custom_block_id,
                                        slice_push_result,
                                    );
                                    
                                    // Map results: first is length, second is slice
                                    self.value_mapper.insert(length_result_id, slice_len_result);
                                    self.value_mapper.insert(slice_result_id, slice_push_result);
                                }
                                Intrinsic::SlicePushFront => {
                                    let result_ids = noir_function.dfg.instruction_results(*noir_instruction_id);
                                    if result_ids.len() != 2 {
                                        panic!("SlicePushFront should return 2 values (length, slice), got {}", result_ids.len());
                                    }
                                    let length_result_id = result_ids[0];
                                    let slice_result_id = result_ids[1];
                                    
                                    // Arguments: [slice_length, slice_contents, ...elements_to_push]
                                    // We drop slice_length (arguments[0])
                                    if arguments.len() < 2 {
                                        panic!("SlicePushFront requires at least 2 arguments (slice_length, slice_contents)");
                                    }
                                    
                                    let slice_id = arguments[1];
                                    let slice_value = &noir_function.dfg.values[slice_id];
                                    let slice_converted = self.convert_value(
                                        &mut custom_function,
                                        custom_block_id,
                                        slice_id,
                                        slice_value,
                                    );
                                    
                                    // Convert all elements to push (arguments[2..])
                                    let values_to_push: Vec<ValueId> = arguments[2..]
                                        .iter()
                                        .map(|arg_id| {
                                            let arg_value = &noir_function.dfg.values[*arg_id];
                                            self.convert_value(
                                                &mut custom_function,
                                                custom_block_id,
                                                *arg_id,
                                                arg_value,
                                            )
                                        })
                                        .collect();
                                    
                                    // Create the SlicePush instruction to get the new slice
                                    let slice_push_result = custom_function.push_slice_push(
                                        custom_block_id,
                                        slice_converted,
                                        values_to_push,
                                        SliceOpDir::Front,
                                    );
                                    
                                    // Create SliceLen instruction on the new slice to get the length
                                    let slice_len_result = custom_function.push_slice_len(
                                        custom_block_id,
                                        slice_push_result,
                                    );
                                    
                                    // Map results: first is length, second is slice
                                    self.value_mapper.insert(length_result_id, slice_len_result);
                                    self.value_mapper.insert(slice_result_id, slice_push_result);
                                }
                                _ => panic!("Unsupported intrinsic: {:?}", intrinsic),
                            }
                        } else {
                            panic!(
                                "Call instruction with non-constant function value not supported: {:?}",
                                function_value
                            );
                        }
                    }

                    NoirInstruction::Not(value) => {
                        let value_value = &noir_function.dfg.values[*value];
                        let value_converted = self.convert_value(
                            &mut custom_function,
                            custom_block_id,
                            *value,
                            value_value,
                        );
                        let result_id =
                            noir_function.dfg.instruction_results(*noir_instruction_id)[0];
                        let not_result = custom_function.push_not(custom_block_id, value_converted);
                        self.value_mapper.insert(result_id, not_result);
                    }
                    NoirInstruction::Cast(input, target) => {
                        let input_value = &noir_function.dfg.values[*input];
                        let input_converted = self.convert_value(
                            &mut custom_function,
                            custom_block_id,
                            *input,
                            input_value,
                        );
                        let target = match target {
                            NumericType::Signed { .. } => todo!(),
                            NumericType::Unsigned { bit_size } => CastTarget::U(*bit_size as usize),
                            NumericType::NativeField => CastTarget::Field,
                        };
                        let result_id =
                            noir_function.dfg.instruction_results(*noir_instruction_id)[0];
                        let cast_result =
                            custom_function.push_cast(custom_block_id, input_converted, target);
                        self.value_mapper.insert(result_id, cast_result);
                    }
                    NoirInstruction::Truncate {
                        value,
                        bit_size,
                        max_bit_size,
                    } => {
                        let value_value = &noir_function.dfg.values[*value];
                        let value_converted = self.convert_value(
                            &mut custom_function,
                            custom_block_id,
                            *value,
                            value_value,
                        );
                        let result_id =
                            noir_function.dfg.instruction_results(*noir_instruction_id)[0];
                        let truncate_result = custom_function.push_truncate(
                            custom_block_id,
                            value_converted,
                            *bit_size as usize,
                            *max_bit_size as usize,
                        );
                        self.value_mapper.insert(result_id, truncate_result);
                    }
                    NoirInstruction::Constrain(l, r, _) => {
                        let left_value = &noir_function.dfg.values[*l];
                        let right_value = &noir_function.dfg.values[*r];
                        let left_converted = self.convert_value(
                            &mut custom_function,
                            custom_block_id,
                            *l,
                            left_value,
                        );
                        let right_converted = self.convert_value(
                            &mut custom_function,
                            custom_block_id,
                            *r,
                            right_value,
                        );

                        custom_function.push_assert_eq(
                            custom_block_id,
                            left_converted,
                            right_converted,
                        );
                    }
                    NoirInstruction::Allocate => {
                        let result_id =
                            noir_function.dfg.instruction_results(*noir_instruction_id)[0];
                        let result_value = &noir_function.dfg.values[result_id];
                        let result_type = result_value.get_type();

                        let converted_type = match &*result_type {
                            noirc_evaluator::ssa::ir::types::Type::Reference(inner) => {
                                self.type_converter.convert_type(&inner)
                            }
                            _ => panic!(
                                "Allocate instruction result should be a reference type, got: {:?}",
                                result_type
                            ),
                        };

                        let alloc_result =
                            custom_function.push_alloc(custom_block_id, converted_type, Empty);

                        self.value_mapper.insert(result_id, alloc_result);
                    }
                    NoirInstruction::Store { address, value } => {
                        let address_value = &noir_function.dfg.values[*address];
                        let value_value = &noir_function.dfg.values[*value];
                        let address_converted = self.convert_value(
                            &mut custom_function,
                            custom_block_id,
                            *address,
                            address_value,
                        );
                        let value_converted = self.convert_value(
                            &mut custom_function,
                            custom_block_id,
                            *value,
                            value_value,
                        );

                        custom_function.push_store(
                            custom_block_id,
                            address_converted,
                            value_converted,
                        );
                    }

                    NoirInstruction::MakeArray { elements, typ } => {
                        let elements_converted: Vec<ValueId> = elements
                            .iter()
                            .map(|e| {
                                self.convert_value(
                                    &mut custom_function,
                                    custom_block_id,
                                    *e,
                                    &noir_function.dfg.values[*e],
                                )
                            })
                            .collect();
                        let result_id =
                            noir_function.dfg.instruction_results(*noir_instruction_id)[0];
                        let converted_typ = self.type_converter.convert_type(typ);
                        let (stp, tp) = match converted_typ.expr {
                            TypeExpr::Array(inner, size) => (SeqType::Array(size), inner),
                            TypeExpr::Slice(inner) => (SeqType::Slice, inner),
                            _ => panic!("Unexpected type in array conversion: {:?}", converted_typ),
                        };

                        // If the element type is a tuple, group flat elements into MkTuple instructions
                        let final_elements = match &tp.expr {
                            TypeExpr::Tuple(field_types) => {
                                let chunk_size = field_types.len();
                                elements_converted
                                    .chunks(chunk_size)
                                    .map(|chunk| {
                                        custom_function.push_mk_tuple(
                                            custom_block_id,
                                            chunk.to_vec(),
                                            field_types.clone(),
                                        )
                                    })
                                    .collect()
                            }
                            _ => elements_converted,
                        };

                        let result_converted = custom_function.push_mk_array(
                            custom_block_id,
                            final_elements,
                            stp,
                            *tp,
                        );
                        self.value_mapper.insert(result_id, result_converted);
                    }
                    NoirInstruction::ArrayGet { array, index, .. } => {
                        let array_value = &noir_function.dfg.values[*array];
                        let index_value = &noir_function.dfg.values[*index];
                        let array_converted = self.convert_value(
                            &mut custom_function,
                            custom_block_id,
                            *array,
                            array_value,
                        );
                        let index_converted = self.convert_value(
                            &mut custom_function,
                            custom_block_id,
                            *index,
                            index_value,
                        );

                        let result_id =
                            noir_function.dfg.instruction_results(*noir_instruction_id)[0];

                        let array_type = array_value.get_type();
                        let converted_array_type = self.type_converter.convert_type(&array_type);
                        
                        match converted_array_type.expr {
                            TypeExpr::Array(array_inner_type, _) | TypeExpr::Slice(array_inner_type) => {
                                match array_inner_type.expr {
                                    TypeExpr::Tuple(_) => {
                                        let tuple_size = array_inner_type.calculate_type_size();
                                        let stride = custom_function.push_u_const(32, tuple_size as u128);

                                        if let Value::NumericConstant { constant, typ: NumericType::Unsigned { bit_size: s } } = index_value {
                                            let tuple_index = custom_function
                                                .push_u_const(*s as usize, constant.to_string().parse::<u128>().unwrap() / (tuple_size as u128));

                                            let tuple_id = custom_function.push_array_get(
                                                custom_block_id,
                                                array_converted,
                                                tuple_index,
                                            );

                                            let tuple_get_result = custom_function.push_tuple_proj(
                                                custom_block_id,
                                                tuple_id,
                                                TupleIdx::Static((constant.to_string().parse::<u128>().unwrap() % (tuple_size as u128)) as usize ),
                                            );

                                            self.value_mapper.insert(result_id, tuple_get_result);
                                        } else {
                                            let tuple_index = custom_function.push_div(
                                                custom_block_id,
                                                index_converted,
                                                stride,
                                            );

                                            let tuple_id = custom_function.push_array_get(
                                                custom_block_id,
                                                array_converted,
                                                tuple_index,
                                            );

                                            let tuple_starting_address = custom_function.push_mul(
                                                custom_block_id,
                                                tuple_index,
                                                stride,
                                            );

                                            let tuple_element_index = custom_function.push_sub(
                                                custom_block_id,
                                                index_converted,
                                                tuple_starting_address,
                                            );

                                            let result_type = noir_function.dfg.values[result_id].get_type();
                                            let converted_result_type = self.type_converter.convert_type(&result_type);
                                            let tuple_get_result = custom_function.push_tuple_proj(
                                                custom_block_id,
                                                tuple_id,
                                                TupleIdx::Dynamic(tuple_element_index, converted_result_type),
                                            );

                                            self.value_mapper.insert(result_id, tuple_get_result);
                                        }
                                    }
                                    _ => {
                                        let array_get_result = custom_function.push_array_get(
                                            custom_block_id,
                                            array_converted,
                                            index_converted,
                                        );
                                        self.value_mapper.insert(result_id, array_get_result);
                                    }
                                }
                            }
                            _ => panic!("ICE: Unexpected type of indexed collection")
                        }

                    }
                    NoirInstruction::ArraySet {
                        array,
                        index,
                        value,
                        ..
                    } => {
                        let array_value = &noir_function.dfg.values[*array];
                        let index_value = &noir_function.dfg.values[*index];
                        let value_value = &noir_function.dfg.values[*value];
                        let array_converted = self.convert_value(
                            &mut custom_function,
                            custom_block_id,
                            *array,
                            array_value,
                        );
                        let index_converted = self.convert_value(
                            &mut custom_function,
                            custom_block_id,
                            *index,
                            index_value,
                        );
                        let value_converted = self.convert_value(
                            &mut custom_function,
                            custom_block_id,
                            *value,
                            value_value,
                        );

                        let result_id =
                            noir_function.dfg.instruction_results(*noir_instruction_id)[0];

                        let array_set_result = custom_function.push_array_set(
                            custom_block_id,
                            array_converted,
                            index_converted,
                            value_converted,
                        );

                        self.value_mapper.insert(result_id, array_set_result);
                    }
                    NoirInstruction::Load { address } => {
                        let address_value = &noir_function.dfg.values[*address];
                        let address_converted = self.convert_value(
                            &mut custom_function,
                            custom_block_id,
                            *address,
                            address_value,
                        );

                        let result_id =
                            noir_function.dfg.instruction_results(*noir_instruction_id)[0];

                        let load_result =
                            custom_function.push_load(custom_block_id, address_converted);

                        self.value_mapper.insert(result_id, load_result);
                    }
                    NoirInstruction::RangeCheck {
                        value,
                        max_bit_size,
                        assert_message: _,
                    } => {
                        let value_value = &noir_function.dfg.values[*value];
                        let value_converted = self.convert_value(
                            &mut custom_function,
                            custom_block_id,
                            *value,
                            value_value,
                        );

                        custom_function.push_rangecheck(
                            custom_block_id,
                            value_converted,
                            *max_bit_size as usize,
                        );
                    }
                    NoirInstruction::IncrementRc { .. } => {
                        // Ignored, we do our own memory management
                    }
                    _ => panic!("Unsupported instruction: {:?}", noir_instruction),
                }
            }

            match block.terminator() {
                Some(TerminatorInstruction::JmpIf {
                    condition,
                    then_destination,
                    else_destination,
                    ..
                }) => {
                    let condition_converted = self.convert_value(
                        &mut custom_function,
                        custom_block_id,
                        *condition,
                        &noir_function.dfg.values[*condition],
                    );
                    let then_destination_converted =
                        *self.block_mapper.get(then_destination).unwrap();
                    let else_destination_converted =
                        *self.block_mapper.get(else_destination).unwrap();
                    custom_function.terminate_block_with_jmp_if(
                        custom_block_id,
                        condition_converted,
                        then_destination_converted,
                        else_destination_converted,
                    );
                }
                Some(TerminatorInstruction::Jmp {
                    destination,
                    arguments,
                    ..
                }) => {
                    let destination_converted = *self.block_mapper.get(destination).unwrap();
                    let arguments_converted = arguments
                        .iter()
                        .map(|id| {
                            self.convert_value(
                                &mut custom_function,
                                custom_block_id,
                                *id,
                                &noir_function.dfg.values[*id],
                            )
                        })
                        .collect();
                    custom_function.terminate_block_with_jmp(
                        custom_block_id,
                        destination_converted,
                        arguments_converted,
                    );
                }
                Some(TerminatorInstruction::Return { return_values, .. }) => {
                    let return_values_converted = return_values
                        .iter()
                        .map(|id| {
                            self.convert_value(
                                &mut custom_function,
                                custom_block_id,
                                *id,
                                &noir_function.dfg.values[*id],
                            )
                        })
                        .collect();
                    custom_function
                        .terminate_block_with_return(custom_block_id, return_values_converted);
                }
                _ => panic!("Unsupported terminator: {:?}", block.terminator()),
            }
        }

        custom_function
    }

    fn convert_value(
        &mut self,
        custom_function: &mut Function<Empty>,
        block: BlockId,
        noir_value_id: NoirValueId,
        noir_value: &Value,
    ) -> ValueId {
        match noir_value {
            Value::NumericConstant { constant, typ } => match typ {
                NumericType::Unsigned { bit_size: s } => {
                    let custom_value_id = custom_function
                        .push_u_const(*s as usize, constant.to_string().parse::<u128>().unwrap());
                    self.value_mapper.insert(noir_value_id, custom_value_id);
                    custom_value_id
                }

                NumericType::NativeField => {
                    let field_value = constant.to_string();
                    let field_value = ark_bn254::Fr::from_str(&field_value).unwrap();
                    let custom_value_id = custom_function.push_field_const(field_value);
                    self.value_mapper.insert(noir_value_id, custom_value_id);
                    custom_value_id
                }
                _ => {
                    panic!("Unsupported numeric type: {:?}", typ);
                }
            },
            Value::Global(global) => custom_function.push_read_global(
                block,
                *self.global_value_mapper.get(&noir_value_id).unwrap() as u64,
                self.type_converter.convert_type(global),
            ),
            _ => *self.value_mapper.get(&noir_value_id).expect(&format!(
                "Value not found: {:?} {:?} {:?}",
                block, noir_value_id, noir_value
            )),
        }
    }
}
