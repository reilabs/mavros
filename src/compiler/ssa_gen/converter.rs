use std::collections::HashMap;

use super::FunctionConverter;
use crate::compiler::ir::r#type::Empty;
use crate::compiler::ir::r#type::Type;
use crate::compiler::ssa::{Const, Function, FunctionId, SSA};
use crate::compiler::ssa_gen::TypeConverter;
use noirc_evaluator::ssa::ir::dfg::DataFlowGraph;
use noirc_evaluator::ssa::ir::function::FunctionId as NoirFunctionId;
use noirc_evaluator::ssa::ir::instruction::Instruction;
use noirc_evaluator::ssa::ir::types::NumericType;
use noirc_evaluator::ssa::ir::value::{Value, ValueId as NoirValueId};
use noirc_evaluator::ssa::ssa_gen::Ssa as NoirSsa;
use std::str::FromStr;

pub struct SsaConverter {
    function_mapper: HashMap<NoirFunctionId, FunctionId>,
    global_value_mapper: HashMap<NoirValueId, usize>,
}

impl SsaConverter {
    pub fn new() -> Self {
        SsaConverter {
            function_mapper: HashMap::new(),
            global_value_mapper: HashMap::new(),
        }
    }

    pub fn convert_globals(
        &mut self,
        globals: &DataFlowGraph,
        custom_ssa: &mut SSA<Empty>,
    ) -> HashMap<NoirValueId, usize> {
        let mut global_types = vec![];
        let mut mapper = HashMap::new();
        let type_converter = TypeConverter::new();

        let init_fn_id = custom_ssa.add_function("globals_init".to_string());
        let deinit_fn_id = custom_ssa.add_function("globals_deinit".to_string());

        let mut work_stack = globals.values.iter().map(|(id, _)| id).collect::<Vec<_>>();

        // Phase 1: Assign global slot indices and collect types
        while let Some(noir_value_id) = work_stack.pop() {
            if mapper.contains_key(&noir_value_id) {
                continue;
            }
            let v = &globals.values[noir_value_id];
            match v {
                Value::NumericConstant { constant: _, typ } => {
                    let converted_type = match typ {
                        NumericType::Unsigned { bit_size: s } => Type::u(*s as usize, Empty),
                        NumericType::NativeField => Type::field(Empty),
                        _ => panic!("Unsupported numeric type for global: {:?}", typ),
                    };
                    let idx = global_types.len();
                    global_types.push(converted_type);
                    mapper.insert(noir_value_id, idx);
                }
                Value::Instruction {
                    instruction,
                    position: _,
                    typ: _,
                } => {
                    let instruction_def = &globals.instructions[*instruction];
                    match instruction_def {
                        Instruction::MakeArray { elements, typ } => {
                            if elements.iter().all(|e| mapper.contains_key(e)) {
                                let converted_type = type_converter.convert_type(typ);
                                let idx = global_types.len();
                                global_types.push(converted_type);
                                mapper.insert(noir_value_id, idx);
                            } else {
                                work_stack.push(noir_value_id);
                                work_stack
                                    .extend(elements.iter().filter(|e| !mapper.contains_key(e)));
                            }
                        }
                        _ => panic!(
                            "ICE: unexpected instruction in globals: {:?}",
                            instruction_def
                        ),
                    }
                }
                _ => panic!("ICE: unexpected value in globals: {:?}", v),
            }
        }

        // Phase 2: Build init function
        {
            let init_fn = custom_ssa.get_function_mut(init_fn_id);
            let entry = init_fn.get_entry_id();

            // Re-walk globals in dependency order (same order as phase 1 produced them)
            for (noir_value_id, idx) in mapper.iter() {
                let v = &globals.values[*noir_value_id];
                match v {
                    Value::NumericConstant { constant, typ } => {
                        let value = match typ {
                            NumericType::Unsigned { bit_size: s } => {
                                init_fn.push_u_const(*s as usize, constant.to_string().parse::<u128>().unwrap())
                            }
                            NumericType::NativeField => {
                                let field_value = ark_bn254::Fr::from_str(&constant.to_string()).unwrap();
                                init_fn.push_field_const(field_value)
                            }
                            _ => unreachable!(),
                        };
                        init_fn.push_init_global(entry, *idx, value);
                    }
                    Value::Instruction { instruction, .. } => {
                        let instruction_def = &globals.instructions[*instruction];
                        match instruction_def {
                            Instruction::MakeArray { elements, typ } => {
                                let elem_type = type_converter.convert_type(typ);
                                let elems: Vec<_> = elements.iter().map(|e| {
                                    let global_idx = *mapper.get(e).unwrap();
                                    init_fn.push_read_global(entry, global_idx as u64, global_types[global_idx].clone())
                                }).collect();
                                let arr = init_fn.push_mk_array(
                                    entry,
                                    elems,
                                    crate::compiler::ssa::SeqType::Array(elements.len()),
                                    elem_type,
                                );
                                init_fn.push_init_global(entry, *idx, arr);
                            }
                            _ => unreachable!(),
                        }
                    }
                    _ => unreachable!(),
                }
            }
            init_fn.terminate_block_with_return(entry, vec![]);
        }

        // Phase 3: Build deinit function
        {
            let deinit_fn = custom_ssa.get_function_mut(deinit_fn_id);
            let entry = deinit_fn.get_entry_id();
            for (i, typ) in global_types.iter().enumerate() {
                if typ.is_heap_allocated() {
                    deinit_fn.push_drop_global(entry, i);
                }
            }
            deinit_fn.terminate_block_with_return(entry, vec![]);
        }

        custom_ssa.set_global_types(global_types);
        custom_ssa.set_globals_init_fn(init_fn_id);
        custom_ssa.set_globals_deinit_fn(deinit_fn_id);

        mapper
    }

    pub fn convert_noir_ssa(&mut self, noir_ssa: &NoirSsa) -> SSA<Empty> {
        let mut custom_ssa = SSA::new();

        for (noir_function_id, _) in noir_ssa.functions.iter() {
            if *noir_function_id == noir_ssa.main_id {
                self.function_mapper
                    .insert(*noir_function_id, custom_ssa.get_main_id());
            } else {
                self.function_mapper.insert(
                    *noir_function_id,
                    custom_ssa
                        .add_function(noir_ssa.functions[noir_function_id].name().to_string()),
                );
            };
        }

        let mut globals_init = false;

        for (noir_function_id, noir_function) in noir_ssa.functions.iter() {
            if !globals_init {
                let globals = DataFlowGraph::from(noir_function.dfg.globals.as_ref().clone());
                self.global_value_mapper = self.convert_globals(&globals, &mut custom_ssa);
                globals_init = true;
            }

            let mut function_converter = FunctionConverter::new(self.global_value_mapper.clone());
            let custom_function =
                function_converter.convert_function(noir_function, &self.function_mapper);
            let function_id = self.function_mapper.get(noir_function_id).unwrap();
            *custom_ssa.get_function_mut(*function_id) = custom_function;
        }

        custom_ssa
    }
}
