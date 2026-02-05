use std::collections::HashMap;

use super::FunctionConverter;
use crate::compiler::ir::r#type::Empty;
use crate::compiler::ssa::{Const, FunctionId, GlobalDef, SSA};
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
    ) -> (Vec<GlobalDef>, HashMap<NoirValueId, usize>) {
        let mut res = vec![];
        let mut mapper = HashMap::new();

        let mut work_stack = globals.values.iter().map(|(id, _)| id).collect::<Vec<_>>();

        while let Some(noir_value_id) = work_stack.pop() {
            if mapper.contains_key(&noir_value_id) {
                continue;
            }
            let v = &globals.values[noir_value_id];
            match v {
                Value::NumericConstant { constant, typ } => match typ {
                    NumericType::Unsigned { bit_size: s } => {
                        res.push(GlobalDef::Const(Const::U(
                            *s as usize,
                            constant.to_string().parse::<u128>().unwrap(),
                        )));
                        mapper.insert(noir_value_id, res.len() - 1);
                    }

                    NumericType::NativeField => {
                        let field_value = constant.to_string();
                        let field_value = ark_bn254::Fr::from_str(&field_value).unwrap();
                        res.push(GlobalDef::Const(Const::Field(field_value)));
                        mapper.insert(noir_value_id, res.len() - 1);
                    }
                    _ => {
                        panic!("Unsupported numeric type for global: {:?}", typ);
                    }
                },
                Value::Instruction {
                    instruction,
                    position: _,
                    typ: _,
                } => {
                    let instruction_def = &globals.instructions[*instruction];
                    match instruction_def {
                        Instruction::MakeArray { elements, typ } => {
                            if elements.iter().all(|e| mapper.contains_key(e)) {
                                let items =
                                    elements.iter().map(|e| *mapper.get(e).unwrap()).collect();
                                res.push(GlobalDef::Array(
                                    items,
                                    TypeConverter::new().convert_type(typ),
                                ));
                                mapper.insert(noir_value_id, res.len() - 1);
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

        (res, mapper)
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
                let (globals, global_value_mapper) = self.convert_globals(&globals);
                custom_ssa.set_globals(globals);
                self.global_value_mapper = global_value_mapper;
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
