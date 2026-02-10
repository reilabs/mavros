use core::panic;
use std::collections::HashMap;

use crate::compiler::{analysis::value_definitions::ValueDefinition, ir::r#type::Empty, pass_manager::{DataPoint, Pass}, ssa::{BinaryArithOpKind, Const, OpCode, TupleIdx}};

pub struct MakeStructAccessStatic {}

impl Pass<Empty> for MakeStructAccessStatic {
    fn run(
        &self,
        ssa: &mut crate::compiler::ssa::SSA<Empty>,
        pass_manager: &crate::compiler::pass_manager::PassManager<Empty>,
    ) {
        self.do_run(ssa, pass_manager.get_value_definitions());
    }

    fn pass_info(&self) -> crate::compiler::pass_manager::PassInfo {
        crate::compiler::pass_manager::PassInfo {
            name: "make_struct_access_static",
            needs: vec![DataPoint::ValueDefinitions],
        }
    }

    fn invalidates_cfg(&self) -> bool {
        false
    }
}

impl MakeStructAccessStatic {
    pub fn new() -> Self {
        Self {}
    }

    pub fn do_run(
        &self,
        ssa: &mut crate::compiler::ssa::SSA<Empty>,
        value_definitions: &crate::compiler::analysis::value_definitions::ValueDefinitions<
            Empty,
        >,
    ) {
        for (function_id, function) in ssa.iter_functions_mut() {
            let value_definitions = value_definitions.get_function(*function_id);
            let mut new_blocks = HashMap::new();
            for (bid, mut block) in function.take_blocks().into_iter() {
                let mut new_instructions = Vec::new();
                for instruction in block.take_instructions().into_iter() {
                    match instruction {
                        OpCode::TupleProj {
                            result: item_val_id, 
                            tuple, 
                            ref idx
                        } => { 
                            if let TupleIdx::Dynamic(tuple_field_index_val_id, _tp) = idx {
                                let tuple_field_index_definition = value_definitions.get_definition(*tuple_field_index_val_id);
                                if let ValueDefinition::Instruction(_, _, OpCode::BinaryArithOp { 
                                    kind: BinaryArithOpKind::Sub,
                                    result: _tuple_field_index_val_id, 
                                    lhs: _flat_array_index_value_id, 
                                    rhs: flat_array_tuple_starting_index_value_id, 
                                }) = tuple_field_index_definition {
                                    let tuple_starting_index_definition = value_definitions.get_definition(*flat_array_tuple_starting_index_value_id);
                                    if let ValueDefinition::Instruction(_, _, OpCode::BinaryArithOp {
                                        kind: BinaryArithOpKind::Mul,
                                        result: _,
                                        lhs: tuple_array_index_value_id,
                                        rhs: mul_stride,
                                    }) = tuple_starting_index_definition {
                                        let tuple_array_index_definition = value_definitions.get_definition(*tuple_array_index_value_id);
                                        if let ValueDefinition::Instruction(_, _, OpCode::BinaryArithOp {
                                            kind: BinaryArithOpKind::Div,
                                            result: _,
                                            lhs: flat_array_index_value_id,
                                            rhs: div_stride,
                                        }) = tuple_array_index_definition {
                                            // Verify the mul and div use the same stride value
                                            let mul_stride_def = value_definitions.get_definition(*mul_stride);
                                            let div_stride_def = value_definitions.get_definition(*div_stride);
                                            let strides_match = match (mul_stride_def, div_stride_def) {
                                                (ValueDefinition::Const(mul_const), ValueDefinition::Const(div_const)) => {
                                                    mul_const == div_const
                                                }
                                                _ => false
                                            };
                                            if !strides_match {
                                                new_instructions.push(instruction);
                                                continue;
                                            }
                                            let flat_array_index_definition = value_definitions.get_definition(*flat_array_index_value_id);
                                            match flat_array_index_definition {
                                                ValueDefinition::Instruction(_, _, OpCode::BinaryArithOp { 
                                                    kind, 
                                                    result: _, 
                                                    lhs: _, 
                                                    rhs, 
                                                }) => {
                                                    match kind {
                                                        BinaryArithOpKind::Mul => {
                                                            new_instructions.push(
                                                                OpCode::TupleProj {
                                                                    result: item_val_id, 
                                                                    tuple, 
                                                                    idx: TupleIdx::Static(0),
                                                                }
                                                            );
                                                        }
                                                        BinaryArithOpKind::Add => {
                                                            let tuple_field_index_val_id_og_definition = value_definitions.get_definition(*rhs);
                                                            if let ValueDefinition::Const(Const::U(_sz, val)) = tuple_field_index_val_id_og_definition {
                                                                new_instructions.push(
                                                                OpCode::TupleProj {
                                                                    result: item_val_id, 
                                                                    tuple, 
                                                                    idx: TupleIdx::Static(*val as usize),
                                                                }
                                                            );
                                                            }
                                                        } 
                                                        _ => panic!("Expected Add or Mul operation for flat_array_index_definition")
                                                    }
                                                } 
                                                ValueDefinition::Const(Const::U(_, val)) => {
                                                    new_instructions.push(
                                                        OpCode::TupleProj {
                                                            result: item_val_id, 
                                                            tuple, 
                                                            idx: TupleIdx::Static(*val as usize),
                                                        }
                                                    );
                                                }
                                                _ => {
                                                    panic!("Unexpected flat_array_index_definition")
                                                }
                                            }
                                        } else {
                                            panic!("Expected div instruction for tuple_array_index_definition")
                                        }
                                    } else {
                                        panic!("Expected multiplication instruction for flat_array_tuple_starting_index_value_id");
                                    }
                                } else {
                                    panic!("Expected dynamic tuple index");
                                }
                            } else {
                                new_instructions.push(instruction);
                            }
                        }
                        _ => {
                            new_instructions.push(instruction.clone());
                        }
                    }
                }
                block.put_instructions(new_instructions);
                new_blocks.insert(bid, block);
            }
            function.put_blocks(new_blocks);
        }
    }
}