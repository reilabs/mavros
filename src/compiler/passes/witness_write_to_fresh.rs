use crate::compiler::{
    analysis::types::TypeInfo,
    ir::r#type::{Type, TypeExpr},
    pass_manager::{PassManager, PassInfo, DataPoint, Pass},
    ssa::{OpCode, SeqType, ValueId, Function, SSA},
    taint_analysis::ConstantTaint,
};

pub struct WitnessWriteToFresh {}

impl Pass<ConstantTaint> for WitnessWriteToFresh {
    fn run(
        &self,
        ssa: &mut SSA<ConstantTaint>,
        pass_manager: &PassManager<ConstantTaint>,
    ) {
        self.do_run(ssa, pass_manager.get_type_info());
    }

    fn pass_info(&self) -> PassInfo {
        PassInfo {
            name: "witness_write_to_fresh",
            needs: vec![DataPoint::Types],
        }
    }

    fn invalidates_cfg(&self) -> bool {
        false
    }

}

impl WitnessWriteToFresh {
    pub fn new() -> Self {
        Self {}
    }

    pub fn do_run(
        &self,
        ssa: &mut SSA<ConstantTaint>,
        type_info: &TypeInfo<ConstantTaint>,
    ) {
        let main_id = ssa.get_main_id();
        let main_function = ssa.get_function_mut(main_id);
        let main_block = main_function.get_block_mut(main_function.get_entry_id());
        let old_params = main_block.take_parameters();
        let old_instructions = main_block.take_instructions();
        
        let mut new_instructions = vec![];

        for (r, tp) in old_params.iter() {
            Self::generate_fresh_witness_for_parameter(
                Some(*r),
                tp.clone(),
                main_function,
                &mut new_instructions,
            );
        }

        new_instructions.extend(old_instructions.into_iter());
        let main_function = ssa.get_function_mut(main_id);
        let main_block = main_function.get_block_mut(main_function.get_entry_id());
        main_block.put_instructions(new_instructions);

        for (function_id, function) in ssa.iter_functions_mut() {
            for (_, block) in function.get_blocks_mut() {
                for instruction in block.get_instructions_mut() {
                    let new_instruction = match instruction {
                        OpCode::WriteWitness { result: r, value: _, witness_annotation: _ } => {
                            let tp = type_info
                                .get_function(*function_id)
                                .get_value_type(r.unwrap());
                            if !tp.is_numeric() {
                                panic!("Expected numeric type, got {:?}", tp);
                            }
                            OpCode::FreshWitness {
                                result: r.unwrap(),
                                result_type: tp.clone(),
                            }
                        }
                        OpCode::Cmp { .. }
                        | OpCode::Cast { .. }
                        | OpCode::MkSeq { .. }
                        | OpCode::Alloc { .. }
                        | OpCode::BinaryArithOp { .. }
                        | OpCode::Truncate { .. }
                        | OpCode::Not { .. }
                        | OpCode::Store { .. }
                        | OpCode::Load { .. }
                        | OpCode::AssertEq { .. }
                        | OpCode::AssertR1C { .. }
                        | OpCode::Call { .. }
                        | OpCode::ArrayGet { .. }
                        | OpCode::ArraySet { .. }
                        | OpCode::SlicePush { .. }
                        | OpCode::SliceLen { .. }
                        | OpCode::Select { .. }
                        | OpCode::ToBits { .. }
                        | OpCode::ToRadix { .. }
                        | OpCode::MemOp { .. }
                        | OpCode::FreshWitness { .. }
                        | OpCode::Constrain { .. }
                        | OpCode::NextDCoeff { .. }
                        | OpCode::PureToWitnessRef { .. }
                        | OpCode::UnboxField { .. }
                        | OpCode::MulConst { .. }
                        | OpCode::BumpD { .. }
                        | OpCode::Rangecheck { .. }
                        | OpCode::Lookup { .. }
                        | OpCode::DLookup { .. }
                        | OpCode::ReadGlobal { .. }
                        | OpCode::Todo { .. } 
                        | OpCode::TupleProj { .. } 
                        | OpCode::MkTuple { .. } 
                        => instruction.clone(),
                    };
                    *instruction = new_instruction;
                }
            }
        }
    }

    fn generate_fresh_witness_for_parameter(
        value_id: Option<ValueId>,
        tp: Type<ConstantTaint>,
        main_function: &mut Function<ConstantTaint>,
        instruction_collector: &mut Vec<OpCode<ConstantTaint>>,
    ) -> ValueId {
        let r = value_id.unwrap_or_else(|| main_function.fresh_value());
        match &tp.expr {
            TypeExpr::Field => {
                instruction_collector.push(OpCode::FreshWitness { 
                    result: r,
                    result_type: Type::field(ConstantTaint::Witness),
                })
            }
            TypeExpr::Array(inner_type, size) => {
                let mut value_ids = vec![];
                for _ in 0..*size {
                    let new_value_id = Self::generate_fresh_witness_for_parameter(
                        None,
                        *inner_type.clone(),
                        main_function,
                        instruction_collector,
                    );
                    value_ids.push(new_value_id);
                }
                instruction_collector.push(OpCode::MkSeq {
                    result: r,
                    elems: value_ids,
                    seq_type: SeqType::Array(*size),
                    elem_type: *inner_type.clone(),
                });
            }    
            TypeExpr::Tuple(child_types) => {
                let mut value_ids = vec![];
                for child_type in child_types.iter() {
                    let new_value_id = Self::generate_fresh_witness_for_parameter(
                        None,
                        child_type.clone(),
                        main_function,
                        instruction_collector,
                    );
                    value_ids.push(new_value_id);
                }
                instruction_collector.push(OpCode::MkTuple {
                    result: r,
                    elems: value_ids,
                    element_types: child_types.clone(),
                });
            }    
            _ => panic!("Unsupported parameter type for witness write to fresh. We only support fields and nested arrays of fields for now"), 
        }
        r
    }
}