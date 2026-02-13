use core::panic;
use std::collections::HashMap;

use tracing::{Level, instrument};

use crate::compiler::{
    flow_analysis::{CFG, FlowAnalysis},
    ir::r#type::{Type, TypeExpr},
    ssa::{CallTarget, CastTarget, Const, Function, FunctionId, OpCode, SSA, TupleIdx, ValueId},
};

pub struct TypeInfo {
    functions: HashMap<FunctionId, FunctionTypeInfo>,
}

impl TypeInfo {
    pub fn get_function(&self, function_id: FunctionId) -> &FunctionTypeInfo {
        self.functions.get(&function_id).unwrap()
    }

    pub fn has_function(&self, function_id: FunctionId) -> bool {
        self.functions.contains_key(&function_id)
    }
}

pub struct FunctionTypeInfo {
    values: HashMap<ValueId, Type>, 
}

impl FunctionTypeInfo {
    pub fn get_value_type(&self, value_id: ValueId) -> &Type {
        self.values.get(&value_id).unwrap()
    }
}

pub struct Types {}

impl Types {
    pub fn new() -> Self {
        Types {}
    }

    pub fn run(
        &self,
        ssa: &SSA,
        cfg: &FlowAnalysis,
    ) -> TypeInfo {
        let mut type_info = TypeInfo {
            functions: HashMap::new(),
        };

        let function_types = ssa
            .iter_functions()
            .map(|(id, func)| (*id, (func.get_param_types(), func.get_returns())))
            .collect::<HashMap<_, _>>();

        for (function_id, function) in ssa.iter_functions() {
            let cfg = cfg.get_function_cfg(*function_id);
            let function_info = self.run_function(function, &function_types, cfg);
            type_info.functions.insert(*function_id, function_info);
        }
        type_info
    }

    #[instrument(skip_all, level = Level::DEBUG, name = "Types::run_function", fields(function = function.get_name()))]
    fn run_function(
        &self,
        function: &Function,
        function_types: &HashMap<FunctionId, (Vec<Type>, &[Type])>,
        cfg: &CFG,
    ) -> FunctionTypeInfo {
        let mut function_info = FunctionTypeInfo {
            values: HashMap::new(),
        };

        for (value_id, const_) in function.iter_consts() {
            match const_ {
                Const::U(size, _) => {
                    function_info.values.insert(*value_id, Type::u(*size));
                }
                Const::Field(_) => {
                    function_info.values.insert(*value_id, Type::field());
                }
                Const::Witness(_) => {
                    function_info.values.insert(*value_id, Type::witness_of(Type::field()));
                }
                Const::FnPtr(_) => {
                    function_info.values.insert(*value_id, Type::function());
                }
            }
        }

        for block_id in cfg.get_domination_pre_order() {
            let block = function.get_block(block_id);

            for param in block.get_parameters() {
                function_info.values.insert(param.0, param.1.clone());
            }

            for instruction in block.get_instructions() {
                self.run_opcode(instruction, &mut function_info, function_types)
                    .expect(&format!("Error running opcode {:?}", instruction));
            }
        }

        function_info
    }

    fn run_opcode(
        &self,
        opcode: &OpCode,
        function_info: &mut FunctionTypeInfo,
        function_types: &HashMap<FunctionId, (Vec<Type>, &[Type])>,
    ) -> Result<(), String> {
        match opcode {
            OpCode::Cmp { kind: _kind, result, lhs, rhs } => {
                let lhs_type = function_info.values.get(lhs).ok_or_else(|| {
                    format!(
                        "Left-hand side value {:?} not found in type assignments",
                        lhs
                    )
                })?;
                let rhs_type = function_info.values.get(rhs).ok_or_else(|| {
                    format!(
                        "Right-hand side value {:?} not found in type assignments",
                        rhs
                    )
                })?;
                let result_type = if lhs_type.is_witness_of() || rhs_type.is_witness_of() {
                    Type::witness_of(Type::u(1))
                } else {
                    Type::u(1)
                };
                function_info.values.insert(*result, result_type);
                Ok(())
            }
            OpCode::BinaryArithOp { kind: _kind, result, lhs, rhs } => {
                let lhs_type = function_info.values.get(lhs).ok_or_else(|| {
                    format!(
                        "Left-hand side value {:?} not found in type assignments",
                        lhs
                    )
                })?;
                let rhs_type = function_info.values.get(rhs).ok_or_else(|| {
                    format!(
                        "Right-hand side value {:?} not found in type assignments",
                        rhs
                    )
                })?;
                function_info
                    .values
                    .insert(*result, lhs_type.get_arithmetic_result_type(rhs_type));
                Ok(())
            }
            OpCode::Alloc { result, elem_type: typ } => {
                function_info
                    .values
                    .insert(*result, typ.clone().ref_of());
                Ok(())
            }
            OpCode::Store { ptr: _, value: _ } => Ok(()),
            OpCode::Load { result, ptr } => {
                let ptr_type = function_info.values.get(ptr).ok_or_else(|| {
                    format!("Pointer value {:?} not found in type assignments", ptr)
                })?;
                if !ptr_type.is_ref() {
                    return Err(format!(
                        "Load operation expects a reference type, got {}",
                        ptr_type
                    ));
                }
                function_info.values.insert(
                    *result,
                    ptr_type.get_refered().clone(),
                );
                Ok(())
            }
            OpCode::MemOp { kind: _, value: _ } => Ok(()),
            OpCode::AssertEq { lhs: _, rhs: _ } => Ok(()),
            OpCode::AssertR1C { a: _, b: _, c: _ } => Ok(()),
            OpCode::Call { results: result, function, args } => {
                match function {
                    CallTarget::Static(fn_id) => {
                        let (param_types, return_types) = function_types
                            .get(fn_id)
                            .ok_or_else(|| format!("Function {:?} not found", fn_id))?;

                        if args.len() != param_types.len() {
                            return Err(format!(
                                "Function {:?} expects {} arguments, got {}",
                                fn_id,
                                param_types.len(),
                                args.len()
                            ));
                        }

                        if result.len() != return_types.len() {
                            return Err(format!(
                                "Function {:?} expects {} return values, got {}",
                                fn_id,
                                return_types.len(),
                                result.len()
                            ));
                        }

                        for (ret, ret_type) in result.iter().zip(return_types.iter()) {
                            function_info.values.insert(*ret, ret_type.clone());
                        }
                        Ok(())
                    }
                    CallTarget::Dynamic(_) => {
                        panic!("Dynamic calls should be eliminated by defunctionalization before type analysis");
                    }
                }
            }
            OpCode::ArrayGet { result, array, index: _ } => {
                let array_type = function_info.values.get(array).ok_or_else(|| {
                    format!("Array value {:?} not found in type assignments", array)
                })?;

                let element_type = array_type.get_array_element();
                function_info.values.insert(
                    *result,
                    element_type,
                );
                Ok(())
            }
            OpCode::ArraySet { result, array, index: _, value: _ } => {
                let array_type = function_info.values.get(array).ok_or_else(|| {
                    format!("Array value {:?} not found in type assignments", array)
                })?;

                function_info.values.insert(*result, array_type.clone());
                Ok(())
            }
            OpCode::SlicePush { result, slice, values: _, dir: _ } => {
                let slice_type = function_info.values.get(slice).ok_or_else(|| {
                    format!("Slice value {:?} not found in type assignments", slice)
                })?;

                function_info.values.insert(*result, slice_type.clone());
                Ok(())
            }
            OpCode::SliceLen { result, slice: _ } => {
                // Result is always u32
                function_info.values.insert(
                    *result,
                    Type::u(32),
                );
                Ok(())
            }
            OpCode::Select { result, cond, if_t: then, if_f: otherwise } => {
                let cond_type = function_info.values.get(cond).ok_or_else(|| {
                    format!("Cond value {:?} not found in type assignments", cond)
                })?;
                let then_type = function_info.values.get(then).ok_or_else(|| {
                    format!("Then value {:?} not found in type assignments", then)
                })?;
                let otherwise_type = function_info.values.get(otherwise).ok_or_else(|| {
                    format!(
                        "Otherwise value {:?} not found in type assignments",
                        otherwise
                    )
                })?;
                // Alternatives must match (after potential WitnessCastInsertion).
                // The matched alternative type comes from unifying the two branches.
                let alt_type = then_type.get_arithmetic_result_type(otherwise_type);
                // If cond is WitnessOf and alternatives are not already WitnessOf,
                // the result is WitnessOf(alt_type). Otherwise result = alt_type.
                let result_type = if cond_type.is_witness_of() && !alt_type.is_witness_of() {
                    Type::witness_of(alt_type)
                } else {
                    alt_type
                };
                function_info.values.insert(*result, result_type);
                Ok(())
            }
            OpCode::WriteWitness { result, value } => {
                let Some(result) = result else {
                    return Ok(());
                };
                let witness_type = function_info.values.get(value).ok_or_else(|| {
                    format!("Witness value {:?} not found in type assignments", value)
                })?;
                function_info.values.insert(
                    *result,
                    Type::witness_of(witness_type.clone()),
                );
                Ok(())
            }
            OpCode::FreshWitness { result: r, result_type: tp } => {
                function_info.values.insert(*r, Type::witness_of(tp.clone()));
                Ok(())
            }
            OpCode::Constrain { a: _, b: _, c: _ } => Ok(()),
            OpCode::NextDCoeff { result: v } => {
                function_info.values.insert(*v, Type::field());
                Ok(())
            }
            OpCode::BumpD { matrix: _, variable: _, sensitivity: _ } => Ok(()),
            OpCode::MkSeq { result: r, elems: _, seq_type: top_tp, elem_type: t } => {
                function_info.values.insert(*r, top_tp.of(t.clone()));
                Ok(())
            }
            OpCode::Cast { result, value, target } => {
                let value_type = function_info
                    .values
                    .get(value)
                    .ok_or_else(|| format!("Value {:?} not found in type assignments", value))?;

                let result_type = match target {
                    CastTarget::Field => {
                        if value_type.is_witness_of() {
                            Type::witness_of(Type::field())
                        } else {
                            Type::field()
                        }
                    }
                    CastTarget::U(size) => {
                        if value_type.is_witness_of() {
                            Type::witness_of(Type::u(*size))
                        } else {
                            Type::u(*size)
                        }
                    }
                    CastTarget::Nop => value_type.clone(),
                    CastTarget::ArrayToSlice => {
                        match &value_type.expr {
                            crate::compiler::ir::r#type::TypeExpr::Array(elem, _len) => {
                                elem.as_ref().clone().slice_of()
                            }
                            _ => panic!("ArrayToSlice cast on non-array type"),
                        }
                    }
                    CastTarget::WitnessOf => Type::witness_of(value_type.clone()),
                };

                function_info.values.insert(*result, result_type);
                Ok(())
            }
            OpCode::Truncate { result, value, to_bits: _, from_bits: _ } => {
                let value_type = function_info
                    .values
                    .get(value)
                    .ok_or_else(|| format!("Value {:?} not found in type assignments", value))?;

                function_info.values.insert(*result, value_type.clone());
                Ok(())
            }
            OpCode::Not { result, value } => {
                let value_type = function_info
                    .values
                    .get(value)
                    .ok_or_else(|| format!("Value {:?} not found in type assignments", value))?;
                function_info.values.insert(*result, value_type.clone());
                Ok(())
            }
            OpCode::ValueOf { result, value } => {
                let value_type = function_info
                    .values
                    .get(value)
                    .ok_or_else(|| format!("Value {:?} not found in type assignments", value))?;
                let inner = match &value_type.expr {
                    TypeExpr::WitnessOf(inner) => inner.as_ref().clone(),
                    _ => panic!("ICE: ValueOf applied to non-WitnessOf type: {}", value_type),
                };
                function_info.values.insert(*result, inner);
                Ok(())
            }
            OpCode::ToBits { result, value: _, endianness: _, count: output_size } => {
                let bit_type = Type::u(1);
                let result_type = bit_type.array_of(*output_size);
                function_info.values.insert(*result, result_type);
                Ok(())
            }
            OpCode::ToRadix { result, value: _, radix: _, endianness: _, count: output_size } => {
                let digit_type = Type::u(8);
                let result_type = digit_type.array_of(*output_size);
                function_info.values.insert(*result, result_type);
                Ok(())
            }
            OpCode::DLookup { target: _, keys: _, results: _ } => Ok(()),
            OpCode::MulConst { result, const_val: _, var } => {
                let var_type = function_info.values.get(var).unwrap();
                function_info.values.insert(*result, var_type.clone());
                Ok(())
            }
            OpCode::Rangecheck { value: v, max_bits: _ } => {
                let v_type = function_info.values.get(v).unwrap();
                if !v_type.strip_witness().is_field() {
                    return Err(format!(
                        "only field types are supported for rangecheck, got {}",
                        v_type
                    ));
                }
                Ok(())
            }
            OpCode::ReadGlobal { result: r, offset: _, result_type: tp } => {
                function_info.values.insert(*r, tp.clone());
                Ok(())
            }
            OpCode::Lookup { target: _, keys: _, results: _ } => Ok(()),
            OpCode::TupleProj { 
                result,
                tuple,
                idx,
            } => {
                if let TupleIdx::Static(sz) = idx {
                    let tuple_type = function_info.values.get(tuple).ok_or_else(|| {
                        format!("Tuple value {:?} not found in type assignments", tuple)
                    })?;
                    let element_type = tuple_type.get_tuple_element(*sz);
                    function_info.values.insert(
                        *result,
                        element_type,
                    );
                    Ok(())
                } else {
                    panic!("Dynamic TupleProj should not appear here")
                }
            }
            OpCode::MkTuple { 
                result,
                elems: _,
                element_types,
            } => {
                function_info.values.insert(*result, Type::tuple_of(element_types.clone()));
                Ok(())
            }
            OpCode::Todo { results, result_types, .. } => {
                if results.len() != result_types.len() {
                    return Err(format!(
                        "Todo opcode has {} results but {} result types",
                        results.len(),
                        result_types.len()
                    ));
                }
                for (result, result_type) in results.iter().zip(result_types.iter()) {
                    function_info.values.insert(*result, result_type.clone());
                }
                Ok(())
            }
            OpCode::InitGlobal { global: _, value: _ } => Ok(()),
            OpCode::DropGlobal { global: _ } => Ok(()),
        }
    }
}
