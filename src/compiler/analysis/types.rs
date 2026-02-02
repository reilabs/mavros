use core::panic;
use std::{collections::HashMap, fmt::{Debug, Display}};

use tracing::{Level, instrument};

use crate::compiler::{
    flow_analysis::{CFG, FlowAnalysis},
    ir::r#type::{CommutativeMonoid, Type},
    ssa::{CastTarget, Const, Function, FunctionId, OpCode, SSA, TupleIdx, ValueId},
};

pub struct TypeInfo<V> {
    functions: HashMap<FunctionId, FunctionTypeInfo<V>>,
}

impl<V> TypeInfo<V> {
    pub fn get_function(&self, function_id: FunctionId) -> &FunctionTypeInfo<V> {
        self.functions.get(&function_id).unwrap()
    }
}

pub struct FunctionTypeInfo<V> {
    values: HashMap<ValueId, Type<V>>, 
}

impl<V> FunctionTypeInfo<V> {
    pub fn get_value_type(&self, value_id: ValueId) -> &Type<V> {
        self.values.get(&value_id).unwrap()
    }
}

pub struct Types {}

impl Types {
    pub fn new() -> Self {
        Types {}
    }

    pub fn run<V: CommutativeMonoid + Display + Eq + Clone + Debug>(
        &self,
        ssa: &SSA<V>,
        cfg: &FlowAnalysis,
    ) -> TypeInfo<V> {
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
    fn run_function<V: CommutativeMonoid + Display + Eq + Clone + Debug>(
        &self,
        function: &Function<V>,
        function_types: &HashMap<FunctionId, (Vec<Type<V>>, &[Type<V>])>,
        cfg: &CFG,
    ) -> FunctionTypeInfo<V> {
        let mut function_info = FunctionTypeInfo {
            values: HashMap::new(),
        };

        for (value_id, const_) in function.iter_consts() {
            match const_ {
                Const::U(size, _) => function_info
                    .values
                    .insert(*value_id, Type::u(*size, V::empty())),
                Const::Field(_) => function_info
                    .values
                    .insert(*value_id, Type::field(V::empty())),
                Const::WitnessRef(_) => function_info
                    .values
                    .insert(*value_id, Type::witness_ref(V::empty())),
            };
        }

        for block in cfg.get_domination_pre_order() {
            let block = function.get_block(block);

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

    fn run_opcode<V: CommutativeMonoid + Display + Eq + Clone>(
        &self,
        opcode: &OpCode<V>,
        function_info: &mut FunctionTypeInfo<V>,
        function_types: &HashMap<FunctionId, (Vec<Type<V>>, &[Type<V>])>,
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
                function_info.values.insert(
                    *result,
                    Type::u(1, lhs_type.get_annotation().op(rhs_type.get_annotation())),
                );
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
            OpCode::Alloc { result, elem_type: typ, result_annotation: annotation } => {
                function_info
                    .values
                    .insert(*result, Type::ref_of(typ.clone(), annotation.clone()));
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
                    ptr_type
                        .get_refered()
                        .clone()
                        .combine_with_annotation(ptr_type.get_annotation()),
                );
                Ok(())
            }
            OpCode::MemOp { kind: _, value: _ } => Ok(()),
            OpCode::AssertEq { lhs: _, rhs: _ } => Ok(()),
            OpCode::AssertR1C { a: _, b: _, c: _ } => Ok(()),
            OpCode::Call { results: result, function: fn_id, args } => {
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
            OpCode::ArrayGet { result, array, index: _ } => {
                let array_type = function_info.values.get(array).ok_or_else(|| {
                    format!("Array value {:?} not found in type assignments", array)
                })?;

                let element_type = array_type.get_array_element();
                function_info.values.insert(
                    *result,
                    element_type.combine_with_annotation(array_type.get_annotation()),
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
                    Type::u(32, V::empty()),
                );
                Ok(())
            }
            OpCode::Select { result, cond: _cond, if_t: then, if_f: otherwise } => {
                let then_type = function_info.values.get(then).ok_or_else(|| {
                    format!("Then value {:?} not found in type assignments", then)
                })?;
                let otherwise_type = function_info.values.get(otherwise).ok_or_else(|| {
                    format!(
                        "Otherwise value {:?} not found in type assignments",
                        otherwise
                    )
                })?;
                function_info.values.insert(
                    *result,
                    then_type.get_arithmetic_result_type(otherwise_type),
                );
                Ok(())
            }
            OpCode::WriteWitness { result, value, witness_annotation: annotation } => {
                let Some(result) = result else {
                    return Ok(());
                };
                let witness_type = function_info.values.get(value).ok_or_else(|| {
                    format!("Witness value {:?} not found in type assignments", value)
                })?;
                function_info.values.insert(
                    *result,
                    witness_type.clone().combine_with_annotation(annotation),
                );
                Ok(())
            }
            OpCode::FreshWitness { result: r, result_type: tp } => {
                function_info.values.insert(*r, tp.clone());
                Ok(())
            }
            OpCode::Constrain { a: _, b: _, c: _ } => Ok(()),
            OpCode::NextDCoeff { result: v } => {
                function_info.values.insert(*v, Type::field(V::empty()));
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
                    CastTarget::Field => Type::field(value_type.get_annotation().clone()),
                    CastTarget::U(size) => Type::u(*size, value_type.get_annotation().clone()),
                    CastTarget::Nop => value_type.clone(),
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
            OpCode::ToBits { result, value, endianness: _, count: output_size } => {
                let value_type = function_info
                    .values
                    .get(value)
                    .ok_or_else(|| format!("Value {:?} not found in type assignments", value))?;
                // Return an array of U(1) values with the same annotation as the input
                let bit_type = Type::u(1, value_type.get_annotation().clone());
                let result_type = Type::array_of(bit_type, *output_size, V::empty());
                function_info.values.insert(*result, result_type);
                Ok(())
            }
            OpCode::ToRadix { result, value, radix: _, endianness: _, count: output_size } => {
                let value_type = function_info
                    .values
                    .get(value)
                    .ok_or_else(|| format!("Value {:?} not found in type assignments", value))?;
                // Return an array of U(8) values (digits) with the same annotation as the input
                let digit_type = Type::u(8, value_type.get_annotation().clone());
                let result_type = Type::array_of(digit_type, *output_size, V::empty());
                function_info.values.insert(*result, result_type);
                Ok(())
            }
            OpCode::DLookup { target: _, keys: _, results: _ } => Ok(()),
            OpCode::PureToWitnessRef { result, value: _, result_annotation: annotation } => {
                function_info
                    .values
                    .insert(*result, Type::witness_ref(annotation.clone()));
                Ok(())
            }
            OpCode::UnboxField { result, value: _ } => {
                function_info
                    .values
                    .insert(*result, Type::field(V::empty()));
                Ok(())
            }
            OpCode::MulConst { result, const_val: _, var } => {
                let var_type = function_info.values.get(var).unwrap();
                function_info.values.insert(*result, var_type.clone());
                Ok(())
            }
            OpCode::Rangecheck { value: v, max_bits: _ } => {
                let v_type = function_info.values.get(v).unwrap();
                if !v_type.is_field() {
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
                        element_type.combine_with_annotation(tuple_type.get_annotation()),
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
                function_info.values.insert(*result, Type::tuple_of(element_types.clone(), V::empty()));
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
        }
    }
}
