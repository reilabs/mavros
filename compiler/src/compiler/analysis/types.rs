//! An analysis pass that gathers (extrinsic) type information wherever needed, avoiding the need to
//! perform detailed bookkeeping of type information whenever transforming the IR.

use mavros_artifacts::FieldConfig;
use tracing::{Level, instrument};

use crate::{
    collections::HashMap,
    compiler::{
        analysis::flow_analysis::{CFG, FlowAnalysis},
        pass_manager::{Analysis, AnalysisId, AnalysisStore},
        ssa::{
            FunctionId, SSAConstantsSnapshot, ValueId,
            hlssa::{CallTarget, Constant, HLFunction, HLSSA, OpCode, Type, TypeExpr},
        },
        util::UNSPREAD_INPUT_MAX,
    },
};

/// The type of a constant, given the signatures of the program's functions.
///
/// `function_returns` answers what a call to each function produces, which is the only part of a
/// [`Constant::FnPtr`] that has a type: see [`TypeExpr::Function`]. Taking it from the callee's own
/// SSA signature rather than from the Noir type it was lowered from is deliberate -- that is the
/// signature `defunctionalize` builds its dispatcher against, so the two cannot drift apart.
pub fn const_value_type(
    value: &Constant,
    function_returns: &dyn Fn(FunctionId) -> Vec<Type>,
) -> Type {
    match value {
        Constant::Int(v) => Type::int(v.bits()),
        Constant::Field(_) => Type::field(),
        Constant::FnPtr(fn_id) => Type::function_returning(function_returns(*fn_id)),
        Constant::Blob(blob) => Type::blob(blob.elem_type.clone(), blob.len()),
    }
}

/// Types every constant in the module-level pool against the functions the SSA knows about.
pub(crate) fn pool_constant_types(
    constants: &SSAConstantsSnapshot<Constant>,
    function_returns: &dyn Fn(FunctionId) -> Option<Vec<Type>>,
) -> HashMap<ValueId, Type> {
    constants
        .iter()
        .filter_map(|(vid, cv)| {
            if let Constant::FnPtr(fn_id) = cv.as_ref()
                && function_returns(*fn_id).is_none()
            {
                return None;
            }
            let typ = const_value_type(cv, &|fn_id| {
                function_returns(fn_id).unwrap_or_else(|| ice!("no signature for {fn_id:?}"))
            });
            Some((*vid, typ))
        })
        .collect()
}

pub(crate) fn push_witness_of_to_leaves(t: Type) -> Type {
    match t.expr {
        TypeExpr::WitnessOf(_) => t,
        TypeExpr::Field | TypeExpr::Int(_) => Type::witness_of(t),
        TypeExpr::Array(inner, n) => push_witness_of_to_leaves(*inner).array_of(n),
        TypeExpr::Slice(inner) => push_witness_of_to_leaves(*inner).slice_of(),
        TypeExpr::Tuple(fields) => {
            Type::tuple_of(fields.into_iter().map(push_witness_of_to_leaves).collect())
        }
        TypeExpr::Blob(..) => t,
        TypeExpr::Ref(_) | TypeExpr::Function(_) => Type::witness_of(t),
    }
}

fn replace_array_element_type(container: &Type, element_type: Type) -> Type {
    match &container.expr {
        TypeExpr::Array(_, size) => element_type.array_of(*size),
        TypeExpr::Slice(_) => element_type.slice_of(),
        TypeExpr::WitnessOf(inner) => {
            Type::witness_of_collapsed(replace_array_element_type(inner, element_type))
        }
        _ => ice!("Type is not an array: {}", container),
    }
}

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
        self.try_get_value_type(value_id).unwrap()
    }

    /// The type of `value_id`, or [`None`] where the analysis never reached it.
    ///
    /// [`Types::run_function`] walks the dominator tree, so a value defined in a block no edge
    /// reaches has no entry here. A caller that iterates the block map instead needs the checked
    /// form.
    pub fn try_get_value_type(&self, value_id: ValueId) -> Option<&Type> {
        self.values.get(&value_id)
    }
}

pub struct Types {}

impl Types {
    pub fn new() -> Self {
        Types {}
    }

    pub fn run(&self, ssa: &HLSSA, cfg: &FlowAnalysis) -> TypeInfo {
        let mut type_info = TypeInfo {
            functions: HashMap::default(),
        };

        let function_types = ssa
            .iter_functions()
            .map(|(id, func)| (*id, (func.get_param_types(), func.get_returns())))
            .collect::<HashMap<_, _>>();

        // The constants side-table is module-level; pre-compute types for every constant
        // `ValueId` so `run_function` can seed `function_info` with them.
        let constant_types = pool_constant_types(&ssa.const_snapshot(), &|fn_id| {
            function_types
                .get(&fn_id)
                .map(|(_, returns)| returns.to_vec())
        });

        // The configured field, threaded through calls so that the width of a `Field` can be read
        // from it rather than a static.
        let field = ssa.field();

        for (function_id, function) in ssa.iter_functions() {
            let cfg = cfg.get_function_cfg(*function_id);
            let function_info =
                self.run_function(function, &function_types, &constant_types, cfg, field);
            type_info.functions.insert(*function_id, function_info);
        }
        type_info
    }

    fn spread_result_type(value_type: &Type) -> Result<Type, String> {
        match &value_type.expr {
            TypeExpr::WitnessOf(inner) => Ok(Type::witness_of(Self::spread_result_type(inner)?)),
            TypeExpr::Int(bits) => {
                // The host word the spread ladders run in, rather than the integer type cap.
                //
                // Looser than the evaluators: this rule answers `int(2n)`, so the input bound a
                // host word implies is half of one, which is what `util::spread_bits` and
                // `specializer::spread` both assert. A `Spread(int(65..=128))` therefore type
                // checks here and panics in every evaluator.
                if *bits > UNSPREAD_INPUT_MAX {
                    return Err(format!(
                        "Spread expects int(n) with n <= {UNSPREAD_INPUT_MAX}, got {}",
                        value_type
                    ));
                }
                Ok(Type::int(bits * 2))
            }
            TypeExpr::Field => Err("Spread does not support field inputs".to_string()),
            _ => Err(format!(
                "Spread expects an integer input, got {}",
                value_type
            )),
        }
    }

    fn unspread_result_types(value_type: &Type) -> Result<(Type, Type), String> {
        match &value_type.expr {
            TypeExpr::WitnessOf(inner) => {
                let (odd, even) = Self::unspread_result_types(inner)?;
                Ok((Type::witness_of(odd), Type::witness_of(even)))
            }
            TypeExpr::Int(bits) => {
                if *bits % 2 != 0 || (*bits / 2) > 64 {
                    return Err(format!(
                        "Unspread expects int(2n) with n <= 64, got {}",
                        value_type
                    ));
                }
                let half_bits = bits / 2;
                Ok((Type::int(half_bits), Type::int(half_bits)))
            }
            TypeExpr::Field => Err("Unspread does not support field inputs".to_string()),
            _ => Err(format!(
                "Unspread expects an integer input, got {}",
                value_type
            )),
        }
    }

    #[instrument(skip_all, level = Level::DEBUG, name = "Types::run_function", fields(function = function.get_name()))]
    pub fn run_function(
        &self,
        function: &HLFunction,
        function_types: &HashMap<FunctionId, (Vec<Type>, &[Type])>,
        constant_types: &HashMap<ValueId, Type>,
        cfg: &CFG,
        field: FieldConfig,
    ) -> FunctionTypeInfo {
        let mut function_info = FunctionTypeInfo {
            values: constant_types.clone(),
        };

        for block_id in cfg.get_domination_pre_order() {
            let block = function.get_block(block_id);

            for param in block.get_parameters() {
                function_info.values.insert(param.0, param.1.clone());
            }

            for instruction in block.get_instructions() {
                self.run_opcode(instruction, &mut function_info, function_types, field)
                    .unwrap_or_else(|e| ice!("Error running opcode {instruction:?}: {e}"));
            }
        }

        function_info
    }

    fn run_opcode(
        &self,
        opcode: &OpCode,
        function_info: &mut FunctionTypeInfo,
        function_types: &HashMap<FunctionId, (Vec<Type>, &[Type])>,
        field: FieldConfig,
    ) -> Result<(), String> {
        match opcode {
            OpCode::Cmp {
                result, lhs, rhs, ..
            } => {
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
                // `bool`, whatever the operands are and however the comparison reads them.
                let result_type = if lhs_type.is_witness_of() || rhs_type.is_witness_of() {
                    Type::witness_of(Type::int(1))
                } else {
                    Type::int(1)
                };
                function_info.values.insert(*result, result_type);
                Ok(())
            }
            OpCode::BinaryArithOp {
                result, lhs, rhs, ..
            } => {
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
                // Width only: an operation's result is as wide as its wider operand, and how the
                // operands are read is the opcode's business, not this rule's.
                function_info
                    .values
                    .insert(*result, lhs_type.get_arithmetic_result_type(rhs_type));
                Ok(())
            }
            OpCode::Alloc { result, value } => {
                let value_type = function_info.values.get(value).ok_or_else(|| {
                    format!("Alloc value {:?} not found in type assignments", value)
                })?;
                function_info
                    .values
                    .insert(*result, value_type.clone().ref_of());
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
                function_info
                    .values
                    .insert(*result, ptr_type.get_refered().clone());
                Ok(())
            }
            OpCode::MemOp { kind: _, value: _ } => Ok(()),
            OpCode::Assert { value: _ } => Ok(()),
            OpCode::AssertCmp {
                kind: _,
                lhs: _,
                rhs: _,
            } => Ok(()),
            OpCode::AssertR1C { a: _, b: _, c: _ } => Ok(()),
            OpCode::Call {
                results: result,
                function,
                args,
                unconstrained: _,
            } => match function {
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

                // A dynamic call is typed from the callee _value_, because the callee itself is not
                // known here. The arguments go unchecked, unlike above, because a `Function` type
                // carries no parameter list to check them against.
                CallTarget::Dynamic(callee) => {
                    let callee_type = function_info.values.get(callee).ok_or_else(|| {
                        format!("Callee value {:?} not found in type assignments", callee)
                    })?;
                    let return_types = callee_type
                        .call_returns()
                        .ok_or_else(|| {
                            format!("Indirect call through a {callee_type}, which is not callable")
                        })?
                        .to_vec();

                    if result.len() != return_types.len() {
                        return Err(format!(
                            "Indirect call through a {} expects {} return values, got {}",
                            callee_type,
                            return_types.len(),
                            result.len()
                        ));
                    }

                    for (ret, ret_type) in result.iter().zip(return_types.iter()) {
                        function_info.values.insert(*ret, ret_type.clone());
                    }
                    Ok(())
                }
            },
            OpCode::ArrayGet {
                result,
                array,
                index,
            } => {
                let array_type = function_info.values.get(array).ok_or_else(|| {
                    format!("Array value {:?} not found in type assignments", array)
                })?;
                let index_type = function_info.values.get(index).ok_or_else(|| {
                    format!("Index value {:?} not found in type assignments", index)
                })?;

                let element_type = array_type.get_array_element();
                let result_type = if index_type.is_witness_of() {
                    push_witness_of_to_leaves(element_type)
                } else {
                    element_type
                };
                function_info.values.insert(*result, result_type);
                Ok(())
            }
            OpCode::ArraySet {
                result,
                array,
                index,
                value,
            } => {
                let array_type = function_info.values.get(array).ok_or_else(|| {
                    format!("Array value {:?} not found in type assignments", array)
                })?;
                let value_type = function_info
                    .values
                    .get(value)
                    .ok_or_else(|| format!("Value {:?} not found in type assignments", value))?;
                let index_type = function_info.values.get(index).ok_or_else(|| {
                    format!("Index value {:?} not found in type assignments", index)
                })?;

                let elem_type = array_type.get_array_element();
                let result_elem_type = if index_type.is_witness_of() {
                    push_witness_of_to_leaves(Type::join(&elem_type, value_type))
                } else {
                    Type::join(&elem_type, value_type)
                };
                let result_type = if result_elem_type == elem_type {
                    array_type.clone()
                } else {
                    replace_array_element_type(array_type, result_elem_type)
                };
                function_info.values.insert(*result, result_type);
                Ok(())
            }
            OpCode::SlicePush {
                result,
                slice,
                values,
                dir: _,
            } => {
                let slice_type = function_info.values.get(slice).ok_or_else(|| {
                    format!("Slice value {:?} not found in type assignments", slice)
                })?;

                let elem_type = slice_type.get_array_element();
                let mut result_elem_type = elem_type.clone();
                for v in values {
                    let value_type = function_info
                        .values
                        .get(v)
                        .ok_or_else(|| format!("Value {:?} not found in type assignments", v))?;
                    result_elem_type = Type::join(&result_elem_type, value_type);
                }
                let result_type = if result_elem_type == elem_type {
                    slice_type.clone()
                } else {
                    replace_array_element_type(slice_type, result_elem_type)
                };
                function_info.values.insert(*result, result_type);
                Ok(())
            }
            OpCode::SlicePop {
                result_slice,
                result_elem,
                slice,
                dir: _,
            } => {
                let slice_type = function_info.values.get(slice).ok_or_else(|| {
                    format!("Slice value {:?} not found in type assignments", slice)
                })?;
                // `result_elem` gets the plain element type; the extra witness-ness of a back
                // pop is applied to types later by untaint. Keep in sync with the `SlicePop`
                // rule in `witness_taint_inference/builder.rs`.
                let elem_type = slice_type.get_array_element();
                function_info
                    .values
                    .insert(*result_slice, slice_type.clone());
                function_info.values.insert(*result_elem, elem_type);
                Ok(())
            }
            OpCode::SliceInsert {
                result,
                slice,
                index,
                value,
            } => {
                let slice_type = function_info.values.get(slice).ok_or_else(|| {
                    format!("Slice value {:?} not found in type assignments", slice)
                })?;
                let value_type = function_info
                    .values
                    .get(value)
                    .ok_or_else(|| format!("Value {:?} not found in type assignments", value))?;
                let index_type = function_info.values.get(index).ok_or_else(|| {
                    format!("Index value {:?} not found in type assignments", index)
                })?;

                let elem_type = slice_type.get_array_element();
                let result_elem_type = if index_type.is_witness_of() {
                    push_witness_of_to_leaves(Type::join(&elem_type, value_type))
                } else {
                    Type::join(&elem_type, value_type)
                };
                let result_type = if result_elem_type == elem_type {
                    slice_type.clone()
                } else {
                    replace_array_element_type(slice_type, result_elem_type)
                };
                function_info.values.insert(*result, result_type);
                Ok(())
            }
            OpCode::SliceRemove {
                result_slice,
                result_elem,
                slice,
                index,
            } => {
                let slice_type = function_info.values.get(slice).ok_or_else(|| {
                    format!("Slice value {:?} not found in type assignments", slice)
                })?;
                let index_type = function_info.values.get(index).ok_or_else(|| {
                    format!("Index value {:?} not found in type assignments", index)
                })?;

                let elem_type = slice_type.get_array_element();
                // A witness-index shift makes every surviving element (and the removed one) a
                // select result.
                let result_elem_type = if index_type.is_witness_of() {
                    push_witness_of_to_leaves(elem_type.clone())
                } else {
                    elem_type.clone()
                };
                let result_slice_type = if result_elem_type == elem_type {
                    slice_type.clone()
                } else {
                    replace_array_element_type(slice_type, result_elem_type.clone())
                };
                function_info
                    .values
                    .insert(*result_slice, result_slice_type);
                function_info.values.insert(*result_elem, result_elem_type);
                Ok(())
            }
            OpCode::SliceLen { result, slice } => {
                let _ = function_info.values.get(slice).ok_or_else(|| {
                    format!("Slice value {:?} not found in type assignments", slice)
                })?;
                function_info.values.insert(*result, Type::int(32));
                Ok(())
            }
            OpCode::Select {
                result,
                cond,
                if_t: then,
                if_f: otherwise,
            } => {
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
                let alt_type = then_type.get_select_result_type(otherwise_type);

                // If cond is WitnessOf and alternatives are not already WitnessOf, the witness
                // influence lands on the result's LEAVES: scalars/refs get wrapped, but containers
                // don't. Note that a slice's length is not tainted here: `purify_witness_slices`
                // has already moved any witness length onto a `log_len` scalar.
                let result_type = if cond_type.is_witness_of() && !alt_type.is_witness_of() {
                    push_witness_of_to_leaves(alt_type)
                } else {
                    alt_type
                };

                function_info.values.insert(*result, result_type);
                Ok(())
            }
            OpCode::WriteWitness { result, value, .. } => {
                let Some(result) = result else {
                    return Ok(());
                };
                let witness_type = function_info.values.get(value).ok_or_else(|| {
                    format!("Witness value {:?} not found in type assignments", value)
                })?;
                function_info
                    .values
                    .insert(*result, Type::witness_of(witness_type.clone()));
                Ok(())
            }
            OpCode::FreshWitness {
                result: r,
                result_type: tp,
            } => {
                function_info
                    .values
                    .insert(*r, Type::witness_of(tp.clone()));
                Ok(())
            }
            OpCode::Constrain { a: _, b: _, c: _ } => Ok(()),
            OpCode::NextDCoeff { result: v } => {
                function_info.values.insert(*v, Type::field());
                Ok(())
            }
            OpCode::BumpD {
                matrix: _,
                variable: _,
                sensitivity: _,
            } => Ok(()),
            OpCode::MkSeq {
                result: r,
                elems: _,
                seq_type: top_tp,
                elem_type: t,
            } => {
                function_info.values.insert(*r, top_tp.of(t.clone()));
                Ok(())
            }
            OpCode::MkSeqOfBlob {
                result: r,
                element_type,
                blob,
            } => {
                let len = match function_info.values.get(blob) {
                    Some(Type {
                        expr: TypeExpr::Blob(_, len),
                    }) => *len,
                    other => ice!("MkSeqOfBlob expected Blob input, got {:?}", other),
                };
                function_info
                    .values
                    .insert(*r, element_type.clone().array_of(len));
                Ok(())
            }
            OpCode::MkRepeated {
                result: r,
                element: _,
                seq_type: top_tp,
                count: _,
                elem_type: t,
            } => {
                function_info.values.insert(*r, top_tp.of(t.clone()));
                Ok(())
            }
            OpCode::Cast {
                result,
                value,
                target,
            } => {
                let value_type = function_info
                    .values
                    .get(value)
                    .ok_or_else(|| format!("Value {:?} not found in type assignments", value))?;

                let result_type = target.result_type(value_type);

                function_info.values.insert(*result, result_type);
                Ok(())
            }
            OpCode::SExt {
                result,
                value,
                from_bits,
                to_bits,
            } => {
                let value_type = function_info
                    .values
                    .get(value)
                    .ok_or_else(|| format!("Value {:?} not found in type assignments", value))?;

                // Widen (signed) to the target width, keeping the witness wrapper.
                let inner = value_type.strip_witness();
                let widened = match &inner.expr {
                    // `from_bits` is a second copy of the operand's own width.
                    TypeExpr::Int(operand_bits) if operand_bits == from_bits => Type::int(*to_bits),
                    TypeExpr::Int(operand_bits) => {
                        return Err(format!(
                            "SExt declares from_bits {from_bits} \
                             for a value of type int{operand_bits}"
                        ));
                    }
                    _ => ice!("SExt on non-integer type: {:?}", value_type),
                };
                let result_type = if value_type.is_witness_of() {
                    Type::witness_of(widened)
                } else {
                    widened
                };
                function_info.values.insert(*result, result_type);
                Ok(())
            }
            OpCode::BitRange {
                result,
                value,
                offset,
                width,
            } => {
                if *width == 0 {
                    return Err("BitRange width must be at least 1".to_string());
                }
                let value_type = function_info
                    .values
                    .get(value)
                    .ok_or_else(|| format!("Value {:?} not found in type assignments", value))?;
                let value_bits = value_type.get_bit_size(field);
                if *offset + *width > value_bits {
                    return Err(format!(
                        "BitRange({}, {}) exceeds source width {} for {}",
                        offset, width, value_bits, value_type
                    ));
                }
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
            OpCode::ToBits {
                result,
                value,
                endianness: _,
                count: output_size,
            } => {
                let value_type = function_info
                    .values
                    .get(value)
                    .ok_or_else(|| format!("Value {:?} not found in type assignments", value))?;
                let bit_type = if value_type.is_witness_of() {
                    Type::witness_of(Type::int(1))
                } else {
                    Type::int(1)
                };
                let result_type = bit_type.array_of(*output_size);
                function_info.values.insert(*result, result_type);
                Ok(())
            }
            OpCode::ToRadix {
                result,
                value,
                radix: _,
                endianness: _,
                count: output_size,
            } => {
                let value_type = function_info
                    .values
                    .get(value)
                    .ok_or_else(|| format!("Value {:?} not found in type assignments", value))?;
                let digit_type = if value_type.is_witness_of() {
                    Type::witness_of(Type::int(8))
                } else {
                    Type::int(8)
                };
                let result_type = digit_type.array_of(*output_size);
                function_info.values.insert(*result, result_type);
                Ok(())
            }
            OpCode::DLookup {
                target: _,
                args: _,
                flag: _,
            } => Ok(()),
            OpCode::AssertConstant { .. } => Ok(()),
            OpCode::MulConst {
                result,
                const_val: _,
                var,
            } => {
                let var_type = function_info.values.get(var).unwrap();
                function_info.values.insert(*result, var_type.clone());
                Ok(())
            }
            OpCode::Rangecheck {
                value: v,
                max_bits: _,
            } => {
                let v_type = function_info.values.get(v).unwrap();
                if !v_type.strip_witness().is_field() {
                    return Err(format!(
                        "only field types are supported for rangecheck, got {}",
                        v_type
                    ));
                }
                Ok(())
            }
            OpCode::ReadGlobal {
                result: r,
                offset: _,
                result_type: tp,
            } => {
                function_info.values.insert(*r, tp.clone());
                Ok(())
            }
            OpCode::Lookup {
                target: _,
                args: _,
                flag: _,
            } => Ok(()),
            OpCode::TupleProj { result, tuple, idx } => {
                let tuple_type = function_info.values.get(tuple).ok_or_else(|| {
                    format!("Tuple value {:?} not found in type assignments", tuple)
                })?;
                let element_type = tuple_type.get_tuple_element(*idx);
                function_info.values.insert(*result, element_type);
                Ok(())
            }
            OpCode::TupleRefProj {
                result,
                tuple_ref,
                idx,
            } => {
                let tuple_ref_type = function_info.values.get(tuple_ref).ok_or_else(|| {
                    format!(
                        "Tuple reference value {:?} not found in type assignments",
                        tuple_ref
                    )
                })?;
                let element_type = tuple_ref_type.get_pointed().get_tuple_element(*idx);
                function_info.values.insert(*result, element_type.ref_of());
                Ok(())
            }
            OpCode::MkTuple {
                result,
                elems: _,
                element_types,
            } => {
                function_info
                    .values
                    .insert(*result, Type::tuple_of(element_types.clone()));
                Ok(())
            }
            OpCode::Todo {
                results,
                result_types,
                ..
            } => {
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
            OpCode::InitGlobal {
                global: _,
                value: _,
            } => Ok(()),
            OpCode::DropGlobal { global: _ } => Ok(()),
            OpCode::Spread { result, value, .. } => {
                let value_type = function_info
                    .values
                    .get(value)
                    .ok_or_else(|| format!("Value {:?} not found in type assignments", value))?;
                let result_type = Self::spread_result_type(value_type)?;
                function_info.values.insert(*result, result_type);
                Ok(())
            }
            OpCode::Unspread {
                result_odd,
                result_even,
                value,
                ..
            } => {
                let value_type = function_info
                    .values
                    .get(value)
                    .ok_or_else(|| format!("Value {:?} not found in type assignments", value))?;
                let (odd_type, even_type) = Self::unspread_result_types(value_type)?;
                function_info.values.insert(*result_odd, odd_type);
                function_info.values.insert(*result_even, even_type);
                Ok(())
            }
            OpCode::Guard { inner, .. } => {
                self.run_opcode(inner, function_info, function_types, field)
            }
        }
    }
}

impl Analysis for TypeInfo {
    fn dependencies() -> Vec<AnalysisId> {
        vec![FlowAnalysis::id()]
    }

    fn compute(ssa: &HLSSA, store: &AnalysisStore) -> Self {
        let cfg = store.get::<FlowAnalysis>();
        Types::new().run(ssa, cfg)
    }
}

// TESTS
// ================================================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::ssa::hlssa::builder::{HLEmitter, HLSSABuilder};
    use mavros_int_semantics::IntBits;

    /// Type an `SExt` whose operand is an eight-bit constant but which declares `from_bits`.
    fn sext_declaring(from_bits: usize) {
        let mut ssa = HLSSA::with_main("main".to_string());
        let main_id = ssa.get_unique_entrypoint_id();
        {
            let mut sb = HLSSABuilder::new(&mut ssa);
            sb.modify_function(main_id, |b| {
                b.function.add_return_type(Type::int(16));
                let entry = b.function.get_entry_id();
                let mut e = b.test_block(entry);
                let v = e.int_const(IntBits::from_u128(8, 0x80));
                let widened = e.sext(v, from_bits, 16);
                e.terminate_return(vec![widened]);
            });
        }
        let flow = FlowAnalysis::run(&ssa);
        let _ = Types::new().run(&ssa, &flow);
    }

    #[test]
    fn a_sext_declaring_its_operands_own_width_types_fine() {
        sext_declaring(8);
    }

    /// The check that lets every consumer of `SExt` read the width off the operand instead.
    #[test]
    #[should_panic(expected = "SExt declares from_bits 16 for a value of type int8")]
    fn a_sext_declaring_a_width_its_operand_does_not_have_is_rejected() {
        sext_declaring(16);
    }

    /// `main(x: int8) { fn_ptr(x) }` calling `callee(value: int8) -> returns` indirectly.
    fn program_calling_indirectly(returns: Vec<Type>) -> (HLSSA, Vec<ValueId>) {
        let mut ssa = HLSSA::with_main("main".to_string());
        let main_id = ssa.get_unique_entrypoint_id();
        let mut builder = HLSSABuilder::new(&mut ssa);

        let (callee_id, ()) = builder.add_function("callee".to_string(), |b| {
            for typ in &returns {
                b.function.add_return_type(typ.clone());
            }
            let entry = b.function.get_entry_id();
            let mut e = b.test_block(entry);
            let value = e.add_parameter(Type::int(8));
            let results = returns.iter().map(|_| value).collect();
            e.terminate_return(results);
        });

        let results = builder.modify_function(main_id, |b| {
            let entry = b.function.get_entry_id();
            let mut e = b.test_block(entry);
            let value = e.add_parameter(Type::int(8));
            let fn_ptr = e.emit_constant(Constant::FnPtr(callee_id));
            let results = e.call_indirect(fn_ptr, vec![value], returns.len());
            e.terminate_return(vec![]);
            results
        });

        (ssa, results)
    }

    /// The results of a call through a function pointer are typed from the pointer's own type.
    #[test]
    fn an_indirect_calls_results_take_the_callees_return_types() {
        let (ssa, results) = program_calling_indirectly(vec![Type::field()]);
        let flow = FlowAnalysis::run(&ssa);

        let types = Types::new().run(&ssa, &flow);
        let main = types.get_function(ssa.get_unique_entrypoint_id());

        assert_eq!(main.get_value_type(results[0]), &Type::field());
    }

    /// A call returning nothing is the other end of the same rule: `Unit` is no results, not one
    /// result of some unit type, so there is nothing to name and nothing to type.
    #[test]
    fn an_indirect_call_returning_nothing_types_fine() {
        let (ssa, results) = program_calling_indirectly(Vec::new());
        assert!(results.is_empty());

        let flow = FlowAnalysis::run(&ssa);
        let _ = Types::new().run(&ssa, &flow);
    }

    /// A callable that arrives as a parameter has no constant to read a signature off, so its
    /// declared type is the answer.
    #[test]
    fn a_callable_parameters_declared_type_types_the_call_through_it() {
        let mut ssa = HLSSA::with_main("apply".to_string());
        let apply = ssa.get_unique_entrypoint_id();
        let mut builder = HLSSABuilder::new(&mut ssa);

        let result = builder.modify_function(apply, |b| {
            b.function.add_return_type(Type::field());
            let entry = b.function.get_entry_id();
            let mut e = b.test_block(entry);
            let callable = e.add_parameter(Type::function_returning(vec![Type::field()]));
            let value = e.add_parameter(Type::field());
            let results = e.call_indirect(callable, vec![value], 1);
            e.terminate_return(results.clone());
            results[0]
        });

        let flow = FlowAnalysis::run(&ssa);
        let types = Types::new().run(&ssa, &flow);

        assert_eq!(
            types.get_function(apply).get_value_type(result),
            &Type::field()
        );
    }

    /// A function pointer's type is read off the callee's own SSA signature, so a mismatch between
    /// it and the call's result count is an error rather than a silently mistyped value.
    #[test]
    #[should_panic(expected = "expects 1 return values, got 2")]
    fn an_indirect_call_taking_more_results_than_the_callee_returns_is_rejected() {
        let (mut ssa, _) = program_calling_indirectly(vec![Type::field()]);
        let main_id = ssa.get_unique_entrypoint_id();
        let extra = ssa.fresh_value();

        for (_, block) in ssa.get_function_mut(main_id).get_blocks_mut() {
            for instruction in block.get_instructions_mut() {
                if let OpCode::Call { results, .. } = instruction {
                    results.push(extra);
                }
            }
        }

        let flow = FlowAnalysis::run(&ssa);
        let _ = Types::new().run(&ssa, &flow);
    }
}
