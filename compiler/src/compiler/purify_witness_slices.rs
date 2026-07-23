//! Purifies witness-length slices into `(physical, log_len)` tuples.

use crate::{
    collections::HashMap,
    compiler::{
        analysis::{
            flow_analysis::FlowAnalysis,
            types::{FunctionTypeInfo, Types},
            witness_info::{FunctionWitnessType, WitnessShape},
            witness_taint_inference::WitnessTaintInference,
        },
        pass_manager::{AnalysisStore, Pass},
        ssa::{
            BlockId, Instruction, SourceLocation, Terminator, ValueId,
            hlssa::{
                CmpKind, HLFunction, HLSSA, LocatedOpCode, OpCode, SliceOpDir, Type, TypeExpr,
                builder::{HLEmitter, HLInstrBuilder},
            },
        },
    },
};

pub struct PurifyWitnessSlices {}

impl PurifyWitnessSlices {
    pub fn new() -> Self {
        Self {}
    }
}

impl Pass for PurifyWitnessSlices {
    fn name(&self) -> &'static str {
        "purify_witness_slices"
    }

    fn run(&self, ssa: &mut HLSSA, _store: &AnalysisStore) {
        let flow = FlowAnalysis::run(ssa);
        let mut wti = WitnessTaintInference::new();
        wti.run(ssa, &flow);

        let flow = FlowAnalysis::run(ssa);
        let types = Types::new().run(ssa, &flow);

        let function_ids: Vec<_> = ssa.get_function_ids().collect();
        for function_id in function_ids {
            if !types.has_function(function_id) {
                continue;
            }
            let Some(fwt) = wti.try_get_function_witness_type(function_id) else {
                continue;
            };
            let type_info = types.get_function(function_id);
            let affected = affected_values(type_info, fwt);
            if affected.is_empty() {
                continue;
            }

            let block_order: Vec<BlockId> = flow
                .get_function_cfg(function_id)
                .get_domination_pre_order()
                .collect();
            let mut function = ssa.take_function(function_id);
            rewrite_function(&mut function, ssa, type_info, &affected, &block_order, fwt);
            ssa.put_function(function_id, function);
        }
    }
}

fn purify_type(ty: &Type, shape: &WitnessShape) -> Type {
    match (&ty.expr, shape) {
        (TypeExpr::Slice { elem, len }, WitnessShape::Array(top, inner)) => {
            let physical = purify_type(elem, inner).slice_of_with_len(len.as_ref().clone());
            if top.is_witness() {
                Type::tuple_of(vec![physical.clone(), Type::u(32)])
            } else {
                physical
            }
        }
        (TypeExpr::Ref(inner_ty), WitnessShape::Ref(_, inner_shape)) => {
            purify_type(inner_ty, inner_shape).ref_of()
        }
        // Note that `Array<WLslice>` is unreachable today
        (TypeExpr::Array(elem, n), WitnessShape::Array(_, inner)) => {
            purify_type(elem, inner).array_of(*n)
        }
        _ => ty.clone(),
    }
}

fn affected_values(
    type_info: &FunctionTypeInfo,
    fwt: &FunctionWitnessType,
) -> HashMap<ValueId, Type> {
    let mut affected: HashMap<ValueId, Type> = HashMap::default();
    for (&v, shape) in fwt.value_witness_types.iter() {
        let ty = type_info.get_value_type(v);
        let pty = purify_type(ty, shape);
        if pty != *ty {
            affected.insert(v, pty);
        }
    }
    affected
}

fn is_wl_slice(v: ValueId, affected: &HashMap<ValueId, Type>) -> bool {
    affected.get(&v).is_some_and(|pty| pty.is_tuple())
}

fn mk_slice_tuple(
    physical: ValueId,
    log_len: ValueId,
    phys_ty: Type,
    function: &mut HLFunction,
    ssa: &mut HLSSA,
    new_instrs: &mut Vec<LocatedOpCode>,
    loc: SourceLocation,
) -> ValueId {
    let mut b = HLInstrBuilder::new(function, ssa, new_instrs, loc);
    b.mk_tuple(vec![physical, log_len], vec![phys_ty, Type::u(32)])
}

fn materialize_pure_slice_tuple(
    slice: ValueId,
    type_info: &FunctionTypeInfo,
    function: &mut HLFunction,
    ssa: &mut HLSSA,
    new_instrs: &mut Vec<LocatedOpCode>,
) -> ValueId {
    let phys_ty = type_info.get_value_type(slice).clone();
    let loc = SourceLocation::synthetic("purify_witness_slices");
    let ll = HLInstrBuilder::new(function, ssa, new_instrs, loc.clone()).slice_len(slice);
    mk_slice_tuple(slice, ll, phys_ty, function, ssa, new_instrs, loc)
}

fn rewrite_function(
    function: &mut HLFunction,
    ssa: &mut HLSSA,
    type_info: &FunctionTypeInfo,
    affected: &HashMap<ValueId, Type>,
    block_order: &[BlockId],
    fwt: &FunctionWitnessType,
) {
    let mut replacement_tuple_map: HashMap<ValueId, ValueId> = HashMap::default();

    let lifted_block_args: HashMap<BlockId, Vec<bool>> = function
        .get_blocks()
        .map(|(bid, block)| {
            let positions = block
                .get_parameters()
                .map(|(v, _)| is_wl_slice(*v, affected))
                .collect();
            (*bid, positions)
        })
        .collect();

    let lifted_returns: Vec<bool> = function
        .iter_returns_mut()
        .zip(&fwt.returns_witness)
        .map(|(ty, shape)| {
            let pty = purify_type(ty, shape);
            let lifted = pty.is_tuple();
            *ty = pty;
            lifted
        })
        .collect();

    for &bid in block_order {
        let old_params = function.get_block_mut(bid).take_parameters();
        let mut new_params = Vec::with_capacity(old_params.len());
        for (v, ty) in old_params {
            if let Some(pty) = affected.get(&v) {
                new_params.push((v, pty.clone()));
                if is_wl_slice(v, affected) {
                    replacement_tuple_map.insert(v, v); // the param value IS the tuple
                }
            } else {
                new_params.push((v, ty));
            }
        }
        function.get_block_mut(bid).put_parameters(new_params);

        let old_instrs = function.get_block_mut(bid).take_instructions();
        let mut new_instrs: Vec<LocatedOpCode> = Vec::new();
        for located in old_instrs {
            let (op, loc) = located.take();
            match op {
                OpCode::SliceLen { result, slice }
                    if replacement_tuple_map.contains_key(&slice) =>
                {
                    let t = replacement_tuple_map[&slice];
                    new_instrs.push(
                        OpCode::TupleProj {
                            result,
                            tuple: t,
                            idx: 1,
                        }
                        .locate(loc),
                    );
                }

                OpCode::SlicePush {
                    result,
                    slice,
                    values,
                    dir,
                } if is_wl_slice(result, affected) => {
                    assert!(
                        dir == SliceOpDir::Back,
                        "purify_witness_slices: front-push into a witness-length slice is not supported yet"
                    );
                    let phys_ty = type_info.get_value_type(result).clone();
                    let (physical, log_len) = {
                        let mut b =
                            HLInstrBuilder::new(function, ssa, &mut new_instrs, loc.clone());
                        if let Some(&t) = replacement_tuple_map.get(&slice) {
                            let p = b.tuple_proj(t, 0);
                            let ll = b.tuple_proj(t, 1);
                            let one = b.u_const(32, 1);
                            let mut physical = p;
                            let mut cursor = ll;
                            for value in &values {
                                let grown = b.slice_push(physical, vec![*value], SliceOpDir::Back);
                                physical = b.array_set(grown, cursor, *value);
                                cursor = b.add(cursor, one);
                            }
                            (physical, cursor)
                        } else {
                            let base_len = b.slice_len(slice);
                            let mut physical = slice;
                            for value in &values {
                                physical = b.slice_push(physical, vec![*value], SliceOpDir::Back);
                            }
                            let bump = b.u_const(32, values.len() as u128);
                            (physical, b.add(base_len, bump))
                        }
                    };
                    let t = mk_slice_tuple(
                        physical,
                        log_len,
                        phys_ty,
                        function,
                        ssa,
                        &mut new_instrs,
                        loc,
                    );
                    replacement_tuple_map.insert(result, t);
                }

                OpCode::ArrayGet {
                    result,
                    array,
                    index,
                } if replacement_tuple_map.contains_key(&array) => {
                    let t = replacement_tuple_map[&array];
                    let physical = {
                        let mut b =
                            HLInstrBuilder::new(function, ssa, &mut new_instrs, loc.clone());
                        let p = b.tuple_proj(t, 0);
                        let ll = b.tuple_proj(t, 1);
                        b.assert_cmp(CmpKind::Lt, index, ll);
                        p
                    };
                    new_instrs.push(
                        OpCode::ArrayGet {
                            result,
                            array: physical,
                            index,
                        }
                        .locate(loc),
                    );
                }

                OpCode::ArraySet {
                    result,
                    array,
                    index,
                    value,
                } if replacement_tuple_map.contains_key(&array) => {
                    let phys_ty = type_info.get_value_type(result).clone();
                    let t = replacement_tuple_map[&array];
                    let (physical, log_len) = {
                        let mut b =
                            HLInstrBuilder::new(function, ssa, &mut new_instrs, loc.clone());
                        let p = b.tuple_proj(t, 0);
                        let ll = b.tuple_proj(t, 1);
                        b.assert_cmp(CmpKind::Lt, index, ll);
                        (b.array_set(p, index, value), ll)
                    };
                    let t2 = mk_slice_tuple(
                        physical,
                        log_len,
                        phys_ty,
                        function,
                        ssa,
                        &mut new_instrs,
                        loc,
                    );
                    replacement_tuple_map.insert(result, t2);
                }

                OpCode::Alloc { result, value } => {
                    let value = if let Some(&t) = replacement_tuple_map.get(&value) {
                        t
                    } else if affected.contains_key(&result) && !affected.contains_key(&value) {
                        materialize_pure_slice_tuple(
                            value,
                            type_info,
                            function,
                            ssa,
                            &mut new_instrs,
                        )
                    } else {
                        value
                    };
                    new_instrs.push(OpCode::Alloc { result, value }.locate(loc));
                }

                OpCode::Store { ptr, value } => {
                    let value = if let Some(&t) = replacement_tuple_map.get(&value) {
                        t
                    } else if affected.contains_key(&ptr) && !affected.contains_key(&value) {
                        materialize_pure_slice_tuple(
                            value,
                            type_info,
                            function,
                            ssa,
                            &mut new_instrs,
                        )
                    } else {
                        value
                    };
                    new_instrs.push(OpCode::Store { ptr, value }.locate(loc));
                }

                OpCode::Load { result, ptr } => {
                    new_instrs.push(OpCode::Load { result, ptr }.locate(loc));
                    if is_wl_slice(result, affected) {
                        replacement_tuple_map.insert(result, result);
                    }
                }

                OpCode::Call {
                    results,
                    function: callee,
                    args,
                    unconstrained,
                } => {
                    let args = args
                        .into_iter()
                        .map(|a| replacement_tuple_map.get(&a).copied().unwrap_or(a))
                        .collect();
                    for &r in &results {
                        if is_wl_slice(r, affected) {
                            replacement_tuple_map.insert(r, r);
                        }
                    }
                    new_instrs.push(
                        OpCode::Call {
                            results,
                            function: callee,
                            args,
                            unconstrained,
                        }
                        .locate(loc),
                    );
                }

                OpCode::Cast {
                    result,
                    value,
                    target,
                } => {
                    let value = replacement_tuple_map.get(&value).copied().unwrap_or(value);
                    new_instrs.push(
                        OpCode::Cast {
                            result,
                            value,
                            target,
                        }
                        .locate(loc),
                    );
                    if is_wl_slice(result, affected) {
                        replacement_tuple_map.insert(result, result);
                    }
                }

                OpCode::Select {
                    result,
                    cond,
                    if_t,
                    if_f,
                } => {
                    let if_t = replacement_tuple_map.get(&if_t).copied().unwrap_or(if_t);
                    let if_f = replacement_tuple_map.get(&if_f).copied().unwrap_or(if_f);
                    new_instrs.push(
                        OpCode::Select {
                            result,
                            cond,
                            if_t,
                            if_f,
                        }
                        .locate(loc),
                    );
                    if is_wl_slice(result, affected) {
                        replacement_tuple_map.insert(result, result);
                    }
                }

                other => {
                    assert!(
                        !other
                            .get_inputs()
                            .chain(other.get_results())
                            .any(|v| affected.contains_key(v)),
                        "purify_witness_slices: witness-length slice flows into an unsupported \
                         opcode: {other:?}"
                    );
                    new_instrs.push(other.locate(loc));
                }
            }
        }

        let terminator = function
            .get_block_mut(bid)
            .take_terminator()
            .expect("terminated block");
        let new_terminator = match terminator {
            Terminator::Jmp(target, args) => {
                let positions = &lifted_block_args[&target];
                let mut new_args = Vec::with_capacity(args.len());
                for (i, arg) in args.into_iter().enumerate() {
                    if positions.get(i).copied().unwrap_or(false) {
                        let t = replacement_tuple_map.get(&arg).copied().unwrap_or_else(|| {
                            materialize_pure_slice_tuple(
                                arg,
                                type_info,
                                function,
                                ssa,
                                &mut new_instrs,
                            )
                        });
                        new_args.push(t);
                    } else {
                        new_args.push(arg);
                    }
                }
                Terminator::Jmp(target, new_args)
            }
            Terminator::JmpIf(cond, t, f) => Terminator::JmpIf(cond, t, f),
            Terminator::Return(values) => {
                let mut new_return_args = Vec::with_capacity(values.len());
                for (i, v) in values.into_iter().enumerate() {
                    if lifted_returns.get(i).copied().unwrap_or(false) {
                        let t = replacement_tuple_map.get(&v).copied().unwrap_or_else(|| {
                            materialize_pure_slice_tuple(
                                v,
                                type_info,
                                function,
                                ssa,
                                &mut new_instrs,
                            )
                        });
                        new_return_args.push(t);
                    } else {
                        new_return_args.push(v);
                    }
                }
                Terminator::Return(new_return_args)
            }
        };

        function.get_block_mut(bid).put_instructions(new_instrs);
        function.get_block_mut(bid).set_terminator(new_terminator);
    }
}
