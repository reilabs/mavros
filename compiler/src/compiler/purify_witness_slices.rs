//! Purifies witness-length slices into `(physical, log_len)` tuples.

use crate::{
    collections::{HashMap, HashSet},
    compiler::{
        analysis::{
            flow_analysis::FlowAnalysis,
            types::{FunctionTypeInfo, Types},
            witness_info::{FunctionWitnessType, WitnessShape},
            witness_taint_inference::WitnessTaintInference,
        },
        ssa::{
            BlockId, Instruction, Terminator, ValueId,
            hlssa::{
                CmpKind, HLFunction, HLSSA, LocatedOpCode, OpCode, SliceOpDir, Type,
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

    pub fn run(&self, ssa: HLSSA, wti: &WitnessTaintInference) -> (HLSSA, bool) {
        let flow = FlowAnalysis::run(&ssa);
        let types = Types::new().run(&ssa, &flow);

        let mut ssa = ssa;
        let mut changed = false;
        let function_ids: Vec<_> = ssa.get_function_ids().collect();
        for function_id in function_ids {
            if !types.has_function(function_id) {
                continue;
            }
            let Some(fwt) = wti.try_get_function_witness_type(function_id) else {
                continue;
            };
            let type_info = types.get_function(function_id);
            let lift = witness_length_slices(type_info, fwt);
            if lift.is_empty() {
                continue;
            }
            changed = true;

            let block_order: Vec<BlockId> = flow
                .get_function_cfg(function_id)
                .get_domination_pre_order()
                .collect();
            let mut function = ssa.take_function(function_id);
            rewrite_function(&mut function, &mut ssa, type_info, &lift, &block_order);
            ssa.put_function(function_id, function);
        }

        (ssa, changed)
    }
}

fn tuple_ty(slice_ty: &Type) -> Type {
    Type::tuple_of(vec![slice_ty.clone(), Type::u(32)])
}

fn rewrite_function(
    function: &mut HLFunction,
    ssa: &mut HLSSA,
    type_info: &FunctionTypeInfo,
    lift: &HashSet<ValueId>,
    block_order: &[BlockId],
) {
    let mut tuple_of: HashMap<ValueId, ValueId> = HashMap::default();

    // Which parameter positions of each block are lifted — captured before we retype params, so
    // terminators know which jmp args feed a (now tuple-typed) lifted param.
    let lifted_positions: HashMap<BlockId, Vec<bool>> = function
        .get_blocks()
        .map(|(bid, block)| {
            let positions = block
                .get_parameters()
                .map(|(v, _)| lift.contains(v))
                .collect();
            (*bid, positions)
        })
        .collect();

    // Return positions that carry a lifted slice (function return types are updated at the end).
    let mut lifted_return: Vec<bool> = vec![false; function.get_returns().len()];

    for &bid in block_order {
        // -- Retype lifted block params to the tuple type (still one param) --
        let old_params = function.get_block_mut(bid).take_parameters();
        let mut new_params = Vec::with_capacity(old_params.len());
        for (v, ty) in old_params {
            if lift.contains(&v) {
                new_params.push((v, tuple_ty(&ty)));
                tuple_of.insert(v, v); // the param value IS the tuple
            } else {
                new_params.push((v, ty));
            }
        }
        function.get_block_mut(bid).put_parameters(new_params);

        // -- Rewrite instructions --
        let old_instrs = function.get_block_mut(bid).take_instructions();
        let mut new_instrs: Vec<LocatedOpCode> = Vec::new();
        for located in old_instrs {
            let (op, loc) = located.take();
            match op {
                // `SliceLen(s)` on a lifted slice is its carried logical length.
                OpCode::SliceLen { result, slice } if tuple_of.contains_key(&slice) => {
                    let t = tuple_of[&slice];
                    new_instrs.push(
                        OpCode::TupleProj {
                            result,
                            tuple: t,
                            idx: 1,
                        }
                        .locate(loc),
                    );
                }

                // Back-push producing a lifted slice: append at the logical cursor, bump length.
                OpCode::SlicePush {
                    result,
                    slice,
                    values,
                    dir,
                } if lift.contains(&result) => {
                    assert!(
                        dir == SliceOpDir::Back,
                        "purify_witness_slices: front-push into a witness-length slice is not supported yet"
                    );
                    let phys_ty = type_info.get_value_type(result).clone();
                    let (physical, log_len) = {
                        let mut b = HLInstrBuilder::new(function, ssa, &mut new_instrs);
                        if let Some(&t) = tuple_of.get(&slice) {
                            // Chained: source is already a tuple. Grow unconditionally (keeps
                            // capacity a pure `max`-of-consts) and write each value at the witness
                            // logical cursor `log_len`; the witness-indexed `ArraySet` is lowered
                            // to a mux by `witness_array`.
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
                            // Pure source: log_len == capacity, so a plain back-push lands at the
                            // cursor.
                            let base_len = b.slice_len(slice);
                            let mut physical = slice;
                            for value in &values {
                                physical = b.slice_push(physical, vec![*value], SliceOpDir::Back);
                            }
                            let bump = b.u_const(32, values.len() as u128);
                            (physical, b.add(base_len, bump))
                        }
                    };
                    let t = {
                        let mut b = HLInstrBuilder::new(function, ssa, &mut new_instrs);
                        b.mk_tuple(vec![physical, log_len], vec![phys_ty, Type::u(32)])
                    };
                    tuple_of.insert(result, t);
                }

                // Element read: bounds-check against the logical length, read from `physical`.
                OpCode::ArrayGet {
                    result,
                    array,
                    index,
                } if tuple_of.contains_key(&array) => {
                    let t = tuple_of[&array];
                    let physical = {
                        let mut b = HLInstrBuilder::new(function, ssa, &mut new_instrs);
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

                // Element write: bounds-check; length unchanged.
                OpCode::ArraySet {
                    result,
                    array,
                    index,
                    value,
                } if tuple_of.contains_key(&array) => {
                    let phys_ty = type_info.get_value_type(result).clone();
                    let t = tuple_of[&array];
                    let (physical, log_len) = {
                        let mut b = HLInstrBuilder::new(function, ssa, &mut new_instrs);
                        let p = b.tuple_proj(t, 0);
                        let ll = b.tuple_proj(t, 1);
                        b.assert_cmp(CmpKind::Lt, index, ll);
                        (b.array_set(p, index, value), ll)
                    };
                    let t2 = {
                        let mut b = HLInstrBuilder::new(function, ssa, &mut new_instrs);
                        b.mk_tuple(vec![physical, log_len], vec![phys_ty, Type::u(32)])
                    };
                    tuple_of.insert(result, t2);
                }

                other => {
                    assert!(
                        !other.get_inputs().any(|v| tuple_of.contains_key(v)),
                        "purify_witness_slices: witness-length slice flows into an unsupported \
                         opcode in v1: {other:?}"
                    );
                    new_instrs.push(other.locate(loc));
                }
            }
        }

        // -- Terminator: carry the tuple whole; materialize one for a pure slice at a lifted edge --
        let terminator = function
            .get_block_mut(bid)
            .take_terminator()
            .expect("terminated block");
        let new_terminator = match terminator {
            Terminator::Jmp(target, args) => {
                let positions = &lifted_positions[&target];
                let mut new_args = Vec::with_capacity(args.len());
                for (i, arg) in args.into_iter().enumerate() {
                    if positions.get(i).copied().unwrap_or(false) {
                        let t = tuple_of.get(&arg).copied().unwrap_or_else(|| {
                            // Pure slice into a lifted param: a full slice's length is both its
                            // capacity and its logical length, so `(arg, SliceLen(arg))` is exact.
                            let phys_ty = type_info.get_value_type(arg).clone();
                            let mut b = HLInstrBuilder::new(function, ssa, &mut new_instrs);
                            let ll = b.slice_len(arg);
                            b.mk_tuple(vec![arg, ll], vec![phys_ty, Type::u(32)])
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
                let mut new_values = Vec::with_capacity(values.len());
                for (i, v) in values.into_iter().enumerate() {
                    if let Some(&t) = tuple_of.get(&v) {
                        lifted_return[i] = true;
                        new_values.push(t);
                    } else {
                        new_values.push(v);
                    }
                }
                Terminator::Return(new_values)
            }
        };

        function.get_block_mut(bid).put_instructions(new_instrs);
        function.get_block_mut(bid).set_terminator(new_terminator);
    }

    // Update function return types for lifted positions.
    for (i, ty) in function.iter_returns_mut().enumerate() {
        if lifted_return[i] {
            *ty = tuple_ty(ty);
        }
    }
}

fn witness_length_slices(
    type_info: &FunctionTypeInfo,
    fwt: &FunctionWitnessType,
) -> HashSet<ValueId> {
    let mut lift: HashSet<ValueId> = HashSet::default();
    for (&v, shape) in fwt.value_witness_types.iter() {
        if !type_info.get_value_type(v).is_slice() {
            continue;
        }
        if let WitnessShape::Array(top, _) = shape {
            if top.is_witness() {
                lift.insert(v);
            }
        }
    }
    lift
}
