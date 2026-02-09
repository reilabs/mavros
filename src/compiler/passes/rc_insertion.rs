use std::fmt::{Debug, Display};

use itertools::Itertools;
use tracing::{Level, debug, instrument, trace};

use crate::compiler::{
    analysis::{
        liveness::{FunctionLiveness, LivenessAnalysis},
        types::FunctionTypeInfo,
    },
    flow_analysis::CFG,
    ir::r#type::{Type, TypeExpr},
    pass_manager::{DataPoint, Pass, PassInfo, PassManager},
    ssa::{CastTarget, Function, MemOp, OpCode, SSA, Terminator, ValueId},
};

pub struct RCInsertion {}

impl<V: Clone + Display + Debug> Pass<V> for RCInsertion {
    fn pass_info(&self) -> PassInfo {
        PassInfo {
            name: "rc_insertion",
            needs: vec![DataPoint::CFG, DataPoint::Types],
        }
    }

    fn run(&self, ssa: &mut SSA<V>, pass_manager: &PassManager<V>) {
        let cfg = pass_manager.get_cfg();

        let liveness = LivenessAnalysis::new().run(ssa, cfg);

        for (function_id, function) in ssa.iter_functions_mut() {
            let cfg = cfg.get_function_cfg(*function_id);
            let liveness = &liveness.function_liveness[function_id];
            let type_info = pass_manager.get_type_info().get_function(*function_id);
            self.run_function(function, cfg, type_info, &liveness);
        }
    }
}

impl RCInsertion {
    pub fn new() -> Self {
        Self {}
    }

    #[instrument(skip_all, level = Level::DEBUG, name = "RCInsertion::run_function", fields(function = function.get_name()))]
    fn run_function<V: Clone + Display + Debug>(
        &self,
        function: &mut Function<V>,
        cfg: &CFG,
        type_info: &FunctionTypeInfo<V>,
        liveness: &FunctionLiveness,
    ) {
        for (block_id, block) in function.get_blocks_mut() {
            // We're traversing the block backwards, dropping everything that's not live
            // after the currently visited instruction.
            // This means new_instructions will be reversed, so some care is needed when
            // inserting drops.
            let mut currently_live = liveness.block_liveness[block_id].live_out.clone();
            let mut new_instructions = vec![];

            match block.get_terminator().unwrap() {
                Terminator::Return(values) => {
                    for (value, count) in values
                        .iter()
                        .sorted_by_key(|v| v.0)
                        .chunk_by(|v1| *v1)
                        .into_iter()
                    {
                        // This is potentially aliasing, so we need to bump RC by 1 for each repetition.
                        // We then decrease by one, because the original value dies here.
                        let count = count.count() - 1;
                        if self.needs_rc(type_info, value) && count > 0 {
                            new_instructions.push(OpCode::MemOp {
                                kind: MemOp::Bump(count),
                                value: *value
                            });
                        }
                    }
                    // We don't drop these values.
                    // The caller will be responsible for them.
                    currently_live.extend(values);
                }
                Terminator::Jmp(_, values) => {
                    for (value, count) in values
                        .iter()
                        .sorted_by_key(|v| v.0)
                        .chunk_by(|v1| *v1)
                        .into_iter()
                    {
                        // This is a potentially aliasing operation, so we need
                        // to bump RC by 1 for each repetition. Then, if the value
                        // dies here, we remove 1 bump.
                        let mut count: usize = count.count();
                        if !currently_live.contains(value) {
                            count -= 1;
                        }
                        if self.needs_rc(type_info, value) && count > 0 {
                            new_instructions.push(OpCode::MemOp {
                                kind: MemOp::Bump(count),
                                value: *value
                            });
                        }
                    }
                    currently_live.extend(values);
                }
                Terminator::JmpIf(_cond, _, _) => {
                    // Not inserting RCs for the condition, because it's
                    // necessarily boolean and therefore doesn't matter if it lives.
                }
            }

            for instruction in block.take_instructions().into_iter().rev() {
                match &instruction {
                    OpCode::BinaryArithOp { kind: _, result: r, lhs: _, rhs: _ } => {
                        new_instructions.push(instruction.clone());
                        let rcd_inputs = instruction
                            .get_inputs()
                            .filter(|v| self.needs_rc(type_info, v))
                            .copied()
                            .collect_vec();
                        for input in rcd_inputs.iter() {
                            if currently_live.contains(input) {
                                new_instructions.push(OpCode::MemOp {
                                    kind: MemOp::Bump(1),
                                    value: *input
                                });
                            }
                        }
                        if self.needs_rc(type_info, r) && !currently_live.contains(r) {
                            panic!("ICE: Result of BinaryArithOp is immediately dropped. This is a bug.")
                        }
                        currently_live.extend(rcd_inputs);
                    }
                    OpCode::UnboxField { result: _, value: v } => {
                        if !currently_live.contains(v) {
                            new_instructions.push(OpCode::MemOp {
                                kind: MemOp::Drop,
                                value: *v
                            });
                        }
                        new_instructions.push(instruction.clone());
                        currently_live.insert(*v);
                    }
                    OpCode::MulConst { result: r, const_val: _, var: v } => {
                        if currently_live.contains(v) {
                            new_instructions.push(OpCode::MemOp {
                                kind: MemOp::Bump(1),
                                value: *v
                            });
                        }
                        if !currently_live.contains(r) {
                            panic!("ICE: Result of MulConst is immediately dropped. This is a bug.")
                        }
                        new_instructions.push(instruction.clone());
                        currently_live.insert(*v);
                    }
                    OpCode::PureToWitnessRef { result: r, value: _, result_annotation: _ } => {
                        if !currently_live.contains(r) {
                            panic!("ICE: Result of PureToWitnessRef is immediately dropped. This is a bug.")
                        }
                        new_instructions.push(instruction.clone());
                    }
                    OpCode::TupleProj { 
                        result, tuple, idx: _,
                    } => {
                        if !currently_live.contains(tuple) {
                            // The tuple dies here, so we drop it _after_ the read.
                            new_instructions.push(OpCode::MemOp {
                                kind: MemOp::Drop,
                                value: *tuple
                            });
                        }
                        if self.needs_rc(type_info, result) {
                            if currently_live.contains(result) {
                                // The result gets a bump to the RC counter, because
                                // it's now both accessed here and in the array.
                                new_instructions.push(OpCode::MemOp {
                                    kind: MemOp::Bump(1),
                                    value: *result
                                });
                            } else {
                                panic!(
                                    "ICE: Result of TupleProj (V{} in block {}) is not live. This is a bug.",
                                    result.0, block_id.0
                                )
                            }
                        } else {
                            trace!(
                                "TupleProj: result={} of type {:?} does not need RC",
                                result.0,
                                type_info.get_value_type(*result)
                            );
                        }
                        new_instructions.push(instruction.clone());
                        currently_live.insert(*tuple);
                    }
                    OpCode::Cast { result: r, value: v, target: CastTarget::Nop } => {
                        // Nop cast aliases result to input in codegen (same frame position).
                        if self.needs_rc(type_info, v) {
                            if !currently_live.contains(r) {
                                panic!("ICE: Result of Cast::Nop is immediately dropped. This is a bug.");
                            }
                            if currently_live.contains(v) {
                                // Both input and output are live — two refs to the same boxed value.
                                new_instructions.push(OpCode::MemOp {
                                    kind: MemOp::Bump(1),
                                    value: *v
                                });
                            }
                            // If only result is live (input dead), no RC op needed — single alias.
                        }
                        currently_live.insert(*v);
                        new_instructions.push(instruction);
                    }
                    // These need to mark their inputs as live, but do not need to bump RCs
                    OpCode::AssertEq { lhs: _, rhs: _ }
                    | OpCode::Cast { result: _, value: _, target: _ }
                    | OpCode::Cmp { kind: _, result: _, lhs: _, rhs: _ }
                    | OpCode::Truncate { result: _, value: _, to_bits: _, from_bits: _ }
                    | OpCode::AssertR1C { a: _, b: _, c: _ }
                    | OpCode::Constrain { a: _, b: _, c: _ }
                    | OpCode::WriteWitness { result: _, value: _, witness_annotation: _ }
                    | OpCode::NextDCoeff { result: _ }
                    | OpCode::Lookup { target: _, keys: _, results: _ }
                    | OpCode::DLookup { target: _, keys: _, results: _ }
                    | OpCode::Not { result: _, value: _ }
                    | OpCode::Todo { .. } => {
                        let rcd_inputs = instruction
                            .get_inputs()
                            .filter(|v| self.needs_rc(type_info, v))
                            .copied()
                            .collect_vec();
                        for input in rcd_inputs.iter() {
                            if !currently_live.contains(input) {
                                new_instructions.push(OpCode::MemOp {
                                    kind: MemOp::Drop,
                                    value: *input
                                });
                            }
                        }
                        currently_live.extend(rcd_inputs);
                        new_instructions.push(instruction)
                    }
                    OpCode::FreshWitness { result: r, result_type: _ } => {
                        // it is possible that fresh_witness is only used for the side effect,
                        // but the actual value is not used.
                        if !currently_live.contains(r) {
                            new_instructions.push(OpCode::MemOp {
                                kind: MemOp::Drop,
                                value: *r
                            });
                        }
                        new_instructions.push(instruction.clone());
                    }
                    OpCode::BumpD { matrix: _, variable: v, sensitivity: _ } => {
                        if !currently_live.contains(v) {
                            new_instructions.push(OpCode::MemOp {
                                kind: MemOp::Drop,
                                value: *v
                            });
                        }
                        new_instructions.push(instruction.clone());
                        currently_live.insert(*v);
                    }
                    OpCode::MkSeq { result, elems: inputs, seq_type: _, elem_type } => {
                        // MkSeq should return an RC counter of 1.
                        new_instructions.push(instruction.clone());
                        // This happens after we push the instruction, so that it
                        // happens before after reversal.
                        if self.type_needs_rc(elem_type) {
                            // This is an aliasing operation. Each use in the array needs a bump.
                            // We then decrease by one, if the original value dies here.
                            for (input, count) in inputs
                                .iter()
                                .sorted_by_key(|v| v.0)
                                .chunk_by(|v1| *v1)
                                .into_iter()
                            {
                                let mut count = count.count();
                                if !currently_live.contains(input) {
                                    count -= 1;
                                }
                                if count > 0 {
                                    new_instructions
                                        .push(OpCode::MemOp {
                                            kind: MemOp::Bump(count),
                                            value: *input
                                        });
                                }
                            }
                        }
                        if !currently_live.contains(result) {
                            panic!("ICE: Result of MkSeq is immediately dropped. This is a bug.")
                            // The line below is the temporary solution if we run into this ever.
                            // It should be debugged properly though, we expect DCE to sweep this
                            // entire instruction.
                            // new_instructions.push(OpCode::MemOp(MemOp::DropAndSweep, *result));
                        }
                        currently_live.extend(inputs);
                    }
                    OpCode::Alloc { result: _, elem_type: _, result_annotation: _ } => todo!(),
                    OpCode::Store { ptr: _, value: _ } => todo!(),
                    OpCode::Load { result: _, ptr: _ } => todo!(),
                    OpCode::Call { results: returns, function: _, args: params, is_unconstrained: _ } => {
                        // Functions take parameters with the correct RC counter
                        // and return results with the correct RC counter.
                        // That means we need to give a bump to each param before the call
                        // and we need to drop any unused returns after the call.
                        for return_id in returns {
                            if self.needs_rc(type_info, return_id)
                                && !currently_live.contains(return_id)
                            {
                                new_instructions.push(OpCode::MemOp {
                                    kind: MemOp::Drop,
                                    value: *return_id
                                });
                            }
                        }
                        new_instructions.push(instruction.clone());
                        // This is an aliasing operation. We need a +1 bump for each use.
                        // We then decrease by one, if the original value dies here.
                        for (param, count) in params
                            .iter()
                            .sorted_by_key(|v| v.0)
                            .chunk_by(|v1| *v1)
                            .into_iter()
                        {
                            let mut count = count.count();
                            if !currently_live.contains(param) {
                                count -= 1;
                            }
                            if self.needs_rc(type_info, param) && count > 0 {
                                new_instructions.push(OpCode::MemOp {
                                    kind: MemOp::Bump(count),
                                    value: *param
                                });
                            }
                        }
                        currently_live.extend(params);
                    }
                    OpCode::ArrayGet { result, array, index: _ } => {
                        if !currently_live.contains(array) {
                            // The array dies here, so we drop it _after_ the read.
                            new_instructions.push(OpCode::MemOp {
                                kind: MemOp::Drop,
                                value: *array
                            });
                        }
                        if self.needs_rc(type_info, result) {
                            if currently_live.contains(result) {
                                // The result gets a bump to the RC counter, because
                                // it's now both accessed here and in the array.
                                new_instructions.push(OpCode::MemOp {
                                    kind: MemOp::Bump(1),
                                    value: *result
                                });
                            } else {
                                panic!(
                                    "ICE: Result of ArrayGet (V{} in block {}) is not live. This is a bug.",
                                    result.0, block_id.0
                                )
                            }
                        } else {
                            trace!(
                                "ArrayGet: result={} of type {:?} does not need RC",
                                result.0,
                                type_info.get_value_type(*result)
                            );
                        }
                        new_instructions.push(instruction.clone());
                        currently_live.insert(*array);
                    }
                    OpCode::SliceLen { result: _, slice } => {
                        // SliceLen returns u32, which doesn't need RC
                        // But we need to keep the slice alive if it's currently live
                        if !currently_live.contains(slice) {
                            // The slice dies here, so we drop it _after_ the read.
                            new_instructions.push(OpCode::MemOp {
                                kind: MemOp::Drop,
                                value: *slice
                            });
                        }
                        new_instructions.push(instruction.clone());
                        currently_live.insert(*slice);
                    }
                    OpCode::ReadGlobal { result: r, offset: _, result_type: _ } => {
                        if self.needs_rc(type_info, r) {
                            if !currently_live.contains(r) {
                                panic!("ICE: Result of ReadGlobal is immediately dropped. This is a bug.")
                            }
                            new_instructions.push(OpCode::MemOp {
                                kind: MemOp::Bump(1),
                                value: *r
                            });
                        }
                        new_instructions.push(instruction.clone());
                        currently_live.insert(*r);
                    }
                    OpCode::ArraySet { result, array, index: _, value } => {
                        new_instructions.push(instruction.clone());
                        if currently_live.contains(array) {
                            // Array set will decrease the RC and oportunistically reuse the storage,
                            // if it notices a refcount of 0. So we need to bump _before_
                            // we enter it.
                            new_instructions.push(OpCode::MemOp {
                                kind: MemOp::Bump(1),
                                value: *array
                            });
                            if self.needs_rc(type_info, array) {
                                new_instructions.push(OpCode::MemOp {
                                    kind: MemOp::Bump(1),
                                    value: *array
                                });
                            }
                        }
                        if self.needs_rc(type_info, value) && currently_live.contains(value) {
                            new_instructions.push(OpCode::MemOp {
                                kind: MemOp::Bump(1),
                                value: *value
                            })
                        }
                        if !currently_live.contains(result) {
                            panic!("ICE: Result of ArraySet is immediately dropped. This is a bug.")
                            // The line below is the temporary solution if we run into this ever.
                            // It should be debugged properly though, we expect DCE to sweep this
                            // entire instruction.
                            // new_instructions.push(OpCode::MemOp(MemOp::Drop, *result));
                        }
                        currently_live.extend(vec![*array, *value]);
                    }
                    OpCode::SlicePush { result, slice, values, dir: _ } => {
                        new_instructions.push(instruction.clone());
                        if currently_live.contains(slice) {
                            // Slice push will decrease the RC and oportunistically reuse the storage,
                            // if it notices a refcount of 0. So we need to bump _before_
                            // we enter it.
                            new_instructions.push(OpCode::MemOp {
                                kind: MemOp::Bump(1),
                                value: *slice
                            });
                            if self.needs_rc(type_info, slice) {
                                new_instructions.push(OpCode::MemOp {
                                    kind: MemOp::Bump(1),
                                    value: *slice
                                });
                            }
                        }
                        let slice_type = type_info.get_value_type(*slice);
                        let elem_type = slice_type.get_array_element();
                        if self.type_needs_rc(&elem_type) {
                            // This is an aliasing operation. Each use in the slice needs a bump.
                            // We then decrease by one, if the original value dies here.
                            for (value, count) in values
                                .iter()
                                .sorted_by_key(|v| v.0)
                                .chunk_by(|v1| *v1)
                                .into_iter()
                            {
                                let mut count = count.count();
                                if !currently_live.contains(value) {
                                    count -= 1;
                                }
                                if count > 0 {
                                    new_instructions
                                        .push(OpCode::MemOp {
                                            kind: MemOp::Bump(count),
                                            value: *value
                                        });
                                }
                            }
                        }
                        if !currently_live.contains(result) {
                            panic!("ICE: Result of SlicePush is immediately dropped. This is a bug.")
                        }
                        let mut live_vals = vec![*slice];
                        live_vals.extend(values.iter().copied());
                        currently_live.extend(live_vals);
                    }
                    OpCode::InitGlobal { global: _, value: v } => {
                        // InitGlobal stores value into a global slot.
                        // If the value needs RC, bump it since the global now holds a reference.
                        let v = *v;
                        if self.needs_rc(type_info, &v) && currently_live.contains(&v) {
                            new_instructions.push(OpCode::MemOp {
                                kind: MemOp::Bump(1),
                                value: v,
                            });
                        }
                        new_instructions.push(instruction);
                        currently_live.insert(v);
                    }
                    OpCode::DropGlobal { global: _ } => {
                        // DropGlobal IS the RC drop itself, just pass through.
                        new_instructions.push(instruction);
                    }
                    OpCode::Select { result: _, cond: _, if_t: v1, if_f: v2 } => {
                        if self.needs_rc(type_info, v1) || self.needs_rc(type_info, v2) {
                            panic!("Unsupported yet");
                        }
                        new_instructions.push(instruction);
                    }
                    OpCode::ToBits { result: r, value: _, endianness: _, count: _ } => {
                        if !currently_live.contains(r) {
                            // We contend with this, because ToBits can be used for a range check.
                            new_instructions.push(OpCode::MemOp {
                                kind: MemOp::Drop,
                                value: *r
                            });
                        }
                        // ToBits should return an RC counter of 1.
                        new_instructions.push(instruction);
                    }
                    OpCode::ToRadix { result: r, value: _, radix: _, endianness: _, count: _ } => {
                        if !currently_live.contains(r) {
                            // We contend with this, because ToRadix can be used for a range check.
                            new_instructions.push(OpCode::MemOp {
                                kind: MemOp::Drop,
                                value: *r
                            });
                        }
                        // ToRadix should return an RC counter of 1.
                        new_instructions.push(instruction);
                    }
                    OpCode::MemOp { kind: _mem_op, value: _value_id } => todo!(),
                    OpCode::Rangecheck { value: _, max_bits: _ } => {
                        new_instructions.push(instruction);
                    }
                    OpCode::MkTuple { 
                        result, 
                        elems, 
                        element_types, 
                    } => {
                        new_instructions.push(instruction.clone());
                        for (input, group) in elems
                            .iter()
                            .zip(element_types)
                            .sorted_by_key(|(v, _)| v.0)
                            .chunk_by(|(v, _)| *v)
                            .into_iter()
                        {
                            let items: Vec<_> = group.collect();
                            let count = items.iter().count();
                            let (_, elem_type) = items[0];
                            
                            if self.type_needs_rc(elem_type) {
                                let mut count = count;
                                if !currently_live.contains(input) {
                                    count -= 1;
                                }
                                if count > 0 {
                                    new_instructions.push(OpCode::MemOp {
                                        kind: MemOp::Bump(count),
                                        value: *input
                                    });
                                }
                            }
                        }
                            
                        if !currently_live.contains(result) {
                            panic!("ICE: Result of MkTuple is immediately dropped. This is a bug.")
                        }
                        currently_live.extend(elems);
                    }
                }
            }
            for param in block.get_parameter_values() {
                if self.needs_rc(type_info, param) && !currently_live.contains(param) {
                    new_instructions.push(OpCode::MemOp {
                        kind: MemOp::Drop,
                        value: *param
                    });
                }
            }
            block.put_instructions(new_instructions.into_iter().rev().collect());
        }

        for (source, target) in cfg.get_edges() {
            let live_out_source = &liveness.block_liveness[&source].live_out;
            let live_in_target = &liveness.block_liveness[&target].live_in;
            let diff = live_out_source
                .difference(live_in_target)
                .filter(|v| self.needs_rc(type_info, v))
                .collect_vec();
            trace!(
                "Dying along edge {} -> {}: [{}]",
                source.0,
                target.0,
                diff.iter().map(|v| v.0).join(", ")
            );
            if diff.is_empty() {
                continue;
            }
            let intermediate_block = function.add_block();
            match function.get_block_mut(source).take_terminator().unwrap() {
                Terminator::JmpIf(cond, t1, t2) => {
                    let t1 = if t1 == target { intermediate_block } else { t1 };
                    let t2 = if t2 == target { intermediate_block } else { t2 };
                    function
                        .get_block_mut(source)
                        .set_terminator(Terminator::JmpIf(cond, t1, t2));
                }
                Terminator::Jmp(_, _) => {
                    debug!("Will panic: {} -> {}", source.0, target.0);
                    debug!(
                        "Source live out: [{}]",
                        liveness.block_liveness[&source]
                            .live_out
                            .iter()
                            .map(|v| v.0)
                            .join(", ")
                    );
                    debug!(
                        "Target live in: [{}]",
                        liveness.block_liveness[&target]
                            .live_in
                            .iter()
                            .map(|v| v.0)
                            .join(", ")
                    );
                    debug!("Difference: [{}]", diff.iter().map(|v| v.0).join(", "));
                    panic!(
                        "ICE: Jmp is not expected - the value should have died in the source block."
                    );
                }
                Terminator::Return(_) => {
                    panic!("ICE: Impossible, CFG says there's an edge here.");
                }
            }
            let intermediate = function.get_block_mut(intermediate_block);
            intermediate.set_terminator(Terminator::Jmp(target, vec![]));
            for value in diff {
                intermediate.push_instruction(OpCode::MemOp {
                    kind: MemOp::Drop,
                    value: *value
                });
            }
        }
    }

    fn needs_rc<V: Clone + Display>(
        &self,
        type_info: &FunctionTypeInfo<V>,
        value: &ValueId,
    ) -> bool {
        let value_type = type_info.get_value_type(*value);
        self.type_needs_rc(&value_type)
    }

    fn type_needs_rc<V: Clone>(&self, value_type: &Type<V>) -> bool {
        match &value_type.expr {
            TypeExpr::Ref(_) => true,
            TypeExpr::Array(_, _) => true,
            TypeExpr::Slice(_) => true,
            TypeExpr::Field => false,
            TypeExpr::U(_) => false,
            TypeExpr::WitnessRef => true,
            TypeExpr::Tuple(_) => true,
            TypeExpr::Function => false,
        }
    }
}
