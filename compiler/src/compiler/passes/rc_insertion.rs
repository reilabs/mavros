//! Handles the insertion of reference count increments (`Bump(n)` which does `rc += n`) and
//! decrements (`Drop`, which does `rc -= 1`) into the SSA in order to ensure correct retention of
//! heap-allocated values at runtime.
//!
//! It also handles cases where values die along edges instead of within a block to actually perform
//! the necessary decrements.

use crate::{collections::HashMap, compiler::util::ice_non_elided_tuple};
use itertools::Itertools;
use tracing::{Level, debug, instrument, trace};

use crate::compiler::{
    analysis::{
        flow_analysis::{CFG, FlowAnalysis},
        liveness::{FunctionLiveness, LivenessAnalysis},
        types::{FunctionTypeInfo, TypeInfo},
    },
    pass_manager::{Analysis, AnalysisId, AnalysisStore, Pass},
    ssa::{
        Instruction, Located, SourceLocation, Terminator, ValueId,
        hlssa::{CastTarget, HLFunction, HLSSA, OpCode, RefCountOp, Type, TypeExpr},
    },
};

pub struct RCInsertion {}

impl Pass for RCInsertion {
    fn name(&self) -> &'static str {
        "rc_insertion"
    }

    fn needs(&self) -> Vec<AnalysisId> {
        vec![FlowAnalysis::id(), TypeInfo::id()]
    }

    fn run(&self, ssa: &mut HLSSA, store: &AnalysisStore) {
        let cfg = store.get::<FlowAnalysis>();

        let liveness = LivenessAnalysis::new().run(ssa, cfg);

        for (function_id, function) in ssa.iter_functions_mut() {
            let cfg = cfg.get_function_cfg(*function_id);
            let liveness = &liveness.function_liveness[function_id];
            let type_info = store.get::<TypeInfo>().get_function(*function_id);
            self.run_function(function, cfg, type_info, &liveness);
        }
    }
}

impl RCInsertion {
    pub fn new() -> Self {
        Self {}
    }

    #[instrument(skip_all, level = Level::DEBUG, name = "RCInsertion::run_function", fields(function = function.get_name()))]
    fn run_function(
        &self,
        function: &mut HLFunction,
        cfg: &CFG,
        type_info: &FunctionTypeInfo,
        liveness: &FunctionLiveness,
    ) {
        let value_locations = Self::value_source_locations(function);
        for (block_id, block) in function.get_blocks_mut() {
            // We're traversing the block backwards, dropping everything that's not live
            // after the currently visited instruction.
            // This means new_instructions will be reversed, so some care is needed when
            // inserting drops.
            let mut currently_live = liveness.block_liveness[block_id].live_out.clone();
            let mut new_instructions = vec![];
            let block_location = block
                .get_instructions_with_source_locations()
                .next()
                .map(|(_, location)| location.clone());

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
                            Self::push_mem_op(
                                &mut new_instructions,
                                RefCountOp::Bump(count),
                                *value,
                                &value_locations,
                                block_location.as_ref(),
                            );
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
                            Self::push_mem_op(
                                &mut new_instructions,
                                RefCountOp::Bump(count),
                                *value,
                                &value_locations,
                                block_location.as_ref(),
                            );
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
                let instruction_location = instruction.location().clone();
                match &*instruction {
                    OpCode::BinaryArithOp {
                        kind: _,
                        result: r,
                        lhs: _,
                        rhs: _,
                    } => {
                        new_instructions.push(instruction.clone());
                        let rcd_inputs = instruction
                            .get_inputs()
                            .filter(|v| self.needs_rc(type_info, v))
                            .copied()
                            .collect_vec();
                        for (input, group) in rcd_inputs
                            .iter()
                            .sorted_by_key(|v| v.0)
                            .chunk_by(|v| *v)
                            .into_iter()
                        {
                            let mut count = group.count();
                            if !currently_live.contains(input) {
                                count -= 1;
                            }
                            if count > 0 {
                                Self::push_mem_op(
                                    &mut new_instructions,
                                    RefCountOp::Bump(count),
                                    *input,
                                    &value_locations,
                                    Some(&instruction_location),
                                );
                            }
                        }
                        if self.needs_rc(type_info, r) && !currently_live.contains(r) {
                            panic!(
                                "ICE: Result of BinaryArithOp is immediately dropped. This is a bug."
                            )
                        }
                        currently_live.extend(rcd_inputs);
                    }
                    OpCode::MulConst {
                        result: r,
                        const_val: _,
                        var: v,
                    } => {
                        if currently_live.contains(v) {
                            Self::push_mem_op(
                                &mut new_instructions,
                                RefCountOp::Bump(1),
                                *v,
                                &value_locations,
                                Some(&instruction_location),
                            );
                        }
                        if !currently_live.contains(r) {
                            panic!("ICE: Result of MulConst is immediately dropped. This is a bug.")
                        }
                        new_instructions.push(instruction.clone());
                        currently_live.insert(*v);
                    }
                    OpCode::TupleProj { .. } | OpCode::TupleRefProj { .. } => {
                        ice_non_elided_tuple()
                    }
                    OpCode::Cast {
                        result: r,
                        value: v,
                        target,
                    } if matches!(target, CastTarget::Nop | CastTarget::ArrayToSlice)
                        || (type_info.get_value_type(*v).is_witness_of()
                            && type_info.get_value_type(*r).is_witness_of()) =>
                    {
                        // Nop/alias cast: result aliases input in codegen (same frame position).
                        // WitnessOf(A) -> WitnessOf(B) casts are also no-ops at runtime.
                        if self.needs_rc(type_info, v) {
                            if !currently_live.contains(r) {
                                panic!(
                                    "ICE: Result of aliasing Cast is immediately dropped. This is a bug."
                                );
                            }
                            if currently_live.contains(v) {
                                // Both input and output are live — two refs to the same boxed value.
                                Self::push_mem_op(
                                    &mut new_instructions,
                                    RefCountOp::Bump(1),
                                    *v,
                                    &value_locations,
                                    Some(&instruction_location),
                                );
                            }
                            // If only result is live (input dead), no RC op needed — single alias.
                        }
                        currently_live.insert(*v);
                        new_instructions.push(instruction);
                    }
                    OpCode::Cast {
                        target: CastTarget::ValueOf,
                        ..
                    } => {
                        panic!("ICE: ValueOf cast should not appear at this stage");
                    }
                    // These need to mark their inputs as live, but do not need to bump RCs
                    OpCode::Assert { .. }
                    | OpCode::AssertCmp { .. }
                    | OpCode::Cast {
                        result: _,
                        value: _,
                        target: _,
                    }
                    | OpCode::Cmp {
                        kind: _,
                        result: _,
                        lhs: _,
                        rhs: _,
                    }
                    | OpCode::SExt {
                        result: _,
                        value: _,
                        from_bits: _,
                        to_bits: _,
                    }
                    | OpCode::BitRange {
                        result: _,
                        value: _,
                        offset: _,
                        width: _,
                    }
                    | OpCode::AssertR1C { a: _, b: _, c: _ }
                    | OpCode::Constrain { a: _, b: _, c: _ }
                    | OpCode::WriteWitness {
                        result: _,
                        value: _,
                        pinned: _,
                    }
                    | OpCode::NextDCoeff { result: _ }
                    | OpCode::Lookup {
                        target: _,
                        args: _,
                        flag: _,
                    }
                    | OpCode::DLookup {
                        target: _,
                        args: _,
                        flag: _,
                    }
                    | OpCode::Not {
                        result: _,
                        value: _,
                    }
                    | OpCode::Spread { .. }
                    | OpCode::Unspread { .. }
                    | OpCode::Todo { .. } => {
                        let rcd_inputs = instruction
                            .get_inputs()
                            .filter(|v| self.needs_rc(type_info, v))
                            .copied()
                            .collect_vec();
                        for input in rcd_inputs.iter() {
                            if !currently_live.contains(input) {
                                Self::push_mem_op(
                                    &mut new_instructions,
                                    RefCountOp::Drop,
                                    *input,
                                    &value_locations,
                                    Some(&instruction_location),
                                );
                            }
                        }
                        currently_live.extend(rcd_inputs);
                        new_instructions.push(instruction)
                    }
                    OpCode::FreshWitness {
                        result: r,
                        result_type: _,
                    } => {
                        // it is possible that fresh_witness is only used for the side effect,
                        // but the actual value is not used.
                        if !currently_live.contains(r) {
                            Self::push_mem_op(
                                &mut new_instructions,
                                RefCountOp::Drop,
                                *r,
                                &value_locations,
                                Some(&instruction_location),
                            );
                        }
                        new_instructions.push(instruction.clone());
                    }
                    OpCode::BumpD {
                        matrix: _,
                        variable: v,
                        sensitivity: _,
                    } => {
                        if !currently_live.contains(v) {
                            Self::push_mem_op(
                                &mut new_instructions,
                                RefCountOp::Drop,
                                *v,
                                &value_locations,
                                Some(&instruction_location),
                            );
                        }
                        new_instructions.push(instruction.clone());
                        currently_live.insert(*v);
                    }
                    OpCode::MkSeq {
                        result,
                        elems: inputs,
                        seq_type: _,
                        elem_type,
                    } => {
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
                                    Self::push_mem_op(
                                        &mut new_instructions,
                                        RefCountOp::Bump(count),
                                        *input,
                                        &value_locations,
                                        Some(&instruction_location),
                                    );
                                }
                            }
                        }
                        if !currently_live.contains(result) {
                            panic!("ICE: Result of MkSeq is immediately dropped. This is a bug.")
                            // The line below is the temporary solution if we run into this ever.
                            // It should be debugged properly though, we expect DCE to sweep this
                            // entire instruction.
                            // Insert a drop/sweep for `result` here if this case becomes valid.
                        }
                        currently_live.extend(inputs);
                    }
                    OpCode::MkSeqOfBlob {
                        result,
                        element_type,
                        blob: _,
                    } => {
                        new_instructions.push(instruction.clone());
                        assert!(
                            !self.type_needs_rc(element_type),
                            "MkSeqOfBlob only supports scalar element types"
                        );
                        if !currently_live.contains(result) {
                            panic!(
                                "ICE: Result of MkSeqOfBlob is immediately dropped. This is a bug."
                            )
                        }
                    }
                    OpCode::MkRepeated {
                        result,
                        element,
                        seq_type: _,
                        count,
                        elem_type,
                    } => {
                        // MkRepeated should return an RC counter of 1.
                        new_instructions.push(instruction.clone());
                        if self.type_needs_rc(elem_type) {
                            // The element is aliased `count` times into the seq.
                            // Each repetition needs a bump; we then decrease by one
                            // if the original value dies here.
                            let mut bump = *count;
                            if !currently_live.contains(element) {
                                bump -= 1;
                            }
                            if bump > 0 {
                                Self::push_mem_op(
                                    &mut new_instructions,
                                    RefCountOp::Bump(bump),
                                    *element,
                                    &value_locations,
                                    Some(&instruction_location),
                                );
                            }
                        }
                        if !currently_live.contains(result) {
                            panic!(
                                "ICE: Result of MkRepeated is immediately dropped. This is a bug."
                            )
                        }
                        currently_live.insert(*element);
                    }
                    OpCode::Alloc { result, value } => {
                        let value = *value;
                        if !currently_live.contains(result) {
                            Self::push_mem_op(
                                &mut new_instructions,
                                RefCountOp::Drop,
                                *result,
                                &value_locations,
                                Some(&instruction_location),
                            );
                        }
                        new_instructions.push(instruction);
                        if self.needs_rc(type_info, &value) && currently_live.contains(&value) {
                            Self::push_mem_op(
                                &mut new_instructions,
                                RefCountOp::Bump(1),
                                value,
                                &value_locations,
                                Some(&instruction_location),
                            );
                        }
                        currently_live.insert(value);
                    }
                    OpCode::Store { ptr, value } => {
                        // In forward order: bump value if still live (aliased into cell),
                        // store, then drop ptr if it dies here.
                        // In reverse: push the drop first, then store, then bump.
                        let ptr = *ptr;
                        let value = *value;
                        if !currently_live.contains(&ptr) {
                            Self::push_mem_op(
                                &mut new_instructions,
                                RefCountOp::Drop,
                                ptr,
                                &value_locations,
                                Some(&instruction_location),
                            );
                        }
                        new_instructions.push(instruction);
                        if self.needs_rc(type_info, &value) && currently_live.contains(&value) {
                            Self::push_mem_op(
                                &mut new_instructions,
                                RefCountOp::Bump(1),
                                value,
                                &value_locations,
                                Some(&instruction_location),
                            );
                        }
                        currently_live.insert(ptr);
                        currently_live.insert(value);
                    }
                    OpCode::Load { result, ptr } => {
                        if !currently_live.contains(ptr) {
                            Self::push_mem_op(
                                &mut new_instructions,
                                RefCountOp::Drop,
                                *ptr,
                                &value_locations,
                                Some(&instruction_location),
                            );
                        }
                        if self.needs_rc(type_info, result) {
                            if currently_live.contains(result) {
                                Self::push_mem_op(
                                    &mut new_instructions,
                                    RefCountOp::Bump(1),
                                    *result,
                                    &value_locations,
                                    Some(&instruction_location),
                                );
                            } else {
                                panic!(
                                    "ICE: Result of Load (V{} in block {}) is not live. This is a bug.",
                                    result.0, block_id.0
                                );
                            }
                        }
                        new_instructions.push(instruction.clone());
                        currently_live.insert(*ptr);
                    }
                    OpCode::Call {
                        results: returns,
                        function: _,
                        args: params,
                        unconstrained: _,
                    } => {
                        // Functions take parameters with the correct RC counter
                        // and return results with the correct RC counter.
                        // That means we need to give a bump to each param before the call
                        // and we need to drop any unused returns after the call.
                        for return_id in returns {
                            if self.needs_rc(type_info, return_id)
                                && !currently_live.contains(return_id)
                            {
                                Self::push_mem_op(
                                    &mut new_instructions,
                                    RefCountOp::Drop,
                                    *return_id,
                                    &value_locations,
                                    Some(&instruction_location),
                                );
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
                                Self::push_mem_op(
                                    &mut new_instructions,
                                    RefCountOp::Bump(count),
                                    *param,
                                    &value_locations,
                                    Some(&instruction_location),
                                );
                            }
                        }
                        currently_live.extend(params);
                    }
                    OpCode::ArrayGet {
                        result,
                        array,
                        index: _,
                    } => {
                        if !currently_live.contains(array) && self.needs_rc(type_info, array) {
                            // The array dies here, so we drop it _after_ the read.
                            // Blobs are not RC'd and need no drop.
                            Self::push_mem_op(
                                &mut new_instructions,
                                RefCountOp::Drop,
                                *array,
                                &value_locations,
                                Some(&instruction_location),
                            );
                        }
                        if self.needs_rc(type_info, result) {
                            if currently_live.contains(result) {
                                // The result gets a bump to the RC counter, because
                                // it's now both accessed here and in the array.
                                Self::push_mem_op(
                                    &mut new_instructions,
                                    RefCountOp::Bump(1),
                                    *result,
                                    &value_locations,
                                    Some(&instruction_location),
                                );
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
                            Self::push_mem_op(
                                &mut new_instructions,
                                RefCountOp::Drop,
                                *slice,
                                &value_locations,
                                Some(&instruction_location),
                            );
                        }
                        new_instructions.push(instruction.clone());
                        currently_live.insert(*slice);
                    }
                    OpCode::ReadGlobal {
                        result: r,
                        offset: _,
                        result_type: _,
                    } => {
                        if self.needs_rc(type_info, r) {
                            if !currently_live.contains(r) {
                                panic!(
                                    "ICE: Result of ReadGlobal is immediately dropped. This is a bug."
                                )
                            }
                            Self::push_mem_op(
                                &mut new_instructions,
                                RefCountOp::Bump(1),
                                *r,
                                &value_locations,
                                Some(&instruction_location),
                            );
                        }
                        new_instructions.push(instruction.clone());
                        currently_live.insert(*r);
                    }
                    OpCode::ArraySet {
                        result,
                        array,
                        index: _,
                        value,
                    } => {
                        new_instructions.push(instruction.clone());
                        if currently_live.contains(array) {
                            // Array set will decrease the RC and oportunistically reuse the storage,
                            // if it notices a refcount of 0. So we need to bump _before_
                            // we enter it.
                            Self::push_mem_op(
                                &mut new_instructions,
                                RefCountOp::Bump(1),
                                *array,
                                &value_locations,
                                Some(&instruction_location),
                            );
                        }
                        if self.needs_rc(type_info, value) && currently_live.contains(value) {
                            Self::push_mem_op(
                                &mut new_instructions,
                                RefCountOp::Bump(1),
                                *value,
                                &value_locations,
                                Some(&instruction_location),
                            )
                        }
                        if !currently_live.contains(result) {
                            panic!("ICE: Result of ArraySet is immediately dropped. This is a bug.")
                            // The line below is the temporary solution if we run into this ever.
                            // It should be debugged properly though, we expect DCE to sweep this
                            // entire instruction.
                            // Insert a drop for `result` here if this case becomes valid.
                        }
                        currently_live.extend(vec![*array, *value]);
                    }
                    OpCode::SlicePush {
                        result,
                        slice,
                        values,
                        dir: _,
                    } => {
                        new_instructions.push(instruction.clone());
                        if currently_live.contains(slice) {
                            // Slice push will decrease the RC and oportunistically reuse the storage,
                            // if it notices a refcount of 0. So we need to bump _before_ we enter it.
                            Self::push_mem_op(
                                &mut new_instructions,
                                RefCountOp::Bump(1),
                                *slice,
                                &value_locations,
                                Some(&instruction_location),
                            );
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
                                    Self::push_mem_op(
                                        &mut new_instructions,
                                        RefCountOp::Bump(count),
                                        *value,
                                        &value_locations,
                                        Some(&instruction_location),
                                    );
                                }
                            }
                        }
                        if !currently_live.contains(result) {
                            panic!(
                                "ICE: Result of SlicePush is immediately dropped. This is a bug."
                            )
                        }
                        let mut live_vals = vec![*slice];
                        live_vals.extend(values.iter().copied());
                        currently_live.extend(live_vals);
                    }
                    OpCode::InitGlobal {
                        global: _,
                        value: v,
                    } => {
                        // InitGlobal stores value into a global slot.
                        // If the value needs RC, bump it since the global now holds a reference.
                        let v = *v;
                        if self.needs_rc(type_info, &v) && currently_live.contains(&v) {
                            Self::push_mem_op(
                                &mut new_instructions,
                                RefCountOp::Bump(1),
                                v,
                                &value_locations,
                                Some(&instruction_location),
                            );
                        }
                        new_instructions.push(instruction);
                        currently_live.insert(v);
                    }
                    OpCode::DropGlobal { global: _ } => {
                        // DropGlobal IS the RC drop itself, just pass through.
                        new_instructions.push(instruction);
                    }
                    OpCode::Select {
                        result: _,
                        cond: _,
                        if_t: v1,
                        if_f: v2,
                    } => {
                        if self.needs_rc(type_info, v1) || self.needs_rc(type_info, v2) {
                            panic!("Unsupported yet");
                        }
                        new_instructions.push(instruction);
                    }
                    OpCode::ToBits {
                        result: r,
                        value: _,
                        endianness: _,
                        count: _,
                    } => {
                        if !currently_live.contains(r) {
                            // We contend with this, because ToBits can be used for a range check.
                            Self::push_mem_op(
                                &mut new_instructions,
                                RefCountOp::Drop,
                                *r,
                                &value_locations,
                                Some(&instruction_location),
                            );
                        }
                        // ToBits should return an RC counter of 1.
                        new_instructions.push(instruction);
                    }
                    OpCode::ToRadix {
                        result: r,
                        value: _,
                        radix: _,
                        endianness: _,
                        count: _,
                    } => {
                        if !currently_live.contains(r) {
                            // We contend with this, because ToRadix can be used for a range check.
                            Self::push_mem_op(
                                &mut new_instructions,
                                RefCountOp::Drop,
                                *r,
                                &value_locations,
                                Some(&instruction_location),
                            );
                        }
                        // ToRadix should return an RC counter of 1.
                        new_instructions.push(instruction);
                    }
                    OpCode::MemOp {
                        kind: _mem_op,
                        value: _value_id,
                    } => todo!(),
                    OpCode::Rangecheck {
                        value: _,
                        max_bits: _,
                    } => {
                        new_instructions.push(instruction);
                    }
                    OpCode::Guard { .. } => {
                        panic!("ICE: Guard should be lowered before RC insertion");
                    }
                    OpCode::MkTuple { .. } => ice_non_elided_tuple(),
                }
            }
            for param in block.get_parameter_values() {
                if self.needs_rc(type_info, param) && !currently_live.contains(param) {
                    Self::push_mem_op(
                        &mut new_instructions,
                        RefCountOp::Drop,
                        *param,
                        &value_locations,
                        block_location.as_ref(),
                    );
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
                intermediate.push_instruction(Self::located_mem_op(
                    RefCountOp::Drop,
                    *value,
                    &value_locations,
                    None,
                ));
            }
        }
    }

    fn value_source_locations(function: &HLFunction) -> HashMap<ValueId, SourceLocation> {
        let mut locations = HashMap::default();
        for (_, block) in function.get_blocks() {
            for instruction in block.get_instructions_with_source_locations() {
                for result in instruction.0.get_results() {
                    locations.insert(*result, instruction.1.clone());
                }
            }
        }
        locations
    }

    fn located_mem_op(
        kind: RefCountOp,
        value: ValueId,
        value_locations: &HashMap<ValueId, SourceLocation>,
        fallback_location: Option<&SourceLocation>,
    ) -> Located<OpCode> {
        let location = value_locations
            .get(&value)
            .or(fallback_location)
            .cloned()
            .expect("ICE: reference-count op emitted without a source location");

        Located::new(OpCode::MemOp { kind, value }, location)
    }

    fn push_mem_op(
        instructions: &mut Vec<Located<OpCode>>,
        kind: RefCountOp,
        value: ValueId,
        value_locations: &HashMap<ValueId, SourceLocation>,
        fallback_location: Option<&SourceLocation>,
    ) {
        instructions.push(Self::located_mem_op(
            kind,
            value,
            value_locations,
            fallback_location,
        ));
    }

    fn needs_rc(&self, type_info: &FunctionTypeInfo, value: &ValueId) -> bool {
        let value_type = type_info.get_value_type(*value);
        self.type_needs_rc(&value_type)
    }

    fn type_needs_rc(&self, value_type: &Type) -> bool {
        match &value_type.expr {
            TypeExpr::Ref(_) => true,
            TypeExpr::Array(_, _) => true,
            TypeExpr::Slice(_) => true,
            TypeExpr::Field => false,
            TypeExpr::U(_) => false,
            TypeExpr::I(_) => false,
            TypeExpr::WitnessOf(_) => true,
            TypeExpr::Tuple(_) => ice_non_elided_tuple(),
            TypeExpr::Function => false,
            TypeExpr::Blob(..) => false,
        }
    }
}
