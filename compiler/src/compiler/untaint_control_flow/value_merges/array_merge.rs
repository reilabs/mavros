//! Merge functional array updates without selecting every unchanged element.
//!
//! # Correctness Contract
//!
//! Consider a branch that starts with array A and builds B by writing a few cells.
//! Instead of selecting every cell of A and B, we start with A and replay those
//! writes. For each write, we select its new value when the changed arm is chosen,
//! or the accumulator's current cell otherwise. After each step the accumulator
//! therefore equals the corresponding version of B in the changed arm and A in
//! the unchanged arm. Replaying in source order preserves this invariant even
//! when indices repeat or alias at runtime.
//!
//! This reasoning requires the writes to start at exactly A. We capture their
//! original SSA provenance before linearization introduces casts and guards. For
//! nested updates, the same argument applies only when the updated child comes
//! from a read of that write's original parent and index. Unsupported or costly
//! plans fall back to the general merge without emitting a partial rewrite.
//!
//! Replay runs outside the original branch, so its accesses must also be safe
//! when that branch is inactive. We use the shared sequence-bounds comparison to
//! clamp invalid pure indices to zero, and assert the original bound under the
//! combined branch predicate. Active invalid writes still fail; an inactive enclosing
//! branch may compute an unused result but must not trap. Known in-bounds indices need no clamp.
//!
//! Once replay has preserved those bounds checks, we forward provable reads from
//! original updates and remove updates that become dead. Other users keep their
//! original values: cleanup never deletes a live update or assumes that distinct
//! dynamic indices cannot alias.

use super::{emit_merge_select, emit_value_conversion};
use crate::{
    collections::{HashMap, HashSet},
    compiler::{
        analysis::types::FunctionTypeInfo,
        passes::shared::{
            seq_bounds::seq_bounds_operands,
            value_replacements::{ReplaceScope, ValueReplacements},
        },
        ssa::{
            Instruction, SourceLocation, Terminator, ValueId,
            hlssa::{
                CastTarget, CmpKind, Constant, HLFunction, HLSSA, OpCode, Type, TypeExpr,
                builder::{HLBlockEmitter, HLEmitter, HLFunctionBuilder, HLInstrBuilder},
            },
        },
    },
};

const MAX_MERGED_WRITES: usize = 100;

#[derive(Clone, Copy)]
enum Base {
    Value(ValueId),
    Element { array: ValueId, index: ValueId },
}

struct ArrayAccess {
    array: ValueId,
    index: ValueId,
    value: Option<ValueId>,
    location: SourceLocation,
}

struct Write {
    original: ValueId,
    index: ValueId,
    value: ValueId,
    location: SourceLocation,
    nested: Option<Plan>,
}

struct Plan {
    typ: Type,
    writes: Vec<Write>,
    cost: usize,
}

pub(super) struct SparseArrayMerge<'a> {
    definitions: HashMap<ValueId, ArrayAccess>,
    constants: HashMap<ValueId, usize>,
    reads: HashMap<ValueId, usize>,
    replayed: HashSet<ValueId>,
    types: &'a FunctionTypeInfo,
}

impl<'a> SparseArrayMerge<'a> {
    /// Snapshot before linearization changes operands and wraps instructions in guards.
    pub(super) fn new(function: &HLFunction, types: &'a FunctionTypeInfo, ssa: &HLSSA) -> Self {
        let mut definitions = HashMap::default();
        for (_, block) in function.get_blocks() {
            for (instruction, location) in block.get_instructions_with_source_locations() {
                let (result, array, index, value) = match instruction {
                    OpCode::ArrayGet {
                        result,
                        array,
                        index,
                    } => (*result, *array, *index, None),
                    OpCode::ArraySet {
                        result,
                        array,
                        index,
                        value,
                    } => (*result, *array, *index, Some(*value)),
                    _ => continue,
                };
                definitions.insert(
                    result,
                    ArrayAccess {
                        array,
                        index,
                        value,
                        location: location.clone(),
                    },
                );
            }
        }
        let mut reads = HashMap::default();
        for access in definitions.values().filter(|access| access.value.is_none()) {
            *reads.entry(access.array).or_insert(0) += 1;
        }
        let constants = definitions
            .values()
            .filter_map(|access| {
                let constant = ssa.get_const(access.index)?;
                let Constant::Int(value) = constant.as_ref() else { return None };
                usize::try_from(value)
                    .ok()
                    .map(|value| (access.index, value))
            })
            .collect();
        Self {
            definitions,
            types,
            constants,
            reads,
            replayed: HashSet::default(),
        }
    }

    fn is_base(&self, value: ValueId, base: Base) -> bool {
        match base {
            Base::Value(expected) => value == expected,
            Base::Element { array, index } => self.definitions.get(&value).is_some_and(|access| {
                access.value.is_none() && access.array == array && access.index == index
            }),
        }
    }

    /// Plan first so a failed match emits no partial rewrite. The shared budget
    /// bounds both write-chain searches and recursive nested-update expansion.
    fn plan(&self, changed: ValueId, base: Base, typ: &Type, budget: &mut usize) -> Option<Plan> {
        let TypeExpr::Array(elem, len) = &typ.expr else { return None };
        if *len == 0 {
            return None;
        }
        let mut current = changed;
        let mut chain = Vec::new();
        while !self.is_base(current, base) {
            if chain.len() >= *budget {
                return None;
            }
            let definition = self.definitions.get(&current)?;
            let value = definition.value?;
            // Witness-indexed writes already require a scan in later lowering.
            if !matches!(
                self.types.get_value_type(definition.index).expr,
                TypeExpr::Int(_)
            ) {
                return None;
            }
            chain.push((current, definition, value));
            current = definition.array;
        }
        if chain.is_empty() {
            return None;
        }
        *budget -= chain.len();
        chain.reverse();
        let read_count: usize = chain
            .iter()
            .map(|(result, _, _)| self.reads.get(result).copied().unwrap_or(0))
            .sum();
        let forwarding_cost = read_count.saturating_mul(chain.len()).saturating_mul(10);
        let writes: Vec<_> = chain
            .into_iter()
            .map(|(original, access, value)| {
                let nested = self.plan(
                    value,
                    Base::Element {
                        array: access.array,
                        index: access.index,
                    },
                    elem,
                    budget,
                );
                Write {
                    original,
                    index: access.index,
                    value,
                    location: access.location.clone(),
                    nested,
                }
            })
            .collect();
        // Count emitted operations conservatively. The general cost excludes casts,
        // while replay includes bounds work and conversion allowances: at most one
        // extra cast per selected leaf, plus the initial base conversion.
        let cost = writes
            .iter()
            .fold(1usize.saturating_add(forwarding_cost), |cost, write| {
                let bounds = if self.index_is_safe(write.index, *len) {
                    0
                } else {
                    5
                };
                cost.saturating_add(2 + bounds).saturating_add(
                    write
                        .nested
                        .as_ref()
                        .map(|plan| plan.cost)
                        .unwrap_or_else(|| merge_cost(elem).saturating_mul(2)),
                )
            });
        (cost < merge_cost(typ)).then(|| Plan {
            typ: typ.clone(),
            writes,
            cost,
        })
    }

    pub(super) fn try_emit(
        &mut self,
        b: &mut HLBlockEmitter<'_>,
        condition: ValueId,
        not_condition: ValueId,
        then_active: ValueId,
        else_active: ValueId,
        lhs: ValueId,
        rhs: ValueId,
        typ: &Type,
    ) -> Option<ValueId> {
        let mut budget = MAX_MERGED_WRITES;
        if let Some(plan) = self.plan(lhs, Base::Value(rhs), typ, &mut budget) {
            let base = emit_value_conversion(rhs, self.types.get_value_type(rhs), typ, b);
            return Some(self.emit_plan(b, condition, then_active, base, &plan));
        }
        let mut budget = MAX_MERGED_WRITES;
        let plan = self.plan(rhs, Base::Value(lhs), typ, &mut budget)?;
        let base = emit_value_conversion(lhs, self.types.get_value_type(lhs), typ, b);
        Some(self.emit_plan(b, not_condition, else_active, base, &plan))
    }

    /// Forward reads through replayed writes before pruning them. Comparisons are
    /// pure, so possible runtime aliases select the latest value without a scan.
    pub(super) fn remove_redundant_updates(self, function: &mut HLFunction, ssa: &mut HLSSA) {
        if self.replayed.is_empty() {
            return;
        }
        let guards: HashMap<_, _> = function
            .get_blocks()
            .flat_map(|(_, block)| {
                block.get_instructions().filter_map(|op| {
                    let (guard, inner) = HLBlockEmitter::unwrap_guard(op);
                    match inner {
                        OpCode::ArraySet { result, .. } => Some((*result, guard)),
                        _ => None,
                    }
                })
            })
            .collect();
        let mut replacements = ValueReplacements::new();
        let blocks: Vec<_> = function.get_blocks().map(|(id, _)| *id).collect();
        for block in blocks {
            let instructions = function.get_block_mut(block).take_instructions();
            let terminator = function
                .get_block_mut(block)
                .take_terminator()
                .expect("ICE: missing terminator before update cleanup");
            let mut rewritten = Vec::new();
            let mut selections = HashMap::default();
            for instruction in instructions {
                let (guard, inner) = HLBlockEmitter::unwrap_guard(instruction.as_ref());
                let forwarded = match inner {
                    OpCode::ArrayGet { result, .. } => self
                        .definitions
                        .get(result)
                        .filter(|read| self.replayed.contains(&read.array))
                        .filter(|read| {
                            matches!(self.types.get_value_type(read.index).expr, TypeExpr::Int(_))
                        })
                        .map(|read| (*result, read)),
                    _ => None,
                };
                let Some((result, read)) = forwarded else {
                    rewritten.push(instruction);
                    continue;
                };
                let mut array = read.array;
                let mut pending = Vec::new();
                let mut value = None;
                while self.replayed.contains(&array) && guards.get(&array) == Some(&guard) {
                    let write = &self.definitions[&array];
                    if write.index == read.index {
                        value = write.value;
                        break;
                    }
                    if !matches!((self.constants.get(&write.index), self.constants.get(&read.index)),
                        (Some(a), Some(b)) if a != b)
                    {
                        pending.push(write);
                    }
                    array = write.array;
                }
                // A different guard can make the original update unobservable. Do not
                // forward across it: the read must retain that update's guarded semantics.
                if array == read.array && value.is_none() {
                    rewritten.push(instruction);
                    continue;
                }
                let mut emitted = Vec::new();
                let mut b = HLInstrBuilder::new(
                    function,
                    ssa,
                    &mut emitted,
                    instruction.location().clone(),
                );
                let target_type = self.types.get_value_type(result);
                let current = match value {
                    Some(value) => emit_value_conversion(
                        value,
                        self.types.get_value_type(value),
                        target_type,
                        &mut b,
                    ),
                    None => {
                        let read = b.array_get(array, read.index);
                        emit_value_conversion(
                            read,
                            &self.types.get_value_type(array).get_array_element(),
                            target_type,
                            &mut b,
                        )
                    }
                };
                let mut current = current;
                for write in pending.into_iter().rev() {
                    let TypeExpr::Int(a) = self.types.get_value_type(write.index).expr else {
                        unreachable!()
                    };
                    let TypeExpr::Int(c) = self.types.get_value_type(read.index).expr else {
                        unreachable!()
                    };
                    let lhs = b.widen_u(write.index, a, a.max(c));
                    let rhs = b.widen_u(read.index, c, a.max(c));
                    let hit = b.cmp(lhs, rhs, CmpKind::Eq);
                    let value = write.value.unwrap();
                    let value = emit_value_conversion(
                        value,
                        self.types.get_value_type(value),
                        target_type,
                        &mut b,
                    );
                    current = b.select(hit, value, current);
                    selections.insert(current, target_type.clone());
                }
                replacements.insert(result, current);
                for op in emitted {
                    let (op, location) = op.take();
                    let op = match guard {
                        Some(condition) => OpCode::Guard {
                            condition,
                            inner: Box::new(op),
                        },
                        None => op,
                    };
                    rewritten.push(op.locate(location));
                }
            }
            let mut fb = HLFunctionBuilder::new(function, ssa);
            let mut b = fb
                .block(block)
                .with_scoped_source_locations("merge_update_cleanup");
            for instruction in rewritten {
                let (_, inner) = HLBlockEmitter::unwrap_guard(instruction.as_ref());
                if let OpCode::Select {
                    result,
                    cond,
                    if_t,
                    if_f,
                } = inner
                    && let Some(typ) = selections.get(result)
                {
                    // Selecting already-computed values is safe even under an inactive
                    // guard. Use pure control flow: the specialization VM has no Select.
                    let (result, cond, if_t, if_f) = (*result, *cond, *if_t, *if_f);
                    b.emit_with_location(instruction.location().clone(), |b| {
                        b.build_if_else_into(
                            cond,
                            vec![(result, typ.clone())],
                            |_| vec![if_t],
                            |_| vec![if_f],
                        );
                    });
                } else {
                    b.emit_located(instruction);
                }
            }
            b.set_terminator(terminator);
        }
        replacements.apply_to_function(function, ReplaceScope::Inputs);

        // A worklist removes the now-dead update chains and their casts/reads.
        // Only replayed stores are eligible: their failure checks were emitted above.
        let mut uses: HashMap<ValueId, usize> = HashMap::default();
        let mut removable = HashMap::default();
        for (_, block) in function.get_blocks() {
            for op in block.get_instructions() {
                for input in op.get_inputs() {
                    *uses.entry(*input).or_default() += 1;
                }
                let (_, inner) = HLBlockEmitter::unwrap_guard(op);
                let result = match inner {
                    OpCode::ArrayGet { result, .. } | OpCode::Cast { result, .. } => Some(*result),
                    OpCode::ArraySet { result, .. } if self.replayed.contains(result) => {
                        Some(*result)
                    }
                    _ => None,
                };
                if let Some(result) = result {
                    removable.insert(result, op.get_inputs().copied().collect::<Vec<_>>());
                }
            }
            match block.get_terminator() {
                Some(Terminator::Jmp(_, args) | Terminator::Return(args)) => {
                    for value in args {
                        *uses.entry(*value).or_default() += 1;
                    }
                }
                Some(Terminator::JmpIf(cond, _, _)) => {
                    *uses.entry(*cond).or_default() += 1;
                }
                None => unreachable!("ICE: merge cleanup requires terminated blocks"),
            }
        }
        let mut work: Vec<_> = removable
            .keys()
            .copied()
            .filter(|v| uses.get(v).copied().unwrap_or(0) == 0)
            .collect();
        let mut dead = HashSet::default();
        while let Some(value) = work.pop() {
            if !dead.insert(value) {
                continue;
            }
            for input in &removable[&value] {
                let count = uses.get_mut(input).unwrap();
                *count -= 1;
                if *count == 0 && removable.contains_key(input) {
                    work.push(*input);
                }
            }
        }
        for (_, block) in function.get_blocks_mut() {
            let mut instructions = block.take_instructions();
            instructions.retain(|op| !op.get_results().any(|v| dead.contains(v)));
            block.put_instructions(instructions);
        }
    }

    fn index_is_safe(&self, index: ValueId, len: usize) -> bool {
        if self.constants.get(&index).is_some_and(|value| *value < len) {
            return true;
        }
        matches!(self.types.get_value_type(index).expr, TypeExpr::Int(bits)
            if bits < usize::BITS as usize && (1usize << bits) <= len)
    }

    fn emit_plan(
        &mut self,
        b: &mut HLBlockEmitter<'_>,
        condition: ValueId,
        active: ValueId,
        mut base: ValueId,
        plan: &Plan,
    ) -> ValueId {
        let TypeExpr::Array(elem, len) = &plan.typ.expr else { unreachable!() };
        for write in &plan.writes {
            self.replayed.insert(write.original);
            base = b.emit_with_location(write.location.clone(), |b| {
                let index_type = self.types.get_value_type(write.index);
                let TypeExpr::Int(bits) = index_type.expr else { unreachable!() };
                // If the entire index type fits, the access is already safe. Otherwise
                // clamp by bounds, not by the witness branch condition: pure loop indices
                // must remain pure, or witness-index lowering would scan the whole array.
                let index = if self.index_is_safe(write.index, *len) {
                    write.index
                } else {
                    let (_, len_cmp, idx_cmp, _) =
                        seq_bounds_operands(b, base, write.index, &plan.typ, index_type);
                    let in_bounds = b.ult(idx_cmp, len_cmp);
                    b.emit(OpCode::Guard {
                        condition: active,
                        inner: Box::new(OpCode::Assert { value: in_bounds }),
                    });
                    // A zero-or-one multiplier preserves the pure integer type and
                    // is supported by both the specialization VM and runtime lowering.
                    let keep = b.cast_to(CastTarget::Int(bits), in_bounds);
                    b.umul(write.index, keep)
                };
                let old = b.array_get(base, index);
                let selected = match &write.nested {
                    Some(nested) => self.emit_plan(b, condition, active, old, nested),
                    None => emit_merge_select(
                        b,
                        condition,
                        write.value,
                        old,
                        None,
                        elem,
                        self.types.get_value_type(write.value),
                        elem,
                    ),
                };
                b.array_set(base, index, selected)
            });
        }
        base
    }
}

fn merge_cost(typ: &Type) -> usize {
    match &typ.expr {
        TypeExpr::Array(elem, len) => len
            .saturating_mul(2usize.saturating_add(merge_cost(elem)))
            .saturating_add(1),
        _ => 1,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::{
        analysis::{flow_analysis::FlowAnalysis, types::Types},
        ssa::{
            FunctionId, Terminator,
            hlssa::{HLSSA, builder::HLFunctionBuilder},
        },
    };

    struct Fixture {
        ssa: HLSSA,
        function: FunctionId,
        base: ValueId,
        other: ValueId,
        changed: ValueId,
        index: ValueId,
        condition: ValueId,
        typ: Type,
    }

    fn fixture(len: usize, count: usize, nested: bool, index_bits: usize) -> Fixture {
        let mut ssa = HLSSA::with_main("merge".to_string());
        let fid = ssa.get_unique_entrypoint_id();
        let mut function = ssa.take_function(fid);
        let entry = function.get_entry_id();
        let scalar = Type::witness_of(Type::field());
        let elem = if nested {
            scalar.clone().array_of(2)
        } else {
            scalar.clone()
        };
        let typ = elem.array_of(len);
        let base = ssa.fresh_value();
        let other = ssa.fresh_value();
        let index = ssa.fresh_value();
        let condition = ssa.fresh_value();
        let value = ssa.fresh_value();
        for (id, ty) in [
            (base, typ.clone()),
            (other, typ.clone()),
            (index, Type::int(index_bits)),
            (condition, Type::witness_of(Type::int(1))),
            (value, scalar),
        ] {
            function.get_block_mut(entry).push_parameter(id, ty);
        }
        function.add_return_type(typ.clone());
        let changed;
        {
            let mut fb = HLFunctionBuilder::new(&mut function, &mut ssa);
            let mut b = fb.test_block(entry);
            let mut array = base;
            for _ in 0..count {
                let val = if nested {
                    let row = b.array_get(array, index);
                    let zero = b.int_const(32, 0);
                    b.array_set(row, zero, value)
                } else {
                    value
                };
                array = b.array_set(array, index, val);
            }
            changed = array;
            b.set_terminator(Terminator::Return(vec![array]));
        }
        ssa.put_function(fid, function);
        Fixture {
            ssa,
            function: fid,
            base,
            other,
            changed,
            index,
            condition,
            typ,
        }
    }

    #[test]
    fn planning_requires_a_supported_chain_and_respects_the_budget() {
        // length, writes, nested, witness index, expected plan
        for (len, count, nested, witness, accepted) in [
            (1024, 1, false, false, true),
            (1024, MAX_MERGED_WRITES, false, false, true),
            (1024, MAX_MERGED_WRITES + 1, false, false, false),
            (1024, 2, true, false, true),
            (3, 1, false, true, false),
            (3, 3, false, false, false),
            (3, MAX_MERGED_WRITES, false, false, false),
            (0, 1, false, false, false),
        ] {
            let mut f = fixture(len, count, nested, 32);
            if witness {
                let function = f.ssa.get_function_mut(f.function);
                let entry = function.get_entry_id();
                for (id, typ) in function.get_block_mut(entry).get_parameters_mut() {
                    if *id == f.index {
                        *typ = Type::witness_of(typ.clone());
                    }
                }
            }
            let types = Types::new().run(&f.ssa, &FlowAnalysis::run(&f.ssa));
            let merger = SparseArrayMerge::new(
                f.ssa.get_function(f.function),
                types.get_function(f.function),
                &f.ssa,
            );
            let mut budget = MAX_MERGED_WRITES;
            let plan = merger.plan(f.changed, Base::Value(f.base), &f.typ, &mut budget);
            assert_eq!(plan.is_some(), accepted);
            if let Some(plan) = plan {
                assert_eq!(plan.writes.len(), count);
                assert_eq!(
                    budget,
                    MAX_MERGED_WRITES - count * if nested { 2 } else { 1 }
                );
                if nested {
                    assert!(
                        plan.writes
                            .iter()
                            .all(|w| w.nested.as_ref().unwrap().writes.len() == 1)
                    );
                }
            }
            let mut budget = MAX_MERGED_WRITES;
            assert!(
                merger
                    .plan(f.changed, Base::Value(f.other), &f.typ, &mut budget)
                    .is_none()
            );
        }
    }

    #[test]
    fn safe_indices_remain_pure_and_narrow_index_bounds_do_not_wrap() {
        for bits in [1, 32, 64, 128] {
            let mut f = fixture(128, 1, false, bits);
            let types = Types::new().run(&f.ssa, &FlowAnalysis::run(&f.ssa));
            let mut function = f.ssa.take_function(f.function);
            let mut merger =
                SparseArrayMerge::new(&function, types.get_function(f.function), &f.ssa);
            let entry = function.get_entry_id();
            function.get_block_mut(entry).take_terminator();
            let result;
            {
                let mut fb = HLFunctionBuilder::new(&mut function, &mut f.ssa);
                let mut b = fb.test_block(entry);
                let not_cond = b.not(f.condition);
                result = merger
                    .try_emit(
                        &mut b,
                        f.condition,
                        not_cond,
                        f.condition,
                        not_cond,
                        f.changed,
                        f.base,
                        &f.typ,
                    )
                    .unwrap();
                b.set_terminator(Terminator::Return(vec![result]));
            }
            f.ssa.put_function(f.function, function);
            let types = Types::new().run(&f.ssa, &FlowAnalysis::run(&f.ssa));
            let ti = types.get_function(f.function);
            assert_eq!(ti.get_value_type(result), &f.typ);
            for (_, block) in f.ssa.get_function(f.function).get_blocks() {
                for op in block.get_instructions() {
                    if let OpCode::Cmp {
                        kind: CmpKind::ULt,
                        lhs,
                        rhs,
                        ..
                    } = op
                    {
                        assert_eq!(ti.get_value_type(*lhs), &Type::int(bits.max(32)));
                        assert_eq!(ti.get_value_type(*rhs), &Type::int(bits.max(32)));
                    }
                    if let OpCode::ArrayGet { index, .. } = op {
                        assert_eq!(ti.get_value_type(*index), &Type::int(bits));
                        if bits == 1 {
                            assert_eq!(*index, f.index);
                        }
                    }
                }
            }
        }
    }
    #[test]
    fn cleanup_removes_replayed_stores_but_preserves_other_users() {
        for keep_original in [false, true] {
            let mut f = fixture(128, 1, false, 32);
            let mut function = f.ssa.take_function(f.function);
            let entry = function.get_entry_id();
            let other_index = f.ssa.fresh_value();
            function
                .get_block_mut(entry)
                .push_parameter(other_index, Type::int(32));
            let first = f.changed;
            {
                let mut fb = HLFunctionBuilder::new(&mut function, &mut f.ssa);
                let mut b = fb.test_block(entry);
                let read = b.array_get(first, other_index);
                f.changed = b.array_set(first, other_index, read);
                b.set_terminator(Terminator::Return(vec![f.changed]));
            }
            f.ssa.put_function(f.function, function);
            let types = Types::new().run(&f.ssa, &FlowAnalysis::run(&f.ssa));
            let mut function = f.ssa.take_function(f.function);
            let mut merger =
                SparseArrayMerge::new(&function, types.get_function(f.function), &f.ssa);
            {
                let mut fb = HLFunctionBuilder::new(&mut function, &mut f.ssa);
                let mut b = fb.test_block(entry);
                let not_cond = b.not(f.condition);
                // Exercise the else-arm path: it must reuse the supplied negation.
                let result = merger
                    .try_emit(
                        &mut b,
                        f.condition,
                        not_cond,
                        f.condition,
                        not_cond,
                        f.base,
                        f.changed,
                        &f.typ,
                    )
                    .unwrap();
                let mut returned = vec![result];
                if keep_original {
                    returned.push(first);
                }
                b.set_terminator(Terminator::Return(returned));
            }
            if keep_original {
                function.add_return_type(f.typ.clone());
            }
            merger.remove_redundant_updates(&mut function, &mut f.ssa);
            let instructions: Vec<_> = function
                .get_blocks()
                .flat_map(|(_, block)| block.get_instructions())
                .collect();
            assert_eq!(
                instructions
                    .iter()
                    .filter(|op| matches!(op, OpCode::Not { .. }))
                    .count(),
                1
            );
            assert!(
                !instructions.iter().any(
                    |op| matches!(op, OpCode::ArraySet { result, .. } if *result == f.changed)
                )
            );
            assert_eq!(
                instructions
                    .iter()
                    .any(|op| matches!(op, OpCode::ArraySet { result, .. } if *result == first)),
                keep_original
            );
            f.ssa.put_function(f.function, function);
            Types::new().run(&f.ssa, &FlowAnalysis::run(&f.ssa));
        }
    }

    #[test]
    fn in_bounds_constants_skip_bounds_work() {
        for index in [0, 127] {
            let mut f = fixture(128, 1, false, 64);
            let constant = f.ssa.add_const(Constant::int(64, index));
            let function = f.ssa.get_function_mut(f.function);
            for (_, block) in function.get_blocks_mut() {
                for op in block.get_instructions_mut() {
                    for input in op.get_inputs_mut() {
                        if *input == f.index {
                            *input = constant;
                        }
                    }
                }
            }
            let types = Types::new().run(&f.ssa, &FlowAnalysis::run(&f.ssa));
            let mut function = f.ssa.take_function(f.function);
            let mut merger =
                SparseArrayMerge::new(&function, types.get_function(f.function), &f.ssa);
            let entry = function.get_entry_id();
            {
                let mut fb = HLFunctionBuilder::new(&mut function, &mut f.ssa);
                let mut b = fb.test_block(entry);
                let not_cond = b.not(f.condition);
                let result = merger
                    .try_emit(
                        &mut b,
                        f.condition,
                        not_cond,
                        f.condition,
                        not_cond,
                        f.changed,
                        f.base,
                        &f.typ,
                    )
                    .unwrap();
                b.set_terminator(Terminator::Return(vec![result]));
            }
            assert!(
                !function
                    .get_block(entry)
                    .get_instructions()
                    .any(|op| matches!(
                        op,
                        OpCode::Cmp { .. } | OpCode::Guard { .. } | OpCode::BinaryArithOp { .. }
                    ))
            );
        }
    }
}
