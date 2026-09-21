//! Merge functional array updates without selecting every unchanged element.
//!
//! # Correctness Contract
//!
//! Consider a branch that starts with array A and builds B by writing a few
//! cells. Instead of selecting every cell of A and B, we start with A and
//! replay those writes. For each write, we select its new value when the
//! changed arm is chosen, or the accumulator's current cell otherwise. After
//! each step the accumulator therefore equals the corresponding version of B in
//! the changed arm and A in the unchanged arm. Replaying in source order
//! preserves this invariant even when indices repeat or alias at runtime.
//!
//! This reasoning requires the writes to start at exactly A. We capture their
//! original SSA provenance before linearization introduces casts and guards.
//! For nested updates, the same argument applies only when the updated child
//! comes from a read of that write's original parent and index. Unsupported or
//! costly plans fall back to the general merge without emitting a partial
//! rewrite.
//!
//! Replay runs outside the original branch, so its accesses must also be safe
//! when that branch is inactive. We use the shared sequence-bounds comparison
//! to clamp invalid pure indices to zero, and assert the original bound under
//! the original branch predicate. Plans that cross a write's guard boundary are
//! rejected, so replay cannot weaken an earlier write's bounds check. Active
//! invalid writes still fail; an inactive enclosing branch may compute an
//! unused result but must not trap. Known in-bounds indices need no clamp.
//!
//! Once replay has preserved those bounds checks, we forward provable reads
//! from original updates and remove updates that become dead. Other users keep
//! their original values: cleanup never deletes a live update or assumes that
//! distinct dynamic indices cannot alias.

use super::{emit_merge_select, emit_value_conversion};
#[cfg(test)]
use crate::compiler::{
    analysis::{
        flow_analysis::FlowAnalysis, types::Types, value_range_analysis::ValueRangeAnalysis,
    },
    ssa::FunctionId,
};
use crate::{
    collections::{HashMap, HashSet},
    compiler::{
        analysis::{types::FunctionTypeInfo, value_range_analysis::FunctionValueRanges},
        passes::shared::{
            seq_bounds::{seq_bounds_operands, widen_comparison_operands},
            value_replacements::{ReplaceScope, ValueReplacements},
        },
        ssa::{
            BlockId, Instruction, SourceLocation, Terminator, ValueId,
            hlssa::{
                CastTarget, CmpKind, Constant, HLFunction, HLSSA, OpCode, Type, TypeExpr,
                builder::{HLBlockEmitter, HLEmitter, HLFunctionBuilder, HLInstrBuilder},
            },
        },
    },
};
use num_traits::ToPrimitive;

const MAX_MERGED_WRITES: usize = 100;

// Costs count SSA operations, not elapsed time or final circuit constraints.
// Constants are interned. Conversions reserve one Cast even when it folds away.
const ARRAY_GET_COST: usize = 1;
const ARRAY_SET_COST: usize = 1;
const ARRAY_CONSTRUCTION_COST: usize = 1;
const CAST_COST: usize = 1;
const SELECT_COST: usize = 1;
// One widening, comparison, assertion, boolean cast, and masking multiply.
const BOUNDS_COST: usize = CAST_COST + 1 + 1 + CAST_COST + 1;
// Two input widening allowances, a comparison, value conversion, and a pure
// choice (three terminators plus one merge parameter). The fallback read is
// charged separately per read. These are conservative upper estimates.
const FORWARDING_STEP_COST: usize = 2 * CAST_COST + 1 + CAST_COST + 3 + 1;

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
    reads: HashMap<ValueId, Vec<ValueId>>,
    uses: HashMap<ValueId, usize>,
    pub(super) ranges: Option<FunctionValueRanges>,
    merge_block: Option<BlockId>,
    replayed: HashSet<ValueId>,
    guards: HashMap<ValueId, Option<ValueId>>,
    types: &'a FunctionTypeInfo,
}

fn count_uses(function: &HLFunction) -> HashMap<ValueId, usize> {
    let mut uses = HashMap::default();
    for (_, block) in function.get_blocks() {
        for op in block.get_instructions() {
            for input in op.get_inputs() {
                *uses.entry(*input).or_insert(0) += 1;
            }
        }
        let args = match block.get_terminator() {
            Some(Terminator::Jmp(_, args) | Terminator::Return(args)) => args.as_slice(),
            Some(Terminator::JmpIf(condition, _, _)) => std::slice::from_ref(condition),
            None => &[],
        };
        for input in args {
            *uses.entry(*input).or_insert(0) += 1;
        }
    }
    uses
}

impl<'a> SparseArrayMerge<'a> {
    /// Cheap structural filter before allocating provenance and use-count maps.
    /// Profitability and exact guard/provenance checks remain the planner's job.
    pub(super) fn has_candidates(
        function: &HLFunction,
        types: &FunctionTypeInfo,
        merges: &HashSet<BlockId>,
    ) -> bool {
        if merges.is_empty() {
            return false;
        }
        let mut incoming = HashSet::default();
        for (_, block) in function.get_blocks() {
            if let Some(Terminator::Jmp(target, args)) = block.get_terminator()
                && merges.contains(target)
            {
                for (arg, (_, typ)) in args
                    .iter()
                    .zip(function.get_block(*target).get_parameters())
                {
                    if matches!(typ.expr, TypeExpr::Array(_, len) if len > 0) {
                        incoming.insert(*arg);
                    }
                }
            }
        }
        function.get_blocks().any(|(_, block)| block.get_instructions().any(|op| {
            matches!(op, OpCode::ArraySet { result, index, .. }
                if incoming.contains(result) && matches!(types.get_value_type(*index).expr, TypeExpr::Int(_)))
        }))
    }

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
                    } => (result, array, index, None),
                    OpCode::ArraySet {
                        result,
                        array,
                        index,
                        value,
                    } => (result, array, index, Some(*value)),
                    _ => continue,
                };
                definitions.insert(
                    *result,
                    ArrayAccess {
                        array: *array,
                        index: *index,
                        value,
                        location: location.clone(),
                    },
                );
            }
        }
        let mut reads = HashMap::default();
        for (result, access) in definitions
            .iter()
            .filter(|(_, access)| access.value.is_none())
        {
            reads
                .entry(access.array)
                .or_insert_with(Vec::new)
                .push(*result);
        }
        // Count original consumers before linearization inserts casts and guards.
        // These wrappers preserve the dependency of each original use; cleanup
        // removes dead wrappers along with replayed stores. Other consumers still
        // retain that dependency. Uses eliminated by earlier merges may remain in
        // this snapshot, conservatively withholding a deletion credit from plan.
        let uses = count_uses(function);
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
            uses,
            ranges: None,
            merge_block: None,
            replayed: HashSet::default(),
            guards: HashMap::default(),
        }
    }

    fn is_base(&self, value: ValueId, base: Base, active: ValueId) -> bool {
        match base {
            Base::Value(expected) => value == expected,
            Base::Element { array, index } => {
                self.guards.get(&value) == Some(&Some(active))
                    && self.definitions.get(&value).is_some_and(|access| {
                        access.value.is_none() && access.array == array && access.index == index
                    })
            }
        }
    }

    /// Plan first so a failed match emits no partial rewrite. The shared budget
    /// bounds both write-chain searches and recursive nested-update expansion.
    fn plan(
        &self,
        changed: ValueId,
        base: Base,
        typ: &Type,
        active: ValueId,
        budget: &mut usize,
    ) -> Option<Plan> {
        let TypeExpr::Array(elem, len) = &typ.expr else { return None };
        if *len == 0 {
            return None;
        }
        let mut current = changed;
        let mut chain = Vec::new();
        while !self.is_base(current, base, active) {
            if chain.len() >= *budget {
                return None;
            }
            let definition = self.definitions.get(&current)?;
            let value = definition.value?;
            if self.guards.get(&current) != Some(&Some(active)) {
                return None;
            }
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
            .map(|(result, _, _)| self.reads.get(result).map_or(0, Vec::len))
            .sum();
        let retained = chain.iter().any(|(result, _, _)| {
            let forwarded_reads = self
                .reads
                .get(result)
                .into_iter()
                .flatten()
                .filter(|read| self.guards.get(*read) == Some(&Some(active)))
                .filter(|read| {
                    matches!(
                        self.types
                            .get_value_type(self.definitions[*read].index)
                            .expr,
                        TypeExpr::Int(_)
                    )
                })
                .count();
            let internal_uses = chain
                .iter()
                .filter(|(_, access, _)| access.array == *result)
                .count();
            // One use of the final array is replaced by this merge (or its parent
            // write for a nested plan). Any other consumer can keep the chain alive.
            self.uses.get(result).copied().unwrap_or(0)
                > internal_uses + forwarded_reads + usize::from(*result == changed)
        });
        let eliminated_cost = if !retained {
            chain
                .len()
                .saturating_mul(ARRAY_SET_COST + CAST_COST + BOUNDS_COST)
        } else {
            0
        };
        let forwarding_cost = read_count.saturating_mul(
            ARRAY_GET_COST.saturating_add(chain.len().saturating_mul(FORWARDING_STEP_COST)),
        );
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
                    active,
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
        let cost =
            writes
                .iter()
                .fold(CAST_COST.saturating_add(forwarding_cost), |cost, write| {
                    let bounds = if self.index_is_safe(write.index, *len) {
                        0
                    } else {
                        BOUNDS_COST
                    };
                    cost.saturating_add(ARRAY_GET_COST + ARRAY_SET_COST + bounds)
                        .saturating_add(write.nested.as_ref().map(|plan| plan.cost).unwrap_or_else(
                            || merge_cost(elem).saturating_add(conversion_cost(elem)),
                        ))
                });
        // The original writes already exist on the general-merge side. Keeping
        // them adds no relative cost; deleting them is a credit to replay. Nested
        // plans retain gross replay cost here, conservatively omitting their credit.
        (cost < merge_cost(typ).saturating_add(eliminated_cost)).then(|| Plan {
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
        self.merge_block = Some(b.block_id());
        let mut budget = MAX_MERGED_WRITES;
        if let Some(plan) = self.plan(lhs, Base::Value(rhs), typ, then_active, &mut budget) {
            let base = emit_value_conversion(rhs, self.types.get_value_type(rhs), typ, b);
            return Some(self.emit_plan(b, condition, then_active, base, &plan));
        }
        let mut budget = MAX_MERGED_WRITES;
        let plan = self.plan(rhs, Base::Value(lhs), typ, else_active, &mut budget)?;
        let base = emit_value_conversion(lhs, self.types.get_value_type(lhs), typ, b);
        Some(self.emit_plan(b, not_condition, else_active, base, &plan))
    }

    /// Capture predicates after linearization, before replay can remove their writes.
    pub(super) fn capture_guards(&mut self, function: &HLFunction) {
        self.guards = function
            .get_blocks()
            .flat_map(|(_, block)| {
                block.get_instructions().filter_map(|op| {
                    let (guard, inner) = HLBlockEmitter::unwrap_guard(op);
                    match inner {
                        OpCode::ArraySet { result, .. } | OpCode::ArrayGet { result, .. } => {
                            Some((*result, guard))
                        }
                        _ => None,
                    }
                })
            })
            .collect();
    }

    /// Test helper for exercising a single rewritten function.
    #[cfg(test)]
    fn capture_ranges(
        &mut self,
        id: FunctionId,
        function: &HLFunction,
        ssa: &HLSSA,
        unknown: &HashSet<ValueId>,
    ) {
        if !self.definitions.values().any(|access| {
            access.value.is_some()
                && matches!(
                    self.types.get_value_type(access.index).expr,
                    TypeExpr::Int(_)
                )
        }) {
            return;
        }
        let cfg = FlowAnalysis::run_function(function);
        let mut signatures = Types::function_types(ssa);
        signatures.insert(id, (function.get_param_types(), function.get_returns()));
        let types = Types::new().run_function(
            function,
            &signatures,
            &Types::constant_types(ssa, &signatures),
            &cfg,
            ssa.field(),
        );
        self.ranges =
            Some(ValueRangeAnalysis::new().run_on_function(function, &cfg, &types, ssa, unknown));
    }

    /// Forward reads through replayed writes before pruning them. Comparisons are
    /// pure, so possible runtime aliases select the latest value without a scan.
    pub(super) fn remove_redundant_updates(self, function: &mut HLFunction, ssa: &mut HLSSA) {
        if self.replayed.is_empty() {
            return;
        }
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
                while self.replayed.contains(&array) && self.guards.get(&array) == Some(&guard) {
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
                    let (lhs, rhs, _) =
                        widen_comparison_operands(&mut b, write.index, a, read.index, c);
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
                    && let Some(typ) = selections.remove(result)
                {
                    // Selecting already-computed values is safe even under an inactive
                    // guard. Pure control flow also keeps array-valued choices type legal:
                    // array Select is not a lowering input, and pure Select is unsupported
                    // by the specialization VM. Every staged choice must be consumed here.
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
            assert!(
                selections.is_empty(),
                "ICE: unlowered array-merge cleanup choices"
            );
            b.set_terminator(terminator);
        }
        replacements.apply_to_function(function, ReplaceScope::Inputs);

        // A worklist removes the now-dead update chains and their casts/reads.
        // Only replayed stores are eligible: their failure checks were emitted above.
        let mut uses = count_uses(function);
        let mut removable = HashMap::default();
        for (_, block) in function.get_blocks() {
            for op in block.get_instructions() {
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
        }
        // DCE handles unrelated dead reads and casts. Here we only need the
        // dependencies of replayed stores: leaving a dead read/cast in that chain
        // keeps earlier stores live until witness-memory lowering expands them.
        let mut related = HashSet::default();
        let mut pending: Vec<_> = self.replayed.iter().copied().collect();
        while let Some(value) = pending.pop() {
            if related.insert(value)
                && let Some(inputs) = removable.get(&value)
            {
                pending.extend(inputs.iter().copied());
            }
        }
        removable.retain(|result, _| related.contains(result));
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
        if let (Some(ranges), Some(block)) = (&self.ranges, self.merge_block) {
            let range = ranges.get_at(block, index);
            if range
                .unsigned()
                .hi()
                .and_then(ToPrimitive::to_usize)
                .is_some_and(|max| max < len)
            {
                return true;
            }
        }
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
            .saturating_mul((2 * ARRAY_GET_COST).saturating_add(merge_cost(elem)))
            .saturating_add(ARRAY_CONSTRUCTION_COST),
        _ => SELECT_COST,
    }
}

// emit_merge_select converts each scalar leaf in an array merge. Unlike its
// two inputs, replay's old cell already has the target type, so only the new
// value needs a conversion allowance.
fn conversion_cost(typ: &Type) -> usize {
    match &typ.expr {
        TypeExpr::Array(elem, len) => len.saturating_mul(conversion_cost(elem)),
        _ => CAST_COST,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::{
        analysis::{
            flow_analysis::FlowAnalysis,
            types::{TypeInfo, Types},
        },
        ssa::{
            FunctionId, Terminator,
            hlssa::{HLSSA, builder::HLFunctionBuilder},
        },
    };
    use mavros_int_semantics::IntBits;

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

    impl Fixture {
        fn merger<'a>(&mut self, types: &'a TypeInfo) -> SparseArrayMerge<'a> {
            let mut function = self.ssa.take_function(self.function);
            let merger = guarded_merger(self, &mut function, types);
            self.ssa.put_function(self.function, function);
            merger
        }

        fn plan(&self, merger: &SparseArrayMerge, budget: &mut usize) -> Option<Plan> {
            merger.plan(
                self.changed,
                Base::Value(self.base),
                &self.typ,
                self.condition,
                budget,
            )
        }
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
                    let zero = b.int_const(IntBits::from_u128(32, 0));
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

    fn guarded_merger<'a>(
        f: &Fixture,
        function: &mut HLFunction,
        types: &'a TypeInfo,
    ) -> SparseArrayMerge<'a> {
        let mut merger = SparseArrayMerge::new(function, types.get_function(f.function), &f.ssa);
        guard_accesses(function, f.condition);
        merger.capture_guards(function);
        merger
    }

    fn guard_accesses(function: &mut HLFunction, condition: ValueId) {
        for (_, block) in function.get_blocks_mut() {
            let instructions = block
                .take_instructions()
                .into_iter()
                .map(|op| {
                    let (op, location) = op.take();
                    let op = match op {
                        OpCode::ArrayGet { .. } | OpCode::ArraySet { .. } | OpCode::Cast { .. } => {
                            OpCode::Guard {
                                condition,
                                inner: Box::new(op),
                            }
                        }
                        _ => op,
                    };
                    op.locate(location)
                })
                .collect();
            block.put_instructions(instructions);
        }
    }

    #[test]
    fn candidate_filter_requires_an_array_merge_fed_by_a_pure_index_write() {
        // array length, witness index, array merge, updated argument, witness merge
        for (len, witness, array, updated, branch, expected) in [
            (128, false, true, true, true, true),
            (128, true, true, true, true, false),
            (0, false, true, true, true, false),
            (128, false, false, true, true, false),
            (128, false, true, false, true, false),
            (128, false, true, true, false, false),
        ] {
            let mut f = fixture(len, 1, false, 32);
            let parameter = f.ssa.fresh_value();
            let function = f.ssa.get_function_mut(f.function);
            let entry = function.get_entry_id();
            if witness {
                for (id, typ) in function.get_block_mut(entry).get_parameters_mut() {
                    if *id == f.index {
                        *typ = Type::witness_of(typ.clone());
                    }
                }
            }
            let merge = function.add_block();
            let arg = if !array {
                f.index
            } else if updated {
                f.changed
            } else {
                f.base
            };
            let typ = if array { f.typ.clone() } else { Type::int(32) };
            function
                .get_block_mut(entry)
                .set_terminator(Terminator::Jmp(merge, vec![arg]));
            function.get_block_mut(merge).push_parameter(parameter, typ);
            function
                .get_block_mut(merge)
                .set_terminator(Terminator::Return(vec![f.base]));
            let types = Types::new().run(&f.ssa, &FlowAnalysis::run(&f.ssa));
            let merges = if branch {
                HashSet::from_iter([merge])
            } else {
                HashSet::default()
            };
            assert_eq!(
                SparseArrayMerge::has_candidates(
                    f.ssa.get_function(f.function),
                    types.get_function(f.function),
                    &merges
                ),
                expected
            );
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
            (3, 3, false, false, true), // deleting the original writes makes replay cheaper
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
            let merger = f.merger(&types);
            let mut budget = MAX_MERGED_WRITES;
            let plan = f.plan(&merger, &mut budget);
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
                    .plan(
                        f.changed,
                        Base::Value(f.other),
                        &f.typ,
                        f.condition,
                        &mut budget
                    )
                    .is_none()
            );
        }
    }

    #[test]
    fn planning_rejects_writes_from_another_guard_or_without_a_guard() {
        let f = fixture(128, 2, false, 32);
        let other_guard = f.ssa.add_const(Constant::int(1, 1));
        let types = Types::new().run(&f.ssa, &FlowAnalysis::run(&f.ssa));
        let mut merger = SparseArrayMerge::new(
            f.ssa.get_function(f.function),
            types.get_function(f.function),
            &f.ssa,
        );
        let first = merger.definitions[&f.changed].array;
        for original_guard in [Some(f.condition), None, Some(other_guard)] {
            merger.guards.insert(f.changed, Some(f.condition));
            merger.guards.insert(first, original_guard);
            let mut budget = MAX_MERGED_WRITES;
            let plan = f.plan(&merger, &mut budget);
            assert_eq!(plan.is_some(), original_guard == Some(f.condition));
        }
        merger.guards.remove(&first);
        let mut budget = MAX_MERGED_WRITES;
        assert!(f.plan(&merger, &mut budget).is_none());
    }

    #[test]
    fn nested_replay_requires_matching_parent_read_and_guard() {
        let mut f = fixture(128, 1, true, 32);
        let types = Types::new().run(&f.ssa, &FlowAnalysis::run(&f.ssa));
        let mut merger = f.merger(&types);
        let child = merger.definitions[&f.changed].value.unwrap();
        let read = merger.definitions[&child].array;
        for (parent, child_guard, read_guard, nested) in [
            (f.base, Some(f.condition), Some(f.condition), true),
            (f.other, Some(f.condition), Some(f.condition), false),
            (f.base, None, Some(f.condition), false),
            (f.base, Some(f.condition), None, false),
            (f.base, Some(f.condition), Some(f.index), false),
        ] {
            merger.definitions.get_mut(&read).unwrap().array = parent;
            merger.guards.insert(child, child_guard);
            merger.guards.insert(read, read_guard);
            let mut budget = MAX_MERGED_WRITES;
            let plan = f.plan(&merger, &mut budget).unwrap();
            assert_eq!(plan.writes[0].nested.is_some(), nested);
        }
    }

    #[test]
    fn only_dead_original_updates_receive_a_deletion_credit() {
        for retained in [false, true] {
            let mut f = fixture(6, 2, false, 32);
            if retained {
                let function = f.ssa.get_function_mut(f.function);
                let entry = function.get_entry_id();
                let first = *function
                    .get_block(entry)
                    .get_instructions()
                    .next()
                    .unwrap()
                    .get_results()
                    .next()
                    .unwrap();
                function
                    .get_block_mut(entry)
                    .set_terminator(Terminator::Return(vec![f.changed, first]));
                function.add_return_type(f.typ.clone());
            }
            let types = Types::new().run(&f.ssa, &FlowAnalysis::run(&f.ssa));
            let merger = f.merger(&types);
            let mut budget = MAX_MERGED_WRITES;
            let plan = f.plan(&merger, &mut budget);
            assert_eq!(plan.is_some(), !retained);
        }
    }

    #[test]
    fn bounds_are_queried_at_the_replay_block() {
        for bounded in [false, true] {
            let mut f = fixture(128, 1, false, 32);
            let mut function = f.ssa.take_function(f.function);
            let entry = function.get_entry_id();
            let body = function.add_block();
            let exit = function.add_block();
            let instructions = function.get_block_mut(entry).take_instructions();
            function.get_block_mut(body).put_instructions(instructions);
            function
                .get_block_mut(body)
                .set_terminator(Terminator::Return(vec![f.changed]));
            function
                .get_block_mut(exit)
                .set_terminator(Terminator::Return(vec![f.base]));
            {
                let mut fb = HLFunctionBuilder::new(&mut function, &mut f.ssa);
                let mut b = fb.test_block(entry);
                if bounded {
                    let len = b.int_const(IntBits::from_u128(32, 128));
                    let condition = b.ult(f.index, len);
                    b.set_terminator(Terminator::JmpIf(condition, body, exit));
                } else {
                    b.set_terminator(Terminator::Jmp(body, vec![]));
                }
            }
            f.ssa.put_function(f.function, function);
            let types = Types::new().run(&f.ssa, &FlowAnalysis::run(&f.ssa));
            let mut function = f.ssa.take_function(f.function);
            let mut merger = guarded_merger(&f, &mut function, &types);
            merger.capture_ranges(f.function, &function, &f.ssa, &HashSet::default());
            merger.merge_block = Some(body);
            assert_eq!(merger.index_is_safe(f.index, 128), bounded);
            // A fact inside the loop/branch body must not be used at its entry.
            merger.merge_block = Some(entry);
            assert!(!merger.index_is_safe(f.index, 128));
        }
    }

    #[test]
    fn pending_merge_arguments_cannot_narrow_index_ranges() {
        for pending in [false, true] {
            let mut f = fixture(128, 1, false, 32);
            let mut function = f.ssa.take_function(f.function);
            let entry = function.get_entry_id();
            let merge = function.add_block();
            let parameter = f.ssa.fresh_value();
            function
                .get_block_mut(merge)
                .push_parameter(parameter, Type::int(32));
            let instructions = function.get_block_mut(entry).take_instructions();
            let index;
            {
                let mut fb = HLFunctionBuilder::new(&mut function, &mut f.ssa);
                let mut b = fb.test_block(entry);
                let zero = b.int_const(IntBits::from_u128(32, 0));
                b.set_terminator(Terminator::Jmp(merge, vec![zero]));
                drop(b);
                let mut b = fb.test_block(merge);
                let one = b.int_const(IntBits::from_u128(32, 1));
                index = b.uadd(parameter, one);
                for instruction in instructions {
                    b.emit_located(instruction);
                }
                b.set_terminator(Terminator::Return(vec![f.changed]));
            }
            f.ssa.put_function(f.function, function);
            let types = Types::new().run(&f.ssa, &FlowAnalysis::run(&f.ssa));
            let function = f.ssa.get_function(f.function);
            let mut merger =
                SparseArrayMerge::new(function, types.get_function(f.function), &f.ssa);
            let unknown = if pending {
                HashSet::from_iter([parameter])
            } else {
                HashSet::default()
            };
            merger.capture_ranges(f.function, function, &f.ssa, &unknown);
            merger.merge_block = Some(merge);
            assert_eq!(merger.index_is_safe(index, 128), !pending);
        }
    }

    #[test]
    fn safe_indices_remain_pure_and_narrow_index_bounds_do_not_wrap() {
        for bits in [1, 32, 64, 128] {
            let mut f = fixture(128, 1, false, bits);
            let types = Types::new().run(&f.ssa, &FlowAnalysis::run(&f.ssa));
            let mut function = f.ssa.take_function(f.function);
            let mut merger = guarded_merger(&f, &mut function, &types);
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
        for (keep_original, nested, read_bits) in
            [(false, false, 32), (true, false, 32), (false, true, 64)]
        {
            let mut f = fixture(128, 1, nested, 32);
            let mut function = f.ssa.take_function(f.function);
            let entry = function.get_entry_id();
            let other_index = f.ssa.fresh_value();
            function
                .get_block_mut(entry)
                .push_parameter(other_index, Type::int(read_bits));
            let first = f.changed;
            let unrelated;
            {
                let mut fb = HLFunctionBuilder::new(&mut function, &mut f.ssa);
                let mut b = fb.test_block(entry);
                let read = b.array_get(first, other_index);
                f.changed = b.array_set(first, other_index, read);
                unrelated = [
                    b.array_get(f.other, other_index),
                    b.cast_to_witness_of(other_index),
                ];
                b.set_terminator(Terminator::Return(vec![f.changed]));
            }
            f.ssa.put_function(f.function, function);
            let types = Types::new().run(&f.ssa, &FlowAnalysis::run(&f.ssa));
            let mut function = f.ssa.take_function(f.function);
            let mut merger = guarded_merger(&f, &mut function, &types);
            {
                let mut fb = HLFunctionBuilder::new(&mut function, &mut f.ssa);
                let mut b = fb.test_block(entry);
                let not_cond = b.not(f.condition);
                // Exercise the else-arm path: it must reuse the supplied negation.
                let result = merger
                    .try_emit(
                        &mut b,
                        not_cond,
                        f.condition,
                        not_cond,
                        f.condition,
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
                    |op| matches!(HLBlockEmitter::unwrap_guard(op).1, OpCode::ArraySet { result, .. } if *result == f.changed)
                )
            );
            assert_eq!(
                instructions
                    .iter()
                    .any(|op| matches!(HLBlockEmitter::unwrap_guard(op).1, OpCode::ArraySet { result, .. } if *result == first)),
                keep_original
            );
            assert!(unrelated.iter().all(|id| {
                instructions
                    .iter()
                    .any(|op| op.get_results().any(|r| r == id))
            }));
            f.ssa.put_function(f.function, function);
            let flow = FlowAnalysis::run(&f.ssa);
            let types = Types::new().run(&f.ssa, &flow);
            for (_, block) in f.ssa.get_function(f.function).get_blocks() {
                assert!(block.get_terminator().is_some());
                for op in block.get_instructions() {
                    // Array-valued choices must become typed phi parameters.
                    if let OpCode::Select { result, .. } = HLBlockEmitter::unwrap_guard(op).1 {
                        assert!(!matches!(
                            types.get_function(f.function).get_value_type(*result).expr,
                            TypeExpr::Array(_, _)
                        ));
                    }
                    if let OpCode::Cmp {
                        lhs,
                        rhs,
                        kind: CmpKind::Eq,
                        ..
                    } = HLBlockEmitter::unwrap_guard(op).1
                    {
                        assert_eq!(
                            types.get_function(f.function).get_value_type(*lhs),
                            &Type::int(read_bits.max(32))
                        );
                        assert_eq!(
                            types.get_function(f.function).get_value_type(*rhs),
                            &Type::int(read_bits.max(32))
                        );
                    }
                }
            }
            use crate::compiler::passes::dead_code_elimination::{Config, DCE};
            DCE::new(Config::pre_r1c()).do_run(&mut f.ssa, &flow);
            assert!(
                f.ssa
                    .get_function(f.function)
                    .get_blocks()
                    .all(|(_, block)| block
                        .get_instructions()
                        .all(|op| !op.get_results().any(|id| unrelated.contains(id))))
            );
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
            let mut merger = guarded_merger(&f, &mut function, &types);
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
                        HLBlockEmitter::unwrap_guard(op).1,
                        OpCode::Cmp { .. } | OpCode::Assert { .. } | OpCode::BinaryArithOp { .. }
                    ))
            );
        }
    }
}
