//! Merge functional array updates without selecting every unchanged element.
//!
//! # Correctness contract
//!
//! Matching uses the SSA before branch linearization inserts casts and guards.
//! A plan requires a chain of functional ArraySets rooted at the exact unchanged
//! SSA value. A nested plan must start at the corresponding ArrayGet of that
//! write's original array and index. Distinct loads are not interchangeable: an
//! intervening store may have changed the referenced array. Indices must be pure
//! integers and every optimized array must have a nonzero, statically known length.
//!
//! For a chain A[k+1] = set(A[k], i[k], v[k]), replay maintains this invariant:
//! when the changed arm is selected, the accumulator equals A[k]; otherwise it
//! equals A[0]. Selecting v[k] against the accumulator's current cell and writing
//! it back preserves that invariant. Writes must therefore stay in source order,
//! including repeated or dynamically aliasing indices. Nested plans apply the
//! same argument to the cell read from the current accumulator. Values outside
//! the active enclosing branch are unobservable; their evaluation must still be safe.
//!
//! The merge executes unconditionally. An out-of-bounds index is clamped to zero,
//! while a separate bounds assertion uses the changed arm's *combined* predicate
//! (including enclosing branches). Thus active invalid writes still fail, and
//! inactive writes neither trap nor change the unchanged arm. Bounds clamping is
//! pure, so it does not introduce witness-index scans. If the whole index type
//! fits, no clamp is needed and the length need not fit in the index's bit width.
//!
//! Planning emits nothing until it succeeds. A rejected plan uses the general
//! merge. Original instructions remain available to other users and ordinary
//! DCE; replacement bounds assertions preserve failures if original writes die.

use super::{emit_merge_select, emit_value_conversion};
use crate::{
    collections::HashMap,
    compiler::{
        analysis::types::FunctionTypeInfo,
        ssa::{
            SourceLocation, ValueId,
            hlssa::{
                CastTarget, HLFunction, LocatedOpCode, OpCode, Type, TypeExpr,
                builder::{HLBlockEmitter, HLEmitter},
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

struct Write {
    index: ValueId,
    value: ValueId,
    location: SourceLocation,
    nested: Option<Plan>,
}

struct Plan {
    typ: Type,
    writes: Vec<Write>,
}

pub(super) struct SparseArrayMerge<'a> {
    definitions: HashMap<ValueId, LocatedOpCode>,
    types: &'a FunctionTypeInfo,
}

impl<'a> SparseArrayMerge<'a> {
    /// Snapshot before linearization changes operands and wraps instructions in guards.
    pub(super) fn new(function: &HLFunction, types: &'a FunctionTypeInfo) -> Self {
        let mut definitions = HashMap::default();
        for (_, block) in function.get_blocks() {
            for (instruction, location) in block.get_instructions_with_source_locations() {
                let result = match instruction {
                    OpCode::ArrayGet { result, .. } | OpCode::ArraySet { result, .. } => *result,
                    _ => continue,
                };
                definitions.insert(result, instruction.clone().locate(location.clone()));
            }
        }
        Self { definitions, types }
    }

    fn is_base(&self, value: ValueId, base: Base) -> bool {
        match base {
            Base::Value(expected) => value == expected,
            Base::Element { array, index } => {
                matches!(self.definitions.get(&value).map(|op| op.as_ref()),
                Some(OpCode::ArrayGet { array: a, index: i, .. }) if *a == array && *i == index)
            }
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
            let OpCode::ArraySet {
                array,
                index,
                value,
                ..
            } = definition.as_ref()
            else {
                return None;
            };
            // Witness-indexed writes already scan the whole array during lowering.
            // Replaying them here adds lookups and scans instead of reducing work.
            if !matches!(self.types.get_value_type(*index).expr, TypeExpr::Int(_)) {
                return None;
            }
            chain.push((*array, *index, *value, definition.location().clone()));
            current = *array;
        }
        if chain.is_empty() {
            return None;
        }
        *budget -= chain.len();
        chain.reverse();
        let writes = chain
            .into_iter()
            .map(|(array, index, value, location)| {
                let nested = self.plan(value, Base::Element { array, index }, elem, budget);
                Write {
                    index,
                    value,
                    location,
                    nested,
                }
            })
            .collect();
        Some(Plan {
            typ: typ.clone(),
            writes,
        })
    }

    pub(super) fn try_emit(
        &self,
        b: &mut HLBlockEmitter<'_>,
        condition: ValueId,
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
        let condition = b.not(condition);
        Some(self.emit_plan(b, condition, else_active, base, &plan))
    }

    fn emit_plan(
        &self,
        b: &mut HLBlockEmitter<'_>,
        condition: ValueId,
        active: ValueId,
        mut base: ValueId,
        plan: &Plan,
    ) -> ValueId {
        let TypeExpr::Array(elem, len) = &plan.typ.expr else { unreachable!() };
        for write in &plan.writes {
            base = b.emit_with_location(write.location.clone(), |b| {
                let index_type = self.types.get_value_type(write.index);
                let TypeExpr::Int(bits) = index_type.expr else { unreachable!() };
                // If the entire index type fits, the access is already safe. Otherwise
                // clamp by bounds, not by the witness branch condition: pure loop indices
                // must remain pure, or witness-index lowering would scan the whole array.
                let index = if bits < usize::BITS as usize && (1usize << bits) <= *len {
                    write.index
                } else {
                    let limit = b.int_const(bits, *len as u128);
                    let in_bounds = b.ult(write.index, limit);
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
        for bits in [1, 32] {
            let mut f = fixture(3, 1, false, bits);
            let types = Types::new().run(&f.ssa, &FlowAnalysis::run(&f.ssa));
            let mut function = f.ssa.take_function(f.function);
            let merger = SparseArrayMerge::new(&function, types.get_function(f.function));
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
}
