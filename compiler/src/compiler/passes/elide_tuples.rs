//! A pass that eliminates all tuple types from the HLSSA.
//!
//! Tuples are spilled into individual values, with the tuple constructor pushed *upward* through
//! `Ref`, `Array`, `Slice` and `WitnessOf` until it disappears entirely:
//!
//! - `(A, B)`              becomes values `A, B`
//! - `Ref<(A, B)>`         becomes `Ref<A>, Ref<B>`
//! - `Array<(A, B), n>`    becomes `Array<A, n>, Array<B, n>`
//! - ...applied recursively.
//!
//! The net effect is that every value whose type *contains* a tuple expands into a fixed, ordered
//! list of tuple-free "leaf" values, so a single `ValueId` is represented by a `Vec<ValueId>`.
//! After this pass runs, no [`TypeExpr::Tuple`], `MkTuple`, `TupleProj` or `TupleRefProj` reaches
//! any subsequent pass: the IR is tuple-free from here through the rest of HLSSA. Several downstream
//! passes still *contain* tuple-handling arms (`untaint_control_flow`, `witness_lowering`,
//! `rc_insertion`, codegen); those are now dead and can be removed as follow-up.
//!
//! This pass is intended to run directly after `PrepareEntryPoint` (which itself synthesizes tuples
//! while reconstructing the entry-point ABI).
//!
//! ## Strategy
//!
//! Types are taken from a single [`TypeInfo`] snapshot computed on the original, type-consistent IR;
//! they are never recomputed mid-pass (the partially rewritten IR is transiently type-inconsistent).
//! Each function is handled in two phases:
//!
//! 1. **Plan:** build a `value_map: ValueId -> Vec<ValueId>` mapping every original value to its
//!    component leaves. Tuple-free values map to themselves (no churn); tuple-bearing values get
//!    freshly minted component ids. `MkTuple`/`TupleProj`/`TupleRefProj` results are *aliased* to
//!    slices of their operands' components rather than allocated (they emit no instruction).
//! 2. **Rewrite (mutating):** flatten function returns, block parameters, instructions and
//!    terminators using the `value_map`. Unreachable blocks (which the type snapshot never typed)
//!    are dropped.

use crate::{
    collections::{HashMap, HashSet},
    compiler::{
        analysis::{flow_analysis::FlowAnalysis, types::TypeInfo},
        pass_manager::{Analysis, AnalysisId, AnalysisStore, Pass},
        ssa::{
            BlockId, FunctionId, Instruction, Terminator, ValueId,
            hlssa::{HLSSA, OpCode, Type, TypeExpr},
        },
    },
};

pub struct ElideTuples {}

impl ElideTuples {
    pub fn new() -> Self {
        ElideTuples {}
    }
}

impl Pass for ElideTuples {
    fn name(&self) -> &'static str {
        "elide_tuples"
    }

    fn needs(&self) -> Vec<AnalysisId> {
        vec![TypeInfo::id(), FlowAnalysis::id()]
    }

    fn run(&self, ssa: &mut HLSSA, store: &AnalysisStore) {
        let old_global_types: Vec<Type> = ssa.get_global_types().to_vec();
        let mut global_offsets: Vec<usize> = Vec::with_capacity(old_global_types.len());
        let mut acc = 0usize;
        for gty in &old_global_types {
            global_offsets.push(acc);
            acc += slot_count(gty);
        }
        let new_global_types: Vec<Type> = old_global_types.iter().flat_map(leaf_types).collect();
        ssa.set_global_types(new_global_types);

        let type_info = store.get::<TypeInfo>();
        let cfg = store.get::<FlowAnalysis>();

        let function_ids: Vec<FunctionId> = ssa.get_function_ids().collect();
        for fid in function_ids {
            let reachable: Vec<BlockId> = cfg
                .get_function_cfg(fid)
                .get_domination_pre_order()
                .collect();
            let value_map = Self::plan_function(ssa, fid, &reachable, type_info);
            Self::rewrite_function(
                ssa,
                fid,
                &reachable,
                &value_map,
                &global_offsets,
                &old_global_types,
            );
        }

        // Verify the tuple-free invariant on every build: one linear pass, and the
        // guarantee it enforces is relied on by every subsequent pass.
        verify_tuple_free(ssa);
    }

    fn preserves(&self) -> Vec<AnalysisId> {
        // Signatures, values and the block set all change, so every cached analysis is invalidated.
        vec![]
    }
}

impl ElideTuples {
    /// Builds the `ValueId -> Vec<ValueId>` component map for the provided function.
    fn plan_function(
        ssa: &HLSSA,
        fid: FunctionId,
        reachable: &[BlockId],
        type_info: &TypeInfo,
    ) -> HashMap<ValueId, Vec<ValueId>> {
        let func = ssa.get_function(fid);
        let fti = type_info.get_function(fid);
        let mut value_map: HashMap<ValueId, Vec<ValueId>> = HashMap::default();

        // Sweep A: every block parameter gets its components, so all parameters are resolved before
        // any instruction (which may reference a dominating block's parameter) is visited.
        for bid in reachable {
            for (pid, ty) in func.get_block(*bid).get_parameters() {
                let comps = alloc_components(ssa, *pid, ty);
                value_map.insert(*pid, comps);
            }
        }

        // Sweep B: instructions in dominator pre-order, so an aliasing op
        // (`MkTuple`/`TupleProj`/`TupleRefProj`) always sees its operands' components already
        // resolved.
        for bid in reachable {
            for instr in func.get_block(*bid).get_instructions() {
                Self::plan_instruction(ssa, instr, fti, &mut value_map);
            }
        }

        // Every result and block parameter is now planned, so the only operands absent from the map
        // are constants. `components()` treats an unmapped value as a single self-component, which
        // is correct only for tuple-free values — assert that no tuple-typed operand ever falls
        // through (which would silently mis-flatten).
        #[cfg(debug_assertions)]
        for bid in reachable {
            for instr in func.get_block(*bid).get_instructions() {
                for v in instr.get_inputs() {
                    if !value_map.contains_key(v) {
                        debug_assert!(
                            !contains_tuple(fti.get_value_type(*v)),
                            "elide_tuples: operand v{} is tuple-typed but was never planned \
                             (a tuple-typed constant?) — components() would mis-flatten it",
                            v.0
                        );
                    }
                }
            }
        }

        value_map
    }

    fn plan_instruction(
        ssa: &HLSSA,
        instr: &OpCode,
        fti: &crate::compiler::analysis::types::FunctionTypeInfo,
        value_map: &mut HashMap<ValueId, Vec<ValueId>>,
    ) {
        match instr {
            // `MkTuple` regroups its elements' components; it emits no instruction.
            OpCode::MkTuple { result, elems, .. } => {
                let comps: Vec<ValueId> = elems
                    .iter()
                    .flat_map(|e| components(value_map, *e))
                    .collect();
                value_map.insert(*result, comps);
            }
            // `TupleProj` selects the slice of components belonging to the projected field.
            OpCode::TupleProj { result, tuple, idx } => {
                let tuple_type = fti.get_value_type(*tuple);
                let elements = tuple_type.get_tuple_elements();
                let offset: usize = elements[..*idx].iter().map(slot_count).sum();
                let width = slot_count(&elements[*idx]);
                let tuple_comps = components(value_map, *tuple);
                value_map.insert(*result, tuple_comps[offset..offset + width].to_vec());
            }
            // `TupleRefProj` selects the slice of reference components belonging to the projected
            // field of the pointed tuple.
            OpCode::TupleRefProj {
                result,
                tuple_ref,
                idx,
            } => {
                let tuple_type = fti.get_value_type(*tuple_ref).get_pointed();
                let elements = tuple_type.get_tuple_elements();
                let offset: usize = elements[..*idx].iter().map(slot_count).sum();
                let width = slot_count(&elements[*idx]);
                let tuple_ref_comps = components(value_map, *tuple_ref);
                value_map.insert(*result, tuple_ref_comps[offset..offset + width].to_vec());
            }
            OpCode::Guard { .. } => panic!("ICE: Guard encountered during tuple elision"),
            // Every genuine result gets freshly-allocated components (or maps to itself when its
            // type is already tuple-free).
            other => {
                for result in other.get_results() {
                    if value_map.contains_key(result) {
                        panic!("ICE: Value encountered before visiting its definition");
                    }
                    let ty = fti.get_value_type(*result);
                    let comps = alloc_components(ssa, *result, ty);
                    value_map.insert(*result, comps);
                }
            }
        }
    }

    /// Rewrites one function's signature, blocks, instructions and terminators in place.
    fn rewrite_function(
        ssa: &mut HLSSA,
        fid: FunctionId,
        reachable: &[BlockId],
        value_map: &HashMap<ValueId, Vec<ValueId>>,
        global_offsets: &[usize],
        old_global_types: &[Type],
    ) {
        let func = ssa.get_function_mut(fid);

        let old_returns = func.take_returns();
        for ty in old_returns {
            for leaf in leaf_types(&ty) {
                func.add_return_type(leaf);
            }
        }

        for bid in reachable {
            let block = func.get_block_mut(*bid);

            let old_params = block.take_parameters();
            let mut new_params = Vec::new();
            for (pid, ty) in old_params {
                let comps = components(value_map, pid);
                let leaves = leaf_types(&ty);
                debug_assert_eq!(comps.len(), leaves.len());
                for (comp, leaf) in comps.into_iter().zip(leaves) {
                    new_params.push((comp, leaf));
                }
            }
            block.put_parameters(new_params);

            let old_instructions = block.take_instructions();
            let mut new_instructions = Vec::with_capacity(old_instructions.len());
            for instr in &old_instructions {
                lower_instruction(
                    instr,
                    value_map,
                    global_offsets,
                    old_global_types,
                    &mut new_instructions,
                );
            }
            block.put_instructions(new_instructions);

            let new_terminator = match block.take_terminator().unwrap() {
                Terminator::Jmp(dest, args) => {
                    Terminator::Jmp(dest, flat_components(value_map, &args))
                }
                Terminator::JmpIf(cond, t, f) => Terminator::JmpIf(single(value_map, cond), t, f),
                Terminator::Return(values) => {
                    Terminator::Return(flat_components(value_map, &values))
                }
            };
            block.set_terminator(new_terminator);
        }

        // Drop unreachable blocks: they were never typed by the snapshot and are dead anyway
        // (RemoveUnreachableBlocks, which runs next, would remove them regardless).
        let keep: HashSet<BlockId> = reachable.iter().copied().collect();
        let mut blocks = func.take_blocks();
        blocks.retain(|id, _| keep.contains(id));
        func.put_blocks(blocks);
    }
}

// ---------------------------------------------------------------------------------------------
// Per-instruction lowering
// ---------------------------------------------------------------------------------------------

/// Lower a single opcode into zero or more tuple-free opcodes, appending them to `out`.
fn lower_instruction(
    op: &OpCode,
    value_map: &HashMap<ValueId, Vec<ValueId>>,
    global_offsets: &[usize],
    old_global_types: &[Type],
    out: &mut Vec<OpCode>,
) {
    match op {
        OpCode::MkTuple { .. } | OpCode::TupleProj { .. } | OpCode::TupleRefProj { .. } => {}

        // Global opcodes carry an absolute slot index that the renumbering in `run` shifts, so they
        // need explicit handling: remap the slot through `global_offsets` and fan out per-leaf.
        OpCode::ReadGlobal {
            result,
            offset,
            result_type,
        } => {
            let base = global_offsets[*offset as usize];
            let leaves = leaf_types(result_type);
            let results = components(value_map, *result);
            debug_assert_eq!(leaves.len(), results.len());
            for (s, (leaf, r)) in leaves.into_iter().zip(results).enumerate() {
                out.push(OpCode::ReadGlobal {
                    result: r,
                    offset: (base + s) as u64,
                    result_type: leaf,
                });
            }
        }
        OpCode::InitGlobal { global, value } => {
            let base = global_offsets[*global];
            for (s, v) in components(value_map, *value).into_iter().enumerate() {
                out.push(OpCode::InitGlobal {
                    global: base + s,
                    value: v,
                });
            }
        }
        OpCode::DropGlobal { global } => {
            let base = global_offsets[*global];
            // The original drop fired only because the whole global was heap-allocated; after the
            // split, drop exactly the leaves that are themselves heap-allocated.
            for (s, leaf) in leaf_types(&old_global_types[*global])
                .into_iter()
                .enumerate()
            {
                if leaf.is_heap_allocated() {
                    out.push(OpCode::DropGlobal { global: base + s });
                }
            }
        }

        OpCode::Alloc { result, elem_type } => {
            let leaves = leaf_types(elem_type);
            let results = components(value_map, *result);
            for (leaf, r) in leaves.into_iter().zip(results) {
                out.push(OpCode::Alloc {
                    result: r,
                    elem_type: leaf,
                });
            }
        }
        OpCode::Load { result, ptr } => {
            let ptrs = components(value_map, *ptr);
            let results = components(value_map, *result);
            for (p, r) in ptrs.into_iter().zip(results) {
                out.push(OpCode::Load { result: r, ptr: p });
            }
        }
        OpCode::Store { ptr, value } => {
            let ptrs = components(value_map, *ptr);
            let values = components(value_map, *value);
            for (p, v) in ptrs.into_iter().zip(values) {
                out.push(OpCode::Store { ptr: p, value: v });
            }
        }
        OpCode::ArrayGet {
            result,
            array,
            index,
        } => {
            let idx = single(value_map, *index);
            let arrays = components(value_map, *array);
            let results = components(value_map, *result);
            for (a, r) in arrays.into_iter().zip(results) {
                out.push(OpCode::ArrayGet {
                    result: r,
                    array: a,
                    index: idx,
                });
            }
        }
        OpCode::ArraySet {
            result,
            array,
            index,
            value,
        } => {
            let idx = single(value_map, *index);
            let arrays = components(value_map, *array);
            let values = components(value_map, *value);
            let results = components(value_map, *result);
            for ((a, v), r) in arrays.into_iter().zip(values).zip(results) {
                out.push(OpCode::ArraySet {
                    result: r,
                    array: a,
                    index: idx,
                    value: v,
                });
            }
        }
        OpCode::MkSeq {
            result,
            elems,
            seq_type,
            elem_type,
        } => {
            let leaves = leaf_types(elem_type);
            let results = components(value_map, *result);
            let elem_components: Vec<Vec<ValueId>> =
                elems.iter().map(|e| components(value_map, *e)).collect();
            for (slot, (leaf, r)) in leaves.into_iter().zip(results).enumerate() {
                let slot_elems: Vec<ValueId> = elem_components.iter().map(|ec| ec[slot]).collect();
                out.push(OpCode::MkSeq {
                    result: r,
                    elems: slot_elems,
                    seq_type: *seq_type,
                    elem_type: leaf,
                });
            }
        }
        OpCode::MkRepeated {
            result,
            element,
            seq_type,
            count,
            elem_type,
        } => {
            let leaves = leaf_types(elem_type);
            let results = components(value_map, *result);
            let elements = components(value_map, *element);
            for (slot, (leaf, r)) in leaves.into_iter().zip(results).enumerate() {
                out.push(OpCode::MkRepeated {
                    result: r,
                    element: elements[slot],
                    seq_type: *seq_type,
                    count: *count,
                    elem_type: leaf,
                });
            }
        }
        OpCode::Call {
            results,
            function,
            args,
            unconstrained,
        } => {
            out.push(OpCode::Call {
                results: flat_components(value_map, results),
                function: function.clone(),
                args: flat_components(value_map, args),
                unconstrained: *unconstrained,
            });
        }
        OpCode::Select {
            result,
            cond,
            if_t,
            if_f,
        } => {
            let c = single(value_map, *cond);
            let then = components(value_map, *if_t);
            let otherwise = components(value_map, *if_f);
            let results = components(value_map, *result);
            for ((t, f), r) in then.into_iter().zip(otherwise).zip(results) {
                out.push(OpCode::Select {
                    result: r,
                    cond: c,
                    if_t: t,
                    if_f: f,
                });
            }
        }
        OpCode::Cast {
            result,
            value,
            target,
        } => {
            let values = components(value_map, *value);
            let results = components(value_map, *result);
            // A tuple-typed cast splits into one cast per leaf component,
            // each targeting the corresponding leaf of the target type.
            let leaf_targets = leaf_types(target);
            assert_eq!(
                leaf_targets.len(),
                results.len(),
                "cast target leaves must match the result component count"
            );
            for ((v, r), leaf) in values.into_iter().zip(results).zip(leaf_targets) {
                out.push(OpCode::Cast {
                    result: r,
                    value: v,
                    target: leaf,
                });
            }
        }
        OpCode::WriteWitness {
            result,
            value,
            pinned,
        } => {
            let values = components(value_map, *value);
            match result {
                Some(result_id) => {
                    let results = components(value_map, *result_id);
                    for (v, r) in values.into_iter().zip(results) {
                        out.push(OpCode::WriteWitness {
                            result: Some(r),
                            value: v,
                            pinned: *pinned,
                        });
                    }
                }
                None => {
                    for v in values {
                        out.push(OpCode::WriteWitness {
                            result: None,
                            value: v,
                            pinned: *pinned,
                        });
                    }
                }
            }
        }
        OpCode::SlicePush {
            dir,
            result,
            slice,
            values,
        } => {
            let slices = components(value_map, *slice);
            let results = components(value_map, *result);
            let value_components: Vec<Vec<ValueId>> =
                values.iter().map(|v| components(value_map, *v)).collect();
            for (slot, (s, r)) in slices.into_iter().zip(results).enumerate() {
                let slot_values: Vec<ValueId> =
                    value_components.iter().map(|vc| vc[slot]).collect();
                out.push(OpCode::SlicePush {
                    dir: *dir,
                    result: r,
                    slice: s,
                    values: slot_values,
                });
            }
        }
        OpCode::SliceLen { result, slice } => {
            // Every component slice shares the original slice's length, so read it off the first
            // leaf. The result is a scalar `u32`, hence a single component.
            let slices = components(value_map, *slice);
            debug_assert!(
                !slices.is_empty(),
                "elide_tuples: SliceLen on a leaf-less slice (e.g. Slice<()>); \
                 the length is unrepresentable after scalarization"
            );
            let r = single(value_map, *result);
            out.push(OpCode::SliceLen {
                result: r,
                slice: slices[0],
            });
        }
        OpCode::Todo {
            payload,
            results,
            result_types,
        } => {
            out.push(OpCode::Todo {
                payload: payload.clone(),
                results: flat_components(value_map, results),
                result_types: result_types.iter().flat_map(leaf_types).collect(),
            });
        }
        OpCode::Guard { condition, inner } => {
            let cond = single(value_map, *condition);
            let mut inner_out = Vec::new();
            lower_instruction(
                inner,
                value_map,
                global_offsets,
                old_global_types,
                &mut inner_out,
            );
            for inner_op in inner_out {
                out.push(OpCode::Guard {
                    condition: cond,
                    inner: Box::new(inner_op),
                });
            }
        }
        // Every remaining opcode operates only on tuple-free (scalar or array-of-scalar) values, so
        // each operand and result has exactly one component. Remap them in place.
        _ => {
            let mut clone = op.clone();
            for id in clone.get_operands_mut() {
                *id = single(value_map, *id);
            }
            out.push(clone);
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Component-map helpers
// ---------------------------------------------------------------------------------------------

/// The component leaves of `value`. Values absent from the map (constants) are their own single
/// component.
fn components(value_map: &HashMap<ValueId, Vec<ValueId>>, value: ValueId) -> Vec<ValueId> {
    value_map
        .get(&value)
        .cloned()
        .unwrap_or_else(|| vec![value])
}

/// The single component of a value that must be tuple-free (panics otherwise — a useful invariant).
fn single(value_map: &HashMap<ValueId, Vec<ValueId>>, value: ValueId) -> ValueId {
    let comps = components(value_map, value);
    assert_eq!(
        comps.len(),
        1,
        "elide_tuples: expected a single-component value, got {} components for v{}",
        comps.len(),
        value.0
    );
    comps[0]
}

/// Flatten a list of values into the concatenation of their components.
fn flat_components(value_map: &HashMap<ValueId, Vec<ValueId>>, values: &[ValueId]) -> Vec<ValueId> {
    values
        .iter()
        .flat_map(|v| components(value_map, *v))
        .collect()
}

/// Allocate components for `value` of type `ty`. Tuple-free types are mapped to the value itself to
/// avoid needless renaming; tuple-bearing types get one fresh id per leaf.
fn alloc_components(ssa: &HLSSA, value: ValueId, ty: &Type) -> Vec<ValueId> {
    if !contains_tuple(ty) {
        vec![value]
    } else {
        (0..slot_count(ty)).map(|_| ssa.fresh_value()).collect()
    }
}

// ---------------------------------------------------------------------------------------------
// Type flattening core
// ---------------------------------------------------------------------------------------------

/// Whether a type contains a tuple anywhere within it.
pub fn contains_tuple(ty: &Type) -> bool {
    match &ty.expr {
        TypeExpr::Tuple(_) => true,
        TypeExpr::Array(inner, _)
        | TypeExpr::Slice(inner)
        | TypeExpr::Ref(inner)
        | TypeExpr::WitnessOf(inner) => contains_tuple(inner),
        TypeExpr::Field
        | TypeExpr::U(_)
        | TypeExpr::I(_)
        | TypeExpr::Function
        | TypeExpr::Blob(..) => false,
    }
}

/// The number of tuple-free leaves a type expands into (0 for unit and aggregates of unit).
fn slot_count(ty: &Type) -> usize {
    match &ty.expr {
        TypeExpr::Tuple(elements) => elements.iter().map(slot_count).sum(),
        TypeExpr::Array(inner, _)
        | TypeExpr::Slice(inner)
        | TypeExpr::Ref(inner)
        | TypeExpr::WitnessOf(inner) => slot_count(inner),
        TypeExpr::Field
        | TypeExpr::U(_)
        | TypeExpr::I(_)
        | TypeExpr::Function
        | TypeExpr::Blob(..) => 1,
    }
}

/// Wrap a leaf in `WitnessOf`, avoiding the (illegal) double wrap when the leaf is already witnessed.
fn witness_of_leaf(leaf: Type) -> Type {
    if leaf.is_witness_of() {
        leaf
    } else {
        Type::witness_of(leaf)
    }
}

/// Flatten a type into its ordered list of tuple-free leaf types, pushing the tuple constructor
/// upward through `Array`/`Slice`/`Ref`/`WitnessOf`.
fn leaf_types(ty: &Type) -> Vec<Type> {
    match &ty.expr {
        TypeExpr::Field
        | TypeExpr::U(_)
        | TypeExpr::I(_)
        | TypeExpr::Function
        | TypeExpr::Blob(..) => vec![ty.clone()],
        TypeExpr::Tuple(elements) => elements.iter().flat_map(leaf_types).collect(),
        TypeExpr::Array(inner, n) => leaf_types(inner)
            .into_iter()
            .map(|leaf| leaf.array_of(*n))
            .collect(),
        TypeExpr::Slice(inner) => leaf_types(inner)
            .into_iter()
            .map(|leaf| leaf.slice_of())
            .collect(),
        TypeExpr::Ref(inner) => leaf_types(inner)
            .into_iter()
            .map(|leaf| leaf.ref_of())
            .collect(),
        TypeExpr::WitnessOf(inner) => leaf_types(inner).into_iter().map(witness_of_leaf).collect(),
    }
}

// ---------------------------------------------------------------------------------------------
// Verification
// ---------------------------------------------------------------------------------------------

/// Assert that no tuple types or tuple opcodes survive anywhere in the SSA. Intended for debug
/// builds; panics with an ICE-style message on the first violation.
pub fn verify_tuple_free(ssa: &HLSSA) {
    for (i, gty) in ssa.get_global_types().iter().enumerate() {
        assert!(
            !contains_tuple(gty),
            "elide_tuples verification: global {} still has tuple type {}",
            i,
            gty
        );
    }
    for (fid, func) in ssa.iter_functions() {
        for ret in func.get_returns() {
            assert!(
                !contains_tuple(ret),
                "elide_tuples verification: fn {:?} return type {} still contains a tuple",
                fid,
                ret
            );
        }
        for (bid, block) in func.get_blocks() {
            for (pid, ty) in block.get_parameters() {
                assert!(
                    !contains_tuple(ty),
                    "elide_tuples verification: fn {:?} block_{} param v{} has tuple type {}",
                    fid,
                    bid.0,
                    pid.0,
                    ty
                );
            }
            for instr in block.get_instructions() {
                verify_op_tuple_free(instr, *fid, *bid);
            }
        }
    }
}

fn verify_op_tuple_free(op: &OpCode, fid: FunctionId, bid: BlockId) {
    let assert_free = |ty: &Type, what: &str| {
        assert!(
            !contains_tuple(ty),
            "elide_tuples verification: fn {:?} block_{} {} still contains tuple type {}",
            fid,
            bid.0,
            what,
            ty
        );
    };
    match op {
        OpCode::MkTuple { .. } => panic!(
            "elide_tuples verification: fn {:?} block_{} still contains a MkTuple",
            fid, bid.0
        ),
        OpCode::TupleProj { .. } => panic!(
            "elide_tuples verification: fn {:?} block_{} still contains a TupleProj",
            fid, bid.0
        ),
        OpCode::TupleRefProj { .. } => panic!(
            "elide_tuples verification: fn {:?} block_{} still contains a TupleRefProj",
            fid, bid.0
        ),
        OpCode::Alloc { elem_type, .. } => assert_free(elem_type, "alloc element type"),
        OpCode::MkSeq { elem_type, .. } | OpCode::MkRepeated { elem_type, .. } => {
            assert_free(elem_type, "sequence element type")
        }
        OpCode::FreshWitness { result_type, .. } => assert_free(result_type, "fresh_witness type"),
        OpCode::ReadGlobal { result_type, .. } => assert_free(result_type, "read_global type"),
        OpCode::Todo { result_types, .. } => {
            for ty in result_types {
                assert_free(ty, "todo result type");
            }
        }
        OpCode::Guard { inner, .. } => verify_op_tuple_free(inner, fid, bid),
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn field() -> Type {
        Type::field()
    }
    fn u32t() -> Type {
        Type::u(32)
    }

    #[test]
    fn scalar_is_single_leaf() {
        assert_eq!(leaf_types(&field()), vec![field()]);
        assert_eq!(slot_count(&field()), 1);
        assert!(!contains_tuple(&field()));
    }

    #[test]
    fn bare_tuple_flattens() {
        let t = Type::tuple_of(vec![field(), u32t()]);
        assert_eq!(leaf_types(&t), vec![field(), u32t()]);
        assert_eq!(slot_count(&t), 2);
        assert!(contains_tuple(&t));
    }

    #[test]
    fn nested_tuple_flattens() {
        // ((Field, u32), Field) -> Field, u32, Field
        let inner = Type::tuple_of(vec![field(), u32t()]);
        let t = Type::tuple_of(vec![inner, field()]);
        assert_eq!(leaf_types(&t), vec![field(), u32t(), field()]);
        assert_eq!(slot_count(&t), 3);
    }

    #[test]
    fn empty_tuple_has_no_leaves() {
        let unit = Type::tuple_of(vec![]);
        assert_eq!(leaf_types(&unit), Vec::<Type>::new());
        assert_eq!(slot_count(&unit), 0);
        assert!(contains_tuple(&unit));
        // Aggregates of unit collapse to zero leaves too.
        assert_eq!(slot_count(&unit.clone().array_of(4)), 0);
        assert_eq!(leaf_types(&unit.ref_of()), Vec::<Type>::new());
    }

    #[test]
    fn ref_of_tuple_pushes_ref_inward() {
        let t = Type::tuple_of(vec![field(), u32t()]).ref_of();
        assert_eq!(leaf_types(&t), vec![field().ref_of(), u32t().ref_of()]);
        assert_eq!(slot_count(&t), 2);
    }

    #[test]
    fn array_of_tuple_becomes_arrays_of_values() {
        let t = Type::tuple_of(vec![field(), u32t()]).array_of(3);
        assert_eq!(
            leaf_types(&t),
            vec![field().array_of(3), u32t().array_of(3)]
        );
        // One whole array per leaf — NOT multiplied by the length.
        assert_eq!(slot_count(&t), 2);
    }

    #[test]
    fn ref_of_array_of_tuple() {
        let t = Type::tuple_of(vec![field(), u32t()]).array_of(2).ref_of();
        assert_eq!(
            leaf_types(&t),
            vec![field().array_of(2).ref_of(), u32t().array_of(2).ref_of()]
        );
    }

    #[test]
    fn witness_of_tuple_distributes_without_double_wrap() {
        // WitnessOf(Tuple(Field, WitnessOf(Field))) -> WitnessOf(Field), WitnessOf(Field)
        let t = Type::witness_of(Type::tuple_of(vec![field(), Type::witness_of(field())]));
        let leaves = leaf_types(&t);
        assert_eq!(
            leaves,
            vec![Type::witness_of(field()), Type::witness_of(field())]
        );
        // No leaf is doubly-wrapped.
        for leaf in &leaves {
            assert!(!matches!(
                &leaf.expr,
                TypeExpr::WitnessOf(inner) if inner.is_witness_of()
            ));
        }
    }

    #[test]
    fn function_is_an_opaque_leaf() {
        let t = Type::tuple_of(vec![field(), Type::function()]);
        assert_eq!(leaf_types(&t), vec![field(), Type::function()]);
        assert!(!contains_tuple(&Type::function()));
    }

    #[test]
    fn global_opcodes_renumber_and_fan_out() {
        // Original globals: [Field, (Field, Array<Field;4>), u32]
        //   slot_counts = [1, 2, 1]  ->  base offsets = [0, 1, 3]
        // so the tuple global at old slot 1 splits into new slots 1,2 and the trailing u32 shifts 2->3.
        let arr = field().array_of(4);
        let tuple_global = Type::tuple_of(vec![field(), arr.clone()]);
        let old_global_types = vec![field(), tuple_global.clone(), u32t()];
        let global_offsets = vec![0usize, 1, 3];

        let mut value_map: HashMap<ValueId, Vec<ValueId>> = HashMap::default();
        value_map.insert(ValueId(10), vec![ValueId(11), ValueId(12)]); // read result
        value_map.insert(ValueId(20), vec![ValueId(21), ValueId(22)]); // init value

        // ReadGlobal of the tuple global -> two reads at offsets 1 and 2 with the leaf types.
        let mut out = Vec::new();
        lower_instruction(
            &OpCode::ReadGlobal {
                result: ValueId(10),
                offset: 1,
                result_type: tuple_global.clone(),
            },
            &value_map,
            &global_offsets,
            &old_global_types,
            &mut out,
        );
        assert_eq!(out.len(), 2);
        match (&out[0], &out[1]) {
            (
                OpCode::ReadGlobal {
                    result: r0,
                    offset: o0,
                    result_type: t0,
                },
                OpCode::ReadGlobal {
                    result: r1,
                    offset: o1,
                    result_type: t1,
                },
            ) => {
                assert_eq!((*r0, *o0, t0.clone()), (ValueId(11), 1, field()));
                assert_eq!((*r1, *o1, t1.clone()), (ValueId(12), 2, arr.clone()));
            }
            _ => panic!("expected two ReadGlobal ops, got {:?}", out),
        }

        // InitGlobal of the tuple global -> two inits at slots 1 and 2.
        let mut out = Vec::new();
        lower_instruction(
            &OpCode::InitGlobal {
                global: 1,
                value: ValueId(20),
            },
            &value_map,
            &global_offsets,
            &old_global_types,
            &mut out,
        );
        assert_eq!(out.len(), 2);
        match (&out[0], &out[1]) {
            (
                OpCode::InitGlobal {
                    global: g0,
                    value: v0,
                },
                OpCode::InitGlobal {
                    global: g1,
                    value: v1,
                },
            ) => {
                assert_eq!((*g0, *v0), (1, ValueId(21)));
                assert_eq!((*g1, *v1), (2, ValueId(22)));
            }
            _ => panic!("expected two InitGlobal ops, got {:?}", out),
        }

        // DropGlobal of the tuple global -> only the heap-allocated leaf (the array at new slot 2).
        let mut out = Vec::new();
        lower_instruction(
            &OpCode::DropGlobal { global: 1 },
            &value_map,
            &global_offsets,
            &old_global_types,
            &mut out,
        );
        assert_eq!(out.len(), 1);
        match &out[0] {
            OpCode::DropGlobal { global } => assert_eq!(*global, 2),
            other => panic!("expected one DropGlobal{{2}}, got {:?}", other),
        }

        // The trailing scalar global keeps its (renumbered) single slot: old 2 -> new 3.
        let mut out = Vec::new();
        lower_instruction(
            &OpCode::ReadGlobal {
                result: ValueId(30),
                offset: 2,
                result_type: u32t(),
            },
            &value_map,
            &global_offsets,
            &old_global_types,
            &mut out,
        );
        assert_eq!(out.len(), 1);
        match &out[0] {
            OpCode::ReadGlobal {
                result,
                offset,
                result_type,
            } => assert_eq!(
                (*result, *offset, result_type.clone()),
                (ValueId(30), 3, u32t())
            ),
            other => panic!("expected one ReadGlobal, got {:?}", other),
        }
    }
}
