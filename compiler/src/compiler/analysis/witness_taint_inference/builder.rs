//! The shared `≥`-graph builder.
//!
//! Given a function instance, its types, its CFG, and the current callee summaries,
//! [`build_graph`] produces a [`TaintGraph`] over [`Position`]s that encodes every taint edge of
//! the function, including the formal input/output wiring needed to extract a summary (phase 1) or
//! to seed concrete inputs and read off concrete shapes (phase 2).
//!
//! Memory is resolved by unification: every copy-shaped flow is two-way (`≡`) at Deref-descended
//! levels and covariant (`≥`) at value levels (see [`GraphBuilder::copy_levels`]), so all aliases
//! of an object share its pointee positions and reference invariance emerges. A store taints the
//! pointee's scalars covariantly (`*ptr ≥ value`, never the reverse) plus the pointer's own handle
//! taint (which slot was written is witness-dependent under a witness-selected pointer), and a
//! load reads them.

use crate::collections::HashMap;
use crate::compiler::{
    analysis::{
        flow_analysis::CFG,
        types::FunctionTypeInfo,
        witness_taint_inference::{
            FunctionSummary, WitnessTaint,
            position::{Descent, Owner, Position, paths_of_type},
        },
    },
    ssa::{
        BlockId, FunctionId, Terminator, ValueId,
        hlssa::{CallTarget, CastTarget, HLFunction, OpCode, Radix, Type, TypeExpr},
        traits::Instruction,
    },
    util::ice_non_elided_tuple,
};

// GRAPH BUILDER
// ================================================================================================

/// Accumulates the `≥` edges for one function instance while [`build_graph`] walks it.
struct GraphBuilder<'a> {
    /// Types of the function's SSA values, used to enumerate the levels of each position.
    types: &'a FunctionTypeInfo,

    /// The callee summaries known so far, instantiated at each constrained call site.
    summaries: &'a HashMap<FunctionId, FunctionSummary>,

    /// The `≥` graph being built.
    graph: WitnessTaint,

    /// Whether the current instruction applied the cf-taint rule.
    cf_taint_applied: bool,
}

impl<'a> GraphBuilder<'a> {
    /// The root position of SSA value `value` (its top level).
    fn value_position(&self, value: ValueId) -> Position {
        Position::root(Owner::Value(value))
    }

    /// The type of SSA value `value`.
    ///
    /// The borrow is tied to the type table, not `self`, so it can be held across `&mut self`
    /// calls without cloning the type.
    fn value_type(&self, value: ValueId) -> &'a Type {
        self.types.get_value_type(value)
    }

    /// Copy taint between two values of type `ty`, level for level (see [`Self::copy_levels`]).
    fn copy_taint(&mut self, dst: ValueId, src: ValueId, ty: &Type) {
        self.copy_levels(self.value_position(dst), self.value_position(src), ty);
    }

    /// The unification copy rule: link `dst_base` and `src_base` level for level over `ty` —
    /// covariantly (`dst·p ≥ src·p`) at value levels, **two-way** (`dst·p ≡ src·p`) at every
    /// Deref-descended level.
    ///
    /// Deref-descended levels name shared mutable memory, and a copy makes the two sides aliases
    /// of it: the equations collapse every alias's pointee positions into one equivalence class,
    /// so a store through any alias is visible through every other — reference invariance without
    /// a points-to analysis. Aliasing is thus equality-based (Steensgaard-style), deliberately
    /// coarser than inclusion-based points-to; see the precision note in the module docs.
    fn copy_levels(&mut self, dst_base: Position, src_base: Position, ty: &Type) {
        for path in paths_of_type(ty) {
            let dst = dst_base.extend(&path);
            let src = src_base.extend(&path);
            if path.contains(&Descent::Deref) {
                self.graph.add_eq(dst, src);
            } else {
                self.graph.add_ge(dst, src);
            }
        }
    }

    /// Map a callee summary's formal position to the matching caller position(s) at a call site.
    fn map_formal(
        &self,
        formal: &Position,
        args: &[ValueId],
        results: &[ValueId],
        call_conditions: &[ValueId],
    ) -> Vec<Position> {
        match &formal.owner {
            Owner::Param(i) => vec![self.value_position(args[*i]).extend(&formal.path)],
            Owner::Return(j) => vec![self.value_position(results[*j]).extend(&formal.path)],
            Owner::Cfg => {
                let mut positions = vec![Position::root(Owner::Cfg)];
                for cond in call_conditions {
                    positions.push(self.value_position(*cond));
                }
                positions
            }
            Owner::Global(g) => vec![Position {
                owner: Owner::Global(*g),
                path: formal.path.clone(),
            }],
            Owner::Top => vec![Position::top()],
            Owner::Value(_) => {
                panic!("ICE: summary referenced a non-formal position {formal:?}")
            }
        }
    }

    /// Add `target ≥ cf_taint` for the function flag and every dominating witness branch condition
    /// — the "writes under witness control flow taint the written leaf" rule.
    fn add_cf_taint_to(&mut self, target: &Position, branch_conditions: &[ValueId]) {
        self.cf_taint_applied = true;
        self.graph
            .add_ge(target.clone(), Position::root(Owner::Cfg));
        for cond in branch_conditions {
            self.graph
                .add_ge(target.clone(), self.value_position(*cond));
        }
    }
}

// BUILDER FUNCTIONS
// ================================================================================================

/// For each block, the branch conditions of every `JmpIf` whose body contains it.
///
/// All dominating conditions are collected, witness or not — their witness-ness is only decided by
/// the solver, and an edge from a pure condition contributes no taint.
///
/// Summary-independent, so callers can compute it once per function and reuse it across every
/// [`build_graph`] call.
pub fn compute_block_conditions(func: &HLFunction, cfg: &CFG) -> HashMap<BlockId, Vec<ValueId>> {
    let mut block_conditions: HashMap<BlockId, Vec<ValueId>> = HashMap::default();
    for (bid, block) in func.get_blocks() {
        if let Some(Terminator::JmpIf(cond, _, _)) = block.get_terminator() {
            for body in cfg.get_if_body(*bid) {
                block_conditions.entry(body).or_default().push(*cond);
            }
        }
        block_conditions.entry(*bid).or_default();
    }
    block_conditions
}

/// Build the `≥` graph for `func`.
///
/// `block_conditions` is the per-block dominating witness-branch conditions from
/// [`compute_block_conditions`].
pub fn build_graph(
    func: &HLFunction,
    types: &FunctionTypeInfo,
    cfg: &CFG,
    block_conditions: &HashMap<BlockId, Vec<ValueId>>,
    summaries: &HashMap<FunctionId, FunctionSummary>,
) -> WitnessTaint {
    let mut builder = GraphBuilder {
        types,
        summaries,
        graph: WitnessTaint::new(),
        cf_taint_applied: false,
    };

    // Formal inputs: each entry parameter value IS its formal `Param` position (an identity, not a
    // copy, hence two-way at every level).
    for (i, (value, ty)) in func.get_entry().get_parameters().enumerate() {
        for path in paths_of_type(ty) {
            builder.graph.add_eq(
                Position {
                    owner: Owner::Param(i),
                    path: path.clone(),
                },
                builder.value_position(*value).extend(&path),
            );
        }
    }

    // Instruction edges.
    let block_ids: Vec<BlockId> = func.get_blocks().map(|(bid, _)| *bid).collect();
    for bid in &block_ids {
        let block = func.get_block(*bid);
        let branch_conditions = block_conditions.get(bid).map_or(&[][..], Vec::as_slice);
        for instr in block.get_instructions() {
            builder.cf_taint_applied = false;
            build_instr(&mut builder, instr, branch_conditions);
            debug_assert_eq!(
                builder.cf_taint_applied,
                writes_under_witness_cf(instr),
                "ICE: cf-taint rule out of sync with writes_under_witness_cf for {instr:?}"
            );
        }
        // terminator
        match block.get_terminator() {
            Some(Terminator::Jmp(target, jump_args)) => {
                let params = func.get_block(*target).get_parameters();
                for ((param, param_type), arg) in params.zip(jump_args.iter()) {
                    builder.copy_taint(*param, *arg, param_type);
                }
            }
            Some(Terminator::JmpIf(cond, _, _)) => {
                // Push the condition into the leaves of the merge-point phi parameters.
                let merge = cfg.get_merge_point(*bid);
                let cond = *cond;
                for (merge_param, merge_type) in func.get_block(merge).get_parameters() {
                    for path in leaf_paths(merge_type) {
                        builder.graph.add_ge(
                            builder.value_position(*merge_param).extend(&path),
                            builder.value_position(cond),
                        );
                    }
                }
            }
            Some(Terminator::Return(values)) => {
                // A return is a copy into the `Return` formals: covariant at value levels, two-way
                // at Deref levels — the Deref levels of a returned value are caller-writable
                // shared memory, exactly like ref-parameter pointees, so taint the caller injects
                // through the returned ref flows back into the returned value's aliases.
                for (j, value) in values.iter().enumerate() {
                    let return_type = &func.get_returns()[j];
                    builder.copy_levels(
                        Position::root(Owner::Return(j)),
                        builder.value_position(*value),
                        return_type,
                    );
                }
            }
            None => {}
        }
    }

    builder.graph
}

fn build_instr(builder: &mut GraphBuilder, instr: &OpCode, branch_conditions: &[ValueId]) {
    match instr {
        OpCode::Cmp {
            result, lhs, rhs, ..
        }
        | OpCode::BinaryArithOp {
            result, lhs, rhs, ..
        } => {
            // scalar result ≥ each operand's top
            builder.graph.add_ge(
                builder.value_position(*result),
                builder.value_position(*lhs),
            );
            builder.graph.add_ge(
                builder.value_position(*result),
                builder.value_position(*rhs),
            );
        }
        OpCode::Cast {
            result,
            value,
            target: CastTarget::WitnessOf,
        } => {
            // A WitnessOf cast is a witness *source* (emitted before inference, e.g. the default
            // element for an array that will hold witness values): its result is unconditionally
            // Witness at the top level, like WriteWitness.
            let ty = builder.value_type(*result);
            builder.copy_taint(*result, *value, ty);
            builder
                .graph
                .add_ge(builder.value_position(*result), Position::top());
        }
        OpCode::Not { result, value }
        | OpCode::Cast { result, value, .. }
        | OpCode::SExt { result, value, .. }
        | OpCode::BitRange { result, value, .. }
        | OpCode::Spread { result, value, .. } => {
            let ty = builder.value_type(*result);
            builder.copy_taint(*result, *value, ty);
        }
        OpCode::Unspread {
            result_odd,
            result_even,
            value,
            ..
        } => {
            let ty = builder.value_type(*result_odd);
            builder.copy_taint(*result_odd, *value, ty);
            let ty = builder.value_type(*result_even);
            builder.copy_taint(*result_even, *value, ty);
        }
        OpCode::Select {
            result,
            cond,
            if_t,
            if_f,
        } => {
            let ty = builder.value_type(*result);
            builder.copy_taint(*result, *if_t, ty);
            builder.copy_taint(*result, *if_f, ty);
            builder.graph.add_ge(
                builder.value_position(*result),
                builder.value_position(*cond),
            );
        }
        OpCode::MkSeq { result, elems, .. } => {
            let element_type = builder
                .value_type(*result)
                .peel_witness()
                .get_array_element();
            for element in elems {
                builder.copy_levels(
                    builder.value_position(*result).child(Descent::Elem),
                    builder.value_position(*element),
                    &element_type,
                );
            }
        }
        OpCode::MkRepeated {
            result, element, ..
        } => {
            let element_type = builder
                .value_type(*result)
                .peel_witness()
                .get_array_element();
            builder.copy_levels(
                builder.value_position(*result).child(Descent::Elem),
                builder.value_position(*element),
                &element_type,
            );
        }
        OpCode::ArrayGet {
            result,
            array,
            index,
        } => {
            let result_type = builder.value_type(*result);
            builder.copy_levels(
                builder.value_position(*result),
                builder.value_position(*array).child(Descent::Elem),
                result_type,
            );
            // a dynamic (or witness) index / array handle taints the read value's leaves
            for path in leaf_paths(result_type) {
                builder.graph.add_ge(
                    builder.value_position(*result).extend(&path),
                    builder.value_position(*index),
                );
                builder.graph.add_ge(
                    builder.value_position(*result).extend(&path),
                    builder.value_position(*array),
                );
            }
        }
        OpCode::ArraySet {
            result,
            array,
            index,
            value,
        } => {
            let element_type = builder
                .value_type(*result)
                .peel_witness()
                .get_array_element();
            builder.copy_levels(
                builder.value_position(*result).child(Descent::Elem),
                builder.value_position(*array).child(Descent::Elem),
                &element_type,
            );
            builder.copy_levels(
                builder.value_position(*result).child(Descent::Elem),
                builder.value_position(*value),
                &element_type,
            );
            // result array handle ≥ source array handle
            builder.graph.add_ge(
                builder.value_position(*result),
                builder.value_position(*array),
            );
            // The written element leaves pick up the index taint and, under witness control flow,
            // the cfg flag — a conditional set leaves the element witness-dependent.
            for path in leaf_paths(&element_type) {
                let written = builder
                    .value_position(*result)
                    .child(Descent::Elem)
                    .extend(&path);
                builder
                    .graph
                    .add_ge(written.clone(), builder.value_position(*index));
                builder.add_cf_taint_to(&written, branch_conditions);
            }
        }
        OpCode::SlicePush {
            result,
            slice,
            values,
            ..
        } => {
            let element_type = builder
                .value_type(*result)
                .peel_witness()
                .get_array_element();
            builder.copy_levels(
                builder.value_position(*result).child(Descent::Elem),
                builder.value_position(*slice).child(Descent::Elem),
                &element_type,
            );
            for value in values {
                builder.copy_levels(
                    builder.value_position(*result).child(Descent::Elem),
                    builder.value_position(*value),
                    &element_type,
                );
            }
            builder.graph.add_ge(
                builder.value_position(*result),
                builder.value_position(*slice),
            );

            // Under witness control flow a conditional push leaves the appended elements
            // witness-dependent.
            for path in leaf_paths(&element_type) {
                let written = builder
                    .value_position(*result)
                    .child(Descent::Elem)
                    .extend(&path);
                builder.add_cf_taint_to(&written, branch_conditions);
            }
        }
        OpCode::SliceLen { .. } => {
            // No edges: the length is structural metadata, Pure for every slice this analysis
            // can express. KNOWN LIMITATION (inherited from the previous pass): a slice whose
            // *length* is witness-dependent has no representation — `leaf_paths` deliberately
            // excludes container-top levels, so nothing ever taints a slice's top and there is
            // no source for SliceLen to read. The phi-merge route (a slice merged at a witness
            // JmpIf) is rejected loudly downstream (`UntaintControlFlow` panics on witness merge
            // selects over slices); the ref route (a witness-conditional store of a slice
            // through a ref) would need WitnessOf-slice support throughout untaint/lowering
            // before this rule can be tightened.
        }
        OpCode::MkSeqOfBlob { .. } => {
            // the blob is compile-time constant data: the result starts Pure. (no edges)
        }
        OpCode::ToBits { result, value, .. } => {
            // each bit ≥ value
            builder.graph.add_ge(
                builder.value_position(*result).child(Descent::Elem),
                builder.value_position(*value),
            );
        }
        OpCode::ToRadix {
            result,
            value,
            radix,
            ..
        } => {
            builder.graph.add_ge(
                builder.value_position(*result).child(Descent::Elem),
                builder.value_position(*value),
            );
            if let Radix::Dyn(radix_value) = radix {
                builder.graph.add_ge(
                    builder.value_position(*result).child(Descent::Elem),
                    builder.value_position(*radix_value),
                );
            }
        }
        OpCode::Alloc { result, value, .. } => {
            // The ref handle itself is Pure but the allocation carries an
            // initial value that is written into the cell.
            let pointee = match &builder.value_type(*result).peel_witness().expr {
                TypeExpr::Ref(inner) => &**inner,
                other => panic!("ICE: Alloc result of a non-ref type {other:?}"),
            };
            let slot = builder.value_position(*result).child(Descent::Deref);
            builder.copy_levels(slot.clone(), builder.value_position(*value), pointee);
        }
        OpCode::Store { ptr, value } => {
            // A store writes the pointee: covariant at the written value levels (`*ptr ≥ value`,
            // never the reverse), two-way at nested-ref levels (storing a ref publishes it — the
            // slot's content and the stored ref become aliases).
            let pointee = match &builder.value_type(*ptr).peel_witness().expr {
                TypeExpr::Ref(inner) => &**inner,
                other => panic!("ICE: Store through a non-ref value of type {other:?}"),
            };
            let slot = builder.value_position(*ptr).child(Descent::Deref);
            builder.copy_levels(slot.clone(), builder.value_position(*value), pointee);

            // A conditional store leaves shared memory witness-dependent. As everywhere, the
            // taintable leaves include a nested ref's *handle* level: which ref a slot holds after
            // a conditional ref store is itself witness-dependent (the unification of the
            // candidates' pointees does not capture that).
            //
            // A store through a witness-selected pointer likewise leaves the written leaves
            // witness-dependent — *which* slot received the value depends on the witness — so the
            // pointer's own handle taint flows into them: the dual of Load's `result ≥ ptr`.
            // (Unification spreads this over every alias of the candidate slots, the usual
            // equality-based over-approximation.)
            for path in leaf_paths(pointee) {
                let written = slot.extend(&path);
                builder
                    .graph
                    .add_ge(written.clone(), builder.value_position(*ptr));
                builder.add_cf_taint_to(&written, branch_conditions);
            }
        }
        OpCode::Load { result, ptr } => {
            // A load reads the pointee: covariant at value levels, two-way at nested-ref levels (a
            // loaded ref aliases the stored one).
            let pointee = match &builder.value_type(*ptr).peel_witness().expr {
                TypeExpr::Ref(inner) => &**inner,
                other => panic!("ICE: Load through a non-ref value of type {other:?}"),
            };
            builder.copy_levels(
                builder.value_position(*result),
                builder.value_position(*ptr).child(Descent::Deref),
                pointee,
            );
            // a witness-selected pointer taints the read value's top
            builder.graph.add_ge(
                builder.value_position(*result),
                builder.value_position(*ptr),
            );
        }
        OpCode::WriteWitness { result, .. } => {
            if let Some(r) = result {
                builder
                    .graph
                    .add_ge(builder.value_position(*r), Position::top());
            }
        }
        OpCode::Call {
            results,
            function: CallTarget::Static(callee),
            args,
            unconstrained,
        } => {
            if *unconstrained {
                // Unconstrained results are Pure; their input dependence re-enters via
                // WriteWitness.
                return;
            }
            // The summary map is pre-populated for every function before the phase-1 worklist runs
            // (it merely starts empty), so a missing entry means the callee was never analyzed —
            // instantiating nothing for it would be silently unsound.
            let summary = builder
                .summaries
                .get(callee)
                .expect("ICE: no summary for constrained static call target");
            // Summary edges share endpoints heavily, so memoize the formal → caller-position
            // mapping per call site.
            let mut mapped: HashMap<&Position, Vec<Position>> = HashMap::default();
            for (summary_sink, summary_source) in &summary.edges {
                for formal in [summary_sink, summary_source] {
                    if !mapped.contains_key(formal) {
                        let positions =
                            builder.map_formal(formal, args, results, branch_conditions);
                        mapped.insert(formal, positions);
                    }
                }
                for caller_sink in &mapped[summary_sink] {
                    for caller_source in &mapped[summary_source] {
                        builder
                            .graph
                            .add_ge(caller_sink.clone(), caller_source.clone());
                    }
                }
            }
        }
        OpCode::Call {
            function: CallTarget::Dynamic(_),
            ..
        } => panic!("ICE: dynamic call target during witness-taint inference"),
        OpCode::ReadGlobal {
            result,
            offset,
            result_type,
        } => {
            // A read copies the program-wide global slot's taint into the result, level for level
            // — two-way at Deref levels, so every read of a ref-carrying global aliases the same
            // program-wide pointee positions.
            builder.copy_levels(
                builder.value_position(*result),
                Position::root(Owner::Global(*offset as usize)),
                result_type,
            );
        }
        OpCode::InitGlobal { global, value } => {
            // An init is a covariant store into the program-wide global slot (two-way at Deref
            // levels), plus the cfg flag on the written leaf if the init runs under witness
            // control flow.
            let slot = Position::root(Owner::Global(*global));
            let value_type = builder.value_type(*value);
            builder.copy_levels(slot.clone(), builder.value_position(*value), value_type);
            for path in leaf_paths(value_type) {
                let written = slot.extend(&path);
                builder.add_cf_taint_to(&written, branch_conditions);
            }
        }
        OpCode::DropGlobal { .. }
        | OpCode::Assert { .. }
        | OpCode::AssertCmp { .. }
        | OpCode::AssertR1C { .. }
        | OpCode::Rangecheck { .. }
        | OpCode::MemOp { .. } => {
            // no taint flow
        }
        OpCode::TupleProj { .. } | OpCode::TupleRefProj { .. } | OpCode::MkTuple { .. } => {
            ice_non_elided_tuple()
        }
        OpCode::Guard { condition, inner } => {
            // Phase 2 discovers and rewires call contexts by matching bare `OpCode::Call` only: a
            // Guard-wrapped constrained call would be analyzed here but never contextualized,
            // cloned, or rewired — silently. Fail loudly instead; nothing emits Guard-wrapped calls
            // before this pass (Guards are introduced by UntaintControlFlow and later).
            assert!(
                !matches!(
                    **inner,
                    OpCode::Call {
                        unconstrained: false,
                        ..
                    }
                ),
                "ICE: Guard-wrapped constrained call during witness-taint inference: {inner:?}"
            );

            // A guard predicates `inner` on `condition` (LowerGuards turns it into a JmpIf whose
            // merge phis pick the inner's results or defaults), so delegate to the inner opcode
            // with the condition joined into the dominating conditions — the cf-taint rule on the
            // inner's writes must see it — and push the condition into every result leaf, exactly
            // as the JmpIf merge rule would after lowering.
            let mut conditions = branch_conditions.to_vec();
            conditions.push(*condition);
            build_instr(builder, inner, &conditions);
            for result in inner.get_results() {
                for path in leaf_paths(builder.value_type(*result)) {
                    builder.graph.add_ge(
                        builder.value_position(*result).extend(&path),
                        builder.value_position(*condition),
                    );
                }
            }
        }
        OpCode::FreshWitness { .. }
        | OpCode::Constrain { .. }
        | OpCode::BumpD { .. }
        | OpCode::NextDCoeff { .. }
        | OpCode::MulConst { .. }
        | OpCode::Lookup { .. }
        | OpCode::DLookup { .. }
        | OpCode::Todo { .. } => {
            panic!("ICE: opcode should not be present during witness-taint inference: {instr:?}")
        }
    }
}

// UTILITY FUNCTIONS
// ================================================================================================

/// Whether executing `op` under witness control flow must taint what it writes — the
/// `add_cf_taint_to` rule for ops whose effect outlives the instruction (functional array/slice
/// updates, stores through refs, global inits).
///
/// Deliberately exhaustive: adding an opcode forces an explicit decision here, and `build_graph`
/// debug-asserts after every instruction that this list agrees with what `build_instr` applied.
fn writes_under_witness_cf(op: &OpCode) -> bool {
    match op {
        OpCode::ArraySet { .. }
        | OpCode::SlicePush { .. }
        | OpCode::Store { .. }
        | OpCode::Alloc { .. }
        | OpCode::InitGlobal { .. } => true,
        OpCode::Guard { inner, .. } => writes_under_witness_cf(inner),
        OpCode::Cmp { .. }
        | OpCode::BinaryArithOp { .. }
        | OpCode::Not { .. }
        | OpCode::Cast { .. }
        | OpCode::SExt { .. }
        | OpCode::BitRange { .. }
        | OpCode::Spread { .. }
        | OpCode::Unspread { .. }
        | OpCode::Select { .. }
        | OpCode::MkSeq { .. }
        | OpCode::MkRepeated { .. }
        | OpCode::ArrayGet { .. }
        | OpCode::SliceLen { .. }
        | OpCode::MkSeqOfBlob { .. }
        | OpCode::ToBits { .. }
        | OpCode::ToRadix { .. }
        | OpCode::Load { .. }
        | OpCode::WriteWitness { .. }
        | OpCode::Call { .. }
        | OpCode::ReadGlobal { .. }
        | OpCode::DropGlobal { .. }
        | OpCode::Assert { .. }
        | OpCode::AssertCmp { .. }
        | OpCode::AssertR1C { .. }
        | OpCode::Rangecheck { .. }
        | OpCode::MemOp { .. }
        | OpCode::TupleProj { .. }
        | OpCode::TupleRefProj { .. }
        | OpCode::MkTuple { .. }
        | OpCode::FreshWitness { .. }
        | OpCode::Constrain { .. }
        | OpCode::BumpD { .. }
        | OpCode::NextDCoeff { .. }
        | OpCode::MulConst { .. }
        | OpCode::Lookup { .. }
        | OpCode::DLookup { .. }
        | OpCode::Todo { .. } => false,
    }
}

/// Paths to the witness "leaves" of `ty` for the purpose of pushing in a scalar taint: scalar
/// leaves (descending arrays/slices), and the *root* of any `Ref` (a ref is opaque to leaf-pushing)
/// — the positions a control-flow taint lands on.
///
/// Container-top levels (array/slice roots) are deliberately not leaves: an array's identity is
/// fully covered by its element taint, and a witness-dependent slice *length* is not expressible
/// in the current shape model (see the `SliceLen` rule in [`build_instr`]).
fn leaf_paths(ty: &Type) -> Vec<Vec<Descent>> {
    let mut out = Vec::new();
    fn go(ty: &Type, prefix: &mut Vec<Descent>, out: &mut Vec<Vec<Descent>>) {
        match &ty.peel_witness().expr {
            TypeExpr::Field
            | TypeExpr::U(_)
            | TypeExpr::I(_)
            | TypeExpr::Function
            | TypeExpr::Blob(..) => out.push(prefix.clone()),
            TypeExpr::Ref(_) => out.push(prefix.clone()),
            TypeExpr::Array(inner, _) | TypeExpr::Slice(inner) => {
                prefix.push(Descent::Elem);
                go(inner, prefix, out);
                prefix.pop();
            }
            TypeExpr::WitnessOf(_) | TypeExpr::Tuple(_) => {}
        }
    }
    go(ty, &mut Vec::new(), &mut out);
    out
}
