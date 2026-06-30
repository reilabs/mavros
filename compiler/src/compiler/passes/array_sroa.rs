//! Scalar replacement of aggregates for **Split fixed-size arrays**.
//!
//! The points-to analysis classifies each array/slice value into a flow-group that is either
//! `Split` (purely local, only ever constant-indexed, never crossing a boundary) or `Collapsed`.
//! `splittable_cells(f, v)` returns `Some(indices)` exactly for a `Split` group.
//!
//! This pass **peels every `Split` array value into one SSA value per cell**, deleting the array
//! aggregate: `MkSeq`/`MkSeqOfBlob`/`MkRepeated`/`ArrayGet`/`ArraySet` collapse into pure dataflow
//! and array-typed phis (block parameters) become per-cell phis. Cells of scalar arrays become
//! plain scalars (which `WitnessTaintInference` can then classify `Pure`); cells of ref arrays
//! become individual refs that the follow-up [`super::mem2reg`] promotes. It is the array analog of
//! tuple elision (which explodes tuples the same way), but driven by the analysis rather than by
//! syntax — so it fires only where aliasing is *proven* separable, never speculatively.
//!
//! ## One-Level Value Rewriting
//!
//! Three facts from the analysis ensure that it can be a simple, one-level value-rewriting pass.
//!
//! - **`Split` never crosses a boundary.** Every parameter, `Call` arg/result, `Return`,
//!   `InitGlobal`, `ReadGlobal`, and `Ref<Array>` store/load is a *collapse trigger*, so a `Split`
//!   array value appears _only_ in `MkSeq`/`MkSeqOfBlob`/`MkRepeated` (def), `ArrayGet`/`ArraySet`
//!   (constant access), the union-copy ops (`ArraySet`-result, `Select`, `Cast`), and
//!   block-param/`Jmp` phis.
//!   This pass therefore never touches function signatures, call sites, returns, or globals — it is
//!   purely intra-function value + phi rewriting. Entry-block array params are function formals and
//!   thus always `Collapsed`, so they are never peeled.
//! - **Every cell is a single component.** A peeled array's element is a scalar, a ref, or a
//!   _deeper_ (always-`Collapsed`) array — never another `Split` array, because an array-typed
//!   element of `MkSeq`/`MkRepeated`/`ArraySet` is itself a collapse trigger (see `array_cells`).
//!   So one cell maps to exactly one value, and the boundary guarantee above means a peeled array
//!   never reaches an instruction this pass keeps verbatim.
//! - **No reference-counting interaction.** `RCInsertion` runs only in the later witgen/AD
//!   pipelines, long after the pre-WTI phases; at this placement there are no `MemOp{Bump,Drop}`
//!   ops.
//!
//! ## Strategy
//!
//! This pass runs in two phases on a per-function basis over the dominator pre-order of its
//! reachable blocks.
//!
//! 1. **Plan:** Create a `value_map: ValueId -> Vec<ValueId>` for the peeled values only. A `Split`
//!    array maps to its `N` per-cell component values (index `k` at position `k` — the cell set is
//!    dense `0..N`). `MkSeq`/`MkSeqOfBlob`/`MkRepeated`/`ArrayGet`/`ArraySet` results *alias*
//!    slices of their operands' components (no fresh ids, no emitted op); `Select`/`Cast` results
//!    and split params get fresh per-cell ids. Everything else is absent from the map and is its
//!    own single component.
//! 2. **Rewrite:**: Split-array block params expand into `N` per-cell params; the aliasing
//!    producers are dropped; `Select`/`Cast` emit one op per cell; every other instruction is kept
//!    with its operands mapped through `single`; `Jmp` args expand a split-array arg into its `N`
//!    cell components (aligned by index across predecessors).

use crate::{
    collections::{HashMap, HashSet},
    compiler::{
        analysis::{
            flow_analysis::FlowAnalysis,
            points_to::PointsTo,
            types::{FunctionTypeInfo, TypeInfo},
        },
        pass_manager::{Analysis, AnalysisId, AnalysisStore, Pass},
        ssa::{
            BlockId, FunctionId, Instruction, Terminator, ValueId,
            hlssa::{Constant, HLSSA, OpCode, Type, TypeExpr},
        },
    },
};

// ARRAY SROA PASS
// ================================================================================================

pub struct ArraySroa {}

impl ArraySroa {
    pub fn new() -> Self {
        ArraySroa {}
    }
}

impl Pass for ArraySroa {
    fn name(&self) -> &'static str {
        "array_sroa"
    }

    fn needs(&self) -> Vec<AnalysisId> {
        vec![PointsTo::id(), TypeInfo::id(), FlowAnalysis::id()]
    }

    fn run(&self, ssa: &mut HLSSA, store: &AnalysisStore) {
        self.do_run(
            ssa,
            store.get::<FlowAnalysis>(),
            store.get::<TypeInfo>(),
            store.get::<PointsTo>(),
        );
    }

    fn preserves(&self) -> Vec<AnalysisId> {
        // New values, params and dropped/rewritten instructions invalidate every cached analysis.
        vec![]
    }
}

impl ArraySroa {
    pub fn do_run(
        &self,
        ssa: &mut HLSSA,
        cfg: &FlowAnalysis,
        types: &TypeInfo,
        points_to: &PointsTo,
    ) {
        let function_ids: Vec<FunctionId> = ssa.get_function_ids().collect();
        for fid in function_ids {
            // Only functions the points-to analysis covers can have `Split` arrays; others answer
            // `None` to every `splittable_cells` query and have no type info to read.
            if !types.has_function(fid) {
                continue;
            }
            let reachable: Vec<BlockId> = cfg
                .get_function_cfg(fid)
                .get_domination_pre_order()
                .collect();
            let plan =
                Self::plan_function(ssa, fid, &reachable, types.get_function(fid), points_to);
            if plan.value_map.is_empty() {
                continue; // nothing splittable in this function
            }
            Self::rewrite_function(ssa, fid, &reachable, &plan);
        }
    }

    /// Plan the per-cell component map for one function.
    fn plan_function(
        ssa: &HLSSA,
        fid: FunctionId,
        reachable: &[BlockId],
        fti: &FunctionTypeInfo,
        points_to: &PointsTo,
    ) -> Plan {
        let func = ssa.get_function(fid);
        let mut plan = Plan::default();

        // Sweep A: split-array block params get `N` fresh per-cell ids, resolved before any
        // instruction (which may reference a dominating block's parameter) is visited. Entry-block
        // params are formals and always `Collapsed`, so this only ever fires for phi params.
        for bid in reachable {
            for (pid, ty) in func.get_block(*bid).get_parameters() {
                if points_to.is_split(fid, *pid) {
                    let n = array_size(ty);
                    let cells: Vec<ValueId> = (0..n).map(|_| ssa.fresh_value()).collect();
                    plan.mark_split(*pid, cells);
                }
            }
        }

        // Sweep B: instructions in dominator pre-order, so an aliasing producer always sees its
        // array operand's components already resolved.
        for bid in reachable {
            for instr in func.get_block(*bid).get_instructions() {
                Self::plan_instruction(ssa, fid, fti, points_to, instr, &mut plan);
            }
        }

        plan
    }

    fn plan_instruction(
        ssa: &HLSSA,
        fid: FunctionId,
        fti: &FunctionTypeInfo,
        points_to: &PointsTo,
        instr: &OpCode,
        plan: &mut Plan,
    ) {
        let is_split = |v: ValueId| points_to.is_split(fid, v);
        match instr {
            // `MkSeq` regroups its elements as cells; emits no instruction. Each element is a
            // single component (a deeper array element is `Collapsed`), so `single` enforces that
            // invariant.
            OpCode::MkSeq { result, elems, .. } if is_split(*result) => {
                let cells: Vec<ValueId> =
                    elems.iter().map(|e| single(&plan.value_map, *e)).collect();
                debug_assert_dense(points_to, fid, *result, cells.len(), fti);
                plan.mark_split(*result, cells);
            }

            // A constant blob array peels into one materialized scalar constant per element.
            OpCode::MkSeqOfBlob { result, blob, .. } if is_split(*result) => {
                let elements = match &*ssa
                    .get_const(*blob)
                    .expect("ICE: MkSeqOfBlob blob is not a constant")
                {
                    Constant::Blob(b) => b.elements.clone(),
                    other => panic!("ICE: MkSeqOfBlob blob is not a Blob constant: {:?}", other),
                };
                let cells: Vec<ValueId> = elements.into_iter().map(|c| ssa.add_const(c)).collect();
                debug_assert_dense(points_to, fid, *result, cells.len(), fti);
                plan.mark_split(*result, cells);
            }

            // A constant-count repeat peels into `N` cells that all alias the one source element
            // (no fresh ids, no emitted op). Only scalar-element fixed-size repeats are classified
            // `Split` (see `array_cells`), so `single` and `array_size` are always satisfied here.
            OpCode::MkRepeated {
                result, element, ..
            } if is_split(*result) => {
                let n = array_size(fti.get_value_type(*result));
                let cell = single(&plan.value_map, *element);
                let cells = vec![cell; n];
                debug_assert_dense(points_to, fid, *result, cells.len(), fti);
                plan.mark_split(*result, cells);
            }

            // `ArrayGet` at a constant index aliases the one cell; emits no instruction.
            OpCode::ArrayGet {
                result,
                array,
                index,
            } if is_split(*array) => {
                let k = const_index(ssa, *index)
                    .expect("ICE: a Split array must be indexed by a constant");
                let cell = components(&plan.value_map, *array)[k];
                // The result is a single value (a scalar, a ref, or a Collapsed inner array)
                // aliased to that cell — recorded in `value_map` but NOT as a split array.
                plan.value_map.insert(*result, vec![cell]);
            }

            // `ArraySet` copies the source cells and overlays one; emits no instruction.
            OpCode::ArraySet {
                result,
                array,
                index,
                value,
            } if is_split(*array) => {
                let k = const_index(ssa, *index)
                    .expect("ICE: a Split array must be set at a constant index");
                let mut cells = components(&plan.value_map, *array);
                cells[k] = single(&plan.value_map, *value);
                plan.mark_split(*result, cells);
            }

            // Union-copy ops over a split array result: one fresh cell id per index, with a
            // per-cell op emitted in the rewrite phase.
            OpCode::Select { result, .. } | OpCode::Cast { result, .. } if is_split(*result) => {
                let n = array_size(fti.get_value_type(*result));
                let cells: Vec<ValueId> = (0..n).map(|_| ssa.fresh_value()).collect();
                plan.mark_split(*result, cells);
            }

            // Guards do not appear pre-WTI; delegate defensively to keep lock-step with the
            // analysis walker (which also delegates).
            OpCode::Guard { inner, .. } => {
                Self::plan_instruction(ssa, fid, fti, points_to, inner, plan)
            }
            _ => {}
        }
    }

    /// Rewrite one function's reachable blocks in place: split params, instructions, terminators.
    fn rewrite_function(ssa: &mut HLSSA, fid: FunctionId, reachable: &[BlockId], plan: &Plan) {
        let func = ssa.get_function_mut(fid);
        for bid in reachable {
            let block = func.get_block_mut(*bid);

            let old_params = block.take_parameters();
            let mut new_params = Vec::with_capacity(old_params.len());
            for (pid, ty) in old_params {
                if plan.split.contains(&pid) {
                    // A peeled split-array param expands into one param per cell, each of the
                    // (witness-aware) element type.
                    let elem = ty.get_array_element();
                    for cell in &plan.value_map[&pid] {
                        new_params.push((*cell, elem.clone()));
                    }
                } else {
                    new_params.push((pid, ty));
                }
            }
            block.put_parameters(new_params);

            let old_instructions = block.take_instructions();
            let mut new_instructions = Vec::with_capacity(old_instructions.len());
            for instr in old_instructions {
                let location = instr.location().clone();
                let mut lowered = Vec::new();
                lower_instruction(&instr, plan, &mut lowered);
                new_instructions.extend(
                    lowered
                        .into_iter()
                        .map(|instruction| instruction.locate(location.clone())),
                );
            }
            block.put_instructions(new_instructions);

            let new_terminator = match block.take_terminator().unwrap() {
                // A split-array `Jmp` arg expands into its `N` cell components, aligned by index
                // with the destination block's expanded params (same flow-group ⇒ same dense cell
                // order).
                Terminator::Jmp(dest, args) => {
                    Terminator::Jmp(dest, flat_components(&plan.value_map, &args))
                }
                Terminator::JmpIf(cond, t, f) => {
                    Terminator::JmpIf(single(&plan.value_map, cond), t, f)
                }
                // `Return` is a boundary, so its values are never split arrays; map each through
                // `single` (preserving arity, since the signature is untouched).
                Terminator::Return(values) => {
                    Terminator::Return(values.iter().map(|v| single(&plan.value_map, *v)).collect())
                }
            };
            block.set_terminator(new_terminator);
        }
    }
}

// PLAN
// ================================================================================================

/// The per-function peeling plan.
#[derive(Default)]
struct Plan {
    /// Original value → its component values. A `Split` array maps to its `N` cells (index `k` at
    /// position `k`); an `ArrayGet` result aliases its single cell. Values absent from the map are
    /// their own single component.
    value_map: HashMap<ValueId, Vec<ValueId>>,

    /// The subset of `value_map` keys that are *peeled `Split` arrays* (as opposed to single-value
    /// `ArrayGet` cell aliases). Drives the rewrite decision (drop / emit-per-cell / keep).
    split: HashSet<ValueId>,
}

impl Plan {
    /// Record `v` as a peeled split array with the given dense per-cell components.
    fn mark_split(&mut self, v: ValueId, cells: Vec<ValueId>) {
        self.value_map.insert(v, cells);
        self.split.insert(v);
    }
}

// INSTRUCTION LOWERING
// ================================================================================================

/// Lower one opcode into zero or more peeled opcodes, appending them to `out`.
fn lower_instruction(op: &OpCode, plan: &Plan, out: &mut Vec<OpCode>) {
    let split = |v: ValueId| plan.split.contains(&v);
    match op {
        // Aliasing producers: dropped entirely (their results are aliased in `value_map`).
        OpCode::MkSeq { result, .. }
        | OpCode::MkSeqOfBlob { result, .. }
        | OpCode::MkRepeated { result, .. }
            if split(*result) => {}
        OpCode::ArrayGet { array, .. } | OpCode::ArraySet { array, .. } if split(*array) => {}

        // Union-copy ops over a split array: one op per cell.
        OpCode::Select {
            result,
            cond,
            if_t,
            if_f,
        } if split(*result) => {
            let c = single(&plan.value_map, *cond);
            let results = &plan.value_map[result];
            let then = components(&plan.value_map, *if_t);
            let otherwise = components(&plan.value_map, *if_f);
            debug_assert_eq!(results.len(), then.len());
            debug_assert_eq!(results.len(), otherwise.len());
            for ((r, t), f) in results.iter().zip(then).zip(otherwise) {
                out.push(OpCode::Select {
                    result: *r,
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
        } if split(*result) => {
            let results = &plan.value_map[result];
            let values = components(&plan.value_map, *value);
            debug_assert_eq!(results.len(), values.len());
            for (r, v) in results.iter().zip(values) {
                out.push(OpCode::Cast {
                    result: *r,
                    value: v,
                    target: target.clone(),
                });
            }
        }

        // Guards do not appear pre-WTI; delegate defensively (one guarded op per produced op).
        OpCode::Guard { condition, inner } => {
            let cond = single(&plan.value_map, *condition);
            let mut inner_out = Vec::new();
            lower_instruction(inner, plan, &mut inner_out);
            for inner_op in inner_out {
                out.push(OpCode::Guard {
                    condition: cond,
                    inner: Box::new(inner_op),
                });
            }
        }

        // Every other instruction is kept verbatim with its operands remapped. No split array ever
        // reaches here (the boundary guarantee), so `single` never sees a multi-cell value — a
        // single-component `ArrayGet` cell alias remaps to its cell value, everything else to
        // itself.
        _ => {
            let mut clone = op.clone();
            for id in clone.get_operands_mut() {
                *id = single(&plan.value_map, *id);
            }
            out.push(clone);
        }
    }
}

// COMPONENT-MAP HELPERS
// ================================================================================================

/// The component values of `value`. Values absent from the map are their own single component.
fn components(value_map: &HashMap<ValueId, Vec<ValueId>>, value: ValueId) -> Vec<ValueId> {
    value_map
        .get(&value)
        .cloned()
        .unwrap_or_else(|| vec![value])
}

/// The single component of a value that must not be a peeled (multi-cell) array — panics otherwise,
/// which is the boundary-guarantee tripwire.
fn single(value_map: &HashMap<ValueId, Vec<ValueId>>, value: ValueId) -> ValueId {
    let comps = components(value_map, value);
    assert_eq!(
        comps.len(),
        1,
        "array_sroa: expected a single-component value, got {} components for v{}",
        comps.len(),
        value.0
    );
    comps[0]
}

/// Flatten a list of values into the concatenation of their components (a split-array arg expands
/// to its cells; everything else stays one value).
fn flat_components(value_map: &HashMap<ValueId, Vec<ValueId>>, values: &[ValueId]) -> Vec<ValueId> {
    values
        .iter()
        .flat_map(|v| components(value_map, *v))
        .collect()
}

// UTILITIES
// ================================================================================================

/// The static size `N` of an array type (witness-peeled).
fn array_size(ty: &Type) -> usize {
    match &ty.peel_witness().expr {
        TypeExpr::Array(_, n) => *n,
        other => panic!("ICE: array_sroa expected an array type, got {:?}", other),
    }
}

/// The constant value of an index, if it resolves to a `Constant::U`.
fn const_index(ssa: &HLSSA, index: ValueId) -> Option<usize> {
    match &*ssa.get_const(index)? {
        Constant::U(_, v) => Some(*v as usize),
        _ => None,
    }
}

/// In debug builds, assert the analysis agrees that `result`'s cell set is exactly the dense
/// `0..count` implied by its array type — the invariant the one-level peel relies on.
fn debug_assert_dense(
    points_to: &PointsTo,
    fid: FunctionId,
    result: ValueId,
    count: usize,
    fti: &FunctionTypeInfo,
) {
    debug_assert_eq!(
        count,
        array_size(fti.get_value_type(result)),
        "array_sroa: cell count disagrees with array type size for v{}",
        result.0
    );
    // The one-level peel relies on a `Split` group's cell set being exactly the dense `0..count`
    // implied by its array type. Phrased as a `debug_assert!` (not a `#[cfg]` block) so `points_to`
    // and `fid` stay referenced in release builds, where the assertion is compiled but not run.
    debug_assert!(
        points_to
            .splittable_cells(fid, result)
            .map_or(true, |cells| cells
                == (0..count).collect::<HashSet<usize>>()),
        "array_sroa: Split cell set for v{} is not dense 0..{}",
        result.0,
        count
    );
}

// TESTS
// ================================================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::{
        analysis::types::Types,
        passes::mem2reg::Mem2Reg,
        ssa::hlssa::{
            Blob, CastTarget, SequenceTargetType,
            builder::{HLEmitter, HLSSABuilder},
        },
    };

    fn fr(n: u64) -> ark_bn254::Fr {
        ark_bn254::Fr::from(n)
    }

    /// Pre-#255 `alloc(Field)`: a `Ref<Field>` cell seeded with an inert default field value.
    /// The constant is interned (never a block instruction), so the seed never shows up in the
    /// opcode `Counts` these tests assert on.
    fn falloc(e: &mut impl HLEmitter) -> ValueId {
        let init = e.field_const(fr(0));
        e.alloc(init)
    }

    fn arr2(elem: Type) -> Type {
        elem.array_of(2)
    }

    /// Run only ArraySroa, building its analyses on the current IR (mirrors the pass-manager
    /// wiring).
    fn run_sroa(ssa: &mut HLSSA) {
        let flow = FlowAnalysis::run(ssa);
        let types = Types::new().run(ssa, &flow);
        let pt = PointsTo::run(ssa, &flow, &types);
        ArraySroa::new().do_run(ssa, &flow, &types, &pt);
    }

    /// Run a fresh Mem2Reg on the current IR (analyses recomputed, as `preserves()=[]` forces).
    fn run_mem2reg(ssa: &mut HLSSA) {
        let flow = FlowAnalysis::run(ssa);
        let types = Types::new().run(ssa, &flow);
        let pt = PointsTo::run(ssa, &flow, &types);
        Mem2Reg::new().do_run(ssa, &flow, &types, &pt);
    }

    /// Counts of the opcodes the transform touches, in one function body.
    #[derive(Default, Debug, PartialEq, Eq)]
    struct Counts {
        mkseq: usize,
        mkseqofblob: usize,
        mkrepeated: usize,
        arrayget: usize,
        arrayset: usize,
        alloc: usize,
        store: usize,
        load: usize,
        select: usize,
    }

    fn counts(ssa: &HLSSA, fid: FunctionId) -> Counts {
        let mut c = Counts::default();
        for (_, block) in ssa.get_function(fid).get_blocks() {
            for inst in block.get_instructions() {
                match inst {
                    OpCode::MkSeq { .. } => c.mkseq += 1,
                    OpCode::MkSeqOfBlob { .. } => c.mkseqofblob += 1,
                    OpCode::MkRepeated { .. } => c.mkrepeated += 1,
                    OpCode::ArrayGet { .. } => c.arrayget += 1,
                    OpCode::ArraySet { .. } => c.arrayset += 1,
                    OpCode::Alloc { .. } => c.alloc += 1,
                    OpCode::Store { .. } => c.store += 1,
                    OpCode::Load { .. } => c.load += 1,
                    OpCode::Select { .. } => c.select += 1,
                    _ => {}
                }
            }
        }
        c
    }

    /// Whether any block parameter of `fid` is array/slice-typed (used to confirm phi splitting).
    fn has_array_param(ssa: &HLSSA, fid: FunctionId) -> bool {
        ssa.get_function(fid).get_blocks().any(|(_, block)| {
            block
                .get_parameters()
                .any(|(_, ty)| ty.peel_witness().is_array_or_slice())
        })
    }

    /// Scalar peel: `[x, y]` read at both constant indices, combined to a scalar — all array ops
    /// go.
    #[test]
    fn scalar_array_peels() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let main_id = ssa.get_unique_entrypoint_id();
        {
            let mut sb = HLSSABuilder::new(&mut ssa);
            sb.modify_function(main_id, |b| {
                b.function.add_return_type(Type::field());
                let entry = b.function.get_entry_id();
                let mut e = b.block(entry);
                let x = e.field_const(fr(7));
                let y = e.field_const(fr(9));
                let arr = e.mk_seq(vec![x, y], SequenceTargetType::Array(2), Type::field());
                let i0 = e.u_const(32, 0);
                let i1 = e.u_const(32, 1);
                let a = e.array_get(arr, i0);
                let bb = e.array_get(arr, i1);
                let s = e.add(a, bb);
                e.terminate_return(vec![s]);
            });
        }
        run_sroa(&mut ssa);
        let c = counts(&ssa, main_id);
        assert_eq!(c.mkseq, 0, "MkSeq peeled away");
        assert_eq!(c.arrayget, 0, "ArrayGets peeled away");
    }

    /// ArraySet peel: a functional update of one cell, then both cells read — no array ops remain.
    #[test]
    fn array_set_peels() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let main_id = ssa.get_unique_entrypoint_id();
        {
            let mut sb = HLSSABuilder::new(&mut ssa);
            sb.modify_function(main_id, |b| {
                b.function.add_return_type(Type::field());
                let entry = b.function.get_entry_id();
                let mut e = b.block(entry);
                let x = e.field_const(fr(1));
                let y = e.field_const(fr(2));
                let z = e.field_const(fr(3));
                let arr = e.mk_seq(vec![x, y], SequenceTargetType::Array(2), Type::field());
                let i0 = e.u_const(32, 0);
                let i1 = e.u_const(32, 1);
                let arr2 = e.array_set(arr, i0, z); // [z, y]
                let a = e.array_get(arr2, i0); // z
                let bb = e.array_get(arr2, i1); // y
                let s = e.add(a, bb);
                e.terminate_return(vec![s]);
            });
        }
        run_sroa(&mut ssa);
        let c = counts(&ssa, main_id);
        assert_eq!(c.mkseq, 0);
        assert_eq!(c.arrayset, 0, "ArraySet peeled away");
        assert_eq!(c.arrayget, 0);
    }

    /// Array-typed phi: two arrays merged through a block parameter, then cell-read. The array
    /// param splits into per-cell params and all array traffic is removed.
    #[test]
    fn array_phi_splits() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let main_id = ssa.get_unique_entrypoint_id();
        {
            let mut sb = HLSSABuilder::new(&mut ssa);
            sb.modify_function(main_id, |b| {
                b.function.add_return_type(Type::field());
                let entry = b.function.get_entry_id();
                let mut e = b.block(entry);
                let cond = e.add_parameter(Type::bool());
                let a = e.field_const(fr(1));
                let bb = e.field_const(fr(2));
                let cc = e.field_const(fr(3));
                let d = e.field_const(fr(4));
                let arr_t = e.mk_seq(vec![a, bb], SequenceTargetType::Array(2), Type::field());
                let arr_f = e.mk_seq(vec![cc, d], SequenceTargetType::Array(2), Type::field());
                let merged = e.build_if_else(
                    cond,
                    vec![arr2(Type::field())],
                    |_| vec![arr_t],
                    |_| vec![arr_f],
                )[0];
                let i0 = e.u_const(32, 0);
                let i1 = e.u_const(32, 1);
                let r0 = e.array_get(merged, i0);
                let r1 = e.array_get(merged, i1);
                let s = e.add(r0, r1);
                e.terminate_return(vec![s]);
            });
        }
        run_sroa(&mut ssa);
        let c = counts(&ssa, main_id);
        assert_eq!(c.mkseq, 0, "both MkSeqs peeled");
        assert_eq!(c.arrayget, 0, "ArrayGets peeled");
        assert!(
            !has_array_param(&ssa, main_id),
            "the array merge param split into per-cell scalar params"
        );
    }

    /// Negative — a single dynamic index collapses the group, so the array is retained untouched.
    #[test]
    fn dynamic_index_is_not_peeled() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let main_id = ssa.get_unique_entrypoint_id();
        {
            let mut sb = HLSSABuilder::new(&mut ssa);
            sb.modify_function(main_id, |b| {
                b.function.add_return_type(Type::field());
                let entry = b.function.get_entry_id();
                let mut e = b.block(entry);
                let n = e.add_parameter(Type::u(32)); // dynamic index
                let x = e.field_const(fr(7));
                let y = e.field_const(fr(9));
                let arr = e.mk_seq(vec![x, y], SequenceTargetType::Array(2), Type::field());
                let got = e.array_get(arr, n);
                e.terminate_return(vec![got]);
            });
        }
        run_sroa(&mut ssa);
        let c = counts(&ssa, main_id);
        assert_eq!(c.mkseq, 1, "dynamic-index array retained");
        assert_eq!(c.arrayget, 1, "dynamic ArrayGet retained");
    }

    /// Negative — an out-of-bounds *constant* index (a legal program whose bounds failure is
    /// deferred to runtime) collapses the group, so the array is retained rather than peeled (which
    /// would index a nonexistent cell and crash the compiler).
    #[test]
    fn out_of_bounds_constant_index_is_not_peeled() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let main_id = ssa.get_unique_entrypoint_id();
        {
            let mut sb = HLSSABuilder::new(&mut ssa);
            sb.modify_function(main_id, |b| {
                let entry = b.function.get_entry_id();
                let mut e = b.block(entry);
                let x = e.field_const(fr(1));
                let y = e.field_const(fr(2));
                let z = e.field_const(fr(3));
                let arr = e.mk_seq(vec![x, y], SequenceTargetType::Array(2), Type::field());
                let oob = e.u_const(32, 5); // index 5 into a length-2 array
                let _ = e.array_set(arr, oob, z);
                e.terminate_return(vec![]);
            });
        }
        run_sroa(&mut ssa);
        let c = counts(&ssa, main_id);
        assert_eq!(c.mkseq, 1, "OOB-indexed array retained");
        assert_eq!(c.arrayset, 1, "OOB ArraySet retained");
    }

    /// Negative — an array that crosses a boundary (passed to a `Call`) collapses and is retained.
    #[test]
    fn boundary_array_is_not_peeled() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let main_id = ssa.get_unique_entrypoint_id();
        {
            let mut sb = HLSSABuilder::new(&mut ssa);
            let sink = sb.ssa().add_function("sink".to_string());
            // sink(a: Array<Field,2>) -> Field: return a[0]
            sb.modify_function(sink, |b| {
                b.function.add_return_type(Type::field());
                let entry = b.function.get_entry_id();
                let mut e = b.block(entry);
                let a = e.add_parameter(arr2(Type::field()));
                let i0 = e.u_const(32, 0);
                let got = e.array_get(a, i0);
                e.terminate_return(vec![got]);
            });
            // main(): arr = [x, y]; return sink(arr)
            sb.modify_function(main_id, |b| {
                b.function.add_return_type(Type::field());
                let entry = b.function.get_entry_id();
                let mut e = b.block(entry);
                let x = e.field_const(fr(7));
                let y = e.field_const(fr(9));
                let arr = e.mk_seq(vec![x, y], SequenceTargetType::Array(2), Type::field());
                let r = e.call(sink, vec![arr], 1)[0];
                e.terminate_return(vec![r]);
            });
        }
        run_sroa(&mut ssa);
        assert_eq!(
            counts(&ssa, main_id).mkseq,
            1,
            "array passed to a Call is a boundary ⇒ collapsed ⇒ retained"
        );
    }

    /// A constant blob array, constant-indexed, peels into per-cell constants.
    #[test]
    fn blob_array_peels_to_constants() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let main_id = ssa.get_unique_entrypoint_id();
        {
            let mut sb = HLSSABuilder::new(&mut ssa);
            sb.modify_function(main_id, |b| {
                b.function.add_return_type(Type::field());
                let entry = b.function.get_entry_id();
                let mut e = b.block(entry);
                let blob = e.emit_constant(Constant::Blob(Blob {
                    elem_type: Type::field(),
                    elements: vec![
                        Constant::Field(fr(1)),
                        Constant::Field(fr(2)),
                        Constant::Field(fr(3)),
                    ],
                }));
                let arr = e.mk_seq_of_blob(Type::field(), blob);
                let i1 = e.u_const(32, 1);
                let got = e.array_get(arr, i1);
                e.terminate_return(vec![got]);
            });
        }
        run_sroa(&mut ssa);
        let c = counts(&ssa, main_id);
        assert_eq!(c.mkseqofblob, 0, "MkSeqOfBlob peeled to per-cell constants");
        assert_eq!(c.arrayget, 0);
    }

    /// Ref-element integration: an `Array<Ref<Field>,2>` of clean locals peels into per-cell refs,
    /// and the follow-up Mem2Reg then promotes the underlying allocs.
    #[test]
    fn ref_array_peels_then_mem2reg_promotes() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let main_id = ssa.get_unique_entrypoint_id();
        {
            let mut sb = HLSSABuilder::new(&mut ssa);
            sb.modify_function(main_id, |b| {
                b.function.add_return_type(Type::field());
                let entry = b.function.get_entry_id();
                let mut e = b.block(entry);
                let ra = falloc(&mut e);
                let rb = falloc(&mut e);
                let c1 = e.field_const(fr(3));
                let c2 = e.field_const(fr(4));
                e.store(ra, c1);
                e.store(rb, c2);
                let arr = e.mk_seq(
                    vec![ra, rb],
                    SequenceTargetType::Array(2),
                    Type::field().ref_of(),
                );
                let i0 = e.u_const(32, 0);
                let i1 = e.u_const(32, 1);
                let r0 = e.array_get(arr, i0); // ra
                let r1 = e.array_get(arr, i1); // rb
                let v0 = e.load(r0);
                let v1 = e.load(r1);
                let s = e.add(v0, v1);
                e.terminate_return(vec![s]);
            });
        }
        // SROA peels the ref-array; the now-exposed per-cell refs are used only as load/store ptrs.
        run_sroa(&mut ssa);
        assert_eq!(counts(&ssa, main_id).mkseq, 0, "ref-array peeled");
        assert_eq!(counts(&ssa, main_id).arrayget, 0);
        // The follow-up Mem2Reg promotes the underlying allocs.
        run_mem2reg(&mut ssa);
        let c = counts(&ssa, main_id);
        assert_eq!(c.alloc, 0, "allocs promoted after peeling");
        assert_eq!(c.store, 0);
        assert_eq!(c.load, 0);
    }

    /// End-to-end headline: a `Ref<Array<Field,2>>` local, constant-indexed. The 1st Mem2Reg
    /// promotes it to an array value, ArraySroa peels that — the result is fully scalarized.
    #[test]
    fn ref_to_array_local_fully_scalarizes() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let main_id = ssa.get_unique_entrypoint_id();
        {
            let mut sb = HLSSABuilder::new(&mut ssa);
            sb.modify_function(main_id, |b| {
                b.function.add_return_type(Type::field());
                let entry = b.function.get_entry_id();
                let mut e = b.block(entry);
                let x = e.field_const(fr(5));
                let y = e.field_const(fr(6));
                let arr = e.mk_seq(vec![x, y], SequenceTargetType::Array(2), Type::field());
                let p = e.alloc(arr); // Ref<Array<Field,2>>, seeded with arr (store folded into the alloc)
                let loaded = e.load(p); // Array<Field,2>
                let i0 = e.u_const(32, 0);
                let got = e.array_get(loaded, i0);
                e.terminate_return(vec![got]);
            });
        }
        // 1st Mem2Reg promotes the Ref<Array> local to the array value; ArraySroa then peels it.
        run_mem2reg(&mut ssa);
        run_sroa(&mut ssa);
        let c = counts(&ssa, main_id);
        assert_eq!(c.alloc, 0, "Ref<Array> local promoted");
        assert_eq!(c.store, 0);
        assert_eq!(c.load, 0);
        assert_eq!(c.mkseq, 0, "array value peeled");
        assert_eq!(c.arrayget, 0);
    }

    /// O1: a constant-count repeat of a *scalar* (`[0; 4]`), functionally updated at one cell and
    /// read at constant indices, peels fully — no `MkRepeated`/`ArraySet`/`ArrayGet` remains.
    #[test]
    fn repeated_scalar_array_peels() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let main_id = ssa.get_unique_entrypoint_id();
        {
            let mut sb = HLSSABuilder::new(&mut ssa);
            sb.modify_function(main_id, |b| {
                b.function.add_return_type(Type::field());
                let entry = b.function.get_entry_id();
                let mut e = b.block(entry);
                let zero = e.field_const(fr(0));
                let arr = e.mk_repeated(zero, SequenceTargetType::Array(4), 4, Type::field()); // [0;4]
                let x = e.field_const(fr(7));
                let i1 = e.u_const(32, 1);
                let updated = e.array_set(arr, i1, x); // [0, 7, 0, 0]
                let i0 = e.u_const(32, 0);
                let a = e.array_get(updated, i0); // 0
                let bb = e.array_get(updated, i1); // 7
                let s = e.add(a, bb);
                e.terminate_return(vec![s]);
            });
        }
        run_sroa(&mut ssa);
        let c = counts(&ssa, main_id);
        assert_eq!(c.mkrepeated, 0, "MkRepeated peeled away");
        assert_eq!(c.arrayset, 0, "ArraySet peeled away");
        assert_eq!(c.arrayget, 0, "ArrayGets peeled away");
    }

    /// O1 negative: a repeat of a *ref* element stays collapsed (all cells alias one object, so
    /// peeling is pointless) — the `MkRepeated` and its `ArrayGet` are retained.
    #[test]
    fn repeated_ref_array_is_not_peeled() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let main_id = ssa.get_unique_entrypoint_id();
        {
            let mut sb = HLSSABuilder::new(&mut ssa);
            sb.modify_function(main_id, |b| {
                b.function.add_return_type(Type::field());
                let entry = b.function.get_entry_id();
                let mut e = b.block(entry);
                let r = falloc(&mut e);
                let c = e.field_const(fr(5));
                e.store(r, c);
                let arr = e.mk_repeated(r, SequenceTargetType::Array(3), 3, Type::field().ref_of()); // [r;3]
                let i0 = e.u_const(32, 0);
                let got = e.array_get(arr, i0); // r
                let v = e.load(got);
                e.terminate_return(vec![v]);
            });
        }
        run_sroa(&mut ssa);
        let c = counts(&ssa, main_id);
        assert_eq!(
            c.mkrepeated, 1,
            "ref repeat retained (peeling refs is pointless)"
        );
        assert_eq!(c.arrayget, 1, "ref repeat's ArrayGet retained");
    }

    /// C1: a locally-built *slice* threaded through an if/else phi is never `Split` (a slice can't be
    /// peeled — `array_size` would panic on it), so both slice constructors are retained and the
    /// pass does not crash.
    #[test]
    fn slice_phi_is_not_peeled() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let main_id = ssa.get_unique_entrypoint_id();
        {
            let mut sb = HLSSABuilder::new(&mut ssa);
            sb.modify_function(main_id, |b| {
                let entry = b.function.get_entry_id();
                let mut e = b.block(entry);
                let cond = e.add_parameter(Type::bool());
                let a = e.field_const(fr(1));
                let bb = e.field_const(fr(2));
                let slice_t = e.mk_seq(vec![a], SequenceTargetType::Slice, Type::field());
                let slice_f = e.mk_seq(vec![bb], SequenceTargetType::Slice, Type::field());
                let _merged = e.build_if_else(
                    cond,
                    vec![Type::field().slice_of()],
                    |_| vec![slice_t],
                    |_| vec![slice_f],
                )[0];
                e.terminate_return(vec![]);
            });
        }
        run_sroa(&mut ssa); // must not panic
        let c = counts(&ssa, main_id);
        assert_eq!(
            c.mkseq, 2,
            "slice constructors retained — slices are never Split"
        );
    }

    /// C1: an Array→Slice cast pulls a slice into the array's flow-group; the whole group collapses
    /// so the array is retained (peeling it would leave the kept `Cast` consuming a deleted
    /// aggregate / index a slice). The pass must not crash.
    #[test]
    fn array_to_slice_cast_is_not_peeled() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let main_id = ssa.get_unique_entrypoint_id();
        {
            let mut sb = HLSSABuilder::new(&mut ssa);
            sb.modify_function(main_id, |b| {
                b.function.add_return_type(Type::field());
                let entry = b.function.get_entry_id();
                let mut e = b.block(entry);
                let x = e.field_const(fr(7));
                let y = e.field_const(fr(9));
                let arr = e.mk_seq(vec![x, y], SequenceTargetType::Array(2), Type::field());
                let _slice = e.cast_to(CastTarget::ArrayToSlice, arr); // unioned into arr's group
                let i0 = e.u_const(32, 0);
                let got = e.array_get(arr, i0);
                e.terminate_return(vec![got]);
            });
        }
        run_sroa(&mut ssa); // must not panic
        let c = counts(&ssa, main_id);
        assert_eq!(
            c.mkseq, 1,
            "array retained — Array→Slice cast collapses the group"
        );
        assert_eq!(c.arrayget, 1);
    }
}
