//! The array flow-group pre-pass that provides per-`Index(k)` cell precision.
//!
//! Arrays are value-semantic in HLSSA, so `ArraySet`, block-parameter phis, `Select`, and casts are
//! *wholesale copies* of an array value: the result inherits all of the source's element cells. For
//! per-constant-index cell precision in the (type-oblivious) solver, a wholesale copy must
//! reproduce every `Index(j)` cell — an enumeration set that is a *global* property of the array's
//! flow-group, not knowable at one instruction. This module computes that set.
//!
//! It partitions a function's array/slice `ValueId`s into flow-groups (a union-find over wholesale-
//! copy edges) and classifies each group:
//!
//! - **`Split`** — the group is *purely local*, *only ever constant-indexed*, and never hits a
//!   collapse trigger. Its live constant indices are tracked as distinct `Index(k)` cells.
//! - **`Collapsed`** — some collapse trigger fired (a dynamic index, `MkRepeated`, a slice, an
//!   array stored/loaded through a `Ref`, or any boundary crossing: a parameter, a `Call`
//!   arg/result, a `Return`/`InitGlobal`/`ReadGlobal`). The group keeps the single `Elem(AllElems)`
//!   cell — today's sound behavior.
//!
//! Because every boundary-crossing or through-`Ref` array is `Collapsed`, the builder's `AllElems`
//! paths (parameter seeding, summary instantiation, `Store`/`Load`) never touch a `Split` group, so
//! there is no seeding/reading mismatch — see [`super::builder`].
//!
//! Construction is **two-pass**. Pass 1 records every union edge and every observation
//! `(value, Index(k) | Collapse)` while pass 2 folds each observation onto its value's *final*
//! union-find representative. Folding after all unions avoids the observe-before-union hazard (a
//! value can change representative as later unions accrue).

use crate::{
    collections::{HashMap, HashSet, UnionFind},
    compiler::{
        analysis::types::FunctionTypeInfo,
        ssa::{
            Terminator, ValueId,
            hlssa::{Constant, HLFunction, HLSSA, OpCode},
        },
    },
};

// PUBLIC RESULT
// ================================================================================================

/// The array-cell flow-group classification for one function.
#[derive(Debug, Clone)]
pub struct ArrayCells {
    /// Every array/slice value mapped to its flow-group representative.
    rep: HashMap<ValueId, ValueId>,

    /// Per-representative facts.
    facts: HashMap<ValueId, GroupFacts>,
}

#[derive(Debug, Clone, Default)]
struct GroupFacts {
    /// Constant indices observed on the group (array literal positions + constant get/set indices).
    indices: HashSet<usize>,

    /// Whether any collapse trigger fired for the group.
    collapsed: bool,
}

impl ArrayCells {
    /// The live constant indices of `v`'s flow-group if it is `Split` (cell-precise), or `None` if
    /// it is `Collapsed` (use `Elem(AllElems)`) or `v` is not a tracked array value.
    pub fn split_indices(&self, v: ValueId) -> Option<&HashSet<usize>> {
        let rep = self.rep.get(&v)?;
        let facts = self.facts.get(rep)?;
        if facts.collapsed {
            None
        } else {
            Some(&facts.indices)
        }
    }

    /// Compute the classification for one function.
    pub fn classify(ssa: &HLSSA, func: &HLFunction, types: &FunctionTypeInfo) -> Self {
        let is_arr = |v: ValueId| types.get_value_type(v).peel_witness().is_array_or_slice();

        let mut uf = UnionFind::default();
        let mut obs: Vec<(ValueId, Obs)> = Vec::new();

        // A parameter array is caller memory (a boundary): collapse.
        for (value, ty) in func.get_entry().get_parameters() {
            if ty.peel_witness().is_array_or_slice() {
                uf.add(*value);
                obs.push((*value, Obs::Collapse));
            }
        }

        for (_, block) in func.get_blocks() {
            for instr in block.get_instructions() {
                process_instr(instr, ssa, &is_arr, &mut uf, &mut obs);
            }
            match block.get_terminator() {
                Some(Terminator::Jmp(target, jump_args)) => {
                    let params = func.get_block(*target).get_parameters();
                    for ((param, param_ty), arg) in params.zip(jump_args.iter()) {
                        if param_ty.peel_witness().is_array_or_slice() {
                            uf.union(*param, *arg);
                        }
                    }
                }
                Some(Terminator::Return(values)) => {
                    for v in values {
                        if is_arr(*v) {
                            uf.add(*v);
                            obs.push((*v, Obs::Collapse));
                        }
                    }
                }
                Some(Terminator::JmpIf(..)) | None => {}
            }
        }

        // Pass 2: fold each observation onto its value's final representative.
        let mut facts: HashMap<ValueId, GroupFacts> = HashMap::default();
        for (v, ob) in obs {
            let rep = uf.find(v);
            let f = facts.entry(rep).or_default();
            match ob {
                Obs::Index(k) => {
                    f.indices.insert(k);
                }
                Obs::Collapse => f.collapsed = true,
            }
        }

        let node_list: Vec<ValueId> = uf.nodes().collect();
        let rep: HashMap<ValueId, ValueId> =
            node_list.into_iter().map(|v| (v, uf.find(v))).collect();

        ArrayCells { rep, facts }
    }
}

// CONSTRUCTION
// ================================================================================================

/// One recorded observation about an array value's flow-group.
enum Obs {
    /// A constant index `k` is accessed on the group.
    Index(usize),

    /// A collapse trigger fired.
    Collapse,
}

/// Process one instruction: union wholesale-copy edges and record observations.
///
/// A `Guard` delegates to its inner op, mirroring [`super::builder`]'s walker, so the pre-pass and
/// the builder stay in lock-step (a guard wrapping a dynamic `ArrayGet` must collapse the group
/// here, or the builder would later hit its const-index `expect`). Guards do not appear pre-WTI, so
/// the delegation is defence-in-depth. Any array value reached only by ops with no arm here is
/// never entered into the union-find, so [`ArrayCells::split_indices`] returns `None` for it — i.e.
/// it is treated as `Collapsed`, the sound default.
fn process_instr(
    instr: &OpCode,
    ssa: &HLSSA,
    is_arr: &impl Fn(ValueId) -> bool,
    uf: &mut UnionFind<ValueId>,
    obs: &mut Vec<(ValueId, Obs)>,
) {
    match instr {
        // Array construction.
        OpCode::MkSeq { result, elems, .. } => {
            uf.add(*result);
            for i in 0..elems.len() {
                obs.push((*result, Obs::Index(i)));
            }
        }
        OpCode::MkRepeated { result, .. } => {
            // A constant-count repeat has statically-separable cells (each index is the one source
            // element), so it *could* be `Split` over `0..count`. Left collapsed deliberately: ref
            // elements all alias one object (peeling is pointless) and scalar elements would merely
            // duplicate one value N times — low ROI for the SROA client.
            uf.add(*result);
            obs.push((*result, Obs::Collapse));
        }
        OpCode::MkSeqOfBlob { result, blob, .. } => {
            // A constant blob array of scalars (no refs). Record each position as a constant cell
            // so a purely-constant-indexed blob array is `splittable_cells`-eligible; collapse only
            // if the blob length is not statically resolvable.
            uf.add(*result);
            match blob_len(ssa, *blob) {
                Some(len) => {
                    for i in 0..len {
                        obs.push((*result, Obs::Index(i)));
                    }
                }
                None => obs.push((*result, Obs::Collapse)),
            }
        }

        // Element access — records the index on the (source) array; a dynamic index collapses.
        OpCode::ArrayGet {
            result,
            array,
            index,
        } => {
            record_index(ssa, obs, *array, *index);
            // A nested inner array extracted from a cell is tracked separately and collapsed.
            if is_arr(*result) {
                uf.add(*result);
                obs.push((*result, Obs::Collapse));
            }
        }
        OpCode::ArraySet {
            result,
            array,
            value,
            index,
        } => {
            uf.union(*result, *array); // result inherits array's unchanged cells
            record_index(ssa, obs, *array, *index);
            // A stored inner array (array-of-arrays) flows into a cell; track it separately,
            // collapsed.
            if is_arr(*value) {
                uf.add(*value);
                obs.push((*value, Obs::Collapse));
            }
        }

        // Wholesale value copies of array-typed values.
        OpCode::Select {
            result, if_t, if_f, ..
        } if is_arr(*result) => {
            uf.union(*result, *if_t);
            uf.union(*result, *if_f);
        }
        OpCode::Cast { result, value, .. }
        | OpCode::Not { result, value }
        | OpCode::SExt { result, value, .. }
        | OpCode::BitRange { result, value, .. }
        | OpCode::Spread { result, value, .. }
            if is_arr(*result) =>
        {
            uf.union(*result, *value);
        }

        // Through-`Ref` and slice flow: collapse.
        OpCode::Store { value, .. } if is_arr(*value) => {
            uf.add(*value);
            obs.push((*value, Obs::Collapse));
        }
        OpCode::Load { result, .. } if is_arr(*result) => {
            uf.add(*result);
            obs.push((*result, Obs::Collapse));
        }
        OpCode::SlicePush {
            result,
            slice,
            values,
            ..
        } => {
            uf.add(*result);
            obs.push((*result, Obs::Collapse));
            uf.add(*slice);
            obs.push((*slice, Obs::Collapse));
            for v in values {
                if is_arr(*v) {
                    uf.add(*v);
                    obs.push((*v, Obs::Collapse));
                }
            }
        }

        // Boundary crossings: collapse every array arg/result/value.
        OpCode::Call { results, args, .. } => {
            for v in args.iter().chain(results.iter()) {
                if is_arr(*v) {
                    uf.add(*v);
                    obs.push((*v, Obs::Collapse));
                }
            }
        }
        OpCode::InitGlobal { value, .. } if is_arr(*value) => {
            uf.add(*value);
            obs.push((*value, Obs::Collapse));
        }
        OpCode::ReadGlobal { result, .. } if is_arr(*result) => {
            uf.add(*result);
            obs.push((*result, Obs::Collapse));
        }

        // Delegate guards to their inner op (mirrors `builder::FnBuilder::build_instr`); guards do
        // not appear pre-WTI, so this is defence-in-depth keeping the two walkers in lock-step.
        OpCode::Guard { inner, .. } => process_instr(inner, ssa, is_arr, uf, obs),

        _ => {}
    }
}

/// Record a constant index as an `Index(k)` observation, or collapse the group on a dynamic index.
fn record_index(ssa: &HLSSA, obs: &mut Vec<(ValueId, Obs)>, array: ValueId, index: ValueId) {
    match const_index(ssa, index) {
        Some(k) => obs.push((array, Obs::Index(k))),
        None => obs.push((array, Obs::Collapse)),
    }
}

fn const_index(ssa: &HLSSA, index: ValueId) -> Option<usize> {
    match &*ssa.get_const(index)? {
        Constant::U(_, v) => Some(*v as usize),
        _ => None,
    }
}

/// The length of a constant blob, if `blob` resolves to a `Constant::Blob`.
fn blob_len(ssa: &HLSSA, blob: ValueId) -> Option<usize> {
    match &*ssa.get_const(blob)? {
        Constant::Blob(b) => Some(b.elements.len()),
        _ => None,
    }
}
