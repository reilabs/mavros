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
//! - **`Split`** — the group is *purely local*, *only ever constant-indexed*, a *fixed-size array*
//!   (never a slice), and never hits a collapse trigger. Its live constant indices are tracked as
//!   distinct `Index(k)` cells. A constant-count `MkRepeated` of a scalar element is `Split` too (its
//!   cells all alias the one source value until an `ArraySet` diverges them).
//! - **`Collapsed`** — some collapse trigger fired (a dynamic or out-of-bounds index, a slice or any
//!   non-fixed-array result, a `MkRepeated` of a ref/array element, an array stored/loaded through a
//!   `Ref`, or any boundary crossing: a parameter, a `Call` arg/result, a
//!   `Return`/`InitGlobal`/`ReadGlobal`). The group keeps the single `Elem(AllElems)` cell — today's
//!   sound behavior.
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
            Instruction, Terminator, ValueId,
            hlssa::{Constant, HLFunction, HLSSA, OpCode, TypeExpr},
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

        // The static element count of a fixed-size array value, or `None` for slices / non-arrays.
        //
        // Used to collapse a group accessed at an out-of-bounds *constant* index (a legal program
        // whose bounds failure is deferred to witgen/runtime) so the splitter never peels it.
        let arr_size = |v: ValueId| -> Option<usize> {
            match &types.get_value_type(v).peel_witness().expr {
                TypeExpr::Array(_, n) => Some(*n),
                _ => None,
            }
        };

        // Whether `v` is a scalar arithmetic leaf (the only element kind worth peeling a
        // `MkRepeated` into — a ref repeat aliases one object and an array repeat is a deeper level).
        let is_scalar = |v: ValueId| -> bool {
            matches!(
                types.get_value_type(v).peel_witness().expr,
                TypeExpr::Field | TypeExpr::U(_) | TypeExpr::I(_)
            )
        };

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
                process_instr(
                    instr, ssa, &is_arr, &arr_size, &is_scalar, &mut uf, &mut obs,
                );
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
    arr_size: &impl Fn(ValueId) -> Option<usize>,
    is_scalar: &impl Fn(ValueId) -> bool,
    uf: &mut UnionFind<ValueId>,
    obs: &mut Vec<(ValueId, Obs)>,
) {
    match instr {
        // Array construction.
        OpCode::MkSeq { result, elems, .. } => {
            uf.add(*result);
            // Only a *fixed-size array* is peelable. A `MkSeq` building a slice
            // (`SequenceTargetType::Slice`) must collapse: the SROA client's `array_size` panics on
            // a slice type, so a slice must never be classified `Split`.
            if arr_size(*result).is_some() {
                for i in 0..elems.len() {
                    obs.push((*result, Obs::Index(i)));
                }
            } else {
                obs.push((*result, Obs::Collapse));
            }

            // An array-typed element flows into a cell of `result` (array-of-arrays): it is a
            // *deeper* array level, which never gets per-cell precision.
            //
            // Track it separately and collapse it, mirroring the `ArraySet` stored-inner-array
            // handling below. This keeps the disjoint-roles invariant honest for `MkSeq` too:
            // without it an inner array reachable only as a `MkSeq` element would stay `Split`,
            // letting the SROA client peel an array that still appears as an aggregate operand of
            // `result`'s (kept or peeled) constructor.
            for elem in elems {
                if is_arr(*elem) {
                    uf.add(*elem);
                    obs.push((*elem, Obs::Collapse));
                }
            }
        }
        OpCode::MkRepeated {
            result, element, ..
        } => {
            uf.add(*result);

            // A constant-count repeat of a *scalar* element into a fixed-size array is peelable: the
            // N cells all alias the one source value (zero duplication in SSA — they share its
            // ValueId) and diverge only when a later `ArraySet` overlays one. Record `0..n` so the
            // group can be `Split`. Collapse otherwise:
            // - a ref element: all cells alias one object, so peeling is pointless;
            // - an array element: a deeper array level (collapsed below), so the cell holds an
            //   aggregate and must not be peeled;
            // - a slice result (`arr_size` is `None`): not peelable at all.
            match arr_size(*result) {
                Some(n) if is_scalar(*element) => {
                    for i in 0..n {
                        obs.push((*result, Obs::Index(i)));
                    }
                }
                _ => obs.push((*result, Obs::Collapse)),
            }

            // An array-typed repeated element is a deeper array level — collapse it.
            if is_arr(*element) {
                uf.add(*element);
                obs.push((*element, Obs::Collapse));
            }
        }
        OpCode::MkSeqOfBlob { result, blob, .. } => {
            // A constant blob array of scalars (no refs). Record each position as a constant cell
            // so a purely-constant-indexed blob array is `splittable_cells`-eligible; collapse only
            // if the blob length is not statically resolvable.
            uf.add(*result);
            match blob_len(ssa, *blob) {
                // Peelable only as a fixed-size array (guard the result type, like `MkSeq`).
                Some(len) if arr_size(*result).is_some() => {
                    for i in 0..len {
                        obs.push((*result, Obs::Index(i)));
                    }
                }
                _ => obs.push((*result, Obs::Collapse)),
            }
        }

        // Element access — records the index on the (source) array; a dynamic index collapses.
        OpCode::ArrayGet {
            result,
            array,
            index,
        } => {
            record_index(ssa, arr_size, obs, *array, *index);
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
            record_index(ssa, arr_size, obs, *array, *index);
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
            // A slice `Select` is not peelable (branches share the result's type); collapse so a
            // slice never lands in a `Split` group.
            if arr_size(*result).is_none() {
                obs.push((*result, Obs::Collapse));
            }
        }
        OpCode::Cast { result, value, .. } if is_arr(*result) => {
            uf.union(*result, *value);
            // An Array→Slice (or the unlikely Slice→Array) cast pulls a non-peelable member into the
            // group; collapse the *whole* unioned group. Collapsing only the slice side would leave
            // the array peelable while the `Cast` consuming it is kept, tripping `single()` in the
            // client's fallback arm.
            if arr_size(*result).is_none() || arr_size(*value).is_none() {
                obs.push((*result, Obs::Collapse));
            }
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
        OpCode::Guard { inner, .. } => {
            process_instr(inner, ssa, is_arr, arr_size, is_scalar, uf, obs)
        }

        // Any opcode without a cell-aware arm above is opaque to the splitter: collapse every array
        // it touches (operand or result).
        //
        // This keeps the `Split` classification in lock-step with what the SROA client can actually
        // peel — producers (`MkSeq`/`MkSeqOfBlob`/`ArraySet`), constant accesses (`ArrayGet`), the
        // cell-preserving copies (`Select`/`Cast`), and phis. An array reaching anything else (e.g.
        // `SliceLen`, `ToBits`/`ToRadix`, `Not`/`SExt`/`BitRange`/`Spread`) must not stay `Split`,
        // or the peel would leave that consumer holding a deleted aggregate. Sound: collapsing only
        // ever blocks a peel.
        other => {
            for v in other.get_inputs().chain(other.get_results()) {
                if is_arr(*v) {
                    uf.add(*v);
                    obs.push((*v, Obs::Collapse));
                }
            }
        }
    }
}

/// Record a constant in-bounds index as an `Index(k)` observation, else collapse the group.
///
/// Collapse triggers: a dynamic index, an unresolvable array size, or an out-of-bounds *constant*
/// index. The last is a legal program (e.g. `let a = [1,2,3]; a[10] = x;`) whose bounds failure is
/// deferred to witgen/runtime — the splitter must not peel it (cell `k` does not exist), so the
/// whole group collapses and the access is left for normal handling.
fn record_index(
    ssa: &HLSSA,
    arr_size: &impl Fn(ValueId) -> Option<usize>,
    obs: &mut Vec<(ValueId, Obs)>,
    array: ValueId,
    index: ValueId,
) {
    match const_index(ssa, index) {
        Some(k) if arr_size(array).map_or(false, |n| k < n) => obs.push((array, Obs::Index(k))),
        _ => obs.push((array, Obs::Collapse)),
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
