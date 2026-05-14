//! Flattens multi-dimensional arrays into one dimension.
//!
//! - `Array<Array<T,N>,M>` becomes `Array<T, M*N>` (recursively for deeper nests).
//! - `Slice<Array<T,N>>` becomes `Slice<T>`.
//! - `Slice<Slice<_>>` panics — not supported by Noir.
//!
//! Indexing into a sub-region of a flat array (which used to produce an inner
//! array value) becomes a `SliceArray` view. Whole-row writes become `BlockSet`.
//! Existing `ArrayGet`/`ArraySet`/`SliceLen` continue to work; at the VM level
//! they accept either an owned array or a view.

use std::collections::HashMap;

use crate::compiler::{
    analysis::types::TypeInfo,
    block_builder::{HLBlockEmitter, HLEmitter},
    ir::r#type::{Type, TypeExpr},
    pass_manager::{Analysis, AnalysisId, AnalysisStore, Pass},
    ssa::{ConstValue, FunctionId, HLSSA, Instruction, OpCode, SeqType, ValueId},
};

pub struct FlattenArrays {}

impl FlattenArrays {
    pub fn new() -> Self {
        Self {}
    }
}

impl Pass for FlattenArrays {
    fn name(&self) -> &'static str {
        "flatten_arrays"
    }

    fn needs(&self) -> Vec<AnalysisId> {
        vec![TypeInfo::id()]
    }

    fn run(&self, ssa: &mut HLSSA, store: &AnalysisStore) {
        let type_info = store.get::<TypeInfo>();
        run_flatten(ssa, type_info);
    }
}

// ---------------------------------------------------------------------------
// Type flattening
// ---------------------------------------------------------------------------

/// Walk `t` and collapse nested array/slice shapes:
/// - `Array<Array<T,N>,M>`  → `Array<T_flat, M * inner_flat_length>`
/// - `Slice<Array<T,N>>`    → `Slice<T_flat>`
/// - `Slice<Slice<_>>`      → panic
pub fn flatten_type(t: &Type) -> Type {
    match &t.expr {
        TypeExpr::Array(inner, n) => {
            let inner_flat = flatten_type(inner);
            match &inner_flat.expr {
                TypeExpr::Array(deeper, m) => (**deeper).clone().array_of(n * m),
                TypeExpr::Slice(_) => {
                    panic!("flatten_arrays: Array of Slice is not supported: {}", t)
                }
                _ => inner_flat.array_of(*n),
            }
        }
        TypeExpr::Slice(inner) => {
            let inner_flat = flatten_type(inner);
            match &inner_flat.expr {
                TypeExpr::Array(deeper, _m) => (**deeper).clone().slice_of(),
                TypeExpr::Slice(_) => {
                    panic!("flatten_arrays: Slice of Slice is not supported by Noir: {}", t)
                }
                _ => inner_flat.slice_of(),
            }
        }
        TypeExpr::Tuple(elems) => Type::tuple_of(elems.iter().map(flatten_type).collect()),
        TypeExpr::Ref(inner) => flatten_type(inner).ref_of(),
        TypeExpr::WitnessOf(inner) => Type::witness_of(flatten_type(inner)),
        TypeExpr::Field | TypeExpr::U(_) | TypeExpr::I(_) | TypeExpr::Function => t.clone(),
    }
}

/// Is `t` (in its *original*, pre-flatten form) an array/slice element whose
/// flattened form would be array-shaped — i.e. one of the cases that a
/// preceding ArrayGet/ArraySet should be rewritten through SliceArray/BlockSet?
fn original_elem_is_aggregate(t: &Type) -> bool {
    let stripped = match &t.expr {
        TypeExpr::WitnessOf(inner) => inner.as_ref(),
        _ => t,
    };
    matches!(stripped.expr, TypeExpr::Array(_, _) | TypeExpr::Slice(_))
}

// ---------------------------------------------------------------------------
// Pass driver
// ---------------------------------------------------------------------------

fn run_flatten(ssa: &mut HLSSA, type_info: &TypeInfo) {
    // Flatten global types.
    let new_globals: Vec<Type> = ssa.get_global_types().iter().map(flatten_type).collect();
    ssa.set_global_types(new_globals);

    let function_ids: Vec<FunctionId> = ssa.get_function_ids().collect();
    for fid in function_ids {
        flatten_function(ssa, fid, type_info);
    }
}

fn flatten_function(ssa: &mut HLSSA, fid: FunctionId, type_info: &TypeInfo) {
    let fn_type_info = type_info.get_function(fid);

    // Snapshot ORIGINAL types per value before mutating anything. We use
    // these to decide whether an op needs rewriting.
    let orig_value_types: HashMap<ValueId, Type> = {
        // Walk every value the type_info knows about. The simplest way is to
        // look at block params and instruction results, but the TypeInfo
        // exposes a per-value `get_value_type`. Since FunctionTypeInfo only
        // has `get_value_type`, gather values from the function.
        let func = ssa.get_function(fid);
        let mut m = HashMap::new();
        for (_, block) in func.get_blocks() {
            for (vid, _) in block.get_parameters() {
                m.insert(*vid, fn_type_info.get_value_type(*vid).clone());
            }
            for instr in block.get_instructions() {
                for r in instr.get_results() {
                    m.insert(*r, fn_type_info.get_value_type(*r).clone());
                }
            }
        }
        m
    };

    let func = ssa.get_function_mut(fid);

    // Rewrite return types.
    for rtp in func.iter_returns_mut() {
        *rtp = flatten_type(rtp);
    }

    // Rewrite block parameter types.
    for (_, block) in func.get_blocks_mut() {
        for (_, tp) in block.get_parameters_mut() {
            *tp = flatten_type(tp);
        }
    }

    // Per-block instruction rewriting.
    let block_ids: Vec<_> = func.get_blocks().map(|(b, _)| *b).collect();
    for bid in block_ids {
        // Take instructions out, walk them, emit replacements via the block
        // emitter so we can fabricate fresh ValueIds and intermediate ops.
        let old_instructions = func.get_block_mut(bid).take_instructions();

        let mut emitter = HLBlockEmitter::new(func, bid);

        // Track definitions of (post-rewrite) MkSeq results so we can splice
        // when assembling outer MkSeqs.
        let mut mkseq_defs: HashMap<ValueId, (Vec<ValueId>, SeqType, Type)> = HashMap::new();
        // Cache of u32 constants emitted in this block, keyed by value.
        let mut u32_const_cache: HashMap<u128, ValueId> = HashMap::new();

        for instr in old_instructions {
            rewrite_instruction(
                instr,
                &orig_value_types,
                &mut emitter,
                &mut mkseq_defs,
                &mut u32_const_cache,
            );
        }
        // BlockEmitter drops back into the function automatically.
        drop(emitter);
    }

    // Rewrite remaining type-bearing fields in instructions (MkSeq.elem_type,
    // MkTuple.element_types, Alloc.elem_type, ReadGlobal.result_type,
    // FreshWitness.result_type, Todo.result_types). These are independent of
    // the structural rewrites above and apply to every survivor instruction.
    let func = ssa.get_function_mut(fid);
    for (_, block) in func.get_blocks_mut() {
        for instr in block.get_instructions_mut() {
            flatten_embedded_types(instr);
        }
    }
}

fn flatten_embedded_types(instr: &mut OpCode) {
    match instr {
        OpCode::MkSeq { elem_type, .. } => {
            *elem_type = flatten_type(elem_type);
        }
        OpCode::MkTuple { element_types, .. } => {
            for tp in element_types.iter_mut() {
                *tp = flatten_type(tp);
            }
        }
        OpCode::Alloc { elem_type, .. } => {
            *elem_type = flatten_type(elem_type);
        }
        OpCode::ReadGlobal { result_type, .. } => {
            *result_type = flatten_type(result_type);
        }
        OpCode::FreshWitness { result_type, .. } => {
            *result_type = flatten_type(result_type);
        }
        OpCode::Todo { result_types, .. } => {
            for tp in result_types.iter_mut() {
                *tp = flatten_type(tp);
            }
        }
        OpCode::Guard { inner, .. } => flatten_embedded_types(inner.as_mut()),
        // No embedded Type fields:
        OpCode::Cmp { .. }
        | OpCode::BinaryArithOp { .. }
        | OpCode::Cast { .. }
        | OpCode::Truncate { .. }
        | OpCode::SExt { .. }
        | OpCode::Not { .. }
        | OpCode::Store { .. }
        | OpCode::Load { .. }
        | OpCode::Assert { .. }
        | OpCode::AssertCmp { .. }
        | OpCode::AssertR1C { .. }
        | OpCode::Call { .. }
        | OpCode::ArrayGet { .. }
        | OpCode::ArraySet { .. }
        | OpCode::SlicePush { .. }
        | OpCode::SliceLen { .. }
        | OpCode::SliceArray { .. }
        | OpCode::BlockSet { .. }
        | OpCode::Select { .. }
        | OpCode::ToBits { .. }
        | OpCode::ToRadix { .. }
        | OpCode::MemOp { .. }
        | OpCode::ValueOf { .. }
        | OpCode::WriteWitness { .. }
        | OpCode::NextDCoeff { .. }
        | OpCode::BumpD { .. }
        | OpCode::Constrain { .. }
        | OpCode::Lookup { .. }
        | OpCode::DLookup { .. }
        | OpCode::MulConst { .. }
        | OpCode::Rangecheck { .. }
        | OpCode::TupleProj { .. }
        | OpCode::InitGlobal { .. }
        | OpCode::DropGlobal { .. }
        | OpCode::Const { .. }
        | OpCode::Spread { .. }
        | OpCode::Unspread { .. } => {}
    }
}

// ---------------------------------------------------------------------------
// Per-instruction rewriting
// ---------------------------------------------------------------------------

fn u32_const(
    emitter: &mut HLBlockEmitter<'_>,
    cache: &mut HashMap<u128, ValueId>,
    v: u128,
) -> ValueId {
    if let Some(id) = cache.get(&v) {
        return *id;
    }
    let id = emitter.u_const(32, v);
    cache.insert(v, id);
    id
}

fn rewrite_instruction(
    instr: OpCode,
    orig_types: &HashMap<ValueId, Type>,
    emitter: &mut HLBlockEmitter<'_>,
    mkseq_defs: &mut HashMap<ValueId, (Vec<ValueId>, SeqType, Type)>,
    u32_cache: &mut HashMap<u128, ValueId>,
) {
    match instr {
        OpCode::ArrayGet {
            result,
            array,
            index,
        } => {
            // Look at the ORIGINAL element type of the array operand.
            let orig_arr_type = orig_types
                .get(&array)
                .expect("ArrayGet: array operand has no recorded original type");
            let orig_elem = orig_arr_type.get_array_element();
            if original_elem_is_aggregate(&orig_elem) {
                // Rewrite to SliceArray. The new view's element stride is the
                // total flat-leaf count of the original inner type.
                let inner_flat = flatten_type(&orig_elem);
                let length = match &inner_flat.expr {
                    TypeExpr::Array(_, n) => *n,
                    TypeExpr::WitnessOf(t) => match &t.expr {
                        TypeExpr::Array(_, n) => *n,
                        _ => unreachable!(
                            "ArrayGet on aggregate element: flat inner not Array: {}",
                            inner_flat
                        ),
                    },
                    _ => unreachable!(
                        "ArrayGet on aggregate element: flat inner not Array: {}",
                        inner_flat
                    ),
                };
                let n_const = u32_const(emitter, u32_cache, length as u128);
                let start = emitter.mul(index, n_const);
                emitter.emit(OpCode::SliceArray {
                    result,
                    array,
                    start,
                    length,
                });
            } else {
                emitter.emit(OpCode::ArrayGet {
                    result,
                    array,
                    index,
                });
            }
        }

        OpCode::ArraySet {
            result,
            array,
            index,
            value,
        } => {
            let orig_arr_type = orig_types
                .get(&array)
                .expect("ArraySet: array operand has no recorded original type");
            let orig_elem = orig_arr_type.get_array_element();
            if original_elem_is_aggregate(&orig_elem) {
                let inner_flat = flatten_type(&orig_elem);
                let length = match &inner_flat.expr {
                    TypeExpr::Array(_, n) => *n,
                    TypeExpr::WitnessOf(t) => match &t.expr {
                        TypeExpr::Array(_, n) => *n,
                        _ => unreachable!(
                            "ArraySet on aggregate element: flat inner not Array: {}",
                            inner_flat
                        ),
                    },
                    _ => unreachable!(
                        "ArraySet on aggregate element: flat inner not Array: {}",
                        inner_flat
                    ),
                };
                let n_const = u32_const(emitter, u32_cache, length as u128);
                let dst_offset = emitter.mul(index, n_const);
                emitter.emit(OpCode::BlockSet {
                    result,
                    array,
                    dst_offset,
                    source: value,
                    length,
                });
            } else {
                emitter.emit(OpCode::ArraySet {
                    result,
                    array,
                    index,
                    value,
                });
            }
        }

        OpCode::MkSeq {
            result,
            elems,
            seq_type,
            elem_type,
        } => {
            let flat_elem = flatten_type(&elem_type);
            if original_elem_is_aggregate(&elem_type) {
                // Aggregate-element MkSeq: needs flattening.
                let inner_len = match &flat_elem.expr {
                    TypeExpr::Array(_, n) => *n,
                    _ => unreachable!(
                        "MkSeq with aggregate elem: flat inner not Array: {}",
                        flat_elem
                    ),
                };
                let leaf_elem_type = match &flat_elem.expr {
                    TypeExpr::Array(inner, _) => (**inner).clone(),
                    _ => unreachable!(),
                };

                // Check if every elem is a known MkSeq that we can splice.
                let splicable = elems.iter().all(|v| mkseq_defs.contains_key(v));
                if splicable {
                    let mut leaves: Vec<ValueId> = Vec::with_capacity(elems.len() * inner_len);
                    for v in &elems {
                        let (sub_elems, _sub_seq_type, _sub_elem_type) =
                            mkseq_defs.get(v).expect("splicable MkSeq missing from defs");
                        // After this pass, the sub-MkSeq's elems are all flat scalar leaves.
                        leaves.extend_from_slice(sub_elems);
                    }
                    let total_len = leaves.len();
                    // Use the original seq_type's shape: outer seq_type stays Array(M)
                    // or Slice. For Array, the new length is total_len; for Slice,
                    // the SeqType::Slice doesn't carry a length.
                    let new_seq_type = match seq_type {
                        SeqType::Array(_outer) => SeqType::Array(total_len),
                        SeqType::Slice => SeqType::Slice,
                        SeqType::Tuple => seq_type,
                    };
                    emitter.emit(OpCode::MkSeq {
                        result,
                        elems: leaves.clone(),
                        seq_type: new_seq_type,
                        elem_type: leaf_elem_type.clone(),
                    });
                    mkseq_defs.insert(result, (leaves, new_seq_type, leaf_elem_type));
                } else {
                    // Fallback: build a flat MkSeq filled with default zeros of the
                    // right total length, then BlockSet each source row in.
                    let outer_n = elems.len();
                    let total_len = outer_n * inner_len;
                    let zero = emit_default_zero(emitter, &leaf_elem_type);
                    let zeros = vec![zero; total_len];
                    let new_seq_type = match seq_type {
                        SeqType::Array(_) => SeqType::Array(total_len),
                        SeqType::Slice => SeqType::Slice,
                        SeqType::Tuple => seq_type,
                    };
                    let init_id = emitter.fresh_value();
                    emitter.emit(OpCode::MkSeq {
                        result: init_id,
                        elems: zeros,
                        seq_type: new_seq_type,
                        elem_type: leaf_elem_type.clone(),
                    });
                    // Chain BlockSets.
                    let mut current = init_id;
                    for (i, source) in elems.iter().enumerate() {
                        let next = if i + 1 == elems.len() {
                            result
                        } else {
                            emitter.fresh_value()
                        };
                        let off = u32_const(emitter, u32_cache, (i * inner_len) as u128);
                        emitter.emit(OpCode::BlockSet {
                            result: next,
                            array: current,
                            dst_offset: off,
                            source: *source,
                            length: inner_len,
                        });
                        current = next;
                    }
                    // If elems was empty, the MkSeq itself is the result.
                    if elems.is_empty() {
                        // Emit a no-op rename: redirect result via an extra MkSeq
                        // (rare path; safe to just emit the original empty MkSeq).
                        emitter.emit(OpCode::MkSeq {
                            result,
                            elems: vec![],
                            seq_type: new_seq_type,
                            elem_type: leaf_elem_type.clone(),
                        });
                    }
                }
            } else {
                // Scalar-element MkSeq: keep as-is (elem_type may still need
                // flattening for nested-inside-Tuple shapes; flatten_embedded_types
                // handles that pass).
                emitter.emit(OpCode::MkSeq {
                    result,
                    elems: elems.clone(),
                    seq_type,
                    elem_type: flat_elem.clone(),
                });
                mkseq_defs.insert(result, (elems, seq_type, flat_elem));
            }
        }

        OpCode::SlicePush {
            dir,
            result,
            slice,
            values,
        } => {
            // If each pushed value is array-typed at the original level,
            // expand each into its inner_len flat leaves (via splice or ArrayGets).
            let elem_orig = orig_types
                .get(&slice)
                .expect("SlicePush: slice operand has no recorded original type")
                .get_array_element();
            if original_elem_is_aggregate(&elem_orig) {
                let inner_flat = flatten_type(&elem_orig);
                let inner_len = match &inner_flat.expr {
                    TypeExpr::Array(_, n) => *n,
                    _ => unreachable!(
                        "SlicePush with aggregate elem: flat inner not Array: {}",
                        inner_flat
                    ),
                };
                let mut new_values: Vec<ValueId> = Vec::with_capacity(values.len() * inner_len);
                for v in &values {
                    if let Some((sub_elems, _, _)) = mkseq_defs.get(v) {
                        new_values.extend_from_slice(sub_elems);
                    } else {
                        for j in 0..inner_len {
                            let idx = u32_const(emitter, u32_cache, j as u128);
                            let leaf = emitter.array_get(*v, idx);
                            new_values.push(leaf);
                        }
                    }
                }
                emitter.emit(OpCode::SlicePush {
                    dir,
                    result,
                    slice,
                    values: new_values,
                });
            } else {
                emitter.emit(OpCode::SlicePush {
                    dir,
                    result,
                    slice,
                    values,
                });
            }
        }

        // -- Lookup tables (array-as-table) --
        OpCode::Lookup {
            target: ref tgt, ..
        }
        | OpCode::DLookup {
            target: ref tgt, ..
        } => {
            if let crate::compiler::ssa::LookupTarget::Array(arr) = tgt {
                let orig_arr_type = orig_types
                    .get(arr)
                    .expect("Lookup: array target has no recorded original type");
                let orig_elem = orig_arr_type.get_array_element();
                assert!(
                    !original_elem_is_aggregate(&orig_elem),
                    "flatten_arrays: Lookup over multi-dimensional array table is not supported"
                );
            }
            emitter.emit(instr);
        }

        // Everything else passes through. `flatten_embedded_types` runs after
        // this pass to rewrite type-fields on these.
        other => {
            emitter.emit(other);
        }
    }
}

fn emit_default_zero(emitter: &mut HLBlockEmitter<'_>, t: &Type) -> ValueId {
    let stripped = match &t.expr {
        TypeExpr::WitnessOf(inner) => inner.as_ref(),
        _ => t,
    };
    match &stripped.expr {
        TypeExpr::Field => emitter.field_const(ark_bn254::Fr::from(0u64)),
        TypeExpr::U(bits) => emitter.u_const(*bits, 0),
        TypeExpr::I(bits) => {
            // No dedicated i_const helper; use Const opcode directly.
            let r = emitter.fresh_value();
            emitter.emit(OpCode::Const {
                result: r,
                value: ConstValue::I(*bits, 0),
            });
            r
        }
        _ => panic!(
            "flatten_arrays: cannot emit a default zero for leaf type {}",
            t
        ),
    }
}
