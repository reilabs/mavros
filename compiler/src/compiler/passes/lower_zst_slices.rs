//! Lowers slices of leaf-less elements to their length.
//!
//! `ElideTuples` flattens every type into tuple-free leaves and drops whatever has none. A slice
//! of a leaf-less element has no leaves either, so it is dropped with its length, and every
//! bounds check against it.
//!
//! This pass runs before the first `ElideTuples` and rewrites `Slice<T>` with leaf-less `T` to a
//! plain `u32`, its length. Each op on such a slice becomes arithmetic on the length plus
//! the bounds check it would normally have, taken from [`seq_bounds`]:
//!
//! - `MkSeq`/`MkRepeated`/`ArrayToSlice` cast: the constant element count.
//! - `SliceLen`: alias the operand (the slice *is* its length).
//! - `ArrayGet`: `assert index < len`; the element is synthesized.
//! - `ArraySet`: `assert index < len`; alias the operand.
//! - `SlicePush`: `len + k`.  `SliceInsert`: `len + 1`, `assert index < len + 1`.
//! - `SlicePop`: `assert 0 < len`, `len - 1`; the popped element is synthesized.
//! - `SliceRemove`: `assert index < len`, `len - 1`; the removed element is synthesized.
//!
//! A fixed-size array of a leaf-less element keeps its length in its type, but the elision drops
//! its accesses and their bounds checks with them. This pass emits that check, again through
//! [`seq_bounds`], and leaves the op for the elision to drop.
//!
//! `Map` casts and guards are generated later so reaching them in this pass is an ICE.

use crate::compiler::{
    analysis::{
        flow_analysis::FlowAnalysis,
        types::{FunctionTypeInfo, TypeInfo},
    },
    pass_manager::{Analysis, AnalysisId, AnalysisStore, Pass},
    passes::shared::{
        seq_bounds::{
            build_lt_bounds_assert_on_len, build_pop_bounds_assert_on_len,
            build_seq_access_bounds_assert,
        },
        value_replacements::{ReplaceScope, ValueReplacements},
    },
    ssa::{
        BlockId, ValueId,
        hlssa::{
            CastTarget, HLFunction, HLSSA, OpCode, SequenceTargetType, Type, TypeExpr,
            builder::{HLEmitter, HLInstrBuilder},
        },
    },
};

use mavros_int_semantics::IntBits;

pub struct LowerZstSlices {}

impl LowerZstSlices {
    pub fn new() -> Self {
        Self {}
    }
}

impl Pass for LowerZstSlices {
    fn name(&self) -> &'static str {
        "lower_zst_slices"
    }

    fn needs(&self) -> Vec<AnalysisId> {
        vec![TypeInfo::id(), FlowAnalysis::id()]
    }

    fn run(&self, ssa: &mut HLSSA, store: &AnalysisStore) {
        Self::do_run(ssa, store.get::<FlowAnalysis>(), store.get::<TypeInfo>());
    }

    fn preserves(&self) -> Vec<AnalysisId> {
        // No block or terminator is added, removed or edited.
        vec![FlowAnalysis::id()]
    }
}

impl LowerZstSlices {
    pub fn do_run(ssa: &mut HLSSA, flow: &FlowAnalysis, types: &TypeInfo) {
        let globals = ssa.get_global_types().iter().map(lower_type).collect();
        ssa.set_global_types(globals);

        for fid in ssa.get_function_ids().collect::<Vec<_>>() {
            let reachable: Vec<BlockId> = flow
                .get_function_cfg(fid)
                .get_domination_pre_order()
                .collect();
            let mut function = ssa.take_function(fid);
            rewrite_function(&mut function, ssa, types.get_function(fid), &reachable);
            ssa.put_function(fid, function);
        }
    }
}

fn rewrite_function(
    function: &mut HLFunction,
    ssa: &mut HLSSA,
    fti: &FunctionTypeInfo,
    reachable: &[BlockId],
) {
    for ty in function.iter_returns_mut() {
        *ty = lower_type(ty);
    }

    let mut aliases = ValueReplacements::new();
    for bid in reachable {
        for (_, ty) in function.get_block_mut(*bid).get_parameters_mut() {
            *ty = lower_type(ty);
        }
        let old = function.get_block_mut(*bid).take_instructions();
        let mut new_instrs = Vec::with_capacity(old.len());
        for instr in old {
            let (op, loc) = instr.take();
            let mut b = HLInstrBuilder::new(function, ssa, &mut new_instrs, loc);
            lower_instruction(op, fti, &mut b, &mut aliases);
        }
        function.get_block_mut(*bid).put_instructions(new_instrs);
    }
    aliases.apply_to_blocks(function, ReplaceScope::Inputs, reachable.iter().copied());
}

fn lower_instruction(
    op: OpCode,
    fti: &FunctionTypeInfo,
    b: &mut HLInstrBuilder<'_>,
    aliases: &mut ValueReplacements,
) {
    let is_zst_slice = |v: ValueId| is_zst_slice_type(fti.get_value_type(v));
    let ty_of = |v: ValueId| fti.get_value_type(v);
    let len_const =
        |b: &mut HLInstrBuilder<'_>, n: usize| b.int_const(IntBits::from_u128(32, n as u128));

    match op {
        OpCode::MkSeq {
            result,
            elems,
            seq_type: SequenceTargetType::Slice,
            elem_type,
        } if is_zero_leaf(&elem_type) => {
            let n = len_const(b, elems.len());
            aliases.insert(result, n);
        }
        OpCode::MkRepeated {
            result,
            seq_type: SequenceTargetType::Slice,
            count,
            elem_type,
            ..
        } if is_zero_leaf(&elem_type) => {
            let n = len_const(b, count);
            aliases.insert(result, n);
        }
        OpCode::Cast {
            result,
            value,
            target: CastTarget::ArrayToSlice,
        } if is_zst_slice(result) => {
            let n = match &ty_of(value).expr {
                TypeExpr::Array(_, n) => *n,
                other => ice!("ArrayToSlice cast of a non-array {other:?}"),
            };
            let n = len_const(b, n);
            aliases.insert(result, n);
        }
        OpCode::Cast {
            target: CastTarget::Map(_),
            ..
        } => ice!("lower_zst_slices: Map cast before ElideTuples"),
        OpCode::SliceLen { result, slice } if is_zst_slice(slice) => aliases.insert(result, slice),
        OpCode::ArrayGet {
            result,
            array,
            index,
        } if is_zst_slice(array) => {
            let (assert, _, _) = build_lt_bounds_assert_on_len(b, array, index, ty_of(index));
            b.emit(assert);
            let elem = synthesize_leafless(b, ty_of(result));
            aliases.insert(result, elem);
        }
        OpCode::ArraySet {
            result,
            array,
            index,
            ..
        } if is_zst_slice(array) => {
            let (assert, _, _) = build_lt_bounds_assert_on_len(b, array, index, ty_of(index));
            b.emit(assert);
            aliases.insert(result, array);
        }
        OpCode::SlicePush {
            result,
            slice,
            values,
            ..
        } if is_zst_slice(slice) => {
            let k = len_const(b, values.len());
            let new_len = b.uadd(slice, k);
            aliases.insert(result, new_len);
        }
        OpCode::SlicePop {
            result_slice,
            result_elem,
            slice,
            ..
        } if is_zst_slice(slice) => {
            let assert = build_pop_bounds_assert_on_len(b, slice);
            b.emit(assert);
            let one = len_const(b, 1);
            let new_len = b.usub(slice, one);
            aliases.insert(result_slice, new_len);
            let elem = synthesize_leafless(b, ty_of(result_elem));
            aliases.insert(result_elem, elem);
        }
        OpCode::SliceInsert {
            result,
            slice,
            index,
            ..
        } if is_zst_slice(slice) => {
            let one = len_const(b, 1);
            let new_len = b.uadd(slice, one);
            let (assert, _, _) = build_lt_bounds_assert_on_len(b, new_len, index, ty_of(index));
            b.emit(assert);
            aliases.insert(result, new_len);
        }
        OpCode::SliceRemove {
            result_slice,
            result_elem,
            slice,
            index,
        } if is_zst_slice(slice) => {
            let (assert, _, _) = build_lt_bounds_assert_on_len(b, slice, index, ty_of(index));
            b.emit(assert);
            let one = len_const(b, 1);
            let new_len = b.usub(slice, one);
            aliases.insert(result_slice, new_len);
            let elem = synthesize_leafless(b, ty_of(result_elem));
            aliases.insert(result_elem, elem);
        }

        // An access into a leaf-less *array*. The elision drops the op, so its bounds check is
        // emitted here and the op kept for the elision to drop.
        OpCode::ArrayGet { array, index, .. } | OpCode::ArraySet { array, index, .. }
            if is_zero_leaf(ty_of(array)) =>
        {
            let assert =
                build_seq_access_bounds_assert(b, array, index, ty_of(array), ty_of(index))
                    .expect("a leaf-less array is an array");
            b.emit(assert);
            b.emit(op);
        }

        // Ops that spell out types get them rewritten.
        OpCode::MkSeq {
            result,
            elems,
            seq_type,
            elem_type,
        } => b.emit(OpCode::MkSeq {
            result,
            elems,
            seq_type,
            elem_type: lower_type(&elem_type),
        }),
        OpCode::MkRepeated {
            result,
            element,
            seq_type,
            count,
            elem_type,
        } => b.emit(OpCode::MkRepeated {
            result,
            element,
            seq_type,
            count,
            elem_type: lower_type(&elem_type),
        }),
        OpCode::MkSeqOfBlob {
            result,
            element_type,
            blob,
        } => b.emit(OpCode::MkSeqOfBlob {
            result,
            element_type: lower_type(&element_type),
            blob,
        }),
        OpCode::MkTuple {
            result,
            elems,
            element_types,
        } => b.emit(OpCode::MkTuple {
            result,
            elems,
            element_types: element_types.iter().map(lower_type).collect(),
        }),
        OpCode::FreshWitness {
            result,
            result_type,
        } => b.emit(OpCode::FreshWitness {
            result,
            result_type: lower_type(&result_type),
        }),
        OpCode::ReadGlobal {
            result,
            offset,
            result_type,
        } => b.emit(OpCode::ReadGlobal {
            result,
            offset,
            result_type: lower_type(&result_type),
        }),
        OpCode::Todo {
            payload,
            results,
            result_types,
        } => b.emit(OpCode::Todo {
            payload,
            results,
            result_types: result_types.iter().map(lower_type).collect(),
        }),

        // No pass before this one emits guards, so a guarded slice op cannot slip through unseen.
        OpCode::Guard { .. } => {
            ice!("lower_zst_slices: unexpected guard before ElideTuples")
        }

        other => b.emit(other),
    }
}

fn synthesize_leafless(b: &mut impl HLEmitter, ty: &Type) -> ValueId {
    match &ty.expr {
        TypeExpr::Tuple(elems) => {
            let values = elems.iter().map(|e| synthesize_leafless(b, e)).collect();
            b.mk_tuple(values, elems.clone())
        }
        TypeExpr::Array(inner, n) => {
            let elem = synthesize_leafless(b, inner);
            b.mk_repeated(elem, SequenceTargetType::Array(*n), *n, (**inner).clone())
        }
        TypeExpr::Ref(inner) => {
            let value = synthesize_leafless(b, inner);
            b.alloc(value)
        }
        TypeExpr::WitnessOf(inner) => {
            let value = synthesize_leafless(b, inner);
            b.cast_to_witness_of(value)
        }
        other => ice!("lower_zst_slices: {other:?} is not leaf-less"),
    }
}

fn lower_type(ty: &Type) -> Type {
    match &ty.expr {
        TypeExpr::Slice(inner) => {
            if is_zero_leaf(inner) {
                Type::int(32)
            } else {
                lower_type(inner).slice_of()
            }
        }
        TypeExpr::Array(inner, n) => lower_type(inner).array_of(*n),
        TypeExpr::Ref(inner) => lower_type(inner).ref_of(),
        TypeExpr::Tuple(elems) => Type::tuple_of(elems.iter().map(lower_type).collect()),
        TypeExpr::WitnessOf(inner) => Type::witness_of(lower_type(inner)),
        TypeExpr::Function(returns) => {
            Type::function_returning(returns.iter().map(lower_type).collect())
        }
        TypeExpr::Field | TypeExpr::Int(_) | TypeExpr::Blob(..) => ty.clone(),
    }
}

fn is_zst_slice_type(ty: &Type) -> bool {
    matches!(&ty.expr, TypeExpr::Slice(inner) if is_zero_leaf(inner))
}

fn is_zero_leaf(ty: &Type) -> bool {
    match &ty.expr {
        TypeExpr::Tuple(elems) => elems.iter().all(is_zero_leaf),
        TypeExpr::Array(inner, _) | TypeExpr::Ref(inner) | TypeExpr::WitnessOf(inner) => {
            is_zero_leaf(inner)
        }
        TypeExpr::Slice(_)
        | TypeExpr::Field
        | TypeExpr::Int(_)
        | TypeExpr::Function(_)
        | TypeExpr::Blob(..) => false,
    }
}
