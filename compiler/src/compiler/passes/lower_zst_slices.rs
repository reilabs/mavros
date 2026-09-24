//! Lowers slices of leaf-less elements to their length.
//!
//! `ElideTuples` flattens every type into tuple-free leaves and drops whatever has none. A slice
//! of a leaf-less element has no leaves either, so it is dropped with its length, and every
//! bounds check against it.
//!
//! This pass runs before the first `ElideTuples` and rewrites `Slice<T>` with leaf-less `T` to a
//! plain `u32`, its length. Each op on such a slice becomes arithmetic on the length plus an
//! explicit bounds check built through [`seq_bounds`]. For every op but `ArrayGet` that is the
//! check the op normally has; `ArrayGet` is normally bounded by the lookup argument of a real
//! read, and a leaf-less read has no lookup, so the check is asserted here instead:
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
//! its accesses, and with them the lookup or assert that would have bounded the index. This pass
//! asserts `index < n` for both `ArrayGet` and `ArraySet`, again through [`seq_bounds`], and
//! leaves the op for the elision to drop.
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
            build_lt_bounds_assert_on_len, build_pop_bounds_assert_on_len, seq_bounds_operands,
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
            let (_, len, index, _) =
                seq_bounds_operands(b, array, index, ty_of(array), ty_of(index));
            b.emit(OpCode::AssertCmp {
                kind: crate::compiler::ssa::hlssa::CmpKind::ULt,
                lhs: index,
                rhs: len,
            });
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::{
        analysis::types::Types,
        ssa::{
            Instruction, Terminator,
            hlssa::{
                BinaryArithOpKind, CmpKind, Constant, SliceOpDir,
                builder::{HLBlockEmitter, HLSSABuilder},
            },
        },
    };

    fn empty_type() -> Type {
        Type::tuple_of(vec![])
    }

    fn empty_value(e: &mut impl HLEmitter) -> ValueId {
        e.mk_tuple(vec![], vec![])
    }

    fn zst_slice() -> Type {
        empty_type().slice_of()
    }

    fn u32_const(n: u128) -> IntBits {
        IntBits::from_u128(32, n)
    }

    fn analyses(ssa: &HLSSA) -> AnalysisStore {
        let flow = FlowAnalysis::run(ssa);
        let types = Types::new().run(ssa, &flow);
        let mut store = AnalysisStore::new();
        store.insert_with_deps::<FlowAnalysis>(flow, vec![]);
        store.insert_with_deps::<TypeInfo>(types, vec![]);
        store
    }

    fn lower(ssa: &mut HLSSA) {
        LowerZstSlices::new().run(ssa, &analyses(ssa));
    }

    fn program(
        ssa: &mut HLSSA,
        returns: Vec<Type>,
        body: impl FnOnce(&mut HLBlockEmitter<'_>) -> Vec<ValueId>,
    ) {
        let main_id = ssa.get_unique_entrypoint_id();
        let mut sb = HLSSABuilder::new(ssa);
        sb.modify_function(main_id, |b| {
            for ty in returns {
                b.function.add_return_type(ty);
            }
            let entry = b.function.get_entry_id();
            let mut e = b.test_block(entry);
            let results = body(&mut e);
            e.terminate_return(results);
        });
    }

    /// `main(s: [()], i: u32)`
    fn lowered(
        returns: Vec<Type>,
        body: impl FnOnce(&mut HLBlockEmitter<'_>, ValueId, ValueId) -> Vec<ValueId>,
    ) -> (HLSSA, ValueId, ValueId) {
        let mut ssa = HLSSA::with_main("main".to_string());
        let mut params = (ValueId(0), ValueId(0));
        program(&mut ssa, returns, |e| {
            let s = e.add_parameter(zst_slice());
            let i = e.add_parameter(Type::int(32));
            params = (s, i);
            body(e, s, i)
        });
        lower(&mut ssa);
        (ssa, params.0, params.1)
    }

    fn entry_ops(ssa: &HLSSA) -> Vec<OpCode> {
        ssa.get_unique_entrypoint()
            .get_blocks()
            .flat_map(|(_, block)| block.get_instructions())
            .cloned()
            .collect()
    }

    fn entry_ops_text(ssa: &HLSSA) -> String {
        format!("{:?}", entry_ops(ssa))
    }

    fn returned(ssa: &HLSSA) -> Vec<ValueId> {
        ssa.get_unique_entrypoint()
            .get_blocks()
            .find_map(|(_, block)| match block.get_terminator() {
                Some(Terminator::Return(values)) => Some(values.clone()),
                _ => None,
            })
            .expect("main returns")
    }

    fn def_of(ssa: &HLSSA, value: ValueId) -> OpCode {
        entry_ops(ssa)
            .into_iter()
            .find(|op| op.get_results().any(|r| *r == value))
            .unwrap_or_else(|| panic!("{value:?} is not instruction-defined"))
    }

    fn int_const(ssa: &HLSSA, value: ValueId) -> Option<IntBits> {
        match ssa.get_const(value).as_deref() {
            Some(Constant::Int(pattern)) => Some(pattern.clone()),
            _ => None,
        }
    }

    fn arith(ssa: &HLSSA, value: ValueId) -> (BinaryArithOpKind, ValueId, Option<IntBits>) {
        match def_of(ssa, value) {
            OpCode::BinaryArithOp { kind, lhs, rhs, .. } => (kind, lhs, int_const(ssa, rhs)),
            other => panic!("{value:?} is not arithmetic: {other:?}"),
        }
    }

    fn asserts(ssa: &HLSSA) -> Vec<(CmpKind, ValueId, ValueId)> {
        entry_ops(ssa)
            .into_iter()
            .filter_map(|op| match op {
                OpCode::AssertCmp { kind, lhs, rhs } => Some((kind, lhs, rhs)),
                _ => None,
            })
            .collect()
    }

    fn get_the_bounds_assert(ssa: &HLSSA) -> (ValueId, ValueId) {
        let all = asserts(ssa);
        assert_eq!(all.len(), 1, "exactly one bounds assert, got {all:?}");
        assert_eq!(all[0].0, CmpKind::ULt);
        (all[0].1, all[0].2)
    }
    #[test]
    fn mk_seq_becomes_its_element_count() {
        let (ssa, ..) = lowered(vec![zst_slice()], |e, _, _| {
            let nothing = empty_value(e);
            vec![e.mk_seq(vec![nothing; 3], SequenceTargetType::Slice, empty_type())]
        });
        assert!(asserts(&ssa).is_empty());
        assert_eq!(int_const(&ssa, returned(&ssa)[0]), Some(u32_const(3)));
    }

    #[test]
    fn mk_repeated_becomes_its_count() {
        let (ssa, ..) = lowered(vec![zst_slice()], |e, _, _| {
            let nothing = empty_value(e);
            vec![e.mk_repeated(nothing, SequenceTargetType::Slice, 5, empty_type())]
        });
        assert!(asserts(&ssa).is_empty());
        assert_eq!(int_const(&ssa, returned(&ssa)[0]), Some(u32_const(5)));
    }

    #[test]
    fn an_array_to_slice_cast_becomes_the_array_length() {
        let (ssa, ..) = lowered(vec![zst_slice()], |e, _, _| {
            let nothing = empty_value(e);
            let array = e.mk_seq(vec![nothing; 2], SequenceTargetType::Array(2), empty_type());
            vec![e.cast_to(CastTarget::ArrayToSlice, array)]
        });
        assert!(asserts(&ssa).is_empty());
        assert_eq!(int_const(&ssa, returned(&ssa)[0]), Some(u32_const(2)));
    }

    #[test]
    fn slice_len_aliases_the_slice() {
        let (ssa, s, _) = lowered(vec![Type::int(32)], |e, s, _| vec![e.slice_len(s)]);
        assert!(asserts(&ssa).is_empty());
        assert_eq!(returned(&ssa)[0], s, "the slice *is* its length");
    }

    #[test]
    fn array_get_is_bounded() {
        let (ssa, s, i) = lowered(vec![empty_type()], |e, s, i| vec![e.array_get(s, i)]);
        assert_eq!(get_the_bounds_assert(&ssa), (i, s));
    }

    #[test]
    fn array_set_is_bounded_and_aliases_the_slice() {
        let (ssa, s, i) = lowered(vec![zst_slice()], |e, s, i| {
            let nothing = empty_value(e);
            vec![e.array_set(s, i, nothing)]
        });
        assert_eq!(get_the_bounds_assert(&ssa), (i, s));
        assert_eq!(returned(&ssa)[0], s);
    }

    #[test]
    fn slice_push_adds_the_number_pushed() {
        let (ssa, s, _) = lowered(vec![zst_slice()], |e, s, _| {
            let nothing = empty_value(e);
            vec![e.slice_push(s, vec![nothing; 2], SliceOpDir::Back)]
        });
        assert!(asserts(&ssa).is_empty(), "a push cannot fail");
        assert_eq!(
            arith(&ssa, returned(&ssa)[0]),
            (BinaryArithOpKind::UAdd, s, Some(u32_const(2)))
        );
    }

    #[test]
    fn slice_insert_is_bounded_against_the_new_length() {
        let (ssa, s, i) = lowered(vec![zst_slice()], |e, s, i| {
            let nothing = empty_value(e);
            vec![e.slice_insert(s, i, nothing)]
        });
        let result = returned(&ssa)[0];
        assert_eq!(get_the_bounds_assert(&ssa), (i, result), "index < len + 1");
        assert_eq!(
            arith(&ssa, result),
            (BinaryArithOpKind::UAdd, s, Some(u32_const(1)))
        );
    }

    #[test]
    fn slice_pop_is_bounded_against_zero() {
        let (ssa, s, _) = lowered(vec![zst_slice()], |e, s, _| {
            let (rest, _) = e.slice_pop(s, SliceOpDir::Back);
            vec![rest]
        });
        let (zero, len) = get_the_bounds_assert(&ssa);
        assert_eq!(int_const(&ssa, zero), Some(u32_const(0)), "0 < len");
        assert_eq!(len, s);
        assert_eq!(
            arith(&ssa, returned(&ssa)[0]),
            (BinaryArithOpKind::USub, s, Some(u32_const(1)))
        );
    }

    #[test]
    fn slice_remove_is_bounded_and_shortens_by_one() {
        let (ssa, s, i) = lowered(vec![zst_slice()], |e, s, i| {
            let (rest, _) = e.slice_remove(s, i);
            vec![rest]
        });
        assert_eq!(get_the_bounds_assert(&ssa), (i, s));
        assert_eq!(
            arith(&ssa, returned(&ssa)[0]),
            (BinaryArithOpKind::USub, s, Some(u32_const(1)))
        );
    }

    #[test]
    fn a_leaf_less_array_access_is_bounded() {
        let mut ssa = HLSSA::with_main("main".to_string());
        let mut index = ValueId(0);
        program(&mut ssa, vec![], |e| {
            let array = e.add_parameter(empty_type().array_of(4));
            let i = e.add_parameter(Type::int(32));
            index = i;
            let nothing = empty_value(e);
            e.array_get(array, i);
            e.array_set(array, i, nothing);
            vec![]
        });
        lower(&mut ssa);

        let all = asserts(&ssa);
        assert_eq!(all.len(), 2, "one bound per access");
        for (kind, lhs, rhs) in all {
            assert_eq!(kind, CmpKind::ULt);
            assert_eq!(lhs, index);
            assert_eq!(int_const(&ssa, rhs), Some(u32_const(4)));
        }
    }

    #[test]
    fn a_leafy_slice_program_is_untouched() {
        let mut ssa = HLSSA::with_main("main".to_string());
        program(&mut ssa, vec![Type::int(32)], |e| {
            let s = e.add_parameter(Type::field().slice_of());
            let i = e.add_parameter(Type::int(32));
            let x = e.add_parameter(Type::field());
            let array = e.add_parameter(Type::field().array_of(4));
            e.array_get(array, i);
            let pushed = e.slice_push(s, vec![x], SliceOpDir::Back);
            e.array_get(pushed, i);
            let set = e.array_set(pushed, i, x);
            let inserted = e.slice_insert(set, i, x);
            let (popped, _) = e.slice_pop(inserted, SliceOpDir::Front);
            let (removed, _) = e.slice_remove(popped, i);
            vec![e.slice_len(removed)]
        });
        let before = entry_ops_text(&ssa);
        let params_before = ssa.get_unique_entrypoint().get_param_types();
        lower(&mut ssa);
        assert_eq!(entry_ops_text(&ssa), before);
        assert_eq!(ssa.get_unique_entrypoint().get_param_types(), params_before);
    }
}
