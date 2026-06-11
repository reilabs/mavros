//! Spills composite `Cast` instructions into concrete element-wise conversions.
//!
//! Earlier pipeline stages (UntaintControlFlow, WitnessLowering) emit a single
//! `Cast` at typed-slot boundaries even when the conversion is element-wise
//! (e.g. `Array<Field, N>` → `Array<WitnessOf(Field), N>`). Backends that
//! distinguish witness references from pure values at runtime (the AD
//! pipeline) need those conversions materialized. This pass, which runs after
//! the witgen/AD pipeline split:
//!
//! - aliases identity casts away,
//! - lowers scalar witness-strip casts (`WitnessOf(X)` → `X`) to `ValueOf`,
//! - expands array conversions: small arrays are unrolled in place, larger
//!   ones become a call to a shared per-type-pair helper function containing
//!   a counted loop (cached by `(src, tgt)` type pair),
//! - keeps scalar numeric casts, scalar witness injections and array→slice
//!   casts as they are (codegen handles those directly).

use crate::compiler::util::ice_non_elided_tuple;
use crate::compiler::{
    analysis::{
        flow_analysis::FlowAnalysis,
        types::{FunctionTypeInfo, TypeInfo},
    },
    pass_manager::{Analysis, AnalysisId, AnalysisStore, Pass},
    passes::fix_double_jumps::ValueReplacements,
    ssa::{
        BlockId, FunctionId, ValueId,
        hlssa::{
            CallTarget, HLFunction, HLSSA, OpCode, SequenceTargetType, Type, TypeExpr,
            builder::{HLBlockEmitter, HLEmitter, HLInstrBuilder, HLSSABuilder},
        },
    },
};

/// Array conversions of at most this many elements are unrolled at the use
/// site; larger ones become a call to a shared per-type-pair helper function
/// containing a counted loop.
const UNROLLED_CONVERSION_LIMIT: usize = 8;

/// A generated array-conversion helper: `fn(src) -> tgt`.
struct ConvertFnEntry {
    src: Type,
    tgt: Type,
    fn_id: FunctionId,
}

pub struct SpillCasts {}

impl Pass for SpillCasts {
    fn name(&self) -> &'static str {
        "spill_casts"
    }

    fn needs(&self) -> Vec<AnalysisId> {
        vec![TypeInfo::id(), FlowAnalysis::id()]
    }

    fn run(&self, ssa: &mut HLSSA, store: &AnalysisStore) {
        self.do_run(ssa, store.get::<TypeInfo>());
    }
}

impl SpillCasts {
    pub fn new() -> Self {
        Self {}
    }

    pub fn do_run(&self, ssa: &mut HLSSA, type_info: &TypeInfo) {
        let mut convert_fns: Vec<ConvertFnEntry> = Vec::new();
        let function_ids: Vec<_> = ssa.get_function_ids().collect();
        for function_id in function_ids {
            if !type_info.has_function(function_id) {
                continue;
            }
            let fn_type_info = type_info.get_function(function_id);
            let mut function = ssa.take_function(function_id);
            self.run_function(&mut function, ssa, fn_type_info, &mut convert_fns);
            ssa.put_function(function_id, function);
        }
    }

    fn run_function(
        &self,
        function: &mut HLFunction,
        ssa: &mut HLSSA,
        type_info: &FunctionTypeInfo,
        convert_fns: &mut Vec<ConvertFnEntry>,
    ) {
        let mut replacements = ValueReplacements::new();
        let block_ids: Vec<BlockId> = function.get_blocks().map(|(bid, _)| *bid).collect();
        for block_id in block_ids {
            let mut block = function.take_block(block_id);
            let old_instructions = block.take_instructions();
            let mut new_instructions = Vec::with_capacity(old_instructions.len());
            for mut instruction in old_instructions {
                let cast_src_type = match &instruction {
                    // Capture the operand type before value replacement: the
                    // original operand is recorded in the pre-pass type info
                    // and replacements are type-preserving.
                    OpCode::Cast { value, .. } => Some(type_info.get_value_type(*value).clone()),
                    _ => None,
                };
                replacements.replace_instruction(&mut instruction);
                match instruction {
                    OpCode::Cast {
                        result,
                        value,
                        target,
                    } => {
                        let src_type = cast_src_type.unwrap();
                        let mut builder = HLInstrBuilder::new(function, ssa, &mut new_instructions);
                        if let Some(replacement) =
                            spill_cast(value, &src_type, &target, &mut builder, convert_fns)
                        {
                            replacements.insert(result, replacement);
                        } else {
                            builder.push(OpCode::Cast {
                                result,
                                value,
                                target,
                            });
                        }
                    }
                    other => new_instructions.push(other),
                }
            }
            block.put_instructions(new_instructions);
            replacements.replace_terminator(block.get_terminator_mut());
            function.put_block(block_id, block);
        }

        // Resolve forward uses of spilled cast results in earlier blocks
        // (block iteration order is arbitrary, not dominance order).
        for (_, block) in function.get_blocks_mut() {
            for instruction in block.get_instructions_mut() {
                replacements.replace_instruction(instruction);
            }
            replacements.replace_terminator(block.get_terminator_mut());
        }
    }
}

/// Decide whether a `src → tgt` cast needs spilling. Returns `Some(value)`
/// with the converted replacement value if the cast was spilled (or is an
/// identity), `None` if the cast must be kept as-is for codegen.
fn spill_cast(
    value: ValueId,
    src: &Type,
    tgt: &Type,
    builder: &mut HLInstrBuilder<'_>,
    convert_fns: &mut Vec<ConvertFnEntry>,
) -> Option<ValueId> {
    if src == tgt {
        return Some(value);
    }
    match (&src.expr, &tgt.expr) {
        // Scalar witness strip: WitnessOf(X) → X (plus a numeric cast if the
        // payload types differ).
        (TypeExpr::WitnessOf(inner), TypeExpr::Field | TypeExpr::U(_) | TypeExpr::I(_)) => {
            let unwrapped = builder.value_of(value);
            if inner.as_ref() == tgt {
                Some(unwrapped)
            } else {
                Some(builder.cast_to(tgt.clone(), unwrapped))
            }
        }
        // Scalar numeric casts, witness-to-witness casts and scalar witness
        // injections are handled directly by codegen.
        (_, TypeExpr::WitnessOf(_)) | (TypeExpr::WitnessOf(_), _) => None,
        (
            TypeExpr::Field | TypeExpr::U(_) | TypeExpr::I(_),
            TypeExpr::Field | TypeExpr::U(_) | TypeExpr::I(_),
        ) => None,
        // Array→slice casts are pure aliases (same heap layout); element
        // types must already agree at this stage.
        (TypeExpr::Array(s, _), TypeExpr::Slice(t)) => {
            assert_eq!(
                s, t,
                "array→slice cast with element conversion is not supported"
            );
            None
        }
        // Element-wise array conversion.
        (TypeExpr::Array(..), TypeExpr::Array(..)) => {
            Some(emit_value_conversion(value, src, tgt, builder, convert_fns))
        }
        (TypeExpr::Tuple(_), _) | (_, TypeExpr::Tuple(_)) => ice_non_elided_tuple(),
        _ => panic!("spill_casts: unsupported cast {:?} -> {:?}", src, tgt),
    }
}

/// Convert a value from `source_type` to `target_type`. Conversions are pure:
/// every read is in bounds by construction and the result is a fresh value, so
/// they are safe to execute unconditionally.
///
/// Large array conversions are outlined into shared per-type-pair helper
/// functions containing a counted loop (see [`get_or_create_convert_fn`]);
/// small ones are unrolled in place.
fn emit_value_conversion(
    value: ValueId,
    source_type: &Type,
    target_type: &Type,
    builder: &mut HLInstrBuilder<'_>,
    convert_fns: &mut Vec<ConvertFnEntry>,
) -> ValueId {
    if source_type == target_type {
        return value;
    }
    match (&source_type.expr, &target_type.expr) {
        // Scalar witness injection (and witness payload conversion).
        (
            TypeExpr::Field | TypeExpr::U(_) | TypeExpr::I(_) | TypeExpr::WitnessOf(_),
            TypeExpr::WitnessOf(_),
        ) => builder.cast_to(target_type.clone(), value),
        // Scalar witness strip.
        (TypeExpr::WitnessOf(inner), TypeExpr::Field | TypeExpr::U(_) | TypeExpr::I(_)) => {
            let unwrapped = builder.value_of(value);
            if inner.as_ref() == target_type {
                unwrapped
            } else {
                builder.cast_to(target_type.clone(), unwrapped)
            }
        }
        // Scalar numeric conversion.
        (
            TypeExpr::Field | TypeExpr::U(_) | TypeExpr::I(_),
            TypeExpr::Field | TypeExpr::U(_) | TypeExpr::I(_),
        ) => builder.cast_to(target_type.clone(), value),
        // Array element conversion.
        (TypeExpr::Array(src_inner, src_size), TypeExpr::Array(tgt_inner, tgt_size)) => {
            assert_eq!(src_size, tgt_size, "array size mismatch in cast spilling");
            if *src_size > UNROLLED_CONVERSION_LIMIT && outlinable_pair(source_type, target_type) {
                emit_conversion_call(value, source_type, target_type, builder, convert_fns)
            } else {
                let mut elems = Vec::with_capacity(*src_size);
                for i in 0..*src_size {
                    let idx = builder.u_const(32, i as u128);
                    let elem = builder.array_get(value, idx);
                    let converted =
                        emit_value_conversion(elem, src_inner, tgt_inner, builder, convert_fns);
                    elems.push(converted);
                }
                builder.mk_seq(
                    elems,
                    SequenceTargetType::Array(*src_size),
                    *tgt_inner.clone(),
                )
            }
        }
        (TypeExpr::Tuple(_), _) | (_, TypeExpr::Tuple(_)) => ice_non_elided_tuple(),
        _ => panic!(
            "cast spilling: unsupported conversion {:?} -> {:?}",
            source_type, target_type
        ),
    }
}

/// Whether a `src → tgt` conversion is expressible by the outlined helper
/// functions: scalar conversions at the leaves of (possibly nested)
/// equally-sized arrays.
fn outlinable_pair(src: &Type, tgt: &Type) -> bool {
    match (&src.expr, &tgt.expr) {
        (TypeExpr::Array(s, n), TypeExpr::Array(t, m)) => {
            n == m && (s == t || outlinable_pair(s, t))
        }
        (
            TypeExpr::Field | TypeExpr::U(_) | TypeExpr::I(_) | TypeExpr::WitnessOf(_),
            TypeExpr::WitnessOf(_),
        ) => true,
        (TypeExpr::WitnessOf(_), TypeExpr::Field | TypeExpr::U(_) | TypeExpr::I(_)) => true,
        (
            TypeExpr::Field | TypeExpr::U(_) | TypeExpr::I(_),
            TypeExpr::Field | TypeExpr::U(_) | TypeExpr::I(_),
        ) => true,
        _ => false,
    }
}

/// Emit a call to the (shared) conversion helper for `src → tgt`.
fn emit_conversion_call(
    value: ValueId,
    src: &Type,
    tgt: &Type,
    builder: &mut HLInstrBuilder<'_>,
    convert_fns: &mut Vec<ConvertFnEntry>,
) -> ValueId {
    let fn_id = get_or_create_convert_fn(src, tgt, builder.ssa, convert_fns);
    let result = builder.ssa.fresh_value();
    builder.push(OpCode::Call {
        results: vec![result],
        function: CallTarget::Static(fn_id),
        args: vec![value],
        unconstrained: false,
    });
    result
}

/// Get or create the conversion helper function for an array `src → tgt`
/// pair. The helper copies the array element by element in a counted loop,
/// converting at the leaves; nested arrays convert through a child helper.
fn get_or_create_convert_fn(
    src: &Type,
    tgt: &Type,
    ssa: &mut HLSSA,
    convert_fns: &mut Vec<ConvertFnEntry>,
) -> FunctionId {
    if let Some(entry) = convert_fns
        .iter()
        .find(|entry| entry.src == *src && entry.tgt == *tgt)
    {
        return entry.fn_id;
    }

    let (src_elem, tgt_elem, size) = match (&src.expr, &tgt.expr) {
        (TypeExpr::Array(s, n), TypeExpr::Array(t, m)) => {
            assert_eq!(n, m, "array size mismatch in cast spilling");
            (s.as_ref().clone(), t.as_ref().clone(), *n)
        }
        _ => panic!(
            "conversion helpers are only generated for array types: {:?} -> {:?}",
            src, tgt
        ),
    };

    // Nested arrays convert through a child helper; create it first.
    let child_fn = match (&src_elem.expr, &tgt_elem.expr) {
        (TypeExpr::Array(..), TypeExpr::Array(..)) if src_elem != tgt_elem => Some(
            get_or_create_convert_fn(&src_elem, &tgt_elem, ssa, convert_fns),
        ),
        _ => None,
    };

    let fn_id = ssa.add_function(format!("convert_{}", convert_fns.len()));
    convert_fns.push(ConvertFnEntry {
        src: src.clone(),
        tgt: tgt.clone(),
        fn_id,
    });

    let mut sb = HLSSABuilder::new(ssa);
    sb.modify_function(fn_id, |b| {
        b.function.add_return_type(tgt.clone());
        let entry_block = b.function.get_entry_id();
        let mut e = b.block(entry_block);
        let param = e.add_parameter(src.clone());

        let default_elem = emit_typed_default(&mut e, &tgt_elem);
        let initial = e.mk_repeated(
            default_elem,
            SequenceTargetType::Array(size),
            size,
            tgt_elem.clone(),
        );
        let converted = e.build_counted_loop(size, vec![(initial, tgt.clone())], |e, i, accs| {
            let elem = e.array_get(param, i);
            let converted = emit_elem_conversion(e, elem, &src_elem, &tgt_elem, child_fn);
            let updated = e.array_set(accs[0], i, converted);
            vec![updated]
        });
        e.terminate_return(vec![converted[0]]);
    });

    fn_id
}

/// Convert one element inside a conversion helper's loop body.
fn emit_elem_conversion(
    e: &mut HLBlockEmitter<'_>,
    value: ValueId,
    src: &Type,
    tgt: &Type,
    child_fn: Option<FunctionId>,
) -> ValueId {
    if src == tgt {
        return value;
    }
    match (&src.expr, &tgt.expr) {
        (_, TypeExpr::WitnessOf(_)) => e.cast_to(tgt.clone(), value),
        (TypeExpr::WitnessOf(inner), TypeExpr::Field | TypeExpr::U(_) | TypeExpr::I(_)) => {
            let unwrapped = e.value_of(value);
            if inner.as_ref() == tgt {
                unwrapped
            } else {
                e.cast_to(tgt.clone(), unwrapped)
            }
        }
        (
            TypeExpr::Field | TypeExpr::U(_) | TypeExpr::I(_),
            TypeExpr::Field | TypeExpr::U(_) | TypeExpr::I(_),
        ) => e.cast_to(tgt.clone(), value),
        (TypeExpr::Array(..), TypeExpr::Array(..)) => {
            let child_fn = child_fn.expect("child conversion helper should have been pre-created");
            e.call(child_fn, vec![value], 1)[0]
        }
        _ => panic!("unsupported element conversion {:?} -> {:?}", src, tgt),
    }
}

/// A zero value of exactly `typ` (witness wrappers included), used to seed
/// conversion-loop accumulators with the type they will hold.
fn emit_typed_default(e: &mut HLBlockEmitter<'_>, typ: &Type) -> ValueId {
    match &typ.expr {
        TypeExpr::Field => e.field_const(ark_bn254::Fr::from(0)),
        TypeExpr::U(size) => e.u_const(*size, 0),
        TypeExpr::I(size) => e.i_const(*size, 0),
        TypeExpr::WitnessOf(inner) => {
            let zero = emit_typed_default(e, inner);
            e.cast_to(typ.clone(), zero)
        }
        TypeExpr::Array(inner, size) => {
            let elem = emit_typed_default(e, inner);
            e.mk_repeated(
                elem,
                SequenceTargetType::Array(*size),
                *size,
                *inner.clone(),
            )
        }
        _ => panic!("unsupported default value type {:?}", typ),
    }
}
