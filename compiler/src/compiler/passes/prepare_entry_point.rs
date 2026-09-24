//! Transforms the SSA so that the program entry point speaks the circuit ABI for the witness rather
//! than Noir-style typed arguments.
//!
//! It does the following four things:
//!
//! 1. **Synthesises a Wrapper for `main`:** The wrapper takes a single `Blob<Field; N>` parameter
//!    holding every flattened input field (the original main's arguments, then — when `main`
//!    declares a return — a one-field *return guard* followed by the declared return values),
//!    performs global initialization, invokes the original `main` and then uses a deep assert to
//!    constrain the return value against the declared counterpart in the witness (see **Return
//!    Guard** below). It then deinitializes the globals. The blob keeps the entry point's
//!    parameter list (and the resulting locals pressure in the generated code) constant
//!    regardless of input size.
//! 2. **Pinned Witness Writes:** A single counted loop writes every blob element to the witness,
//!    pinned so that DCE cannot remove the writes downstream, while accumulating the witness
//!    values into an array.
//! 3. **Input Reconstruction:** The original typed input values are rebuilt from slices of that
//!    witness array via per-type reconstruct functions (which also range-check integers).
//! 4. **Handling of Unconstrained Calls:** Any calls that are unconstrained are modified to write
//!    the unconstrained result to the witness. It also handles range-checking of integers, and
//!    recurses into arrays and tuples. This ensures that we bind the untrusted/unconstrained
//!    results into the constraint system.
//!
//! # Return Guard
//!
//! The declared return in the blob is optional: a `u1` guard slot between the flattened
//! parameters and returns gates the check. Each leaf asserts `guard * (result - declared) == 0`,
//! so `guard = 0` (missing `return` input, columns zero-filled) disables the check entirely.
//!
//! Enforcing the declared return is a verifier obligation: pinning only the parameter and return
//! columns accepts any return value (the prover picks `guard = 0`), so a verifier must require
//! `guard == 1` as a public input. Relying on the column being boolean is not sound either: the
//! booleanity constraint lives in the guard's reconstruct function and survives only as long as
//! the guarded asserts consuming it do.
//!
//! Cost: one witness column, one blob slot, and one booleanity row per program; each leaf check
//! stays a single R1C (see `assert_eq_deep_guarded`).

use crate::{
    collections::HashMap,
    compiler::{
        pass_manager::{AnalysisStore, Pass},
        ssa::{
            BlockId, FunctionId, Located, SourceLocation, ValueId,
            hlssa::{
                CallTarget, CastTarget, HLSSA, OpCode, SequenceTargetType, Type, TypeExpr,
                builder::{HLBlockEmitter, HLEmitter, HLSSABuilder},
            },
        },
    },
};
use mavros_int_semantics::IntBits;

/// Origin of the synthetic source location the wrapper's return-value assertion carries.
pub const RETURN_CHECK_ORIGIN: &str = "public return value check";

pub struct PrepareEntryPoint {
    main_is_unconstrained: bool,
    abi_has_return: Option<bool>,
}

struct PrepareFnEntry {
    hlssa_type: Type,
    fn_id: FunctionId,
}

struct ReconstructFnEntry {
    hlssa_type: Type,
    fn_id: FunctionId,
}

impl Pass for PrepareEntryPoint {
    fn name(&self) -> &'static str {
        "prepare_entry_point"
    }

    fn run(&self, ssa: &mut HLSSA, _store: &AnalysisStore) {
        Self::wrap_main(ssa, self.main_is_unconstrained, self.abi_has_return);
        Self::process_unconstrained_calls(ssa);
    }
}

impl PrepareEntryPoint {
    /// Defaults to signature-based return presence for hand-built SSA only.
    /// Noir lowering materializes unit as one result, so callers compiling Noir
    /// must also supply `with_abi_return` (as `Driver` does).
    pub fn new(main_is_unconstrained: bool) -> Self {
        Self {
            main_is_unconstrained,
            abi_has_return: None,
        }
    }

    /// The Noir ABI distinguishes unit (no public return) from an empty struct even
    /// though both have an empty-tuple HLSSA result. Hand-built SSA defaults to its signature.
    pub fn with_abi_return(mut self, has_return: bool) -> Self {
        self.abi_has_return = Some(has_return);
        self
    }

    fn wrap_main(ssa: &mut HLSSA, main_is_unconstrained: bool, abi_has_return: Option<bool>) {
        let original_main_id = ssa.get_unique_entrypoint_id();
        let original_main = ssa.get_unique_entrypoint();
        let param_types = original_main.get_param_types();
        let return_types = original_main.get_returns().to_vec();

        let globals_init_fn = ssa.get_globals_init_fn();
        let globals_deinit_fn = ssa.get_globals_deinit_fn();

        ssa.get_unique_entrypoint_mut()
            .set_name("original_main".to_string());

        // Reconstruct functions rebuild each typed input value from its
        // flattened field representation, range-checking integers on the way.
        let has_return = abi_has_return.unwrap_or(!return_types.is_empty());
        let guard_type = Type::int(1);
        let mut reconstruct_fns = Vec::new();
        for typ in param_types
            .iter()
            .chain(return_types.iter().filter(|_| has_return))
        {
            Self::get_or_create_reconstruct_fn(typ, ssa, &mut reconstruct_fns);
        }
        if has_return {
            Self::get_or_create_reconstruct_fn(&guard_type, ssa, &mut reconstruct_fns);
        }

        let total_fields = Self::entry_blob_field_count(&param_types, &return_types, has_return);

        let wrapper_id = ssa.add_function("wrapper_main".to_string());
        let mut sb = HLSSABuilder::new(ssa);
        sb.modify_function(wrapper_id, |b| {
            let entry_block = b.function.get_entry_id();
            let mut e = b
                .block(entry_block)
                .with_source_location(SourceLocation::synthetic("wrapper_main"));

            let blob_param = e.add_parameter(Type::blob(Type::field(), total_fields));

            // witness[0] must be constant one, emitted by the program.
            let one = e.field_const(e.field().constant(1u64));
            e.pinned_write_witness(one);

            // Write every input field to the witness in blob order, collecting
            // the witness values into an array for the reconstructions below.
            // The writes are pinned so they survive DCE even when their results
            // become unused (e.g. when inter-procedural DCE prunes call args to
            // original_main).
            let witness_inputs = (total_fields > 0).then(|| {
                let initial_array =
                    Self::emit_default_witness_array(&mut e, &Type::field(), total_fields);
                let copied = e.build_counted_loop(
                    total_fields,
                    vec![(initial_array, Type::field().array_of(total_fields))],
                    |e, i, accumulators| {
                        let elem = e.array_get(blob_param, i);
                        let witness = e.pinned_write_witness(elem);
                        let updated = e.array_set(accumulators[0], i, witness);
                        vec![updated]
                    },
                );
                copied[0]
            });

            // Rebuild each typed input value from its slice of the witness array.
            let mut offset = 0usize;
            let mut input_value = |e: &mut HLBlockEmitter<'_>, typ: &Type| {
                let witness_inputs =
                    witness_inputs.expect("a typed input implies a non-empty input blob");
                let width = Self::flattened_field_count(typ);
                let value = match &typ.expr {
                    TypeExpr::Field => {
                        let index = e.int_const(IntBits::from_u128(32, offset as u128));
                        e.array_get(witness_inputs, index)
                    }
                    TypeExpr::Int(_) | TypeExpr::Array(_, _) | TypeExpr::Tuple(_) => {
                        let child = Self::emit_reconstruct_child_input_array(
                            e,
                            witness_inputs,
                            offset,
                            width,
                        );
                        let fn_id = Self::find_reconstruct_fn(typ, &reconstruct_fns);
                        e.call(fn_id, vec![child], 1)[0]
                    }
                    _ => todo!("Not implemented yet"),
                };
                offset += width;
                value
            };

            let mut arg_values = Vec::new();
            for typ in &param_types {
                arg_values.push(input_value(&mut e, typ));
            }

            let return_guard = has_return.then(|| input_value(&mut e, &guard_type));

            let mut return_input_values = Vec::new();
            if has_return {
                for typ in &return_types {
                    return_input_values.push(input_value(&mut e, typ));
                }
            }

            if let Some(init_fn) = globals_init_fn {
                e.call(init_fn, vec![], 0);
            }

            let results = if main_is_unconstrained {
                e.call_unconstrained(original_main_id, arg_values, return_types.len())
            } else {
                e.call(original_main_id, arg_values, return_types.len())
            };
            e.emit_with_location(SourceLocation::synthetic(RETURN_CHECK_ORIGIN), |e| {
                let Some(guard) = return_guard else { return };
                let guard_field = e.cast_to_field(guard);
                for ((result, public_input), return_type) in results
                    .iter()
                    .zip(return_input_values.iter())
                    .zip(return_types.iter())
                {
                    Self::assert_eq_deep_guarded(
                        e,
                        guard_field,
                        *result,
                        *public_input,
                        return_type,
                    );
                }
            });

            if let Some(deinit_fn) = globals_deinit_fn {
                e.call(deinit_fn, vec![], 0);
            }

            e.terminate_return(vec![]);
        });
        ssa.set_unique_entrypoint(wrapper_id);
    }

    fn assert_eq_deep_guarded(
        b: &mut HLBlockEmitter<'_>,
        guard_field: ValueId,
        result: ValueId,
        public_input: ValueId,
        typ: &Type,
    ) {
        match &typ.expr {
            TypeExpr::Array(inner, size) => {
                for i in 0..*size {
                    let index = b.int_const(IntBits::from_u128(32, i as u128));
                    let result_elem = b.array_get(result, index);
                    let input_elem = b.array_get(public_input, index);
                    Self::assert_eq_deep_guarded(b, guard_field, result_elem, input_elem, inner);
                }
            }
            TypeExpr::Tuple(element_types) => {
                for (i, elem_type) in element_types.iter().enumerate() {
                    let result_elem = b.tuple_proj(result, i);
                    let input_elem = b.tuple_proj(public_input, i);
                    Self::assert_eq_deep_guarded(
                        b,
                        guard_field,
                        result_elem,
                        input_elem,
                        elem_type,
                    );
                }
            }
            _ => {
                let result_field = b.ensure_field(result, typ);
                let input_field = b.ensure_field(public_input, typ);
                let diff = b.usub(result_field, input_field);
                let gated = b.umul(guard_field, diff);
                let zero = b.field_const(b.field().constant(0u64));
                b.assert_eq(gated, zero);
            }
        }
    }

    /// Process unconstrained call results: flatten to Fields, WriteWitness,
    /// rangecheck + reconstruct, and replace original results.
    fn process_unconstrained_calls(ssa: &mut HLSSA) {
        // Pre-collect callee return types (need immutable ssa access)
        let mut callee_return_types: HashMap<FunctionId, Vec<Type>> = HashMap::default();
        let func_ids: Vec<FunctionId> = ssa.get_function_ids().collect();
        for &fid in &func_ids {
            for (_, block) in ssa.get_function(fid).get_blocks() {
                for instr in block.get_instructions() {
                    if let OpCode::Call {
                        unconstrained: true,
                        function: CallTarget::Static(callee_id),
                        ..
                    } = instr
                    {
                        callee_return_types
                            .entry(*callee_id)
                            .or_insert_with(|| ssa.get_function(*callee_id).get_returns().to_vec());
                    }
                }
            }
        }

        let mut prepare_fns = Vec::new();
        for return_types in callee_return_types.values() {
            for return_type in return_types {
                if !Self::is_zero_width(return_type) {
                    Self::get_or_create_prepare_fn(return_type, ssa, &mut prepare_fns);
                }
            }
        }

        // Process each function/block: single pass over instructions, handling
        // all unconstrained calls inline so indices never go stale.
        let mut sb = HLSSABuilder::new(ssa);
        for fid in func_ids {
            let block_ids: Vec<BlockId> = sb
                .ssa()
                .get_function(fid)
                .get_blocks()
                .map(|(id, _)| *id)
                .collect();
            sb.modify_function(fid, |fb| {
                for bid in block_ids {
                    let block = fb.function.get_block_mut(bid);
                    let old_instructions = block.take_instructions();

                    let mut new_instructions = Vec::new();

                    for mut instr in old_instructions {
                        let location = instr.location().clone();
                        if let OpCode::Call {
                            unconstrained: true,
                            results,
                            function: CallTarget::Static(callee_id),
                            ..
                        } = &mut *instr
                        {
                            let return_types = callee_return_types
                                .get(callee_id)
                                .expect("unconstrained static call return types should be known");
                            assert_eq!(
                                results.len(),
                                return_types.len(),
                                "ICE: unconstrained call result count does not match callee return count"
                            );

                            let original_results = results.clone();
                            let fresh_results = original_results
                                .iter()
                                .zip(return_types)
                                .map(|(original, typ)| {
                                    if Self::is_zero_width(typ) {
                                        *original
                                    } else {
                                        fb.ssa.fresh_value()
                                    }
                                })
                                .collect::<Vec<_>>();
                            *results = fresh_results.clone();

                            new_instructions.push(instr);
                            for ((original_result, fresh_result), return_type) in original_results
                                .into_iter()
                                .zip(fresh_results)
                                .zip(return_types.iter())
                            {
                                // Keep the original call result for zero-width values:
                                // it is already defined, and has no witness fields to prepare.
                                if Self::is_zero_width(return_type) {
                                    continue;
                                }
                                let prepare_fn = Self::find_prepare_fn(return_type, &prepare_fns);
                                new_instructions.push(Located::new(
                                    OpCode::Call {
                                        results: vec![original_result],
                                        function: CallTarget::Static(prepare_fn),
                                        args: vec![fresh_result],
                                        unconstrained: false,
                                    },
                                    location.clone(),
                                ));
                            }
                        } else {
                            new_instructions.push(instr);
                        }
                    }

                    let block = fb.function.get_block_mut(bid);
                    block.put_instructions(new_instructions);
                }
            });
        }
    }

    /// Unlike ABI field counting, this also accepts internal types. Slices retain
    /// their length even with zero-width elements, so they cannot be skipped here.
    fn is_zero_width(typ: &Type) -> bool {
        match &typ.expr {
            TypeExpr::Tuple(fields) => fields.iter().all(Self::is_zero_width),
            TypeExpr::Array(inner, size) => *size == 0 || Self::is_zero_width(inner),
            TypeExpr::WitnessOf(inner) => Self::is_zero_width(inner),
            _ => false,
        }
    }

    fn find_prepare_fn(typ: &Type, prepare_fns: &[PrepareFnEntry]) -> FunctionId {
        prepare_fns
            .iter()
            .find(|entry| entry.hlssa_type == *typ)
            .map(|entry| entry.fn_id)
            .expect("prepare function should have been pre-created")
    }

    fn unique_prepare_fn_name(prepare_fns: &[PrepareFnEntry]) -> String {
        format!("prepare_{}", prepare_fns.len())
    }

    fn unique_reconstruct_fn_name(reconstruct_fns: &[ReconstructFnEntry]) -> String {
        format!("reconstruct_{}", reconstruct_fns.len())
    }

    pub(crate) fn entry_blob_field_count(
        param_types: &[Type],
        return_types: &[Type],
        has_return: bool,
    ) -> usize {
        param_types
            .iter()
            .chain(return_types.iter().filter(|_| has_return))
            .map(Self::flattened_field_count)
            .sum::<usize>()
            + usize::from(has_return)
    }

    /// How many elements of the entry point's input blob a value of `typ` occupies.
    ///
    /// An integer takes one for the moment. A parameter is written to a single witness column and
    /// range-checked to its own width there, and a range check bounds a _field element_. As a
    /// result the widest integer one slot carries is `widest_injective_int_bits`, and below that
    /// there is no width at which one slot is not enough.
    ///
    /// A wider parameter would need a slot per limb here, and a reconstruction that combines them —
    /// which is a witnessed wide addition and shift. Supporting this is future work.
    pub fn flattened_field_count(typ: &Type) -> usize {
        match &typ.expr {
            TypeExpr::Field | TypeExpr::Int(_) => 1,
            TypeExpr::Array(inner, size) => Self::flattened_field_count(inner) * size,
            TypeExpr::Tuple(element_types) => {
                element_types.iter().map(Self::flattened_field_count).sum()
            }
            TypeExpr::WitnessOf(inner) => Self::flattened_field_count(inner),
            _ => todo!("Not implemented yet"),
        }
    }

    fn get_or_create_prepare_fn(
        typ: &Type,
        ssa: &mut HLSSA,
        prepare_fns: &mut Vec<PrepareFnEntry>,
    ) -> FunctionId {
        if let Some(entry) = prepare_fns.iter().find(|entry| entry.hlssa_type == *typ) {
            return entry.fn_id;
        }

        let fn_id = ssa.add_function(Self::unique_prepare_fn_name(prepare_fns));
        prepare_fns.push(PrepareFnEntry {
            hlssa_type: typ.clone(),
            fn_id,
        });

        let child_fns = match &typ.expr {
            TypeExpr::Array(inner, _) => {
                vec![Some(Self::get_or_create_prepare_fn(
                    inner,
                    ssa,
                    prepare_fns,
                ))]
            }
            TypeExpr::Tuple(element_types) => element_types
                .iter()
                .map(|elem_type| {
                    (!Self::is_zero_width(elem_type))
                        .then(|| Self::get_or_create_prepare_fn(elem_type, ssa, prepare_fns))
                })
                .collect(),
            _ => Vec::new(),
        };

        let mut sb = HLSSABuilder::new(ssa);
        sb.modify_function(fn_id, |b| {
            b.function.add_return_type(typ.clone());
            let entry_block = b.function.get_entry_id();
            let location = SourceLocation::synthetic(b.function.get_name());
            let mut e = b.block(entry_block).with_source_location(location);
            let param = e.add_parameter(typ.clone());
            let result = Self::emit_prepare_body(&mut e, param, typ, &child_fns);
            e.terminate_return(vec![result]);
        });

        fn_id
    }

    fn emit_prepare_body(
        e: &mut HLBlockEmitter<'_>,
        value_id: ValueId,
        typ: &Type,
        child_fns: &[Option<FunctionId>],
    ) -> ValueId {
        match &typ.expr {
            TypeExpr::Field => e.write_witness(value_id),
            TypeExpr::Int(size) => {
                let cast_back = CastTarget::Int(*size);
                let as_field = e.cast_to_field(value_id);
                let witness = e.write_witness(as_field);

                if *size == 1 {
                    let zero = e.field_const(e.field().constant(0));
                    let one = e.field_const(e.field().constant(1));
                    let x_sub_1 = e.usub(witness, one);
                    let x_times_x_sub_1 = e.umul(witness, x_sub_1);
                    e.assert_eq(x_times_x_sub_1, zero);
                } else {
                    e.rangecheck(witness, *size);
                }

                e.cast_to(cast_back, witness)
            }
            TypeExpr::Array(inner, size) => {
                let child_fn = child_fns
                    .first()
                    .copied()
                    .flatten()
                    .expect("array prepare function should have child function");
                let initial_array = Self::emit_default_witness_array(e, inner, *size);
                let prepared_array = e.build_counted_loop(
                    *size,
                    vec![(initial_array, typ.clone())],
                    |e, index, accumulators| {
                        let current_array = accumulators[0];
                        let elem = e.array_get(value_id, index);
                        let prepared = e.call(child_fn, vec![elem], 1);
                        let updated_array = e.array_set(current_array, index, prepared[0]);
                        vec![updated_array]
                    },
                );
                prepared_array[0]
            }
            TypeExpr::Tuple(element_types) => {
                let mut elems = Vec::with_capacity(element_types.len());
                for (i, child_fn) in child_fns.iter().enumerate() {
                    let elem = e.tuple_proj(value_id, i);
                    let prepared = match child_fn {
                        Some(child_fn) => e.call(*child_fn, vec![elem], 1)[0],
                        None => elem,
                    };
                    elems.push(prepared);
                }
                e.mk_tuple(elems, element_types.clone())
            }
            TypeExpr::WitnessOf(_) => value_id,
            _ => e.write_witness(value_id),
        }
    }

    fn find_reconstruct_fn(typ: &Type, reconstruct_fns: &[ReconstructFnEntry]) -> FunctionId {
        reconstruct_fns
            .iter()
            .find(|entry| entry.hlssa_type == *typ)
            .map(|entry| entry.fn_id)
            .expect("reconstruct function should have been pre-created")
    }

    fn get_or_create_reconstruct_fn(
        typ: &Type,
        ssa: &mut HLSSA,
        reconstruct_fns: &mut Vec<ReconstructFnEntry>,
    ) -> FunctionId {
        if let Some(entry) = reconstruct_fns
            .iter()
            .find(|entry| entry.hlssa_type == *typ)
        {
            return entry.fn_id;
        }

        match &typ.expr {
            TypeExpr::Array(inner, _) => {
                Self::get_or_create_reconstruct_fn(inner, ssa, reconstruct_fns);
            }
            TypeExpr::WitnessOf(inner) => {
                Self::get_or_create_reconstruct_fn(inner, ssa, reconstruct_fns);
            }
            TypeExpr::Tuple(element_types) => {
                for elem_type in element_types {
                    Self::get_or_create_reconstruct_fn(elem_type, ssa, reconstruct_fns);
                }
            }
            _ => {}
        }

        let fn_id = ssa.add_function(Self::unique_reconstruct_fn_name(reconstruct_fns));
        reconstruct_fns.push(ReconstructFnEntry {
            hlssa_type: typ.clone(),
            fn_id,
        });

        let mut sb = HLSSABuilder::new(ssa);
        sb.modify_function(fn_id, |b| {
            b.function.add_return_type(typ.clone());
            let entry_block = b.function.get_entry_id();
            let location = SourceLocation::synthetic(b.function.get_name());
            let mut e = b.block(entry_block).with_source_location(location);

            let input_len = Self::flattened_field_count(typ);
            let input_array = e.add_parameter(Type::field().array_of(input_len));
            let result = Self::emit_reconstruct_body(&mut e, typ, input_array, reconstruct_fns);
            e.terminate_return(vec![result]);
        });

        fn_id
    }

    fn emit_reconstruct_body(
        e: &mut HLBlockEmitter<'_>,
        typ: &Type,
        input_array: ValueId,
        reconstruct_fns: &[ReconstructFnEntry],
    ) -> ValueId {
        match &typ.expr {
            TypeExpr::Field => {
                let zero = e.int_const(IntBits::zero(32));
                e.array_get(input_array, zero)
            }
            TypeExpr::Int(size) => {
                let zero = e.int_const(IntBits::zero(32));
                let field_param = e.array_get(input_array, zero);
                if *size == 1 {
                    let zero = e.field_const(e.field().constant(0));
                    let one = e.field_const(e.field().constant(1));
                    let x_sub_1 = e.usub(field_param, one);
                    let x_times_x_sub_1 = e.umul(field_param, x_sub_1);
                    e.assert_eq(x_times_x_sub_1, zero);
                } else {
                    e.rangecheck(field_param, *size);
                }

                e.cast_to(CastTarget::Int(*size), field_param)
            }
            TypeExpr::Array(inner, size) => {
                let child_fn = Self::find_reconstruct_fn(inner, reconstruct_fns);
                let child_width = Self::flattened_field_count(inner);
                let initial_array = Self::emit_default_witness_array(e, inner, *size);
                let reconstructed_array = e.build_counted_loop(
                    *size,
                    vec![(initial_array, typ.clone())],
                    |e, index, accumulators| {
                        let current_array = accumulators[0];
                        let child_input = Self::emit_reconstruct_child_input_array_at_index(
                            e,
                            input_array,
                            index,
                            child_width,
                        );
                        let reconstructed = e.call(child_fn, vec![child_input], 1);
                        let updated_array = e.array_set(current_array, index, reconstructed[0]);
                        vec![updated_array]
                    },
                );
                reconstructed_array[0]
            }
            TypeExpr::Tuple(element_types) => {
                let mut elems = Vec::with_capacity(element_types.len());
                let mut field_offset = 0;
                for elem_type in element_types {
                    let child_fn = Self::find_reconstruct_fn(elem_type, reconstruct_fns);
                    let child_width = Self::flattened_field_count(elem_type);
                    let child_input = Self::emit_reconstruct_child_input_array(
                        e,
                        input_array,
                        field_offset,
                        child_width,
                    );
                    let reconstructed = e.call(child_fn, vec![child_input], 1);
                    elems.push(reconstructed[0]);
                    field_offset += child_width;
                }
                e.mk_tuple(elems, element_types.clone())
            }
            TypeExpr::WitnessOf(inner) => {
                let inner_fn = Self::find_reconstruct_fn(inner, reconstruct_fns);
                let reconstructed = e.call(inner_fn, vec![input_array], 1);
                e.cast_to_witness_of(reconstructed[0])
            }
            _ => todo!("Not implemented yet"),
        }
    }

    /// Build a placeholder value that is witness-typed at the leaves. These
    /// seed the fill-loop accumulators of this pass, whose every slot is
    /// overwritten with a witness-leaved value before use. Making the
    /// placeholders witness-typed up front keeps the accumulator's inferred
    /// type stable across the loop, so untaint does not have to emit an
    /// (unrolled) pure→witness array conversion at the loop boundary.
    fn emit_default_witness_value(e: &mut HLBlockEmitter<'_>, typ: &Type) -> ValueId {
        match &typ.expr {
            TypeExpr::Field => {
                let zero = e.field_const(e.field().constant(0));
                e.cast_to_witness_of(zero)
            }
            TypeExpr::Int(size) => {
                let zero = e.int_const(IntBits::zero(*size));
                e.cast_to_witness_of(zero)
            }
            TypeExpr::Array(inner, size) => Self::emit_default_witness_array(e, inner, *size),
            TypeExpr::WitnessOf(inner) => Self::emit_default_witness_value(e, inner),
            TypeExpr::Tuple(element_types) => {
                let mut elems = Vec::with_capacity(element_types.len());
                for elem_type in element_types {
                    elems.push(Self::emit_default_witness_value(e, elem_type));
                }
                e.mk_tuple(elems, element_types.clone())
            }
            _ => todo!("Not implemented yet"),
        }
    }

    fn emit_default_witness_array(
        e: &mut HLBlockEmitter<'_>,
        inner: &Type,
        size: usize,
    ) -> ValueId {
        if size == 0 {
            return e.mk_seq(Vec::new(), SequenceTargetType::Array(0), inner.clone());
        }
        let default_elem = Self::emit_default_witness_value(e, inner);
        e.mk_repeated(
            default_elem,
            SequenceTargetType::Array(size),
            size,
            inner.clone(),
        )
    }

    fn emit_reconstruct_child_input_array(
        e: &mut HLBlockEmitter<'_>,
        input_array: ValueId,
        start: usize,
        len: usize,
    ) -> ValueId {
        let start = e.int_const(IntBits::from_u128(32, start as u128));
        Self::emit_reconstruct_child_input_array_from(e, input_array, start, len)
    }

    fn emit_reconstruct_child_input_array_at_index(
        e: &mut HLBlockEmitter<'_>,
        input_array: ValueId,
        index: ValueId,
        width: usize,
    ) -> ValueId {
        let start = if width == 1 {
            index
        } else {
            let width_value = e.int_const(IntBits::from_u128(32, width as u128));
            e.umul(index, width_value)
        };
        Self::emit_reconstruct_child_input_array_from(e, input_array, start, width)
    }

    /// Copy `width` consecutive fields out of `input_array`, starting at the
    /// dynamic index `start`, into a fresh `Array<Field; width>`. Uses a
    /// counted loop rather than unrolled reads so the emitted code stays small
    /// for wide aggregates.
    fn emit_reconstruct_child_input_array_from(
        e: &mut HLBlockEmitter<'_>,
        input_array: ValueId,
        start: ValueId,
        width: usize,
    ) -> ValueId {
        // Up to one element there is nothing to loop over.
        if width <= 1 {
            let fields = (0..width)
                .map(|_| e.array_get(input_array, start))
                .collect();
            return e.mk_seq(fields, SequenceTargetType::Array(width), Type::field());
        }

        let initial_array = Self::emit_default_witness_array(e, &Type::field(), width);
        let copied = e.build_counted_loop(
            width,
            vec![(initial_array, Type::field().array_of(width))],
            |e, i, accumulators| {
                let src_index = e.uadd(start, i);
                let elem = e.array_get(input_array, src_index);
                let updated = e.array_set(accumulators[0], i, elem);
                vec![updated]
            },
        );
        copied[0]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::ssa::Terminator;

    #[test]
    fn zero_width_unconstrained_results_keep_their_definitions_without_preparation() {
        let unit = Type::tuple_of(vec![]);
        for return_type in [
            unit.clone(),
            unit.clone().array_of(2),
            Type::field().array_of(0),
            unit.clone().slice_of(),
            Type::field().slice_of(),
            Type::tuple_of(vec![unit, Type::field()]),
        ] {
            let mut ssa = HLSSA::with_main("main".into());
            let main = ssa.get_unique_entrypoint_id();
            let callee = ssa.add_function("callee".into());
            let mut sb = HLSSABuilder::new(&mut ssa);
            sb.modify_function(callee, |b| {
                b.function.add_return_type(return_type.clone());
                let entry = b.function.get_entry_id();
                let mut e = b
                    .block(entry)
                    .with_source_location(SourceLocation::synthetic("test"));
                let value = e.add_parameter(return_type.clone());
                e.terminate_return(vec![value]);
            });
            let mut result = None;
            sb.modify_function(main, |b| {
                b.function.add_return_type(return_type.clone());
                let entry = b.function.get_entry_id();
                let mut e = b
                    .block(entry)
                    .with_source_location(SourceLocation::synthetic("test"));
                let arg = e.add_parameter(return_type.clone());
                let value = e.call_unconstrained(callee, vec![arg], 1)[0];
                result = Some(value);
                e.terminate_return(vec![value]);
            });
            PrepareEntryPoint::process_unconstrained_calls(&mut ssa);
            let ops: Vec<_> = ssa
                .get_function(main)
                .get_entry()
                .get_instructions()
                .collect();
            let zero_width = PrepareEntryPoint::is_zero_width(&return_type);
            assert_eq!(ops.len(), if zero_width { 1 } else { 2 });
            let OpCode::Call {
                results,
                function: CallTarget::Static(target),
                unconstrained: true,
                ..
            } = ops[0]
            else {
                panic!("original unconstrained call must survive");
            };
            assert_eq!(*target, callee);
            if zero_width {
                assert_eq!(results, &[result.unwrap()]);
            }
            let Some(Terminator::Return(values)) =
                ssa.get_function(main).get_entry().get_terminator()
            else {
                panic!("missing return");
            };
            assert_eq!(values, &[result.unwrap()]);
            for (_, f) in ssa.iter_functions() {
                if f.get_name().starts_with("prepare_") {
                    assert!(!PrepareEntryPoint::is_zero_width(&f.get_param_types()[0]));
                }
            }
        }
    }

    #[test]
    fn absent_abi_return_has_no_slots_or_return_reconstruction() {
        let mut ssa = HLSSA::with_main("main".into());
        let main = ssa.get_unique_entrypoint_id();
        HLSSABuilder::new(&mut ssa).modify_function(main, |b| {
            b.function.add_return_type(Type::int(8));
            let entry = b.function.get_entry_id();
            let mut e = b
                .block(entry)
                .with_source_location(SourceLocation::synthetic("test"));
            let value = e.int_const(IntBits::zero(8));
            e.terminate_return(vec![value]);
        });
        PrepareEntryPoint::wrap_main(&mut ssa, false, Some(false));
        assert_eq!(
            ssa.get_unique_entrypoint().get_param_types(),
            vec![Type::blob(Type::field(), 0)]
        );
        assert_eq!(
            ssa.iter_functions().count(),
            2,
            "unused return reconstruction should not be generated"
        );
    }
}
